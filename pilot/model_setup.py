"""Load Qwen2.5-VL and register forward hooks on every LLM decoder layer.

Hooks capture the last-token hidden state per sample into a dict keyed by layer index.
Always remove hooks in a try/finally — leaked hooks accumulate tensors and silently OOM.
"""
from contextlib import contextmanager
from typing import Dict, Tuple
import torch
from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration

import config


def _get_decoder_layers(model):
    """Return the LLM decoder layer list across transformers versions."""
    if hasattr(model.model, "language_model"):
        return model.model.language_model.layers
    return model.model.layers


def load_model_and_processor():
    """Returns (model, processor). Model is in eval mode on config.DEVICE."""
    print(f"Loading {config.MODEL_NAME} on {config.DEVICE} ({config.DTYPE})...")
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        config.MODEL_NAME,
        torch_dtype=config.DTYPE,
        device_map=config.DEVICE if config.DEVICE != "mps" else None,
        low_cpu_mem_usage=True,
    )
    if config.DEVICE == "mps":
        model = model.to("mps")
    model.eval()

    # max_pixels caps per-image vision tokens. Default is 1280*28*28 (~1M) per image,
    # which OOMs on MPS for multi-image inputs (3+ images crashes 16 GB Macs).
    # 256*28*28 (~200K per image) keeps a 4-image example under ~800K total visual pixels.
    processor = AutoProcessor.from_pretrained(
        config.MODEL_NAME,
        min_pixels=128 * 28 * 28,
        max_pixels=256 * 28 * 28,
    )

    n_layers = len(_get_decoder_layers(model))
    print(f"  loaded. LLM decoder has {n_layers} layers.")
    return model, processor


@contextmanager
def capture_last_token_hidden_states(model, store: Dict[int, torch.Tensor]):
    """Context manager that hooks every LLM decoder layer.

    On exit, hooks are removed even if the forward pass raises. `store` is mutated
    in-place: store[layer_idx] = tensor of shape (batch, hidden_dim).
    """
    layers = _get_decoder_layers(model)
    hooks = []

    def make_hook(idx: int):
        def fn(module, inputs, output):
            hs = output[0] if isinstance(output, tuple) else output
            store[idx] = hs[:, -1, :].detach().to("cpu").float()
        return fn

    try:
        for i, layer in enumerate(layers):
            hooks.append(layer.register_forward_hook(make_hook(i)))
        yield store
    finally:
        for h in hooks:
            h.remove()


def _format_question(question: str, options: Dict[str, str], n_images: int) -> str:
    """Format a BLINK question as constrained multiple-choice asking for the letter.

    Hidden states are unaffected by the prompt format choice for our purposes.
    """
    options_lines = "\n".join(f"{letter}. {text}" for letter, text in options.items())
    image_prefix = f"You are shown {n_images} image(s).\n" if n_images > 1 else ""
    return (
        f"{image_prefix}{question}\n\n"
        f"Options:\n{options_lines}\n\n"
        f"Answer with the letter only."
    )


def run_single(model, processor, images, question: str,
               options: Dict[str, str] = None) -> Tuple[str, Dict[int, torch.Tensor]]:
    """One inference pass with 1-4 images. Returns (generated_text, {layer_idx: hidden_state}).

    `images` is a list of PIL images. If options is provided, the question is formatted as
    multiple-choice asking for a letter; otherwise free-form.
    """
    if not isinstance(images, (list, tuple)):
        images = [images]

    prompt_text = _format_question(question, options, len(images)) if options else question

    # Build chat content with one image block per image, then the text block.
    content = [{"type": "image", "image": img} for img in images]
    content.append({"type": "text", "text": prompt_text})
    messages = [{"role": "user", "content": content}]

    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = processor(text=[text], images=list(images), padding=True, return_tensors="pt").to(model.device)

    hidden_states: Dict[int, torch.Tensor] = {}
    max_new = 16 if options else 64
    with capture_last_token_hidden_states(model, hidden_states), torch.no_grad():
        out = model.generate(**inputs, max_new_tokens=max_new, do_sample=False)

    gen_ids = out[:, inputs.input_ids.shape[1]:]
    generated = processor.batch_decode(gen_ids, skip_special_tokens=True)[0].strip()

    return generated, hidden_states
