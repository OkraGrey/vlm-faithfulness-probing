"""Load BLINK from HuggingFace.

BLINK has 14 task types; each example yields (image, question, options_dict, answer_letter).

Schema notes (from running against the live HF dataset):
- choices is a list of option texts, no letter prefixes (e.g. ["yes", "no"]).
- answer is formatted "(B)" — we strip the parens and use the letter to index into choices.
"""
import string
from datasets import load_dataset
from PIL import Image
from typing import Iterator, Tuple, Dict, List

import config


def _gather_images(ex) -> List[Image.Image]:
    """Collect all non-None image_N fields (image_1 .. image_4) from a BLINK example.

    Many BLINK tasks (Multi-view_Reasoning, Jigsaw, Visual_Similarity, etc.) require
    multiple images per question; only using image_1 starves the model of context.
    """
    images = []
    for i in range(1, 5):
        key = f"image_{i}"
        if key in ex and ex[key] is not None:
            images.append(ex[key])
    return images


def load_blink_samples(task: str = config.BLINK_TASK,
                       n: int = config.NUM_SAMPLES,
                       split: str = "val") -> Iterator[Tuple[List[Image.Image], str, Dict[str, str], str]]:
    """Yield n examples from BLINK[task][split].

    Returns: (list of PIL images, question, options dict {A:..., B:..., ...}, answer letter).
    The image list has 1-4 entries depending on the task.
    """
    ds = load_dataset("BLINK-Benchmark/BLINK", task, split=split)
    letters = string.ascii_uppercase

    for i in range(min(n, len(ds))):
        ex = ds[i]
        images = _gather_images(ex)
        question = ex["question"]
        choices = ex["choices"]
        answer = ex["answer"].strip("() ")
        options = {letters[j]: choice for j, choice in enumerate(choices)}
        yield images, question, options, answer


def load_blink_multi(tasks: List[str] = None,
                     n_per_task: int = None,
                     split: str = None) -> Iterator[Tuple[str, int, List[Image.Image], str, Dict[str, str], str]]:
    """Yield samples from multiple BLINK tasks.

    Returns: (task_name, sample_idx, list of PIL images, question, options dict, answer letter).
    """
    tasks = tasks if tasks is not None else config.BLINK_TASKS
    n_per_task = n_per_task if n_per_task is not None else config.SAMPLES_PER_TASK
    split = split if split is not None else config.BLINK_SPLIT

    for task in tasks:
        for idx, (images, question, options, answer) in enumerate(
            load_blink_samples(task=task, n=n_per_task, split=split)
        ):
            yield task, idx, images, question, options, answer


if __name__ == "__main__":
    print(f"Loading 1 sample from each of {len(config.BLINK_TASKS)} BLINK tasks (peek at schema)...")
    by_task = {}
    for task, idx, imgs, q, opts, ans in load_blink_multi(n_per_task=1):
        by_task[task] = len(imgs)
        print(f"\n[{task}]")
        print(f"  num_images: {len(imgs)}")
        print(f"  question:   {q[:90]}")
        print(f"  options:    {opts}")
        print(f"  answer:     {ans} -> {opts.get(ans, '?')}")
    print(f"\nImages-per-question by task: {by_task}")
