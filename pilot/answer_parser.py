"""Parse free-form VLM output into a multiple-choice letter (A, B, C, ...).

Robust to:
- Letter answers with various wrappers: "A", "(A)", "A.", "A:", "Answer: A"
- Option-text answers: "yes" matches the option mapped to "yes"
- Lowercased / sentence-case starts: "Yes, the car..."

Returns None when parsing fails — caller decides whether to skip or count as wrong.
"""
import re
from typing import Dict, Optional


# Match a standalone letter A-Z at the start of the string (with optional wrappers/punctuation).
_LETTER_PATTERN = re.compile(
    r"""^\s*                # leading whitespace
        (?:answer[:\s]+)?   # optional "Answer:" prefix
        [\(\[]?             # optional opening paren/bracket
        ([A-Z])             # the letter (captured)
        (?:[\)\]\.\,\:\s]|$)  # closing wrapper, punctuation, OR end of string
    """,
    re.VERBOSE | re.IGNORECASE,
)


def parse_answer(generated: str, options: Dict[str, str]) -> Optional[str]:
    """Extract the chosen letter from a model's free-form output.

    Args:
        generated: model's raw output text.
        options: {"A": "yes", "B": "no", ...} — letter to text mapping.

    Returns:
        The chosen letter (e.g. "A"), or None if parsing failed.
    """
    if not generated:
        return None

    text = generated.strip()
    text_lower = text.lower()
    valid_letters = set(options.keys())

    # Strategy 1: explicit letter at start (e.g. "A", "(B)", "Answer: A.")
    m = _LETTER_PATTERN.match(text)
    if m:
        letter = m.group(1).upper()
        if letter in valid_letters:
            return letter

    # Strategy 2: option text appears at start (e.g. "yes, the car..." → option "A: yes")
    # Sort options by text length descending so longer texts take precedence over substrings.
    sorted_opts = sorted(
        ((letter, str(text_val).lower().strip()) for letter, text_val in options.items()),
        key=lambda kv: -len(kv[1]),
    )
    for letter, opt_text in sorted_opts:
        if opt_text and (text_lower.startswith(opt_text + " ")
                          or text_lower.startswith(opt_text + ",")
                          or text_lower.startswith(opt_text + ".")
                          or text_lower == opt_text):
            return letter

    # Strategy 3: option text appears anywhere in the first 80 characters
    head = text_lower[:80]
    matches = [letter for letter, opt_text in sorted_opts if opt_text and opt_text in head]
    if len(matches) == 1:
        return matches[0]

    return None


def is_correct(parsed: Optional[str], gold: str) -> bool:
    """True iff parsed letter matches gold letter exactly. Unparsed counts as wrong."""
    return parsed is not None and parsed == gold


if __name__ == "__main__":
    # Quick self-tests
    cases = [
        # (generated, options, expected_letter)
        ("A", {"A": "yes", "B": "no"}, "A"),
        ("(B)", {"A": "yes", "B": "no"}, "B"),
        ("Answer: A.", {"A": "yes", "B": "no"}, "A"),
        ("Yes, the car is beneath the cat.", {"A": "yes", "B": "no"}, "A"),
        ("No, the car is not under the cat.", {"A": "yes", "B": "no"}, "B"),
        ("The bowl is in front of the apples.", {"A": "yes", "B": "no"}, None),
        ("There are 3 dogs in the image.", {"A": "2", "B": "3", "C": "4"}, "B"),
        ("", {"A": "yes", "B": "no"}, None),
    ]
    failed = 0
    for gen, opts, expected in cases:
        got = parse_answer(gen, opts)
        ok = got == expected
        marker = "✓" if ok else "✗"
        if not ok:
            failed += 1
        print(f"  {marker}  parse({gen!r:50s}, {opts}) = {got!r}, expected {expected!r}")
    print(f"\n{len(cases) - failed}/{len(cases)} passed")
