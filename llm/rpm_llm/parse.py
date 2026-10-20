"""Extracts the chosen letter from a free-text answer. Unparseable answers return None and
are scored as wrong."""
import re

_FINAL = re.compile(r"correct\s+answer\s+is\s*[:\-]?", re.I)
_TAIL_PATTERNS = (
    re.compile(r"\(\s*(?:option\s+)?([A-H])\s*\)", re.I),          # "Circle (C)", "(Option C)"
    re.compile(r"\b(?:option|choice|answer)\s*[:\-]?\s*([A-H])\b"),  # "Option C"
    re.compile(r"(?<![A-Za-z])([A-H])(?![A-Za-z])"),                 # "**C**.", "C", "'C'"
)
_LENIENT = re.compile(r"\b(?:answer|option|choice)\b(?:\s+is)?\s*[:\-]?\s*[\*\"'(]*([A-H])(?![A-Za-z])")


def _from_tail(tail):
    tail = tail.strip().split("\n\n")[0][:80]
    for pat in _TAIL_PATTERNS:
        m = pat.search(tail)
        if m:
            return m.group(1).upper()
    return None


def parse_answer(text, lenient=False):
    """Letter after the last 'The correct answer is', else None.
    With lenient=True, falls back to the last 'answer/option/choice X' anywhere in the text."""
    if not text:
        return None
    for m in reversed(list(_FINAL.finditer(text))):
        letter = _from_tail(text[m.end():])
        if letter:
            return letter
    if lenient:
        hits = _LENIENT.findall(text)
        if hits:
            return hits[-1].upper()
    return None
