#!/usr/bin/env python3
"""Numbers in the thesis drafts that the results files do not contain.

Every decimal number in the thesis text of docs/thesis/*.md (the "Notes for you" and the reference lists left out)
is looked up, as a whole token, in the files the results are recorded in: RESULTS, SYMMETRY_RESULTS, ARITHMETIC,
FINDINGS, the lab notes and the reference checks. A number that is not found is printed with its file and line, to be
checked by hand. Finding a number is weak evidence (the same digits can occur for another reason); not finding one
is what this script is for. Section, figure, table and theorem numbers are skipped.

    python3 analysis/thesis_number_check.py              # every chapter and the appendices
    python3 analysis/thesis_number_check.py ch6          # files whose name contains ch6
"""
import glob, os, re, sys

SOURCES = ["docs/RESULTS.md", "docs/SYMMETRY_RESULTS.md", "docs/ARITHMETIC.md", "docs/FINDINGS.md",
           "docs/PYNEURALOGIC_NOTES.md", "docs/thesis/REFERENCE_CHECKS.md"]
NUM = re.compile(r"(?<![\w.·⁻])(\d{1,3}(?:,\d{3})*\.\d+|\d+\.\d+)(?![\w.]*\d)")
LABEL = re.compile(r"(Sections?|Chapters?|Figures?|Tables?|Theorem|Lemma|Statements?|Proposition|Definitions?|"
                   r"Corollary|§|notes|items?|v)\s*(\d+(\.\d+)*\s*(,|and|to|–)\s*)*$", re.I)


def thesis_text(path):
    """Lines of thesis text: up to the notes, without the reference lists."""
    out, skip = [], False
    for k, line in enumerate(open(path, encoding="utf-8"), 1):
        if line.startswith("## Notes for you"):
            break
        if line.startswith("## "):
            skip = line.startswith("## References")
        if not skip:
            out.append((k, line.rstrip("\n")))
    return out


def tokens(text):
    return set(m.group(1).replace(",", "") for m in NUM.finditer(text))


if __name__ == "__main__":
    corpus = set()
    for f in SOURCES:
        corpus |= tokens(open(f, encoding="utf-8").read())
    files = sorted(glob.glob("docs/thesis/*.md"))
    files = [f for f in files if not os.path.basename(f).isupper() and "REFERENCE" not in f]
    if len(sys.argv) > 1:
        files = [f for f in files if any(a in f for a in sys.argv[1:])]
    total = 0
    for f in files:
        misses = []
        for k, line in thesis_text(f):
            for m in NUM.finditer(line):
                if LABEL.search(line[:m.start()]):
                    continue
                if m.group(1).replace(",", "") not in corpus:
                    misses.append((k, m.group(1), line.strip()))
        total += len(misses)
        print(f"{f}: {len(misses)} number(s) not found")
        for k, n, line in misses:
            print(f"  line {k}: {n}   | {line[:150]}")
    print(f"\n{total} in all. Each needs a look: a rounding, a number derived in the text, or a mistake.")
