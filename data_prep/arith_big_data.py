#!/usr/bin/env python3
"""100,000 puzzles of I-RAVEN-X Arithmetic for the grokking runs (docs/ARITHMETIC.md): rows of 10, values 0..49, IBM's generator
with its own --rule Arithmetic, so every attribute of every puzzle follows Arithmetic, plus or minus, Type included. Puzzles 0-99999
train, 100000-100299 pick the epoch, 100300-101299 are unseen. IBM's script stops at 10,000, so its get_sample is called here.
Its row shuffle draws from an unseeded generator, so a rerun gives other puzzles.

    python3 data_prep/arith_big_data.py
"""
import json, os, sys

sys.path.insert(0, os.path.expanduser("~/raven-large-language-models/src/datasets/generation"))
from iravenx_task import get_sample, set_seeds

N, MAXVAL = 10, 50
TRAIN, SEL, UNSEEN = 100000, 300, 1000
PATH = f"data/iravenx_arith/center_single_shuffle_n_{N}_maxval_{MAXVAL}.json"


def main():
    set_seeds(2026)
    samples = {str(i): get_sample(N, MAXVAL, "Arithmetic", "shuffle") for i in range(TRAIN + SEL + UNSEEN)}
    os.makedirs(os.path.dirname(PATH), exist_ok=True)
    json.dump(samples, open(PATH, "w"), default=str)
    print(f"{len(samples)} puzzles in {PATH}")


if __name__ == "__main__":
    main()
