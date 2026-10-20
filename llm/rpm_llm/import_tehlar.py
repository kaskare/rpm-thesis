"""Converts Tehlar's saved Pixtral outputs (Mistral/output/<model>/<prompt>/<dataset>/<variant>/
<variant>_run_<k>_results.csv) into run.py records, re-parsing the full responses.

  python -m rpm_llm.import_tehlar Mistral/output results/tehlar_pixtral.jsonl
"""
import csv
import json
import re
import sys
from pathlib import Path

from .data import CONFIGS
from .parse import parse_answer

_RUN = re.compile(r"_run_(\d+)_results\.csv$")


def split_name(image_path):
    """Both path styles he used -> (variant, config, stem)."""
    parts = Path(image_path).parts
    stem = Path(image_path).stem
    if parts[-2] in CONFIGS:
        return parts[-3], parts[-2], stem
    name = stem  # flat style: <variant>_<config>_RAVEN_<i>_<split>
    variant, rest = name[:8], name[9:]
    config = max((c for c in CONFIGS if rest.startswith(c + "_")), key=len)
    return variant, config, rest[len(config) + 1:]


def convert(root, out):
    n = 0
    with open(out, "w") as fo:
        for f in sorted(Path(root).glob("*/*/*/*/*_run_*_results.csv")):
            model, prompt, dataset = f.parts[-5], f.parts[-4], f.parts[-3]
            run = int(_RUN.search(f.name).group(1))
            for row in csv.DictReader(open(f, newline="")):
                variant, config, stem = split_name(row["image_path"])
                text = row["answer"]
                pred = parse_answer(text)
                rec = dict(provider="mistral", model=model, prompt=prompt, effort=None, run=run,
                           id=f"{dataset}/{variant}/{config}/{stem}", variant=f"{dataset}/{variant}",
                           config=config, target=row["correct_answer"], status="ok",
                           prediction=pred, lenient=parse_answer(text, lenient=True),
                           correct=pred == row["correct_answer"], response=text,
                           tehlar_prediction=row["prediction"], tehlar_correct=row["correct"] == "True")
                fo.write(json.dumps(rec) + "\n")
                n += 1
    print(f"{n} records -> {out}")


if __name__ == "__main__":
    convert(sys.argv[1], sys.argv[2])
