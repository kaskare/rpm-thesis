import hashlib
import json

import numpy as np
import pytest
from PIL import Image
import io

from rpm_llm import run as run_mod
from rpm_llm.data import sample, LETTERS
from rpm_llm.parse import parse_answer
from rpm_llm.prompts import TEHLAR
from rpm_llm.providers import Reply
from rpm_llm.render import render_png
from rpm_llm.score import load, summarise



@pytest.fixture(scope="module")
def DATA(tmp_path_factory):
    """Synthetic I-RAVEN layout: 2 puzzles per configuration and split."""
    root = tmp_path_factory.mktemp("iraven")
    rng = np.random.default_rng(0)
    from rpm_llm.data import CONFIGS
    i = 0
    for c in CONFIGS:
        (root / c).mkdir()
        for split in ("train", "val", "test"):
            for _ in range(2):
                img = rng.integers(0, 256, (16, 160, 160), dtype=np.uint8)
                t = int(rng.integers(0, 8))
                np.savez(root / c / f"RAVEN_{i}_{split}.npz", image=img, target=t, predict=(t + 1) % 8)
                i += 1
    return root


@pytest.mark.parametrize("text,expected", [
    ("... The correct answer is: C", "C"),
    ("The correct answer is: **F**.", "F"),
    ("The correct answer is: D**", "D"),
    ("The correct answer is: 'E'.", "E"),
    ("The correct answer is: Circle (Option C).", "C"),
    ("The correct answer is: Triangle (A).", "A"),
    ("The correct answer is: **Option H**\n\n```markdown", "H"),
    ("The correct answer is:\n\n**B**", "B"),
    ("The correct answer is: _", None),
    ("The correct answer is: Small Square.", None),
    ("I think it is option C. Final: The correct answer is: G", "G"),
    ("The correct answer is: A. Wait, The correct answer is: B", "B"),
    ("No idea.", None),
    ("", None),
])
def test_parse(text, expected):
    assert parse_answer(text) == expected


def test_parse_lenient_only_when_asked():
    t = "After elimination the answer is **E**."
    assert parse_answer(t) is None
    assert parse_answer(t, lenient=True) == "E"


def test_prompt_is_tehlars():
    # sha256 of the mistral_prompt string in Tehlar's Mistral/main.py (thesis_paper @ 05874b9)
    assert hashlib.sha256(TEHLAR.encode()).hexdigest() == "6a3b5d7b4ba70c78c82e8faa4ea803df4b002c80b0de4b6abf16741bd10feb0a"


def test_sample_uses_target_and_is_deterministic(DATA):
    ps = sample(DATA, ("test",))
    assert len(ps) == 14 and {p.split for p in ps} == {"test"}
    assert sample(DATA, ("train", "val", "test"), 1, seed=3) == sample(DATA, ("train", "val", "test"), 1, seed=3)
    assert len(sample(DATA, ("train", "val", "test"), 1)) == 7
    for p in ps:
        with np.load(p.path) as d:
            assert p.target == int(d["target"]) != int(d["predict"])


def test_render_matches_tehlar_png_size(DATA):
    p = sample(DATA, ("train", "val", "test"))[0]
    img = Image.open(io.BytesIO(render_png(p.path)))
    assert img.size == (1280, 672) and img.mode == "L"


def test_end_to_end_with_fake_provider(DATA, tmp_path, monkeypatch):
    calls = []

    class Fake:
        def __init__(self, model, **kw):
            pass

        def __call__(self, prompt, png):
            calls.append(png)
            if len(calls) == 2:
                raise RuntimeError("rate limited")
            return Reply("reasoning... The correct answer is: **A**", "end_turn", "fake")

    monkeypatch.setitem(run_mod.PROVIDERS, "fake", Fake)
    man, out = tmp_path / "m.jsonl", tmp_path / "r.jsonl"
    run_mod.main(["--data", str(DATA), "--splits", "train,val,test", "--per-config", "1",
                  "--write-manifest", str(man)])
    args = ["--manifest", str(man), "--provider", "fake", "--model", "fake", "--runs", "2",
            "--limit", "3", "--workers", "1", "--out", str(out)]
    run_mod.main(args)
    recs = [json.loads(l) for l in out.read_text().splitlines()]
    assert len(recs) == 6 and sum(r["status"] == "error" for r in recs) == 1
    run_mod.main(args)  # resume retries only the failed item
    recs = load([out])
    assert len(recs) == 6 and all(r["status"] == "ok" for r in recs)
    assert all(r["prediction"] == "A" for r in recs)
    report = summarise(recs)
    assert "API errors not scored: 0" in report
