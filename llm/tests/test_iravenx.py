import gzip
import json

import pytest

from rpm_llm import iravenx
from rpm_llm import run as run_mod
from rpm_llm.data import load_manifest, save_manifest
from rpm_llm.providers import Reply


def fake_iravenx(path, n=4, count=3):
    """Same layout as IBM's iravenx_task.py output: n*n-1 context panels + 8 candidates."""
    data = {}
    for i in range(count):
        rpm = [[{"Type": str(r), "Size": str(c), "Color": "7", "Angle": "99"}]
               for r in range(n) for c in range(n)][: n * n - 1]
        rpm += [[{"Type": str(n - 1), "Size": str(k), "Color": "7", "Angle": "5"}] for k in range(8)]
        data[str(i)] = {"rules": [{"Type": "Constant"}], "rpm": rpm, "target": str(n - 1)}
    with gzip.open(path, "wt") as f:
        json.dump(data, f)
    return path


@pytest.fixture
def xfile(tmp_path):
    iravenx.load.cache_clear()
    return fake_iravenx(tmp_path / "center_single_shuffle_n_4_maxval_10.json.gz")


def test_describe_shows_last_rows(xfile):
    text = iravenx.describe(f"{xfile}#0", nshow=3)
    lines = text.splitlines()
    assert lines[0] == "Matrix (4 columns; the last 3 of 4 rows are shown):"
    assert lines[1] == "Row 1: (1, 0, 7), (1, 1, 7), (1, 2, 7), (1, 3, 7)"
    assert lines[3] == "Row 3: (3, 0, 7), (3, 1, 7), (3, 2, 7), ?"
    assert lines[-1] == "H: (3, 7, 7)" and "99" not in text


def test_sample_and_manifest_roundtrip(xfile, tmp_path):
    ps = iravenx.sample(xfile, first=2)
    assert [p.id for p in ps] == [f"iravenx/center_single_shuffle_n_4_maxval_10/{i}" for i in (0, 1)]
    assert ps[0].answer == "D"
    save_manifest(ps, tmp_path / "m.jsonl")
    assert load_manifest(tmp_path / "m.jsonl") == ps


def test_run_iravenx(xfile, tmp_path, monkeypatch):
    seen = []

    class Fake:
        def __init__(self, model, **kw):
            pass

        def __call__(self, prompt, png):
            seen.append((prompt, png))
            return Reply("The correct answer is: D", "end_turn", "fake")

    monkeypatch.setitem(run_mod.PROVIDERS, "fake", Fake)
    out = tmp_path / "r.jsonl"
    run_mod.main(["--iravenx", str(xfile), "--prompt", "iravenx_text", "--provider", "fake",
                  "--model", "m", "--out", str(out)])
    assert len(seen) == 3 and all(png is None and "{puzzle}" not in p for p, png in seen)
    recs = [json.loads(line) for line in out.read_text().splitlines()]
    assert all(r["input"] == "iravenx" and r["nshow"] == 3 and r["correct"] for r in recs)
    with pytest.raises(SystemExit):
        run_mod.main(["--iravenx", str(xfile), "--provider", "fake", "--model", "m", "--out", str(out)])
