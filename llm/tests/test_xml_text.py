import random

import pytest

from rpm_llm import xml_labels
from rpm_llm.xml_labels import describe, parse

GRID4 = [[0.25, 0.25, 0.5, 0.5], [0.25, 0.75, 0.5, 0.5], [0.75, 0.25, 0.5, 0.5], [0.75, 0.75, 0.5, 0.5]]


def write_xml(path, slots_per_panel, rng):
    panels = []
    for slots in slots_per_panel:
        pos = [GRID4[s] for s in slots]
        ents = "".join(f'<Entity bbox="{GRID4[s]}" Type="{rng.randint(1, 5)}" Size="{rng.randint(0, 5)}" '
                       f'Color="{rng.randint(0, 9)}" Angle="0"/>' for s in slots)
        panels.append(f'<Panel><Struct name="Singleton"><Component id="0" name="Grid">'
                      f'<Layout Position="{pos}" Number="{len(slots)}">{ents}</Layout>'
                      f'</Component></Struct></Panel>')
    path.write_text(f"<Data><Panels>{''.join(panels)}</Panels><Rules/></Data>")


@pytest.fixture
def d4(tmp_path):
    xml_labels._CANON.clear()
    rng = random.Random(0)
    cfg = tmp_path / "distribute_four"
    cfg.mkdir()
    write_xml(cfg / "RAVEN_0_train.xml", [[0, 1, 2, 3]] * 16, rng)
    test = [[3]] * 8 + [[0, 3], [1], [2], [0, 1, 2], [3], [1, 2], [0], [0, 1, 2, 3]]
    write_xml(cfg / "RAVEN_1_test.xml", test, rng)
    return cfg / "RAVEN_1_test.xml"


def test_slot_from_bbox_not_position_list(d4):
    panels = parse(str(d4), "distribute_four")
    # one object in the bottom-right slot: Position lists only that slot, so its index is 0,
    # but the canonical slot is 3 -> global position 4
    assert [o["pos"] for o in panels[0]] == [4]
    assert [o["pos"] for o in panels[8]] == [1, 4]


def test_describe_text(d4):
    text = describe(str(d4), "distribute_four")
    assert "Layout: a 2x2 grid of slots." in text
    assert "Row 1, column 1: bottom-right:" in text
    assert "Row 3, column 3: ?" in text
    assert text.count("\n") == 3 + 8 + 1 + 2 + 8 - 1
    assert "\nH: top-left:" in text and "; bottom-right:" in text


def test_run_text_mode_sends_no_image(d4, tmp_path, monkeypatch):
    import json
    import numpy as np
    from rpm_llm import run as run_mod
    from rpm_llm.providers import Reply

    np.savez(d4.with_suffix(".npz"), image=np.zeros((16, 8, 8), np.uint8), target=2, predict=2)
    seen = []

    class Fake:
        def __init__(self, model, **kw):
            pass

        def __call__(self, prompt, png):
            seen.append((prompt, png))
            return Reply("The correct answer is: C", "end_turn", "fake")

    monkeypatch.setitem(run_mod.PROVIDERS, "fake", Fake)
    out = tmp_path / "r.jsonl"
    run_mod.main(["--data", str(d4.parent.parent), "--provider", "fake", "--model", "m",
                  "--prompt", "xml_text", "--out", str(out)])
    (prompt, png), = seen
    assert png is None and "Row 3, column 3: ?" in prompt and "{puzzle}" not in prompt
    rec = json.loads(out.read_text())
    assert rec["input"] == "xml" and rec["correct"]


def test_parser_matches_train_pynl():
    import ast
    from pathlib import Path
    src = Path(__file__).resolve().parents[2] / "pynl" / "train_pynl.py"
    if not src.exists():
        pytest.skip("train_pynl.py not found")

    def defs(path, swap_eval):
        tree = ast.parse(path.read_text())
        out = {}
        for node in tree.body:
            name = getattr(node, "name", None) or getattr(getattr(node, "targets", [None])[0], "id", None)
            if name not in ("POS_MAP", "_canonical", "parse"):
                continue
            if isinstance(node, ast.FunctionDef) and ast.get_docstring(node):
                node.body = node.body[1:]
            if swap_eval:
                for n in ast.walk(node):
                    if isinstance(n, ast.Call) and getattr(n.func, "id", None) == "eval":
                        n.func = ast.Attribute(value=ast.Name("ast", ast.Load()), attr="literal_eval", ctx=ast.Load())
            out[name] = ast.dump(node, annotate_fields=False)
        return out

    mine = defs(Path(xml_labels.__file__), False)
    theirs = defs(src, True)
    assert set(mine) == set(theirs) == {"POS_MAP", "_canonical", "parse"}
    assert mine == theirs
