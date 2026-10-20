# Superseded PyNeuraLogic experiments (moved here 2026-10-20)

Kept because `thesis/PYNEURALOGIC_NOTES.md` §4–§8 report their results. They ran from the repository
root when the current code lived there (`from train_pynl import ...`); to rerun one, copy it into
`pynl/` and prefix its script path in the job file with `pynl/`.

| file | what (notes §12) |
|---|---|
| `train_pynl_softmax.py` + `submit_pynl_softmax.sh` | pure PyNeuraLogic, 8-way softmax (16.0%) |
| `train_pynl_aligned.py` | pure, structurally corrected: shared `W` via `edge/3`, identity residual, 2-layer head |
| `train_pynl_torch.py` + `submit_pynl_hybrid.sh`, `submit_pynl_hybrid384.sh` | hybrid, `center_single` (52.9%) |
| `train_ladder.py` + `submit_ladder.sh`, `submit_ladder_full.sh` | the 4-level ablation, `--level N` |
| `submit_pynl.sh` | the early pure trainer in `train_pynl.py` (27.2%, per-candidate binary scoring) |
| `submit_all_constellations.sh`, `submit_full.sh`, `submit_sweep.sh`, `submit_layout_grid.sh` | `train_all_constellations.py` runs before `submit_max.sh` (first joint runs, the timed-out full run, the padding sweep, the layout grid search) |
| `pynl_iraven_minimal.ipynb` | the first annotated minimal notebook |
