# Superseded on 2026-10-28

Moved here when the generative facet model (`pynl/bench_hub_grounding.py`) and the torch baselines on its
task (`baselines/`) replaced these models. Not runnable from here as is; scripts expect to sit in `pynl/`,
`torch_gnn/` or `scripts/` and run from the repository root. Their notes sections are in git history
(commit 4380e83, `thesis/PYNEURALOGIC_NOTES.md` §1, §4–§14, §16, §19). Their experiment folders and large logs
were deleted.

| folder | what | headline result |
|---|---|---|
| `pynl_gnn_style/` | The GNN-style PyNeuraLogic model: object/slot graph, torch embeddings and loss through the PyNeuraLogic torch bridge (`train_all_constellations.py`), its I-RAVEN-X variant (`train_iravenx.py`), evaluation (`eval_pynl_test.py`), per-rule analysis (`analyze_constellation.py`) and jobs | 68.49% I-RAVEN test (68.46% end to end); 46% on I-RAVEN-X 3×3 |
| `notebooks/`, `thesis_html/` | Its walkthrough, end-to-end and pixels-to-answer notebooks and their executed copies; the porting-stage report "Expressiveness Is Not Trainability" | |
| `torch_gnn/`, `scripts/` | The original object-level torch GNN (`train_object_vsa_gnn_frontend.py`, `eval_gnn_test.py`, graph drawing, jobs) | 45.3 ± 12.4% I-RAVEN test after the position fixes (3 seeds) |
| `tables/` | LaTeX tables of those two models; the 68.81% / 68.83% I-RAVEN numbers in them are invalid (§10 of the old notes) | |

`pynl/pixel_pipeline.py` lost its `solve` stage and `pynl/sanity_checks.py` its `model` check, both of which
ran the GNN-style model; the early hub and ordered templates were removed from `pynl/bench_hub_grounding.py`.
