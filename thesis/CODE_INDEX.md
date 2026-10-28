# Code index

Where the code behind the current models lives and how it connects. Scripts and job files are in `pynl/`
(PyNeuraLogic) and `baselines/` (torch), the notebook in `notebooks/`, the ResNet-18 frontend in `frontend/`;
run everything from the repository root. Superseded code is in `archive/` (bottom of this page). Findings:
`thesis/FINDINGS.md`; plan: `thesis/THESIS_PLAN_symmetry.md`; lab notes: `thesis/PYNEURALOGIC_NOTES.md`.

## Reading order

1. `pynl/bench_hub_grounding.py`: the generative facet model, the main model (notes §17–§18, §20–§24).
2. `pynl/facet_symmetry_check.py` and `pynl/dataset_symmetry_check.py`: the symmetries of the model and of
   the data (§21–§22).
3. `baselines/train_baseline.py`: the torch baselines on the same task (§23).
4. `pynl/pynl_symbolic.py`: the symbolic models, the "rules given" level (§15).

## The generative facet model (ground-truth attributes)

| file | what it is | where to look |
|---|---|---|
| `bench_hub_grounding.py` | Generates the missing panel's description on a graph grounded differently for every puzzle, then picks the closest candidate. | row length and code length: `configure` · two-part layouts: `COMPONENTS` · column roles: `col_names` · value codes and targets: `code`, `facet_target`, `facet_decode` · template: `build_facet_template` · facts: `facet_facts`, `puzzle_facts` · data: `load`, `load_iravenx` · generate-then-match metrics: `evaluate` · training and selection: `main` |
| `train_pynl.py` | The I-RAVEN XML parser and position map the facet model imports (`parse`, `POS_MAP`). It also still holds the first pure-PyNeuraLogic center_single trainer. | |
| `facet_model.ipynb` | Loads a facet checkpoint (`template_best.bin`, `best.json`) and shows one puzzle: the grounded network, printed rules, generated vs true answer, candidate distances, accuracy on val 300–399. Section 6: copy links on I-RAVEN-X 3×10 next to the plain model, with the trained gates of one puzzle and exact generation per rule. Executed copy: `thesis/facet_model.html`. | |
| `facet_viz.py` | Drawings for the notebook. `ground_graph` reads the network PyNeuraLogic grounds for one sample (built without chain pruning) as atom-to-atom edges; `draw_network` puts one attribute of it on the puzzle grid; `gate_means` reads a sample's trained copy gates. | |
| `facet_symmetry_check.py` | Which symmetries each template variant has, on random weights and (`--trained`) on checkpoints (§21, §24). | |
| `dataset_symmetry_check.py` | Which symmetries the I-RAVEN and I-RAVEN-X data have: which rules survive each transformation, the generators' statistics, the structure of the candidate sets (§22). | |
| `facet_rule_eval.py` | Rebuilds any I-RAVEN-X checkpoint from its `best.json`; matched, exact and exact value per rule on the unseen puzzles, optionally with rows 1 ↔ 2 or middle columns 2 ↔ 3 swapped. | |
| `submit_facet_long.sh`, `submit_column_roles.sh` | 150-epoch runs; the column-role runs (§24). Other runs used `sbatch --wrap` (command in notes §18.1). | |

Options of `bench_hub_grounding.py --template facet`:
* `--widths 32,64,128` widens the layers; `--iravenx FILE` trains on I-RAVEN-X; `--save-dir` saves the best
  checkpoint (`template_best.bin` + `best.json`).
* `--copy-links` adds gated copy links from every context panel's code to the output (§20).
  * `--copy-gates content` chooses gates by content and row role, which keeps rows 1 and 2 interchangeable (§21).
  * `--copy-shuffled` is the control: the copied codes come from another puzzle.
* `--columns roles`: first / middle / last columns, so the operands of Arithmetic are interchangeable (§24).
* Ablations: `--values onehot`, `--per-attr-weights`, `--no-rows`; `--visible-selectors` restores the bug of §17.

## Torch baselines (`baselines/`, notes §23)

| file | what it is |
|---|---|
| `prepare_baseline_data.py` | pynl env: writes the facet model's puzzles, splits and weight counts as arrays (`experiments/baselines/data/<dataset>.npz`), plus `bench_hub_grounding`'s own targets for a check. |
| `train_baseline.py` | nvsa env, GPU: on the facet model's task (same targets, selection and metrics), sized to the facet model's weights (`--scale` for more). Architectures: `mlp`, `gnn` (all-to-all over objects), `attr-abs` and `attr-rel` (all-to-all over object attributes, absolute or relative labels: plan models A and B). Regularization: `--norm`, `--dropout`, `--drop-edge`, `--wd`, `--schedule cosine`, `--warmup`, `--clip`, `--augment rows`. `--check-symmetry`, and transformed test sets after training or with `--eval-only`. |
| `submit_baselines.sh`, `submit_sweep.sh`, `submit_regularized.sh`, `submit_attr_gnn.sh` | equal weights (2 seeds); 4×, 16×, 64×; the regularization grid; models A and B. |
| `summarize_baselines.py` | the table, next to the facet model; `--per-rule`. |

## The symbolic models (ground-truth attributes, notes §15)

| file | what it is |
|---|---|
| `pynl_symbolic.py` | Attribute values, rows and slot sets as symbols; crisp relations scored with learned weights. `--model prims` (generic building blocks) or `--model rules` (RAVEN's rule types). |
| `rpm_symbolic.py` | Puzzle reader for the above: numbers, slot sets and per-slot attributes per component. |
| `rpm_reference_solvers.py` | The same two vocabularies as plain, non-learned Python solvers. |
| `pynl_symbolic_e2e.py` | Scores a trained symbolic model end to end (ResNet perception → symbolic facts) and with XML labels. |
| `submit_symbolic.sh`, `submit_symbolic_e2e.sh`, `submit_symbolic_extra.sh` | training grid; test split end to end; learning curves and random initial weights. |

## Perception and data checks

| file | what it is |
|---|---|
| `frontend/train_frontend.py`, `predict_frontend.py`, `extract_objects.py` | The ResNet-18 perception frontend (torch, trained on RAVEN). Checkpoints: `log/universal_frontend_correct_proportional_mean1`, `_mean3`. |
| `pynl/pixel_pipeline.py`, `submit_pixels.sh` | `perceive` (env `nvsa`, GPU): runs the frontend over every I-RAVEN panel into `experiments/frontend_preds/`; `panels_from_preds` decodes them for the end-to-end runs. |
| `torch_gnn/gnn_solver.py` | Kept for `load_frontend`, which `pixel_pipeline.py` uses. |
| `gnn/` | The inherited data package: `gnn.data.canonical_positions` (the slot fix of §10) and `CONFIG_POS_MAP`, used by the symbolic reader, the frontend and the data checks. |
| `pynl/sanity_checks.py` | `splits` (no train/test duplicates), `shortcut` (candidates-only baseline), `images` (XML labels match the pixels). Each check has a negative control. |

## How the files connect

```
bench_hub_grounding.py ──imports──> train_pynl.py (parse, POS_MAP)
facet_symmetry_check.py, facet_rule_eval.py, facet_viz.py, prepare_baseline_data.py ──import──> bench_hub_grounding.py
pynl_symbolic.py, rpm_reference_solvers.py ──import──> rpm_symbolic.py ──> gnn/data.py
pynl_symbolic_e2e.py ──imports──> pixel_pipeline.py (panels_from_preds) ──> torch_gnn/gnn_solver.py, frontend/
```

## Archive

- `archive/superseded_2026-10-28/`: the GNN-style PyNeuraLogic model (`train_all_constellations.py` and its
  jobs, evaluation and notebooks), the original torch GNN (`torch_gnn/` training and evaluation scripts,
  `scripts/` GNN jobs) and stale LaTeX tables. See its `MANIFEST.md`.
- `archive/pyneuralogic_experiments/`: the porting-stage PyNeuraLogic models (pure, softmax, aligned, hybrid,
  ladder).
- `archive/pyneuralogic_2026-03-31_to_05-06/`: the April 2026 spatial PyNeuraLogic GNN.
