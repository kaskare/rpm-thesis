# PyNeuraLogic code index

Where the code behind the current PyNeuraLogic models lives and how it connects. Scripts and job
files below are in `pynl/`, notebooks in `notebooks/`, the ResNet-18 frontend in `frontend/` and the
torch GNN in `torch_gnn/`; run everything from the repository root. Line numbers are as of 2026-10-19.

## Reading order

1. `train_all_constellations.py` — the GNN-style model (best result).
2. `pynl_symbolic.py` — the symbolic models.
3. `archive/pyneuralogic_2026-03-31_to_05-06/scripts/raven_pyneuralogic.py` — your April model, for comparison.

## Files

### The GNN-style model (64.00% I-RAVEN test)

| file | what it is | where to look |
|---|---|---|
| `train_all_constellations.py` | One model for all 7 constellations: torch embeddings feed a PyNeuraLogic message-passing template, and torch computes the loss and steps its optimiser. | graph: `edges` :56, `structural_facts` :75 · template: `build_template` :82 (3 layers :90, head :101) · torch side + bridge: `AllConstellations` :121 · CONCAT order probe: `_recover_order` :149 · node embedding: `forward` :175, `encode_puzzle` :191 · training loop: `main` :227 (loss :364) |
| `train_pynl.py` | XML parser and position map used by the model above (`parse` :78, `POS_MAP` :19). It also holds the earlier pure-PyNeuraLogic center_single trainer. | |
| `submit_max.sh` | SLURM chain that trains it: `ARM=conv` (2000/constellation) or `ARM=data` (6000), dim 128, 3 layers, `--no-same-panel`. | |
| `submit_ablations.sh` | Five 12-epoch arms at 2000/constellation: control, `--edge-labels rowcol`, `--per-type-weights` (with within-panel edges), `--absent-pos`, all three. These switches in `train_all_constellations.py` are off by default. | |
| `eval_pynl_test.py` | Test-split evaluation per constellation, next to the 3-seed torch GNN means. | `load_model` :48 |

Run: `experiments/max_data_L2`. Best validation is 68.52% (epoch 78); the 64.00% test figure is from the epoch-62 checkpoint.

The torch ↔ PyNeuraLogic bridge is library code, not ours:
`~/.conda/envs/pynl/lib/python3.11/site-packages/neuralogic/nn/torch_function.py`.

### From pixels (63.96% test)

| file | what it is |
|---|---|
| `pixel_pipeline.py` | `perceive` (env `nvsa`, GPU): the ResNet-18 frontend reads attributes from every panel. `solve` (env `pynl`): the model above answers from those predictions. |
| `submit_pixels.sh` | `STAGE=perceive` or `STAGE=solve`. |

The frontend itself is torch-only: `frontend/train_frontend.py` and `frontend/predict_frontend.py`.

### The symbolic models (ground-truth attributes)

| file | what it is | where to look |
|---|---|---|
| `pynl_symbolic.py` | Attribute values, rows and slot sets as symbols. Crisp relations (equal, successor ±1/±2, sum/difference, set union/difference/shift) are scored with learned weights. `--model prims` uses generic building blocks: 97.58% val. `--model rules` uses RAVEN's rule types: 98.44% val. | facts: `puzzle_tuples` :64, `background_tuples` :110 · template: `build_template` :248 · CONCAT order: `probe_sample` :385, `read_order` :393 · Python cross-check of the template: `python_counts` :131, `python_rule_counts` :209 · training: `main` :435 |
| `rpm_symbolic.py` | Puzzle reader for the above: numbers, slot sets and per-slot attributes per component. | `read_puzzle` :41 |
| `rpm_reference_solvers.py` | The same two vocabularies as plain, non-learned Python solvers: 96.34% and 97.75% val. | `solve_primitives` :109, `solve_rules` :171 |
| `submit_symbolic.sh` | Array job: 100/300 puzzles per constellation × lr 0.01/0.003, `MODEL=prims` or `MODEL=rules`. | |
| `pynl_symbolic_e2e.py` | Scores a trained symbolic model end to end (ResNet perception → symbolic facts) and with XML labels, next to the non-learned reference solvers. Verifies its converter against `read_puzzle`. | `to_symbolic`, `template_scores` |
| `submit_symbolic_e2e.sh` | Test-split run of the above for both vocabularies. | |
| `submit_symbolic_extra.sh` | Learning curves (10, 30, 1000 puzzles/constellation) and random initial weights (`--init-seed`, 3 seeds). | |

### Checks and walkthroughs

| file | what it is |
|---|---|
| `sanity_checks.py` | `splits` (no train/test duplicates), `shortcut` (candidates-only baseline), `images` (XML labels match the pixels), `model` (candidate-order invariance; foreign/blank context). Each check has a negative control. |
| `pynl_best_architecture_walkthrough.ipynb` | Step-by-step walkthrough of the GNN-style model. Executed copy: `thesis/pynl_best_architecture_walkthrough.html`. |
| `pynl_iraven_solver.ipynb` | End to end with few comments: puzzle images, the ResNet-18 perception, then the reasoner defined inline, through to accuracy end to end and with XML labels. Includes an exact-match check against `train_all_constellations.py`. Executed copy: `thesis/pynl_iraven_solver.html`. |
| `pixels_to_answer.ipynb` | One puzzle from images to answer. Executed copy: `thesis/pixels_to_answer.html`. |

### `archive/pyneuralogic_2026-03-31_to_05-06/scripts/raven_pyneuralogic.py`

The April model, kept for comparison: pure PyNeuraLogic, one binary sample per candidate, objects as nodes. It scored near chance (9–19.5% test per constellation). Template at :405, dataset at :364, evaluation at :455.

## How the files connect

```
train_all_constellations.py ──imports──> train_pynl.py (parse, POS_MAP)
eval_pynl_test.py, sanity_checks.py, pixel_pipeline.py ──import──> train_all_constellations.py
pynl_symbolic.py, rpm_reference_solvers.py ──import──> rpm_symbolic.py ──> gnn/data.py
```

## Earlier steps

- `archive/pyneuralogic_experiments/`: `train_pynl_softmax.py`, `train_pynl_torch.py`, `train_pynl_aligned.py`, `train_ladder.py` and their job scripts: the porting steps and the 4-level ablation behind the hybrid design (notes §4–§8).
- Torch GNN baseline: `torch_gnn/train_object_vsa_gnn_frontend.py` and the `gnn/` package.

The full story, numbers and bugs are in `thesis/PYNEURALOGIC_NOTES.md`.
