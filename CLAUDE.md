# I-RAVEN and I-RAVEN-X: symmetries as inductive bias in PyNeuraLogic models (thesis)

Findings in order: `thesis/FINDINGS.md`. Plan: `thesis/THESIS_PLAN_symmetry.md`. Lab notes:
`thesis/PYNEURALOGIC_NOTES.md`. Code map: `thesis/CODE_INDEX.md`. Project summary for Claude projects:
`thesis/claude_project/PROJECT_KNOWLEDGE.md`.

## Environment
- PyNeuraLogic work: conda env `pynl` (Python 3.11, neuralogic 0.9.2, CPU/JVM only, never a GPU).
  ```
  source /mnt/appl/software/Anaconda3/2024.02-1/etc/profile.d/conda.sh && conda activate pynl
  export PYTHONNOUSERSITE=1 JAVA_HOME=/mnt/appl/software/Java/21.0.8 && export PATH=$JAVA_HOME/bin:$PATH
  ```
  `ml load Java/21` silently fails on amd compute nodes; always pin JAVA_HOME.
- Torch baselines and GPU perception: conda env `nvsa` (Python 3.7, torch 1.11). Do not upgrade it.
- SLURM: `amd` / `amdlong` / `amdfast` (CPU), `amdgpu` / `amdgpufast` (GPU). Compute nodes cannot see the
  session's /tmp scratch.

## Rules
- Never shut down the shared login node. Plain scripts are fine there; CPU-heavy Jupyter kernels are
  SIGKILLed. Execute notebooks with `sbatch ... jupyter nbconvert --execute --inplace`.
- Native PyNeuraLogic training (`model.build(Settings(...))`) must pass `iso_value_compression=False`.
- Facet checkpoints are `template_best.bin` + `best.json` (the template is rebuilt from its args); torch
  baselines save `model_best.pt` + `best.json`. Copy them aside before reading from a directory a live job
  writes to.
- Selection puzzles (I-RAVEN val 0–299 per constellation, I-RAVEN-X 6000–6299) pick the epoch; report on
  unseen puzzles (I-RAVEN val 300+, I-RAVEN-X 6300+). Score the test split once per final checkpoint.
- `origin` is the user's private repo `kaskare/rpm-thesis` (pushed from RCI with a write deploy key,
  SSH host alias `github-rpm`). `ibm` is IBM's public upstream: never push or open PRs there.
  Data, experiments and logs are far too large for git.
- The user prefers "end to end" over "from pixels", and compact notebooks.

## Key code
Scripts live in folders but run from the repo root (`python3 pynl/bench_hub_grounding.py ...`), so `data/`
and `experiments/` paths are unchanged. Python puts a script's own folder on `sys.path` and scripts add the
root, so imports stay bare (`from train_pynl import parse`).
- `pynl/bench_hub_grounding.py --template facet`: the main model, a generative facet model on per-puzzle
  graphs (copy links, content gates, column roles); `facet_symmetry_check.py`, `dataset_symmetry_check.py`,
  `facet_rule_eval.py`, `facet_viz.py`; jobs `submit_facet_long.sh`, `submit_column_roles.sh`.
- `baselines/`: torch MLP, all-to-all object GNN and attribute-level GNNs (A/B) on the facet model's task.
- `pynl/pynl_symbolic.py` + `rpm_symbolic.py`: symbolic models; `rpm_reference_solvers.py`: non-learned
  references; `pynl_symbolic_e2e.py`: end to end.
- `pynl/pixel_pipeline.py` + `submit_pixels.sh`: the ResNet-18 frontend over every panel
  (`experiments/frontend_preds/`); `frontend/`: the frontend itself; `gnn/`: the inherited data package
  (slot fix), still imported.
- `notebooks/facet_model.ipynb` (it `chdir`s to the repo root); executed copy `thesis/facet_model.html`.
- `llm/`: LLM benchmark on I-RAVEN (images end to end, or text with XML labels); see `llm/README.md`.
- Superseded code is in `archive/` (the GNN-style PyNeuraLogic model and the original torch GNN in
  `archive/superseded_2026-10-28/`); not runnable from there as is.
