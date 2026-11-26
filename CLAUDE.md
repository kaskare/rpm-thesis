# I-RAVEN and I-RAVEN-X: symmetries as inductive bias in PyNeuraLogic models (thesis)

Findings in order: `docs/FINDINGS.md`. All result tables: `docs/RESULTS.md` (update it with every new
result); what each symmetry buys, by model and dataset: `docs/SYMMETRY_RESULTS.md` (update it too); per rule
with and without Arithmetic: `docs/ARITHMETIC.md`. Plan: `docs/THESIS_PLAN_symmetry.md`. Lab notes:
`docs/PYNEURALOGIC_NOTES.md`. Code map: `docs/CODE_INDEX.md`. Project summary for Claude projects:
`docs/claude_project/PROJECT_KNOWLEDGE.md`.

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
Layout (since 2026-11-08):
- `core/`: the models, their training and their scoring, nothing else. `core/pynl/` (PyNeuraLogic),
  `core/baselines/` (torch), `core/frontend/` (ResNet-18 perception), `core/gnn/` (the inherited I-RAVEN data
  package, slot fix). Nothing in `core/` imports from outside it, except IBM's `raven/` and `torch_gnn/`.
- `analysis/`: checks, inspections and drawings that read models or data (symmetry checks, viz, sanity checks).
- `data_prep/`: data generation and conversion (rule twins, Arithmetic-only sets, arrays and images for torch).
- `jobs/pynl/`, `jobs/baselines/`, `jobs/frontend/`: SLURM job scripts only.
- `notebooks/`: all notebooks. `docs/`: all written output (results, findings, notes, plan, executed notebook
  pages, figures); `docs/private/` is git-excluded.
- `red_team/`: one critical review per markdown file in `docs/`.

Scripts run from the repo root (`python3 core/pynl/bench_hub_grounding.py ...`), so `data/` and `experiments/`
paths are unchanged. Scripts outside `core/` put the `core/` folder they need on `sys.path`, so imports stay bare
(`import bench_hub_grounding as B`).
- `core/pynl/bench_hub_grounding.py --template facet`: the main model, a generative facet model on per-puzzle
  graphs (copy links, content gates, column roles, row fold); `core/pynl/facet_rule_eval.py` scores
  checkpoints, `core/pynl/facet_e2e_eval.py` scores them end to end on the ViT recognizer's panels
  (`jobs/pynl/submit_facet_e2e.sh`), `core/pynl/facet_symmetry_check.py` (the transforms), `core/pynl/facet_stream.py` (streaming
  with weight decay); jobs `jobs/pynl/submit_facet_long.sh`, `jobs/pynl/submit_column_roles.sh`.
- `core/baselines/train_baseline.py`: torch MLP, all-to-all object GNN, attribute-level GNNs (A/B) and ViT on
  the facet model's task; data from `data_prep/prepare_baseline_data.py`. `core/baselines/test_eval.py` scores a
  checkpoint once on the test split (`test.json`; the main-table baselines and the plain facet checkpoint have had
  their pass, RESULTS §1).
- `core/pynl/pynl_symbolic.py` + `rpm_symbolic.py`: symbolic models; `rpm_reference_solvers.py`: non-learned
  references; `pynl_symbolic_e2e.py`: end to end.
- `core/pynl/pixel_pipeline.py` + `jobs/pynl/submit_pixels.sh`: the ResNet-18 frontend over every panel
  (`experiments/frontend_preds/`); `core/frontend/`: the frontend itself.
- `core/pynl/value_axis.py`: the value-axis model (values as constants, a convolution along them; notes §27);
  drawings `analysis/value_axis_viz.py`; jobs `jobs/pynl/submit_value_axis.sh`. On the rule twins (sum, midpoint;
  `jobs/pynl/submit_rule_twins.sh`, `analysis/twins_seeds.py`, `analysis/rule_orbits.py`): `--seed`, `--select exact`,
  `--scale step|pow2` for rows read in their own unit (`docs/ARITHMETIC.md`, "The scaling kernel"), and `--gate`
  ("The rule gate"; `analysis/twins_probe.py` reads a checkpoint's row descriptor) and `--steps` ("The row
  descriptor reads the steps"). Proofs for the twins in plain words: `docs/thesis/ch3_pricing_proofs.md`, checked by
  `analysis/pricing_checks.py`.
- `notebooks/facet_model.ipynb` and `notebooks/value_axis.ipynb` (they `chdir` to the repo root); executed
  copies `docs/facet_model.html`, `docs/value_axis.html`. `notebooks/symmetry_briefing.ipynb` goes with the
  supervisor briefing page (`docs/symmetry_briefing.html`).
- `llm/`: LLM benchmark on I-RAVEN (images end to end, or text with XML labels); see `llm/README.md`.
- Superseded code is in `archive/` (the GNN-style PyNeuraLogic model and the original torch GNN in
  `archive/superseded_2026-10-28/`); not runnable from there as is.
