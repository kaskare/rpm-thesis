# I-RAVEN: PyNeuraLogic vs torch GNN (thesis)

Full project state: `thesis/claude_project/PROJECT_KNOWLEDGE.md`. Lab notes: `thesis/PYNEURALOGIC_NOTES.md`.

## Environment
- PyNeuraLogic work: conda env `pynl` (Python 3.11, neuralogic 0.9.2, CPU/JVM only, never a GPU).
  ```
  source /mnt/appl/software/Anaconda3/2024.02-1/etc/profile.d/conda.sh && conda activate pynl
  export PYTHONNOUSERSITE=1 JAVA_HOME=/mnt/appl/software/Java/21.0.8 && export PATH=$JAVA_HOME/bin:$PATH
  ```
  `ml load Java/21` silently fails on amd compute nodes; always pin JAVA_HOME.
- Torch GNN and GPU perception: conda env `nvsa` (Python 3.7). Do not upgrade it.
- SLURM: `amdfast` (CPU), `amdgpufast` (GPU). Compute nodes cannot see the session's /tmp scratch.

## Rules
- Never shut down the shared login node. Plain scripts are fine there; CPU-heavy Jupyter kernels are
  SIGKILLed. Execute notebooks with `sbatch ... jupyter nbconvert --execute --inplace`.
- Native PyNeuraLogic training (`model.build(Settings(...))`) must pass `iso_value_compression=False`.
- Checkpoints of the GNN-style model are two files: `model_best.pt` + `template_best.bin`. Copy them
  aside before reading from a directory a live job writes to.
- Validation puzzles 0–299 per constellation are used for checkpoint selection; use 300+ for demos.
  Score the test split once per final checkpoint.
- `origin` is the user's private repo `kaskare/rpm-thesis` (pushed from RCI with a write deploy key,
  SSH host alias `github-rpm`). `ibm` is IBM's public upstream: never push or open PRs there.
  Data, experiments and logs are far too large for git.
- The user prefers "end to end" over "from pixels", and compact notebooks.

## Key code
- `train_all_constellations.py`: best model (graph, template, torch bridge, training); run
  `experiments/max_data_L2` via `submit_max.sh`.
- `pynl_symbolic.py` + `rpm_symbolic.py`: symbolic models; `rpm_reference_solvers.py`: non-learned references.
- `pixel_pipeline.py` + `submit_pixels.sh`: end to end through the ResNet-18 frontend.
- `eval_pynl_test.py`, `sanity_checks.py`; notebook `pynl_iraven_solver.ipynb`.
- `llm/`: LLM benchmark on I-RAVEN (images end to end, or text with XML labels); see `llm/README.md`.
- `thesis/CODE_INDEX.md`: where each part of the PyNeuraLogic code lives. Superseded experiment
  scripts are in `archive/pyneuralogic_experiments/` (not runnable from there as is).
