# Code index

Where the code behind the current models lives and how it connects. Since 2026-11-08: the models, their training
and scoring in `core/` (`core/pynl/` PyNeuraLogic, `core/baselines/` torch, `core/frontend/` the ResNet-18 frontend,
`core/gnn/` the I-RAVEN data package); checks, inspections and drawings in `analysis/`; data generation in
`data_prep/`; SLURM job scripts in `jobs/pynl/` and `jobs/baselines/`; notebooks in `notebooks/`; written output in
`docs/`. File names in the tables below are bare; their folder follows that split. Run everything from the
repository root. Superseded code is in `archive/` (bottom of this page). Findings:
`docs/FINDINGS.md`; plan: `docs/THESIS_PLAN_symmetry.md`; lab notes: `docs/PYNEURALOGIC_NOTES.md`.

## Reading order

1. `core/pynl/bench_hub_grounding.py`: the generative facet model, the main model (notes §17–§18, §20–§24).
2. `core/pynl/facet_symmetry_check.py` and `analysis/dataset_symmetry_check.py`: the symmetries of the model and of
   the data (§21–§22).
3. `core/baselines/train_baseline.py`: the torch baselines on the same task (§23).
4. `core/pynl/pynl_symbolic.py`: the symbolic models, the "rules given" level (§15).

## The generative facet model (ground-truth attributes)

| file | what it is | where to look |
|---|---|---|
| `bench_hub_grounding.py` | Generates the missing panel's description on a graph grounded differently for every puzzle, then picks the closest candidate. | row length and code length: `configure` · two-part layouts: `COMPONENTS` · column roles: `col_names` · value codes and targets: `code`, `facet_target`, `facet_decode` · template: `build_facet_template` · facts: `facet_facts`, `puzzle_facts` · data: `load`, `load_iravenx` · generate-then-match metrics: `evaluate` · training and selection: `main` |
| `train_pynl.py` | The I-RAVEN XML parser and position map the facet model imports (`parse`, `POS_MAP`). It also still holds the first pure-PyNeuraLogic center_single trainer. | |
| `facet_model.ipynb` | Loads a facet checkpoint (`template_best.bin`, `best.json`) and shows one puzzle: the grounded network, printed rules, generated vs true answer, candidate distances, accuracy on val 300–399. Section 6: copy links on I-RAVEN-X 3×10 next to the plain model, with the trained gates of one puzzle and exact generation per rule. Executed copy: `docs/facet_model.html`. | |
| `facet_viz.py` | Drawings for the notebook. `ground_graph` reads the network PyNeuraLogic grounds for one sample (built without chain pruning) as atom-to-atom edges; `draw_network` puts one attribute of it on the puzzle grid; `gate_means` reads a sample's trained copy gates. | |
| `facet_symmetry_check.py` | Which symmetries each template variant has, on random weights and (`--trained`) on checkpoints (§21, §24). | |
| `dataset_symmetry_check.py` | Which symmetries the I-RAVEN and I-RAVEN-X data have: which rules survive each transformation, the generators' statistics, the structure of the candidate sets (§22). | |
| `facet_rule_eval.py` | Rebuilds any I-RAVEN-X checkpoint from its `best.json`; matched, exact and exact value per rule on the unseen puzzles, optionally transformed (rows 1 ↔ 2, middle columns 2 ↔ 3, size ↔ colour, mirrored layouts, rows ↔ columns of a 3×3 context with `transpose`), or (`--iravenx FILE`) on rows of another length. Also I-RAVEN checkpoints (per constellation, and per rule from the XML). Job `submit_rescore_exact.sh` rescores a list of saved checkpoints (used after the exact-scorer fix of 2026-11-08); `submit_transpose.sh` scores the I-RAVEN checkpoints and the reference solvers on transposed puzzles. `--test` scores a checkpoint once on the test range (I-RAVEN-X puzzles 8000.., I-RAVEN test 0..) and writes `test.json`; job `submit_test_pass.sh` runs the test pass of Chapters 6 and 7 over the lists in `experiments/test_pass/` (facet and value-axis checkpoints, the latter through `value_axis.py --eval --eval-start 8000`), and `analysis/test_pass_summary.py` sets each checkpoint's unseen score next to its test score. | |
| `facet_e2e_eval.py` | The facet model end to end on I-RAVEN: context panels and candidates from the ViT slot recognizer's predictions (`experiments/recognizer/vit64/preds_<split>.npy`) instead of the XML labels. Three passes: labels (must reproduce the checkpoint's record), hard (argmax), soft (thermometer bit k = P(value > k) where the recognizer is unsure). Matched on perceived and on label candidates, exact against the true answer, per layout, split by whether all 16 panels were read right. Writes `<ckpt>/e2e_<split>.json`. Job `jobs/pynl/submit_facet_e2e.sh` (`SPLIT=test` once per final checkpoint). |
| `submit_facet_long.sh`, `submit_column_roles.sh` | 150-epoch runs; the column-role and row-fold runs (§24–§25). Other runs used `sbatch --wrap` (command in notes §18.1). | |
| `value_axis.py` | The value-axis model (notes §27): values as constants, a kernel over offsets ±2, pair comparisons pooled over values, exact sums through `plus`, scoring per value with shared weights; `--no-plus`, `--per-value` (the value-symmetry ablation), `--learned-plus` (the `plus` facts used by learned rules, no built-in sums or checks; `--state-next` also moves the state along `next`), `--peano` (`plus` derived from the zero and `next` by two rules instead of the table), `--check-shift`, `--eval DIR` (another range or row length), `--seed` (seeds the backend: initial weights and sample order), `--select matched|exact` (what picks the saved epoch), `--scale step|pow2` (with `--ring --no-plus`: the kernel's offsets counted in each row's own unit, its first step or the largest power of two in it, so the model is also equivariant to scaling the values, or to doubling; `docs/ARITHMETIC.md`, "The scaling kernel"), `--check-scale` (random weights: values multiplied and shifted on the ring), `--gate` (the rule descriptor gates the question row's kernel features in the answer, a sigmoid per feature; "The rule gate"). | template: `build_value_template` · facts: `value_facts` · scoring: `to_codes`, `score` `--steps`: the row descriptor also reads each neighbouring pair's step directly, the first panel's kernel feature at the second panel's value (`docs/ARITHMETIC.md`, "The row descriptor reads the steps"). |
| `arith_op_data.py` | I-RAVEN-X 3×10 kept to puzzles whose Arithmetic is only plus or only minus (`data/iravenx_ops/`). Job `submit_arith_op.sh`: column roles on each, and the two-operator checkpoint scored on the same puzzles. |
| `arith_big_data.py`, `facet_stream.py` | The grokking test (`docs/ARITHMETIC.md`): 100,000 I-RAVEN-X 3×10 puzzles, every attribute Arithmetic (`data/iravenx_arith/`), and the facet model trained on them, grounding chunks at a time, with decoupled weight decay in Python. Jobs `submit_arith_big.sh` and `jobs/baselines/submit_arith_big.sh`; `jobs/baselines/submit_grok_small.sh` trains GNN B on the first N of them in the grokking papers' setting. |
| `rule_twins.py` | I-RAVEN-X puzzles with Arithmetic swapped for a test rule (sum, midpoint, mirror, lookup), the symmetry checks and the untrained references (`docs/ARITHMETIC.md`). Jobs `submit_rule_twins.sh`, `jobs/baselines/submit_twins.sh`. `value_axis.py --ring --kernel --pair` is the shift-equivariant reader of both operands, `--eval DIR --dump FILE` writes its generated values, and `bench_hub_grounding.py --select exact` picks the epoch by exact. |
| `train_vit_recognizer.py`, `prepare_recognizer_data.py` | The perception stage for the facet model end to end: the black-box ViT's architecture reading one panel, one query token per universal slot and one shared head (occupancy, type, size, colour); `predict` writes per-slot probabilities for val 300.. and test. Data: every panel of 2000 training puzzles per layout and of the evaluation splits, labelled from the XML by the facet model's parser. Job `jobs/frontend/submit_recognizer.sh` (prep, train, predict). `--per-layout N` trains on the first N puzzles of every layout, for the learning curve end to end (`PER_LAYOUT`, `PREDS` in `jobs/pynl/submit_facet_e2e.sh`). |
| `submit_wrong_symmetry.sh` | Wrong-symmetry controls on I-RAVEN-X 3×3: `bench_hub_grounding.py --columns tie12` (columns 1 and 2 tied, out of support) and `--fold-values` (v and maxval − v coded alike, conflicting); `facet_rule_eval.py` transforms `cols12` and `reflect` check them. |
| `rule_orbits.py` | Sum against midpoint with no network: learners that know only a symmetry (none, every panel +s, one operand with the answer, shift with doubling, shift with every scaling), fit by majority vote per orbit on the rule-twin puzzles; `--complete` also fits on the complete rows (`docs/ARITHMETIC.md`, "Sum and midpoint compared"). Job `submit_twins_dump.sh` dumps trained twin checkpoints' answers for it. |
| `twins_figure.py` | Thesis figure: sum and midpoint under the convolution, with and without `plus`, weights per value and the counting learner, the ceiling drawn in (numbers from `docs/ARITHMETIC.md`, hard-coded; 100 and 600 puzzles are means of three seeds with one sd). Writes `docs/figures/twins_sum_midpoint.{png,pdf}`; base Anaconda's Python (pynl has no matplotlib). |
| `twins_seeds.py` | The rule twins over seeds (`docs/ARITHMETIC.md`, "Three seeds" and "The scaling kernel"): per seed, mean and sd for the shared, per-value, `plus`, doubled and scaled models on sum and midpoint at 100 and 600 puzzles, epoch picked by whole-puzzle exact on selection, read from each run's `history.json`. `--dumps experiments/twins_eval` splits the unseen midpoint rows by what training covered: the pair (per value), or the first step and, for a new first step, its odd part (models with the shift); the rows that start like a Progression row in the model's unit; for the scaled models, rows by whether the question row shares a value with a complete row; and the rows by how many training question rows fall in their orbit. Also the gated models (`sharedgate`, `scaledgate`) and those whose row descriptor reads the steps (`sharedsteps`, `doubledsteps`, `scaledsteps`). Runs: `jobs/pynl/submit_rule_twins.sh` with `SEED` and `SELECT=exact`; dumps: `jobs/pynl/submit_twins_dump.sh`. |
| `twins_probe.py` | Does a trained value-axis model's row descriptor carry the row's rule (`docs/ARITHMETIC.md`, "The rule gate")? Loads saved rule-twin checkpoints, queries `rd("r1", A)` and `ctx2(A)` on unseen midpoint puzzles and prints the largest difference per feature between attributes under midpoint and under Progression. pynl env; about a minute a run. |
| `pricing_checks.py` | The counts behind `docs/thesis/ch3_pricing_proofs.md`, nothing trained: which twin tables keep which move; the expected share of unseen rows whose orbit n training rows reach, from the table alone, next to the counting learner of `rule_orbits.py`; the ceilings; and a puzzle-wide, shift-equivariant function that solves sum. numpy-free. |
| `twins_linear_fit.py` | The rule twins with values read as numbers and no training (`docs/ARITHMETIC.md`, "Two checks without training"): fits c = p·a + q·b on each puzzle's two complete rows and applies it to the question row; exact % per rule on the unseen puzzles. numpy only. |
| `submit_grok_fold.sh` | The row fold in the grokking papers' setting on Arithmetic-only 3×10 puzzles: `facet_stream.py --columns fold` with small training sets (grounded once, `--n` ≤ `--chunk`), decoupled weight decay and `--init-scale` (`docs/ARITHMETIC.md`, "Does it grok?"). |
| `value_axis_next_arithmetic.ipynb` | The value axis with `plus` used by learned rules and `next` steps on Arithmetic: its rules, training on 3×3 from scratch, one puzzle walked through (the values the state reaches, P per value), the trained 3×10 checkpoint, every learned-plus run. Executed copy: `docs/value_axis_next_arithmetic.html`; job `submit_notebook.sh` executes any notebook. |
| `autotopism_check.py` | brute force on Z_5 and Z_7: the triples of maps that keep addition and midpoint (`docs/READING_LIST_arithmetic_symmetry.md`) |
| `learned_plus_check.py` | Inside a `value_axis.py --learned-plus` checkpoint, on unseen puzzles: the values the row state reaches (no model), the state's length and direction at each, per path (adding, taking away, `next`), and P(answer); `--random` is the untrained control (`docs/ARITHMETIC.md`, "Numbers as objects"). | `reach` |
| `value_axis_viz.py` | Drawings for the value-axis notebook: `draw_value_graph` places every ground atom of one attribute on the value axis (lanes = panels), `draw_kernel` the learned kernel. Uses `facet_viz.ground_graph(..., out_len=153)`. | |
| `symmetry_briefing.ipynb` | Goes with the supervisor briefing page, in its eight sections: every table read from the run directories (`best.json`, `history.json`, job logs), the symmetry check of §21 run live on random weights, the row fold (its rules, weights at rows of 5 / 10 / 20, symmetries on 3×10, row-length scores from the logs, a recursive toy sum trained in PyNeuraLogic), the data and rule symmetry counts, the Arithmetic algebra (the shift test, the convolution's limit, the moves (s, t, s + t), sums in thermometer codes), the grokking curves. Runs in about a minute. Executed copy: `docs/symmetry_briefing.html`. |
| `value_axis.ipynb` | One unseen I-RAVEN-X puzzle through the value-axis model: the printed template, the grounded graph per attribute, the kernel, the answer distribution, accuracy per rule, the shift symmetry with trained weights, values 0–100, and every run's `best.json`. Executed copy: `docs/value_axis.html`. | |
| `submit_value_axis.sh`, `submit_symmetry_ablation.sh` | Value-axis runs (plus, noplus, pervalue, zero; §27); the symmetry ablations with labels and separate weights (§26). `SEED` (here and in `submit_column_roles.sh`, which also has `RUN=named`) passes `bench_hub_grounding.py --seed`. | |

Options of `bench_hub_grounding.py --template facet`:
* `--widths 32,64,128` widens the layers; `--iravenx FILE` trains on I-RAVEN-X; `--save-dir` saves the best
  checkpoint (`template_best.bin` + `best.json`).
* `--copy-links` adds gated copy links from every context panel's code to the output (§20).
  * `--copy-gates content` chooses gates by content and row role, which keeps rows 1 and 2 interchangeable (§21).
  * `--copy-shuffled` is the control: the copied codes come from another puzzle.
* `--columns roles`: first / middle / last columns, so the operands of Arithmetic are interchangeable (§24).
  `--columns fold`: roles, but each row is read by a recursive fold, one shared step per column (§25).
  Neither names a column, so their checkpoints run on rows of any length. With them, `--iravenx A,B` trains on
  files of several row lengths at once (one value range; `--n` per file; notes §25, mixed row lengths).
* Ablations: `--values onehot`, `--per-attr-weights`, `--no-rows`; `--visible-selectors` restores the bug of §17.
  `--label-components` / `--label-attributes` remove the component or attribute symmetry at nearly equal
  capacity (§26; runs: `submit_symmetry_ablation.sh`).

## Torch baselines (`core/baselines/`, data in `data_prep/`, notes §23)

| file | what it is |
|---|---|
| `prepare_baseline_images.py` | nvsa env: the eight context panels of every I-RAVEN puzzle at 80×80, in `iraven.npz`'s order (`experiments/baselines/data/iraven_images.npz`), for `--arch vit`. |
| `prepare_baseline_data.py` | pynl env: writes the facet model's puzzles, splits and weight counts as arrays (`experiments/baselines/data/<dataset>.npz`), plus `bench_hub_grounding`'s own targets for a check. |
| `train_baseline.py` | nvsa env, GPU: on the facet model's task (same targets, selection and metrics), sized to the facet model's weights (`--scale` for more). Architectures: `mlp`, `gnn` (all-to-all over objects), `attr-abs` and `attr-rel` (all-to-all over object attributes, absolute or relative labels: plan models A and B). `vit`: a plain vision transformer over the I-RAVEN matrix as one image, end to end (the black-box vision reference). `rec-tf` (2026-11-23): the facet model's ViT slot recognizer (`--recognizer`, frozen or `--finetune` with its label loss kept, `--aux`) with a plain transformer as the reasoner, the control for supervision. Regularization: `--norm`, `--dropout`, `--drop-edge`, `--wd`, `--schedule cosine` (fixed 2026-11-23; `--schedule alternating` reproduces the 8 runs of 2026-11-02 to 11-23 whose cosine never decayed, FINDINGS 53), `--warmup`, `--clip`, `--augment rows`. `--check-symmetry`, and transformed test sets after training or with `--eval-only`. |
| `rule_eval.py` | nvsa env, GPU: an I-RAVEN baseline run by rule and layout on the unseen puzzles, with `facet_rule_eval.py`'s definitions (`rule_eval.json` in the run directory). |
| `test_eval.py` | nvsa env, GPU: the main table's I-RAVEN baseline checkpoints on the test split, once each (`jobs/baselines/submit_test_once.sh`). A checkpoint must first reproduce its validation record; the result goes to `test.json` in the run directory, and a run that has one is skipped; `TEST_RUNS` replaces the list. Test arrays and images: `data_prep/prepare_baseline_data.py --test`, `data_prep/prepare_baseline_images.py --test`. |
| `submit_baselines.sh`, `submit_sweep.sh`, `submit_regularized.sh`, `submit_attr_gnn.sh`, `submit_final.sh`, `submit_vit.sh` | equal weights (2 seeds); 4×, 16×, 64×; the regularization grid; models A and B; the regularized setting on every dataset, second seeds and 4×/16×; the ViT at 1× and 16×. |
| `submit_rec_tf.sh` | the supervision control (`--arch rec-tf`): frozen recognizer at reasoner widths 64 and 224, 300 to 6000 puzzles per layout; fine-tuned at width 64, 300 to 3000. The baseline job scripts take `SCHEDULE` (default cosine; `alternating` for the runs before 2026-11-23) and `TAGSUFFIX`, added to the run's tag (the reruns with the fixed schedule: `_cosfix`; their job ids in `experiments/baselines/cosfix_jobs.txt`). |
| `submit_cnn.sh` | the small ResNet over the matrix at 1× and 16× (RESULTS §1), with the ViT's recipe; the first runs (job 11932164) came from a script outside the repo with the same arguments. |
| `summarize_baselines.py` | the table, next to the facet model; `--per-rule`. |

## The symbolic models (ground-truth attributes, notes §15)

| file | what it is |
|---|---|
| `pynl_symbolic.py` | Attribute values, rows and slot sets as symbols; crisp relations scored with learned weights. `--model prims` (generic building blocks) or `--model rules` (RAVEN's rule types). |
| `rpm_symbolic.py` | Puzzle reader for the above: numbers, slot sets and per-slot attributes per component. |
| `rpm_reference_solvers.py` | The same two vocabularies as plain, non-learned Python solvers; `--transpose` exchanges rows and columns of the context first. |
| `pynl_symbolic_e2e.py` | Scores a trained symbolic model end to end (ResNet perception → symbolic facts) and with XML labels. |
| `submit_symbolic.sh`, `submit_symbolic_e2e.sh`, `submit_symbolic_extra.sh` | training grid; test split end to end; learning curves and random initial weights. |

## Perception and data checks

| file | what it is |
|---|---|
| `core/frontend/train_frontend.py`, `predict_frontend.py`, `extract_objects.py` | The ResNet-18 perception frontend (torch, trained on RAVEN). Checkpoints: `log/universal_frontend_correct_proportional_mean1`, `_mean3`. |
| `core/pynl/pixel_pipeline.py`, `submit_pixels.sh` | `perceive` (env `nvsa`, GPU): runs the frontend over every I-RAVEN panel into `experiments/frontend_preds/`; `panels_from_preds` decodes them for the end-to-end runs. |
| `torch_gnn/gnn_solver.py` | Kept for `load_frontend`, which `pixel_pipeline.py` uses. |
| `core/gnn/` | The inherited data package: `gnn.data.canonical_positions` (the slot fix of §10) and `CONFIG_POS_MAP`, used by the symbolic reader, the frontend and the data checks. |
| `analysis/sanity_checks.py` | `splits` (no train/test duplicates), `shortcut` (candidates-only baseline), `images` (XML labels match the pixels). Each check has a negative control. |

## The thesis text

| file | what it is |
|---|---|
| `docs/thesis/*.md` | The chapter drafts, the abstract with Chapter 1, and the appendices (Markdown, the source until the LaTeX is frozen). `OPEN_ITEMS.md`: what is left for the author; `REFERENCE_CHECKS.md`: how every citation was checked. |
| `thesis_latex/md2tex.py` | Converts the drafts into `thesis_latex/chapters/*.tex` for the ctuthesis class (citations through its table `CITES`, references to sections and figures as `\ref`, hand fixes in `POST`). `thesis_latex/main.tex` holds the front matter; `references.bib` is a copy of `docs/thesis/references.bib`. Compile on Overleaf with pdfLaTeX and BibTeX. |
| `analysis/thesis_figures.py` | Draws the thesis figures into `docs/figures/` (PDF and PNG), with matplotlib only; the data-factor plot reads its numbers from RESULTS §7. |
| `analysis/thesis_number_check.py` | Lists numbers in the drafts that the results files do not contain, to be checked by hand. |
| `red_team/thesis.md` | An examiner's objections to the drafts, and whether the text answers each. |

## How the files connect

```
bench_hub_grounding.py ──imports──> train_pynl.py (parse, POS_MAP)
facet_symmetry_check.py, facet_rule_eval.py, facet_viz.py, prepare_baseline_data.py ──import──> bench_hub_grounding.py
pynl_symbolic.py, rpm_reference_solvers.py ──import──> rpm_symbolic.py ──> core/gnn/data.py
pynl_symbolic_e2e.py ──imports──> pixel_pipeline.py (panels_from_preds) ──> torch_gnn/gnn_solver.py, core/frontend/
```

## Archive

- `archive/superseded_2026-10-28/`: the GNN-style PyNeuraLogic model (`train_all_constellations.py` and its
  jobs, evaluation and notebooks), the original torch GNN (`torch_gnn/` training and evaluation scripts,
  `scripts/` GNN jobs) and stale LaTeX tables. See its `MANIFEST.md`.
- `archive/pyneuralogic_experiments/`: the porting-stage PyNeuraLogic models (pure, softmax, aligned, hybrid,
  ladder).
- `archive/pyneuralogic_2026-03-31_to_05-06/`: the April 2026 spatial PyNeuraLogic GNN.
