# I-RAVEN and I-RAVEN-X with PyNeuraLogic: project knowledge

State as of 2026-10-28, rewritten after superseded models were archived. Written for use as Claude project
knowledge. Numbers are measured unless marked otherwise. More detail: `thesis/FINDINGS.md` (findings in
order), `thesis/PYNEURALOGIC_NOTES.md` (lab notes), `thesis/THESIS_PLAN_symmetry.md` (plan).

---

## 1. The project in one paragraph

A master's thesis at CTU (cluster RCI, login node `login4.rci.cvut.cz`, account `kassyask`) on solving
Raven's Progressive Matrices (I-RAVEN, and IBM's I-RAVEN-X) with **PyNeuraLogic**, a differentiable logic
programming framework developed at CTU. The repository started as a clone of IBM's NVSA RAVEN code.

**The main model is the facet model**, a generative model on a graph grounded differently for every
puzzle:
- one node per (object, attribute) and thermometer value codes;
- it generates the missing panel and then picks the closest candidate;
- it trains natively in PyNeuraLogic.

| dataset | facet model, matched (exact) |
|---|---|
| I-RAVEN, unseen validation | **97.3%** (69.1%) |
| I-RAVEN-X 3×3 | 99.2–100% |
| I-RAVEN-X 3×10, values 0–49 / 0–99 / 0–999 | 85.4 / 84.9 / 81.2%; with gated copy links **97.8 / 96.2 / 94.3%** |

ARLC reaches 95.0 / 95.0 / 90.6 on I-RAVEN-X 3×10, and LLMs at most 84. A symbolic PyNeuraLogic model with
the rule vocabulary given reaches 98.33% on the I-RAVEN test split (96.31% end to end through a ResNet-18
frontend).

**Thesis framing (chosen 2026-10-28):** symmetries as inductive bias. A symmetry built into the model
(provably, by not naming constants in the template) should reduce the training data needed, raise
accuracy at a fixed number of weights, and generalize to transformed puzzles. Plain torch models on the
same task (MLP, all-to-all GNNs) are the references.

---

## 2. Data

### I-RAVEN-10000
* Location: `data/I-RAVEN-10000/<constellation>/RAVEN_<id>_<split>.npz` + `.xml`. 7 constellations, each
  6000 train / 2000 val / 2000 test: `center_single`, `distribute_four`, `distribute_nine`,
  `left_center_single_right_center_single`, `up_center_single_down_center_single`,
  `in_center_single_out_center_single`, `in_distribute_four_out_center_single`.
* `.npz`: `image` (16, 160, 160) greyscale panels; `target` (answer index 0–7). Panels 0–7 are the context,
  8–15 the candidates. `.xml`: objects with centre-based `bbox`, `Type`, `Size`, `Color`, `Angle`; the rules.
* **22 universal positions.** `train_pynl.POS_MAP`:
  - center_single [0], distribute_four [1–4], distribute_nine [5–13];
  - left/right [14, 15], up/down [16, 17];
  - in/out [0, 9], in_distribute_four [0, 18–21].
* **Slots come from box geometry** (`gnn.data.canonical_positions`). I-RAVEN's XML `Position` lists only
  the occupied slots, so `Position.index(bbox)` is wrong there, and about a third of the grid puzzles
  become unanswerable.
* **Attributes:** type 0–4 (triangle … circle), size 0–5, colour 0–9 (0 = white … 9 = black). Angle
  follows no rule.
* **Rules** are row-wise, one per component and attribute: Constant, Progression (±1/±2), Arithmetic, and
  Distribute_Three (the same multiset in every row).
  - Arithmetic on colour and number: c = a ± b. On size: a + b + 1 or a − b − 1. On positions: union or
    difference.
  - Position Progression is a cyclic shift.
* **Candidates:** every combination of changing or keeping 3 attributes of the answer (2³). The I-RAVEN
  candidates-only shortcut is at chance; on RAVEN it scores 90–100%.

### I-RAVEN-X (IBM's generator)
* `data/iravenx/center_single_shuffle_n_<n>_maxval_<m>.json`: center_single, rows of n panels (n = 3 or
  10), values 0..maxval for type, size and colour alike. Puzzles 0–5999 train, 6000–6299 selection,
  6300–7299 unseen.
* Arithmetic: plus c_n = c_1 + … + c_{n−1}, minus c_1 = c_2 + … + c_n. Distribute rows are cyclic shifts of
  row 1 by ±1. Distribute draws values up to maxval **inclusive**, one more than the codes hold, so the
  facet model's I-RAVEN-X exact scores are slightly off. Matching is unaffected.

### Protocol
The epoch is picked on selection puzzles (I-RAVEN val 0–299 per constellation; I-RAVEN-X 6000–6299).
Results are reported on unseen puzzles (val 300–1299; 6300–7299). The test split is scored once per final
checkpoint.

---

## 3. Environment and cluster rules
* **`pynl`** (Python 3.11, neuralogic 0.9.2, CPU/JVM only):
  `source /mnt/appl/software/Anaconda3/2024.02-1/etc/profile.d/conda.sh && conda activate pynl`.
  Always set:
  - `export JAVA_HOME=/mnt/appl/software/Java/21.0.8` (`ml load Java/21` silently fails on amd nodes);
  - `export PATH=$JAVA_HOME/bin:$PATH`;
  - `export PYTHONNOUSERSITE=1`.

  Cap the JVM with `neuralogic.set_max_memory_size(GB)`. Native training needs
  `Settings(..., iso_value_compression=False)`.
* **`nvsa`** (Python 3.7, torch 1.11, CUDA): the torch baselines and the ResNet-18 frontend. Do not upgrade.
* **Partitions:** `amd` (24 h), `amdlong` (3 days), `amdfast` (CPU); `amdgpu` (24 h), `amdgpufast` (4 h,
  A100).
* **Facet run sizes:** the full I-RAVEN run needs a 300 GB JVM heap (grounding 30 min, ~14 min per epoch).
  I-RAVEN-X 3×10 at values 0–49 needs 64–96 GB and takes ~5 h for 60 epochs. Compute nodes cannot see the
  session's `/tmp`.
* **Login node:** never shut it down. Plain scripts are fine; CPU-heavy Jupyter kernels are killed. Execute
  notebooks with `sbatch … jupyter nbconvert --execute --inplace`.
* **Git:**
  - `origin` = the user's private `kaskare/rpm-thesis` (SSH alias `github-rpm`); `ibm` = IBM's public
    upstream: never push or open PRs there.
  - Data, experiments and logs are not in git.
  - Commits end with "Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>".

---

## 4. The facet model (`pynl/bench_hub_grounding.py --template facet`)
* **Task:** generate the missing panel per component, then pick the closest candidate (ties count as
  wrong). The output per component is:
  - the mean thermometer code of each attribute;
  - the object-count code;
  - the 22-slot occupancy.

  Loss: MSE. Candidates are never seen in training.
* **Graph** (grounded per puzzle; no position nodes, no padding):
  - facets `f(O, A)` exchange messages with same-position facets in another column of the row (a weight
    per column pair);
  - panel-component nodes `vF(P, C, A)` (number is a fourth attribute);
  - row relations `rel(Rw, C, A)` per complete row, averaged into `ctx`;
  - the answer `ans(C, A)` from the question row's panels and `ctx`;
  - a separate position channel with the same row structure.

  Attribute and component are rule variables (shared weights). Structural atoms are hidden.
* **Settings:** widths 32 → 64 → 128, lr 3e-4, 6000 puzzles, 60–80 epochs. Checkpoint = `template_best.bin`
  + `best.json`.
* **Options:**
  - `--copy-links`: gated copy of context codes to the output, an operation prior;
  - `--copy-gates content`: gates by content and row role, so the row symmetry is kept;
  - `--columns roles`: first / middle / last columns, the operand-order symmetry of Arithmetic;
  - `--iravenx FILE`;
  - ablations `--values onehot`, `--per-attr-weights`, `--no-rows`.
* **Built-in symmetries** (theorem: a template is blind to the constants it does not name), verified to
  ~1e-16:
  - object order;
  - candidate order;
  - rows 1 ↔ 2;
  - the components of L-R and U-D, for attributes;
  - size ↔ colour;
  - with roles, the middle columns.

## 5. Results

| model | I-RAVEN matched (exact) | I-RAVEN-X 3×10, values 0–49 |
|---|---|---|
| facet | **97.3** (69.1); per layout 99.3 / 95.5 / 89.8 / 99.5 / 99.7 / 99.6 / 97.8 | 85.4 (15.0); 89.1 after 150 epochs |
| facet + copy links, per-panel gates | – | **97.8** (60.0), but 39.5% on Distribute puzzles after a row swap |
| facet + copy links, content gates | – | 94.9 (51.7), unchanged by a row swap |
| shuffled-copy control | – | 87.3 |
| torch MLP, equal weights / best regularized | 60.6 / 72.8 | 83.5 / 84.4 (exact 2.6) |
| torch all-to-all object GNN, equal / best regularized | 87.2 / 91.3 | 71.6 / 84.0 (exact 3.3) |
| symbolic rules (test split) | 98.33 (end to end 96.31) | – |

* **Exact value per rule, 3×10, values 0–49,** in the order Constant / Progression / Distribute / Arithmetic:
  - facet: 93 / 77 / 10 / 19;
  - copy links: 100 / 93 / 87 / 30;
  - content gates: 100 / 98 / 60 / 30.
* **The baselines' weight count does not help:** at 4–64× the weights they still trail.
* **Column roles (running, 2026-10-28):** 92.7% at epoch 25, against the named-column model's 85.0, with a
  fifth of the weights.
* **The attribute-level GNN pair A/B** (absolute vs relative labels) is queued.

## 6. Pitfalls found (PyNeuraLogic and data)
* An unweighted body atom adds its value to the rule's sum. Visible selector atoms hid the positions until
  2026-10-25 (grids 76.9 → 92.4% after the fix).
* Chain pruning drops the transformation of single-rule atoms, so `rel` and `ans` are written as two rules.
* CONCAT follows the grounder's order. Every sample needs its own query atom. `build_dataset` may be called
  once. Embeddings must be weighted facts.
* Copy gates must start closed (bias −6). A weight given as values must be a flat list to act as a vector.

## 7. Files
See `thesis/CODE_INDEX.md`. In short:
* `pynl/`: `bench_hub_grounding.py` (facet model), `facet_symmetry_check.py`, `dataset_symmetry_check.py`,
  `facet_rule_eval.py`, `facet_viz.py`, the symbolic models, `pixel_pipeline.py` (perception cache) and
  `sanity_checks.py`;
* `baselines/`: the torch baselines;
* `notebooks/facet_model.ipynb`;
* `archive/`: superseded models, including the GNN-style PyNeuraLogic model and the original torch GNN.

## 8. Notebook and writing preferences
* Say "end to end", never "from pixels". Keep notebooks compact.
* An executed notebook plus its HTML is the deliverable. The cluster is not shared, and no passwords are
  stored.
* Answers should be clear and direct.

## 9. Open items
* Score the facet model once on the I-RAVEN test split, and end to end.
* Seeds for the facet runs: most numbers are one run.
* Plan items E1–E4: models A/B, learning curves, transformed test sets, wrong-symmetry controls.
