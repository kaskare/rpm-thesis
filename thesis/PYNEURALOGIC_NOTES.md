# I-RAVEN and I-RAVEN-X with PyNeuraLogic — lab notes

Working notes, kept as a record of what was measured and how. The findings that still hold, in the order
they were made: `thesis/FINDINGS.md`. The thesis plan: `thesis/THESIS_PLAN_symmetry.md`.

**Pruned 2026-10-28.** Sections on superseded models were removed: the porting-stage PyNeuraLogic models
(§4–§8), the GNN-style PyNeuraLogic model (§9, §11.4–11.5, §12–§14, §16), the original torch GNN (most of §1,
the spatial-relation bug in §10) and the April spatial GNN (§19). They are in git history (commit 4380e83);
their code is in `archive/`. Section numbers are kept so references stay valid.

---

## 1. RAVEN's answer-set leakage

### 1.1 Measured with the original torch GNN (since archived)

| | `object_only` | `+ unified` | Δ |
|---|---|---|---|
| RAVEN | 81.09% | 96.45% | **+15.4** |
| I-RAVEN | 68.44% | 48.28% | **−20.2** |

Same code, same hyperparameters, sign flip. RAVEN builds its 8 candidates by perturbing
the correct answer one attribute at a time, so the answer is the attribute-wise centroid
of the candidate set — recoverable without looking at the context. `unified_graph` puts
all 8 candidates in one graph where they exchange messages, which is exactly the
mechanism needed to exploit that. I-RAVEN kills the bias, and there the same mechanism
costs 20 points.

**Consequence:** RAVEN numbers from models that see all candidates together are not real; work on
I-RAVEN, and keep candidates from seeing each other. The generative models never see them at all.

## 2. Environment

PyNeuraLogic **0.9.2** in conda env **`pynl`** (`~/.conda/envs/pynl`, Python 3.11.16,
JPype 1.7.1, torch 2.14.0+cpu). The old `nvsa` env is Python 3.7.12, which caps
`neuralogic` at 0.8.0 — 0.9.x needs ≥3.10, so it cannot be upgraded in place.

Three environment traps:

* **`ml load Java/21` silently does nothing on the amd compute nodes.** `JAVA_HOME` comes
  back empty and JPype dies with `JVMNotFoundException`, even though the module works on
  the login node. Pin `JAVA_HOME=/mnt/appl/software/Java/21.0.8` and verify `libjvm.so`.
* **`~/.local/lib/python3.11/site-packages` shadows the conda env** (it carries
  transformers / huggingface-hub / gymnasium). Always `export PYTHONNOUSERSITE=1`.
* **The default JVM heap kills a Jupyter kernel** (`DeadKernelError`).
  `neuralogic.set_max_memory_size(16)` fixes it. Only bites in notebooks.

PyNeuraLogic is **JVM/CPU only** — never request a GPU. Compute scales ~d² and the engine
is largely single-threaded (~1.5 cores utilised regardless of `--cpus-per-task`).

### 2.1 API changes 0.8 → 0.9 that break the old code

| old (0.8.0) | new (0.9.2) |
|---|---|
| `from neuralogic.core import Template` | **gone** — use `Model()`, `m += rule`, `m.build(Settings())` |
| `from neuralogic.nn import get_evaluator` | **gone** — `neuralogic.nn.Trainer` |
| `from neuralogic.optim import Adam` | module gone — `from neuralogic.nn import Adam` |
| `neuralogic.inference.inference_engine` | **module gone** — closest is `core.Grounder` |
| `m[R.h0/1] = Metadata(...)` | `m += (R.h0/1) \| Metadata(...)` |
| — | predicate metadata **rejects `aggregation`** — that goes on the *rule* |

---

## 3. PyNeuraLogic framework gotchas

Each of these cost real debugging time and none produce a clear error message.

1. **Embedding tables must be declared as weighted facts.** `R.emb_col(V.C)[DIM,]` in a
   rule body with no `R.emb_col("c3")[DIM,]` facts silently fails to ground; you get
   `Query [score] not matched anywhere in the template`, which points at the wrong place.
2. **Every sample needs a unique query atom.** With a nullary `Sample(R.score[label], …)`
   all 8 candidates share one query and PyNeuraLogic **merges them** — 3440 samples came
   back as 2240 with no error. Key on `R.score(gid)`. This also means duplicating samples
   to oversample positives does nothing; copies are deduplicated.
3. **`build_dataset()` may be called only once per built model** — the second call throws
   `ArrayIndexOutOfBoundsException`. Build train+val together and split the list.
4. **`Aggregation.CONCAT` does not emit in the order you named the constants.** It uses
   the grounder's own permutation — `[6,0,4,3,1,2,7,5]` here. Without correcting it the
   head scores candidate *i* against candidate *j*'s target and the model cannot learn.
   The order depends only on the constants, **not on template depth** (verified identical
   for layers 0/1/2), so a depth-0 identity companion can be probed to recover it.
5. **Rules sharing a head combine with `SUM`** (`relation_combination = Combination.SUM`);
   `Aggregation.AVG` applies *within* one rule, over its groundings.
6. **Constant tensor inputs to the torch wrapper need `requires_grad=True`.** Otherwise
   autograd never records the `NeuraLogic` node, backward never reaches the template
   weights, and you get `element 0 of tensors does not require grad`.
7. **Weight sharing across topologies breaks `backward`.** Two `NeuraLogic` modules built
   from one `Model` do share weights (`A.model is B.model` → True) and forward correctly,
   but the second topology's backward throws
   `Incompatible dimensions … matrix increment by vector` — trainer state binds to
   whichever topology grounded first. This rules out one-module-per-constellation.
8. **An unweighted body atom adds its value to the rule's sum.** A "selector" such as
   `obj(P, O)` (value 1.0) in `occ(P, C) :- obj(P, O), ocomp(O, C), pos(O)` adds 1.0 to every
   component, so under `Aggregation.SUM` each object contributes its one-hot **+ 2 everywhere**.
   This is LRNN semantics, not a defect; atoms that should only ground the rule must be declared
   `R.hidden.obj(...)` (safe from 0.9.2 on, which fixed weight misalignment around hidden atoms).
   It hid the object positions from the facet model; see §17.

---

## 10. The slot bug in the inherited data pipeline

**`gnn/data.py::parse_ground_truth` derived an object's slot from
`all_positions.index(bbox)`, where `all_positions` is the layout's XML `Position`
attribute. On I-RAVEN that attribute lists only the slots a panel actually occupies**, so
the index means "k-th object present", not "which grid slot".

**The bug is I-RAVEN-specific.** The two datasets write `Position` differently:

| dataset | `Position` lists | `Position.index(bbox)` is | bug |
|---|---|---|---|
| RAVEN-10000 | **every** slot of the layout | the true slot | dormant |
| I-RAVEN-10000 | only the **occupied** slots | rank among present objects | live |

```
distribute_four, one panel     RAVEN:   3 objects, Position lists 4 bboxes
                               I-RAVEN: 2 objects, Position lists 2 bboxes
```

I-RAVEN's generator rewrites the XML when it rebuilds the candidates, and in doing so shrinks
`Position` to the occupied slots. Code written and tested against RAVEN is therefore correct
there and silently wrong on I-RAVEN — which is how it went unnoticed.

For any panel holding fewer objects than the constellation's maximum, position identity is
destroyed — a lone object at slot 3 and a lone object at slot 0 both become local index 0.

```
distribute_four   Position-list lengths seen: {1, 2, 3, 4}   true distinct bboxes: 4
distribute_nine   Position-list lengths seen: {1 … 9}        true distinct bboxes: 9
center_single     always length 1                            1
```

### Measured impact

| constellation | puzzles with indistinguishable candidates |
|---|---|
| `center_single`, `left/right`, `up/down`, `in/out` | 0/40 each |
| **`distribute_four`** | **13/40** |
| **`distribute_nine`** | **13/40** |
| **`in_distribute_four_out`** | **12/40** |

Roughly a third of the puzzles in the three variable layouts are **unanswerable** with the old parser:
two or more candidates encode identically and one of them is the answer. The original torch GNN's results
on those layouts came from these corrupted inputs.

### The fix

`canonical_positions(config, dataset_root)` derives the slot from **bbox geometry**, using
the union of every bbox ever seen in that component, cached per constellation, and hard-fails
if the count ≠ `len(CONFIG_POS_MAP[config])`. Applied to both `gnn/data.py:43` and
`train_pynl.py`.

Regression check — panels whose parse changed:

| constellation | changed | duplicates after |
|---|---|---|
| the four fixed-count ones | **0**/480 each | 0 |
| `distribute_four` | 334/480 (70%) | 0 |
| `distribute_nine` | **449/480 (94%)** | 0 |
| `in_distribute_four_out` | 369/480 (77%) | 0 |

Zero change where counts are fixed (the old code was accidentally right there), and all
duplicates gone.

### Second sweep (2026-09-29): the ResNet frontend and its tools

The same line was copied into three more files, all of them about the **ResNet-18 perception
frontend** rather than the GNN:

| file | function | fixed |
|---|---|---|
| `train_frontend.py` | `parse_xml_panel_targets` — the ResNet's **training labels** | ✓ |
| `predict_frontend.py` | `parse_ground_truth` — ground truth shown next to predictions | ✓ |
| `extract_objects.py` | `parse_ground_truth` — ground truth for object extraction | ✓ |

All three now call `gnn.data.canonical_positions`, so there is one slot decoder in the repo.
(Two copies remain under `archive/`; they are archived and left untouched.)

**Impact: none on existing results.** The frontend was trained on **RAVEN-10000**, where the
bug is dormant, so its labels were correct: on RAVEN the original parser's positions equal the
true geometric slots on every panel checked (0 of 1600 differ across distribute_four,
distribute_nine, in_distribute_four and center_single). Its reported 95% panel accuracy is
therefore against correct labels, and its checkpoints stay valid. Nothing in the scripts or logs ran these three files on I-RAVEN.
The fix matters for any *future* I-RAVEN use (e.g. `train_frontend.py --dataset
data/I-RAVEN-10000`), where the old code would have mislabelled 73–94% of panels in the
variable-layout constellations.

Regression, first 200 test puzzles per constellation, fixed vs original code, panels changed
(of 3200):

| constellation | RAVEN | I-RAVEN | fixed vs `gnn.data` on I-RAVEN |
|---|---|---|---|
| the four fixed-count ones | 0 | 0 | 0 |
| `distribute_four` | **0** | 2335 (73%) | 0 |
| `distribute_nine` | **0** | 2995 (94%) | 0 |
| `in_distribute_four_out` | **0** | 2442 (76%) | 0 |

Identical on RAVEN in all three files (existing checkpoints and RAVEN numbers unaffected);
on I-RAVEN they change exactly where the bug bites and agree with the already-fixed
`gnn/data.py` parser on every panel.

**Rule for new code:** never derive a slot from `Position.index(bbox)`. Use
`gnn.data.canonical_positions(config, dataset_root)[(component_index, tuple(bbox))]`.

### Rotation is *not* a missing attribute

`Angle` is on every `<Entity>` and captured by neither pipeline. Across 420 puzzles in all
7 constellations, the rules govern only **Type (660), Size (660), Color (660),
Number/Position (506), Number (81), Position (73)** — `Angle` appears **zero** times. It is
a noise distractor, so dropping it is correct, and after the position fix no candidate
collisions remain without it.

---

## 11. End-to-end from pixels

Everything above reasons over ground-truth attributes read from the XML. To get a number
from raw images, the XML is replaced by what the **ResNet-18 perception frontend** sees.

### 11.1 The perception frontend (`train_frontend.py`)

An NVSA-style "universal frontend", trained on **RAVEN-10000**, all 7 constellations jointly.

* **Input:** one panel at a time, 160×160 grayscale (raw 0–255), repeated to 3 channels for
  the ImageNet-pretrained ResNet-18. First conv stride 1 and no max-pool, so small objects
  keep their resolution. Output: a 512-d vector per panel.
* **Target space:** a **frozen** random bipolar codebook of 6622 rows = 22 universal positions ×
  (5 types × 6 sizes × 10 colours + 1 "object here" row). Each attribute row is the
  element-wise product of four random ±1 vectors (type ⊙ size ⊙ colour ⊙ position). Only the
  ResNet trains.
* **Loss:** cosine similarity to every row, times a learned scale, under the additive
  cross-entropy of the NVSA paper. The best runs use one term per object, averaged
  (`--per-object-loss --loss-reduction mean`), so a multi-object panel learns to output the
  superposition of its objects' vectors.
* **Readout (training eval):** every row with cosine above τ = 0.23 is a detection; a panel is
  correct only if the detected set matches exactly.
* **Best checkpoints:** `log/universal_frontend_correct_proportional_mean3` (95.26% panel
  accuracy) and `…_mean1` (95.04%, the one wired into `gnn_solver.py`).

Its labels were correct despite the §10 bug, because the bug is dormant on RAVEN.

### 11.2 The pipeline (`pixel_pipeline.py`, `submit_pixels.sh`)

`perceive` (GPU, env `nvsa`) runs both frontends over every I-RAVEN val + test panel (448,000), stores per
panel and position the best attribute match and its cosine (`experiments/frontend_preds/`), and scores
perception against the ground truth. The end-to-end runs (`pynl_symbolic_e2e.py`) decode these.

* **Decoding:** at each position the constellation can use, take the best of its 300 attribute rows; the
  object exists iff that cosine > τ = 0.23. One object per position by construction. The constellation
  label comes from the dataset folder, as in the XML pipeline.
* **No remapping needed:** the frontend's codebook positions, `train_pynl.POS_MAP` and
  `gnn.config.CONFIG_POS_MAP` are identical, and the two ground-truth parsers agree on every panel checked.
* **Built-in check:** the decoder is asserted equal to the frontend's own `decode_w_index`.
* **Cost:** perception 10 min on one A100.

### 11.3 Perception on I-RAVEN (test, `mean3`)

| constellation | panels exactly right | puzzles with all 16 right |
|---|---|---|
| center_single, left/right, up/down, in/out | 100.00% | 100.00% |
| distribute_four | 99.99% | 99.90% |
| distribute_nine | 96.02% | 76.45% |
| in_distribute_four | 96.34% | 71.55% |
| **overall** | **98.91%** | **92.56%** |

Higher than the 95.3% the frontend reported during training, because the decoding differs:
the training eval counts every row above threshold (two hits at one position = wrong), the
pipeline takes the single best row per position among the constellation's positions. Only
474 of 224,000 panels had a detection outside the constellation's positions.

## 15. Symbolic templates (`pynl_symbolic.py`)

A different kind of PyNeuraLogic model: no embeddings and no message passing. Attribute values, rows of
three panels and slot sets are symbols; crisp rules (MIN combination, MAX aggregation) derive which
relations hold on each row; learned weights score each candidate row. One sample per puzzle, 8-way
softmax, native PyNeuraLogic training (`iso_value_compression=False` is required).

* **Facts** (`puzzle_tuples`): per panel and component, values (`val`), per-slot attribute tuples
  (`tup`), slot sets (`occset`), and set algebra over the puzzle's slot sets (union, difference,
  intersection, cyclic shift, row permutation). Background knowledge: steps ±1/±2, a ± b + k.
* **`--model prims`:** a generic vocabulary (equalities, steps, sums, set relations, slot-wise equality,
  same multiset) and the analogy principle "a relation both context rows show should hold on the
  candidate row"; one weight per (attribute, primitive).
* **`--model rules`:** RAVEN's rule families written as compositions of those primitives; one weight per
  (attribute, rule family).

| model | unseen val (300+/constellation) |
|---|---|
| template, primitives (`symb_prims_n300_lr0.01`) | **97.58%** |
| template, rules (`symb_rules_n300_lr0.01`) | **98.44%** |
| non-learned reference, primitives / rules (`rpm_reference_solvers.py`, val 2000/cons) | 96.34% / 97.75% |

* The non-learned rule solver never ranks a wrong candidate above the answer: 95.74% unique right,
  4.26% ties (mostly 2-way), 0.00% wrong; ties concentrate where objects come and go (distribute_nine
  10.2%, distribute_four 7.6%, in_distribute_four 5.9%). Learned weights break some ties.
* The learned primitive weights rediscovered the rules: position union +15.8, colour c = a + b + 0
  +11.0; type arithmetic and wrong colour offsets negative.
* Caveats: the best epoch is chosen on the same validation puzzles; the vocabulary was chosen after
  measuring this dataset's rule semantics; the relations are designed, not learned.
* **Test split, scored once (2026-10-19, `pynl_symbolic_e2e.py`, `experiments/symb_e2e_*_test.json`):**

| model | XML labels | end to end (ResNet-18) |
|---|---|---|
| template, rules | **98.33%** | **96.31%** |
| template, primitives | 97.55% | 95.71% |
| reference solver, rules (non-learned) | 97.78% | 95.74% |
| reference solver, primitives (non-learned) | 96.42% | 94.73% |

  Per constellation (rules, XML / end to end): center_single 99.90 / 99.90, distribute_four
  95.98 / 95.94, distribute_nine 95.46 / 85.44, left/right 99.95 / 99.95, up/down 99.95 / 99.95,
  in/out 99.85 / 99.85, in_distribute_four 97.26 / 93.16. All of the end-to-end loss sits where
  perception is imperfect (puzzles fully perceived: distribute_nine 76.5%, in_distribute_four 71.6%).
* **Learning curve** (unseen val; initial weights all 0.1 = non-learned counting): primitives start at
  96.01% and reach 97.13 / 97.35 / 97.49 / 97.58 / 97.58% with 10 / 30 / 100 / 300 / 1000 puzzles per
  constellation; rules start at 97.96%, do not improve with 10 or 30 puzzles (training drifts to 97.6%),
  and reach 98.25 / 98.44 / 98.49% with 100 / 300 / 1000.
* **Random initial weights** (uniform 0–0.2, 3 seeds, 300/constellation): every seed converges to the same
  validation accuracy as the equal start (97.58% primitives, 98.44% rules) from starts of 95.7–96.4% and
  97.6–97.9%. The result does not depend on the initialisation.

## 17. Facet model: positions hidden by visible selector atoms (found and fixed 2026-10-25)

**Where.** The generative per-puzzle graph model, `pynl/bench_hub_grounding.py --template facet`
(objects as nodes, one facet node per attribute, panel-attribute nodes, ordered row relation per
attribute, positions as a separate occupancy channel).

**Symptom.** Trained on all seven constellations, distribute_four and distribute_nine stayed far below
in_distribute_four, whose inner component is the same 2×2 grid (val 300–1299, widened model, epoch 5):
distribute_four 58.5%, distribute_nine 52.1%, in_distribute_four 87.2%. The same widths trained on
distribute_nine alone reached 90.1%.

**Not the graph.** The grid component is built identically in the three layouts (1000 unseen puzzles each):

| | distribute_four | in_distribute_four (inner) | distribute_nine |
|---|---|---|---|
| objects per panel | 2.24 | 2.23 | 4.65 |
| same-position edges per puzzle | 19.3 | 19.2 | 40.0 |
| rows with a fixed layout | 18% | 18% | 14% |
| answers uniform in type / size / colour | 82 / 87 / 86% | 88 / 86 / 88% | 79 / 83 / 85% |

**Diagnosis** (epoch-5 checkpoint, `Modified_attributes` from the XML):
* Answer preferred over a candidate differing only in one attribute: type / size / colour 97.7–99.6% in
  every grid; **position 46.4 / 49.6 / 53.8%** (chance) in distribute_four / distribute_nine /
  in_distribute_four; **number 71.4 / 57.3 / 98.5%**.
* Generated positions exactly right: **0.0%** in distribute_four and distribute_nine, 20.2% in
  in_distribute_four.
* Mean predicted occupancy: distribute_four puts 0.00 on its own slots 1–4 and 0.55 / 0.47 / 0.58 / 0.62
  on slots 18–21 (in_distribute_four's inner grid); distribute_nine 0.00 on 5–13 and 0.69–0.73 on 18–21.
* Moving a puzzle's objects to other slot ids (1–4 → 5–8) or permuting them left the predicted positions
  unchanged to two decimals; left/right components (slots 14, 15) were also predicted at 18–21; only
  slot 0 (center_single, outer objects) was predicted correctly.

**Cause.** `occ(P, C) :- obj(P, O), ocomp(O, C), pos(O)` with `Aggregation.SUM`: the two unweighted
structural atoms each add 1.0 to every component (gotcha 8, §3). A panel with objects at slots 1, 3, 4
reached the model as `[6, 7, 6, 7, 7, 6, 6, …]` instead of `[0, 1, 0, 1, 1, 0, 0, …]`: +2 per object in all
22 dimensions, up to +18 in distribute_nine. After the row-relation tanh only the object count got
through, so the decoder could not tell the grid layouts apart and put every grid's objects on one slot
set, 18–21. The attribute channels were not affected in the same way: they aggregate with AVG, so the
selectors there add a constant rather than a count-scaled offset.

**Why in_distribute_four still scored 87%.** The one slot set the decoder used is its own, so its
count-based occupancy landed on the right slots (number decisions 98.5%); and only 21% of its puzzles have
all three candidate-separating attributes in the grid (the rest involve the easy outer object), against
92–95% in distribute_four and distribute_nine. The joint-vs-alone gap on distribute_nine has the same
origin: trained alone, the count-based guesses landed on distribute_nine's own slots 5–13.

**Fix.** `build_facet_template(..., hidden=True)` (the default) declares every structural atom hidden
(`obj`, `ocomp`, `samepos`, `col`, `row`, `row3`, `full`, `qcomp`: 17 occurrences); the weighted
`puzzle(G)` bias is unchanged. `--visible-selectors` reproduces the old behaviour. Checked: the occupancy
of the panel above is now `[0, 1, 0, 1, 1, 0, …]`; in an untrained model, permuting a puzzle's objects
moves the predicted occupancy by 0.10 against 0.03 before; both versions still memorise 15 grid puzzles.
PyNeuraLogic's own GNN modules had the same problem with valued edge atoms (fixed upstream in 0.9.2 by
making them hidden).

**Re-runs** (started 2026-10-25 21:19): all constellations, widened, hidden (job 11882029,
`experiments/facet_all_w32-64-128_hidden`); and distribute_four + distribute_nine + in_distribute_four,
2000 each, hidden vs `--visible-selectors` (jobs 11882030 / 11882031), 40 epochs, widths 32 → 64 → 128,
lr 3e-4, val 300–1299:

| | distribute_four | distribute_nine | in_distribute_four | all three | exact generation |
|---|---|---|---|---|---|
| hidden (fixed), epoch picked on val 0–299 (20) | 94.5 | 86.2 | 96.4 | 92.4 | 20.8 |
| hidden (fixed), epoch 40 | 93.5 | 88.4 | 96.4 | 92.8 | 22.9 |
| visible (old), epoch picked on val 0–299 (30) | 71.0 | 73.4 | 86.3 | 76.9 | 2.6 |

The full all-constellation run (11882029) is still training.

## 18. Generative per-puzzle object graphs: the facet model, results and assessment (2026-10-24/25)

### 18.1 Setup
* **Task.** Generate the missing panel, per component: type / size / colour histograms, object count and
  occupied positions. Then pick the candidate closest to what was generated. Candidates are never seen in
  training; the loss is MSE against the true answer's description. Two metrics: *matched* (closest
  candidate is right; chance 12.5%; the true description matches 99.1% on I-RAVEN, 100% on I-RAVEN-X)
  and *exact generation* (the decoded panel equals the answer; stricter).
* **Input.** XML ground-truth attributes (perception assumed perfect).
* **Protocol.** I-RAVEN: train on train puzzles, pick the epoch on val 0–299, report val 300–1299. I-RAVEN-X:
  train 0–5999, pick on 6000–6299, report 6300–7299. **Test splits untouched.** Every number below is one
  seed and one validation run.
* **Command** (the full run; JVM heap 300 GB; grounding took 30 min):
  `python3 pynl/bench_hub_grounding.py --template facet --constellation all --widths 32,64,128 --lr 3e-4
  --n 6000 --n-sel 300 --n-val 1000 --n-train-eval 300 --train-epochs 80 --eval-every 5
  --save-dir experiments/facet_all_w32-64-128_hidden --jvm-gb 300`. For I-RAVEN-X, add
  `--iravenx data/iravenx/center_single_shuffle_n_10_maxval_50.json`.
* **Code.** `pynl/bench_hub_grounding.py --template facet`. Options: `--widths 32,64,128`, `--values`,
  `--per-attr-weights`, `--no-rows`, `--visible-selectors`, `--iravenx FILE`, `--save-dir`.
  `configure(n_cols, maxval)` sets row length and value-code length.

### 18.2 The graph (built by grounding, different for every puzzle)
* **Nodes.**
  * Facets `f(O, A)`: one per object and attribute (type, size, colour). Values are 10-bit thermometer codes,
    or maxval bits for I-RAVEN-X.
  * Panel-component values `vF(P, C, A)`; number is a fourth attribute.
  * Row relations `rel(Rw, C, A)` for the complete rows.
  * Answers `ans(C, A)`.
  * A position channel with the same row structure over the component's occupancy.
  * No position nodes, no padding.
* **Edges.**
  * Facet ↔ facet: same attribute, same position, another column of the row (a weight per column pair).
  * Facet ↔ its panel node.
  * Panel nodes → row node: a weight per column.
  * Rows 1–2 and the question row's panels → answer.
* **Weights.** Attribute and component are variables, so one weight set serves all of them.
* **Structural atoms are hidden** (§17).
* **Priors**, as against "no priors":
  * attributes are independent, and behave alike (shared weights);
  * values have an order (thermometer codes);
  * panels have a column order;
  * same position means correspondence;
  * panels and components are units;
  * with the row relation (level 2), "the same relation holds in every row".
* **Not given:** rule names, arithmetic facts, rule labels, candidates.

### 18.3 What moved the numbers (distribute_nine, val 300–1299)
| step | unseen matched |
|---|---|
| hub: one mixed vector per object, rows unordered (2000 puzzles) | 24.7 |
| ordered: a weight per column pair, row-3 decoder | 31.4 |
| facet: one node per attribute, thermometer values, panel-attribute nodes | 73.6 |
| facet, 6000 puzzles | 78.0 |
| facet, widths 32 → 64 → 128, lr 3e-4 (lr 1e-3 was unstable: 70.3) | 90.1 |
| selector fix, 3 grid layouts trained jointly (2000 each) | 86.2 (d9), 92.4 over the three |

Training-side changes did not help (cross-entropy 23.5, training on all three rows 21.9, chain pruning off
24.2; all with the ordered template). The gains came from the representation.

### 18.4 Current results (runs in progress; update when finished)
* **I-RAVEN, all 7 constellations, fixed, widths 32 → 64 → 128, lr 3e-4, 6000 per constellation** (job
  11882029, 80 epochs, finished 2026-10-26, `experiments/facet_all_w32-64-128_hidden`). At the epoch picked on
  val 0–299 (epoch 55): **97.3% matched, 69.1% exact** on val 300–1299 (last epoch: 97.0 / 68.4):

  | center | 2×2 | 3×3 | L-R | U-D | O-IC | O-IG |
  |---|---|---|---|---|---|---|
  | 99.3 | 95.5 | 89.8 | 99.5 | 99.7 | 99.6 | 97.8 |

  Progress: 92.9 (epoch 5), 95.2 (10), 96.3 (15), then 96–97.3 from epoch 25 on. The same model with visible
  selectors was at 84.7 / 14.2 at epoch 5. The notebook (`thesis/facet_model.html`) was re-run on this
  checkpoint: example distribute_four val 303 is picked correctly, though the generated panel has one object
  too many; val 300–399 gives 97.6% matched. The 150-epoch repeat (job 11887180, `experiments/facet_ep150_all`)
  peaked at epoch 65 with 97.4% matched / 69.3% exact: no gain over 60 epochs.
* **I-RAVEN-X 3×3, values 0–9** (`experiments/facet_iravenx_n3_m10`). **99.2%** at the epoch picked on
  val 6000–6299 (epoch 10), **100.0%** from epoch 15 on, 95.4% exact. Copying the last panel matches 4.7%.
* **I-RAVEN-X 3×10** (IBM generator, seed 1234, no confounders), jobs 11883234–36, 60 epochs, finished
  2026-10-26. Unseen matched at the epoch picked on selection / at the last epoch (exact in brackets):
  * maxval 50: **85.4** (epoch 35; 15.0) / 87.8 (22.6)
  * maxval 100: **84.9** (epoch 60; 6.4) / 84.9 (6.4)
  * maxval 1000: **81.2** (epoch 50; 0.0) / 83.4 (0.0)

  Copy-panel baselines 1.9–2.8%; the true description matches 100%. Positions are always exact, and
  per-attribute exactness falls with the range (type 68 / 44 / 3% at the last epoch). Still rising at
  epoch 60, so 150-epoch runs were started (`pynl/submit_facet_long.sh`, jobs 11887180–83,
  `experiments/facet_ep150_*`), together with the all-constellation I-RAVEN run.

### 18.5 Where it stands against the literature
| setting | ours | reference |
|---|---|---|
| learned rules, ground-truth attributes, I-RAVEN | 97.3 (val 300–1299; test not yet scored) | ARLC 92.6 (trained on 2×2 only!), MLP 87.1, GPT-3 86.5, Learn-VRF 79.5 |
| hand-written rules, same input | – | our symbolic template 98.33 (test), NVSA backend about 97–99 (RAVEN) |
| pixels, classifying candidates | not run | Triple-CFN 99.1, RS-Tran 98.7, SAViR-T 98.1, DRNet 97.6 (self-reported). Independent re-runs (A-I-RAVEN, IJCAI 2025): DRNet 90.9, PredRNet 88.8, SCL 83.4, CPCNet 70.4, STSN 59.0 |
| I-RAVEN-X 3×3 | 99.2 / 100 | ARLC learned 98.4, programmed 97.2, GPT-4 93.2, Llama-3 85.0 |
| I-RAVEN-X 3×10, maxval 50 / 100 / 1000 | 85.4 / 84.9 / 81.2 (60 epochs) | ARLC learned 95.0 / 95.0 / 90.6 (trained once at 50, reconfigured), programmed 100 / 100 / 99.7, GPT-4 82.2 / 79.6 / 76.6 |

Protocol differences to state:
* ARLC trains on 2×2 only and transfers to the other constellations, and from maxval 50 to larger ranges.
  We train per constellation set and per range; thermometer codes cannot stretch past their length.
* Image models see candidates during training and classify among them. Ours generates without candidates.

### 18.6 Analyses worth keeping
* **Grids are hardest across the I-RAVEN literature.** In SRAN's own table (the I-RAVEN paper) and in
  NVSA's and ARLC's, learned models are weakest on 2×2, 3×3 and O-IG. I-RAVEN-X (Hersche et al., NEURMAD
  2025; arXiv 2412.05586 and 2510.17496) keeps only center_single on purpose: LLMs did *better* on the grids,
  and the grids can be solved from the last row alone.
* **Candidate structure.** Only 21% of in_distribute_four puzzles need all three candidate-separating
  decisions inside the grid, against 92–95% for distribute_four / distribute_nine. Per-configuration
  accuracies are therefore not directly comparable.
* **ARLC's grid scores** follow from "position decisions at chance, everything else right":
  1 − ½ × (share of puzzles whose candidates differ in position).

  | | predicted | ARLC reported |
  |---|---|---|
  | 2×2 | 84.7 | 83.4–84.9 |
  | 3×3 | 83.2 | 80.0–81.7 |
  | O-IG | 88.2 | 89.4–90.3 |

  ARLC encodes a panel's positions as one pattern symbol, and its add/subtract rule formula cannot express
  OR, AND-NOT or cyclic shifts. That is a design limit, not our bug.
* **Error analysis by attribute** (answer vs a candidate differing only in one attribute) pinpoints failures
  by attribute and rule. It is how the selector bug was found (§17).
* **Cost.** Grounding takes 10–165 ms per puzzle, once. Training takes 2–66 ms per puzzle per epoch on CPU;
  all of I-RAVEN takes about 14 min per epoch at widths 32 → 64 → 128.

### 18.7 Findings for the thesis, ranked
1. Generative per-puzzle object graphs in PyNeuraLogic: design, headline numbers, and the table in 18.3:
   the right lifted representation brings a learned model close to hand-written rules.
2. The prior levels (none / same relation in every row / given rules). This needs the post-fix ablations,
   and ideally a generative level 3.
3. Semantic pitfalls of lifted templates: visible selectors (§17), chain pruning dropping transformations,
   hidden atoms merging identical groundings. Plus how each was diagnosed.
4. Error analysis and the benchmark observations in 18.6.
5. Cost (18.6), the slot bug (§10), the symbolic templates (§15) and their end-to-end cost.
(Superseded on 2026-10-28 by the symmetry framing: `thesis/THESIS_PLAN_symmetry.md`.)

### 18.8 Needed before these are claims
1. Score the final checkpoint once on the test split, and end to end with the cached ResNet-18 predictions
   (`experiments/frontend_preds/`). About half a day after the run ends.
2. Re-run the ablations with hidden selectors: values, weight sharing, level 1 vs 2. About a day, in parallel.
3. Three seeds of the final configuration.
4. ARLC's protocol: train on 2×2, test on all seven (unseen slot ids may hurt). About half a day.
5. Optional: a generative level 3 (symbolic rules as a generator), so all three levels share one setting.

## 20. I-RAVEN-X 3×10: why the facet model fell short, and gated copy links (2026-10-26)

### 20.1 Error analysis by rule
The 60-epoch 3×10 checkpoints, on unseen puzzles 6300–7299. For each attribute, the generated value
(= the number of thermometer bits on) is compared with the true one, grouped by that attribute's rule
label. Script and output: `experiments/facet_iravenx_analysis/` (`rule_errors.py`, job 11897470).

| rule | values 0–49: exact | mean error (% of range) | values 0–99: exact | values 0–999: exact | mean error (% of range) |
|---|---|---|---|---|---|
| Constant | 93% | 0.3 | 72% | 5% | 0.6 |
| Progression | 77% | 0.7 | 58% | 5% | 0.7 |
| Arithmetic | 19% | 3.3 | 11% | 1% | 2.6 |
| Distribute-n | 10% | 7.6 | 4% | 0.5% | 6.6 |

Two separate problems:
* **Distribute-n is a limit of the structure.** The missing panel is one specific panel of row 1 or 2
  (the rows are cyclic shifts of the same n values). The answer node sees rows 1–2 only through `ctx`,
  the AVERAGE of two tanh row summaries, so the single panel is gone. The only remaining route is
  arithmetic (the row-1 sum minus the 9 known row-3 values), which needs 1-in-10,000 precision through
  tanh layers. In the one puzzle inspected, the wrong candidates reused another value of the same row, so
  an approximate answer lands near a distractor.
* **Precision is a limit of the value code.** Constant and Progression are almost right (under 1% of the
  range off), but a 1000-bit thermometer squeezed into 128 tanh units cannot keep single-value resolution,
  and an off-by-5 error changes the per-bit-averaged MSE by about 0.001. The values 0–999 run's training
  MSE stayed at 0.0010 from epoch 5 to 60.

Memorisation is not the problem: without copy links the model memorises 200 3×10 puzzles (values 0–49)
to 198/200 exact in 50 epochs. The model learns rules that do not carry over to unseen puzzles.

### 20.2 Gated copy links (`--copy-links`, `build_facet_template(copy=True)`)
* **Facts.** Every context panel P (all 3n−1) gets `pc(P, C, A)`: its component's value code, scaled
  to ±`COPY_SCALE` (4), for type, size, colour and number.
* **Gates.** `gate(P, C, A)` is a sigmoid of `ans`, with its own weights per panel. Its bias starts at
  `GATE_BIAS0` = −6, so gates start at about 0.0025. It is written as two rules so chain pruning keeps
  the sigmoid; a hidden `pc` atom only binds A.
* **Copy.** `copy(C, A) <= gate(P, C, A), pc(P, C, A)` with ELPRODUCT, summed over panels, then added to
  the output logits `oa` beside the existing decoder.
* **Unchanged without the flag.** The flag-off template is byte-identical to the committed one
  (checked: I-RAVEN, 3×10 values 0–49, 3×3 values 0–9; default, visible selectors, per-attribute
  weights, no rows).
* **Gate initialisation matters.** With gates starting at 0.5, the 29 copies swamped the logits:
  memorising 200 puzzles reached only 34/200 exact at epoch 50 (121/200 at epoch 150). With the −6
  bias it reached 195/200 at epoch 50, and MSE 0.00029 against 0.00086 without copy links.
* **Gotcha.** A weight given as values must be a flat list (`[b] * THERM`) to act as a vector. A
  THERM×1 nested list is read as a matrix and fails with "Incompatible summation of matrix plus vector".

### 20.3 Results (3×10, unseen 6300–7299, epoch picked on 6000–6299)
| values | facet, 60 epochs | facet, 150 epochs | **facet + copy links, 60 epochs** | ARLC learned | best LLM/LRM |
|---|---|---|---|---|---|
| 0–49 | 85.4 (exact 15.0) | 89.1 (23.8) | **97.8 (60.0)**, job 11899232 | 95.0 | – |
| 0–99 | 84.9 (6.4) | 86.3 (6.5) | **96.2 (30.8)**, epoch 45, job 11901474 | 95.0 | DeepSeek R1 84.0 |
| 0–999 | 81.2 (0.0) | 85.4 (0.1) | **94.3 (15.0)**, epoch 45, job 11904250 (11901475 ran out of memory at 180 GB while grounding; 420 GB worked) | 90.6 | DeepSeek R1 82.8 |

At values 0–49, copy links beat 150 epochs of the plain model already at epoch 5 (94.6%). Type became
almost always exact (96% against 60%), size and colour 79% (against 51–52%). There is a slight overfit:
99.3% matched on training puzzles. LLM/LRM numbers are from arXiv 2510.17496 (Table 1, no
confounders); o1 reached 86.0 on a 100-puzzle subset.

**Notebook (2026-10-28).** `notebooks/facet_model.ipynb` §6 shows the copy-link model next to the plain one
on unseen puzzles 6300–6599: 86.3 → 98.0% matched; exact generation per rule: Distribute 11.5 → 96.9,
Constant 94.1 → 100, Progression 79.5 → 89.3, Arithmetic 25.2 → 31.9. On puzzle 6300 (size Distribute,
shift +1, answer 43), the four open size gates (0.92–0.97) are the panels that hold the answer under either
shift: (1,8) and (2,9) hold 43, (1,2) and (2,1) hold 24 and 22 (shift −1). The gates do not pick the
direction; the copied codes cancel between 22 and 43 and the plain path decides those bits. The notebook's
graph picture is now read from the grounded network (`pynl/facet_viz.py`). The earlier picture drew only the
facets and same-position edges, although the model always had its panel, row, context and answer nodes.

### 20.4 What the copy links are, as an inductive bias
Copy links add an inductive bias, a weak and domain-general one:

| built in | still learned |
|---|---|
| the answer's value MAY be one of the context values | WHETHER to copy (gates start closed) |
| any single context panel can be read directly, not only through the row summary | WHICH panel to copy (a gate per panel) |
| a copied value passes through unchanged (exact precision) | WHEN to copy (gates depend on `ans`, where the rule is represented) |

* **No rule is named.** Arithmetic and Progression still go through the old path. The assumption
  "outputs often reuse inputs" is the same one behind pointer networks (Vinyals et al. 2015) and the copy
  mechanism in summarisation (Gu et al. 2016; See et al. 2017).
* **Algorithmic alignment.** In the terms of Xu et al. (ICLR 2020), two of the four rules ARE copies, so
  the model now has a path shaped like the solution.
* **Placement among the prior levels.** It is neither level 3 (no rule set is given) nor level 2, which
  constrains *relations* ("the same relation holds in every row"). The copy links constrain *what the
  answer is built from*. Present it as a separate axis: **priors on operations** (copy) vs **priors on
  rules**.
* **Claim it supports:** a generic operation prior closes most of the gap to ARLC without telling the
  model any rule.

**Caveats to address before claiming it:**
1. **It was designed after looking at the errors** on validation puzzles (Distribute-n failing). Apply it
   everywhere with fixed settings, score the untouched test split once, and justify it as a general
   mechanism (pointer networks), not as "the fix for Distribute".
2. **It bundles two effects:** access to individual panels and exact precision. Ablation: copy through
   the tanh panel nodes `vF` instead of the raw codes. If accuracy holds, access was the key; if it drops,
   precision was.
3. **Test it on I-RAVEN too.** It must at least hold 97.3% there (distribute_nine may gain). A bias that
   helps only the dataset it was designed for is weaker evidence.

Reading list on inductive bias: `thesis/READING_LIST_inductive_bias.md`.

## 21. Symmetries and the copy-link bias, with proofs (2026-10-27)

Questions: are the copy links really an inductive bias? Which symmetries does the facet model have built in,
where, and how?

Files:
- Check script: `pynl/facet_symmetry_check.py`.
- Outputs: `experiments/facet_symmetry_check_11904305.out`;
  `experiments/facet_iravenx_analysis/rule_errors_copy_11904259.out` and `rows_swap_11904346.out`.

### 21.1 Definitions
* **Symmetry.** A transformation g of a puzzle whose effect on the answer is known:
  - *invariance*: f(g·x) = f(x);
  - *equivariance*: f(g·x) = g·f(x).

  A model has it **built in** when it holds for every weight setting. Then training cannot lose it, and the
  data need not teach it.
* **Inductive bias.** Anything other than the data that decides which hypothesis the learner ends with
  (Mitchell 1980). It comes in two kinds:
  - *restriction*: which functions the model can represent at all;
  - *preference*: which of them training reaches, through parameterisation, initialisation and the optimiser.

  A built-in symmetry is a restriction.

### 21.2 Theorem: a template is blind to constants it does not name
Let T be a template and F the facts of one sample. Let σ be a one-to-one renaming of constants that leaves
every constant written in T unchanged. Then every atom a keeps its value:

  val over σ(F) of σ(a) = val over F of a.

*Proof.*
1. A ground rule is a rule r of T with its variables substituted by θ.
2. σ leaves T's constants alone, so σ(rθ) = r(σ∘θ): a ground rule of the same r. Its body holds in σ(F)
   exactly when rθ's body holds in F.
3. σ⁻¹ also leaves T's constants alone, so this pairs the ground rules of the two networks one to one.
4. Weights belong to rules, not to groundings, so paired ground rules have the same weights.
5. The aggregations (AVG, SUM) see a multiset of body values, which σ maps one to one. ELPRODUCT and CONCAT
   combine the atoms of a single body, which σ maps atom by atom.
6. Base case, the facts: val over σ(F) of σ(f) = val over F of f, by the definition of σ(F). Induction over
   the depth of the acyclic ground network then carries this to every atom. ∎

A change to the puzzle is therefore a built-in symmetry whenever:
- it can be written as such a renaming, that is, σ(F) equals the facts of g(puzzle); and
- T names none of the renamed constants.

This is the logic-program form of a GNN's permutation equivariance. Two limits:
- The condition is sufficient, not necessary.
- A named constant only *allows* the symmetry to break. The random-weight checks show that it does break.

Floating-point summation order leaves differences of about 1e-16.

### 21.3 Where the facet model's symmetries come from
Random weights: N(0, 1/fan-in), 3 draws, 70 I-RAVEN val puzzles (300+). The last column is the max |change
in the output|.

| change to the puzzle | as a renaming of | named in T by | built in for | broken by | random weights |
|---|---|---|---|---|---|
| order of objects in a panel | object names `p3o1` | nothing | all variants | – | ≤ 8e-16 |
| rows 1 ↔ 2 | panels `p0..p{n−1}` ↔ `p{n}..p{2n−1}`, rows `r1` ↔ `r2` | per-panel copy gates `gate("p{p}")`; level 1 `vF("p{p}")` | level 2, per-attribute weights, content gates | panel copy gates (0.99), level 1 (0.46) | 3e-16 |
| type ↔ colour (output blocks swap) | `type` ↔ `color` | per-attribute weights only | shared weights (`V.A`) | per-attribute weights (0.83) | 0 |
| components swapped (L-R mirror), attribute outputs | `k0` ↔ `k1` | nothing | all | – | 0 |
| the same, position outputs | not a renaming: the slots are *values* of the one-hot `pos` | – | none | dense occupancy weights | 0.37–0.73 |
| order of the candidates | candidates are not in the facts | – | all (generate, then match) | – | 0 |
| control: columns 1 ↔ 2 | not a renaming: `col(P, "c1")` is named, and row arguments are ordered | – | none | per-column weights | 0.49–0.96 |

Trained checkpoints. "Answers changed" counts puzzles whose chosen candidate changes; "right" is before →
after.
* **I-RAVEN, level 2 (70 puzzles).** Every built-in check: 0 answers changed. Columns 1 ↔ 2: 33 changed,
  68 → 36 right; that swap also changes the true answer under Progression and Arithmetic. Mirrored positions:
  1e-4. This symmetry is not built in, but it was *learned*.
* **I-RAVEN-X 3×10, values 0–49 (300 unseen).**
  - plain model: 0 changed under the row swap;
  - per-panel copy gates: **101/300 changed, 294 → 194 right** (§21.5).

Also invariant by input design: orientation. The facts carry only type, size, colour and position.

### 21.4 Copy links are an inductive bias: a preference, not a restriction
1. **Same data and protocol.** The two runs' `best.json` arguments differ only in `copy_links` and
   `save_dir`. Same 6000 puzzles, targets, MSE, Adam 3e-4, epochs and epoch selection.
2. **No new information.** `pc(P, C, A)` = 4·(2·pt(P, C, A) − 1). `pt` (the AVG of the `fv` codes; number
   from `nm`) is already an atom of the plain model. The links add a path, not information. No rule label is
   used.
3. **Nested hypothesis spaces.** The copy term is Σ_P gate_P ⊙ pc_P, bounded by 4·(3n−1)·max gate. As the
   gate biases go to −∞, the copy model's outputs converge to the plain model's with the same other weights.
   Nothing the plain model can express is excluded, so this is a **preference** bias.
4. **What it prefers** (a sketch, not a proof). "The answer is panel P's value" is reached exactly, for any
   value range, by opening one gate: O(1) weights.

   Through the plain path the value must cross 128 tanh units and a per-bit decoder. Holding v/m in one unit
   and thresholding it at every integer needs slopes of order m. This fits the scaling:
   - plain: 85.4 / 84.9 / 81.2 at values 0–49 / 0–99 / 0–999;
   - copy links: 97.8 / 96.2 / 94.3 (run 11904250 for 0–999).
5. **The gains land where the bias says.** Exact generation per rule on 1000 unseen puzzles, plain → copy:

   | rule | values 0–49 | values 0–99 |
   |---|---|---|
   | Constant | 93.4 → **100.0** | 72.1 → **98.7** |
   | Distribute-n | 9.8 → **86.8** | 4.3 → **71.6** |
   | Progression | 77.4 → 92.9 | 57.8 → 53.0 |
   | Arithmetic | 18.6 → 30.2 | 10.5 → 15.9 |

   At 0–49, puzzles without Arithmetic now match 100.0% (556); with Arithmetic, 95.0% (444).
6. **Not yet excluded: capacity.** Panel gates add 187k weights, 29 × (50·128 + 50): +13% on 1,410k. The
   0–49 gains on Progression and Arithmetic (+15.5, +11.6) are not copies. They may come from:
   - the extra weights;
   - the main path no longer carrying the copy rules;
   - seed variance (one run each).

   Control running (§21.6).
7. **Not a symmetry.** The copy path commutes with any relabelling of the values only while the gates stay
   fixed. The gates read thermometer codes, so the model as a whole is not relabelling-equivariant.

### 21.5 The cost: per-panel gates learned Distribute-n by address
The theorem predicts the break: `gate("p{p}")` names panels. Swapping rows 1 and 2 leaves the correct answer
unchanged under all four rules: for Distribute-n, the answer is the value of row 1 that row 3 lacks. Result
on 1000 unseen puzzles:

| copy model, before → after the swap | matched | exact value |
|---|---|---|
| puzzles with a Distribute attribute (646) | 98.3 → **39.5** | |
| puzzles without one (354) | 96.9 → 97.5 | |
| Distribute-n values | | 86.8 → **1.2** |
| Constant / Progression / Arithmetic values | | 100 → 99.9 / 92.9 → 92.8 / 30.2 → 28.9 |

The plain model is identical before and after.

**What the model learned.** The per-panel gates learned "copy the panel at the address set by the shift
step". They did not learn "copy the value row 3 lacks".
- Why the address does not move: `ctx` is an AVG over the two rows, so `ans` and the gates do not change
  under the swap.
- Why the address rule now fails: the swap also breaks the generator's constant step. Row 3 would be row 2
  shifted by the step from row 1 to row 2 only if 3s ≡ 0 (mod n), which never holds for n = 10.
- So the address rule is right in distribution and wrong off it: a shortcut in Geirhos et al.'s sense, not an
  in-distribution error.

### 21.6 Content gates and the shuffled control (2026-10-27, values 0–49, 60 epochs)
Both flags are off by default; the flag-off and panel-gate templates are byte-identical to before.
* **`--copy-gates content`** (job 11905907):
  - `gate(P, C, A) <= ans(C, A) ⊙ vF(P, C, A)`, with weights per row role (complete row via
    `inrow` + `full`; question row via `inrow(P, "r3")`).
  - The template names no panel and no complete row, so swapping rows 1 and 2 is built in again: 4.4e-16 on
    random weights.
  - Copying becomes choice by content (as in attention), not by address. It has 1/7 of the gate weights.
* **`--copy-shuffled`** (job 11905908): the copied codes come from the next puzzle of the same
  constellation. Only `pc` facts differ (checked). Same weights, no information.

Results, 1000 unseen puzzles, epoch picked on selection (45 for both new runs):

| run | matched | exact, all three attributes |
|---|---|---|
| plain | 85.4 | 15.0 |
| copy links, per-panel gates | 97.8 | 60.0 |
| copy links, content gates | **94.9** | 51.7 |
| shuffled control | **87.3** | 21.5 |

* **The gain is the bias, not the capacity.** The control has the same weights as the copy model and gains
  only 1.9 points over plain. Copy links with real codes gain 12.4.
  - **Bias or just more connections?** The control keeps every connection, gate and weight; only what flows
    through the copy path is uninformative. So of the 12.4 points, about 1.9 can be credited to the extra
    connections and weights, and the rest to the information they carry: the bias "the answer may be a
    context value".
  - **More connections can hurt here.** Gates that start half open swamped the logits: memorising 200
    puzzles reached 34/200 exact, against 195/200 with closed gates (§20.2). Per-panel gates learned a
    shortcut that collapses under a row swap (§21.5).
  - **Open gap.** The shuffled codes are wrong values that can actively mislead, so the control might
    understate what uninformative connections would give. A blank-code control (copied codes all zero) would
    close this; one 5-hour run, not run yet.
* **Content gates keep most of the gain and lose the shortcut.** Row swap, per rule
  (`rows_swap_content_11908129.out`):

  | | before → after the swap |
  |---|---|
  | matched, puzzles with a Distribute attribute (646) | 94.0 → 94.0 |
  | matched, puzzles without one (354) | 96.6 → 96.6 |
  | exact: Constant / Progression / Arithmetic | 100 / 98.2 / 30.0, unchanged |
  | exact: Distribute-n | 59.8 → 59.8 |

  - Every number is unchanged, as the theorem guarantees.
  - Distribute-n exact: plain 9.8, panel gates 86.8 (1.2 after the swap), content gates 59.8. Most of the
    Distribute gain survives without addresses, so the model does choose by content.
  - Progression is higher than with panel gates: 98.2 vs 92.9.
  - Arithmetic stays at 30: no copy rule covers it (§22.4).

## 22. Symmetries of the data: I-RAVEN and I-RAVEN-X (2026-10-27)

Question: which symmetries do the datasets themselves have? §21 asked the same of the model.

Files:
- Check script: `pynl/dataset_symmetry_check.py`. Outputs: `experiments/dataset_symmetry_check_iraven.out`
  (1000 train puzzles per I-RAVEN constellation) and `experiments/dataset_symmetry_check.out` (6000 per I-RAVEN-X
  file; its I-RAVEN candidate line predates the fixed candidate test).
- Generators read: I-RAVEN `~/SRAN/I-RAVEN` (`Rule.py`, `main.py`, `build_tree.py`, `const.py`); I-RAVEN-X
  `~/raven-large-language-models/src/datasets/generation/iravenx_task.py`.

### 22.1 Three levels
Take a transformation g of a complete matrix (answer filled in) that keeps the missing cell in place.
1. **Task symmetry.** Every row still obeys a rule of the same family, possibly with another parameter, so the
   answer moves with g: f(g·x) = g·f(x). Building g into a model costs nothing on the right solution.
2. **Data symmetry.** Also P(g·x) = P(x): the generator produces g·x as often as x. The training data then
   already shows both, so even a model without g sees no reason to tell them apart.
3. **Neither.**

The gap between 1 and 2 is where shortcuts live. A rule that is right on everything the generator produces can
still be wrong on g·x when g·x never occurs. That is §21.5: the per-panel gates found such a rule.

### 22.2 Which rules survive which transformation
The share of rule instances (one attribute of one puzzle) that still satisfy a rule of the same family after g.
Order of the columns: Constant / Progression / Arithmetic / Distribute.
0 instances fail their own label. Non-uniform panels in grids are excluded: there the attribute is noise.

| g | I-RAVEN (type, size, colour, number) | I-RAVEN-X 3×10 (all three value ranges) | I-RAVEN-X 3×3, values 0–9 |
|---|---|---|---|
| rows 1 ↔ 2 | 100 / 100 / 100 / 100 | 100 / 100 / 100 / **0** (0 even with any shift step) | 100 / 100 / 100 / 100 |
| columns 1 ↔ 2 | 100 / 0 / 51 (plus only) / 100 | 100 / 0 / 50 (plus only) / 0 | 100 / 0 / 55 / 100 |
| columns 2 ↔ 3 (operands only) | – | 100 / 0 / **100** / 0 | – |
| matrix transposed | 58 / 3 / 1 / 100 | – | – |
| values reflected, v → max − v | 100 / 100 / **0** / 100 | 100 / 100 / **0** / 100 | 100 / 100 / 0 / 100 |
| columns reversed (moves the missing cell) | 100 / 100 / 100 / 100 | 100 / 100 / 100 / 100 | 100 / 100 / 100 / 100 |

Positions in the grid layouts, 2×2 / 3×3. Constant, union/difference and Distribute are set rules, and they
survive every slot permutation: 100%. Progression on Position survives only some:

| slots mapped by | left-right | up-down | 180° | 90° | diagonal |
|---|---|---|---|---|---|
| Progression on Position | 100 / 0.7 | 100 / 0.7 | **100 / 100** | 0.6 / 0 | 0.6 / 0 |

Union and difference do not survive column reversal (2.5 / 0): a union cannot be undone.

### 22.3 What the generators decide
1. **Rows 1 ↔ 2.**
   - *I-RAVEN: a data symmetry, with one small exception.* Each row starts from an independent resample, and
     the same rule objects build it. Distribute_Three picks its direction 50/50 (`np.random.uniform() >= 0.5`;
     4920 left vs 5023 right over all attributes). For n = 3, the swap maps one direction onto the other.
     - The exception is colour Arithmetic. C12 and C22 may not both be 0 (so + and − stay distinguishable),
       and the generator enforces this by resampling row 2. C12 = 0 in 28.9% (644/2228), C22 = 0 in 20.4%.
   - *I-RAVEN-X 3×10: not even a task symmetry for Distribute.* Constant, Progression and Arithmetic rows
     are i.i.d. Distribute rows are `np.roll(row 1, kδ)` with δ = ±1 (2437 vs 2551). After the swap no step
     fits, because that needs 3s ≡ 0 (mod 10).
     - The answer is still "the value row 3 lacks". So the swap is a symmetry of that weaker rule ("every row
       holds the same n values"), but it leaves the generator's support.
     - 3×3 is different: 3δ ≡ 0 (mod 3), so the swap survives there.
   - *Consequence:* on I-RAVEN and I-RAVEN-X 3×3 the swap is a data symmetry, so building it in only saves
     the data from teaching it. On 3×10 it is a task symmetry, not a data symmetry: the swapped puzzle never
     occurs, but its answer is still "the value row 3 lacks". Building it in costs nothing and removes the
     address shortcut that per-panel gates learned (§21.5).
     - This is the gap §21.5's address rule lived in.
2. **Columns are not a symmetry.** Progression dies under any column swap. What survives is the order-free
   part of Arithmetic, which is a task symmetry but not a data symmetry:
   - **I-RAVEN-X.** A row holds 9 operands and their sum (plus: sum last; minus: sum first). Any permutation
     of the operand columns keeps the rule. But the generator draws the operands one after another from a
     shrinking range, shuffles them, then swaps a zero out of the last operand slot. Result (values 0–49,
     plus), share of zero operands by column: c1 78%, c2 67%, …, c8 55%, c9 0%. Rows are mostly zeros: the
     mean Arithmetic value is 7.4 against about 24.5 for the other rules.
   - **I-RAVEN.** c2 is drawn from what c1 leaves. Means for plus: c1 1.99 vs c2 1.10 (size); 4.41 vs 2.29
     (colour).
3. **Time reversal (columns reversed)** maps Progression d → −d, plus ↔ minus, and the Distribute
   direction. In complete matrices it is a data symmetry of both generators. I-RAVEN-X even builds δ < 0 as
   the mirror of δ > 0, starting from the right. It moves the missing cell, though, so it relates puzzles
   with different blanks: a symmetry for a "fill any cell" model, not for ours.
4. **Value relabelling follows the structure each rule uses.**
   - Constant and Distribute use only equality, so any one-to-one relabelling keeps them.
   - Progression uses order and successor, so it keeps reflection, and translation inside the range.
   - Arithmetic uses addition with 0, so it keeps nothing: reflection 0%.

   In the data, reflection is exact for the first three rules. I-RAVEN-X Progression values average
   24.3–24.7, against a range midpoint of 24.5.
   - Quirk: Distribute draws with Python's `random.randint(0, maxval)`, which includes maxval. Its values
     reach 50 at "values 0–49"; the other rules stop at 49.
5. **Grid positions.** Position-Progression is `(slot + d) mod 4 | 9` in reading order (`Rule.py`). Only the
   grid symmetries that map the reading-order cycle onto itself survive:
   - 2×2: left-right, up-down and 180°;
   - 3×3: 180° only.

   d is ±1/±2 with balanced signs, so these are data symmetries.
6. **Attributes.**
   - *I-RAVEN-X:* size and colour come from the same function, rule list and range. Size ↔ colour is
     therefore exact (1451–1543 per rule each). Type has no Arithmetic, so type ↔ size or colour holds only on
     the non-Arithmetic part.
   - *I-RAVEN:* there is no exact attribute swap. The attributes have 5, 6 and 10 values, size's Arithmetic
     has a +1, and type has no Arithmetic.
7. **Components.**
   - L-R and U-D have identical constraints, and both components show matching rule counts and means: an
     exact symmetry.
   - O-IC and O-IG are not exchangeable. The outer object is always white (colour Constant, 1000/1000) and
     size ≥ 3 (mean 4.0 vs 2.4).
   - Rules act per component, so swapping components is still a task symmetry in every layout.
8. **Candidates.**
   - The answer index is uniform in both datasets.
   - I-RAVEN-X candidate sets are 100% products {a, a′} × {b, b′} × {c, c′}. I-RAVEN's are products in 100%
     of the single-object layouts and 81–89% of the grids.
   - On a product set, flipping any attribute between its two values permutes the set. Such flips reach
     every candidate from every other (a transitive group action), so any function of the candidates alone
     treats all eight alike. Measured, the mode guess gives exactly 12.5% on product sets.
   - On the other grid sets it gives 14.5–15.7%: changing Number resamples positions per candidate. This is a
     small leak.
   - In I-RAVEN-X, the answer holds the larger value of its pair in 50.5–54.2% of attributes.
9. **Nuisance.** Angle is random and rule-free in both generators, and our facts drop it. This invariance is
   exact by input design.

### 22.4 What this means for the models
| symmetry | data symmetry? | facet model |
|---|---|---|
| objects in a panel; candidate order | yes | built in |
| rows 1 ↔ 2 | I-RAVEN yes (but colour Arithmetic); 3×10 no (Distribute leaves the support) | built in, except per-panel gates |
| L-R / U-D components | yes | built in (attributes); learned for positions (1e-4, §21.3) |
| O-IC / O-IG components | task only | built in, harmless |
| size ↔ colour (I-RAVEN-X); type ↔ colour | yes; task only (no type Arithmetic) | built in (shared `V.A`) |
| value reflection / translation | Constant, Progression, Distribute only | not built in (thermometer codes) |
| grid slots: L-R, U-D, 180° (2×2); 180° (3×3) | yes | not built in (dense occupancy weights) |
| Arithmetic operand order | task only (zero pattern) | not built in (per-column weights) |
| columns reversed | complete matrices only | unused |
| columns; transposition; 90° grid rotation | no | correctly not built in |

* Where the model builds in a symmetry the generator also has, the bias is free: the data would teach it anyway.
* Where it builds in a task symmetry the generator lacks (rows in 3×10), the bias removes shortcuts the data
  would allow. Per-panel gates gave that bias up, and §21.5 shows the price.
* The I-RAVEN-X weak spot, Arithmetic (30% exact with copy links), follows a symmetry the model does not
  have. The answer is a sum over operand columns, whatever their order. A structural fix in that direction:
  aggregate the middle columns 2..n−1 with SUM instead of giving each its own weights. That builds the
  symmetry over columns 2..n−1 in for both plus and minus, and treats columns 1 and n as their own roles.

## 23. Torch baselines: an MLP and an all-to-all GNN on the facet model's task (2026-10-28)
Two plain torch models, trained on a GPU, on everything the facet model gets except its architecture:
- the same puzzles and splits;
- thermometer codes as inputs and one sample per component;
- the generated target (mean code per attribute, count code, occupied positions), sigmoid outputs with MSE;
- the epoch picked on the selection split;
- the same metrics: matched (closest candidate, ties wrong) and exact (decoded panel = decoded answer).

Candidates are never seen in training. Before training, the trainer reproduces `bench_hub_grounding.py`'s
targets and decoded answers for the first 20 puzzles of every split.

Code in `baselines/`:
- `prepare_baseline_data.py` (pynl env) writes the puzzles as arrays;
- `train_baseline.py` (nvsa env, GPU) trains;
- `submit_baselines.sh`, `submit_sweep.sh` and `submit_regularized.sh` are the jobs;
- `summarize_baselines.py` prints the tables.

The models:
- **MLP.** The context panels flattened: per panel and position slot, an occupancy bit and three attribute
  codes, plus the component asked for. Three hidden ReLU layers of width H.
- **All-to-all GNN.** One node per context object plus a query node. A node's input is its row, column,
  slot and component one-hots, its three codes and a query flag. Every node sends to every other:
  m_ij = ReLU(W_i h_i + W_j h_j), averaged over j, then h_i = LayerNorm(h_i + ReLU(U [h_i; m̄_i])). Three
  layers; the query node is decoded.

H is the width whose weight count is closest to the facet model's: 264k on I-RAVEN, 1.6M on I-RAVEN-X 3×10
(the copy-link model).

Caveat on exact scores. I-RAVEN-X Distribute draws values up to maxval inclusive, one more than the codes
hold. The baselines score exact generation against the decoded true code, so they are unaffected. The facet
model's I-RAVEN-X exact scores are slightly off, most at 3×3 with values 0–9.

### 23.1 At the facet model's weight count (2 seeds, unseen puzzles, matched % (exact %))
| dataset | facet | facet + copy links | MLP | all-to-all GNN |
|---|---|---|---|---|
| I-RAVEN | **97.3** (69.1) | – | 60.6 ± 0.8 (17.1) | 87.2 ± 1.1 (51.5) |
| I-RAVEN-X 3×3, values 0–9 | **99.2** (92.7) | – | 92.8 ± 0.2 (64.8) | 92.0 ± 0.4 (58.5) |
| 3×10, values 0–49 | 85.4 (15.0) | **97.8** (60.0) | 83.5 ± 0.4 (3.9) | 71.6 ± 0.1 (1.2) |
| 3×10, values 0–99 | 84.9 (6.4) | **96.2** (30.8) | 80.0 ± 2.3 (0.5) | 64.7 ± 5.4 (0.0) |
| 3×10, values 0–999 | 81.2 (0.0) | **94.3** (15.0) | 74.3 ± 3.2 (0.0) | 54.1 ± 3.1 (0.0) |

I-RAVEN by layout, matched %, seed 0 (seed 1 within 3 points, except the MLP on up-down: 72.1):

| model | center | 2×2 | 3×3 | L-R | U-D | in-out | in-out grid |
|---|---|---|---|---|---|---|---|
| facet | 99.3 | 95.5 | 89.8 | 99.5 | 99.7 | 99.6 | 97.8 |
| GNN | 98.7 | 73.3 | 64.6 | 99.0 | 98.7 | 99.7 | 82.0 |
| MLP | 87.8 | 32.7 | 20.3 | 74.0 | 63.0 | 90.5 | 52.1 |

* The GNN shares weights across objects. It matches the facet model wherever a component holds one object,
  and loses 22–25 points on the grids, where Number and Position need same-slot objects compared across
  panels; all-to-all averaging mixes every object with every other.
* The MLP has its own weights per slot: it must learn every slot and both halves of L-R and U-D separately.
* On I-RAVEN-X 3×10 the candidates of an attribute differ between two values, so a rough answer often picks
  the right one: the MLP comes within 2 points of the plain facet model on matching (83.5 vs 85.4). Exact
  generation tells them apart (3.9 vs 15.0, 60.0 with copy links). Exact per rule at values 0–49, MLP / GNN:
  Constant 45 / 36, Progression 43 / 27, Distribute 26 / 11, Arithmetic 6 / 7 (facet: 93 / 77 / 10 / 19,
  with copy links 100 / 93 / 87 / 30). The baselines do not even copy a constant value exactly.

### 23.2 More weights: 4×, 16×, 64× the facet model's (seed 0)
| | weights | I-RAVEN MLP | I-RAVEN GNN | 3×10 0–49 MLP | 3×10 0–49 GNN |
|---|---|---|---|---|---|
| 1× (2 seeds) | 264k / 1.6M | 60.6 (17.1) | 87.2 (51.5) | 83.5 (3.9) | 71.6 (1.2) |
| 4× | 1.1M / 6.4M | 68.9 (25.2) | 86.3 (50.9) | **87.2** (2.5) | **72.3** (1.2) |
| 16× | 4.3M / 26M | **71.0** (27.3) | **88.5** (58.8) | 86.4 (2.1) | 64.9 (1.3) |
| 64× | 17M / 102M | 69.9 (26.0) | 86.5 (54.9) | 84.6 (2.4), lr 1e-4 | 63.3 (0.3) |

* Weights do not close the gap: the MLP levels off near 70% on I-RAVEN (2×2 and 3×3 grids at 37–41 and
  29–34%), the GNN near 88%. Neither reaches the facet model's 97.3% at 64 times its weights.
* The larger models peak earlier (I-RAVEN MLP: epoch 131, 58, 44 at 4×, 16×, 64×): they fit the training
  puzzles faster, not the rules.
* On I-RAVEN-X 3×10 the MLP at 4× matches slightly better than the plain facet model (87.2 vs 85.4), 10
  points below copy links. Its exact generation does not improve with size; the GNN gets worse.
* The I-RAVEN-X MLP at 64× did not move from its first epoch at learning rate 1e-3 (11.5% throughout; run kept
  in `experiments/baselines/runs_failed/`); rerun at 1e-4.

### 23.3 Regularization
Seed 0, at the facet model's weight count (`submit_regularized.sh`, job 11913187, 40 runs).
- **Every run:** LayerNorm after each hidden layer of the MLP, cosine learning rate after 5 warm-up epochs,
  gradient norm clipped at 1.
- **Grid:** dropout 0 / 0.1 / 0.3 × weight decay 0 / 0.05 / 0.5 (AdamW, weight matrices only).
- **Extra runs at dropout 0.1, weight decay 0.05:** GNN message dropout 0.2; row-swap augmentation, I-RAVEN
  only.
- The setting is picked on the selection split.

| | unregularized (2 seeds) | best on selection | setting |
|---|---|---|---|
| I-RAVEN MLP | 60.6 (17.1) | **72.8** (29.9) | weight decay 0.5, no dropout |
| I-RAVEN GNN | 87.2 (51.5) | **91.3** (56.5) | weight decay 0.5, no dropout |
| 3×10, values 0–49, MLP | 83.5 (3.9) | 84.4 (2.6) | weight decay 0.5, dropout 0.3 |
| 3×10, values 0–49, GNN | 71.6 (1.2) | **84.0** (3.3) | weight decay 0.5, dropout 0.3 |

I-RAVEN by layout, best settings:

| model | center | 2×2 | 3×3 | L-R | U-D | in-out | in-out grid |
|---|---|---|---|---|---|---|---|
| facet | 99.3 | 95.5 | 89.8 | 99.5 | 99.7 | 99.6 | 97.8 |
| GNN | 99.3 | 80.6 | 72.9 | 99.1 | 99.3 | 99.6 | 88.4 |
| MLP | 95.2 | 46.6 | 37.8 | 86.7 | 86.8 | 95.3 | 61.3 |

* **Weight decay did the work.** The largest value in the grid, 0.5, won all four cells, so a larger one
  might gain more.
* **Dropout 0.3 ruins the I-RAVEN MLP** (21–28%). GNN message dropout hurts on both datasets at the same
  setting: I-RAVEN 89.3 → 86.1, I-RAVEN-X 66.0 → 62.6.
* **Row-swap augmentation helps a little:** MLP 67.5 → 70.4, GNN 89.3 → 90.5 (same setting without it). The
  facet model has this symmetry built in.
* **The gap narrows but stays.**
  - I-RAVEN: 91.3 vs 97.3, with the grids still 15–17 points behind.
  - 3×10: the regularized GNN matches nearly as often as the plain facet model (84.0 vs 85.4), but generates
    exactly in 3.3% vs 15.0% (60.0 with copy links).
  - Exact per rule, MLP / GNN: Constant 53 / 61, Progression 36 / 42, Distribute 21 / 25, Arithmetic 11 / 10.
    The plain facet model: 93 / 77 / 10 / 19. So the baselines generate Distribute better than the plain facet
    model; copy links reach 87, content gates 60.

### 23.4 Final baselines: every dataset, 2 seeds, and more weights regularized (2026-10-29)
`baselines/submit_final.sh`, job 11914716 (24 runs). The setting picked in §23.3 is run unchanged, not re-tuned
per dataset: LayerNorm (MLP), weight decay 0.5, dropout 0 on I-RAVEN and 0.3 on I-RAVEN-X, cosine schedule with
5 warm-up epochs, clipping at 1. Weight counts as in §23.1 (facet model's; with copy links on 3×10).

**At the facet model's weight count** (matched % (exact %), 2 seeds):

| dataset | facet (best variant) | MLP | MLP, regularized | GNN | GNN, regularized |
|---|---|---|---|---|---|
| I-RAVEN | **97.3** (69.1) | 60.6 ± 0.8 (17.1) | 72.0 ± 1.1 (27.5) | 87.2 ± 1.1 (51.5) | 90.5 ± 1.2 (55.1) |
| I-RAVEN-X 3×3, 0–9 | **99.2** (92.7) | 92.8 ± 0.2 (64.8) | 95.1 ± 0.4 (74.3) | 92.0 ± 0.4 (58.5) | 93.9 ± 0.2 (67.7) |
| 3×10, 0–49 | **97.8** (60.0) | 83.5 ± 0.4 (3.9) | 84.5 ± 0.1 (2.7) | 71.6 ± 0.1 (1.2) | 85.3 ± 1.9 (4.2) |
| 3×10, 0–99 | **96.2** (30.8) | 80.0 ± 2.3 (0.5) | 76.3 ± 1.4 (0.1) | 64.7 ± 5.4 (0.0) | 76.8 ± 2.1 (0.2) |
| 3×10, 0–999 | **94.3** (15.0) | 74.3 ± 3.2 (0.0) | 73.8 ± 0.2 (0.0) | 54.1 ± 3.1 (0.0) | 55.1 ± 1.0 (0.0) |

(facet on 3×10: copy links with per-panel gates; without them 85.4 / 84.9 / 81.2.)

**More weights, regularized** (seed 0):

| weights | I-RAVEN MLP | I-RAVEN GNN | 3×10 0–49 MLP | 3×10 0–49 GNN |
|---|---|---|---|---|
| 1× (2 seeds) | 72.0 (27.5) | 90.5 (55.1) | 84.5 (2.7) | 85.3 (4.2) |
| 4× | 79.4 (40.4) | 92.9 (62.6) | 84.1 (2.6) | 83.4 (2.6) |
| 16× | 81.2 (46.1) | **93.8** (63.7) | 82.3 (1.8) | 72.8 (1.2) |

I-RAVEN by layout, regularized at 16×: GNN 99.5 / 89.6 / 75.2 / 99.8 / 99.7 / 99.9 / 92.8, MLP 97.5 / 57.8 / 45.2 /
96.1 / 97.9 / 99.3 / 74.7 (center, 2×2, 3×3, L-R, U-D, in-out, in-out grid; facet 99.3 / 95.5 / 89.8 / 99.5 /
99.7 / 99.6 / 97.8).

* **Regularization helps on I-RAVEN** (MLP +11.4, GNN +3.3) **and on 3×3** (+1.9 to +2.3), helps the GNN on
  3×10 (+12 to +14 at values 0–49 and 0–99), and does nothing at values 0–999. It costs the MLP 3.7 points at 0–99; selection
  then prefers the unregularized MLP.
* **On I-RAVEN, regularized models keep gaining with weights**, unlike the unregularized ones (§23.2): the GNN
  reaches 93.8 (63.7) at 16× (4.3M weights), the strongest object-level baseline. It is still 3.5 points below
  the facet model at 1/16 of its weights, and 14.6 below on the 3×3 grid.
* **On I-RAVEN-X 3×10 more weights hurt** even with regularization (GNN 85.3 → 72.8 at 16×). Exact generation
  stays at 1–4% at every size and setting.
* **3×3, values 0–9 is the dataset where the baselines come closest:** the regularized MLP matches 95.1
  (74.3), exact per rule 99.7 / 94.7 / 93.4 / 64.4 (Constant / Progression / Distribute / Arithmetic), against
  the facet model's 99.2 (92.7). Few values and three columns leave little to generalize across.

**Are they fair baselines?** Yes, for what they are meant to show: the same puzzles, splits, inputs, targets,
metrics and selection rule as the facet model; matched weights, then 4–64× more; a 40-run regularization grid
picked on selection; 2 seeds where it matters. Limits:
- one setting carried across datasets, tuned on two of them;
- weight decay 0.5 was the edge of the grid;
- the 64× runs are unregularized;
- object granularity is a design choice: the attribute-level GNNs A and B (`thesis/notes_attr_gnn_draft.md`)
  are the stronger torch comparison (I-RAVEN A 95.0 (62.5), B 96.8 (67.9); 3×10 A 95.0 (27.7), B 85.8 (3.5)).

## 24. Column roles: the operand-order symmetry of I-RAVEN-X Arithmetic (2026-10-28)
The prediction from §22.4, tested as plan item E4 (`thesis/THESIS_PLAN_symmetry.md`). I-RAVEN-X Arithmetic is
c_n = c_1 + … + c_{n−1} (plus) or c_1 = c_2 + … + c_n (minus). In both, columns 2..n−1 are interchangeable
operands, while the named-column template gives each column its own weights.

**`--columns roles`** (`bench_hub_grounding.py`):
- The col(P, ·) facts say first / mid / last instead of c1..cn (`col_names`).
- The facet layers get one weight set per role pair: 7 pairs instead of n(n−1) = 90 column pairs.
- `rel`, `ans`, `relp` and `ansp` read the first and the last column through one rule each, and the middle
  columns through ONE rule with the panel as a variable, averaged.
- The template then names no middle column, so permuting columns 2..n−1 is built in.
- For n = 3 the roles are the columns. With the flag off, templates and facts are identical to before
  (checked for all five variants, n = 3 and n = 10).

**Costs nothing for the other rules.** The swap of two middle columns keeps Constant and Arithmetic (§22.2:
100%) and sends Progression and Distribute puzzles out of the support (0%). A model with the symmetry
therefore loses no answer it needs (plan §1, "out of support"). Progression stays computable: the answer is
linear in c_1 and the sum of the middle columns.

**Random weights** (`facet_symmetry_check.py`, 6 unseen puzzles, 2 draws), max |output difference|:

| transformation | named columns, 1,411k w | column roles, 288k w | roles + content gates, 314k w | roles + panel gates, 475k w |
|---|---|---|---|---|
| middle columns 2 ↔ 3 | 0.38 | **4.4e-16** | **1.0e-15** | 0.87 |
| rows 1 ↔ 2 | 0.0 | 3.3e-16 | 1.0e-15 | 1.0 |
| control: columns 1 ↔ 2 | 0.46 | 0.48 | 0.81 | 0.95 |

Per-panel gates name panels and break both symmetries, so the copy variant uses content gates.

**Runs** (`pynl/submit_column_roles.sh`, 60 epochs, the settings of the runs they are compared with; per-rule
evaluation by `pynl/facet_rule_eval.py`, also on the transformed puzzles):
- `facet_iravenx_n10_m50_roles` (288k weights), job 11913673;
- `facet_iravenx_n10_m50_roles_content` (314k), job 11913674;
- `_roles_wide` and `_roles_wide_content`, widths 64,144,288: 1,390k and 1,448k weights, matched to the
  named-column models (1,411k and 1,436k), jobs 11913675–76.

The narrow runs answer "does the symmetry help with a fifth of the weights?", the wide ones "at equal weights?".
Same-script reference on the named-column checkpoints: job 11913677.

Prediction: Arithmetic exact rises (named columns: 18.6 plain, 30.0 with content gates); Constant,
Progression and Distribute stay where they are.

**Reference, named columns** (job 11913677; reproduces §20.4 and §21.6 exactly, which checks
`facet_rule_eval.py`). Unseen puzzles, before → after exchanging middle columns 2 and 3 in every row:

| checkpoint | matched | exact | Constant | Progression | Distribute | Arithmetic |
|---|---|---|---|---|---|---|
| plain, 1,411k w | 85.4 → 70.2 | 15.0 → 11.6 | 93.4 → 93.4 | 77.4 → 73.2 | 9.8 → 4.3 | 18.6 → 17.4 |
| content gates, 1,436k w | 94.9 → 74.9 | 51.7 → 23.9 | 100 → 100 | 98.2 → 84.3 | 59.8 → 13.8 | 30.0 → 30.2 |

The last four columns give the exact value per rule. After the swap, Progression and Distribute puzzles are
out of the generator's support. Their answer by the weaker rule is unchanged: Distribute's value is still the
one row 3 lacks, and Progression's answer is still determined by c_1 and the sum. The named models lose 15–20
points of matching there, which a model with roles cannot lose, by construction.

**Result, plain column roles** (job 11913673, 288k weights, epoch 55 picked on selection; unseen 6300–7299):

| | matched | exact | exact value: Constant | Progression | Distribute | Arithmetic |
|---|---|---|---|---|---|---|
| named columns, 1,411k w | 85.4 | 15.0 | 93.4 | 77.4 | 9.8 | 18.6 |
| **column roles, 288k w** | **94.1** | **29.0** | **98.6** | **94.3** | **22.0** | **40.5** |
| (copy links, content gates, 1,436k w) | 94.9 | 51.7 | 100 | 98.2 | 59.8 | 30.0 |

* **The prediction holds for Arithmetic:** 18.6 → 40.5% exact, more than copy links with content gates reach
  (30.0). Building the operand-order symmetry in did what copy links could not.
* **The other rules improved too,** where I predicted no change: Progression +17, Distribute +12, Constant
  +5. The roles also remove 80% of the weights; per-column weights were spending data on telling columns
  apart.
* **Exactly symmetric on the trained model:** swapping middle columns 2 and 3, or rows 1 and 2, leaves every
  number unchanged (the named models lose 15–20 points of matching, see above).
* Matching 94.1% approaches the copy-link models without any copy path. Distribute stays the weak rule: it
  needs a single panel of row 1, which the copy links supply.

**Result, column roles + content copy gates** (job 11913674, 314k weights, epoch 60):

| | matched | exact | Constant | Progression | Distribute | Arithmetic |
|---|---|---|---|---|---|---|
| named columns + content gates, 1,436k w | 94.9 | 51.7 | 100 | 98.2 | 59.8 | 30.0 |
| **roles + content gates, 314k w** | **96.3** | 43.3 | 99.6 | 94.8 | 42.8 | **37.5** |

* Matching and Arithmetic improve over the named-column copy model (+1.4, +7.5); exact generation falls (−8.4),
  all of it in Distribute (−17) and a little in Progression (−3.4). A possible reason, not tested: the gates
  read `ans`, which with roles sees the question row's middle columns only as an average, so it may carry less
  about which value is missing; with named columns `ans` had a weight per column.
* Again identical under the middle-column and row swaps.

**Row-length transfer of the roles models** (job 11914475): trained on rows of 10, scored on 1000 puzzles with
rows of 5 and 20 (values 0–49; same weights, the template names no column):

| rows of | model | matched | exact | Constant | Progression | Distribute | Arithmetic |
|---|---|---|---|---|---|---|---|
| 10 (trained) | roles / + content gates | 94.1 / 96.3 | 29.0 / 43.3 | 98.6 / 99.6 | 94.3 / 94.8 | 22.0 / 42.8 | 40.5 / 37.5 |
| 5 | roles / + content gates | 65.6 / 66.3 | 4.0 / 5.2 | 99.0 / 99.8 | 23.8 / 9.8 | 4.2 / 16.7 | 4.3 / 5.1 |
| 20 | roles / + content gates | 50.1 / 49.2 | 4.6 / 4.6 | 98.7 / 99.8 | 11.0 / 2.8 | 7.3 / 11.2 | 6.5 / 4.5 |

* Only Constant transfers. Everything that depends on how many columns a row has breaks: Arithmetic (a sum of
  n−2 middle values, read as their average), Progression (c_1 + (n−1)·step) and Distribute (n values).
  As predicted for Arithmetic; Progression breaking was not predicted, but follows from the same count.
* So the roles generalize over the ORDER of the middle columns but not over their NUMBER. That is the test
  the row fold has to pass (§25).

**At the named-column model's weight count** (`--columns roles --widths 64,144,288`, jobs 11913675–76, 60 epochs):

| 3×10, values 0–49 | weights | matched | exact | Constant | Progression | Distribute | Arithmetic |
|---|---|---|---|---|---|---|---|
| named columns | 1,411k | 85.4 | 15.0 | 93.4 | 77.4 | 9.8 | 18.6 |
| column roles | 288k | 94.1 | 29.0 | 98.6 | 94.3 | 22.0 | 40.5 |
| column roles, wide (epoch 55) | 1,390k | **94.9** | 28.5 | 97.1 | 94.8 | 20.9 | 36.6 |
| roles + content gates | 314k | **96.3** | **43.3** | 99.6 | 94.8 | 42.8 | 37.5 |
| roles + content gates, wide (epoch 30) | 1,448k | 93.1 | 30.0 | 99.4 | 80.2 | 27.5 | 25.1 |

* **At equal weights the operand-order symmetry is worth 9.5 points of matching** (94.9 vs 85.4) and 13.5 of exact.
  The comparison in the table above was at a fifth of the weights; this one removes the capacity question.
* **Width does not help the roles model**: 4.8× the weights gives 94.9 vs 94.1. With content gates the wide model is
  worse (93.1 vs 96.3; its selection picked epoch 30, the final epoch had 94.3 / 32.4).
* Both wide models are identical under the middle-column and row swaps, as built.
* Row-length transfer of the wide models (job 11914476): rows of 5 63.9 / 67.7, rows of 20 49.3 / 50.1 (without /
  with content gates); again only Constant survives (97–100% exact). Width changes nothing here either.

## 25. A recursive row fold: the step symmetry and other row lengths (2026-10-28)
Plan E4, suggested by the supervisor's paper (`thesis/PAPER_NOTES_sourek2021.md`): recursion is the construct
only NeuraLogic expresses. `--columns fold` (`bench_hub_grounding.py`) keeps the column roles everywhere
except the attribute channel's reading of a row, which becomes a fold:

```
acc(P, C, A)  :- col(P, first), vF(P, C, A)                       start
acc(Q, C, A)  :- next(P, Q), Wh·acc(P, C, A) + Wx·vF(Q, C, A)      one step per column, tanh
rel(Rw, C, A) :- acc of the row's last panel + vF of its first panel
ans(C, A)     :- acc of the question row's last context panel + ctx
```

(Each step is written as two rules so chain pruning keeps its tanh; the start state has one rule, so it stays
linear in vF.)
* **Symmetry it builds in:** one transition for every step, the same rule applied at every position of the row
  (Progression's step, a running sum). It gives up the operand-order symmetry: a fold reads the row in order.
  Random weights: rows 1 ↔ 2, size ↔ colour and object order exact (≤4.4e-16); middle columns 2 ↔ 3 change the
  output (0.041), as they should.
* **Row length:** neither the fold nor the roles name a column, so their weight layout is identical for rows of
  5, 10 and 20 (roles 288,328 weights, fold 304,712): a checkpoint trained on 10 runs on any length.
  Named-column models cannot.
* **Toy check** (PyNeuraLogic 0.9.2): a two-rule recursive template trained on rows of 3–5 values summed rows of
  8 and 10 to a mean error of 0.009 (sums 0–90). It needed the structural atoms hidden (§17); visible, it
  learned nothing.
* **Test sets at other lengths:** IBM's `iravenx_task.py` (`~/raven-large-language-models/...`) with
  `--n 5` / `--n 20 --maxval 50 --nconf 0`, seed 1234: `data/iravenx/center_single_shuffle_n_{5,20}_maxval_50.json`.
  Rows of 30 would break its Progression sampler at values 0–49. Longer rows also narrow Progression's start
  values, so the value distribution shifts as well.

**Runs** (`submit_column_roles.sh`, RUN=fold / fold_content, 60 epochs, jobs 11914473–74): trained on 3×10, scored
on unseen rows of 10, and on rows of 5 and 20. The roles checkpoints get the same row-length scoring (jobs
11914475–76).

**Predictions:**
- **Rows of 10:** the fold at least matches the roles on Progression and below them on Arithmetic (order
  sensitivity).
- **Other lengths:** the roles average over the middle columns, which hides their number, so roles models
  should break on Arithmetic at rows of 5 and 20. The fold should transfer better.
- **Risk:** precision over 9–19 tanh steps.

**Snapshot mid-training** (2026-10-29; checkpoints copied to `experiments/snapshot_fold_30`,
`snapshot_fold_content_20`; job 11914793). Matched (exact), and exact value per rule
(Constant / Progression / Distribute / Arithmetic):

| model, rows of | 10 (trained) | 5 | 20 |
|---|---|---|---|
| fold, epoch 30 | 91.1 (22.8); 99.9 / 94.8 / 15.1 / 26.7 | 49.6 (0.2); 38.6 / 9.9 / 9.2 / 0.0 | 48.9 (4.6); 99.7 / 5.9 / 12.4 / 2.6 |
| fold + content gates, epoch 20 | 93.2 (32.1); 99.9 / 86.5 / 28.8 / 21.5 | 57.5 (6.8); 97.4 / 19.2 / 16.9 / 2.0 | 48.4 (9.5); 100 / 10.5 / 25.2 / 2.8 |
| column roles (final, for reference) | 94.1 (29.0); 98.6 / 94.3 / 22.0 / 40.5 | 65.6 (4.0) | 50.1 (4.6) |

* At rows of 10 the fold equals the roles on Constant and Progression, and is below them on Arithmetic (26.7 vs
  40.5), as predicted from its order sensitivity.
* **It does not transfer to other row lengths either.** At rows of 5 the plain fold even loses Constant (38.6%).
  Trained only on rows of 10, nothing asks the recurrence to mean the same after 4 steps as after 9: the
  readouts learned to decode the state reached after exactly 9 steps. The toy (§25) transferred because it was
  linear and trained on mixed lengths (3–5).
* So recursion makes length transfer *possible* (same weights at any length), not automatic. The fair test
  trains on mixed row lengths (e.g. 5, 7, 10) and tests on 20; both the fold and the roles templates can train
  on mixed lengths as they are.

**Final** (60 epochs; epoch picked on selection: fold 60, fold + content gates 30). Same format:

| model, rows of | 10 (trained) | 5 | 20 |
|---|---|---|---|
| fold | 92.2 (28.0); 99.9 / **99.2** / 15.1 / 36.2 | 53.8 (0.4); 41.7 / 13.4 / 9.8 / 1.2 | 55.1 (5.9); 99.1 / 2.8 / 15.6 / 3.0 |
| fold + content gates | 93.7 (35.0); 99.6 / 89.8 / 32.4 / 29.4 | 61.0 (8.1); 94.9 / 22.4 / 19.8 / 1.0 | 60.5 (10.1); 99.9 / 14.5 / 27.5 / 3.4 |
| column roles | 94.1 (29.0); 98.6 / 94.3 / 22.0 / 40.5 | 65.6 (4.0) | 50.1 (4.6) |
| column roles + content gates | 96.3 (43.3); 99.6 / 94.8 / 42.8 / 37.5 | 66.3 (5.2) | 49.2 (4.6) |

* **Progression: 99.2% exact, the best of any model** (copy links with content gates 98.2, roles 94.3). The step
  symmetry the fold builds in is Progression's own.
* **Arithmetic: 36.2, below the roles (40.5)**, as predicted from order sensitivity; Distribute unchanged (15.1).
  Overall the fold is 1.9 points below the roles at a few more weights (305k vs 288k).
* **Swaps:** rows 1 ↔ 2 identical (built in); middle columns 2 ↔ 3 cost 0.5 (fold) and 0.6 points (fold +
  content gates), against 15 for named columns. A fold *can* tell the middle columns apart but learns not to;
  the data teaches the symmetry the template does not have.
* **Row length: still no transfer.** Rows of 5 give 54–61, rows of 20 give 55–61; only Constant survives (Arithmetic
  1–3%). At rows of 20 the fold models are 5–11 points above the roles, at rows of 5 they are 5–12 below, but none
  is near the trained length. The conclusion of the snapshot holds: one trained length does not make the
  recurrence length-independent. Mixed-length training is the open test.

## 26. Symmetry ablations: components and attributes (2026-10-29)
Plan E1/E2 for two symmetries the facet model has built in but whose benefit was never measured: the two
components of left-right and up-down layouts, and the attributes (size ↔ colour is a data symmetry of
I-RAVEN-X, §22). The question is what each symmetry buys in accuracy, in training data, and on transformed
puzzles.

**How the symmetry is removed** (`--label-components`, `--label-attributes`, `build_facet_template(labels=...)`):
- One-hot facts name what the symmetry makes interchangeable: `cid(k0) = [1, 0]`, `cid(k1) = [0, 1]` (into
  every panel-component summary `vF` and `vP`), or `aid(type|size|color|number)` (into `vF`).
- The weights stay shared; only 0.4–1k weights are added. So this removes the symmetry at nearly equal
  capacity, unlike `--per-attr-weights`, which triples the weights.
- Random weights (`facet_symmetry_check.py`): component labels break the mirror (0.37), not type ↔ colour (0);
  attribute labels break type ↔ colour (0.41), not the mirror (0); rows stay exact. Defaults unchanged
  (checked for all template variants, I-RAVEN and 3×10).

**Runs** (`pynl/submit_symmetry_ablation.sh`, jobs 11914707–15; scored by `facet_rule_eval.py`, which now also
handles I-RAVEN checkpoints and the transforms `mirror` and `sizecolor`):
- Components: I-RAVEN left-right + up-down, default vs labelled, at 100 / 1000 / 6000 puzzles per layout
  (300 / 120 / 60 epochs); scored as is and mirrored.
- Attributes: I-RAVEN-X 3×10 values 0–49 with column roles, default (6000: `facet_iravenx_n10_m50_roles`;
  600 new) vs labelled, at 600 / 6000 puzzles; scored as is and with size ↔ colour swapped.

**Predictions:**
- **Full data:** little difference. The data has these symmetries, so it teaches them; the torch baselines
  learned the left-right symmetry from I-RAVEN without it built in.
- **Small data:** the default (symmetric) model ahead.
- **Transformed puzzles:** the default model identical by construction; the labelled one may drop.

**Results** (matched % (exact %); I-RAVEN: val 300–1299 per layout; I-RAVEN-X: unseen 6300–7299; epoch picked on
selection puzzles):

| data | puzzles | symmetric | labelled | symmetric, transformed | labelled, transformed |
|---|---|---|---|---|---|
| I-RAVEN L-R + U-D | 100 per layout | 86.6 (40.4) | 87.3 (41.3) | 86.6 (40.4) | 87.1 (41.1) |
| | 1000 | 99.3 (95.6) | 99.6 (95.7) | 99.3 (95.6) | 99.6 (95.7) |
| | 6000 | 99.9 (97.7) | 99.8 (98.7) | 99.9 (97.7) | 99.8 (98.7) |
| I-RAVEN-X 3×10 0–49, roles | 600 | 66.3 (16.4) | 64.6 (14.1) | 66.3 (16.4) | 65.1 (14.3) |
| | 6000 | 94.1 (29.0) | 93.9 (28.5) | identical | 93.9 (29.1) |

Transformed = mirrored (L-R + U-D) or size ↔ colour swapped (I-RAVEN-X). Exact value per rule, I-RAVEN-X: at 600,
symmetric 99.8 / 84.7 / 5.3 / 8.7 vs labelled 98.3 / 77.3 / 5.0 / 8.1 (Constant / Progression / Distribute /
Arithmetic); at 6000, 98.6 / 94.3 / 22.0 / 40.5 vs 99.3 / 94.8 / 23.4 / 42.7.

* **Removing the exact symmetry costs nothing measurable.** All differences are within about 2 points, and at
  L-R + U-D the labelled model is, if anything, ahead (one seed each). The small-data prediction failed; only
  I-RAVEN-X at 600 puzzles points the predicted way (−1.7 matched, −7 on Progression).
* **The labelled models stay almost invariant after training.** Mirroring or swapping size and colour moves their
  matching by 0.2–0.5 points, although with random weights the labels break the symmetry clearly (0.37, 0.41). The
  data is symmetric, so the labels carry no information about the answer and training does not use them.
* **What the ablation keeps is the larger part of the bias.** The labels break the symmetry but leave the weights
  shared: both components (both attributes) are still read by the same rules, and each rule still trains on both.
  So this measures exact invariance on top of sharing, and finds it worth little here. The sharing itself is
  removed only by separate weights per component or attribute (`--per-attr-weights`, 3× the weights); the one
  earlier test of that (2000 distribute_nine puzzles, dim 32: separate 75.4 vs shared 73.6 matched, job 11877948)
  was within noise too.
* **Contrast with the symmetries that did matter:** the row symmetry (copy links with per-panel gates lose 38
  points on swapped rows, §21.5) and operand order (column roles, §24: +8.7 matched, a fifth of the weights). Both
  are symmetries where the alternative has many more weights or a shortcut that conflicts with the data. Components
  and attributes in the facet model are neither: two components and four attributes, sharing kept either way.
* **Not a conclusion on its own:** see §26.1, where removing the weight sharing does cost accuracy at small data.

### 26.1 Separate weights per component and per attribute
The labels above break the invariance but keep the sharing. Here the sharing goes too:
- **Components:** `--per-comp-weights` (`build_facet_template(shared_comps=False)`) writes every rule with weights
  once per component, `k0` and `k1`, with the component as a constant (and `ocomp(O, k)` in the object-level rules,
  which name no component). Weight-free rules and all metadata are unchanged.
- **Attributes:** `--per-attr-weights` (`shared=False`), which existed: every attribute rule once per attribute.
- All 18 earlier template variants are unchanged, string for string (I-RAVEN and 3×10; default, copy links, both
  gates, level 1, labels, roles, fold, per-attribute).
- Random weights (`facet_symmetry_check.py`): per-component weights break the mirror (0.78) and keep rows and
  type ↔ colour exact; per-attribute weights with roles break type ↔ colour (0.90) and keep rows and middle
  columns exact (≤4.4e-16).

Two arms each (`submit_symmetry_ablation.sh`, arms `sep` and `sepm`):

| | shared (default) | separate, same widths | separate, matched weights |
|---|---|---|---|
| L-R + U-D, components | 266,528 (32, 64, 128) | 533,056 (2×) | 265,532 (22, 45, 90) |
| I-RAVEN-X roles, attributes | 288,328 (32, 64, 128) | 932,190 (3.2×) | 288,214 (18, 35, 69) |

Same training sizes and epochs as the label runs (L-R + U-D 100 / 1000 / 6000, I-RAVEN-X 600 / 6000).

Jobs 11916358–67 (sep: 58–62, sepm: 63–67).

**Results** (matched (exact); all 10 runs):

| data | puzzles | shared | labels | separate, same widths | separate, matched weights |
|---|---|---|---|---|---|
| L-R + U-D | 100 | **86.6** (40.4) | 87.3 (41.3) | 75.6 (22.2) | 74.6 (19.6) |
| | 1000 | **99.3** (95.6) | 99.6 (95.7) | 97.9 (84.5) | 98.5 (84.5) |
| | 6000 | 99.9 (97.7) | 99.8 (98.7) | 100.0 (97.8) | 100.0 (98.8) |
| I-RAVEN-X roles | 600 | **66.3** (16.4) | 64.6 (14.1) | 52.5 (4.1) | 53.0 (3.6) |
| | 6000 | **94.1** (29.0) | 93.9 (28.5) | 90.5 (17.8) | 88.1 (17.6) |

* **Removing the sharing costs what removing the invariance did not.** Separate weights per component lose 11–12
  points of matching and 18–21 of exact at 100 puzzles per layout, 11 points of exact at 1000, and nothing at
  6000. Separate weights per attribute lose 13–14 points at 600 puzzles and 6 at 6000 (matched count).
* **It is the sharing, not the weight count:** with twice or three times the weights (same widths) the separate
  models are no better than at the matched count (75.6 vs 74.6; 52.5 vs 53.0).
* **The gain shrinks with data**, as predicted for a data symmetry (§21–§22): the data teaches the symmetry
  eventually, and at 6000 puzzles per layout the components no longer matter.
* So the label ablation (§26) measured the wrong thing: an invariance on top of shared weights. The inductive
  bias of these symmetries is the weight sharing they allow.

* The separate attribute weights at the same widths (932k) peaked at the last epoch, 60: 90.5 (17.8), still below
  the shared 94.1 (29.0); exact per rule 93.4 / 80.4 / 16.2 / 25.7 against 98.6 / 94.3 / 22.0 / 40.5. The
  components at 6000 are level in every arm (99.9–100.0).


## 27. The value axis: values as constants, a convolution along them (2026-10-29)
`pynl/value_axis.py` (I-RAVEN-X, one object per panel). Every value 0..maxval is a constant `v0..vM`, and a
panel's attribute is a fact `x(P, A, v17)`. Background facts give the value axis its structure: `off(U, V, k)`
for V = U + k (k = −2..2) and `plus(U, W, V)` for U + W = V. No rule names a value.

**Template** (10,817 weights at width 32; the facet model with column roles has 288k):
- `h(P, A, V) :- x(P, A, U), off(U, V, k)`, one weight per offset: a kernel along the value axis, the same at every
  value. h says "V is k above P's value".
- `pv(P, Q, A, V)` compares neighbouring panels at the same value; `pd(P, Q, A)` sums it over V. So a pair's
  descriptor depends only on the offset between their values, not on the values.
- `rd(Rw, A)`: the row's pairs averaged; counts per row and value `pr`, distinct values `nd`, overlap with the
  question row `ov`.
- `acc`, `accm`: exact running sums along a row, `acc(Q, A, V) :- next(P, Q), acc(P, A, U), x(Q, A, W),
  plus(U, W, V)` with a product of the operands' values summed over `plus`, i.e. the convolution of their value
  distributions. `pchk` / `mchk`: does a complete row satisfy plus (last = sum of the rest) or minus (first = sum
  of the rest). `sa`, `ma`: the question row's answer under plus and under minus.
- `ctx(A)`: which rule, from the value-free descriptors and checks. `o(A, V)`: two tanh layers shared across
  all V (a 1×1 convolution) over ctx, h of the question row's last panel, the counts, sa and ma at V. Softmax
  over the values per attribute; MSE on the one-hot answer.
- Number and position are constant in I-RAVEN-X; for scoring they are taken from the last panel. Scoring turns the
  value distributions into thermometer codes, so matched and exact are `bench_hub_grounding.evaluate`'s.

**Checks:**
- Running sums, plus/minus checks, counts and distinct values on real Arithmetic puzzles: exact (45 of 45
  queries).
- Random weights (`--check-shift`), every context value of one attribute moved by k ∈ {+1, +3, −2}, in puzzles
  whose values stay 2 away from the ends of the range: without plus the output moves by k exactly (≤3e-17, all
  130–181 puzzles per case); with plus it does so in 94–97% of them, and differs (≤2.8e-3) where the shift changes
  a sum, as addition should. Control: the same outputs compared without moving them differ by 2.8e-2 on
  average, so the exact match is not a flat output.
- The weights do not depend on the value range (the same 29 weight tensors at 0–50 and 0–100), nor on the row
  length: a checkpoint trained on values 0–49 runs on 0–99 and on rows of 5 or 20.

**Why it should help, and the Arithmetic question.** The facet model reads a 50-bit thermometer code with a
weight per bit: "+2 from 17" and "+2 from 33" are different inputs to learn. Here they are the same computation,
moved along the axis, so every training puzzle teaches every value. For Arithmetic the global shift is not a
symmetry ((u+k) + (w+k) = (u+w) + 2k), and the template does not impose it: `plus` is not preserved by the shift.
What addition does have is equivariance per operand: shift one operand by k and the sum moves by k (shift the
operands by k₁ and k₂, the sum moves by k₁ + k₂). A convolution of value distributions has exactly this property,
and `plus` computes one. So the plus pathway is the same at every value without being shift-invariant.
`plus` is background knowledge (like NVSA's binding); what is learned is when to use it.

**Runs** (`pynl/submit_value_axis.sh`): smoke (job 11916971); then the full model, without plus, and at 600
puzzles; each scored also on values 0–99 and rows of 5 and 20.

**Smoke run** (500 training puzzles, 20 epochs, 300 unseen puzzles 6300..; job 11916971): best on selection at
epoch 5, unseen **99.3% matched, 98.3% exact**; exact value per rule Constant 100 / Progression 100 / Distribute
100 / Arithmetic 98.5. Every epoch from 5 to 20 is within 0.7 points. The facts hold only the 29 context panels.
For scale: the column-roles facet model reaches 94.1 (29.0) with 6000 puzzles.

* What does the work is not settled by this run: the model has the value symmetry *and* primitives the facet
  model lacks (exact sums, per-row value counts, offsets of ±2 from the last panel). `noplus` takes out the sums;
  the value-symmetry ablation (weights per value instead of per offset) is not built yet.

Jobs 11917071–74 (plus, noplus, plus600, plus100); smoke checkpoint on values 0–99 and rows of 5 / 20: 11917075.

**Ablation of the value symmetry: weights per value** (`--per-value`, `build_value_template(per_value=True)`).
Every rule with weights whose atoms carry a value is written once per value constant, with its own weights:
the kernel (`h`, per value and offset), the comparison and pooling (`pv`, `pd`), and the scoring layers
(`o`, `o2`, `s`). The value-free part (`rd`, `ctx`), the exact sums and the counts are unchanged. So the
template names every value, as the facet model does with a weight per code bit, and the primitives stay.
- 335,475 weights with plus (332,147 without), against 10,817 shared; the facet model with roles has 288k.
- The default templates are unchanged, string for string (with and without plus).
- Cannot run on another value range (weights tied to values 0–50); rows of 5 and 20 still work.
- Random weights, without plus: shifting the values no longer moves the output (exact in 0 of 130–181 puzzles
  per case, max difference 0.2; the shared template: all exact, ≤8e-17).
- Runs at 6000 / 600 / 100 puzzles: jobs 11917285–87.

**Results so far** (unseen 6300–7299; matched (exact); exact value per rule C / P / D / A):

| run | weights | 3×10, 0–49 | per rule | values 0–99 | rows of 5 |
|---|---|---|---|---|---|
| 100 puzzles | 10,817 | 99.0 (95.5) | 99.9 / 98.5 / 100 / 97.2 | 98.1 (96.4) | 59.7 (32.7) |
| 600 puzzles | 10,817 | 99.9 (98.3) | 100 / 100 / 100 / 99.8 | 100.0 (99.5) | 75.8 (52.5) |
| 6000 puzzles, without plus (epoch 45) | 10,689 | 84.0 (58.4) | 100 / 100 / 100 / 11.1 | | |

* Without `plus` the other three rules stay perfect and Arithmetic falls to 11%, near the share of answers that
  local features can reach (within ±2 of the last panel: 19%; in rows 1–2: 27%). So the sums come from the
  given relation; the convolution with a ±2 kernel cannot add. A sum shifts by an amount that depends on the
  other operand: a kernel spanning the range, tied to the values through the position of zero.

**More results** (2026-10-29):

| run | 3×10, 0–49 | per rule C / P / D / A | values 0–99 | rows of 5 | rows of 20 |
|---|---|---|---|---|---|
| 6000 puzzles (epoch 5) | 100.0 (98.4) | 100 / 100 / 100 / 100 | 100.0 (99.6) | 89.8 (76.6) | 98.6 (62.7) |
| 6000, without plus (epoch 45) | 84.0 (58.4) | 100 / 100 / 100 / 11.1 | 78.9 (58.7) | 57.0 (24.4) | 80.1 (58.6) |
| weights per value, 6000 (epoch 40) | 98.4 (95.7) | 100 / 100 / 100 / 94.9 | cannot run | 55.7 (17.3) | 98.9 (96.9) |
| weights per value, 600 | 92.7 (79.5) | 98.0 / 87.8 / 97.7 / 86.8 | cannot run | 63.6 (23.4) | 87.6 (64.4) |
| weights per value, 100 | 38.3 (3.6) | 60.5 / 30.8 / 21.4 / 22.3 | cannot run | 27.3 (0.6) | 33.5 (3.4) |

* **The value symmetry is what makes the model data-efficient.** With the same primitives (sums, counts, offsets),
  weights per value instead of per offset drop matching from 99.0 to 38.3 at 100 puzzles and from 99.9 to 92.7 at
  600 (exact 95.5 → 3.6 and 98.3 → 79.5), with 31 times the weights. Arithmetic too (97.2 → 22.3 at 100): the
  sum is given, but reading it at each value has to be learned separately for each value.
* **Weights per value at 6000 puzzles:** 98.4% matched (95.7 exact), Arithmetic 94.9, best on selection at
  epoch 40 of 60. That is where shared weights are at 100 puzzles (99.0, 95.5 exact): without the value
  symmetry the model needs about 60 times the data. Rows of 5: 55.7 (17.3), worse than at 600 puzzles (63.6);
  per-value scoring layers fit the sums of rows of 10 more tightly.

* **Rows of 20 transfer better than rows of 5.** Shared weights: 98.6 (62.7) at 6000 puzzles, 92.4 (84.4) at
  600, 97.4 (82.3) at 100; Constant, Distribute and Arithmetic stay at 99–100% exact, Progression drops (51.2 /
  89.2 / 81.7). Weights per value: 98.9 (96.9) at 6000 (Progression 99.8, Arithmetic 96.6), 87.6 (64.4) at 600,
  33.5 (3.4) at 100. At 100 and 600 puzzles the value symmetry wins here too; at 6000 the per-value model is the
  best on rows of 20 but the worst on rows of 5 (55.7). Scoring needed the named last panel in `sa`/`ma` (the
  `qlast(P), acc(P, …)` join did not finish in 10 h with the BUP grounder).
