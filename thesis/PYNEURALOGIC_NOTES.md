# Porting the RAVEN GNN solver to PyNeuraLogic — findings

Working notes from rebuilding the PyNeuraLogic strand of this project from scratch,
with the goal of matching the torch GNN's I-RAVEN accuracy.

**Status (2026-10-19) — supersedes every number further down.**

| I-RAVEN test, all 7 constellations, 2000 each | accuracy |
|---|---|
| **PyNeuraLogic level 2, final** (`experiments/max_data_L2`, epoch 126 of 130), with XML labels | **68.49%** |
| same checkpoint, **end to end** through the ResNet-18 frontend (§11) | **68.46%** |
| PyNeuraLogic level 2 at epoch 62 (earlier milestone) | 64.00% (63.96% end to end) |
| torch GNN, both position fixes, 3 seeds (`experiments/gnn_FIXREL_s{42,43,44}`) | 45.25% ± 12.4 (41.74 / 35.01 / 59.02) |
| torch GNN, published (wrong positions, not reproducible) | 68.81% |
| PyNeuraLogic level 2 at 2000/constellation (`allcons_v2_L2`) | 51.78% |
| **symbolic PyNeuraLogic, rules** (§15): test, with XML labels / end to end | **98.33% / 96.31%** |

* **Final run:** 130 epochs in chained 3-day links, finished 2026-10-19; best validation 70.86% at
  epoch 126; the test split was scored once on that checkpoint. Per constellation (XML / end to end):
  center_single 85.70 / 85.70, distribute_four 40.85 / 40.85, distribute_nine 38.20 / 38.05,
  left/right 83.10 / 83.10, up/down 84.00 / 84.00, in/out 90.25 / 90.25, in_distribute_four
  57.30 / 57.30. LaTeX: `table_iraven_final_test.tex`.
* **Against the published torch model:** level overall (68.49 vs 68.81) and +6 to +10 on the four
  constellations the position bug never touched (85.70 vs 75.40, 83.10 vs 73.40, 84.00 vs 74.20,
  90.25 vs 84.45). Its numbers on the other three come from corrupted inputs.
* **Data was the binding constraint:** 2000/constellation saturates near 54.4% validation. The full-data
  model is a warm start from that run ("epoch 62" = 38 epochs at 2000 + 24 at 6000); no from-scratch
  full-data run at dim 128 exists. Validation was still creeping up at the end (10-epoch means
  68.5 → 69.4% over epochs 91–130) while the training loss kept falling.
* **One seed.** The torch GNN varies 35–59% across seeds; PyNeuraLogic has a single run.
* Checked not to be an artefact (`sanity_checks.py`, each check with a control that proves it can fail):
  0 train/test duplicates, uniform answer index, the candidates-only shortcut at chance on I-RAVEN
  (90–100% on RAVEN), XML labels match the rendered images, the model is exactly invariant to candidate
  order, and a foreign or blank context collapses it to chance. Full audit list: §14.
* **Running (launched 2026-10-19):** GNN ablations for the layout weak spot (§16).

**Bottom line (porting stage, center_single only).** The relational structure transfers
cleanly; the optimisation machinery does not. A pure PyNeuraLogic port reaches 27.2% on `center_single`; moving the numerical
parts (embeddings, normalisation, loss, optimiser) into PyTorch while keeping the same
logic template reaches 52.9%. Along the way, a **position-decoding bug in the original
torch data pipeline** turned up that makes ~32% of puzzles in three constellations
literally unanswerable.

---

## 1. The torch baseline — what actually won

Numbers pulled directly from `best_val_acc` inside each `experiments/*/model_best.pt`,
not from the summary doc.

| | best config | val | test |
|---|---|---|---|
| RAVEN | `gnn` + `object_only` + `unified` | 96.45% | 96.41% |
| **I-RAVEN** | `gnn` + `granular` + `object_only` | **68.83%** | **68.81%** |

`gnn` = `RelGNLayer`, plain relational message passing, **no attention**.

* **GAT failed outright** — every variant landed 15.17–26.53% against a 12.5% chance
  floor. Several runs stopped at epoch 1/15/16/54, so read this as "did not train",
  not "attention doesn't help".
* **GCN was mid**: 87.49% RAVEN, 64.49% I-RAVEN.
* **`object_only` is the single biggest factor**: dropping panel super-nodes moves RAVEN
  38.40% → 81.09% and I-RAVEN 34.38% → 68.44%. The `same_pos_cross_panel` edges that
  replace the panel scaffolding do the real work.

### 1.1 RAVEN's answer-set leakage, measured

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

**Consequence: 96.4% is not a real number.** Any port should target the I-RAVEN config and
keep candidates from seeing each other. *(Correction, later: the I-RAVEN 68.8% was trained on
the corrupted positions of §10 and is not valid either. The fixed-position retrain scores
55.09% test.)*

### 1.2 The winning architecture

`emb 256 / hidden 384 / 3 layers / lr 1e-3 / dropout 0.1 / batch 8 / AdamW wd 1e-4`

* **Graph** (`gnn/graph.py:47`, per candidate — 8 separate graphs per puzzle): 8 context
  panels + 1 candidate. Objects only. With `granular`, each object also gets 4 satellite
  attribute nodes. **17 typed relations** (`gnn/config.py:14`).
* **Encoder** (`gnn/model.py:68`): plain **sum** of `nn.Embedding` lookups over pos,
  color, shape, size, kind, role, panel-grid.
* **Layer** (`gnn/model.py:150`), ×3:
  ```
  msg   = W([x_src + rel_emb ; x_dst ; rel_emb])
  agg   = scatter_mean over incoming edges, per destination
  x_out = LayerNorm(x + dropout(out_proj(agg)))
  ```
* **Readout** (`gnn/model.py:381`): mean-pool context, mean-pool candidate,
  `MLP([ctx ; cand]) → 1 logit`.
* **Loss** (`gnn/train.py:102`): `F.cross_entropy(logits[batch, 8], target)` — one
  **8-way softmax ranking** the candidates. Note the non-unified path still scores each
  candidate on its *own* graph then stacks (`gnn/model.py:409`): independent scoring,
  joint normalisation.

---

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

## 4. What failed, and the elimination

Five hypotheses for why the pure port underperforms. **Four were real defects; none was
the main cause.**

| hypothesis | test | verdict |
|---|---|---|
| Class imbalance (1 pos : 7 neg) | plateau loss 0.3810 ≈ 0.3768, the constant base-rate optimum. Retrained on a **balanced** 1-pos/1-neg set | **rejected** — pairwise acc stayed 49–53% for 12 epochs |
| Additive readout | body atoms combine by `SUM`, so `score = W₁·ctx + W₂·cand` with **no interaction term**; `ctx` is identical across a puzzle's candidates so it cancels under argmax | **real, minor**: SUM 13.8% → ELPRODUCT 18.8% → **CONCAT 20.0%** |
| `tanh` saturation / no LayerNorm | each layer sums 8 terms with nothing renormalising; `out std` decayed to ~1e-8 | **real**: TANH collapsed (1.7e-08), NORM 1.4e-01, LEAKY_RELU 1.6e-01 — but acc only 16.2 → 18.8% |
| Binary vs 8-way softmax | built the softmax version (below); loss started at ln(8)=2.0794 and reached 2.0785 after 300 epochs | **rejected** — flat at the uniform floor for 5.5 hours |
| Capacity / scale | 300 epochs at dim 64, 2000 puzzles | **rejected** — peaked 27.2% at epoch 89 then *degraded*; last-50-epoch mean 20.6% vs first-50 19.2% |

Fixing the three real defects moved a toy from 13.8% → 21.2%. Scaling moved it to 27.2%.
The gap to 75.4% stayed ~48 points.

### 4.1 Expressing 8-way softmax in PyNeuraLogic

It *is* possible, and worth recording:

```python
m += (R.answer(V.G) <= (R.cand_of(V.B, V.G), R.score(V.B))) \
     | Metadata(aggregation=Aggregation.CONCAT)      # 8 scalars -> one 8-vector
Sample(R.answer(gid)[onehot], facts)                  # target as one-hot
Settings(error_function=CrossEntropy(with_logits=True))   # softmaxes internally
```

One sample = one puzzle, with all 8 candidate sub-graphs present but **fully disjoint**
(context duplicated per branch, no candidate-candidate edges). That reproduces torch's
`forward_one` + `stack` exactly and avoids the leakage. Total work is unchanged:
8 samples × 9 objects becomes 1 sample × 72.

At full scale it never learned — loss 2.0785 vs the 2.0794 floor after 300 epochs.

---

## 5. Structural differences torch vs PyNeuraLogic

"Same GNN structure" is true only of the *relational* part.

| # | torch | naive pynl port | why it matters |
|---|---|---|---|
| 1 | `LayerNorm` every layer | **nothing** | at hidden 384 the loss diverged to **3643** in 2 epochs |
| 2 | batch size 8, AdamW | `build_dataset(batch_size=1)`, the default | 16 000 noisy Adam steps/epoch instead of 2 000 |
| 3 | `x + f(agg)` — **identity** skip | `h_k(X) :- h_{k-1}(X)[d,d]` — a **learned matrix** in the skip path | not a residual at all; destroys the gradient-flow property |
| 4 | one `scatter_mean` over **all** incoming edges | AVG *within* each relation, then SUM across relations | with 3 active relation types the magnitude is ~3× torch's, feeding #1 |
| 5 | **one shared `W`** for all 17 relations, identity via additive `rel_emb` | separate `[d,d]` per relation | **3.7× the parameters** (6.64M vs 1.79M at hidden 384), zero sharing |
| 6 | 2-layer `score_head`, xavier gain 0.1, zeroed bias | single `Linear(2D→1)` | no hidden layer in the readout |
| 7 | dropout 0.1, weight decay 1e-4 | none | no regularisation |

Fixing #1–#5 (`train_pynl_aligned.py`) moved the 10-puzzle overfit from 60–70% to **90%**.

Also: at hidden 384 with lr 1e-3 the loss goes **2.52 → 3643 → 43.4**. lr 1e-4 is stable
but plateaus fast; lr 1e-5 is stable and still descending. Needing a 100× smaller LR is
itself the missing-normalisation symptom.

---

## 6. The decisive diagnostic: can it overfit 10 puzzles?

A model with >1M parameters should memorise 10 examples to 100% in tens of epochs.

| variant | best TRAIN acc on 10 puzzles |
|---|---|
| pure PyNeuraLogic, as first written | 60–70% after 120 epochs (still 50% at epoch 200 of 800) |
| pure PyNeuraLogic, structurally aligned | 90% after 400 epochs |
| **hybrid with torch head** | **100% at epoch 14** (~15 seconds) |

This is the cleanest result in the whole exercise. **The template was always expressive
enough; optimisation was the entire gap.** Ruling out overfitting, data volume and
generalisation, since none can explain a failure to fit the training set itself.

---

## 7. The ablation ladder

`train_ladder.py`, one script with `--level N` so hyperparameters cannot drift.
Identical settings throughout: `emb 64 / hidden 64 / 2 layers / lr 1e-3 / nl-lr 1e-4 /
batch 8 / 80 epochs / 2000 train / 500 val`, `center_single`.

| level | torch owns | PyNeuraLogic owns | best val | min train loss |
|---|---|---|---|---|
| 1 | emb, proj, **head**, loss, opt | message passing | 32.6% | 0.180 |
| **2** | emb, proj, loss, opt | MP + **head** | **39.0%** | 0.255 |
| 3 | loss, opt only (**0 torch params**) | **emb** + MP + head | 20.8% | 0.636 |
| 4 | — | **everything** | 15.4% | 1.402 |

**The drop is localised, and not where expected.**

* **Moving the scoring head into PyNeuraLogic *helped*** (+6.4). Readout, context pooling
  and per-candidate scoring are all learnable as rules.
* **The collapse is entirely 2 → 3: −18.2 points.** That step moves *only* the attribute
  embeddings from `nn.Embedding` into a `[d, n_values]` matrix over a one-hot —
  mathematically the same operation.
* **3 → 4 costs a further 5.4**, isolating the training loop (`CrossEntropy` + `Adam` +
  batching) as real but secondary.
* Train loss tells it more sharply: L1–L2 reach ~0.2; **L4 never fits the training set**
  (1.402 against a 2.079 uniform floor).

**So the narrow failure is: learning input embeddings inside the differentiable-logic
framework.** That points at initialisation and gradient scale on fact-weight matrices,
not at anything about logic or graph structure.

*Caveat: single seed per level. The 18-point 2→3 gap is unlikely to be noise; the 6.4-point
1↔2 ordering could plausibly flip. Worth 3 seeds before citing.*

---

## 8. The hybrid

`train_pynl_torch.py` — PyNeuraLogic for relational message passing, PyTorch for
everything numeric (embeddings, LayerNorm, scoring head, 8-way softmax CE, AdamW,
gradient accumulation).

All 8 context panels and all 8 candidates in **one graph, one forward pass**. Candidates
connect to the context but **not to each other**, preserving the no-leakage property.

Full scale (`center_single`, 6000/2000, emb 128 / hidden 128, 3 layers, 150 epochs,
12h48m, 24.5 GB peak):

```
epoch    loss    val acc
    1  2.0301     16.0%
   20  1.4773     26.0%
   60  0.8854     47.5%
  121  0.3899     52.9%   <- best
  150  0.3055     51.2%
```

**52.9%**, monotone loss 2.03 → 0.305, no divergence or collapse. Mean val went 22.7%
(first 20 epochs) → 51.3% (last 20). **Not converged** — loss was still falling at
epoch 150, minimum at 147.

| approach | best val, I-RAVEN `center_single` |
|---|---|
| pure PyNeuraLogic, binary | 27.2% |
| pure PyNeuraLogic, 8-way softmax | 16.0% |
| **hybrid** | **52.9%** |
| torch GNN (test split) | 75.4% |
| chance | 12.5% |

### 8.1 The uncomfortable implication

In the hybrid the symbolic component is reduced to **pure topology** — a message-passing
schedule over a fixed graph. No attribute constants, no grounding over attribute values,
no logical embeddings. All semantic content lives in torch. **And that is the version
that works.**

The more of the symbolic machinery moved out of PyNeuraLogic, the better it performed.
A reviewer will ask what the logic is buying; for the 52.9% model the honest answer is
"the graph topology, and not much else". The pure version is the one that actually does
neurosymbolic reasoning — and it is stuck at 27.2%. **That contrast is the result.**

---

## 9. All seven constellations

Earlier work was `center_single` only — one of seven, and the easiest.

**Object counts vary**, which breaks the static-graph requirement:

| constellation | objects/panel | slots |
|---|---|---|
| `center_single` | always 1 | 1 |
| `left/right`, `up/down`, `in/out` | always 2 | 2 |
| `distribute_four` | **1–4** | 4 |
| `in_distribute_four_out_center_single` | **2–5** | 5 |
| `distribute_nine` | **1–9** | 9 |

One-module-per-constellation with shared weights **fails on backward** (§3.7). The working
approach is a **common 9-slot topology** — 16 panels × 9 slots = 144 nodes, 2808 edges,
measured 484 ms/puzzle at hidden 128 (8.8× `center_single`), with a learned pad embedding
for inactive slots.

**Slot semantics matter:** slot *k* is the *k*-th **universal position** of that
constellation, not the *k*-th object present. A puzzle is entirely one constellation, so
slot *k* denotes the same position in all 16 panels and same-slot cross-panel edges link
matching positions. Verified: 4243 objects, zero mismatches.

`train_all_constellations.py --level {1,2}`, 500 train / 200 val per constellation.

### 9.1 Overfit probe, all 7 at once (level 2)

**92.9% — 13 of 14 puzzles memorised**, six constellations at 100% including
`distribute_nine`. Only `distribute_four` stuck at 50%, with loss pinned at exactly
**0.0496** for 50 epochs.

Isolating `distribute_four` (5 puzzles): saturates at **80%**, loss flat at **0.4159** for
150 epochs. And `ln(8)/5 = 0.4159` exactly — so four puzzles are fit perfectly and one
sits at *precisely* uniform. The model emits identical logits for all 8 candidates on it.

That signature led to the next section.

---

## 10. The position bugs (torch pipeline)

Two decoding bugs: the slot bug below (I-RAVEN only) and a spatial-relation bug (both
datasets, third subsection).

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

### Measured impact on the **torch** data pipeline

| constellation | puzzles with indistinguishable candidates | torch test acc |
|---|---|---|
| `center_single` | 0/40 | 75.40% |
| `left/right` | 0/40 | 73.40% |
| `up/down` | 0/40 | 74.20% |
| `in/out` | 0/40 | 84.45% |
| **`distribute_four`** | **13/40** | **55.85%** |
| **`distribute_nine`** | **13/40** | **47.25%** |
| **`in_distribute_four_out`** | **12/40** | **71.15%** |

Affected constellations average **58.1%**; unaffected **76.9%**. The three worst results
are exactly the three broken ones. Roughly a third of those puzzles are **unanswerable** —
two or more candidates encode identically and one of them is the answer.

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
duplicates gone. `center_single` is untouched, so every PyNeuraLogic-vs-torch comparison
above remains valid.

**Your torch I-RAVEN results predate this fix.** Re-running
`train_object_vsa_gnn_frontend.py` is the cheapest high-value experiment available.
*(Done: the fixed-position retrain, `experiments/gnn_FIXEDPOS_granular_object_only`, scores
55.09% test.)*

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

### Third bug (2026-09-30): wrong spatial relations in the torch graph

The torch graph adds within-panel edges typed by spatial relation (`left_of`, `above`,
`same_row`, `overlap`, …) via `gnn/graph.py::infer_relations(pi, pj, centers)`, which looks
centres up by the objects' **universal position id**. But `gnn/data.py::build_position_centers`
keyed centres by the **rank** of each bbox in a sorted list of all distinct bboxes — a
different numbering — and also treated the centre-based bboxes as corner-based. Universal
position 0 (the panel centre, true (0.5, 0.5)) was given (0.33, 0.33).

Measured against exact bbox centres, first 200 test puzzles per constellation:

| constellation | within-panel relations wrong | example: labelled vs true |
|---|---|---|
| distribute_four | 33.4% | 4→2: below, **left_of** vs below, same_col |
| distribute_nine | 75.1% | 7→10: above, **left_of** vs above, same_col |
| left/right | 0.0% | correct by coincidence |
| up/down | **100%** | upper→lower: **left_of, same_row** vs above, same_col |
| in/out | **100%** | outer→inner: **above, left_of** vs overlap |
| in_distribute_four | 79.5% | 0→19: **above, same_col** vs below, left_of |

**Scope:** every torch GNN run, RAVEN and I-RAVEN, published and retrained. The labels are
wrong but *consistent* (a fixed function of the position pair), so the GNN could still use
them as arbitrary edge types — but distinct geometric relations get merged, and nothing
about "spatial reasoning" can be claimed from those models. The PyNeuraLogic model uses no
spatial relations and is unaffected.

**Fix:** `gnn.data.universal_position_centers(dataset_root)` returns the true centre of each
universal position from bbox geometry (`(bbox[1], bbox[0])`, checked against the pixel masks).
`train_object_vsa_gnn_frontend.py` uses it by default (`--legacy-centers` restores the old
behaviour), and `eval_gnn_test.py` now evaluates with the centres **stored in the checkpoint**,
so every old checkpoint reproduces its reported number exactly (verified: stored centres equal
the legacy function's output). With the fix, 0 relations are wrong in every constellation.

Retrain with both fixes, three seeds: `submit_gnn_fixrel.sh` → `experiments/gnn_FIXREL_s42`,
`_s43`, `_s44`.

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

Two chained jobs, because the stages need different environments:

| stage | where | does |
|---|---|---|
| `perceive` | GPU, env `nvsa` | runs both frontends over every I-RAVEN val + test panel (448,000); stores per panel and position the best attribute match and its cosine; scores perception against the ground truth |
| `solve` | CPU, env `pynl` | picks the frontend with the better **val** perception (never test), decodes its test detections, and scores the same puzzles from pixels and from XML with the same PyNeuraLogic checkpoint |

* **Decoding**, identical in both stages: at each position the constellation can use, take the
  best of its 300 attribute rows; the object exists iff that cosine > τ = 0.23. One object per
  position by construction. The constellation label comes from the dataset folder, as in the
  XML pipeline.
* **No remapping needed:** the frontend's codebook positions, `train_pynl.POS_MAP` and
  `gnn.config.CONFIG_POS_MAP` are identical, and the two ground-truth parsers agree on every
  panel checked.
* **Built-in checks:** the decoder is asserted equal to the frontend's own `decode_w_index`;
  the stored ground truth is asserted equal to `train_pynl.parse` on 140 puzzles.
* **Cost:** perception 10 min on one A100 (outputs cached in `experiments/frontend_preds/`);
  solving ~90 min on CPU. Re-scoring a newer reasoner only needs the `solve` stage.

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

### 11.4 Result: pixels cost nothing

| constellation | all 16 perceived | from pixels | from XML | Δ |
|---|---|---|---|---|
| center_single | 100.0% | 78.80 | 78.80 | 0.00 |
| distribute_four | 99.9% | 35.25 | 35.25 | 0.00 |
| distribute_nine | 76.5% | 35.95 | 35.65 | +0.30 |
| left/right | 100.0% | 79.70 | 79.70 | 0.00 |
| up/down | 100.0% | 78.70 | 78.70 | 0.00 |
| in/out | 100.0% | 86.65 | 86.65 | 0.00 |
| in_distribute_four | 71.5% | 52.70 | 53.25 | −0.55 |
| **overall** | **92.6%** | **63.96** | **64.00** | **−0.04** |

(`experiments/max_data_L2` at epoch 62, I-RAVEN test.)

Why it is robust: when all 16 panels are perceived correctly the input is identical to the
XML, so the answer is too — the whole difference comes from the ~1,040 puzzles with at least
one misperceived panel. On those, pixels and XML stay within 2 points (distribute_nine
32.9% vs 31.7%, in_distribute_four 53.3% vs 55.3%). A typical error is one small object in
one crowded panel, which rarely changes the winning candidate, in puzzles the reasoner
already finds hard.

**Caveat — image overlap.** The frontend was trained on RAVEN, which shares images with
I-RAVEN test: 8% of distribute_nine panels (0 whole contexts), 33% of in_distribute_four
panels (51% of whole contexts), and every center_single panel. Perception on **unseen**
panels is 95.7% / 95.4% versus 98–100% on seen ones, so perception is inflated by under a
point where it matters, and the end-to-end number barely moves. center_single cannot be
separated (no unseen panels) but is the easiest case, one large object per panel. The
reasoner never saw RAVEN, so this does not touch it.

### 11.5 Can the whole thing be trained end to end?

**Technically yes.** Exactly one step blocks gradients:

```
pixels → ResNet → cosine vs codebook → [threshold + argmax → integer IDs] → embedding lookup → PyNeuraLogic → 8 logits
                                              ↑ the only non-differentiable step
```

The fix is to replace the hard decode with **soft attribute probabilities**: per position, a
softmax over the 300 codebook similarities, marginalised into P(type), P(size), P(color), plus
a soft existence probability. The node embedding becomes the probability-weighted sum of the
embedding rows, blended with the pad token by the existence probability. With one-hot
probabilities this reproduces the current model exactly, which gives a clean regression
test. It is the same trick NVSA's own end-to-end training uses (`raven/main_nvsa_marg.py`:
`metric_fc` turns the ResNet output into marginal probabilities, `marg_prob`, and the loss
backpropagates into the ResNet).

Everything downstream already passes gradients: level 2 trains its torch embeddings
*through* the PyNeuraLogic layer, and the training loop runs backward once per puzzle, so the
gradient would continue into the ResNet. A GPU node (A100, 128 CPUs, 1 TB RAM) holds the
ResNet and the JVM side by side. Speed would be set by the reasoner (~6.3 h per full-data
epoch); the ResNet should add only a small overhead.

**The catch: it cannot help accuracy much.** Perception costs 0.04 points (§11.4), and joint
training can at best recover what perception loses. The limit is the reasoner — 64% even
with perfect inputs, ~35% on distribute_four/nine. End-to-end training buys a methodological
claim ("fully differentiable, trained from pixels to answer"), not a better number.

Two versions, with very different risk:

1. **Joint fine-tuning from the trained parts** (ResNet `mean3` + reasoner `max_data_L2`),
   keeping the attribute loss as an auxiliary term so the ResNet's "colour 3" does not drift
   away from actual colour 3. Low risk; same accuracy expected; earns the end-to-end claim.
   Estimated a day to build and test, then 2–3 days of GPU training.
2. **Perception learned only from the answer**, with no attribute labels — can a logic
   reasoner *teach* perception? High risk. NVSA gets away with it because its reasoner is a
   fixed rule executor with no learnable parts, so its gradients are meaningful from step one;
   here both sides would learn from one weak signal (which of 8 candidates is right), which
   tends to collapse or stall.

**Recommendation:** version 1 if the thesis narrative needs "end to end", presented as
confirming the pipeline trains jointly without loss. For a higher number, the effort
belongs in the reasoner. *Not built yet.*

---

## 12. Files

| file | what |
|---|---|
| `train_pynl.py` | pure PyNeuraLogic, per-candidate binary scoring (27.2%). Also hosts `parse()`, `POS_MAP`, `canonical_positions` |
| `train_pynl_softmax.py` | pure PyNeuraLogic, 8-way softmax (16.0%) |
| `train_pynl_aligned.py` | pure, structurally corrected — shared `W` via `edge/3`, identity residual, 2-layer head |
| `train_pynl_torch.py` | hybrid, `center_single` (52.9%) |
| `train_ladder.py` | the 4-level ablation, `--level N` |
| `train_all_constellations.py` | all 7 jointly, `--level {1,2}`, 9-slot padded topology |
| `pynl_iraven_minimal.ipynb` | annotated minimal notebook with executed outputs |
| `pynl_best_architecture_walkthrough.ipynb` | walkthrough of the best model (`max_data_L2`), importing the real training code |
| `eval_pynl_test.py` | full test-split evaluation of a PyNeuraLogic run (reference column: fixed-position torch GNN) |
| `sanity_checks.py` | `splits` / `shortcut` / `images` / `model` checks that the result is not an artefact |
| `pixel_pipeline.py` | end-to-end from pixels: `perceive` (ResNet, GPU) and `solve` (PyNeuraLogic, CPU) — §11 |
| `train_frontend.py`, `predict_frontend.py`, `extract_objects.py` | ResNet-18 perception frontend and its tools (position parse fixed, §10) |
| `gnn/data.py` | **both torch bugs fixed here**: `canonical_positions` (slot decoder) and `universal_position_centers` (spatial-relation centres) |
| `analyze_constellation.py` | per-rule / per-object-count failure analysis (two-component rule keys fixed 2026-09-30) |
| `archive/pyneuralogic_2026-03-31_to_05-06/` | the previous PyNeuraLogic attempt, with MANIFEST |

Moved on 2026-10-20 to `archive/pyneuralogic_experiments/`: `train_pynl_softmax.py`,
`train_pynl_aligned.py`, `train_pynl_torch.py`, `train_ladder.py`, `pynl_iraven_minimal.ipynb` and the
submit scripts `submit_pynl*.sh`, `submit_ladder*.sh`, `submit_full.sh`, `submit_sweep.sh`,
`submit_layout_grid.sh`, `submit_all_constellations.sh`. LaTeX tables are in `thesis/tables/`.
The current code then moved into folders: `pynl/` (PyNeuraLogic scripts and jobs), `notebooks/`,
`frontend/` (ResNet-18), `torch_gnn/` (torch GNN). File names in earlier sections are unchanged.

Submit scripts: `submit_pynl*.sh`, `submit_ladder*.sh`, `submit_all_constellations.sh`,
`submit_max.sh` (chained resumable jobs behind `max_data_L2`), `submit_pixels.sh` (§11),
`submit_gnn_fixrel.sh` (torch retrain with both fixes, three seeds).

**Stale, do not cite:** `table_i_raven_per_constellation_test.tex`, `table_406.tex` and
`scripts/BEST_MODELS_RESULTS.md` still report the invalid 68.81% / 68.83% I-RAVEN numbers.
Regenerate them from the three-seed retrain and the final PyNeuraLogic checkpoint.
PyNeuraLogic jobs are CPU-only on `amdlong`, `JAVA_HOME` pinned, `PYTHONNOUSERSITE=1`.

---

## 13. Open questions

1. **A trustworthy torch baseline.** The 55.09% retrain is 10–13 points below the published
   model even where the position bug never applied — but the published run is itself not
   reproducible (§14), and both used wrong spatial relations (§10). The three-seed retrain
   with both fixes (`gnn_FIXREL_s*`) settles the baseline and its spread.
2. **Why can't PyNeuraLogic learn input embeddings?** The narrow, localised failure from
   §7. Initialisation and gradient scale on `[d, n_values]` fact-weight matrices are the
   place to look.
3. ~~The hybrid was never converged / generalisation gap.~~ **Answered:** it was data. At
   6000/constellation the level-2 model reaches 64.0% test; at 2000 it saturates near 54%.
   Open detail: the full-data model is a warm start; no from-scratch full-data run at dim 128.
4. **The reasoner is now the bottleneck, not perception** (§11.4). Re-measured on the epoch-62
   checkpoint: distribute_four / distribute_nine score 66–69% when the layout stays constant
   and ~30% otherwise; Number rules 26–29%, Position-Progression worst at 18–22%; in
   distribute_four's inner grid, Position-Progression 25.5%. The model handles attribute rules
   and fixed layouts and fails when the set of occupied slots changes. Panel layout features
   and within-panel edges both hurt when tried; this is where accuracy is left.
5. **End-to-end training** (§11.5): version 1 is low-risk and earns the claim; version 2
   (perception from the answer alone) is the research question.
6. **Seeds.** Every PyNeuraLogic number here is a single run; the torch baseline now gets three.

### A framing that fits the evidence

*Expressiveness is not trainability.* GNNs are provably a special case of differentiable
logic programming, and that equivalence invites the assumption that anything a GNN learns,
a template can learn. These experiments show that is false in practice and localise why:
the declarative part transfers cleanly, while normalisation, skip topology, batching and
embedding optimisation — none of which the framework gives you vocabulary for — are what
make deep GNNs trainable.

The RAVEN leakage ablation (§1.1) and the position bug (§10) are independently
publishable observations about the benchmark and its standard tooling.

---

## 14. Audit (2026-09-30)

Everything found while checking this work for gaps and inconsistencies, with its status.

| finding | impact | status |
|---|---|---|
| **Wrong spatial relations in the torch graph** (§10, third bug): centres keyed by bbox rank, looked up by universal id; 33–100% of within-panel relations wrong in 5 of 6 multi-object constellations | every torch GNN run; not PyNeuraLogic | **fixed**; old checkpoints still reproduce; three-seed retrain running |
| **Published torch 68.8% not reproducible.** `ep405` and `ep406` have identical recorded arguments and seed 42, yet finished at **27.35%** and **68.83%**; `ep405` ran ~10× faster per epoch, pointing to different code. `ep406`'s first 88 epochs have no surviving log, and the training code was edited after it finished | the "old model" comparison rests on one unreproducible run | disclosed; baseline replaced by the three-seed retrain |
| **Warm start.** "Epoch 62" = 38 epochs at 2000/constellation (identical to `allcons_v2`) + 24 at 6000 | wording of the headline | disclosed in the status table |
| **Repeated test evaluation.** Test scored at epoch 56, epoch 62 and in the pixel run; never used to pick a checkpoint | mild | done: the final checkpoint (epoch 126) was scored once on 2026-10-19 |
| **Stale result files** with the invalid 68.81% / 68.83% (§12) | thesis tables | new `table_iraven_final_test.tex` (2026-10-19); the old files are left unchanged |
| **Sanity checks ran on epoch 56**, headline on 62 | consistency | re-run on epoch 62, same outcome |
| **Weak-spot claim came from the 2000/constellation model** via an analysis script that merged the two components' rules in in_distribute_four | the claim in §13.4 | script fixed; re-measured on epoch 62, claim holds (§13.4) |
| **`train_pynl.parse` treated centre-based bboxes as corner-based** | spatial relations of the early pure-PyNeuraLogic scripts only; their reported numbers are center_single, which has no object pairs | fixed; no reported number changes |

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

## 16. GNN ablations for the layout weak spot (running, 2026-10-19)

`submit_ablations.sh`, same budget as `laygrid_nosp_off` (2000/constellation, 12 epochs, 45.1% validation),
new switches in `train_all_constellations.py`, all off by default (verified: with defaults the outputs
equal the old code exactly on the trained checkpoint):

| arm | change |
|---|---|
| `ablate_control` | the best configuration again (seed noise against `laygrid_nosp_off`) |
| `ablate_rowcol` | `--edge-labels rowcol`: cross-panel edges tagged same-row / same-column (same 1,656 edges) |
| `ablate_sp_pertype` | within-panel edges on, `--per-type-weights`: one message matrix per edge type, each type averaged separately |
| `ablate_absentpos` | `--absent-pos`: empty slots carry their position and panel embedding |
| `ablate_combined` | all three |


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

**Affected results.** Every facet run before 2026-10-25 21:19: distribute_nine alone 78.0% (dim 32) and
90.1% (32 → 64 → 128), all constellations 83.8% (dim 32), the cancelled widened all-constellation run
(84.7% at epoch 5), and the ablations. The hub and ordered templates also have visible selectors, but
there positions are part of each object's input vector, so they only add constant offsets.

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

**Ablations (before the fix, distribute_nine, 2000 puzzles, dim 32; to be repeated).**
* One-hot values: 64.4 (thermometer 71.2–75.4). Ordinal codes matter.
* Separate weights per attribute: 75.4, within noise of shared.
* No row relation (level 1): 72.1, level with level 2.
* Run-to-run spread is about ±1–2 points. Picking the epoch on 300 puzzles is noisy (±2.4).

### 18.4 Current results (runs in progress; update when finished)
* **I-RAVEN, all 7 constellations, fixed, widths 32 → 64 → 128, lr 3e-4, 6000 per constellation** (job
  11882029, 80 epochs, `experiments/facet_all_w32-64-128_hidden`). Epoch 5: **92.9% matched, 55.5% exact**:

  | center | 2×2 | 3×3 | L-R | U-D | O-IC | O-IG |
  |---|---|---|---|---|---|---|
  | 97.9 | 89.0 | 72.3 | 97.9 | 97.5 | 98.8 | 96.8 |

  The same run with visible selectors was at 84.7 / 14.2 at epoch 5. The grid quick check reached
  86–88 on 3×3, so distribute_nine should still rise.
* **I-RAVEN-X 3×3, values 0–9** (`experiments/facet_iravenx_n3_m10`). **99.2%** at the epoch picked on
  val 6000–6299 (epoch 10), **100.0%** from epoch 15 on, 95.4% exact. The GNN-style model got 46% on the
  same data. Copying the last panel matches 4.7%.
* **I-RAVEN-X 3×10** (IBM generator, seed 1234, no confounders), jobs 11883234–36. Epoch 5:
  * maxval 50: 77.4 (exact 6.1)
  * maxval 100: 76.6 (exact 2.1)
  * maxval 1000: pending

### 18.5 Where it stands against the literature
| setting | ours | reference |
|---|---|---|
| learned rules, ground-truth attributes, I-RAVEN | 92.9 (epoch 5) | ARLC 92.6 (trained on 2×2 only!), MLP 87.1, GPT-3 86.5, Learn-VRF 79.5 |
| hand-written rules, same input | – | our symbolic template 98.33 (test), NVSA backend about 97–99 (RAVEN) |
| pixels, classifying candidates | not run | Triple-CFN 99.1, RS-Tran 98.7, SAViR-T 98.1, DRNet 97.6 (self-reported). Independent re-runs (A-I-RAVEN, IJCAI 2025): DRNet 90.9, PredRNet 88.8, SCL 83.4, CPCNet 70.4, STSN 59.0 |
| I-RAVEN-X 3×3 | 99.2 / 100 | ARLC learned 98.4, programmed 97.2, GPT-4 93.2, Llama-3 85.0 |
| I-RAVEN-X 3×10, maxval 50 / 100 / 1000 | pending | ARLC learned 95.0 / 95.0 / 90.6 (trained once at 50, reconfigured), programmed 100 / 100 / 99.7, GPT-4 82.2 / 79.6 / 76.6 |

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
  all of I-RAVEN takes about 14 min per epoch at widths 32 → 64 → 128. The GNN-style model through the
  torch bridge took 6.5–9.6 h per epoch.

### 18.7 Findings for the thesis, ranked
1. Generative per-puzzle object graphs in PyNeuraLogic: design, headline numbers, and the table in 18.3.
   This moves the thesis from "PyNeuraLogic matches a torch GNN" (68.5 vs 45) to "the right lifted
   representation brings a learned model close to hand-written rules".
2. The prior levels (none / same relation in every row / given rules). This needs the post-fix ablations,
   and ideally a generative level 3.
3. Semantic pitfalls of lifted templates: visible selectors (§17), chain pruning dropping transformations,
   hidden atoms merging identical groundings. Plus how each was diagnosed.
4. Error analysis and the benchmark observations in 18.6.
5. Cost (18.6) and the earlier chapters: the GNN port, the pipeline bugs (§10), the symbolic templates (§15),
   end to end costing almost nothing.

### 18.8 Needed before these are claims
1. Score the final checkpoint once on the test split, and end to end with the cached ResNet-18 predictions
   (`experiments/frontend_preds/`). About half a day after the run ends.
2. Re-run the ablations with hidden selectors: values, weight sharing, level 1 vs 2. About a day, in parallel.
3. Three seeds of the final configuration.
4. ARLC's protocol: train on 2×2, test on all seven (unseen slot ids may hurt). About half a day.
5. Optional: a generative level 3 (symbolic rules as a generator), so all three levels share one setting.
