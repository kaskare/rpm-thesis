# I-RAVEN solver in PyNeuraLogic — project knowledge

State as of 2026-10-19. Written from the full working history of the project, for use as Claude
project knowledge. Numbers are measured unless marked as estimates.

---

## 1. The project in one paragraph

A master's thesis at CTU (cluster: RCI, login node `login4.rci.cvut.cz`, account `kassyask`).
Goal: solve I-RAVEN Raven's Progressive Matrices with a **PyNeuraLogic** model (a differentiable
logic program, developed at CTU) and compare it with a **torch GNN** solver built earlier in the same
repository. The repository is a clone of IBM's NVSA RAVEN code
(`IBM/neuro-vector-symbolic-architectures-raven`), extended with the user's own GNN, PyNeuraLogic,
symbolic and perception work. The final PyNeuraLogic model (epoch 126 of 130) reaches **68.49%
I-RAVEN test with XML labels and 68.46% end to end** through a ResNet-18 frontend, against
**45.25 ± 12.4** for the torch GNN retrained over 3 seeds (best seed 59.02%). That is level with the old
published torch number, 68.81%, which was trained on buggy positions and is not reproducible. A separate
**symbolic** PyNeuraLogic model reaches **97.6–98.4%** on ground-truth attributes.

---

## 2. Dataset: I-RAVEN-10000

* Location: `data/I-RAVEN-10000/<constellation>/RAVEN_<id>_<split>.npz` + `.xml`.
* 7 constellations, each 6000 train / 2000 val / 2000 test (42,000 / 14,000 / 14,000).
  `center_single`, `distribute_four`, `distribute_nine`, `left_center_single_right_center_single`,
  `up_center_single_down_center_single`, `in_center_single_out_center_single`,
  `in_distribute_four_out_center_single`.
* `.npz`: `image` (16, 160, 160) uint8 greyscale panels; `target` (answer index 0–7). Panels 0–7 are the
  context (3×3 matrix minus the last cell), 8–15 the candidates.
* `.xml`: per panel, objects with `bbox` (centre-based, `[y_centre, x_centre, h, w]`), `Type`, `Size`,
  `Color`, `Angle`; plus the puzzle's rules.
* **22 universal positions** (distinct boxes across all layouts). `train_pynl.POS_MAP`:
  center_single [0]; distribute_four [1–4]; distribute_nine [5–13]; left/right [14, 15];
  up/down [16, 17]; in/out [0, 9] (outer = centre_single's box, inner = d9's centre cell);
  in_distribute_four [0, 18–21]. Slots in distribute_four: 0 = top-left, 1 = top-right,
  2 = bottom-left, 3 = bottom-right.
* Attributes: type 0–4 = triangle, square, pentagon, hexagon, circle (XML Type − 1); size 0–5;
  colour 0–9 with **0 = white … 9 = black** (greys 255, 224, 196, 168, 140, 112, 84, 56, 28, 0).
  Angle is not used by any rule.
* Rules (one per component and attribute, row-wise): Constant, Progression, Arithmetic,
  Distribute_Three. **Measured semantics in this data:** Progression ±1/±2; Arithmetic on Colour and
  Number c = a ± b, on Size c = a + b + 1 or a − b − 1, on Position union / difference;
  Distribute_Three = same multiset in every row; Position-Progression = cyclic shift over all slots
  (±1 on 4 slots, ±1/±2 on 9); Constant = one value across the row, or slot-wise equal when the layout
  is fixed (attributes are noise when the layout changes).
* **Candidates (I-RAVEN):** the 8 candidates are every combination of changing / keeping 3 attributes
  of the answer (2³), so each wrong candidate breaks 1–3 rules.
* **Validation protocol used everywhere:** training checkpoints are selected on val puzzles 0–299 per
  constellation; val puzzles 300+ are "unseen" and used for demos and quick checks; the test split is
  scored once at the end.

---

## 3. Environment and cluster rules

* **conda env `pynl`** (`~/.conda/envs/pynl`): Python 3.11.16, neuralogic **0.9.2**, JPype 1.7.1,
  torch 2.14 (CPU). Activate via `/mnt/appl/software/Anaconda3/2024.02-1/etc/profile.d/conda.sh`.
* **conda env `nvsa`**: Python 3.7, holds the torch GNN and the GPU perception pipeline. Cannot run
  neuralogic 0.9 (needs Python ≥ 3.10). Leave it alone.
* Always: `export JAVA_HOME=/mnt/appl/software/Java/21.0.8` (`ml load Java/21` silently does nothing on
  the amd compute nodes), `export PATH=$JAVA_HOME/bin:$PATH`, `export PYTHONNOUSERSITE=1`
  (`~/.local` shadows the env).
* PyNeuraLogic is JVM/CPU only: never request a GPU for it. Cap the JVM:
  `neuralogic.set_max_memory_size(GB)`. Silence the Java log: `neuralogic.logging.clear_handlers()`.
* **Login node rules:** never shut it down (shared); plain scripts are fine, but CPU-heavy **Jupyter
  kernels get SIGKILLed** there. Execute notebooks on a compute node:
  `sbatch -p amdfast -c 16 --mem=48G -t 0:45:00 --wrap "source …conda.sh && conda activate pynl &&
  export PYTHONNOUSERSITE=1 JAVA_HOME=… && jupyter nbconvert --to notebook --execute --inplace
  --ExecutePreprocessor.kernel_name=python3 <nb>.ipynb && jupyter nbconvert --to html <nb>.ipynb
  --output-dir thesis"`.
* Partitions: `amdfast` (CPU), `amdgpufast` (GPU). SLURM scripts use `--mail-user` with the user's
  email. Compute nodes cannot see the session's `/tmp` scratch; pipe scripts via stdin
  (`srun … python3 - < script.py`) or keep them under `$HOME`.
* **Git:** almost everything is uncommitted (88 untracked items, incl. the dataset, checkpoints, logs).
  `origin` is IBM's public upstream repo; `gh` is not installed; no git identity is configured; the
  user's email appears in ~15 SLURM scripts. A PR to IBM would publish unpublished thesis work — not
  done. Recommended: a local branch with code + notebooks + notes (no data/checkpoints/logs) plus a
  `.gitignore`, pushed later to the user's own private repo.

---

## 4. The best PyNeuraLogic model (GNN-style, "level 2")

Code: `pynl/train_all_constellations.py` (index of all code: `thesis/CODE_INDEX.md`).
Run: `experiments/max_data_L2` via `pynl/submit_max.sh ARM=data` (chained 3-day SLURM links, `--resume`).

**Settings:** `--level 2 --embed-dim 128 --hidden-dim 128 --layers 3 --lr 1e-3 --nl-lr 1e-4
--batch-size 8 --agg avg --no-same-panel`, layout features off, 6000 puzzles/constellation,
all 7 constellations jointly, warm-started from the 2000/constellation run (`allcons_v2`):
"epoch 62" = 38 epochs at 2000 + 24 at 6000.

**Graph (one per puzzle, identical for every puzzle → grounded once):**
* 144 nodes `p{panel}_s{slot}`: 16 panels × 9 slots (9 = distribute_nine's maximum). Slot k = the k-th
  universal position of the puzzle's constellation; unused slots are padding.
* Edges, same slot only: `ctx_ctx` (every context panel ↔ every other, 56 directed per slot),
  `ctx_cand` and `cand_ctx` (every context panel ↔ every candidate), none between candidates.
  1,656 edges. No within-panel edges (`same_panel` exists as an option, +1,152 edges, off), no row/column
  structure (panel 0 links to panel 7 exactly as to panel 1).
* Facts: `edge(S, D, T)`, `is_context(O)`, `in_cand(O, C)`, and one 128-d weighted fact `feat(node)`
  per node, rewritten for every puzzle.

**Node embedding (torch):** 5 integer ids per occupied slot — position (0–21), colour (0–9), type (0–4),
size (0–5), panel (context 0–7; **every candidate = 8**) — each looked up in its own learned table and
**summed**. Empty slots get one learned `absent` vector instead (initialised at zero, trained; at epoch 78
norm 5.7, dense; carries no position or panel). Then `proj` = Linear(128, 128) + LayerNorm + ReLU.

**Template (PyNeuraLogic):**
```
rel_emb(T)[128]  for T in same_panel, ctx_ctx, ctx_cand, cand_ctx   (same_panel unused)
h0(O) :- feat(O).                                                    identity
per layer k = 1..3:
  hk(D) :- edge(S,D,T), h{k-1}(S)[128,128], rel_emb(T).   AVG        messages (one shared matrix)
  hk(D) :- edge(S,D,T), h{k-1}(D)[128,128].               AVG        self
  hk(X) :- h{k-1}(X).                                                identity residual
  hk/1 → NORM
candp(C) :- in_cand(O,C), h3(O)[128,128].  AVG, LeakyReLU            one vector per candidate
ctxp     :- is_context(O), h3(O)[128,128]. AVG, LeakyReLU            one vector for the context
sh(C)    :- candp(C)[128,128], ctxp[128,128].  LeakyReLU             combine
score(C)[1,128] :- sh(C).                                            one number per candidate
out :- score(C).  CONCAT                                             8 scores
```
"Shared weights" = one message matrix for all edge types, which differ only by an additive learned
vector r_T (the April model had separate matrices per relation).

**Sizes:** 23,552 torch weights (5 tables 6,656 + absent 128 + proj 16,768) + 164,480 template weights
(10 matrices 128×128, 4 edge vectors, 1×128 score) ≈ 188k. No biases in the template.

**Training and the torch ↔ PyNeuraLogic bridge** (`neuralogic.nn.torch_function.NeuraLogic`):
* At construction the wrapper grounds one sample (template + 144 placeholder facts + edges) into a fixed
  Java network; it forces `iso_value_compression=False`, `chain_pruning=False`. Template weights live
  in Java; torch sees only an empty placeholder parameter (so AdamW is built on params not named `nl.*`).
* Forward per puzzle: torch embeddings → 144×128 tensor → `set_fact_value` on each `feat` fact →
  `model.test` → 8 numbers → torch reorders them with `perm` → `F.cross_entropy` (the softmax is in
  torch, the template returns raw scores).
* Backward: torch sends ∂loss/∂scores (= softmax − one-hot) into Java; Java backpropagates through the
  template, **updates the template weights immediately with its own Adam (lr 1e-4) on every backward**,
  and returns ∂loss/∂feat (144×128) to torch, which continues into proj and the embeddings; **AdamW
  (lr 1e-3, wd 1e-4) steps every 8 puzzles**. `opt.zero_grad()` never reaches Java. A finite-difference
  check (step 1e-3) agrees with autograd within 0.4%, with and without NORM.
* **CONCAT order:** the grounder emits the 8 scores in its own order; `_recover_order` probes it once at
  startup (candidate k's slots carry value k+1) and stores `perm`.
* **Checkpoints are two files:** `model_best.pt` (torch state, incl. `perm`) + `template_best.bin`
  (`Model.save` = pickled template state_dict). `Model.save` stores weights only, so **NeuraLogic's Adam
  state restarts at every resume** (torch's AdamW state is restored) — an asymmetry of the chained runs.
* The Java template initialisation is **not** seeded by `torch.manual_seed`; seed runs need it seeded
  separately.
* Cost: ~0.5 s per puzzle forward; ~6.5–9.6 h per epoch at 6000/constellation. Grounding is constant.

**Why this design (and why the April model failed):** the April model (`archive/…/raven_pyneuralogic.py`)
used real objects as nodes (graph shape varies → regrounding per sample), 4 scaled numbers as features
(type/5, size/6, colour/10, pos/22 through one linear map), 6 spatial relations from a fake 6-column
position grid plus `same_pos` plus right/down neighbour-panel edges, separate weights per relation, a
learned skip matrix, no normalisation, and **8 binary yes/no samples per puzzle** (one per candidate,
scored alone; 7 of 8 targets are 0). It trained per constellation on 500 puzzles at D=8 and scored
9–19.5% test (chance 12.5%); full-data runs ran out of memory (Java heap / 200 GB). The fixes:
a fixed slot graph (ground once), learned embedding tables, whole puzzle with 8-way softmax,
identity residual + NORM, embeddings/loss/optimiser in torch, more data. Pure PyNeuraLogic stalled at
~27% on center_single; moving the numeric parts to torch lifted it to ~53%.

---

## 5. Results

### PyNeuraLogic GNN-style model
| run | data | result |
|---|---|---|
| `max_data_L2`, epoch 62 | 6000/cons | **64.00% test** (XML labels), **63.96% end to end** (14,000 test puzzles) |
| **`max_data_L2`, final (130 epochs, finished 2026-10-19)** | 6000/cons | best val 70.86% at epoch 126 → **68.49% test (XML labels), 68.46% end to end** (scored once, 2026-10-19) |
| `allcons_v2_L2` | 2000/cons | 51.78% test; saturates ~54.4% val (`max_conv_L2` ran 150 epochs, best 54.38% @ 50) |
| `full_iraven_L2` (dim 256, from scratch) | 6000/cons | failed, 27.7% val |
| level 1 (torch scoring head) | 2000/cons | 46.2% test |
| pure PyNeuraLogic | center_single | ~27%; level-4 pure variant ~15% |

**Final test by constellation (epoch 126, XML / end to end):** center_single 85.70 / 85.70;
distribute_four 40.85 / 40.85; distribute_nine 38.20 / 38.05; left/right 83.10 / 83.10;
up/down 84.00 / 84.00; in/out 90.25 / 90.25; in_distribute_four 57.30 / 57.30; overall 68.49 / 68.46.
Puzzles with every panel perceived correctly: 92.6%. Results file:
`experiments/frontend_preds/solve_universal_frontend_correct_proportional_mean3_test.json`.
Against the old published torch model on the 4 constellations the position bug never touched:
85.70 vs 75.40, 83.10 vs 73.40, 84.00 vs 74.20, 90.25 vs 84.45 (+6 to +10). On the 3 affected
constellations the old numbers (55.85, 47.25, 71.15) come from buggy inputs and are not comparable.

Epoch-62 test by constellation (XML / end to end): center_single 78.8 / 78.8; distribute_four
35.25 / 35.25; distribute_nine 35.65 / 35.95; left/right 79.7 / 79.7; up/down 78.7 / 78.7;
in/out 86.65 / 86.65; in_distribute_four 53.25 / 52.7. Epoch-62 results file backed up as
`experiments/frontend_preds/solve_universal_frontend_correct_proportional_mean3_test_epoch62.json`.

Best checkpoint (epoch 126) validation by constellation: center_single 83.7, distribute_four 46.7,
distribute_nine 41.3, left/right 87.3, up/down 84.7, in/out 92.7, in_distribute_four 59.7. Training
was flattening (10-epoch val means 68.5 → 68.7 → 68.6 → 69.4 over epochs 91–130; loss 0.64 → 0.57).

**Where it fails** (audit on epoch 62): distribute_four/nine ~67% when the layout stays constant,
~30% when it changes; Number rules 26–29%; Position-Progression 18–22%. The model handles attribute
rules and fixed layouts and fails when the set of occupied slots changes.

### Torch GNN baseline
| version | test |
|---|---|
| published `ep406` (`gnn` + `granular` + `object_only`) | 68.81% (9,634 / 14,000); trained on buggy positions; not reproducible (`ep405`, same args + seed 42, gave 27.35%) |
| retrain, fixed positions, wrong spatial relations | 55.09% (one seed) |
| retrain with both fixes, 3 seeds (`experiments/gnn_FIXREL_s{42,43,44}`, 406 epochs) | 41.74 / 35.01 / 59.02 → **45.25 ± 12.4** |

On the 4 constellations the position bug never touched, PyNeuraLogic also beats the old published
model (78.80 vs 75.40, 79.70 vs 73.40, 78.70 vs 74.20, 86.65 vs 84.45).

### Symbolic models (ground-truth attributes, unseen val puzzles 300+)
| model | val |
|---|---|
| `pynl_symbolic.py --model prims` (generic primitives, learned weights) | **97.58%** (300/cons, lr 0.01) |
| `pynl_symbolic.py --model rules` (RAVEN rule families, learned weights) | **98.44%** |
| non-learned reference: primitives / rules (`rpm_reference_solvers.py`, 2000/cons) | 96.34% / 97.75% |

**Test split (scored once, 2026-10-19, `experiments/symb_e2e_{prims,rules}_test.json`):** rules template
**98.33% with XML labels, 96.31% end to end**; primitives 97.55% / 95.71%; non-learned references
97.78% / 95.74% (rules) and 96.42% / 94.73% (primitives). The end-to-end loss is entirely in
distribute_nine (95.5 → 85.4) and in_distribute_four (97.3 → 93.2), the two constellations with imperfect
perception. In the literature table this end-to-end number sits between ALANS-V (93.5) and Rel-SAR / NVSA
(98.1 / 99.0), all of which also use attribute-trained perception with designed reasoning.
Learning curve: primitives 96.01% (equal start) → 97.13 / 97.35 / 97.49 / 97.58 / 97.58% at 10 / 30 / 100 /
300 / 1000 puzzles per constellation; rules 97.96% start, no gain at 10–30, then 98.25 / 98.44 / 98.49%.
Random initial weights (3 seeds) converge to exactly the same validation accuracy.

The reference rule solver **never ranks a wrong candidate above the answer**: 95.74% unique right, 4.26%
ties (mostly 2-way), 0.00% wrong. Ties concentrate where objects come and go (distribute_nine 10.2%,
distribute_four 7.6%, in_distribute_four 5.9%). Primitives: 93.06% / 6.90% / 0.04%. Learned weights
rediscovered the rules (position union +15.8, colour c = a + b +11.0, type arithmetic and wrong colour
offsets negative). Caveat: best epoch chosen on the same val set; no random-init or learning-curve runs yet.

### Perception (ResNet-18 frontend)
`log/universal_frontend_correct_proportional_mean3` (trained on RAVEN attribute labels; epoch 198,
95.26%). ResNet-18 (no maxpool, 11.4M weights, input 1×160×160 → 512-d, Tanh) → cosine against a frozen
codebook of 6,622 vectors (22 positions × 301 rows: 300 type/size/colour combinations + 1) → per position
the best of 300 rows; object exists if cosine > 0.23. On I-RAVEN test: 98.9% of panels and 92.6% of
puzzles perceived exactly (distribute_nine 76.5%, in_distribute_four 71.6% of puzzles; others ~100%).
RAVEN shares some images with I-RAVEN test; perception on unseen panels ~95.5% vs 98–100% seen — a small
effect. Predictions cached in `experiments/frontend_preds/`.

### Literature (I-RAVEN)
* From pixels with attribute-trained perception: NVSA 99.0, Rel-SAR 98.1, ALANS-V 93.5, algebraic
  (Xu et al.) 93.2 — all with fixed/exact rule executors.
* From pixels, answers only (reported / re-run by others): Rel-SAR 98.0; DRNet 97.6 / 90.9; PredRNet
  96.5 / 88.8; STSN 95.7 / 59.0 ± 18.5; SCL 95.0 / 83–84; Rel-Base 91.1 / 89.6; NVSA end-to-end 88.1;
  MRNet 83.5 / 75–87; PrAE 77.0 / 71.1 or 19.5; SRAN 60.8 / 58.2; CoPINet 46.1; LEN 41.4; MXGNet 33.1;
  WReN 23.8 / 18.4.
* Ground-truth attributes: exact executors ~97–99; MLP 87.1; GPT-3 (text) 86.5; Learn-VRF 81.0–84.1.
* Framing: accuracy cannot be the headline (≈20 points below an MLP on the same input). Claims the
  evidence supports: the PyNeuraLogic port beats its torch counterpart; why pure PyNeuraLogic fails and
  what fixes it; end to end costs nothing; two bugs found in the standard pipeline; reproducibility
  problems mirror the field's.

---

## 6. Bugs found (and fixed)

1. **I-RAVEN position decoding:** `Position.index(bbox)` is wrong on I-RAVEN only (its XML `Position`
   lists occupied slots, RAVEN's lists every slot) → ~32% of puzzles in three constellations
   unanswerable in the torch pipeline. Fixed with geometric `gnn.data.canonical_positions`; all live
   parsers use it. RAVEN results and the RAVEN-trained frontend are unaffected.
2. **Torch GNN spatial relations:** `build_position_centers` keyed centres by sorted-bbox rank while
   `infer_relations` indexed by universal position → 33–100% of within-panel relations wrong.
   Fixed with `gnn.data.universal_position_centers` (default; `--legacy-centers` for old runs).
3. **Old published 68.81% not reproducible** (see §5); `ep406`'s training code changed afterwards.
4. **PyNeuraLogic gotchas** (none give a clear error):
   * native training must pass `iso_value_compression=False` — the default merges neurons with equal
     initial values and froze the symbolic models (98% → 62%);
   * CONCAT emits in the grounder's order → probe it;
   * underivable queries are silently dropped → add a bias rule `score(K) :- cand(K)`;
   * samples sharing a query atom are merged (3440 → 2240) → unique query per sample; duplicating
     samples does nothing;
   * `build_dataset()` only once per built model;
   * rules sharing a head combine with SUM, AVG applies within a rule;
   * one template across two topologies breaks backward (rules out one module per constellation);
   * torch-wrapper inputs need `requires_grad=True`;
   * a built sample's target is read-only (set via `b._java_sample.target = ValueFactory().get_value(...)[1]`);
   * template facts are not learnable in native training; a probe's +1 offset needs `combination=PRODUCT`;
   * Python `|` vs `<=` precedence in rules needs parentheses.
5. **Stale files:** `table_406.tex`, `table_i_raven_per_constellation_test.tex`,
   `scripts/BEST_MODELS_RESULTS.md` still show the invalid 68.81% / 68.83% numbers.

---

## 7. Experiments that did not help

* Within-panel edges: behind at matched epochs (epoch 5: 19.8 vs 27.0%) and 55% slower — but only a
  5-epoch single-seed test; likely diluted by shared weights + one AVG over all edges.
* Panel layout features (object count, occupancy): 45.1% → 26.7% / 34.3% at 12 epochs.
* dim 256; GAT in torch; panel super-nodes in torch (object_only was the biggest torch factor).

## 8. Improvement ideas (not yet run)

1. **Symbolic reasoner end to end** (perceived attributes → `pynl_symbolic.py`): estimated ~90%; a
   different model (designed relations); a hybrid (symbolic scores as GNN inputs) is a middle path.
2. **GNN panel-wide rules:** (a) relabel `ctx_ctx` as same-row / same-column / other (no extra edges);
   (b) within-panel edges with their own weights or per-type aggregation; (c) add the slot's position
   embedding to `absent`. Test each at 2000/cons × 12 epochs against `laygrid_nosp_off` (45.1%).
3. Data: swap rows 1 and 2 (valid augmentation); generate fresh puzzles with the public I-RAVEN generator
   (protocol caveat).
4. LR-decay fine-tune; 3 seeds + ensembling (seeds are needed for reporting anyway).

---

## 9. PyNeuraLogic 0.9.x (what changed, what we use)

0.9.0 (2026-06-07): `Template` → `Model`; Python ≥ 3.10; optimisers in `neuralogic.nn`;
`Model.save/load` (pickle); built-in `Trainer` + callbacks; epochs passed to `train()`.
0.9.1 (2026-08-16): static graphs (`build_static_dataset`); weighted-query and torch-sync fixes.
0.9.2 (2026-08-30): Glorot default init; `set_fact_value` no longer writes through a JVM-shared (zero)
weight — this matters for our torch wrapper; fixed weights saved/loaded; torch-aligned tests; built-in
GNN layers (GCN, SAGE, GIN, GATv2, RGCN…) checked against torch_geometric; `weight_decay`, clipping.
We use: `Model`, `neuralogic.nn.Adam`, `Model.save`, epochs in `train()`, `.fixed()`, and benefit from the
0.9.2 fixes. Unused: `Trainer`, static graphs (the torch wrapper does the same), built-in GNN layers.

---

## 10. Files

PyNeuraLogic scripts and jobs are in `pynl/`, notebooks in `notebooks/`, the torch GNN in `torch_gnn/`,
the ResNet-18 frontend in `frontend/`. Everything runs from the repository root.

| file | role |
|---|---|
| `train_all_constellations.py` | GNN-style model: graph :56, template :82, torch side :121, CONCAT probe :149, forward :175, encode :191, training :227 |
| `train_pynl.py` | XML parser `parse` and `POS_MAP` |
| `pynl_symbolic.py`, `rpm_symbolic.py` | symbolic models (prims / rules) and their puzzle reader |
| `rpm_reference_solvers.py` | non-learned primitive and rule solvers |
| `eval_pynl_test.py` | test evaluation + per-constellation torch means |
| `pixel_pipeline.py`, `submit_pixels.sh` | end to end: `perceive` (nvsa, GPU) and `solve` (pynl, CPU) |
| `sanity_checks.py` | splits / shortcut / images / model checks, each with a negative control |
| `submit_max.sh`, `submit_symbolic.sh`, `submit_gnn_fixrel.sh` | SLURM jobs |
| `pynl_iraven_solver.ipynb` (+ `thesis/pynl_iraven_solver.html`) | compact end-to-end notebook, model inline, verified identical to the training code |
| `pynl_best_architecture_walkthrough.ipynb`, `pixels_to_answer.ipynb` | longer walkthroughs (HTML in `thesis/`) |
| `thesis/CODE_INDEX.md` | index of the PyNeuraLogic code; superseded experiments in `archive/pyneuralogic_experiments/` |
| `thesis/PYNEURALOGIC_NOTES.md` | the detailed lab notes (status, bugs §10, pixels §11, files §12, open questions §13, audit §14); status block predates the 3-seed torch and symbolic results |
| `thesis/pyneuralogic_report.html` | report, published as https://claude.ai/artifact/JZNH9G6j2DaYVNKG4BUVfn (v3, stale: lacks symbolic + 3-seed numbers) |
| `archive/pyneuralogic_2026-03-31_to_05-06/` | the April work (`scripts/raven_pyneuralogic.py`) |

---

## 11. Notebook decisions the user made

* `pynl_iraven_solver.ipynb`: title "PyNeuraLogic I-RAVEN solver"; no "minimal walkthrough" wording;
  no sbatch instructions; no training-time numbers; says **"end to end"**, never "from pixels"; compares
  only against the old torch GNN **68.81% "(trained with the position bug)"**; starts from the images,
  shows the puzzle, the ResNet-18 stage, then the reasoner; example = unseen distribute_four val #303
  (`RAVEN_1273_val`), answered correctly (candidate 7, 97.1%), with a comment that it was picked as one
  the model gets right and §10 gives the real accuracy.
* The user does not want to share the cluster or store passwords; an executed notebook (outputs saved)
  plus HTML is the deliverable. A laptop-runnable bundle (~210 MB: checkpoints, frontend, a few puzzles)
  is possible if a supervisor asks to run it.

## 12. The example puzzle (#303), explained

Rules: Number/Position Constant, Type Progression (+1), Size Constant (slot-wise), Colour Arithmetic
(3rd = 1st − 2nd). Row 3: square → pentagon → **hexagon**; colour 7 − 7 = **0 (white)**; 3 objects at
top-left, top-right, bottom-right with sizes 4, 2, 5. Candidate 7 satisfies all; candidate 0 is identical
except grey (colour 4) → breaks only the colour rule; candidate 1 breaks only layout; 5 only type; 6 all
three. The model gave 7: 97.1%, 0: 1.7%, 1: 1.1% — its runners-up are exactly the one-attribute-off
candidates.

## 13. Work launched on 2026-10-19

* **GNN ablations** (`submit_ablations.sh`, job 11825875, ~1–1.5 days): 2000/cons × 12 epochs against
  `laygrid_nosp_off` (45.1% val): `ablate_control`, `ablate_rowcol` (`--edge-labels rowcol`),
  `ablate_sp_pertype` (within-panel edges + `--per-type-weights`), `ablate_absentpos` (`--absent-pos`),
  `ablate_combined`. The switches are off by default; with defaults the code reproduces the trained
  checkpoint's outputs exactly (verified). `eval_pynl_test.load_model` reads them from a run's args.
* **Symbolic models end to end** (`pynl_symbolic_e2e.py`, job 11825915) — done: rules 98.33% XML /
  96.31% end to end on the test split (see §5).
* **Symbolic learning curves and random init** (`submit_symbolic_extra.sh`, job 11825919) — done (see §5).
* Done the same day: notebook headline updated to the final numbers and re-run (example still correct,
  93.7%; its 350-puzzle sample shows 66.0%, noise against the 14,000-puzzle test); notes status, §15
  (symbolic) and §16 (ablations) written; report republished as version 4; `table_iraven_final_test.tex`
  created; everything committed to the local branch `pyneuralogic-thesis` (commit fd9a035, 117 files,
  `.gitignore` keeps data, checkpoints, logs, `*.out` and `out/` out; not pushed anywhere).

## 14. Open items

* Read the results of §13 when the jobs finish and fold them into the notes, report and this file.
* Seeds for the full PyNeuraLogic model (each full run took ~19 days; decide after the ablations, since
  a better architecture would make seeds of the current one less useful).
* Hybrid GNN + symbolic scores, or the symbolic model as the main end-to-end system (depends on §13).
* Push `pyneuralogic-thesis` to a private repository of the user's (needs their own credentials; never
  IBM's upstream).
