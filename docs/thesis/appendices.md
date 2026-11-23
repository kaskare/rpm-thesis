# Appendices (draft, 2026-11-23)

Six appendices: the PyNeuraLogic pitfalls met in this work (A), the checks made on the data (B), the facet template as
PyNeuraLogic prints it (C), the full result tables (D), two negative results (E) and how to reproduce the main results
(F).

Conventions are those of the chapters. Cells are matched % with exact % in brackets. I-RAVEN numbers are on the unseen
validation puzzles (300–1299 of every layout) unless marked *test*; I-RAVEN-X numbers are on puzzles 6300–7299. The
epoch is always picked on separate selection puzzles.

## Appendix A. PyNeuraLogic pitfalls

All PyNeuraLogic models in this thesis run on PyNeuraLogic 0.9.2, which executes on the Java virtual machine and on
CPUs only. The five pitfalls below each cost real debugging time, and none of them produces an error message that
points at its cause. Each is given with its symptom, its cause and the fix used in the code. A.6 lists the smaller
ones.

### A.1 Isomorphic value compression must be off for native training

**Symptom.** The symbolic templates, whose rules are given and only their weights learned (Section 5.1), trained
badly: training accuracy collapsed, and the scores of some candidates did not move at all during training. Nothing was
reported as wrong.

**Cause.** `Settings(iso_value_compression=True)`, the default, merges neurons whose *initial values* coincide, as an
optimisation for inference. In those models every weight starts at the same value, so the score of a candidate can
start out equal to the value of a constant fact. The two neurons are then merged, and the candidate's score stays
frozen at that constant for the whole of training. The optimisation is correct for a fixed network and wrong for one
whose weights are about to change.

**Fix.** Every native training call in the code builds the model with

```python
model.build(Settings(optimizer=Adam(lr=args.lr), error_function=MSE(), iso_value_compression=False))
```

(`core/pynl/bench_hub_grounding.py`, `value_axis.py`, `pynl_symbolic.py`, `facet_stream.py` and the scoring scripts).
PyNeuraLogic's torch wrapper, `neuralogic.nn.torch_function.NeuraLogic`, already switches it off, which is why the
problem appears only in native training. The comment at the build call in `core/pynl/pynl_symbolic.py` records how
large the collapse was.

### A.2 Chain pruning removes the transformation of atoms with a single rule

**Symptom.** A nonlinearity declared on an atom, such as `(R.rel / 3) | tanh`, has no effect. The network trains and
its outputs look plausible, but part of it is linear where the template says it is not.

**Cause.** The default `chain_pruning=True` removes every atom neuron of the grounded network that has a single
incoming rule neuron, and its transformation goes with it. The query atom keeps its own. This acts on the ground
network, not on the template text: an atom whose template has two rules, of which only one grounds for a given
puzzle, loses its transformation as well. Aggregation does not protect an atom either: one rule averaged over several
groundings still feeds the atom through one neuron. All three cases were checked on a toy template, where a tanh
declared on a single-rule atom was applied with pruning off and dropped with it on, and kept in both settings once
the same sum was split over two rules.

**Fix.** Turning chain pruning off is one option; on an early template it did not help (FINDINGS 9). The code keeps
the default and splits the sum of an atom that needs its nonlinearity over two rules. In the facet template (Appendix
C), `rel` and `ans`, and `relp` and `ansp` in the position channel, are written this way, which the code explains in
a comment: without these two tanh layers everything after `vF` would be one linear map, which cannot express that the
relation of the complete rows decides how the question row's panels combine. The copy gates and every step of the row
fold are split in the same way so that their sigmoid and tanh survive. The other atoms with a single grounded rule
keep the declared tanh only on paper: `f0`, `pv1`, `pv2`, `ctx`, `vP` and `ctxp`, and `vF` for the number attribute,
whose second rule has no facets to ground on. Every result in this thesis was obtained with the template as printed in
Appendix C and with this behaviour.

### A.3 Unweighted selector atoms add 1.0 unless declared hidden

**Symptom.** The facet model, trained on all seven layouts, stayed far behind on the 2×2 and 3×3 grids, although its
grid component is built exactly as in the in-out grid layout, where it did well. It placed the objects of every grid on
one set of slots, and moving a puzzle's objects to other slots did not change the positions it generated.

**Cause.** In a lifted relational network an unweighted body atom contributes its value to the rule's sum. A
"selector" atom that is meant only to ground the rule, such as `obj(P, O)` or `ocomp(O, C)` in the occupancy rule as
it was written before the fix (printed with `--visible-selectors`),

```
occ(P, C) :- obj(P, O), ocomp(O, C), pos(O). [transformation=identity, aggregation=sum]
```

has the value 1.0, so it adds 1.0 to every component of the result. Each object therefore added its one-hot slot
*plus 2 in every slot*, and the occupancy vector of a panel told the model little more than how many objects it held.
This is the semantics of the formalism and not a defect of the library. The attribute channels suffered less, because
they average over objects, so there the selectors add a constant instead of an offset that grows with the count.

**Fix.** Atoms that only ground a rule are declared hidden, `R.hidden.obj(...)`, printed with a leading `*` in
Appendix C. `build_facet_template(..., hidden=True)` does this for every structural atom (`obj`, `ocomp`, `samepos`,
`col`, `row`, `row3`, `full`, `qcomp`), and `--visible-selectors` restores the old behaviour as a control. Trained on the
three grid layouts, the model went from 76.9% to 92.4% matched (FINDINGS 10). Hidden atoms are safe from version 0.9.2
on, which fixed a misalignment of weights around them. The weighted fact `puzzle(G)`, which serves as a bias, is
unchanged.

### A.4 The Java runtime must be pinned on the compute nodes

**Symptom.** A job that runs on the login node dies on an `amd` compute node with JPype's `JVMNotFoundException`, before
any PyNeuraLogic code is reached.

**Cause.** `ml load Java/21` silently does nothing on the amd compute nodes of the cluster: `JAVA_HOME` comes back empty,
although the same module works on the login node. A second trap sits in the same place: a user site-packages folder
for Python 3.11 shadows the conda environment with other versions of some libraries.

**Fix.** Every job script that runs PyNeuraLogic pins the runtime and switches the user site off:

```bash
source /mnt/appl/software/Anaconda3/2024.02-1/etc/profile.d/conda.sh && conda activate pynl
export PYTHONNOUSERSITE=1 JAVA_HOME=/mnt/appl/software/Java/21.0.8 && export PATH=$JAVA_HOME/bin:$PATH
```

### A.5 Grounding memory

**Symptom.** Large runs die while grounding, before the first epoch (an early run on I-RAVEN-X with rows of 10 and values
0–1000 died at 180 GB). A Jupyter kernel running PyNeuraLogic dies with the default heap of the virtual machine, and
CPU-heavy kernels are killed on the login node anyway.

**Cause.** PyNeuraLogic grounds every puzzle into a network of its own and keeps all of them in the heap of the Java
virtual machine. Memory therefore grows with the number of puzzles, with the length of the value codes and with the
value range. Grounding takes 10–165 ms per puzzle, once (FINDINGS 13), but on the largest runs it took 30–45 minutes in
all, and a grounded wide column-roles puzzle takes up to 12 MB. Updates are one puzzle at a time, since native training
does not batch. By contrast, the torch baselines of the same size train on one GPU in a few GB.

**Fix.** The heap is set explicitly, `neuralogic.set_max_memory_size(...)`, through `--jvm-gb` in
`bench_hub_grounding.py`, and the job requests the memory that the run needs. The largest runs in RESULTS §9 are:

| run | memory requested / peak | time |
|---|---|---|
| facet + copy links, I-RAVEN-X 3×10, values 0–1000 (5,426k weights) | 420 GB / 344 GB | 16.3 h |
| facet + content gates, I-RAVEN, all seven layouts | 420 GB / 327 GB | 15.0 h |
| row fold, 3×10, values 0–50 | 160 GB / 71 GB | 5.0 h |

For the 100,000 Arithmetic-only puzzles of Appendix E, which do not fit a node, `core/pynl/facet_stream.py` grounds the
puzzles in chunks, again in every epoch. The rescoring job starts one process per checkpoint so that the virtual
machine starts with an empty heap each time, and notebooks are executed as batch jobs on compute nodes.

### A.6 Smaller pitfalls

The other pitfalls listed in FINDINGS 5, in brief:
- embedding tables must be declared as weighted facts, otherwise the rule silently fails to ground and the error names
  the query;
- every sample needs its own query atom, because samples with the same query are merged without warning;
- `build_dataset()` may be called only once per built model;
- `Aggregation.CONCAT` follows the grounder's order of the constants, not the order in which they were named;
- rules with the same head are summed;
- `Transformation.NORM` rescales any non-zero vector to unit length and leaves a zero vector at zero. In the facet
  model value 0 is the all-zero thermometer code, and since the template has no biases, a value of 1e-6 in its place
  flips outputs. This is why the facet model cannot take graded inputs from the recognizer without retraining (FINDINGS
  39).

---

## Appendix B. Checks on the data

### B.1 The torch trainer reproduces the facet model's targets

The torch baselines run in another environment (Python 3.7, torch, no PyNeuraLogic), so they cannot call the facet
model's code. `data_prep/prepare_baseline_data.py` therefore writes the facet model's puzzles and splits as plain
arrays, and the baselines compute the targets and metrics from these arrays with their own code. To make sure that
both compute the same thing, the same script also stores what the facet pipeline itself computes for the first
puzzles of every split:
- `check_target`: the facet model's target, `facet_target`, for the answer panel of each of the first 20 puzzles, per
  component;
- `check_spec`: the decoded answer, `decode(describe(...))`, as one integer vector of occupancy bits and value counts
  per attribute.

Before training, `core/baselines/train_baseline.py` recomputes both from the arrays for every split and stops unless
the targets agree to 1e-6 and the decoded answers are identical. One exception is built in. I-RAVEN-X's Distribute rule
draws values up to the maximum inclusive, one more than the codes and counts hold (FINDINGS 20). The code of that value
stays distinct, so matching is unaffected, but the facet pipeline's decoder writes the value into the next attribute's
counts. Such answers are left out of the decoded comparison, and the baselines score exact generation against the
decoded true code. The facet model's own scorer had the same flaw: it capped exact generation at 95.6% on 3×3 and at
98.4% on 3×10. It was fixed and the I-RAVEN-X checkpoints were rescored on 2026-11-08; the facet model's 3×3 exact went
from 92.7 to 97.0, and on 3×10 nothing moved by more than 0.6. Matching, the choice of epoch and the per-rule exact
values were never affected.

The weights are checked in the same spirit. The baselines are sized to the facet model's weight count, which the
script reads by building the facet template, and `train_baseline.py` asserts that every model it builds has the number
of weights it was sized to. All 102 torch checkpoints agree with their `best.json` (RESULTS §9). For the test pass,
`core/baselines/test_eval.py` checks that the images and the label arrays list the same puzzles, and every checkpoint
has to reproduce its recorded validation numbers exactly before it is scored on test once (RESULTS §1).

### B.2 What the generators do to each rule

Section 3.5 gives, for each transformation of a puzzle, the share of rule instances that still satisfy a rule of the
same kind afterwards, measured on 1000 training puzzles per I-RAVEN layout and 6000 per I-RAVEN-X file
(`analysis/dataset_symmetry_check.py`). No instance fails its own label before the transformation. This section adds
what the generators' code says about the same transformations, which explains the shares.

- **Rows 1 and 2 exchanged.** In the I-RAVEN generator each row starts from a fresh draw and is built by the same rule
  objects, and Distribute Three picks its direction with equal probability, so for rows of three the exchange maps one
  direction onto the other. The exchange is a data symmetry, with one small exception: for Arithmetic on colour the
  generator redraws row 2 so that plus and minus stay distinguishable. In I-RAVEN-X with rows of 10, Constant,
  Progression and Arithmetic rows are drawn independently, but a Distribute row is the first row rotated by a constant
  step. After the exchange no step fits. The exchanged puzzle is therefore never generated, yet its answer, the value
  that the third row lacks, stays the same: a symmetry of the task but not of the data. With rows of three the exchange
  survives, because a rotation by three steps leaves a row of three unchanged.
- **Columns.** Progression dies under any exchange of columns, because a Progression row read in another order is not
  one. Constant survives every exchange, and Distribute survives an exchange of columns 1 and 2 in rows of three but
  not in rows of 10, where its rows are rotations. Of Arithmetic, what survives is the part that does not depend on the
  order of the operands: the plus rows under an exchange of columns 1 and 2, and every Arithmetic row of I-RAVEN-X
  under an exchange of middle columns. The generators do not draw the operands symmetrically, though. I-RAVEN draws
  the second operand from what the first leaves, and I-RAVEN-X draws its operands one after another from a shrinking
  range, shuffles them and then moves a zero out of the last operand slot. Most operands are zero, and the share of
  zeros depends on the column. Operand order is therefore a symmetry of the task but not of the data (Section 6.6).
- **Values reflected**, v mapped to max − v. Constant and Distribute use only equality, and Progression uses order and
  the successor, so all three keep their rule. Arithmetic uses addition with zero and keeps nothing.
- **Columns reversed**, which moves the missing panel. It maps a Progression step d to −d, plus to minus and one
  Distribute direction to the other, and it is a data symmetry of both generators on complete matrices. Because it moves
  the blank, it relates puzzles with different questions, so it is a symmetry for a model that fills any cell, not for
  ours. Union and difference on positions do not survive it: a union cannot be undone.
- **Grid positions.** Progression on Position moves through the slots in reading order, cyclically, so only the grid
  maps that keep that cycle keep the rule: on the 2×2 grid the left-right and up-down mirrors and the half turn, on the
  3×3 grid the half turn only. The set rules on positions (Constant, union and difference, Distribute) survive every
  permutation of the slots.
- **Attributes.** In I-RAVEN-X, size and colour are drawn by the same function with the same rules and range, so
  exchanging them is an exact data symmetry. Type never carries Arithmetic (B.3), so exchanging type with size or colour
  holds only away from Arithmetic. I-RAVEN has no exact exchange of attributes: type, size and colour have 5, 6 and 10
  values, Arithmetic on size has an extra +1, and type has no Arithmetic.
- **Components.** The left and right components, and the upper and lower ones, have identical constraints and matching
  rule counts and means, so exchanging them is an exact data symmetry. The inner and outer components of the in-out
  layouts are not exchangeable: the outer object is always white and larger. Rules act on each component separately, so
  exchanging components is still a symmetry of the task in every layout.
- **Nuisance.** `Angle` is drawn at random and no rule governs it (FINDINGS 3). The facts leave it out, so the models
  are invariant to it by construction.

### B.3 Rules per attribute in I-RAVEN-X

On the I-RAVEN-X training puzzles 0–5999, size and colour each carry the four rules equally often, about 25% each,
while type never carries Arithmetic and carries each of the other three rules a third of the time (RESULTS §7). A
recount from the data file for rows of 10 and values 0–50, made for this appendix, agrees. Two consequences are used
in the thesis. The attribute symmetry that the facet model builds in is exact between size and colour and holds for
type only away from Arithmetic. And when weight sharing across attributes is priced in training puzzles (Section 6.2),
the factor counted in advance is 2 on puzzles with Arithmetic, where only size and colour contribute examples, and
between 2 and 3 overall; the runs met both counts.

### B.4 Other checks

Three checks were made before any model was trained (FINDINGS 2 and 4, `analysis/sanity_checks.py`, each with a
negative control). I-RAVEN's annotation lists only the occupied slots of a grid, and deriving a slot from an object's
index in that list, as the inherited code did, made about a third of the grid puzzles unanswerable; slots are now
derived from the bounding boxes. No puzzle occurs in two splits, and the answer index is uniform. A model that sees only
the eight candidates is at chance on I-RAVEN, against 90–100% on RAVEN, and the annotation matches the images.

---

## Appendix C. The facet template

This is the facet template of Chapter 4 as PyNeuraLogic prints it, in the configuration of the main I-RAVEN model
(widths 32, 64 and 128; 10-bit codes; 22 slots; 266,528 weights). It was printed by building the template, without
grounding or training:

```bash
python3 -c "import sys; sys.path.insert(0, 'core/pynl'); import bench_hub_grounding as B; \
print(B.build_facet_template(32, 2, widths=[32, 64, 128]))"
```

### C.1 Notation

- `h(X, Y) :- b1, b2.` is a rule with head `h` and body atoms `b1`, `b2`. Upper-case arguments are variables, lower-case
  ones are constants.
- `{m, n} b(...)` multiplies the value of `b`, a vector of length n, by a learned m×n matrix. A body atom without
  braces enters with its own value, unweighted.
- `*b(...)` marks a hidden atom: it takes part in grounding but adds nothing to the sum (Appendix A.3).
- `[aggregation=avg]` after a rule says how its groundings for one head are combined; `sum` is the other choice used
  here, and the library's default is the average. Rules with the same head are summed.
- `h/2 [transformation=tanh]` sets the activation of every atom of predicate `h` with two arguments. Under the default
  chain pruning it is applied only where an atom has more than one grounded rule (Appendix A.2).
- `[combination=concat]` joins the body atoms into one vector instead of summing them; `elproduct` multiplies them
  element by element.

The facts of a puzzle are: `fv(O, A)`, the thermometer code of object O's value of attribute A; `nm(P, C)`, the code of
the number of objects of component C in panel P; `pos(O)`, the one-hot slot of O; `obj(P, O)` and `ocomp(O, C)`, which
place an object in its panel and component; `samepos(O1, O)`, for two objects in the same slot of two panels of one
row; `col(P, c)`, the column of a panel; `row(Rw, P1, P2, P3)` and `full(Rw)` for the two complete rows; `row3(P1, P2)`
for the two context panels of the question row; `qcomp(G, C)`, the component that the sample asks for; and `puzzle(G)`,
a scalar fact of value 1. Panels are `p0` to `p7`, components `k0` and `k1`, and A ranges over `type`, `size`, `color`
and `number`.

### C.2 The template

```
f0(O, A) :- {32, 10} fv(O, A).
f0/2 [transformation=tanh]
pv1(P, C, A) :- *obj(P, O), *ocomp(O, C), {64, 32} f0(O, A). [aggregation=avg]
f1(O, A) :- *obj(P, O), *ocomp(O, C), {64, 64} pv1(P, C, A).
f1(O, A) :- *samepos(O1, O), *obj(P, O), *col(P, c1), *obj(Q, O1), *col(Q, c2), {64, 32} f0(O1, A). [aggregation=avg]
f1(O, A) :- *samepos(O1, O), *obj(P, O), *col(P, c1), *obj(Q, O1), *col(Q, c3), {64, 32} f0(O1, A). [aggregation=avg]
f1(O, A) :- *samepos(O1, O), *obj(P, O), *col(P, c2), *obj(Q, O1), *col(Q, c1), {64, 32} f0(O1, A). [aggregation=avg]
f1(O, A) :- *samepos(O1, O), *obj(P, O), *col(P, c2), *obj(Q, O1), *col(Q, c3), {64, 32} f0(O1, A). [aggregation=avg]
f1(O, A) :- *samepos(O1, O), *obj(P, O), *col(P, c3), *obj(Q, O1), *col(Q, c1), {64, 32} f0(O1, A). [aggregation=avg]
f1(O, A) :- *samepos(O1, O), *obj(P, O), *col(P, c3), *obj(Q, O1), *col(Q, c2), {64, 32} f0(O1, A). [aggregation=avg]
pv1/3 [transformation=tanh]
f1(O, A) :- {64, 32} f0(O, A).
f1/2 [transformation=norm]
pv2(P, C, A) :- *obj(P, O), *ocomp(O, C), {128, 64} f1(O, A). [aggregation=avg]
f2(O, A) :- *obj(P, O), *ocomp(O, C), {128, 128} pv2(P, C, A).
f2(O, A) :- *samepos(O1, O), *obj(P, O), *col(P, c1), *obj(Q, O1), *col(Q, c2), {128, 64} f1(O1, A). [aggregation=avg]
f2(O, A) :- *samepos(O1, O), *obj(P, O), *col(P, c1), *obj(Q, O1), *col(Q, c3), {128, 64} f1(O1, A). [aggregation=avg]
f2(O, A) :- *samepos(O1, O), *obj(P, O), *col(P, c2), *obj(Q, O1), *col(Q, c1), {128, 64} f1(O1, A). [aggregation=avg]
f2(O, A) :- *samepos(O1, O), *obj(P, O), *col(P, c2), *obj(Q, O1), *col(Q, c3), {128, 64} f1(O1, A). [aggregation=avg]
f2(O, A) :- *samepos(O1, O), *obj(P, O), *col(P, c3), *obj(Q, O1), *col(Q, c1), {128, 64} f1(O1, A). [aggregation=avg]
f2(O, A) :- *samepos(O1, O), *obj(P, O), *col(P, c3), *obj(Q, O1), *col(Q, c2), {128, 64} f1(O1, A). [aggregation=avg]
pv2/3 [transformation=tanh]
f2(O, A) :- {128, 64} f1(O, A).
f2/2 [transformation=norm]
pt(P, C, A) :- *obj(P, O), *ocomp(O, C), fv(O, A). [transformation=identity, aggregation=avg]
pt(P, C, number) :- nm(P, C). [transformation=identity]
pt/3 [transformation=identity]
vF(P, C, A) :- {128, 10} pt(P, C, A).
vF(P, C, A) :- *obj(P, O), *ocomp(O, C), {128, 128} f2(O, A). [aggregation=avg]
vF/3 [transformation=tanh]
rel(Rw, C, A) :- *row(Rw, P1, P2, P3), {128, 128} vF(P1, C, A), {128, 128} vF(P2, C, A).
rel(Rw, C, A) :- *row(Rw, P1, P2, P3), {128, 128} vF(P3, C, A).
ctx(C, A) :- *full(Rw), {128, 128} rel(Rw, C, A). [aggregation=avg]
ans(C, A) :- *row3(P1, P2), {128, 128} vF(P1, C, A), {128, 128} vF(P2, C, A).
ans(C, A) :- {128, 128} ctx(C, A).
oa(G, A) :- {10, 1} puzzle(G), *qcomp(G, C), {10, 128} ans(C, A).
rel/3 [transformation=tanh]
ctx/2 [transformation=tanh]
ans/2 [transformation=tanh]
oa/2 [transformation=identity]
occ(P, C) :- *obj(P, O), *ocomp(O, C), pos(O). [transformation=identity, aggregation=sum]
occ/2 [transformation=identity]
vP(P, C) :- {32, 22} occ(P, C).
vP/2 [transformation=tanh]
relp(Rw, C) :- *row(Rw, P1, P2, P3), {64, 32} vP(P1, C), {64, 32} vP(P2, C).
relp(Rw, C) :- *row(Rw, P1, P2, P3), {64, 32} vP(P3, C).
relp/2 [transformation=tanh]
ctxp(C) :- *full(Rw), {64, 64} relp(Rw, C). [aggregation=avg]
ctxp/1 [transformation=tanh]
ansp(C) :- *row3(P1, P2), {128, 32} vP(P1, C), {128, 32} vP(P2, C).
ansp(C) :- {128, 64} ctxp(C).
ansp/1 [transformation=tanh]
op(G) :- {22, 1} puzzle(G), *qcomp(G, C), {22, 128} ansp(C).
op/1 [transformation=identity]
out(G) :- oa(G, type), oa(G, size), oa(G, color), oa(G, number), op(G). [transformation=identity, combination=concat]
out/1 [transformation=sigmoid]
```

### C.3 What each rule does

**The facets, `f0` to `f2`.**
- `f0(O, A) :- {32, 10} fv(O, A)` projects each object's code of one attribute to width 32. The rule has the attribute as
  a variable, so one matrix serves type, size and colour; this is where the attribute symmetry is built in.
- `pv1(P, C, A)` is the summary of a panel's component for the next layer: the average, over the objects of panel P in
  component C, of their projected facets.
- The first `f1` rule lets an object read the summary of its own panel and component.
- The six `f1` rules with `samepos` let an object read the facets of the objects in the same slot in the other columns
  of its row, averaged over such partners. Each ordered pair of columns has its own matrix (column 1 reading column 2,
  column 1 reading column 3, and so on). This is where "the same position means correspondence" and the order of the
  columns enter. The row is not named, so rows 1, 2 and 3 use the same matrices.
- The last `f1` rule is the residual. Since the width grows from 32 to 64, it is a learned projection rather than the
  identity.
- `f1/2 [transformation=norm]` normalises each facet to unit length.
- The `pv2` and `f2` rules repeat the layer, from width 64 to 128.

**The value of a component in a panel, `pt` and `vF`.**
- `pt(P, C, A)` is the average code of the component's objects, unweighted. For a thermometer code this is, bit by bit,
  the fraction of objects whose value exceeds that bit's threshold, which is the form of the target. For the number
  attribute, `pt(P, C, number)` is the code of the object count.
- `vF(P, C, A)` adds a projection of `pt` to the average of the final facets. Written as two rules, it keeps its tanh
  for type, size and colour. For number only the first rule grounds, so there it is linear (Appendix A.2).

**The rows, `rel`, `ctx` and `ans`.**
- The two `rel` rules form the relation of a complete row: each of its three panels enters through the matrix of its
  column. The sum is split over two rules only so that the tanh survives pruning. The row Rw is a variable, so the two
  complete rows are read by the same weights.
- `ctx(C, A)` averages the relation over the two complete rows. This is the prior "the same relation holds in every
  row", and the average makes rows 1 and 2 interchangeable.
- `ans(C, A)` combines the question row's two panels, each through its column's matrix, with the context. It is the
  model's answer for one attribute of one component.
- `oa(G, A)` decodes the answer to the 10 code bits. The weighted fact `puzzle(G)` gives each bit a bias of its own:
  "value greater than k" needs its own threshold for every k, and the template has no other bias. The hidden
  `qcomp(G, C)` selects the component that the sample asks for.

**The positions, `occ` to `op`.**
- `occ(P, C)` sums the one-hot slots of a component's objects into its occupancy. This is the rule of Appendix A.3,
  with the selectors now hidden.
- `vP`, `relp`, `ctxp` and `ansp` give the occupancy the same row structure as the attributes, with widths 32, 64 and
  128, and `op(G)` decodes it to the 22 slots with a bias per slot.

**The output, `out`.** The four decoded codes and the occupancy are concatenated and passed through a sigmoid. The loss
is the squared error against the description of the true answer's component.

No rule names an object, a complete row or a component; they appear only as variables. Of the attributes, only
`number` is named, by the rule that gives it its count as input, and `out` names all four to fix the order of the
outputs. By the renaming theorem of Section 3.2 the model is therefore blind to reordering objects, to exchanging rows 1
and 2, to exchanging the two components, and to exchanging type, size and colour together with their outputs. The rules
do name the columns, `c1`, `c2` and `c3`, and the question row is set apart by a predicate of its own, `row3`.

### C.4 Column roles and the row fold

Both variants are I-RAVEN-X models (Chapter 6 and Appendix E). Every line not mentioned below is the same as in C.2.
With **column roles** the facts give each panel one of `col(P, first)`, `col(P, mid)` and `col(P, last)`, the facet
rules read role pairs instead of column pairs, and the rows are read by one rule per role, averaged:

```
f1(O, A) :- *samepos(O1, O), *obj(P, O), *col(P, first), *obj(Q, O1), *col(Q, mid), {64, 32} f0(O1, A). [aggregation=avg]
    (and the other five role pairs, in f1 and f2 alike)
rel(Rw, C, A) :- *full(Rw), *inrow(P, Rw), *col(P, first), {128, 128} vF(P, C, A). [aggregation=avg]
rel(Rw, C, A) :- *full(Rw), *inrow(P, Rw), *col(P, mid), {128, 128} vF(P, C, A). [aggregation=avg]
rel(Rw, C, A) :- *full(Rw), *inrow(P, Rw), *col(P, last), {128, 128} vF(P, C, A). [aggregation=avg]
ans(C, A) :- *inrow(P, r3), *col(P, first), {128, 128} vF(P, C, A). [aggregation=avg]
ans(C, A) :- *inrow(P, r3), *col(P, mid), {128, 128} vF(P, C, A). [aggregation=avg]
```

The `mid` rule has the panel as a variable and averages over all middle columns, so the template names no middle
column, and any exchange of them is built in: the operands of Arithmetic become interchangeable. The position channel
(`relp`, `ansp`) changes in the same way. Since no column is named by its index, the same template runs on rows of any
length.

The **row fold** keeps the roles everywhere except in the attribute channel's reading of a row, which becomes a
recursion along the row with the facts `next(P, Q)` for neighbouring panels and `qlast(P)` for the last context panel
of the question row:

```
acc(P, C, A) :- *col(P, first), {128, 128} vF(P, C, A).
acc(Q, C, A) :- *next(P, Q), {128, 128} acc(P, C, A).
acc(Q, C, A) :- *next(P, Q), {128, 128} vF(Q, C, A).
rel(Rw, C, A) :- *full(Rw), *inrow(P, Rw), *col(P, last), {128, 128} acc(P, C, A).
rel(Rw, C, A) :- *full(Rw), *inrow(P, Rw), *col(P, first), {128, 128} vF(P, C, A).
ans(C, A) :- *qlast(P), {128, 128} acc(P, C, A).
acc/3 [transformation=tanh]
```

with `ans(C, A) :- {128, 128} ctx(C, A)` unchanged. The first `acc` rule starts the state at the row's first panel.
The next two are one step: the state at panel Q is a matrix times the state at its left neighbour plus a matrix times
Q's own value, through a tanh, with the same two matrices at every step. A step is written as two rules so that its tanh
survives pruning; the start state has a single rule and stays linear in `vF`. A complete row's relation reads the state
at its last panel together with its first panel, and the answer reads the state at the question row's last context
panel. The fold builds in one transition for every step, Progression's symmetry, and gives up the operand order, since
it reads a row in order.

### C.5 Content gates

Copy links with content gates add these rules to C.2:

```
gate(P, C, A) :- {10, 128} ans(C, A), {10, 128} vF(P, C, A), *inrow(P, Rw), *full(Rw). [combination=elproduct]
gate(P, C, A) :- [-6.0, -6.0, -6.0, -6.0, -6.0, -6.0, -6.0, -6.0, -6.0, -6.0] puzzle(G), *qcomp(G, C), *pc(P, C, A), *inrow(P, Rw), *full(Rw).
gate(P, C, A) :- {10, 128} ans(C, A), {10, 128} vF(P, C, A), *inrow(P, r3). [combination=elproduct]
gate(P, C, A) :- [-6.0, -6.0, -6.0, -6.0, -6.0, -6.0, -6.0, -6.0, -6.0, -6.0] puzzle(G), *qcomp(G, C), *pc(P, C, A), *inrow(P, r3).
copy(C, A) :- gate(P, C, A), pc(P, C, A). [combination=elproduct, aggregation=sum]
oa(G, A) :- *qcomp(G, C), copy(C, A).
gate/3 [transformation=sigmoid]
copy/2 [transformation=identity]
```

The fact `pc(P, C, A)` holds a context panel's code, scaled to ±4. A gate is a sigmoid of the answer and the panel's own
value, multiplied element by element, with one set of weights for the panels of the complete rows and one for the
question row: the panel is a variable, so rows 1 and 2 stay interchangeable. The bias, given as a learned vector that
starts at −6, keeps every gate nearly closed at the start. `copy(C, A)` sums the gated codes over the panels and adds
them to the output logits beside the decoder, so a copied value skips the tanh layers and keeps its precision. The
per-panel gates of the main table instead give every context panel gate rules of its own, `gate(p0, C, A)`,
`gate(p1, C, A)` and so on, computed from the answer alone with weights of their own. That names the panels and gives
up the symmetry between rows 1 and 2 (Section 6.5).

---

## Appendix D. Full tables

The tables are compact versions of RESULTS §1, §3, §7, §11 and §14, with the numbers copied unchanged. "Weights" are given
per dataset, in the order of the columns, for the datasets a row has results on.

### D.1 Main table

Unseen puzzles, matched (exact). The facet model's exact on I-RAVEN-X was rescored after the scorer fix of Appendix B.1,
except for values 0–100 and 0–1000, which can rise by at most 0.4 and 0.1.

| model | weights | I-RAVEN | 3×3, 0–10 | 3×10, 0–50 | 3×10, 0–100 | 3×10, 0–1000 |
|---|---|---|---|---|---|---|
| **Facet model (PyNeuraLogic)** | | | | | | |
| facet, named columns, 60–80 epochs | 267k / 267k / 1,411k / 1,425k / 1,685k | 97.3 (69.1) | **99.2** (97.0) | 85.4 (15.0) | 84.9 (6.4) | 81.2 (0.0) |
| the same checkpoint, end to end through the ViT slot recognizer | 267k + 326k recognizer | 97.1 (68.8) | – | – | – | – |
| facet, 150 epochs | 267k / 1,411k / 1,425k / 1,685k | 97.4 (69.3) | – | 89.1 (23.8) | 86.3 (6.5) | 85.4 (0.1) |
| + copy links, per-panel gates | 277k / 277k / 1,598k / 1,799k / 5,426k | **98.2** (**72.1**) | 99.8 (99.3) | 97.8 (60.6) | **96.2** (30.8) | **94.3** (15.0) |
| + copy links, content gates | 272k / 272k / 1,436k | 97.4 (68.7) | 99.8 (98.5) | 94.9 (52.3) | – | – |
| column roles | 288k | (= named at 3 columns) | – | 94.1 (29.2) | – | – |
| column roles + content gates | 314k | – | – | 96.3 (43.7) | – | – |
| column roles, widths 64/144/288 (equal weights to named) | 1,390k | – | – | 94.9 (28.6) | – | – |
| row fold | 283k / 283k / 305k | 95.8 (63.0) | 99.6 (98.9) | 92.2 (28.0) | – | – |
| row fold + content gates | 330k | – | – | 93.7 (35.0) | – | – |
| **Symbolic, rules given** | | | | | | |
| symbolic template, rules (PyNeuraLogic) | 41 | **98.33** *test*; end to end 96.31 | – | – | – | – |
| symbolic template, primitives | 94 | 97.55 *test*; end to end 95.71 | – | – | – | – |
| reference solver, rules (nothing learned) | 0 | 97.78 *test*; end to end 95.74 | – | – | – | – |
| reference solver, primitives | 0 | 96.42 *test*; end to end 94.73 | – | – | – | – |
| **Torch baselines** (same task and metrics) | | | | | | |
| MLP, facet's weights | 264k / 267k / 1,597k / 1,802k / 5,383k | 60.6 ± 0.8 (17.1) | 92.8 ± 0.2 (64.8) | 83.5 ± 0.4 (3.9) | 80.0 ± 2.3 (0.5) | 74.3 ± 3.2 (0.0) |
| all-to-all object GNN, facet's weights | 265k / 265k / 1,593k / 1,798k / 5,430k | 87.2 ± 1.1 (51.5) | 92.0 ± 0.4 (58.5) | 71.6 ± 0.1 (1.2) | 64.7 ± 5.4 (0.0) | 54.1 ± 3.1 (0.0) |
| MLP, regularized | 264k / 266k / 1,599k / 1,803k / 5,384k | 72.0 ± 1.1 (27.5) | 95.1 ± 0.4 (74.3) | 84.5 ± 0.1 (2.7) | 76.3 ± 1.4 (0.1) | 73.8 ± 0.2 (0.0) |
| object GNN, regularized | 265k / 265k / 1,593k / 1,798k / 5,430k | 90.5 ± 1.2 (55.1) | 93.9 ± 0.2 (67.7) | 85.3 ± 1.9 (4.2) | 76.8 ± 2.1 (0.2) | 55.1 ± 1.0 (0.0) |
| strongest object-level baseline, picked on selection | 4,258k / 267k / 25,556k / 1,802k / 5,383k | 93.8 (63.7), GNN 16× reg. | 92.8 (64.8), MLP | 86.4 (2.1), MLP 16× | 80.0 (0.5), MLP | 74.3 (0.0), MLP |
| attribute GNN A, absolute labels | 266k / 1,596k | 95.0 (62.5) | – | 95.0 (27.7) | – | – |
| attribute GNN B, relative labels (symmetric) | 267k / 1,599k | 96.8 (67.9) | – | 85.8 (3.5) | – | – |
| **Black-box vision, end to end** | | | | | | |
| ViT over the matrix as one image, width 64 | 335k / 335k / 428k | 79.1 (37.6) | 35.4 (2.6) | 52.9 (0.0) | – | – |
| ViT, width 224 | 3,753k / 3,753k / 4,079k | 93.4 (64.4) | 47.5 (5.6) | 43.1 (0.4) | – | – |
| ViT, rows 1 and 2 built in, width 224 | 3,709k / 3,709k / 3,919k | 92.1 (62.1) | 44.0 (5.6) | 40.1 (0.0) | – | – |
| **Literature (reported, other protocols)** | | | | | | |
| ARLC, learned (trains on 2×2 only) | | 92.6 | 98.4 | 95.0 | 95.0 | 90.6 |
| ARLC, programmed | | – | 97.2 | 100 | 100 | 99.7 |
| GPT-4 | | – | 93.2 | 82.2 | 79.6 | 76.6 |
| DeepSeek R1 | | – | – | – | 84.0 | 82.8 |

Two seeds are given as mean ± sd; every other row is one run. "Regularized" is the setting picked on the selection
puzzles of I-RAVEN and of 3×10 with values 0–50, run unchanged on every dataset. The literature rows see the candidates
or train differently and are not directly comparable.

**The value axis** is a test bed that is given addition and features in the shape of the rules (Section 4.5). It is
compared only with itself, never with the models above.

| value axis | weights | I-RAVEN | 3×3, 0–10 | 3×10, 0–50 | 3×10, 0–100, trained on 0–50 |
|---|---|---|---|---|---|
| 6000 puzzles (I-RAVEN: per layout, slots and count added) | 18.0k on I-RAVEN, 10.8k | 99.3 (75.1) | 99.8 (99.7) | 100.0 (100.0) | 100.0 (100.0) |
| 6000 puzzles, without `plus` | 17.8k on I-RAVEN, 10.7k | 94.2 (59.0) | 94.4 (86.3) | 84.0 (59.6) | 78.9 (60.5) |
| 6000 puzzles, `plus` used by learned rules (no built-in sums or checks) | 15.8k | – | 99.6 (99.4) | 100.0 (100.0) | 99.2 (99.0) |

### D.2 I-RAVEN test split

Test puzzles 0–1999 of every layout, 14,000 puzzles, one pass per checkpoint (2026-11-14). Every checkpoint first
reproduced its recorded validation numbers exactly. Matched (exact); the layouts are matched.

| model | weights | validation | **test** | centre | 2×2 | 3×3 | left-right | up-down | in-out | in-out grid |
|---|---|---|---|---|---|---|---|---|---|---|
| facet, XML labels | 267k | 97.31 (69.09) | **97.19** (68.35) | 99.6 | 95.0 | 89.1 | 99.8 | 99.7 | 99.8 | 97.5 |
| facet, end to end (ViT slot recognizer) | 267k + 326k | 97.13 (68.79) | **97.03** (68.09) | 99.6 | 95.0 | 89.1 | 99.8 | 99.7 | 99.8 | 96.4 |
| attribute GNN B (symmetric) | 267k | 96.76 (67.94) | **96.79** (67.84) | 99.8 | 95.2 | 85.5 | 100.0 | 99.9 | 99.7 | 97.5 |
| attribute GNN A | 266k | 95.03 (62.53) | **95.09** (62.51) | 99.9 | 89.5 | 84.4 | 100.0 | 99.9 | 100.0 | 92.0 |
| object GNN, regularized, 16× | 4,258k | 93.79 (63.66) | **93.44** (63.66) | 99.8 | 88.2 | 74.5 | 99.8 | 99.8 | 99.8 | 92.2 |
| object GNN, regularized, seed 0 | 265k | 91.31 (56.51) | 91.71 (56.43) | 99.2 | 82.1 | 73.0 | 99.3 | 98.9 | 99.7 | 89.8 |
| object GNN, regularized, seed 1 | 265k | 89.66 (53.71) | 89.98 (53.84) | 99.4 | 80.4 | 71.8 | 99.0 | 98.7 | 99.5 | 81.0 |
| MLP, regularized, seed 0 | 264k | 72.81 (29.94) | 73.55 (29.59) | 94.4 | 46.4 | 38.8 | 91.2 | 87.1 | 95.3 | 61.6 |
| MLP, regularized, seed 1 | 264k | 71.29 (25.07) | 71.89 (24.81) | 91.0 | 46.9 | 37.4 | 84.7 | 84.3 | 95.0 | 64.0 |
| ViT, width 224, end to end | 3,753k | 93.40 (64.39) | **93.10** (64.36) | 99.1 | 92.8 | 80.5 | 99.8 | 99.6 | 99.8 | 80.0 |
| ViT, rows 1 and 2 built in, width 224 | 3,709k | 92.07 (62.14) | 92.26 (62.40) | 99.5 | 88.8 | 77.8 | 99.9 | 99.7 | 99.7 | 80.5 |
| ViT, width 64, end to end | 335k | 79.09 (37.61) | 78.69 (37.36) | 89.3 | 57.2 | 55.4 | 95.8 | 95.6 | 89.9 | 67.7 |
| ViT, rows 1 and 2 built in, width 64 | 323k | 89.43 (55.30) | 89.79 (54.52) | 98.7 | 79.7 | 71.3 | 99.2 | 99.0 | 99.0 | 81.6 |

Means of the two seeds on test: object GNN regularized 90.8 ± 1.2 (55.1), and with the facet's weights but without
regularization 87.75 ± 1.1 (52.0); MLP regularized 72.7 ± 1.2 (27.2), and with the facet's weights 60.8 ± 0.1 (17.1).
No checkpoint moves by more than 0.8 points between the two splits, and the order of the models is the same on both.

### D.3 Transformed and transposed puzzles

Does a model give the same answers when the puzzle is transformed? Each transformation is either a data symmetry or one
under which the answer stays the same (Appendix B.2). Matched %; "identical" means every answer is the same, and "built
in" that this holds by construction.

| model | as is | rows 1 and 2 swapped | middle columns 2 and 3 swapped | size and colour swapped |
|---|---|---|---|---|
| **I-RAVEN-X 3×10, values 0–50** | | | | |
| facet, named columns | 85.4 | 85.4 (identical) | 70.2 | identical (built in) |
| + copy links, per-panel gates | 97.8 | ≈60.0 (puzzles with Distribute: 98.3 to 39.5) | – | – |
| + copy links, content gates | 94.9 | 94.9 (identical) | 74.9 | – |
| column roles (± content gates) | 94.1 / 96.3 | identical | identical | identical |
| column roles, wide (± content gates) | 94.9 / 93.1 | identical | identical | – |
| row fold (± content gates) | 92.2 / 93.7 | identical | 91.7 / 93.1 | – |
| attribute GNN B (symmetric) | 85.8 | identical | identical | identical |
| attribute GNN A | 95.0 | 55.5 (Distribute exact 59.9 to 1.0) | 82.0 | 95.0 |
| object GNN, regularized (seed 0) | 84.0 | 52.7 | 71.6 | 85.0 |
| MLP, regularized (seed 0) | 84.4 | 47.7 | 63.9 | 83.5 |
| ViT, width 64 | 52.9 | 38.5 | 47.3 | – |
| ViT, width 224 | 43.1 | 37.9 | 44.4 | – |
| ViT, rows 1 and 2 built in, width 224 | 40.1 | identical | 39.9 | – |

On I-RAVEN, where these are data symmetries, the models that lack them barely move: attribute GNN A 95.0 as is, 95.0
with rows swapped and 95.0 mirrored; the regularized object GNN (seed 0) 91.3, 91.3 and 91.1; the regularized MLP
(seed 0) 72.8, 72.4 and 73.0; the ViT at width
224 93.4 and 93.4, at width 64 79.1 and 78.9. The facet model and GNN B are identical by construction.

**Transposed I-RAVEN puzzles** (2026-11-22, no training). Rows and columns of the 3×3 context are exchanged, so the
rules run down the columns; the answer keeps its place. Facet models on 1000 validation puzzles per layout, matched
(exact); reference solvers on ground-truth attributes, 2000 validation puzzles per layout.

| model | as is | transposed | exact by rule, transposed: Constant / Progression / Distribute / Arithmetic |
|---|---|---|---|
| facet | 97.3 (69.1) | 45.0 (3.9) | 39.6 / 39.8 / 82.1 / 13.0 (as is 83.7 / 89.7 / 88.7 / 83.8) |
| facet, copy links, content gates | 97.4 (68.7) | 45.1 (3.7) | 40.4 / 38.3 / 81.8 / 13.3 |
| row fold | 95.8 (63.0) | 45.1 (4.1) | 39.3 / 41.4 / 81.4 / 13.8 |
| reference: RAVEN's rules along rows | 97.8 | 47.7 | – |
| reference: generic row relations (primitives) | 96.3 | 72.5 | – |

All three facet models fall to the same 45%, close to RAVEN's rules applied along rows. Distribute Three keeps most of
its exact score because its values form a Latin square, which transposing keeps. This is the cost of the prior "rules
run along rows" when the generator turns the matrix, not its worth in data.

### D.4 Symmetry ablations, seeds and the data factor

**One symmetry removed** (Section 6.1): the components of the left-right and up-down layouts on I-RAVEN, and the
attributes of I-RAVEN-X 3×10 (values 0–50, column roles). One run per cell; its seed was not recorded.

| data | training puzzles | shared, symmetric | labels (shared weights) | separate weights, same widths | separate weights, matched count |
|---|---|---|---|---|---|
| I-RAVEN left-right + up-down (components) | 100 per layout | **86.6** (40.4) | 87.3 (41.3) | 75.6 (22.2) | 74.6 (19.6) |
| | 1000 | **99.3** (95.6) | 99.6 (95.7) | 97.9 (84.5) | 98.5 (84.5) |
| | 6000 | 99.9 (97.7) | 99.8 (98.7) | 100.0 (97.8) | 100.0 (98.8) |
| I-RAVEN-X 3×10, column roles (attributes) | 600 | **66.3** (16.4) | 64.6 (14.2) | 52.5 (4.1) | 53.0 (3.6) |
| | 6000 | **94.1** (29.2) | 93.9 (28.6) | 90.5 (17.8), still rising at epoch 60 | 88.1 (17.6) |

Weights: components 266,528 shared, 533,056 separate, 265,532 matched; attributes 288,328 shared, 932,190 separate,
288,214 matched. Labels add 320 or 512.

**Three seeds of the deciding cells** (2026-11-22). Per seed, then mean ± sd (exact).

| data, puzzles | shared, symmetric | labels | separate, matched count | gain over matched, per seed |
|---|---|---|---|---|
| components, 100 | 86.6 / 87.3 / 90.2: **88.0 ± 1.9** (42.4 ± 3.0) | 87.3 / 89.5 / 91.6: 89.5 ± 2.1 (44.6 ± 3.2) | 74.6 / 72.2 / 74.8: 73.8 ± 1.4 (19.4 ± 1.4) | 12.0 / 15.1 / 15.4: **+14.2 ± 1.9** |
| attributes, 600 | 66.3 / 68.8 / 63.4: **66.2 ± 2.7** (13.3 ± 2.8) | 64.6 / 67.0 / 62.9: 64.8 ± 2.1 (13.8 ± 0.3) | 53.0 / 53.1 / 51.9: 52.7 ± 0.7 (3.6 ± 0.0) | 13.3 / 15.7 / 11.5: **+13.5 ± 2.1** |
| attributes, 6000 | 94.1 / 94.2 / 94.4: **94.2 ± 0.2** (27.6 ± 2.0) | one run, 93.9 | 88.1 / 89.4 / 86.3: 87.9 ± 1.6 (15.4 ± 2.7) | 6.0 / 4.8 / 8.1: **+6.3 ± 1.7** |

The first value in each cell is the single run of the table above. Labels against the symmetry, per seed: +1.4 ± 0.8 on
components and −1.3 ± 0.7 on attributes.

**Three seeds of the 3×10 facet models**, 6000 puzzles, values 0–50. Exact value per rule.

| model | matched: first / seed 1 / seed 2 | mean ± sd (exact) | Constant | Progression | Distribute | Arithmetic | middle columns swapped |
|---|---|---|---|---|---|---|---|
| named columns | 85.4 / 88.4 / 88.5 | 87.4 ± 1.8 (18.4 ± 3.0) | 95.8 ± 2.5 | 86.0 ± 7.5 | 11.3 ± 1.3 | 20.7 ± 2.2 | 71.1 ± 1.0 |
| column roles | 94.1 / 94.2 / 94.4 | **94.2 ± 0.2** (27.6 ± 2.0) | 97.5 ± 2.8 | 92.8 ± 1.6 | 22.9 ± 0.8 | **39.6 ± 1.3** | identical |
| row fold | 92.2 / 91.1 / 91.5 | 91.6 ± 0.6 (26.7 ± 1.1) | 99.8 ± 0.2 | **97.2 ± 1.7** | 15.6 ± 0.5 | 32.4 ± 4.6 | 91.2 ± 0.5 |

Per seed, column roles over named columns: matched +6.8 ± 1.6, Arithmetic +18.9 ± 3.4, Distribute +11.6 ± 0.7,
Progression +6.8 ± 8.9. The fold over column roles on Progression: +4.4 ± 0.9, at every seed.

**The data factor, counted before the runs** (2026-11-22). The arm is separate weights at the same widths, so that each
copy is the same network as the shared model's.
- *Components.* With separate weights each copy trains only on its own component's samples, and under the data symmetry
  both components' samples have the same distribution. Predicted: separate weights at 200 puzzles equal the shared model
  at 100, a factor of exactly 2, the order of the group. Met: separate at 200, seeds 1–3, 87.9 / 88.0 / 88.2, **88.1 ±
  0.2** (42.2 ± 2.7 exact), against **88.0 ± 1.9** (42.4 ± 3.0) for shared at 100.
- *Attributes.* Predicted from the rule counts of Appendix B.3: a factor of 2 on puzzles with Arithmetic and between 2
  and 3 overall.

| separate weights, seeds 1–3 | matched | with Arithmetic (444) | without (556) | exact | Arithmetic exact |
|---|---|---|---|---|---|
| 1200 puzzles | 61.9 / 63.0 / 62.8: **62.6 ± 0.6** | 60.1 / 62.2 / 60.1: 60.8 ± 1.2 | 63.3 / 63.7 / 64.9: 64.0 ± 0.8 | 9.3 ± 0.9 | 8.6 ± 1.4 |
| 1800 puzzles | 69.1 / 70.8 / 69.6: **69.8 ± 0.9** | 68.7 / 70.5 / 70.3: 69.8 ± 1.0 | 69.4 / 71.0 / 69.1: 69.8 ± 1.0 | 13.2 ± 1.1 | 11.4 ± 1.4 |
| shared, 600 (for comparison) | 66.2 ± 2.7 | 63.0 ± 2.9 | 68.7 ± 2.8 | 13.3 ± 2.8 | 7.8 ± 1.0 |

Both predictions were met: the shared model at 600 lies between separate weights at 1200 and at 1800, a factor of about
2.5 read linearly, about 2 on puzzles with Arithmetic and about 3 on the rest.

**The value symmetry in the value-axis model** (3×10, values 0–50, `plus` given; compared with itself only).

| training puzzles | shared across values (10.8k) | weights per value (335k) |
|---|---|---|
| 100 | **99.0** (97.3) | 38.3 (4.0) |
| 600 | **99.9** (99.9) | 92.7 (81.0) |
| 6000 | **100.0** (**100.0**) | 98.4 (97.4) |

### D.5 Rule twins

I-RAVEN-X with rows of 3 and values 0–50, Arithmetic replaced by a test rule (`data_prep/rule_twins.py`). Exact % on
the test rule at 100 / 600 / 6000 puzzles, on unseen puzzles, epoch picked by exact on the selection puzzles. No model
is given `plus` except in the last row. The rows are read against each other as in Chapter 7: shared against per-value
weights, and with `plus` against without.

| model | weights | sum | midpoint | mirror | lookup |
|---|---|---|---|---|---|
| shared, shift-equivariant | 14,785 | 4.7 / 6.9 / 7.1 | 57.4 / 89.3 / 92.7 | 45.3 / 78.3 / 88.1 | 35.9 / 74.3 / 77.7 |
| per value | 459,635 | 4.9 / 19.3 / 69.8 | 5.7 / 17.8 / 75.2 | 4.6 / 15.7 / 63.2 | 53.0 / 97.5 / 97.5 |
| facet (600 / 6000) | 278,088 | 31.2 / 65.5 | 24.0 / 76.6 | 14.7 / 41.9 | 60.3 / 97.1 |
| MLP, mean of 2 seeds | 277,828 | 2.7 / 3.0 / 26.4 | 4.8 / 5.9 / 21.7 | 2.9 / 2.9 / 10.2 | 2.8 / 5.6 / 17.7 |
| memorising answered rows | – | 4.7 / 24.1 / 90.7 | 5.0 / 21.6 / 90.5 | 4.0 / 19.7 / 91.7 | 65.3 / 100 / 100 |
| shared + `plus` used by learned rules (100 / 600) | 19,937 | **97.6 / 99.8** | 70.5 / 91.5 | – | – |

**Three seeds** (2026-11-14), mean ± sd of seeds 1–3; one selection rule for every cell, fixed before the runs.

| model | sum, 100 puzzles | sum, 600 | midpoint, 100 | midpoint, 600 |
|---|---|---|---|---|
| shared, shift-equivariant | 4.5 ± 2.3 | 6.2 ± 0.8 | 60.3 ± 2.7 | 91.0 ± 3.4 |
| per value | 4.8 ± 0.6 | 19.4 ± 1.1 | 5.0 ± 0.5 | 18.0 ± 0.9 |
| shared + `plus` used by learned rules | 98.2 ± 0.8 | 99.7 ± 0.4 | 63.9 ± 3.9 | 91.7 ± 1.8 |

**Scaling, the rule gate and the steps, on midpoint** (2026-11-15 to 17). *Doubled* reads every row in units of the
largest power of two in its first step, *scaled* in units of its first step; *with the gate* multiplies the question
row's features by a sigmoid of the rule descriptor; *with the steps* lets the row descriptor read each neighbouring
pair's step directly. Exact %, seeds 1 / 2 / 3 (mean ± sd).

| model | 100 puzzles | 600 puzzles |
|---|---|---|
| shared | 57.2 / 62.0 / 61.8 (60.3 ± 2.7) | 87.1 / 93.1 / 92.9 (91.0 ± 3.4) |
| shared with the gate | 72.5 / 65.7 / 61.6 | 90.3 / 88.1 / 92.5 |
| shared with the steps | 54.9 / 63.4 / 61.4 (59.9 ± 4.5) | **96.0 / 99.4 / 99.4** (98.3 ± 1.9) |
| doubled | 55.6 / 48.7 / 45.3 (49.9 ± 5.3) | 65.5 / 69.5 / 67.5 (67.5 ± 2.0) |
| doubled with the steps | **90.1 / 91.3 / 86.1** (89.2 ± 2.7) | **99.4 / 99.0 / 99.2** (99.2 ± 0.2) |
| scaled | 19.0 / 100.0 / 21.0 | 29.5 / 100.0 / 20.2 |
| scaled with the gate | 23.2 / 20.2 / 24.0 | 20.2 / 20.2 / 30.5 |
| scaled with the steps | **100.0 / 100.0 / 100.0** | **100.0 / 100.0 / 100.0** |
| counting learners (no network): shared, doubled, scaled | 80.0, 94.3, 100 | 99.8, 99.8, 100 |

**Sum with the steps** (2026-11-22). The prediction, written before the runs, was that each model stays at its
ceiling, because the steps are read inside a row and nothing measures from the question row to row 1. Exact %, seeds
1 / 2 / 3 (mean ± sd).

| model | 100 puzzles | 600 puzzles | ceiling |
|---|---|---|---|
| shared with the steps | 4.9 / 2.0 / 5.1 (4.0 ± 1.8) | 8.1 / 6.7 / 8.1 (7.6 ± 0.8) | 7.9 |
| doubled with the steps | 4.3 / 3.0 / 3.0 (3.4 ± 0.8) | 5.9 / 4.5 / 5.1 (5.2 ± 0.7) | 7.2 |
| scaled with the steps | 1.4 / 1.2 / 1.0 (1.2 ± 0.2) | 2.4 / 1.0 / 1.6 (1.6 ± 0.7) | 1.4 |
| without the steps, means: shared, doubled, scaled | 4.5, 3.6, 2.0 | 6.2, 4.1, 2.6 | |

Every mean is at or under its ceiling within sampling error: with 507 rows the rate is fixed to about ±1.2 points near
7.9 and ±0.5 near 1.4.

### D.6 The test pass of Chapters 5 to 7

At the end of the work every final checkpoint behind an I-RAVEN-X, rule-twin or components number of Chapters 5 to 7
was scored once on puzzles no run had touched (2026-11-23, 187 checkpoints): puzzles 8000 to 9999 of each I-RAVEN-X
or twins file (2,000 puzzles), and the I-RAVEN test split 0 to 1999 of left-right and up-down for the components. The
rule mix of these ranges is that of the unseen ones (Arithmetic on 16.9% of attributes in both on rows of 10). The
unseen column is the cell as the chapters report it. Matched (exact) unless noted; three seeds where a ± is given.

**Chapter 6, one symmetry removed.**

| data, puzzles | arm | unseen | test |
|---|---|---|---|
| components, 100 | shared, symmetric | 88.0 ± 1.9 (42.4) | 88.4 ± 1.3 (40.6) |
| | separate weights, matched count | 73.8 ± 1.4 (19.4) | 72.9 ± 1.0 (19.1) |
| | gain per seed | +14.2 ± 1.9 | **+15.5 ± 0.9** |
| components, 200 | separate weights, same widths | 88.1 ± 0.2 (42.2) | **88.3 ± 0.6** (41.1) |
| attributes, 600 | shared, symmetric | 66.2 ± 2.7 (13.3) | 68.3 ± 2.9 (13.2) |
| | separate weights, matched count | 52.7 ± 0.7 (3.6) | 55.4 ± 0.8 (3.7) |
| | gain per seed | +13.5 ± 2.1 | **+12.9 ± 3.6** |
| attributes, 1,200 / 1,800 | separate weights, same widths | 62.6 ± 0.6 / 69.8 ± 0.9 | 65.0 ± 0.1 / 72.3 ± 1.0 |
| attributes, 6,000 | shared, symmetric | 94.2 ± 0.2 (27.6) | 94.3 ± 0.5 (28.6) |
| | separate weights, matched count | 87.9 ± 1.6 (15.4) | 89.2 ± 0.8 (17.2) |
| | gain per seed | +6.3 ± 1.7 | **+5.2 ± 1.3** |
| value axis, 100 / 600 | shared across values | 99.0 (97.3) / 99.9 (99.9) | 98.8 (97.5) / 100.0 (100.0) |
| | weights per value | 38.3 (4.0) / 92.7 (81.0) | 38.2 (4.6) / 94.3 (80.5) |

The counted data factor holds on test. On components, separate weights at 200 puzzles match shared weights at 100 to a
tenth of a point. On attributes, shared weights at 600 puzzles lie between separate weights at 1,200 and 1,800, about
2.5 read linearly; on the 888 test puzzles with Arithmetic separate weights are 1.6 points behind at twice the puzzles
and 7.7 ahead at three times, on the 1,112 without they are 4.6 behind at twice and 1.1 ahead at three times. At 600
to 1,800 puzzles every arm on attributes sits 2.0 to 2.7 points higher on test, all alike, so the gains move by less
than their spread. Labels against the symmetry: +0.5 ± 0.7 on components, −1.3 ± 2.2 on attributes.

**Chapter 5 and Section 6.6, facet models on I-RAVEN-X.**

| model | unseen | test |
|---|---|---|
| named columns, rows of 3, values 0–10 | 99.2 (97.0) | 99.6 (97.7) |
| named columns, rows of 10, values 0–50 | 87.4 ± 1.8 (18.4); Arithmetic 20.7 ± 2.2 | 87.1 ± 2.0 (18.6); Arithmetic 20.9 ± 0.5 |
| column roles | 94.2 ± 0.2 (27.6); Arithmetic 39.6 ± 1.3 | 94.3 ± 0.5 (28.6); Arithmetic **36.0 ± 2.1** |
| row fold | 91.6 ± 0.6 (26.7); Progression 97.2 ± 1.7 | 91.4 ± 0.4 (26.9); Progression 97.9 ± 1.0 |
| copy links, per-panel gates, values 0–50 / 0–100 / 0–1000 | 97.8 (60.6) / 96.2 (30.8) / 94.3 (15.0) | 96.5 (61.5) / 97.0 (31.6) / 95.4 (16.6) |
| copy links, content gates | 94.9 (52.3) | 95.8 (51.5) |
| columns 1 and 2 tied, rows of 3, 6,000 / 600 puzzles | 98.6 (95.1) / 92.1 (79.2) | 98.8 (94.2) / 92.6 (78.4) |
| v and 10 − v folded, 6,000 / 600 puzzles | 46.6 (13.4) / 39.7 (14.0) | 47.1 (14.2) / 42.8 (16.6) |

Column roles' Arithmetic drops by 3.6 points, at every seed. The seeds share the same 507 unseen Arithmetic rows, so
this is 1.4 standard errors of that sample, but it is the direction that choosing column roles with those puzzles in view
would leave. Their lead over named columns is +15.1 ± 1.8 per seed on test against +18.9 ± 3.4 on unseen. Copy links
lose 1.3 points matched on rows of 10 (2.1 standard errors); no other cell of this table moves by more than 1.1, except
the folded control at 600 puzzles, which stays at its ceiling.

**Chapter 7, the rule twins.** Exact % on the tested rule, unseen → test; 963 midpoint and 1,014 sum rows on test.

| model | sum, 100 | sum, 600 | midpoint, 100 | midpoint, 600 |
|---|---|---|---|---|
| shared, shift built in | 4.5 → 3.4 | 6.2 → 5.6 | 60.3 → 57.8 | 91.0 → 89.6 |
| per value | 4.8 → 4.4 | 19.4 → 17.1 | 5.0 → 5.6 | 18.0 → 19.0 |
| shared, `plus` used by learned rules | 98.2 → 97.6 | 99.7 → 99.0 | 63.9 → 61.5 | 91.7 → 91.0 |
| doubled | 3.6 → 3.2 | 4.1 → 3.6 | 49.9 → 48.0 | 67.5 → 66.2 |
| scaled, seeds 1 / 2 / 3 | 3.2 / 0.8 / 2.0 → 2.1 / 1.3 / 0.9 | 3.4 / 1.0 / 3.4 → 2.0 / 1.2 / 2.0 | 19.0 / 100 / 21.0 → 19.1 / 100 / 21.2 | 29.5 / 100 / 20.2 → 29.7 / 100 / 21.0 |
| shared with the steps | 4.0 → 3.6 | 7.6 → 6.6 | 59.9 → 59.4 | 98.3 → 97.4 |
| doubled with the steps | 3.4 → 2.2 | 5.2 → 4.0 | 89.2 → 88.2 | 99.2 → 99.1 |
| scaled with the steps | 1.2 → 1.3 | 1.6 → 1.2 | 100 → 100 | 100 → 100 |
| facet model, one run (600 / 6,000) | – | 31.2 → **24.7** / 65.5 → 64.4 | – | 24.0 → 21.8 / 76.6 → 78.0 |

Every sum mean stays at or under its ceiling within sampling error (7.9, 7.2 and 1.4; 1,014 rows fix a rate near 1.4
to about ±0.4). The ladder keeps the counted order, and the shared model with the steps stays about 20 points under its
count at 100 puzzles. The facet model on sum at 600 puzzles is the largest move of the whole pass against its sampling
error (2.6 standard errors); its lead over midpoint at 600 shrinks from 7.2 points to 2.9, and midpoint's lead at 6,000
grows from 11.1 to 13.6. The cells at 6,000 puzzles of the shared and per-value models saved the epoch picked by
matched, not the one their table reads, so their test pass compares with that checkpoint only: shared sum 7.5 → 8.4,
midpoint 92.9 → 91.0, per value sum 53.5 → 51.6, midpoint 75.2 → 73.8. The count of 0 of 101 unseen first steps was
made on the unseen puzzles and was not repeated on test.

---

## Appendix E. Negative results

### E.1 The row fold does not transfer to other row lengths

**What was tried.** Neither column roles nor the row fold (Appendix C.4) name a column by its index, so their weights do
not depend on the row length: a checkpoint trained on rows of 10 runs unchanged on rows of 5 or 20, with 288,328
weights for the roles and 304,712 for the fold. The fold seemed the better candidate for transfer, because it applies
one learned transition per column, the same at every step, and a toy recursive template trained on rows of 3–5 values
had summed rows of 8–10 exactly (FINDINGS 23). Both models were trained on I-RAVEN-X 3×10 with values 0–50 and scored
on unseen puzzles with rows of 5 and of 20, drawn by IBM's generator (`iravenx_task.py --n 5 / 20 --maxval 50 --nconf 0`,
seed 1234). When neither transferred, both were trained again on rows of 5 and 10 together, 6000 puzzles of each, with
the epoch picked on 300 selection puzzles of each. The prediction, written before that run, was that rows of 20 rise
well above the fold's 55% if one training length were the cause.

**What happened.** Neither transferred, after one training length or after two:

| model | rows of 5 | rows of 10 | rows of 20 |
|---|---|---|---|
| column roles | 65.6 (4.0) | 94.1 (29.2), trained | 50.1 (4.6) |
| column roles + content gates | 66.3 (5.2) | 96.3 (43.7), trained | 49.2 (4.6) |
| row fold | 53.8 (0.4) | 92.2 (28.0), trained | 55.1 (5.9) |
| row fold + content gates | 61.0 (8.1) | 93.7 (35.0), trained | 60.5 (10.1) |
| row fold, trained on rows of 5 and 10 | 87.9 (7.4), trained | 90.4 (23.3), trained | 47.7 (4.1) |
| column roles, trained on rows of 5 and 10 | 79.3 (14.2), trained | 93.6 (27.3), trained | 54.0 (7.9) |

On the lengths it was not trained on, every model stays between 47.7 and 66.3 matched, and only Constant survives. After
training on rows of 5 and 10, the exact value per rule on rows of 20 (Constant / Progression / Distribute / Arithmetic) is
99.7 / 2.9 / 11.3 / 4.7 for the fold and 98.3 / 38.6 / 5.8 / 3.9 for the roles. The second training length made rows of
20 slightly worse for the fold (47.7 against 55.1) and did not lift the roles (54.0 against 50.1), so the prediction
failed: one training length is not the cause (FINDINGS 33). The fold's Progression, 99.2 exact at the trained length and
its best rule (FINDINGS 23), falls to 2.9 on rows of 20. The exact cells on rows of 5 and 20 were scored before the
scorer fix of Appendix B.1 and can rise by at most 0.7 and 1.3 points, which changes nothing here.

On I-RAVEN, with rows of three, the fold has nothing to transfer and is below the facet model: 95.8 (63.0) against
97.3 (69.1), with Arithmetic exact at 75.9 against 83.8 (FINDINGS 23). It is therefore used as an I-RAVEN-X model only,
where it is the best model on Progression (97.2 ± 1.7 exact over three seeds, D.4).

**What it means.** Recursion makes transfer to other lengths possible, since the same weights apply at any length, but
it does not make it happen. Trained on rows of 10, nothing asks the recurrence to mean the same after 4 steps as after 9,
and the readouts learned to decode the state reached after the trained number of steps. Training on two lengths did not
change that. The run left two explanations open that it cannot separate: a state carried through up to 19 tanh steps
loses precision, and at longer rows the generator's Progression rows start from a narrower range of values, so the
values themselves shift. The roles, for their part, are blind to the order of the middle columns but not to their
number (FINDINGS 22). After this result the row-length line of work was closed, and the planned test on rows of 100, which was conditional on a
transfer, was not run.

### E.2 No grokking on Arithmetic

**Why it was tried.** On I-RAVEN-X with rows of 10, Arithmetic's exact score stays low in every learned model (D.4).
One explanation is that the training sets are too small or too briefly trained, and that a model would find
the sum late, after memorising, as in the grokking literature (Power et al. 2022; Nanda et al. 2023; Liu et al. 2023;
Varma et al. 2023). The setting most likely to show it is a training set small enough to memorise but above the critical size below which memorising stays the
cheaper solution (Varma et al. 2023), many updates, decoupled weight decay, and possibly larger initial weights (Liu
et al. 2023).

**Setup.** 100,000 I-RAVEN-X puzzles with rows of 10 and values 0–50 in which every attribute follows Arithmetic
(`data_prep/arith_big_data.py`), followed by 300 selection and 1000 unseen puzzles of the same kind. Training on these is
a diagnostic, not a benchmark result: the rule is known in training and test, so the runs ask whether a model can
learn the sum at all, not how it does on Arithmetic among the other rules. Grokking would show as Arithmetic exact high
on the training puzzles long before it rises on the unseen ones.

**What happened.** Arithmetic exact %, type, size and colour pooled, training / unseen (RESULTS §12).

| model | weights | training puzzles | weight decay | training / unseen |
|---|---|---|---|---|
| column roles, wide | 1,390k | 100,000 | 0.1, decoupled, per puzzle | 51.6 / 52.4 at 40,000 puzzles, 38.7 / 38.8 at 120,000; cancelled |
| attribute GNN B, 16×, no dropout | 25.6M | 100,000 | 0.1, constant learning rate | 99.2 / 99.3 (epoch 45) |
| attribute GNN B, 16×, grokking papers' setting | 25.6M | first 1000 / 3000 / 10000 | 1, β2 0.98 | best 51.6 / 46.3, 19.0 / 20.1, 26.5 / 27.1; loss rising by 12,000 steps, cancelled |
| attribute GNN B, 16× | 25.6M | first 1000 | 0 / 0.1 | 99.8 / **98.3**, 99.7 / 97.0 (by 16,000 steps) |
| attribute GNN B, 16× | 25.6M | first 3000 | 0.1 | 99.1 / **99.0** (12,000 steps) |
| attribute GNN B, 16× | 25.6M | first 100 | 0 / 0.1 | 100 / 6.4, 100 / 10.4 (300,000 steps) |
| attribute GNN B, 16× | 25.6M | first 300 | 0.1 | 100 / 61.7 at 16,000 steps, then down to 24–30 by 300,000 |
| attribute GNN B, 1× | 1.6M | first 1000 | 0 | 100 / **98.0** (epoch 1750 of 2500) |
| row fold | 305k | first 1000 | 0 | best 90.7 / **63.4** (epoch 110), then 98 / 48 by 400,000 updates |
| row fold, initial weights ×1 / ×3 | 305k | first 1000 | 1, decoupled, every 100 puzzles | best 25.7 / 24.6, 23.9 / 24.5: training never fits |
| row fold | 305k | first 300 | 1 | best 31.7 / 29.0: training never fits |
| row fold | 305k | first 1000 | 0.03 / 0.01 | best 86.1 / **65.7** (epoch 125), 81.1 / 58.7 (epoch 95); over the last 100 epochs 86 / 43 and 97 / 46 |

No run grokked. The runs fall into three groups. With 1000 or more training puzzles and the recipe of the
100,000-puzzle run, GNN B learns the sum, with training and unseen accuracy rising together, so there is no memorised
stage for a late generalisation to follow. In the grokking papers' own setting its training loss rose instead, at every
size tried, and nothing was memorised either. With 100 or 300 puzzles GNN B memorises, and the unseen accuracy never
comes back in 300,000 steps; at 300 it even reaches 61.7 and then loses it. The fold memorises without decay and with
gentle decay: the unseen accuracy peaks near epoch 100 and then sinks, over 400,000 to 600,000 updates. With decay 1
the fold never fits the training puzzles at all. The decay is counted per update, and an update here is one puzzle, so
an epoch of 1000 puzzles brings a thousand decay steps, where the grokking papers train on full or large batches with
far fewer steps per epoch. The grid of decay values missed the window between memorising and not fitting, and the second
round with gentler decay did not find it either (FINDINGS 35).

**What it changed.** One thing carried over to the benchmark. GNN B at 1×, trained on the ordinary mixed-rule 3×10
puzzles with the recipe that learned the sum here (no dropout, no decay, a constant learning rate, 40,000 steps, two
seeds), gets Arithmetic exact 45.1 and 54.0 at the epoch picked on selection, against 22.7 with the regularized setting
of D.1 (RESULTS §5), and 97.9 (45.5) and 98.2 (56.5) on whole puzzles. The 22.7 therefore came from dropout and decay, not from the
mixing of rules, and this is the best Arithmetic of any model here that has to learn the sum on 3×10 (column roles:
40.5). The regularized setting was chosen on I-RAVEN and holds the torch baselines back on 3×10. The other baselines
were not rerun without it, and the facet models trained for 60–150 epochs against these 425, so the 3×10 comparisons
between the facet family and the torch baselines are not like for like (FINDINGS 37, RESULTS §12).

---

## Appendix F. Reproduction

### F.1 Environment

- **PyNeuraLogic models**: conda environment `pynl` with Python 3.11 and PyNeuraLogic 0.9.2, on CPU nodes only, with the
  Java runtime pinned as in Appendix A.4. Native training always passes `iso_value_compression=False` (A.1).
- **Torch baselines, the ViT recognizer and the ResNet-18 frontend**: conda environment `nvsa` with Python 3.7 and
  torch 1.11, on GPU nodes.
- **Cluster**: SLURM, CPU partitions `amd`, `amdlong` and `amdfast`, GPU partitions `amdgpu` and `amdgpufast`. Every job
  file in `jobs/` names its partition, memory and time, and the usage at its top.
- Every script runs from the repository root, for example
  `python3 core/pynl/bench_hub_grounding.py --template facet ...`, so the `data/` and `experiments/` paths resolve.

### F.2 Where each main result comes from

| result | script | job file |
|---|---|---|
| Facet model on I-RAVEN, 97.3 (69.1), Chapter 5 | `core/pynl/bench_hub_grounding.py --template facet --constellation all --widths 32,64,128 --lr 3e-4 --n 6000 --n-sel 300 --n-val 1000 --n-train-eval 300 --train-epochs 80 --eval-every 5 --save-dir experiments/facet_all_w32-64-128_hidden --jvm-gb 300` | submitted with `sbatch --wrap` (job 11882029); 150 epochs and the fold on I-RAVEN: `jobs/pynl/submit_facet_long.sh` |
| Facet model end to end and its one test pass, 97.19 and 97.03 | `core/pynl/facet_e2e_eval.py`; the recognizer `core/frontend/train_vit_recognizer.py` with data from `data_prep/prepare_recognizer_data.py` | `jobs/frontend/submit_recognizer.sh`, then `jobs/pynl/submit_facet_e2e.sh` (job 12028100; `SPLIT=test` once per final checkpoint, job 12031767) |
| Facet model on I-RAVEN-X, copy links and content gates | `bench_hub_grounding.py --iravenx FILE`, `--copy-links`, `--copy-gates content`; per-rule scores `core/pynl/facet_rule_eval.py` | `sbatch --wrap` as above with `--iravenx`; `jobs/pynl/submit_fill_table.sh`, `jobs/pynl/submit_facet_long.sh`; rescoring after the scorer fix `jobs/pynl/submit_rescore_exact.sh` |
| Column roles, row fold, their seeds and mixed row lengths (Chapter 6, Appendix E.1) | `bench_hub_grounding.py --columns roles` or `--columns fold`, `--seed` | `jobs/pynl/submit_column_roles.sh` (`RUN=plain`, `content`, `wide`, `fold`, `fold_content`, `named`, `fold_mixed`, `roles_mixed`; `SEED`) |
| Symmetry ablations, their seeds and the data factor (Sections 6.2–6.3, D.4) | `bench_hub_grounding.py --label-components`, `--label-attributes`, `--per-comp-weights`, `--per-attr-weights`; summary `analysis/seeds_summary.py` | `jobs/pynl/submit_symmetry_ablation.sh` (`RUN`, `SEED`) |
| Learning curves, the unseen layout and the held-out rule on an attribute (Sections 5.4, 6.4) | `bench_hub_grounding.py`; `core/baselines/train_baseline.py` | `jobs/pynl/submit_facet_curves.sh`, `jobs/baselines/submit_curves.sh` |
| Wrong symmetries (Section 6.7) | `bench_hub_grounding.py --columns tie12`, `--fold-values`; checks by `facet_rule_eval.py` | `jobs/pynl/submit_wrong_symmetry.sh` |
| Transformed and transposed puzzles (D.3) | `facet_rule_eval.py` (its transforms, `transpose`); `core/pynl/rpm_reference_solvers.py --transpose`; the baselines' `--eval-only` | `jobs/pynl/submit_transpose.sh` (job 12074421) |
| Symmetries of the model and of the data (Chapter 3, Appendix B.2) | `core/pynl/facet_symmetry_check.py`, `analysis/dataset_symmetry_check.py` | no job file; run from the repository root |
| Torch baselines (D.1) | `data_prep/prepare_baseline_data.py`, `data_prep/prepare_baseline_images.py`, `data_prep/render_iravenx.py`; `core/baselines/train_baseline.py`; table `analysis/summarize_baselines.py` | `jobs/baselines/submit_baselines.sh`, `submit_sweep.sh`, `submit_regularized.sh`, `submit_final.sh`, `submit_attr_gnn.sh`, `submit_vit.sh`, `submit_iravenx_vit.sh` |
| The test pass of Chapters 5 to 7 (D.6) | `facet_rule_eval.py --test`, `value_axis.py --eval --eval-start 8000`; summary `analysis/test_pass_summary.py` | `jobs/pynl/submit_test_pass.sh` (lists in `experiments/test_pass/`) |
| The learning curve end to end (Section 5.4) | `train_vit_recognizer.py --per-layout N`; `facet_e2e_eval.py` | `jobs/frontend/submit_recognizer.sh` (`PER_LAYOUT`), then `jobs/pynl/submit_facet_e2e.sh` (`CKPT`, `PREDS`) |
| The baselines' one test pass (D.2) | `core/baselines/test_eval.py`; test arrays from `prepare_baseline_data.py --test` | `jobs/baselines/submit_test_once.sh` (job 12032179) |
| Symbolic models and reference solvers | `core/pynl/pynl_symbolic.py`, `rpm_symbolic.py`, `rpm_reference_solvers.py`, `pynl_symbolic_e2e.py` | `jobs/pynl/submit_symbolic.sh`, `submit_symbolic_e2e.sh`, `submit_symbolic_extra.sh` |
| ResNet-18 perception for the symbolic models end to end | `core/frontend/train_frontend.py`, `core/pynl/pixel_pipeline.py` | `jobs/pynl/submit_pixels.sh` |
| Value axis (Section 4.5, D.1, D.4) | `core/pynl/value_axis.py`, `core/pynl/value_axis_iraven.py` | `jobs/pynl/submit_value_axis.sh`, `jobs/pynl/submit_value_axis_iraven.sh` |
| Rule twins (Chapter 7, D.5) | `data_prep/rule_twins.py`; `value_axis.py` with `--ring --kernel --pair`, `--scale`, `--gate`, `--steps`, `--seed`, `--select exact`; `analysis/twins_seeds.py`, `twins_probe.py`, `rule_orbits.py`, `twins_linear_fit.py`; the counts behind the proofs `analysis/pricing_checks.py` | `jobs/pynl/submit_rule_twins.sh`, `jobs/pynl/submit_twins_dump.sh`, `jobs/baselines/submit_twins.sh` |
| Grokking (Appendix E.2) | `data_prep/arith_big_data.py`, `core/pynl/facet_stream.py` | `jobs/pynl/submit_arith_big.sh`, `jobs/pynl/submit_grok_fold.sh`, `jobs/baselines/submit_arith_big.sh`, `jobs/baselines/submit_grok_small.sh` |
| Plus and minus apart (RESULTS §13) | `data_prep/arith_op_data.py` | `jobs/pynl/submit_arith_op.sh` |
| Checks on the data (Appendix B.4) | `analysis/sanity_checks.py` | no job file |
| Notebooks and their HTML copies | `notebooks/*.ipynb` | `jobs/pynl/submit_notebook.sh` |

`docs/CODE_INDEX.md` describes every file in more detail, and `docs/RESULTS.md` gives the job number behind each cell.

### F.3 What a rerun reproduces

- **Seeds.** The early PyNeuraLogic runs were not seeded, and for the single runs of the ablation table (D.4) the seed was
  not recorded. A rerun of such a run gives a number within the spread of its seeds, not the same digits. Runs made
  since `--seed` was added, and the torch baselines, carry a recorded seed.
- **Checkpoints.** A facet checkpoint is `template_best.bin` with `best.json`; the template is rebuilt from the arguments
  stored in `best.json`, and `facet_rule_eval.py` rescores any I-RAVEN-X checkpoint that way. A torch checkpoint is
  `model_best.pt` with `best.json`.
- **The test split.** The I-RAVEN checkpoints of D.2 and every final checkpoint of Chapters 5 to 7 (D.6) have had their
  one test pass, and `test_eval.py`, `facet_rule_eval.py --test` and `value_axis.py` skip or refuse any run that
  already has a `test.json`. The I-RAVEN models other than these, the copy-link variants and the row fold on I-RAVEN,
  and the CNNs were never scored on test. Someone reproducing the work should train a new checkpoint, pick its epoch
  on the selection puzzles, and score test once.

### F.4 The attached medium

The medium holds the code and the written record, not the data or the experiments, which are far too large:
- `core/`: the models, their training and their scoring (`core/pynl/` for PyNeuraLogic, `core/baselines/` for torch,
  `core/frontend/` for perception, `core/gnn/` for the I-RAVEN data package with the slot fix), together with IBM's
  `raven/` and `torch_gnn/gnn_solver.py`, which `core/` imports;
- `analysis/`, `data_prep/` and `jobs/`: the checks, the data preparation and every SLURM job file named in F.2;
- `notebooks/` and their executed HTML copies in `docs/`;
- `docs/`: the result tables, the findings, the lab notes, the code index and the thesis drafts;
- `llm/`, the LLM benchmark on I-RAVEN, and `archive/`, the superseded code, kept for reference and not runnable as is.

Not on the medium, and how to rebuild it:
- `data/`. I-RAVEN is drawn by the generator of the I-RAVEN paper, and I-RAVEN-X by IBM's `iravenx_task.py`; the files
  with rows of 5 and 20 were drawn with seed 1234 (E.1). The derived sets come from the scripts in `data_prep/`: the
  rule twins, the Arithmetic-only puzzles, the one-operator puzzles, the I-RAVEN-X images and the arrays for the
  baselines and the recognizer.
- `experiments/` and `log/`: checkpoints, predictions and job logs. Each is rebuilt by the job in F.2.
