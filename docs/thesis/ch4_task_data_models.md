# Chapter 4. Task, data and models (draft, 2026-11-14)

Chapter 3 (theory) is not drafted. This chapter uses two things from it: a template cannot tell apart constants it
does not name, and a symmetry is a data symmetry, out of support, or conflicting, by what it does to valid puzzles.
Numbers are from `docs/RESULTS.md`. Notes for you are at the end.

---

## 4.1 The task: generate the answer, then match it

Most solvers of Raven's matrices score the eight candidates and are trained to give the right one the highest score.
All models in this thesis do something else. They **generate a description of the missing panel** from the eight
context panels, and only afterwards is the description compared with the candidates. The candidates are never an input
and never part of the loss.

**The description.** A panel is described per component. For a component, the description holds, for each of type,
size and colour, the mean code of its objects' values; the number of its objects; and which position slots are
occupied. Values are written in a thermometer code: value v turns on the first v bits, and value 0 turns on none. On
I-RAVEN the codes have 10 bits and there are 22 position slots over all layouts, so a component's description is 62
numbers between 0 and 1. On I-RAVEN-X a code has one bit per value step, 10 to 1,000 bits.

**Training.** A sample asks for one component of the missing panel. The loss is the squared error between the model's
62 outputs and the description of the true answer. Layouts with two components give two samples per puzzle.

**Two measures.**
- *Matched*: the candidate whose description is closest to the generated one, summed over components, is the right
  answer. Ties count as wrong, and guessing gives 12.5%. This is the usual accuracy of the benchmark. The true
  description itself picks the right candidate in 99.1% of I-RAVEN puzzles and in all I-RAVEN-X puzzles.
- *Exact*: the generated panel, decoded, equals the answer: every attribute's values, the number of objects and their
  positions. Exact does not use the candidates at all.

Exact is much the stricter measure, for two reasons. A model can match without knowing a value precisely, because the
wrong candidates are often far away: on I-RAVEN-X with rows of 10, one of our models matches 85% and generates 15%
exactly. And in the grid layouts of I-RAVEN the generator gives an attribute a rule only where the objects of a panel
are uniform in it; elsewhere the values are noise that no model can predict, and exact counts them. We report matched
with exact in brackets, as in 97.2 (68.4).

Generating has one consequence for the design of every model here: a shortcut through the candidate set, the flaw of
RAVEN, is impossible by construction, on any dataset.

## 4.2 Data, splits and protocol

| dataset | puzzles | training | picks the epoch | reported during development | test |
|---|---|---|---|---|---|
| I-RAVEN, seven layouts | 10,000 per layout | train 0–5999 per layout | validation 0–299 | validation 300–1299 (7,000 puzzles) | test 0–1999 (14,000 puzzles) |
| I-RAVEN-X, per row length and range | 10,000 | 0–5999 | 6000–6299 | 6300–7299 (1,000 puzzles) | 8000–9999, not used |

**Protocol.** A model is trained on the training puzzles, or on the first n of them per layout for the learning
curves. Its checkpoint is the epoch with the best matched score on the selection puzzles. Its score is reported on
puzzles that played no part in that choice. For I-RAVEN these are 7,000 validation puzzles, which we call *unseen*.
They were looked at for the whole of the work, so the final I-RAVEN numbers of Chapter 5 come from the test split,
which every final checkpoint was scored on exactly once, after all choices were fixed. For I-RAVEN-X the unseen
puzzles are the reported ones and the test range was not used.

**Seeds.** Most results are a single run. Where two seeds exist the text gives the mean and the standard deviation;
for the object-level baselines the two seeds differ by 1 to 2 points, which is the scale against which single-run
differences should be read.

**Checks on the data.** Three checks were made before any model was trained on it. I-RAVEN's annotation lists only
the occupied slots of a grid, and reading a slot from an object's index in that list, as the code we inherited did,
left about a third of the grid puzzles unanswerable; slots are now derived from the bounding boxes. No puzzle occurs in
two splits. And a model that sees only the eight candidates is at chance on I-RAVEN, where it scores 90–100% on RAVEN.

**The symmetries the data has.** Chapter 6 depends on knowing which transformations of a puzzle the generators
respect. We measured this on the data and read it off the generators' code. The table gives the share of rule
instances that still satisfy a rule of the same kind after the transformation, as Constant / Progression / Arithmetic
/ Distribute.

| transformation | I-RAVEN | I-RAVEN-X, rows of 3 | I-RAVEN-X, rows of 10 |
|---|---|---|---|
| rows 1 and 2 exchanged | 100 / 100 / 100 / 100 | 100 / 100 / 100 / 100 | 100 / 100 / 100 / **0** |
| columns 1 and 2 exchanged | 100 / **0** / 51 / 100 | 100 / **0** / 55 / 100 | 100 / **0** / 50 / **0** |
| values reflected, v → max − v | 100 / 100 / **0** / 100 | 100 / 100 / **0** / 100 | 100 / 100 / **0** / 100 |
| matrix transposed | 58 / 3 / 1 / 100 | – | – |

Three readings are used later.
- *Rows 1 and 2* can be exchanged in I-RAVEN and in I-RAVEN-X with rows of 3: the generator produces the exchanged
  puzzle as often as the original (with one small exception, a constraint on Arithmetic over colours). With rows of 10 it does not, because a Distribute row is there the first row
  shifted by one step, and after the exchange no step fits. The answer is still right, though: it is the value the
  third row lacks. So the exchange is a symmetry of the task and not of the data, and a model without it can learn a
  rule that holds only on what the generator produces (Section 6.4).
- *Columns 1 and 2* cannot be exchanged. A Progression row stops being a Progression row. The exchanged puzzle is not
  wrong, it simply never occurs: the transformation leads out of the data.
- *Reflecting the values* keeps Constant, Progression and Distribute rows valid, and equally likely, with the
  reflected answer. A model that cannot see the difference between v and max − v would have to give a puzzle and its
  reflection the same answer, and they need different ones.

Further symmetries of the generators: the two components of the left-right and up-down layouts follow identical
distributions; in I-RAVEN-X, size and colour are drawn by the same function; and the order of objects in a panel and
of the candidates means nothing.

## 4.3 Object recognition

The relational models take the objects of each panel as input: for each object its slot, type, size and colour.
During development these come from the dataset's annotation ("with XML labels"). For the end-to-end results they come
from a recognizer that reads the image of one panel.

**The recognizer** is a vision transformer with the patch embedding and the blocks of our black-box baseline (Section
4.6): depth 6, width 64, patches of 16 pixels, 325,718 weights. In place of one answer token it has one learned query
token per position slot, 22 in all, and one head shared by all slots, which says whether the slot holds an object and
gives a distribution over its type, size and colour. It reads panels one at a time and knows nothing of rows or
puzzles. It was trained on the panels of 2,000 puzzles per layout, with the epoch picked on the selection puzzles.

| | panels read entirely right | puzzles with all 16 panels right |
|---|---|---|
| unseen validation puzzles | 99.35% | 95.69% |
| test | 99.40% | 95.64% |

By layout it reads every panel right in the centre, 2×2, left-right and up-down layouts, 99.92% of 3×3 panels, 99.99%
of in-out panels and 95.56% of in-out grid panels. The small inner grid is its weak spot.

**From recognizer to model.** A slot becomes an object when its probability of being occupied exceeds one half, and
each attribute takes its most probable value. The candidates are read by the recognizer as well, so nothing in the
end-to-end pipeline comes from the annotation at test time. We also tried passing the recognizer's uncertainty on, by
setting each thermometer bit to the probability that the value exceeds it. On the unseen puzzles this changed nothing
(97.13% both ways), so the simpler mapping is used.

The recognizer is trained on annotated objects, apart from the reasoning model. The pipeline is end to end at test
time, not in training.

## 4.4 The facet model

The main model is a template in PyNeuraLogic. For each puzzle it is grounded on that puzzle's facts into a network of
its own.

**Facts.** A puzzle is described by its panels `p0 … p7`, the row and column of each, the objects of each panel, the
component each object belongs to, and for each object and each of type, size and colour a *facet*: a node that
carries the thermometer code of that value. Facts also say which objects of neighbouring panels sit in the same
position. Rule names, arithmetic facts and candidates are not among the facts.

**Rules.** With P a panel, C a component, A an attribute, O an object and Rw a row:

```
f(O, A)       from the object's own value, its panel's summary, and the facets of the objects in the
              same position in the other columns of its row (two layers)
vF(P, C, A)   the component's value in a panel: the mean of its objects' codes and of their facets
rel(Rw, C, A) a complete row:  vF of its three panels, each through the weights of its column
ans(C, A)     the question row's two panels, each through its column's weights, and the mean of
              rel over the two complete rows
```

The number of objects is treated as a fourth attribute. Positions have a channel of their own with the same row
structure, on the occupancy of the component. The output for the queried component is the concatenation of `ans` over
the attributes and the position channel. The layers have widths 32, 64 and 128, and the template has 266,528 weights.

**Training.** Adam with learning rate 3·10⁻⁴, one puzzle per update, 80 epochs on the 42,000 training puzzles of all
seven layouts together. The checkpoint is epoch 55.

**What the template does not name.** Every rule above has the attribute, the component, the object and the complete
row as variables. By the theorem of Chapter 3, the model therefore cannot distinguish:

| not named | so the model is unchanged by | random weights, largest change of the output |
|---|---|---|
| objects | reordering the objects of a panel | 8·10⁻¹⁶ |
| the two complete rows | exchanging rows 1 and 2 | 3·10⁻¹⁶ |
| components | exchanging the two components (attribute outputs) | 0 |
| attributes | exchanging type and colour, with the outputs | 0 |
| candidates | reordering the candidates | 0 (they are not an input) |

It does name the columns, so exchanging two columns changes the output (by 0.5 to 1.0 with random weights), as it
should: Progression has a direction. It names no position slot either, but the occupancy is a vector with one entry
per slot and its weights are dense, so mirror symmetries of the grids are learned and not built in.

**What is built in, stated as priors.** Objects are units, and a panel is a set of them. Attributes are independent
and behave alike. Values have an order, through the thermometer code. Columns have an order. The same relation holds
in every row. The objects in the same position of neighbouring panels correspond.

## 4.5 Variants used in the experiments

The experiments of Chapters 6 and 7 change one thing at a time. These are the changes.

**A symmetry removed, in two ways** (Section 6.1).
- *Labels.* A fact gives each component, or each attribute, a fixed one-hot vector that the rules read. The weights
  stay shared; the model can now tell the components apart. This adds 320 or 512 weights.
- *Separate weights.* Every rule is written once per component, or attribute, with its own weights. To keep the
  comparison fair this is run twice: at the same widths, which doubles or triples the weights, and at narrower widths
  that give the same number of weights as the shared model.

**Copy links** (Section 6.4). A direct connection from each context panel's value to the output, opened or closed by
a learned gate. With *per-panel gates* each of the eight panels has its own gate weights, which names the panels and
gives up the symmetry between rows 1 and 2. With *content gates* a gate is computed from the panel's own value with
weights shared over the panels of a row, which keeps it.

**Column roles** (rows of 10). The columns of a row are first, middle and last, and the eight middle columns enter
through one rule with the panel as a variable. The middle columns can then be exchanged, which is the symmetry of
Arithmetic's operands. This variant has 288k weights against 1,411k for named columns, and is the model on which the
attribute symmetry is removed in Section 6.1.

**Two wrong symmetries** (Section 6.5). *Columns tied*: columns 1 and 2 share their weights and enter symmetrically,
so the model cannot tell (a, b, c) from (b, a, c). *Values folded*: v and max − v get the same input code, so the
model cannot tell a puzzle from its reflection. Both are exact with random weights.

**The value axis** (Sections 6.2 and 7). A second template for I-RAVEN-X, built to put a symmetry on the values
themselves. In the facet model a value is a code, read through one weight per bit, so "two more than 17" and "two
more than 33" are different things to learn. In the value-axis template every value is a constant, `v0 … v50`, a
panel's attribute is a fact `x(P, A, v17)`, and background facts `off(U, V, k)` say that V lies k steps above U. A
rule

```
h(P, A, V) :- x(P, A, U), off(U, V, k)        one weight per offset k
```

is then a convolution along the line of values: the same weights at every value. No rule names a value, so the
template is equivariant to shifting all values, to about 10⁻¹⁶ with random weights, and a checkpoint trained on
values 0–50 runs unchanged on 0–100. Its ablation writes every rule once per value with separate weights (335k
weights against 10.8k), which names the values and removes the symmetry.

This template is a test bed and not a competitor of the facet model. Besides the convolution it is given features in
the shape of the rules: exact running sums along a row through a background relation `plus(U, W, V)` for U + W = V,
the number of distinct values in a row, and offsets from the last panel. With them it answers I-RAVEN-X puzzles
almost perfectly, and that says little, since the hard part was handed to it. We therefore compare the value axis
only with itself: shared weights against weights per value, and with `plus` against without. In one variant the given
sums are removed and only the `plus` facts are kept, so that learned rules have to use them (Chapter 7).

## 4.6 Baselines

Every baseline is trained on the facet model's task: the same puzzles and splits, the same target description, the
same loss, the same choice of epoch and the same two measures. The trainer refuses to run unless it reproduces the
facet pipeline's targets on a sample of puzzles. Only the architecture differs. Each is sized to the facet model's
number of weights, about 265k on I-RAVEN.

| model | input | what it shares |
|---|---|---|
| MLP | all context panels as one vector: per panel and slot, occupancy and the three codes | nothing |
| object GNN | one node per object, with its row, column, slot and component as labels; every node connected to every other | weights over objects |
| attribute GNN A | the facet model's graph: one node per object and attribute; nodes labelled with row, column, component, attribute and slot | weights over objects, as in B, but the labels let it tell everything apart |
| attribute GNN B | the same graph; nodes labelled only relatively: complete or question row, column, query | the facet model's symmetries: rows 1 and 2, components, attributes |
| ViT | the images of the eight context panels as one picture, cut into 16-pixel patches | weights over patches |
| ViT, rows built in | the same, with the two complete rows given as a set | also rows 1 and 2 |

**A and B** are the pair that separates two things the facet model changes at once: the granularity of the graph and
the symmetries. Both are written in PyTorch on the facet model's graph. B has the facet model's symmetries and A does
not, so the object GNN against A measures granularity, A against B measures the symmetries, and B against the facet
model measures what the lifted framework itself adds.

**The ViT** is the black-box vision reference the assignment asks for: depth 6, pre-norm blocks, at width 64 (335k
weights, the closest to the facet model) and width 224 (3,753k). It generates the same description as every other
model. Its target and the candidates it is matched against come from the annotation, so it is given labels for the
answer and the candidates, and images for the context.

**Training.** MLP, object GNN and the attribute GNNs: AdamW, 300 epochs, with the regularisation picked on the
selection puzzles from a grid of 40 runs of the MLP and the object GNN (weight decay 0.5, cosine schedule, gradient clipping; layer normalisation in
the MLP). The plain versions without regularisation are reported too. The ViT uses the usual recipe for a transformer
trained from scratch (learning rate 5·10⁻⁴, weight decay 0.05, cosine schedule with warm-up, dropout 0.1), not tuned
here. Larger versions of the MLP and the object GNN, up to 64 times the weights, were trained to check that size is
not what they lack.

**Symbolic references.** Two models mark the level reached when the rules are given: a template in which the four
rules are written out as logic and only 41 weights are learned, and a solver in plain Python with nothing learned.
They show what remains hard once the rules are known, and they are the only models here that see the candidates.

**What each model is given.**

| | objects | rules | candidates | symmetries built in |
|---|---|---|---|---|
| symbolic references | labels | yes | yes | – |
| facet model | labels, or the recognizer | no | no | objects, rows 1 and 2, components, attributes |
| attribute GNN B | labels | no | no | the same |
| attribute GNN A | labels | no | no | objects |
| object GNN | labels | no | no | objects |
| MLP | labels | no | no | none |
| ViT | images of the context; labels for the target | no | no | patches |

---

## Notes for you (not thesis text)

- **Section 4.2's symmetry table belongs to Chapter 3 in the outline (3.5, 3.7).** I put a short version here because
  Chapter 3 is not drafted and Chapter 6 cannot be read without it. Move it when Chapter 3 exists.
- **The rule listing in 4.4 is a paraphrase** of `build_facet_template`'s docstring. The thesis should print the real
  template, or an excerpt, in an appendix.
- **"The trainer refuses to run unless it reproduces the facet pipeline's targets"** is the `check_*` arrays of
  `data_prep/prepare_baseline_data.py`, on the first 20 puzzles of each split.
- **Not described on purpose:** the row fold, the hub and ordered templates, the CNN, the ViT with a filter per patch,
  the capacity sweeps beyond one sentence (outline §5, cut).
- **The value axis is introduced as a test bed in 4.5**, as the outline decided. Its I-RAVEN version (18.0k weights,
  99.3 matched) is left out; say if you want one sentence on it.
- **One thing to check in the code before this is final:** the recognizer's input size and whether its query tokens
  attend to the patches in every block. I described only what its docstring states.
- **The 62 numbers:** 3 × 10 code bits, 10 bits for the object count, 22 slots (`meta` of
  `experiments/baselines/data/iraven.npz`).
