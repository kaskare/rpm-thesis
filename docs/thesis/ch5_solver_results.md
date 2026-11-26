# Chapter 5. The solver: accuracy, end to end, and against the baselines (draft, 2026-11-14)

Cells are matched % with exact % in brackets.

## 5.1 I-RAVEN, test split

Every model below was scored once on the 14,000 test puzzles, after its checkpoint and all settings were fixed
(Section 4.2). Before its test pass each baseline checkpoint was scored again on the unseen validation puzzles and
had to reproduce its recorded numbers, which all did exactly.

| model | weights | given | test, matched (exact) |
|---|---|---|---|
| symbolic template, rules given | 41 | labels, rules, candidates | 98.3 |
| **facet model** | 267k | labels | **97.2** (68.4) |
| **facet model, end to end** | 267k + 326k | images | **97.0** (68.1) |
| attribute GNN B, the facet model's symmetries | 267k | labels | 96.8 (67.8) |
| attribute GNN A, absolute labels | 266k | labels | 95.1 (62.5) |
| object GNN, 16 times the weights | 4,258k | labels | 93.4 (63.7) |
| object GNN, 2 seeds | 265k | labels | 90.8 ± 1.2 (55.1) |
| MLP, 2 seeds | 264k | labels | 72.7 ± 1.2 (27.2) |
| ViT, width 224 | 3,753k | images of the context | 93.1 (64.4) |
| ViT, width 64 | 335k | images of the context | 78.7 (37.4) |
| ViT, rows 1 and 2 built in, width 224 / 64 | 3,709k / 323k | images of the context | 92.3 (62.4) / 89.8 (54.5) |

The object GNN and the MLP are the regularised versions; without regularisation they reach 87.8 ± 1.1 and 60.8 ± 0.1.

Four things can be read from the table.

**The model generates the answers of I-RAVEN without being given the rules or seeing the candidates.** It answers
97.2% of test puzzles, 1.1 points below the template in which the rules are written out.

**Test and validation agree.** The facet model scored 97.3 on the unseen validation puzzles and 97.2 on test; end to
end, 97.1 and 97.0. No model in the table moves by more than 0.7 points between the two splits, and the order of the
models is the same. The validation puzzles were looked at for weeks, and this did not inflate the numbers.

**The order follows the structure given to the model.** A flat vector of all panels gives 73%. Objects as nodes give
91%. One node per attribute of each object gives 95%. The symmetries of rows, components and attributes on top of
that give 97%. Sixteen times the weights do not make up one step: the large object GNN stays below the small attribute
GNN.

**Against black-box vision.** The vision transformer that reads the context images reaches 93.1% with 3.75 million
weights. The facet model behind its recognizer reaches 97.0% with 592 thousand in total, 3.9 points more with a sixth
of the weights. At a comparable size, the transformer with 335k weights reaches 78.7%. The two pipelines are not
given the same supervision: the recognizer is trained on the annotated objects of every panel, while the transformer
gets the annotation of the answer only. What the comparison shows is the value of that intermediate representation
together with the structure built on it, and the next sections separate the two.

**Against published solvers.** Our transformer is a plain one, not an architecture built for these puzzles.
Małkiński and Mańdziuk (2025a) retrained thirteen such architectures under one protocol, on the same 6,000 training
puzzles per layout, each with an auxiliary loss on the puzzle's rules. Three of the strongest, as their authors report
them and as rerun:

| I-RAVEN test, 6,000 training puzzles per layout | reported | rerun under one protocol |
|---|---|---|
| SCL | 95.0 | 83.4 |
| PredRNet | 96.5 | 88.8 |
| DRNet | 97.6 | 90.9 |
| our vision transformer, width 224 | | 93.1 |
| facet model, end to end | | 97.0 |

These models pick one of the eight candidates. Ours generate the answer, and matched accuracy counts it right when the
closest candidate is the answer, so the two scores are close but are not the same task. The rerun models also learn
from the labels of the rules, which ours never see. Within those limits our transformer is a fair stand-in for black-box
vision at this amount of data, not a weak one.

## 5.2 By layout

Matched % on test.

| model | centre | 2×2 | 3×3 | left-right | up-down | in-out | in-out grid |
|---|---|---|---|---|---|---|---|
| symbolic template, rules given | 99.9 | 96.0 | 95.5 | 100.0 | 100.0 | 99.9 | 97.3 |
| facet model | 99.6 | 95.0 | 89.1 | 99.8 | 99.7 | 99.8 | 97.5 |
| facet model, end to end | 99.6 | 95.0 | 89.1 | 99.8 | 99.7 | 99.8 | 96.4 |
| attribute GNN B | 99.8 | 95.2 | 85.5 | 100.0 | 99.9 | 99.7 | 97.5 |
| attribute GNN A | 99.9 | 89.5 | 84.4 | 100.0 | 99.9 | 100.0 | 92.0 |
| object GNN, 16× | 99.8 | 88.2 | 74.5 | 99.8 | 99.8 | 99.8 | 92.2 |
| MLP, seed 0 | 94.4 | 46.4 | 38.8 | 91.2 | 87.1 | 95.3 | 61.6 |
| ViT, width 224 | 99.1 | 92.8 | 80.5 | 99.8 | 99.6 | 99.8 | 80.0 |

Every model except the MLP solves the four layouts whose components hold a single object. The differences are in the
grids. The 3×3 grid is the hardest layout for every learned model, the facet model included: it loses 6.4 points
there to the rules-given template, and less than one point on any other layout. The vision transformer loses most on
the 3×3 grid and the in-out grid. Training it on the grid layouts alone, with finer patches or larger images, moved
no grid layout by more than 8 points; it memorised (training loss near zero, selection accuracy 35–42%). The same grid puzzles were learned once
the single-object layouts were added to the training set, so at this amount of data the transformer needs the easy
layouts to find the task at all.

## 5.3 End to end

Replacing the annotation by the recognizer costs 0.16 points on test (97.19 to 97.03), and 0.18 on the validation
puzzles. All of it falls on the in-out grid, the one layout the recognizer misreads; the other six layouts are
unchanged to the puzzle.

| test puzzles | share | matched |
|---|---|---|
| all 16 panels read right | 95.6% | 97.2 |
| at least one panel misread | 4.4% (611 puzzles) | 92.6 |

A misread panel is usually not fatal. The model still matches 92.6% of the puzzles in which the recognizer made an
error, because one wrong value among eight panels moves the generated description a little, and the right candidate
often stays the closest.

For comparison, the rules-given template loses 2.0 points behind an earlier recognizer, a ResNet-18 that reads all
16 panels right in 92.6% of puzzles.

One limit was found here. The facet template has no bias terms and normalises its facet vectors, and value 0 is the
code with no bit on. A code of 10⁻⁶ in place of 0 is therefore rescaled to full size and read as a large value.
The model cannot take graded inputs as it stands, which is why the recognizer's uncertainty is not passed on (Section
4.3). Training the two parts jointly would need this changed first.

## 5.4 By amount of training data

The models were retrained on the first 300, 1,000 and 3,000 puzzles of every layout, with the number of updates kept
the same. These runs were scored on the unseen validation puzzles only, with one seed. The relational models read the
annotation. The facet model was also run end to end: the recognizer of Section 4.3 was trained from scratch on the same
first puzzles of every layout, and the facet checkpoint of that size read its panels (at 6,000 puzzles it is the
recognizer of Section 5.3, trained on 2,000).

| training puzzles per layout | facet | facet, end to end | GNN B | GNN A | object GNN | MLP | ViT 224 | ViT 224, rows built in |
|---|---|---|---|---|---|---|---|---|
| 300 | 90.0 (53.3) | 89.8 (53.1) | 87.0 (53.0) | 86.1 (50.8) | 29.6 (1.1) | 24.9 (0.5) | 15.6 (0.2) | 14.6 (0.1) |
| 1,000 | 94.1 (62.7) | 93.9 (62.4) | 93.9 (64.0) | 93.4 (61.5) | 68.7 (21.9) | 48.1 (7.7) | 18.6 (0.2) | 23.8 (0.5) |
| 3,000 | 96.1 (66.0) | 95.8 (65.7) | 96.8 (68.8) | 95.9 (65.3) | 86.5 (49.8) | 64.5 (19.1) | 45.8 (4.5) | 62.0 (13.3) |
| 6,000 | 97.3 (69.1) | 97.1 (68.8) | 96.8 (67.9) | 95.0 (62.5) | 90.5 (55.1) | 72.0 (27.5) | 93.4 (64.4) | 92.1 (62.1) |

**Reading the panels costs little at any size.** End to end the facet model loses 0.2 to 0.3 points at every size,
almost all of it on the in-out grid, the layout the recognizer misreads most. Trained on 300 puzzles per layout the
recognizer reads all 16 panels of a puzzle right in 90% of puzzles, and on 3,000 in 96%. Where it misreads a panel the
facet model still answers 78% of those puzzles at 300 and 89% at 3,000.

**The gap is largest when data is scarce.** From 300 puzzles per layout the facet model answers 90% end to end and
the vision transformer 16%, barely above the 12.5% of guessing. The transformer needs twenty times the puzzles to pass
the facet model's score at 300. This is the comparison the assignment asks for: 4 points at 6,000 puzzles, 74 at 300.

**The labels help only once data is plentiful.** The recognizer is trained on the labelled objects of all 16 panels
of every training puzzle, the transformer on the answer's description alone, so the end-to-end curve removes the
annotation at test time but not its help in training. To separate the two, a black-box reasoner was given the same
perception: the recognizer of each size reads the context panels, and a plain transformer over its slot states
generates the answer (Section 4.6). It has the facet model's labels and none of its structure.

| training puzzles per layout | facet, end to end | recognizer + transformer 64 | recognizer + transformer 224 | the same, recognizer fine-tuned, 64 |
|---|---|---|---|---|
| 300 | 89.8 (53.1) | 18.9 (0.4) | 17.6 (0.3) | 20.4 (0.5) |
| 1,000 | 93.9 (62.4) | 35.9 (1.8) | 32.5 (2.0) | 76.3 (33.4) |
| 3,000 | 95.8 (65.7) | 87.6 (56.0) | 88.8 (60.6) | 90.0 (59.8) |
| 6,000 | 97.1 (68.8) | 91.4 (59.1) | 95.1 (69.9) | – |

With 300 or 1,000 puzzles per layout the labels do not rescue the black box. It memorises the training puzzles and
stays near guessing, as the vision transformer does, 71 and 58 points below the facet model reading the same
recognizer. Training the recognizer along with it lifts 1,000 puzzles to 76%, and does nothing at 300. From 3,000
puzzles the labels carry most of the vision transformer's gap: the reasoner is 6 to 8 points below the facet model at
3,000, and at 6,000, with 14 times the facet model's weights, 2.0 points below in matched and level in exact. What it
still misses is on the 2×2 and 3×3 grids, the layouts with the most objects. It needs about ten times the puzzles to
reach the 90% the facet model gets from 300. Of the 74 points at 300 puzzles, about 4 come with the labels and 71 with
the structure. These runs have one seed each.

**Most of that gap is the representation.** At 300 puzzles the three models on the attribute graph have 86 to 90%
and the object GNN has 30%, with the same labels as input. The step from a node per object to a node per attribute is
worth more than 56 points there. The symmetries are worth 1 to 4 points among the three attribute models. On I-RAVEN, then, the
graph carries the result and the symmetries refine it. Chapter 6 shows where symmetries carry more: on values, and on
what the training data never showed.

## 5.5 The symmetries carry the result, not the framework

The attribute GNN B is written in PyTorch and has the facet model's symmetries. It reaches 96.8% on test against the
facet model's 97.2%, and stays within 3 points of it at every amount of data, ahead at 3,000. The 0.4 points on test
are one run each; two seeds of the object GNN differ by 1.7. We therefore do not claim that the lifted framework makes
a model more accurate. By layout, the facet model's lead is the 3×3 grid (89.1 against 85.5), and the two are within
0.2 points everywhere else.

What the template contributes is of another kind. The symmetries of B had to be engineered into its node labels and
checked one by one. In the template they follow from which constants are left unnamed, one proof covers all of them,
and removing one is a change of a line (Chapter 6).

## 5.6 Where the facet model fails

**Not on a particular rule.** It matches 97.3% of the validation puzzles that contain an Arithmetic rule and 97.3% of
those that do not. Generating type, size and colour exactly, by the rule the attribute follows:

| | Constant | Progression | Distribute Three | Arithmetic |
|---|---|---|---|---|
| facet model | 83.7 | 89.7 | 88.7 | 83.8 |
| ViT, width 224 | 82.4 | 89.7 | 86.6 | 86.3 |

Constant is the lowest because of the grids, where panels that are not uniform hold noise values (Section 4.1).

**On number and position.** When the number of objects in a grid changes along the row, the model generates the right
count in 44 to 49% of cases and the right positions in 51 to 71%. This is why exact is 68% where matched is 97%: the
candidates rarely differ in these, so matching does not need them. It is also the place where the rules-given
template gains its 6 points on the 3×3 grid. Position rules are set operations and cyclic shifts over slots, and the
facet model's position channel has none of that structure.

## 5.7 I-RAVEN-X

On I-RAVEN-X with rows of 3 and values 0–10 the plain facet model reaches 99.2 (97.0), with 95.0% of Arithmetic values
exact.

With rows of 10 the picture changes, and we report it without a ranking. The plain model reaches 87.4 ± 1.8% matched at
values 0–50 over three seeds and generates 18.4 ± 3.0% of answers exactly; Arithmetic values are exact in 20.7 ± 2.2% of
cases. Variants that build in more raise this: with column roles, 94.2 ± 0.2 (27.6 ± 2.0) and 39.6 ± 1.3% on
Arithmetic, with a fifth of the weights (36.0 ± 2.1% on Arithmetic on the test range, scored once at the end; Appendix
D.6). A torch
attribute GNN B trained for seven times as many epochs without regularisation reaches 97.9 (45.5) and 98.2 (56.5) in
two seeds, level with the best facet variant. These runs differ in training length and recipe, most have one seed,
and the baselines' recipe was tuned on I-RAVEN, so they do not support an order of models, and we give none. What they
do support: no model that has to learn sums generates more than about half of the Arithmetic values exactly on rows of
10, and matching hides this. Rows of 10 are used in the rest of the thesis for experiments inside one model family,
where only one thing changes.

## 5.8 Cost

The lifted approach has a real cost. PyNeuraLogic grounds every puzzle into its own network and keeps all of them in
memory, and it updates the weights one puzzle at a time, on CPU.

| run | peak memory | time |
|---|---|---|
| facet model with content gates, I-RAVEN, all layouts | 327 GB | 15 h |
| facet model with copy links, rows of 10, values 0–1000 | 344 GB | 16 h |
| value axis with `plus` in learned rules, rows of 10 | 104–161 GB | 3–7 h |

Grounding takes 10 to 165 ms per puzzle and is done once; an epoch over all of I-RAVEN takes about 14 minutes. The
torch baselines of the same size train on one GPU in a few gigabytes. Scoring is cheap by comparison: the whole test
split, with labels and end to end, took 15 minutes and 16 GB.
