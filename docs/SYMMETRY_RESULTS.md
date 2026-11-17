# Symmetry results

What each built-in symmetry buys, per model and dataset. Selected from `docs/RESULTS.md`, which stays the full
record (last updated 2026-11-08: facet exact on I-RAVEN-X rescored after a scorer fix, RESULTS notes on the main table).

Cells are matched % (exact %) on unseen puzzles, one seed unless noted. "With" and "without" mean the same model
with one symmetry built in or taken out, so only that symmetry differs.

**In short.** Built-in symmetry saves training data and weights, removes shortcuts and reaches what training never showed.
On puzzles like the training ones the gain shrinks with data: at full data it is gone on I-RAVEN and survives on 3×10, mainly
for operand order. On what training never showed it stays large at full data: a layout never seen (99.1 vs 13–24) and a rule
never seen on an attribute (87.7 vs 18–24). Most of the gain comes from weight sharing. For data symmetries, exact invariance
on top of sharing adds nothing measurable on puzzles like the training ones; it does on the unseen rule and attribute
(84.1 vs 42.3). For task symmetries it is what keeps a shortcut out.

## 1. Overview

A *data* symmetry: the generator produces the transformed puzzle as often as the original. A *task* symmetry: the
answer stays right, but the generator may never produce the transformed puzzle (notes §22). Shortcuts live in that gap.

| symmetry | kind | prior | model, dataset | what it buys | key result |
|---|---|---|---|---|---|
| components (left-right, up-down) | data | generator | facet, I-RAVEN L-R + U-D | less data | 100 puzzles: 86.6 vs 74.6 |
| attributes (size ↔ colour) | data | general | column roles, 3×10 | less data | 600 puzzles: 66.3 vs 53.0 |
| value shift | task; not for Arithmetic | general | value axis, 3×10 | less data, fewer weights, unseen values | 100 puzzles: 99.0 vs 38.3 |
| operand order (middle columns) | task | generator | column roles, 3×10 | accuracy on full data, fewer weights | 94.9 vs 85.4 at equal size |
| rows 1 ↔ 2 | data on I-RAVEN and 3×3, task on 3×10 | general | content gates, 3×10 | removes a shortcut | swapped rows: 94.9 vs ≈60.0 |
| one step per column (row fold) | task; Progression only | mixed | row fold, 3×10 | fewer weights, Progression | 305k vs 1,411k; Progression 99.2 vs 77.4 |
| components | data | generator | facet, attribute GNNs, I-RAVEN | an unseen layout | up-down never trained: 99.1–99.9 vs 13–24 |
| attributes | data | general | facet, attribute GNN B, I-RAVEN | an unseen rule on an attribute | size exact 87.7 and 84.1 vs 18–24 |
| rows 1 ↔ 2 | data on I-RAVEN | general | ViT, I-RAVEN | less data | 3000 puzzles: 70.7 vs 42.7 |
| sum's moves, one operand with the answer (`plus` used by learned rules) | task | general on small numbers (here 0–50, past them) | value axis, rule twins, rows of 3 | less data | sum, 100 puzzles: 98.2 ± 0.8 vs 4.5 ± 2.3 exact (3 seeds) |
| value scaling, v → kv + s (rows read in units of their first step, or of its power of two) | task | not classified | value axis, rule twins, rows of 3 | unseen first steps and less data, once the row descriptor reads the steps (§6) | midpoint, 100 puzzles, 3 seeds: scaled 100.0 at every seed and doubled 89.2 ± 2.7, vs 59.9 ± 4.5 with the shift alone; rows with a new first step: 101 of 101 (scaled), 64–69 of the 72 the count allows (doubled), vs 0–1 |

*Prior* (added 2026-11-09; thesis outline, Ch 7): **general** is what a person brings to any such test (objects, comparing and sorting values, small-number addition, "the rule repeats"; Chollet 2019 §II.1.3 and §III.1.2 limit `plus` to numbers below about 10); **generator** is read from how RAVEN is built (its layouts, which columns are operands). Attributes count as general because any attribute may carry any rule; the fixed attribute list itself is a generator prior. Rows 1 ↔ 2 is general (the rule repeats), but that rules run along rows only is a generator prior, held by every model here and not priced. Suggestive only: the general priors carry the largest gains, the generator priors smaller ones that mostly vanish at full data.

## 2. Less training data

The gain is largest with few puzzles and shrinks with more, once the model can learn the task at all.

| symmetry | model, dataset | puzzles | with | without | gain |
|---|---|---|---|---|---|
| components | facet, I-RAVEN L-R + U-D | 100 per layout | 86.6 (40.4) | 74.6 (19.6) | **+12.0** |
| | | 1000 | 99.3 (95.6) | 98.5 (84.5) | +0.8 |
| | | 6000 | 99.9 (97.7) | 100.0 (98.8) | −0.1 |
| attributes | column roles, 3×10, 0–50 | 600 | 66.3 (16.4) | 53.0 (3.6) | **+13.3** |
| | | 6000 | 94.1 (29.2) | 88.1 (17.6) | +6.0 |
| value shift | value axis, 3×10, 0–50 | 100 | 99.0 (97.3) | 38.3 (4.0) | **+60.7** |
| | | 600 | 99.9 (99.9) | 92.7 (81.0) | +7.2 |
| | | 6000 | 100.0 (100.0) | 98.4 (97.4) | +1.6 |
| rows 1 ↔ 2 | ViT, width 64, I-RAVEN | 1000 per layout | 24.3 (0.4) | 19.2 (0.2) | +5.1 |
| | | 3000 | 70.7 (24.2) | 42.7 (3.5) | **+28.0** |
| | | 6000 | 89.4 (55.3) | 79.1 (37.6) | +10.3 |
| rows 1 ↔ 2 | ViT, width 224, I-RAVEN | 1000 | 23.8 (0.5) | 18.6 (0.2) | +5.2 |
| | | 3000 | 62.0 (13.3) | 45.8 (4.5) | **+16.2** |
| | | 6000 | 92.1 (62.1) | 93.4 (64.4) | −1.3 |

- **The ViT learns late.** Up to 1000 puzzles neither version gets past 25, so its gain peaks at 3000. With 14× the weights the
  gain has gone by 6000.
- **Equal size:** components 266k vs 266k, attributes 288k vs 288k, rows within 4%. Giving the "without" models the same widths instead
  (2–3× the weights) does not change this: 75.6 and 52.5.
- **Value shift is not at equal size:** 10.8k vs 335k, with the rules given. Without the symmetry the model needs some
  60× the data for the same accuracy.

## 3. Accuracy on full data and fewer weights

I-RAVEN-X 3×10, values 0–50, 6000 puzzles.

| model | symmetry added | matched (exact) | the rule it targets, exact | weights |
|---|---|---|---|---|
| facet, named columns | – | 85.4 (15.0) | Arithmetic 18.6, Progression 77.4 | 1,411k |
| column roles | operand order | **94.1** (29.2) | Arithmetic **40.5** | 288k |
| column roles, wide | operand order | **94.9** (28.6) | Arithmetic 36.6 | 1,390k |
| row fold | one step per column | 92.2 (28.0) | Progression **99.2** | 305k |

Operand order gives the largest gain on full data, and the equal-size row shows it comes from the symmetry and not from
being smaller. Arithmetic's exact score stays low in every learned model.

## 4. Robustness: the same puzzle, transformed

A symmetric model gives identical answers. Models without the symmetry can use a shortcut that only works on what the
generator produces. I-RAVEN-X 3×10, values 0–50, matched:

| model | as is | rows 1 ↔ 2 | middle columns 2 ↔ 3 |
|---|---|---|---|
| copy links, per-panel gates (no row symmetry) | 97.8 | **≈60.0** | – |
| copy links, content gates | 94.9 | identical | 74.9 |
| facet, named columns | 85.4 | identical | 70.2 |
| column roles | 94.1 | identical | identical |
| column roles + content gates | 96.3 | identical | identical |
| row fold | 92.2 | identical | 91.7 |

- **The shortcut:** per-panel gates learned Distribute by address, because on 3×10 Distribute rows are row 1 shifted by a
  constant step. Swapping rows drops their Distribute puzzles from 98.3 to 39.5. So the table's highest as-is score is
  not a solution of the rule, and content gates give up 2.9 points to lose the shortcut.
- **Column roles + content gates** is the best model with both guarantees: 96.3 (43.7) at 314k weights.
- **On I-RAVEN and 3×3** these are data symmetries, so the training data already teaches them: labelled and separate
  models move by at most 0.8 points under mirror and size ↔ colour.

## 5. Reach beyond the training data

| test | model | result |
|---|---|---|
| unseen values, trained 0–50, scored 0–100 | value axis, 6000 / 100 puzzles | 100.0 (100.0) / 98.1 (96.8) |
| | weights per value, thermometer-code models | cannot run |
| rows of 5 / 20, trained on rows of 10 | value axis, 6000 puzzles | 89.8 (78.9) / 98.6 (63.7) |
| | weights per value, 6000 puzzles | 55.7 (17.0) / 98.9 (98.2) |
| | column roles | 65.6 (4.0) / 50.1 (4.6) |
| | row fold | 53.8 (0.4) / 55.1 (5.9) |
| rows of 20, trained on rows of 5 and 10 | column roles / row fold | 54.0 (7.9) / 47.7 (4.1) |
| unseen layout, trained without up-down, matched on it | facet / attribute GNN B / A | **99.1** / **99.5** / **99.9** |
| | object GNN / ViT 224 / ViT 224 with rows / MLP | 23.6 / 14.0 / 13.7 / 13.3 (chance 12.5) |
| the same, Arithmetic, type, size and colour exact with the occupancy given | facet / attribute GNN B / A | 94.3 / 97.6 / 98.5 |
| | object GNN / ViT 224 / ViT 224 with rows / MLP | 19.9 / 16.6 / 15.1 / 12.2 |
| unseen rule on an attribute: no Progression on Size in training, size exact on it | facet / attribute GNN B | **87.7** / **84.1** (89.9 / 89.4 trained on it, 3000 puzzles) |
| | attribute GNN A, same sharing with labels | 42.3 (85.9) |
| | ViT 224 with rows / object GNN / MLP / ViT 224 | 23.6 / 20.6 / 19.7 / 18.5 (65.3 / 83.2 / 68.0 / 48.9) |

Value shift gives real reach. So do components and attributes, at full data on I-RAVEN, where sections 2–3 show no gain left:
every model without them stays under 24 on the layout it never saw and on the held-out pair.
The labels in GNN A cost it 41.8 points there and nothing on the unseen layout. Row length does not follow from these
symmetries: roles are blind to the order of the middle columns but not to their number, and the fold transfers only
Constant, whether trained on one row length or on two (rows of 5 and 10).

## 6. Where symmetry does not help

- **Exact invariance on top of sharing, for data symmetries.** Labels break the symmetry but keep every weight shared, and nothing is lost:

  | data, puzzles | shared, exact symmetry | labels, shared weights |
  |---|---|---|
  | components, 100 / 1000 | 86.6 / 99.3 | 87.3 / 99.6 |
  | attributes, 600 / 6000 | 66.3 / 94.1 | 64.6 / 93.9 |

  That holds only on puzzles like the training ones. With a rule held out from one attribute, the labelled torch model
  falls to 42.3 where the symmetric one keeps 84.1 (section 5).

- **Full data.** Components fall to −0.1 and value shift to +1.6 at 6000 puzzles. Attributes keep +6.0 on 3×10 (+3.6 at
  the same widths, where that model was still improving).
- **The row fold on I-RAVEN.** 95.8 (63.0) vs the plain facet model's 97.3 (69.1): with three panels, per-column weights
  already read each position, so one shared step gains nothing.
- **Arithmetic.** It is a relation, and it lacks the value shift built in here: one operand can move if the sum moves with
  it, but all panels cannot move alike. On 3×10 value shift carries the other rules at 100 exact without
  `plus`, and Arithmetic falls to 11.1. On small ranges it holds 70.7 (3×3) and 65.7 (I-RAVEN), partly
  because few values leave few sums (`docs/ARITHMETIC.md`). Operand order is the only built-in symmetry that helps it.
  Rule twins test this directly, rows of 3, values 0–50: at 600 puzzles the shift-equivariant model gets 89.3 on midpoint,
  a rule of sum's size with the shift symmetry, and 6.9 on sum, while weights per value get 17.8 and 19.3. A table of 51
  entries with no symmetry gets 97.5 with weights per value. Over three seeds (2026-11-14) the four numbers are
  91.0 ± 3.4 and 6.2 ± 0.8, 18.0 ± 0.9 and 19.4 ± 1.1. So a rule's symmetry pays only when the model has it, and no
  symmetry hurts only when there are many facts (`docs/ARITHMETIC.md`).
  With `plus` as background facts and only its use learned (one matrix per kind of step, the same for every value),
  the value axis on 3×10 gets Arithmetic 100, and 88.7 when the state may also move along `next`, against 11.1
  without `plus` (value-axis rows, compared with each other only). The learned walk keeps to the trained row length:
  44.8 and 1.3 at rows of 20. The `plus` facts can be derived from the zero and `next` by the per-operand rule
  u + (w + 1) = (u + w) + 1 with identical outputs, but that rule is written in, not learned.
  On the rule twins this is direct (rows of 3, values 0–50): the shared ring model gets sum 4.7 / 6.9 at 100 / 600
  puzzles; given `plus`, which carries sum's own move (one operand with the answer), 97.6 / 99.8, while midpoint stays
  at 70.5 / 91.5. Over three seeds: sum 4.5 ± 2.3 / 6.2 ± 0.8 without `plus` and 98.2 ± 0.8 / 99.7 ± 0.4 with it;
  midpoint 60.3 ± 2.7 / 91.0 ± 3.4 without and 63.9 ± 3.9 / 91.7 ± 1.8 with, so `plus` neither helps nor hurts
  midpoint. So sum is not a rule without symmetry: its symmetry is as large as midpoint's, and pays the same once
  the model has it (`docs/ARITHMETIC.md`, "Sum and midpoint compared").

- **A symmetry the rule has, which also merges it with another rule** (scaling on the rule twins, 2026-11-15; RESULTS
  §11, `docs/ARITHMETIC.md`, "The scaling kernel"). Midpoint keeps its answer when every value is multiplied by the same
  number, and so does sum. Two models have it exactly on the ring of 101 values, with the shared model's weights:
  rows read in units of the largest power of two in their first step (doubled), or of the first step itself (scaled).

  | midpoint, exact %, 3 seeds | shared (shift only) | doubled | scaled |
  |---|---|---|---|
  | 100 puzzles | 60.3 ± 2.7 | 49.9 ± 5.3 | 19.0 / 100.0 / 21.0 |
  | 600 puzzles | 91.0 ± 3.4 | 67.5 ± 2.0 | 29.5 / 100.0 / 20.2 |
  | rows with a new first step whose odd part was seen (72), 100 puzzles | 0 / 0 / 0 | 50 / 43 / 39 | 18 / 72 / 19 |
  | rows that start like a Progression row, right, 600 puzzles | 0 / 0 / 0 of 33 | 1 / 11 / 1 of 152 | 149 / 505 / 102 of 505 |

  The symmetry reaches exactly the unseen first steps a count named beforehand. The totals fall for a reason the count
  left out: the data's Progression always steps by 1 or 2, so the size of the first step tells the rules apart, and a
  model that reads rows in a unit no longer sees it. Such rows are continued as a Progression unless the network reads
  the rule from the complete rows, which it did at one seed of three. This is the gap of section 1 from the other
  side: scaling is a task symmetry, the generator never scales a Progression, and the shortcut that lives in that gap
  was carrying accuracy. The shared model already loses its 33 rows with first step ±2 this way.

  Why the network reads the rule at one seed only was then measured (2026-11-16, `docs/ARITHMETIC.md`, "The rule
  gate"). A gate from the rule descriptor on the answer layer changed nothing: scaled with the gate 20 to 30% at all
  three seeds, shared with the gate none of the 33 rows. The cause is one layer earlier. In the 22 failing runs of 24
  the row descriptor, a sum over all values, is the same vector for a midpoint row and a Progression row (difference
  below 0.0001 per feature); in the two runs that learn midpoint it differs by 1.4 and 1.5. So the lost accuracy is
  not shown to be a cost of scaling: it is the cost of a model that cannot read the complete rows' rule, which scaling
  makes necessary on every row and the shift alone on 33 of 505. A row descriptor that carries the steps was the
  next change.

  It was then built, with the predictions written first (2026-11-17, `docs/ARITHMETIC.md`, "The row descriptor reads
  the steps"; `value_axis.py --steps`, the same symmetries). Midpoint, exact %, three seeds:

  | | shared | doubled | scaled |
  |---|---|---|---|
  | 100 puzzles, before | 60.3 ± 2.7 | 49.9 ± 5.3 | 19.0 / 100.0 / 21.0 |
  | 100 puzzles, the row descriptor reads the steps | 59.9 ± 4.5 | 89.2 ± 2.7 | 100.0 / 100.0 / 100.0 |
  | 600 puzzles, before | 91.0 ± 3.4 | 67.5 ± 2.0 | 29.5 / 100.0 / 20.2 |
  | 600 puzzles, the row descriptor reads the steps | 98.3 ± 1.9 | 99.2 ± 0.2 | 100.0 / 100.0 / 100.0 |
  | counting learner, 100 / 600 puzzles | 80.0 / 99.8 | 94.3 / 99.8 | 100 / 100 |

  So scaling does not cost accuracy: more of the rule's symmetry built in means less data at every step, as the
  count says, once the model can read the complete rows. The shared model's 33 rows are all right at 600 puzzles. The
  split of the unseen first steps holds as before: none of the 29 rows the count excludes for the doubled model, 64
  to 69 of the 72 it allows, all 101 for the scaled model.

- **The scope of the rule twins: values as constants** (2026-11-17; `docs/ARITHMETIC.md`, "Two checks without
  training"). With values read as numbers and the linear form given, a fit on each puzzle's two complete rows gets sum
  99.6, midpoint 99.8 and mirror 99.8 with no training, and 1.7 on lookup. So the twins price a symmetry for models
  that treat values as constants with an order. Row by row the count holds: with no training row in its orbit a row is
  never right (0 of 306, three seeds pooled), and with one it is right 74% of the time at 100 puzzles and 99% at 600.

- **A symmetry the data does not have** (facet model, I-RAVEN-X 3×3, values 0–10; RESULTS §7, "Wrong symmetries").
  Each is built in exactly (output difference 0 under its transformation, random weights).

  | built in | kind | 600 puzzles | 6000 puzzles |
  |---|---|---|---|
  | nothing extra | – | 95.1 (82.2) | 99.2 (92.7) |
  | columns 1 ↔ 2 tied | out of support | 92.1 (79.2) | 98.6 (95.1) |
  | v ↔ 10 − v in the input | conflicting | 39.7 (14.0) | 46.6 (13.4) |

  The out-of-support tie costs little at 6000 puzzles: Distribute and Arithmetic lose 3 and 5 points exact. At 600 it
  costs 3 points matched, all on Arithmetic (62.4 against 76.2 exact). The conflicting fold sits
  just under the ceiling the count predicts (50–53% of attribute values exact against 54–56), and more data barely
  moves it.

Limits of this evidence: nearly every row is one seed (the rule twins, the scaling kernel, the rule gate and the step-reading runs at 100 and 600 puzzles have three), and the value-axis rows have rules given, so they compare only
with each other.

## 7. Extra: the baselines

Same task and metrics. These comparisons mostly differ in more than one symmetry, so they support sections 2–6 and do
not replace them. MLP and object GNN at full data are regularized, mean of 2 seeds; the rest one seed.

**At about the facet model's size:**

| model | I-RAVEN | 3×3, 0–10 | 3×10, 0–50 |
|---|---|---|---|
| facet, for reference | 97.3 (69.1), 267k | 99.2 (97.0), 267k | 85.4 (15.0), 1,411k |
| attribute GNN B, relative labels (symmetric) | 96.8 (67.9), 267k | – | 85.8 (3.5), 1,599k |
| attribute GNN A, absolute labels | 95.0 (62.5), 266k | – | 95.0 (27.7), 1,596k |
| object GNN | 90.5 (55.1), 265k | 93.9 (67.7), 265k | 85.3 (4.2), 1,593k |
| MLP | 72.0 (27.5), 264k | 95.1 (74.3), 266k | 84.5 (2.7), 1,599k |
| ViT, width 64 | 79.1 (37.6), 335k | 35.4 (2.6), 335k | 52.9 (0.0), 428k |
| ViT, width 224 (14× the weights) | 93.4 (64.4), 3,753k | 47.5 (5.6), 3,753k | 43.1 (0.4), 4,079k |
| ViT, width 224, rows 1 ↔ 2 built in | 92.1 (62.1), 3,709k | 44.0 (5.6), 3,709k | 40.1 (0.0), 3,919k |
| CNN | 39.4 (2.1), 286k | – | – |

**Less training data** (I-RAVEN, puzzles per layout):

| puzzles | facet | GNN B | GNN A | object GNN | MLP | ViT, width 64 | ViT, width 224 |
|---|---|---|---|---|---|---|---|
| 300 | 90.0 (53.3) | 87.0 (53.0) | 86.1 (50.8) | 29.6 (1.1) | 24.9 (0.5) | 15.1 (0.1) | 15.6 (0.2) |
| 1000 | 94.1 (62.7) | 93.9 (64.0) | 93.4 (61.5) | 68.7 (21.9) | 48.1 (7.7) | 19.2 (0.2) | 18.6 (0.2) |
| 3000 | 96.1 (66.0) | 96.8 (68.8) | 95.9 (65.3) | 86.5 (49.8) | 64.5 (19.1) | 42.7 (3.5) | 45.8 (4.5) |
| 6000 | 97.3 (69.1) | 96.8 (67.9) | 95.0 (62.5) | 90.5 (55.1) | 72.0 (27.5) | 79.1 (37.6) | 93.4 (64.4) |

**Transformed puzzles** (seed 0 for MLP and object GNN):

| model | I-RAVEN: as is → rows 1 ↔ 2 | 3×10: as is → rows 1 ↔ 2 | 3×10: middle columns 2 ↔ 3 |
|---|---|---|---|
| attribute GNN B | 96.8 → identical | 85.8 → identical | identical |
| attribute GNN A | 95.0 → 95.0 | 95.0 → **55.5** | 82.0 |
| object GNN | 91.3 → 91.3 | 84.0 → **52.7** | 71.6 |
| MLP | 72.8 → 72.4 | 84.4 → **47.7** | 63.9 |
| ViT, width 224 | 93.4 → 93.4 | 43.1 → **37.9** | 44.4 |
| ViT, width 224, rows 1 ↔ 2 built in | 92.1 → identical | 40.1 → identical | 39.9 |
| ViT, width 64 | 79.1 → 78.9 | 52.9 → **38.5** | 47.3 |

What they add:
- **Attributes on their own carry most of it.** At 300 puzzles the two attribute GNNs have 86–87 and the object GNN 29.6. GNN
  B has the facet model's symmetries in plain torch and stays within 3 points of it at every size, so the symmetries
  do the work and not the logic language.
- **A vs B is the labels test in torch.** B has the symmetries, A adds one-hot labels. B leads by 0.5 to 1.8 at every
  size, about the 1.6 points between object GNN seeds, as with labels in the facet model. The real gap only shows on the
  held-out pair (section 5).
- **A's 3×10 lead is the same shortcut.** It beats B by 9.2, then its Distribute exact falls from 59.9 to 1.0 when rows
  swap.
- **Objects as a set:** on I-RAVEN the object GNN beats the MLP by 18.5, and on 3×10, with one object per panel, the two
  are level (85.3 vs 84.5). That fits: the symmetry has nothing to act on there.
- **Every baseline without the row symmetry fails the 3×10 swap, and none fails it on I-RAVEN**, which matches the kinds
  in section 1: rows 1 ↔ 2 are a task symmetry on 3×10 and a data symmetry on I-RAVEN. The ViT is no exception, 52.9 → 38.5
  at width 64 and 43.1 → 37.9 at width 224. With the rows built in it gets 40.1, so every ViT sits at 38–40 without the row
  order, and 9.5× the weights add nothing on 3×10.
- **ViT and the grids:** resolution does not explain the loss at this amount of data. Trained on the grids alone, finer
  patches and larger images stay at the control (2×2 / 3×3 / in-out grid 31.5 / 29.9 / 38.6 and 38.6 / 29.2 / 32.4, control
  37.6 / 28.3 / 43.0), and all three memorise. The same 18,000 grid puzzles with 18,000 single-object puzzles added get
  82.7 / 73.4 / 73.8 (`docs/RESULTS.md` §7). Finer images on the fully trained ViT were not tested.
- **ViT and translation:** it is not translation invariant. Trained without shifted images, width 224 falls from 93.4 to
  43.3 at a 1-pixel shift and 21.8 at 16 pixels. Sharing one filter across patches still helps: at width 64 a separate
  filter per patch gets 49.8 (9.2) with 12× the weights, against 79.1 (37.6); at 1000 puzzles both fail (22.5, 19.2).
- **The CNN is not the better vision model here.** It has 39.4 (2.1), and 50.5 (6.3) at 16×. It holds up better at 1 pixel
  (46.6 of 50.5) and is not invariant either (26.4 at 8 pixels). It differs from the ViT in more than filters.

## 8. Not run

The value-shift ablation on I-RAVEN (weights per value).
