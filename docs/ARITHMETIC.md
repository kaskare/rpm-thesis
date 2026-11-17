# Arithmetic

Which models do well on every rule except Arithmetic, on every dataset? Last updated 2026-11-02.

Cells are exact % per rule on unseen puzzles, in the order **Constant / Progression / Distribute / Arithmetic**.
I-RAVEN counts type, size and colour together, as `docs/RESULTS.md` §4 does; I-RAVEN-X counts every attribute value
(§5). One seed each.

| model | I-RAVEN | 3×3, 0–10 | 3×10, 0–50 |
|---|---|---|---|
| value axis, without `plus` | 84.9 / 97.7 / 98.1 / 65.7 | 100 / 100 / 99.9 / 70.7 | **100 / 100 / 100 / 11.1** |
| value axis, with `plus` (rules given) | 85.3 / 99.4 / 99.6 / 98.5 | 100 / 100 / 100 / 99.4 | 100 / 100 / 100 / 100 |
| copy links, per-panel gates | 84.5 / 90.0 / 89.4 / 85.5 | 99.6 / 100 / 100 / 98.3 | 100 / 92.9 / 86.8 / 30.2 |
| copy links, content gates | 84.3 / 89.3 / 88.9 / 81.7 | 99.6 / 100 / 100 / 97.5 | 100 / 98.2 / 59.8 / 30.0 |
| row fold | 82.1 / 88.8 / 88.4 / 75.9 | 99.6 / 100 / 99.6 / 98.3 | 99.9 / 99.2 / 15.1 / 36.2 |
| plain facet, named columns | 83.7 / 89.7 / 88.7 / 83.8 | 99.6 / 99.5 / 99.3 / 95.0 | 93.4 / 77.4 / 9.8 / 18.6 |

- **The value axis on I-RAVEN** gains nothing from 6000 puzzles: 1000 per layout gave 85.3 / 99.8 / 99.8 / 98.3 with `plus`
  and 84.8 / 97.6 / 97.9 / 66.4 without.
- **Without `plus` on I-RAVEN** drops every given arithmetic relation: the `plus` table and the union and difference
  sizes of the question row's position sets. Position Arithmetic is not in the table, but the count of objects it
  sets is part of the model's matched score.
- **Without `plus`, Arithmetic is not lost everywhere.** It is 65.7 on I-RAVEN and 70.7 on 3×3, against 11.1 on 3×10.
  Part of this is the data. A reference that sees only differences (a lookup table fit on the training puzzles, from
  the question row's differences and the sign of the complete rows to the answer's step) names no value, and it gets
  47.5 on 3×3, 43.4 on the I-RAVEN center layout and 2.4 on 3×10. With few values many sums can be guessed: 44.6% of
  the 3×3 Arithmetic question rows have a zero operand, so the answer repeats a panel. The gap between that reference
  and the model is not explained.
- **Constant on I-RAVEN** is held down by noise on the grids, which no model can generate (§4). On single-object layouts
  the value axis has 100.
- **Per-panel gates' Distribute on 3×10** is an address shortcut, not a solution of the rule (`docs/SYMMETRY_RESULTS.md`
  section 4).
- **Epoch choice.** These cells use the epoch picked by matched on the selection puzzles, and exact can move while matched
  does not. Plain facet on 3×10 has 15.0 exact overall at that epoch (35) and 22.6 at the last, so its 18.6 and 9.8 may be
  too low, and the gain to 40.5 with column roles too high. Column roles move less (29.0 to 33.0). Not rerun.

The value axis without `plus` also keeps the pattern on other 3×10 test sets, trained on rows of 10, values 0–50:

| test set | Constant / Progression / Distribute / Arithmetic |
|---|---|
| values 0–100 | 100 / 100 / 100 / 7.0 |
| rows of 20 | 100 / 95.1 / 100 / 11.6 |
| rows of 5 | 100 / 100 / **24.8** / 2.9 |

## Why Arithmetic behaves like a lookup table

Constant, Progression and Distribute can be solved by comparing panels: the same value, the same step, the same set of
values. Arithmetic cannot. To a model that only compares panels, c = a + b is not a pattern but a list of separate facts,
and the answer has to be looked up. The reason is that addition needs a zero, and the panels do not define one.

**Values are places, not amounts.** A value is a position on a scale, such as a size level or a colour level. So there are
two kinds of quantity:

- a *place*: a value, such as 20;
- a *step*: the difference of two places, such as 20 − 10 = +10.

Steps can be added to each other, and a step can be added to a place. Two places cannot be added until the scale has a
starting point. Temperature works the same way:

| | in °C | the same temperatures in K |
|---|---|---|
| the two values | 10 and 20 | 283 and 293 |
| halfway between them | 15 | 288, which is 15 °C |
| their sum | 30 | 576, which is 303 °C |

Halfway does not depend on the scale. The sum does, because the two scales have different zeros.

**What zero is.** Zero is the value z that changes nothing when added: a ⊕ z = a for every a. It is the place that means "no
amount", and it is what turns a place into an amount: the amount at b is its distance from zero, b − z. Adding then means
moving a by that amount:

```
a ⊕ b = a + (b − z) = a + b − z
```

Every z gives a consistent addition. With z = 0, 10 ⊕ 20 = 30. With z = 3, 10 ⊕ 20 = 27, and 10 ⊕ 3 = 10, as a zero should.

**The panels do not say where zero is.** It is a convention of the generator, not something drawn. In 2,500 puzzle files of
I-RAVEN's center layout, colour follows c = a + b or c = a − b in all 635 puzzles with Arithmetic on colour, so its zero is
index 0. Size follows c = a + b + 1 or c = a − b − 1 in all 600 with Arithmetic on size, so its zero is index −1, one below
the smallest size, which no panel can show. Renumbering removes the difference: count sizes from 1 instead of 0, and size
follows c = a + b or c = a − b, the same as colour. That is one fixed number per attribute, so it is bookkeeping. It moves
the label of zero, and it does not remove the need for one: after it, a further shift still breaks the rule. Index 0 is not
"nothing" either: it is a white object or the smallest object, and it is still drawn.

**A shift moves the zero.** A shift adds the same s to every panel: (a, b, c) becomes (a + s, b + s, c + s). Rows of Constant,
Progression and Distribute stay valid, as long as the values stay in range. A row of addition does not:

```
(a + s) + (b + s) = c + 2s,  but the shifted answer is c + s
```

For example, 10 + 20 = 30. Shifted by 5 the row reads 15, 25, 35, but 15 + 25 = 40. The shifted row is not nonsense: it is
correct for the addition whose zero is 5, because 15 + 25 − 5 = 35. A shift by s turns the addition with zero z into the
addition with zero z + s. So addition has no shift symmetry: the rule depends on where zero is, and a shift changes exactly
that.

**What does repeat from row to row.** All three rows of a puzzle use the same operation, and that is a real symmetry: rows 1
and 2 can be exchanged and the puzzle stays valid. In 611 of 611 training puzzles of the center layout with Arithmetic on
colour, the three rows have the same operator. Every rule has this symmetry, and most models here build it in. It is not
the one Arithmetic lacks.

The numbers do not repeat. Puzzle 150, where the operation is a + b:

| row | a | b | c | step a → b | step b → c |
|---|---|---|---|---|---|
| 1 | 4 | 1 | 5 | −3 | +4 |
| 2 | 7 | 2 | 9 | −5 | +7 |
| 3 | 3 | 2 | ? = 5 | −1 | +3 |

Rows 1 and 2 have the same two steps in only 5 of the 611 puzzles. If the steps did repeat, say +3 then −2 in every row, the
rule would be shift-symmetric: 1, 4, 2 and 5, 8, 6 follow it from different starting places, and one row would show how to
solve the others, as in Progression. In addition, the second step is c − b = a, the row's own first value. So the steps
change with where the row starts, and that is what "no shift symmetry" looks like in a puzzle.

Any rule c = pa + qb takes the same test. A shift gives c + (p + q)s, so the rule is shift-symmetric exactly when p + q = 1.
Subtraction has p + q = 0 and fails too. Written with the first step d = b − a, the rule is

```
c = b − p·d + (p + q − 1)·b
```

The last term uses a place as an amount, and that is where a zero is needed. It disappears exactly when p + q = 1.

**A convolution does not help.** A convolution uses the same weights at every value: the model has one weight for "+5", not
one per value (`core/pynl/value_axis.py`). This is what makes it so strong on the other three rules. It also means that when
every panel moves by s, the answer moves by s:

```
f(a + s, b + s) = f(a, b) + s
```

Taking s = −b gives f(a, b) = b + f(a − b, 0). So every such model that reads a and b has the form

```
f(a, b) = b + h(b − a)
```

Whatever its kernel, width or depth, it can only take a step from b, and it chooses that step from the first step alone.
Addition needs the step c − b = a, and the first step does not determine a:

| a, b | first step b − a | a + b | step needed, c − b |
|---|---|---|---|
| 5, 15 | +10 | 20 | +5 |
| 10, 20 | +10 | 30 | +10 |
| 15, 25 | +10 | 40 | +15 |

The same input needs three different outputs, so h can be right on only one row per first step. In the test below (values
0–50, a, b ≥ 1) there are 97 first steps and 1,225 rows, so at most 97 / 1,225 = 7.9% can be right, and 8.1 is measured on the
unseen puzzles. The trained model gets 7.1. On small ranges the limit is much higher, because there are few rows (see the
bullets above).

**Addition is a convolution, but of a different kind.** Write a panel as a one-hot vector e(a). Convolving two panels gives

```
(e(a) ∗ e(b))(c) = Σ_u e(a)(u) · e(b)(c − u)
```

which is 1 exactly at c = a + b. This is what the `plus` table computes. It differs from the model's learned convolution in
two ways:

- **The kernel is a panel.** It changes with every row, while a learned kernel is one fixed set of weights.
- **It is not shift-symmetric.** e(a + s) ∗ e(b + s) = e(a + b + 2s). The peak moves by 2s, so it lands on the right value only
  if zero is known.

The `plus` table supplies both. It is a list of facts such as plus(2, 3, 5): 66 of them on values 0–10 and 1,326 on 0–50. Facts
such as plus(0, w, w) fix the zero. With the table, the value axis has Arithmetic 100 on 3×10; without it, 11.1. That 100 is a
lookup in a table the model was given, not something it learned.

**Addition has symmetries of its own.** "No shift symmetry" above means no shift of every panel by the same s. A row stays
valid when a, b and c move by s₁, s₂ and s₃ exactly when s₁ + s₂ = s₃, and equal shifts meet that only at s = 0. (For midpoint
the condition is s₁ + s₂ = 2s₃, which equal shifts always meet.) Counted on the table for values 0–50:

| move | (2, 5, 7) becomes | still a + b = c? | `plus` facts it keeps, of those in range |
|---|---|---|---|
| every panel +1 | (3, 6, 8) | no, 3 + 6 = 9 | 0 of 1,275 |
| a and c +1 | (3, 5, 8) | yes | 1,275 of 1,275 |
| b and c +1 | (2, 6, 8) | yes | 1,275 of 1,275 |
| a +1, b −1 | (3, 4, 7) | yes | 1,275 of 1,275 |
| a and b swapped | (5, 2, 7) | yes | 1,326 of 1,326 |

These moves do collapse the table: start from 0 + 0 = 0, apply (s, t, s + t), and all 1,326 facts follow. So addition is one
fact, where zero is, plus these moves. It looks like a list of separate facts only to a model that cannot make them. The
convolution cannot, because it moves every panel alike. The `plus` rule can: move one operand by s and the sum moves by s. So
the table is how the model is given addition's own symmetry. In the terms of the literature, these moves are autotopisms and
equal maps are automorphisms (`docs/READING_LIST_arithmetic_symmetry.md`).

**Finding the zero from the puzzle does not remove the table.** A complete row reveals the zero, z = a₁ + b₁ − c₁, so

```
c₃ = a₃ + b₃ − z = a₃ + b₃ + c₁ − a₁ − b₁
```

The coefficients sum to 1, so this relation over five panels is shift-symmetric, and a convolution could express it. But in
steps it reads

```
c₃ − a₃ = (b₃ − a₁) + (c₁ − b₁)
```

which is an addition of two steps. It needs no zero, but it is still an addition, now a table on steps, and the model must read
five panels at once. Shift symmetry can remove the zero, not the addition. `--find-zero` tries this route and has no finished
result.

**So a model has two options:**

- **It names every value.** It can then tell the rows apart, but it has to meet the sums one at a time. Weights per value
  score 4.9 / 19.3 / 69.8 at 100 / 600 / 6000 puzzles, close to plain memorising at 4.7 / 24.1 / 90.7.
- **It shares weights across values.** It then learns the other rules from few puzzles, but it cannot tell the sums apart.

The table also grows quickly: 1,225 rows on rows of 3 with values 0–50, and about 12.6 billion on 3×10 with nine operands, where
memorising cannot work. That link to the low 3×10 scores is an inference, since the test below has rows of 3. Addition is not
without structure: moving one operand moves the answer by the same amount, (a + s, b, c + s). Models with ordered codes use a
little of this (facet 31.2 at 600 puzzles, memorising 24.1), but no model that learns the sum passes 40.5 on 3×10.

**Midpoint: the same size, no zero needed.** Midpoint is c = (a + b) / 2, with a ≠ b and a + b even. It looks like addition, but
it is a place plus a step:

```
c = a + (b − a) / 2
```

That reads "from a, go half the way to b", and no zero appears. Here p = q = 1/2, so p + q = 1, and the formula above gives
c = b − d/2, which means h(d) = −d/2. The same first step as before:

| a, b | first step b − a | (a + b) / 2 | step needed, c − b |
|---|---|---|---|
| 5, 15 | +10 | 10 | −5 |
| 10, 20 | +10 | 15 | −5 |
| 15, 25 | +10 | 20 | −5 |

The same input needs the same output. All 41 rows with first step +10 are one fact, and the whole table of 1,250 rows is 50
facts, one for each first step (±2, ±4, …, ±50). Addition stays at 1,225, because every row has its own pair of first step and
answer step.

Side by side, on values 0–50:

| | addition, c = a + b | midpoint, c = (a + b) / 2 |
|---|---|---|
| rows in the table | 1,225 | 1,250 |
| p + q | 2 | 1 |
| every panel shifted by s | the sum moves by 2s, the row breaks | the midpoint moves by s, the row holds |
| needs a zero | yes | no |
| step from b to c | a, a place used as an amount | −(b − a) / 2, the first step alone |
| facts for a model with the convolution | 1,225, on only 97 first steps | 50 |
| best possible from the first step, rule told | 8.1 | 100 |
| exact at 600 puzzles, weights per value | 19.3 | 17.8 |
| exact at 600 puzzles, convolution | 6.9 | 89.3 |

The last two rows are the point. Without the convolution the two rules are equally hard, because both are tables of the same
size. With it, midpoint collapses to 50 facts, and addition cannot collapse at all.

## Is the missing symmetry what makes it hard?

The claim under test: Arithmetic lacks the value shift the model builds in, so to that model one sum teaches nothing about
another, and that is why it needs so much data. New puzzles keep everything in I-RAVEN-X and replace Arithmetic with a test rule (rows of 3,
values 0–50; `data_prep/rule_twins.py`, `data/iravenx_twins/`). No model is given `plus`.

| rule | row | table entries | rows kept by a shift | by reflection | by swapping a, b |
|---|---|---|---|---|---|
| sum (control) | c = a + b, a, b ≥ 1 | 1,225 | 0% | 0% | 100% |
| midpoint | c = (a + b) / 2, a ≠ b | 1,250 | 100% | 100% | 100% |
| mirror | c = 2a − b, a ≠ b | 1,250 | 100% | 100% | 0% |
| lookup | b = g(a), c = g(b) | 51 | 0% | 0% | 0% |

- **Midpoint is sum's twin:** the only rule c = pa + qb with both the shift symmetry and sum's operand swap.
- **Mirror is the guard:** midpoint's answer lies between its operands, so errors shrink, while sum's grow. Mirror uses the
  same triples with the columns moved, so its errors grow too.
- **Lookup has no symmetry:** `g` is a fixed table on the values. It is not a permutation, because a permutation
  commutes with its own powers and so keeps a hidden symmetry. Colour refinement separates all 51 values, which leaves only
  the identity.

**Similar difficulty is measured, not assumed.** A learner that only memorises rows it has met, with nothing trained, gets
these exact scores on the unseen puzzles:

| puzzles | sum | midpoint | mirror | lookup |
|---|---|---|---|---|
| 100 | 12.4 | 13.1 | 11.3 | 95.0 |
| 600 | 56.8 | 52.7 | 48.7 | 100 |
| 1000 | 74.0 | 70.3 | 66.2 | 100 |
| 6000 | 100 | 100 | 100 | 100 |

Reading only the differences in the question row, with the rule told, the best possible score is 8.1 on sum, 100 on
midpoint and mirror, and 82.0 on lookup. With 51 rows, b − a works as a fingerprint of the row, so a rule with no symmetry can still be memorised.

**Models.** Shared: the value axis on a ring (`--ring --kernel 50 --pair --no-plus`), exactly shift-equivariant, checked to
7e-17 with random weights, 14,785 weights. The ring matters: on a line, a kernel as wide as the range would let pooling read
how far a value is from the ends, which is the zero. Per value: the same template with weights per value, 459,635 weights.
Facet: plain, 278,088 weights. MLP: regularized, 2 seeds.

**Predictions.** Without the symmetry built in, sum, midpoint and mirror should be about equally hard, and lookup far
easier. With it, midpoint and mirror should come from very few puzzles, and sum should stay near 8.1.

**The claim is refuted if** sharing does not help midpoint, or lookup is as hard as sum for the per-value models, or the
models without the symmetry find midpoint far easier than sum. In the last case the rules were not matched in difficulty,
and mirror shows whether that came from the shrinking errors.

## Result

The claim holds. A rule of sum's size comes from few puzzles when its symmetry is built in, is as hard as sum when it is
not, and a rule with no symmetry but few facts is easy. Exact % on the test rule, at 100 / 600 / 6000 puzzles:

| model | sum | midpoint | mirror | lookup |
|---|---|---|---|---|
| shared, shift-equivariant | 4.7 / 6.9 / 7.1 | **57.4 / 89.3 / 92.7** | 45.3 / 78.3 / 88.1 | 35.9 / 74.3 / 77.7 |
| per value | 4.9 / 19.3 / 69.8 | 5.7 / 17.8 / 75.2 | 4.6 / 15.7 / 63.2 | **53.0 / 97.5 / 97.5** |
| facet (600 / 6000 only) | 31.2 / 65.5 | 24.0 / 76.6 | 14.7 / 41.9 | 60.3 / 97.1 |
| MLP, mean of 2 seeds | 2.7 / 3.0 / 26.4 | 4.8 / 5.9 / 21.7 | 2.9 / 2.9 / 10.2 | 2.8 / 5.6 / 17.7 |
| memorising answered rows, untrained | 4.7 / 24.1 / 90.7 | 5.0 / 21.6 / 90.5 | 4.0 / 19.7 / 91.7 | 65.3 / 100 / 100 |

- **Sharing helps a rule that has the symmetry.** Midpoint goes from 5.7 to 57.4 at 100 puzzles and from 17.8 to 89.3 at 600.
  The shared model at 600 puzzles beats weights per value at 6000 (75.2), with 3% of the weights. Mirror goes from 4.6 to 45.3.
- **The same model gets nothing on sum:** 7.1 at 6000 puzzles, at its limit. This follows from the algebra, so it checks that
  the symmetry is really built in. It is not a discovery.
- **Without the symmetry, midpoint is no easier than sum.** Per value: 5.7 against 4.9, 17.8 against 19.3, 75.2 against 69.8.
  Facet: 24.0 against 31.2, 76.6 against 65.5. A rule's symmetry pays only if the model has it.
- **No symmetry does not mean hard.** Lookup against sum, per value: 53.0 against 4.9, 97.5 against 19.3. Facet: 60.3 against
  31.2, 97.1 against 65.5.
- **A symmetry the rule lacks costs points.** On lookup the shared model trails weights per value by 17–23 points, and on sum
  by 62.7 at 6000 puzzles.
- **Weights per value memorise.** At 100 and 600 puzzles they stay close to the memorising row, which counts only answered
  rows because only the answer panel has a loss. At 600 they stay within about 3 points from epoch 70 to 200, so this is not
  undertraining. On 3×10 there are about 12.6 billion rows, so memorising cannot work there, which fits the low Arithmetic
  scores above. That last step is an inference; this test has rows of 3.

So Arithmetic's cost needs both: many facts, and no symmetry in the model to collapse them. Few facts with no symmetry is easy (lookup),
and many facts with the symmetry built in is easy too (midpoint).

**Against the three refutation tests:** sharing helps midpoint; lookup is far easier than sum for weights per value; midpoint
is at most 11 points ahead of sum without the symmetry, and behind it on the facet model at 600 puzzles. None is met.

**What did not go as predicted:**
- **The MLP.** Lookup is not easier for it: 17.7 against sum's 26.4 at 6000 puzzles. It barely learns any rule here (Constant
  about 46, Progression about 16 at 6000), so this row says little. Its thermometer input may favour linear relations such as
  sum over a table. That is untested.
- **Mirror is harder than sum without the symmetry:** 63.2 against 69.8 (per value), 41.9 against 65.5 (facet), 10.2 against
  26.4 (MLP). At 6000 puzzles sum sits between its two twins on per value and facet. Mirror's larger coefficient or its lost
  operand swap could cause this; the test does not separate them.
- **The shared model stops short of 100 on midpoint and mirror.** It reads only the question row, so it cannot tell these rows
  from Progression when the first step is ±1 or ±2. At 6000 puzzles it gets 0 of 33 such midpoint rows, 32 of them answered as
  Progression, and 469 of the other 472. On mirror it gets 0 of 81 (78 as Progression) and 421 of the other 422. With the rule
  not told, the best possible from the first step is 7.3 / 93.5 / 83.9 / 77.7, so the model sits at that limit. Reading the rule
  from the complete rows should clear this; not tried. Mirror's 88.1 is one epoch; the others are about 84.

**Limits:**
- **Epoch choice.** These cells pick the epoch by exact on the selection puzzles. This was changed after the first results:
  matched sits at 97–99% here, so it picked epochs as early as 10 of 60, and the facet row was rerun for it (jobs
  11933739–46). By matched, the 6000 cells are shared 7.5 / 92.9 / 83.7 / 78.5, per value 53.5 / 75.2 / 56.5 / 99.2, facet
  57.4 / 62.2 / 42.5 / 85.0 and MLP 21.7 / 20.8 / 8.8 / 18.9. The verdict is the same.
- **6000 is not converged.** Per value and facet are still rising at their last epochs. The gaps at 100 and 600 are the firm
  ones.
- **Noise.** One seed each except the MLP, about 500 rows per cell, so roughly ±4. Three seeds of the shared and
  per-value models on sum and midpoint at 100 and 600 puzzles came later ("Three seeds" below): sd 0.5 to 3.4.
- **One row, not the whole puzzle.** Sum breaks shift symmetry as a relation within a row. Across five panels,
  c3 = a3 + b3 + c1 − a1 − b1 has coefficients summing to 1, so a shift-equivariant model could reach it. `--find-zero` tries
  this and has no finished result.

## Sum and midpoint compared (2026-11-08)

Why does midpoint come from few puzzles once the convolution is built in, while sum gets nothing from it? The short
answer: **the two rules are equally symmetric, but in different directions, and the convolution builds in only one of
them.**

### The raw rules
Both are linear relations on a row (a, b, c), on values 0–50:

| | sum | midpoint |
|---|---|---|
| the rule | a + b = c | a + b = 2c |
| rows in the table (rule twins) | 1,225 (a, b ≥ 1) | 1,250 (a ≠ b, a + b even) |
| moves (s₁, s₂, s₃) of a, b, c that keep every row valid | s₁ + s₂ = s₃ | s₁ + s₂ = 2s₃ |
| every panel moved alike, (s, s, s) | **no**: s + s ≠ s | **yes**: s + s = 2s |
| one operand moved with the answer, (s, 0, s) | **yes** | no: it needs (s, 0, s/2) |
| reflection v → 50 − v | no | yes |
| the table as one row plus moves | 0 + 0 = 0 and the moves | 0, 0, 0 and the moves (with a + b even) |

Each rule's moves form a group with two free parameters, and each table is one row moved around by that group. So
neither rule is a list of separate facts in itself; both are one fact and a group. They differ only in which moves
belong to the group. The convolution's move, every panel alike, belongs to midpoint's group and not to sum's.
Addition's own move, one operand with the answer, belongs to sum's and not to midpoint's. (These moves are the
autotopisms of the reading list, `docs/READING_LIST_arithmetic_symmetry.md`; `analysis/autotopism_check.py` counts
them on Z₅ and Z₇.)

### What a built-in move does: learners that know only a symmetry
`analysis/rule_orbits.py` has no network. Each learner predicts c from (a, b) and must respect one group of moves: if
the input moves, its answer moves the same way. So it learns one answer per *orbit* of inputs, a training row teaches
its whole orbit, and it is fit by majority vote per orbit:
- **table**: no move, one answer per pair (a, b), which is memorising;
- **every panel +s**: the convolution's move. Then c = b + g(b − a), one answer per first step b − a;
- **one operand with the answer**: addition's moves. Then c = a + b + k, a single number k for the whole table.

Fit on the question rows of the first N rule-twin puzzles (the rows that carry a loss in every model here), and scored
on the unseen puzzles 6300–7299, exact %:

| puzzles | sum: table | sum: every panel +s | sum: one operand | midpoint: table | midpoint: every panel +s | midpoint: one operand |
|---|---|---|---|---|---|---|
| 10 | 0.0 | 0.0 | **100** | 1.0 | 16.4 | 0.8 |
| 100 | 4.7 | 3.6 | **100** | 5.0 | **80.0** | 3.4 |
| 300 | 12.6 | 5.3 | **100** | 12.5 | **98.8** | 2.2 |
| 600 | 24.1 | 6.3 | **100** | 21.6 | **99.8** | 2.2 |
| 6000 | 90.7 | 7.3 | **100** | 90.5 | **100** | 2.2 |
| unlimited data (fit on the whole table) | 100 | 6.9 | 100 | 100 | 100 | 2.2 |
| orbits of inputs | 1,225 | 97 | 1 | 1,250 | 50 | 1 |

1. **With a move from the rule's own group, the table collapses to its orbits.** Midpoint has 50 orbits under the
   convolution's move, one per first step, so each training row teaches about 25 rows: at 100 puzzles 32 of the 50
   first steps have been seen, and they cover 80% of the unseen rows. Sum has one orbit under its own moves: the 8
   Arithmetic rows of 10 puzzles fix k = 0, and everything follows.
2. **With a move from outside the rule's group, the learner is not just unhelped, it is capped.** Every row of an orbit
   gets the same answer step, while the rule needs a different one for each: sum under the convolution's move stays
   below 8, midpoint under addition's moves at 2.2, however much data.
3. **With no move, the two rules cost the same:** 4.7 against 5.0 at 100 puzzles, 24.1 against 21.6 at 600. These
   are the "memorising answered rows" of the results table above.

So on midpoint the convolution's move reaches 80% with 100 puzzles, where memorising needs several thousand (34.7 at
1000, 90.5 at 6000), and on sum it is worth nothing. Addition's moves are worth even more on sum and nothing on
midpoint. **A built-in symmetry pays exactly when it is one of the rule's own
moves.** It is a property of the pair, model and rule, not of either alone. That is the whole difference between the
two rules: same size, same amount of symmetry, and only midpoint's lies along the axis the convolution shares weights
over.

Fitting on the complete rows too (three rows a puzzle, though no model has a loss on them) moves every column up the
same way: midpoint with the convolution's move 91.5 at 100 puzzles, sum with it still 5.9.

### The trained models against these learners
The shared, shift-equivariant value axis (results above) gets 57.4 / 89.3 / 92.7 on midpoint at 100 / 600 / 6000
puzzles, against 80.0 / 99.8 / 100 for the learner with the same move. Two things separate them:
- **The learner is told the rule.** The network must also tell midpoint rows from Progression rows, and with a first
  step of ±1 or ±2 it cannot from the question row alone: at 6000 puzzles that is 33 of the 505 rows, 32 answered as
  Progression. The other 472 it gets 469 right, which is the learner's 100.
- **At 100 puzzles the network gets less out of the orbits it has seen** (57.4 against 80.0).

**It learns one answer per first step, not the law "halfway".** The saved checkpoints' answers on the unseen puzzles
(job 11974989), split by whether the row's first step b − a occurs in any training question row. The saved
checkpoint is the epoch picked by matched, so its totals differ from the table's (47.1 at 100 puzzles, not 57.4).
The same split from checkpoints picked by exact, at three seeds, is under "Three seeds" below: the zero holds in each.

| checkpoint | first step seen in training | first step never seen |
|---|---|---|
| shared, midpoint, 100 puzzles | 58.9 (of 404) | **0.0** (of 101) |
| shared, midpoint, 300 | 77.0 (of 499) | 0.0 (of 6) |
| shared, midpoint, 600 | 89.5 (of 504) | 0.0 (of 1) |
| shared, mirror, 100 | 62.4 (of 356) | 0.0 (of 147) |
| shared, mirror, 600 | 79.9 (of 493) | 0.0 (of 10) |

| checkpoint | pair (a, b) seen in training | pair never seen |
|---|---|---|
| per value, midpoint, 100 | 88.0 (of 25) | 1.7 (of 480) |
| per value, midpoint, 600 | 64.2 (of 109) | 2.0 (of 396) |

Not one row with an unseen first step is right, at any size. The answer −(b − a) / 2 is linear in the first step, but
the network does not carry it from one first step to the next: it fills the orbits the convolution gives it, one by
one, as the counting learner does, only less efficiently. Weights per value do the same with pairs. So the collapse
is exactly the built-in symmetry's, and nothing beyond it is learned here.

Weights per value follow the table learner: 5.7 / 17.8 / 75.2 against 5.0 / 21.6 / 90.5. Sum on the shared model:
4.7 / 6.9 / 7.1, at the cap.

### Giving the model sum's moves
`plus(U, W, V)` is addition's move written as a relation: (u + s) + w = v + s keeps the fact, and the Peano rule
u + (w + 1) = (u + w) + 1 is the same move with s = 1. A state that walks through `plus` (`--learned-plus`, "Numbers
as objects" above) therefore moves one operand with the answer. Prediction from the learners: with it, sum should come
from as few puzzles as midpoint does with the convolution. Runs: the shared ring model of the results above with
`--learned-plus` added (19,937 weights; the kernel stays a ring, only the `plus` facts do not wrap), on sum and
midpoint at 100 and 600 puzzles (jobs 11974990–93). Exact % on the test rule, epoch picked by exact on the
selection puzzles, as in the results table:

| shared ring model | sum, 100 / 600 puzzles | midpoint, 100 / 600 puzzles |
|---|---|---|
| without `plus` (results above) | 4.7 / 6.9 | 57.4 / 89.3 |
| **with `plus` used by learned rules** | **97.6 / 99.8** | 70.5 / 91.5 |
| counting learner, addition's moves | 100 / 100 | 3.4 / 2.2 |
| counting learner, every panel +s | 3.6 / 6.3 | 80.0 / 99.8 |

**The prediction holds.** Given its own moves, sum comes from 100 puzzles (97.6), as midpoint does from the
convolution's; without them it is stuck below 8. Midpoint keeps what the convolution gave it (the network still has
it), and is no worse for the extra facts. It is no better either: over three seeds 63.9 ± 3.9 against 60.3 ± 2.7 at
100 puzzles ("Three seeds" below), so the 70.5 here was a high draw.
Picked by matched instead, the cells are 98.2 / 99.0 and 70.5 / 90.1. The sum cells in this table were picked by exact
on the test rule; by exact on the whole puzzle, as everywhere else, that run gives 98.2 / 99.2. One seed each here;
three seeds: sum 98.2 ± 0.8 / 99.7 ± 0.4, midpoint 63.9 ± 3.9 / 91.7 ± 1.8.

So "Arithmetic has no symmetry" was wrong as stated: it has one, of the same size as midpoint's, and a model that has
it learns sum as easily. What it lacks is the shift symmetry the convolution builds in.

**Three cautions on this comparison (2026-11-13).**
- **"Equally symmetric" is true of every rule c = pa + qb.** Its rows stay valid under (s₁, s₂, p·s₁ + q·s₂), always two
  free parameters. What separates the rules is whether the common shift (s, s, s) is among them, and that holds exactly
  when p + q = 1: midpoint, mirror, Progression and Constant yes, sum no. That is the statement to make.
- **Sum's full group is the rule up to one number.** A function that respects every move (s₁, s₂, s₁ + s₂) is a + b + k,
  which is why the counting learner with addition's moves has a single orbit. So `plus` gives sum its whole group, while
  the convolution gives midpoint a one-parameter part of its group (50 orbits). 97.6 against 57.4 at 100 puzzles reflects
  that, not an easier rule. `plus` is the relation given as background facts; calling it "sum's symmetry" undersells how
  much is given.
- **The moves are groups on the integers, not on 0–50.** A shift can leave the range, midpoint needs a + b even, and
  the shared model's ring wraps where the data does not. The orbit counts above are made on the actual tables, so the
  numbers stand; the group language is exact only away from the edges.

### Three seeds (2026-11-14)

Every cell above is one run. Here: seeds 1–3 of the shared, per-value and `plus` models on sum and midpoint at 100 and
600 puzzles (36 runs, jobs 12034883–12034918, 13 to 82 minutes each, peak 66 GB). `--seed` calls
`neuralogic.manual_seed`; the same seed reproduces a run exactly, and earlier runs drew a fresh random seed each.

**One rule for every cell, fixed before the runs:** the epoch with the highest whole-puzzle exact on the selection
puzzles, the first at that maximum; the cell is exact % on the test rule on the unseen puzzles. The seeded runs save
that checkpoint. `analysis/twins_seeds.py` prints the table. "First run" is the earlier unseeded run under the same rule.

| model | rule | puzzles | seed 1 / 2 / 3 | mean ± sd | first run |
|---|---|---|---|---|---|
| shared | sum | 100 | 5.9 / 5.7 / 1.8 | 4.5 ± 2.3 | 4.7 |
| shared | sum | 600 | 5.5 / 7.1 / 6.1 | 6.2 ± 0.8 | 6.9 |
| shared | midpoint | 100 | 57.2 / 62.0 / 61.8 | **60.3 ± 2.7** | 57.4 |
| shared | midpoint | 600 | 87.1 / 93.1 / 92.9 | **91.0 ± 3.4** | 89.3 |
| per value | sum | 100 | 4.1 / 5.3 / 4.9 | 4.8 ± 0.6 | 4.9 |
| per value | sum | 600 | 20.7 / 18.7 / 18.7 | 19.4 ± 1.1 | 19.3 |
| per value | midpoint | 100 | 5.5 / 5.0 / 4.6 | 5.0 ± 0.5 | 5.7 |
| per value | midpoint | 600 | 19.0 / 17.6 / 17.4 | 18.0 ± 0.9 | 17.8 |
| shared + `plus` | sum | 100 | 98.4 / 98.8 / 97.2 | **98.2 ± 0.8** | 98.2 |
| shared + `plus` | sum | 600 | 99.8 / 100.0 / 99.2 | **99.7 ± 0.4** | 99.2 |
| shared + `plus` | midpoint | 100 | 66.1 / 66.1 / 59.4 | 63.9 ± 3.9 | 70.5 |
| shared + `plus` | midpoint | 600 | 89.7 / 93.3 / 92.3 | 91.7 ± 1.8 | 91.5 |

- **Every conclusion stands.** The largest sd is 3.9 points. The effects are 55 and 73 points (midpoint, shared against
  per value, at 100 and 600 puzzles) and 94 points (sum, with `plus` against without, at 100).
- **The twins are matched without the symmetry:** per value 4.8 ± 0.6 against 5.0 ± 0.5, and 19.4 ± 1.1 against 18.0 ± 0.9.
- **Sum under the shift stays under its ceiling in all six cells** (1.8 to 7.1 against 7.9). Over all 120 evaluations of
  the six runs, four are above 7.9, the highest 8.7 (44 of 507 rows). The ceiling is a count on the whole table; on 507
  sampled rows it is 7.9 ± 1.2, so these are within sampling error.
- **Seed 3's 1.8 at 100 puzzles is an arbitrary epoch, not a worse model.** Whole-puzzle exact on selection stays at
  57–58% in these runs, since only the puzzles without sum come out right, so its maximum (epoch 10 here) says nothing
  about sum. That run's 20 evaluations average 4.6.
- **`plus` does not help midpoint.** 63.9 ± 3.9 against 60.3 ± 2.7 at 100 puzzles and 91.7 ± 1.8 against 91.0 ± 3.4 at
  600. The first run's 70.5 was a high draw, above all three seeds.
- **The older `plus` cells on sum (97.6 / 99.8) were picked differently:** by exact on the test rule, not on the whole
  puzzle. Under the rule here that run gives 98.2 / 99.2.
- **Still one run each:** the 6000-puzzle column, mirror, lookup, the facet rows.

**Rows by whether training covered them, from the checkpoints in the table** (job 12034926,
`analysis/twins_seeds.py --dumps experiments/twins_eval`). The totals of each split equal the cells above. Seeds 1 / 2 / 3:

| checkpoint | first step seen in training | first step never seen |
|---|---|---|
| shared, midpoint, 100 puzzles | 71.5 / 77.5 / 77.2 % (of 404) | **0 / 0 / 0** (of 101) |
| shared, midpoint, 600 | 87.3 / 93.3 / 93.1 % (of 504) | 0 / 0 / 0 (of 1) |

| checkpoint | pair (a, b) seen in training | pair never seen |
|---|---|---|
| per value, midpoint, 100 | 24 / 21 / 20 (of 25) | 4 / 4 / 3 (of 480) |
| per value, midpoint, 600 | 88 / 83 / 80 (of 109) | 8 / 6 / 8 (of 396) |

The zero holds at every seed, and now comes from the model in the table (the earlier split used checkpoints picked by
matched, 47.1 in total). The counting learner with the same move gets 80.0 at 100 puzzles by being right on every
covered row; the network is right on 72–78% of them.

Figure: `docs/figures/twins_sum_midpoint.png` (and `.pdf`), drawn by `analysis/twins_figure.py` from the tables above
(100 and 600 puzzles: means of the three seeds with one sd; 6000 and the counting learner: one run).

### The scaling kernel (2026-11-15)

The shared model learns one answer per first step, and gets 0 of 101 rows whose first step is new. Midpoint has a second
symmetry the model lacks: multiply every value by the same number k and the row stays valid (ka + kb = 2kc). Sum has it
too. If the count is right, building it in must make new first steps answerable, and the count says which.

**How it is built.** The kernel's offsets are counted in each row's own unit: V = U + k·S, one weight per k as before.
The unit S is a constant the template does not name, like the values. Background facts `mul(bk, S, D)` say that step D
is k units of S on the model's ring of 101 values. 101 is prime, so every step has every multiple and every fraction on
the ring, a half-step included. One rule derives a row's unit from its first two panels:

- **scaled** (`--scale step`): the unit is the row's first step. Multiplying every value by any number and adding any
  number then renames values and steps and keeps every background fact, so the model is exactly equivariant to all of
  it (the theorem of notes §21). It can only answer c = b + g·(b − a), one number g per context.
- **doubled** (`--scale pow2`): the unit is the largest power of two in the first step. The model is equivariant to
  doubling, as long as the doubled first steps stay within ±50, and learns one answer per odd part of the first step.

Both have the shared model's 14,785 weights and its ring. A row whose first two values are equal has no unit and no
kernel atoms; the counts carry it (Constant).

**Checked with random weights** (`--check-scale`, max difference between the output of a transformed puzzle and the
transformed output):

| model | every value +9 | ×2 | ×3, ÷2, reflected |
|---|---|---|---|
| shared | 2e-17 | 2e-2 (not equivariant) | 2e-2 |
| doubled | 2e-17 | 2e-17 in the 68 to 81 of 100 puzzles where the doubled first steps stay in range, 1e-2 in the others | 1e-2 |
| scaled | 2e-17 | 2e-17 | 2e-17 |

On 100 puzzles per attribute (job 12039067, `log/check_scale_12039067.out`); a puzzle that is not moved differs from
itself moved by about 1e-2, so 1e-2 means no symmetry.

**Predictions by counting, made before the runs** (`analysis/rule_orbits.py`, learners "doubling" and "scaling"; no
network):

| | shared | doubled | scaled |
|---|---|---|---|
| midpoint: orbits of first steps | 50 | 26 | 1 |
| midpoint: counting learner at 100 / 600 puzzles | 80.0 / 99.8 | 94.3 / 99.8 | 100 / 100 |
| the 101 unseen rows with a new first step, 100 puzzles | 0 right (measured) | 72 answerable (their odd part was seen), 29 not | all 101 answerable |
| sum: ceiling on the whole table | 97 of 1,225, 7.9% | 88, 7.2% | 17, 1.4% |

The counting learner is told the rule; the network must also recognise it, so it will fall short of the learner's
totals, as the shared model does (60 against 80).

**The predictions fail if** the doubled model gets rows right among the 29 or none among the 72, if the scaled model
leaves the rows with a new first step at zero, or if sum goes clearly above its ceiling.

**Results** (24 runs, jobs 12039041–12039065, 26 to 93 minutes each; seeds 1–3, the selection rule of "Three seeds";
`analysis/twins_seeds.py --dumps experiments/twins_eval`). Exact % on the test rule, unseen puzzles:

| model | rule | puzzles | seed 1 / 2 / 3 | mean ± sd |
|---|---|---|---|---|
| shared | midpoint | 100 | 57.2 / 62.0 / 61.8 | 60.3 ± 2.7 |
| doubled | midpoint | 100 | 55.6 / 48.7 / 45.3 | 49.9 ± 5.3 |
| scaled | midpoint | 100 | 19.0 / **100.0** / 21.0 | two outcomes, see below |
| shared | midpoint | 600 | 87.1 / 93.1 / 92.9 | 91.0 ± 3.4 |
| doubled | midpoint | 600 | 65.5 / 69.5 / 67.5 | 67.5 ± 2.0 |
| scaled | midpoint | 600 | 29.5 / **100.0** / 20.2 | two outcomes |
| shared | sum | 100 | 5.9 / 5.7 / 1.8 | 4.5 ± 2.3 |
| doubled | sum | 100 | 3.7 / 3.6 / 3.4 | 3.6 ± 0.2 |
| scaled | sum | 100 | 3.2 / 0.8 / 2.0 | 2.0 ± 1.2 |
| shared | sum | 600 | 5.5 / 7.1 / 6.1 | 6.2 ± 0.8 |
| doubled | sum | 600 | 4.1 / 3.7 / 4.3 | 4.1 ± 0.3 |
| scaled | sum | 600 | 3.4 / 1.0 / 3.4 | 2.6 ± 1.4 |

**The rows with a new first step came out as counted.** Unseen midpoint rows right at 100 puzzles, seeds 1 / 2 / 3:

| rows | shared | doubled | scaled |
|---|---|---|---|
| first step new, its odd part seen (72) | 0 / 0 / 0 | **50 / 43 / 39** | 18 / 72 / 19 |
| first step and odd part new (29) | 0 / 0 / 0 | **0 / 0 / 0** | 4 / 29 / 4 |
| first step seen (404) | 289 / 313 / 312 | 231 / 203 / 190 | 74 / 404 / 83 |

- The doubled model is right on 39 to 50 of the 72 rows where the shared model is right on none, and on none of the 29,
  at every seed. That is the line the count drew.
- The scaled model has no zero group. In seed 2 it is right on all 101, and on every other row: 100.0% from epoch 20
  with 100 puzzles. In seeds 1 and 3 it is right on about one row in five, the same share in each group, as a model
  that cannot see the first step must be.

**What the count missed: rows that start like a Progression row.** The totals fell (49.9 against 60.3 at 100 puzzles,
67.5 against 91.0 at 600). The data's Progression has the steps ±1 and ±2. Read in a unit, a midpoint row whose first
step is one unit has the same first two panels as a Progression row; only the two complete rows above it then say which
rule it is. On those rows the network continues the row by its first step (it generates 2b − a). At 600 puzzles,
seeds 1 / 2 / 3:

| model | rows that start like a Progression row | right among them | continued as a Progression | right among the other rows |
|---|---|---|---|---|
| shared | 33 (first step ±2) | **0 / 0 / 0** | 33 / 33 / 33 | 440 / 470 / 469 of 472 |
| doubled | 152 (first step ±2, ±4, ±8, ±16, ±32) | 1 / 11 / 1 | 148 / 139 / 150 | 330 / 340 / 340 of 353 |
| scaled | 505 (every row) | 149 / **505** / 102 | 356 / 0 / 403 | none |

- **Each symmetry merges more midpoint rows with Progression:** 6.5% of the rows, then 30%, then all of them. The
  first step's size was the network's cue for the rule, and a scaling symmetry hides it by construction.
- **The shared model has the same failure on fewer rows, and it was already recorded** (above, at 6,000 puzzles: none
  of the 33 rows with first step ±2, 469 of the other 472). It holds at every seed and both sizes, and is the plateau
  near 93%: seeds 2 and 3 are right on 470 and 469 of the other 472 rows. The count should have carried it over.
- **The information is in the model's input.** The rule descriptor reads the complete rows, where midpoint and
  Progression differ, and seed 2 of the scaled model uses it. The other runs do not. Why the network usually fails to
  use it is not measured; the answer layer adds the context to the kernel's features and does not gate them.
- **At 100 puzzles the doubled model is right on 80 / 70 / 62% of its other rows** (281, 246, 218 of 353); the shared
  model on 61 / 66 / 66% of its other rows. With the 152 rows counted wrong the counting learner's total is 69.9%, not 94.3.

**Sum.** The doubled model stays under its ceiling (3.6 ± 0.2 and 4.1 ± 0.3 against 7.2). The scaled model gets 2.0 ± 1.2
and 2.6 ± 1.4, and no cell is above 3.4. With this symmetry a model can give one multiple g per context. On the whole
table the best multiple is right on 17 of 1,225 rows (1.4%). On the 507 unseen rows the best is g = 1, continuing the
row by its first step: it is right on the 17 rows with b = 2a (3.4%; the counting learner's 2.0 was the table's best
multiple scored on these rows, not their own best). At 600 puzzles seeds 1 and 3 answer g = 1 on 496 of the 497 rows
with a ≠ b and are right on exactly those 17 rows; seed 2 answers other multiples and is right on 5 rows (dumps: jobs
12039193–95). So the scaled model treats sum as it treats midpoint in its failed seeds: as a Progression.

**Each prediction:**

| prediction | outcome |
|---|---|
| doubled: none of the 29 rows right | held, 0 of 29 at three seeds |
| doubled: rows among the 72 right | held, 39 to 50 of 72 |
| scaled: rows with a new first step not at zero | held: 22, 101 and 23 of 101; but only one seed in three learns the rule |
| sum stays under its ceiling | held for doubled; held for scaled too: no cell above the 3.4% that one multiple reaches on these rows |
| totals follow the counting learner | **failed**: the count did not include the rows that merge with Progression |

**What this shows.** A scaling symmetry makes new first steps answerable in exactly the rows the count names. It also
merges rows of different rules, which the count over one rule's table cannot see, and the network then has to tell the
rules apart from the context, which it does at one seed of three. **What it does not show:** that the scaled model learns
midpoint in general, or that scaling is a better bias than the shift alone; on the totals it is worse. The scaled
variant is also given most of the rule, as `plus` gives sum: it can only learn the one number g. Three seeds, 200
epochs, no tuning.

### The rule gate (2026-11-16)

Two failures above have the same shape. The scaled model learns midpoint at one seed of three, and the shared model is
wrong on every midpoint row whose first step is ±2. In both, the question row's first two panels do not tell midpoint
from Progression. The two complete rows do, and the rule descriptor reads them. One explanation was open: the answer
layer only adds the descriptor to the kernel's features, so it cannot switch a feature off by the rule.

**The change** (`--gate`). A gate `gt(A) = sigmoid(W·ctx2(A) + b)`, one number per feature, multiplies the question
row's kernel features (`h` at its last panel and `pv` of its last pair, through their weights) before the answer
layer's tanh. The descriptor is also still added. The gate names no value and is the same at every value, so the value
symmetries stay. It adds one matrix and a bias, 1,056 weights (15,841 in all). The facet model's content gates are the
same construction (notes §21).

**Predictions, made before the runs.** Midpoint, 100 and 600 puzzles, seeds 1–3, the data and the selection rule of
"Three seeds".

| model | without the gate | with the gate, if the answer layer is the cause |
|---|---|---|
| scaled, 100 and 600 puzzles | 100.0 at seed 2; 19 to 30 at seeds 1 and 3 | 100 at every seed |
| shared, 600 puzzles | 91.0 ± 3.4, none of the 33 rows with first step ±2 | the 33 rows right, about 99 |
| shared, 100 puzzles | 60.3 ± 2.7, none of the 33 | no prediction |

The second prediction is the weaker one. In the scaled model every midpoint row teaches the gate, and the descriptor of
a midpoint row is the same whatever its first step. In the shared model only the training question rows with first
step ±2 teach it: 18 of them at 600 puzzles, 4 at 100 (213 at 6,000, where the model without the gate got none of the
33 right). And the descriptor of a midpoint row differs with its first step, so the gate must also carry over to
complete rows it was not taught on.

**The predictions fail if** the scaled model with the gate stays below 100 at some seed, or the shared model with the
gate is still right on none of the 33 rows at 600 puzzles.

**Found in the old dumps while the gated runs were training** (before any of their results;
`analysis/twins_seeds.py --dumps`, third split). The failed seeds of the scaled model without the gate do follow the
context, through a different input. Unseen rows right, by whether the question row shares a value with one of the
complete rows (the overlap `ov`, an input of the rule descriptor), seeds 1 / 2 / 3:

| rows, scaled model without the gate | 100 puzzles | 600 puzzles |
|---|---|---|
| midpoint, no shared value (397) | 0 / 397 / 16 | 47 / 397 / 0 |
| midpoint, a shared value (108) | 96 / 108 / 90 | 102 / 108 / 102 |
| Progression, no shared value (710) | 710 / 710 / 679 | 663 / 710 / 710 |
| Progression, a shared value (124) | 98 / 124 / 96 | 93 / 124 / 99 |

In seeds 1 and 3 a midpoint row with no shared value is continued as a Progression, and one with a shared value is
answered as a midpoint in 90 to 102 of 108 rows. With a shared value the model also tells Progression from midpoint in
most rows (93 to 99 of 124). So the answer layer can switch on the context without a gate; what the failed seeds do not
use is the complete rows' step pattern (`rd`) when nothing else marks the puzzle. That speaks against the explanation
the gate tests, and was not known when the predictions were written.

**Results** (12 runs, jobs 12043421–32, 21 to 82 minutes; dumps 12043446–57). Exact % on midpoint, unseen puzzles,
seeds 1 / 2 / 3:

| model | 100 puzzles | 600 puzzles |
|---|---|---|
| shared | 57.2 / 62.0 / 61.8 (60.3 ± 2.7) | 87.1 / 93.1 / 92.9 (91.0 ± 3.4) |
| shared with the gate | 72.5 / 65.7 / 61.6 (66.6 ± 5.5) | 90.3 / 88.1 / 92.5 (90.3 ± 2.2) |
| scaled | 19.0 / 100.0 / 21.0 | 29.5 / 100.0 / 20.2 |
| scaled with the gate | 23.2 / 20.2 / 24.0 | 20.2 / 20.2 / 30.5 |

| rows right, seeds 1 / 2 / 3 | 100 puzzles | 600 puzzles |
|---|---|---|
| shared with the gate: the 33 rows with first step ±2 | 0 / 0 / 0 (all continued as a Progression) | 0 / 0 / 0 (all continued) |
| shared with the gate: the other 472 rows | 366 / 332 / 311 | 456 / 445 / 467 |
| scaled with the gate: midpoint, no shared value (397) | 19 / 1 / 19 | 1 / 0 / 47 |
| scaled with the gate: midpoint, a shared value (108) | 98 / 101 / 102 | 101 / 102 / 107 |

| prediction | outcome |
|---|---|
| scaled with the gate: 100 at every seed | **failed**: no run of six learns the rule; the highest cell at any evaluated epoch is 30.5 |
| shared with the gate, 600 puzzles: the 33 rows right | **failed**: none of the 33 at any seed, at either size |

- The gate changes nothing that can be told from three seeds. The shared model's totals are within its spread (the 66.6
  at 100 puzzles is 1.2 sd of the gated runs above 60.3). The scaled model without the gate learned the rule at seed 2,
  with it at no seed; a seed draws different initial weights in the two templates, so this is two runs of six against
  none of six and does not show that the gate hurts.
- The gated scaled runs fail as the others did: right when the question row shares a value with a complete row.
- The symmetries are kept (job 12043433, random weights, 100 puzzles): scaled with the gate 2e-17 for all six
  transforms in 100 of 100 puzzles; shared with the gate 3e-17 for the shift and 2e-2 for every scaling, as without it.

**Where the rule is lost: the row descriptor is a constant** (`analysis/twins_probe.py`, job 12043845; saved
checkpoints, 30 unseen puzzles, no training). The model reads a complete row through `rd(Rw, A)`: the features of each
neighbouring pair, summed over the 101 values (`pd`), averaged over the row's pairs. The probe asks a checkpoint for
`rd` of the first row and compares the attributes under midpoint with those under Progression: the largest difference
per feature between the two means (features lie in −1..1).

| runs | learns midpoint | `rd` of a midpoint row against a Progression row | `rd` among midpoint rows with different steps |
|---|---|---|---|
| scaled, seed 2, 100 and 600 puzzles | 100.0, 100.0 | **1.5, 1.4** | 0 (one unit: every midpoint row is the same row) |
| scaled, seeds 1 and 3 (4 runs) | 19.0 to 29.5 | 5e-11 to 7e-05 | 0 |
| scaled with the gate (6 runs) | 20.2 to 30.5 | 1e-14 to 2e-05 | 0 |
| shared (6 runs) | none of the 33 rows | 6e-14 to 1e-05 | 9e-13 to 4e-06 |
| shared with the gate (6 runs) | none of the 33 rows | 1e-13 to 2e-07 | 7e-12 to 2e-06 |

- **In all 22 runs that fail, the descriptor of a midpoint row and of a Progression row is the same vector** to four
  decimals or more. In the shared runs it is also the same for midpoint rows with different first steps: it says
  nothing about the row's steps at all. In the two runs that learn the rule it differs by 1.4 and 1.5.
- So the rule never reaches the answer layer in the failed runs, and no gate on that layer could have helped. In the
  failed runs the means of the rule descriptor `ctx2` under the two rules differ by at most 0.1 to 0.2 per feature,
  against a spread of 0.6 to 1.7 among the midpoint attributes (the overlap and the counts change from puzzle to
  puzzle). In the two working runs the means differ by 1.8, with a spread of 0.4 to 0.5.
- **Why the descriptor can be constant** (read off the template, not measured). `pd` sums the pair's features over
  every value of the ring: Σ_V tanh(W₁·h(P, V) + W₂·h(Q, V)). On the ring each panel's kernel weights appear once in
  that sum wherever the panel's value is, so without the tanh the sum is (W₁ + W₂)·Σ_k w_k for every pair. The pair's
  step enters only through the curvature of the tanh. Why training grows that part at one initialisation and not at
  the others is not measured.
- This explains both earlier failures with one cause: the shared model's 33 rows (every seed, every size, 6,000
  puzzles included) and the scaled model's two outcomes. The doubled model's 152 rows were not probed.

**What this shows.** The scaled model's failure at two seeds of three is not shown to be a cost of the scaling
symmetry. It is a model that cannot read which rule the complete rows follow, in every run but two; scaling only made
that reading necessary on every row, where the shared model needs it on 33 of 505. The symmetry prediction itself, the
split of the unseen first steps, is untouched. **What it does not show:** that a row descriptor which carries the
steps would make the scaled model learn midpoint at every seed. The next change (run on 2026-11-17, "The row descriptor reads the steps" below): read a pair's step directly,
as the first panel's kernel feature at the second panel's value (`h(P, A, V)` at Q's value is the kernel weight of
exactly that step), instead of pooled over all values.

### Two checks without training (2026-11-17)

Both read data that was already there; nothing was trained. They were made to judge two ideas for the thesis: to lead
with which unseen rows a model reaches, and to order the built-in moves as a ladder of priors.

**One training row in an orbit is enough.** `analysis/twins_seeds.py --dumps`, fourth split. The unseen midpoint rows by
how many training question rows fall in their orbit (the first step for the shared model, the pair (a, b) for weights
per value), rows right, three seeds pooled. There are 57 midpoint question rows in 100 training puzzles and 311 in 600.

| training rows in the orbit | 0 | 1 | 2 | 3–4 | 5–8 | 9+ |
|---|---|---|---|---|---|---|
| shared, 100 puzzles | 0 of 303 | 422 of 573 (74%) | 286 of 417 (69%) | 172 of 186 (92%) | 34 of 36 | – |
| shared, 600 puzzles | 0 of 3 | 71 of 72 (99%) | 2 of 3 | 89 of 90 (99%) | 696 of 768 (91%) | 521 of 579 (90%) |
| per value, 100 puzzles | 11 of 1,440 (1%) | 65 of 75 (87%) | – | – | – | – |
| per value, 600 puzzles | 22 of 1,188 (2%) | 200 of 270 (74%) | 45 of 51 (88%) | 6 of 6 | – | – |

- **No training row in the orbit, no answer.** 0 of 306 for the shared model, and 1 to 2% for weights per value, where
  guessing one of 101 values gives 1%. The count's "not reachable" holds row by row.
- **One row is enough for most of an orbit:** 74% at 100 puzzles and 99% at 600. More rows in the orbit help at 100
  puzzles (92% from three or four) and add nothing at 600. So what a single row teaches depends on how much the rest
  of the model has been trained, and the network's shortfall against the counting learner (60 against 80 at 100
  puzzles) is not a fixed number of rows an orbit needs.
- **The two lowest cells at 600 puzzles are the Progression-like rows.** First steps −2 and +2 have 8 and 10 training
  rows, so their 33 unseen rows a seed (14 and 19), all continued as a Progression ("The rule gate"), sit in the last two
  bins. They are 99 of the 133 wrong rows. Without them the bins are 696 of 726 (96%) and 521 of 522.

**Values read as numbers: a fit with no training** (`analysis/twins_linear_fit.py`). For each attribute, c = p·a + q·b
is fitted on the puzzle's own two complete rows by least squares and applied to the question row. No other puzzle is
used. Exact % on the unseen puzzles:

| data | its fourth rule | Progression | Constant | Distribute Three |
|---|---|---|---|---|
| sum | 99.6 (505 of 507) | 98.7 | 100 | 0 |
| midpoint | 99.8 (504 of 505) | 99.3 | 100 | 0 |
| mirror | 99.8 (502 of 503) | 98.7 | 100 | 0 |
| lookup | 1.7 (8 of 479) | 98.6 | 100 | 0 |

- **The three linear twins are solved without one training puzzle,** once values are numbers and the rule's form,
  linear in the two operands, is given. In the terms of "What a built-in move does" this is the far end: the whole form
  is built in, and the two complete rows fix its two numbers.
- **It gets nothing where the rule is not linear:** 0 on Distribute Three and 1.7 on lookup.
- **Its few misses are puzzles that do not fix the rule.** On the sum, midpoint and mirror puzzles all 32 misses on
  the linear rules (13, 7 and 12) are puzzles whose two complete rows have rank below 2, so they leave p and q open.
- **So the twins measure what a model pays for treating values as constants with an order,** as every template here
  does. That is the scope of the comparison between sum and midpoint, and the thesis has to state it. A reader from the
  vector-symbolic side will ask: as far as I remember, the neuro-symbolic model that I-RAVEN-X was built around
  (Hersche et al. 2024) works on value encodings with additive structure. Check the paper before citing it.
- **What it adds to the ladder of built-in moves** for midpoint at 100 puzzles, orbits left to learn against exact %:
  1,250 (per value) 5.0; 50 (shift) 60.3; 26 (doubled) 49.9; 1 (scaled) 19.0 / 100.0 / 21.0; the form given, values as
  numbers, 99.8 with no training. The counting learners say the middle of the ladder should rise (80.0, 94.3, 100);
  the trained models fall there, for the reason found in "The rule gate". A row descriptor that carries the steps
  restores the order: 59.9, 89.2 and 100.0 ("The row descriptor reads the steps" below).

### The counts behind the proofs (2026-11-17)

`docs/thesis/ch3_pricing_proofs.md` states eleven claims about the twins with a proof in plain words each. Three
needed a count that was not here yet (`analysis/pricing_checks.py`, nothing trained).

**Coverage from the table alone.** With rows drawn evenly from the table (the generator samples three of its rows
per attribute), the expected share of unseen rows whose orbit n training rows reach is the sum over orbits of
P(o)·(1 − (1 − P(o))ⁿ), with P(o) the orbit's share of the table. Midpoint, expected / reached by the counting learner
on the actual training puzzles (57, 311 and 3,040 question rows):

| built in | orbits | 100 puzzles | 600 | 6,000 | rows for 90% |
|---|---|---|---|---|---|
| nothing (one answer per pair) | 1,250 | 4.5 / 5.0 | 22.0 / 21.6 | 91.2 / 90.5 | 2,878 |
| every panel +s | 50 | 74.8 / 80.0 | 98.7 / 99.8 | 100 / 100 | 107 |
| +s and ×2 | 26 | 90.5 / 94.3 | 99.3 / 99.8 | 100 / 100 | 55 |
| +s and ×k | 1 | 100 / 100 | 100 / 100 | 100 / 100 | 1 |

The formula gives the counting learner within 6 points before a puzzle is drawn. The shift cuts the rows needed for
90% by a factor of 27.

**Which table keeps which move** (rows that are rows again, of those that stay in range): the shift, sum 0 of 30,325,
midpoint and mirror 40,400 of 40,400; scaling by 2, 3 and 5, all three rules every row; one operand with the answer,
sum 39,200 of 40,425 (the rest land on a = 0, which the table leaves out), midpoint and mirror 0 of 51,450.

**The ceiling on sum belongs to learners that answer from the question row.** The function
c₃ = b₃ + (c₁ − b₁) + (a₃ − a₁) uses only differences, follows a shift of the whole puzzle, and is right on 507 of 507
unseen sum rows, because c₁ − b₁ = a₁ in a sum row. So the puzzle-wide shift alone does not cap sum at 7.9%. The
shared model stays under the ceiling because of how it reads the puzzle: a value's offsets to the question row's two
panels, the count of panels at that value in each row, and a descriptor that is the same at every value. It never
measures from the question row to another row. "Sum and midpoint compared" above said "a shift-equivariant model
cannot fit sum"; read it as "a learner that answers from the question row". Chapter 7 (7.3, 7.4) is corrected.

**How sharp "under the ceiling" is** (a fact check of the earlier wording, 2026-11-17). Every cell of the shared model
on sum is 7.1 or less at the epoch picked on the selection puzzles. Over all 172 evaluated epochs of its nine runs
(six seeded, three first runs), 8 are above 7.9, the highest 9.7 in the run on 6,000 puzzles, whose epochs average
7.9. The 507 unseen sum rows fix a 7.9% rate to about 1.2 points, and the ceiling is on the expected accuracy: fitted
to the unseen rows themselves, one answer per orbit would get 28.4% of them. In 48 of the 507 rows the answer is a
value already in the puzzle, which the template can see. So the model sits at the ceiling; "under it" was too strong.

### The row descriptor reads the steps (2026-11-17)

"The rule gate" ended on a change that was not built. It is built here, and it decides the middle of the ladder above.

**The change** (`value_axis.py --steps`). The row descriptor `rd` also reads each neighbouring pair's step directly:
`ps(P, Q, A) = tanh(W h(P, A, V))` at the one value V that panel Q has. `h(P, A, V)` is the kernel weight of the
offset from P's value to V, so at Q's value it is the weight of exactly the pair's step, and with `--scale` of the step
in the row's unit. `rd` averages a second weight matrix over a row's `ps`, next to the pooled `pd`. No value is named,
so the symmetries stay. 2,048 more weights: 16,833.

What a complete row then looks like to the model, as the two steps of its pairs:

| row | shared | doubled | scaled |
|---|---|---|---|
| Progression, step δ | δ, δ | ±1, ±1 | 1, 1 |
| midpoint, first step d = m·2^j with m odd | d, −d/2 | m, −m/2 | 1, −1/2 |

In the scaled model every midpoint row gives one pair of kernel weights and every Progression row another.

**Predictions, written before the runs.** Midpoint, 100 and 600 puzzles, seeds 1 to 3, the selection rule of "Three
seeds": 18 runs (`sharedsteps`, `doubledsteps`, `scaledsteps`).

| | predicted | fails if |
|---|---|---|
| scaled with the steps | 100 at every seed and both sizes, as seed 2 reached without them | any of the six runs is below 90 |
| doubled with the steps, the 152 rows that begin like a Progression row, 600 puzzles | more than half right at every seed (they were 1 / 11 / 1); the total above 67.5 at every seed | 11 or fewer at two seeds |
| shared with the steps, the 33 rows with first step ±2, 600 puzzles | some right at two seeds or more (18 training rows teach them; the weaker test) | none at all three seeds |
| the probe, all 18 runs | a midpoint row's `rd` differs from a Progression row's by more than 0.1 in some feature (it was below 0.0001 in 22 runs of 24) | below 0.01 in any run |
| unseen first steps, 100 puzzles | shared still 0 of 101; doubled none of the 29 and more of the 72 than 39 to 50; scaled all 101 | shared or doubled right on a row the count excludes |
| the symmetries, random weights | unchanged, 1e-16 | – |

No prediction for the shared model at 100 puzzles: 4 training rows have first step ±2.

**What each outcome would mean.** If the scaled model learns midpoint at every seed, the accuracy lost in "The scaling
kernel" was the row descriptor's and the ladder's order is restored in its middle. If its descriptor separates the
rules and it still fails, the cause is somewhere else again and the cost may be the symmetry's after all.

**Wired in, before the runs** (random weights, 40 unseen puzzles): 16,833 weights in all three models. The largest
difference between the mean `rd` of a midpoint row and of a Progression row is 0.26, 0.34 and 0.39 with the steps
(shared, doubled, scaled) against 0.04, 0.06 and 0.15 without. In the scaled model all midpoint rows still give one
`rd` (spread 7e-16).

**Runs:** training jobs 12049211 to 12049245 (odd numbers), their dumps 12049212 to 12049246 (even), the symmetry
check 12049210 (`log/check_steps_12049210.out`), the probe 12049249. Submitted 2026-11-17.

**Results, 100 puzzles.** Exact % on midpoint, unseen puzzles, seeds 1 / 2 / 3.

| model | without the steps | with the steps |
|---|---|---|
| shared | 57.2 / 62.0 / 61.8 (60.3 ± 2.7) | 54.9 / 63.4 / 61.4 (59.9 ± 4.5) |
| doubled | 55.6 / 48.7 / 45.3 (49.9 ± 5.3) | **90.1 / 91.3 / 86.1** (89.2 ± 2.7) |
| scaled | 19.0 / 100.0 / 21.0 | **100.0 / 100.0 / 100.0** |

| unseen midpoint rows right, with the steps, seeds 1 / 2 / 3 | shared | doubled | scaled |
|---|---|---|---|
| first step seen in training (404) | 277 / 319 / 310 | 391 / 392 / 368 | 404 / 404 / 404 |
| new first step, odd part seen (72) | 0 / 1 / 0 | 64 / 69 / 67 | 72 / 72 / 72 |
| new first step, new odd part (29) | 0 / 0 / 0 | 0 / 0 / 0 | 29 / 29 / 29 |
| rows that begin like a Progression row in the model's unit | 17 / 27 / 19 of 33 | 152 / 149 / 147 of 152 | 505 / 505 / 505 of 505 |

- **The scaled model learns midpoint at every seed:** all 505 unseen rows from 100 puzzles, the 101 rows with a new
  first step included, and every Progression row beside them (834 of 834). The prediction is met at this size.
- **The doubled model rises from 49.9 to 89.2,** against 94.3 for its counting learner. It is right on 64 to 69 of the
  72 rows the count allows (39 to 50 without the steps) and on none of the 29 it excludes. It no longer continues a
  midpoint row as a Progression: 147 to 152 of the 152 rows that begin like one are right, where 1 to 11 were at 600
  puzzles without the steps.
- **The shared model's 33 rows, for which no prediction was made at this size:** 17 / 27 / 19 right. In the twelve
  earlier shared runs, with and without the gate, none was ever right. Its total does not move (59.9 against 60.3):
  it gains about 21 of the 33 rows and loses as many among the other 472 (260 / 293 / 291 right, against 289 / 313 /
  312 without the steps).
- **One row against the count.** The shared model at seed 2 is right on 1 of the 101 rows with a first step it never
  saw; the prediction said none. Guessing among 101 values gives about 1 such row per seed, and the earlier shared
  runs had 0 of 303, so this is at the rate of chance. As written, that prediction is missed by one row.

**Results, 600 puzzles.** Exact % on midpoint, unseen puzzles, seeds 1 / 2 / 3. All 18 runs completed 200 epochs.

| model | without the steps | with the steps |
|---|---|---|
| shared | 87.1 / 93.1 / 92.9 (91.0 ± 3.4) | **96.0 / 99.4 / 99.4** (98.3 ± 1.9) |
| doubled | 65.5 / 69.5 / 67.5 (67.5 ± 2.0) | **99.4 / 99.0 / 99.2** (99.2 ± 0.2) |
| scaled | 29.5 / 100.0 / 20.2 | **100.0 / 100.0 / 100.0** |

| unseen midpoint rows right, with the steps, seeds 1 / 2 / 3 | shared | doubled | scaled |
|---|---|---|---|
| rows that begin like a Progression row in the model's unit | 33 / 33 / 33 of 33 | 152 / 151 / 152 of 152 | 505 / 505 / 505 of 505 |
| the other rows | 452 / 469 / 469 of 472 | 350 / 349 / 349 of 353 | – |
| the one row with a first step training never had | 0 / 0 / 0 | 0 / 0 / 0 | 1 / 1 / 1 |

- **The scaled model is at 100.0 in all six runs,** at every evaluated epoch from the tenth on. With one number left
  to learn (`docs/thesis/ch3_pricing_proofs.md`, Statement 7) it is right as soon as it can read which rule the
  complete rows follow.
- **The shared model's 33 rows are all right at every seed.** No run of the twelve without the steps had one of them.
  Its total goes from 91.0 to 98.3, against 99.8 for its counting learner.
- **The doubled model goes from 67.5 to 99.2,** against 99.8 for its counting learner.
- **No model continues a midpoint row as a Progression any more:** 0 such answers in the nine runs at 600 puzzles,
  where the shared model gave 33 of 33 and the doubled model 139 to 150 of 152.

**The probe** (`analysis/twins_probe.py`, job 12049249, `log/twins_probe_12049249.out`; 30 unseen puzzles). The largest
difference per feature between the mean `rd` of a midpoint row and of a Progression row:

| runs | midpoint exact | `rd`: midpoint row against Progression row | spread among midpoint rows |
|---|---|---|---|
| shared with the steps (6) | 54.9 to 99.4 | 0.87 to 1.9 | 1.3 to 1.6 |
| doubled with the steps (6) | 86.1 to 99.4 | 1.7 to 2.0 | 1.1 to 1.5 |
| scaled with the steps (6) | 100.0 | 1.3 to 1.7 | below 5e-16 |
| for comparison, 22 of the 24 runs without the steps | fail | below 0.0001 | – |

The row descriptor now separates the two rules in every run. In the scaled model every midpoint row still has one
descriptor, as it must.

**The symmetries** (`log/check_steps_12049210.out`, random weights, 100 puzzles). Scaled with the steps: every scaling
and shift tested, at most 3.6e-17, in 100 of 100 puzzles. Doubled with the steps: the shift 3e-17, doubling 3e-17 in
the puzzles whose first steps stay in range, tripling 1.7e-2. Shared with the steps: the shift 3e-17, every scaling
2e-2 to 3e-2. Unchanged from the models without the steps.

**The predictions.**

| prediction | outcome |
|---|---|
| scaled with the steps: 100 at every seed and both sizes | **met:** 100.0 in six runs of six |
| doubled with the steps, 600 puzzles: more than half of the 152 rows right at every seed, total above 67.5 | **met:** 152 / 151 / 152; 99.4 / 99.0 / 99.2 |
| shared with the steps, 600 puzzles: some of the 33 rows right at two seeds or more | **met,** by more than was predicted: 33 / 33 / 33 |
| the probe: `rd` differs by more than 0.1 in all 18 runs | **met:** 0.87 at the least |
| unseen first steps, 100 puzzles: doubled none of the 29 and more than 39 to 50 of the 72; scaled all 101 | **met:** 0 of 29, 64 / 69 / 67 of 72; 101 of 101 |
| unseen first steps, 100 puzzles: shared still 0 of 101 | **missed by one row:** 0 / 1 / 0, the rate of guessing |
| the symmetries unchanged | **met** |

**What this shows.** The accuracy lost in "The scaling kernel" was the row descriptor's and not the symmetry's. With a
descriptor that carries the steps, every rung of the ladder is above the one below it, as the count says:

| midpoint, exact %, 3 seeds | weights per value | shared | doubled | scaled |
|---|---|---|---|---|
| orbits left to learn | 1,250 rows | 50 | 26 | 1 |
| counting learner, 100 / 600 puzzles | 5.0 / 21.6 | 80.0 / 99.8 | 94.3 / 99.8 | 100 / 100 |
| trained, with the steps, 100 / 600 puzzles | 5.0 ± 0.5 / 18.0 ± 0.9 (no steps) | 59.9 ± 4.5 / 98.3 ± 1.9 | 89.2 ± 2.7 / 99.2 ± 0.2 | 100.0 / 100.0 |

The failure in "The rule gate" had one cause in both models, and one change removes it in both: the shared model's
33 rows and the scaled model's two outcomes.

**What it does not show.**
- **Why the pooled descriptor collapses in training.** With random weights it differs between the two rules by 0.04
  to 0.15 ("Wired in" above); after training it was below 0.0001 in 22 runs. That was not studied.
- **That the shared model reaches its count at 100 puzzles.** It stays 20 points under (59.9 against 80.0), and the
  steps do not move its total there. The gap is in rows whose first step training had: 277 to 319 of 404 right.
- **Anything on sum.** The models with the steps were run on midpoint only. That sum stays at its ceiling in them is
  expected and not measured.
- **More than three seeds show.** The scaled result has no spread at all; the others have standard deviations of 0.2
  to 4.5 points.

## Plus and minus apart

Is Arithmetic hard because one model must learn two operators? Along a row of 10, plus is last = sum of the others and minus
is last = first − sum of the middle. Column roles (288k weights) trained on the 3×10 puzzles whose Arithmetic uses only one
operator (`data_prep/arith_op_data.py`; a puzzle with a minus attribute leaves the plus set even if its other attribute uses plus),
against the two-operator checkpoint on the same unseen puzzles. Exact value on Arithmetic, epoch picked by matched.

| trained on | training puzzles | plus attributes | minus attributes | plus exact (751 puzzles) | minus exact (769 puzzles) |
|---|---|---|---|---|---|
| both operators | 6000 | 1501 | 1477 | **49.5** (of 206) | 29.8 (of 228) |
| plus only | 4622 | 1347 | 0 | 35.0 | – |
| minus only | 4588 | 0 | 1323 | – | **34.2** |

- **One operator is not easier.** Alone, both reach about 35. Together, plus gains 14.5 points and minus loses 4.4.
- **Minus puzzles help plus.** The plus-only model saw nearly as many plus attributes (1347 against 1501) but no minus, and
  lost 14.5 points (about 4 standard errors on 206). Our inference: the model reads minus rows as sums too (minus puts the
  sum in the first column), so they add training data for plus, while minus is pulled towards plus. Not separated: the
  plus-only set also has 1378 fewer puzzles of the other rules, and its Progression fell from 94.2 to 91.1.
- **The minus gain is within noise** (1.4 standard errors on 228). One seed each.
- **Minus needs the zero as much as plus** (checked 2026-11-05 on every complete row of puzzles 0–9999). Shift every
  value of a row by 1 or 2: Constant and Progression rows stay valid in 100%, plus and minus rows in 0%, on 3×3 and 3×10.
  For example, 5, 2, 3 (5 = 2 + 3) becomes 6, 3, 4, and 6 ≠ 3 + 4. A − b = c is b + c = a: the difference a − b stays
  put under the shift while c moves, so a step is equated with a value, and that needs the zero. A shift-invariant model
  would merge a valid minus row with an invalid one. What collapses Arithmetic rows correctly is to shift the zero with
  the values (6 − 1 = (3 − 1) + (4 − 1)), which is `--find-zero`.

## Numbers as objects: `plus` used by learned rules (2026-11-05/06)

**The idea** (the supervisor, meeting of 5 November; plan M4). Numbers live in the same relational domain as the
objects:
- an object's attribute is an edge to a number node;
- the number nodes know `next` and `plus` (1 + 2 = 3) as background knowledge;
- messages travel as in a GNN: from a circle to its size 1, through `plus` with 2 to 3, and back to an object of
  size 3.

Arithmetic then needs no addition table learned from codes. `plus` joins three nodes at once, a hyperedge, which a GNN
on pairwise edges does not have.

**Why this is a new test.** The value axis (notes §27) already has the parts: values as constants, the hop
`x(P, A, V)`, the offsets (`off`, which include next), and the `plus` facts. But its Arithmetic 100 comes from six
rules with no weights that are the rule itself:
- the running sum along a row (`acc`, `accm`);
- the check that a complete row adds up (`pchk`, `mchk`);
- the answers (`sa`, `ma`).

`value_axis.py --learned-plus` removes those six and keeps the `plus` facts, so learned rules have to use them.

### What the model does
Everything else of the value axis stays: the convolution along the values, the counts, and the scoring per value.
In their place, a state walks along each row, per attribute, on the number nodes (15,841 weights at width 32):

```
st(P, A, V)  :- col(P, first), x(P, A, V)                                     start at the first panel's value
st(Q, A, V)  :- next(P, Q), st(P, A, U), x(Q, A, W), plus(U, W, V)            add the next panel's value W
st(Q, A, V)  :- next(P, Q), st(P, A, U), x(Q, A, W), plus(V, W, U)            take W away
ck(Rw, A)    :- full(Rw), col(L, last), next(P, L), st(P, A, V), x(L, A, V)   a complete row: the state meets the last panel
ctx(A)       :- full(Rw), ck(Rw, A)                                           ... which tells the model the rule
o(A, V)      :- st(p_qlast, A, V)                                             the question row: the state scores value V
```

1. **Start.** At a row's first panel the state sits on that panel's value, as a learned 32-vector.
2. **Step.** From panel P to the next panel Q, whose value is W, the state at U moves through `plus` in two ways, each
   with its own 32 × 32 matrix and a tanh:
   - to U + W (adding);
   - to U − W (taking away).

   The same two matrices serve every value and every step: one rule for the whole number line, like the row fold's one
   rule for the whole row.
3. **Back to a panel.**
   - In a complete row, the state before the last panel is read at the last panel's value (`ck`, its own matrix), and
     this feeds `ctx`, the model's summary of which rule the rows follow.
   - In the question row, the state at the last context panel is read at every value V and feeds the score of V for
     the missing panel.

Nothing says "the last value is the sum of the others". The model has to learn:
- that the check matters;
- which path, adding or taking away, the complete rows follow;
- how to turn the question row's states into the answer.

`x`, `next`, `col` and `plus` are hidden atoms: they only bind variables and carry no value into the network.

**Variant `--state-next`** (20,961 weights). The state can also move along `next` at each step, by −2..+2 on the
number line, one 32 × 32 matrix per offset. Many more values then carry a state (5.0 of 11 on 3×3, 43.3 of 51 on 3×10,
against 1.4 and 5.7 without), and the weights have to prefer the `plus` steps to the `next` steps.

### An example
Puzzle 6300, colour, 3×3 with values 0–10, the plus form. The question row is 1, 1, ?, and the answer is 2. With
`--state-next` the state at the second panel exists at four values:

| value | reached by | length of the state | P(value) |
|---|---|---|---|
| 0 | taking away, 1 − 1 | 3.59 | 0.000 |
| 1 | a `next` step | 4.85 | 0.000 |
| **2** | **adding, 1 + 1** | 3.79 | **1.000** |
| 3 | a `next` step | 3.30 | 0.000 |

The answer's state is not the longest. The model picks it by its direction.

### Results
I-RAVEN-X 3×3, values 0–10, 6000 training puzzles, 60 epochs, the epoch picked on matched (selection puzzles
6000–6299), unseen puzzles 6300–7299. Exact is the argmax of the value distribution.

| value axis | weights | matched (exact) | Arithmetic exact | trained on 3×3, scored on 3×10, values 0–50: matched (exact), Arithmetic |
|---|---|---|---|---|
| without `plus` (job 11932568) | 10.7k | 94.4 (86.3) | 70.7 | 31.1 (5.7), 4.5 |
| **`plus` used by learned rules** (job 11962271, epoch 10) | 15.8k | 99.6 (99.4) | **98.8** | 57.7 (32.3), 5.9 |
| **the same, the state also along `next`** (job 11962272, epoch 25) | 21.0k | **100.0** (**99.9**) | **99.8** | 24.5 (1.0), 2.8 |
| the rule given (job 11926965) | 10.8k | 99.8 (99.7) | 99.4 | 65.2 (46.2), 76.7 |

Plus and minus puzzles apart: 99.1 and 97.8 (`plus` steps only), 100 and 99.3 (also along `next`).

### How it worked
Checked with `analysis/learned_plus_check.py` on 1000 unseen puzzles: 484 Arithmetic attributes, 349 plus and 135 minus.
It reads the state at the question row's last context panel, at every value where it exists.

1. **Which values carry a state is fixed by the logic, before any weight acts.** The answer is always among them:
   1.4 values with `plus` steps only, 5.0 with `next` steps too. The grounding computes U + W and U − W exactly; the
   weights never do arithmetic.
2. **The state's direction records how it got there.** Group the states by path (adding at every step, taking away at
   every step, or with a `next` step) and put each state with the nearest average direction (cosine):

   | weights | adding | taking away | `next` | cosine, adding vs `next` |
   |---|---|---|---|---|
   | trained | 100% | 100% | 79.4% | −0.01 |
   | untrained (the control) | 100% | 100% | 82.2% | +0.33 |

   The labels are there before training: each kind of step multiplies by its own matrix, so the paths point in
   different directions. Training pushed adding and `next` apart, to right angles.
3. **Training learned to read the labels.** P(answer) on Arithmetic:

   | | untrained | trained |
   |---|---|---|
   | `plus` steps only | 0.10 (argmax right 45.9%) | 0.986 (98.8%) |
   | also along `next` | 0.10 (argmax right 11.8%) | 0.998 (99.8%) |

   The read-out combines two things:
   - the complete rows say adding or taking away (through `ck` and `ctx`);
   - the question row's states say which value got there by that path.

   Untrained weights already pick a reached value 45.9% of the time when only 1.4 values are reached; with `next`
   steps they are near chance.
4. **Length is not the signal.** The answer's state is the longest of those reached in 25% of cases with `next`
   steps and 82.6% without.

### On 3×10: eight steps (2026-11-06)
Trained on I-RAVEN-X 3×10, values 0–50, the same settings. The question row has nine context panels, so the state
takes eight steps.

| value axis, trained on 3×10 | weights | matched (exact) | Arithmetic exact | values 0–100: matched (exact), Arithmetic | rows of 5 / 20: matched, Arithmetic |
|---|---|---|---|---|---|
| without `plus` | 10.7k | 84.0 (59.6) | 11.1 | 78.9 (60.5) | 57.0 / 80.1 |
| **`plus` used by learned rules** (job 11962269, epoch 5) | 15.8k | **100.0** (**100.0**) | **100** | 99.2 (99.0), 100 | 63.8 / 71.2; 12.7 / 44.8 |
| **the same, the state also along `next`** (job 11962270, epoch 15) | 21.0k | 99.3 (94.4) | **88.7** | 98.4 (92.9), 84.9 | 61.9 / 74.7; 1.8 / 1.3 |
| the rule given | 10.8k | 100.0 (100.0) | 100 | 100.0 (100.0) | 89.8 / 98.6; – / 100 |

- **`plus` steps only: 100 from epoch 5 on, as predicted.** Here the grounding nearly gives the rule. The check atom
  `ck` exists in every Arithmetic row but in only 4.4% of Progression rows and 1.3% of Distribute rows, and the
  question row's state exists at 5.7 values, always including the answer. The weights only have to learn that `ck`
  marks Arithmetic, and which path to read.
- **Also along `next`: 88.7, the real test.** The state now exists at 43.3 of 51 values, so neither its existence nor
  `ck`'s says much, and the weights have to keep the `plus` path apart from the `next` paths over eight steps. They
  do on most Arithmetic attributes, but not stably: unseen Arithmetic moves between 81.2 and 97.0 across epochs, and
  selection (on matched, 300 puzzles) picked epoch 15.
- **Wider values carry over** (Arithmetic 100 and 84.9 at 0–100): the same two matrices serve every value.
- **Other row lengths do not.** Arithmetic falls to 12.7 / 44.8 (rows of 5 / 20) and 1.8 / 1.3; the rule given keeps
  100 at rows of 20. Probably because the read-out learned what a path looks like after eight steps through tanh,
  and at another length the states have passed a different number of matrices. Not checked inside the checkpoints.

### What it shows, and what it does not
- **Given `plus` as background knowledge, learned rules use it.** One step through `plus` is learned almost perfectly
  from 6000 puzzles, also when other moves along the number line compete. Without `plus` the same model gets 70.7.
- **The learned part is the reading, not the addition.** The grounding adds exactly, and the separate matrices label
  the paths by construction. This is the division of labour the supervisor described: arithmetic from background
  knowledge, its use learned.
- **3×3 is a single step:** c3 = c1 + c2 or c3 = c1 − c2. Trained on 3×3, neither variant carries over to rows of 10
  (Arithmetic 5.9 and 2.8, against 76.7 for the rule given): the learned step does not stay an exact walk over 8 steps.
  With `next` steps even Constant falls (51.1).
- **Trained on 3×10, eight steps are learned too:** 100 with `plus` steps only, where the grounding nearly gives the
  rule, and 88.7 when `next` steps compete. Without `plus` the same model gets 11.1. The walk is learned for the
  trained row length only.
- One seed each.

### Addition from the zero and `next` (Peano), 2026-11-06
Every value is one more than the one before, starting at 0. That is enough for all addition of non-negative integers.
`value_axis.py --peano` replaces the `plus` table with exactly that knowledge. At values 0–50, the table's 1,326 facts
become:
- the zero, `v0`;
- `next` on the values, `succ(v_i, v_i+1)`: 50 facts;
- two rules with no weights, Peano's definition of addition:

```
plus(U, v0, U)   :- val(U).                                     u + 0 = u
plus(U, W2, V2)  :- plus(U, W, V), succ(W, W2), succ(V, V2).    u + (w + 1) = (u + w) + 1
```

The second rule is the symmetry addition does have: move one operand up by one, and the sum moves up by one, while
the other operand stays put. The convolution's symmetry moves every value together, and a + b = c does not survive
that. The first rule anchors the recursion at the zero, the one value the template names. The grounder unrolls it
like the row fold, along the numbers instead of along the row, to any range.

**Check: it derives exactly the table.** The same trained checkpoint was loaded into both templates and run on the
same unseen puzzles (300 for 3×3, 100 for 3×10):

| checkpoint | all three values right: table / zero + `next` | largest output difference | grounding time, table → zero + `next` |
|---|---|---|---|
| 3×3, the rule given | 99.7 / 99.7 | 2.2e-16 | 4 s → 5 s |
| 3×3, `plus` used by learned rules | 99.7 / 99.7 | 2.2e-16 | 3 s → 5 s |
| 3×3, the same, also along `next` | 100.0 / 100.0 | 2.2e-16 | 3 s → 6 s |
| 3×10, values 0–50, the rule given | 100.0 / 100.0 | 1.0e-17 | 25 s → 555 s |

So every model here that uses `plus` behaves the same when its only arithmetic knowledge is the zero and `next`.

**Cost.** The grounder derives the sums again for every puzzle: 1.5–2× slower on 3×3, 22× on 3×10 (5.5 s a puzzle,
about 11 hours for the 7300 puzzles of a run). The networks are identical, so the 3×10 runs keep the table.

**Training from scratch on 3×3** with `--peano`, the same settings:

| zero + `next` | matched (exact) | Arithmetic exact | trained on 3×3, scored on 3×10: matched (exact), Arithmetic |
|---|---|---|---|
| `plus` steps only (job 11962979, epoch 15) | 99.9 (99.8) | 99.6 | 54.5 (27.4), 23.5 |
| also along `next` (job 11962980, epoch 5) | 99.8 (99.5) | 100.0 | 17.5 (0.2), 3.2 |

Level with the table runs (98.8 and 99.8), as it must be: for the same weights the outputs are the same. The runs are
not seeded, so the gaps (99.6 against 98.8, and 23.5 against 5.9 on 3×10) are between two runs of one network, not
an effect of the zero and `next`.

**What it shows:** the knowledge arithmetic needs is small (a zero, `next` and one recursive rule) and fits in a
lifted template. **What it does not:** the recursion is written in, not learned. Weights cannot change which atoms
exist, so a model that found u + (w + 1) = (u + w) + 1 by itself would need rule learning, not weight learning.

## Does it grok?

**Arithmetic-only training is a diagnostic, not a benchmark result** (2026-11-08). Every attribute of every puzzle,
in training and in the test, follows Arithmetic, so the model is told the rule and no other rule shares its weights.
These runs ask whether a model can learn the sum at all. How a model does on Arithmetic is judged only on the
mixed-rule puzzles, where it must also recognise the rule (`docs/RESULTS.md` §1 and §5).

Does Arithmetic on 3×10 fall to much more data and weight decay? 100,000 puzzles in which every attribute follows Arithmetic
(`data_prep/arith_big_data.py`), with 300 selection and 1000 unseen puzzles of the same kind. Grokking would show as Arithmetic exact
high on training puzzles long before it rises on unseen ones.

| model | weights | weight decay | job | Arithmetic exact, train / unseen |
|---|---|---|---|---|
| column roles, wide | 1,390k | 0 → 0.1, per puzzle | 11941252 | 51.6 / 52.4 at 40,000 puzzles, 38.7 / 38.8 at 120,000; cancelled |
| attribute GNN B, 16×, no dropout | 25.6M | 0.1 at a constant learning rate, per batch of 64 | 11942529 | 99.2 / **99.3** (epoch 45 of 115); cancelled |

**No grokking: train and unseen never part.** At every evaluation of both models the two agree within 6 points, so there is
no memorised stage for a late generalisation to follow. With 100,000 Arithmetic-only puzzles GNN B learns the sum: 42.3 at
epoch 5, 98.1 at epoch 10 (a million puzzles seen), 99.3 at its best. The same architecture at 1× gets 22.7 on the 6000
mixed-rule puzzles (`RESULTS.md` §5), so once it is told the rule, data and width are enough to learn the sum, with no
symmetry added. That says nothing yet about Arithmetic among the other rules.

**The GNN's evaluations swing because the weights spike, not because of dropout.** Unseen exact by epoch: 98.1, 5.9, 81.6,
73.6, 62.7, 66.0, 64.3, 99.3, 98.0, ... 99.5, 69.5, 97.5, 73.3, 99.2, 43.8 (epochs 10 to 115). At epoch 15 the snapshot's error
on training puzzles was 0.025 against a mean training loss of 0.0018 over that epoch, so the weights jump between good and bad
points under a constant learning rate of 1e-3 and Adam's β2 of 0.999; selection keeps the best. The dropout 0.3 runs had the
same swings (decay 0.1: 30, 71, 30) and then sank for good: by epoch 30, exact 0.1 at decay 0.1, 0.0 at 0.25 (training loss
rising, 0.0085 to 0.0133) and 0.0 at 0.5 by epoch 20; cancelled. Decay 1 with dropout collapsed first (loss 0.022 to 0.059 by
epoch 25). 64× (102M weights) does not start, loss flat at 0.170 for 20 epochs.

**The facet model peaked early and then lost to the decay.** After 40,000 puzzles it had 52.4; from 60,000 on it held at 37 to
43 while the weights' rms fell from 0.046 to 0.030 and the unseen error rose from 0.0109 to 0.0128. Decay is applied per
puzzle, like AdamW at batch 1, so its pull against the updates is AdamW's at 0.1, but over 64 times as many steps per epoch.

### Small training sets

Grokking needs a training set the model can memorise. GNN B at 16× on the first N of the 100,000 puzzles, 300,000 steps of 64
puzzles for every N, an evaluation every 2000 steps (`jobs/baselines/submit_grok_small.sh`).

**The grokking papers' setting does not fit.** Weight decay 1, β2 0.98, learning rate 1e-3 held constant, no dropout: at every
N the training loss fell for about 6000 steps and then rose, and train and unseen stayed equal, so nothing was memorised.
Cancelled after 12,000 steps.

| N | best: train / unseen (step) | at 12,000 steps: train / unseen, training loss |
|---|---|---|
| 1000 | 51.6 / 46.3 (4000) | 12.6 / 12.7, 0.0222 |
| 3000 | 19.0 / 20.1 (8000) | 13.5 / 13.1, 0.0273 |
| 10000 | 26.5 / 27.1 (6000) | 17.9 / 19.3, 0.0154 |

A likely reason, our inference: GNN B puts a LayerNorm after every layer, so most weights only set directions, and decay that
shrinks them makes every step turn them further. At learning rate times decay 1e-3 the turns are too large to fit (van
Laarhoven 2017, "L2 regularization versus batch and weight normalization", from memory, not reread). The swings are another
matter: the run without decay below has them too (step 12,000), so they come from Adam at a constant 1e-3, not from decay.

**With the 100,000-puzzle run's recipe** (decay 0.1, β2 0.999), only N changing, and a control without decay, GNN B learns
Arithmetic from 1000 puzzles, with no decay at all, and never memorises first. Arithmetic exact, train / unseen, every 2000
steps (thousands):

| N | weight decay | job | curve | best (step) |
|---|---|---|---|---|
| 1000 | 0 | 11950968 | 2: 37/36, 4: 76/72, 6: 99/97, 10: 100/98, 12: 57/55, 14: 100/98, 18: 99/98 | 99.8 / **98.3** (16,000) |
| 1000 | 0.1 | 11950969 | 2: 26/26, 4: 65/59, 8: 95/90, 10: 100/97, 14: 99/92, 18: 100/96 | 99.7 / 97.0 (10,000) |
| 3000 | 0.1 | 11950970 | 2: 33/33, 4: 47/46, 6: 86/85, 8: 42/42, 10: 95/94, 12: 99/99 | 99.1 / **99.0** (12,000) |

Cancelled after 12,000 to 18,000 steps. Train runs ahead of unseen by at most 6 points, so there is no memorised stage here
either. 1000 puzzles hold 3000 Arithmetic attributes (type, size and colour all follow it), as many as the 6000 mixed-rule
puzzles hold (1501 plus, 1477 minus), on which GNN B at 1× with dropout 0.3 and decay 0.5 gets 22.7. That gap is not
explained: the rules are mixed there, the model is 16 times smaller, and dropout 0.3 is what sank the 100,000-puzzle runs.

**Not the size** (2026-11-08). GNN B at 1× (1.6M weights) on the same first 1000 Arithmetic-only puzzles, no decay, the
same recipe (job 11978018): unseen exact 98.6 / 97.5 / 98.0 per attribute (type, size, colour), 99.5 matched, train 100
(epoch 1750 of 2500, picked on selection). Train and unseen rose together, with two collapses and recoveries on the way
(epochs 625–750 and 1375), so this is no grokking either. So the gap to 22.7 is the mixing of rules or the earlier run's
regularisation. GNN B at 1× on the mixed-rule
puzzles with this recipe (no dropout, no decay, constant learning rate 1e-3, 40,000 steps, epoch picked by matched; seeds 0
and 1, jobs 11978453–54) separates the two: Arithmetic exact **45.1 / 54.0** at the epoch picked by matched, 50.6 / 49.2 at
the last epoch, with the whole puzzle at 99.2 (63.9) / 99.3 (61.5). So the 22.7 came from dropout and decay, not from the
mixing. Mixing still costs: 98.0 when every attribute is Arithmetic, about 50 among the other rules. This is also the
best Arithmetic of any model here that has to learn the sum on 3×10 (column roles: 40.5), and the other torch baselines
have not been rerun with this recipe (RESULTS §12).

Smaller still, run to the end (300,000 steps), Arithmetic exact, train / unseen:

| N | weight decay | job | curve | best (step) |
|---|---|---|---|---|
| 100 | 0.1 | 11950991 | 100 / 6 at 2000 steps, 7 to 9 to 200,000, 10 to 12 after 250,000 | 100 / 10.4 (258,000) |
| 300 | 0.1 | 11950992 | 100 / 62 at 22,000 steps, then down: 44 at 52,000, 31 at 102,000, 24 to 30 after 250,000 | 100 / **61.7** (16,000) |
| 100 | 0 | 11950993 | 100 / 6 from 2000 steps to 300,000 | 100 / 6.4 |

**No grokking in 300,000 steps.** Train reaches 100 by 2000 steps on 100 puzzles and by 22,000 on 300, and decay does not
bring unseen up afterwards. On 100 puzzles it drifts from 7 to about 11 over the last 50,000 steps, against a flat 6 without decay; on 300
puzzles generalisation is lost instead, from 62 to under 30, while train stays at 100. The decay was not too short: learning
rate times decay is 1e-4 per step, 30 e-folds over the run. Training spikes stay (train 14 at 242,000 steps on 100 puzzles,
11 at 72,000 on 300). At 300 puzzles it reaches 62 and then loses it, at 1000 it reaches 98 at once: at this learning rate and decay
there is no size with a memorise-then-generalise stage.

### What makes grokking possible (the literature, read 2026-11-08)
Checked against the papers' text (arXiv HTML versions):
- **Weight decay** has the largest effect in the original paper: "adding weight decay has a very large effect on data
  efficiency, more than halving the amount of samples needed" (Power et al. 2022, AdamW, weight decay 1, learning rate
  1e-3, β2 0.98, minibatch 512 or half the training set, 10⁵ updates). Nanda et al. 2023 (full batch, AdamW, weight
  decay 1, 30% of the data, 40,000 epochs) see three phases, memorisation, circuit formation and cleanup, where "weight
  decay removes the memorization components", and no grokking without regularisation.
- **A memorising and a generalising solution must both exist, the generalising one slower but more efficient** (larger
  output at the same weight norm), so decay favours it in the end. Below a critical dataset size memorising stays more
  efficient: then no grokking, or "ungrokking", where a model that generalised falls back (Varma et al. 2023). Above
  it, train and test rise together. The critical size does not depend on the decay's strength; how long it takes does.
- **Large initial weights.** Omnigrok (Liu, Michaud, Tegmark 2023) induces grokking on MNIST, an MLP with MSE loss
  (1000 training samples, AdamW, initial weights multiplied by α = 9), and explains it by the training loss being flat
  in the weight norm while the test loss has a minimum ("LU mechanism"); holding the norm fixed "almost eliminat[es]
  grokking". Kumar et al. 2024 get grokking without any decay from a network that starts in the lazy regime and later
  learns features; a small output scale removes it.
- **Not needed in every setting:** Prieto et al. 2025 trace grokking without regularisation to floating-point limits
  of softmax cross-entropy, which does not apply to an MSE loss. Grokfast (Lee et al. 2024) speeds grokking up by
  amplifying the slow part of the gradient, together with weight decay.

So the setting most likely to show it here is: a training set small enough to memorise but not too small, decoupled
weight decay of about 1, many updates, and optionally larger initial weights, with an MSE loss like ours.

### The row fold in that setting (2026-11-08)
Arithmetic-only, as above: a diagnostic of whether the fold can learn the sum, not a benchmark result.
The recursive row fold (`--columns fold`, 305k weights on 3×10, notes §25) gets Arithmetic exact 36.2 on the mixed 3×10 data.
Here it trains on Arithmetic-only 3×10 puzzles (values 0–50, every attribute Arithmetic), with `core/pynl/facet_stream.py`:
one update per puzzle, Adam at learning rate 3e-4, decoupled weight decay applied every 100 puzzles, a small training
set grounded once and kept, and `--init-scale` multiplying every initial weight. Scored every 5 epochs (10 at 300
puzzles) on 300 training puzzles and 1000 unseen ones (`jobs/pynl/submit_grok_fold.sh`).

| run | training puzzles | weight decay | initial weights | updates | job | Arithmetic exact, train / unseen |
|---|---|---|---|---|---|---|
| control | 1000 | 0 | ×1 | 400,000 | 11974980 | best 90.7 / **63.4** (epoch 110); 97.9 / 47.8 at the end |
| decay | 1000 | 1 | ×1 | 400,000 | 11974981 | best 25.7 / 24.6; train never above 26.6 |
| decay, large start | 1000 | 1 | ×3 | 400,000 | 11974982 | best 23.9 / 24.5; train never above 25.2 |
| decay, smaller set | 300 | 1 | ×1 | 300,000 | 11974983 | best 31.7 / 29.0; train never above 34.2 |
| gentler decay | 1000 | 0.03 | ×1 | 600,000 | 11978016 | best 86.1 / **65.7** (epoch 125); last 100 epochs about 86 / 43 |
| gentlest decay | 1000 | 0.01 | ×1 | 600,000 | 11978017 | best 81.1 / 58.7 (epoch 95); last 100 epochs about 97 / 46 |

Learning rate times decay is 3e-4 per update at decay 1, so weights that no gradient holds shrink e-fold every 3,300
updates. Grokking would show as train near 100 long before unseen rises.

**No grokking in the first round, for two different reasons.**
- **Without decay the fold memorises and stays there.** Train climbs to 95 by 155,000 updates and 98 by 300,000;
  unseen peaks at 63.4 at 110,000 and then sinks to 44–53, while the unseen error rises (0.0102 → 0.0141) and the
  weights grow (rms 0.104 → 0.206). That is the memorising stage the grokking papers start from, with no late turn in
  400,000 updates. Note that 63.4 is well above the 36.2 the fold gets on the 6000 mixed-rule puzzles, which hold as
  many Arithmetic attributes (2978 against 3000 here): Arithmetic with no other rules beside it is easier for it.
- **With decay 1 it never memorises.** Within 10 epochs the weights fall from rms 0.10 to 0.02, and train and unseen
  stay together at 10–25 for the whole run, with or without the larger start (the ×3 start is pulled in by epoch 10)
  and at 300 puzzles too. One update per puzzle means 1000 decay steps an epoch, and the Adam noise each step adds
  balances the decay at about √(lr / 2·wd) ≈ 0.012 per weight, far below what fitting needs. The papers' decay of 1
  is for full or large batches, about a hundred times fewer steps per epoch than here.

So the grid missed the window between them. The second round (decay 0.03 and 0.01, 600 epochs) keeps the equilibrium
near 0.07–0.12, where the no-decay run memorised, while still pulling for 2 to 7 e-folds over the run.
