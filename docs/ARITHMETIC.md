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
- **Noise.** One seed each except the MLP, about 500 rows per cell, so roughly ±4.
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
- **At 100 puzzles the network gets less out of the orbits it has seen** (57.4 against 80.0). Does it answer rows
  whose first step no training row had, which would mean it learned "halfway" as a law rather than one answer per
  first step? The trained checkpoints' answers are being dumped for this (job 11974989): *running*.

Weights per value follow the table learner: 5.7 / 17.8 / 75.2 against 5.0 / 21.6 / 90.5. Sum on the shared model:
4.7 / 6.9 / 7.1, at the cap.

### Giving the model sum's moves
`plus(U, W, V)` is addition's move written as a relation: (u + s) + w = v + s keeps the fact, and the Peano rule
u + (w + 1) = (u + w) + 1 is the same move with s = 1. A state that walks through `plus` (`--learned-plus`, "Numbers
as objects" above) therefore moves one operand with the answer. Prediction from the learners: with it, sum should come
from as few puzzles as midpoint does with the convolution. Runs: the shared ring model of the results above with
`--learned-plus` added (19,937 weights; the kernel stays a ring, only the `plus` facts do not wrap), on sum and
midpoint at 100 and 600 puzzles (jobs 11974990–93): *running*.

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

Does Arithmetic on 3×10 fall to much more data and weight decay? 100,000 puzzles in which every attribute follows Arithmetic
(`data_prep/arith_big_data.py`), with 300 selection and 1000 unseen puzzles of the same kind. Grokking would show as Arithmetic exact
high on training puzzles long before it rises on unseen ones.

| model | weights | weight decay | job | Arithmetic exact, train / unseen |
|---|---|---|---|---|
| column roles, wide | 1,390k | 0 → 0.1, per puzzle | 11941252 | 51.6 / 52.4 at 40,000 puzzles, 38.7 / 38.8 at 120,000; cancelled |
| attribute GNN B, 16×, no dropout | 25.6M | 0.1 at a constant learning rate, per batch of 64 | 11942529 | 99.2 / **99.3** (epoch 45 of 115); cancelled |

**No grokking: train and unseen never part.** At every evaluation of both models the two agree within 6 points, so there is
no memorised stage for a late generalisation to follow. With 100,000 puzzles GNN B learns Arithmetic outright: 42.3 at epoch 5,
98.1 at epoch 10 (a million puzzles seen), 99.3 at its best. The same architecture at 1× gets 22.7 on the 6000 mixed-rule
puzzles (`RESULTS.md` §5), so data and width are enough, with no symmetry added.

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
The recursive row fold (`--columns fold`, 305k weights on 3×10, notes §25) gets Arithmetic exact 36.2 on the mixed 3×10 data.
Here it trains on Arithmetic-only 3×10 puzzles (values 0–50, every attribute Arithmetic), with `core/pynl/facet_stream.py`:
one update per puzzle, Adam at learning rate 3e-4, decoupled weight decay applied every 100 puzzles, a small training
set grounded once and kept, and `--init-scale` multiplying every initial weight. Scored every 5 epochs (10 at 300
puzzles) on 300 training puzzles and 1000 unseen ones (`jobs/pynl/submit_grok_fold.sh`).

| run | training puzzles | weight decay | initial weights | updates | job | Arithmetic exact, train / unseen |
|---|---|---|---|---|---|---|
| control | 1000 | 0 | ×1 | 400,000 | 11974980 | *running* |
| decay | 1000 | 1 | ×1 | 400,000 | 11974981 | *running* |
| decay, large start | 1000 | 1 | ×3 | 400,000 | 11974982 | *running* |
| decay, smaller set | 300 | 1 | ×1 | 300,000 | 11974983 | *running* |

Learning rate times decay is 3e-4 per update, so weights that no gradient holds shrink e-fold every 3,300 updates.
Grokking would show as train near 100 long before unseen rises.
