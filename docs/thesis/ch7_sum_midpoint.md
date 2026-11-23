# Chapter 7. One model, two rules: sum and midpoint (draft, 2026-11-17)

Cells are exact % on the tested rule, on unseen puzzles, about 500 rows per cell. Sum and midpoint at 100 and 600
puzzles have three seeds for the shared, per-value and `plus` models, and so do the models of Section 7.7; every other
cell is one run, so roughly ±4 points.

Chapter 6 found that a symmetry saves data when the data has it and caps the model when it conflicts. Those were
different symmetries in different models. This chapter holds the model fixed and changes only the rule, with two rules
built to be alike in everything but one property. The one built-in symmetry then saves most of the data on one rule
and caps the model on the other, and both amounts are computed before training.

The question came from the benchmark. Arithmetic is the rule every model that learns from data finds hardest on
I-RAVEN-X (Section 5.7). Is that because a sum is a hard function, because its table is large, or because the models
lack the right symmetry for it?

## 7.1 Two rules of one form

Take a row (a, b, c) of values between 0 and 50.

| | sum | midpoint |
|---|---|---|
| rule | c = a + b | c = (a + b) / 2 |
| rows in the table | 1,225 (a, b ≥ 1) | 1,250 (a ≠ b, a + b even) |
| exchange of a and b keeps the row valid | yes | yes |
| adding s to all three keeps the row valid | **no** | **yes** |

Both are rules of the form c = pa + qb. Both have a table of about 1,250 rows, and both are symmetric in their
operands. They differ in one algebraic property. Adding the same number s to every value turns (a, b, c) into
(a + s, b + s, c + s), and the rule c = pa + qb still holds for it exactly when p + q = 1 (Section 2.4). Midpoint has
p + q = 1, sum has p + q = 2. In words: midpoint needs only the differences between values, and sum needs to know
where zero is. Of the benchmark's rules, Constant, Progression and Distribute Three survive the common shift as well;
Arithmetic is the only one that does not.

It would be wrong to say that sum has no symmetry. The rows of any rule c = pa + qb stay valid under the moves
(s₁, s₂, p·s₁ + q·s₂): for sum, one operand may move if the answer moves with it. The two rules have equally many such
moves. What separates them is whether the common shift (s, s, s) is among them.

**The puzzles.** We generated I-RAVEN-X puzzles with rows of 3 and values 0–50 in which Arithmetic is replaced by a
test rule, and everything else is kept: the other three rules, the attributes, the splits. Two further rules guard the
comparison. *Mirror*, c = 2a − b, uses midpoint's triples with the columns moved; it keeps the common shift, and
unlike midpoint its answer lies outside its operands, as a sum's does. *Lookup*, b = g(a) and c = g(b) for a fixed
random table g, has no symmetry at all and only 51 rows.

## 7.2 The models

**Shared.** The value-axis template of Section 4.5 with a kernel over the whole range, closed into a ring so that the
ends of the range are not special, and without `plus`. It has 14,785 weights and is exactly equivariant to the common
shift: with random weights, shifting the input moves the output by the same amount to 7·10⁻¹⁷.

**Per value.** The same template with a separate copy of every rule for each value: 459,635 weights and no symmetry
on the values.

**The shared model's forward pass.** For one attribute, $x_p$ is panel $p$'s value and $v$ runs over the ring of 101
values. The kernel has one learned vector $w_k$ for each offset $k$ from −50 to 50. A row $\rho$ is read through its
pairs of neighbouring panels $(p, q)$; $(p_a, p_b)$ is the pair of the question row, and rows 1 and 2 are the complete
ones. For a row, $\pi_\rho(v)$ counts its panels at value $v$, $\delta_\rho$ is its number of distinct values, and
$\kappa_\rho$ the number of values a complete row shares with the question row. Capital letters are learned matrices,
$a_i$ and $b_i$ learned vectors, $n$ a learned vector that gives each value its score:

```math
h_p(v) &= w_{v - x_p} && \text{the kernel: value } v \text{ seen from panel } p\\
e_{p,q}(v) &= \tanh\big( M_1 h_p(v) + M_2 h_q(v) \big) && \text{a pair of neighbours, at } v\\
d_\rho &= M_4 \operatorname*{mean}_{(p,q) \in \rho} M_3 \sum_{v} e_{p,q}(v) && \text{the row descriptor}\\
c &= \tanh\Big( N_1 \operatorname*{mean}_{\rho = 1,2} d_\rho + N_2 d_3 + a_1 \operatorname*{mean}_{\rho = 1,2} \delta_\rho + a_2 \delta_3 && \text{the rule descriptor}\\
&\qquad\qquad + a_3 \operatorname*{mean}_{\rho = 1,2} \kappa_\rho + b_1 \Big)\\
c' &= \tanh(N_3 c + b_2)\\
o(v) &= \tanh\Big( N_4 c' + K_1 h_{p_b}(v) + K_2\, e_{p_a,p_b}(v) && \text{the answer at } v\\
&\qquad\qquad + a_4 \sum_{\rho = 1,2} \pi_\rho(v) + a_5\, \pi_3(v) \Big)\\
P(v) &= \operatorname{softmax}_v \; n^{\top} \tanh\big( N_5\, o(v) + b_3 \big)
```

The answer for each attribute is the distribution $P$ over the values, trained by the squared error against the one-hot
code of the true value. Nothing on the right of these equations depends on $v$ except through $v - x_p$ and the counts
at $v$, so shifting every value moves $P$ with it: that is the symmetry. It also shows Statement 5's point. The score of
$v$ reads $v$'s offsets to the two panels of the question row and a descriptor $c'$ that is the same for every $v$; no
term measures from a panel of the question row to a panel of another row. The variants change one line each. Per
value, $w$ and every matrix that meets a value are indexed by $v$ as well. With `plus` in learned rules, a state
carried along a row through the `plus` facts adds a term to $o(v)$ and to $c$. In the scaled and doubled models of
Section 7.7, the offset $v - x_p$ is counted in the row's unit. The row descriptor that reads the steps adds
$M_6 \operatorname{mean}_{(p,q)} M_5 w_{x_q - x_p}$ to $d_\rho$, under a tanh.

Two further models have no value symmetry: the facet model, and a learner that only memorises the rows it has been
asked about and is otherwise untrained.

**What these models share: a value is a symbol.** In the two value-axis models a value is a constant on an ordered
axis. In the facet model it is a code that carries the order. The memorising learner treats values as labels. None of
them is told that the tested rule is linear in its operands, and none computes with values as numbers. This is the
setting of the thesis: a template knows a constant only through the facts about it, and the question is what each of
those facts is worth. It is not the only setting. NVSA and ARLC make addition an operation of the representation
(Section 2.5), and with values read as numbers the twins need no training at all. Section 7.8 measures that and
places it at the end of the same scale.

## 7.3 Predictions

A learner that answers from the question row and is equivariant to the common shift must answer (a + s, b + s) with
its answer to (a, b) plus s. It can therefore only compute c = b + g(b − a) for some function g of the first step
b − a. All rows with the same first step form one orbit, and the learner has one number to learn per orbit
(Section 3.6, Statements 2 to 4).

The shared model sees the whole puzzle, and its symmetry is the shift of all nine panels together. That symmetry
alone would not cap sum: c₃ = b₃ + (c₁ − b₁) + (a₃ − a₁) uses only differences and is right on every sum puzzle,
because a complete sum row shows where zero is (Statement 5). The predictions below apply to the shared model
because of how it reads the puzzle. It scores a candidate value by its offsets to the two panels of the question row,
by how many panels of each row sit exactly at that value, and by a descriptor that is the same for every value. It
never measures from a panel of the question row to a panel of another row. Apart from values that coincide between
rows, it is a learner of the form above.

**Midpoint.** The answer is c = b − (b − a)/2, which has this form. The 1,250 rows fall into 50 orbits, one for each
even first step from −50 to 50 except 0. A model with the shift has 50 things to learn where a model without it has
1,250. A learner that does nothing but keep one answer per orbit, fitted on the first 100 training puzzles, has seen
32 of the 50 first steps, and these cover 80% of the unseen rows.

**Sum.** The answer is c = b + a, and a is not a function of b − a. Within an orbit every row needs a different g,
and the model has one. It can be right on one row of each orbit. Sum's 1,225 rows fall into 97 orbits, so at most
97 of 1,225 rows can be right: **7.9%**, whatever the amount of data. This is the bound of Wang et al. (2023a,
Theorem 4.3) for incorrect equivariance, evaluated on sum's table (Section 3.6.3).

**Without the symmetry** the two rules should cost the same, since they differ in nothing else. The memorising learner
confirms that the tables are equally hard to cover: it gets 4.7 / 24.1 / 90.7 on sum and 5.0 / 21.6 / 90.5 on midpoint
at 100 / 600 / 6,000 puzzles.

These are the numbers a counting learner reaches. The claim would be refuted if sharing did not help midpoint, if the
models without the symmetry found midpoint much easier than sum, or if lookup were as hard as sum for them.

## 7.4 Results

Exact % on the tested rule at 100 / 600 / 6,000 training puzzles.

| model | weights | sum | midpoint | mirror | lookup |
|---|---|---|---|---|---|
| shared, shift built in | 14,785 | 4.7 / 6.9 / 7.1 | **57.4 / 89.3 / 92.7** | 45.3 / 78.3 / 88.1 | 35.9 / 74.3 / 77.7 |
| per value | 459,635 | 4.9 / 19.3 / 69.8 | 5.7 / 17.8 / 75.2 | 4.6 / 15.7 / 63.2 | 53.0 / 97.5 / 97.5 |
| facet model (600 / 6,000) | 278,088 | 31.2 / 65.5 | 24.0 / 76.6 | 14.7 / 41.9 | 60.3 / 97.1 |
| memorising, untrained | – | 4.7 / 24.1 / 90.7 | 5.0 / 21.6 / 90.5 | 4.0 / 19.7 / 91.7 | 65.3 / 100 / 100 |
| counting learner with the shift | – | 3.6 / 6.3 / 7.3 | 80.0 / 99.8 / 100 | 70.8 / 98.0 / 100 | – |

Each cell above is one run. The cells for sum and midpoint at 100 and 600 puzzles were then repeated with three
seeds, under one rule for the epoch that was fixed beforehand: the highest whole-puzzle exact score on the selection
puzzles. Mean ± standard deviation:

| model | sum, 100 puzzles | sum, 600 | midpoint, 100 | midpoint, 600 |
|---|---|---|---|---|
| shared, shift built in | 4.5 ± 2.3 | 6.2 ± 0.8 | **60.3 ± 2.7** | **91.0 ± 3.4** |
| per value | 4.8 ± 0.6 | 19.4 ± 1.1 | 5.0 ± 0.5 | 18.0 ± 0.9 |

Figure 7.1 shows the same numbers.

![**Figure 7.1.** Exact % on sum and on midpoint against training puzzles, for the convolution along the values (the shift built in), the same model with learned use of `plus`, and weights per value. Means of three seeds at 100 and 600 puzzles, one run at 6,000. The dashed line is sum's ceiling under the shift, 7.9%, counted before training; the dotted line is the counting learner with the shift.](../figures/twins_sum_midpoint.png)

The single runs of the first table agree with these means to within 3 points. The same comparison as a two-by-two
table, with the means at 100 / 600 puzzles:

| | the model has the shift | the model has no value symmetry |
|---|---|---|
| the rule keeps the shift (midpoint) | 60.3 / 91.0 | 5.0 / 18.0 |
| the rule does not (sum) | 4.5 / 6.2, ceiling 7.9 | 4.8 / 19.4 |

**Without the symmetry the twins are equally hard.** With weights per value, sum and midpoint are within 1.4 points
of each other at 100 and 600 puzzles over three seeds, and within 6 points at 6,000, and both track the memorising
learner. The facet model has no consistent winner: sum is
ahead at 600 puzzles and midpoint at 6,000. Nothing separates the two rules for a model that cannot use the shift.

**With the shift built in they part completely.** The same 14,785 weights reach 91.0% on midpoint and 6.2% on sum
from 600 puzzles, with standard deviations of 3.4 and 0.8 points. On midpoint the shared model at 600 puzzles is ahead of weights per value at 6,000 (75.2), with 3% of
the weights and a tenth of the data. Mirror behaves like midpoint, so the gain does not come from midpoint's answers
lying between its operands.

**Sum stays at its ceiling.** 7.1% at 6,000 puzzles against the 7.9% computed in advance, and between 1.8% and
7.1% in each of the six seeded runs at 100 and 600 puzzles, at the epoch picked on the selection puzzles. The ceiling
is on the expected accuracy, and the 507 unseen sum rows fix a 7.9% rate only to about 1.2 points. Over all 172
evaluated epochs of these runs, 8 are above 7.9 and the highest is 9.7, in the run on 6,000 puzzles, whose epochs
average 7.9. The model is at the ceiling, not measurably under it. That a model which reads the question row through a
shift-equivariant kernel cannot fit a rule that does not keep the shift follows from the algebra of Section 7.3 and is
no discovery. It is a check that
the symmetry is really built in, and it shows what a mismatched symmetry costs here: 63 points against weights per
value at 6,000 puzzles, on a rule the model could otherwise learn.

**No symmetry is not what makes a rule hard.** Lookup has none, and weights per value learn it from 600 puzzles
(97.5), because it has 51 rows. Sum is expensive for a model without the right symmetry because its table is large
*and* nothing collapses it. A large table with the right symmetry is cheap (midpoint, shared), and a small table needs
none (lookup).

## 7.5 What the network generalises is what its orbits allow

The shared network reaches 60.3 / 91.0 / 92.7 on midpoint where the counting learner reaches 80.0 / 99.8 / 100. Part
of the distance has a plain cause: the counting learner is told which rule a row follows, and the network is not. A
midpoint row whose first step is ±2 looks, from the question row alone, like a Progression row. At 6,000 puzzles the
network gets none of those 33 rows right and 469 of the other 472.

The more telling comparison is on rows whose first step never occurred in a training question. The checkpoints are
those of the seeded table in Section 7.4, seeds 1 / 2 / 3.

| shared model, midpoint | first step seen in training | first step never seen |
|---|---|---|
| 100 training puzzles | 71.5 / 77.5 / 77.2% of 404 rows | **0 / 0 / 0** of 101 rows |
| 600 | 87.3 / 93.3 / 93.1% of 504 | 0 / 0 / 0 of 1 |

Not one row with an unseen first step is answered correctly, at either size or any seed. This is what the model's
parameterisation implies. The answer is linear in the first step, c = b − (b − a)/2, but the kernel has one free weight
vector per offset and nothing ties the vectors of two offsets together, so the template has no way to carry "half the
step back" from the steps it has seen to one it has not (Statement 11 holds for it nearly as it stands). The model fills
the orbits the convolution gives it, one at a time, as the counting learner does, only less efficiently. Weights per
value do the same one level down: at 100 puzzles they get 20 to 24 of the 25 rows whose pair (a, b) they have seen and
3 or 4 of the other 480. In both models, what generalises is exactly what the built-in symmetry merges. The result is
not a finding about neural networks in general, which with a smooth kernel or numbers as inputs could do otherwise. Its
use is that the count says, before training, which rows a model of this family will get.

How many training rows an orbit needs can be read from the same checkpoints. Pooled over the three seeds, an unseen
row with exactly one training question in its orbit is answered correctly in 422 of 573 cases at 100 puzzles (74%)
and in 71 of 72 at 600 (99%). With three or four training questions it is 92% at 100 puzzles. One row per orbit is
thus enough once the rest of the network has been trained, which is what the count assumes. The pairs that weights
per value get right without having seen them, 11 of 1,440 over the three seeds, are the rate of guessing among 101
values.

## 7.6 The reversal: giving the model the relation sum needs

If the argument is right, sum should become as cheap as midpoint in a model that has sum's own moves. The move
"one operand and the answer together" is what a relation `plus(U, W, V)` for U + W = V expresses when it is supplied
as background facts: (u + s) + w = v + s. We added these facts to the shared model, with learned rules that walk
through them (Section 4.5), and removed nothing else. The model has 19,937 weights. *Prediction from the counting
learner:* with sum's moves, sum is learned from the first few puzzles; midpoint keeps what the convolution gives it.

| shared model, three seeds | sum, 100 / 600 puzzles | midpoint, 100 / 600 puzzles |
|---|---|---|
| without `plus` | 4.5 ± 2.3 / 6.2 ± 0.8 | 60.3 ± 2.7 / 91.0 ± 3.4 |
| with `plus` used by learned rules | **98.2 ± 0.8 / 99.7 ± 0.4** | 63.9 ± 3.9 / 91.7 ± 1.8 |

The model that was held under 8% learns sum from 100 puzzles. Midpoint is where it was: the differences of 3.6 and
0.7 points are inside the spread of the seeds, so the added facts neither help nor hurt it.

How much this gives away has to be said plainly. A function that respects every one of sum's moves is a + b + k for a
single number k. Sum's full group of moves is thus the rule itself up to one constant, and the counting learner with
these moves has a single orbit and is right after ten puzzles. The convolution gave midpoint one of the two directions
of its group, which left 50 orbits to learn; `plus` gives sum everything, which leaves one. That is why 98.2 is above
60.3, and it does not mean sum is the easier rule. `plus` is the relation handed over as background knowledge. What the
network learns is when to use it and how to read the result. Learning sum then becomes choosing and reading a given
relation, much as in models with programmed rules, PrAE (Zhang et al., 2021) and ARLC (Camposampiero et al., 2024).
The section measures the price of that knowledge, not a way of learning arithmetic. It also has a cost in size: there is
one `plus` fact per pair of values, so their number grows with the square of the value range, and the value axis was
not run on I-RAVEN-X's range of 1,000. The general statement is this: the more of a rule's group is built in, the fewer
orbits are left to learn, and at one orbit the rule has been given.

The same switch on the benchmark's own Arithmetic, in puzzles where the four rules are mixed:

| value axis, Arithmetic exact | without `plus` | with `plus` in learned rules |
|---|---|---|
| I-RAVEN-X, rows of 3, values 0–10 | 70.7 | 98.8 |
| I-RAVEN-X, rows of 10, values 0–50 | 11.1 | 100 (88.7 when other moves along the values compete) |

Without `plus` the other three rules stay at 100% exact on rows of 10 and Arithmetic falls to 11.1%. These rows are
value-axis models compared with one another. The models of Chapter 5, which have to learn sums from codes, are in the
right-hand column of the two-by-two table: they have no symmetry on the values, and for them a large table is simply
expensive.

## 7.7 A second symmetry: scaling

Section 7.5 ended on a claim that can be tested: what generalises is what the built-in symmetry merges. If that is
right, a model with a second symmetry of midpoint must answer rows with unseen first steps, in the orbits the new
symmetry merges and in no others. Midpoint has such a symmetry. Multiply every value by the same number k and the row
stays valid: ka + kb = 2kc. In the terms of Section 2.4, the shift makes the zero of the value scale a convention, and
scaling makes its unit a convention too: together they are Stevens's interval scale, except that on the ring a scaling
may also be negative, which his interval scale, being ordered, excludes.

**The model.** The shared model's kernel has one weight per offset between two values. The new kernel counts offsets
in each row's own unit: a value V lies k units from a value U, V = U + k·S, with one weight per k. The unit S is a
constant like the values, and the template never names it. Background facts say which step is k units of S on the
model's ring of 101 values. Since 101 is prime, every step has every multiple and every fraction on the ring, half a
step included. A rule derives a row's unit from its first two panels, in one of two ways.

- **Doubled.** The unit is the largest power of two in the row's first step. The model is equivariant to doubling all
  values, and first steps that differ by a power of two fall into one orbit: 26 orbits in place of 50.
- **Scaled.** The unit is the first step itself. The model is equivariant to v → kv + s for every k and s, and all
  first steps are one orbit. It can only answer c = b + g·(b − a) for a number g it has to learn. For midpoint g is
  −1/2. As `plus` does for sum (Section 7.6), this hands over the rule up to one number.

Both have the shared model's 14,785 weights. With random weights, the output of a transformed puzzle equals the
transformed output to 2·10⁻¹⁷ on 100 puzzles, for every k and s in the scaled model and for doubling in the doubled
one wherever the doubled steps stay in range. These are exact on the ring; on the values 0 to 50 a scaling sends most
rows out of range, and Section 3.7.1 says what that leaves. The shared model has neither symmetry (a difference of 2·10⁻²).

**Predictions, made before training.** At 100 training puzzles, 101 of the 505 unseen midpoint rows have a first step
that no training question had. For 72 of them a training question had the same odd part of the step, for 29 none had.

| | shared | doubled | scaled |
|---|---|---|---|
| orbits of first steps, midpoint | 50 | 26 | 1 |
| the 72 rows: new first step, odd part seen | none right (measured) | can be right | can be right |
| the 29 rows: new first step, new odd part | none right (measured) | none right | can be right |
| ceiling on sum | 7.9% | 7.2% | 1.4% |

**Results.** Exact % on the tested rule, three seeds, the selection rule of Section 7.4.

| | shared | doubled | scaled, seeds 1 / 2 / 3 |
|---|---|---|---|
| midpoint, 100 puzzles | 60.3 ± 2.7 | 49.9 ± 5.3 | 19.0 / **100.0** / 21.0 |
| midpoint, 600 | 91.0 ± 3.4 | 67.5 ± 2.0 | 29.5 / **100.0** / 20.2 |
| sum, 100 | 4.5 ± 2.3 | 3.6 ± 0.2 | 3.2 / 0.8 / 2.0 |
| sum, 600 | 6.2 ± 0.8 | 4.1 ± 0.3 | 3.4 / 1.0 / 3.4 |

| unseen midpoint rows right at 100 puzzles, seeds 1 / 2 / 3 | shared | doubled | scaled |
|---|---|---|---|
| the 72 rows | 0 / 0 / 0 | **50 / 43 / 39** | 18 / 72 / 19 |
| the 29 rows | 0 / 0 / 0 | **0 / 0 / 0** | 4 / 29 / 4 |

The line the count drew holds at every seed. The doubled model answers more than half of the 72 rows on which the
shared model answers none, and none of the 29. The scaled model has no group at zero; at one seed it answers all 101,
and every other row, from 100 puzzles.

**What the count left out.** The totals fell, and the scaled model failed at two seeds of three. The cause is the
confusion of Section 7.5, enlarged. The benchmark's Progression steps by 1 or 2. In a model that reads rows in a
unit, a midpoint row whose first step is one unit begins exactly like a Progression row, and the network continues it
as one: it generates 2b − a. At 600 puzzles, seeds 1 / 2 / 3:

| model | midpoint rows that begin like a Progression row | right among them | right among the other rows |
|---|---|---|---|
| shared | 33 of 505 (first step ±2) | 0 / 0 / 0 | 440 / 470 / 469 of 472 |
| doubled | 152 (first step a power of two) | 1 / 11 / 1 | 330 / 340 / 340 of 353 |
| scaled | 505 (every row) | 149 / **505** / 102 | none |

In the scaled model a midpoint question row and a Progression question row are the same picture, two panels one unit
apart. Only the two complete rows above tell the rules apart. At one seed the model uses them: it is then right on all
505 rows from 100 puzzles, from the twentieth epoch on. At the other two seeds it continues four rows in five as a
Progression.

**Why two seeds fail.** Two explanations were tested on the saved runs. The first was the answer layer: it adds the
rule descriptor to the kernel's features and cannot switch a feature off by the rule. A gate from the rule descriptor
on those features (1,056 more weights, the symmetries unchanged) was predicted to bring the scaled model to 100% at
every seed and to repair the shared model's 33 rows. Both predictions failed: the gated scaled model reaches 20 to 30%
at all three seeds and both sizes, and the gated shared model is right on none of the 33 rows.

The second was measured directly. The model reads a complete row through a row descriptor: the features of each
neighbouring pair of panels, summed over all values and averaged over the row. Asked for that descriptor on unseen
puzzles, the saved networks give (largest difference per feature, on a scale of −1 to 1):

| runs | midpoint | row descriptor: a midpoint row against a Progression row |
|---|---|---|
| scaled, seed 2, 100 and 600 puzzles | 100.0, 100.0 | 1.5, 1.4 |
| scaled, seeds 1 and 3; scaled with the gate (10 runs) | 19.0 to 30.5 | below 0.0001 |
| shared, with and without the gate (12 runs) | none of the 33 rows | below 0.0001 |

In the 22 runs that fail, the descriptor is one vector whatever the row: it does not carry the row's steps, so the rule
never reaches the answer. In the two runs that succeed it separates the rules. The template allows this. Summed over
every value of the ring, the pair's features contain each kernel weight once wherever the two panels lie, so the sum
depends on their step only through the curvature of the tanh. Why training grows that part at one initialisation and
not at the others was not measured.

What the failed runs use instead is a cue that sits in the rule descriptor directly: whether the question row shares
a value with a complete row. At 600 puzzles they are right on 101 to 107 of the 108 midpoint rows with a shared value
and on 0 to 47 of the 397 without.

**The repair.** If that is the cause, a row descriptor that carries the steps must remove both failures, the scaled
model's two outcomes and the shared model's 33 rows. We let the descriptor read each neighbouring pair's step
directly: the first panel's kernel feature at the second panel's value, which is the kernel weight of exactly that
step, counted in the row's unit where rows are read in one. Nothing else changes. The model has 2,048 more weights
and the same symmetries (4·10⁻¹⁷ with random weights). *Predictions, written before the runs:* the scaled model reaches
100% at every seed; the doubled model answers more than half of its 152 rows that begin like a Progression row; the
shared model answers some of its 33; the descriptor separates the two rules in every run.

| midpoint, exact %, seeds 1 / 2 / 3 | 100 puzzles | 600 puzzles |
|---|---|---|
| shared, the descriptor reads the steps | 54.9 / 63.4 / 61.4 | 96.0 / 99.4 / 99.4 |
| doubled, the descriptor reads the steps | 90.1 / 91.3 / 86.1 | 99.4 / 99.0 / 99.2 |
| scaled, the descriptor reads the steps | **100.0 / 100.0 / 100.0** | **100.0 / 100.0 / 100.0** |

All four hold. The scaled model is right on all 505 rows in each of its six runs, from the first evaluated epoch on.
The shared model answers its 33 rows at every seed at 600 puzzles, and 17 to 27 of them at 100; in twelve runs
without the repair it had answered none. The doubled model answers 151 or 152 of its 152 at 600 puzzles and 147 to
152 at 100. In all 18 runs the
descriptor of a midpoint row differs from that of a Progression row by 0.9 to 2.0. The line the count drew among the
unseen first steps holds as before: at 100 puzzles the doubled model answers 64 to 69 of the 72 rows it may and none
of the 29 it may not, and the scaled model all 101. One further prediction was missed by one row: the shared model
was to stay at none of the 101 and is right on one, at one seed, which is the rate of guessing among 101 values.

Sum stays capped, as counted. The doubled model gets 3.6 and 4.1% against a ceiling of 7.2. The scaled model gets at
most 3.4%: at two seeds it continues every sum row by its first step and is right on exactly the 17 unseen rows with
b = 2a, which is the most that one multiple reaches on these rows.

**What this adds.** First, the claim of Section 7.5 survived a test it could have failed: the rows that became
answerable are the ones the count named, and the rows it excluded stayed at zero. Second, a symmetry acts on all the
data at once. Scaling is a symmetry of every rule here, but the generator never scales a Progression, so the size of
the first step told the rules apart, and the shared model was using it. The symmetry removed that cue together with
the differences it was meant to remove. This is the gap of Section 6.5 seen from the other side: there a symmetry
closed a shortcut that hurt on transformed puzzles, here it closed one that helped on the benchmark's own. A count over
one rule's table cannot see this; a count over the question rows of all rules would. Third, the scaling model that
succeeds learns one number, so its 100% is the rule given, not the rule found. Fourth, the lost accuracy was not the price of the symmetry. With the cue gone, the model has to read the rule from
the complete rows, and the first row descriptor did not carry it in 22 runs of 24. The shared model had the same
defect and met it on 33 rows of 505; the scaled model met it on every row. With a descriptor that reads the steps,
more of the rule's symmetry means less data at every step: 59.9, 89.2 and 100.0% from 100 puzzles for the shift,
doubling and scaling.

**Limits.** Three seeds, 200 epochs and no tuning. The ladder as printed is the second attempt: the first descriptor
failed, the cause was measured, and the repair was predicted before it was run. Why the pooled descriptor loses the
steps in training was not studied; with random weights it still separates the two rules a little. On sum the models
with the repaired descriptor stay at their ceilings, as predicted before the runs: 4.0 and 7.6, 3.4 and 5.2, 1.2 and
1.6 at 100 and 600 puzzles against 7.9, 7.2 and 1.4. The repair was to the descriptor, and on sum the limit is where
the answer is read: from the question row only. The doubled model's runs without the repair were not probed. Scaling is exact
on the ring of 101 values, not on the range 0 to 50.

## 7.8 A ladder of priors, and values read as numbers

Sections 7.6 and 7.7 ended on the same remark: the more of a rule's group is built in, the fewer orbits are left to
learn, and at one orbit the rule has been given. This section puts the models of the chapter in that order and adds
the one that is missing at the top.

**The top of the ladder.** Every model so far treats a value as a symbol (Section 7.2). Read the values as numbers
instead, and assume the form c = pa + qb. Then nothing has to be learned from other puzzles: the two complete rows of
a puzzle are two equations for p and q. We fit them by least squares for each attribute and apply them to the
question row. The fit is not told which rule an attribute follows. *Prediction:* it solves every rule of this form
and no other.

| exact %, unseen puzzles, no training | the tested rule | Progression | Constant | Distribute Three |
|---|---|---|---|---|
| sum puzzles | 99.6 | 98.7 | 100 | 0 |
| midpoint puzzles | 99.8 | 99.3 | 100 | 0 |
| mirror puzzles | 99.8 | 98.7 | 100 | 0 |
| lookup puzzles | 1.7 | 98.6 | 100 | 0 |

It does. Sum, midpoint and mirror are solved without one training puzzle, and so are Progression and Constant, which
have the same form. On the sum, midpoint and mirror puzzles, each of its 32 misses on these rules is a puzzle whose
two complete rows do not determine p and q. On lookup and Distribute Three, which are not linear, it gets 1.7% and
nothing.

**The ladder.** Midpoint at 100 training puzzles, in the order of how much is built in. The counting learner keeps
one answer per orbit and is told the rule (Section 7.3).

| built in | left to learn | counting learner | trained model |
|---|---|---|---|
| nothing about the values (weights per value) | 1,250 rows | 5.0 | 5.0 ± 0.5 |
| the shift: only differences count | 50 orbits | 80.0 | 59.9 ± 4.5 |
| the shift and doubling | 26 orbits | 94.3 | 89.2 ± 2.7 |
| the shift and scaling: only ratios of differences count | 1 orbit, one number | 100 | 100.0 / 100.0 / 100.0 |
| the linear form, values as numbers | nothing from other puzzles | – | 99.8 with no training |

The three rows with a symmetry are the models whose row descriptor reads the steps (Section 7.7). With the first
descriptor they reached 60.3 ± 2.7, 49.9 ± 5.3 and 19.0 / 100.0 / 21.0. At 600 puzzles the same four models reach
18.0, 98.3, 99.2 and 100.0.

The same for sum:

| built in | left to learn | counting learner | trained model |
|---|---|---|---|
| nothing about the values (weights per value) | 1,225 rows | 4.7 | 4.8 ± 0.6 |
| the shift, with the answer read from the question row | a ceiling of 7.9% | 3.6 | 4.5 ± 2.3 |
| addition's own moves (`plus`) | 1 orbit, one number | 100 | 98.2 ± 0.8 |
| the linear form, values as numbers | nothing from other puzzles | – | 99.6 with no training |

**What the ladder says.**

- **Each rung is a piece of knowledge about numbers.** That values lie on a line (the shift), that the line has no
  fixed unit (scaling), that values can be added (`plus`), that the rule is linear. Each has a price in training
  puzzles, and the count gives that price before training.
- **The trained models end in the count's order, at the second attempt.** The count says every rung needs less data
  than the one below it, and from 100 puzzles the networks reach 5.0, 59.9, 89.2 and 100.0%. They stay under the count
  where many orbits are left: by 20 points with the shift, by 5 with doubling, by nothing with scaling. The order held
  only once the model could read the rule from the complete rows. With the first row descriptor the two scaling rungs
  fell below the rung beneath them (Section 7.7), with the same groups and the same counts. The count bounds what a
  model of this family can reach and orders the rungs; how close a network comes depends also on how it describes a
  row, and the 20 points with the shift alone are not explained.
- **A prior and the answer are two ends of one scale.** At the top nothing is left to learn across puzzles, and at one
  orbit a single number is. What is left to learn, counted in orbits, is a unit on that scale. Chollet (2019, pp. 1 and 43) asks that skill be reported together with the priors and the experience behind it (Section 2.4). For one rule and one family
  of models, the amount of prior can here be counted.
- **The top rung has its price too.** The linear form gives nothing on Distribute Three and on lookup, as the shift
  gives nothing on sum. The worth of a prior belongs to the pair of model and rule at every rung.

Which rung a model should start from is not something the count decides. Models in which addition is an operation of
the representation start from the upper rungs (Section 2.5). A template that treats values as constants starts at the
bottom, and this chapter measured what each step up costs it.

## 7.9 What the experiment shows, and its limits

**What it shows.** A symmetry prior is not good or bad in itself. The same convolution is worth an order of magnitude
of data on midpoint and costs nearly everything on sum. Its worth belongs to the pair of model and rule, and for this
pair it was computed before training: 50 orbits in place of 1,250 rows on one side, a ceiling of 7.9% on the other.
The trained networks sat at the ceiling, came close to the saving (59.9% against the count's 74.8% at 100
puzzles, 91.0% against 98.7% at 600, and 98.3% once the row descriptor read the steps), and, as their weights per
offset imply, got no row the count did not allow. Section 7.8 puts the same statement
on one scale: from weights per value to the linear form, each piece of built-in knowledge leaves less to learn, down
to nothing.

**Objections.**
- *"Read the values as numbers and both rules are trivial."* They are: a linear fit on a puzzle's two complete rows
  solves them with no training (Section 7.8). That model is given the form of the rule. The twins test what a model
  pays that is not given it, which is the position of every template in this thesis, and of any model on a rule that
  is not linear, where the numbers do not help.
- *"A shift-equivariant model cannot fit a rule that is not shift-equivariant. That is obvious."* For a model that
  answers from the question row it is, and Section 7.4 says so. What is not obvious is that the two rules are otherwise
  equally hard, and that the size of the gain on midpoint, and which rows it covers, are given by a count.
- *"Your model sees the complete rows, and from them a shift-equivariant function can compute a sum."* It can
  (Section 7.3). The ceiling holds for this template because it never measures from one row to another, and Section
  7.3 says that this is a property of the template and not of the shift.
- *"Midpoint is not a rule of the benchmark."* It is a probe, built to isolate one property. The evidence on the
  benchmark is Section 7.6 and the value-shift ablation of Section 6.2.
- *"These are not groups on the values 0 to 50."* Correct. A shift can leave the range, midpoint needs a + b even, and
  the shared model's ring wraps around where the data does not. The orbit counts were made on the actual tables, so the
  numbers stand; the group language is exact only away from the ends of the range.

**Limits.**
- *The test range came last.* Every checkpoint of this chapter was scored once on puzzles 8000 to 9999 at the end of
  the work, with 963 midpoint and 1,014 sum rows (Appendix D.6). The seeded cells move by at most 3.2 points, each
  within the sampling error of the 500 rows reported here. Every sum mean stays at or under its ceiling within sampling
  error, and the ladder keeps its order: with the steps at 100 puzzles, 59.4, 88.2 and 100 against the counts 80.0,
  94.3 and 100. The facet model on sum at 600 puzzles moves the most, from 31.2 to 24.7, so its lead over midpoint
  there shrinks from 7 points to 3. The cells at 6,000 puzzles of the shared and per-value models cannot be tested as
  reported, because their runs saved another epoch than the one the table reads.
- Sum and midpoint at 100 and 600 puzzles have three seeds, with standard deviations up to 3.9 points. Three runs fix
  a standard deviation only to about half its size. The cells at 6,000 puzzles, mirror, lookup and the facet model are
  one run of about 500 rows, so ±4 points. The effects reported are 50 points and more; differences of a few points
  between cells mean nothing.
- The epoch is picked by exact score on the selection puzzles here, because matched is near 100% for every model on
  these puzzles and picks arbitrary epochs. The conclusion is the same under either choice. For sum in the shared
  model the choice is arbitrary under either: the whole-puzzle score does not move with a rule the model cannot
  learn, and one seed's cell at 100 puzzles (1.8%) is such an epoch.
- The runs at 6,000 puzzles with weights per value and the facet model were still improving when they stopped. The
  comparisons at 100 and 600 puzzles are the firm ones.
- The experiment is on rows of 3. That Arithmetic on rows of 10 is hard for the models of Chapter 5 for the same
  reason is an inference: their table there has billions of rows and they have no symmetry that collapses it.
- Every trained model here treats values as symbols (Section 7.2). The comparison between sum and midpoint is a
  statement about such models; with values as numbers and the linear form given, neither rule needs training.
- The ladder of Section 7.8 has three seeds per rung, and its three rungs with a symmetry use the row descriptor that
  was repaired after the first one failed. It is a second attempt, with the repair predicted before it was run. The
  first attempt shows that, for one group and one orbit count, the way a model describes a row can decide between 20%
  and 100%. The count predicts the order and the reach of the rungs, not the accuracy of a given network.
