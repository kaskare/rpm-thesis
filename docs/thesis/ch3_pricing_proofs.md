# Chapter 3, Sections 3.6 and 3.7: what a symmetry is worth, with proofs (draft, 2026-11-17)

Each statement below comes with why it holds, in plain words, and with the measurement that backs it. The checks that
need no training are printed by `analysis/pricing_checks.py`, `analysis/rule_orbits.py` and
`analysis/twins_linear_fit.py`; the trained numbers are from `docs/ARITHMETIC.md` and `docs/RESULTS.md` §11. The
renaming theorem itself (Section 3.2) is in `docs/PYNEURALOGIC_NOTES.md` §21 and is not repeated here. Notes for you
are at the end.

---

## 3.6 The words

A **row** is three values (a, b, c). In the **question row** the third is missing, and a **learner** is anything that
turns the two values (a, b) into an answer for c. We write f(a, b) for that answer.

A **move** changes a row in a fixed way. Three moves matter here:

- the **shift**: add the same number s to all three values;
- **scaling**: multiply all three values by the same number k;
- **one operand with the answer**: add s to a and to c, or to b and to c.

A rule **keeps** a move if every row of the rule is again a row of the rule after the move. A learner **has** a move
if moving its input moves its answer in the same way: for the shift, f(a + s, b + s) = f(a, b) + s. The **orbit** of
an input is everything the moves turn it into. Values are the whole numbers 0 to 50. Where a statement divides, it
is made on the ring of 101 values that the models use, where every step has a half and every multiple.

## 3.6.1 Which rule keeps which move

**Statement 1.** A rule of the form c = pa + qb keeps the shift exactly when p + q = 1. Every such rule keeps
scaling. Sum keeps the move of one operand with the answer, and midpoint does not.

*Why.* Shift a row of the rule by s. The new third value is c + s. The rule asks for p(a + s) + q(b + s), which is
pa + qb + (p + q)s, that is c + (p + q)s. The two agree for every s exactly when p + q = 1. Midpoint has ½ + ½,
mirror (c = 2a − b) has 2 − 1, Progression (c = 2b − a) has −1 + 2 and Constant (c = b) has 0 + 1: all keep the shift.
Sum has 1 + 1 = 2 and does not. For scaling, multiply the rule by k: kc = p(ka) + q(kb), which holds whatever p and q
are. For the last move, add s to a and to c in a sum row: c + s = (a + s) + b holds; in a midpoint row the rule asks
for c + s/2.

*Checked on the tables* (rows that are rows again after the move, of those that stay in range):

| | shift | scaling by 2, 3, 5 | one operand with the answer |
|---|---|---|---|
| sum, 1,225 rows | 0 of 30,325 | 465 of 465 | 39,200 of 40,425 |
| midpoint, 1,250 rows | 40,400 of 40,400 | 490 of 490 | 0 of 51,450 |
| mirror, 1,250 rows | 40,400 of 40,400 | 490 of 490 | 0 of 51,450 |

The 3% that sum loses under its own move are rows pushed onto a = 0, which the table leaves out.

## 3.6.2 A learner with the shift, and what it saves

**Statement 2.** A learner that has the shift can only compute c = b + g(b − a), for some list of numbers g with one
entry per first step b − a.

*Why.* Shift the input by −a. The learner must answer (0, b − a) with f(a, b) − a. So f(a, b) = a + f(0, b − a): its
answer is a plus something that depends on the first step alone. Writing g(d) for f(0, d) − d gives
f(a, b) = b + g(b − a). All inputs with the same first step form one orbit, and the learner has one free number for
each orbit.

**Statement 3 (the data saved).** If the rule keeps the shift, one training row teaches the learner every row with
the same first step. With rows drawn evenly from a table in which orbit o holds a share P(o) of the rows, the expected
share of unseen rows whose orbit n training rows have reached is

    the sum over orbits of P(o) · (1 − (1 − P(o))ⁿ).

*Why.* A rule that keeps the shift is itself of the form c = b + g*(b − a), by the argument of Statement 2 applied to
the rule. A training row with first step d shows g*(d), and that is the right entry for every row with that first
step. An unseen row is therefore answered correctly exactly when its first step occurred in training. A given orbit
is missed by one training row with chance 1 − P(o) and by n independent rows with chance (1 − P(o))ⁿ. An unseen row
lies in orbit o with chance P(o). Adding up over the orbits gives the formula. The same holds for any set of moves
the rule keeps, with its own orbits.

*Checked on midpoint.* "Expected" uses the table alone. "Reached" is the counting learner on the actual training
puzzles, which keeps one answer per orbit and is told the rule. 100 and 600 training puzzles hold 57 and 311 midpoint
question rows.

| built in | orbits | 100 puzzles: expected / reached / trained network | 600 puzzles | rows for 90% |
|---|---|---|---|---|
| nothing, one answer per pair (a, b) | 1,250 | 4.5 / 5.0 / 5.0 ± 0.5 | 22.0 / 21.6 / 18.0 ± 0.9 | 2,878 |
| the shift | 50 | 74.8 / 80.0 / 59.9 ± 4.5 | 98.7 / 99.8 / 98.3 ± 1.9 | 107 |
| the shift and doubling | 26 | 90.5 / 94.3 / 89.2 ± 2.7 | 99.3 / 99.8 / 99.2 ± 0.2 | 55 |
| the shift and scaling | 1 | 100 / 100 / 100.0 | 100 / 100 / 100.0 | 1 |

The formula gives the counting learner's score to within 6 points before a single puzzle is drawn. The shift cuts
the rows needed for 90% from 2,878 to 107, a factor of 27. The trained networks are within 4 points of the counted
value without a symmetry. With one they are under it where many orbits are left and little data is given: by 20
points with the shift and by 5 with doubling at 100 puzzles, by 2 or less at 600, and by nothing with scaling. The
three networks with a symmetry are those whose row descriptor reads the steps (Section 7.7); Section 3.7 says why
that matters.

## 3.6.3 The ceiling, and whose ceiling it is

**Statement 4 (the ceiling).** On a rule in which the rows of an orbit all need different answers, a learner of the
form of Statement 2 is right on at most one row of each orbit. For sum under the shift that is 97 rows of 1,225: 7.9%.

*Why.* On a sum row the answer is b + a, so the learner is right exactly when g(b − a) = a. Rows with the same first
step have different a, and g holds one number for that step. It can match one of them. Sum's table has 97 different
first steps. With rows drawn evenly, the best a learner can expect is one right row per orbit, 97 in 1,225. In
general the count per orbit is the largest group of its rows that agree on the answer; that is the bound of Wang et
al. (2023), evaluated here on a table.

*Checked.* Fitted on the whole table, the counting learner with the shift is right on 97 of 1,225 rows. With the
shift and doubling it is 88 (7.2%), with the shift and scaling 17 (1.4%). The trained shared model reaches 4.5 ± 2.3
and 6.2 ± 0.8 at 100 and 600 puzzles and 7.1 at 6,000, at the epoch picked on the selection puzzles.

*How sharp the check is.* The ceiling is on the expected accuracy. The unseen set has 507 sum rows, on which one
standard deviation of a 7.9% rate is 1.2 points. Over all 172 evaluated epochs of the nine runs, 8 are above 7.9 and
the highest is 9.7, in the run on 6,000 puzzles, whose epochs average 7.9. So the model sits at the ceiling, within
what chance allows on this many rows, and the measurement cannot exclude a point or two above it. Statement 5 says
why a point or two would not contradict anything.

**Statement 5 (the ceiling belongs to learners that answer from the question row).** A model that has the shift of
the whole puzzle, all nine panels together, is not capped on sum by that symmetry alone.

*Why.* Here is a function of the whole puzzle that uses only differences:

    c₃ = b₃ + (c₁ − b₁) + (a₃ − a₁).

Shift every panel by s and the answer moves by s, so the function has the shift. On a sum puzzle c₁ − b₁ is a₁, so
the function returns b₃ + a₁ + a₃ − a₁ = a₃ + b₃, the right answer. A complete sum row shows where zero is, and a
learner that can measure from a panel of one row to a panel of another can use it.

*Checked.* The function is right on 507 of 507 unseen sum rows and follows a shift of the whole puzzle on all of them.

*What this means for the trained model.* Statements 2 to 4 are proved for a learner that sees the question row. The
template sees the whole puzzle. It stays under the ceiling because of how it reads the puzzle, which can be read off
its rules: the score of a candidate value V is computed from V's offsets to the two panels of the question row, from
how many panels of each row sit exactly at V, and from a descriptor of the puzzle that is the same for every V. No
rule measures the offset between a panel of the question row and a panel of another row. So apart from values that
happen to coincide between rows, the template is a learner of the form of Statement 2, with a g that may depend on
the descriptor. The rows of an attribute are drawn independently, so the descriptor of the complete rows says nothing
about where the question row sits. For the template the ceiling is therefore a prediction that follows from its
structure, and not a consequence of the puzzle-wide shift alone. The coincidences are a real way past it, though a
small one: in 48 of the 507 unseen sum rows the answer is a value that already stands in the puzzle, and the template
sees which values those are. The runs sit at the ceiling and not clearly above it.

In the terms of Wang et al. (2023): shifting one row turns a sum row into a row that needs another answer, which is
their incorrect equivariance, the case their lower bound covers. Shifting a whole sum puzzle turns it into something
that is not a sum puzzle at all, which is their extrinsic equivariance; the bound does not apply to it, and they show
that it can help or hurt.

## 3.6.4 More of the rule's moves

**Statement 6 (doubling).** A learner that has the shift and doubling has one free number for each odd part of the
first step. On the ring, a row with a first step it never saw is answerable exactly when a training row had a first
step with the same odd part.

*Why.* By Statement 2 the learner computes b + g(d). Doubling the input must double the answer's offset from b, so
g(2d) = 2g(d). Write a step as d = m·2ʲ with m odd. Then g(d) = 2ʲ·g(m): the entry for the odd part fixes the
entries for all its doublings, and one training row with any of them fixes g(m). Midpoint's even first steps between
−50 and 50 have 26 odd parts.

*Checked.* At 100 puzzles 101 unseen midpoint rows have a new first step: 72 with an odd part that training had,
29 without. The trained doubled model is right on 50 / 43 / 39 of the 72 and on 0 / 0 / 0 of the 29, over three
seeds. The model with the shift alone is right on none of the 101.

**Statement 7 (scaling).** On the ring, a learner that has the shift and every scaling computes c = a + t·(b − a)
for one number t, on every row with a ≠ b.

*Why.* Any two different values (a, b) are the image of (0, 1) under "multiply by b − a, then add a". The learner
has both moves, so f(a, b) = (b − a)·f(0, 1) + a. Its only free number is t = f(0, 1). Midpoint is t = ½,
Progression is t = 2, mirror is t = −1.

*A consequence that is also proved.* With scaling built in, every question row with two different values is the same
input, (0, 1). The question row then says nothing about which rule applies. Midpoint and Progression need different
t, so a model with scaling can tell them apart only from the complete rows.

*Checked.* The counting learner with these moves is right on every midpoint row after the first training row. On sum
its ceiling is 17 rows of 1,225. The consequence is what Section 7.7 measured, in both directions. With a row
descriptor that came out the same for every row, the scaled model continued four midpoint rows in five as a
Progression at two seeds of three. With a descriptor that reads each pair's step, it is right on all 505 midpoint rows
in six runs of six, from 100 puzzles and from 600, and the descriptor separates the two rules in every run.

**Statement 8 (addition's own moves).** A learner that has both moves of one operand with the answer computes
c = a + b + k for one number k.

*Why.* Move the first operand by −a and the second by −b: the learner must answer (0, 0) with f(a, b) − a − b. So
f(a, b) = a + b + f(0, 0).

*Checked.* The counting learner with these moves is right on every sum row after one training row. The shared model
given the `plus` facts reaches 98.2 ± 0.8 on sum from 100 puzzles and 99.7 ± 0.4 from 600.

**Statement 9 (one orbit is the rule, up to one answer).** If the moves a learner has can turn any input into any
other, the learner is fixed by its answer to one input.

*Why.* Pick an input x. Every other input is some move h applied to x, and the learner must answer it with h applied
to its answer to x. Nothing else is free.

This is why the number of orbits is the right unit: it counts the answers that are left to learn. Statements 7 and 8
are the two cases with one orbit, and in both the model has been handed its rule up to one number.

## 3.6.5 Values read as numbers

**Statement 10.** Let the rule be c = pa + qb, and let the two complete rows of a puzzle be (a₁, b₁, c₁) and
(a₂, b₂, c₂). If a₁b₂ ≠ a₂b₁, the two rows fix p and q, a least-squares fit finds them, and its answer for the
question row is exact. Nothing is learned from other puzzles.

*Why.* The two rows are two equations, a₁p + b₁q = c₁ and a₂p + b₂q = c₂. They have exactly one solution when
a₁b₂ ≠ a₂b₁, and the rule's own p and q are a solution. Least squares returns the exact solution when there is one.
When a₁b₂ = a₂b₁ the rows leave a whole line of solutions, and the fit returns the shortest, which need not be the
rule's.

*Checked* (`analysis/twins_linear_fit.py`, unseen puzzles, no training):

| puzzles | the fourth rule | Progression | Constant | Distribute Three |
|---|---|---|---|---|
| sum | 99.6 | 98.7 | 100 | 0 |
| midpoint | 99.8 | 99.3 | 100 | 0 |
| mirror | 99.8 | 98.7 | 100 | 0 |
| lookup | 1.7 | 98.6 | 100 | 0 |

All 44 misses on the linear rules (13, 7, 12 and 12) are puzzles with a₁b₂ = a₂b₁. There is no miss among the others.
Lookup and Distribute Three are not of the form, and the fit gets 1.7% and nothing.

## 3.6.6 Reach: what is proved and what is measured

**Statement 11.** For a learner with one free number per orbit, training rows from other orbits put no condition on
the number of an orbit that training never reached.

*Why.* By Statement 2 the learner's answer on a row depends on g at that row's first step only. A training row with
another first step involves another entry of g. So every setting of the unreached entry fits the training rows
equally well, and the training data cannot make it right.

*Proved for:* the counting learner. *Measured for the networks:* the kernel has one free weight vector per offset,
and nothing ties the vectors of different offsets together, but a network could still carry "half the step back"
from the steps it has seen. It does not. Pooled over three seeds:

| training rows in the row's orbit | 0 | 1 | 3–4 |
|---|---|---|---|
| shared model, 100 puzzles | 0 of 303 | 422 of 573 (74%) | 172 of 186 (92%) |
| shared model, 600 puzzles | 0 of 3 | 71 of 72 (99%) | 89 of 90 (99%) |
| weights per value, 100 puzzles (orbit: the pair) | 11 of 1,440 (1%) | 65 of 75 (87%) | – |

With no training row in its orbit a row is never right, beyond the 1% that guessing among 101 values gives. With one
it usually is. So which unseen rows a model of this family answers can be listed from the training set before
training: the rows whose orbit it reached.

## 3.7 The ladder

Statements 2, 6, 7 and 10 put the models in one order, by how much of the rule's group is built in and how many
answers are left to learn. Midpoint at 100 training puzzles:

| built in | left to learn | expected from the table | counting learner | trained model |
|---|---|---|---|---|
| nothing about the values | 1,250 rows | 4.5 | 5.0 | 5.0 ± 0.5 |
| the shift | 50 orbits | 74.8 | 80.0 | 59.9 ± 4.5 |
| the shift and doubling | 26 orbits | 90.5 | 94.3 | 89.2 ± 2.7 |
| the shift and scaling | 1 orbit, one number | 100 | 100 | 100.0 / 100.0 / 100.0 |
| the linear form, values as numbers | nothing from other puzzles | – | – | 99.8, no training |

For sum the ladder has other rungs, because its moves are others: nothing (4.8 ± 0.6); the shift, which sum does not
keep (4.5 ± 2.3 at a ceiling of 7.9); its own moves (98.2 ± 0.8); the linear form (99.6, no training).

The count says each rung needs less data than the one below, and the trained models follow it up the whole ladder.
That took two attempts. With the first row descriptor the two scaling rungs fell below the rung beneath them: 49.9 ±
5.3 with doubling and 19.0 / 100.0 / 21.0 with scaling. Statement 7's consequence names the reason: with rows read in
a unit the rule has to come from the complete rows, and that descriptor did not carry it. A descriptor that reads each
pair's step was then built, with its outcome predicted before the runs, and gives the table above
(`docs/ARITHMETIC.md`, "The row descriptor reads the steps").

## 3.7.1 What is not proved

- **That the trained network reaches the count.** The count is what a learner with one answer per orbit gets. The
  networks come within 4 points of it without a symmetry, and with the shift they stay 20 points under it at 100
  puzzles and 2 at 600.
- **That a network cannot generalise past its orbits.** Statement 11 is proved for the counting learner. For the
  networks it is a measurement on one family of templates, in which nothing but the built-in moves ties weights
  together. A model with other ties, such as a smooth kernel, could do otherwise.
- **The ceiling for the template.** Statement 5: it follows from how the template reads the puzzle, with coincidences
  of values between rows left over. It is not a theorem about the puzzle-wide shift. The runs sit at it: 7.1 or less
  at every picked epoch, and above 7.9 at 8 of 172 evaluated epochs, 9.7 at the most.
- **Anything about groups on 0 to 50.** A shift can leave the range and midpoint needs a + b even. The counts are made
  on the actual tables. Statements 6 and 7 are exact on the ring of 101 values.

---

## What is known already, and what is added here

All of these were found by search on 2026-11-17 and read as abstracts only, except where Chapter 2 says otherwise.

| statement | standing | source |
|---|---|---|
| 1 | textbook: a weighted sum of points needs no origin when the weights sum to 1 | Section 2.4 (Stevens 1946; Baez 2009) |
| 2, 6 to 9 | elementary; no novelty claimed. The form in 8 is a known functional equation | reading list: Reem 2017 |
| 3 | the gain from an invariance as more effective samples is known in general: for a finite group the effective number of samples is multiplied by the group's size | Tahmasebi and Jegelka 2023; Sannai et al. 2021; Elesedy and Zaidi 2021; Bietti et al. 2021 |
| 3, added here | the exact count on a finite table with orbits of unequal size, and its test against a counting learner and trained networks | – |
| 4 | the bound | Wang et al. 2023 |
| 5 | the two cases are Wang et al.'s incorrect and extrinsic equivariance; the example and the reading of the template are added here | Wang et al. 2023 |
| 10 | models that make addition an operation of the representation | Section 2.5 (NVSA, ARLC) |
| 11 | what trained networks do on inputs outside the training support is studied for Boolean functions (they find the interpolator of lowest degree); reasoning over symbols never seen in training is proved to fail for fully connected networks | Abbe et al. 2023; Boix-Adserà et al. 2024 |

- Abbe, E., Bengio, S., Lotfi, A., Rizk, K. (2023). Generalization on the unseen, logic reasoning and degree
  curriculum. *ICML*; JMLR 25 (2024). arXiv:2301.13105.
- Boix-Adserà, E., Saremi, O., Abbe, E., Bengio, S., Littwin, E., Susskind, J. (2024). When can transformers reason
  with abstract symbols? *ICLR*. arXiv:2310.09753.
- Sannai, A., Imaizumi, M., Kawano, M. (2021). Improved generalization bounds of group invariant / equivariant deep
  networks via quotient feature spaces. *UAI*. arXiv:1910.06552.
- Tahmasebi, B., Jegelka, S. (2023). The exact sample complexity gain from invariances for kernel regression.
  *NeurIPS*. arXiv:2303.14269.

---

## Notes for you (not thesis text)

- **Statement 5 changes a sentence of Chapter 7.** Sections 7.3 and 7.4 said that a model equivariant to the common
  shift can only compute b + g(b − a) and so cannot fit sum. That is true of a learner that answers from the question
  row. The template has the shift of the whole puzzle, and that alone would not cap sum (the function in Statement 5).
  I changed the two sentences and added a paragraph to 7.3. An examiner who knows Wang et al. could have raised it.
- **The same point opens a run you have not made.** A kernel between a panel and the panel above it would give the
  template the cross-row offsets that the function in Statement 5 uses. The model would keep the shift and could in
  principle learn sum from differences. That is "the zero found from the complete rows", which `--find-zero` did with
  the `plus` facts. Not run, and not needed for the argument.
- **Statement 3 is the cleanest new piece.** The coverage formula predicts the counting learner from the table alone
  (74.8 against 80.0, 98.7 against 99.8, 4.5 against 5.0, 22.0 against 21.6, 91.2 against 90.5). It makes "the data a
  symmetry saves" a number: 2,878 rows against 107 for 90%.
- **Nothing here is deep mathematics.** Each proof is two or three lines. The contribution is that every statement is
  tied to a count made before training and to a measurement. Do not let the chapter claim more.
- **The sources are abstracts.** Tahmasebi and Jegelka's "multiplied by the group's size" is from the search summary
  of their abstract. Read the papers before citing them for more than their existence.
- **Voice.** "We", as in the other chapters.
