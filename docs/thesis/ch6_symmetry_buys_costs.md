# Chapter 6. What a symmetry buys and what a wrong one costs (draft, 2026-11-14)

Cells are matched % with exact % in brackets, on unseen puzzles, one seed unless marked.

Chapter 5 compared models that differ in many things at once. This chapter changes one symmetry at a time inside one
model, and compares each outcome with what Chapter 3 predicts. The predictions, in the order of the sections:

1. A symmetry the data has merges inputs that need the same answer. A model with it built in has fewer things to
   learn, by a factor of up to the size of the group. The gain should be largest when training puzzles are few, grow
   with the group, and disappear once the training set covers every case by itself.
2. That gain comes from sharing weights. A model that shares them but is not exactly invariant should lose nothing on
   puzzles like the training ones.
3. Exact invariance matters where the data cannot teach the symmetry: on inputs the training set never contained, and
   where the generator leaves a gap between what is valid and what it produces. A symmetry of the task in that gap
   costs nothing on the right solution, and where the rule itself ignores what the symmetry exchanges, it helps that
   rule even at full data.
4. A symmetry that leads out of the data sets no ceiling. One that maps valid puzzles onto valid puzzles with another
   answer sets a ceiling that can be counted in advance, and data does not lift it.

## 6.1 Design

Each experiment trains the same template twice or more on the same puzzles: with the symmetry, and with it removed in
one of the ways of Section 4.5. *Labels* keep all weights shared and let the model tell the things apart. *Separate
weights* give each component, attribute or value its own copy of every rule. The separate-weight models are run at the
same number of weights as the shared model, by narrowing them, so that a difference is not a difference in size. Every
run takes the same number of updates.

Three symmetries are tested in Sections 6.2 and 6.3, each a symmetry of the data it is tested on (Section 3.5):

| symmetry | group | model | data |
|---|---|---|---|
| the two components | 2 elements | facet model | I-RAVEN, left-right and up-down layouts |
| the attributes | size and colour exchangeable; type for three of four rules | facet model, column roles | I-RAVEN-X, rows of 10, values 0–50 |
| a shift of all values | about 50 elements | value axis, rules' primitives given | I-RAVEN-X, rows of 10, values 0–50 |

## 6.2 A symmetry the data has saves training data

| symmetry | training puzzles | built in | removed, same number of weights | gain |
|---|---|---|---|---|
| components | 100 per layout, 3 seeds | 88.0 ± 1.9 (42.4 ± 3.0) | 73.8 ± 1.4 (19.4 ± 1.4) | **+14.2 ± 1.9** |
| | 1,000 | 99.3 (95.6) | 98.5 (84.5) | +0.8 |
| | 6,000 | 99.9 (97.7) | 100.0 (98.8) | −0.1 |
| attributes | 600, 3 seeds | 66.2 ± 2.7 (13.3 ± 2.8) | 52.7 ± 0.7 (3.6 ± 0.0) | **+13.5 ± 2.1** |
| | 6,000, 3 seeds | 94.2 ± 0.2 (27.6 ± 2.0) | 87.9 ± 1.6 (15.4 ± 2.7) | +6.3 ± 1.7 |
| value shift | 100 | 99.0 (97.3) | 38.3 (4.0) | **+60.7** |
| | 600 | 99.9 (99.9) | 92.7 (81.0) | +7.2 |
| | 6,000 | 100.0 (100.0) | 98.4 (97.4) | +1.6 |

The gains with three seeds are the means of the gains per seed; each seed's gain is within one standard deviation
of the mean (components 12.0 to 15.4, attributes 11.5 to 15.7 at 600 and 4.8 to 8.1 at 6,000). The value-shift rows are not at equal size: the shared model has 10.8k weights and the model with weights per value
335k, because a copy of every rule is needed for each of 51 values.

**The gain is largest with little data and shrinks as data grows**, in all three cases. For the components it is gone
at 6,000 puzzles: by then the training set shows each component every rule often enough.

**The gain grows with the group.** A group of two buys 14 points at 100 puzzles. The shift of about fifty values buys
61. Put as data, the model without the value shift is still behind after sixty times the puzzles: 98.4% at 6,000
against 99.0% at 100. That factor is more than sixty, larger than the group, but the two models also differ thirty times
in size, so it is not the symmetry's alone. The direction is consistent with the sample-complexity results of Section
2.4. The size of the saving we counted only where the models are otherwise equal, next.

**The saving, counted in puzzles before the runs.** Section 2.4 cites proofs, for kernel methods, that a symmetry saves data by a factor of up to
the size of its group. For the components the factor can be read from the template. Each training sample is one component of one
puzzle. With separate weights the copy for one component trains only on that component's samples, while the shared
network trains on both. The two components follow the same distribution, so a separate model given twice the puzzles gives each copy what the shared network gets: the factor is exactly 2.
For the components this count is close to a tautology. What the run tests is that nothing else is at work: neither
a gain the shared network draws from serving both components, nor a loss from it. For the attributes it is not one number. Size
and colour carry the four rules equally often, but type never carries Arithmetic. The shared network therefore sees
twice the examples a separate copy sees on Arithmetic, and up to three times on the other rules. We wrote these
predictions down, then trained the separate model at the same widths as the shared one on twice and three times the
data:

| symmetry, puzzles scored | shared weights | separate weights, more data | predicted | measured |
|---|---|---|---|---|
| components, all | 100: 88.0 ± 1.9 | 200: 88.1 ± 0.2 | equal (factor 2) | equal |
| attributes, with Arithmetic | 600: 63.0 ± 2.9 | 1,200: 60.8 ± 1.2 | equal (factor 2) | within 2.2 points |
| attributes, without Arithmetic | 600: 68.7 ± 2.8 | 1,200: 64.0 ± 0.8 | separate behind (factor above 2) | 4.7 behind |
| attributes, all | 600: 66.2 ± 2.7 | 1,200: 62.6 ± 0.6; 1,800: 69.8 ± 0.9 | between the two (factor 2 to 3) | between the two |

Three seeds each, matched %. On the components, where the count is nearly a tautology, it holds to a tenth of a point:
the shared network draws nothing from serving both components beyond the doubled data. The attributes are the real
test, because there the count is not the size of the group, and the pattern is the counted one. At twice the data the gap closes where only size and colour teach the rule, and stays where
type also does; at three times the data the separate model is level on the puzzles without Arithmetic (69.8 against
68.7) and ahead on those with it. The attributes are worth between two and three times the data, about two and a half,
which is what the count of examples allowed. Built-in sharing is worth the training data it reuses, and no more. Figure 6.1 shows both
comparisons. The same checkpoints scored once on the test range at the end of the work give the same reading:
separate weights at 200 puzzles 88.3 ± 0.6 against shared at 100 88.4 ± 1.3, and on attributes shared at 600 (68.3)
between separate at 1,200 (65.0) and 1,800 (72.3) (Appendix D.6).

![**Figure 6.1.** What a symmetry is worth in training data. Matched % on unseen puzzles against the number of training puzzles, for the facet model with one set of weights shared by the two components or by the three attributes, and with separate weights, at the same widths or at a matched weight count. Without the symmetry the model needs twice the data for the components and two to three times for the attributes, as counted before the runs. Bars: one standard deviation over three seeds.](../figures/data_factor.png)

**Giving the removed-symmetry models their full width does not help them.** At the same widths as the shared model,
with two to three times the weights, they score 75.6 on components at 100 puzzles and 52.5 on attributes at 600,
within a point of the narrowed versions.

**The same holds in a model that is not relational.** Giving the vision transformer the two complete rows as a set
builds in the exchange of rows 1 and 2. At 3,000 puzzles per layout this raises the small transformer from 42.4 to
62.8 (means of three seeds; the gain per seed is 7.6 to 28.0 points) and the large one from 45.8 to 62.0 (one seed). At 6,000 the small one keeps a gain of 10 points and the large one none
(92.1 against 93.4). Below 3,000 neither version learns the task, so there is nothing for the symmetry to multiply.

## 6.3 The gain is the sharing, not the exact invariance

Labels remove the invariance and keep the sharing. With random weights, a labelled model changes its output by 0.4
when the components are mirrored. Trained, it does as well as the exactly invariant model:

| | training puzzles | exactly symmetric | labels, weights shared | separate weights |
|---|---|---|---|---|
| components | 100, 3 seeds | 88.0 ± 1.9 | 89.5 ± 2.1 | 73.8 ± 1.4 |
| | 1,000 | 99.3 | 99.6 | 98.5 |
| attributes | 600, 3 seeds | 66.2 ± 2.7 | 64.8 ± 2.1 | 52.7 ± 0.7 |
| | 6,000 | 94.2 ± 0.2 (3 seeds) | 93.9 | 87.9 ± 1.6 (3 seeds) |

The reason is visible in the template. A labelled model adds a few hundred weights for the labels and keeps every
other weight in a rule that serves both components, so every training puzzle still trains every weight. And because
the two components follow the same distribution, nothing in the data rewards treating them differently: the trained
labelled models change by at most 0.8 points when a puzzle is mirrored, and by at most 1.1 when size and colour are
exchanged. Per seed, labels score 1.4 ± 0.8 points above the symmetric model on components and 1.3 ± 0.7 below it on
attributes. They have learned the symmetry they were not
given.

The same comparison exists in PyTorch. Attribute GNN A is B with absolute labels on its nodes. On I-RAVEN it trails B
by 0.5 to 1.8 points at every amount of data (Section 5.4), which is the distance between two seeds.

So on puzzles like the training ones, a data symmetry pays through weight sharing, and exactness adds nothing
measurable. The next two sections are the cases where it does.

## 6.4 What training never showed

**An attribute that never carried a rule.** We removed from training and selection every puzzle in which size follows
Progression, and scored how often the size is generated exactly on unseen puzzles that have this pair.

| model | pair held out | pair in training (3,000 puzzles) |
|---|---|---|
| facet model | **87.7** | 89.9 |
| attribute GNN B, symmetric | **84.1** | 89.4 |
| attribute GNN A, the same with labels | 42.3 | 85.9 |
| object GNN | 20.6 | 83.2 |
| MLP | 19.7 | 68.0 |
| ViT, width 224 | 18.5 | 48.9 |

The models that cannot tell size from colour apply to size what they learned about Progression on colour and type,
and lose 2 to 5 points. The labelled model A, which matched B everywhere in Section 6.3, loses 44. The data could
teach A that the attributes behave alike only where it showed them behaving alike. Here exact invariance is the whole
difference between two models with the same sharing.

**A layout that was never trained.** Trained on six layouts without up-down and scored on up-down:

| | facet | GNN B | GNN A | object GNN | ViT 224 | MLP |
|---|---|---|---|---|---|---|
| matched on up-down | 99.1 | 99.5 | 99.9 | 23.6 | 14.0 | 13.3 |

All three models on the attribute graph answer the unseen layout as well as a trained one, and the others are near
the 12.5% of guessing. Labelled A transfers as well as the symmetric models, so this is not an effect of exact
symmetry. It comes from the graph: on it an object's type, size and colour are processed without reference to the slot
the object sits in, and up-down differs from the trained left-right layout only in its slots. Exact generation is 0
for every model on this layout, because the output bits for its two slots were never trained.

**Values that were never trained.** The value-axis model has no weight that belongs to a value. A checkpoint trained
on values 0–50 was run on puzzles with values 0–100 and scores 100.0 (100.0); trained on 100 puzzles, 98.1 (96.8).
The version with weights per value cannot be run on the wider range at all, and neither can any model that reads
thermometer codes.

## 6.5 A symmetry given up lets a shortcut in

On rows of 10, exchanging rows 1 and 2 is a symmetry of the task but not of the data: the exchanged puzzle has the same
answer and the generator never produces it (Section 4.2). The training data therefore cannot teach it, and a model
without it is free to learn something that holds only on what the generator produces.

Copy links show this. They let the model pass a context panel's value straight to the output, which suits Constant and
Distribute, whose answers are values of the context. With a gate per panel, the model can tell the panels apart.

| I-RAVEN-X, rows of 10, values 0–50 | as generated | rows 1 and 2 exchanged |
|---|---|---|
| copy links, a gate per panel | **97.8** | about 60 |
| copy links, gates computed from content | 94.9 | 94.9, identical |
| plain facet model | 87.4 ± 1.8 (3 seeds) | identical |

The model with a gate per panel has the highest score on the generated puzzles and loses 38 points when the rows are
exchanged. On puzzles with a Distribute attribute it falls from 98.3 to 39.5. In the generated data a Distribute row is
the first row moved by one step, so the value that the third row lacks always stands at the same place in the first
two, and the gates learned to fetch it from that address. They did not learn the rule. Gates computed from the content
of a panel cannot name an address; that model gives up 2.9 points on the generated puzzles and answers the exchanged
ones identically.

The gain of copy links is real information and not just extra connections: a control with the same links wired to the
wrong puzzle's panels gains 1.9 points where the proper links gain 12.4.

The baselines repeat the pattern, and they show the role of the gap.

| matched, as generated → rows exchanged | I-RAVEN (a data symmetry) | rows of 10 (a task symmetry only) |
|---|---|---|
| attribute GNN B, symmetric | identical | identical |
| attribute GNN A | 95.0 → 95.0 | 95.0 → **55.5** |
| object GNN | 91.3 → 91.3 | 84.0 → **52.7** |
| MLP | 72.8 → 72.4 | 84.4 → **47.7** |
| ViT, width 64 | 79.1 → 78.9 | 52.9 → **38.5** |

On I-RAVEN no model needs the symmetry built in, because the data has it and teaches it. On rows of 10 every model
without it fails the exchange. Model A's exact score on Distribute falls from 59.9 to 1.0: its lead over B on the
generated puzzles was the same shortcut.

A benchmark score, then, does not tell a rule from a regularity of the generator. Building in a symmetry of the task
that the generator lacks closes the gap between the two, at a small cost on the benchmark.

## 6.6 A symmetry of the task that buys accuracy at full data

On I-RAVEN-X with rows of 10, an Arithmetic row is a sum: its last value is the sum of the nine before it, or its first
value is the sum of the nine after it. The middle columns are interchangeable operands. Exchanging two of them keeps
the row valid with the same answer, and it keeps the answer of a Distribute row too, which is the value the row lacks
whatever the order. But the generator never produces most of the exchanged rows: it draws the operands one after
another from a shrinking range and places the zeros among them in a fixed pattern (Section 3.5). So the exchange is a
symmetry of the task outside the support of the data, like the exchange of rows in Section 6.5.

The plain facet model names every column and gives each of the ten its own weights. *Column roles* name only the first
and the last column and read the eight middle columns through one rule, with the panel as a variable (Section 3.3).
Any exchange of middle columns is then built in. Progression rows do not survive the exchange, but their answer can
still be computed from the first value and the middle values taken together, so the symmetry costs Progression no
answer. *Prediction, written before the run:* Arithmetic improves, and nothing else loses.

| rows of 10, 6,000 puzzles, 3 seeds | named columns | column roles |
|---|---|---|
| matched | 87.4 ± 1.8 | **94.2 ± 0.2** |
| exact, Arithmetic | 20.7 ± 2.2 | **39.6 ± 1.3** |
| exact, Distribute | 11.3 ± 1.3 | 22.9 ± 0.8 |
| exact, Progression | 86.0 ± 7.5 | 92.8 ± 1.6 |
| exact, Constant | 95.8 ± 2.5 | 97.5 ± 2.8 |
| matched, two middle columns exchanged | 71.1 ± 1.0 | identical |
| weights | 1,411k | 288k |

Arithmetic's exact score doubles, as predicted, and Distribute's doubles as well, which fits a rule whose answer does
not depend on the order. On the test range, scored once at the end, the gain is smaller: Arithmetic 36.0 ± 2.1 against
20.9 ± 0.5, an increase of 1.7 times rather than 1.9, while matched (93.7, 94.5, 94.7 against 87.1 ± 2.0) and
Distribute (21.6 against 10.5) hold (Appendix D.6). The drop is within the sampling error of the 507 Arithmetic rows
that all three seeds share, but it is the direction a choice made with those puzzles in view would leave, and column
roles were chosen that way, so the test number is the safer one to quote. Progression and Constant move within the spread of the seeds. The named model loses 16 points
when two middle columns are exchanged: it reads the order that the generator imposes. The roles model also has a fifth
of the weights, so the size of the model could be the cause. It is not: at equal size, with widths 64, 144 and 288 and
1,390k weights, the roles model matches 94.9 (one run).

This is the largest gain from a symmetry at 6,000 puzzles in this chapter, 6.8 ± 1.6 points per seed. The data could
not have taught it, because the exchanged orders never occur. Arithmetic stays the weak rule all the same: fewer than
half of its values are generated exactly.

## 6.7 A wrong symmetry

Two symmetries the data does not have were built into the facet model, exactly, on I-RAVEN-X with rows of 3 and values
0–10. The predictions were written down before the runs.

**Columns 1 and 2 tied: out of the data.** The model cannot tell (a, b, c) from (b, a, c). For Constant and Distribute
rows, and for sums, the exchanged row is valid with the same answer, so for them the tie is a correct symmetry. For
Progression and for differences the exchanged row never occurs. Nowhere does the tie ask for two different answers to
one input. *Prediction:* no ceiling; at most a cost in data.

| exact, Constant / Progression / Distribute / Arithmetic | 6,000 puzzles | 600 puzzles |
|---|---|---|
| plain (matched 99.2 and 95.1) | 99.6 / 99.5 / 99.3 / 95.0 | 99.6 / 99.3 / 89.2 / 76.2 |
| columns 1 and 2 tied (matched 98.6 and 92.1) | 99.6 / 100.0 / 96.3 / 89.9 | 99.6 / 99.5 / 94.7 / 62.4 |

No ceiling appears: the tied model matches 98.6% at 6,000 puzzles, and Progression, whose rows the tie breaks, is
generated exactly every time. The cost is on Arithmetic, 5 points at 6,000 puzzles and 14 at 600. Distribute gains 5
points at 600 puzzles, as a rule for which the tie is a correct symmetry should.

**Values folded: in conflict with the data.** The values v and 10 − v get the same input code. A puzzle and its
reflection are then one input, both occur equally often, and their answers differ unless the answer is the middle
value. *Prediction, counted on the data:* at most 54.4%, 56.0% and 55.0% of Constant, Progression and Distribute
values can be exact. Arithmetic rows do not survive reflection, so they have no forced ceiling.

| | 600 puzzles | 6,000 puzzles |
|---|---|---|
| plain, matched | 95.1 | 99.2 |
| values folded, matched | 39.7 | 46.6 |
| values folded, values exact (type / size / colour) | 57 / 52 / 49 | 52 / 53 / 50 |

The folded model generates about half of all values exactly, at the counted ceiling to within the noise of a thousand
puzzles. Ten times the data moves it from 49–57% to 50–53%, which is no movement. A plain model with a tenth of the
data is far ahead of it.

## 6.8 Summary

| what the symmetry does to valid puzzles | prediction | measured |
|---|---|---|
| maps them to equally likely puzzles with the corresponding answer | saves data, by up to the size of the group; nothing at full data | +14 (group of 2, 3 seeds) to +61 points (about 50) at 100 puzzles; −0.1 to +6 at 6,000; counted: without the components' symmetry exactly twice the puzzles (88.1 at 200 against 88.0 at 100) |
| the same, but the model only shares weights and is not invariant | no loss on puzzles like the training ones | within 1.5 points of the invariant model (3 seeds) |
| | a loss on what training never showed | 42.3 against 84.1 on a held-out rule and attribute |
| maps them to valid puzzles the generator never produces | no loss on the right solution; closes a shortcut | 94.9 unchanged against 97.8 → 60 under the exchange |
| | where the rule ignores the order, a gain on that rule | Arithmetic exact 39.6 against 20.7 (test: 36.0 against 20.9); matched 94.2 against 87.4 at 6,000 (3 seeds) |
| maps them out of the valid puzzles | no ceiling, a cost in data | 98.6 against 99.2 at 6,000; 92.1 against 95.1 at 600 |
| maps them to valid puzzles with another answer | a ceiling, counted in advance | about half of the values, at 600 puzzles and at 6,000 |

Every row came out on the side predicted. The sizes are single runs, and Chapter 8 lists what that leaves open. The
next chapter tests the first and the last row of this table on one model with two rules, where the prediction is a
number.
