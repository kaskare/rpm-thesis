# Abstract and Chapter 1 (draft, 2026-11-13)

## Abstract

Neural networks that read the images directly can solve generated Raven's Progressive Matrices, but only with a great
deal of data: a vision transformer trained on I-RAVEN answers 19% of puzzles from 1,000 training puzzles per layout and
93% from 6,000. This thesis studies the alternative, structure built in before training, and asks what a built-in
symmetry is worth to a relational model, and whether that can be known before training.

The models are lifted relational neural networks in PyNeuraLogic, where a template of weighted rules is grounded into a
different network for every puzzle. We prove that a template cannot tell apart constants it does not name, so a symmetry
is built in by leaving the transformed things unnamed, and holds exactly for every setting of the weights. With this
method we build a model that generates the missing panel from a graph of the puzzle's objects, without being given the
rules or seeing the answer candidates. On the I-RAVEN test set it reaches 97.2% from object labels and 97.0% end to end
behind an object recognizer trained on the labelled objects of every panel. The vision transformer, which is given the
label of the answer only, reaches 93.1% with six times the weights.

We then put a price on symmetries. Counting the orbits of a rule's table under the model's symmetry predicts how much
data the symmetry saves, and the ceiling it sets when the rule does not share it. Two rules matched in form and size,
sum and midpoint, test this. From 600 puzzles a shift-equivariant model that reads the answer off the question row
reaches 91% on midpoint and 6% on sum, whose predicted ceiling is 7.9%; given addition as background knowledge, it
learns sum from 100 puzzles. On the benchmark, removing the shift symmetry costs 61 points of accuracy at 100 training
puzzles and 2 at 6,000. Counted before the runs, the symmetry between the attributes of an object is worth two to three times the
training data, and the model without it needs that much more. A symmetry that conflicts with the data holds the model
near the ceiling computed in advance. The counts bound what a network can reach; whether it gets there also depends on
how it reads the puzzle, and the thesis reports where it falls short.

**Keywords:** Raven's Progressive Matrices, inductive bias, symmetry, equivariance, lifted relational neural networks,
graph neural networks, neuro-symbolic learning, sample efficiency

---

## Chapter 1. Introduction

### 1.1 Motivation

A Raven's Progressive Matrix shows eight panels arranged in a three-by-three grid, with the ninth left empty. Each
panel holds simple figures, and the figures change along the rows according to rules the solver is not told: a shape
stays the same, a size grows by one step, a colour is the sum of the two before it. The solver has to find the rules in
the two complete rows, apply them to the third, and pick the missing panel from eight candidates. Raven (1938) designed
the test to measure reasoning that does not depend on language or schooling, and the rules behind its items were later
analysed by Carpenter et al. (1990).

Machine learning took over the format through generated datasets. In PGM (Barrett et al., 2018) and RAVEN (Zhang et al., 2019a) a program produces every puzzle. For RAVEN it picks one of seven layouts, gives each attribute of the objects
(type, size, colour, number and position) one of four rules (Constant, Progression, Arithmetic, Distribute Three), fills
the rows accordingly and renders the panels. I-RAVEN (Hu et al., 2021) repaired the way RAVEN draws its wrong answers,
and I-RAVEN-X (Hersche et al., 2024) lengthened the rows to ten panels and widened the value ranges up to a thousand.
These three are the datasets of this thesis.

The first neural solvers were black boxes: networks that map the sixteen images of a puzzle to the index of the answer
(Barrett et al., 2018; Zhuo and Kankanhalli, 2020). Two things about them motivate this work.

The first is that they find shortcuts. In RAVEN every wrong candidate differs from the answer in one attribute, so the
answer is the candidate that most resembles the others, and a model that never looks at the matrix can find it (Hu et
al., 2021). In our own check, a model that sees only the eight candidates scores 90 to 100% on RAVEN and is at chance on
I-RAVEN. A high score on such a benchmark may therefore say little about reasoning, which is the general problem of
shortcut learning (Geirhos et al., 2020).

The second is that they need a great deal of data. We trained a vision transformer on I-RAVEN to generate the answer
from the image of the matrix. With 300 training puzzles per layout it answers 16% of unseen puzzles, where guessing
gives 12.5%. With 1,000 it answers 19%, with 3,000 it answers 46%, and only with 6,000 does it reach 93%. The
information is in the small sets all the same: a model that is given the objects of each panel, the facet model of
Chapter 4, reaches 90% from the same 300 puzzles. It still reaches 90% when an object recognizer, trained on the
labelled panels of those 300 puzzles, reads the images for it, although that recognizer learns from far more labels
than the transformer, which sees only the answer.

Chollet (2019) argues that this is the number that matters. Skill at a fixed task can be bought with unlimited
training data or with unlimited prior knowledge, so skill alone says little about a system; what is informative is how
much experience it needed, given the priors it started with. We take this as a rule for reporting. Every model in this
thesis states what is built into it, and its accuracy is reported against the number of training puzzles.

What can be built in is the subject of a long argument. One side holds that general methods and more computation
overtake hand-built structure in the end (Sutton, 2019). The other holds that structure which matches the problem,
relational structure in particular, is what lets a model generalise from little data (Battaglia et al., 2018).
Geometric deep learning gives the second view a precise form: the structure to build in is a symmetry, a set of
transformations of the input under which the answer does not change or changes in a known way (Bronstein et al., 2021),
and there are proofs that a model with the right invariance generalises strictly better (Elesedy and Zaidi, 2021) and,
for kernel methods, needs fewer examples by up to the size of the group (Bietti et al., 2021; Tahmasebi and Jegelka,
2023).

In practice, though, a prior is still chosen by intuition and judged afterwards by the accuracy it gives. That leaves
two questions unanswered each time. How many training examples did the prior replace? And what would it have cost had it
been the wrong one? A prior that fits its benchmark looks the same, in a results table, as a prior that is right in
general.

### 1.2 The question and the approach

This thesis asks: **what is a built-in symmetry worth to a relational model, and can that be known before training?**

Raven's matrices suit the question for a reason that is usually held against them. The data is generated, so its
symmetries are known exactly. The two complete rows of a puzzle can be exchanged. In a layout with two groups of
objects, the groups can be exchanged. Size and colour follow the same rules and can be exchanged. Most rules survive when every value is shifted by
the same amount, and one does not. Some of these transformations turn a valid puzzle into another puzzle the generator
could have produced, some into a valid puzzle it never produces, and some into a puzzle whose right answer is a different
one. A symmetry can thus be the right one, a harmless one or a wrong one (correct, extrinsic and incorrect
equivariance in the terms of Wang et al., 2023a), and which it is can be read off the data.

The models are lifted relational neural networks (Šourek et al., 2021), written in PyNeuraLogic. Such a model is a
template: a set of weighted rules over variables. For each puzzle the template is grounded on the facts that describe
that puzzle, and the result is a neural network whose shape follows the puzzle: more objects give more neurons, but the
weights belong to the rules and are the same everywhere. Like a graph neural network (Zhou et al., 2020), the template
shares its weights over a structure; unlike one, it lets the designer say which structure, rule by rule. Whatever a rule
refers to only through a variable, it treats alike. So in a template a symmetry is a decision about what to name.

Our claim is that the worth of such a decision can be computed. A symmetry merges the inputs it maps onto one another
into orbits, and a model that has the symmetry built in must treat all inputs of an orbit alike. If the rule to be
learned treats them alike too, the model has only as many things to learn as there are orbits, and the number of orbits
measures the data it needs. If the rule asks for something different on each input of an orbit, the model can be right
on at most one of them, and the number of orbits over the number of inputs is a ceiling on its accuracy that no amount
of data lifts. Both numbers come from
counting, before any model is trained.

### 1.3 Contributions

**A method for building symmetries into a relational model.** We prove that the output of a template does not change
when constants that it does not name are renamed throughout a puzzle's facts (Chapter 3). The statement is the familiar
invariance of relational models under isomorphism and its proof is short; what we add is its use as one test that
every symmetry in the thesis passes or fails, before any training. A transformation of the puzzle
that amounts to such a renaming is then a symmetry of the model. Leaving something unnamed builds in every permutation
of it, and background facts narrow that group down: facts that
order the values, for instance, leave only the shifts. The symmetry is exact and holds for every setting of the weights,
which we confirm on random weights to about 10⁻¹⁶. With this method we build the facet model (Chapter 4), a graph for
each puzzle with one node per attribute of each object. It generates a description of the missing panel and only then
picks the closest candidate, so it is never told the rules and never sees the candidates, and the shortcut above is
closed by construction. On the I-RAVEN test set it answers 97.2% of puzzles from ground-truth object labels, and 97.0%
end to end, when a recognizer built on the same transformer as our black-box baseline names the objects of every panel,
the candidates' included (Chapter 5). On the same test puzzles the black-box transformer reaches 93.1%, with six times
the weights of the whole pipeline. The two are not given the same supervision: the recognizer learns from the labelled
objects of every panel, the transformer from the label of the answer only.

**A price for a symmetry.** The orbit count above gives, for a pair of model and rule, the data a symmetry saves and
the ceiling a mismatched one sets (Chapter 3). It needs no network: a learner that only counts, one answer per orbit,
realises it. It describes a learner that shares nothing across orbits, which the value models of Chapter 7 do by
construction. A network reaches the count only if it can also tell the orbits apart, and Chapter 7 shows a case where
it could not until the part of the model that describes a row was changed. The ceiling is the error bound of Wang et al. (2023a, Theorem 4.3) for a symmetry the task does not have, evaluated on
a rule's table; the count of the data saved, and a test in which both numbers are predicted, are ours.

**Measurements that test the price.** The sharpest is a pair of rules built for the purpose (Chapter 7). In a row
(a, b, c), sum sets c = a + b and midpoint sets c = (a + b)/2. The two have the same form and tables of nearly the same
size, and models with no symmetry on the values find them equally hard. They differ in one property: adding the same
number to every panel keeps a midpoint row valid and breaks a sum row. A model with that shift built in, a convolution
along the axis of values, should then learn midpoint from few puzzles and be unable to learn sum, and the count says how
few and how unable. From 600 puzzles the model reaches 91.0% on midpoint and 6.2% on sum, against a ceiling of 7.9%
(means of three seeds, here and below). The ceiling follows from the shift together with the way the model reads the
puzzle, from the question row alone: a model that measured from one row to another would not have it (Section 3.6.3).
With separate weights for every value, and so without the shift, the same template reaches 18.0% and 19.4%. Trained on
100 puzzles, the model gets none of the 101 midpoint rows right that fall in an orbit absent from its training puzzles,
at any seed, and about three quarters of the others. Its weights per offset make that unavoidable; the count says in
advance which rows those are. And when addition itself is supplied as background facts, the same model learns
to use it and gets sum from 100 puzzles (98.2%). Chapter 7 says how
much that gives away: a rule's full group of moves is the rule, up to one constant. A symmetry is thus not good or bad in
itself. Its worth belongs to the pair of model and rule.

The same law holds on the benchmark (Chapter 6). Removing a symmetry the data has costs most when data is scarce: 61
points for the value shift at 100 puzzles, 14 and 13 points for the symmetries between components and between
attributes at 100 and 600 (three seeds), and between 0 and 6 points at 6,000. Counted in training puzzles before the
runs, the components' symmetry is worth a factor of two, the size of its group, which follows almost from how the
samples are formed, and the attributes' between two and three, which does not. The models without them need that much
more data. A symmetry of the task that the data
lacks, the order of Arithmetic's operands, still adds 7 points at 6,000 puzzles. A symmetry that maps valid puzzles to puzzles with
another answer holds the model near the ceiling the count gives, about half of all values, whether it trains on 600
puzzles or on 6,000. And a model that names
what it should not, here the position of a panel, learns a shortcut through it: after the two complete rows are
exchanged, its accuracy on puzzles with a Distribute rule falls from 98.3% to 39.5%, while the symmetric version is unchanged.

### 1.4 What the thesis does not claim

It does not present the best solver of Raven's matrices. The facet model is the demonstration that the method yields a
working model, not the result.

It does not claim that PyNeuraLogic makes a model more accurate. A graph neural network written in PyTorch with the same
symmetries reaches 96.8% on the I-RAVEN test set, against the facet model's 97.2%. The gains reported here belong to the symmetries.
What the lifted template adds is that a symmetry is declared, not engineered, and that one proof covers every symmetry
built this way.

It says nothing about human intelligence. Every score is measured on puzzles from the same generator as the training
puzzles, with a fixed list of attributes and with rules that run along rows only, which the tests people take do not
guarantee. The rows-only prior has a price: on I-RAVEN puzzles whose matrix is transposed, so that the rules run down
the columns, every facet model falls from about 97% to 45%. The priors studied here were chosen with the generator in
view, and Chapter 8 separates those a person would bring to any such test from those that only this family of
benchmarks rewards.

The recognizer is trained apart from the reasoning model, on the labelled objects of every panel, where the black-box
model is given labels for the answer only. Chapter 5 states what each model is given.

It does not claim that our black-box baseline is the best vision model. It is a plain vision transformer trained from
scratch. At the same amount of data, three published architectures for these puzzles, SCL, PredRNet and DRNet, reach 83
to 91% when retrained under one protocol, and 95 to 98% as their authors report them (Małkiński and Mańdziuk, 2025a).
They pick among the candidates, where ours generates the answer. Section 5.1 sets the numbers side by side.

The comparisons with little data in Chapters 6 and 7 have three seeds; most other numbers come from a single training
run, and the text says which.

### 1.5 The assignment

The assignment asks for symmetry-aware neural architectures that combine object recognition with inductive priors on
the logical composition of the objects, built on graph or relational neural networks, and for an improvement on
black-box vision models. Chapters 3 and 4 give the architectures and the priors, and Section 4.3 the object
recognition. Chapter 5 gives the comparison with a black-box vision model on the same task, where the gain is largest
when training puzzles are few. Chapters 6 and 7
measure what each prior contributes.

### 1.6 Structure

Chapter 2 introduces Raven's matrices and the three datasets, graph neural networks, lifted relational neural networks,
and symmetry as inductive bias. Chapter 3 gives the theory: the renaming theorem, how background facts choose the group,
the three kinds of symmetry, and the orbit count. Chapter 4 describes the task, the data, the recognizer, the facet
model and the baselines. Chapter 5 reports the solver's accuracy, with labels and end to end, against the baselines.
Chapter 6 measures what a matched symmetry buys and what a broken or wrong one costs. Chapter 7 is the study of sum and
midpoint. Chapter 8 discusses where the priors come from, what is specific to lifted templates, and the limits of the
work.

---

## References used above

- Tahmasebi, B., Jegelka, S. (2023). The exact sample complexity gain from invariances for kernel regression.
  *NeurIPS*. arXiv:2303.14269.
- Barrett, D. G. T., Hill, F., Santoro, A., Morcos, A. S., Lillicrap, T. (2018). Measuring abstract reasoning in neural
  networks. *ICML*.
- Battaglia, P. W., et al. (2018). Relational inductive biases, deep learning, and graph networks. arXiv:1806.01261.
- Bietti, A., Venturi, L., Bruna, J. (2021). On the sample complexity of learning under geometric stability. *NeurIPS*.
- Bronstein, M. M., Bruna, J., Cohen, T., Veličković, P. (2021). Geometric deep learning: grids, groups, graphs,
  geodesics, and gauges. arXiv:2104.13478.
- Carpenter, P. A., Just, M. A., Shell, P. (1990). What one intelligence test measures: a theoretical account of the
  processing in the Raven Progressive Matrices test. *Psychological Review* 97(3), 404–431.
- Chollet, F. (2019). On the measure of intelligence. arXiv:1911.01547.
- Elesedy, B., Zaidi, S. (2021). Provably strict generalisation benefit for equivariant models. *ICML*.
- Geirhos, R., et al. (2020). Shortcut learning in deep neural networks. *Nature Machine Intelligence* 2, 665–673.
- Hersche, M., Camposampiero, G., Wattenhofer, R., Sebastian, A., Rahimi, A. (2024). Towards learning to reason:
  comparing LLMs with neuro-symbolic on arithmetic relations in abstract reasoning. arXiv:2412.05586.
- Hu, S., Ma, Y., Liu, X., Wei, Y., Bai, S. (2021). Stratified rule-aware network for abstract visual reasoning. *AAAI*.
- Raven, J. C. (1938). *Progressive Matrices: A Perceptual Test of Intelligence*. H. K. Lewis, London.
- Wang, D., Zhu, X., Park, J. Y., Jia, M., Su, G., Platt, R., Walters, R. (2023a). A general theory of correct,
  incorrect, and extrinsic equivariance. *NeurIPS*. arXiv:2303.04745.
- Šourek, G., Železný, F., Kuželka, O. (2021). Beyond graph neural networks with lifted relational neural networks.
  *Machine Learning* 110, 1695–1738.
- Sutton, R. (2019). The bitter lesson. Essay, incompleteideas.net.
- Zhang, C., Gao, F., Jia, B., Zhu, Y., Zhu, S.-C. (2019a). RAVEN: a dataset for relational and analogical visual
  reasoning. *CVPR*.
- Zhou, J., et al. (2020). Graph neural networks: a review of methods and applications. *AI Open* 1, 57–81.
- Zhuo, T., Kankanhalli, M. (2020). Solving Raven's Progressive Matrices with neural networks. arXiv:2002.01646.
