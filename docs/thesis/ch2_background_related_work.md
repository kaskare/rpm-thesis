# Chapter 2. Background and related work (draft, 2026-11-14)

Follows `abstract_introduction.md`. Chapter 3 (theory) is not drafted here. Notes for you are at the end; citations
marked † are from memory and have not been opened in this project.

---

## 2.1 Raven's matrices and the three datasets

**The format.** A Raven's Progressive Matrix is a three-by-three grid of panels with the last one missing. The figures
change along the rows by rules the solver has to find, and the answer is chosen from eight candidates (Raven, 1938).

**RAVEN.** Zhang et al. (2019) generate such puzzles by program. A puzzle has one of seven layouts: a single centred
object, a 2×2 grid, a 3×3 grid, two objects side by side (left-right), two objects one above the other (up-down), one
object inside another (in-out), and a 2×2 grid inside an object (in-out grid). The last four layouts have two
*components*, each with its own rules. Every object has a type, a size and a colour, with 5, 6 and 10 values, and a
grid component also has the number of its objects and their positions. Each attribute follows one of four rules along
every row:

| rule | a row (a, b, c) satisfies |
|---|---|
| Constant | a = b = c |
| Progression | b − a = c − b, a fixed step of ±1 or ±2 |
| Arithmetic | c = a + b or c = a − b; on positions, the union or difference of the occupied slots |
| Distribute Three | the three panels hold three fixed values, in a different order in each row |

The object's orientation follows no rule and acts as a distractor.

**I-RAVEN.** RAVEN builds each wrong candidate by changing one attribute of the answer. The answer is then the
candidate that shares the most attribute values with the others, and a model can find it without looking at the
matrix. Hu et al. (2021) showed this and regenerated the candidates so that the vote is balanced. In our own check, a
model given only the eight candidates scores 90–100% on RAVEN and is at chance on I-RAVEN (Chapter 4). All I-RAVEN
experiments here use 10,000 puzzles per layout.

**I-RAVEN-X.** Hersche et al. (2024) extend the single-object layout in two directions: rows of n panels instead of
three, and values from 0 to a maximum of up to 1,000 instead of ten. Arithmetic then sums or subtracts n − 1 operands,
and a model that stores sums has far more of them to store. I-RAVEN-X has no images; a puzzle is its attribute values.
We use rows of 3 with values 0–10, and rows of 10 with values 0–50, 0–100 and 0–1,000.

**What generated puzzles are not.** The tests people take consist of a few dozen hand-made items, ordered from easy to
hard, with no training set. The generated datasets keep the format and part of the rule list. In the RAVEN family the
rules run along rows only, the list of attributes is fixed, and training and test puzzles come from one program. We
measured how much depends on the row direction: after transposing an I-RAVEN matrix, 3% of Progression instances and 1%
of Arithmetic instances still satisfy their rule, against 100% for Distribute Three. Barrett et al. (2018) themselves
call their generated task a different problem from the psychometric test. We therefore draw no conclusion about human
reasoning from any number in this thesis, and return to what the generator contributes in Chapter 8.

## 2.2 Graph neural networks

A graph neural network keeps a vector for every node of a graph and updates it in rounds: each node collects messages
from its neighbours, combines them with an operation that does not depend on their order, such as a sum or a mean, and
passes the result through a learned function (Zhou et al., 2020). The same functions serve every node and every edge.
Two consequences matter here.

First, the network accepts graphs of any size, because its weights belong to the update rule and not to particular
nodes. Second, renumbering the nodes changes nothing: the output of a node moves with the node. This permutation
symmetry is the one symmetry every graph network has by construction. Battaglia et al. (2018) describe the general
point: an architecture expresses assumptions about the data, which they call relational inductive biases, and a graph
network assumes that entities and their relations matter and their order does not.

What a plain graph network does not decide is everything else: which things are nodes, which edges exist, and which
labels the nodes carry. A label such as "row 1" or "the left component", given to a node as an input feature, lets the
network treat that row or component differently from the others. Chapter 4 uses this. Two of our baselines are graph
networks on the same graph, and differ only in whether nodes carry such absolute labels.

Edges join two nodes. A relation among three things, such as u + w = v among three numbers, is a hyperedge, and a
network that passes messages along pairs has to be extended to use it.

## 2.3 Lifted relational neural networks

Lifted relational neural networks (Šourek et al., 2018, 2021) describe a model as a *template*: a set of rules in the
style of logic programming, with weights attached. A rule such as

```
h(X) :- w1 : h(Y), w2 : edge(Y, X)
```

says how the value of `h` at any X is computed from the values at every Y joined to it. The variables stand for any
constant. A *sample* supplies the constants and the facts about them: the objects of one puzzle, their attributes, the
panels they sit in.

**Grounding.** For each sample, the template and the facts are grounded: every way of substituting the sample's
constants for a rule's variables that makes the rule's body derivable becomes one copy of the rule. The copies form a
computation graph, with a neuron for each fact, each ground rule body, each aggregation over the bodies that share a
head, and each derived atom (Šourek et al., 2021, Section 3.2). A different sample gives a different network. The
weights do not change: in the authors' words, they are shared "in a scheme determined by the lifted structure of the
template". Variables that occur only in the body of a rule are aggregated over, which corresponds to quantifying over
them.

**Relation to graph networks.** A message-passing layer is one such rule over `edge` facts, and Šourek et al. (2021)
reproduce standard graph networks as templates. Templates go further in three ways that this thesis uses: rules over
relations of any arity, so a three-way relation is as natural as an edge; rules that derive new relations from the
facts before any learning takes place; and recursive rules, which unfold to whatever depth the sample's structure has.
The authors list attention and gating as constructs that do not fit naturally.

**PyNeuraLogic** is the Python front end of the NeuraLogic engine. All our relational models are written in it and
trained with its own optimizer, on CPU. The engine grounds each puzzle once and keeps the ground networks in memory,
which sets the cost reported in Chapter 5.

The property this thesis builds on is the sentence quoted above. Chapter 3 turns it into a theorem: a template cannot
tell apart two constants that it does not name, so its output is unchanged by any renaming of such constants that maps
the sample's facts onto themselves.

## 2.4 Symmetry as inductive bias

**Bias is necessary.** A learner that prefers no hypothesis over another cannot generalise beyond its training
examples (Mitchell, 1980), and averaged over all problems no learner beats another (Wolpert and Macready, 1997). An
advantage always comes from an assumption that fits the problem. The open question is which assumptions to make, and
the field disagrees on how many: Sutton (2019) argues that general methods with more computation have overtaken
hand-built knowledge again and again, while Marcus (2018) and Lake et al. (2017) argue that objects, relations and
compositional structure have to be built in.

**Symmetry.** Let a group G act on the inputs of a function f. The function is *invariant* if f(g·x) = f(x) for every
g in G, and *equivariant* if f(g·x) = g·f(x), where G also acts on the outputs. A model has a symmetry *built in* when
this holds for every setting of its weights, and has merely *learned* it when it holds, approximately, after training.
Geometric deep learning (Bronstein et al., 2021) reads the successful architectures this way: convolutional networks
are equivariant to translations (Cohen and Welling, 2016†), networks on sets and graphs to permutations (Zaheer et
al., 2017†), and the architecture follows from the choice of domain and group.

**Symmetry is weight sharing.** Ravanbakhsh et al. (2017†) show that a layer is equivariant to a permutation group
exactly when its weights are tied along the orbits of the group, and Kondor and Trivedi (2018) that a layer equivariant
to a compact group must be a convolution over that group. Chapter 3 proves the lifted counterpart: in a template the
tying is done by variables, and the group is whatever the unnamed constants and the background facts leave free.

**What a symmetry is worth.** A symmetry that the target function shares shrinks the set of functions the model can
represent without removing the target, so fewer examples are needed. Sokolić et al. (2017†) bound the generalisation
error of an invariant classifier through the size of the input space after the group's orbits are merged, which for a
finite group of size T is smaller by a factor of up to T. Elesedy and Zaidi (2021) prove a strict generalisation
benefit for equivariant models, and Bietti et al. (2021) give sample-complexity gains
proportional to the size of the group for kernel methods. These results say that the gain grows with the group. They
are stated for function classes and data distributions in general; they do not say what a particular symmetry is
worth on a particular rule, which is what we count in Chapter 3.

**What a wrong symmetry costs.** A model can also have a symmetry that the task lacks. Wang et al. (2023) sort this
into three cases, by what the symmetry does to the data. Equivariance is *correct* where the transformed input has
the transformed label, *incorrect* where the transformed input also occurs in the data but with a different label, and
*extrinsic* where the transformed input lies outside the data distribution. For incorrect equivariance they prove a
lower bound on the error, computed orbit by orbit: an invariant model gives every input of an orbit the same label, so
on each orbit it loses at least the probability mass that does not carry the orbit's commonest label (their Theorem
4.3). They also show that extrinsic equivariance can help or hurt. Petrache and Trivedi (2023†) analyse the trade-off
between the approximation error of a misspecified symmetry and its generalisation gain. The three kinds of symmetry in
our Chapter 3 (data symmetry, out of support, conflicting) are these three cases, reached from the generator's side,
and the ceiling we compute for a conflicting symmetry is their bound evaluated on a rule's table. What we add is the
other half of the count, the data a correct symmetry saves, and matched experiments in which both halves are
predicted before training. Softer forms exist as well: a model can be given a preference for a symmetry and not a
constraint (Finzi et al., 2021†; Wang et al., 2022†). Chapter 6 has a small result of that kind, where a model that
shares all its weights but is not exactly invariant does as well as the exactly invariant one.

**Shortcuts.** A model without the right constraint may fit the training data by a rule that holds only there (Geirhos
et al., 2020). RAVEN's candidate sets are one example. Chapter 6 shows another, in which a model that is allowed to
tell panels apart learns one rule by the panel's address.

**Structure that matches the computation.** Xu et al. (2020) show that a network learns a task more easily when its
structure mirrors the algorithm that solves it, and Xu et al. (2021) that a network extrapolates on an algorithmic
task only when the task's non-linearities are built into the architecture or the features. Chapter 7 gives a case: a
model learns sums from a hundred puzzles once addition is available as a relation, and not at all without it.

**Values on a line without a zero.** A symmetry that Chapter 7 turns on is the shift of all values by a constant.
Stevens (1946) classifies scales by the transformations they survive: on an interval scale the zero is a convention,
and only differences mean something. In geometric terms such values form an affine line, on which points can be
subtracted but not added, and a weighted sum of points is defined without reference to an origin exactly when the
weights sum to one (Baez, 2009). A rule c = pa + qb therefore survives a common shift exactly when p + q = 1. The
midpoint of two values satisfies this and their sum does not.

**Priors and the measurement of skill.** Chollet (2019) argues that skill at a task can be bought with training data
or with built-in knowledge, so a score is informative only together with both. He names the priors a fair test may
assume, among them objects, elementary geometry, and numbers below about ten with addition and subtraction, and he
counts generation from one fixed program as a weakness of a benchmark, since learning the program solves every task.
RAVEN is such a benchmark. We use his terms in Chapter 8 to sort our priors into those a person would bring to any
such test and those read off the generator.

## 2.5 Solvers of Raven's matrices

**Black-box image models.** The first neural solvers map the images to an answer and are trained on the answer's index.
WReN (Barrett et al., 2018) scores each candidate with a relation network over panel embeddings. Zhuo and Kankanhalli
(2020), the reference named in this thesis's assignment, train convolutional networks on the task, with and without
answer labels. Later models add structure that matches the generator: CoPINet contrasts the candidates (Zhang et al.,
2019b†), SRAN encodes rows hierarchically (Hu et al., 2021), and SCL shares networks across objects, attributes and
relations (Wu et al., 2020†). Several of these are symmetry priors in our sense, and we credit them as such. All of
them classify among the candidates. Reported accuracies on I-RAVEN now reach 98–99%, but independent reruns under one
protocol are considerably lower (Małkiński and Mańdziuk, 2022; see the notes below for the figures we have).

**Object-centric and graph models.** A second group first finds the objects and then reasons over them, which is the
plan the assignment asks for. Spratley et al. (2020) attend to objects before relating panels, Wang et al. (2020†)
build a multiplex graph over the objects of a puzzle, and Mondal et al. (2023†) combine slot attention with a
transformer. Our facet model belongs here, with two differences: its graph has one node per attribute of an object,
not per object, and it generates the answer.

**Neuro-symbolic models.** PrAE (Zhang et al., 2021a) recognises the objects of each panel as probability
distributions over attribute values, infers which of the generator's rules each attribute follows, executes the rule to
predict the answer's attributes, and picks the nearest candidate. The rules are given. This is the closest precedent
for our task of generating the answer and then matching, and for the symbolic reference models of Chapter 4. ALANS
(Zhang et al., 2021b) removes the given rules by representing values as matrices and learning operators on them. NVSA
(Hersche et al., 2023) represents attributes as high-dimensional vectors and writes each rule as an expression in
vector-symbolic algebra; ARLC (Camposampiero et al., 2024†) learns such expressions from data. In these models
addition is an operation of the representation, so they compare with our models that are given addition, not with
those that have to learn sums (Chapter 7).

**Language models.** Given the attribute values as text, large language models solve many of these puzzles without
training on them. Hersche et al. (2024) report that GPT-4 answers 93% of I-RAVEN-X puzzles with rows of 3 and 77–82%
with rows of 10, and that on Arithmetic at range 50 it reaches 30% where ARLC reaches 82%.

**Arithmetic in neural networks.** Networks that are not given addition learn it poorly. Trask et al. (2018) find
that networks trained on sums fail outside the trained range and describe the failure as memorisation. Small
transformers trained on modular addition first memorise and, much later, generalise (Power et al., 2022), by
representing each number as an angle so that adding becomes rotating (Nanda et al., 2023). Mohamadi et al. (2024)
prove, for networks in the kernel regime, that a learner that treats values as unnamed labels must see a constant
fraction of the addition table. That is
the situation of a model with separate weights for every value, and in Chapter 7 such a model tracks a learner that
only memorises.

## 2.6 Where this thesis stands

Three things separate this work from the solvers above.

- **The question is the price of a prior, not the score.** The models above each add structure and report accuracy.
  We take one model family and add or remove one symmetry at a time, at several amounts of training data, and compare
  each result with a number computed before training.
- **The symmetry is declared.** In a template a symmetry is the absence of a name, and one theorem covers every
  symmetry built this way. The architectures above are each engineered for their symmetry, and each needs its own
  argument.
- **The answer is generated and the candidates are never seen**, in training or as input. Only PrAE and its successors
  share this, and they are given the rules or the operations.

---

## References added in this chapter

- Baez, J. (2009). Torsors made easy. math.ucr.edu/home/baez/torsors.html.
- Camposampiero, G., Hersche, M., Terzić, A., Wattenhofer, R., Sebastian, A., Rahimi, A. (2024). Towards learning
  abductive reasoning using VSA distributed representations. *NeSy*. †
- Cohen, T., Welling, M. (2016). Group equivariant convolutional networks. *ICML*. †
- Finzi, M., Benton, G., Wilson, A. G. (2021). Residual pathway priors for soft equivariance constraints. *NeurIPS*. †
- Hersche, M., Zeqiri, M., Benini, L., Sebastian, A., Rahimi, A. (2023). A neuro-vector-symbolic architecture for
  solving Raven's progressive matrices. *Nature Machine Intelligence* 5, 363–375.
- Kondor, R., Trivedi, S. (2018). On the generalization of equivariance and convolution in neural networks to the
  action of compact groups. *ICML*.
- Lake, B. M., Ullman, T. D., Tenenbaum, J. B., Gershman, S. J. (2017). Building machines that learn and think like
  people. *Behavioral and Brain Sciences* 40.
- Małkiński, M., Mańdziuk, J. (2022). Deep learning methods for abstract visual reasoning: a survey on Raven's
  Progressive Matrices. arXiv:2201.12382.
- Marcus, G. (2018). Deep learning: a critical appraisal. arXiv:1801.00631.
- Mitchell, T. M. (1980). The need for biases in learning generalizations. Rutgers technical report CBM-TR-117.
- Mohamadi, M. A., Li, Z., Wu, L., Sutherland, D. J. (2024). Why do you grok? A theoretical analysis of grokking
  modular addition. *ICML*.
- Mondal, S. S., Webb, T., Cohen, J. D. (2023). Learning to reason over visual objects. *ICLR*. †
- Nanda, N., Chan, L., Lieberum, T., Smith, J., Steinhardt, J. (2023). Progress measures for grokking via mechanistic
  interpretability. *ICLR*.
- Petrache, M., Trivedi, S. (2023). Approximation-generalization trade-offs under (approximate) group equivariance.
  *NeurIPS*. †
- Power, A., Burda, Y., Edwards, H., Babuschkin, I., Misra, V. (2022). Grokking: generalization beyond overfitting on
  small algorithmic datasets. arXiv:2201.02177.
- Ravanbakhsh, S., Schneider, J., Póczos, B. (2017). Equivariance through parameter-sharing. *ICML*. †
- Sokolić, J., Giryes, R., Sapiro, G., Rodrigues, M. (2017). Generalization error of invariant classifiers.
  *AISTATS*. †
- Šourek, G., Aschenbrenner, V., Železný, F., Schockaert, S., Kuželka, O. (2018). Lifted relational neural networks:
  efficient learning of latent relational structures. *JAIR* 62, 69–100. †
- Spratley, S., Ehinger, K., Miller, T. (2020). A closer look at generalisation in RAVEN. *ECCV*.
- Stevens, S. S. (1946). On the theory of scales of measurement. *Science* 103, 677–680.
- Trask, A., Hill, F., Reed, S., Rae, J., Dyer, C., Blunsom, P. (2018). Neural arithmetic logic units. *NeurIPS*.
- Wang, D., Zhu, X., Park, J. Y., Jia, M., Su, G., Platt, R., Walters, R. (2023). A general theory of correct,
  incorrect, and extrinsic equivariance. *NeurIPS*. arXiv:2303.04745.
- Wang, D., Jamnik, M., Liò, P. (2020). Abstract diagrammatic reasoning with multiplex graph networks. *ICLR*. †
- Wang, R., Walters, R., Yu, R. (2022). Approximately equivariant networks for imperfectly symmetric dynamics.
  *ICML*. †
- Wolpert, D. H., Macready, W. G. (1997). No free lunch theorems for optimization. *IEEE Transactions on Evolutionary
  Computation* 1(1), 67–82.
- Wu, Y., Dong, H., Grosse, R., Ba, J. (2020). The scattering compositional learner: discovering objects, attributes,
  relationships in analogical reasoning. arXiv:2007.04212. †
- Xu, K., Li, J., Zhang, M., Du, S. S., Kawarabayashi, K., Jegelka, S. (2020). What can neural networks reason about?
  *ICLR*.
- Xu, K., Zhang, M., Li, J., Du, S. S., Kawarabayashi, K., Jegelka, S. (2021). How neural networks extrapolate: from
  feedforward to graph neural networks. *ICLR*.
- Zaheer, M., Kottur, S., Ravanbakhsh, S., Póczos, B., Salakhutdinov, R., Smola, A. (2017). Deep sets. *NeurIPS*. †
- Zhang, C., Jia, B., Gao, F., Zhu, Y., Lu, H., Zhu, S.-C. (2019b). Learning perceptual inference by contrasting.
  *NeurIPS*. †
- Zhang, C., Jia, B., Zhu, S.-C., Zhu, Y. (2021a). Abstract spatial-temporal reasoning via probabilistic abduction
  and execution. *CVPR*.
- Zhang, C., Xie, S., Jia, B., Wu, Y. N., Zhu, S.-C., Zhu, Y. (2021b). Learning algebraic representation for
  systematic generalization in abstract reasoning. arXiv:2111.12990 (ECCV 2022).

Already listed with Chapter 1: Barrett 2018, Battaglia 2018, Bietti 2021, Bronstein 2021, Chollet 2019, Elesedy and
Zaidi 2021, Geirhos 2020, Hersche 2024, Hu 2021, Raven 1938, Šourek 2021, Sutton 2019, Zhang 2019, Zhou 2020, Zhuo and
Kankanhalli 2020.

---

## Notes for you (not thesis text)

- **Wang et al. (2023) is a precedent for two things the outline calls yours. Read it first.** Checked on arXiv on
  14 Nov (2303.04745, NeurIPS 2023; title, authors, the definitions in its Section 3 and Theorem 4.3). It defines
  correct, incorrect and extrinsic equivariance, which are your data symmetry, conflicting and out of support. Its
  Theorem 4.3 bounds the error of an invariant model from below by the mass, per orbit, outside the orbit's commonest
  label, which is your ceiling for the value fold; it has regression and equivariant versions too, which I have not
  read and which would be the ones that cover sum under the shift. So outline §3.6 should not say the ceiling is new,
  and the three kinds in §3.5 should carry their names beside yours. What stays yours: the theorem for lifted
  templates, the data-saved side of the count, the matched twins with both numbers predicted, the result that the
  network adds nothing to the count, and the reversal with `plus`. An earlier paper by the same group (Wang et al.,
  ICLR 2023, on latent symmetry) is where I believe the three terms first appear; that one is from memory.
- **Every † citation is from memory.** Venues, years and author lists need checking, and the claims attributed to
  Sokolić et al., Ravanbakhsh et al., Petrache and Trivedi, and Finzi et al. need the papers opened. The arXiv number
  for Wu et al. (SCL) is my best recollection.
- **Still unread in the project, cited for one sentence each:** Zhuo and Kankanhalli (2020), which the assignment
  names, so read it and expand its sentence; PrAE, described from the project's notes; Carpenter et al. (1990), which
  this chapter no longer leans on.
- **Reported accuracies for image models are left out of the text on purpose.** The lab notes have: self-reported on
  I-RAVEN, Triple-CFN 99.1, RS-Tran 98.7, SAViR-T 98.1, DRNet 97.6; independent reruns (A-I-RAVEN, IJCAI 2025), DRNet
  90.9, PredRNet 88.8, SCL 83.4, STSN 59.0. I could not tie these to references I can vouch for, and the sentence
  citing the survey for "reruns are lower" should cite the A-I-RAVEN paper once you have it open.
- **ARLC's training protocol** ("trained on the 2×2 layout only") is in the lab notes and is not repeated here; check
  it in the paper before Chapter 5 mentions ARLC's 92.6.
- **The I-RAVEN-X numbers for GPT-4 and ARLC** are from Hersche et al. (2024), which the reading list marks as read to
  Section VI.
- **Section 2.1's rule table simplifies Arithmetic** to rows of three. For I-RAVEN-X the text says n − 1 operands.
- **The human-test paragraph is deliberately short** (outline §5: one paragraph). The long outline has the full table
  of differences if you want it back.
