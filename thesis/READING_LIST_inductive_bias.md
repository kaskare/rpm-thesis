# Reading list: inductive bias in neural networks

Saved 2026-10-26. Papers on the role of inductive bias, for the thesis framing of the three prior levels
(none / "the same relation holds in every row" / the full rule set). References were written from memory;
the arXiv IDs of Webb, Kerg and Benny were checked, and the others should be confirmed before citing.

Start with the four marked ★: they carry the thesis argument.

## Foundations

- [ ] **Mitchell (1980)**, *The need for biases in learning generalizations*. Rutgers tech report CBM-TR-117.
  Coined the term: without a bias, a learner has no reason to prefer one generalisation over another.
- [ ] **Wolpert & Macready (1997)**, *No free lunch theorems for optimization*. IEEE Trans. Evol. Comput. 1(1).
  Averaged over all problems no learner wins; every advantage comes from a bias that fits the problem.
- [ ] ★ **Battaglia et al. (2018)**, *Relational inductive biases, deep learning, and graph networks*.
  [arXiv:1806.01261](https://arxiv.org/abs/1806.01261). The structure of the model is the prior; GNNs as the
  example. Closest to the thesis framing.
- [ ] **Bronstein, Bruna, Cohen & Veličković (2021)**, *Geometric Deep Learning: Grids, Groups, Graphs,
  Geodesics, and Gauges*. [arXiv:2104.13478](https://arxiv.org/abs/2104.13478). Symmetry as the way to
  build bias into architectures.

## The debate: built-in priors or scale

- [ ] ★ **Sutton (2019)**, *The Bitter Lesson*.
  [incompleteideas.net](http://www.incompleteideas.net/IncIdeas/BitterLesson.html). General methods plus
  compute beat built-in knowledge in the long run.
- [ ] ★ **Marcus (2018)**, *Deep learning: a critical appraisal*
  ([arXiv:1801.00631](https://arxiv.org/abs/1801.00631)) and *Innateness, AlphaZero, and artificial
  intelligence* ([arXiv:1801.05667](https://arxiv.org/abs/1801.05667)). The case for built-in structure.
- [ ] **Lake, Ullman, Tenenbaum & Gershman (2017)**, *Building machines that learn and think like people*.
  Behavioral and Brain Sciences. [arXiv:1604.00289](https://arxiv.org/abs/1604.00289). Priors for objects,
  causality, compositionality.
- [ ] **Goyal & Bengio (2022)**, *Inductive biases for deep learning of higher-level cognition*.
  Proc. R. Soc. A. [arXiv:2011.15091](https://arxiv.org/abs/2011.15091). Which biases would let networks
  reason systematically.

## Structure that matches the algorithm (explains our results)

- [ ] ★ **Xu et al. (ICLR 2020)**, *What can neural networks reason about?*
  [arXiv:1905.13211](https://arxiv.org/abs/1905.13211). Algorithmic alignment: a network learns a task more
  easily when its structure mirrors the algorithm. Explains the copy links (Constant and Distribute-n are
  copies).
- [ ] **Xu et al. (ICLR 2021)**, *How neural networks extrapolate: from feedforward to graph neural networks*.
  [arXiv:2009.11848](https://arxiv.org/abs/2009.11848). MLPs extrapolate linearly; relevant to the
  I-RAVEN-X value ranges.
- [ ] **Veličković & Blundell (2021)**, *Neural algorithmic reasoning*. Patterns.
  [arXiv:2105.02761](https://arxiv.org/abs/2105.02761). With the CLRS benchmark,
  [arXiv:2205.15659](https://arxiv.org/abs/2205.15659).
- [ ] **Webb et al. (2023/2024)**, *The relational bottleneck as an inductive bias for efficient
  abstraction*. Trends in Cognitive Sciences. [arXiv:2309.06629](https://arxiv.org/abs/2309.06629).
  Reasoning over relations instead of features; tested on RPM-like tasks.
- [ ] **Kerg et al. (2022)**, *On neural architecture inductive biases for relational tasks*.
  [arXiv:2206.05056](https://arxiv.org/abs/2206.05056).

## What models learn when the bias is wrong or missing

- [ ] **Locatello et al. (ICML 2019)**, *Challenging common assumptions in the unsupervised learning of
  disentangled representations*. [arXiv:1811.12359](https://arxiv.org/abs/1811.12359). Without biases or
  supervision the true factors are not identifiable.
- [ ] ★ **Geirhos et al. (2020)**, *Shortcut learning in deep neural networks*. Nature Machine Intelligence.
  [arXiv:2004.07780](https://arxiv.org/abs/2004.07780). The general frame for RAVEN's answer-set shortcut.
  Also Geirhos et al. (ICLR 2019), *ImageNet-trained CNNs are biased towards texture*,
  [arXiv:1811.12231](https://arxiv.org/abs/1811.12231).
- [ ] **McCoy, Pavlick & Linzen (2019)**, *Right for the wrong reasons*.
  [arXiv:1902.01007](https://arxiv.org/abs/1902.01007). And **Gururangan et al. (2018)**, *Annotation
  artifacts in natural language inference data*, [arXiv:1803.02324](https://arxiv.org/abs/1803.02324).
  Shortcuts in language datasets.
- [ ] **Lake & Baroni (2018)**, *Generalization without systematicity* (SCAN).
  [arXiv:1711.00350](https://arxiv.org/abs/1711.00350).
- [ ] **D'Amour et al. (2020)**, *Underspecification presents challenges for credibility in modern machine
  learning*. [arXiv:2011.03395](https://arxiv.org/abs/2011.03395).

## Implicit bias: what networks prefer on their own

- [ ] **Zhang et al. (ICLR 2017)**, *Understanding deep learning requires rethinking generalization*.
  [arXiv:1611.03530](https://arxiv.org/abs/1611.03530). Networks memorise random labels.
- [ ] **Neyshabur, Tomioka & Srebro (2015)**, *In search of the real inductive bias*.
  [arXiv:1412.6614](https://arxiv.org/abs/1412.6614).
- [ ] **Rahaman et al. (2019)**, *On the spectral bias of neural networks*.
  [arXiv:1806.08734](https://arxiv.org/abs/1806.08734).
- [ ] **Valle-Pérez, Camargo & Louis (2019)**, *Deep learning generalizes because the parameter-function map
  is biased towards simple functions*. [arXiv:1805.08522](https://arxiv.org/abs/1805.08522).

## RPM-specific

- [ ] **Hu et al. (AAAI 2021)**, *Stratified rule-aware network for abstract visual reasoning* (I-RAVEN).
  [arXiv:2002.06838](https://arxiv.org/abs/2002.06838). Found the answer-set shortcut in RAVEN.
- [ ] **Spratley, Ehinger & Miller (ECCV 2020)**, *A closer look at generalisation in RAVEN*.
- [ ] **Benny, Pekar & Wolf (CVPR 2021)**, *Scale-localized abstract reasoning* (RAVEN-FAIR).
  [arXiv:2009.09405](https://arxiv.org/abs/2009.09405).
- [ ] **Barrett et al. (ICML 2018)**, *Measuring abstract reasoning in neural networks* (PGM).
  [arXiv:1807.04225](https://arxiv.org/abs/1807.04225). Rule labels as meta-targets.
- [ ] **Małkiński & Mańdziuk (2022)**, *Deep learning methods for abstract visual reasoning: a survey on
  Raven's Progressive Matrices*. [arXiv:2201.12382](https://arxiv.org/abs/2201.12382).
