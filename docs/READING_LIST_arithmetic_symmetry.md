# Reading list: does Arithmetic have a symmetry, and why is arithmetic hard for AI?

Saved 2026-11-02. Each entry says how far it was checked: **read** (the passage was opened and read), **abstract** (abstract
page only), **snippet** (search result only, open it before citing). Start with the ones marked ★.

## What the sources support

| statement | standing | source |
|---|---|---|
| Without an origin, points can be subtracted but not added | textbook | Affine space; Baez |
| A weighted sum of points is origin-free when the coefficients sum to 1 | textbook | Affine space |
| If adding a constant changes nothing, the zero is a convention | classic | Stevens 1946 |
| The same map on a, b and c is an **automorphism** | definition | Allsop 2025; Falcón |
| Three maps, one each for a, b and c, are an **autotopism** | definition | Allsop 2025; Falcón |
| Every autotopism of an abelian group is x → θ(x + u), y → θ(y + v), z → θ(z + u + v) | published | Allsop 2025, citing Belousov 1981 |
| On the reals, f(x + y) = g(x) + h(y) gives g = cx + a, h = cx + b, f = cx + a + b | published | Reem 2017, Theorem 9.1 |
| Equivariance to a compact group means convolution over that group | published | Kondor & Trivedi 2018 |

Two steps are ours, from those results:

- **Shifting every value alike is not a symmetry of addition.** With θ the identity, the same map on all three needs
  u = v = u + v, so u = v = 0.
- **Midpoint is addition with its rows and columns relabelled** (an isotopy), on Z_p with p odd:
  (a + b)/2 = a/2 + b/2. So the two have the same symmetries in number, and differ only in which of them use the same map
  on all three.

`analysis/autotopism_check.py` confirms both by brute force, over every candidate triple on Z_5 and Z_7. Z_5:

| rule | triples that keep it | of those, the same map on all three | shifting every value is one of them |
|---|---|---|---|
| addition | 100 | 4 | no |
| midpoint | 100 | 20 | yes |

The sets of triples for addition equal the published form exactly. No source read names Z_n; it is covered as an abelian
group.

## What they do not support

- **"Arithmetic has no symmetry."** False. It has autotopisms, including operand order and moving one operand with the sum.
- **"Arithmetic is hard because it lacks a symmetry."** Nothing read establishes this. Kiani 2024 shows that symmetry alone
  does not remove the hardness of learning by gradient descent. Mohamadi 2024 finds symmetry *in the learner* behind the
  hardness: a learner that treats values as unnamed labels has to see much of the table.
- **Exact statements about our data.** The theorems are about groups. Values 0–50 without wrapping are not a group, so the
  moves hold only while they stay in range. They are exact on the ring, not on 0–50.
- **"Humans look arithmetic up."** This is still debated (Ashcraft against Uittenhove).

"Symmetry" has three meanings in our documents, and they should be kept apart:

1. a rule **is** a symmetry: one transformation takes each panel to the next (Arithmetic is not one, F 29);
2. a rule **has** a symmetry: a transformation of its valid rows leaves them valid (Arithmetic has);
3. a model **has** a symmetry: its output moves with the transformation (the convolution has the shift of every value).

## How others make addition work

Each of these turns adding into composing elements of a group, and each has a zero, given or learned:

| method | a value becomes | adding becomes | zero |
|---|---|---|---|
| network after grokking (Nanda 2023) | an angle | a rotation | angle 0 |
| fractional power encoding (Sun 2025) | a base vector bound with itself x times | binding | no binding |
| ALANS (Zhang 2021) | a matrix power applied to M₀ | matrix products | M₀, learned |
| value axis with `plus` | a constant | a table lookup | `v0`, given |

## Places, steps and zero

- [ ] ★ **Affine space**, [Wikipedia](https://en.wikipedia.org/wiki/Affine_space). Read (first third). "There is no predefined
  concept of adding or multiplying points together." Alice and Bob agree exactly when "the sum of the coefficients in a
  linear combination is 1": our p + q = 1.
- [ ] ★ **Baez (2009)**, [Torsors Made Easy](https://math.ucr.edu/home/baez/torsors.html). Read. "But you can't add elements of
  a G-torsor X." Energy, voltage, phase, temperature and dates as torsors. Picking a point makes the torsor a group.
- [ ] ★ **Stevens (1946)**, *On the theory of scales of measurement*. Science 103. Read. Scales defined by the transformations
  they survive: x' = ax + b for interval scales, multiplying only for ratio scales. "The zero point on an interval scale is
  a matter of convention or convenience." Later critics dispute how far this rules statistics in or out (Velleman &
  Wilkinson 1993, snippet).
- [ ] **Principal homogeneous space**, [Wikipedia](https://en.wikipedia.org/wiki/Principal_homogeneous_space). Read. The formal
  definition of a torsor.
- [ ] **Level of measurement**, [Wikipedia](https://en.wikipedia.org/wiki/Level_of_measurement). Read (first third).

## Symmetries of an operation

- [ ] ★ **Latin square**, [Wikipedia](https://en.wikipedia.org/wiki/Latin_square). Read (first half). Addition tables are Latin
  squares. Conjugates permute the roles of row, column and symbol, which is how a + b = c also gives c − b = a.
- [ ] ★ **Allsop (2025)**, *Isotopisms of quadratic quasigroups*.
  [arXiv:2506.02446](https://arxiv.org/abs/2506.02446). Read. The paragraph after Theorem 1.4 states the autotopism group of an
  abelian group, G² ⋊ Aut(G). The rest of the paper is about other quasigroups.
- [ ] **Small Latin squares and quasigroups**,
  [Wikipedia](https://en.wikipedia.org/wiki/Small_Latin_squares_and_quasigroups). Read. The clearest definition of isotopism.
- [ ] **Falcón (2012)**, *Cycle structures of autotopisms of the Latin squares of order up to 11*. Ars Combinatoria 103.
  [arXiv:0709.2973](https://arxiv.org/abs/0709.2973). Read for the definitions.
- [ ] **Belousov (1981)**, *Elements of quasigroup theory: a special course*. The source Allsop cites. Not read.
- [ ] **Cauchy's functional equation**, [Wikipedia](https://en.wikipedia.org/wiki/Cauchy%27s_functional_equation). Read. The maps
  that keep addition with one map are x → cx, so scaling, not shifting.
- [ ] **Reem (2017)**, *Remarks on the Cauchy functional equation and variations of it*.
  [arXiv:1002.3721](https://arxiv.org/abs/1002.3721). Read section 9. Pexider's equation, the real-valued form of the triples.

## Symmetry and learning

- [ ] ★ **Kondor & Trivedi (2018)**, *On the generalization of equivariance and convolution in neural networks to the action of
  compact groups*. ICML. [arXiv:1802.03690](https://arxiv.org/abs/1802.03690). Abstract and theorem statement. Why a shift-symmetric
  layer has to be a convolution.
- [ ] ★ **Kiani, Le, Lawrence, Jegelka & Weber (2024)**, *On the hardness of learning under symmetries*.
  [arXiv:2401.01869](https://arxiv.org/abs/2401.01869). Abstract. Are symmetries enough to remove the hardness of learning? "We
  answer this question in the negative." It covers complete function classes, not addition.
- [ ] ★ **Mohamadi, Li, Wu & Sutherland (2024)**, *Why do you grok? A theoretical analysis of grokking modular addition*. ICML.
  [arXiv:2407.12332](https://arxiv.org/abs/2407.12332). Read to section 4.2. "No permutation-equivariant model can achieve small
  population error on modular addition unless it sees at least a constant fraction of all possible data points", for kernel
  methods near initialization. Rich networks need far fewer. Closest theory to our weights-per-value model tracking
  memorising; the setting differs, so it is an analogy.
- [ ] ★ **Dawes, Segert, Krishnamurthy & Cohen (2025)**, *A group theoretic analysis of the symmetries underlying base addition
  and their learnability by neural networks*. [arXiv:2507.10678](https://arxiv.org/abs/2507.10678). Read to section 6. Carrying as
  a cocycle. Learnability correlates with the carry function's structure, and recurrence helps through "weight sharing over
  time", like our running sum. It is about digits, and the link is a correlation across carry functions.

## How networks learn addition

- [ ] ★ **Nanda, Chan, Lieberum, Smith & Steinhardt (2023)**, *Progress measures for grokking via mechanistic interpretability*.
  [arXiv:2301.05217](https://arxiv.org/abs/2301.05217). Abstract. Small transformers "convert addition to rotation about a circle".
- [ ] **Chughtai, Chan & Nanda (2023)**, *A toy model of universality*.
  [arXiv:2302.03025](https://arxiv.org/abs/2302.03025). Abstract. Composition in any finite group through representation theory.
- [ ] **Morwani, Edelman, Oncescu, Zhao & Kakade (2023)**, *Feature emergence via margin maximization: case studies in algebraic
  tasks*. ICLR 2024. [arXiv:2311.07568](https://arxiv.org/abs/2311.07568). Abstract. Proves the Fourier features in stylized networks.
- [ ] **Power, Burda, Edwards, Babuschkin & Misra (2022)**, *Grokking*.
  [arXiv:2201.02177](https://arxiv.org/abs/2201.02177). Abstract. Where this line of work started.
- [ ] **He, Wang, Chen & Yang (2026)**, *On the mechanism and dynamics of modular addition*.
  [arXiv:2602.16849](https://arxiv.org/abs/2602.16849). Snippet.

## Networks and numbers

- [ ] ★ **Trask, Hill, Reed, Rae, Dyer & Blunsom (2018)**, *Neural arithmetic logic units*.
  [arXiv:1808.00508](https://arxiv.org/abs/1808.00508). Read. The failure "is better characterized by memorization than by systematic
  abstraction".
- [ ] ★ **Xu, Zhang, Li, Du, Kawarabayashi & Jegelka (2020)**, *How neural networks extrapolate*.
  [arXiv:2009.11848](https://arxiv.org/abs/2009.11848). Abstract. GNNs extrapolate algorithmic tasks by "encoding task-specific
  non-linearities in the architecture or features", which is what `plus` does.
- [ ] **Madsen & Johansen (2019)**, *Measuring arithmetic extrapolation performance*.
  [arXiv:1910.01888](https://arxiv.org/abs/1910.01888). Abstract. 4,800 experiments: "consistently learning arithmetic extrapolation
  is challenging".
- [ ] **Saxton, Grefenstette, Hill & Kohli (2019)**, *Analysing mathematical reasoning abilities of neural models*.
  [arXiv:1904.01557](https://arxiv.org/abs/1904.01557). Abstract.
- [ ] **Kaiser & Sutskever (2015)**, *Neural GPUs learn algorithms*
  ([arXiv:1511.08228](https://arxiv.org/abs/1511.08228)), and **Price, Zaremba & Sutskever (2016)**, *Extensions and limitations
  of the Neural GPU* ([arXiv:1611.00736](https://arxiv.org/abs/1611.00736)). Snippets. Long addition from a repeated local step.

## Raven's matrices

- [ ] ★ **Zhang, Xie, Jia, Wu, Zhu & Zhu (2021)**, *Learning algebraic representation for systematic generalization in abstract
  reasoning* (ALANS). ECCV 2022. [arXiv:2111.12990](https://arxiv.org/abs/2111.12990). Read to section 5. Values as matrix powers,
  "the zero element as a learnable matrix", operators induced by regression.
- [ ] ★ **Hersche, Camposampiero, Wattenhofer, Sebastian & Rahimi (2024)**, *Towards learning to reason: comparing LLMs with
  neuro-symbolic on arithmetic relations in abstract reasoning*. [arXiv:2412.05586](https://arxiv.org/abs/2412.05586). Read to
  section VI. Introduces I-RAVEN-X. On 3×10 at range 50, arithmetic accuracy is 81.6 for ARLC, 30.4 for GPT-4 and 1.5 for
  Llama-3. ARLC's addition comes from its encoding, so it compares with the value axis with `plus`, not with models that
  learn the sum (our reading).
- [ ] **Sun et al. (2025)**, *Systematic abductive reasoning via diverse relation representations in vector-symbolic
  architecture*. [arXiv:2501.11896](https://arxiv.org/abs/2501.11896). Read in part. "Addition + in the real number domain can be
  represented by the binding operation."
- [ ] **Hersche et al. (2023)**, neuro-vector-symbolic architecture for Raven's matrices. Nature Machine Intelligence.
  [arXiv:2203.04571](https://arxiv.org/abs/2203.04571). Snippet.

## Why language models get arithmetic wrong

Mostly about digits, tokens and carries, so a different problem from ours, where each value is one symbol.

- [ ] ★ **Baeumel, van Genabith & Ostermann (2025)**, *The lookahead limitation: why multi-operand addition is hard for LLMs*.
  BlackboxNLP. [arXiv:2502.19981](https://arxiv.org/abs/2502.19981). Snippet. A one-digit lookahead works for two operands and fails
  for more, which is the case closest to our nine numbers.
- [ ] **Bie, Dick, Kulesza, Raghavan, Raman & Vassilvitskii (2026)**, *AI-rithmetic*.
  [arXiv:2602.10416](https://arxiv.org/abs/2602.10416). Read the abstract and introduction. Frontier models fail on digits through
  misalignment and carrying.
- [ ] **Singh & Strouse (2024)**, *Tokenization counts*. [arXiv:2402.14903](https://arxiv.org/abs/2402.14903). Abstract.
- [ ] **Wang, Chen, Yu & Xiang (2025)**, *How do autoregressive transformers solve full addition?* EMNLP. Read the abstract.
  [PDF](https://aclanthology.org/2025.emnlp-main.643.pdf). Attention heads that handle carries of different lengths.
- [ ] **Mirzadeh et al. (2024)**, *GSM-Symbolic*. ICLR 2025. [arXiv:2410.05229](https://arxiv.org/abs/2410.05229). Snippet. Sensitive to
  changed numbers and irrelevant clauses.

## People

- [ ] **Ashcraft (1992)**, *Cognitive arithmetic: a review of data and theory*. Cognition 44. Snippet. Facts retrieved from an
  associative memory network.
- [ ] **Uittenhove, Thevenot & Barrouillet (2016)** and **Chen & Campbell (2018)**. Snippets. Fast compiled counting against
  retrieval. So "lookup" is not settled for people either.
