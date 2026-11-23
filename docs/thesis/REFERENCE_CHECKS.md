# Reference checks for the thesis drafts (2026-11-23)

This file goes with `references.bib`. It records, for every work cited as (Author, Year) in
`abstract_introduction.md` and `ch2` to `ch8`, what was checked and where, and every sentence in the drafts that
the source does not support as written. The chapter files were not edited; Section 8 lists the changes they need,
chapter by chapter, with the sentence quoted.

Bibliographic details were checked against arXiv (its API and the PDFs), Crossref (journals, Springer, IEEE, AAAI,
IJCAI and the NeurIPS proceedings from 2022 on), the PMLR volume pages (ICML, AISTATS, UAI), the NeurIPS proceedings
index (2017 to 2021) and OpenReview (ICLR). DBLP refuses automated clients, so it was not used. Where the drafts cite
a paper for what it proves or says, that part of the paper was read in the PDF; page numbers below are those of the
version named. Each `.bib` entry carries one of three marks in the comment above it:

- **checked**: the details are right and, where the drafts cite the work for a claim, the source supports it;
- **corrected**: the drafts' reference list or a sentence in the text differs from the source, and the entry in
  `references.bib` is the corrected one;
- **unchecked**: the source could not be read, with the reason.

Where only the bibliographic details and the abstract were checked, the table in Section 7 says so.

**Labels for the two Wang et al. papers of 2023.** The chapters now use 2023a for the NeurIPS paper ("A general
theory…", `wang2023general`) and 2023b for the ICLR paper ("The surprising effectiveness…", `wang2023surprising`).
This file follows that: where it writes "Wang et al. (2023)" without a letter it means the NeurIPS paper. One caution,
not verified here: biblatex sorts these two by title and puts the NeurIPS paper first, but natbib's `plainnat` sorts
by the full author list, where the ICLR paper (second author Park) may come before the NeurIPS paper (second author
Zhu). In LaTeX the letters come from the style, so cite by key and check the letters in the compiled PDF.

---

## 1. The findings that change the text, most important first

1. **Wang et al. (2023) cover the ceiling, through their Theorem 4.3, but not as the drafts cite it.** Their
   equivariant bound is for regression (squared error). The 7.9% follows from their *classification* theorem for
   invariant models, once the answer is written relative to the moved input (c − b for the shift). The drafts should
   say this in one line and cite Theorem 4.3, not "the equivariant form". The data-saved half, the template's reading
   of the puzzle (Statement 5), and the lifted-template theorem are not in the paper. Details in Section 2.
2. **"GPT-4 answers 93% of I-RAVEN-X puzzles with rows of 3" (Ch2 2.5) is wrong.** The 93.2% is on I-RAVEN's centre
   layout (500 test puzzles). All of Hersche et al.'s I-RAVEN-X results are on rows of 10.
3. **PrAE does see the candidates in training.** It is trained by a cross-entropy over the eight candidates, with the
   ground-truth rules as an auxiliary loss (its Section 3.3). Ch2 2.6 says that only PrAE and its successors share
   "the candidates are never seen, in training or as input". They share the generate-then-match design, not that.
4. **Sokolić et al. (2017): the error bound shrinks by up to √T, not T.** What shrinks by up to T is the covering
   number of the input space, or equivalently the number of samples needed for the same bound.
5. **"The effective number of samples is multiplied by the group's size" (Ch3 3.8) holds for Tahmasebi and Jegelka
   and for Bietti et al., not for Elesedy and Zaidi or Sannai et al.** The last two prove gains of other forms.
6. **Wolpert and Macready (1997) is about optimisation algorithms.** For "averaged over all problems no learner beats
   another" the source is Wolpert (1996), *Neural Computation*.
7. **The survey of Małkiński and Mańdziuk does not say that reruns are lower.** Their A-I-RAVEN paper (IJCAI 2025)
   does, with the numbers in the lab notes. The survey has also been published, in *ACM Computing Surveys* (2025).
8. **Baez's torsor page does not contain the weights-sum-to-one statement** that Ch2 2.4 cites it for. Gallier's
   *Basics of Affine Geometry* (Lemma 2.2.1) does.
9. **"They also show that extrinsic equivariance can help or hurt"**: the NeurIPS paper shows the harm; the help is
   their ICLR 2023 paper, which also introduced the three names. It needs its own entry (`wang2023surprising`).
10. Smaller corrections: ALANS is ECCV 2022 (cite as 2022, not 2021b); the ICML title of Mohamadi et al. reads
    "Analysis *on*"; the first author of the multiplex-graph paper is *Duo* Wang; Hersche et al. (2024) now has a
    journal version (2026); I-RAVEN-X values run from 0 to m − 1, not to m.

---

## 2. Wang et al. (2023), *A general theory of correct, incorrect, and extrinsic equivariance*

Read: arXiv 2303.04745v2 (the NeurIPS 2023 camera-ready), Sections 1 to 7 and Appendices A to C. Authors, title and
venue in the drafts are right; Crossref gives NeurIPS 36, pp. 40006–40029.

### What Section 3 says

The setting (p. 3): a ground-truth function f from inputs to labels, a density p over the inputs, and a model class
that is "arbitrarily expressive except that it is constrained to be equivariant with respect to a group G". The
error is the 0–1 error for classification and the squared error for regression, both under p.

Definitions 3.1 to 3.3 (p. 3) classify a model against f as a whole; Definitions 3.5 to 3.7 (pp. 3–4) do it
pointwise, for a pair (x, g) with p(x) > 0:

- **correct** if p(gx) > 0 and f(gx) = g f(x);
- **incorrect** if p(gx) > 0 and f(gx) ≠ g f(x);
- **extrinsic** if p(gx) = 0.

Definition 3.9 splits all pairs into a correct, an incorrect, an extrinsic and an undefined set. The paper says on
p. 2 that the three names come from the same group's earlier paper (Wang, Park, Sortur, Wong, Walters and Platt,
ICLR 2023, their reference [58]) and that the pointwise version is the new part. The note in Ch2 that the terms first
appear there is therefore right.

### What Section 4 says

- **Theorem 4.3** (p. 5). For a G-*invariant* classifier, the error is at least the integral over a fundamental
  domain of the *total dissent* of each orbit: the probability mass of the orbit that does not carry the orbit's
  majority label (Definition 4.2; footnote 3 says the majority label need not have more than half the mass). The
  density p need not be invariant. The bound is tight "assuming universal approximation": some invariant model
  attains it. For a discrete group the integral is a sum (Appendix B, equation 8).
- **Proposition 4.4** (p. 5) writes the dissent through the incorrect set. **Proposition 4.7** (p. 6): for a finite
  group whose action preserves the density, the error is at least 1 − Σ q·c_q, where q is the share of an orbit
  that carries its majority label; this is the earlier paper's Proposition 4.1, which Theorem 4.3 generalises.
- **Theorem 4.8** (p. 6), invariant regression: the squared error is at least the probability-weighted variance of f
  over each orbit.
- **Theorem 4.9 and Corollary 4.10** (p. 7), equivariant regression: the same with f first "G-stabilised", that is,
  each output moved back by the inverse transformation, g⁻¹f(gx), before the variance is taken.

There is **no theorem for equivariant classification**. The abstract's "invariant or equivariant networks in
classification or regression settings" means invariant classification, invariant regression and equivariant
regression; the Discussion (p. 10) lists exactly these three.

Section 5 (pp. 7–8) shows by an example that extrinsic equivariance can make a problem harder (lower Rademacher
complexity, an exclusive-or the invariant model cannot fit). Section 6 (pp. 8–10) finds trained networks close to
the bounds: a D4-invariant network on printed digits, bound 90%, measured 92.5% (a rotated 9 is not exactly a 6 in
every font); a flip-equivariant robot policy, bound 0.5 + 0.5c, matched closely. The Discussion names the main
limitation: the bounds need the density over the domain.

### What it already covers of the drafts' ceiling argument (Ch3 3.6.3, Ch7 7.3)

- **The ceiling itself.** Statement 4 counts, for sum under the shift, one right row per orbit: 97 of 1,225, 7.9%.
  That is Theorem 4.3 evaluated on the uniform distribution over sum's question rows, after one step the paper does
  not write out for classification: a learner with the shift, f(a + s, b + s) = f(a, b) + s, is the same as an
  invariant learner of the offset f(a, b) − b, and it is wrong exactly when that offset is wrong. With the offset
  c − b as the label (for sum, c − b = a, which differs on every row of an orbit), Theorem 4.3 gives the count.
  This is the paper's own "G-stabilised" label of Section 4.3, used with the 0–1 error instead of the squared error.
- **Tightness.** The paper's remark that the bound is attained by some invariant model is the drafts' "a learner
  that only counts, one answer per orbit, realises it".
- **The line 0 to 50.** Theorem 4.3 needs no invariant density. That matters here: on the line, the shift moves many
  sum rows out of the table (those pairs are pointwise extrinsic) and others onto different sum rows (pointwise
  incorrect). The theorem handles the mixture; Proposition 4.7 would not, since the uniform table is not
  shift-invariant.
- **The value reflection of Ch6 6.7** ("about half of all values"). This is the paper's own case: a group of two
  with labels that differ inside an orbit, like their 6 and 9. Proposition 4.7 or Theorem 4.3 gives it directly.
- **The three kinds of Ch3 3.5.** Conflicting is incorrect equivariance and out of support is extrinsic, exactly.
  The pointwise version is what the drafts need, since one move can be correct on one rule and incorrect on another
  (the shift on midpoint and on sum). The data symmetry is a special case of correct equivariance: their definition
  asks only that g·x occur in the data, not that it occur as often as x.
- **Statement 5's two cases.** Shifting one row of a sum puzzle gives a valid question row with another answer
  (incorrect); shifting the whole puzzle gives a puzzle the generator never produces (extrinsic). Both readings agree
  with the definitions. On an orbit with only one point in the data the total dissent is zero, so the precise form of
  "the bound does not apply" is "the bound is zero there".
- **The check that trained models sit at the bound** has a precedent in their Section 6.

### What it does not cover

- **The data a correct symmetry saves.** The paper bounds the error that remains with unlimited data. It says so on
  p. 3 ("Our work is complimentary to these and quantifies approximation error"), leaving generalisation to Elesedy
  and Zaidi and others. Statement 3, the coverage formula, and the 2,878 against 107 rows are not in it.
- **Which group a model has.** The paper assumes the model is exactly equivariant to G and otherwise unrestricted.
  It has nothing on the step that Statement 5 needs: the template's built-in symmetry is the shift of the whole
  puzzle, under which sum puzzles are extrinsic and the bound is zero, and the 7.9% applies to the template only
  because its rules read the question row through offsets and never measure across rows. That argument, and the
  48 coincidences it leaves, are the drafts' own; it is also why the ceiling for the template is a prediction from
  the rules and not a theorem.
- **Finite-data behaviour.** Nothing on which unseen rows a trained model answers (Statement 11, the 0 of 303).
- **The renaming theorem**, the matched twins with both numbers predicted, the reversal with `plus`, and the ladder.
- **Equivariant classification** as such (the one-line rewriting above is needed).

The paper's stated limitation, that the density must be known, is worth turning around in the thesis: a generated
table is a setting where the density is known exactly, so the bound can be computed before training.

### Text changes for this paper

See Section 8, items 1.2, 1.3, 2.10 to 2.12, 3.1 to 3.4 and 7.1. In short: cite Theorem 4.3 by number, add the
one-line rewriting to the offset, say "a case of" correct equivariance, replace "the bound does not apply" by "the
bound is zero there", credit the help of extrinsic equivariance to the ICLR paper, and add the citation in Ch7 7.3.

---

## 3. Zhuo and Kankanhalli (2020), *Solving Raven's Progressive Matrices with neural networks*

Read: arXiv 2002.01646v2, all nine pages. An arXiv preprint; no peer-reviewed venue was found.

**What it does and finds.** The paper treats RAVEN (the original dataset, not I-RAVEN) as image classification. It
stacks the sixteen panels, eight of the matrix and eight candidates, as the input channels of a ResNet and trains it
with a cross-entropy on the answer's index (Section 3.1). Its subject is over-fitting. ImageNet pre-training and depth
help: a pre-trained ResNet-50 answers 86.3% of RAVEN's test puzzles, against 70.6% without pre-training, 77.2% for
ResNet-18, and the 84.4% human score that RAVEN reports; CoPINet's 91.4% is the best in their table (Tables 1 and 2).
Trained on one layout and tested on others, the ResNet-50 drops: from 60.8% on Center to 41–44% on the left-right,
up-down and in-out layouts (Table 3), and the authors conclude that supervised models "heavily rely on the training
data" and do not generalise to new problems (Section 5). The paper also proposes the first unsupervised method,
MCPT: each candidate completes a third row, a network scores all ten rows, and a two-hot target says that the two
complete rows should score high. The training signal is thus only the assumption that the third row obeys the same
rules as the first two. MCPT reaches 28.5% without any labels, against 12.5% for guessing (Table 6).

Three points matter for the thesis. The numbers are on RAVEN, before Hu et al. (2021) showed that its candidates give
the answer away, so part of the 86% may come from that. The paper varies no training-set size, so it says nothing
about data efficiency. And MCPT's target is a structural prior of exactly the kind the thesis builds in (the same
rules in every row), used here as the only supervision.

**A sentence for Ch2 2.5** (replacing "train convolutional networks on the task, with and without answer labels"):

> Zhuo and Kankanhalli (2020), the reference named in this thesis's assignment, give a ResNet the sixteen panels as
> one stack of images and train it on the answer's index. With ImageNet pre-training a ResNet-50 answers 86% of
> RAVEN's test puzzles, above the human score that RAVEN reports, but it falls by 17 to 19 points when tested on
> layouts it was not trained on. Without answer labels, trained only on the assumption that the third row obeys the
> rules of the first two, a network reaches 28.5%, against 12.5% for guessing. Their results are on RAVEN, before
> its candidate shortcut was found.

The sentence in Ch1 1.1 ("networks that map the sixteen images of a puzzle to the index of the answer") describes the
paper correctly and can stay.

---

## 4. Chollet (2019), *On the measure of intelligence*: pages and quotations

Read: arXiv 1911.01547v2. The printed page numbers equal the PDF page numbers.

| where in the drafts | quotation or claim | source | verdict |
|---|---|---|---|
| Ch8 8.4 | "unlimited priors allow system developers to 'buy' unbounded performance on any given task" | p. 25, §II.1.3 | verbatim (the sentence goes on: "with no implications with regard to generalization abilities") |
| Ch8 8.4 | numbers "may be added or subtracted" | p. 26, §II.1.3, the Core Knowledge item "Natural numbers and elementary arithmetic" | verbatim |
| Ch8 8.4 | quantities "smaller than approximately 10" in his own benchmark | p. 49, §III.1.2: "All quantities featured in ARC are smaller than approximately 10." | verbatim |
| Ch8 8.4 | generating every task from one program is "a weakness", since reverse-engineering the program would solve them all | p. 51, §III.1.3: "We perceive programmatic generation from a static 'master' program as a weakness, as it implies that merely reverse-engineering the generative program shared across tasks … would be sufficient to fully solve all tasks." | verbatim; the paraphrase is faithful |
| Ch1 1.1 | skill can be bought with unlimited data or unlimited priors; what is informative is the experience needed, given the priors | abstract p. 1; p. 19, §II.1.1 (training data as the other way to "buy" performance); p. 43, §II.2.3 (a comparison needs shared priors and compares experience-efficiency) | supported |
| Ch2 2.4 | "He names the priors a fair test may assume, among them objects, elementary geometry, and numbers below about ten with addition and subtraction" | p. 26 lists four areas: objectness and elementary physics, agentness and goal-directedness, natural numbers and elementary arithmetic, elementary geometry and topology. "Below about ten" is from ARC, p. 49 | merges two places; see change 2.13 |
| Ch2 2.4 | generation from one fixed program counted as a weakness | p. 51 | supported |
| Ch7 7.8 | "asks that skill be reported together with the priors and the experience behind it" | abstract p. 1 ("priors, and experience, as critical pieces to be accounted for"); p. 43 | supported |
| Ch8 8.4 | "A fair test may assume a short list of them: objects, elementary geometry, and numbers" | p. 26 has four items | add "among them", or name agentness too |

Ch8's opening line says the quotations were checked on 9 November; they were rechecked against v2 on 23 November and
all four are exact.

---

## 5. The sample-complexity papers (Ch1 1.1, Ch2 2.4, Ch3 3.8, Ch6 6.2)

**Sokolić et al. (2017).** Read: arXiv 1610.04574v3, Sections 1 to 3 (pp. 1–5). Their Theorems 1 and 2 bound the
generalisation error by a term of the form √(N/m), with N the covering number of the input space (non-invariant) or of
the *base space* left once the transformations are factored out (invariant). Theorem 3 (p. 5) shows that the base
space's covering number is at most 1/T of the input space's, for a set of T transformations that keep the transformed
copies apart and do not shrink distances. The contributions (p. 2) and Remark 1 (p. 5) state the result: the
generalisation error of an invariant method "may be up to a factor √T smaller". So the drafts' "smaller by a factor
of up to T" is right for the covering number and wrong for the error. Read in samples, the invariant learner reaches
the non-invariant bound with up to T times fewer examples. T is a set of transformations, not necessarily a group, and
the input space must factor as transformations times base space. **Corrected.**

**Elesedy and Zaidi (2021).** Read: arXiv 2102.10333v2 (the ICML version is shortened), Sections 1, 2 and 6. Lemma 6
(p. 6) is general: for squared loss, an input distribution invariant under a compact group and an equivariant target,
averaging any predictor over the group lowers its test error by the squared norm of its anti-symmetric part. It is
therefore strictly better unless the predictor is already equivariant. Theorem 7 (invariant, p. 7) and Theorem 13
(equivariant, p. 10) give the exact gap for minimum-norm least squares with Gaussian inputs, σ²·dim A/(n − d − 1) when
n > d + 1. Here dim A is the dimension of the linear maps the symmetry rules out. "A strict generalisation benefit for
equivariant models" (Ch2 2.4) is right. "Needs fewer examples" (Ch1 1.1) follows in their linear setting (lower error
at every n) but is not how they state it. The gain scales with dim A, not with the group's size, so the paper does
not belong under "the effective number of samples is multiplied by the group's size" (Ch3 3.8). **Checked**, with
that one misattribution.

**Bietti et al. (2021).** Read: arXiv 2106.07148v2, abstract and Section 1 (pp. 1–2). For kernel ridge regression on
the sphere with a kernel invariant to a finite group of permutations of the coordinates, the sample complexity
improves "by a factor equal to the size of the group", valid "when the sample size is large enough"; the paper
extends this to stability under subsets of permutations. "Sample-complexity gains proportional to the size of the
group for kernel methods" (Ch2 2.4) is right; adding "for large enough samples" would be exact. The NeurIPS title is
*On the sample complexity of learning under geometric stability*, as in the drafts; arXiv v2 adds "invariance and".
**Checked.**

**Tahmasebi and Jegelka (2023).** Read: arXiv 2303.14269v2, abstract and Sections 1 and 4 (Theorem 4.1 and
equation 7, p. 5). For kernel ridge regression on a compact manifold, with a target invariant under a Lie group,
they give minimax-optimal rates. For a finite group acting effectively the gain "effectively multiplies the number of
samples by the group size": n samples act like n·|G|. For groups of positive dimension the gain is a lower dimension
in the exponent. The Ch3 note that this came from a search summary can be dropped; the claim is in the abstract and
in Theorem 4.1. **Checked.**

**Sannai et al. (2021)**, cited with them in Ch3 3.8. Read: abstract. The bound improves through the volume of a
quotient feature space; for DeepSets by √(n!). This is again a square root in an error bound, not a multiplication of
samples. **Checked**, same misattribution as Elesedy and Zaidi.

What this means for the text: "for a finite group the effective number of samples is multiplied by the group's size"
should cite Tahmasebi and Jegelka (2023) and Bietti et al. (2021), both for kernel methods. Sokolić et al. give the
same factor read in samples, through a bound. Elesedy and Zaidi and Sannai et al. prove gains of other forms. Ch6 6.2's
"Section 2.4 says a symmetry saves data by a factor of up to the size of its group" is then supported, for kernel
methods; Ch6's own note suggests "is consistent with" for the next sentence, and that is the right strength.

---

## 6. The RAVEN papers and Stevens

**Hersche et al. (2024).** Read: arXiv 2412.05586v1, Sections I, II.B, IV.E, V and Tables II, IV, V, VI. I-RAVEN-X
(Section II.B, p. 3) extends I-RAVEN's centre layout with a configurable number of columns g and a value range m.
Values lie in [0, m − 1]. For Arithmetic the first g − 1 panels are added to give the last; for minus, the first
column is the sum of the others, so "sums or subtracts n − 1 operands" (Ch2 2.1) is right. The paper uses I-RAVEN-X
only as attribute values in text prompts and does not render images. The numbers:

| | rows of 3, range 5–10 (I-RAVEN centre) | rows of 10, m = 50 | m = 100 | m = 1000 |
|---|---|---|---|---|
| GPT-4, task accuracy (Table V, p. 10) | 93.2 | 82.2 | 79.6 | 76.6 |
| GPT-4, Arithmetic (Table VI) | 73.6 | 30.4 | 25.1 | 8.4 |
| ARLC learned, Arithmetic (Table VI) | 99.9 | 81.6 | 78.8 | 61.6 |
| ARLC programmed, Arithmetic (Table VI) | 100.0 | 99.8 | 100.0 | 99.5 |

So "77–82% with rows of 10" and "30% where ARLC reaches 82%" are right (for the learned ARLC). "93% of I-RAVEN-X puzzles
with rows of 3" is not: the 93.2% is on 500 test puzzles of I-RAVEN's centre layout. ARLC on I-RAVEN-X is trained at
range 50 and only reconfigured for larger ranges (p. 9; p. 8 says 45). Crossref lists a journal version, *Neurosymbolic
Artificial Intelligence* 2 (2026), doi:10.1177/29498732261419316; it was not read, and the `.bib` keeps the preprint
the drafts cite with a note. **Corrected** (text).

**Camposampiero et al. (2024), ARLC.** Read: arXiv 2406.19121v3, abstract, Sections 3 and 4.1, Table 2 (p. 11).
ARLC is trained on the 2×2 layout only and tested on all seven ("exclusively trained on the 2x2 constellation", p. 11;
PrAE is the other model in the table trained this way). It is given the panels' ground-truth attribute labels
("Following previous works which assume a perfect perception", Section 3). Average accuracy on I-RAVEN: 92.4 with
rules programmed, **92.6 ± 0.2 with rules programmed and then trained**, 92.4 ± 1.5 learned from scratch. The lab
notes' "92.6, trained on 2×2 only" is right but belongs to the programmed-then-trained version. If Chapter 5 cites it,
say so, and say that it is given the attributes. Details checked on Crossref (Springer LNCS, NeSy 2024, pp. 370–385).
**Checked.**

**Zhang et al. (2021a), PrAE.** Read: arXiv 2103.14230v2, abstract and Section 3 (pp. 3–6). The description in Ch2
2.5 is right. An object CNN gives attribute distributions for all sixteen panels. A symbolic backend abduces rules from
a given set of logical constraints ("we assume a set of symbolic logical constraints describing rules is available")
and executes them to predict the answer's attributes, and the candidate with the smallest Jensen–Shannon divergence is
chosen. But training (Section 3.3, p. 5) turns those divergences into a distribution over the eight candidates and
minimises a cross-entropy with the answer's index, by REINFORCE, plus an auxiliary loss on the ground-truth rules.
No attribute labels are used. So PrAE sees the candidates in training and as input. **Corrected** (Ch2 2.6).

**Małkiński and Mańdziuk (2025), A-I-RAVEN and I-RAVEN-Mesh** (IJCAI 2025, pp. 5932–5940). Read: the proceedings
PDF, Section 4 and Table 2 (p. 5). Thirteen models retrained under one protocol (Adam, learning rate 10⁻³ with a
plateau schedule, no model-specific tuning, no augmentation, three seeds, 42,000 / 14,000 / 14,000 puzzles, every
model with an auxiliary rule loss). Test accuracy on I-RAVEN, rerun against reported: DRNet 90.9 against 97.6,
PredRNet 88.8 against 96.5, SCL 83.4 against 95.0, CPCNet 70.4 against 98.5, STSN 59.0 against 95.7. These are the lab
notes' figures. Not every rerun is lower: MRNet gets 86.7 against a reported 83.5. The highest *reported* figure in
the table is 98.5 (CPCNet); the 99.1 of Triple-CFN in the lab notes is not in this paper and was not checked.
**Checked**; it should replace the survey as the source for "reruns are lower".

**Małkiński and Mańdziuk, survey.** arXiv 2201.12382v2 (2022) was searched for statements about reruns or
reproducibility; there are none; it collects reported numbers. Published as *ACM Computing Surveys* 57(7), article
166 (2025). **Corrected.**

**Wu et al. (2020), SCL.** Read: arXiv 2007.04212v1, abstract and Section 3. The arXiv number in the drafts is
right. SCL composes an object network, an attribute network and a relationship network, and its "scattering
transformation" shares one network across the groups the vector is split into, so "shares networks across objects,
attributes and relations" is a fair summary. An arXiv preprint; no venue found. **Checked.**

**Stevens (1946).** Details from Crossref (*Science* 103(2684), 677–680). The content was read from a scanned copy
(a course-reading PDF), not from the publisher. Table 1 (p. 678) gives the interval scale the "general linear group
x′ = ax + b" and the ratio scale x′ = ax. On p. 679 the zero of an interval scale "is a matter of convention or
convenience, as is shown by the fact that the scale form remains invariant when a constant is added". Ch2 2.4's
sentence is supported. One point for Ch7: Stevens's groups are nested ("Each mathematical group in the column is
contained in the group immediately above it", p. 678), so his interval group keeps order (a > 0). The ring's negative
scalings are outside it, as the Ch7 note suspected. **Checked.**

---

## 7. Every work cited, with its status

"Details" means authors, title, year, venue and identifier. A claim marked "not re-read" was compared with the
abstract or with the project's reading list, not with the paper's text.

| key | cited in | status | what was checked, and where |
|---|---|---|---|
| `abbe2023generalization` | Ch3 3.8 | checked | details: PMLR 202:31–60; arXiv (JMLR 2024 extension). Claim: abstract (min-degree interpolator on unseen inputs, for Boolean functions) |
| `baez2009torsors` | Ch2 2.4, Ch3 3.8 | corrected | the page itself (27 Dec 2009). Supports "points can be subtracted but not added" and "a vector space that has forgotten its origin"; does not contain the weights-sum-to-one statement |
| `barrett2018measuring` | Ch1, Ch2 | checked | details: PMLR 80:511–520. Claim: arXiv p. 8, "our models were solving a very different problem from that solved by human subjects taking Raven-style IQ tests" |
| `battaglia2018relational` | Ch1, Ch2 | checked | details: arXiv 1806.01261. Claim not re-read |
| `bietti2021sample` | Ch1, Ch2, Ch3 | checked | details: NeurIPS 2021 index. Claim: abstract and Section 1 (Section 5 above) |
| `boixadsera2024when` | Ch3 3.8 | checked | details: OpenReview, ICLR 2024. Claim: abstract ("classical fully-connected networks, which we prove fail to learn to reason") |
| `bronstein2021geometric` | Ch1, Ch2 | checked | details: arXiv 2104.13478. Claim not re-read |
| `camposampiero2024towards` | Ch2 2.5 | checked | details: Crossref (LNCS, NeSy 2024, pp. 370–385). Claim: Section 4.1, Table 2 (Section 6 above) |
| `carpenter1990what` | Ch1 1.1 | unchecked | details checked on Crossref. Paywalled; the free DTIC copy refused automated download. Cited for "the rules behind its items were later analysed" |
| `chollet2019measure` | Ch1, Ch2, Ch7, Ch8 | checked | arXiv v2, pp. 1, 19, 25, 26, 43, 47, 49, 51 (Section 4 above) |
| `cohen2016group` | Ch2 2.4 | checked | details: PMLR 48:2990–2999. Claim needs rewording (change 2.6) |
| `elesedy2021provably` | Ch1, Ch2, Ch3 | checked | details: PMLR 139:2959–2969. Claim: Sections 1, 6 (Section 5 above); misattributed in Ch3 3.8 |
| `finzi2021residual` | Ch2 2.4 | checked | details: NeurIPS 2021 index. Claim: abstract (soft priors from hard constraints) |
| `geirhos2020shortcut` | Ch1, Ch2 | checked | details: Crossref, *Nat. Mach. Intell.* 2(11):665–673. Claim is the paper's subject; not re-read |
| `hersche2023neuro` | Ch2 2.5 | checked | details: Crossref, *Nat. Mach. Intell.* 5(4):363–375. Claim not re-read |
| `hersche2024towards` | Ch1, Ch2 | corrected | Sections II.B, IV.E, V, Tables II–VI (Section 6 above); journal version 2026 |
| `hu2021stratified` | Ch1, Ch2 | checked | details: Crossref, AAAI 35(2):1567–1574. Claim: arXiv p. 2, "the correct answer could be inferred even without the presence of the context matrix" |
| `kondor2018generalization` | Ch2 2.4 | checked | details: PMLR 80:2747–2755. Claim: abstract (convolution necessary and sufficient, "given some natural constraints") |
| `lake2017building` | Ch2 2.4 | checked | details: Crossref, *BBS* 40, e253. Claim not re-read |
| `malkinski2025deep` (survey) | Ch2 2.5 | corrected | published in *ACM Comput. Surv.* 57(7), 2025; does not support the reruns sentence |
| `marcus2018deep` | Ch2 2.4 | checked | details: arXiv 1801.00631. Claim not re-read |
| `mitchell1980need` | Ch2 2.4 | checked | the author's PDF, Sections 1–3 ("biases are necessary for the inductive leap") |
| `mohamadi2024why` | Ch2 2.5, Ch7 notes | corrected | title: PMLR 235:35934–35967 reads "Analysis on". Claim: Section 3.1.2, Theorem 3.4 (p. 8); see change 2.17 |
| `mondal2023learning` | Ch2 2.5 | checked | details: OpenReview, ICLR 2023. Claim not re-read |
| `nanda2023progress` | Ch2 2.5 | checked | details: OpenReview, ICLR 2023. Claim: abstract ("convert addition to rotation about a circle") |
| `petrache2023approximation` | Ch2 2.4 | checked | details: Crossref, NeurIPS 36:61936–61959. Claim: abstract |
| `power2022grokking` | Ch2 2.5 | checked | details: arXiv 2201.02177. Claim: abstract |
| `ravanbakhsh2017equivariance` | Ch2 2.4, Ch6 notes | checked | details: PMLR 70:2892–2901. Claim: abstract ("equivariant with respect to G-action iff G explains the symmetries of the network parameters") and Section 2 |
| `raven1938progressive` | Ch1, Ch2 | unchecked | a 1938 book, not available online. Cited for the test's purpose and its format; see change 2.1 |
| `reem2017remarks` | Ch3 3.8 | checked | details: arXiv journal reference and Crossref, *Aequationes Math.* 91(2):237–264. Claim as in the project's reading list (Theorem 9.1); not re-read |
| `sannai2021improved` | Ch3 3.8 | checked | details: PMLR 161:771–780. Claim: abstract; misattributed in Ch3 3.8 |
| `sokolic2017generalization` | Ch2 2.4 | corrected | Sections 1–3, Theorems 1–3, Remark 1 (Section 5 above) |
| `sourek2018lifted` | Ch2 2.3 | checked | details: Crossref, *JAIR* 62:69–100. Cited only with the 2021 paper |
| `sourek2021beyond` | Ch1, Ch2 | checked | details: arXiv journal reference and OpenAlex, *Mach. Learn.* 110(7):1695–1738. Quote: Section 3.2, Example 2 (arXiv p. 12). Attention and gating: end of Section 5.1 (arXiv p. 19) |
| `spratley2020closer` | Ch2 2.5 | checked | details: Crossref, ECCV 2020, LNCS pp. 601–616. Claim not re-read |
| `stevens1946theory` | Ch2, Ch3, Ch7 | checked | Table 1 (p. 678) and the interval scale (p. 679), from a scan (Section 6 above) |
| `sutton2019bitter` | Ch1, Ch2 | checked | the essay page (13 March 2019), first paragraph |
| `tahmasebi2023exact` | Ch3 3.8 | checked | details: Crossref, NeurIPS 36:55616–55646. Claim: abstract, Theorem 4.1 (p. 5) |
| `trask2018neural` | Ch2 2.5 | checked | details: NeurIPS 2018 index. Claim: abstract and the project's reading-list quote ("better characterized by memorization") |
| `wang2020abstract` | Ch2 2.5 | checked | details: OpenReview, ICLR 2020 (Duo Wang, Mateja Jamnik, Pietro Liò). Claim not re-read |
| `wang2022approximately` | Ch2 2.4 | checked | details: PMLR 162:23078–23091 (Rui Wang). Claim: abstract |
| `wang2023general` | Ch1, Ch2, Ch3, Ch7 notes | checked | Sections 1–7, Appendices A–C (Section 2 above) |
| `wolpert1997no` | Ch2 2.4 | corrected | details: Crossref. The paper is about optimisation; replace by `wolpert1996lack` |
| `wu2020scattering` | Ch2 2.5 | checked | abstract and Section 3 |
| `xu2020what` | Ch2 2.4 | checked | details: OpenReview, ICLR 2020. Claim: abstract (alignment lowers sample complexity) |
| `xu2021how` | Ch2 2.4 | checked | details: OpenReview, ICLR 2021. Claim: abstract; "relies on", not "only when" (change 2.9) |
| `zaheer2017deep` | Ch2 2.4 | checked | details: NeurIPS 2017 index. Covers sets, not graphs (change 2.6) |
| `zhang2019learning` (CoPINet) | Ch2 2.5 | checked | details: NeurIPS 2019 index. Claim not re-read |
| `zhang2019raven` | Ch1, Ch2 | checked | details: Crossref (IEEE, CVPR 2019, pp. 5312–5322). Seven layouts and the noise attribute Orientation: arXiv Section 3. The value counts 5, 6, 10 were not found in the main text |
| `zhang2021abstract` (PrAE) | Ch2 2.5, 2.6 | corrected | Section 3 (Section 6 above) |
| `zhang2022learning` (ALANS) | Ch2 2.5 | corrected | details: Crossref, ECCV 2022, LNCS pp. 692–709; year 2022, not 2021b. Claim as in the project's reading list |
| `zhou2020graph` | Ch1, Ch2 | checked | details: Crossref, *AI Open* 1:57–81. Claim is textbook GNN description; not re-read |
| `zhuo2020solving` | Ch1, Ch2 | checked | read in full (Section 3 above) |
| `malkinski2025airaven` | suggested for Ch2 2.5 | checked | Section 4, Table 2 (Section 6 above) |
| `wang2023surprising` | suggested for Ch2 2.4 | checked | details: OpenReview, ICLR 2023; its role confirmed by `wang2023general`, p. 2 and Section 5 |
| `wolpert1996lack` | suggested for Ch2 2.4 | checked | details only (Crossref, *Neural Comput.* 8(7):1341–1390); content paywalled, so keep the sentence general |
| `gallier2011geometric` | suggested for Ch2 2.4 | checked | Lemma 2.2.1, read in the author's course notes of this chapter; the book's page number not checked |

Every (Author, Year) in the drafts has a key: 53 cited works and 4 suggested additions. The † marks in Ch2's reference
list can go: every † work has now been checked. Of them, Sokolić et al. needed a correction, and the sentence that
cites Cohen and Welling and Zaheer et al. needs rewording (change 2.6).

---

## 8. Text changes, chapter by chapter

Each item quotes the sentence as it stands on 23 November and proposes a change. None was made.

### Chapter 1 (`abstract_introduction.md`)

- **1.1** (1.1): "there are proofs that a model with the right invariance needs fewer examples (Elesedy and Zaidi,
  2021; Bietti et al., 2021)". Acceptable. More exact: "there are proofs that a model with the right invariance
  generalises strictly better (Elesedy and Zaidi, 2021) and, for kernel methods, needs fewer examples by up to the
  size of the group (Bietti et al., 2021; Tahmasebi and Jegelka, 2023)".
- **1.2** (1.2): "A symmetry can thus be the right one, a harmless one or a wrong one, and which it is can be read off
  the data." Add the source of the three kinds, as the chapter's own note asks: "… a wrong one (correct, extrinsic and
  incorrect equivariance in the terms of Wang et al., 2023), and which it is …".
- **1.3** (1.3): "The ceiling is the error bound of Wang et al. (2023) for a symmetry the task does not have,
  evaluated on a rule's table". Name the theorem: "… the error bound of Wang et al. (2023, Theorem 4.3) for a
  symmetry the task does not have …".
- **1.4** (1.1): "Raven (1938) designed the test to measure reasoning that does not depend on language or schooling"
  is unchecked. Keep it only after reading a source. Zhuo and Kankanhalli say the same and cite J. Raven (2000),
  *Cognitive Psychology* 41(1):1–48, which may be easier to obtain; it was not checked either.
- **1.5** Reference list: as in `references.bib`; Zhang et al. (2019) becomes 2019a, since CoPINet is 2019b.

### Chapter 2 (`ch2_background_related_work.md`)

- **2.1** (2.1): "A Raven's Progressive Matrix is a three-by-three grid of panels with the last one missing … the
  answer is chosen from eight candidates (Raven, 1938)." Unchecked. From memory only, and worth checking: the 1938
  test also had items of other formats (the early sets are pattern completions with six options), so the
  three-by-three form with eight candidates may describe the later sets and the Advanced Matrices rather than the
  test as a whole. "In its best-known form" would be safe, or cite Carpenter et al. (1990) after reading it.
- **2.2** (2.1): "values from 0 to a maximum of up to 1,000 instead of ten" and "rows of 10 with values 0–50, 0–100
  and 0–1,000". Hersche et al. draw values from [0, m − 1], with m up to 1,000. The project's archived I-RAVEN-X
  loader says the same ("Every attribute takes values 0..maxval-1"). Check the generator, then write "values from 0
  to m − 1, with m up to 1,000" and "values 0–49, 0–99 and 0–999" if that is what the data holds.
- **2.3** (2.1): "I-RAVEN-X has no images; a puzzle is its attribute values." Fine; "the paper uses it only as
  attribute values" is the exact form.
- **2.4** (2.4): "averaged over all problems no learner beats another (Wolpert and Macready, 1997)". Replace the
  citation by Wolpert (1996), *The lack of a priori distinctions between learning algorithms*; the 1997 paper is
  about optimisation.
- **2.5** (2.4): "Ravanbakhsh et al. (2017†) show that a layer is equivariant to a permutation group exactly when its
  weights are tied along the orbits of the group". Supported by their abstract ("iff G explains the symmetries of the
  network parameters"). Remove the †.
- **2.6** (2.4): "convolutional networks are equivariant to translations (Cohen and Welling, 2016†), networks on sets
  and graphs to permutations (Zaheer et al., 2017†)". Cohen and Welling extend translation equivariance to other
  groups, and Deep Sets is about sets only. Suggested: "convolutional networks are equivariant to translations, group
  convolutions extend this to other groups (Cohen and Welling, 2016), and networks on sets are invariant or
  equivariant to permutations (Zaheer et al., 2017), as graph networks are (Section 2.2)".
- **2.7** (2.4): "Sokolić et al. (2017†) bound the generalisation error of an invariant classifier through the size
  of the input space after the group's orbits are merged, which for a finite group of size T is smaller by a factor
  of up to T." Suggested: "Sokolić et al. (2017) bound the generalisation error of an invariant classifier through
  the covering number of the base space that is left when the transformations are factored out. For T
  transformations that keep the transformed copies apart, this number is up to T times smaller, so the bound is up to
  √T smaller, which is the bound of a non-invariant classifier trained on up to T times as many examples."
- **2.8** (2.4): "Elesedy and Zaidi (2021) prove a strict generalisation benefit for equivariant models, and Bietti
  et al. (2021) give sample-complexity gains proportional to the size of the group for kernel methods." Correct. Two
  short additions make it exact: "(for squared loss and a symmetric input distribution; exactly computed for linear
  models)" after the first, "once the sample is large enough" after the second, and Tahmasebi and Jegelka (2023)
  beside Bietti et al.
- **2.9** (2.4): "Xu et al. (2021) that a network extrapolates on an algorithmic task only when the task's
  non-linearities are built into the architecture or the features". The paper says success "relies on" this, and
  calls it a hypothesis with theoretical and empirical evidence. Suggested: "… that a graph network's success in
  extrapolating algorithmic tasks relies on the task's non-linearities being built into the architecture or the
  features".
- **2.10** (2.4): "Equivariance is *correct* where the transformed input has the transformed label". Their
  definition also asks that the transformed input occur in the data. Suggested: "… *correct* where the transformed
  input also occurs in the data and has the transformed label".
- **2.11** (2.4): "For incorrect equivariance they prove a lower bound on the error, computed orbit by orbit: … (their
  Theorem 4.3)." Correct. Worth adding: "The bound needs no assumption that the data's distribution is symmetric, and
  some invariant model attains it."
- **2.12** (2.4): "They also show that extrinsic equivariance can help or hurt." Suggested: "Their earlier paper
  showed that extrinsic equivariance can help (Wang et al., 2023b); this one shows that it can also hurt (their
  Section 5)." This needs the entry `wang2023surprising`. Labels as now used in the chapters: 2023a is the NeurIPS
  paper (`wang2023general`), 2023b the ICLR paper (`wang2023surprising`).
- **2.13** (2.4): "He names the priors a fair test may assume, among them objects, elementary geometry, and numbers
  below about ten with addition and subtraction". Suggested: "… among them objects, elementary geometry, and small
  numbers with addition and subtraction (in his own benchmark, below about ten)".
- **2.14** (2.4): "a weighted sum of points is defined without reference to an origin exactly when the weights sum to
  one (Baez, 2009)". Baez's page does not say this. Cite Gallier (2011, Lemma 2.2.1) for it, and keep Baez for the
  sentence before ("points can be subtracted but not added"). The "exactly when" (the converse) is in neither source,
  but Statement 1 of Chapter 3 proves the version the thesis needs.
- **2.15** (2.5): Zhuo and Kankanhalli: replace the sentence with the one in Section 3 of this file.
- **2.16** (2.5): "Reported accuracies on I-RAVEN now reach 98–99%, but independent reruns under one protocol are
  considerably lower (Małkiński and Mańdziuk, 2022; see the notes below for the figures we have)." Suggested:
  "Reported accuracies on I-RAVEN reach about 98%, but when Małkiński and Mańdziuk (2025a) retrained thirteen of these
  models under one protocol, most scored far lower: DRNet 90.9% against a reported 97.6%, SCL 83.4% against 95.0%,
  STSN 59.0% against 95.7%." Use 99% only after checking the Triple-CFN paper.
- **2.17** (2.5): "Mohamadi et al. (2024) prove, for networks in the kernel regime, that a learner that treats values
  as unnamed labels must see a constant fraction of the addition table." Two precisions: the table is that of
  *modular* addition, and the theorem is for permutation-equivariant *kernel methods*, which describe networks early
  in training. Suggested: "… prove that a permutation-equivariant kernel method, which is how a network behaves early
  in training, must see a constant fraction of the table of modular addition before it generalises."
- **2.18** (2.5): "PrAE (Zhang et al., 2021a) recognises the objects … The rules are given." Correct. Add one
  sentence: "It is trained on the answer's index, through a choice among the eight candidates, with the rules of each
  training puzzle as an auxiliary target." Cite ALANS as Zhang et al. (2022), which makes PrAE plain 2021.
- **2.19** (2.5): "Hersche et al. (2024) report that GPT-4 answers 93% of I-RAVEN-X puzzles with rows of 3 and 77–82%
  with rows of 10, and that on Arithmetic at range 50 it reaches 30% where ARLC reaches 82%." Suggested: "… that GPT-4,
  given the attribute values as text, answers 93% of I-RAVEN's centre puzzles and 77–82% of I-RAVEN-X puzzles with
  rows of 10, and that on Arithmetic at range 50 it reaches 30% where the learned ARLC reaches 82%."
- **2.20** (2.6): "The answer is generated and the candidates are never seen, in training or as input. Only PrAE and
  its successors share this, and they are given the rules or the operations." Suggested: "… Only PrAE and its
  successors also generate the answer before matching, and they are given the rules or the operations; PrAE is still
  trained through a choice among the candidates."
- **2.21** Reference list: as in `references.bib`. Remove the † marks; the survey is *ACM Computing Surveys* 2025;
  Wang, D. (2020) is Duo Wang; add Wang et al. (2023b, ICLR), Małkiński and Mańdziuk (2025a, IJCAI), Wolpert (1996)
  and Gallier (2011).
- **2.22** Notes for you: the notes on Wang et al.'s earlier paper ("from memory"), the SCL arXiv number ("best
  recollection") and ARLC's protocol can now say "checked".

### Chapter 3 (`ch3_pricing_proofs.md`)

- **3.1** (3.5): "Wang et al. (2023) call this correct equivariance." Suggested: "This is a case of what Wang et al.
  (2023) call correct equivariance; their definition asks only that g·x occur in the data, not that it occur as
  often." Optionally add after the list: "Their definitions are made point by point, so one move can be correct on
  some rules and incorrect on others, as the shift is."
- **3.2** (3.6.3): "In general the count per orbit is the largest group of its rows that agree on the answer; that is
  the bound of Wang et al. (2023), evaluated here on a table." Suggested: "In general the count per orbit is the
  largest group of its rows whose answers agree once the move is undone; for the shift, the rows with the same c − b.
  Written this way a learner with the shift is a learner of c − b that ignores the shift, and the count is the bound
  of Wang et al. (2023, Theorem 4.3) for such learners, evaluated here on a table."
- **3.3** (3.6.3, after Statement 5): "Shifting a whole sum puzzle turns it into something that is not a sum puzzle at
  all, which is their extrinsic equivariance; the bound does not apply to it, and they show that it can help or hurt."
  Suggested: "… which is their extrinsic equivariance. On such orbits only one puzzle occurs in the data and their
  bound is zero. Extrinsic equivariance can still hurt (their Section 5) or help (Wang et al., 2023b)."
- **3.4** (3.8, table): row "3": keep Tahmasebi and Jegelka 2023 and Bietti et al. 2021 for "the effective number of
  samples is multiplied by the group's size", and move Elesedy and Zaidi 2021 and Sannai et al. 2021 to a separate
  clause ("strict gains and improved bounds of other forms"). Row "4": "Theorem 4.3 of Wang et al. 2023, applied to
  the answer relative to the moved input". The line above the table, "read as abstracts only", is now out of date for
  Wang et al., Tahmasebi and Jegelka, Elesedy and Zaidi, Bietti et al. and Sokolić et al.
- **3.5** Notes for you: "Tahmasebi and Jegelka's 'multiplied by the group's size' is from the search summary of their
  abstract" can go; it is in the abstract and in Theorem 4.1.

### Chapter 5 (`ch5_solver_results.md`)

- **5.1** No citation in the text. If ARLC's 92.6 is added (the notes consider it): "ARLC reaches 92.6% on I-RAVEN with
  its rules programmed and then trained, 92.4% learned from scratch, trained on the 2×2 layout only and given the
  panels' attributes (Camposampiero et al., 2024, Table 2)."

### Chapter 6 (`ch6_symmetry_buys_costs.md`)

- **6.1** (6.2): "Section 2.4 says a symmetry saves data by a factor of up to the size of its group." Supported once
  Section 2.4 names Tahmasebi and Jegelka and Bietti et al. Adding "for kernel methods" keeps it exact.
- **6.2** (6.2): "The direction is the one the sample-complexity results of Section 2.4 predict." The chapter's note
  suggests "is consistent with"; that is the right strength, since those results are for other model classes.

### Chapter 7 (`ch7_sum_midpoint.md`)

- **7.1** (7.3, Sum): after "97 of 1,225 rows can be right: **7.9%**, whatever the amount of data." add: "This is the
  bound of Wang et al. (2023, Theorem 4.3) for incorrect equivariance, evaluated on sum's table (Section 3.6.3)." The
  note "in its equivariant form" should go: their equivariant theorem is for squared error, and the 7.9% comes from
  their classification theorem after the rewriting of item 3.2.
- **7.2** (7.7): "In the terms of Section 2.4, the shift makes the zero of the value scale a convention, and scaling
  makes its unit one." "Makes its unit one" can be read as "sets the unit to 1". Suggested: "… the shift makes the zero
  of the value scale a convention, and scaling makes its unit a convention too: together they are Stevens's interval
  scale, except that on the ring a scaling may also be negative, which his interval scale, being ordered, excludes."
- **7.3** (7.8): "Chollet (2019) asks that skill be reported together with the priors and the experience behind it
  (Section 2.4)." Supported (his p. 1 and p. 43); a page reference can be added.

### Chapter 8 (`ch8_discussion_conclusion.md`)

- **8.1** (8.4): "A fair test may assume a short list of them: objects, elementary geometry, and numbers". Chollet's
  list has four items. Suggested: "… a short list of them, among them objects, elementary geometry, and numbers …".
- **8.2** (8.4): the four quotations and their pages (25, 26, 49, 51) are exact. The opening line can say "checked
  against the arXiv PDF (v2) on 23 November".

---

## 9. What is unchecked, and why

- **Raven (1938).** A book from 1938, not online. The drafts cite it for the purpose of the test (Ch1 1.1) and for
  its format, a three-by-three grid with eight candidates (Ch2 2.1). See changes 1.4 and 2.1.
- **Carpenter et al. (1990).** Paywalled at APA. Crossref lists a free DTIC technical-report version (ADA221448), but
  the DTIC server returned a web page instead of the PDF to an automated request; a browser may get it. The drafts
  cite it only for "the rules behind its items were later analysed by Carpenter et al. (1990)" (Ch1 1.1).
- **Wolpert (1996)**, suggested as a replacement: details only, content paywalled.
- **Claims marked "not re-read" in Section 7** (Battaglia, Bronstein, Geirhos, Lake, Marcus, Zhou, NVSA, CoPINet,
  Spratley, Mondal, MXGNet, ALANS, Reem): the details are checked, and each is cited for one general sentence that
  matches its abstract or the project's reading list. None of them carries a number or a quotation in the drafts.

## 10. How the checks were made

Downloads went to the session's scratch folder only; nothing was added to the repository except this file and
`references.bib`. No personal or project details were sent anywhere; the queries were paper titles, author names and
identifiers. Rate limits on Semantic Scholar, OpenAlex search and (late in the session) the arXiv API meant that two
arXiv identifiers (CoPINet, Mondal et al.), the OpenReview identifier of the multiplex-graph paper and three LNCS
volume numbers could not be confirmed; they are left out of `references.bib` rather than given from memory.

---

## 11. The appendices (`appendices.md`, checked 2026-11-23)

Only Appendix E.2 cites literature: "as in the grokking literature (Power et al. 2022; Nanda et al. 2023; Liu et al.
2023; Varma et al. 2023)". All four now have keys. Power et al. and Nanda et al. are cited with the same years as in
`references.bib` (`power2022grokking`, `nanda2023progress`); the appendix writes the citations without the comma the
chapters use ("Power et al., 2022"), which the LaTeX citation commands will make uniform.

- **Liu et al. (2023), Omnigrok** (`liu2023omnigrok`, ICLR 2023): **checked**. Supports "possibly larger initial
  weights": a large initialisation induces grokking, also on MNIST with 1,000 training points (abstract, Section 3).
- **Varma et al. (2023)** (`varma2023explaining`, arXiv preprint; submitted to ICLR 2024, not accepted as far as
  OpenReview shows): **checked**. A critical dataset size below which memorising stays the more efficient solution,
  and "ungrokking" (abstract, Sections 1–4).
- **Power et al. (2022)**: **checked**. The setting E.2 calls "the grokking papers' own" (weight decay 1, β2 0.98) is
  in their Appendix A, p. 6: AdamW, learning rate 10⁻³, weight decay 1, β1 0.9, β2 0.98, minibatch 512 or half the
  training set, 10⁵ updates. The weight-decay quotation in `docs/ARITHMETIC.md` is on p. 4.
- **Nanda et al. (2023)**: **checked**. Full-batch AdamW, learning rate 0.001, weight decay 1, 30% of the pairs,
  40,000 epochs (p. 3), which supports "the grokking papers train on full or large batches".

**Text change for E.2.** "The setting most likely to show it is a training set small enough to memorise, many
updates, decoupled weight decay, and possibly larger initial weights." Varma et al. add a lower limit, and E.2's own
results at 100 and 300 puzzles look like it. Suggested: "… a training set small enough to memorise but above the
critical size below which memorising stays the cheaper solution (Varma et al. 2023), many updates, decoupled weight
decay, and possibly larger initial weights (Liu et al. 2023)." Optional: cite Loshchilov and Hutter's AdamW paper
for "decoupled weight decay"; it is not in `references.bib` and was not checked.
