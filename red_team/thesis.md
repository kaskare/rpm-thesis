# An examiner's read of the thesis drafts (23 November 2026)

These are the objections an examiner is most likely to raise, worst first. Each says whether the text now answers it.

1. **"The trained networks land on the counts" claimed too much.** For the data saved, they don't quite: with the shift,
   59.9% at 100 puzzles where the count allows 74.8%, and 91.0% against 98.7% at 600. *Fixed* in 7.9 and 8.1, which
   now give both numbers. The ceiling and the factor of two counted in training puzzles were met.
2. **"The factor of two for components is true by construction."** Nearly so: at twice the data each separate copy sees
   what the shared network sees. *Answered* in 6.2, which now says so and says what the run does test. The attributes
   are the real test (2 to 3 times, met at both points).
3. **Most numbers are one run.** *Answered* in 8.6. The small-data comparisons of Chapters 6 and 7, column roles and the
   row fold have three seeds. The I-RAVEN main result, the 1,000 and 6,000 component cells, the value-shift ablation
   and the wrong symmetries are still one run each.
4. **I-RAVEN-X numbers come from puzzles looked at throughout the work.** *Answered:* new limit in 8.6.
5. **The value axis is given the rules' primitives.** *Answered* in 4.5, 6.2 and 8.6: it is compared with itself only.
6. **The theorem is the familiar permutation equivariance.** *Answered* in 3.2 and 3.8: the proof is short. What the
   chapter adds is that every symmetry is tied to a check without training or to a count made before the runs.
7. **Generator priors carry part of the accuracy.** *Answered* in 1.4 and 8.4 with the transposition result: about 97%
   falls to 45%.
8. **The end-to-end pipeline uses object labels that the ViT never gets.** *Answered* in 1.4 and 5.3.
9. **The ceiling is Wang et al.'s.** *Answered:* credited to their Theorem 4.3 in 1.3, 3.6.3 and 7.3, with the step that
   turns the shift into an invariance.
10. **The rule twins are an invented task.** Partly answered in 7.1 and 7.9: they were built so that only one property
    differs. Arithmetic on the benchmark is linked to them by inference only, as 8.6 says.

Still open, for the author: the items in `docs/thesis/OPEN_ITEMS.md`, Part 1.

# An external review (23 November 2026), and what was done about it

A reader sent the drafts back with six major objections and some minor ones. Each is paraphrased here with its
verdict after checking the drafts, and what changed.

| objection | verdict | what was done |
|---|---|---|
| 1a. End to end against the ViT compares dense object supervision with the answer's label only | right; 5.1 said so, the abstract and 1.3 did not | the abstract, 1.3 and 1.4 now say what each is trained on |
| 1b. The curve of 5.4 sets models reading labels against a ViT reading pixels and calls 74 points "the improvement over black-box vision" | right | the recognizer retrained on 300, 1,000 and 3,000 puzzles per layout and the facet model scored end to end at each size: 89.8 / 93.9 / 95.8, 0.2–0.3 below the labels (RESULTS §8). 5.4 now reports this column, drops the caution paragraph, and says the 74 points measure the object representation together with its labels, not two pixel learners on equal supervision; 1.1 and 8.8 say the same |
| 1c. The ViT is a strawman | partly: published RPM solvers pick among candidates, ours generate | 5.1 gains a table of SCL, PredRNet and DRNet, as reported and as rerun under one protocol (Małkiński and Mańdziuk 2025a, already checked): 83–91% rerun, below our ViT's 93.1% at the same data; 1.4 says so |
| 2a. 0 of 101 unseen steps is forced by one weight per offset | right | 3.6.6, 7.5 and 1.3 present it as what the parameterisation implies, with the count saying in advance which rows |
| 2b. The 7.9% sum ceiling comes from reading the question row only, not from the shift | already in Statement 5, 7.3 and 8.7; the summaries said otherwise | the abstract, 1.3, 3.7, 7.8's table and 8.1 now name both causes |
| 2c. The factor of two for components is trivial | conceded already in 6.2 | 6.2, 1.3 and the abstract lead with the attributes as the real test |
| 3. The ladder was repaired after its predictions failed; the count does not predict a network's accuracy | right on the framing; the repair was already reported as a second attempt | 3.7, 3.7.1, 7.8, 7.9 and 8.1: the count bounds a learner that shares nothing across orbits and orders the rungs; reaching it needs a descriptor that keeps orbits apart; the 59.9 against 80.0 gap is unexplained |
| 4a. I-RAVEN-X has never had a test pass | right | every final checkpoint of Chapters 6 and 7 scored once on puzzles 8000–9999 (I-RAVEN-X, twins) or the I-RAVEN test split (components): `jobs/pynl/submit_test_pass.sh`. Done for all 187 (RESULTS §14, Appendix D.6): the gains, the data factor and the twins' ceilings and ladder hold. The facet model on sum at 600 puzzles drops 31.2 → 24.7 (2.6 standard errors, one run); column roles' Arithmetic 39.6 → 36.0 at every seed (1.4 standard errors). 4.2, 5.7, 6.6, 7's limits and 8.6 give the test numbers |
| 4b. Single seeds and few seeds | partly: the facet against GNN B difference is already disclaimed (5.6) | not rerun; the main I-RAVEN table and the 6,000-puzzle cells stay one run each, as 8.6 says |
| 5a. `plus` reduces arithmetic to rule selection and grows with the square of the range | right, and the review's M³ is M² | 7.6 and 8.6: like programmed-rule models (PrAE, ARLC); the facts grow with the square of the range; never run on values to 1,000 |
| 5b. The ring of 101 values | partly: the data never leaves where ring and integers agree | 3.7.1 and 7.7 say which statements hold on the ring only and that scaling is out of support on 0–50 |
| 6a. The renaming theorem is standard isomorphism invariance | already conceded in 3.2 and 3.8 | 1.3 now says so where the contribution is listed |
| 6b. A pairwise GNN can carry triples through factor nodes; sharing is a line of code in tensor frameworks too | right | 8.5 softened on both |
| Minor: repetition between Chapter 1 and 8.1 | right | 8.1 rewritten as a synthesis with fewer restated numbers |
| Minor: no equations of the forward pass in the main text | right | 4.4 and 7.2 give the forward pass as equations, written from the template as it runs |
| Minor: the appendices were missing from the text the reader got | right; the combined text file leaves them out | the combined files are rebuilt with the appendices |

**Still open after this round.** A black-box model given the same object labels as the recognizer (1b: the curve now
removes the labels at test time but not their help in training); more seeds for the single-run cells (4b); the new
architecture the review recommended as a baseline (1c). The value axis was never run on values to 1,000 (5a).
