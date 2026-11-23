# Open items for the thesis (23 November 2026)

This file replaces the "Notes for you" sections that used to close every chapter draft. Part 1 lists what is still
open, worst first: things only you can do or decide, then text to check, then optional work. Part 2 keeps every
chapter's notes as they stood on 23 November, word for word, so nothing is lost. Part 3 says where each chapter's
numbers come from.

## Part 1. What is still open

### Only you can do these

1. **The assignment PDF from KOS**, to go after the title page.
2. **The title.** The working title is "What a Symmetry Is Worth: Pricing Inductive Bias in Relational Models of
   Raven's Matrices". Outline §1 suggests putting the assignment's words (symmetry-aware, object recognition,
   relational neural networks) into the subtitle or the abstract.
3. **The declaration and the acknowledgements.** You sign the declaration. The acknowledgements are yours to write.
4. **Credit for the reflection relation in Section 3.3,** and one result to pass on to your supervisor (Appendix
   E.1): see `docs/private/OPEN_ITEMS_private.md`.
6. **Two sources nobody could check.** Raven (1938) is a book from 1938 that is not online. Carpenter et al. (1990) is
   paywalled; a free DTIC report version (ADA221448) may open in a browser. Chapter 1 cites them for what the test is
   for, and Chapter 2 for its format. Either read them, or keep the sentences as they are now: "in its best-known form"
   in 2.1 is safe, but the 1.1 sentence on what Raven designed the test for still rests on memory. Zhuo and
   Kankanhalli say the same and cite J. Raven (2000), *Cognitive Psychology* 41(1), which may be easier to get.
7. **The Overleaf build log**, once the LaTeX exists. I cannot compile here.

### The LaTeX (`thesis_latex/`)

- **Compile it on Overleaf** with pdfLaTeX and BibTeX, and send me the log. I could not compile here, so expect a
  few errors on the first run. The class is `ctuthesis`; if Overleaf does not find it, start from Overleaf's
  ctuthesis template and copy the files in.
- **Fill in the TODOs in `main.tex`:** your name, study programme, the submission date, Gustav's titles, the
  assignment PDF, the acknowledgements.
- **Check what Claude drafted (marked CHECK):** the Czech title and subtitle, the Czech abstract and keywords, the
  department, and the class keys `doctype = M` and `faculty = F3`.
- **The declaration.** The English text in `main.tex` is the usual ČVUT wording. Check it against FEL's current rules.
  Check in particular whether the current ČVUT guideline on ethical principles for theses (a version of 31 October
  2025 is on cvut.cz; I could not read the PDF here) asks you to state how AI tools were used. This thesis was drafted
  with a lot of help from Claude, so that statement matters.
- **Labels a and b for Wang et al. 2023.** The PDF prints whichever letters the bibliography style assigns; they may
  be the other way round from the Markdown drafts. Citations go by key, so nothing breaks, but the drafts and the PDF
  may then differ in the letters.
- **When to stop regenerating.** Until you start editing the `.tex` files, rerun `python3 thesis_latex/md2tex.py`
  after any change to the drafts. After that, edit only the `.tex` files.

### Text to check before it is final

8. **5.3's explanation of why a misread panel is not fatal** is a reading, not a measurement.
9. **5.6's last two sentences** (position rules are set operations, and the position channel has none of that
   structure) are the lab notes' assessment (§18.6). Check that you agree.
10. **8.4's table of priors** has two judgement calls: "the same rule in every row" is called general although "rows"
    come from the generator, and "any attribute can carry a rule" is called general although the attribute list is
    not.
11. **8.5's claim that the baseline's symmetries "each needed its own check"** refers to `--check-symmetry` in
    `core/baselines/train_baseline.py`. You know best whether "engineered one at a time" is fair.
12. **Wang et al. (2023a) are credited as the reference check found** (`REFERENCE_CHECKS.md`, Section 2): the ceiling
    is their Theorem 4.3, applied to the answer relative to the moved input. Their equivariant theorem is for squared
    error and is not the one used.

### Optional work

13. **A rescoring job for the whole-puzzle exact scores in 6.7's tie table.** The plain run's recorded exact predates
    the fix of the exact scorer. The per-rule values in the table were never affected.
14. **Why tying columns 1 and 2 costs Arithmetic** (Section 6.7) is not explained, because it was not tested. A guess:
    for a difference, the model must now recover which operand is larger from the values.
15. **A ViT run at 4,500 puzzles** would make "twenty times the puzzles" in Chapter 5 tighter. The ViT passes 90%
    only at 6,000 per layout (93.4) and is at 45.8 at 3,000.
16. **One sentence in 7.4 on the lookup column** of the shared model (35.9 / 74.3 / 77.7). It reaches 77.7 because with
    51 rows the first step almost identifies the row; the best possible from differences is 82.0.
17. **A footnote in 7.4 on mirror.** Without the symmetry mirror is harder than sum (63.2 against 69.8 per value), which
    was neither predicted nor explained. An examiner who reads the table will see it.
18. **A back-reference to Mohamadi et al. (2024) in 7.4**, next to "both track the memorising learner".
19. **ARLC's 92.6% in Chapter 5**, if you want a neuro-symbolic number next to ours. The reference check has the
    sentence: rules programmed and then trained, 92.4% learned from scratch, trained on the 2×2 layout only and given
    the panels' attributes.
20. **A job file for the runs submitted with `sbatch --wrap`** (the main I-RAVEN facet run and the named-column runs on
    I-RAVEN-X), so that Appendix F has a job for every result.
21. **A README for the attached medium.** The repository's `README.md` is still IBM's NVSA readme. Appendix F needs
    one that says what the thesis is and points to `docs/CODE_INDEX.md`.
22. **Commit `analysis/seeds_summary.py` and `jobs/pynl/submit_transpose.sh`** before the medium is made; Appendix F
    names both.

### Deliberately left out (no action needed)

- Chapter 1: the 3×10 comparisons between models, the row fold, grokking, row-length transfer, and any number from the
  value axis as a solver.
- Chapter 4: the hub and ordered templates, the CNN, the ViT with a filter per patch, the capacity sweeps beyond one
  sentence, and the value axis on I-RAVEN (18.0k weights, 99.3 matched).
- Chapter 5: copy links on I-RAVEN (98.2 on validation, never scored on test), the value axis on I-RAVEN, the published
  accuracies of image models (now quoted in 2.5 from the A-I-RAVEN reruns only).
- Chapter 6: the ViT with a filter per patch, translation of the ViT's input. The unseen-layout result is credited to
  the graph, not to symmetry, because labelled GNN A transfers as well as B.
- Chapter 7: Peano addition (one paragraph in Chapter 8), grokking (Appendix E.2), plus and minus apart, the MLP row.
- Chapter 8: LLMs.

## Part 2. The chapters' notes as they stood on 23 November

### abstract_introduction.md

- **End to end is 97.13%** (68.8 exact) on the 7,000 unseen validation puzzles, against 97.31 from labels (job 12028100,
  `docs/RESULTS.md` §8). The weights: 267k facet + 326k recognizer against the ViT's 3,753k.
- **The test pass is done** (2026-11-14, one pass, 14,000 puzzles): 97.19% from labels and 97.03% end to end, 0.1 below
  the validation numbers. The abstract and 1.3 now carry them.
- **The baselines have their test pass too** (14 Nov, RESULTS §1): ViT 93.10, torch GNN B 96.79. The abstract, 1.3 and
  1.4 now compare everything on the test split. The learning-curve numbers in 1.1 (16 / 19 / 46 / 93%, and the facet
  model's 90% from 300 puzzles) are validation numbers; those runs have no test pass.
- **The sum and midpoint numbers are means of three seeds** since 14 Nov (`docs/ARITHMETIC.md`, "Three seeds"): 91.0 ±
  3.4 and 6.2 ± 0.8 at 600 puzzles, 18.0 and 19.4 with weights per value, 98.2 ± 0.8 for sum with `plus` at 100. The
  single runs they replace were 89.3, 6.9, 17.8, 19.3 and 97.6. "About three quarters of the others" is 71.5 / 77.5 /
  77.2% over the seeds (the earlier 59% came from a checkpoint picked by matched).
- **Wang et al. (2023) is credited in 1.3** for the ceiling (see the note in `ch2_background_related_work.md`). The
  three kinds of symmetry in 1.2 are theirs as well (correct, extrinsic, incorrect); add the citation there once you
  have read the paper.
- **Read before you cite:** Carpenter et al. (1990) and Raven (1938) are cited only for what the test is; I have not
  checked them, nor has the project. Elesedy and Zaidi (2021) and Bietti et al. (2021) are on the reading list and are
  cited for one sentence. Zhuo and Kankanhalli (2020) is on the assignment's list and still unread.
- **The reference details are from memory** and need checking against the papers, page numbers and venues included.
- **"We"** is used throughout. Change to "I" or the passive if your department prefers.
- **1.4 states the torch result on purpose.** The examiner will find it in Chapter 5 anyway, and stated here it reads as
  a finding, not a concession.
- **Left out on purpose:** the 3×10 comparisons between models, the row fold, grokking, row-length transfer, and any
  number from the value axis as a solver.

### ch2_background_related_work.md

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
- **Every  citation is from memory.** Venues, years and author lists need checking, and the claims attributed to
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

### ch3_pricing_proofs.md

- **Sections 3.1 to 3.5 are new (23 Nov).** Sources: lab notes §21.1 to §21.3 (definitions, theorem, checks), §22
  (the data's symmetries), §24 (column roles), §26 and §26.1 (labels, separate weights), §27 (the value axis and its
  shift check). The table of rule survival moved here from Section 4.2, which now points to it.
- **Wang et al.'s names** (correct, extrinsic, incorrect) are used as Chapter 2 describes them. Confirm them against
  the paper once the reference check is done; the proof of their bound is cited, not repeated.
- **The reflection relation in 3.3** is described as a way of writing a symmetry and was not run. On whom to credit,
  see `docs/private/OPEN_ITEMS_private.md`.
- **"Out of support" has two cases here**, valid and not valid, because the thesis needs both (rows of 10 in 6.5,
  columns tied in 6.7). Wang et al. have one name for both.
- **Section 6 numbers in this chapter** assume the new Section 6.6 on column roles, with the wrong symmetries moving
  to 6.7.
- The source paragraph that opened the old draft: the checks that need no training are printed by
  `analysis/pricing_checks.py`, `analysis/rule_orbits.py` and `analysis/twins_linear_fit.py`; the trained numbers are
  from `docs/ARITHMETIC.md` and `docs/RESULTS.md` §11.

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
- **The sources were checked on 23 Nov** (`docs/thesis/REFERENCE_CHECKS.md`): Wang et al., Tahmasebi and Jegelka,
  Elesedy and Zaidi, Bietti et al. and Sokolić et al. were read for what this chapter cites them for.
- **Voice.** "We", as in the other chapters.

### ch4_task_data_models.md

- **Section 4.2's symmetry table moved to Section 3.5** (23 Nov), with the three readings it carried; 4.2 points to it.
- **The rule listing in 4.4 is a paraphrase** of `build_facet_template`'s docstring. The thesis should print the real
  template, or an excerpt, in an appendix.
- **"The trainer refuses to run unless it reproduces the facet pipeline's targets"** is the `check_*` arrays of
  `data_prep/prepare_baseline_data.py`, on the first 20 puzzles of each split.
- **Not described on purpose:** the row fold, the hub and ordered templates, the CNN, the ViT with a filter per patch,
  the capacity sweeps beyond one sentence (outline §5, cut).
- **The value axis is introduced as a test bed in 4.5**, as the outline decided. Its I-RAVEN version (18.0k weights,
  99.3 matched) is left out; say if you want one sentence on it.
- **One thing to check in the code before this is final:** the recognizer's input size and whether its query tokens
  attend to the patches in every block. I described only what its docstring states.
- **The 62 numbers:** 3 × 10 code bits, 10 bits for the object count, 22 slots (`meta` of
  `experiments/baselines/data/iraven.npz`).

### ch5_solver_results.md

- **The baselines' test pass was done today** (job 12032179, `core/baselines/test_eval.py`; RESULTS §1). 15
  checkpoints, each scored once, each after reproducing its validation record exactly. Their test passes are spent,
  like the facet model's.
- **Learning curves and by-rule tables are validation numbers.** Only the 6,000-puzzle checkpoints have test numbers.
  I kept the two apart in the text and said which is which.
- **"Twenty times the puzzles"**: the ViT passes 90% only at 6,000 per layout (93.4), and is at 45.8 at 3,000. If you
  want a tighter statement, a run at 4,500 would give it.
- **The symbolic template's 98.3 is from its own earlier test pass** (RESULTS §1), with exact not defined for it.
- **5.7 follows the outline's decision** to print no 3×10 ranking (§2 row 17, §6 item 2). If you rerun GNN A, the
  object GNN and the MLP without regularisation and train a facet model as long, this section can become a table.
- **Left out:** copy links on I-RAVEN (98.2 on validation, never scored on test), the row fold, the value axis on
  I-RAVEN, the literature rows. The copy-link number is higher than the plain model's; it has no test pass, and its
  per-panel gates break the row symmetry (Chapter 6), so I did not use it as the headline.
- **5.3's explanation of why a misread panel is not fatal** is my reading, not something measured.
- **5.6's last two sentences** (position rules are set operations; the position channel has none of that structure)
  are the notes' assessment (§18.6); check you agree before it goes in.

### ch6_symmetry_buys_costs.md

- **The value-shift ablation is on the value axis with its rule features and `plus` given**, so it is compared with
  itself only, and it is not at equal size. The text says both. A width-matched per-value model does not exist.
- **"A factor of about sixty, close to the size of the group"** rests on two cells: 99.0 at 100 puzzles shared, 98.4
  at 6,000 per value. It is suggestive, one seed, and the per-value model also has 30 times the weights. Keep the
  hedge or cut the sentence.
- **Whole-puzzle exact is left out of 6.7's tie table on purpose.** The plain run's recorded 92.7 predates the fix of
  the exact scorer (rescored: 97.0) and I could not confirm which scorer the tied run's 95.1 used. The per-rule values
  were never affected by that bug, so the table uses them. Worth one rescoring job if you want the bracketed numbers.
- **The fold's measured values are per attribute, all rules pooled; the ceilings are per rule.** `facet_rule_eval.py`
  cannot give per-rule values for hedged outputs (RESULTS §7). Type reads 57 at 600 puzzles, just above the 54–56
  ceiling; with about 1,000 values that is within noise, and the text says "to within the noise".
- **Why the tie costs Arithmetic** is not explained in the text because it was not tested. My guess: for a difference
  the model must now recover which operand is larger from the values.
- **The unseen-layout result is credited to the graph, not to symmetry**, because labelled GNN A transfers as well as
  B. The earlier docs list it under symmetry gains; I think that overstates it.
- **Ravanbakhsh, Elesedy and Zaidi, Bietti:** 6.2's "the proportionality that the sample-complexity results predict"
  points at Section 2.4. Those papers prove it for other model classes; say "is consistent with" if you want to be
  safe.
- **Column roles are back (23 Nov)** as Section 6.6, with three seeds, as you decided. The wrong symmetries are now
  6.7 and the summary 6.8. The equal-size wide model (94.9) is one run.
- **Left out:** the row fold, the ViT with a filter per patch, translation of the ViT's input.

### ch7_sum_midpoint.md

- **Order follows outline §3**, steps 1 to 8, with the two-by-two table and the three objections.
- **The ceiling is credited in 7.3 (23 Nov)** to Wang et al.'s Theorem 4.3. It is their classification theorem, after
  the answer is written relative to the moved input; their equivariant theorem is for squared error
  (`docs/thesis/REFERENCE_CHECKS.md`, Section 2).
- **Mohamadi et al. (2024)** belongs in 7.4 next to "both track the memorising learner": a learner that treats values
  as unnamed labels must see a constant fraction of the table. Section 2.5 already says it; one back-reference is
  enough.
- **All three cheap runs of outline §3 are done.** Three seeds at 100 and 600 puzzles and the 0 of 101 from the
  checkpoints in the table (2026-11-14; `docs/ARITHMETIC.md`, "Three seeds"), and the scaling kernel (2026-11-15; "The
  scaling kernel"), now Section 7.7.
- **How 7.7 came out, plainly.** The prediction about which unseen rows become answerable held exactly. With the first
  row descriptor the totals went down; the cause was measured, the repair predicted, and with it every rung needs less
  data than the one below ("The repair"). It is now a strong section, but a second attempt: keep the failure in the
  text. The 33 rows of the shared model were already in 7.5; the repair also answers them at 600 puzzles.
- **The gate (2026-11-16) was my guess and it was wrong.** Both of its predictions failed (`docs/ARITHMETIC.md`, "The
  rule gate"). The probe of the saved checkpoints then located the cause one layer earlier: the row descriptor is a
  constant in 22 runs of 24 (`analysis/twins_probe.py`). "Why two seeds fail" in 7.7 reports both. The gated runs can
  shrink to one sentence if the section is too long; the probe table is the part to keep.
- **The next run for 7.7 was made on 2026-11-17 and came out as predicted** (`docs/ARITHMETIC.md`, "The row descriptor
  reads the steps"; 18 runs). It is "The repair" in 7.7, and it changed the fourth point of "What this adds", the
  limits, and the ladder of 7.8. The section's ending is now "scaling gives the rule from 100 puzzles once the model
  can read the rows". Keep the first, failed version in the text: the failure, its measured cause and the predicted
  repair are the argument, and a reader should see that the ladder is a second attempt. One prediction was missed by
  one row (the shared model right on 1 of 303 rows with an unseen first step); the text says so.
- **"Section 2.4" in 7.7** is the paragraph on Stevens (1946). Stevens's interval scale allows v → kv + s with k > 0;
  the ring also allows negative k. Check the wording against the paper before keeping "its unit one".
- **What the seeds changed in this chapter.** No conclusion. One sentence: `plus` does not improve midpoint (the
  single run's 70.5 at 100 puzzles was a high draw; three seeds give 63.9 ± 3.9 against 60.3 ± 2.7). The `plus`
  cells on sum were 97.6 / 99.8 in the single run, picked by exact on the test rule; the seeded cells use the
  whole-puzzle rule like every other cell. The first table of 7.4 still shows the single runs, so that all four
  rules and 6,000 puzzles sit in one table; you may prefer to put the means into its 100 and 600 cells.
- **The row for 300 puzzles is gone from 7.5** (0 of 6 unseen first steps, 77.0% of the others): it came from an
  unseeded checkpoint picked by matched. It is in `docs/ARITHMETIC.md` if you want it back.
- **The figure is redrawn** with the seed means and one-sd whiskers at 100 and 600 puzzles.
- **"A tenth of the data"** in 7.4: 600 against 6,000 puzzles.
- **The lookup column of the shared model** (35.9 / 74.3 / 77.7) is not discussed. A shift-equivariant model should be
  capped on lookup too; it reaches 77.7 because with 51 rows the first step b − a almost identifies the row (the docs
  give 82.0 as the best possible from differences). One sentence in 7.4 could say so; I left it out to keep the
  section on the twins.
- **Mirror without the symmetry is harder than sum** (63.2 against 69.8 per value, 41.9 against 65.5 facet). Not
  predicted and not explained; `docs/ARITHMETIC.md` lists it under "what did not go as predicted". An examiner who
  reads the table will see it, so a footnote is safer than silence.
- **The `plus` facts do not wrap** while the kernel is a ring; `docs/ARITHMETIC.md` notes it.
- **Values as numbers and the ladder of priors are in (2026-11-17).** The scope paragraph closes 7.2, the ladder is
  the new 7.8 (the old 7.8 is now 7.9; no other chapter referred to it by number), 7.5 gained the rows-per-orbit
  numbers, and 7.9 gained one sentence, one objection and two limits. Sources: `docs/ARITHMETIC.md`, "Two checks
  without training"; `analysis/twins_linear_fit.py`; `analysis/twins_seeds.py --dumps`. To check before you keep it:
  - The linear fit is a reference with no training, not one of the thesis's models. Keep it out of any ranking.
  - The middle of the ladder contradicted its own count with the first row descriptor (60.3, then 49.9, then 19 to
    100). The table now shows the models with the repaired descriptor (59.9, 89.2, 100.0) and says so underneath.
  - The ladder is not new mathematics: a symmetry with one orbit fixes the rule. Its use here is as an accounting of
    priors. Do not let the section claim more than that.
  - The Chollet sentence rests on Section 2.4's summary of his paper. Check it against the paper.
  - "Only ratios of differences count" for the scaled model is the invariant of v → kv + s; on the ring k may be
    negative (see the Stevens note above).
  - Chapter 8's section on priors can now point to 7.8. I did not change Chapter 8.
- **Not included:** Peano addition from zero and `next`, the grokking runs, plus and minus apart, the MLP row (it
  barely learns any rule on these puzzles). Outline §5 puts the first in two paragraphs of this chapter or Chapter 8;
  I gave it one paragraph in Chapter 8.

### ch8_discussion_conclusion.md

- **8.4's table is the long outline's, shortened.** Two rows are judgement calls you may want to move: "the same rule
  in every row" (general, but "rows" is the generator's) and "any attribute can carry a rule" (general, though the
  attribute list is not). The long outline marks the row fold and copy links as mixed; both are cut here with their
  models.
- **Chollet's page numbers and quotations** were checked against the arXiv PDF (v2) on 23 November
  (`docs/thesis/REFERENCE_CHECKS.md`, Section 4).
- **"Sum from 5% to 98%"** is 4.5 → 98.2 at 100 puzzles on the twins, means of three seeds (Section 7.6).
- **8.1's "little or nothing at 6,000"**: −0.1 for components, +1.6 for the value shift, +6.0 for attributes. The
  introduction says "between 0 and 6 points"; use one wording in both.
- **8.5 claims the baseline's symmetries "each needed its own check".** That is `--check-symmetry` in
  `core/baselines/train_baseline.py`; true as far as I can see, but you wrote that code path with me over several
  sessions and know best whether "engineered one at a time" is fair.
- **8.5's sentence that a pairwise message-passing network "has no place" for `plus`** follows `docs/ARITHMETIC.md`
  ("Numbers as objects": a hyperedge). Hypergraph networks exist; "without being redesigned" is there for that reason.
- **Future work item 1 was the scaling kernel.** It was run on 2026-11-15 and is now Section 7.7; the item is replaced
  by what that run left open. The gate it first proposed was run on 2026-11-16 and failed; the item now names the row
  descriptor, the cause the probe found (`docs/ARITHMETIC.md`, "The rule gate"). The descriptor was repaired on 2026-11-17
  and the item now names the next open path, sum from differences across rows. Sections 8.1 and 8.2 do not mention
  scaling yet. One sentence would fit in 8.2: more of a rule's symmetry means less data at every step, once the model
  can read the rule from the complete rows (Sections 7.7 and 7.8).
- **The conclusion uses test-split numbers** for 97.2, 97.0 and 93.1, and validation numbers for 90 against 16 (the
  learning curve has no test pass).
- **Not discussed:** grokking, row-length transfer, the row fold, LLMs. Outline §5 sends the first three to an
  appendix of negative results, one page each.

### appendices.md

- **"Every number appears in RESULTS, FINDINGS or SYMMETRY_RESULTS."** I read this as every result number. Code and
  configuration values (widths, learning rates, the Java path, job numbers, the −6 gate bias in the printout) come from
  the code, `CLAUDE.md` or the lab notes. To keep to the rule I left out numbers that live only in the lab notes or in
  code comments: the size of the symbolic models' collapse under `iso_value_compression` (it is in the comment in
  `pynl_symbolic.py`), the §17 diagnosis numbers, and the per-transformation shares of notes §22.2. B.2 points to
  Section 3.5 for those shares instead, which prints them as of 23 Nov; if that table leaves Chapter 3, B.2 needs it
  back.
- **C.1's default aggregation** was read from the installed library (`rule_aggregation = Aggregation.AVG` in
  `neuralogic/core/settings/settings_proxy.py`). The printout marks every rule whose aggregation matters explicitly.
- **B.3's recount.** I counted the rules per attribute in training puzzles 0–5999 of the values-0–50 file; every share is
  within one point of 25% (size, colour) or of a third (type). The exact counts are not in RESULTS, so the appendix gives
  only the shares RESULTS states. Add the counts to RESULTS §7 if you want them in the table.
- **A.2 is partly new.** That pruning acts on the ground network was checked on 2026-11-23 on a toy template
  (pruning on and off, one rule, one rule averaged over two groundings, two rules of which one grounds, and the sum
  split over two rules). The list of facet atoms whose tanh is inactive (`f0`, `pv1`, `pv2`, `ctx`, `vP`, `ctxp`, and
  `vF` for number) follows from that and from the printout; it is not yet in the lab notes or FINDINGS 5. It changes no
  result, since every model was trained this way. Chapter 4 does not mention the activations, so nothing there needs
  changing; a sentence in the lab notes, §3, would keep the record complete.
- **The printout in C.2 comes from the working tree**, where `core/pynl/bench_hub_grounding.py` has uncommitted changes.
  Reprint it after the final commit; the command is in the appendix. C.4 and C.5 are the lines that differ from it in
  the roles, fold and content-gate templates, printed the same way.
- **Two files that F.2 names are not tracked by git yet**: `analysis/seeds_summary.py` and `jobs/pynl/submit_transpose.sh`.
  Commit them before the medium is made. Building the medium from `git ls-files` leaves out everything git ignores or
  excludes.
- **The repository's `README.md` is still IBM's NVSA readme.** The medium needs a short README of its own that says
  what the thesis is and points to `docs/CODE_INDEX.md` and Appendix F.
- **The facet runs on I-RAVEN-X with named columns, and the main I-RAVEN run, have no job file.** They were submitted with
  `sbatch --wrap`, with the command of notes §18.1. If the medium should have a job for every result, a small
  `jobs/pynl/submit_facet_main.sh` would close the gap.
- **E.1 exists although row-length work is parked**, because the outline sends it to this appendix as a negative result.
  It reports the runs that were made and proposes nothing new.
- **E.2's literature** (Power et al. 2022, Nanda et al. 2023, Liu et al. 2023, Varma et al. 2023) is as summarised in
  `docs/ARITHMETIC.md`, "What makes grokking possible"; the references still have to go into the bibliography.

## Part 3. Where each chapter's numbers come from

The opening notes of the drafts, as they stood:

- **abstract_introduction.md:** Working title: *What a Symmetry Is Worth: Pricing Inductive Bias in Relational Models of Raven's Matrices*. Numbers are from `docs/RESULTS.md`. Notes for you, not for the thesis, are at the end.
- **ch2_background_related_work.md:** Follows `abstract_introduction.md`. Chapter 3 (theory) is not drafted here. Notes for you are at the end; citations marked are from memory and have not been opened in this project.
- **ch4_task_data_models.md:** This chapter uses two things from Chapter 3: a template cannot tell apart constants it does not name (Section 3.2), and a symmetry is a data symmetry, out of support, or conflicting, by what it does to valid puzzles (Section 3.5). Numbers are from `docs/RESULTS.md`. Notes for you are at the end.
- **ch5_solver_results.md:** Numbers are from `docs/RESULTS.md` (§1 for the test split, §4 for layouts, rules and learning curves, §8 for perception, §9 for cost). Cells are matched % with exact % in brackets. Notes for you are at the end.
- **ch6_symmetry_buys_costs.md:** Numbers are from `docs/RESULTS.md` §3 and §7 and `docs/SYMMETRY_RESULTS.md`. Cells are matched % with exact % in brackets, on unseen puzzles, one seed unless marked. Notes for you are at the end.
- **ch7_sum_midpoint.md:** Numbers are from `docs/ARITHMETIC.md` ("Is the missing symmetry what makes it hard?", "Sum and midpoint compared") and `docs/RESULTS.md` §11. Cells are exact % on the tested rule, on unseen puzzles, about 500 rows per cell. Sum and midpoint at 100 and 600 puzzles have three seeds for the shared, per-value and `plus` models (`docs/ARITHMETIC.md`, "Three seeds"), and so do the two scaling models of Section 7.7 ("The scaling kernel") and the gated models there ("The rule gate") and the models of "The repair" ("The row descriptor reads the steps"); every other cell is one run, so roughly ±4 points. The untrained fit of Section 7.8 and the rows per orbit in Section 7.5 are from "Two checks without training". Notes for you are at the end.
- **ch8_discussion_conclusion.md:** Draws on Chapters 5 to 7. The quotations from Chollet (2019) were checked against the arXiv PDF on 9 November (long outline, Chapter 7). Notes for you are at the end.
- **appendices.md:** Six appendices: the PyNeuraLogic pitfalls met in this work (A), the checks made on the data (B), the facet template as PyNeuraLogic prints it (C), the full result tables (D), two negative results (E) and how to reproduce the main results (F). Numbers are from `docs/RESULTS.md`, `docs/FINDINGS.md` and `docs/SYMMETRY_RESULTS.md`. Notes for you are at the end. Conventions are those of the chapters. Cells are matched % with exact % in brackets. I-RAVEN numbers are on the unseen validation puzzles (300–1299 of every layout) unless marked *test*; I-RAVEN-X numbers are on puzzles 6300–7299. The epoch is always picked on separate selection puzzles.
