# Findings, in the order they were made (2026-10-28)

The findings that still hold, oldest first. Superseded models and results are left out: the porting-stage
PyNeuraLogic models, the GNN-style PyNeuraLogic model, the original torch GNN, the April spatial GNN, the
hub and ordered templates, and facet runs before the selector fix. Details and numbers:
`docs/PYNEURALOGIC_NOTES.md` (section numbers below). Plan: `docs/THESIS_PLAN_symmetry.md`.

Metrics: *matched* = the candidate closest to the generated answer is right (ties count as wrong);
*exact* = the generated panel equals the answer. I-RAVEN numbers are unseen validation puzzles 300–1299 per
layout, I-RAVEN-X numbers puzzles 6300–7299; the epoch is picked on separate selection puzzles. Test splits
are untouched except where marked.

## September 2026: the benchmark and its tooling
1. **RAVEN leaks the answer through its candidates; I-RAVEN does not** (§1.1). Letting the 8 candidates
   exchange messages gained 15.4 points on RAVEN and lost 20.2 on I-RAVEN. So: I-RAVEN only, and candidates
   never see each other. The later generative models never see candidates at all.
2. **I-RAVEN's XML lists only the occupied slots** (§10). Deriving a slot from `Position.index(bbox)`, as
   the inherited code did, made about a third of grid puzzles unanswerable. Slots now come from box
   geometry (`gnn.data.canonical_positions`), used by all current code.
3. **`Angle` is a distractor**: no rule governs it (§10).
4. **The data is clean** (`sanity_checks.py`). No train/test duplicates, uniform answer index, a
   candidates-only shortcut at chance on I-RAVEN (90–100% on RAVEN), XML labels match the images.
5. **PyNeuraLogic pitfalls** (§3; later items §17, §20). Each costs debugging time without a clear error:
   - embeddings must be weighted facts;
   - every sample needs its own query atom;
   - `build_dataset` may be called once;
   - CONCAT follows the grounder's order;
   - rules with one head add up;
   - an unweighted body atom adds its value;
   - chain pruning drops the transformation of single-rule atoms;
   - native training needs `iso_value_compression=False`;
   - `Transformation.NORM` rescales any non-zero vector to full size and leaves a zero vector at zero, so a template
     without biases jumps at an all-zero input: in the facet model value 0 is the all-zero thermometer code, and 1e-6
     in its place flips outputs by 1.0 (found 2026-11-13 with soft codes from the recognizer; item 39).

## October 2026, first half: perception and the symbolic models
6. **Perception is nearly solved** (§11.1–11.3). The ResNet-18 frontend, trained on RAVEN, reads 98.9% of
   I-RAVEN test panels exactly and all 16 panels of 92.6% of puzzles. The misses sit in distribute_nine
   and in_distribute_four. Caveat: RAVEN and I-RAVEN test share some images.
7. **Symbolic templates (2026-10-19, §15)**, the "rules given" level: relations are written as crisp rules
   and only their weights are learned.

   | model | I-RAVEN test, XML labels | end to end (ResNet-18) |
   |---|---|---|
   | rules | **98.33%** | **96.31%** |
   | primitives | 97.55% | 95.71% |
   | non-learned reference, rules | 97.78% | 95.74% |
   | non-learned reference, primitives | 96.42% | 94.73% |

   - The end-to-end loss sits where perception is imperfect.
   - The learned primitive weights rediscover RAVEN's rules.
   - Training barely matters: equal start weights already give 96–98%, and random starts converge to the
     same accuracy.

## 2026-10-24/26: the facet model, a generative model on per-puzzle graphs
8. **Generating the answer instead of choosing it** (§18.1). The model outputs the missing panel's
   description and only then picks the closest candidate, so candidate shortcuts are impossible by
   construction. Exact generation becomes a stricter metric: on I-RAVEN-X 3×10, matching overstates what
   a model knows (85% matched, 15% exact).
9. **The representation moved the numbers, training did not** (§18.3, distribute_nine):

   | template | matched |
   |---|---|
   | one node per object (hub) | 24.7 |
   | a weight per column pair (ordered) | 31.4 |
   | one node per object attribute, thermometer values (facet) | 73.6 |
   | facet, 6000 puzzles | 78.0 |
   | facet, widths 32 → 64 → 128 | 90.1 |

   Cross-entropy, training on all three rows and turning chain pruning off did not help. Attribute
   granularity was the decisive step.
10. **Visible selector atoms hid the positions (2026-10-25, §17).** Unweighted structural atoms added +2
    per object to every occupancy dimension. Declared hidden, the three grid layouts went from 76.9% to
    92.4%.
11. **Facet model results** (§18.4). All in PyNeuraLogic's own training, without a torch bridge:

    | dataset | matched (exact) | notes |
    |---|---|---|
    | I-RAVEN, all 7 layouts | **97.3%** (69.1%) | test split: 97.19 (68.35), item 40 |
    | I-RAVEN-X 3×3 | **99.2%**, 100% from epoch 15 | |
    | I-RAVEN-X 3×10, values 0–50 / 0–100 / 0–1000 | 85.4 / 84.9 / 81.2 | 89.1 / 86.3 / 85.4 after 150 epochs |

    I-RAVEN by layout: center 99.3, 2×2 95.5, 3×3 89.8, L-R 99.5, U-D 99.7, in-out 99.6, in-out grid 97.8.
    150 epochs gain nothing over 60 on I-RAVEN.
    Arithmetic is no weak spot on I-RAVEN: 97.3% matched with an Arithmetic rule and without. Exact per
    rule over type, size and colour: Constant 83.7, Progression 89.7, Distribute 88.7, Arithmetic 83.8
    (I-RAVEN-X 3×10: Arithmetic 18.6). The weak part is Number and Position under a changing count (44–71%).
12. **Against the literature** (§18.5–18.6).
    - On I-RAVEN with learned rules: 97.3 against ARLC's 92.6. ARLC trains on 2×2 only, a different
      protocol.
    - Grids are the weak spot of every learned model. ARLC's grid scores follow exactly from "position
      decisions at chance".
    - Per-layout accuracies are not comparable across layouts, because the candidates differ in different
      attributes.
13. **Cost** (§18.6). Grounding takes 10–165 ms per puzzle, once. All of I-RAVEN trains at about 14 min per
    epoch on CPU.
14. **What limits 3×10 (2026-10-26, §20.1).** Exact value per rule at values 0–50: Constant 93,
    Progression 77, Arithmetic 19, Distribute 10.
    - Distribute is a limit of the structure: the answer reads rows 1–2 only through an average.
    - At 0–1000, precision is a limit of the value code.
    - The model memorises training puzzles fine.
15. **Gated copy links** (§20.2–20.4), an operation prior: "the answer may be a context value". No rule is
    named.

    | values | facet | facet + copy links | ARLC | LLMs |
    |---|---|---|---|---|
    | 0–50 | 85.4 | **97.8** (60.0 exact) | 95.0 | – |
    | 0–100 | 84.9 | **96.2** | 95.0 | at most 84 |
    | 0–1000 | 81.2 | **94.3** | 90.6 | at most 84 |

## 2026-10-27: symmetries of the model and of the data
16. **Theorem: a template is blind to the constants it does not name** (§21.2). Renaming such constants
    leaves every atom value unchanged. So the facet model has these symmetries built in, all verified to
    about 1e-16 on random weights:
    - the order of objects in a panel;
    - the order of the candidates;
    - rows 1 ↔ 2;
    - the two components of left-right and up-down, for attributes;
    - size ↔ colour.
17. **Breaking a symmetry lets a shortcut in** (§21.5–21.6). Per-panel copy gates name panels, and they
    learned Distribute by address.
    - Swapping rows 1 and 2 drops their Distribute puzzles from 98.3% to 39.5%.
    - Content gates keep the symmetry: 94.9% (52.3 exact), unchanged by the swap.
    - A shuffled-source control with the same weights and connections gains only 1.9 points (87.3), so
      the copy-link gain (12.4) is the information carried, not extra connections. Extra connections can
      even hurt: gates that start open swamped the model. A blank-code control would close the last gap.
18. **Which symmetries the data has** (§22).
    - **Rows 1 ↔ 2** are interchangeable in the I-RAVEN generator: each row is generated independently, and
      Distribute's direction is 50/50. One small exception: for colour Arithmetic the generator re-draws
      row 2 so that + and − stay distinguishable. The swap is also a data symmetry of I-RAVEN-X 3×3.
    - **On I-RAVEN-X 3×10, rows 1 ↔ 2 are only a task symmetry.** Distribute rows are row 1 shifted by a
      constant step, which no longer fits after a swap, so the generator never produces the swapped puzzle.
      Its answer, "the value row 3 lacks", is unchanged. Building the swap in costs nothing and removes the
      address shortcut of per-panel gates.
    - **Other symmetries:** size ↔ colour is a data symmetry of I-RAVEN-X. Value reflection breaks
      Arithmetic.
    - **The weak spot:** I-RAVEN-X Arithmetic follows a symmetry the facet model lacked, the order of the
      operands.

## 2026-10-28: baselines, and symmetry as the thesis
19. **Plain torch models on the facet task** (§23). An MLP and an all-to-all object GNN get everything but
    the architecture: same targets, selection and metrics.

    | | MLP | GNN | facet |
    |---|---|---|---|
    | I-RAVEN, equal weights | 60.6 | 87.2 | 97.3 |
    | I-RAVEN, best of 4×–64× weights | 71.0 | 88.5 | |
    | I-RAVEN, best regularized (weight decay) | 72.8 | 91.3 | |
    | 3×10, values 0–50, matched | 83.5 | 71.6 | 85.4 |
    | 3×10, values 0–50, exact | 3.9 | 1.2 | 15.0; 60.0 with copy links |
    | 3×10, best regularized, matched (exact) | 84.4 (2.6) | 84.0 (3.3) | |

    - **Equal weights:** the GNN matches the facet model wherever a component holds one object, and loses
      15–25 points on the grids.
    - **More weights (4× to 64×) do not close the gap.** Larger models peak earlier.
    - **Regularization:** weight decay helps most. Teaching the row symmetry by augmentation adds 1–3
      points.
20. **I-RAVEN-X's generator draws Distribute values up to maxval inclusive**, one more than the codes
    hold (§23). The facet model's exact scorer decoded the answer through a description that cannot hold
    maxval, so it capped exact at 95.6% on 3×3 and 98.4% on 3×10; fixed and rescored 2026-11-08. The facet
    model's 3×3 exact is 97.0, not 92.7 (copy links 99.3, content gates 98.5, row fold 98.9); 3×10 moved at
    most 0.6. Matching, epoch choice and the per-rule exact values were never affected.
21. **Thesis framing:** built-in symmetries as inductive bias, measured on training data needed, accuracy
    at fixed weights, and generalization. What a symmetry σ does to valid puzzles decides its cost:
    - **σ keeps puzzles valid** (a data symmetry): building it in is free.
    - **σ makes them invalid** (out of support): no representational cost.
    - **σ keeps them valid but moves the answer the wrong way** (conflicting): a hard ceiling.

    So tying columns 1 and 2 does not make Progression unlearnable.
22. **Operand order built in: column roles (§24).** Columns are first / middle / last, and the middle ones
    share one rule. The symmetry is exact (4e-16).

    | 3×10, values 0–50 | matched | exact | Arithmetic exact | weights |
    |---|---|---|---|---|
    | named columns | 85.4 | 15.0 | 18.6 | 1,411k |
    | **column roles** | **94.1** | **29.2** | **40.5** | 288k |
    | copy links, content gates | 94.9 | 52.3 | 30.0 | 1,436k |

    - The predicted gain on Arithmetic is there, and beyond what copy links reached.
    - Progression (+17), Distribute (+12) and Constant (+5) improved too, which was not predicted.
    - Three seeds (2026-11-22; RESULTS §7): named 87.4 ± 1.8, roles 94.2 ± 0.2 matched; Arithmetic exact 20.7 ± 2.2
      against 39.6 ± 1.3. The first named run was the weakest: per seed the gain is +6.8 ± 1.6 matched, and of the
      unpredicted gains only Distribute's holds (+11.6 ± 0.7); Progression is +6.8 ± 8.9, Constant +1.7 ± 3.2.
    - Swapping middle columns or rows changes nothing on the trained model. The named models lose 15–20
      points of matching under the same swaps.
    - With content gates: 96.3% matched, but 43.7% exact (Distribute falls). Still a quarter of the weights.
    - At equal weights (widths 64/144/288, 1,390k) the roles model matches 94.9% against 85.4% for named
      columns: the gain is the symmetry, not the smaller model. Width does not help it (94.1 at 288k).
    - **No transfer to other row lengths.** On rows of 5 and 20 they match 66 and 50%; only Constant
      survives. The roles are blind to the order of the middle columns but not to their number.
23. **A recursive row fold (§25).** This is the construct only NeuraLogic expresses: one transition
    per column, the same at every step, and independent of the row length. A toy recursive template
    trained on rows of 3–5 values summed rows of 8–10 exactly.

    | 3×10, values 0–50 | matched | Progression exact | Arithmetic exact | rows of 5 / 20 |
    |---|---|---|---|---|
    | column roles | 94.1 | 94.3 | 40.5 | 65.6 / 50.1 |
    | **row fold** | 92.2 | **99.2** | 36.2 | 53.8 / 55.1 |
    | row fold + content gates | 93.7 | 89.8 | 29.4 | 61.0 / 60.5 |

    - Progression is the fold's own symmetry (one step, the same everywhere), and the fold is best of all
      models on it. Three seeds: 97.2 ± 1.7 exact, against 92.8 ± 1.6 for the roles (+4.4 ± 0.9, every seed) and
      86.0 ± 7.5 for named columns; matched 91.6 ± 0.6, Arithmetic 32.4 ± 4.6 (RESULTS §7).
    - It reads rows in order, so it loses to the roles on Arithmetic, as predicted. Swapping middle columns
      costs it only 0.5 points: the data teaches what the template leaves open.
    - **Trained on one row length, it does not transfer** (54–61% on rows of 5 and 20, only Constant
      survives). Recursion makes transfer possible, not automatic; training on rows of 5 and 10 together does not
      fix it either (finding 33).
    - **On I-RAVEN (rows of 3) it is below the facet model:** 95.8% matched (63.0 exact) against 97.3 (69.1), at
      283k weights against 266k. Arithmetic exact drops most (75.9 vs 83.8); Progression gains nothing (88.8 vs
      89.7), since with three panels the facet model already reads each position with its own weights. The fold
      levels off by epoch 20 (96.1) and oscillates after, where the facet model keeps improving to epoch 55.
      So the fold is kept as an I-RAVEN-X model only (decided 2026-10-30).

24. **Final baselines (§23.4): the regularized setting on every dataset, 2 seeds, and up to 16× weights.**
    - The strongest object-level baseline on I-RAVEN is the regularized GNN at 16× the weights: 93.8 (63.7),
      3.5 points below the facet model and 14.6 below on the 3×3 grid.
    - On I-RAVEN-X 3×10 no size or setting generates exactly more than 4%; more weights hurt.
    - 3×3, values 0–10 is closest: regularized MLP 95.1 (74.3) against 99.2 (97.0).
    - The attribute-level GNNs are stronger: on I-RAVEN, B (symmetric) 96.8 (67.9) and A 95.0 (62.5), so
      granularity does most of the work there and the built-in symmetries add a little. On 3×10, B learned
      much worse than A (85.8 vs 95.0, one seed), and A's lead partly comes from reading Distribute by
      position: swapping rows drops it to 55.5.

25. **The value axis (§27): values as constants, a convolution along them.** Closer to the symbolic
    template than to the facet model: it is given primitives (exact sums through a `plus` relation, value
    counts per row, offsets of ±2 from the last panel) and learns which rule applies and how to combine
    them. 10.8k weights.

    | 3×10, values 0–50 | matched | exact | Arithmetic exact | on values 0–100 (never seen) |
    |---|---|---|---|---|
    | value axis, 100 puzzles | 99.0 | 97.3 | 97.2 | 98.1 (96.8) |
    | value axis, 600 puzzles | 99.9 | 99.9 | 99.8 | 100.0 (99.9) |
    | value axis, 6000 puzzles | **100.0** | **100.0** | **100.0** | **100.0** (100.0) |
    | value axis without `plus`, 6000 puzzles | 84.0 | 59.6 | 11.1 | 78.9 (60.5) |
    | facet, column roles, 6000 puzzles | 94.1 | 29.2 | 40.5 | cannot run |

    - No weight belongs to a value, so a model trained on values 0–50 runs on 0–100 at the same accuracy.
    - Rows of another length transfer only partly: rows of 5 give 60–90% matched (more with more training
      data); Distribute breaks most. Rows of 20 transfer better: 92–99% matched, with Constant, Distribute
      and Arithmetic at 99–100% exact and Progression dropping (51–89%).
    - Without `plus`, Constant, Progression and Distribute stay at 100% exact and Arithmetic falls to 11%:
      the sums come from the given relation, not from the convolution. The facet model learns approximate
      sums (40.5% exact, mostly within ±2), which is why it matches more often (94.1 vs 84.0).
    - **Trained on 3×3** (values 0–10, rows of 3): 99.8% matched (99.7 exact) against the facet model's 99.2
      (97.0); exact 100 / 100 / 100 / 99.4 per rule. The same checkpoint on 3×10 with values 0–50, never seen:
      65.2% (46.2); Distribute 100 and Arithmetic 76.7 exact carry over, Progression does not (36.9). The
      76.7 goes through the given `plus`, so it does not rank against models that learn the sum.
    - Exact here is the argmax of the value distribution (corrected 2026-10-31, §27): read through the facet
      model's thermometer code it had been the distribution's median, up to 16 points lower.
    - What the symmetry contributes is open until the runs with weights per value (the symmetry removed,
      335k weights) are in.

26. **Can the convolution be kept without `plus`? (2026-10-29, question and answer verbatim.)**

    Question: "im curious if the convolution can be kept without the plus defined"

    Answer:

    > **Can the convolution be kept without plus?** Two separate answers.
    >
    > **1. Keep the convolution, drop plus: that's the no-plus run.** It's still grounding on a busy node, so there are no results yet. I expect Constant, Progression and Distribute to stay near 100%, and Arithmetic to collapse. Without plus, the only clues the model has at a candidate value are its offset (±2) from the last panel and how often it appears in the rows. Among the 506 unseen Arithmetic attributes, the answer is within ±2 of the last panel in only 19%, and appears in rows 1–2 in 27%. That's roughly its ceiling, and the model can't tell which of those cases it's in.
    >
    > **2. Can a convolution learn addition without being told? Not with the kernel we have, but nearly.**
    > - **Why the current kernel can't:** it has 5 learned weights for offsets −2…2, so it can only shift by fixed amounts. Adding means shifting the running sum by however much the next panel holds, a shift that depends on the input. In convolution terms, the kernel has to be the other operand's value distribution: (p ∗ q)(v) = Σₖ q(k)·p(v−k).
    > - **What that needs:**
    >   - shifts across the whole range, not ±2;
    >   - a link between "value w" and "shift by w". That link is just knowing where zero is: w is the value that lies w steps above zero.
    > - **The rule:** with full-range shift facts and zero, plus doesn't have to be a table; one rule derives it: `acc(Q, V) :- acc(P, U), x(Q, W), off(zero, W, K), off(U, V, K)`. In words: find the shift K that takes zero to W, then move the running sum U by K.
    >
    > This connects back to your shift question. The template would then name exactly one value, zero, and everything else stays shift-symmetric. That's the least addition can need: (u+k) + (w+k) ≠ (u+w) + k precisely because a sum depends on where zero is.
    >
    > **What's learnable here: the zero itself.** Give zero(Z) a learned weight per value, normalized over the values, and the model has to discover from data that v0 is the origin. Addition would then be convolution plus one learned origin, and nothing else value-specific.
    >
    > To be honest about it: the full-range shift facts hold the same information as the plus table (about 1,300 facts either way). What changes is that addition stops being given as a primitive. It's built from translation plus an origin, and the origin is learned. That's a cleaner story for the thesis than "we handed it plus".

    Added afterwards: the no-plus run has finished. Constant, Progression and Distribute stay at 100% exact,
    Arithmetic falls to 11.1% (item 25), as the answer predicted.

27. **The symmetries help through weight sharing, most with little data (§21.2, §26, §26.1, §27).**

    **Results** (unseen puzzles, matched % (exact %), one seed each):

    | | training puzzles | shared | labels | separate weights, same widths | separate weights, same weight count |
    |---|---|---|---|---|---|
    | components (I-RAVEN left-right + up-down) | 100 per layout | **86.6** (40.4) | 87.3 (41.3) | 75.6 (22.2) | 74.6 (19.6) |
    | | 1000 | **99.3** (95.6) | 99.6 (95.7) | 97.9 (84.5) | 98.5 (84.5) |
    | | 6000 | 99.9 (97.7) | 99.8 (98.7) | 100.0 (97.8) | 100.0 (98.8) |
    | attributes (I-RAVEN-X 3×10, column roles) | 600 | **66.3** (16.4) | 64.6 (14.2) | 52.5 (4.1) | 53.0 (3.6) |
    | | 6000 | **94.1** (29.2) | 93.9 (28.6) | 90.5 (17.8) | 88.1 (17.6) |

    Over three seeds (2026-11-22; RESULTS §7) the small-data gains hold: components at 100, 88.0 ± 1.9 against
    73.8 ± 1.4 (+14.2 ± 1.9); attributes at 600, 66.2 ± 2.7 against 52.7 ± 0.7 (+13.5 ± 2.1), and at 6000,
    94.2 ± 0.2 against 87.9 ± 1.6 (+6.3 ± 1.7). Labels stay within 1.5 points of the shared model (89.5 ± 2.1, 64.8 ± 2.1).

    | value symmetry, value-axis model (same given primitives) | shared across values (10.8k weights) | weights per value (335k) |
    |---|---|---|
    | 100 training puzzles | **99.0** (97.3) | 38.3 (4.0) |
    | 600 | **99.9** (99.9) | 92.7 (81.0) |
    | 6000 | **100.0** (**100.0**) | 98.4 (97.4) |

    **Why weight sharing is what a symmetry is, in a template.** The theorem of §21.2 says a renaming σ of
    constants (k0 ↔ k1, type ↔ colour, v17 → v19) leaves every output unchanged when two things hold:
    - the template names none of the renamed constants. Then every ground rule rθ is paired with the ground
      rule r(σ∘θ) of the same rule r, and **weights belong to rules, not to groundings** (step 4): both
      groundings use the same weights. That pairing is the weight sharing. Writing the component as a variable,
      `vF(P, C, A) :- ...`, means one weight set serves k0 and k1;
    - the transformed puzzle's facts are exactly σ of the original's: the facts may not tie a constant to
      anything the renaming does not move.

    There are two ways to break the symmetry, and they violate different conditions:
    - **Labels** (`--label-components`: facts `cid(k0) = [1, 0]`, `cid(k1) = [0, 1]` fed into every
      panel-component summary; `--label-attributes`: a one-hot `aid(A)`). The template is unchanged and still
      shares every weight. But the facts now tie each component to a fixed vector. Mirroring a puzzle moves
      the objects from k0 to k1 while `cid(k0)` stays `[1, 0]`, so the mirrored puzzle's facts are no longer a
      renaming of the original's, and the second condition fails. With random weights the output changes
      (0.37 for the mirror, 0.41 for type ↔ colour).
    - **Separate weights** (`--per-comp-weights`, `--per-attr-weights`, `--per-value`). Every rule with weights
      is written once per component (attribute, value) with that constant in it. The template now names the
      constants, the first condition fails, and the pairing of step 4 is gone: k0's rules and k1's rules have
      different weights.

    **Why the labels broke the symmetry and the model still trained as well, and stayed almost symmetric.**
    - Everything except the label is still shared. A labelled model adds only 320 (components) or 512
      (attributes) weights to 266k / 288k, a learned vector per component added before the tanh of `vF`. Every
      other weight still sits in a rule with the component as a variable, so every training puzzle, left or
      right, trains the same weights. The data efficiency of the shared model is untouched.
    - The label carries no information about the answer. Left-right and up-down components are generated by
      the same rule distribution (a data symmetry, §22), so under the data the best predictor treats k0 and k1
      alike. The gradients on the two label vectors come from statistically identical samples and pull them the
      same way; nothing rewards a difference.
    - So the trained labelled models are almost invariant: mirroring or swapping size and colour moves them by
      0.2–1.1 points (three seeds). Labels test whether *exact* invariance matters on top of sharing, and it does not, here.

    **Why separate weights cost accuracy.**
    - Each copy of a rule trains only on its own component (attribute, value). At 100 puzzles per layout each
      component's rules see half the examples the shared rules see, and must learn the same function from
      them independently.
    - It is the sharing, not the capacity: the same widths (2–3× the weights) score like the matched count
      (75.6 vs 74.6, 52.5 vs 53.0).
    - The data teaches the symmetry eventually: with 6000 puzzles per layout both copies learn the same function
      and the gap is gone (100.0 vs 99.9). The built-in symmetry is worth most where data is scarce.

    **The value symmetry in the value-axis model, and how its convolution works.**
    - *Values are constants.* A panel's attribute is a fact `x(P, A, v17)`. Background facts give the axis its
      geometry: `off(U, V, k)` for V = U + k, k = −2..2. The plus relation gives addition.
    - *The kernel.* `h(P, A, V) :- W_k · x(P, A, U), off(U, V, k)`, one learned 32-number code W_k per offset k:
      160 weights. Each panel has one value, so the input is a single 1 on the axis, and the convolution stamps
      the five codes around it: panel at 17 gets W₋₂ at 15 … W₊₂ at 19, and no atom elsewhere. h(P, A, V) means
      "V is k above P's value". The trained codes are about as distinct as random ones (cosines −0.38 to 0.07);
      their job is to tell the offsets apart.
    - *Comparing panels.* For neighbouring panels P, Q, `pv(P, Q, A, V)` = tanh(B·h(P, A, V) + C·h(Q, A, V)) at every
      value where either window reaches, and `pd(P, Q, A)` sums it over V. The list of pairs met along the axis
      depends only on the offset between the two values, so pd is an offset code: 0 for Constant, ±1 / ±2 for
      Progression's step, "far apart" for anything beyond 4. Averaged over a row (`rd`) and combined with value
      counts and the plus checks, it says which rule holds (`ctx`), without naming any value.
    - *Scoring every value.* `o(A, V)` and `o2(A, V)` are two tanh layers with the same weights at every V (a 1×1
      convolution). At each candidate value they read ctx, the last known panel's code h(qlast, A, V) (so
      "step +2" fires where V is 2 above it), how often V occurs in each row (Distribute), and the plus / minus
      answers at V (Arithmetic). A softmax over V gives the answer.
    - *The symmetry.* V appears only as a variable in every rule with weights, so by §21.2 shifting all values
      moves the output with them (checked with random weights: exact to 8e-17 without plus; with plus wherever
      the shift does not change a sum, since addition depends on where zero is).
    - *The ablation* (`--per-value`) writes each of those rules once per value with its own weights: a kernel per
      (value, offset), a comparison and pooling per value, scoring layers per value. Same primitives, 335k
      weights. A pattern learned at 17 now says nothing about 33. With 100 puzzles there are 300 answers (three
      attributes) spread over 51 values, about 6 per value, and each value's scoring layers must learn from
      those alone: 38.3% matched instead of 99.0.
    - *Why even Arithmetic falls (97.2 → 22.3% exact at 100 puzzles), though the sum is given.* The sum `sa(A, V)`
      is exact, but deciding "the rows pass the plus check, so put the answer where sa is" is done by the
      scoring layers, now separately at each value. About 17% of attribute values are Arithmetic, so 100
      puzzles give roughly one Arithmetic answer per value to learn that from.

    At 6000 puzzles weights per value reach 98.4% matched (97.4 exact; Arithmetic 94.9%), about what shared
    weights reach with 100 puzzles (99.0, 97.3 exact; Arithmetic 97.2). Without the symmetry the model needs
    some 60 times the data for the same accuracy. On rows of 5 it transfers worse: 55.7% (17.0 exact) against
    89.8% (78.9) with shared weights. On rows of 20 it transfers best of all (98.9%, 98.2 exact, against 98.6%,
    63.7 exact); with 100 and 600 puzzles shared weights stay ahead there too (97.4 vs 33.5, 92.4 vs 87.6).

28. **A plain ViT, end to end, comes close on images (2026-10-31).** It reads the matrix as one image and
    generates the answer's code; its target and the matching still use the XML labels, so this is not the
    literature's setting.
    - Width 64 (335,038 weights): 79.1% matched (37.6 exact). Width 224 (3,752,958 weights): 93.4% (64.4), the level
      of the strongest object-level baseline on attributes (93.8) and 3.9 points under the facet model.
    - No rule is its weak spot: type, size and colour exact 82.4 / 89.7 / 86.6 / 86.3 for Constant, Progression,
      Distribute, Arithmetic (facet 83.7 / 89.7 / 88.7 / 83.8), and matched 93.8 with Arithmetic, 92.9 without.
    - The loss is on grids. It matches the facet model on single-object layouts (99.3–99.6) and falls behind on
      2×2 (93.0 vs 95.5), 3×3 (81.8 vs 89.8) and the in-out grid (81.1 vs 97.8).
    - Resolution does not explain that at this amount of data. Trained on the grids alone (18,000 puzzles), finer
      patches and larger images stay near the control, 33–47 against 28–43, and all three memorise. The same 18,000 grid puzzles with
      18,000 single-object puzzles added reach 73–83, which suggests the ViT picks up the rules on the easy layouts.
    - It needs 14 times the facet model's weights, and it shares weights too: one filter for every patch, one
      block for every token. The facet model's own end-to-end score is still missing, so the two have not yet been
      compared on the same inputs.

29. **Not every rule is a symmetry: Arithmetic is a relation.** Constant, Progression and Distribute each have a
    transformation that takes one panel or row to the next: nothing, a fixed shift, a permutation. A sum or
    difference has none, because the step depends on the content of two panels and differs from row to row.
    - The value transformations that keep the other three rules break it: reflecting the values keeps 100% of them
      and 0% of Arithmetic (§22.2), and shifting them does the same.
    - On 3×10 the value axis's convolution carries the other three at 100% exact with no `plus`, and Arithmetic falls
      to 11.1, against 100 with it (item 25). On small ranges it stays at 70.7 (3×3) and 65.7 (I-RAVEN),
      partly because few values leave few sums: a lookup on differences alone gets 47.5 and 43.4 there, 2.4 on 3×10.
      The relation has to be given or learned; the shift symmetry does not contain it.
    - It can be read as a shift by a panel's own value, but that needs a fixed zero, which breaks the shift symmetry
      (item 26).
    - It is a relation with symmetries of its own. Its operands can be reordered, and building that in lifts Arithmetic
      exact from 18.6 to 40.5 (item 22). A row also stays valid when a, b and c move by s₁, s₂ and s₃ with s₁ + s₂ = s₃, so
      one operand can move if the sum moves with it. Equal shifts, the ones the convolution has, are what this
      excludes. The `plus` rule follows these moves (`docs/ARITHMETIC.md`).
    - For models that have to learn the sum it is usually the lowest rule. On 3×10 it is lowest in 8 of 13, among
      them all five with copy links or content gates; Distribute is lowest in the other five. Structure repairs
      Distribute (9.8 → 86.8); no learned model passes 40.5 on Arithmetic. On 3×3 it is lowest in the facet
      model (95.0 against 99.3 or more) and in the regularized MLP and GNN (64.4 / 69.4 against 87.7 or more). On
      I-RAVEN it is lowest in the row fold, the small ViT and the regularized MLP and GNN, level with Constant in
      the facet model, and second to noisy Constant in the big ViT, whose
      lowest rule on single-object layouts it is. The value axis's 100 is no counterexample, because the sum is
      given.

30. **Arithmetic is costly because it has many facts and the model has no symmetry to collapse them (2026-11-01).** Four rules,
    rows of 3, values 0–50 (`docs/ARITHMETIC.md`, RESULTS §11). Midpoint has as many facts as sum and keeps the
    shift symmetry; lookup has 51 facts and no symmetry. At 600 puzzles the shift-equivariant model gets 89.3 on
    midpoint and 6.9 on sum; weights per value get 17.8 and 19.3, and 97.5 on lookup. Without the symmetry midpoint
    is no easier than sum, so a rule's symmetry pays only if the model has it, and weights per value stay close to
    plain memorising of answered rows. Mirror, midpoint's triples with the columns moved, is harder than sum
    without the symmetry, for a reason the test does not separate. The MLP does not show lookup as easier (17.7
    against 26.4 at 6000), but it barely learns any rule here. One seed per model except the MLP; three seeds for
    sum and midpoint at 100 and 600 puzzles are item 42.

## 2026-11-05/06: after the meeting
31. **Given `plus` as background knowledge, learned rules learn to use it (3×3) (notes §27, `docs/ARITHMETIC.md`).**
    The supervisor's numbers as objects: values are nodes that know `plus`, and a learned state walks along a row
    through `plus`, adding or taking away the next panel's value, with no built-in sum, check or answer. On
    I-RAVEN-X 3×3, values 0–10, 6000 puzzles: Arithmetic exact 98.8, and 99.8 when the state can also move along
    `next` (without `plus`: 70.7; the rule given: 99.4).
    - The grounding adds exactly, and each kind of step's own matrix labels the path by the state's direction, even
      with untrained weights.
    - Training learned to read those labels: P(answer) 0.10 → 0.998.
    - It is one step, and trained on 3×3 it does not carry over to rows of 10 (Arithmetic 5.9 and 2.8).
    - **Trained on 3×10, eight steps:** Arithmetic 100 with `plus` steps only, where the grounding nearly gives the
      rule (the check atom exists in every Arithmetic row and in 1–4% of the others), and 88.7 when the state can also
      move along `next` (without `plus`: 11.1). Values 0–100 carry over (100, 84.9); other row lengths do not (rows of
      20: 44.8 and 1.3, the rule given 100).
32. **Addition needs only the zero and `next` (2026-11-06, `docs/ARITHMETIC.md`).** Two weight-free rules,
    u + 0 = u and u + (w + 1) = (u + w) + 1, derive exactly the `plus` table: trained checkpoints give the same outputs
    in both templates (max difference 2e-16). Trained from scratch on 3×3 with them: Arithmetic 99.6 and 100. The
    recursion is written in, not learned, and grounding it costs 22× on 3×10.
33. **Two training row lengths do not make the row fold length-independent (2026-11-06, plan M2.1, notes §25).**
    Trained on rows of 5 and 10 together, rows of 20 get 47.7% matched (fold) and 54.0 (column roles), against 55.1 and
    50.1 after rows of 10 alone. The prediction was a clear rise if the single training length were the cause; there is
    none, and Progression exact at rows of 20 falls to 2.9 in the fold.
34. **Sum and midpoint are equally symmetric; only midpoint's symmetry lies along the convolution (2026-11-08,
    `docs/ARITHMETIC.md`, "Sum and midpoint compared").** Each rule is one row moved by a two-parameter group of
    moves: (s₁, s₂, s₁ + s₂) for sum, (s₁, s₂, (s₁ + s₂)/2) for midpoint. The convolution's move, every panel +s, is
    in midpoint's group and not in sum's; one operand moved with the answer is in sum's and not midpoint's.
    - Learners that know only one move (`analysis/rule_orbits.py`, no network), 100 puzzles: sum 100 with its own
      move and 3.6 with the convolution's; midpoint 80.0 with the convolution's and 3.4 with sum's.
    - The shared value axis given sum's move (`plus` used by learned rules): sum 97.6 at 100 puzzles and 99.8 at 600,
      against 4.7 / 6.9 without it; midpoint unchanged (70.5 / 91.5).
    - The network learns one answer per orbit: of the midpoint rows whose first step no training row had, it gets
      0 of 101 right at 100 puzzles, against 58.9% of the others.
    - Three seeds of these cells: item 42.
35. **The row fold does not grok Arithmetic in the first round (2026-11-08).** On 1000 Arithmetic-only 3×10 puzzles (a
    diagnostic: the rule is known, so this asks whether it can learn the sum, not how it does on the benchmark),
    without decay it memorises (train 98) while unseen peaks at 63.4 and falls; with decoupled decay 1 at one update
    per puzzle it never fits (train ≤ 27). With decay 0.03 and 0.01 it memorises as without decay: unseen peaks at 65.7
    and 58.7 near epoch 100 and settles at 43–46 while train reaches 86–97, over 600,000 updates. No grokking at any decay.

36. **A wrong symmetry costs what the data predicts (2026-11-08, plan E4, RESULTS §7).** Facet model, I-RAVEN-X 3×3,
    values 0–10, each symmetry built in exactly.
    - Out of support, columns 1 ↔ 2 tied: 98.6 (95.1) at 6000 puzzles against 99.2 (92.7). No ceiling, as predicted;
      Distribute and Arithmetic lose 3 and 5 points exact. At 600 puzzles 92.1 (79.2) against 95.1 (82.2): Arithmetic
      loses 14 points exact and Distribute gains 5.
    - Conflicting, v and 10 − v given the same input code: 46.6 (13.4) at 6000 and 39.7 (14.0) at 600. 50–53% of
      attribute values are exact, just under the 54–56 ceiling the count gives.
    - So the three kinds of symmetry (F 21) behave as the classification says: a data symmetry is free, an out-of-support
      one costs little, a conflicting one sets a ceiling that more data does not lift.

37. **Without dropout and decay, the symmetric torch GNN is level with the best facet variant on 3×10 and has the best
    learned Arithmetic (2026-11-09, RESULTS §12).** Attribute GNN B at 1× (1.6M weights), 6000 mixed-rule puzzles, 425
    epochs, two seeds: 97.9 (45.5) and 98.2 (56.5) at the epoch picked on selection, Arithmetic exact 45.1 and 54.0; at
    the last epoch 99.2 (63.9) and 99.3 (61.5), Arithmetic 50.6 and 49.2. The regularized run had 85.8 (3.5) and 22.7.
    The facet family's best are 97.8 (60.6) with per-panel copy links and 40.5 on Arithmetic with column roles.
    - The symmetries are GNN B's too, so this supports them and not PyNeuraLogic as such.
    - The regularized recipe was chosen on I-RAVEN. GNN A, the object GNN and the MLP were not rerun without it, and
      the facet models trained for 60–150 epochs. The 3×10 comparisons with the baselines are open until that is done.
38. **A ViT slot recognizer reads 99.35% of panels exactly (2026-11-09, RESULTS §8).** The black-box ViT's backbone
    reading one panel, 325,718 weights: all 16 panels right in 95.69% of unseen puzzles (ResNet-18: 98.91% and 92.56%),
    3×3 grid panels 99.92% (ResNet-18: 96.02). The facet model has not been scored on its output yet.

39. **End to end, the facet model loses 0.18 points (2026-11-13, RESULTS §8).** On the recognizer's panels, candidates
    included, it matches 97.13% of the 7000 unseen puzzles (68.8 exact) against 97.31 from the labels; all 13 lost
    puzzles are in the in-out grid. The labels pass reproduces the checkpoint's record exactly. Soft codes add
    nothing, and raw ones break the model: value 0 is the all-zero code, the template has no biases, and `NORM`
    rescales any non-zero vector to full size, so 1e-6 in place of 0 flips outputs. That is a limit of the template as
    built: it cannot take graded inputs without retraining.

40. **On the test split the facet model scores 97.19% from labels and 97.03% end to end (2026-11-14, one pass,
    RESULTS §8).** 68.35 and 68.09 exact, 14,000 puzzles. Both are 0.1 below the validation numbers, so the validation
    puzzles were not overfitted by selection. The pass for this checkpoint is spent.

41. **The baselines' test pass confirms the validation table (2026-11-14, one pass each, RESULTS §1).** On the same
    14,000 test puzzles: GNN B 96.79 (67.84), GNN A 95.09 (62.51), object GNN regularized 90.8 ± 1.2 (2 seeds; 93.44 at
    16×), MLP regularized 72.7 ± 1.2, ViT width 224 93.10 (64.36), width 64 78.69. Every checkpoint first reproduced its
    validation record exactly. No checkpoint moves by more than 0.8 points between the splits and the order is unchanged. The
    facet model leads GNN B by 0.4 points (all of it on the 3×3 grid, 89.1 against 85.5) and, end to end, the ViT by 3.9
    points with a sixth of the weights.
42. **Three seeds leave the sum and midpoint results where they were (2026-11-14, `docs/ARITHMETIC.md`, "Three seeds",
    RESULTS §11).** Seeds 1–3 of the shared, per-value and `plus` models at 100 and 600 puzzles, 36 runs, one selection
    rule fixed beforehand (whole-puzzle exact on the selection puzzles). Exact % on the test rule, mean ± sd, at 100 /
    600 puzzles:
    - shared: sum 4.5 ± 2.3 / 6.2 ± 0.8 (ceiling 7.9), midpoint 60.3 ± 2.7 / 91.0 ± 3.4;
    - per value: sum 4.8 ± 0.6 / 19.4 ± 1.1, midpoint 5.0 ± 0.5 / 18.0 ± 0.9, so the twins are matched;
    - shared with `plus`: sum 98.2 ± 0.8 / 99.7 ± 0.4, midpoint 63.9 ± 3.9 / 91.7 ± 1.8.
    - Midpoint rows with a first step no training question had: 0 of 101 right at every seed, now from the checkpoints
      in the table; 72–78% of the 404 others.
    - One sentence changes: `plus` does not help midpoint (the first run's 70.5 at 100 puzzles was a high draw). The
      older `plus` cells on sum, 97.6 / 99.8, had been picked by exact on the test rule; by the common rule 98.2 / 99.2.
    - The 6000-puzzle cells, mirror, lookup and the facet rows are still one run each.
43. **A scaling symmetry reaches the unseen first steps a count named, and costs accuracy by merging midpoint with
    Progression; the cost is revised in item 47 (2026-11-15, `docs/ARITHMETIC.md`, "The scaling kernel", RESULTS §11).** The shared model with each row
    read in its own unit, the unit an unnamed constant: the first step (scaled; equivariant to v → kv + s on the ring
    of 101) or the largest power of two in it (doubled). Same 14,785 weights, symmetry exact with random weights.
    Predictions were made by counting before the 24 runs (seeds 1–3, sum and midpoint, 100 and 600 puzzles).
    - Held: of the 101 unseen midpoint rows with a new first step (100 puzzles), the doubled model is right on 50 / 43 /
      39 of the 72 whose odd part was seen and on 0 / 0 / 0 of the other 29; the shared model on none of the 101.
    - Held in one seed: the scaled model is right on all 505 midpoint rows from 100 puzzles in seed 2. In seeds 1 and 3
      it is right on a fifth to three tenths and continues the rest as a Progression.
    - Not predicted: the totals fall, doubled 49.9 ± 5.3 and 67.5 ± 2.0 against shared 60.3 ± 2.7 and 91.0 ± 3.4. A
      midpoint row whose first step is one unit starts like a Progression row (the data's Progression steps are ±1 and
      ±2), and the network continues it as one: 0 of 33 right for the shared model (first step ±2), 1 / 11 / 1 of 152
      for doubled (a power of two), 149 / 505 / 102 of 505 for scaled (every row), at 600 puzzles.
    - So the shared model's plateau near 93% on midpoint is its 33 rows with first step ±2: seeds 2 and 3 are right on
      470 and 469 of the other 472. The size of the first step was the cue for the rule; the complete rows also name
      the rule, and the network used them at one seed of three.
    - Sum: doubled 3.6 ± 0.2 / 4.1 ± 0.3 (ceiling 7.2); scaled at most 3.4, which is the 17 unseen rows with b = 2a, where continuing a row by its first step is right (the most one multiple can reach on these rows).
44. **The scaled model's failure at two seeds is the row descriptor's, not the answer layer's (2026-11-16,
    `docs/ARITHMETIC.md`, "The rule gate", RESULTS §11).** Item 43 left open why the network reads the rule from the
    complete rows at one seed of three.
    - Tested first, with predictions written before the runs: a gate from the rule descriptor on the answer layer's
      kernel features (`--gate`, 1,056 more weights, symmetries unchanged). Both predictions failed. Scaled with the
      gate: 23.2 / 20.2 / 24.0 and 20.2 / 20.2 / 30.5 on midpoint at 100 and 600 puzzles. Shared with the gate: none
      of its 33 rows with first step ±2 at any seed, totals 66.6 ± 5.5 and 90.3 ± 2.2.
    - Measured on the 24 saved checkpoints (`analysis/twins_probe.py`): the row descriptor `rd`, the pairs' features
      summed over all values. In the 22 runs that fail it is the same vector for a midpoint row and a Progression row
      (largest difference per feature 1e-14 to 7e-05). In the two runs that learn midpoint (scaled, seed 2, both sizes)
      it differs by 1.4 and 1.5.
    - The failed scaled runs use the value overlap instead: right on 90 to 107 of the 108 midpoint rows whose question
      row shares a value with a complete row, on 0 to 47 of the 397 others.
    - One cause for two failures: the shared model's 33 rows (every seed and size) and the scaled model's two outcomes.
      The cost in item 43 is therefore not shown to be a cost of scaling. A row descriptor that reads a pair's step
      directly was built next (item 47).

45. **The rule twins are a test for models that treat values as constants; with values as numbers they need no
    training** (2026-11-17; `docs/ARITHMETIC.md`, "Two checks without training"; nothing trained).
    - A least-squares fit of c = p·a + q·b on a puzzle's two complete rows, applied to its question row, gets sum 99.6,
      midpoint 99.8, mirror 99.8, Progression 98.6 to 99.3 and Constant 100 on the unseen puzzles. It gets 1.7 on lookup
      and 0 on Distribute Three.
    - So the comparison between sum and midpoint holds for models whose values are constants with an order. The thesis
      has to state that scope.
    - Row by row, the count holds: no training row in a row's orbit, never right (0 of 306, three seeds pooled); one
      training row, right 74% of the time at 100 puzzles and 99% at 600.

46. **The price of a symmetry on the twins is proved statement by statement, and one earlier claim was too strong**
    (2026-11-17; `docs/thesis/ch3_pricing_proofs.md`, `analysis/pricing_checks.py`; nothing trained).
    - The data saved has a formula from the table alone: the expected share of unseen rows reached by n training rows
      is the sum over orbits of P(o)·(1 − (1 − P(o))ⁿ). It gives the counting learner within 6 points (74.8 against
      80.0 with the shift at 100 puzzles, 22.0 against 21.6 without a symmetry at 600). Rows needed for 90%: 2,878
      with nothing built in, 107 with the shift, 55 with doubling, 1 with scaling.
    - **Too strong before:** "a shift-equivariant model cannot fit sum". The puzzle-wide shift alone does not cap sum:
      c₃ = b₃ + (c₁ − b₁) + (a₃ − a₁) uses only differences and is right on 507 of 507 sum rows. The 7.9% ceiling holds
      for learners that answer from the question row, and for the shared model because it never measures from the
      question row to another row. Chapter 7 (7.3, 7.4) is corrected.
    - **Also too strong:** "sum stays under its ceiling". It is 7.1 or less at every picked epoch, but 8 of 172
      evaluated epochs are above 7.9 (9.7 at the most), and 507 rows fix the rate only to 1.2 points. The model sits at
      the ceiling.
    - With scaling built in, every question row is the same input, so the rule can only come from the complete rows.
      That is proved (Statement 7) and is the reason behind item 43's cost.

47. **Scaling does not cost accuracy: the row descriptor did** (2026-11-17, 18 runs, predictions written first;
    `docs/ARITHMETIC.md`, "The row descriptor reads the steps"; RESULTS §11).
    - The change: the row descriptor also reads each neighbouring pair's step directly, as the first panel's kernel
      feature at the second panel's value (`value_axis.py --steps`, 2,048 more weights, the symmetries unchanged).
    - Midpoint, three seeds, 100 / 600 puzzles: scaled 100.0 at every seed at both sizes (it was 19.0 / 100.0 / 21.0 and
      29.5 / 100.0 / 20.2); doubled 89.2 ± 2.7 / 99.2 ± 0.2 (49.9 / 67.5); shared 59.9 ± 4.5 / 98.3 ± 1.9 (60.3 / 91.0).
    - The shared model's 33 rows with first step ±2, never right in twelve earlier runs, are all right at 600 puzzles
      at every seed, and 17 to 27 of them at 100.
    - The probe: the row descriptor separates midpoint from Progression in all 18 runs (0.87 to 2.0 per feature;
      below 0.0001 in 22 of 24 runs before). So items 43 and 44 are closed: one cause, one change.
    - The ladder of built-in moves now rises at every rung, as counted: 5.0, 59.9, 89.2, 100.0 at 100 puzzles.
    - Six predictions met; one missed by one row (the shared model right on 1 of 303 rows with an unseen first
      step, the rate of guessing). Not studied: why the pooled descriptor collapses.

48. **On sum, the steps change nothing: the cap is where the answer is read** (2026-11-22, 18 runs, prediction written
    first; `docs/ARITHMETIC.md`, "Sum with the steps"; RESULTS §11).
    - The three models of item 47 on sum, three seeds, 100 / 600 puzzles: shared 4.0 / 7.6, doubled 3.4 / 5.2, scaled
      1.2 / 1.6, against ceilings of 7.9, 7.2 and 1.4. Every mean is at or under its ceiling within sampling error.
    - So the descriptor was the cause on midpoint only. On sum the answer is read from the question row, and nothing
      measures from it to row 1, where c₁ − b₁ = a₁ shows the zero.
    - Next, not run: a template that measures across rows. It needs care, since a template that chains offsets has
      sum's solving function, c₃ = b₃ + (c₁ − b₁) + (a₃ − a₁), in it by construction.

49. **Rules along rows: the prior costs half the accuracy when the matrix is turned** (2026-11-22, no training;
    RESULTS §3, `jobs/pynl/submit_transpose.sh`).
    - On transposed I-RAVEN puzzles (rules down the columns, the answer in its place) the facet model, copy links with
      content gates and the row fold all fall to 45.0–45.1 matched (3.7–4.1 exact), from 95.8–97.4.
    - RAVEN's rules applied along rows keep 47.7 on the same kind of puzzles, generic row relations 72.5. The models fail
      as a row-rule solver fails; Distribute Three, whose Latin square survives transposing, keeps 82 exact.
    - This is the prior's cost off the generator, not its worth in data, which needs rows and columns both as lines.

50. **A data symmetry is worth the data it reuses, as counted: 2× for components, 2–3× for attributes** (2026-11-22/23, 9 runs, prediction written
    first; RESULTS §7, "The data factor").
    - Count: one sample per (puzzle, component); separate weights train each component's copy on its own samples only,
      and the two components have the same distribution. So separate weights at 2N puzzles give each copy what the
      shared network gets at N. Attributes: size and colour carry the four rules equally, type never Arithmetic, so the
      factor is 2 on Arithmetic and 2 to 3 overall.
    - Components, three seeds each: separate weights at 200 puzzles 88.1 ± 0.2 (42.2 exact) against shared at 100
      88.0 ± 1.9 (42.4). Met to a tenth of a point.
    - Attributes at 1200 against shared at 600: with Arithmetic 60.8 ± 1.2 against 63.0 ± 2.9 (factor 2, met), without
      64.0 ± 0.8 against 68.7 ± 2.8 (behind, as allowed). At 1800, 69.8 ± 0.9 against 66.2 ± 2.7 overall: the factor
      is between 2 and 3 (about 2.5), about 2 on puzzles with Arithmetic and about 3 on the rest. Met at both points.
    - Sharing buys data and, on components, nothing else. The value shift's factor (above sixty against a group of about
      fifty) comes with thirty times the weights, so it is not the symmetry's alone; Chapter 6 now says so.

51. **The test pass leaves Chapters 5 to 7 where they were** (2026-11-23, 187 checkpoints, one pass each; RESULTS §14).
    - Every final checkpoint behind an I-RAVEN-X, twins or components number, scored once on puzzles 8000–9999 (I-RAVEN-X,
      twins) or the I-RAVEN test split (components). The rule mix is the same as in the unseen range.
    - The symmetry gains: components at 100 puzzles +15.5 ± 0.9 (unseen +14.2 ± 1.9), attributes at 600 +12.9 ± 3.6
      (+13.5 ± 2.1), at 6000 +5.2 ± 1.3 (+6.3 ± 1.7). The counted data factor: separate weights at 200 puzzles 88.3 ± 0.6
      against shared at 100 88.4 ± 1.3; on attributes shared at 600 between separate at 1200 and 1800, about 2.5, 2 on
      puzzles with Arithmetic and 3 on the rest.
    - The twins: sum under each ceiling within sampling error, the ladder in the counted order, the shared model still
      with the steps 20 points under its count at 100 puzzles (59.4 against 80.0).
    - The largest move against its sampling error: the facet model on sum at 600 puzzles, 31.2 → 24.7 (one run, 2.6
      standard errors); the sign of its comparison with midpoint holds (2.9 points instead of 7.2). Column roles'
      Arithmetic on 3×10 drops 3.6 points at every seed, 39.6 ± 1.3 → 36.0 ± 2.1: 1.4 standard errors of the shared
      puzzle sample, in the direction their choice on those puzzles would leave. Their lead over named columns is
      +15.1 ± 1.8 on test instead of +18.9 ± 3.4.
    - Exact in `best.json` is the old scorer for facet I-RAVEN-X runs before 2026-11-08; the pass uses the fixed one.

52. **End to end, the learning curve loses 0.2–0.3 points at every size** (2026-11-23, RESULTS §8). With the recognizer
    retrained on the same 300, 1000 and 3000 puzzles per layout, the facet model gets 89.8, 93.9 and 95.8 matched,
    against 90.0, 94.1 and 96.1 from labels; the loss is almost all in the in-out grid. At 300 puzzles the pipeline is
    74 points above the ViT on the same puzzles (15.1 / 15.6). The supervision still differs: the recognizer is trained
    on the labelled objects of all 16 panels, the ViT on the answer's index only. Item 54 gives a black-box reasoner
    the same labels.

53. **A schedule bug hit 8 torch runs** (2026-11-23, fixed; corrected and rerun 2026-11-24). From 2026-11-02, when gradient
    accumulation (`--accum`) came in, `train_baseline.py`'s cosine schedule read a step count that each batch overwrote
    with its loss, so after warm-up the rate was full and zero in turn (`--schedule alternating` reproduces it). It was
    first taken to affect all 149 runs with `--schedule cosine`, and all were queued for a rerun; the reruns showed
    otherwise. The trainer of 2026-10-28 and 2026-10-31 in git has no such overwrite, and 99 reruns of runs made before
    2026-11-02 reproduce them, 95 to the last digit and 4 within float or GPU noise. The 8 runs that had the bug: the
    ViT at width 224 and its row-symmetric version on 3×10, the two finer-grid ViTs (RESULTS §7), and seeds 1 and 2 of
    the ViT with and without rows built in at 3000 puzzles. Their reruns with the fix all score higher, by 2 to 11 points
    matched, and replace them. One comparison changes direction: on 3×10 the ViT with rows built in was 3.0 points under
    the plain one (40.1 against 43.1) and is now 0.7 above it (46.0 against 45.3), and 6.7 above it on puzzles with rows 1
    and 2 exchanged. The gain from rows built in at 3000 puzzles stays the same, +20.0 ± 10.1 against +20.4 ± 11.1. The
    reruns of unaffected runs still queued were cancelled on 2026-11-25.

54. **With the same labels, a black-box reasoner needs the data the facet model does not** (2026-11-24, RESULTS §8).
    The recognizer of item 52 reads the context panels and a plain transformer over its slot states generates the
    answer (`--arch rec-tf`, fixed schedule, one seed). At 300 and 1000 puzzles per layout it gets 18.9 and 35.9, near
    guessing like the ViT, against the facet model's 89.8 and 93.9 end to end with the same recognizer; it memorises
    the training puzzles. At 3000 it gets 87.6 to 90.0, and at 6000 with 14× the weights 95.1 (69.9), 2.0 below the facet
    model in matched and level in exact; what remains is on the 2×2 and 3×3 grids. So at 300 puzzles the 74 points over
    the ViT are about 4 from the labels and 71 from the structure; from 3000 the labels explain most of the ViT's gap.
    Fine-tuning the recognizer with its label loss kept helps at 1000 (76.3), not at 300.

## Still open
- Rerun GNN A, the object GNN and the MLP on 3×10 without dropout and decay, and train a facet model as long (item 37).
- Seeds for the facet runs: most numbers are one run. Chapter 6's small-data comparisons and the 6000 attributes pair
  now have three seeds, and so do named columns and the row fold on 3×10 (RESULTS §7). The rest is one run.
- The attribute-level torch GNN pair (with and without symmetries), learning curves and transformed test
  sets (plan E1–E3). The wrong-symmetry controls (E4) are done (item 36).
