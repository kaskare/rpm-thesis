# Findings, in the order they were made (2026-10-28)

The findings that still hold, oldest first. Superseded models and results are left out: the porting-stage
PyNeuraLogic models, the GNN-style PyNeuraLogic model, the original torch GNN, the April spatial GNN, the
hub and ordered templates, and facet runs before the selector fix. Details and numbers:
`thesis/PYNEURALOGIC_NOTES.md` (section numbers below). Plan: `thesis/THESIS_PLAN_symmetry.md`.

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
   - native training needs `iso_value_compression=False`.

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
    | I-RAVEN, all 7 layouts | **97.3%** (69.1%) | test split not yet scored |
    | I-RAVEN-X 3×3 | **99.2%**, 100% from epoch 15 | |
    | I-RAVEN-X 3×10, values 0–49 / 0–99 / 0–999 | 85.4 / 84.9 / 81.2 | 89.1 / 86.3 / 85.4 after 150 epochs |

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
14. **What limits 3×10 (2026-10-26, §20.1).** Exact value per rule at values 0–49: Constant 93,
    Progression 77, Arithmetic 19, Distribute 10.
    - Distribute is a limit of the structure: the answer reads rows 1–2 only through an average.
    - At 0–999, precision is a limit of the value code.
    - The model memorises training puzzles fine.
15. **Gated copy links** (§20.2–20.4), an operation prior: "the answer may be a context value". No rule is
    named.

    | values | facet | facet + copy links | ARLC | LLMs |
    |---|---|---|---|---|
    | 0–49 | 85.4 | **97.8** (60.0 exact) | 95.0 | – |
    | 0–99 | 84.9 | **96.2** | 95.0 | at most 84 |
    | 0–999 | 81.2 | **94.3** | 90.6 | at most 84 |

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
    - Content gates keep the symmetry: 94.9% (51.7 exact), unchanged by the swap.
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
    | 3×10, values 0–49, matched | 83.5 | 71.6 | 85.4 |
    | 3×10, values 0–49, exact | 3.9 | 1.2 | 15.0; 60.0 with copy links |
    | 3×10, best regularized, matched (exact) | 84.4 (2.6) | 84.0 (3.3) | |

    - **Equal weights:** the GNN matches the facet model wherever a component holds one object, and loses
      15–25 points on the grids.
    - **More weights (4× to 64×) do not close the gap.** Larger models peak earlier.
    - **Regularization:** weight decay helps most. Teaching the row symmetry by augmentation adds 1–3
      points.
20. **I-RAVEN-X's generator draws Distribute values up to maxval inclusive**, one more than the codes
    hold (§23). The facet model's I-RAVEN-X exact scores are therefore slightly off, most at 3×3. Matching
    is unaffected.
21. **Thesis framing:** built-in symmetries as inductive bias, measured on training data needed, accuracy
    at fixed weights, and generalization. What a symmetry σ does to valid puzzles decides its cost:
    - **σ keeps puzzles valid** (a data symmetry): building it in is free.
    - **σ makes them invalid** (out of support): no representational cost.
    - **σ keeps them valid but moves the answer the wrong way** (conflicting): a hard ceiling.

    So tying columns 1 and 2 does not make Progression unlearnable.
22. **Operand order built in: column roles (§24).** Columns are first / middle / last, and the middle ones
    share one rule. The symmetry is exact (4e-16).

    | 3×10, values 0–49 | matched | exact | Arithmetic exact | weights |
    |---|---|---|---|---|
    | named columns | 85.4 | 15.0 | 18.6 | 1,411k |
    | **column roles** | **94.1** | **29.0** | **40.5** | 288k |
    | copy links, content gates | 94.9 | 51.7 | 30.0 | 1,436k |

    - The predicted gain on Arithmetic is there, and beyond what copy links reached.
    - Progression (+17), Distribute (+12) and Constant (+5) improved too, which was not predicted.
    - Swapping middle columns or rows changes nothing on the trained model. The named models lose 15–20
      points of matching under the same swaps.
    - With content gates: 96.3% matched, but 43.3% exact (Distribute falls). Still a quarter of the weights.
    - At equal weights (widths 64/144/288, 1,390k) the roles model matches 94.9% against 85.4% for named
      columns: the gain is the symmetry, not the smaller model. Width does not help it (94.1 at 288k).
    - **No transfer to other row lengths.** On rows of 5 and 20 they match 66 and 50%; only Constant
      survives. The roles are blind to the order of the middle columns but not to their number.
23. **A recursive row fold (§25).** This is the construct only NeuraLogic expresses: one transition
    per column, the same at every step, and independent of the row length. A toy recursive template
    trained on rows of 3–5 values summed rows of 8–10 exactly.

    | 3×10, values 0–49 | matched | Progression exact | Arithmetic exact | rows of 5 / 20 |
    |---|---|---|---|---|
    | column roles | 94.1 | 94.3 | 40.5 | 65.6 / 50.1 |
    | **row fold** | 92.2 | **99.2** | 36.2 | 53.8 / 55.1 |
    | row fold + content gates | 93.7 | 89.8 | 29.4 | 61.0 / 60.5 |

    - Progression is the fold's own symmetry (one step, the same everywhere), and the fold is best of all
      models on it.
    - It reads rows in order, so it loses to the roles on Arithmetic, as predicted. Swapping middle columns
      costs it only 0.5 points: the data teaches what the template leaves open.
    - **Trained on one row length, it does not transfer** (54–61% on rows of 5 and 20, only Constant
      survives). Recursion makes transfer possible, not automatic; mixed-length training is the open test.

24. **Final baselines (§23.4): the regularized setting on every dataset, 2 seeds, and up to 16× weights.**
    - The strongest object-level baseline on I-RAVEN is the regularized GNN at 16× the weights: 93.8 (63.7),
      3.5 points below the facet model and 14.6 below on the 3×3 grid.
    - On I-RAVEN-X 3×10 no size or setting generates exactly more than 4%; more weights hurt.
    - 3×3, values 0–9 is closest: regularized MLP 95.1 (74.3) against 99.2 (92.7).
    - The attribute-level GNNs are stronger: on I-RAVEN, B (symmetric) 96.8 (67.9) and A 95.0 (62.5), so
      granularity does most of the work there and the built-in symmetries add a little. On 3×10, B learned
      much worse than A (85.8 vs 95.0, one seed), and A's lead partly comes from reading Distribute by
      position: swapping rows drops it to 55.5.

25. **The value axis (§27): values as constants, a convolution along them.** Closer to the symbolic
    template than to the facet model: it is given primitives (exact sums through a `plus` relation, value
    counts per row, offsets of ±2 from the last panel) and learns which rule applies and how to combine
    them. 10.8k weights.

    | 3×10, values 0–49 | matched | exact | Arithmetic exact | on values 0–99 (never seen) |
    |---|---|---|---|---|
    | value axis, 100 puzzles | 99.0 | 95.5 | 97.2 | 98.1 (96.4) |
    | value axis, 600 puzzles | 99.9 | 98.3 | 99.8 | 100.0 (99.5) |
    | value axis, 6000 puzzles | **100.0** | **98.4** | **100.0** | **100.0** (99.6) |
    | value axis without `plus`, 6000 puzzles | 84.0 | 58.4 | 11.1 | 78.9 (58.7) |
    | facet, column roles, 6000 puzzles | 94.1 | 29.0 | 40.5 | cannot run |

    - No weight belongs to a value, so a model trained on values 0–49 runs on 0–99 at the same accuracy.
    - Rows of another length transfer only partly: rows of 5 give 60–90% matched (more with more training
      data); Distribute breaks most. Rows of 20 transfer better: 92–99% matched, with Constant, Distribute
      and Arithmetic at 99–100% exact and Progression dropping (51–89%).
    - Without `plus`, Constant, Progression and Distribute stay at 100% exact and Arithmetic falls to 11%:
      the sums come from the given relation, not from the convolution. The facet model learns approximate
      sums (40.5% exact, mostly within ±2), which is why it matches more often (94.1 vs 84.0).
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
    | attributes (I-RAVEN-X 3×10, column roles) | 600 | **66.3** (16.4) | 64.6 (14.1) | 52.5 (4.1) | 53.0 (3.6) |
    | | 6000 | **94.1** (29.0) | 93.9 (28.5) | 90.5 (17.8) | 88.1 (17.6) |

    | value symmetry, value-axis model (same given primitives) | shared across values (10.8k weights) | weights per value (335k) |
    |---|---|---|
    | 100 training puzzles | **99.0** (95.5) | 38.3 (3.6) |
    | 600 | **99.9** (98.3) | 92.7 (79.5) |
    | 6000 | **100.0** (**98.4**) | 98.4 (95.7) |

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
      0.2–0.5 points. Labels test whether *exact* invariance matters on top of sharing, and it does not, here.

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

    At 6000 puzzles weights per value reach 98.4% matched (95.7 exact; Arithmetic 94.9%), about what shared
    weights reach with 100 puzzles (99.0, 95.5 exact; Arithmetic 97.2). Without the symmetry the model needs
    some 60 times the data for the same accuracy. On rows of 5 it transfers worse: 55.7% (17.3 exact) against
    89.8% (76.6) with shared weights. On rows of 20 it transfers best of all (98.9%, 96.9 exact, against 98.6%,
    62.7 exact); with 100 and 600 puzzles shared weights stay ahead there too (97.4 vs 33.5, 92.4 vs 87.6).

## Still open
- Score the facet model once on the I-RAVEN test split, and end to end.
- Seeds for the facet runs: most numbers are one run.
- The attribute-level torch GNN pair (with and without symmetries), learning curves, transformed test
  sets, and the wrong-symmetry controls (plan E1–E4).
