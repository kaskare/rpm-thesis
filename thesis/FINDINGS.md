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
    - A shuffled-source control with the same weights gains only 1.9 points (87.3), so the copy-link gain
      is the bias, not capacity.
18. **Which symmetries the data has** (§22). Rows 1 ↔ 2 keep every I-RAVEN rule, but break Distribute on
    I-RAVEN-X 3×10. Size ↔ colour is a data symmetry of I-RAVEN-X. Value reflection breaks Arithmetic. The
    I-RAVEN-X weak spot, Arithmetic, follows a symmetry the facet model lacks: the order of the operands.

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
    - Swapping two middle columns costs the named-column models 15–20 points of matching.
    - At epoch 25 the roles model matches 92.7% against the named model's 85.0%, with a fifth of the
      weights (288k vs 1,411k).
    - Runs are still going: copy-gate and weight-matched versions.

## Still open
- Score the facet model once on the I-RAVEN test split, and end to end.
- Seeds for the facet runs: most numbers are one run.
- The attribute-level torch GNN pair (with and without symmetries), learning curves, transformed test
  sets, and the wrong-symmetry controls (plan E1–E4).
