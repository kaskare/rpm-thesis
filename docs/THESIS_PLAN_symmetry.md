# Thesis plan: symmetries as inductive bias (2026-10-28)

**Claim.** Building the task's symmetries into the model, rather than leaving them for the data to teach:
1. reduces the training data needed;
2. raises accuracy at a fixed number of weights;
3. generalizes to transformed puzzles and to settings not seen in training.

Expressivity is not a claim: a built-in symmetry removes functions; it never adds any.

## 1. Definitions
A symmetry σ acts on a puzzle and on its answer. A model has σ **built in** when it maps σ(puzzle) to
σ(answer) by construction. In a PyNeuraLogic template this holds when σ only renames constants the template
does not name (theorem, notes §21.2). Which σ the data has: notes §22.

**Three kinds of σ, by what σ does to valid puzzles.** Only the third can cost a built-in model accuracy.
This corrects an earlier claim: tying columns 1 and 2 does *not* make Progression unlearnable.

| kind | σ maps a valid puzzle to | examples | building σ in |
|---|---|---|---|
| data symmetry | a valid puzzle, answer moved along by σ | rows 1 ↔ 2 (I-RAVEN), L-R / U-D components, size ↔ colour (I-RAVEN-X) | free; the data no longer has to teach it |
| out of support | an invalid puzzle (never seen) | columns 1 ↔ 2 for Progression; middle columns for Progression and Distribute; rows 1 ↔ 2 for 3×10 Distribute | costs no representational power and removes shortcuts (§21.5); the needed function may be harder to learn |
| conflicting | a valid puzzle, but the built-in model moves the answer the wrong way | invariance to value reflection, where the data needs equivariance | a hard ceiling, computable from the data |

## 2. Models: graph × symmetry
To credit a gain to symmetry rather than to the facet model's sparser graph, cross the two:

| | all-to-all graph (torch, GPU) | facet graph (PyNeuraLogic) |
|---|---|---|
| **symmetries named** | A. facet-level GNN, absolute labels | C. facet model with the symmetries named, one switch each |
| **symmetries built in** | B. facet-level GNN, relative labels | D. facet model (current) |

- **A and B (new).** These replace the object-level GNN with attribute-level nodes:
  - one node per (object, attribute);
  - one node per panel component for count and occupancy;
  - query nodes for the missing panel.

  All nodes are connected to all others.
  - A names everything: row, column, slot, component and attribute one-hots per node.
  - B has the facet model's symmetries and no more:
    - node flags for "question row", "query" and "component asked for";
    - edge labels for same row, same slot, same attribute and same component;
    - columns stay named, as in the facet model;
    - row index, component and attribute are never named. Weights are shared across attributes, and
      messages are tagged by whether both ends are the same attribute.
- **C.** One switch per symmetry:
  - rows: separate rules for r1 and r2 instead of `full(Rw)`;
  - components: separate rules for k0 and k1;
  - attributes: `--per-attr-weights`, which exists.
- A vs B and C vs D measure the symmetry; A vs C and B vs D measure the graph.
- **References (done):** the object-level MLP and GNN (notes §23).

## 3. Experiments
**E1. Accuracy at the facet model's weight count.**
- Models: A–D plus the references.
- Data: I-RAVEN and I-RAVEN-X 3×10, values 0–50; then values 0–100 and 0–1000.
- Already have (§23): D beats the references at equal weights and at 64×. The capacity sweep shows weights
  alone do not close the gap.
- Running: regularized references (§23.3).

**E2. Training data.**
- Learning curves for A–D and the references, at 300 / 1000 / 3000 / 6000 training puzzles (per layout on
  I-RAVEN).
- Prediction: a built-in data symmetry shifts the curve left, and the gain shrinks with more data. The early
  ablation (§18.3: shared vs separate attribute weights within noise at 2000 puzzles) may hold only at full
  data.

**E3. Generalization.**
- **Transformed test sets.** Apply each data symmetry to the unseen puzzles: rows 1 ↔ 2, swapped components,
  swapped size and colour values. A built-in model answers identically (check numerically, as in §21);
  measure the drop for the others. Existing checkpoints only, no training.
- **Learned vs built in.** Row-swap augmentation on the references (queued, §23.3) against B and D.
- **Transfer across layouts.** For example, train without U-D and test on U-D. First check that the
  position channel allows it: U-D uses slots 16–17, never seen in L-R.

**E4. Predictions from the symmetry analysis.**
- **Operand order for I-RAVEN-X Arithmetic** (§22.4).
  - Arithmetic is symmetric in columns 2..n−1 for plus and minus alike. The same swap is out of support for
    Progression and Distribute, so building it in costs them nothing.
  - Implementation in D:
    - replace the `col(P, c_j)` facts with roles first / middle / last;
    - the middle columns enter `rel` and `ans` through one rule with a panel variable;
    - the facet layers get one weight set per role pair.
  - Run plain and with content copy gates; per-panel gates name panels and would break the symmetry.
  - Prediction: Arithmetic exact rises (now 18.6 plain, 30.2 with copy links at 0–50); the other rules are
    unchanged.
- **Wrong symmetries.**
  - Columns 1 ↔ 2 tied (I-RAVEN), out of support for Progression. Prediction: no ceiling; at most slower
    learning or lower accuracy at a fixed size.
  - Value-reflection invariance: v and max − v get the same code (I-RAVEN-X), which conflicts with the data.
    Prediction: about 50% exact on Constant, Progression and Distribute values; Arithmetic unaffected.

## 4. Order of work
1. E4 operand order for Arithmetic (notes §24): plain run done, Arithmetic exact 18.6 → 40.5%; the rest
   running. E4b, the recursive row fold (§25), is running too, with row-length transfer (E3).
2. E3 transformed test sets on the existing checkpoints.
3. A and B, the facet-level torch GNNs.
4. C, the facet model with symmetries named.
5. E2 learning curves over A–D.
6. E4 wrong symmetries.
7. E3 transfer across layouts, if the feasibility check passes.

## 4b. Plan after the meeting of 5 November 2026
Transcript (local only, git-excluded): `docs/private/meeting_2026-11-05_transcript.md`. In order of work:

**M1. Correct the row-fold claim (now).** The meeting heard that the fold, trained on 3×10, works on 3×100.
Measured (§25): its weights load on any row length (304,712 at rows of 5, 10 and 20), but trained on rows of
10 it matches 53.8 / 55.1% on rows of 5 / 20 and keeps only Constant. Rows of 100 were never run. Tell the
supervisor, with M2 as the plan to make it true.

**M2. Row-length transfer for the fold.** *Parked 2026-11-06 at the user's request, after M2.1 showed no transfer:
no further row-length runs (M2.2 and M2.3 not run); the results so far stay as they are.* The point the supervisor valued most: one
lifted rule unfolds onto rows of any length. It is a hybrid of a relational GNN and a recurrent network,
which fixed-size networks cannot be. Make it measured:
1. **Mixed lengths.** Train the fold, and column roles for comparison, on rows of 5 and 10 together. Test on
   rows of 20. The data exists: `data/iravenx/center_single_shuffle_n_{5,10,20}_maxval_50.json`, 10,000
   puzzles each. Code: `bench_hub_grounding.py` sets the row length once for the whole run (`configure`), so
   the row length must become per puzzle (`puzzle_facts`, `facet_facts`, `load_iravenx`). Prediction: if the
   cause is the single training length, rows of 20 rise well above 55%. The linear toy fold
   (`notebooks/symmetry_briefing.ipynb` §1) sums rows of 20 exactly after training on rows of 3–5.
   *Done 2026-11-06 (jobs 11961921–22, notes §25): no transfer.* Rows of 20: fold 47.7, roles 54.0, against 55.1
   and 50.1 after rows of 10 alone; only Constant carries over.
2. **Warm start.** Start from the 3×10 checkpoint, train on rows of 20, and compare with the same run from
   scratch: does learning 3×10 first make 3×20 cheaper? Needs a flag that loads `template_best.bin` before
   training.
3. **Rows of 100**, only if step 1 transfers. In IBM's generator Progression needs values above 2·(n − 1),
   so rows of 100 need values 0–200 at least. The thermometer codes, and with them the weights, depend on
   the value range, so the 3×10 and 3×20 training sets must be generated at that range too. Arithmetic over
   99 operands below 200 is mostly zeros, so report it apart.

**M3. Reflection as a relation (left-right ↔ up-down).** Suggested in the meeting: rather than sharing
weights by construction, write one reflection relation with the axis as an argument (horizontal,
vertical), so one rule with one set of weights serves both layouts.
- **Today:** the attributes already have this symmetry, because the template names no layout. Trained
  without up-down, the facet model matches 99.1% on it (RESULTS, "Transfer to an unseen layout"). But the
  position channel names its 22 slots (left-right 14–15, up-down 16–17), so exact on the unseen layout is 0.
- **Plan:**
  - write the slots of the two layouts as (axis, first / second);
  - check on random weights that renaming horizontal ↔ vertical leaves the output unchanged (§21.2);
  - rerun the unseen-layout test.
- **Prediction:** exact on up-down rises from 0, and matched stays near 99.

**M4. Arithmetic: numbers as objects (an experiment, then writing).** The supervisor's picture:
- numbers live in the same relational domain as the objects;
- an attribute hops to a number object, which knows `next` and `plus` (1 + 2 = 3);
- information goes from circle to 1, then to 3, and back to the circle, the way a GNN passes messages.

**The value-axis model (§27) has the parts of it:**
- values as constants (the number objects), with the hop `x(P, A, V)`;
- neighbouring values (`off`, which includes next);
- `plus` facts as background knowledge, which Šourek et al. (2021) allow in a template.

**It also has more:** the Arithmetic rule itself, with no weights.
- the running sum along a row (`acc`);
- the check that a complete row's last value (or its first) is the sum of the others (`pchk`, `mchk`);
- the plus and minus answers (`sa`, `ma`).

So its Arithmetic 100 shows that the rule works when given. It does not show that a model learns to use `plus`.

**The new test:**
- **Model:** the value axis with the `plus` facts but without `acc`, `pchk`, `mchk`, `sa` and `ma`. Learned rules pass
  messages from a panel's value through `plus` to other values and back to panels.
- **Compare with:** the value axis without `plus` (Arithmetic 11.1) and with the rule given (100).
- **Prediction:** open.
- *Built 2026-11-05* (`value_axis.py --learned-plus`, notes §27): a learned recurrence whose state at a value
  moves to the next panel through `plus`, adding or taking away that panel's value. The grounding alone nearly
  gives the rule: on 3×10 the + / − paths reach 5.7 values at the end of an Arithmetic question row, always
  including the answer. `--state-next` also lets the state move along `next`, so it reaches most values and the
  weights must find the `plus` steps; that is the real test. Jobs 11962269–72 (3×10 and 3×3). 3×3 done:
  Arithmetic 98.8 (plus steps only) and 99.8 (also along next), written up in `docs/ARITHMETIC.md`. 3×10 done:
  Arithmetic 100 and 88.7, values 0–100 carry over, other row lengths do not. `--peano` derives `plus` from the
  zero and `next` (two rules, no weights); same outputs as the table (2026-11-06), and from scratch on 3×3
  Arithmetic 99.6 and 100 (jobs 11962979–80).

**Writing:** reasoning over entities taken from background knowledge. `plus` is a three-way relation (a
hyperedge), which a GNN on pairwise edges does not have.

**M5. Arithmetic: the math (writing).** Read `docs/READING_LIST_arithmetic_symmetry.md` (Baez, *Torsors Made
Easy*, first), then write the section in plain words:
- Attribute values form an affine line, a torsor of the shifts: differences mean something, the zero does not.
- Constant, Progression and Distribute survive shifting every value, so a convolution along the values
  (shift-equivariant, no origin) expresses them.
- a + b = c does not survive the shift: it needs the zero.

The notebook's checks back each step: the shift test p + q = 1, the convolution's bound, and the moves
(s, t, s + t). Write it only once it is clear.

**M6. Background and scope (writing).**
- Background chapter from geometric deep learning (Bronstein, Bruna, Cohen & Veličković 2021, and the slides
  from the meeting): symmetry groups and equivariance first, then the neurosymbolic version, weight sharing
  in a template as a symmetry (§21.2).
- The core holds without the LLM benchmark (`llm/`): keep it optional.
- Thesis material by theme: the symmetries; the encodings (thermometer codes, values as constants); the
  levels of abstraction (facet model, column roles, row fold, value axis).

**M7. For the next meeting.** A short presentation of the ideas and results, built from the briefing page.

**Order:**
1. M1 now.
2. M2.1, the code change and then the mixed-length run (done, no transfer). M2 parked after it.
3. M3.
4. M4's test (done).
5. M4–M6, writing alongside the runs.
6. M7 before the next meeting.

## 5. Already in the notes
| evidence | where |
|---|---|
| theorem: renaming unnamed constants preserves the model | §21.2 |
| breaking the row symmetry (per-panel gates) let a shortcut in: Distribute 98.3 → 39.5 after a row swap; content gates restore it (94.9) | §21.5, §21.6 |
| which symmetries I-RAVEN and I-RAVEN-X have | §22 |
| facet model vs MLP and all-to-all GNN, equal weights and 4×–64× | §23 |
