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
- Data: I-RAVEN and I-RAVEN-X 3×10, values 0–49; then values 0–99 and 0–999.
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
  - Prediction: Arithmetic exact rises (now 18.6 plain, 30.2 with copy links at 0–49); the other rules are
    unchanged.
- **Wrong symmetries.**
  - Columns 1 ↔ 2 tied (I-RAVEN), out of support for Progression. Prediction: no ceiling; at most slower
    learning or lower accuracy at a fixed size.
  - Value-reflection invariance: v and max − v get the same code (I-RAVEN-X), which conflicts with the data.
    Prediction: about 50% exact on Constant, Progression and Distribute values; Arithmetic unaffected.

## 4. Order of work
1. E4 operand order for Arithmetic (requested 2026-10-28; running, notes §24).
2. E3 transformed test sets on the existing checkpoints.
3. A and B, the facet-level torch GNNs.
4. C, the facet model with symmetries named.
5. E2 learning curves over A–D.
6. E4 wrong symmetries.
7. E3 transfer across layouts, if the feasibility check passes.

## 5. Already in the notes
| evidence | where |
|---|---|
| theorem: renaming unnamed constants preserves the model | §21.2 |
| breaking the row symmetry (per-panel gates) let a shortcut in: Distribute 98.3 → 39.5 after a row swap; content gates restore it (94.9) | §21.5, §21.6 |
| which symmetries I-RAVEN and I-RAVEN-X have | §22 |
| facet model vs MLP and all-to-all GNN, equal weights and 4×–64× | §23 |
