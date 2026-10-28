# Draft: plan models A and B, attribute-level torch GNNs (2026-10-28)

Draft for the lab notes (§25), kept separate while another session reorganizes them. Plan:
`thesis/THESIS_PLAN_symmetry.md` §2. Code: `baselines/train_baseline.py --arch attr-abs | attr-rel`; job script
`baselines/submit_attr_gnn.sh`.

## Design
Same task, protocol and metrics as the other torch baselines (§23).
- **One graph per sample.** A sample is a (puzzle, queried component) pair, and its graph holds only that
  component's objects; the facet model never mixes components either.
- **Attribute channel.** Each object gets one node per attribute: type, size, colour and number. Number is the
  object count of the panel's component, put on every object. Each node's input is its thermometer code. The
  last four nodes are queries, one per attribute.
- **Position channel.** A separate stack with its own weights, like the facet model's occ / vP / relp / ansp.
  It has one node per context panel holding the component's occupancy, plus a query node.
- **Layer.** m_ij = ReLU(W_i h_i + W_j h_j + E e_ij), averaged over the present j ≠ i, then
  h_i = LayerNorm(h_i + ReLU(U [h_i; m̄_i])). Three layers.
  - Edge labels e_ij, the same for A and B. Attribute channel: same row, panel, object, attribute and slot.
    Position channel: same row.
  - Heads: one shared head decodes the four attribute queries; a second decodes the position query.
- **B (`attr-rel`)** labels nodes only relatively: complete or question row, column role (first / middle /
  last) and a query flag. It names no row index, column index, component, attribute or slot. Nor does it have a
  "same column" edge: the facet model with column roles is invariant to permuting middle columns within each
  row.
- **A (`attr-abs`)** = B plus one-hots for row, column, component, attribute and slot. The position nodes add
  row, column and component.
- **Size.** Width matched to the facet model's weight count:

  | | A | B |
  |---|---|---|
  | I-RAVEN | width 102, 266k weights | width 103, 267k |
  | I-RAVEN-X 3×10 | width 253, 1,596k | width 254, 1,599k |

## Symmetry check
`--check-symmetry`: random initial weights, float64, selection puzzles. The table gives the max |output
difference| from an exactly symmetric model. That model gives:
- the same output for objects, rows 1 ↔ 2 and columns 2 ↔ 3;
- the size and colour blocks exchanged for size ↔ colour;
- the other component's attribute output for the mirror.

| transformation | MLP | object GNN | A (`attr-abs`) | B (`attr-rel`) |
|---|---|---|---|---|
| I-RAVEN-X 3×10: objects in a panel | 0 | 0 | 0 | 0 |
| I-RAVEN-X: rows 1 ↔ 2 | 5.9e-2 | 4.7e-2 | 1.6e-2 | **1.3e-15** |
| I-RAVEN-X: middle columns 2 ↔ 3 | 3.3e-2 | 1.1e-2 | 2.7e-3 | **1.3e-15** |
| I-RAVEN-X: size ↔ colour | 0.18 | 2.3 | 0.92 | **1.5e-15** |
| I-RAVEN: objects in a panel | – | 8.9e-16 | 1.1e-15 | 1.1e-15 |
| I-RAVEN: rows 1 ↔ 2 | – | 0.11 | 4.6e-2 | **8.9e-16** |
| I-RAVEN: left-right / up-down mirror (attributes) | – | 0.88 | 1.4 | **0** |

B has exactly the symmetries of the facet model with `--columns roles`; A has none of them.

## Runs
Job 11914165, `submit_attr_gnn.sh`, 300 epochs, seed 0. Setting: the object GNN's best regularized one (§23.3),
i.e. weight decay 0.5, cosine learning rate after 5 warm-up epochs, gradient norm clipped at 1. Dropout 0 on
I-RAVEN and 0.3 on I-RAVEN-X 3×10. Batch 32 on I-RAVEN, 64 on I-RAVEN-X.

After training, each run scores its best checkpoint on the transformed unseen puzzles (`best.json`
"transforms"). `--eval-only` did the same for the reference MLP and object-GNN runs (job 11914166).

### Reference baselines on transformed puzzles (jobs 11914166, 11914187)
Unseen puzzles; the cells give matched %, with exact % in brackets.

| run | original | rows 1 ↔ 2 | columns 2 ↔ 3 | size ↔ colour | mirror |
|---|---|---|---|---|---|
| I-RAVEN MLP, weight decay 0.5 | 72.8 (29.9) | 72.4 (29.2) | – | – | 73.0 (29.8) |
| I-RAVEN GNN, weight decay 0.5 | 91.3 (56.5) | 91.3 (56.4) | – | – | 91.1 (56.6) |
| I-RAVEN MLP, dropout 0.1, weight decay 0.05 | 67.5 (21.6) | 68.1 (21.8) | – | – | 67.6 (21.7) |
| same + row-swap augmentation | 70.4 (23.5) | 70.9 (23.4) | – | – | 70.5 (23.8) |
| I-RAVEN GNN, dropout 0.1, weight decay 0.05 | 89.3 (55.1) | 89.5 (55.0) | – | – | 89.1 (54.8) |
| same + row-swap augmentation | 90.5 (55.7) | 90.5 (55.8) | – | – | 90.4 (55.3) |
| 3×10 values 0–49 MLP, weight decay 0.5, dropout 0.3 | 84.4 (2.6) | **47.7** (1.4) | **63.9** (2.1) | 83.5 (3.0) | – |
| 3×10 values 0–49 GNN, weight decay 0.5, dropout 0.3 | 84.0 (3.3) | **52.7** (2.1) | **71.6** (2.1) | 85.0 (4.4) | – |

* **Data symmetries are learned from the data.** On I-RAVEN, rows 1 ↔ 2 and the mirror are symmetries of the
  generator (§22): every swapped puzzle is another valid one. The baselines, which have neither built in, lose
  at most 0.6 points under them, with or without augmentation.
* **Task-only symmetries are not.** On 3×10, swapping rows 1 ↔ 2 moves Distribute puzzles out of the
  generator's support, and swapping two middle columns moves Distribute and Progression puzzles out (§22.2). Their answer by the weaker rule stays the same:
  Distribute's is still the value row 3 lacks. The baselines lose 13–37 points: they learned the generator's
  layout, a shortcut of the kind per-panel gates learned (§21.5).
* **Size ↔ colour** is a data symmetry of I-RAVEN-X, and the baselines are unaffected by it (83.5–85.0).
* The named-column facet models lose 15–20 points under the column swap (§24 reference: 85.4 → 70.2 plain,
  94.9 → 74.9 content gates). Models with the symmetry built in (B, facet with roles) cannot lose any.

A AND B RESULTS PENDING
