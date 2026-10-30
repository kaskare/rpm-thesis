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

### Results, A and B (job 11914165)
Run directories: `experiments/baselines/runs/{iraven,iravenx_n10_m50}_{attr-abs,attr-rel}_reg_do{0,0.3}_wd0.5_s0`.
Each run took 10.3 h (I-RAVEN) or 27 min (I-RAVEN-X) on one A100. The tables give unseen puzzles at the epoch
picked on the selection split, as matched % with exact % in brackets.

**I-RAVEN** (1000 unseen puzzles per layout):

| model | weights | epoch | unseen | rows 1 ↔ 2 | mirror |
|---|---|---|---|---|---|
| facet | 267k | | **97.3** (69.1) | built in | built in |
| **B `attr-rel`** | 267k | 293 | **96.8** (67.9) | identical | identical |
| **A `attr-abs`** | 266k | 290 | 95.0 (62.5) | 95.0 (62.4) | 95.0 (62.5) |
| object GNN, weight decay 0.5 | 264k | | 91.3 (56.5) | 91.3 (56.4) | 91.1 (56.6) |
| MLP, weight decay 0.5 | 264k | | 72.8 (29.9) | 72.4 (29.2) | 73.0 (29.8) |

| model | center | 2×2 | 3×3 | L-R | U-D | in-out | in-out grid |
|---|---|---|---|---|---|---|---|
| facet | 99.3 | 95.5 | 89.8 | 99.5 | 99.7 | 99.6 | 97.8 |
| B | 99.7 | 93.6 | 87.2 | 99.9 | 99.3 | 99.8 | 97.8 |
| A | 99.6 | 89.4 | 83.6 | 100 | 100 | 99.9 | 92.7 |
| object GNN | 99.3 | 80.6 | 72.9 | 99.1 | 99.3 | 99.6 | 88.4 |

Exact per part, A / B: position 69.1 / 79.0, type 71.2 / 77.3, size 70.7 / 76.3, colour 67.8 / 72.0. Both runs
were still improving: the best epochs were 290 and 293 of 300.

**I-RAVEN-X 3×10, values 0–49** (unseen 6300–7299). The last four columns give the exact value per rule:

| model | weights | matched | exact | Constant | Progression | Distribute | Arithmetic |
|---|---|---|---|---|---|---|---|
| facet, named columns | 1,411k | 85.4 | 15.0 | 93.4 | 77.4 | 9.8 | 18.6 |
| facet, named + content gates | 1,436k | 94.9 | 51.7 | 100 | 98.2 | 59.8 | 30.0 |
| facet, named + copy links (panel gates) | 1,598k | **97.8** | **60.0** | 100 | 93 | 87 | 30 |
| facet, column roles | 288k | 94.1 | 29.0 | 98.6 | 94.3 | 22.0 | **40.5** |
| facet, roles + content gates | 314k | 96.3 | 43.3 | 99.6 | 94.8 | 42.8 | 37.5 |
| facet, roles wide (running, epoch 40 of 60) | 1,390k | 93.9 | 26.2 | | | | |
| facet, roles wide + content gates (running, epoch 35) | 1,448k | 94.1 | 28.9 | | | | |
| **A `attr-abs`**, epoch 250 | 1,596k | 95.0 | 27.7 | 95.2 | 73.9 | 59.9 | 18.0 |
| **B `attr-rel`**, epoch 126 | 1,599k | 85.8 | 3.5 | 54.7 | 39.4 | 12.3 | 22.7 |
| object GNN, weight decay 0.5, dropout 0.3 | 1.6M | 84.0 | 3.3 | 61 | 42 | 25 | 10 |
| MLP, same setting | 1.6M | 84.4 | 2.6 | 53 | 36 | 21 | 11 |

The wide roles runs (jobs 11913675–76) are still training. Their rows give the last logged epoch, not yet
picked on selection.

Transformed unseen puzzles, I-RAVEN-X 3×10:

| model | original | rows 1 ↔ 2 | columns 2 ↔ 3 | size ↔ colour |
|---|---|---|---|---|
| facet, named columns | 85.4 (15.0) | built in | 70.2 (11.6) | built in |
| facet, named + content gates | 94.9 (51.7) | built in | 74.9 (23.9) | built in |
| facet, column roles (± content gates) | 94.1 (29.0) / 96.3 (43.3) | identical | identical | identical |
| **A** | 95.0 (27.7) | **55.5** (13.6) | **82.0** (17.7) | 95.0 (27.9) |
| **B** | 85.8 (3.5) | identical | identical | identical |
| object GNN | 84.0 (3.3) | 52.7 (2.1) | 71.6 (2.1) | 85.0 (4.4) |
| MLP | 84.4 (2.6) | 47.7 (1.4) | 63.9 (2.1) | 83.5 (3.0) |

A's exact value per rule after the row swap: Constant 95.0, Progression 76.4, Distribute **1.0**, Arithmetic 19.2.
After the column swap: 94.6, 75.8, **23.1**, 17.6.

* **On I-RAVEN, granularity does most of the work.** Attribute-level nodes alone (A against the object GNN,
  same regularization) gain 3.7 points of matching and 6.0 of exact. On the 2×2 and 3×3 grids they gain 9–11.
  B comes within 0.5 points of the facet model, 2–3 behind on the two grids and level on the in-out grid.
* **On I-RAVEN, building in the data symmetries adds a little** (B against A): +1.8 matched, +5.4 exact,
  +3.6 to +5.1 on the grids.
  - One seed each; the object GNN's two unregularized seeds differed by about 1 point.
  - B drops four labels at once (row, component, attribute, slot), so the gain cannot be split among them.
  - A learned the symmetries from the data: the row swap and the mirror move it by at most 0.1. The facet
    model's label ablation (§26) behaved the same way; whether the symmetries help is open until the
    unshared-weights runs are in.
* **On 3×10, B learned much worse than A:** 85.8 (3.5) against 95.0 (27.7). This is one seed at one setting,
  the object GNN's dropout 0.3.
  - B fits the training puzzles worse throughout. Its training loss was 0.023 at its selected epoch 126, and
    0.0096 against A's 0.0057 at epoch 300. So the gap is not overfitting.
  - Its curve is unstable: selection matching swings between 47 and 87% from epoch 40 to 200. At epoch 300 it
    scored 77.1 (14.0) on unseen puzzles, still far below A.
  - The symmetries alone do not explain this. The facet model with the same ones (column roles) improved on
    named columns, 85.4 (15.0) → 94.1 (29.0), at a fifth of the weights.
  - **Untested explanation:** the two graphs reach the needed function differently. In the facet template,
    the middle columns enter `rel` and `ans` through one dedicated rule averaged over exactly those panels. B
    must pick the same average out of a mean over all ~120 nodes, using edge labels. Plan §1 allowed for
    this: an out-of-support symmetry costs no representational power, but "the needed function may be harder
    to learn".
  - This is the graph × symmetry interaction that the 2 × 2 design of plan §2 was meant to expose. The row
    and column symmetries help the facet graph (+8.7 matched, +14.0 exact) and hurt the all-to-all graph
    (−9.2, −24.2). The facet comparison is still unequal in weights until the wide roles runs finish.
* **A's lead on 3×10 is partly a layout shortcut.**
  - A matches about as well as the column-roles facet model (95.0 vs 94.1), with 5.5 times the weights.
  - It generates Distribute far better (59.9 vs 22.0 exact), but by address. Swapping rows 1 ↔ 2 drops its
    Distribute exact to 1.0 and its matching to 55.5, the same failure as the per-panel gates (§21.5: 98.3 →
    39.5). Swapping columns 2 ↔ 3 drops matching to 82.0.
  - Constant and Progression are unaffected by either swap. B and the roles models are unchanged by
    construction.
  - Against the object GNN, A gains 11 points of matching and 24 of exact: granularity again.

**Open:**
- B on 3×10 at other settings, to see whether its gap is optimization: dropout 0 or 0.1, learning rate, two
  seeds. About 30 min per run.
- Second seeds of A and B on I-RAVEN to firm up the +1.8. About 10 h per run.
