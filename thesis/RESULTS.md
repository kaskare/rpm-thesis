# Results tables

Every result that still holds, in one place (last updated 2026-10-29). How each was obtained: the lab notes
(`thesis/PYNEURALOGIC_NOTES.md`, section numbers below); in what order: `thesis/FINDINGS.md`.

**Conventions.**
- Cells are **matched % (exact %)**:
  - *matched*: the candidate closest to the generated answer is the right one; ties count as wrong; chance
    12.5%;
  - *exact*: the generated panel equals the answer.
- **Splits:**
  - I-RAVEN: unseen validation puzzles 300–1299 per layout (7000), the epoch picked on val 0–299;
  - I-RAVEN test: marked *test*;
  - I-RAVEN-X: unseen puzzles 6300–7299, the epoch picked on 6000–6299.
- **Seeds:** one seed, unless "± sd (2 seeds)".
- **Inputs:** ground-truth attributes; "end to end" means through the ResNet-18 frontend.
- **Pending results:** marked *running*.

## 1. Main table

| model | weights | I-RAVEN | I-RAVEN-X 3×3, 0–9 | 3×10, 0–49 | 3×10, 0–99 | 3×10, 0–999 |
|---|---|---|---|---|---|---|
| **Facet model (PyNeuraLogic)** | | | | | | |
| facet, named columns, 60–80 epochs | 266k / 1,411k | **97.3** (69.1) | **99.2** (92.7) | 85.4 (15.0) | 84.9 (6.4) | 81.2 (0.0) |
| facet, 150 epochs | | 97.4 (69.3) | – | 89.1 (23.8) | 86.3 (6.5) | 85.4 (0.1) |
| + copy links, per-panel gates | 1,598k | – | – | 97.8 (60.0) | **96.2** (30.8) | **94.3** (15.0) |
| + copy links, content gates | 1,436k | – | – | 94.9 (51.7) | – | – |
| + copy links, shuffled control | 1,598k | – | – | 87.3 (21.5) | – | – |
| column roles | 288k | (= named at 3 columns) | – | 94.1 (29.0) | – | – |
| column roles + content gates | 314k | – | – | 96.3 (43.3) | – | – |
| column roles, widths 64/144/288 (equal weights to named) | 1,390k | – | – | 94.9 (28.5) | – | – |
| column roles + content gates, widths 64/144/288 | 1,448k | – | – | 93.1 (30.0) | – | – |
| row fold | 305k | *running* (job 11921354) | – | 92.2 (28.0) | – | – |
| row fold + content gates | 330k | – | – | 93.7 (35.0) | – | – |
| **Value axis (PyNeuraLogic, §27)** | | | | | | |
| value axis, 6000 puzzles | 10.8k | – | – | **100.0** (**98.4**) | 100.0 (99.6), trained on 0–49 | – |
| value axis, 600 puzzles | 10.8k | – | – | 99.9 (98.3) | 100.0 (99.5), trained on 0–49 | – |
| value axis, 100 puzzles | 10.8k | – | – | 99.0 (95.5) | 98.1 (96.4), trained on 0–49 | – |
| value axis, 500 puzzles (smoke, 300 unseen) | 10.8k | – | – | 99.3 (98.3) | 98.5 (97.5), trained on 0–49 | – |
| value axis, 6000 puzzles, without plus | 10.7k | – | – | 84.0 (58.4) | 78.9 (58.7), trained on 0–49 | – |
| value axis, weights per value (symmetry removed), 600 puzzles | 335k | – | – | 92.7 (79.5) | cannot run | – |
| value axis, weights per value, 100 puzzles | 335k | – | – | 38.3 (3.6) | cannot run | – |
| value axis, weights per value, 6000 puzzles | 335k | – | – | 98.4 (95.7) | cannot run | – |
| **Symbolic (rules given)** | | | | | | |
| symbolic template, rules (PyNeuraLogic) | one per (attribute, rule) | **98.33** *test*; end to end 96.31 | – | – | – | – |
| symbolic template, primitives | | 97.55 *test*; end to end 95.71 | – | – | – | – |
| reference solver, rules (Python, nothing learned) | 0 | 97.78 *test*; end to end 95.74 | – | – | – | – |
| reference solver, primitives | 0 | 96.42 *test*; end to end 94.73 | – | – | – | – |
| **Torch baselines** (same task and metrics, §23) | | | | | | |
| MLP, facet's weights | 264k / 267k / 1.6M | 60.6 ± 0.8 (17.1) | 92.8 ± 0.2 (64.8) | 83.5 ± 0.4 (3.9) | 80.0 ± 2.3 (0.5) | 74.3 ± 3.2 (0.0) |
| all-to-all object GNN, facet's weights | same | 87.2 ± 1.1 (51.5) | 92.0 ± 0.4 (58.5) | 71.6 ± 0.1 (1.2) | 64.7 ± 5.4 (0.0) | 54.1 ± 3.1 (0.0) |
| MLP, regularized | same | 72.0 ± 1.1 (27.5) | 95.1 ± 0.4 (74.3) | 84.5 ± 0.1 (2.7) | 76.3 ± 1.4 (0.1) | 73.8 ± 0.2 (0.0) |
| object GNN, regularized | same | 90.5 ± 1.2 (55.1) | 93.9 ± 0.2 (67.7) | 85.3 ± 1.9 (4.2) | 76.8 ± 2.1 (0.2) | 55.1 ± 1.0 (0.0) |
| strongest object-level baseline, picked on selection | up to 16× | 93.8 (63.7) GNN 16× reg. | 92.8 (64.8) MLP | 86.4 (2.1) MLP 16× | 80.0 (0.5) MLP | 74.3 (0.0) MLP |
| attribute GNN A, absolute labels | 266k / 1.6M | 95.0 (62.5) | – | 95.0 (27.7) | – | – |
| attribute GNN B, relative labels (symmetric) | 267k / 1.6M | 96.8 (67.9) | – | 85.8 (3.5) | – | – |
| **Literature (reported, other protocols)** | | | | | | |
| ARLC, learned (trains on 2×2 only) | | 92.6 | 98.4 | 95.0 | 95.0 | 90.6 |
| ARLC, programmed | | – | 97.2 | 100 | 100 | 99.7 |
| GPT-4 | | – | 93.2 | 82.2 | 79.6 | 76.6 |
| DeepSeek R1 | | – | – | – | 84.0 | 82.8 |
| MLP (as reported with ARLC) | | 87.1 | – | – | – | – |

Notes on the main table:
- **Weights:** for 3×10 the facet weight counts are at values 0–49.
- **Copy links:** the per-panel gates break the row symmetry; see §3.
- **The facet model's I-RAVEN-X exact scores are slightly off** (generator draws Distribute values up to maxval
  inclusive, §23). Matching is unaffected.
- **Torch baselines:**
  - "regularized" is the setting picked on selection puzzles of I-RAVEN and 3×10 0–49 (§23.3), run unchanged
    on every dataset with 2 seeds: LayerNorm (MLP), weight decay 0.5, dropout 0 on I-RAVEN and 0.3 on
    I-RAVEN-X, cosine schedule, clipping.
  - "strongest" is the run with the highest selection accuracy among all object-level runs on that dataset
    (MLP and GNN, 1× to 64×, with and without regularization, §23.4). Selection is 300 puzzles, so it does not
    always pick the best unseen score (3×3: the regularized MLP reaches 95.1).
  - A and B are one seed each (notes draft `thesis/notes_attr_gnn_draft.md`).
- **Literature:** from the respective papers; they see candidates or train differently, so they are not
  directly comparable.

## 2. Row-length transfer (trained on rows of 10, values 0–49)

| model | rows of 5 | rows of 10 (trained) | rows of 20 |
|---|---|---|---|
| column roles | 65.6 (4.0) | 94.1 (29.0) | 50.1 (4.6) |
| column roles + content gates | 66.3 (5.2) | 96.3 (43.3) | 49.2 (4.6) |
| column roles, wide (1,390k) | 63.9 (2.8) | 94.9 (28.5) | 49.3 (4.1) |
| column roles + content gates, wide (1,448k) | 67.7 (6.0) | 93.1 (30.0) | 50.1 (5.1) |
| row fold | 53.8 (0.4) | 92.2 (28.0) | 55.1 (5.9) |
| row fold + content gates | 61.0 (8.1) | 93.7 (35.0) | 60.5 (10.1) |
| value axis, 6000 puzzles | 89.8 (76.6) | 100.0 (98.4) | 98.6 (62.7) |
| value axis, 6000 puzzles, without plus | 57.0 (24.4) | 84.0 (58.4) | 80.1 (58.6) |
| value axis, 600 puzzles | 75.8 (52.5) | 99.9 (98.3) | 92.4 (84.4) |
| value axis, weights per value, 6000 puzzles | 55.7 (17.3) | 98.4 (95.7) | 98.9 (96.9) |
| value axis, weights per value, 600 puzzles | 63.6 (23.4) | 92.7 (79.5) | 87.6 (64.4) |
| value axis, weights per value, 100 puzzles | 27.3 (0.6) | 38.3 (3.6) | 33.5 (3.4) |
| value axis, 100 puzzles | 59.7 (32.7) | 99.0 (95.5) | 97.4 (82.3) |
| value axis, 500 puzzles (smoke) | 58.6 (34.2) | 99.3 (98.3), 300 unseen | 86.6 (78.5) |

Named-column models and the torch baselines cannot be run on another row length: their weights are tied to
the columns. Test sets: IBM's `iravenx_task.py --n 5 / 20 --maxval 50 --nconf 0`, seed 1234.

Rows of 20, exact value per rule (Constant / Progression / Distribute / Arithmetic): shared weights 6000
100 / 51.2 / 100 / 100, 600 100 / 89.2 / 100 / 100, 100 100 / 81.7 / 100 / 99.1; weights per value 6000
100 / 99.8 / 100 / 96.6, 600 98.0 / 67.0 / 96.5 / 87.3, 100 61.8 / 27.3 / 16.7 / 24.4.

## 3. Transformed puzzles (does a model give the same answers when the puzzle is transformed?)

Each transformation is either a symmetry of the data or one under which the answer stays the same
(notes §22). "–" = not applicable or not run.

| model | as is | rows 1 ↔ 2 | middle columns 2 ↔ 3 | size ↔ colour | left-right / up-down mirrored |
|---|---|---|---|---|---|
| **I-RAVEN** | | | | | |
| facet | 97.3 | identical (built in) | – | – | identical (built in) |
| attribute GNN B (symmetric) | 96.8 | identical | – | – | identical |
| attribute GNN A | 95.0 | 95.0 | – | – | 95.0 |
| object GNN, regularized (seed 0) | 91.3 | 91.3 | – | – | 91.1 |
| MLP, regularized (seed 0) | 72.8 | 72.4 | – | – | 73.0 |
| **I-RAVEN-X 3×10, values 0–49** | | | | | |
| facet, named columns | 85.4 | 85.4 (identical) | 70.2 | identical (built in) | – |
| + copy links, per-panel gates | 97.8 | ≈60.0 (puzzles with Distribute 98.3 → 39.5) | – | – | – |
| + copy links, content gates | 94.9 | 94.9 (identical) | 74.9 | – | – |
| column roles (± content gates) | 94.1 / 96.3 | identical | identical | identical | – |
| column roles, wide (± content gates) | 94.9 / 93.1 | identical | identical | – | – |
| row fold (± content gates) | 92.2 / 93.7 | identical | 91.7 / 93.1 | – | – |
| attribute GNN B (symmetric) | 85.8 | identical | identical | identical | – |
| attribute GNN A | 95.0 | 55.5 (Distribute exact 59.9 → 1.0) | 82.0 | 95.0 | – |
| object GNN, regularized (seed 0) | 84.0 | 52.7 | 71.6 | 85.0 | – |
| MLP, regularized (seed 0) | 84.4 | 47.7 | 63.9 | 83.5 | – |

The rows-swap row for per-panel gates combines §21.5's two groups (646 puzzles with a Distribute attribute,
354 without).

## 4. I-RAVEN by layout (matched %)

| model | center | 2×2 | 3×3 | L-R | U-D | in-out | in-out grid |
|---|---|---|---|---|---|---|---|
| facet | 99.3 | 95.5 | 89.8 | 99.5 | 99.7 | 99.6 | 97.8 |
| row fold (job 11921354) | *running* | *running* | *running* | *running* | *running* | *running* | *running* |
| symbolic rules (*test*) | 99.90 | 95.98 | 95.46 | 99.95 | 99.95 | 99.85 | 97.26 |
| attribute GNN B (symmetric) | 99.7 | 93.6 | 87.2 | 99.9 | 99.3 | 99.8 | 97.8 |
| attribute GNN A | 99.6 | 89.4 | 83.6 | 100 | 100 | 99.9 | 92.7 |
| object GNN, regularized, 16× | 99.5 | 89.6 | 75.2 | 99.8 | 99.7 | 99.9 | 92.8 |
| object GNN, regularized (2 seeds) | 99.2 | 80.1 | 72.5 | 99.2 | 98.9 | 99.5 | 84.0 |
| object GNN, facet's weights (seed 0) | 98.7 | 73.3 | 64.6 | 99.0 | 98.7 | 99.7 | 82.0 |
| MLP, regularized, 16× | 97.5 | 57.8 | 45.2 | 96.1 | 97.9 | 99.3 | 74.7 |
| MLP, regularized (2 seeds) | 93.2 | 46.8 | 37.6 | 85.0 | 84.9 | 95.0 | 61.8 |
| MLP, facet's weights (seed 0) | 87.8 | 32.7 | 20.3 | 74.0 | 63.0 | 90.5 | 52.1 |

**Facet model on I-RAVEN by rule** (job 11920318, `facet_rule_eval.py`, 1000 unseen puzzles per layout;
reproduces 97.3 (69.1)). Rules from each puzzle's XML, per component. Exact: the component's value histogram
for that attribute (its object count for Number, its positions for Position and Number/Position).

| part | Constant | Progression | Distribute | Arithmetic |
|---|---|---|---|---|
| type, size and colour | 83.7 | 89.7 | 88.7 | 83.8 |
| type | 81.4 | 90.8 | 89.5 | – |
| size | 81.8 | 90.3 | 90.4 | 83.9 |
| colour | 87.0 | 87.1 | 85.0 | 83.7 |
| Number/Position | 99.5 | – | – | – |
| Number | – | 44.4 | 49.0 | 47.6 |
| Position | – | 50.9 | 70.7 | 63.5 |

Matched: 97.3% of the 4031 puzzles with an Arithmetic rule and 97.3% of the 2969 without.

## 5. I-RAVEN-X 3×10, values 0–49: exact value per rule (%)

| model | Constant | Progression | Distribute | Arithmetic |
|---|---|---|---|---|
| facet, named columns | 93.4 | 77.4 | 9.8 | 18.6 |
| + copy links, per-panel gates | **100** | 92.9 | **86.8** | 30.2 |
| + copy links, content gates | **100** | 98.2 | 59.8 | 30.0 |
| column roles | 98.6 | 94.3 | 22.0 | **40.5** |
| column roles + content gates | 99.6 | 94.8 | 42.8 | 37.5 |
| column roles, wide (1,390k) | 97.1 | 94.8 | 20.9 | 36.6 |
| column roles + content gates, wide (1,448k) | 99.4 | 80.2 | 27.5 | 25.1 |
| row fold | 99.9 | **99.2** | 15.1 | 36.2 |
| row fold + content gates | 99.6 | 89.8 | 32.4 | 29.4 |
| value axis, 6000 puzzles | 100 | 100 | 100 | 100 |
| value axis, 600 puzzles | 100 | 100 | 100 | 99.8 |
| value axis, 100 puzzles | 99.9 | 98.5 | 100 | 97.2 |
| value axis without plus, 6000 puzzles | 100 | 100 | 100 | 11.1 |
| value axis, weights per value, 6000 puzzles | 100 | 100 | 100 | 94.9 |
| value axis, weights per value, 600 puzzles | 98.0 | 87.8 | 97.7 | 86.8 |
| value axis, weights per value, 100 puzzles | 60.5 | 30.8 | 21.4 | 22.3 |
| attribute GNN A | 95.2 | 73.9 | 59.9 | 18.0 |
| attribute GNN B (symmetric) | 54.7 | 39.4 | 12.3 | 22.7 |
| object GNN, regularized (2 seeds) | 58.4 | 43.2 | 24.9 | 10.7 |
| MLP, regularized (2 seeds) | 50.3 | 34.2 | 19.8 | 10.5 |

Count of attribute values per rule (of 3000): Constant 837, Progression 792, Distribute 865, Arithmetic 506.

## 6. Torch baselines in detail (§23)

**More weights** (seed 0; 1× is the mean of 2 seeds):

| weights | I-RAVEN MLP | I-RAVEN GNN | 3×10 0–49 MLP | 3×10 0–49 GNN |
|---|---|---|---|---|
| 1× (264k / 1.6M) | 60.6 (17.1) | 87.2 (51.5) | 83.5 (3.9) | 71.6 (1.2) |
| 4× | 68.9 (25.2) | 86.3 (50.9) | 87.2 (2.5) | 72.3 (1.2) |
| 16× | 71.0 (27.3) | 88.5 (58.8) | 86.4 (2.1) | 64.9 (1.3) |
| 64× (17M / 102M) | 69.9 (26.0) | 86.5 (54.9) | 84.6 (2.4), lr 1e-4 | 63.3 (0.3) |
| 1×, regularized (2 seeds) | 72.0 (27.5) | 90.5 (55.1) | 84.5 (2.7) | 85.3 (4.2) |
| 4×, regularized | 79.4 (40.4) | 92.9 (62.6) | 84.1 (2.6) | 83.4 (2.6) |
| 16×, regularized | 81.2 (46.1) | **93.8** (63.7) | 82.3 (1.8) | 72.8 (1.2) |

**Regularization at 1×** (setting picked on selection; 40 runs, §23.3):

| | without | best | setting |
|---|---|---|---|
| I-RAVEN MLP | 60.6 (17.1) | 72.8 (29.9) | weight decay 0.5 |
| I-RAVEN GNN | 87.2 (51.5) | 91.3 (56.5) | weight decay 0.5 |
| 3×10 MLP | 83.5 (3.9) | 84.4 (2.6) | weight decay 0.5, dropout 0.3 |
| 3×10 GNN | 71.6 (1.2) | 84.0 (3.3) | weight decay 0.5, dropout 0.3 |
| I-RAVEN, + row-swap augmentation | MLP 67.5 → 70.4; GNN 89.3 → 90.5 | | at dropout 0.1, weight decay 0.05 |

**The regularized setting on every dataset** (2 seeds, not re-tuned; §23.4):

| | I-RAVEN | 3×3, 0–9 | 3×10, 0–49 | 3×10, 0–99 | 3×10, 0–999 |
|---|---|---|---|---|---|
| MLP | 60.6 → 72.0 | 92.8 → 95.1 | 83.5 → 84.5 | 80.0 → 76.3 | 74.3 → 73.8 |
| object GNN | 87.2 → 90.5 | 92.0 → 93.9 | 71.6 → 85.3 | 64.7 → 76.8 | 54.1 → 55.1 |

Exact per rule on 3×3, values 0–9, regularized MLP / GNN: Constant 99.7 / 98.1, Progression 94.7 / 88.8,
Distribute 93.4 / 87.7, Arithmetic 64.4 / 69.4.

## 7. Symmetry ablations (§26, §27): one symmetry removed

**Components and attributes in the facet model** (unseen puzzles; matched (exact)):

| data | training puzzles | shared, symmetric (default) | labels (shared weights) | separate weights, same widths | separate weights, matched count |
|---|---|---|---|---|---|
| I-RAVEN L-R + U-D (components) | 100 per layout | **86.6** (40.4) | 87.3 (41.3) | 75.6 (22.2) | 74.6 (19.6) |
| | 1000 | **99.3** (95.6) | 99.6 (95.7) | 97.9 (84.5) | 98.5 (84.5) |
| | 6000 | 99.9 (97.7) | 99.8 (98.7) | 100.0 (97.8) | 100.0 (98.8) |
| I-RAVEN-X 3×10, 0–49, column roles (attributes) | 600 | **66.3** (16.4) | 64.6 (14.1) | 52.5 (4.1) | 53.0 (3.6) |
| | 6000 | **94.1** (29.0) | 93.9 (28.5) | 90.5 (17.8), still rising at epoch 60 | 88.1 (17.6) |

- Weights: components 266k shared, 533k separate, 266k matched (widths 22, 45, 90); attributes 288k shared, 932k
  separate, 288k matched (18, 35, 69). Labels add 320 / 512.
- Transformed puzzles (mirror, size ↔ colour): the shared models are identical by construction; labelled and separate
  models move by at most 0.8 points (they learn the symmetry from the symmetric data).
- One seed each.

**The value symmetry in the value-axis model** (3×10, values 0–49, plus relation given; matched (exact)):

| training puzzles | shared across values (10.8k) | weights per value (335k) |
|---|---|---|
| 100 | **99.0** (95.5) | 38.3 (3.6) |
| 600 | **99.9** (98.3) | 92.7 (79.5) |
| 6000 | **100.0** (**98.4**) | 98.4 (95.7) |

## 8. Perception (ResNet-18 frontend, trained on RAVEN; I-RAVEN test)

| | panels exactly right | puzzles with all 16 right |
|---|---|---|
| center, L-R, U-D, in-out | 100.00 | 100.00 |
| 2×2 / 3×3 / in-out grid | 99.99 / 96.02 / 96.34 | 99.90 / 76.45 / 71.55 |
| overall | 98.91 | 92.56 |
