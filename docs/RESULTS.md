# Results tables

Every result that still holds, in one place (last updated 2026-11-14). How each was obtained: the lab notes
(`docs/PYNEURALOGIC_NOTES.md`, section numbers below); in what order: `docs/FINDINGS.md`.

**Conventions.**
- Cells are **matched % (exact %)**:
  - *matched*: the candidate closest to the generated answer is the right one; ties count as wrong; chance
    12.5%;
  - *exact*: the generated panel equals the answer. For the value axis this is the argmax of its value
    distribution, its own answer (since 2026-10-31). Decoded through the facet model's thermometer code, which
    reads off the distribution's median instead, it came out up to 16 points lower where the distribution is
    spread (3×3: 95.3 vs 99.7; 3×10: 98.4 vs 100.0). The per-rule exacts were argmax all along. † marks the
    one value-axis cell still code-decoded (the smoke run's rows of 10, read from its training log).
- **Splits:**
  - I-RAVEN: unseen validation puzzles 300–1299 per layout (7000), the epoch picked on val 0–299;
  - I-RAVEN test: marked *test*;
  - I-RAVEN-X: unseen puzzles 6300–7299, the epoch picked on 6000–6299.
- **Seeds:** one seed, unless "± sd (2 seeds)".
- **Inputs:** ground-truth attributes; "end to end" means through the ResNet-18 frontend, except the
  black-box ViT rows, which read the eight context panels themselves (their target is still the facet code,
  and the candidates are still compared through their XML labels).
- **Pending results:** marked *running*.

## 1. Main table

| model | weights | I-RAVEN | I-RAVEN-X 3×3, 0–10 | 3×10, 0–50 | 3×10, 0–100 | 3×10, 0–1000 |
|---|---|---|---|---|---|---|
| **Facet model (PyNeuraLogic)** | | | | | | |
| facet, named columns, 60–80 epochs | 267k / 267k / 1,411k / 1,425k / 1,685k | 97.3 (69.1) | **99.2** (97.0) | 85.4 (15.0) | 84.9 (6.4) | 81.2 (0.0) |
| facet, named columns, **end to end** through the ViT slot recognizer (same checkpoint; hard mapping, candidates perceived; job 12028100) | 267k + 326k recognizer | 97.1 (68.8) | – | – | – | – |
| the same checkpoint on the **test split**, scored once (2026-11-14, job 12031767): labels / end to end | 267k / + 326k | **97.19** (68.35) *test* / **97.03** (68.09) *test* | – | – | – | – |
| facet, 150 epochs | 267k / 1,411k / 1,425k / 1,685k | 97.4 (69.3) | – | 89.1 (23.8) | 86.3 (6.5) | 85.4 (0.1) |
| + copy links, per-panel gates (I-RAVEN: job 11932362, epoch 60) | 277k / 277k / 1,598k / 1,799k / 5,426k | **98.2** (**72.1**) | 99.8 (99.3) | 97.8 (60.6) | **96.2** (30.8) | **94.3** (15.0) |
| + copy links, content gates (I-RAVEN: job 11932572, epoch 40) | 272k / 272k / 1,436k | 97.4 (68.7) | 99.8 (98.5) | 94.9 (52.3) | – | – |
| + copy links, shuffled control | 1,598k | – | – | 87.3 (21.5, old exact scorer) | – | – |
| column roles | 288k | (= named at 3 columns) | – | 94.1 (29.2) | – | – |
| column roles + content gates | 314k | – | – | 96.3 (43.7) | – | – |
| column roles, widths 64/144/288 (equal weights to named) | 1,390k | – | – | 94.9 (28.6) | – | – |
| column roles + content gates, widths 64/144/288 | 1,448k | – | – | 93.1 (30.0) | – | – |
| row fold | 283k / 283k / 305k | 95.8 (63.0) | 99.6 (98.9) | 92.2 (28.0) | – | – |
| row fold + content gates | 330k | – | – | 93.7 (35.0) | – | – |
| **Value axis (PyNeuraLogic, §27): addition and rule features given** | | | | | | |
| value axis, 6000 puzzles per layout, slots and count added (job 11932367, epoch 25) | 18.0k | **99.3** (**75.1**) | – | – | – | – |
| value axis, 1000 puzzles per layout, slots and count added (job 11932368, epoch 55) | 18.0k | 99.2 (74.7) | – | – | – | – |
| value axis, 6000 puzzles | 10.8k | – | **99.8** (**99.7**) | **100.0** (**100.0**) | 100.0 (100.0), trained on 0–50 | – |
| value axis, 600 puzzles | 10.8k | – | – | 99.9 (99.9) | 100.0 (99.9), trained on 0–50 | – |
| value axis, 100 puzzles | 10.8k | – | – | 99.0 (97.3) | 98.1 (96.8), trained on 0–50 | – |
| value axis, 500 puzzles (smoke, 300 unseen) | 10.8k | – | – | 99.3 (98.3†) | 98.5 (97.9), trained on 0–50 | – |
| value axis, 6000 puzzles, without plus (I-RAVEN: job 11932570, epoch 15) | 17.8k on I-RAVEN, 10.7k | 94.2 (59.0) | 94.4 (86.3) | 84.0 (59.6) | 78.9 (60.5), trained on 0–50 | – |
| value axis, 1000 puzzles per layout, without plus or the union and difference sizes (job 11932569, epoch 35) | 17.8k | 94.2 (59.4) | – | – | – | – |
| value axis, weights per value (symmetry removed), 600 puzzles | 335k | – | – | 92.7 (81.0) | cannot run | – |
| value axis, weights per value, 100 puzzles | 335k | – | – | 38.3 (4.0) | cannot run | – |
| value axis, weights per value, 6000 puzzles | 335k | – | – | 98.4 (97.4) | cannot run | – |
| value axis, plus used by learned rules (no built-in sums or checks; §27), 6000 puzzles (jobs 11962271, 11962269) | 15.8k | – | 99.6 (99.4) | 100.0 (100.0) | 99.2 (99.0), trained on 0–50 | – |
| value axis, plus used by learned rules, the state also along next (jobs 11962272, 11962270) | 21.0k | – | 100.0 (99.9) | 99.3 (94.4) | 98.4 (92.9), trained on 0–50 | – |
| **Symbolic (rules given)** | | | | | | |
| symbolic template, rules (PyNeuraLogic) | 41 | **98.33** *test*; end to end 96.31 | – | – | – | – |
| symbolic template, primitives | 94 | 97.55 *test*; end to end 95.71 | – | – | – | – |
| reference solver, rules (Python, nothing learned) | 0 | 97.78 *test*; end to end 95.74 | – | – | – | – |
| reference solver, primitives | 0 | 96.42 *test*; end to end 94.73 | – | – | – | – |
| **Torch baselines** (same task and metrics, §23) | | | | | | |
| MLP, facet's weights | 264k / 267k / 1,597k / 1,802k / 5,383k | 60.6 ± 0.8 (17.1) | 92.8 ± 0.2 (64.8) | 83.5 ± 0.4 (3.9) | 80.0 ± 2.3 (0.5) | 74.3 ± 3.2 (0.0) |
| all-to-all object GNN, facet's weights | 265k / 265k / 1,593k / 1,798k / 5,430k | 87.2 ± 1.1 (51.5) | 92.0 ± 0.4 (58.5) | 71.6 ± 0.1 (1.2) | 64.7 ± 5.4 (0.0) | 54.1 ± 3.1 (0.0) |
| MLP, regularized | 264k / 266k / 1,599k / 1,803k / 5,384k | 72.0 ± 1.1 (27.5) | 95.1 ± 0.4 (74.3) | 84.5 ± 0.1 (2.7) | 76.3 ± 1.4 (0.1) | 73.8 ± 0.2 (0.0) |
| object GNN, regularized | 265k / 265k / 1,593k / 1,798k / 5,430k | 90.5 ± 1.2 (55.1) | 93.9 ± 0.2 (67.7) | 85.3 ± 1.9 (4.2) | 76.8 ± 2.1 (0.2) | 55.1 ± 1.0 (0.0) |
| strongest object-level baseline, picked on selection | 4,258k / 267k / 25,556k / 1,802k / 5,383k | 93.8 (63.7) GNN 16× reg. | 92.8 (64.8) MLP | 86.4 (2.1) MLP 16× | 80.0 (0.5) MLP | 74.3 (0.0) MLP |
| attribute GNN A, absolute labels | 266k / 1,596k | 95.0 (62.5) | – | 95.0 (27.7) | – | – |
| attribute GNN B, relative labels (symmetric) | 267k / 1,599k | 96.8 (67.9) | – | 85.8 (3.5) | – | – |
| **Black-box vision, end to end** (reads the images; same task and metrics) | | | | | | |
| ViT over the matrix as one image, width 64 (job 11931310) | 335k / 335k / 428k | 79.1 (37.6) | 35.4 (2.6) | 52.9 (0.0) | – | – |
| ViT, width 224 (job 11931310) | 3,753k / 3,753k / 4,079k | 93.4 (64.4) | 47.5 (5.6) | 43.1 (0.4) | – | – |
| ViT, rows 1 ↔ 2 built in, width 64 | 323k | 89.4 (55.3) | – | – | – | – |
| ViT, rows 1 ↔ 2 built in, width 224 | 3,709k / 3,709k / 3,919k | 92.1 (62.1) | 44.0 (5.6) | 40.1 (0.0) | – | – |
| ViT, width 64, a separate filter per patch | 4,019k | 49.8 (9.2) | – | – | – | – |
| CNN (small ResNet, panels stacked as channels) | 286k | 39.4 (2.1) | – | – | – | – |
| CNN, 16× | 4,270k | 50.5 (6.3) | – | – | – | – |
| **Literature (reported, other protocols)** | | | | | | |
| ARLC, learned (trains on 2×2 only) | | 92.6 | 98.4 | 95.0 | 95.0 | 90.6 |
| ARLC, programmed | | – | 97.2 | 100 | 100 | 99.7 |
| GPT-4 | | – | 93.2 | 82.2 | 79.6 | 76.6 |
| DeepSeek R1 | | – | – | – | 84.0 | 82.8 |
| MLP (as reported with ARLC) | | 87.1 | – | – | – | – |

Notes on the main table:
- **Weights** are learnable parameters, given per dataset in column order for the datasets a row has results
  on; §9 has the exact counts, read from the checkpoints. They grow with the value range on 3×10, because the
  thermometer codes do.
  - The torch baselines were sized to the plain facet model on I-RAVEN and 3×3, and to the facet model with
    per-panel copy links on 3×10. At 0–1000 that is 3.2× the named-column facet model's weights.
  - ViT widths move in steps of 32, so the closest to the facet model's weights is 1.26× (width 64), and the
    closest to 16× is 14× (width 224).
  - The I-RAVEN-X images are drawn by `data_prep/render_iravenx.py`: 80 px panels and 16 px patches on 3×3, 160 px
    panels and 32 px patches on 3×10, where one value step is one pixel.
- **Copy links:** the per-panel gates break the row symmetry; see §3.
- **Exact for the facet model on I-RAVEN-X was rescored on 2026-11-08** (jobs 11977983–86). The generator draws values
  up to maxval inclusive, and `bench_hub_grounding.evaluate` decoded the answer through a description that cannot hold
  maxval, so perfect outputs scored 95.6% exact on 3×3 (0–10) and 98.4% on 3×10 (0–50); fixed, notes §23. The
  3×3 cells rose 3.6–4.4 points; on 3×10 0–50, 13 of 18 runs are unchanged, the rest rose 0.1–0.6. Matching,
  epoch choice and the per-rule exact values (thermometer bits summed) were never affected; the rescored runs
  reproduce their matched exactly.
  - Not rescored, still the old scorer, with the most they can rise: 0–100 (+0.4), 0–1000 (+0.1), rows of 5 / 20
    (+0.7 / +1.3), the shuffled control (+1.6; `facet_rule_eval.py` cannot rebuild its shuffled copy sources), and
    the transformed-puzzle tables (§3).
- **Torch baselines:**
  - "regularized" is the setting picked on selection puzzles of I-RAVEN and 3×10 0–50 (§23.3), run unchanged
    on every dataset with 2 seeds: LayerNorm (MLP), weight decay 0.5, dropout 0 on I-RAVEN and 0.3 on
    I-RAVEN-X, cosine schedule, clipping.
  - "strongest" is the run with the highest selection accuracy among all object-level runs on that dataset
    (MLP and GNN, 1× to 64×, with and without regularization, §23.4). Selection is 300 puzzles, so it does not
    always pick the best unseen score (3×3: the regularized MLP reaches 95.1).
  - A and B are one seed each (notes draft `docs/notes_attr_gnn_draft.md`).
- **Literature:** from the respective papers; they see candidates or train differently, so they are not
  directly comparable.

**I-RAVEN test split, one pass per checkpoint** (2026-11-14; test puzzles 0–1999 of every layout, 14,000 puzzles; facet
job 12031767, baselines job 12032179 with `core/baselines/test_eval.py`). Every baseline checkpoint first reproduced its
recorded validation numbers exactly, then was scored on test once; the result is in the run's `test.json`, and a run that
has one is skipped. These passes are spent. The sweeps the baselines' settings were picked from got no test pass.
Matched (exact); layouts are matched on test.

| model | weights | validation | **test** | center | 2×2 | 3×3 | L-R | U-D | in-out | in-out grid |
|---|---|---|---|---|---|---|---|---|---|---|
| facet, XML labels | 267k | 97.31 (69.09) | **97.19** (68.35) | 99.6 | 95.0 | 89.1 | 99.8 | 99.7 | 99.8 | 97.5 |
| facet, end to end (ViT slot recognizer) | 267k + 326k | 97.13 (68.79) | **97.03** (68.09) | 99.6 | 95.0 | 89.1 | 99.8 | 99.7 | 99.8 | 96.4 |
| attribute GNN B (symmetric) | 267k | 96.76 (67.94) | **96.79** (67.84) | 99.8 | 95.2 | 85.5 | 100.0 | 99.9 | 99.7 | 97.5 |
| attribute GNN A | 266k | 95.03 (62.53) | **95.09** (62.51) | 99.9 | 89.5 | 84.4 | 100.0 | 99.9 | 100.0 | 92.0 |
| object GNN, regularized, 16× | 4,258k | 93.79 (63.66) | **93.44** (63.66) | 99.8 | 88.2 | 74.5 | 99.8 | 99.8 | 99.8 | 92.2 |
| object GNN, regularized, seed 0 | 265k | 91.31 (56.51) | 91.71 (56.43) | 99.2 | 82.1 | 73.0 | 99.3 | 98.9 | 99.7 | 89.8 |
| object GNN, regularized, seed 1 | 265k | 89.66 (53.71) | 89.98 (53.84) | 99.4 | 80.4 | 71.8 | 99.0 | 98.7 | 99.5 | 81.0 |
| object GNN, facet's weights, seed 0 | 265k | 88.00 (54.94) | 88.52 (55.04) | 99.0 | 73.8 | 66.8 | 99.3 | 99.2 | 99.7 | 81.8 |
| object GNN, facet's weights, seed 1 | 265k | 86.47 (48.07) | 86.98 (49.05) | 98.4 | 71.8 | 63.2 | 97.6 | 97.3 | 99.0 | 81.5 |
| MLP, regularized, seed 0 | 264k | 72.81 (29.94) | 73.55 (29.59) | 94.4 | 46.4 | 38.8 | 91.2 | 87.1 | 95.3 | 61.6 |
| MLP, regularized, seed 1 | 264k | 71.29 (25.07) | 71.89 (24.81) | 91.0 | 46.9 | 37.4 | 84.7 | 84.3 | 95.0 | 64.0 |
| MLP, facet's weights, seed 0 | 264k | 60.06 (16.37) | 60.72 (16.63) | 89.3 | 33.3 | 25.6 | 72.9 | 63.1 | 89.5 | 51.3 |
| MLP, facet's weights, seed 1 | 264k | 61.17 (17.91) | 60.90 (17.50) | 85.5 | 29.0 | 24.6 | 72.2 | 73.5 | 91.9 | 49.6 |
| ViT, width 224, end to end | 3,753k | 93.40 (64.39) | **93.10** (64.36) | 99.1 | 92.8 | 80.5 | 99.8 | 99.6 | 99.8 | 80.0 |
| ViT, width 64, end to end | 335k | 79.09 (37.61) | 78.69 (37.36) | 89.3 | 57.2 | 55.4 | 95.8 | 95.6 | 89.9 | 67.7 |
| ViT, rows 1 ↔ 2 built in, width 224 | 3,709k | 92.07 (62.14) | 92.26 (62.40) | 99.5 | 88.8 | 77.8 | 99.9 | 99.7 | 99.7 | 80.5 |
| ViT, rows 1 ↔ 2 built in, width 64 | 323k | 89.43 (55.30) | 89.79 (54.52) | 98.7 | 79.7 | 71.3 | 99.2 | 99.0 | 99.0 | 81.6 |

- Means of the two seeds on test: object GNN regularized 90.8 ± 1.2 (55.1), with the facet's weights 87.75 ± 1.1 (52.0);
  MLP regularized 72.7 ± 1.2 (27.2), with the facet's weights 60.8 ± 0.1 (17.1).
- **Test and validation agree.** No checkpoint moves by more than 0.8 points matched, and the order of the models is the same
  on both splits. The facet model keeps its lead over GNN B (97.19 against 96.79, 0.4 points, one seed each) and over the
  ViT end to end (97.03 against 93.10, with 592k weights against 3,753k).
- By layout, the facet model's lead over GNN B is the 3×3 grid (89.1 against 85.5); on the other six layouts the two are
  within 0.2 points. The ViT's deficit is the 3×3 grid and the in-out grid (80.5 and 80.0).
- The copy-link variants, the row fold, the value axis and the CNNs were not scored on test.

## 2. Row-length transfer (trained on rows of 10, values 0–50)

| model | rows of 5 | rows of 10 (trained) | rows of 20 |
|---|---|---|---|
| column roles | 65.6 (4.0) | 94.1 (29.2) | 50.1 (4.6) |
| column roles + content gates | 66.3 (5.2) | 96.3 (43.7) | 49.2 (4.6) |
| column roles, wide (1,390k) | 63.9 (2.8) | 94.9 (28.6) | 49.3 (4.1) |
| column roles + content gates, wide (1,448k) | 67.7 (6.0) | 93.1 (30.0) | 50.1 (5.1) |
| row fold | 53.8 (0.4) | 92.2 (28.0) | 55.1 (5.9) |
| row fold + content gates | 61.0 (8.1) | 93.7 (35.0) | 60.5 (10.1) |
| row fold, trained on rows of 5 and 10 (job 11961921, epoch 40) | 87.9 (7.4), trained | 90.4 (23.3) | 47.7 (4.1) |
| column roles, trained on rows of 5 and 10 (job 11961922, epoch 55) | 79.3 (14.2), trained | 93.6 (27.3) | 54.0 (7.9) |
| value axis, 6000 puzzles | 89.8 (78.9) | 100.0 (100.0) | 98.6 (63.7) |
| value axis, 6000 puzzles, without plus | 57.0 (23.9) | 84.0 (59.6) | 80.1 (60.2) |
| value axis, plus used by learned rules, 6000 puzzles | 63.8 (43.3) | 100.0 (100.0) | 71.2 (26.4) |
| value axis, plus used by learned rules, also along next, 6000 puzzles | 61.9 (31.6) | 99.3 (94.4) | 74.7 (59.7) |
| value axis, 600 puzzles | 75.8 (68.8) | 99.9 (99.9) | 92.4 (90.6) |
| value axis, weights per value, 6000 puzzles | 55.7 (17.0) | 98.4 (97.4) | 98.9 (98.2) |
| value axis, weights per value, 600 puzzles | 63.6 (23.7) | 92.7 (81.0) | 87.6 (65.4) |
| value axis, weights per value, 100 puzzles | 27.3 (0.8) | 38.3 (4.0) | 33.5 (3.6) |
| value axis, 100 puzzles | 59.7 (32.6) | 99.0 (97.3) | 97.4 (84.8) |
| value axis, 500 puzzles (smoke) | 58.6 (34.2) | 99.3 (98.3†), 300 unseen | 86.6 (82.3) |

Named-column models and the torch baselines cannot be run on another row length: their weights are tied to
the columns. Test sets: IBM's `iravenx_task.py --n 5 / 20 --maxval 50 --nconf 0`, seed 1234.

Rows of 20, exact value per rule (Constant / Progression / Distribute / Arithmetic): shared weights 6000
100 / 51.2 / 100 / 100, 600 100 / 89.2 / 100 / 100, 100 100 / 81.7 / 100 / 99.1; weights per value 6000
100 / 99.8 / 100 / 96.6, 600 98.0 / 67.0 / 96.5 / 87.3, 100 61.8 / 27.3 / 16.7 / 24.4; `plus` used by learned rules
38.1 / 68.9 / 100 / 44.8, also along next 99.3 / 99.9 / 100 / 1.3; row fold trained on rows of 5 and 10
99.7 / 2.9 / 11.3 / 4.7, column roles trained on rows of 5 and 10 98.3 / 38.6 / 5.8 / 3.9.

**Training on rows of 5 and 10 together** (plan M2.1, 6000 puzzles of each, epoch picked on 300 selection puzzles of
each) does not lift rows of 20: the fold gets 47.7, below its 55.1 when trained on rows of 10 alone, and the roles 54.0
(50.1). The prediction was that rows of 20 rise well above 55 if the single training length were the cause, so it is
not. Rows of 5 are a training length here, so 87.9 and 79.3 are not transfer.

**Value axis trained on 3×3** (job 11926965; 6000 puzzles, epoch 10 picked on selection; the data's values are
0–10). Unseen 3×3: 99.8 (99.7); exact per rule 100 / 100 / 100 / 99.4 (facet 99.2 (97.0)). The same checkpoint
on 3×10, values 0–50 (rows 3× longer, values 5× wider than it saw): 65.2 (46.2); exact per rule 93.9 / 36.9 /
**100** / **76.7**.

**Value axis with `plus` used by learned rules, trained on 3×3** (2026-11-06; `docs/ARITHMETIC.md`, "Numbers as
objects"). Exact per rule on unseen 3×3: `plus` steps only (job 11962271, epoch 10) 100 / 100 / 100 / 98.8; the
state also along `next` (job 11962272, epoch 25) 100 / 100 / 100 / 99.8; without `plus` (job 11932568) 100 / 100 /
99.9 / 70.7. On 3×10, values 0–50: 57.7 (32.3), per rule 100 / 45.1 / 99.8 / 5.9; and 24.5 (1.0), 51.1 / 12.0 /
20.9 / 2.8. Trained on 3×10, values 0–50 (jobs 11962269, epoch 5, and 11962270, epoch 15): Arithmetic exact 100
and 88.7, with the other rules at 100 (§5); values 0–100 carry over (Arithmetic 100 and 84.9), other row lengths do
not (rows of 5 / 20: 12.7 / 44.8 and 1.8 / 1.3).

**`plus` derived from the zero and next** (`--peano`, 2026-11-06): the trained checkpoints above, loaded into the
zero-and-next template, give the same outputs (max difference 2e-16 on 3×3, 1e-17 on 3×10). Trained from scratch on
3×3: `plus` steps only (job 11962979, epoch 15) 99.9 (99.8), per rule 100 / 100 / 100 / 99.6; also along `next` (job
11962980, epoch 5) 99.8 (99.5), 100 / 100 / 99.4 / 100. Scored on 3×10, values 0–50: 54.5 (27.4), Arithmetic 23.5; and
17.5 (0.2), 3.2. Level with the table runs, as it must be: the network is the same, and the runs are not seeded, so
the gaps are between two runs, not an effect of the zero and next.

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
| ViT, width 224 | 93.4 | 93.4 | – | – | – |
| ViT, width 64 | 79.1 | 78.9 | – | – | – |
| ViT, rows 1 ↔ 2 built in, widths 224 / 64 | 92.1 / 89.4 | identical | – | – | – |
| **I-RAVEN-X 3×3, values 0–10** | | | | | |
| ViT, width 224 | 47.5 | 46.4 | – | – | – |
| ViT, width 64 | 35.4 | 35.1 | – | – | – |
| ViT, rows 1 ↔ 2 built in, width 224 | 44.0 | identical | – | – | – |
| **I-RAVEN-X 3×10, values 0–50** | | | | | |
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
| ViT, width 64 | 52.9 | 38.5 | 47.3 | – | – |
| ViT, width 224 | 43.1 | 37.9 | 44.4 | – | – |
| ViT, rows 1 ↔ 2 built in, width 224 | 40.1 | identical | 39.9 | – | – |

The rows-swap row for per-panel gates combines §21.5's two groups (646 puzzles with a Distribute attribute,
354 without).

## 4. I-RAVEN by layout (matched %)

| model | center | 2×2 | 3×3 | L-R | U-D | in-out | in-out grid |
|---|---|---|---|---|---|---|---|
| facet | 99.3 | 95.5 | 89.8 | 99.5 | 99.7 | 99.6 | 97.8 |
| row fold (job 11921354, epoch 40) | 98.6 | 93.9 | 86.2 | 98.6 | 98.2 | 99.0 | 96.0 |
| + copy links, per-panel gates (job 11932362, epoch 60) | 99.6 | 96.8 | 93.3 | 99.9 | 100.0 | 99.5 | 98.0 |
| + copy links, content gates (job 11932572, epoch 40) | 99.6 | 95.6 | 89.2 | 99.8 | 99.8 | 100.0 | 97.9 |
| symbolic rules (*test*) | 99.90 | 95.98 | 95.46 | 99.95 | 99.95 | 99.85 | 97.26 |
| attribute GNN B (symmetric) | 99.7 | 93.6 | 87.2 | 99.9 | 99.3 | 99.8 | 97.8 |
| attribute GNN A | 99.6 | 89.4 | 83.6 | 100 | 100 | 99.9 | 92.7 |
| object GNN, regularized, 16× | 99.5 | 89.6 | 75.2 | 99.8 | 99.7 | 99.9 | 92.8 |
| object GNN, regularized (2 seeds) | 99.2 | 80.1 | 72.5 | 99.2 | 98.9 | 99.5 | 84.0 |
| object GNN, facet's weights (seed 0) | 98.7 | 73.3 | 64.6 | 99.0 | 98.7 | 99.7 | 82.0 |
| MLP, regularized, 16× | 97.5 | 57.8 | 45.2 | 96.1 | 97.9 | 99.3 | 74.7 |
| MLP, regularized (2 seeds) | 93.2 | 46.8 | 37.6 | 85.0 | 84.9 | 95.0 | 61.8 |
| MLP, facet's weights (seed 0) | 87.8 | 32.7 | 20.3 | 74.0 | 63.0 | 90.5 | 52.1 |
| ViT, width 224, reads the images (job 11931310, epoch 271) | 99.4 | 93.0 | 81.8 | 99.6 | 99.3 | 99.6 | 81.1 |
| ViT, width 64, reads the images (job 11931310, epoch 264) | 89.8 | 56.2 | 57.9 | 96.7 | 95.1 | 90.0 | 67.9 |
| ViT, rows 1 ↔ 2 built in, width 224 (job 11932145, epoch 291) | 99.0 | 89.2 | 77.4 | 99.4 | 99.8 | 99.7 | 80.0 |
| ViT, rows 1 ↔ 2 built in, width 64 (job 11932145, epoch 279) | 97.6 | 78.2 | 71.0 | 99.6 | 99.3 | 98.7 | 81.6 |
| ViT, width 64, a separate filter per patch (epoch 213) | 65.8 | 31.5 | 31.3 | 57.8 | 63.8 | 58.8 | 39.7 |
| CNN, 16× (job 11932164, epoch 171) | 56.9 | 37.1 | 35.5 | 60.5 | 62.4 | 59.2 | 41.7 |
| CNN (job 11932164, epoch 33) | 48.4 | 30.9 | 27.5 | 44.6 | 45.9 | 46.2 | 32.4 |
| value axis, 1000 puzzles, rules given (job 11932368, epoch 55) | 99.3 | 98.3 | 98.4 | 99.9 | 99.4 | 99.7 | 99.5 |
| value axis, 6000 puzzles, rules given (job 11932367, epoch 25) | 99.5 | 98.6 | 99.0 | 99.6 | 99.5 | 99.6 | 99.2 |
| value axis, 6000 puzzles, without plus (job 11932570, epoch 15) | 91.8 | 94.6 | 92.6 | 93.9 | 92.1 | 96.5 | 97.8 |

**Copy links on I-RAVEN by rule** (jobs 11932362 and 11932572, type, size and colour exact). Per-panel gates: 84.5 / 90.0 /
89.4 / 85.5, content gates: 84.3 / 89.3 / 88.9 / 81.7, in the order Constant / Progression / Distribute / Arithmetic.

**Facet model / row fold on I-RAVEN by rule** (jobs 11920318 and 11921354, `facet_rule_eval.py`, 1000 unseen
puzzles per layout; facet reproduces 97.3 (69.1), fold 95.8 (63.0)). Cells: facet / fold. Rules from each puzzle's XML, per component. Exact: the component's value histogram
for that attribute (its object count for Number, its positions for Position and Number/Position).

| part | Constant | Progression | Distribute | Arithmetic |
|---|---|---|---|---|
| type, size and colour | 83.7 / 82.1 | 89.7 / 88.8 | 88.7 / 88.4 | 83.8 / **75.9** |
| type | 81.4 / 80.1 | 90.8 / 90.3 | 89.5 / 89.3 | – |
| size | 81.8 / 79.9 | 90.3 / 90.4 | 90.4 / 90.8 | 83.9 / 77.1 |
| colour | 87.0 / 85.4 | 87.1 / 84.5 | 85.0 / 83.6 | 83.7 / 74.6 |
| Number/Position | 99.5 / 99.7 | – | – | – |
| Number | – | 44.4 / 46.8 | 49.0 / 44.6 | 47.6 / 45.1 |
| Position | – | 50.9 / 48.7 | 70.7 / 71.3 | 63.5 / 63.5 |

Matched: facet 97.3% of the 4031 puzzles with an Arithmetic rule and 97.3% of the 2969 without; fold 95.3 / 96.5.

**ViT on I-RAVEN by rule** (jobs 11931809–10, `core/baselines/rule_eval.py`, the same definitions on the same 7000
puzzles; reproduces 93.4 (64.4) and 79.1 (37.6)). Cells: width 224 / width 64.

| part | Constant | Progression | Distribute | Arithmetic |
|---|---|---|---|---|
| type, size and colour | 82.4 / 76.0 | 89.7 / 73.2 | 86.6 / 68.2 | 86.3 / 65.1 |
| type | 79.2 / 69.3 | 90.4 / 72.6 | 86.9 / 61.8 | – |
| size | 78.6 / 72.6 | 88.3 / 72.3 | 84.6 / 70.1 | 87.0 / 60.9 |
| colour | 87.7 / 84.3 | 90.1 / 75.3 | 89.0 / 76.2 | 85.6 / 69.5 |
| Number/Position | 99.3 / 96.7 | – | – | – |
| Number | – | 57.6 / 34.3 | 65.1 / 38.2 | 63.4 / 40.1 |
| Position | – | 51.3 / 13.9 | 61.4 / 23.2 | 65.6 / 23.7 |

Matched, width 224 / 64: 93.8 / 78.1 with an Arithmetic rule, 92.9 / 80.4 without.

**ViT, width 224, by layout and rule** (type, size and colour exact). On grids, Constant is low because panels
that are not uniform hold noise values there (§22.2), which no model can generate.

| layout | matched (exact) | Constant | Progression | Distribute | Arithmetic |
|---|---|---|---|---|---|
| center | 99.4 (97.3) | 99.9 | 99.6 | 98.6 | 97.6 |
| L-R | 99.6 (97.3) | 99.9 | 99.9 | 99.8 | 97.8 |
| U-D | 99.3 (97.1) | 100 | 100 | 99.9 | 97.2 |
| in-out | 99.6 (97.5) | 100 | 99.9 | 99.6 | 96.3 |
| 2×2 | 93.0 (41.4) | 37.9 | 86.0 | 79.6 | 79.6 |
| in-out grid | 81.1 (12.5) | 74.9 | 78.5 | 71.6 | 74.0 |
| 3×3 | 81.8 (7.6) | 11.0 | 41.4 | 34.5 | 37.4 |

**Regularized MLP and object GNN by rule on I-RAVEN** (job 11932364, `core/baselines/rule_eval.py`, mean of 2 seeds;
type, size and colour exact): MLP 74.5 / 69.0 / 63.2 / 38.2, GNN 78.8 / 78.5 / 78.2 / 68.7 (Constant / Progression /
Distribute / Arithmetic). Distribute on the single-object layouts: MLP 79.3–95.1, GNN 99.8–99.9.

**Facet model on 3×3, values 0–10, by rule** (job 11932363, epoch 10, 1000 unseen puzzles; reproduces 99.2 (92.7) under the old exact scorer, 99.2 (97.0) after its fix):
Constant 99.6, Progression 99.5, Distribute 99.3, Arithmetic 95.0.

**Copy links and row fold on 3×3, values 0–10, by rule** (jobs 11932361 and 11932360, both epoch 10). Copy links:
Constant 99.6, Progression 100, Distribute 100, Arithmetic 98.3. Row fold: 99.6, 100, 99.6, 98.3.

**Value axis on I-RAVEN** (`core/pynl/value_axis_iraven.py`, job 11932368, 1000 puzzles per layout, 18,018 weights, epoch 55
of 60 picked on selection; besides `plus` it is given the slot descriptors and the union and difference sizes of the
question row's sets). Matched (exact) by layout: center 99.3 (99.1), 2×2 98.3 (46.0), 3×3 98.4 (33.6), L-R 99.9 (98.7),
U-D 99.4 (98.6), in-out 99.7 (99.2), in-out grid 99.5 (47.9). The low grid exacts are Constant's noise: Constant 36.4
/ 31.5 / 81.0 on the grids, the other rules 97.3 or more.

| part | Constant | Progression | Distribute | Arithmetic |
|---|---|---|---|---|
| type, size and colour | 85.3 | 99.8 | 99.8 | 98.3 |
| type | 83.1 | 99.8 | 99.8 | – |
| size | 83.3 | 99.9 | 99.9 | 99.0 |
| colour | 88.5 | 99.8 | 99.6 | 97.7 |
| Number/Position | 100 | – | – | – |
| Number | – | 99.7 | 99.6 | 98.8 |
| Position | – | 90.8 | 96.1 | 97.4 |

Distribute by layout: 100 / 99.3 / 99.4 / 99.9 / 100 / 99.9 / 99.5 (order of §4).

**Value axis without `plus`, on I-RAVEN and 3×3** (jobs 11932569 and 11932568). I-RAVEN, 1000 puzzles per layout, no
union or difference sizes either, 17,826 weights, epoch 35: 94.2 (59.4); type, size and colour together 84.8 / 97.6 /
97.9 / 66.4 (size 74.0, colour 58.7 on Arithmetic); Number 99.0 / 98.5 / 74.1; Position 89.7 / 93.9 / 76.9. 3×3, 10,689
weights, epoch 35: 94.4 (86.3), 100 / 100 / 99.9 / 70.7. So Arithmetic without `plus` is not 11.1 on small ranges. A
reference that sees only differences, a lookup table fit on the training puzzles, gets 47.5 on 3×3, 43.4 on the I-RAVEN
center layout and 2.4 on 3×10; 44.6% of 3×3 Arithmetic question rows have a zero operand. The model's lead over that
reference is not explained.

**Content gates on 3×3, values 0–10, by rule** (job 11932571, epoch 10, 1000 unseen puzzles, 271,668 weights): 99.6 / 100
/ 100 / 97.5; 99.8 (98.5) overall.

**Learning curves** (I-RAVEN, puzzles per layout, seed 0; `jobs/baselines/submit_curves.sh`, `jobs/pynl/submit_facet_curves.sh`).
Every run takes the same number of updates: epochs scale with 6000/n. At 6000, MLP and object GNN are regularized, mean of
2 seeds. The ViT is not trained on shifted images.

| puzzles | facet | GNN B | GNN A | object GNN | MLP | ViT 64 | ViT 224 | ViT 64, rows built in | ViT 224, rows built in |
|---|---|---|---|---|---|---|---|---|---|
| 300 | 90.0 (53.3) | 87.0 (53.0) | 86.1 (50.8) | 29.6 (1.1) | 24.9 (0.5) | 15.1 (0.1) | 15.6 (0.2) | 16.0 (0.1) | 14.6 (0.1) |
| 1000 | 94.1 (62.7) | 93.9 (64.0) | 93.4 (61.5) | 68.7 (21.9) | 48.1 (7.7) | 19.2 (0.2) | 18.6 (0.2) | 24.3 (0.4) | 23.8 (0.5) |
| 3000 | 96.1 (66.0) | 96.8 (68.8) | 95.9 (65.3) | 86.5 (49.8) | 64.5 (19.1) | 42.7 (3.5) | 45.8 (4.5) | 70.7 (24.2) | 62.0 (13.3) |
| 6000 | 97.3 (69.1) | 96.8 (67.9) | 95.0 (62.5) | 90.5 (55.1) | 72.0 (27.5) | 79.1 (37.6) | 93.4 (64.4) | 89.4 (55.3) | 92.1 (62.1) |

Facet by rule at 1000: 81.8 / 85.6 / 84.8 / 75.8. GNN A and B at 3000 are level with or above their 6000 runs, so the
curve has flattened and the last step is within a run's noise.

## 5. I-RAVEN-X 3×10, values 0–50: exact value per rule (%)

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
| value axis, plus used by learned rules, 6000 puzzles | 100 | 100 | 100 | 100 |
| value axis, plus used by learned rules, the state also along next, 6000 puzzles | 100 | 100 | 100 | 88.7 |
| value axis, weights per value, 6000 puzzles | 100 | 100 | 100 | 94.9 |
| value axis, weights per value, 600 puzzles | 98.0 | 87.8 | 97.7 | 86.8 |
| value axis, weights per value, 100 puzzles | 60.5 | 30.8 | 21.4 | 22.3 |
| attribute GNN A | 95.2 | 73.9 | 59.9 | 18.0 |
| attribute GNN B (symmetric), regularized | 54.7 | 39.4 | 12.3 | 22.7 |
| attribute GNN B, no dropout or decay, 425 epochs (seeds 0 / 1, last epoch; §12) | 100 / 99.8 | 100 / 99.7 | 82.1 / 79.0 | **50.6 / 49.2** |
| object GNN, regularized (2 seeds) | 58.4 | 43.2 | 24.9 | 10.7 |
| MLP, regularized (2 seeds) | 50.3 | 34.2 | 19.8 | 10.5 |

Count of attribute values per rule (of 3000): Constant 837, Progression 792, Distribute 865, Arithmetic 506.
The value-axis rows get their sums from the given `plus` relation: compare them with one another, not with the
other rows, which have to learn the sum.

## 6. Torch baselines in detail (§23)

**More weights** (seed 0; 1× is the mean of 2 seeds):

| weights | I-RAVEN MLP | I-RAVEN GNN | 3×10 0–50 MLP | 3×10 0–50 GNN |
|---|---|---|---|---|
| 1× (264–265k / 1,593–1,597k) | 60.6 (17.1) | 87.2 (51.5) | 83.5 (3.9) | 71.6 (1.2) |
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

| | I-RAVEN | 3×3, 0–10 | 3×10, 0–50 | 3×10, 0–100 | 3×10, 0–1000 |
|---|---|---|---|---|---|
| MLP | 60.6 → 72.0 | 92.8 → 95.1 | 83.5 → 84.5 | 80.0 → 76.3 | 74.3 → 73.8 |
| object GNN | 87.2 → 90.5 | 92.0 → 93.9 | 71.6 → 85.3 | 64.7 → 76.8 | 54.1 → 55.1 |

Exact per rule on 3×3, values 0–10, regularized MLP / GNN: Constant 99.7 / 98.1, Progression 94.7 / 88.8,
Distribute 93.4 / 87.7, Arithmetic 64.4 / 69.4.

## 7. Symmetry ablations (§26, §27): one symmetry removed

**Components and attributes in the facet model** (unseen puzzles; matched (exact)):

| data | training puzzles | shared, symmetric (default) | labels (shared weights) | separate weights, same widths | separate weights, matched count |
|---|---|---|---|---|---|
| I-RAVEN L-R + U-D (components) | 100 per layout | **86.6** (40.4) | 87.3 (41.3) | 75.6 (22.2) | 74.6 (19.6) |
| | 1000 | **99.3** (95.6) | 99.6 (95.7) | 97.9 (84.5) | 98.5 (84.5) |
| | 6000 | 99.9 (97.7) | 99.8 (98.7) | 100.0 (97.8) | 100.0 (98.8) |
| I-RAVEN-X 3×10, 0–50, column roles (attributes) | 600 | **66.3** (16.4) | 64.6 (14.2) | 52.5 (4.1) | 53.0 (3.6) |
| | 6000 | **94.1** (29.2) | 93.9 (28.6) | 90.5 (17.8), still rising at epoch 60 | 88.1 (17.6) |

- Weights: components 266,528 shared, 533,056 separate, 265,532 matched (widths 22, 45, 90); attributes 288,328
  shared, 932,190 separate, 288,214 matched (18, 35, 69). Labels add 320 / 512.
- Transformed puzzles (mirror, size ↔ colour): the shared models are identical by construction; labelled and separate
  models move by at most 0.8 points (they learn the symmetry from the symmetric data).
- One seed each.

**The value symmetry in the value-axis model** (3×10, values 0–50, plus relation given; matched (exact)):

| training puzzles | shared across values (10.8k) | weights per value (335k) |
|---|---|---|
| 100 | **99.0** (97.3) | 38.3 (4.0) |
| 600 | **99.9** (99.9) | 92.7 (81.0) |
| 6000 | **100.0** (**100.0**) | 98.4 (97.4) |

**Transfer to an unseen layout** (trained and selected on six layouts without up-down, scored on up-down, 1000 unseen
puzzles; left-right is the mirror layout, which is trained). Matched %; chance 12.5:

| model | up-down, never seen | left-right, trained |
|---|---|---|
| facet | **99.1** | 99.3 |
| attribute GNN B | **99.5** | 99.8 |
| attribute GNN A | **99.9** | 99.9 |
| object GNN | 23.6 | 97.6 |
| ViT, width 224 | 14.0 | 98.1 |
| ViT, rows 1 ↔ 2 built in, width 224 | 13.7 | 99.2 |
| MLP | 13.3 | 86.6 |

Exact on up-down is 0 for every model, because the position bits of an unseen layout were never trained. The facet model's
type, size and colour on up-down are 99.9 / 98.4 / 99.3 / 94.3 exact by rule (Constant / Progression / Distribute Three /
Arithmetic). The torch models on the same score, with the occupancy given: GNN B 100.0 / 100.0 / 100.0 / 97.6, GNN A 100.0 /
100.0 / 100.0 / 98.5, object GNN 28.6 / 23.0 / 31.9 / 19.9, ViT 20.5 / 14.8 / 16.7 / 16.6, ViT rows built in 21.6 / 14.2 / 17.5 /
15.1, MLP 17.1 / 20.5 / 15.8 / 12.2.

**A rule on an attribute it was never seen on** (no puzzle with Progression on Size in training or selection, which leaves
28,067 training puzzles, about 4000 per layout, so the fair reference lies between 3000 and 6000). Exact % on size among the
2577 unseen components with Progression on Size:

| model | held out | with it, 3000 puzzles | with it, 6000 puzzles | size, other rules, held out |
|---|---|---|---|---|
| facet | **87.7** | 89.9 | 90.3 | 85.6 |
| attribute GNN B | **84.1** | 89.4 | 88.0 | 85.1 |
| attribute GNN A | 42.3 | 85.9 | 84.2 | 84.6 |
| ViT, rows 1 ↔ 2 built in, width 224 | 23.6 | 65.3 | 86.9 | 73.9 |
| object GNN | 20.6 | 83.2 | 81.2 | 75.7 |
| MLP | 19.7 | 68.0 | 73.6 | 66.0 |
| ViT, width 224 | 18.5 | 48.9 | 88.3 | 65.0 |

Matched on the 2368 puzzles with that pair: facet 97.6, GNN B 97.4, GNN A 89.5, object GNN 67.5, ViT rows built in 65.4, MLP
52.9, ViT 48.3. The facet model loses 2.2 points against 3000 puzzles and GNN B 5.3. GNN A has the same sharing as B plus
labels, and loses 43.6.

**Rows 1 ↔ 2 built into the ViT** (the same ViT with the two rows given as a set; same-size comparison in the learning curves
above). Width 64: 79.1 → 89.4 at 6000 and 42.7 → 70.7 at 3000. Width 224: 93.4 → 92.1 and 45.8 → 62.0. Neither learns
at 300 or 1000. Rows built in do not help transfer: up-down stays at chance, 13.7.

**Weight sharing across patches in the ViT** (width 64): a separate filter per patch, 4,019k weights, gets 49.8 (9.2) at 6000
against 79.1 (37.6) with one filter, 335k; at 1000 both fail, 22.5 (0.3) and 19.2 (0.2).

**Why the ViT loses on grids** (width 224, one seed; matched % on 1000 unseen puzzles per layout; panel and patch sides in
pixels). The last three rows are the runs above, read by layout:

| trained on | puzzles | of them grid | panel, patch | 2×2 | 3×3 | in-out grid |
|---|---|---|---|---|---|---|
| grids only | 18,000 | 18,000 | 80, 16 | 37.6 | 28.3 | 43.0 |
| grids only, finer patches (3,861k) | 18,000 | 18,000 | 80, 8 | 31.5 | 29.9 | 38.6 |
| grids only, larger images (3,904k) | 18,000 | 18,000 | 160, 16 | 38.6 | 29.2 | 32.4 |
| all layouts | 21,000 | 9,000 | 80, 16 | 41.8 | 37.1 | 41.8 |
| all but up-down | 36,000 | 18,000 | 80, 16 | 82.7 | 73.4 | 73.8 |
| all layouts | 42,000 | 18,000 | 80, 16 | 93.0 | 81.8 | 81.1 |

Exact is at most 0.4 in the three grids-only runs. All three memorise: training loss ends at 0.0002–0.0025 while selection
stays at 33–36, with the best epochs at 54, 108 and 168. Finer images change nothing at this amount of data. The same
18,000 grid puzzles reach 73–83 once 18,000 single-object puzzles are added, and 9,000 grid puzzles with 12,000 others match
the grids-only runs. Finer images on a ViT that has learned the task (42,000 puzzles) were not tested.

**Wrong symmetries** (plan E4, 2026-11-08; facet model, I-RAVEN-X 3×3, values 0–10; jobs 11978039, 11978041–42, 11978387–88). A symmetry
the data does not have, built in exactly (max output difference 0 under its transformation with random weights).
Against the plain model (6000 puzzles: 99.2 (92.7), exact per rule 99.6 / 99.5 / 99.3 / 95.0). Exact per rule is
Constant / Progression / Distribute / Arithmetic.

| built in | kind | prediction | 6000 puzzles | 600 puzzles (600 epochs) |
|---|---|---|---|---|
| none (plain, named columns) | – | – | 99.2 (92.7); per rule 99.6 / 99.5 / 99.3 / 95.0 | 95.1 (82.2); per rule 99.6 / 99.3 / 89.2 / 76.2 |
| columns 1 ↔ 2 tied (`--columns tie12`) | out of support: the swap breaks Progression rows | no ceiling; at most a data cost | 98.6 (95.1); per rule 99.6 / 100.0 / 96.3 / 89.9 | 92.1 (79.2); per rule 99.6 / 99.5 / 94.7 / 62.4 |
| v ↔ 10 − v in the input (`--fold-values`) | conflicting: the reflected puzzle is valid with another answer | Constant / Progression / Distribute exact at most 54.4 / 56.0 / 55.0; Arithmetic no forced cap | 46.6 (13.4); per attribute 52 / 53 / 50 | 39.7 (14.0); per attribute 57 / 52 / 49 |

- **Out of support costs little at full data, more at 600 puzzles.** Tying columns 1 and 2 keeps 98.6 matched at 6000
  puzzles; Distribute and Arithmetic lose 3 and 5 points exact. No ceiling, as predicted. At 600 puzzles it costs 3 points
  matched (92.1 against 95.1), all of it on Arithmetic (62.4 against 76.2 exact), while Distribute gains (94.7 against 89.2):
  Distribute does not care about column order, Arithmetic's operands do.
- **Conflicting caps the model where the count says.** Folding v and 10 − v leaves 50–53% of attribute values exact (type,
  size, colour, all rules pooled), just under the 54–56 ceiling the count gives Constant, Progression and Distribute. More
  data barely moves it (600 → 6000: 39.7 → 46.6 matched, 14.0 → 13.4 exact). Per attribute is the bench's decoding, which picks one side of
  the reflected pair. `facet_rule_eval.py`'s per-rule exact sums the thermometer bits instead, which lands between v and
  10 − v when the model hedges, so its per-rule numbers for these runs (Constant 17.9, Progression 36.3) understate and are
  left out.

## 8. Perception (ResNet-18 frontend, trained on RAVEN; I-RAVEN test)

| | panels exactly right | puzzles with all 16 right |
|---|---|---|
| center, L-R, U-D, in-out | 100.00 | 100.00 |
| 2×2 / 3×3 / in-out grid | 99.99 / 96.02 / 96.34 | 99.90 / 76.45 / 71.55 |
| overall | 98.91 | 92.56 |

**ViT slot recognizer** (`core/frontend/train_vit_recognizer.py`, width 64, 325,718 weights, trained on 2000 I-RAVEN
puzzles per layout, epoch 22 of 30 picked on val 0–299; jobs 11978381–83). Not yet used by any reasoning model.

| | panels exactly right | puzzles with all 16 right |
|---|---|---|
| center, 2×2, L-R, U-D | 100.00 | – |
| 3×3 / in-out / in-out grid | 99.92 / 99.99 / 95.56 | – |
| overall, val 300–1299 | 99.35 | 95.69 |
| overall, test | 99.40 | 95.64 |

It reads the 3×3 grid far better than the ResNet-18 (99.92 against 96.02); its weak layout is the in-out grid.

**The facet model end to end** (`core/pynl/facet_e2e_eval.py`, job 12028100; checkpoint `experiments/facet_all_w32-64-128_hidden`,
val 300–1299, 7000 puzzles; the test split follows below). Context panels and candidates come from the recognizer.
Exact is against the true answer.

| input | matched | matched on the labels' candidates | exact | in-out grid | the other six layouts |
|---|---|---|---|---|---|
| XML labels | 97.31 | 97.31 | 69.09 | 97.8 | as recorded |
| recognizer, hard (argmax) | **97.13** | 97.23 | 68.79 | 96.5 | unchanged |
| recognizer, soft where unsure | 97.13 | 97.24 | 68.86 | 96.5 | unchanged |

- The labels pass reproduces the checkpoint's record exactly (6812 matched, 4836 exact), from the recognizer data's label
  array and in slot order, so the scorer and the slot conventions are right.
- Reading the panels costs 0.18 points, 13 puzzles of 7000, all in the in-out grid, the one layout the recognizer
  misreads. Where all 16 panels were read right the model matches 97.33; in the 302 puzzles with a misread panel, 92.72.
- Soft codes change nothing here: the recognizer is unsure of too few bits. Raw soft codes cannot be used at all: value 0
  is the all-zero thermometer code, the template has no biases, and `NORM` rescales any non-zero vector to full size, so
  1e-6 in place of 0 flips outputs by 1.0. The soft pass therefore keeps bits below 0.01 and above 0.99 exact.
- For comparison, the symbolic rules lose 2.0 points through the ResNet-18 (98.33 → 96.31, test), and the black-box ViT
  gets 93.4 (64.4) on these puzzles with 3,753k weights.

**Test split, one pass** (2026-11-14, job 12031767; test puzzles 0–1999 of every layout, 14,000 puzzles; the mapping was
fixed to hard on the unseen split first). This checkpoint's test pass is now spent: do not score it on test again.

| input | matched | exact | center | 2×2 | 3×3 | L-R | U-D | in-out | in-out grid |
|---|---|---|---|---|---|---|---|---|---|
| XML labels | **97.19** | 68.35 | 99.6 | 95.0 | 89.1 | 99.8 | 99.7 | 99.8 | 97.5 |
| recognizer, hard | **97.03** | 68.09 | 99.6 | 95.0 | 89.1 | 99.8 | 99.7 | 99.8 | 96.4 |

The test numbers sit 0.1 below the unseen validation ones (97.31 and 97.13), so weeks of looking at those puzzles did not
inflate them. End to end costs 0.16 points, again all in the in-out grid; in the 611 puzzles with a misread panel the
model still matches 92.64. The recognizer reads all 16 panels right in 95.64% of test puzzles. The baselines' test
passes are in §1 ("I-RAVEN test split, one pass per checkpoint"): ViT 93.10, GNN B 96.79.

## 9. Weight counts

Learnable parameters, counted on 2026-10-31 from the saved checkpoints (`template_best.bin`, `model_best.pt`).
All 102 torch checkpoints agree with their `best.json`.

| model | I-RAVEN | 3×3, 0–10 | 3×10, 0–50 | 3×10, 0–100 | 3×10, 0–1000 |
|---|---|---|---|---|---|
| facet, named columns (60–80 or 150 epochs) | 266,528 | 266,528 | 1,410,632 | 1,425,082 | 1,685,182 |
| + copy links, per-panel gates (also shuffled control at 0–50) | 276,848 | 276,848 | 1,597,682 | 1,799,182 | 5,426,182 |
| + copy links, content gates | 271,668 | 271,668 | 1,436,332 | – | – |
| column roles | – | – | 288,328 | – | – |
| column roles + content gates | – | – | 314,028 | – | – |
| column roles, wide | – | – | 1,389,960 | – | – |
| column roles + content gates, wide | – | – | 1,447,660 | – | – |
| row fold | 282,912 | 282,912 | 304,712 | – | – |
| row fold + content gates | – | – | 330,412 | – | – |
| value axis (I-RAVEN adds the slot block and the count) | 18,018 | 10,817 | 10,817 | same checkpoint | – |
| value axis, without plus | 17,826 | 10,689 | 10,689 | same checkpoint | – |
| value axis, plus used by learned rules | – | 15,841 | 15,841 | same checkpoint | – |
| value axis, plus used by learned rules, the state also along next | – | 20,961 | 20,961 | same checkpoint | – |
| value axis, weights per value | – | – | 335,475 | – | – |
| symbolic template, rules / primitives | 41 / 94 | – | – | – | – |
| MLP | 264,061 | 267,037 | 1,596,930 | 1,802,348 | 5,383,288 |
| MLP, regularized (adds LayerNorm) | 264,343 | 265,813 | 1,598,766 | 1,803,482 | 5,383,642 |
| object GNN, plain and regularized | 264,898 | 265,454 | 1,593,366 | 1,798,274 | 5,430,102 |
| attribute GNN A | 266,252 | – | 1,595,743 | – | – |
| attribute GNN B | 267,008 | – | 1,599,256 | – | – |
| MLP 4× / 16× / 64× | 1,065,247 / 4,265,311 / 17,062,612 | – | 6,392,322 / 25,555,878 / 102,255,930 | – | – |
| MLP, regularized 4× / 16× | 1,066,333 / 4,261,042 | – | 6,389,662 / 25,556,476 | – | – |
| object GNN 4× / 16× / 64× (regularized: 4×, 16×) | 1,062,942 / 4,257,502 / 17,069,122 | – | 6,387,276 / 25,567,296 / 102,226,074 | – | – |
| ViT, width 64 / width 224 | 335,038 / 3,752,958 | 335,038 / 3,752,958 | 428,190 / 4,078,590 | – | – |
| ViT, rows 1 ↔ 2 built in, width 64 / width 224 | 322,558 / 3,709,278 | – / 3,709,278 | – / 3,918,878 | – | – |
| ViT, width 64, a separate filter per patch | 4,019,390 | – | – | – | – |
| CNN / CNN 16× | 286,282 / 4,269,548 | – | – | – | – |

**Compute cost** (peak memory from SLURM's MaxRSS, CPU nodes, one run each; 2026-11-08). PyNeuraLogic grounds every
puzzle into its own network and keeps it in the JVM, so memory grows with the number of puzzles, the code length and
the value range. This is a real cost of the approach: the torch baselines of the same size train on one GPU in a few
GB.

| run | job | memory requested / peak | time |
|---|---|---|---|
| facet + copy links, I-RAVEN-X 3×10, values 0–1000 (5,426k weights) | 11904250 | 420 GB / **344 GB** (an earlier try died at 180 GB while grounding) | 16.3 h |
| facet + content gates, I-RAVEN, all seven layouts | 11932572 | 420 GB / 327 GB | 15.0 h |
| row fold, 3×10, values 0–50 | 11914473 | 160 GB / 71 GB | 5.0 h |
| value axis, `plus` used by learned rules, also along `next`, 3×10 | 11962270 | 180 GB / 161 GB | 6.8 h |
| value axis, `plus` used by learned rules, 3×10 | 11962269 | 120 GB / 104 GB | 3.0 h |

- Grounding alone took 30–45 minutes on the largest runs; `plus` derived from the zero and `next` (`--peano`) grounds
  22× slower on 3×10 than the table.
- A grounded wide column-roles puzzle takes up to 12 MB, so 100,000 puzzles do not fit a node and
  `core/pynl/facet_stream.py` grounds them in chunks, again every epoch.
- Updates are one puzzle at a time (no batching in native PyNeuraLogic training), so an epoch costs about as many
  steps as there are puzzles.

## 10. What the models do inside

**Are the rules readable from the facet model's row relation?** Unseen I-RAVEN puzzles, 200 per layout. A linear
classifier predicts an attribute's rule from `rel(Rw, C, A)`, the mean over rows 1 and 2, fitted on half of the puzzles
and scored on the other half (`analysis/facet_rel_vectors.py`, `analysis/facet_rel_analysis.py`). Untrained: the same template at
its random initial weights, one draw. Raw input: the thermometer codes of the six panels. AMI: k-means against the rules,
no labels used.

| layouts | trained | untrained | raw input | commonest rule | AMI trained / untrained |
|---|---|---|---|---|---|
| single-object | 95.6 | **94.0** | 54.4 | 31.6 | 0.182 / 0.159 |
| grids | **93.7** | 79.5 | 54.4 | 37.9 | 0.186 / 0.095 |

Per attribute, trained / untrained: single-object type 100.0 / 100.0, size 96.3 / 94.0, colour 94.0 / 88.3; grids type
99.8 / 88.5, size 95.8 / 80.5, colour 87.8 / 68.8.

**A trained model's answers when the picture moves** (`analysis/vit_translation.py`, matched on unseen puzzles; the
training images are never shifted):

| model | 0 px | 1 px | 2 px | 4 px | 8 px | 16 px | 32 px |
|---|---|---|---|---|---|---|---|
| ViT, width 224 | 93.4 | 43.3 | 37.8 | 31.8 | 23.5 | 21.8 | 20.2 |
| ViT, width 64 | 79.1 | 35.0 | 32.1 | 27.2 | 21.1 | 19.7 | 14.9 |
| CNN, 16× | 50.5 | 46.6 | 42.4 | 33.4 | 26.4 | 32.8 | 21.8 |
| CNN | 39.4 | 34.8 | 33.0 | 26.0 | 21.9 | 23.8 | 18.8 |

Neither is invariant. The CNN loses less at 1 px, 8–12% against the ViT's 54–56%, but it starts far lower, and it differs from
the ViT in more than filters, so this table does not say how much of that is translation.

## 11. Rule twins: is the missing symmetry what makes Arithmetic hard?

I-RAVEN-X rows of 3, values 0–50, with Arithmetic swapped for a test rule (`data_prep/rule_twins.py`). Design, refutation tests and
limits: `docs/ARITHMETIC.md`. Exact % on the test rule at 100 / 600 / 6000 puzzles, on unseen puzzles, epoch picked by exact
on the selection puzzles. No model is given `plus`. For the facet rows, that epoch was picked with the old exact scorer
(notes on the main table), which counted 2.4–3.7% of these puzzles wrong even when perfect; the per-rule exacts here
are unaffected, the epoch choice may not be.

| model | weights | sum | midpoint | mirror | lookup |
|---|---|---|---|---|---|
| shared, shift-equivariant | 14,785 | 4.7 / 6.9 / 7.1 | 57.4 / 89.3 / 92.7 | 45.3 / 78.3 / 88.1 | 35.9 / 74.3 / 77.7 |
| per value | 459,635 | 4.9 / 19.3 / 69.8 | 5.7 / 17.8 / 75.2 | 4.6 / 15.7 / 63.2 | 53.0 / 97.5 / 97.5 |
| facet (600 / 6000) | 278,088 | 31.2 / 65.5 | 24.0 / 76.6 | 14.7 / 41.9 | 60.3 / 97.1 |
| MLP, mean of 2 seeds | 277,828 | 2.7 / 3.0 / 26.4 | 4.8 / 5.9 / 21.7 | 2.9 / 2.9 / 10.2 | 2.8 / 5.6 / 17.7 |
| memorising answered rows | – | 4.7 / 24.1 / 90.7 | 5.0 / 21.6 / 90.5 | 4.0 / 19.7 / 91.7 | 65.3 / 100 / 100 |
| shared + `plus` used by learned rules (jobs 11974990–93; 100 / 600) | 19,937 | **97.6 / 99.8** | 70.5 / 91.5 | – | – |

**Three seeds** (2026-11-14, jobs 12034883–12034918, `analysis/twins_seeds.py`; per seed in `docs/ARITHMETIC.md`,
"Three seeds"). Mean ± sd of seeds 1–3, exact % on the test rule on unseen puzzles. One rule for every cell, fixed
before the runs: the epoch with the highest whole-puzzle exact on the selection puzzles.

| model | sum, 100 puzzles | sum, 600 | midpoint, 100 | midpoint, 600 |
|---|---|---|---|---|
| shared, shift-equivariant | 4.5 ± 2.3 | 6.2 ± 0.8 | 60.3 ± 2.7 | 91.0 ± 3.4 |
| per value | 4.8 ± 0.6 | 19.4 ± 1.1 | 5.0 ± 0.5 | 18.0 ± 0.9 |
| shared + `plus` used by learned rules | 98.2 ± 0.8 | 99.7 ± 0.4 | 63.9 ± 3.9 | 91.7 ± 1.8 |

- The single runs above lie inside the seeds' range or within 0.2 points of it, except `plus` on midpoint at 100
  puzzles (70.5 against 59.4–66.1). So `plus` does not help midpoint: 63.9 ± 3.9 against 60.3 ± 2.7.
- The `plus` cells on sum above (97.6 / 99.8) were picked by exact on the test rule; under the rule here that run
  gives 98.2 / 99.2.
- Sum on the shared model is under its ceiling of 7.9 in all six seeded cells (1.8 to 7.1).
- Midpoint rows whose first step no training question had, shared model at 100 puzzles: 0 of 101 right at each seed,
  against 71.5 / 77.5 / 77.2% of the 404 others, from the checkpoints in this table (job 12034926).
- The 6000-puzzle column, mirror, lookup and the facet rows remain one run each.

**The scaling kernel** (2026-11-15, jobs 12039041–12039065; `docs/ARITHMETIC.md`, "The scaling kernel"). The shared model
with every row read in its own unit: its first step (*scaled*, `--scale step`, equivariant to v → kv + s on the ring of
101 values) or the largest power of two in its first step (*doubled*, `--scale pow2`, equivariant to doubling). 14,785
weights each; both symmetries exact with random weights on 100 puzzles (job 12039067). Same selection rule, seeds 1–3,
exact % on the test rule on unseen puzzles:

| model | sum, 100 puzzles | sum, 600 | midpoint, 100 | midpoint, 600 |
|---|---|---|---|---|
| shared (shift only) | 4.5 ± 2.3 | 6.2 ± 0.8 | 60.3 ± 2.7 | 91.0 ± 3.4 |
| doubled | 3.6 ± 0.2 | 4.1 ± 0.3 | 49.9 ± 5.3 | 67.5 ± 2.0 |
| scaled, seeds 1 / 2 / 3 | 3.2 / 0.8 / 2.0 | 3.4 / 1.0 / 3.4 | 19.0 / 100.0 / 21.0 | 29.5 / 100.0 / 20.2 |

Unseen midpoint rows right, seeds 1 / 2 / 3 (`analysis/twins_seeds.py --dumps experiments/twins_eval`):

| rows | shared | doubled | scaled |
|---|---|---|---|
| 100 puzzles: first step new, its odd part seen (72) | 0 / 0 / 0 | 50 / 43 / 39 | 18 / 72 / 19 |
| 100 puzzles: first step and odd part new (29) | 0 / 0 / 0 | 0 / 0 / 0 | 4 / 29 / 4 |
| 600 puzzles: rows that start like a Progression row in the model's unit | 0 / 0 / 0 of 33 | 1 / 11 / 1 of 152 | 149 / 505 / 102 of 505 |
| 600 puzzles: the other rows | 440 / 470 / 469 of 472 | 330 / 340 / 340 of 353 | none |

- The split of the rows with a new first step was predicted by counting before the runs (72 answerable, 29 not, for
  the doubled model; all 101 for the scaled one) and held at every seed.
- Not predicted: a midpoint row whose first step is one unit has the first two panels of a Progression row, and the
  network continues it as a Progression (it generates 2b − a on 139 to 150 of the doubled model's 152 such rows, and on
  all 33 of the shared model's). The scaled model merges every row; one seed of three tells the rules apart by the
  complete rows and is then right on all 505 rows from 100 puzzles.
- Sum: doubled under its ceiling of 7.2; scaled at most 3.4, which is the 17 unseen rows with b = 2a, where continuing a row by its first step is right (the most one multiple can reach on these rows).

**The rule gate** (2026-11-16, jobs 12043421–32; `docs/ARITHMETIC.md`, "The rule gate"). Why does the scaled model learn
midpoint at one seed of three? Tested first: the answer layer. `--gate` multiplies the question row's kernel features
by a sigmoid of the rule descriptor, feature by feature (15,841 weights, 1,056 more; the symmetries stay exact with
random weights, job 12043433). Predicted before the runs: scaled with the gate at 100 at every seed; shared with the
gate right on its 33 rows with first step ±2 at 600 puzzles. Exact % on midpoint, unseen puzzles, seeds 1 / 2 / 3:

| model | 100 puzzles | 600 puzzles | the 33 rows with first step ±2, 600 puzzles |
|---|---|---|---|
| shared | 57.2 / 62.0 / 61.8 | 87.1 / 93.1 / 92.9 | 0 / 0 / 0 |
| shared with the gate | 72.5 / 65.7 / 61.6 | 90.3 / 88.1 / 92.5 | 0 / 0 / 0 |
| scaled | 19.0 / 100.0 / 21.0 | 29.5 / 100.0 / 20.2 | |
| scaled with the gate | 23.2 / 20.2 / 24.0 | 20.2 / 20.2 / 30.5 | |

Both predictions failed. Measured next on the saved checkpoints (`analysis/twins_probe.py`, job 12043845, 30 unseen
puzzles): the row descriptor `rd` of the first row, for attributes under midpoint against attributes under Progression,
largest difference per feature (features in −1..1).

| runs | midpoint, exact % | `rd`: midpoint row against Progression row |
|---|---|---|
| scaled, seed 2 (100 and 600 puzzles) | 100.0, 100.0 | 1.5, 1.4 |
| scaled, seeds 1 and 3 (4 runs) | 19.0 to 29.5 | 5e-11 to 7e-05 |
| scaled with the gate (6 runs) | 20.2 to 30.5 | 1e-14 to 2e-05 |
| shared (6 runs) | 57.2 to 93.1, none of the 33 rows | 6e-14 to 1e-05 |
| shared with the gate (6 runs) | 61.6 to 92.5, none of the 33 rows | 1e-13 to 2e-07 |

- In the 22 runs that fail, the row descriptor is one vector for every row, so the complete rows cannot tell the model
  the rule. In the two runs that learn midpoint it separates the rules.
- What the failed scaled runs use instead: at 600 puzzles they are right on 101 to 107 of the 108 midpoint rows whose
  question row shares a value with a complete row, and on 0 to 47 of the 397 others (seeds 1 and 3 without the gate,
  all three with it; `analysis/twins_seeds.py --dumps`, third split).
- Run on 2026-11-17: a row descriptor that reads a pair's step directly ("The row descriptor reads the steps" below).

**Learners that know only a symmetry** (`analysis/rule_orbits.py`, no network; fit by majority vote per orbit on the
training question rows, rule told; `docs/ARITHMETIC.md`, "Sum and midpoint compared"). Exact % at 100 / 600 / 6000
puzzles, and with unlimited data:

| built-in move | sum | midpoint | mirror |
|---|---|---|---|
| none (memorising) | 4.7 / 24.1 / 90.7; 100 | 5.0 / 21.6 / 90.5; 100 | 4.0 / 19.7 / 91.7; 100 |
| every panel +s (the convolution's) | 3.6 / 6.3 / 7.3; 6.9 | 80.0 / 99.8 / 100; 100 | 70.8 / 98.0 / 100; 100 |
| one operand with the answer (addition's) | 100 / 100 / 100; 100 | 3.4 / 2.2 / 2.2; 2.2 | 0.8 / 1.0 / 1.2; 2.0 |
| every panel +s and ×2 (the doubled model's) | 3.0 / 5.1 / 6.1; 6.3 | 94.3 / 99.8 / 100; 100 | 88.9 / 100 / 100; 100 |
| every panel +s and ×k (the scaled model's) | 1.0 / 0.6 / 2.0; 2.0 | 100 / 100 / 100; 100 | 100 / 100 / 100; 100 |

**Two checks without training** (2026-11-17; `docs/ARITHMETIC.md`, same heading). Unseen midpoint rows right by how
many training question rows fall in the row's orbit (first step for shared, pair for per value), three seeds pooled
(`analysis/twins_seeds.py --dumps`):

| training rows in the orbit | 0 | 1 | 2 | 3–4 | 5–8 | 9+ |
|---|---|---|---|---|---|---|
| shared, 100 puzzles | 0 of 303 | 422 of 573 (74%) | 286 of 417 (69%) | 172 of 186 (92%) | 34 of 36 | – |
| shared, 600 puzzles | 0 of 3 | 71 of 72 (99%) | 2 of 3 | 89 of 90 (99%) | 696 of 768 (91%) | 521 of 579 (90%) |
| per value, 100 puzzles | 11 of 1,440 (1%) | 65 of 75 (87%) | – | – | – | – |
| per value, 600 puzzles | 22 of 1,188 (2%) | 200 of 270 (74%) | 45 of 51 (88%) | 6 of 6 | – | – |

With no training row in the orbit a model is at chance or below; one row is enough for most of an orbit. At 600 puzzles
99 of the 133 wrong rows are the rows with first step ±2, which sit in the last two bins.

A fit with no training, c = p·a + q·b on the puzzle's two complete rows (`analysis/twins_linear_fit.py`), exact % on
the unseen puzzles: sum 99.6, midpoint 99.8, mirror 99.8, lookup 1.7; Progression 98.6 to 99.3, Constant 100,
Distribute Three 0. A reference with values read as numbers and the linear form given; it is not ranked against the
learned models.

**The counts behind the proofs** (2026-11-17; `analysis/pricing_checks.py`, nothing trained; `docs/ARITHMETIC.md`,
same heading; proofs in `docs/thesis/ch3_pricing_proofs.md`). Midpoint, share of unseen rows whose orbit the training
rows reach: expected from the table alone / reached by the counting learner.

| built in | orbits | 100 puzzles | 600 | 6,000 | rows for 90% |
|---|---|---|---|---|---|
| nothing (one answer per pair) | 1,250 | 4.5 / 5.0 | 22.0 / 21.6 | 91.2 / 90.5 | 2,878 |
| every panel +s | 50 | 74.8 / 80.0 | 98.7 / 99.8 | 100 / 100 | 107 |
| +s and ×2 | 26 | 90.5 / 94.3 | 99.3 / 99.8 | 100 / 100 | 55 |
| +s and ×k | 1 | 100 / 100 | 100 / 100 | 100 / 100 | 1 |

A function of the whole puzzle built from differences only, c₃ = b₃ + (c₁ − b₁) + (a₃ − a₁), is right on 507 of 507
unseen sum rows: the 7.9% ceiling belongs to learners that answer from the question row, not to the puzzle-wide shift.
The shared model is 7.1 or less on sum at every picked epoch; over all 172 evaluated epochs of its nine runs, 8 are
above 7.9 (9.7 at the most), and 507 rows fix the rate only to 1.2 points.

**The row descriptor reads the steps** (2026-11-17, jobs 12049211–46, probe 12049249; `value_axis.py --steps`;
`docs/ARITHMETIC.md`, same heading; predictions written before the runs). The row descriptor also reads each
neighbouring pair's step directly, 2,048 more weights (16,833), the same symmetries. Exact % on midpoint, unseen
puzzles, seeds 1 / 2 / 3:

| model | 100 puzzles | 600 puzzles |
|---|---|---|
| shared | 57.2 / 62.0 / 61.8 (60.3 ± 2.7) | 87.1 / 93.1 / 92.9 (91.0 ± 3.4) |
| shared with the steps | 54.9 / 63.4 / 61.4 (59.9 ± 4.5) | **96.0 / 99.4 / 99.4** (98.3 ± 1.9) |
| doubled | 55.6 / 48.7 / 45.3 (49.9 ± 5.3) | 65.5 / 69.5 / 67.5 (67.5 ± 2.0) |
| doubled with the steps | **90.1 / 91.3 / 86.1** (89.2 ± 2.7) | **99.4 / 99.0 / 99.2** (99.2 ± 0.2) |
| scaled | 19.0 / 100.0 / 21.0 | 29.5 / 100.0 / 20.2 |
| scaled with the steps | **100.0 / 100.0 / 100.0** | **100.0 / 100.0 / 100.0** |
| counting learners: shared, doubled, scaled | 80.0, 94.3, 100 | 99.8, 99.8, 100 |

| rows right with the steps, seeds 1 / 2 / 3 | shared | doubled | scaled |
|---|---|---|---|
| 100 puzzles: new first step, odd part seen (72) | 0 / 1 / 0 | 64 / 69 / 67 | 72 / 72 / 72 |
| 100 puzzles: new first step, new odd part (29) | 0 / 0 / 0 | 0 / 0 / 0 | 29 / 29 / 29 |
| 100 puzzles: rows that begin like a Progression row | 17 / 27 / 19 of 33 | 152 / 149 / 147 of 152 | 505 / 505 / 505 of 505 |
| 600 puzzles: rows that begin like a Progression row | 33 / 33 / 33 of 33 | 152 / 151 / 152 of 152 | 505 / 505 / 505 of 505 |

- Six predictions met, one missed by one row (the shared model is right on 1 of 303 rows with a first step it never
  saw, the rate of guessing).
- The probe: in all 18 runs the row descriptor of a midpoint row differs from a Progression row's by 0.87 to 2.0 per
  feature; it was below 0.0001 in 22 of the 24 runs without the steps.
- The accuracy lost with the scaling kernel was the row descriptor's, not the symmetry's. Not run: sum with the steps.

## 12. Arithmetic-only training (does it grok?): a diagnostic, not a benchmark result

I-RAVEN-X 3×10, values 0–50, every attribute Arithmetic (`data_prep/arith_big_data.py`); 300 selection and 1000 unseen puzzles of
the same kind. Arithmetic exact %, type, size and colour pooled. Details: `docs/ARITHMETIC.md`, "Does it grok?".
The rule is known in training and test, so these rows say whether a model can learn the sum, not how it does on Arithmetic
in the benchmark: compare models on Arithmetic only on the mixed-rule puzzles (§1, §5).

| model | weights | weight decay | train / unseen |
|---|---|---|---|
| column roles, wide | 1,390k | 0.1, decoupled, per puzzle | 51.6 / 52.4 at 40,000 puzzles, 38.7 / 38.8 at 120,000; cancelled |
| attribute GNN B, 16×, no dropout | 25.6M | 0.1, constant learning rate | 99.2 / 99.3 (epoch 45) |
| attribute GNN B, 16×, first 1000 / 3000 / 10000 puzzles | 25.6M | 1, β2 0.98 | best 51.6 / 46.3, 19.0 / 20.1, 26.5 / 27.1; loss rising by 12,000 steps, cancelled |
| attribute GNN B, 16×, first 1000 puzzles | 25.6M | 0 / 0.1 | 99.8 / **98.3**, 99.7 / 97.0 (by 16,000 steps) |
| attribute GNN B, 16×, first 3000 puzzles | 25.6M | 0.1 | 99.1 / **99.0** (12,000 steps) |
| attribute GNN B, 16×, first 100 puzzles | 25.6M | 0 / 0.1 | 100 / 6.4, 100 / 10.4 (300,000 steps) |
| attribute GNN B, 16×, first 300 puzzles | 25.6M | 0.1 | 100 / 61.7 at 16,000 steps, then down to 24–30 by 300,000 |
| row fold, first 1000 puzzles, no decay (job 11974980) | 305k | 0 | best 90.7 / **63.4** (epoch 110), then train 98 / unseen 48 by 400,000 updates |
| row fold, first 1000 puzzles (jobs 11974981–82) | 305k | 1, decoupled, every 100 puzzles; ×1 / ×3 initial weights | best 25.7 / 24.6, 23.9 / 24.5: train never fits |
| row fold, first 300 puzzles (job 11974983) | 305k | 1 | best 31.7 / 29.0: train never fits |
| row fold, first 1000 puzzles (jobs 11978016–17) | 305k | 0.03 / 0.01 | best 86.1 / **65.7** (epoch 125), 81.1 / 58.7 (epoch 95); over the last 100 epochs train 86 / unseen 43 and 97 / 46: memorised, no grokking in 600,000 updates |
| attribute GNN B, **1×**, first 1000 puzzles (job 11978018) | 1.6M | 0 | 100 / **98.0** (epoch 1750 of 2500, picked on selection; 98.6 / 97.5 / 98.0 per attribute, 99.5 matched) |

For comparison on the benchmark, **mixed rules**: attribute GNN B at 1× gets 22.7 on Arithmetic (§5) with dropout 0.3 and decay 0.5;
with the recipe above instead (no dropout, no decay, constant learning rate 1e-3, 40,000 steps; seeds 0 and 1, jobs
11978453–54) it gets **45.1 / 54.0** at the epoch picked by matched on selection (132 / 176 of 425), and 50.6 / 49.2 at the
last epoch. Whole puzzles: 97.9 (45.5) / 98.2 (56.5) picked, 99.2 (63.9) / 99.3 (61.5) at the end; per rule at the end
Constant 100 / 99.8, Progression 100 / 99.7, Distribute 82.1 / 79.0. So the 22.7 was the regularisation, not the mixing of
rules, and this is the best Arithmetic of any model that has to learn the sum on 3×10 (column roles: 40.5). **The
regularized recipe of §1 and §5 was chosen on I-RAVEN and holds the torch baselines back on 3×10; GNN A, the object GNN and
the MLP have not been rerun without it, and the facet models saw 60–150 epochs against these 425.** Until that is done, the
3×10 comparisons between the facet family and the torch baselines are not like for like.

## 13. Plus and minus apart

Column roles (288k weights) on the I-RAVEN-X 3×10 puzzles whose Arithmetic uses one operator (`data_prep/arith_op_data.py`),
against the two-operator checkpoint on the same unseen puzzles. Exact value on Arithmetic, epoch picked by matched; one seed.
Details: `docs/ARITHMETIC.md`, "Plus and minus apart".

| trained on | puzzles | plus (751 unseen) | minus (769 unseen) |
|---|---|---|---|
| both operators | 6000 | **49.5** | 29.8 |
| plus only | 4622 | 35.0 | – |
| minus only | 4588 | – | **34.2** |
