# Audit of the original code (Tehlar, thesis_paper @ 05874b9)

Scope: the repository only holds the Pixtral-12B experiment (Mistral API) on the 3_comp / 4_comp
I-RAVEN variants. The GPT-4o / VRB code described in the thesis is not in it.

Checked facts
- Labels are right: all 224 answer-sheet rows and all 4160 saved result rows match `target` in the npz.
- The renderer is fine apart from cosmetics (panels in the right order, candidates A-H in npz order).
- The I-RAVEN `Position.index` bug does not apply: the pipeline only uses images.

## Bugs

| # | where | bug | effect |
|---|---|---|---|
| 1 | Mistral/main.py:109,162 | `range(1, len_data)` skips the last row of every answer sheet | 13 of 14 puzzles per variant evaluated |
| 2 | main.py:112,118 + answer sheets | none of the 224 image paths in the committed answer sheets exists (3_comp: flat layout that was removed; 4_comp: `../4_comp/...`); `encode_image` returns None and the request is sent as `base64,None` without error | committed code cannot reproduce the results; it would silently query the model without an image |
| 3 | main.py:129 | PNG sent with `image/jpeg` MIME type | stricter APIs reject it |
| 4 | main.py:11 | `os.environ["MISTRAL_API_KEY"]` read at import | crashes when unset, so `--api_key` is useless |
| 5 | main.py:14, run.sh | default model is Mathstral-7B (text only) and run.sh does not pass `--model` | run.sh as committed sends images to a text model |
| 6 | main.py:157, results.py:151 | writer saves `output/<dataset>/<variant>/<model>/mistral_prompt/<variant>_run_k...`, reader expects `output/<model>/<prompt>/<dataset>/<variant>/...`; the `ft_prompt` text is not in the repo | outputs were moved by hand; the second prompt cannot be reproduced |
| 7 | main.py:140 | all responses held in memory, saved after the last run, no retry | one API error loses the whole experiment |
| 8 | main.py:170 | answer = everything after `The correct answer is: `, compared by string equality | `**F**.`, `C\n`, `D**` count as wrong |
| 9 | results.py:117-119 | `process_row` receives a string and indexes `row["prediction"]` | `apply_regex` crashes; the committed `regexed/` files come from an older version |
| 10 | results.py:125 + regexed CSVs | `correct` column never recomputed after regex | 508 of 4160 rows have prediction == answer but `correct=False`; the "after regex" accuracy equals the raw one exactly |
| 11 | results.py:112 | default pattern `[\*]*([A-Za-z])[\*]*` takes the first letter of the text: "Triangle (Option A)" → T, "Option H" → O, "hexagon" → h; the curated `re_patterns` list is never used | wrong letters extracted |
| 12 | results.py:119 | regex only applied to the cut-off prediction, never to the full response | answers outside the expected phrase are lost |
| 13 | processor.py:192 | ground truth read from `predict` (the generator's built-in solver), not `target` | equal on these 224 puzzles, wrong in general |
| 14 | answers_generator.py:43,56 | hard-codes 3_comp and writes `answers/`, while main.py reads hand-edited `data/` sheets (two formats for 4_comp) | data sheets not reproducible from the script |
| 15 | requirements.txt | lists `base64` and `csv` (standard library) and misses pandas, regex, matplotlib, opencv | `pip install -r` fails or is incomplete |
| 16 | processor.py:154 | candidate separators drawn 10 px inside the candidate panels | cosmetic |

## Method problems (not code bugs)
- 14 puzzles per variant (2 per configuration, train/val/test mixed), 13 evaluated: one puzzle is 7.7 points.
- 10 runs are samples at the API's default temperature; temperature and seed are not recorded.
- No baseline besides chance. The answer letters are not uniform: on the 208 evaluated puzzles,
  always answering H scores 15.9%.

## Re-scored Pixtral results (fixed parser, unparseable counted wrong)
Source: `results/tehlar_pixtral_rescored.md`, from his saved responses, 208 puzzles × 10 runs.

| prompt | his raw | his "regexed" | re-scored | 95% CI | unparseable |
|---|---|---|---|---|---|
| default_prompt | 5.0% | 5.0% | 14.6% ± 1.3 (over runs) | 13.2–16.2 | 0.8% |
| ft_prompt | 0.05% | 0.05% | 15.0% ± 2.1 | 13.6–16.6 | 3.7% |

Chance is 12.5% and always-H is 15.9%, so Pixtral-12B is not above a trivial baseline.
With ft_prompt it answers H in 41% of cases.
