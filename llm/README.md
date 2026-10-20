# LLMs on I-RAVEN

Benchmarks multimodal LLMs on I-RAVEN with the method of Tehlar (CTU bachelor thesis, 2025), rewritten
from scratch after an audit of his code (`TEHLAR_AUDIT.md`), plus a text input with XML labels.

- **End to end (images):** one PNG per puzzle (3x3 grid, missing cell marked with a cross, candidates
  A-H below), Tehlar's prompt (`--prompt tehlar`, byte-identical to what his code sent).
- **With XML labels (text):** `--prompt xml_text`; panels written out as text from the `.xml` via the
  same `parse`/`POS_MAP` as `pynl/train_pynl.py` (a test checks they stay identical).

The answer is the letter after the last "The correct answer is:". Unparseable answers count as wrong.
The ground truth is the npz `target`.

| file | role |
|---|---|
| `rpm_llm/data.py` | finds and samples npz puzzles, manifests |
| `rpm_llm/render.py` | puzzle → PNG |
| `rpm_llm/xml_labels.py` | XML → text |
| `rpm_llm/prompts.py` | `tehlar`, `tehlar_clean`, `xml_text` |
| `rpm_llm/providers.py` | `anthropic`; `openai`, also any OpenAI-compatible server (vLLM/Ollama for Gemma and Qwen, Mistral) |
| `rpm_llm/run.py` | runs a model, appends JSONL, resumes and retries failed calls |
| `rpm_llm/score.py` | accuracy per run and pooled (95% CI), per constellation, unparseable rate, chance and majority-letter baselines |
| `rpm_llm/import_tehlar.py` | re-scores Tehlar's saved Pixtral outputs (`results/tehlar_pixtral_rescored.md`) |

```sh
cd llm
pip install numpy pillow anthropic openai pytest
python -m pytest -q tests

python -m rpm_llm.run --data ../data/I-RAVEN-10000 --splits test --per-config 50 --seed 0 --write-manifest manifests/iraven_test_350.jsonl
python -m rpm_llm.run --manifest manifests/iraven_test_350.jsonl --provider anthropic --model claude-opus-5-5 --effort high --runs 3 --out results/claude-opus-5-5.jsonl
python -m rpm_llm.run --manifest manifests/iraven_test_350.jsonl --provider anthropic --model claude-opus-5-5 --effort high --prompt xml_text --runs 3 --out results/claude-opus-5-5_xml.jsonl
python -m rpm_llm.run --manifest manifests/iraven_test_350.jsonl --provider openai --model gpt-5 --effort high --runs 3 --out results/gpt-5.jsonl
python -m rpm_llm.run --manifest manifests/iraven_test_350.jsonl --provider openai --base-url http://localhost:8000/v1 --model Qwen/Qwen2.5-VL-7B-Instruct --max-tokens 4096 --temperature 0 --runs 3 --out results/qwen2.5-vl-7b.jsonl
python -m rpm_llm.score results/*.jsonl --md results/report.md
```
Keys come from `ANTHROPIC_API_KEY` / `OPENAI_API_KEY` (or `--api-key-env NAME`). Claude 4.7+ models
reject `--temperature`. Tehlar's own 224 puzzles: clone github.com/JakubTehlar/thesis_paper and use
`--data <clone>/Datasets --splits train,val,test`.
