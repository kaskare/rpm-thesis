"""Runs a model on a fixed puzzle set and appends one JSON line per (run, puzzle).
Re-running the same command resumes: finished items are skipped, failed ones are retried.

  python -m rpm_llm.run --manifest manifests/tehlar_224.jsonl --provider anthropic \
      --model claude-opus-5-5 --runs 3 --out results/claude-opus-5-5.jsonl
"""
import argparse
import hashlib
import json
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from .data import load_manifest, sample, save_manifest
from .parse import parse_answer
from .prompts import PROMPTS, TEXT_PROMPTS
from .providers import PROVIDERS
from .render import render_png
from .xml_labels import describe


def done_keys(path):
    keys = set()
    if Path(path).exists():
        with open(path) as f:
            for line in f:
                r = json.loads(line)
                if r.get("status") == "ok":
                    keys.add((r["run"], r["id"]))
    return keys


def main(argv=None):
    ap = argparse.ArgumentParser()
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--manifest", help="JSONL puzzle list (from --data ... --write-manifest)")
    src.add_argument("--data", help="root folder with I-RAVEN .npz files")
    ap.add_argument("--splits", default="test", help="comma separated: train,val,test")
    ap.add_argument("--per-config", type=int, help="puzzles per (variant, configuration); default all")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--write-manifest", help="save the sampled puzzle list here and exit")
    ap.add_argument("--provider", choices=PROVIDERS)
    ap.add_argument("--model")
    ap.add_argument("--base-url")
    ap.add_argument("--api-key-env", default="OPENAI_API_KEY")
    ap.add_argument("--prompt", choices=PROMPTS, default="tehlar")
    ap.add_argument("--runs", type=int, default=1)
    ap.add_argument("--max-tokens", type=int, default=32000)
    ap.add_argument("--effort", help="Anthropic output_config.effort / OpenAI reasoning_effort")
    ap.add_argument("--temperature", type=float)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--limit", type=int, help="first N puzzles only (smoke test)")
    ap.add_argument("--out")
    a = ap.parse_args(argv)

    puzzles = (load_manifest(a.manifest) if a.manifest
               else sample(a.data, tuple(a.splits.split(",")), a.per_config, a.seed))
    if a.write_manifest:
        save_manifest(puzzles, a.write_manifest)
        print(f"{len(puzzles)} puzzles -> {a.write_manifest}")
        return
    if not (a.provider and a.model and a.out):
        ap.error("--provider, --model and --out are required to run")
    if a.limit:
        puzzles = puzzles[: a.limit]

    prompt = PROMPTS[a.prompt]
    prompt_sha = hashlib.sha256(prompt.encode()).hexdigest()[:12]
    call = PROVIDERS[a.provider](a.model, max_tokens=a.max_tokens, effort=a.effort,
                                 temperature=a.temperature, base_url=a.base_url, api_key_env=a.api_key_env)
    skip = done_keys(a.out)
    jobs = [(r, p) for r in range(1, a.runs + 1) for p in puzzles if (r, p.id) not in skip]
    print(f"{len(puzzles)} puzzles x {a.runs} runs, {len(jobs)} to do", file=sys.stderr)
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    lock = threading.Lock()
    meta = dict(provider=a.provider, model=a.model, prompt=a.prompt,
                input="xml" if a.prompt in TEXT_PROMPTS else "image", prompt_sha=prompt_sha,
                effort=a.effort, temperature=a.temperature, max_tokens=a.max_tokens, base_url=a.base_url)

    def work(run, p):
        rec = dict(meta, run=run, id=p.id, variant=p.variant, config=p.config, split=p.split, target=p.answer)
        t0 = time.time()
        try:
            if a.prompt in TEXT_PROMPTS:
                rep = call(prompt.format(puzzle=describe(p.path[:-4] + ".xml", p.config)), None)
            else:
                rep = call(prompt, render_png(p.path))
            pred = parse_answer(rep.text)
            rec.update(status="ok", prediction=pred, lenient=parse_answer(rep.text, lenient=True),
                       correct=pred == p.answer, response=rep.text, stop_reason=rep.stop_reason,
                       served_model=rep.model, usage=rep.usage)
        except Exception as e:
            rec.update(status="error", error=f"{type(e).__name__}: {e}")
        rec["seconds"] = round(time.time() - t0, 2)
        with lock, open(a.out, "a") as f:
            f.write(json.dumps(rec) + "\n")
        return rec

    n_ok = n_correct = n_err = 0
    with ThreadPoolExecutor(a.workers) as ex:
        futs = [ex.submit(work, r, p) for r, p in jobs]
        for i, fu in enumerate(as_completed(futs), 1):
            rec = fu.result()
            if rec["status"] == "ok":
                n_ok += 1
                n_correct += rec["correct"]
            else:
                n_err += 1
                print(f"error {rec['id']} run {rec['run']}: {rec['error']}", file=sys.stderr)
            if i % 20 == 0 or i == len(futs):
                print(f"{i}/{len(futs)}  acc {n_correct}/{n_ok}  errors {n_err}", file=sys.stderr)
    if n_err:
        print(f"{n_err} calls failed; re-run the same command to retry them", file=sys.stderr)


if __name__ == "__main__":
    main()
