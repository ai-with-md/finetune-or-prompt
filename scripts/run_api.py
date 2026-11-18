"""Ask Claude Haiku 5.5 and Sonnet 5.5 for the intent of each test message.

Six runs: each model with no examples, 1 and 5 labelled examples per intent.
Haiku answers all 3,080 test messages; Sonnet answers 20 per intent (1,540) to
keep the cost down. The 5 examples per intent are the same 385 messages the
fine-tuned and frozen models were trained on at that size, so all approaches
learn from the same labels.

Every answer is appended to results/api/<run>.jsonl as soon as it arrives, so a
run that stops resumes where it left off and no message is paid for twice. The
prompt is fixed per run; if it changes, the script refuses to mix old and new
answers. Spending stops before CAP_DOLLARS, counting the worst case (no cache
hits) for every request about to be sent.

    .venv-train/bin/python scripts/run_api.py --dry-run   # plan and cost, no paid calls
    .venv-train/bin/python scripts/run_api.py --pilot     # 50 Sonnet messages, checks caching
    .venv-train/bin/python scripts/run_api.py             # every run

Needs ANTHROPIC_API_KEY in the environment or in a .env file in the repo root.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import pandas as pd

from fopstudy.costs import PRICES, request_cost
from fopstudy.data import load_labels, load_split, per_class_sample, validation_split
from fopstudy.prompting import (API_DIR, MAX_TOKENS, MODEL_IDS, THINKING, Config, answer_schema,
                                parse_answer, prompt_hash, read_records, summarise, system_prompt,
                                total_spend, user_message, worst_case_cost)
from fopstudy.results import RESULTS_DIR, save_predictions

CAP_DOLLARS = 15.0
CONFIGS = [Config("haiku", 0), Config("haiku", 1), Config("haiku", 5),
           Config("sonnet", 0, 20), Config("sonnet", 1, 20), Config("sonnet", 5, 20)]
PILOT_CONFIG, PILOT_SIZE = Config("sonnet", 5, 20), 50
WORKERS = 4          # requests in flight at once, after the first has written the cache
CHUNK = 40           # the spending cap is checked before every chunk
MAX_ERRORS = 20
ROOT = Path(__file__).resolve().parent.parent


def load_env():
    """Read KEY=value lines from .env without printing anything."""
    path = ROOT / ".env"
    if not path.exists():
        return
    for line in path.read_text().splitlines():
        line = line.strip()
        if line and not line.startswith("#") and "=" in line:
            key, value = line.split("=", 1)
            os.environ.setdefault(key.strip(), value.strip().strip('"').strip("'"))


def plan(labels, fit, test):
    """The prompt and the test rows of every run.

    Rows are asked in a fixed shuffled order (the test set is sorted by intent),
    so the pilot and any run that stops part-way cover all intents.
    """
    runs = {}
    for c in CONFIGS:
        examples = per_class_sample(fit, c.shots, seed=0) if c.shots else None
        subset = per_class_sample(test, c.test_per_class, seed=0) if c.test_per_class else test
        rows = np.random.default_rng(0).permutation(sorted(subset.index.tolist())).tolist()
        runs[c] = {"system": system_prompt(labels, examples), "rows": rows}
    return runs


def request_params(c: Config, system: str, text: str, labels) -> dict:
    return dict(
        model=MODEL_IDS[c.model],
        max_tokens=MAX_TOKENS,
        thinking=THINKING[c.model],
        system=[{"type": "text", "text": system, "cache_control": {"type": "ephemeral"}}],
        messages=[{"role": "user", "content": user_message(text)}],
        output_config={"format": answer_schema(labels)},
    )


def count_prompt_tokens(client, c: Config, system: str, labels) -> int:
    """Input tokens of one request, from the free token-counting endpoint."""
    params = request_params(c, system, "How do I check my balance?", labels)
    params.pop("max_tokens")
    try:
        return client.messages.count_tokens(**params).input_tokens
    except Exception as e:  # counting is only for the estimate and the cap; fall back safely
        print(f"{c.name}: token counting failed ({type(e).__name__}); using a rough estimate", flush=True)
        return int(estimate_tokens(c, system) * 1.5)


def estimate_tokens(c: Config, system: str) -> int:
    """Rough count without an API key: about 4 characters a token, 30% more on Haiku 5.5."""
    tokens = (len(system) + 300) / 4
    return int(tokens * (1.3 if c.model == "haiku" else 1.0))


def expected_cost(c: Config, prompt_tokens: int, n: int, output_tokens: int = 15) -> float:
    """One cache write, then cache reads; the customer message (about 30 tokens) is never cached."""
    p = PRICES[c.model]
    cached = max(prompt_tokens - 30, 0)
    first = request_cost(p, 30, output_tokens, cache_write_tokens=cached)
    rest = request_cost(p, 30, output_tokens, cache_read_tokens=cached)
    return first + max(n - 1, 0) * rest


def ask(client, c: Config, system: str, phash: str, row: int, text: str, label: int, labels) -> dict:
    start = time.perf_counter()
    resp = client.messages.create(**request_params(c, system, text, labels))
    latency = time.perf_counter() - start
    answer = next((b.text for b in resp.content if b.type == "text"), "")
    u = resp.usage
    return {"row": row, "label": label, "pred": parse_answer(answer, labels), "answer": answer,
            "stop_reason": resp.stop_reason, "model_key": c.model, "model": resp.model,
            "input_tokens": u.input_tokens, "output_tokens": u.output_tokens,
            "cache_read_tokens": u.cache_read_input_tokens or 0,
            "cache_write_tokens": u.cache_creation_input_tokens or 0,
            "latency_s": round(latency, 4), "request_id": resp._request_id, "prompt_hash": phash}


def run(client, c: Config, spec: dict, test, labels, prompt_tokens: int, limit=None) -> bool:
    """Answer the remaining rows of one run. Returns True if the run is complete."""
    import anthropic

    path = API_DIR / f"{c.name}.jsonl"
    path.parent.mkdir(parents=True, exist_ok=True)
    system, phash = spec["system"], prompt_hash(spec["system"])
    prompt_file = API_DIR / "prompts" / f"{c.name}.txt"
    prompt_file.parent.mkdir(parents=True, exist_ok=True)
    prompt_file.write_text(system)

    done = read_records(path)
    if any(r["prompt_hash"] != phash for r in done):
        sys.exit(f"{c.name}: saved answers used a different prompt. Move {path} aside before rerunning.")
    done_rows = {r["row"] for r in done}
    todo = [r for r in spec["rows"] if r not in done_rows]
    if limit is not None:
        todo = todo[:max(limit - len(done_rows), 0)]
    print(f"{c.name}: {len(done_rows)} saved, {len(todo)} to ask", flush=True)

    lock = threading.Lock()
    errors = 0
    fatal = (anthropic.AuthenticationError, anthropic.PermissionDeniedError,
             anthropic.BadRequestError, anthropic.NotFoundError)

    def save(record):
        with lock, open(path, "a") as f:
            f.write(json.dumps(record) + "\n")

    def one(row):
        return ask(client, c, system, phash, row, test.at[row, "text"], int(test.at[row, "label"]), labels)

    # The first request alone, so it writes the cache before the others read it.
    chunks = [todo[:1]] + [todo[i:i + CHUNK] for i in range(1, len(todo), CHUNK)] if todo else []
    with ThreadPoolExecutor(WORKERS) as pool:
        for chunk in chunks:
            spent = total_spend()
            worst = worst_case_cost(c.model, prompt_tokens, len(chunk))
            if spent + worst > CAP_DOLLARS:
                print(f"Stopping: ${spent:.2f} spent, the next {len(chunk)} requests could cost up to "
                      f"${worst:.2f}, cap ${CAP_DOLLARS:.2f}", flush=True)
                return False
            futures = {pool.submit(one, row): row for row in chunk}
            for fut in as_completed(futures):
                try:
                    save(fut.result())
                except fatal:
                    raise
                except anthropic.APIError as e:
                    errors += 1
                    print(f"  row {futures[fut]}: {type(e).__name__}, will retry on the next run", flush=True)
                    if errors > MAX_ERRORS:
                        sys.exit("Too many API errors; stopping. Rerun later to resume.")
    records = read_records(path)
    complete = {r["row"] for r in records} >= set(spec["rows"])
    if complete:
        records = sorted(records, key=lambda r: r["row"])
        save_predictions(c.model, c.shots, 0, [r["label"] for r in records],
                         [r["pred"] for r in records], rows=[r["row"] for r in records])
    return complete


def write_summary(runs):
    rows = []
    for c in runs:
        records = read_records(API_DIR / f"{c.name}.jsonl")
        if len(records) == len(runs[c]["rows"]):
            rows.append({"run": c.name, "model": MODEL_IDS[c.model], "shots": c.shots, **summarise(records)})
    if rows:
        pd.DataFrame(rows).round(6).to_csv(RESULTS_DIR / "runs_api.csv", index=False)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true", help="show the plan and cost; no paid calls")
    parser.add_argument("--pilot", action="store_true", help=f"answer {PILOT_SIZE} Sonnet messages and check caching")
    args = parser.parse_args()

    load_env()
    labels = load_labels()
    train, test = load_split("train"), load_split("test")
    fit, _ = validation_split(train, fraction=0.1, seed=0)
    runs = plan(labels, fit, test)

    client = None
    if os.environ.get("ANTHROPIC_API_KEY"):
        import anthropic
        client = anthropic.Anthropic(max_retries=5)
    elif not args.dry_run:
        sys.exit("No ANTHROPIC_API_KEY found. Add it to .env in the repo root.")

    tokens = {}
    for c, spec in runs.items():
        tokens[c] = count_prompt_tokens(client, c, spec["system"], labels) if client else estimate_tokens(c, spec["system"])

    spent = total_spend()
    if args.dry_run:
        source = "counted by the API" if client else "estimated (no key yet)"
        print(f"Prompt tokens {source}. Spent so far ${spent:.2f}, cap ${CAP_DOLLARS:.2f}.\n")
        total = total_worst = 0.0
        for c, spec in runs.items():
            left = len(spec["rows"]) - len(read_records(API_DIR / f"{c.name}.jsonl"))
            exp, worst = expected_cost(c, tokens[c], left), worst_case_cost(c.model, tokens[c], left)
            total, total_worst = total + exp, total_worst + worst
            note = "  (near the 512-token cache minimum)" if tokens[c] < 700 else ""
            print(f"{c.name:14s} {left:5d} requests  {tokens[c]:6d} prompt tokens  "
                  f"expected ${exp:6.2f}  without caching ${worst:6.2f}{note}")
        print(f"\nExpected total ${total:.2f}; without caching ${total_worst:.2f} "
              f"(the cap stops it at ${CAP_DOLLARS:.2f}).")
        return

    if args.pilot:
        c = PILOT_CONFIG
        run(client, c, runs[c], test, labels, tokens[c], limit=PILOT_SIZE)
        records = read_records(API_DIR / f"{c.name}.jsonl")[:PILOT_SIZE]
        s = summarise(records)
        later = records[1:]
        checks = {
            "every answer is a valid intent": s["invalid"] == 0,
            "every answer finished normally": all(r["stop_reason"] == "end_turn" for r in records),
            "requests after the first read the prompt from the cache":
                all(r["cache_read_tokens"] >= 0.9 * tokens[c] for r in later),
        }
        print(f"\nPilot: {s['requests']} messages, accuracy {s['accuracy']:.1%}, "
              f"cost ${s['cost_total']:.3f}, median {s['latency_p50_ms']:.0f} ms, "
              f"{s['output_tokens_mean']:.0f} output tokens per answer")
        for name, ok in checks.items():
            print(f"  {'PASS' if ok else 'FAIL'}  {name}")
        out = s["output_tokens_mean"]
        rest = sum(expected_cost(k, tokens[k], len(runs[k]["rows"]) - len(read_records(API_DIR / f"{k.name}.jsonl")),
                                 output_tokens=round(out)) for k in runs)
        print(f"Spent so far ${total_spend():.2f}; the remaining runs should cost about ${rest:.2f}.")
        return

    for c, spec in runs.items():
        if not run(client, c, spec, test, labels, tokens[c]):
            break
        print(f"{c.name}: done, {summarise(read_records(API_DIR / f'{c.name}.jsonl'))}", flush=True)
    write_summary(runs)
    print(f"Spent in total ${total_spend():.2f}", flush=True)


if __name__ == "__main__":
    main()
