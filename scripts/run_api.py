"""Ask Claude Haiku 5.5 and Sonnet 5.5 for the intent of each test message.

Six runs: each model with no examples, 1 and 5 labelled examples per intent.
Haiku answers all 3,080 test messages; Sonnet answers 20 per intent (1,540) to
keep the cost down. The 5 examples per intent are the same 385 messages the
fine-tuned and frozen models were trained on at that size, so all approaches
learn from the same labels.

Two more runs (--retrieved) use all 9,000 training labels the way a prompt
can: for each message, the 10 labelled training messages most similar to it
(by the frozen bge-small embeddings) go into the request. Both models answer
the same 1,540 messages. These runs have their own budget, RETRIEVED_BUDGET.

Every answer is appended to results/api/<run>.jsonl as soon as it arrives, so a
run that stops resumes where it left off and no message is paid for twice. The
prompt is fixed per run; if it changes, the script refuses to mix old and new
answers. Spending stops before CAP_DOLLARS, counting the worst case (no cache
hits) for every request about to be sent.

    .venv-train/bin/python scripts/run_api.py --dry-run   # plan and cost, no paid calls
    .venv-train/bin/python scripts/run_api.py --pilot     # 50 Sonnet messages, checks caching
    .venv-train/bin/python scripts/run_api.py             # the six runs
    .venv-train/bin/python scripts/run_api.py --retrieved --dry-run   # the same, for the two retrieved runs

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
from fopstudy.data import DATA_DIR, load_labels, load_split, per_class_sample, validation_split
from fopstudy.prompting import (API_DIR, MAX_TOKENS, MODEL_IDS, THINKING, Config, answer_schema,
                                nearest_examples, parse_answer, prompt_hash, read_records, record_cost,
                                summarise, system_prompt, total_spend, user_message, worst_case_cost)
from fopstudy.results import RESULTS_DIR, save_predictions

CAP_DOLLARS = 15.0
CONFIGS = [Config("haiku", 0), Config("haiku", 1), Config("haiku", 5),
           Config("sonnet", 0, 20), Config("sonnet", 1, 20), Config("sonnet", 5, 20)]
PILOT_CONFIG, PILOT_SIZE = Config("sonnet", 5, 20), 50
RETRIEVED = 10
RETRIEVED_CONFIGS = [Config("haiku", 0, 20, retrieved=RETRIEVED), Config("sonnet", 0, 20, retrieved=RETRIEVED)]
RETRIEVED_BUDGET = 2.50  # dollars for the two retrieved runs together, on top of the overall cap
RETRIEVED_PILOT = 20
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


def plan(labels, fit, train, test, configs):
    """The prompt and the test rows of every run, and each message's examples for retrieved runs.

    Rows are asked in a fixed shuffled order (the test set is sorted by intent),
    so the pilot and any run that stops part-way cover all intents.
    """
    runs = {}
    for c in configs:
        examples = per_class_sample(fit, c.shots, seed=0) if c.shots else None
        subset = per_class_sample(test, c.test_per_class, seed=0) if c.test_per_class else test
        rows = np.random.default_rng(0).permutation(sorted(subset.index.tolist())).tolist()
        runs[c] = {"system": system_prompt(labels, examples), "rows": rows, "examples": {}}
        if c.retrieved:
            runs[c]["examples"] = retrieved_examples(labels, fit, train, test, rows, c.retrieved)
    return runs


def retrieved_examples(labels, fit, train, test, rows, k):
    """For each test row, its k most similar training messages (from the training part only)."""
    paths = [DATA_DIR / f"emb_bge-small_{split}.npy" for split in ("train", "test")]
    if not all(p.exists() for p in paths):
        sys.exit("Embeddings not found. Run scripts/frozen_baseline.py first.")
    train_emb, test_emb = (np.load(p) for p in paths)
    pool = train.index.get_indexer(fit.index)  # positions of the 9,000 training messages
    top = nearest_examples(test_emb[test.index.get_indexer(rows)], train_emb[pool], k)
    return {row: [(fit["text"].iat[j], labels[int(fit["label"].iat[j])]) for j in idx]
            for row, idx in zip(rows, top)}


def request_params(c: Config, system: str, text: str, labels, examples=None) -> dict:
    return dict(
        model=MODEL_IDS[c.model],
        max_tokens=MAX_TOKENS,
        thinking=THINKING[c.model],
        system=[{"type": "text", "text": system, "cache_control": {"type": "ephemeral"}}],
        messages=[{"role": "user", "content": user_message(text, examples)}],
        output_config={"format": answer_schema(labels)},
    )


def count_prompt_tokens(client, c: Config, system: str, labels, examples=None) -> int:
    """Input tokens of one request, from the free token-counting endpoint."""
    params = request_params(c, system, "How do I check my balance?", labels, examples)
    params.pop("max_tokens")
    try:
        return client.messages.count_tokens(**params).input_tokens
    except Exception as e:  # counting is only for the estimate and the cap; fall back safely
        print(f"{c.name}: token counting failed ({type(e).__name__}); using a rough estimate", flush=True)
        return int(estimate_tokens(c, system, examples) * 1.5)


def estimate_tokens(c: Config, system: str, examples=None) -> int:
    """Rough count without an API key: about 4 characters a token, 30% more on Haiku 5.5."""
    tokens = (len(system) + len(user_message("", examples)) + 300) / 4
    return int(tokens * (1.3 if c.model == "haiku" else 1.0))


def expected_cost(c: Config, prompt_tokens: int, n: int, output_tokens: int = 15, uncached: int = 30) -> float:
    """One cache write, then cache reads; the user turn (`uncached` tokens) is never cached."""
    p = PRICES[c.model]
    cached = max(prompt_tokens - uncached, 0)
    first = request_cost(p, uncached, output_tokens, cache_write_tokens=cached)
    rest = request_cost(p, uncached, output_tokens, cache_read_tokens=cached)
    return first + max(n - 1, 0) * rest


def token_counts(client, c: Config, spec: dict, labels) -> dict:
    """Prompt tokens per request (the longest, for the cap) and how many of them are not cached."""
    count = (lambda ex=None: count_prompt_tokens(client, c, spec["system"], labels, ex)) if client else \
        (lambda ex=None: estimate_tokens(c, spec["system"], ex))
    base = count()
    if not spec["examples"]:
        return {"prompt": base, "uncached": 30}
    by_length = sorted(spec["examples"].values(), key=lambda ex: sum(len(t) for t, _ in ex))
    typical, longest = by_length[len(by_length) // 2], by_length[-1]
    return {"prompt": count(longest), "uncached": count(typical) - base + 30}


def run_spend(c: Config) -> float:
    return sum(record_cost(r) for r in read_records(API_DIR / f"{c.name}.jsonl"))


def ask(client, c: Config, system: str, phash: str, row: int, text: str, label: int, labels,
        examples=None) -> dict:
    start = time.perf_counter()
    resp = client.messages.create(**request_params(c, system, text, labels, examples))
    latency = time.perf_counter() - start
    answer = next((b.text for b in resp.content if b.type == "text"), "")
    u = resp.usage
    return {"row": row, "label": label, "pred": parse_answer(answer, labels), "answer": answer,
            "stop_reason": resp.stop_reason, "model_key": c.model, "model": resp.model,
            "input_tokens": u.input_tokens, "output_tokens": u.output_tokens,
            "cache_read_tokens": u.cache_read_input_tokens or 0,
            "cache_write_tokens": u.cache_creation_input_tokens or 0,
            "latency_s": round(latency, 4), "request_id": resp._request_id, "prompt_hash": phash}


def run(client, c: Config, spec: dict, test, labels, prompt_tokens: int, limit=None, budget=None) -> bool:
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
        return ask(client, c, system, phash, row, test.at[row, "text"], int(test.at[row, "label"]), labels,
                   spec["examples"].get(row))

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
            if budget is not None and budget["spent"]() + worst > budget["dollars"]:
                print(f"Stopping: these runs have spent ${budget['spent']():.2f}, the next {len(chunk)} requests "
                      f"could cost up to ${worst:.2f}, budget ${budget['dollars']:.2f}", flush=True)
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
        approach, size = (c.name, "all") if c.retrieved else (c.model, c.shots)
        save_predictions(approach, size, 0, [r["label"] for r in records],
                         [r["pred"] for r in records], rows=[r["row"] for r in records])
    return complete


def write_summary(runs):
    rows = []
    for c in runs:
        records = read_records(API_DIR / f"{c.name}.jsonl")
        if len(records) == len(runs[c]["rows"]):
            rows.append({"run": c.name, "model": MODEL_IDS[c.model], "shots": c.shots, "retrieved": c.retrieved,
                         **summarise(records)})
    if rows:
        pd.DataFrame(rows).round(6).to_csv(RESULTS_DIR / "runs_api.csv", index=False)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true", help="show the plan and cost; no paid calls")
    parser.add_argument("--pilot", action="store_true", help=f"answer {PILOT_SIZE} Sonnet messages and check caching")
    parser.add_argument("--retrieved", action="store_true", help="the two runs with retrieved examples")
    args = parser.parse_args()

    load_env()
    labels = load_labels()
    train, test = load_split("train"), load_split("test")
    fit, _ = validation_split(train, fraction=0.1, seed=0)
    configs = RETRIEVED_CONFIGS if args.retrieved else CONFIGS
    runs = plan(labels, fit, train, test, configs)
    budget = {"dollars": RETRIEVED_BUDGET, "spent": lambda: sum(run_spend(k) for k in RETRIEVED_CONFIGS)} \
        if args.retrieved else None

    client = None
    if os.environ.get("ANTHROPIC_API_KEY"):
        import anthropic
        client = anthropic.Anthropic(max_retries=5)
    elif not args.dry_run:
        sys.exit("No ANTHROPIC_API_KEY found. Add it to .env in the repo root.")

    counts = {c: token_counts(client, c, spec, labels) for c, spec in runs.items()}
    tokens = {c: n["prompt"] for c, n in counts.items()}

    spent = total_spend()
    if args.dry_run:
        source = "counted by the API" if client else "estimated (no key yet)"
        print(f"Prompt tokens {source}. Spent so far ${spent:.2f}, cap ${CAP_DOLLARS:.2f}.\n")
        total = total_worst = 0.0
        for c, spec in runs.items():
            left = len(spec["rows"]) - len(read_records(API_DIR / f"{c.name}.jsonl"))
            exp = expected_cost(c, tokens[c], left, output_tokens=17, uncached=counts[c]["uncached"])
            worst = worst_case_cost(c.model, tokens[c], left)
            total, total_worst = total + exp, total_worst + worst
            note = "  (near the 512-token cache minimum)" if tokens[c] < 700 else ""
            print(f"{c.name:18s} {left:5d} requests  {tokens[c]:6d} prompt tokens (longest), "
                  f"{counts[c]['uncached']:4d} not cached (typical)  expected ${exp:6.2f}  "
                  f"without caching ${worst:6.2f}{note}")
        stop = f"the budget stops these runs at ${RETRIEVED_BUDGET:.2f}" if args.retrieved else \
            f"the cap stops it at ${CAP_DOLLARS:.2f}"
        print(f"\nExpected total ${total:.2f}; without caching ${total_worst:.2f} ({stop}).")
        return

    if args.pilot and args.retrieved:
        c = RETRIEVED_CONFIGS[1]
        run(client, c, runs[c], test, labels, tokens[c], limit=RETRIEVED_PILOT, budget=budget)
        records = read_records(API_DIR / f"{c.name}.jsonl")[:RETRIEVED_PILOT]
        s = summarise(records)
        checks = {
            "every answer is a valid intent": s["invalid"] == 0,
            "every answer finished normally": all(r["stop_reason"] == "end_turn" for r in records),
            "requests after the first read the fixed prompt from the cache":
                all(r["cache_read_tokens"] > 0 for r in records[1:]),
        }
        print(f"\nPilot: {s['requests']} messages, accuracy {s['accuracy']:.1%}, cost ${s['cost_total']:.3f}, "
              f"{s['input_tokens_mean']:.0f} uncached and {s['cache_read_tokens_mean']:.0f} cached input tokens")
        for name, ok in checks.items():
            print(f"  {'PASS' if ok else 'FAIL'}  {name}")
        per = s["cost_total"] / s["requests"]
        left = sum(len(runs[k]["rows"]) - len(read_records(API_DIR / f"{k.name}.jsonl")) for k in runs)
        print(f"Sonnet at ${per * 1000:.2f} per 1,000; these runs so far ${budget['spent']():.2f} of "
              f"${RETRIEVED_BUDGET:.2f}; {left} requests left.")
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
        if not run(client, c, spec, test, labels, tokens[c], budget=budget):
            break
        print(f"{c.name}: done, {summarise(read_records(API_DIR / f'{c.name}.jsonl'))}", flush=True)
    write_summary(plan(labels, fit, train, test, CONFIGS + RETRIEVED_CONFIGS))
    print(f"Spent in total ${total_spend():.2f}", flush=True)


if __name__ == "__main__":
    main()
