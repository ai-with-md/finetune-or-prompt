"""Prompts, answer parsing and spend tracking for the hosted-model runs.

Nothing here calls the API, so it is tested without a key. scripts/run_api.py
does the calling and keeps one JSON line per answered message in results/api/,
which is the record of every request: the answer, the token counts and the time
it took. Predictions, costs and speeds are all computed from those lines.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Optional

import numpy as np
import pandas as pd

from fopstudy.costs import PRICES, request_cost
from fopstudy.results import RESULTS_DIR

API_DIR = RESULTS_DIR / "api"
MODEL_IDS = {"haiku": "claude-haiku-5-5", "sonnet": "claude-sonnet-5-5"}
# The lowest thinking setting each model accepts: no extended thinking.
THINKING = {"haiku": {"type": "disabled"}, "sonnet": {"type": "between_tools"}}
MAX_TOKENS = 100

INSTRUCTIONS = (
    "You sort messages that customers send to a bank's support team.\n"
    "Every message has exactly one intent from the list below. Choose the intent that best "
    "matches what the customer is asking about, and answer with its name exactly as written.\n\n"
    "Intents:\n"
)


@dataclass(frozen=True)
class Config:
    """One prompted run.

    Either a fixed set of `shots` labelled examples of each intent in the
    system prompt, or, with `retrieved`, that many labelled examples chosen for
    each message: the training messages most similar to it.
    """

    model: str
    shots: int
    test_per_class: Optional[int] = None  # None: the whole test set
    retrieved: int = 0

    @property
    def name(self) -> str:
        if self.retrieved:
            return f"{self.model}_retrieved{self.retrieved}"
        return f"{self.model}_shots{self.shots}"


def system_prompt(labels: list[str], examples: Optional[pd.DataFrame] = None) -> str:
    """The fixed part of every request: instructions, the intents and any examples.

    It is identical byte for byte across the requests of a run, so it is cached
    after the first one. `examples` has `text` and `label` columns and is grouped
    by intent in label order, keeping the order of rows within an intent.
    """
    prompt = INSTRUCTIONS + "\n".join(labels)
    if examples is not None and len(examples):
        blocks = []
        for label, group in examples.groupby("label", sort=True):
            lines = "\n".join(f"- {_one_line(t)}" for t in group["text"])
            blocks.append(f"{labels[label]}\n{lines}")
        prompt += "\n\nLabelled examples of each intent:\n\n" + "\n\n".join(blocks)
    return prompt


def _one_line(text: str) -> str:
    return " ".join(str(text).split())


def user_message(text: str, examples: Optional[list[tuple[str, str]]] = None) -> str:
    """The part of a request that changes every time: the message, after any retrieved examples.

    `examples` are (text, intent name) pairs, shown in the order given.
    """
    message = f"Customer message: {_one_line(text)}"
    if not examples:
        return message
    lines = "\n".join(f'- "{_one_line(t)}": {intent}' for t, intent in examples)
    return f"Labelled examples of similar messages:\n{lines}\n\n{message}"


def nearest_examples(query: np.ndarray, pool: np.ndarray, k: int) -> np.ndarray:
    """Positions in `pool` of the k rows most similar to each query row, most similar last.

    Rows are L2-normalised embeddings, so the dot product is the cosine similarity.
    The most similar example goes last, nearest the message it is meant to help with.
    """
    scores = np.atleast_2d(query) @ pool.T
    top = np.argsort(-scores, axis=1, kind="stable")[:, :k]
    return top[:, ::-1]


def answer_schema(labels: list[str]) -> dict:
    """Structured output that only allows one of the intent names."""
    return {"type": "json_schema", "schema": {
        "type": "object",
        "properties": {"intent": {"type": "string", "enum": list(labels)}},
        "required": ["intent"],
        "additionalProperties": False,
    }}


def prompt_hash(system: str) -> str:
    return hashlib.sha256(system.encode("utf-8")).hexdigest()[:16]


def parse_answer(text: str, labels: list[str]) -> int:
    """Label index of the answer, or -1 if it is not a valid intent."""
    try:
        intent = json.loads(text)["intent"]
    except (ValueError, TypeError, KeyError):
        return -1
    return labels.index(intent) if intent in labels else -1


def record_cost(record: dict) -> float:
    return request_cost(PRICES[record["model_key"]], record["input_tokens"], record["output_tokens"],
                        cache_read_tokens=record["cache_read_tokens"],
                        cache_write_tokens=record["cache_write_tokens"])


def read_records(path: Path) -> list[dict]:
    if not Path(path).exists():
        return []
    with open(path) as f:
        return [json.loads(line) for line in f if line.strip()]


def total_spend(api_dir: Path = API_DIR) -> float:
    """Dollars spent so far, from the token counts of every saved answer."""
    return sum(record_cost(r) for p in sorted(Path(api_dir).glob("*.jsonl")) for r in read_records(p))


def worst_case_cost(model: str, prompt_tokens: int, n_requests: int) -> float:
    """Cost if no request were served from the cache: the guard for the spending cap."""
    return n_requests * request_cost(PRICES[model], prompt_tokens, MAX_TOKENS)


def summarise(records: Iterable[dict]) -> dict:
    """Accuracy, invalid answers, tokens, cost and speed of one run."""
    df = pd.DataFrame(list(records))
    df["cost"] = [record_cost(r) for r in df.to_dict("records")]
    lat = df["latency_s"].to_numpy() * 1000
    return {
        "requests": len(df),
        "accuracy": float((df["pred"] == df["label"]).mean()),
        "invalid": int((df["pred"] < 0).sum()),
        "cache_hit_share": float((df["cache_read_tokens"] > 0).mean()),
        "input_tokens_mean": float(df["input_tokens"].mean()),
        "cache_read_tokens_mean": float(df["cache_read_tokens"].mean()),
        "output_tokens_mean": float(df["output_tokens"].mean()),
        "cost_total": float(df["cost"].sum()),
        "cost_per_1k": float(1000 * df["cost"].mean()),
        "latency_p50_ms": float(np.median(lat)),
        "latency_p95_ms": float(np.percentile(lat, 95)),
    }
