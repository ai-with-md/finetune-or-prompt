"""The prompt must be byte-stable (or caching fails) and every answer must map to an intent."""

import json

import numpy as np
import pandas as pd
import pytest

from fopstudy import prompting
from fopstudy.costs import PRICES, request_cost

LABELS = ["card_arrival", "exchange_rate", "top_up_failed"]


def examples():
    return pd.DataFrame({"text": ["Where is my card?", "Top up  did not\nwork", "Rate today?",
                                  "Card still not here"],
                         "label": [0, 2, 1, 0]})


def test_system_prompt_lists_every_intent_and_groups_examples():
    zero = prompting.system_prompt(LABELS)
    assert all(label in zero for label in LABELS)
    assert "examples" not in zero
    few = prompting.system_prompt(LABELS, examples())
    assert few.startswith(zero)
    # Grouped by intent in label order, rows kept in order, whitespace flattened.
    assert few.index("- Where is my card?") < few.index("- Card still not here") < few.index("exchange_rate\n- Rate")
    assert "- Top up did not work" in few


def test_system_prompt_is_identical_every_time():
    a = prompting.system_prompt(LABELS, examples())
    b = prompting.system_prompt(LABELS, examples().copy())
    assert a == b
    assert prompting.prompt_hash(a) == prompting.prompt_hash(b)
    assert prompting.prompt_hash(a) != prompting.prompt_hash(prompting.system_prompt(LABELS))


def test_answer_schema_allows_only_the_intents():
    schema = prompting.answer_schema(LABELS)["schema"]
    assert schema["properties"]["intent"]["enum"] == LABELS
    assert schema["additionalProperties"] is False


@pytest.mark.parametrize("text, expected", [
    ('{"intent": "exchange_rate"}', 1),
    ('{"intent": "not_an_intent"}', -1),
    ("exchange_rate", -1),
    ("", -1),
    ('{"other": "exchange_rate"}', -1),
    ("[1, 2]", -1),
])
def test_parse_answer(text, expected):
    assert prompting.parse_answer(text, LABELS) == expected


def record(row, label, pred, model="haiku", read=0, write=0, latency=0.5):
    return {"row": row, "label": label, "pred": pred, "model_key": model, "input_tokens": 30,
            "output_tokens": 12, "cache_read_tokens": read, "cache_write_tokens": write,
            "latency_s": latency}


def test_spend_and_summary(tmp_path):
    recs = [record(0, 1, 1, write=5000, latency=1.0), record(1, 2, 2, read=5000), record(2, 0, -1, read=5000)]
    with open(tmp_path / "haiku_shots5.jsonl", "w") as f:
        f.writelines(json.dumps(r) + "\n" for r in recs)
    p = PRICES["haiku"]
    expected = (request_cost(p, 30, 12, cache_write_tokens=5000)
                + 2 * request_cost(p, 30, 12, cache_read_tokens=5000))
    assert prompting.total_spend(tmp_path) == pytest.approx(expected)
    s = prompting.summarise(prompting.read_records(tmp_path / "haiku_shots5.jsonl"))
    assert s["requests"] == 3
    assert s["accuracy"] == pytest.approx(2 / 3)
    assert s["invalid"] == 1
    assert s["cache_hit_share"] == pytest.approx(2 / 3)
    assert s["latency_p50_ms"] == pytest.approx(500)


def test_worst_case_is_the_uncached_price():
    assert prompting.worst_case_cost("sonnet", 6000, 10) == pytest.approx(
        10 * (6000 * 2.00 + prompting.MAX_TOKENS * 10.00) / 1e6)


def test_every_model_has_an_id_and_a_thinking_setting():
    assert set(prompting.MODEL_IDS) == set(prompting.THINKING) == set(PRICES)


def test_retrieved_examples_go_before_the_message():
    assert prompting.user_message("Hi  there") == "Customer message: Hi there"
    msg = prompting.user_message("Where is it?", [("Card  not here", "card_arrival"), ("Rate?", "exchange_rate")])
    assert msg == ('Labelled examples of similar messages:\n- "Card not here": card_arrival\n'
                   '- "Rate?": exchange_rate\n\nCustomer message: Where is it?')
    assert prompting.Config("sonnet", 0, 20, retrieved=10).name == "sonnet_retrieved10"
    assert prompting.Config("sonnet", 5, 20).name == "sonnet_shots5"


def test_nearest_examples_most_similar_last():
    pool = np.array([[1.0, 0.0], [0.0, 1.0], [0.6, 0.8], [0.8, 0.6]])
    top = prompting.nearest_examples(np.array([1.0, 0.0]), pool, k=3)
    assert top.tolist() == [[2, 3, 0]]
    assert prompting.nearest_examples(pool[:2], pool, k=1).tolist() == [[0], [1]]
