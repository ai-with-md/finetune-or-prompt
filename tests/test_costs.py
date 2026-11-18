"""Cost and break-even arithmetic, checked against sums done by hand."""

import math

import pytest

from fopstudy.costs import (PRICES, TokenPrice, api_cost_per_1k, break_even_volume,
                            error_cost_to_break_even, monthly_cost, request_cost, server_capacity,
                            server_monthly_cost)

HAIKU = PRICES["haiku"]
SONNET = PRICES["sonnet"]


def test_request_cost_by_hand():
    assert request_cost(HAIKU, 1_000_000, 0) == pytest.approx(0.10)
    # 650 input and 10 output tokens: (650 * 0.10 + 10 * 0.50) / 1e6
    assert request_cost(HAIKU, 650, 10) == pytest.approx(7e-5)


def test_batch_halves_the_price():
    assert request_cost(HAIKU, 650, 10, batch=True) == pytest.approx(3.5e-5)


def test_cached_prefix_is_charged_at_the_cache_read_price():
    # 9,000 cached + 30 fresh input + 10 output on Sonnet: (900 + 60 + 100) / 1e6 per request
    assert api_cost_per_1k(SONNET, 30, 10, cached_tokens=9000) == pytest.approx(1.06)
    assert request_cost(SONNET, 0, 0, cache_write_tokens=1_000_000) == pytest.approx(2.50)


def test_cache_reads_are_cheaper_than_fresh_input_for_every_model():
    for price in PRICES.values():
        assert price.cache_read < price.input < price.cache_write


def test_monthly_cost_adds_fixed_usage_and_errors():
    # 50 fixed + 200 requests at $1 per 1,000 + 200 * 10% wrong * $0.50
    assert monthly_cost(200, per_1k=1.0, fixed=50, error_rate=0.1, cost_per_error=0.5) == pytest.approx(60.2)
    with pytest.raises(ValueError):
        monthly_cost(-1)


def test_break_even_on_price_alone():
    # $50 a month of fixed cost against $0.07 per 1,000 hosted requests
    assert break_even_volume(0.07, 50) == pytest.approx(50 / 0.07e-3)


def test_break_even_counts_what_errors_cost():
    # saving per request: 0.001 in price + (10% - 5%) * $0.02 in errors = 0.002
    v = break_even_volume(1.0, 50, api_error_rate=0.10, selfhost_error_rate=0.05, cost_per_error=0.02)
    assert v == pytest.approx(25_000)


def test_bills_are_equal_at_the_break_even_volume():
    kwargs = dict(api_per_1k=1.3, fixed_monthly=120, selfhost_per_1k=0.05)
    v = break_even_volume(**kwargs, api_error_rate=0.12, selfhost_error_rate=0.07, cost_per_error=0.03)
    hosted = monthly_cost(v, per_1k=1.3, error_rate=0.12, cost_per_error=0.03)
    selfhost = monthly_cost(v, per_1k=0.05, fixed=120, error_rate=0.07, cost_per_error=0.03)
    assert hosted == pytest.approx(selfhost)


def test_never_breaks_even_when_self_hosting_saves_nothing():
    assert break_even_volume(0.0, 50) == math.inf
    assert break_even_volume(1.0, 50, api_error_rate=0.0, selfhost_error_rate=0.2, cost_per_error=1.0) == math.inf


def test_free_self_hosting_breaks_even_at_zero():
    assert break_even_volume(1.0, 0) == 0


def test_error_cost_to_break_even_by_hand_and_round_trip():
    # extra cost per request: 50 / 100,000 - 0.07 / 1,000 = 0.00043; divided by 8 fewer errors per 100
    c = error_cost_to_break_even(100_000, 0.07, 50, api_error_rate=0.15, selfhost_error_rate=0.07)
    assert c == pytest.approx(0.00043 / 0.08)
    v = break_even_volume(0.07, 50, api_error_rate=0.15, selfhost_error_rate=0.07, cost_per_error=c)
    assert v == pytest.approx(100_000)


def test_error_cost_edge_cases():
    assert error_cost_to_break_even(1e6, 1.0, 50, 0.1, 0.1) == 0.0  # already cheaper
    assert error_cost_to_break_even(10, 0.07, 50, 0.1, 0.1) == math.inf  # no fewer errors
    with pytest.raises(ValueError):
        error_cost_to_break_even(0, 0.07, 50, 0.1, 0.05)


def test_custom_price():
    p = TokenPrice(input=1.0, output=2.0, cache_write=1.25, cache_read=0.1)
    assert api_cost_per_1k(p, 1000, 1000) == pytest.approx(3.0)


def test_server_month_and_capacity():
    assert server_monthly_cost(0.10) == pytest.approx(73.0)
    # 10 ms a request is 100 a second: 730 hours * 3,600 seconds * 100
    assert server_capacity(10) == pytest.approx(262_800_000)
    assert server_capacity(10, slowdown=2, utilisation=0.5) == pytest.approx(65_700_000)
    with pytest.raises(ValueError):
        server_capacity(10, utilisation=0)
