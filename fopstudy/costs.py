"""Cost per 1,000 requests and the break-even monthly volume.

The decision is a comparison of two monthly bills:

    hosted(V)      = V / 1000 * (price per 1,000 requests + errors per 1,000 * cost per error)
    self-hosted(V) = fixed monthly cost + V / 1000 * (same, for the small model)

The hosted model costs nothing when idle and a little per request. The
self-hosted model costs a fixed amount every month (a server, upkeep, and the
one-time labelling and training spread over some months) and almost nothing per
request. The break-even volume is where the two bills cross.
"""

from __future__ import annotations

import math
from dataclasses import dataclass


@dataclass(frozen=True)
class TokenPrice:
    """Prices in US dollars per million tokens."""

    input: float
    output: float
    cache_write: float
    cache_read: float


# Anthropic's published API prices, checked on this date. Prices change: recheck before reuse.
PRICES_CHECKED = "2026-11-15"
PRICES = {
    "haiku": TokenPrice(input=0.10, output=0.50, cache_write=0.125, cache_read=0.01),
    "sonnet": TokenPrice(input=2.00, output=10.00, cache_write=2.50, cache_read=0.10),
}
# The Batch API halves every token price, and stacks with caching.
BATCH_DISCOUNT = 0.5


def request_cost(price: TokenPrice, input_tokens: int, output_tokens: int,
                 cache_read_tokens: int = 0, cache_write_tokens: int = 0,
                 batch: bool = False) -> float:
    """Dollar cost of one request. input_tokens counts only the uncached input."""
    cost = (input_tokens * price.input + output_tokens * price.output
            + cache_read_tokens * price.cache_read
            + cache_write_tokens * price.cache_write) / 1e6
    return cost * (BATCH_DISCOUNT if batch else 1.0)


def api_cost_per_1k(price: TokenPrice, input_tokens: int, output_tokens: int,
                    cached_tokens: int = 0, batch: bool = False) -> float:
    """Steady-state cost of 1,000 requests.

    cached_tokens is a shared prompt prefix read from the cache on every request;
    the occasional cache write is ignored because it is spread over many requests.
    """
    return 1000 * request_cost(price, input_tokens, output_tokens,
                               cache_read_tokens=cached_tokens, batch=batch)


def monthly_cost(volume: float, per_1k: float = 0.0, fixed: float = 0.0,
                 error_rate: float = 0.0, cost_per_error: float = 0.0) -> float:
    """Monthly bill for `volume` requests, counting what wrong answers cost."""
    if volume < 0:
        raise ValueError("volume cannot be negative")
    return fixed + volume / 1000 * per_1k + volume * error_rate * cost_per_error


def break_even_volume(api_per_1k: float, fixed_monthly: float, selfhost_per_1k: float = 0.0,
                      api_error_rate: float = 0.0, selfhost_error_rate: float = 0.0,
                      cost_per_error: float = 0.0) -> float:
    """Monthly requests above which self-hosting is cheaper. math.inf if it never is."""
    if fixed_monthly < 0:
        raise ValueError("fixed_monthly cannot be negative")
    saving_per_request = ((api_per_1k - selfhost_per_1k) / 1000
                          + (api_error_rate - selfhost_error_rate) * cost_per_error)
    if saving_per_request <= 0:
        return math.inf
    return fixed_monthly / saving_per_request


def error_cost_to_break_even(volume: float, api_per_1k: float, fixed_monthly: float,
                             api_error_rate: float, selfhost_error_rate: float,
                             selfhost_per_1k: float = 0.0) -> float:
    """Cost of one wrong answer at which the two bills are equal at this volume.

    Above it, self-hosting wins. Returns 0.0 if self-hosting already wins with
    free errors, and math.inf if it makes at least as many errors and costs more.
    """
    if volume <= 0:
        raise ValueError("volume must be positive")
    extra_cost_per_request = fixed_monthly / volume + (selfhost_per_1k - api_per_1k) / 1000
    if extra_cost_per_request <= 0:
        return 0.0
    fewer_errors = api_error_rate - selfhost_error_rate
    if fewer_errors <= 0:
        return math.inf
    return extra_cost_per_request / fewer_errors
