"""Tests for tweet_index/ratelimit.py.

The bucket arithmetic is checked with tiny budgets so the tests stay fast. The
sustained behaviour against a real provider is a separate exercise; this file
only proves the pacing maths and that concurrent callers cannot exceed a
budget between them.
"""

import threading
import time

import pytest

from tweet_index.ratelimit import Budget, RateLimiter, from_config


# ---------------------------------------------------------------------------
# Budget: one refilling bucket
# ---------------------------------------------------------------------------

def test_budget_starts_full():
    b = Budget(per_minute=60)
    assert b.available == 60
    assert b.wait_for(60, time.monotonic()) == 0


def test_headroom_lowers_the_cap():
    b = Budget(per_minute=1000, headroom=0.10)
    assert b.limit == pytest.approx(900)


def test_budget_makes_you_wait_when_empty():
    b = Budget(per_minute=60)            # one unit per second
    now = time.monotonic()
    b.refill(now)
    b.take(60)
    assert b.wait_for(1, now) == pytest.approx(1.0, abs=0.01)
    assert b.wait_for(5, now) == pytest.approx(5.0, abs=0.01)


def test_budget_refills_over_time():
    b = Budget(per_minute=60)
    now = time.monotonic()
    b.refill(now)
    b.take(60)
    b.refill(now + 10)                   # ten seconds later
    assert b.available == pytest.approx(10, abs=0.01)


def test_refill_never_exceeds_the_cap():
    b = Budget(per_minute=60)
    b.refill(time.monotonic() + 3600)
    assert b.available == 60


def test_giving_back_never_exceeds_the_cap():
    b = Budget(per_minute=60)
    b.give_back(1000)
    assert b.available == 60


# ---------------------------------------------------------------------------
# RateLimiter: two budgets at once
# ---------------------------------------------------------------------------

def test_a_call_passes_straight_through_when_there_is_room():
    limiter = RateLimiter(requests_per_minute=600, tokens_per_minute=600000,
                          headroom=0)
    started = time.monotonic()
    limiter.acquire(100)
    assert time.monotonic() - started < 0.05
    assert limiter.calls == 1


def test_the_request_budget_can_be_the_constraint():
    limiter = RateLimiter(requests_per_minute=120, tokens_per_minute=10**9,
                          headroom=0)
    limiter.drain()                      # past the initial burst allowance
    delay = limiter._claim(1)
    assert delay > 0, "the next call should have to wait on requests"


def test_the_token_budget_can_be_the_constraint():
    # Requests effectively unlimited, 60 tokens per minute.
    limiter = RateLimiter(requests_per_minute=10**6, tokens_per_minute=60,
                          headroom=0)
    limiter.drain()
    delay = limiter._claim(10)
    assert delay > 0, "the next call should have to wait on tokens"


def test_whichever_budget_is_tighter_wins():
    limiter = RateLimiter(requests_per_minute=10**6, tokens_per_minute=60,
                          headroom=0)
    limiter.drain()
    # 30 tokens needs 30s; one request needs none. The wait is the longer.
    assert limiter._claim(30) == pytest.approx(30, abs=0.5)


def test_settling_refunds_an_overestimate():
    limiter = RateLimiter(requests_per_minute=10**6, tokens_per_minute=1000,
                          headroom=0)
    reserved = limiter.acquire(700)
    assert limiter.tokens.available == pytest.approx(300, abs=1)
    limiter.settle(reserved, 90)          # the call really cost 90
    assert limiter.tokens.available == pytest.approx(910, abs=1)


def test_settling_charges_an_underestimate():
    limiter = RateLimiter(requests_per_minute=10**6, tokens_per_minute=1000,
                          headroom=0)
    reserved = limiter.acquire(100)
    limiter.settle(reserved, 400)         # it cost more than we reserved
    assert limiter.tokens.available == pytest.approx(600, abs=1)


def test_refunds_cannot_push_the_bucket_above_its_cap():
    limiter = RateLimiter(requests_per_minute=10**6, tokens_per_minute=1000,
                          headroom=0)
    reserved = limiter.acquire(10)
    limiter.settle(reserved, 0)
    assert limiter.tokens.available <= limiter.tokens.limit


# ---------------------------------------------------------------------------
# The point of the exercise: budgets hold under concurrency
# ---------------------------------------------------------------------------

def test_threads_cannot_exceed_the_request_budget_between_them():
    """24 threads against a 60/min budget must still land on 60/min."""
    limiter = RateLimiter(requests_per_minute=600, tokens_per_minute=10**9,
                          headroom=0)          # 10 per second
    limiter.drain()                            # measure the steady state
    done = []

    def worker():
        for _ in range(5):
            limiter.acquire(1)
            done.append(1)

    threads = [threading.Thread(target=worker) for _ in range(24)]
    started = time.monotonic()
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    elapsed = time.monotonic() - started

    assert len(done) == 120
    rate = len(done) / elapsed
    assert rate <= 11, f"paced at {rate:.1f}/s, above the 10/s budget"


def test_concurrent_token_spend_stays_within_budget():
    limiter = RateLimiter(requests_per_minute=10**6, tokens_per_minute=6000,
                          headroom=0)          # 100 tokens per second
    limiter.drain()
    spent = []

    def worker():
        for _ in range(10):
            reserved = limiter.acquire(10)
            spent.append(10)
            limiter.settle(reserved, 10)

    threads = [threading.Thread(target=worker) for _ in range(8)]
    started = time.monotonic()
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    elapsed = time.monotonic() - started

    assert sum(spent) == 800
    per_second = sum(spent) / elapsed
    assert per_second <= 110, f"spent {per_second:.0f} tokens/s, budget is 100"


def test_async_acquire_also_paces():
    """The asyncio path shares _claim, but check it really waits."""
    import asyncio

    limiter = RateLimiter(requests_per_minute=10**6, tokens_per_minute=120,
                          headroom=0)          # 2 tokens per second
    limiter.drain()

    async def go():
        started = time.monotonic()
        await limiter.acquire_async(4)         # needs 2 seconds of refill
        return time.monotonic() - started

    waited = asyncio.run(go())
    assert waited >= 1.5, f"returned after only {waited:.2f}s"


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

def test_from_config_builds_a_limiter():
    cfg = {"rate_limits": {"openai": {"requests_per_minute": 500,
                                      "tokens_per_minute": 200000,
                                      "headroom": 0.10}}}
    limiter = from_config(cfg, "openai")
    assert limiter.name == "openai"
    assert limiter.requests.limit == pytest.approx(450)
    assert limiter.tokens.limit == pytest.approx(180000)


def test_from_config_returns_none_for_an_unknown_provider():
    assert from_config({"rate_limits": {}}, "openai") is None
    assert from_config({}, "openai") is None


def test_stats_reports_what_happened():
    limiter = RateLimiter(600, 10**9, headroom=0, name="test")
    limiter.acquire(1)
    limiter.acquire(1)
    stats = limiter.stats()
    assert stats["calls"] == 2
    assert stats["name"] == "test"
    assert stats["calls_per_second"] > 0
