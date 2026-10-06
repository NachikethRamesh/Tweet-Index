"""A token bucket that paces calls against two budgets at once.

Providers limit both how many requests you may send per minute and how many
tokens those requests may carry. Hitting either returns HTTP 429. A plain
concurrency number cannot keep you under both, because token cost per call
varies: concurrency 24 survived a 49-call burst against OpenAI and then lost
3,260 of 5,909 calls on a sustained run, because the token budget never
recovered.

So each budget gets a bucket that refills continuously, and a call waits until
both have room.

Token cost is not known until the reply arrives, so a caller reserves an
estimate first and settles up afterwards:

    reservation = limiter.acquire(estimated_tokens)
    ... make the call ...
    limiter.settle(reservation, actual_tokens)

Settling matters: reserving 700 and spending 90 would otherwise waste 87% of
the budget. The refund puts it back.

Section 3.2 names `aiolimiter` for this. It is asyncio-only and paces a single
budget, while the enrichment path is threaded and needs two budgets, so this
module is used instead. It offers both a blocking and an async acquire, so
threaded and asyncio callers share one implementation.
"""

import asyncio
import threading
import time


class Budget:
    """One continuously refilling bucket, in units per minute."""

    def __init__(self, per_minute, headroom=0.0):
        self.limit = per_minute * (1.0 - headroom)
        self.per_second = self.limit / 60.0
        self.available = self.limit
        self.updated = time.monotonic()

    def refill(self, now):
        self.available = min(self.limit,
                             self.available + (now - self.updated) * self.per_second)
        self.updated = now

    def wait_for(self, amount, now):
        """Seconds until `amount` is available. 0 when it already is."""
        self.refill(now)
        if amount <= self.available:
            return 0.0
        if self.per_second <= 0:
            return float("inf")
        return (amount - self.available) / self.per_second

    def take(self, amount):
        self.available -= amount

    def give_back(self, amount):
        self.available = min(self.limit, self.available + amount)


class RateLimiter:
    """Paces calls against a requests-per-minute and tokens-per-minute budget.

    `headroom` keeps a margin below the published limits, because the
    provider's accounting and ours will never agree exactly.
    """

    def __init__(self, requests_per_minute, tokens_per_minute, headroom=0.10,
                 name=""):
        self.name = name
        self.requests = Budget(requests_per_minute, headroom)
        self.tokens = Budget(tokens_per_minute, headroom)
        self._lock = threading.Lock()
        self.calls = 0
        self.waited = 0.0          # total seconds spent waiting
        self.started = time.monotonic()

    def _claim(self, estimated_tokens):
        """Take the budget if both have room, else say how long to wait."""
        with self._lock:
            now = time.monotonic()
            delay = max(self.requests.wait_for(1, now),
                        self.tokens.wait_for(estimated_tokens, now))
            if delay > 0:
                return delay
            self.requests.take(1)
            self.tokens.take(estimated_tokens)
            self.calls += 1
            return 0.0

    def acquire(self, estimated_tokens=0):
        """Block until there is room. Returns the reservation to settle."""
        while True:
            delay = self._claim(estimated_tokens)
            if delay <= 0:
                return estimated_tokens
            with self._lock:
                self.waited += delay
            time.sleep(min(delay, 1.0))     # re-check at least once a second

    async def acquire_async(self, estimated_tokens=0):
        """The same, for asyncio callers."""
        while True:
            delay = self._claim(estimated_tokens)
            if delay <= 0:
                return estimated_tokens
            with self._lock:
                self.waited += delay
            await asyncio.sleep(min(delay, 1.0))

    def settle(self, reserved, actual_tokens):
        """Correct the reservation once the real token count is known."""
        difference = actual_tokens - reserved
        with self._lock:
            if difference > 0:
                self.tokens.take(difference)
            elif difference < 0:
                self.tokens.give_back(-difference)

    def drain(self):
        """Empty both buckets.

        A token bucket starts full, so the first minute may legitimately burst
        up to the whole budget -- the provider allows that too. Tests that want
        to observe the steady-state rate drain it first.
        """
        with self._lock:
            now = time.monotonic()
            self.requests.refill(now)
            self.tokens.refill(now)
            self.requests.take(self.requests.available)
            self.tokens.take(self.tokens.available)

    def stats(self):
        elapsed = max(time.monotonic() - self.started, 1e-6)
        with self._lock:
            return {
                "name": self.name,
                "calls": self.calls,
                "seconds": elapsed,
                "calls_per_second": self.calls / elapsed if elapsed else 0.0,
                "seconds_waiting": self.waited,
                "requests_per_minute_cap": self.requests.limit,
                "tokens_per_minute_cap": self.tokens.limit,
            }


def from_config(cfg, provider):
    """Build a limiter from the rate_limits block of config.yaml."""
    limits = (cfg.get("rate_limits") or {}).get(provider)
    if not limits:
        return None
    return RateLimiter(
        requests_per_minute=limits["requests_per_minute"],
        tokens_per_minute=limits["tokens_per_minute"],
        headroom=limits.get("headroom", 0.10),
        name=provider,
    )
