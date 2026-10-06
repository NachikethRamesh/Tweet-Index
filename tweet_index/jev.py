"""The TypeSafe Jev client: typed judgments with calibrated probabilities.

Section 12 of the technical design. Jev never generates text. You hand it a
state (one tweet) and a set of questions, each with fixed options, and it
answers every question with a probability or a position along named levels.
That is why the vocabulary is fixed and the LLM, not Jev, proposes concepts.

Everything Jev is used for goes through two functions (Section 12.3), so the
whole dependency can be replaced by a small LLM returning JSON if TypeSafe
changes terms or disappears:

    judge_tweet(state, questions) -> answers
    relevance(states, topic)      -> [(id, level, confidence)]

The model is pinned. Never `jev-latest`: an alias moves when TypeSafe ships,
which would silently shift every threshold tuned against it.
"""

import json
import os
from datetime import datetime, timezone
from pathlib import Path

# Input tokens are charged; output tokens are free (Section 12.1).
USD_PER_MILLION_INPUT = 0.042

_client = None
_limiter = None
_model = None


class JevError(RuntimeError):
    """A call failed, or came back in a shape the design did not expect."""


def set_limiter(limiter):
    """Install the shared rate limiter (Section 12.1: 18 requests/second)."""
    global _limiter
    _limiter = limiter


def configure(model):
    """Pin the model for this process. Call once, before any judging."""
    global _model
    if not model:
        raise JevError("no jev model configured")
    if "latest" in model:
        raise JevError(f"refusing to use the moving alias {model!r}; "
                       "pin an exact version such as jev-1.13.0")
    _model = model


def model():
    if _model is None:
        raise JevError("call configure(model) first")
    return _model


def sdk():
    """The typesafe_sdk module, imported lazily so tests need no key."""
    try:
        import typesafe_sdk
    except ImportError as exc:        # pragma: no cover
        raise JevError("the typesafe-sdk package is not installed; "
                       "pip install -r requirements.txt") from exc
    return typesafe_sdk


def client():
    """The synchronous client, created once per process."""
    global _client
    if _client is None:
        if not os.environ.get("TYPESAFE_API_KEY"):
            raise JevError(
                "TYPESAFE_API_KEY is not set. Put it in .env as "
                "TYPESAFE_API_KEY=... (that file is git-ignored).")
        _client = sdk().TypeSafeClient(model=model())
    return _client


def close():
    """Close the synchronous client, if one was opened."""
    global _client
    if _client is not None:
        _client.close()
        _client = None


def async_client():
    """A fresh asynchronous client, for the Phase 2 relevance rerank.

    Use it with `async with`, which closes it properly. The asynchronous
    client has `aclose()` rather than `close()` (confirmed live 2026-10-04):

        async with jev.async_client() as client:
            resp = await client.system_one(state=..., questions=...)
    """
    if not os.environ.get("TYPESAFE_API_KEY"):
        raise JevError("TYPESAFE_API_KEY is not set.")
    return sdk().AsyncTypeSafeClient(model=model())


# ---------------------------------------------------------------------------
# Questions
# ---------------------------------------------------------------------------

# Section 10.3. Descriptive levels, not 0-10 numbers: Jev's own documentation
# says numeric calibration is weak, and descriptive levels are judged more
# consistently.
SUBSTANCE_LEVELS = [
    "No information: a joke, a reaction, or a bare link.",
    "An opinion or claim with no support.",
    "A useful pointer: a link, paper, or resource with a short description.",
    "Explains something with reasoning, examples, or data.",
    "In-depth: a detailed thread, write-up, or primary source.",
]

# Section 11.2.
RELEVANCE_LEVELS = [
    "Unrelated to the topic.",
    "Mentions the topic only in passing.",
    "Partly about the topic; it is one of several subjects.",
    "Mainly about the topic.",
    "Squarely about the topic; someone researching it would want this.",
]


def substance_question():
    """One Score question: how substantive is this tweet?"""
    return sdk().Score(
        instructions="How substantive is `tweet`, taking `quoted_tweet`, "
                     "`image_text`, `card` and `links` into account? "
                     "A `community_note` disputing the tweet lowers substance.",
        criteria=SUBSTANCE_LEVELS,
    )


def relevance_question(topic):
    """One Score question: how relevant is this tweet to a topic?

    The topic travels as a structured field of instructions rather than inside
    the state, so the state stays the same small per-tweet object used for
    tagging (Section 11.2).
    """
    return sdk().Score(
        instructions={
            "question": "How relevant is `tweet` to `topic`? Use `summary`, "
                        "`quoted_tweet` and `image_text` as context.",
            "topic": topic,
        },
        criteria=RELEVANCE_LEVELS,
    )


def concept_questions(concepts):
    """One Noul (yes/no with a probability) per concept."""
    noul = sdk().Noul
    return {c["id"]: noul(instructions=c["question"]) for c in concepts}


# ---------------------------------------------------------------------------
# State
# ---------------------------------------------------------------------------

MAX_FIELD_CHARS = 600      # Section 12.2: large irrelevant state hurts accuracy

# For judging the tweet itself, 600 characters is too tight. §12.2's warning is
# about *irrelevant* state; a long post's own text is the evidence, and cutting
# it off makes a detailed thread look like a one-liner. The faithfulness audit
# showed this directly: with a 600-character clip, 92.7% of posts over 2,000
# characters were judged against evidence that had been cut away. Judgment
# states use JUDGE_CHARS for the tweet's own text and the tighter limit for
# everything around it.
JUDGE_CHARS = 4000


def clip(value, limit=MAX_FIELD_CHARS):
    if not value:
        return None
    value = str(value).strip()
    return value[:limit] if value else None


def column(row, name):
    """A column that may not be in this query's SELECT."""
    return row[name] if name in row.keys() else None


def join(*parts):
    kept = [p.strip() for p in parts if p and p.strip()]
    return " - ".join(kept) or None


def tweet_state(row, link_titles=(), image_text=None, media=(),
                text_limit=JUDGE_CHARS):
    """One tweet as a Jev state (Section 10.3).

    Exactly one tweet per state. Everything the enrichment call saw is
    included, because a judge that cannot see the evidence judges the wrong
    thing: `article` and `media` are here for that reason, beyond the fields
    Section 10.3 lists.
    """
    return {
        "tweet": {"author": row["author_handle"],
                  "text": clip(row["text"], text_limit)},
        "quoted_tweet": clip(row["quoted_text"], text_limit),
        "summary": clip(column(row, "context")),
        "image_text": clip(image_text),
        "card": clip(join(row["card_title"], column(row, "card_description"))),
        "article": clip(join(column(row, "article_title"),
                             column(row, "article_preview"))),
        "translation_en": clip(column(row, "translation_en")),
        "community_note": clip(row["community_note"]),
        "media": list(media),
        "links": [t for t in link_titles if t][:5],
    }


# ---------------------------------------------------------------------------
# Calls
# ---------------------------------------------------------------------------

def usage_tokens(response):
    """Input tokens from a response.

    Section 12.1 marks the field name as needing verification, so several
    spellings are accepted and the one actually seen is recorded in
    the evaluation notes.
    """
    for holder in (getattr(response, "usage", None), response):
        if holder is None:
            continue
        for name in ("input_tokens", "prompt_tokens", "tokens_in",
                     "total_tokens", "tokens"):
            value = getattr(holder, name, None)
            if isinstance(value, int):
                return value, name
    return 0, None


def judge_tweet(state, questions):
    """One call: every question answered about one tweet.

    Returns (answers, version, input_tokens).
    """
    reserved = None
    if _limiter is not None:
        reserved = _limiter.acquire(estimate_tokens(state, questions))
    try:
        response = client().system_one(state=state, questions=questions)
    except Exception as exc:
        if reserved is not None:
            _limiter.settle(reserved, 0)
        raise JevError(f"{type(exc).__name__}: {exc}") from exc

    tokens, _field = usage_tokens(response)
    if reserved is not None:
        _limiter.settle(reserved, tokens)
    version = getattr(response, "model", None) or model()
    return response.answers, version, tokens


def estimate_tokens(state, questions):
    """A rough token cost, for reserving rate-limit budget before the call."""
    chars = len(json.dumps(state, ensure_ascii=False, default=str))
    chars += sum(len(str(q)) for q in questions.values())
    return int(chars / 3.75) + 50


# ---------------------------------------------------------------------------
# Cost log
# ---------------------------------------------------------------------------

class CostLog:
    """Accumulates spend for a run and appends one line per run to a file.

    Output tokens are free, so only input tokens are counted (Section 12.1).
    """

    def __init__(self, path=None):
        self.path = Path(path) if path else None
        self.calls = 0
        self.input_tokens = 0
        self.failures = 0

    def record(self, input_tokens):
        self.calls += 1
        self.input_tokens += input_tokens

    def record_failure(self):
        self.failures += 1

    @property
    def usd(self):
        return self.input_tokens / 1e6 * USD_PER_MILLION_INPUT

    def summary(self):
        return (f"{self.calls} Jev calls, {self.input_tokens:,} input tokens, "
                f"${self.usd:.4f}, {self.failures} failed")

    def write(self, what):
        if not self.path:
            return
        self.path.parent.mkdir(parents=True, exist_ok=True)
        stamp = datetime.now(timezone.utc).isoformat(timespec="seconds")
        with self.path.open("a", encoding="utf-8") as fh:
            fh.write(f"{stamp} {what} model={_model} {self.summary()}\n")
