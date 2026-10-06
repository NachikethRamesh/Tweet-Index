"""One function for talking to a large language model: complete_json().

Section 4 asks for this file to be provider-agnostic, so the rest of the code
never imports a vendor SDK. Everything the caller needs is a system prompt, a
user message, and a model name; what comes back is a parsed dict plus the token
counts, so cost can be reported honestly.

Only OpenAI is wired up. Swapping provider means editing this file and nothing
else.
"""

import base64
import json
import os
import re

PROVIDER = "openai"

# Roughly how many characters of prompt make one token, measured over 100 real
# enrichment calls. Used only to reserve budget before a call; the reservation
# is corrected from the reply's own usage figures.
CHARS_PER_TOKEN = 3.75

_client = None
_limiter = None


def set_limiter(limiter):
    """Install the shared rate limiter. Without one, calls are unpaced."""
    global _limiter
    _limiter = limiter


def limiter():
    return _limiter


class LLMError(RuntimeError):
    """A call failed, or came back as something other than the JSON asked for."""


def client():
    """The OpenAI client, created once per process.

    The key is read from the OPENAI_API_KEY environment variable, usually via
    the .env file. It is never passed on the command line or written to a log.
    """
    global _client
    if _client is None:
        if not os.environ.get("OPENAI_API_KEY"):
            raise LLMError(
                "OPENAI_API_KEY is not set. Put it in .env as "
                "OPENAI_API_KEY=sk-... (that file is git-ignored).")
        try:
            from openai import OpenAI
        except ImportError as exc:        # pragma: no cover
            raise LLMError("the openai package is not installed; "
                           "pip install -r requirements.txt") from exc
        # Retries 429s and 5xx with backoff. The design says 5 (Section 8.2);
        # 8 is used because a long run sits close to the token-per-minute
        # limit, and five attempts against a continuously saturated bucket is
        # not enough -- 3,260 of 5,909 calls were lost that way on 2026-10-04.
        _client = OpenAI(max_retries=8)
    return _client


FENCE = re.compile(r"^\s*```(?:json)?\s*(.*?)\s*```\s*$", re.DOTALL)


def strip_code_fences(text):
    """Unwrap ```json ... ``` if the model adds it anyway."""
    match = FENCE.match(text or "")
    return match.group(1) if match else text


def data_url(image_bytes, mime="image/jpeg"):
    """An image as a data URL, so nothing has to be publicly reachable."""
    return f"data:{mime};base64,{base64.b64encode(image_bytes).decode('ascii')}"


def complete_json(system, user, model, schema=None, max_output_tokens=400,
                  reasoning_effort=None, image=None, image_mime="image/jpeg"):
    """Ask for one JSON object back.

    `schema` is an optional JSON Schema. When given, the model is held to it
    exactly, so the reply cannot be missing a field or wrapped in prose. When
    omitted, the model is only told to return valid JSON.

    `reasoning_effort` matters on reasoning models, where thinking tokens are
    drawn from the same budget as the answer: if reasoning fills
    max_output_tokens, the reply arrives with no content at all. Lowering it
    leaves more room for the answer and costs less.

    Returns (parsed_dict, usage) where usage is
    {"input_tokens": int, "output_tokens": int, "reasoning_tokens": int}.
    """
    if schema is None:
        response_format = {"type": "json_object"}
    else:
        response_format = {
            "type": "json_schema",
            "json_schema": {"name": "result", "strict": True, "schema": schema},
        }

    # With an image, the user message becomes a list of parts. The image
    # travels as a base64 data URL rather than a link, so the provider never
    # has to reach a public URL and a local copy is enough.
    if image is None:
        user_content = user
    else:
        user_content = [
            {"type": "text", "text": user},
            {"type": "image_url",
             "image_url": {"url": data_url(image, image_mime)}},
        ]

    request = {
        "model": model,
        "max_completion_tokens": max_output_tokens,
        "response_format": response_format,
        "messages": [{"role": "system", "content": system},
                     {"role": "user", "content": user_content}],
    }
    if reasoning_effort:
        request["reasoning_effort"] = reasoning_effort

    # Reserve budget before calling: the prompt's own length plus the most the
    # reply could cost. Both are corrected below from the real usage figures.
    reserved = None
    if _limiter is not None:
        estimate = int((len(system) + len(user)) / CHARS_PER_TOKEN) \
                   + max_output_tokens
        if image is not None:
            estimate += 1200        # a tweet photo costs roughly this much
        reserved = _limiter.acquire(estimate)

    try:
        reply = client().chat.completions.create(**request)
    except Exception as exc:              # network, auth, rate limit, refusal
        if reserved is not None:
            _limiter.settle(reserved, 0)   # the call cost us nothing; refund
        raise LLMError(f"{type(exc).__name__}: {exc}") from exc

    usage = getattr(reply, "usage", None)
    if reserved is not None:
        spent = ((getattr(usage, "prompt_tokens", 0) or 0)
                 + (getattr(usage, "completion_tokens", 0) or 0))
        _limiter.settle(reserved, spent)

    choice = reply.choices[0]
    why = getattr(choice, "finish_reason", None)
    text = strip_code_fences(choice.message.content or "")
    if not text.strip():
        raise LLMError(f"the model returned an empty reply (finish={why})")
    try:
        parsed = json.loads(text)
    except json.JSONDecodeError as exc:
        # finish_reason matters here: 'length' means the budget ran out, while
        # 'content_filter' means generation was halted and no bigger budget
        # will help.
        raise LLMError(f"reply was not JSON (finish={why}): "
                       f"{text[:160]!r}") from exc

    details = getattr(usage, "completion_tokens_details", None)
    return parsed, {
        "input_tokens": getattr(usage, "prompt_tokens", 0) or 0,
        "output_tokens": getattr(usage, "completion_tokens", 0) or 0,
        "reasoning_tokens": getattr(details, "reasoning_tokens", 0) or 0,
    }
