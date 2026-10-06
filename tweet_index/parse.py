"""Turn one X timeline entry into a flat dictionary of columns.

This follows the listing in Section 7.3 of the technical design. The shape
of X's payload is deeply nested and inconsistent, so every lookup goes
through dig(), which returns None instead of raising when a step is
missing. Nothing here touches the network or the database: give it an
entry, get back a dict.
"""

import html
from datetime import datetime, timezone


def dig(d, *keys):
    """Walk nested dicts, returning None as soon as a step is missing."""
    for k in keys:
        if not isinstance(d, dict):
            return None
        d = d.get(k)
    return d


def unwrap(r):
    """Return the tweet inside a result, or None if it is not a tweet.

    Check the type BEFORE unwrapping: the tweet inside a
    TweetWithVisibilityResults has no __typename at all.
    """
    if not isinstance(r, dict):
        return None
    t = r.get("__typename")
    if t == "TweetWithVisibilityResults":
        return r.get("tweet")
    if t in ("Tweet", None):
        return r
    return None                          # TweetTombstone, TweetUnavailable, ...


def iter_entries(node):
    """Yield timeline entries (entryId 'tweet-<id>') anywhere in a payload.

    Works for both feed roots; quoted tweets are NOT entries.
    """
    if isinstance(node, dict):
        if str(node.get("entryId", "")).startswith("tweet-"):
            yield node
            return
        for v in node.values():
            yield from iter_entries(v)
    elif isinstance(node, list):
        for v in node:
            yield from iter_entries(v)


def post_text(r):
    """The post's text, in full and with HTML entities decoded.

    A long post keeps its complete text in note_tweet; legacy.full_text is
    truncated at 280 characters. For short posts, display_text_range is
    measured on the ESCAPED text, so slice first and unescape second --
    doing it the other way round shifts every index.
    """
    note = dig(r, "note_tweet", "note_tweet_results", "result", "text")
    if note:
        return note.strip()              # long post; already unescaped
    leg = r.get("legacy") or {}
    txt, rng = leg.get("full_text", ""), leg.get("display_text_range")
    if rng:
        txt = txt[rng[0]:rng[1]]         # range counts the ESCAPED text
    return html.unescape(txt).strip()


def url_entities(r):
    """Every link entity on a post: legacy entities plus the note's own set.

    A long post keeps its links only in the note's entity_set, so both
    places have to be read (Section 7.2).
    """
    return ((dig(r, "legacy", "entities", "urls") or []) +
            (dig(r, "note_tweet", "note_tweet_results", "result",
                 "entity_set", "urls") or []))


def media_entities(r):
    return dig(r, "legacy", "extended_entities", "media") or []


def clean_text(r):
    """The post's text with media links removed and author links expanded.

    Two kinds of t.co link can appear in the text, and they are handled
    differently:

      * a link to the post's own photo or video, which carries no meaning
        and is removed;
      * a link the author typed, which is replaced by the real URL it
        points at, so the stored text says "economist.com/..." rather than
        an opaque "t.co/abc123".

    After this, no stored text contains a t.co link at all.
    """
    text = post_text(r)

    for m in media_entities(r):          # the post's own media: drop
        if m.get("url"):
            text = text.replace(m["url"], "")

    for u in url_entities(r):            # links the author typed: expand
        if u.get("url") and u.get("expanded_url"):
            text = text.replace(u["url"], u["expanded_url"])

    return text.strip()


def article(r):
    """(title, preview_text) of an X Article, if the post carries one."""
    a = dig(r, "article", "article_results", "result") or {}
    return a.get("title"), a.get("preview_text")


def card(r):
    """(title, description) of a link preview card, if present."""
    vals = {b.get("key"): dig(b, "value", "string_value")
            for b in dig(r, "card", "legacy", "binding_values") or []}
    return vals.get("title"), vals.get("description")


def to_iso(s):
    """X's 'Thu Sep 17 09:18:21 +0000 2026' to ISO 8601 UTC."""
    if not s:
        return None
    return datetime.strptime(s, "%a %b %d %H:%M:%S %z %Y") \
                   .astimezone(timezone.utc).isoformat()


def parse_entry(entry):
    """One timeline entry to one row's worth of fields, or None to skip it.

    Returns None for tombstones, unavailable tweets and anything without a
    rest_id, so ingest can log the entry and carry on.
    """
    r = unwrap(dig(entry, "content", "itemContent", "tweet_results", "result"))
    if not r or not r.get("rest_id"):
        return None

    rt = unwrap(dig(r, "legacy", "retweeted_status_result", "result"))
    if rt:
        r = rt                           # store the original, not the wrapper

    leg = r.get("legacy") or {}
    user = dig(r, "core", "user_results", "result") or {}
    handle = dig(user, "core", "screen_name") or dig(user, "legacy", "screen_name")

    q = unwrap(dig(r, "quoted_status_result", "result")) or {}
    q_user = dig(q, "core", "user_results", "result") or {}

    a_title, a_prev = article(r)
    qa_title, qa_prev = article(q)
    c_title, c_desc = card(r)

    tr = dig(r, "grok_translated_post_with_availability", "data") or {}

    links = [u.get("expanded_url") for u in url_entities(r)]

    return {
        "id": r["rest_id"],
        "text": clean_text(r),
        "author_handle": handle,
        "author_name": dig(user, "core", "name") or dig(user, "legacy", "name"),
        "posted_at": to_iso(leg.get("created_at")),
        "lang": leg.get("lang"),
        "quoted_id": q.get("rest_id"),
        "quoted_author": (dig(q_user, "core", "screen_name")
                          or dig(q_user, "legacy", "screen_name")),
        "quoted_text": (clean_text(q) if q else None) or
                       "\n".join(p for p in (qa_title, qa_prev) if p) or None,
        "in_reply_to_id": leg.get("in_reply_to_status_id_str"),
        "initial_tweet_id": dig(r, "edit_control", "initial_tweet_id"),
        "card_title": c_title,
        "card_description": c_desc,
        "article_title": a_title or qa_title,
        "article_preview": a_prev or qa_prev,
        "translation_en": (tr.get("translation")
                           if tr.get("destination_language") == "en" else None),
        "community_note": dig(r, "birdwatch_pivot", "subtitle", "text"),
        "save_order": entry.get("sortIndex"),
        # The feed decides is_like / is_bookmark. These two are the payload's
        # own opinion, kept so ingest can cross-check and log disagreements.
        "x_favorited": leg.get("favorited"),
        "x_bookmarked": leg.get("bookmarked"),
        "url": f"https://x.com/{handle}/status/{r['rest_id']}",
        "media": [(m["type"], m["media_url_https"])
                  for m in media_entities(r)],
        "links": list(dict.fromkeys(u for u in links if u)),   # dedupe, keep order
    }
