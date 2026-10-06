"""Build synthetic X timeline responses for the tests.

The original test suite ran against real captures from the author's own
account. Those are personal data and are not in this repository, so this
module generates payloads of the same shape with invented tweets instead.

The shape is what `tweet_index.parse` expects, and only the parts the parser
reads are filled in -- X sends a great deal more per entry than this.

    python tests/make_fixtures.py        # write a sample pair to /tmp and
                                         # print what is in them

Each builder returns a plain dict, so a test can compose exactly the case it
needs rather than hunting for a tweet that happens to exhibit it.
"""

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))


def user(handle="alice_dev", name="Alice Dev"):
    return {"result": {"core": {"screen_name": handle, "name": name}}}


def tweet(tweet_id, text, handle="alice_dev", name="Alice Dev",
          created_at="Mon Sep 29 10:00:00 +0000 2026", lang="en",
          favorited=False, bookmarked=False, display_range=None,
          urls=None, media=None, quoted=None, card=None, article=None,
          translation=None, community_note=None, reply_to=None,
          retweet_of=None, typename="Tweet"):
    """One tweet result, as it sits inside a timeline entry.

    `display_range` is X's display_text_range. Leaving it None means "the
    whole text is the post", which is the common case; setting it is how the
    trailing media t.co link is marked as not part of the text.
    """
    legacy = {
        "full_text": text,
        "created_at": created_at,
        "lang": lang,
        "favorited": favorited,
        "bookmarked": bookmarked,
        "entities": {"urls": list(urls or [])},
    }
    if display_range is not None:
        legacy["display_text_range"] = list(display_range)
    if media:
        legacy["entities"]["media"] = list(media)
        legacy["extended_entities"] = {"media": list(media)}
    if reply_to:
        legacy["in_reply_to_status_id_str"] = reply_to
    if retweet_of:
        legacy["retweeted_status_result"] = {"result": retweet_of}

    result = {
        "__typename": typename,
        "rest_id": tweet_id,
        "legacy": legacy,
        "core": {"user_results": user(handle, name)},
    }
    if quoted:
        result["quoted_status_result"] = {"result": quoted}
    if card:
        result["card"] = card
    if article:
        result["article"] = article
    if translation:
        result["grok_translated_post_with_availability"] = {
            "data": translation}
    if community_note:
        result["birdwatch_pivot"] = {"subtitle": {"text": community_note}}
    return result


def entry(tweet_result, sort_index="1900000000000000000"):
    """A timeline entry wrapping one tweet result."""
    return {
        "entryId": f"tweet-{tweet_result.get('rest_id', 'unknown')}",
        "sortIndex": sort_index,
        "content": {
            "entryType": "TimelineTimelineItem",
            "itemContent": {
                "itemType": "TimelineTweet",
                "tweet_results": {"result": tweet_result},
            },
        },
    }


def tombstone(entry_id="tweet-9000000000000000001",
              sort_index="1800000000000000000"):
    """A deleted or withheld tweet. The parser must skip these."""
    return {
        "entryId": entry_id,
        "sortIndex": sort_index,
        "content": {
            "entryType": "TimelineTimelineItem",
            "itemContent": {
                "itemType": "TimelineTombstone",
                "tweet_results": {
                    "result": {
                        "__typename": "TweetTombstone",
                        "tombstone": {"text": {
                            "text": "This Post is unavailable."}},
                    }
                },
            },
        },
    }


def cursor(value="DAABCgABGv_x", direction="Bottom"):
    return {"entryId": f"cursor-bottom-{value}",
            "content": {"entryType": "TimelineTimelineCursor",
                        "value": value, "cursorType": direction}}


def with_visibility(tweet_result, limited_actions="limited_replies"):
    """A tweet wrapped in TweetWithVisibilityResults.

    X uses this for posts with reply limits or sensitive-content notices. The
    parser has to look past the wrapper, which is a case worth a test because
    the `__typename` must be read before unwrapping.
    """
    return {
        "__typename": "TweetWithVisibilityResults",
        "tweet": tweet_result,
        "limitedActionResults": {"limited_actions": [
            {"action": limited_actions}]},
    }


def media_item(kind="photo", url="https://pbs.twimg.com/media/AAA.jpg",
               tco="https://t.co/mediaLINK"):
    return {"type": kind, "media_url_https": url, "url": tco,
            "expanded_url": "https://x.com/alice_dev/status/1/photo/1"}


def url_item(tco="https://t.co/shortLINK",
             expanded="https://example.com/an-article"):
    return {"url": tco, "expanded_url": expanded,
            "display_url": "example.com/an-article"}


def card(title="A Linked Page", description="What the linked page is about"):
    return {"legacy": {"binding_values": [
        {"key": "title", "value": {"string_value": title, "type": "STRING"}},
        {"key": "description",
         "value": {"string_value": description, "type": "STRING"}},
    ]}}


def article(title="A Long Read", preview="The opening of the long read."):
    return {"article_results": {"result": {
        "title": title,
        "preview_text": preview,
    }}}


def page(entries, feed="bookmarks", with_cursor=True):
    """A whole response page, nested the way X nests it.

    The two feeds are nested differently, and the userscript classifies a
    response by its shape rather than by the GraphQL operation name:
    bookmarks arrive under `data.bookmark_timeline_v2`, likes under
    `data.user.result.timeline` with `scribeConfig.page` set to 'favorites'.
    The Python parser walks whatever it is given and so does not care, but
    these payloads keep the real shapes: anyone reading them to learn the
    format should see the truth, and the userscript tests depend on it.
    """
    items = list(entries)
    if with_cursor:
        items = items + [cursor()]
    instructions = [{"type": "TimelineAddEntries", "entries": items}]
    if feed == "bookmarks":
        return {"data": {"bookmark_timeline_v2": {
            "timeline": {"instructions": instructions}}}}
    return {"data": {"user": {"result": {"timeline": {"timeline": {
        "instructions": instructions,
        "metadata": {"scribeConfig": {"page": "favorites"}},
    }}}}}}


def capture(pages, feed="bookmarks",
            captured_at="2026-09-29T10:00:00.000Z"):
    """A capture file as the userscript writes it."""
    return {
        "kind": feed,
        "captured_at": captured_at,
        "pages": [{"json": p} for p in pages],
    }


# --------------------------------------------------------------------------
# A standard set, used by the tests
# --------------------------------------------------------------------------

def sample_bookmarks():
    """Three pages of bookmarks covering the cases the parser handles."""
    plain = tweet("1000000000000000001", "A plain bookmarked post.",
                  bookmarked=True)

    # Text ends before the media t.co link: display_text_range marks it.
    body = "A post with a photo."
    with_photo = tweet(
        "1000000000000000002", f"{body} https://t.co/mediaLINK",
        bookmarked=True, display_range=[0, len(body)],
        media=[media_item()])

    # An author link, which the parser expands to its real destination.
    linked = tweet(
        "1000000000000000003",
        "Worth reading: https://t.co/shortLINK",
        bookmarked=True, urls=[url_item()], card=card())

    quoted_inner = tweet("1000000000000000010",
                         "The original claim, stated at length.",
                         handle="bob_writes", name="Bob Writes")
    reaction = tweet("1000000000000000004", "This.", bookmarked=True,
                     quoted=quoted_inner)

    noted = tweet("1000000000000000005", "A confident but wrong claim.",
                  bookmarked=True,
                  community_note="Context: the cited paper says otherwise.")

    wrapped = with_visibility(
        tweet("1000000000000000006", "A post with reply limits.",
              bookmarked=True))

    longread = tweet("1000000000000000007", "", bookmarked=True,
                     article=article())

    translated = tweet(
        "1000000000000000008", "Bu bir Türkçe gönderidir.",
        bookmarked=True, lang="tr",
        translation={"destination_language": "en",
                     "translation": "This is a post in Turkish."})

    # Also in the likes feed below, so ingest sees it in both.
    both = tweet("1000000000000000009", "Saved twice, in both feeds.",
                 bookmarked=True, favorited=True)

    return [
        page([entry(plain, "1900000000000000009"),
              entry(with_photo, "1900000000000000008"),
              entry(linked, "1900000000000000007")], "bookmarks"),
        page([entry(reaction, "1900000000000000006"),
              entry(noted, "1900000000000000005"),
              tombstone()], "bookmarks"),
        page([entry(wrapped, "1900000000000000004"),
              entry(longread, "1900000000000000003"),
              entry(translated, "1900000000000000002"),
              entry(both, "1900000000000000001")], "bookmarks"),
    ]


def sample_likes():
    """Two pages of likes, one tweet overlapping the bookmarks."""
    a = tweet("2000000000000000001", "A liked post about testing.",
              handle="carol_qa", name="Carol QA", favorited=True)
    b = tweet("2000000000000000002", "A liked post with no text at all.",
              handle="dan_designs", favorited=True, media=[media_item("video")])
    both = tweet("1000000000000000009", "Saved twice, in both feeds.",
                 favorited=True, bookmarked=True)
    reply = tweet("2000000000000000003", "Replying to something.",
                  favorited=True, reply_to="2000000000000000001")
    return [
        page([entry(a, "1800000000000000004"),
              entry(b, "1800000000000000003")], "likes"),
        page([entry(both, "1800000000000000002"),
              entry(reply, "1800000000000000001")], "likes"),
    ]


def write_captures(directory):
    """Write one bookmarks and one likes capture file. Returns both paths."""
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    bookmarks = directory / "capture_bookmarks_2026-09-29T10-00-00-000Z.json"
    likes = directory / "capture_likes_2026-09-29T10-05-00-000Z.json"
    bookmarks.write_text(
        json.dumps(capture(sample_bookmarks(), "bookmarks"), indent=1),
        encoding="utf-8")
    likes.write_text(
        json.dumps(capture(sample_likes(), "likes"), indent=1),
        encoding="utf-8")
    return bookmarks, likes


if __name__ == "__main__":
    import tempfile

    from tweet_index.parse import iter_entries, parse_entry

    out = Path(tempfile.gettempdir()) / "tweet-index-fixtures"
    paths = write_captures(out)
    for path in paths:
        blob = json.loads(path.read_text(encoding="utf-8"))
        entries = [e for p in blob["pages"] for e in iter_entries(p["json"])]
        parsed = [t for t in (parse_entry(e) for e in entries) if t]
        print(f"{path.name}")
        print(f"  pages {len(blob['pages'])}, entries {len(entries)}, "
              f"parsed {len(parsed)}, skipped {len(entries) - len(parsed)}")
        for t in parsed:
            print(f"    {t['id']}  @{t['author_handle']:14} "
                  f"{(t['text'] or '(no text)')[:44]}")
