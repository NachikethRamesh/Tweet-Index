"""The parser, against synthetic timeline payloads.

One test per case the parser exists to handle. The payloads come from
`make_fixtures.py`, which builds the shape X sends with invented tweets, so
each case can be composed exactly rather than found in a real capture.
"""

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from tests import make_fixtures as mk                # noqa: E402
from tweet_index.parse import (                      # noqa: E402
    clean_text, dig, iter_entries, parse_entry, to_iso, unwrap,
)


def parsed(tweet_result, sort_index="1900000000000000000"):
    """Parse one tweet result through a timeline entry."""
    return parse_entry(mk.entry(tweet_result, sort_index))


# --------------------------------------------------------------------------
# dig, unwrap, iter_entries
# --------------------------------------------------------------------------


def test_dig_walks_nested_keys():
    assert dig({"a": {"b": {"c": 1}}}, "a", "b", "c") == 1


@pytest.mark.parametrize("args", [
    ("a", "missing"), ("missing",), ("a", "b", "c", "too", "deep"),
])
def test_dig_returns_none_rather_than_raising(args):
    assert dig({"a": {"b": {"c": 1}}}, *args) is None


def test_dig_copes_with_a_non_dict_midway():
    assert dig({"a": "a string"}, "a", "b") is None


def test_unwrap_looks_past_a_visibility_wrapper():
    inner = mk.tweet("1", "hello")
    assert unwrap(mk.with_visibility(inner))["rest_id"] == "1"


def test_unwrap_leaves_a_plain_tweet_alone():
    t = mk.tweet("1", "hello")
    assert unwrap(t) is t


def test_unwrap_rejects_a_type_that_is_not_a_tweet():
    """Checking __typename BEFORE unwrapping is the point: a tombstone is
    refused outright rather than unwrapped into something half-formed."""
    stone = {"__typename": "TweetTombstone", "tombstone": {"text": {}}}
    assert unwrap(stone) is None
    assert unwrap({"__typename": "TweetUnavailable"}) is None


def test_unwrap_accepts_a_result_with_no_typename():
    """The tweet inside a visibility wrapper carries no __typename at all."""
    bare = {"rest_id": "1", "legacy": {}}
    assert unwrap(bare) is bare


@pytest.mark.parametrize("value", [None, "a string", 42, []])
def test_unwrap_returns_none_for_anything_that_is_not_a_dict(value):
    assert unwrap(value) is None


def test_iter_entries_finds_entries_at_any_depth(bookmark_pages):
    found = [e for page in bookmark_pages for e in iter_entries(page)]
    assert len(found) == 10
    assert all(e["entryId"].startswith("tweet-") for e in found)


def test_iter_entries_does_not_yield_quoted_tweets():
    """A quoted tweet is nested inside an entry but is not an entry."""
    quoted = mk.tweet("222", "the original")
    page = mk.page([mk.entry(mk.tweet("111", "a reaction", quoted=quoted))])
    ids = [e["entryId"] for e in iter_entries(page)]
    assert ids == ["tweet-111"]


def test_iter_entries_ignores_cursors():
    page = mk.page([mk.entry(mk.tweet("111", "x"))], with_cursor=True)
    assert len(list(iter_entries(page))) == 1


# --------------------------------------------------------------------------
# What parse_entry skips
# --------------------------------------------------------------------------


def test_a_tombstone_is_skipped():
    assert parse_entry(mk.tombstone()) is None


def test_a_tweet_with_no_rest_id_is_skipped():
    t = mk.tweet("1", "x")
    del t["rest_id"]
    assert parsed(t) is None


def test_an_unavailable_tweet_is_skipped():
    t = mk.tweet("1", "x", typename="TweetUnavailable")
    del t["rest_id"]
    assert parsed(t) is None


def test_the_fixtures_contain_exactly_one_tombstone(bookmark_entries):
    skipped = [e for e in bookmark_entries if parse_entry(e) is None]
    assert len(skipped) == 1


# --------------------------------------------------------------------------
# The ordinary fields
# --------------------------------------------------------------------------


def test_the_plain_fields_come_through():
    row = parsed(mk.tweet("1000000000000000001", "A plain bookmarked post.",
                          handle="alice_dev", name="Alice Dev"))
    assert row["id"] == "1000000000000000001"
    assert row["text"] == "A plain bookmarked post."
    assert row["author_handle"] == "alice_dev"
    assert row["author_name"] == "Alice Dev"
    assert row["lang"] == "en"
    assert row["url"] == ("https://x.com/alice_dev/status/"
                          "1000000000000000001")


def test_the_sort_index_is_kept_as_save_order():
    row = parsed(mk.tweet("1", "x"), sort_index="1899999999999999999")
    assert row["save_order"] == "1899999999999999999"


def test_created_at_becomes_iso():
    row = parsed(mk.tweet("1", "x",
                          created_at="Mon Sep 29 10:00:00 +0000 2026"))
    assert row["posted_at"].startswith("2026-09-29T10:00:00")


@pytest.mark.parametrize("value", [None, ""])
def test_to_iso_returns_none_for_an_empty_value(value):
    assert to_iso(value) is None


def test_to_iso_raises_on_a_date_it_cannot_read():
    """Recording the real behaviour: it does not swallow a bad date. X has
    always sent this format, and a silent None would hide a format change."""
    with pytest.raises(ValueError):
        to_iso("not a date")


def test_to_iso_converts_to_utc():
    assert to_iso("Mon Sep 29 12:00:00 +0200 2026").startswith(
        "2026-09-29T10:00:00")


def test_a_reply_keeps_what_it_replies_to():
    row = parsed(mk.tweet("1", "x", reply_to="999"))
    assert row["in_reply_to_id"] == "999"


def test_the_payloads_own_flags_are_reported_not_trusted():
    row = parsed(mk.tweet("1", "x", favorited=True, bookmarked=False))
    assert row["x_favorited"] is True
    assert row["x_bookmarked"] is False


# --------------------------------------------------------------------------
# Text cleaning: the cases that caused real bugs
# --------------------------------------------------------------------------


def test_the_posts_own_media_link_is_stripped():
    """display_text_range marks where the text ends; the trailing t.co link
    to the post's own photo is not part of what was written."""
    body = "A post with a photo."
    row = parsed(mk.tweet("1", f"{body} https://t.co/mediaLINK",
                          display_range=[0, len(body)],
                          media=[mk.media_item()]))
    assert row["text"] == body
    assert "t.co" not in row["text"]


def test_an_author_link_is_expanded_to_its_destination():
    row = parsed(mk.tweet("1", "Worth reading: https://t.co/shortLINK",
                          urls=[mk.url_item()]))
    assert row["text"] == "Worth reading: https://example.com/an-article"
    assert "t.co" not in row["text"]


def test_no_stored_text_contains_a_tco_link(bookmark_entries, like_entries):
    for e in bookmark_entries + like_entries:
        row = parse_entry(e)
        if row and row["text"]:
            assert "t.co" not in row["text"], row["id"]


def test_links_are_collected_and_deduplicated():
    same = mk.url_item("https://t.co/a", "https://example.com/one")
    again = mk.url_item("https://t.co/b", "https://example.com/one")
    other = mk.url_item("https://t.co/c", "https://example.com/two")
    row = parsed(mk.tweet("1", "two links", urls=[same, again, other]))
    assert row["links"] == ["https://example.com/one",
                            "https://example.com/two"]


def test_clean_text_on_a_tweet_with_nothing_in_it():
    assert clean_text(mk.tweet("1", "")) == ""


def test_escaped_entities_are_unescaped():
    row = parsed(mk.tweet("1", "Tom &amp; Jerry &lt;3"))
    assert row["text"] == "Tom & Jerry <3"


# --------------------------------------------------------------------------
# Media
# --------------------------------------------------------------------------


def test_media_is_listed_as_kind_and_url():
    row = parsed(mk.tweet("1", "x", media=[mk.media_item("photo")]))
    assert row["media"] == [("photo",
                             "https://pbs.twimg.com/media/AAA.jpg")]


def test_a_video_is_recorded_as_a_video():
    row = parsed(mk.tweet("1", "x", media=[mk.media_item("video")]))
    assert row["media"][0][0] == "video"


def test_several_pictures_keep_their_order():
    items = [mk.media_item("photo", f"https://pbs.twimg.com/media/{n}.jpg")
             for n in "ABC"]
    row = parsed(mk.tweet("1", "x", media=items))
    assert [u.rsplit("/", 1)[-1] for _k, u in row["media"]] == [
        "A.jpg", "B.jpg", "C.jpg"]


def test_no_media_is_an_empty_list():
    assert parsed(mk.tweet("1", "x"))["media"] == []


# --------------------------------------------------------------------------
# Quoted posts, cards, articles, translations, notes
# --------------------------------------------------------------------------


def test_a_quoted_post_is_captured_separately():
    quoted = mk.tweet("222", "The original claim, stated at length.",
                      handle="bob_writes")
    row = parsed(mk.tweet("111", "This.", quoted=quoted))
    assert row["text"] == "This."
    assert row["quoted_id"] == "222"
    assert row["quoted_author"] == "bob_writes"
    assert row["quoted_text"] == "The original claim, stated at length."


def test_no_quoted_post_leaves_those_fields_empty():
    row = parsed(mk.tweet("1", "x"))
    assert row["quoted_id"] is None
    assert row["quoted_text"] is None


def test_a_card_title_and_description_are_read():
    row = parsed(mk.tweet("1", "x", card=mk.card("A Linked Page", "About it")))
    assert row["card_title"] == "A Linked Page"
    assert row["card_description"] == "About it"


def test_an_article_title_and_preview_are_read():
    row = parsed(mk.tweet("1", "", article=mk.article("A Long Read", "Opens")))
    assert row["article_title"] == "A Long Read"
    assert row["article_preview"] == "Opens"


def test_an_article_in_a_quoted_post_is_used_as_the_quoted_text():
    quoted = mk.tweet("222", "", article=mk.article("Quoted Long Read", "Op"))
    row = parsed(mk.tweet("111", "Read this", quoted=quoted))
    assert row["article_title"] == "Quoted Long Read"
    assert "Quoted Long Read" in row["quoted_text"]


def test_an_english_translation_is_kept():
    row = parsed(mk.tweet(
        "1", "Bu bir Türkçe gönderidir.", lang="tr",
        translation={"destination_language": "en",
                     "translation": "This is a post in Turkish."}))
    assert row["translation_en"] == "This is a post in Turkish."


def test_a_translation_into_another_language_is_ignored():
    """Only English translations are stored; the index is English."""
    row = parsed(mk.tweet(
        "1", "A post", translation={"destination_language": "fr",
                                    "translation": "Un message"}))
    assert row["translation_en"] is None


def test_no_translation_block_at_all():
    assert parsed(mk.tweet("1", "x"))["translation_en"] is None


def test_a_community_note_is_kept():
    row = parsed(mk.tweet("1", "A confident but wrong claim.",
                          community_note="Context: the paper says otherwise."))
    assert row["community_note"] == "Context: the paper says otherwise."


def test_a_visibility_wrapped_tweet_parses_normally():
    row = parse_entry(mk.entry(mk.with_visibility(
        mk.tweet("1", "A post with reply limits.", handle="alice_dev"))))
    assert row["id"] == "1"
    assert row["text"] == "A post with reply limits."
    assert row["author_handle"] == "alice_dev"


def test_a_retweet_stores_the_original():
    original = mk.tweet("222", "The original post", handle="bob_writes")
    row = parsed(mk.tweet("111", "RT @bob_writes: The original post",
                          retweet_of=original))
    assert row["text"] == "The original post"
    assert row["author_handle"] == "bob_writes"


# --------------------------------------------------------------------------
# The whole synthetic set
# --------------------------------------------------------------------------


def test_every_parsed_row_has_the_fields_the_database_needs(
        bookmark_entries, like_entries):
    required = ["id", "url", "author_handle", "text", "media", "links"]
    for e in bookmark_entries + like_entries:
        row = parse_entry(e)
        if row:
            for key in required:
                assert key in row, f"{row['id']} is missing {key}"
            assert row["id"] and row["url"]


def test_one_tweet_appears_in_both_feeds(bookmark_entries, like_entries):
    ids = lambda es: {parse_entry(e)["id"] for e in es   # noqa: E731
                      if parse_entry(e)}
    overlap = ids(bookmark_entries) & ids(like_entries)
    assert overlap == {"1000000000000000009"}
