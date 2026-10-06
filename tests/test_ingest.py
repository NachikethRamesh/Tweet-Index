"""Ingest: capture files into the database, idempotently.

Every test builds its own database in tmp_path and its own capture files from
`make_fixtures.py`. Nothing here touches a real database or a real capture.
"""

import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import ingest                                        # noqa: E402
from tests import make_fixtures as mk                 # noqa: E402
from tweet_index.db import connect, count             # noqa: E402


@pytest.fixture
def db(tmp_path):
    conn = connect(tmp_path / "test.db")
    yield conn
    conn.close()


@pytest.fixture
def logs(tmp_path):
    return tmp_path / "logs"


def ingest_both(conn, captures, logs):
    bookmarks, likes = captures
    return (ingest.ingest_file(conn, bookmarks, logs),
            ingest.ingest_file(conn, likes, logs))


# --------------------------------------------------------------------------
# A first run
# --------------------------------------------------------------------------


def test_the_tables_exist_on_a_fresh_database(db):
    for table in ("tweets", "media", "links", "enrichment", "embeddings",
                  "tweet_judgments", "queries", "ingested_files"):
        assert count(db, table) == 0


def test_ingesting_both_feeds_stores_every_parseable_tweet(db, captures, logs):
    book, like = ingest_both(db, captures, logs)
    # 9 of 10 bookmark entries parse (one tombstone), 4 of 4 likes, and one
    # tweet is in both feeds, so 12 distinct rows.
    assert book["seen"] == 9
    assert book["skipped"] == 1
    assert like["seen"] == 4
    assert like["skipped"] == 0
    assert count(db, "tweets") == 12


def test_the_tombstone_is_counted_and_logged(db, captures, logs):
    book, _ = ingest_both(db, captures, logs)
    assert book["skipped"] == 1
    assert "TweetTombstone" in book["reasons"]
    assert (logs / "ingest.log").exists()


def test_the_feed_decides_the_flags(db, captures, logs):
    ingest_both(db, captures, logs)
    row = db.execute(
        "SELECT is_like, is_bookmark FROM tweets WHERE id = ?",
        ("1000000000000000001",)).fetchone()
    assert (row["is_bookmark"], row["is_like"]) == (1, 0)
    row = db.execute(
        "SELECT is_like, is_bookmark FROM tweets WHERE id = ?",
        ("2000000000000000001",)).fetchone()
    assert (row["is_bookmark"], row["is_like"]) == (0, 1)


def test_a_tweet_in_both_feeds_ends_up_with_both_flags(db, captures, logs):
    ingest_both(db, captures, logs)
    rows = db.execute(
        "SELECT id FROM tweets WHERE is_like = 1 AND is_bookmark = 1"
    ).fetchall()
    assert [r["id"] for r in rows] == ["1000000000000000009"]


def test_media_rows_are_created(db, captures, logs):
    ingest_both(db, captures, logs)
    rows = db.execute(
        "SELECT tweet_id, kind, remote_url FROM media ORDER BY id").fetchall()
    assert len(rows) == 2                      # one photo, one video
    assert {r["kind"] for r in rows} == {"photo", "video"}
    assert all(r["remote_url"].startswith("https://") for r in rows)


def test_link_rows_are_created_with_the_expanded_url(db, captures, logs):
    ingest_both(db, captures, logs)
    rows = db.execute("SELECT url FROM links").fetchall()
    assert [r["url"] for r in rows] == ["https://example.com/an-article"]


def test_the_file_is_recorded_as_ingested(db, captures, logs):
    ingest_both(db, captures, logs)
    rows = db.execute(
        "SELECT filename, kind, entries, skipped FROM ingested_files"
        " ORDER BY filename").fetchall()
    assert len(rows) == 2
    kinds = {r["kind"] for r in rows}
    assert kinds == {"bookmarks", "likes"}


# --------------------------------------------------------------------------
# Running it again
# --------------------------------------------------------------------------


def test_the_same_file_twice_adds_nothing(db, captures, logs):
    ingest_both(db, captures, logs)
    before = count(db, "tweets")

    again = ingest.ingest_file(db, captures[0], logs)

    assert again["skipped_file"] is True
    assert again["new"] == 0
    assert count(db, "tweets") == before


def test_a_second_capture_of_the_same_tweets_adds_no_rows(db, tmp_path, logs):
    """A fresh capture a week later overlaps heavily with the last one."""
    first = mk.write_captures(tmp_path / "run1")
    ingest.ingest_file(db, first[0], logs)
    before = count(db, "tweets")

    # The same content, under a different filename.
    second = mk.write_captures(tmp_path / "run2")
    renamed = second[0].with_name("capture_bookmarks_2026-10-06T10-00-00Z.json")
    second[0].rename(renamed)
    result = ingest.ingest_file(db, renamed, logs)

    assert result["seen"] == 9              # all parsed again
    assert result["new"] == 0               # none of them new
    assert count(db, "tweets") == before


def test_first_seen_and_raw_file_are_not_overwritten(db, tmp_path, logs):
    first = mk.write_captures(tmp_path / "run1")
    ingest.ingest_file(db, first[0], logs)
    original = db.execute(
        "SELECT first_seen, raw_file FROM tweets WHERE id = ?",
        ("1000000000000000001",)).fetchone()

    second = mk.write_captures(tmp_path / "run2")
    renamed = second[0].with_name("capture_bookmarks_2026-10-06T10-00-00Z.json")
    second[0].rename(renamed)
    ingest.ingest_file(db, renamed, logs)

    after = db.execute(
        "SELECT first_seen, raw_file FROM tweets WHERE id = ?",
        ("1000000000000000001",)).fetchone()
    assert after["first_seen"] == original["first_seen"]
    assert after["raw_file"] == original["raw_file"]


def test_flags_are_or_ed_not_replaced(db, tmp_path, logs):
    """Ingesting likes after bookmarks must not clear is_bookmark."""
    captures = mk.write_captures(tmp_path / "raw")
    ingest.ingest_file(db, captures[0], logs)          # bookmarks
    row = db.execute("SELECT is_like, is_bookmark FROM tweets WHERE id = ?",
                     ("1000000000000000009",)).fetchone()
    assert (row["is_bookmark"], row["is_like"]) == (1, 0)

    ingest.ingest_file(db, captures[1], logs)          # likes
    row = db.execute("SELECT is_like, is_bookmark FROM tweets WHERE id = ?",
                     ("1000000000000000009",)).fetchone()
    assert (row["is_bookmark"], row["is_like"]) == (1, 1)


def test_media_and_links_are_not_duplicated_on_a_second_pass(db, tmp_path,
                                                             logs):
    first = mk.write_captures(tmp_path / "run1")
    ingest.ingest_file(db, first[0], logs)
    media_before = count(db, "media")
    links_before = count(db, "links")

    second = mk.write_captures(tmp_path / "run2")
    renamed = second[0].with_name("capture_bookmarks_2026-10-06T10-00-00Z.json")
    second[0].rename(renamed)
    ingest.ingest_file(db, renamed, logs)

    assert count(db, "media") == media_before
    assert count(db, "links") == links_before


# --------------------------------------------------------------------------
# Odd input
# --------------------------------------------------------------------------


def test_a_capture_with_no_entries_is_harmless(db, tmp_path, logs):
    path = tmp_path / "capture_bookmarks_empty.json"
    path.write_text(json.dumps(mk.capture([mk.page([])], "bookmarks")),
                    encoding="utf-8")
    result = ingest.ingest_file(db, path, logs)
    assert result["seen"] == 0
    assert count(db, "tweets") == 0


def test_a_capture_of_only_a_tombstone(db, tmp_path, logs):
    page = {"data": {"bookmark_timeline_v2": {"timeline": {"instructions": [
        {"type": "TimelineAddEntries", "entries": [mk.tombstone()]}]}}}}
    path = tmp_path / "capture_bookmarks_stone.json"
    path.write_text(json.dumps(mk.capture([page], "bookmarks")),
                    encoding="utf-8")
    result = ingest.ingest_file(db, path, logs)
    assert result["seen"] == 0
    assert result["skipped"] == 1
    assert count(db, "tweets") == 0


def test_the_backup_copy_is_made(tmp_path):
    captures = mk.write_captures(tmp_path / "raw")
    backup = tmp_path / "backup"
    copied = ingest.back_up_raw(captures[0], backup)
    assert copied is not None
    assert (backup / captures[0].name).exists()
    assert (backup / captures[0].name).stat().st_size == \
        captures[0].stat().st_size


def test_the_backup_is_not_remade_if_it_is_already_there(tmp_path):
    captures = mk.write_captures(tmp_path / "raw")
    backup = tmp_path / "backup"
    ingest.back_up_raw(captures[0], backup)
    assert ingest.back_up_raw(captures[0], backup) is None


def test_no_backup_directory_configured_is_not_an_error(tmp_path):
    captures = mk.write_captures(tmp_path / "raw")
    assert ingest.back_up_raw(captures[0], None) is None


def test_last_run_reports_what_happened(db, captures, logs):
    ingest.LAST_RUN.clear()
    book, like = ingest_both(db, captures, logs)
    # ingest_file alone does not fill LAST_RUN; main() does. Check the
    # per-file numbers it returns instead, which is what main sums.
    assert book["new"] + like["new"] == 12
    assert book["seen"] + like["seen"] == 13     # one tweet counted twice
