"""Tests for tweet_index/embed.py and embed_all.py's planning logic.

Most of these need no model at all: assembling the text, hashing it, deciding
what to re-embed, and the cosine search are all plain Python and NumPy. One
test at the end loads the real model to check that a vector really comes back
unit length and that similar sentences score higher than unrelated ones.
"""

import numpy as np
import pytest

import embed_all
from tweet_index.db import connect, count
from tweet_index.embed import (DOC_PARTS, MAX_CHARS, SOURCES, doc_text,
                               embed_docs,
                               embed_query, from_blob, load_matrix, search,
                               text_hash, to_blob)

MODEL = "BAAI/bge-small-en-v1.5"
PREFIX = "Represent this sentence for searching relevant passages: "


def row(**kwargs):
    """A stand-in for a database row: every doc part, empty unless given."""
    return {part: kwargs.get(part) for part in DOC_PARTS}


# ---------------------------------------------------------------------------
# doc_text: which text gets embedded
# ---------------------------------------------------------------------------

def test_doc_text_joins_the_parts_in_order():
    text = doc_text(row(text="the post", quoted_text="the quote",
                        translation_en="the translation",
                        card_title="card title",
                        card_description="card description",
                        article_title="article title",
                        article_preview="article preview"))
    assert text.split("\n") == ["the post", "the quote", "the translation",
                               "card title", "card description",
                               "article title", "article preview"]


def test_doc_text_skips_missing_parts():
    assert doc_text(row(text="only this")) == "only this"


def test_doc_text_skips_empty_and_whitespace_parts():
    text = doc_text(row(text="first", quoted_text="", card_title="   ",
                        article_title="last"))
    assert text == "first\nlast"


def test_doc_text_strips_each_part():
    assert doc_text(row(text="  padded  ", card_title="\ntitle\n")) == \
        "padded\ntitle"


def test_doc_text_is_empty_for_a_tweet_with_no_text():
    """The 241 image-only posts. embed_all skips these rather than embed "".''"""
    assert doc_text(row()) == ""
    assert doc_text(row(text="", quoted_text=None)) == ""


def test_doc_text_truncates_long_posts():
    text = doc_text(row(text="x" * 5000))
    assert len(text) == MAX_CHARS


def test_doc_text_truncation_keeps_the_opening():
    text = doc_text(row(text="the important opening. " + "filler " * 1000))
    assert text.startswith("the important opening.")
    assert len(text) == MAX_CHARS


def test_doc_text_accepts_a_sqlite_row(tmp_path):
    """doc_text indexes by column name, so a real sqlite3.Row must work.

    `context` lives in enrichment rather than tweets, so the query joins the
    two exactly as embed_all.plan does.
    """
    conn = connect(tmp_path / "t.db")
    with conn:
        conn.execute(
            "INSERT INTO tweets (id, url, text, card_title, first_seen, raw_file)"
            " VALUES ('1','u','body','a card','now','f.json')")
    columns = ", ".join(f"{SOURCES[c]}.{c} AS {c}" for c in DOC_PARTS)
    stored = conn.execute(
        f"SELECT {columns} FROM tweets t"
        f" LEFT JOIN enrichment en ON en.tweet_id = t.id"
        f" LEFT JOIN (SELECT tweet_id,"
        f"                   GROUP_CONCAT(ocr_text, char(10)) AS image_text"
        f"            FROM media WHERE TRIM(COALESCE(ocr_text,'')) <> ''"
        f"            GROUP BY tweet_id) im ON im.tweet_id = t.id"
        f" LEFT JOIN (SELECT tweet_id,"
        f"                   GROUP_CONCAT(title, char(10)) AS link_titles"
        f"            FROM links WHERE TRIM(COALESCE(title,'')) <> ''"
        f"            GROUP BY tweet_id) lk ON lk.tweet_id = t.id").fetchone()
    assert doc_text(stored) == "body\na card"
    conn.close()


def test_doc_text_includes_image_text_last():
    """Section 9 puts image text after the enrichment summary."""
    text = doc_text(row(text="a post", context="what it is about",
                        image_text="words in the picture"))
    assert text.split("\n") == ["a post", "what it is about",
                                "words in the picture"]


def test_a_tweet_with_only_image_text_is_embeddable():
    """An image-only post becomes searchable once its picture is read."""
    assert doc_text(row(image_text="SALE 50% OFF\nA shop window.")) == \
        "SALE 50% OFF\nA shop window."


def test_doc_text_includes_the_enrichment_summary():
    """Section 9 puts the summary after the tweet's own text."""
    text = doc_text(row(text="the post", context="what it is about"))
    assert text == "the post\nwhat it is about"


def test_doc_text_summary_comes_last():
    text = doc_text(row(text="post", quoted_text="quote",
                        article_title="article", context="summary"))
    assert text.split("\n") == ["post", "quote", "article", "summary"]


def test_a_tweet_with_only_a_summary_is_embeddable():
    """An image-only post becomes searchable once it has a summary."""
    assert doc_text(row(context="a football photo from a fan account")) == \
        "a football photo from a fan account"


def test_an_empty_summary_does_not_stop_embedding():
    """The tweet's own text still carries it; only both being empty skips it."""
    assert doc_text(row(text="still has its own words", context="")) == \
        "still has its own words"


def test_no_text_and_no_summary_is_skipped():
    assert doc_text(row(text="", context="")) == ""
    assert doc_text(row()) == ""


# ---------------------------------------------------------------------------
# text_hash: how reruns know what changed
# ---------------------------------------------------------------------------

def test_hash_is_stable():
    assert text_hash("some text") == text_hash("some text")


def test_hash_changes_with_the_text():
    assert text_hash("some text") != text_hash("some text!")


def test_hash_is_hex_sha256():
    digest = text_hash("anything")
    assert len(digest) == 64
    int(digest, 16)             # raises if it is not hexadecimal


def test_hash_handles_non_ascii():
    assert text_hash("我が家のおすすめインテリア") != text_hash("something else")


# ---------------------------------------------------------------------------
# blobs: vectors in and out of SQLite
# ---------------------------------------------------------------------------

def test_blob_round_trip():
    vector = np.arange(384, dtype=np.float32) / 384
    assert np.array_equal(from_blob(to_blob(vector)), vector)


def test_blob_is_float32():
    assert len(to_blob(np.zeros(384, dtype=np.float32))) == 384 * 4


# ---------------------------------------------------------------------------
# embed_all.plan: what gets embedded, what is skipped, what is left alone
# ---------------------------------------------------------------------------

def insert(conn, tweet_id, **parts):
    columns = ["id", "url", "first_seen", "raw_file"] + list(parts)
    values = [tweet_id, f"https://x.com/a/status/{tweet_id}", "now", "f.json"]
    values += list(parts.values())
    if "text" not in parts:                 # text is NOT NULL in the schema
        columns.append("text")
        values.append("")
    marks = ",".join("?" * len(values))
    with conn:
        conn.execute(f"INSERT INTO tweets ({','.join(columns)})"
                     f" VALUES ({marks})", values)


@pytest.fixture
def conn(tmp_path):
    c = connect(tmp_path / "plan.db")
    yield c
    c.close()


def test_plan_embeds_everything_on_a_fresh_database(conn):
    insert(conn, "1", text="first")
    insert(conn, "2", text="second")
    todo, skipped, unchanged = embed_all.plan(conn, MODEL, False)
    assert sorted(t[0] for t in todo) == ["1", "2"]
    assert skipped == []
    assert unchanged == 0


def test_plan_skips_tweets_with_no_text(conn):
    insert(conn, "1", text="has text")
    insert(conn, "2")                          # image-only post
    insert(conn, "3", text="   ")              # whitespace only
    todo, skipped, unchanged = embed_all.plan(conn, MODEL, False)
    assert [t[0] for t in todo] == ["1"]
    assert sorted(skipped) == ["2", "3"]


def test_plan_uses_the_other_columns_when_text_is_empty(conn):
    """A bare-link post can still be searchable through its card or article."""
    insert(conn, "1", card_title="Between the spreadsheets")
    todo, skipped, _ = embed_all.plan(conn, MODEL, False)
    assert [t[0] for t in todo] == ["1"]
    assert skipped == []


def test_plan_leaves_unchanged_tweets_alone(conn):
    insert(conn, "1", text="first")
    todo, _, _ = embed_all.plan(conn, MODEL, False)
    embed_all.store(conn, todo, np.zeros((1, 384), dtype=np.float32), MODEL)

    todo2, skipped, unchanged = embed_all.plan(conn, MODEL, False)
    assert todo2 == []
    assert unchanged == 1


def test_plan_re_embeds_when_the_text_changes(conn):
    insert(conn, "1", text="first")
    todo, _, _ = embed_all.plan(conn, MODEL, False)
    embed_all.store(conn, todo, np.zeros((1, 384), dtype=np.float32), MODEL)

    with conn:
        conn.execute("UPDATE tweets SET text='rewritten' WHERE id='1'")
    todo2, _, unchanged = embed_all.plan(conn, MODEL, False)
    assert [t[0] for t in todo2] == ["1"]
    assert unchanged == 0


def test_plan_re_embeds_when_a_new_column_is_filled(conn):
    """Phase 4 adding OCR text must trigger a re-embed (Section 9)."""
    insert(conn, "1", text="a screenshot")
    todo, _, _ = embed_all.plan(conn, MODEL, False)
    embed_all.store(conn, todo, np.zeros((1, 384), dtype=np.float32), MODEL)

    with conn:
        conn.execute("UPDATE tweets SET card_title='now with a card'"
                     " WHERE id='1'")
    todo2, _, _ = embed_all.plan(conn, MODEL, False)
    assert [t[0] for t in todo2] == ["1"]


def test_plan_re_embeds_when_the_model_changes(conn):
    insert(conn, "1", text="first")
    todo, _, _ = embed_all.plan(conn, MODEL, False)
    embed_all.store(conn, todo, np.zeros((1, 384), dtype=np.float32), MODEL)

    todo2, _, unchanged = embed_all.plan(conn, "some-other-model", False)
    assert [t[0] for t in todo2] == ["1"]
    assert unchanged == 0


def test_plan_redo_all_ignores_the_hash(conn):
    insert(conn, "1", text="first")
    todo, _, _ = embed_all.plan(conn, MODEL, False)
    embed_all.store(conn, todo, np.zeros((1, 384), dtype=np.float32), MODEL)

    todo2, _, unchanged = embed_all.plan(conn, MODEL, True)
    assert [t[0] for t in todo2] == ["1"]
    assert unchanged == 0


def test_plan_accounts_for_every_tweet(conn):
    for i in range(5):
        insert(conn, str(i), text=f"tweet {i}")
    insert(conn, "empty")
    todo, skipped, unchanged = embed_all.plan(conn, MODEL, False)
    assert len(todo) + len(skipped) + unchanged == count(conn, "tweets")


def test_store_upserts_rather_than_duplicating(conn):
    insert(conn, "1", text="first")
    todo, _, _ = embed_all.plan(conn, MODEL, False)
    embed_all.store(conn, todo, np.zeros((1, 384), dtype=np.float32), MODEL)
    embed_all.store(conn, todo, np.ones((1, 384), dtype=np.float32), MODEL)
    assert count(conn, "embeddings") == 1


def test_store_records_the_model_and_hash(conn):
    insert(conn, "1", text="first")
    todo, _, _ = embed_all.plan(conn, MODEL, False)
    embed_all.store(conn, todo, np.zeros((1, 384), dtype=np.float32), MODEL)
    stored = conn.execute("SELECT model_used, text_hash FROM embeddings").fetchone()
    assert stored["model_used"] == MODEL
    assert stored["text_hash"] == text_hash("first")


# ---------------------------------------------------------------------------
# search: cosine over the matrix
# ---------------------------------------------------------------------------

def unit(*values):
    v = np.array(values, dtype=np.float32)
    return v / np.linalg.norm(v)


def test_search_orders_by_cosine():
    ids = ["a", "b", "c"]
    matrix = np.vstack([unit(1, 0), unit(0, 1), unit(1, 1)])
    results = search(unit(1, 0), ids, matrix, 3)
    assert [tid for tid, _ in results] == ["a", "c", "b"]
    assert results[0][1] == pytest.approx(1.0, abs=1e-6)


def test_search_respects_k():
    ids = ["a", "b", "c"]
    matrix = np.vstack([unit(1, 0), unit(0, 1), unit(1, 1)])
    assert len(search(unit(1, 0), ids, matrix, 2)) == 2


def test_search_k_larger_than_the_corpus():
    ids = ["a"]
    matrix = np.vstack([unit(1, 0)])
    assert len(search(unit(1, 0), ids, matrix, 50)) == 1


def test_search_on_an_empty_index():
    assert search(unit(1, 0), [], np.zeros((0, 2), dtype=np.float32), 5) == []


def test_search_scores_are_plain_floats():
    """They go into JSON in the query log, so they cannot be numpy types."""
    results = search(unit(1, 0), ["a"], np.vstack([unit(1, 0)]), 1)
    assert type(results[0][1]) is float


def test_load_matrix_round_trips_vectors(conn):
    insert(conn, "1", text="first")
    insert(conn, "2", text="second")
    vectors = np.vstack([unit(*([1] + [0] * 383)), unit(*([0, 1] + [0] * 382))])
    todo, _, _ = embed_all.plan(conn, MODEL, False)
    embed_all.store(conn, sorted(todo), vectors, MODEL)

    ids, matrix = load_matrix(conn)
    assert ids == ["1", "2"]
    assert matrix.shape == (2, 384)
    assert matrix.dtype == np.float32


def test_load_matrix_on_an_empty_database(conn):
    ids, matrix = load_matrix(conn)
    assert ids == []
    assert matrix.shape == (0, 384)


# ---------------------------------------------------------------------------
# the real model: one test, because loading it costs a couple of seconds
# ---------------------------------------------------------------------------

def test_the_model_returns_unit_vectors_and_sensible_similarities():
    texts = ["a workout for building core strength",
             "how to price a software startup"]
    vectors = embed_docs(texts, MODEL)
    assert vectors.shape == (2, 384)
    assert vectors.dtype == np.float32
    # Unit length is what makes the dot product a cosine.
    assert np.allclose(np.linalg.norm(vectors, axis=1), 1.0, atol=1e-5)

    query = embed_query("core exercises", MODEL, PREFIX)
    assert np.isclose(np.linalg.norm(query), 1.0, atol=1e-5)

    sims = vectors @ query
    assert sims[0] > sims[1], "the workout tweet should win on 'core exercises'"
