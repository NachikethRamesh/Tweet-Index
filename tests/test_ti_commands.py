"""Tests for the ti.py shell's /open, /more, /status and /history.

Everything builds its own database in tmp_path; nothing opens data/index.db.
The searching itself is not exercised -- these commands all work from state
the shell already holds, or from the queries table, and that is the point of
testing them separately.
"""

import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import ti                                                    # noqa: E402
from tweet_index import spend                                 # noqa: E402
from tweet_index.db import connect                            # noqa: E402


@pytest.fixture
def scratch(tmp_path, monkeypatch):
    db = tmp_path / "scratch.db"
    conn = connect(db)
    conn.execute(
        "INSERT INTO tweets (id, url, author_handle, text, is_like,"
        " is_bookmark, first_seen, raw_file) VALUES"
        " ('1', 'https://x.com/a/status/1', 'a', 'a core workout', 1, 0,"
        "  '2026-10-05T00:00:00', 'f.json')")
    conn.execute(
        "INSERT INTO tweets (id, url, author_handle, text, is_like,"
        " is_bookmark, first_seen, raw_file) VALUES"
        " ('2', 'https://x.com/b/status/2', 'b', 'both feeds', 1, 1,"
        "  '2026-10-05T00:00:00', 'f.json')")
    conn.execute(
        "INSERT INTO ingested_files (filename, kind, ingested_at, entries,"
        " skipped) VALUES ('f.json', 'likes', '2026-10-05T12:00:00', 2, 0)")
    conn.commit()
    conn.close()

    downloads = tmp_path / "Downloads"
    downloads.mkdir()
    monkeypatch.setattr(ti, "model", lambda name: None)
    monkeypatch.setattr(ti, "load_matrix", lambda conn, table: ([], None))

    shell = ti.Shell(db=str(db))
    shell.cfg["paths"]["downloads"] = str(downloads)
    return shell, downloads, db


def fake_row(tid, handle, url):
    return {"id": tid, "author_handle": handle, "url": url,
            "text": f"text of {tid}", "posted_at": "2026-01-01T00:00:00",
            "context": "", "is_like": 1, "is_bookmark": 0}


def fake_result(tid, handle, final=0.5):
    url = f"https://x.com/{handle}/status/{tid}"
    return {"row": fake_row(tid, handle, url),
            "cosine": 0.5, "relevance": 3.0, "confidence": 0.9,
            "substance": 1.0, "substance_note": "judged",
            "band": "top", "final": final}


@pytest.fixture
def searched(scratch):
    """A shell holding a last search of 12 results, 8 of them printed."""
    shell, _downloads, _db = scratch
    shell.last = {
        "topic": "core workout",
        "results": [fake_result(str(i), f"author{i}", 1.0 - i / 100)
                    for i in range(1, 13)],
        "shown": 8,
        "kind": "ask",
    }
    return shell


@pytest.fixture
def no_browser(monkeypatch):
    """Capture what /open would have opened instead of opening it."""
    opened = []
    monkeypatch.setattr(ti.webbrowser, "open", opened.append)
    return opened


# --------------------------------------------------------------------------
# /open
# --------------------------------------------------------------------------


def test_open_without_a_search(scratch, capsys):
    shell, _d, _db = scratch
    shell.do_open("1")
    assert "Nothing searched yet" in capsys.readouterr().out


def test_open_opens_the_right_url(searched, capsys, no_browser):
    searched.do_open("3")
    assert no_browser == ["https://x.com/author3/status/3"]
    assert "@author3" in capsys.readouterr().out


def test_open_reaches_a_result_that_was_never_printed(searched, no_browser):
    """Only 8 were shown, but all 12 are addressable."""
    searched.do_open("11")
    assert no_browser == ["https://x.com/author11/status/11"]


@pytest.mark.parametrize("arg", ["0", "13", "99", "-1"])
def test_open_out_of_range(searched, capsys, no_browser, arg):
    searched.do_open(arg)
    out = capsys.readouterr().out
    assert "no result" in out and "12" in out
    assert no_browser == []


@pytest.mark.parametrize("arg", ["", "   ", "abc", "3 4", "1.5"])
def test_open_with_a_bad_number(searched, capsys, no_browser, arg):
    searched.do_open(arg)
    assert "Which one?" in capsys.readouterr().out
    assert no_browser == []


def test_open_survives_a_browser_that_fails(searched, capsys, monkeypatch):
    def boom(url):
        raise RuntimeError("no browser here")
    monkeypatch.setattr(ti.webbrowser, "open", boom)
    searched.do_open("1")
    out = capsys.readouterr().out
    assert "Could not open a browser" in out
    # The link is still printed, so the result is not lost.
    assert "https://x.com/author1/status/1" in out


def test_open_a_result_with_no_link(scratch, capsys, no_browser):
    shell, _d, _db = scratch
    result = fake_result("1", "a")
    result["row"]["url"] = None
    shell.last = {"topic": "t", "results": [result], "shown": 1,
                  "kind": "ask"}
    shell.do_open("1")
    assert "no link stored" in capsys.readouterr().out
    assert no_browser == []


# --------------------------------------------------------------------------
# /more
# --------------------------------------------------------------------------


def test_more_shows_the_next_batch(searched, capsys):
    searched.do_more("")
    out = capsys.readouterr().out
    assert "9-12 of 12" in out
    assert "@author9" in out and "@author12" in out
    assert "@author8" not in out              # not repeated
    assert searched.last["shown"] == 12


def test_more_at_the_end(searched, capsys):
    searched.last["shown"] = 12
    searched.do_more("")
    assert "That is all 12 results" in capsys.readouterr().out


def test_more_without_a_search(scratch, capsys):
    shell, _d, _db = scratch
    shell.do_more("")
    assert "Nothing searched yet" in capsys.readouterr().out


def test_more_makes_no_api_calls(searched, monkeypatch):
    """The whole point: those results were scored by the original search."""
    def forbidden(*args, **kwargs):
        raise AssertionError("/more must not call out")
    monkeypatch.setattr(ti.ask, "rank_smart", forbidden)
    monkeypatch.setattr(ti.ask, "connect_jev", forbidden)
    searched.do_more("")


def test_more_batches_ten_at_a_time(scratch, capsys):
    shell, _d, _db = scratch
    shell.last = {"topic": "t",
                  "results": [fake_result(str(i), f"a{i}")
                              for i in range(1, 31)],
                  "shown": 5, "kind": "ask"}
    shell.do_more("")
    assert "6-15 of 30" in capsys.readouterr().out
    assert shell.last["shown"] == 15
    shell.do_more("")
    assert "16-25 of 30" in capsys.readouterr().out


def test_more_works_after_a_basic_search(scratch, capsys):
    shell, _d, _db = scratch
    shell.last = {"topic": "t",
                  "results": [{"row": fake_row(str(i), f"a{i}",
                                               f"https://x.com/a{i}"),
                               "cosine": 0.5} for i in range(1, 15)],
                  "shown": 10, "kind": "basic"}
    shell.do_more("")
    out = capsys.readouterr().out
    assert "11-14 of 14" in out
    assert "@a11" in out


# --------------------------------------------------------------------------
# /status
# --------------------------------------------------------------------------


def test_status_reports_counts_and_spend(scratch, capsys):
    shell, _d, _db = scratch
    shell.do_status("")
    out = capsys.readouterr().out
    assert "Tweets" in out
    assert "bookmarks" in out and "likes" in out and "in both" in out
    assert "2026-10-05 12:00" in out
    assert "Waiting       0 capture files" in out
    assert "Spend so far in" in out


def test_status_counts_both_feeds_separately(scratch, capsys):
    shell, _d, _db = scratch
    shell.do_status("")
    out = capsys.readouterr().out
    # Two tweets: one like only, one in both.
    assert "bookmarks   1" in out
    assert "likes       2" in out
    assert "in both     1" in out


def test_status_mentions_a_waiting_capture(scratch, capsys):
    shell, downloads, _db = scratch
    (downloads / "capture_likes_a.json").write_text("{}", encoding="utf-8")
    shell.do_status("")
    out = capsys.readouterr().out
    assert "Waiting       1 capture file" in out
    assert "/update" in out


def test_status_shows_spend_by_provider(scratch, capsys, monkeypatch,
                                        tmp_path):
    shell, _d, _db = scratch
    logs = tmp_path / "logs"
    monkeypatch.setattr(ti, "ROOT", tmp_path)
    (tmp_path / "data").mkdir(exist_ok=True)
    spend.record(logs, "openai", 0.0057, "images")
    spend.record(logs, "typesafe", 0.0009, "substance")
    monkeypatch.setattr(ti.spend, "month",
                        lambda conn, d: ({"openai": 0.0057,
                                          "typesafe": 0.0009}, 3))
    shell.do_status("")
    out = capsys.readouterr().out
    assert "openai" in out and "0.0057" in out
    assert "typesafe" in out and "0.0009" in out
    assert "0.0066" in out                    # the total
    assert "3 searches" in out


# --------------------------------------------------------------------------
# /history
# --------------------------------------------------------------------------


def log_search(db, when, topic, payload):
    conn = connect(db)
    conn.execute(
        "INSERT INTO queries (asked_at, topic, config, results, latency_ms,"
        " jev_tokens) VALUES (?,?,?,?,?,?)",
        (when, topic, "{}", payload, 700, 1000))
    conn.commit()
    conn.close()


def test_history_with_nothing_logged(scratch, capsys):
    shell, _d, _db = scratch
    shell.do_history("")
    assert "No searches recorded yet" in capsys.readouterr().out


def test_history_shows_date_topic_and_best_result(scratch, capsys):
    shell, _d, db = scratch
    log_search(db, "2026-10-01T09:30:00", "core workout",
               json.dumps([{"id": "1", "final": 0.4},
                           {"id": "2", "final": 0.9}]))
    log_search(db, "2026-10-02T11:00:00", "leadership habits",
               json.dumps([{"id": "1", "final": 0.7}]))
    shell.conn = connect(db)

    shell.do_history("")
    out = capsys.readouterr().out
    assert "2026-10-01 09:30" in out
    assert "core workout" in out and "leadership habits" in out
    # The best result, not the first in the list.
    assert "@b 0.900" in out
    # Newest first.
    assert out.index("leadership habits") < out.index("core workout")


def test_history_caps_at_ten(scratch, capsys):
    shell, _d, db = scratch
    for n in range(15):
        log_search(db, f"2026-10-05T09:{n:02d}:00", f"subject{n}", "[]")
    shell.conn = connect(db)
    shell.do_history("")
    out = capsys.readouterr().out
    assert len([ln for ln in out.splitlines() if "subject" in ln]) == 10
    assert "subject14" in out
    assert "subject0 " not in out


def test_history_survives_an_unreadable_blob(scratch, capsys):
    shell, _d, db = scratch
    log_search(db, "2026-10-05T09:00:00", "t", "not json at all")
    shell.conn = connect(db)
    shell.do_history("")
    assert "(unreadable)" in capsys.readouterr().out


def test_history_handles_an_empty_result_list(scratch, capsys):
    shell, _d, db = scratch
    log_search(db, "2026-10-05T09:00:00", "t", "[]")
    shell.conn = connect(db)
    shell.do_history("")
    assert "(nothing)" in capsys.readouterr().out


def test_history_names_a_tweet_no_longer_stored(scratch, capsys):
    shell, _d, db = scratch
    log_search(db, "2026-10-05T09:00:00", "t",
               json.dumps([{"id": "missing", "final": 0.5}]))
    shell.conn = connect(db)
    shell.do_history("")
    assert "(deleted)" in capsys.readouterr().out


# --------------------------------------------------------------------------
# Dispatch and the arrow keys
# --------------------------------------------------------------------------


@pytest.mark.parametrize("line,expected", [
    ("/open 3", "open 3"),
    ("/more", "more"),
    ("/status", "status"),
    ("/history", "history"),
    ("open 3", "open 3"),
    ("status", "status"),
    ("more", "more"),
    ("history", "history"),
    # A topic that begins with a command word is still a search.
    ("opening a restaurant in paris", "ask opening a restaurant in paris"),
    ("history of rome", "ask history of rome"),
    ("more cowbell", "ask more cowbell"),
    ("statusing things", "ask statusing things"),
])
def test_dispatch(line, expected):
    assert ti.Shell.precmd(None, line) == expected


def test_arrow_history_needs_rawinput():
    """use_rawinput False would read stdin directly and lose the arrows."""
    assert ti.Shell.use_rawinput is True


# --------------------------------------------------------------------------
# The spend ledger
# --------------------------------------------------------------------------


def test_ledger_attributes_each_stage_to_the_right_provider(tmp_path):
    """The faithfulness check is a Jev call, despite its label."""
    logs = tmp_path / "logs"
    spend.record_stages(logs, [
        {"stage": "images", "cost": 0.0057},
        {"stage": "summaries", "cost": 0.0037},
        {"stage": "substance", "cost": 0.0009},
        {"stage": "faithfulness", "cost": 0.0007},
        {"stage": "ingest", "cost": 0.0},
        {"stage": "links", "cost": 0.0},
    ])
    rows = spend.read_ledger(logs)
    got = {r["what"]: r["provider"] for r in rows}
    assert got == {"images": "openai", "summaries": "openai",
                   "substance": "typesafe", "faithfulness": "typesafe"}
    # Free stages are not recorded at all.
    assert "ingest" not in got and "links" not in got


def test_ledger_sums_by_provider_for_the_month(tmp_path):
    logs = tmp_path / "logs"
    spend.record(logs, "openai", 0.01, "images")
    spend.record(logs, "openai", 0.02, "summaries")
    spend.record(logs, "typesafe", 0.03, "substance")
    totals, searches = spend.month(None, logs)
    assert round(totals["openai"], 4) == 0.03
    assert round(totals["typesafe"], 4) == 0.03
    assert searches == 0


def test_ledger_ignores_other_months(tmp_path):
    logs = tmp_path / "logs"
    spend.record(logs, "openai", 0.01, "images")
    totals, _ = spend.month(None, logs, when="1999-01")
    assert totals == {}


def test_ledger_skips_a_torn_line(tmp_path):
    logs = tmp_path / "logs"
    spend.record(logs, "openai", 0.01, "images")
    with (logs / spend.LEDGER).open("a", encoding="utf-8") as fh:
        fh.write('{"at": "2026-10-05", "provi\n')      # a crash mid-write
    rows = spend.read_ledger(logs)
    assert len(rows) == 1


def test_ledger_never_raises_on_a_bad_directory(tmp_path):
    """A ledger is not worth losing a run for."""
    spend.record(tmp_path / "a" / "b" / "c", "openai", 0.01, "images")
    spend.record(logs_dir=tmp_path, provider="openai", usd=None, what="x")
    assert spend.read_ledger(tmp_path) == []


def test_ledger_reports_when_it_started(tmp_path):
    logs = tmp_path / "logs"
    assert spend.ledger_started(logs) is None
    spend.record(logs, "openai", 0.01, "images")
    assert spend.ledger_started(logs) is not None


def test_more_never_repeats_a_result_already_shown(scratch, capsys,
                                                   monkeypatch):
    """The bands are picked by band, not by position.

    A below-threshold result can outscore a maybe and sit between the bands
    in the raw score order. Slicing that raw list for /more reprinted the
    maybe. /ask therefore builds one display order: printed first, then the
    rest by score.
    """
    shell, _d, _db = scratch

    top = fake_result("t1", "topper", 0.90)
    top["band"] = "top"
    dropped = fake_result("d1", "dropped", 0.70)     # outscores the maybe
    dropped["band"] = "drop"
    maybe = fake_result("m1", "maybeish", 0.60)
    maybe["band"] = "maybe"

    monkeypatch.setattr(ti.ask, "connect_jev", lambda cfg: None)
    monkeypatch.setattr(shell, "log", lambda *a, **k: None)
    monkeypatch.setattr(ti.ask, "rank_smart", lambda *a, **k: {
        "corpus": 100, "candidates": [1, 2, 3],
        "scored": [1, 2, 3], "failures": [],
        "results": [top, dropped, maybe],        # raw score order
        "tops": [top], "maybes": [maybe],
        "jev_tokens": 10,
        "timings": {"stage 1 cosine": 0.0, "stage 2 jev": 0.0,
                    "stage 3 blend": 0.0},
    })

    shell.do_ask("anything")
    first = capsys.readouterr().out
    assert "@topper" in first and "@maybeish" in first
    assert "@dropped" not in first            # below the thresholds

    shell.do_more("")
    second = capsys.readouterr().out
    assert "@dropped" in second
    # The maybe was already shown; it must not come round again.
    assert "@maybeish" not in second


def test_open_numbers_match_what_was_displayed(scratch, capsys, monkeypatch,
                                               no_browser):
    """/open 2 must be the second thing on screen, not the second by score."""
    shell, _d, _db = scratch
    top = fake_result("t1", "topper", 0.90)
    top["band"] = "top"
    dropped = fake_result("d1", "dropped", 0.70)
    dropped["band"] = "drop"
    maybe = fake_result("m1", "maybeish", 0.60)
    maybe["band"] = "maybe"

    monkeypatch.setattr(ti.ask, "connect_jev", lambda cfg: None)
    monkeypatch.setattr(shell, "log", lambda *a, **k: None)
    monkeypatch.setattr(ti.ask, "rank_smart", lambda *a, **k: {
        "corpus": 100, "candidates": [1, 2, 3], "scored": [1, 2, 3],
        "failures": [], "results": [top, dropped, maybe],
        "tops": [top], "maybes": [maybe], "jev_tokens": 10,
        "timings": {"stage 1 cosine": 0.0, "stage 2 jev": 0.0,
                    "stage 3 blend": 0.0},
    })
    shell.do_ask("anything")
    capsys.readouterr()

    shell.do_open("2")                        # the maybe, shown as 2
    assert no_browser == ["https://x.com/maybeish/status/m1"]
