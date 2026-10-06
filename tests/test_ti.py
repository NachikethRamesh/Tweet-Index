"""Tests for the ti.py shell: command parsing, dispatch and /update.

Everything here builds its own database in tmp_path. Nothing opens
a real database.

The searching itself is not tested here -- that would need the embedding
model and a Jev key, and the ranking is already covered by test_ask.py. What
is tested is the shell around it: how a line becomes a command, what happens
with no captures waiting, and that a failure in a command does not end the
session.
"""

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import ti                                                    # noqa: E402
from tweet_index.db import connect                            # noqa: E402

# --------------------------------------------------------------------------
# split_n: the optional -n N on a topic
# --------------------------------------------------------------------------


@pytest.mark.parametrize("line,topic,n", [
    ("core workout", "core workout", None),
    ("core workout -n 8", "core workout", 8),
    ("-n 3 core workout", "core workout", 3),
    ("core workout --n 12", "core workout", 12),
    ("", "", None),
    ("   ", "", None),
    # A bad number is left in the topic rather than guessed at.
    ("core workout -n abc", "core workout -n abc", None),
    # -n with nothing after it is not a flag.
    ("core workout -n", "core workout -n", None),
    ('"quoted topic" -n 2', "quoted topic", 2),
    # 0 and negatives clamp to 1: showing no results is never what was meant.
    ("topic -n 0", "topic", 1),
    ("topic -n -5", "topic", 1),
])
def test_split_n(line, topic, n):
    assert ti.Shell.split_n(line) == (topic, n)


# --------------------------------------------------------------------------
# precmd: slashes, bare text, and the commands that need no slash
# --------------------------------------------------------------------------


@pytest.mark.parametrize("line,expected", [
    ("/ask core workout", "ask core workout"),
    ("/basic core workout", "basic core workout"),
    ("/update", "update"),
    ("/help", "help"),
    ("/quit", "quit"),
    # Bare text is a search.
    ("core workout", "ask core workout"),
    ("what did I save about evals", "ask what did I save about evals"),
    # Typed without the slash, these are still commands.
    ("update", "update"),
    ("help", "help"),
    ("quit", "quit"),
    ("basic core workout", "basic core workout"),
    # Blank stays blank.
    ("", ""),
    ("   ", ""),
    # A topic that merely begins with a command word is still a search.
    ("asking for a friend", "ask asking for a friend"),
    ("helpful leadership habits", "ask helpful leadership habits"),
])
def test_precmd(line, expected):
    assert ti.Shell.precmd(None, line) == expected


def test_precmd_leaves_a_slashed_topic_alone():
    """A slash means the first word is the command, whatever it is."""
    assert ti.Shell.precmd(None, "/ask /basic") == "ask /basic"


# --------------------------------------------------------------------------
# A shell on a scratch database
# --------------------------------------------------------------------------


@pytest.fixture
def scratch(tmp_path, monkeypatch):
    """A real but tiny database, and a Downloads folder we control."""
    db = tmp_path / "scratch.db"
    conn = connect(db)
    conn.execute(
        "INSERT INTO tweets (id, url, author_handle, text, is_like,"
        " is_bookmark, first_seen, raw_file) VALUES"
        " ('1', 'https://x.com/a/status/1', 'a', 'a core workout', 1, 0,"
        "  '2026-10-05T00:00:00', 'f.json')")
    conn.execute(
        "INSERT INTO ingested_files (filename, kind, ingested_at, entries,"
        " skipped) VALUES ('f.json', 'likes', '2026-10-05T12:00:00', 1, 0)")
    conn.commit()
    conn.close()

    downloads = tmp_path / "Downloads"
    downloads.mkdir()

    # Build the shell without loading the embedding model or the vectors:
    # both are slow, and neither is what these tests are about.
    monkeypatch.setattr(ti, "model", lambda name: None)
    monkeypatch.setattr(ti, "load_matrix", lambda conn, table: ([], None))

    shell = ti.Shell(db=str(db))
    shell.cfg["paths"]["downloads"] = str(downloads)
    return shell, downloads, db


def test_startup_reports_the_corpus(scratch):
    shell, _downloads, _db = scratch
    assert "1 tweets" in shell.intro or "1," in shell.intro
    assert "2026-10-05 12:00" in shell.intro


def test_startup_mentions_waiting_captures(scratch, tmp_path):
    shell, downloads, db = scratch
    (downloads / "capture_likes_x.json").write_text("{}", encoding="utf-8")
    fresh = ti.Shell(db=str(db))
    fresh.cfg["paths"]["downloads"] = str(downloads)
    line = fresh.startup_lines(0.1, 0.01)
    assert "1 capture file waiting" in line
    assert "/update" in line


def test_startup_says_nothing_waiting(scratch):
    shell, _downloads, _db = scratch
    assert "No new captures waiting" in shell.intro


def test_update_with_no_files_does_nothing(scratch, capsys):
    """The case that matters: /update must not start a run with no input."""
    shell, _downloads, db = scratch
    before = connect(db).execute("SELECT COUNT(*) FROM tweets").fetchone()[0]

    shell.do_update("")

    out = capsys.readouterr().out
    assert "No capture files waiting" in out
    assert "README" in out
    # Nothing ran: no stage lines, no summary.
    assert "1/7" not in out
    assert "New tweets" not in out
    after = connect(db).execute("SELECT COUNT(*) FROM tweets").fetchone()[0]
    assert after == before


def test_empty_topic_asks_for_one_rather_than_searching(scratch, capsys):
    shell, _downloads, _db = scratch
    shell.do_ask("")
    assert "Give me something to search for" in capsys.readouterr().out
    shell.do_basic("   ")
    assert "Give me something to search for" in capsys.readouterr().out


def test_empty_topic_makes_no_jev_connection(scratch, monkeypatch):
    """An empty topic must not reach for a key it does not need."""
    called = []
    monkeypatch.setattr(ti.ask, "connect_jev",
                        lambda cfg: called.append(cfg))
    shell, _downloads, _db = scratch
    shell.do_ask("")
    assert called == []


# --------------------------------------------------------------------------
# Errors never end the session
# --------------------------------------------------------------------------


def test_onecmd_survives_an_exception(scratch, capsys):
    shell, _downloads, _db = scratch

    def boom(arg):
        raise RuntimeError("something broke")

    shell.do_ask = boom
    assert shell.onecmd("ask anything") is False      # False = keep looping
    assert "something broke" in capsys.readouterr().out


@pytest.mark.parametrize("message,expected", [
    ("api_key not found", "Check .env"),
    ("Connection refused", "The network did not answer"),
    ("getaddrinfo failed", "The network did not answer"),
    ("database is locked", "The database is busy"),
    ("no such table: embeddings", "missing a table"),
])
def test_errors_explain_themselves(scratch, capsys, message, expected):
    shell, _downloads, _db = scratch
    shell.report(RuntimeError(message))
    assert expected in capsys.readouterr().out


def test_an_unknown_error_still_names_itself(scratch, capsys):
    shell, _downloads, _db = scratch
    shell.report(ValueError("a surprise"))
    out = capsys.readouterr().out
    assert "ValueError: a surprise" in out


def test_quit_stops_the_loop(scratch, capsys):
    shell, _downloads, _db = scratch
    assert shell.do_quit("") is True
    assert "Bye." in capsys.readouterr().out


def test_blank_line_does_nothing(scratch):
    shell, _downloads, _db = scratch
    assert shell.emptyline() is False
