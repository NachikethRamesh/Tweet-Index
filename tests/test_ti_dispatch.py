"""How a typed line becomes a command.

The rule: a leading slash always means "command". Without one, the line is a
search unless it can only be a command -- which is why "more cowbell" and
"history of rome" are searches while "more" and "history" are not.
"""

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import ti                                                    # noqa: E402

P = ti.Shell.precmd


@pytest.mark.parametrize("line,expected", [
    # A slash is unambiguous, always.
    ("/ask core workout", "ask core workout"),
    ("/basic core workout", "basic core workout"),
    ("/open 3", "open 3"),
    ("/more", "more"),
    ("/status", "status"),
    ("/history", "history"),
    ("/update", "update"),
    ("/help", "help"),
    ("/help ask", "help ask"),
    ("/quit", "quit"),
    # Even a slashed line whose argument looks like a command.
    ("/ask /basic", "ask /basic"),
    ("/more cowbell", "more cowbell"),
])
def test_a_slash_means_a_command(line, expected):
    assert P(None, line) == expected


@pytest.mark.parametrize("line,expected", [
    # Commands that take an argument are recognised by their first word.
    ("ask core workout", "ask core workout"),
    ("basic core workout", "basic core workout"),
    ("open 3", "open 3"),
    # Commands that take none are recognised only when typed alone.
    ("more", "more"),
    ("status", "status"),
    ("history", "history"),
    ("update", "update"),
    ("help", "help"),
    ("quit", "quit"),
    ("exit", "exit"),
    # Case does not matter for recognition.
    ("MORE", "MORE"),
    ("Status", "Status"),
])
def test_bare_commands(line, expected):
    assert P(None, line) == expected


@pytest.mark.parametrize("line", [
    # These begin with a no-argument command's name and carry more words,
    # so they can only be searches.
    "more cowbell",
    "history of rome",
    "history of the peloponnesian war",
    "status anxiety",
    "update my resume",
    "help me write a cold intro",
    "quit smoking",
    # These merely start with the letters of a command.
    "opening a restaurant in paris",
    "asking for a raise",
    "basically everything about evals",
    "statusing things",
    "historical fiction",
    # An ordinary topic.
    "core workout",
    "what have I saved about evals",
    "GPU architecture and CUDA",
])
def test_bare_text_is_a_search(line):
    assert P(None, line) == f"ask {line}"


def test_help_with_a_command_name_is_help():
    assert P(None, "help ask") == "help ask"
    assert P(None, "help more") == "help more"
    assert P(None, "help open") == "help open"


def test_help_with_anything_else_is_a_search():
    assert P(None, "help me choose a laptop") == "ask help me choose a laptop"
    assert P(None, "help nonsense") == "ask help nonsense"


@pytest.mark.parametrize("line", ["", "   ", "\t", "\n"])
def test_blank_stays_blank(line):
    assert P(None, line) == ""


def test_whitespace_is_trimmed():
    assert P(None, "   core workout   ") == "ask core workout"
    assert P(None, "  /status  ") == "status"


def test_every_command_in_the_lists_has_a_handler():
    """A name in ARG_COMMANDS or BARE_COMMANDS with no do_ method would be
    dispatched and then reported as unknown."""
    for name in ti.Shell.ARG_COMMANDS + ti.Shell.BARE_COMMANDS:
        attr = "do_EOF" if name == "eof" else f"do_{name}"
        assert hasattr(ti.Shell, attr), f"{name} has no {attr}"


@pytest.mark.parametrize("line,expected", [
    # PowerShell writes a byte-order mark when piping into a program, which
    # would hide the slash and turn a command into a search.
    ("\ufeff/quit", "quit"),
    ("\ufeff/status", "status"),
    ("\ufeffcore workout", "ask core workout"),
    ("\u200b/more", "more"),
    ("\ufeff", ""),
])
def test_an_invisible_prefix_does_not_hide_the_command(line, expected):
    assert P(None, line) == expected
