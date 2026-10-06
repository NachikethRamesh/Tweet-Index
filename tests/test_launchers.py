"""The bin/ launchers, and printing non-ASCII when output is redirected.

These check the properties that broke in practice rather than the wording:
that the launchers find the project from their own location, that they set
UTF-8, and that ti.py prints a tweet with emoji in it even when stdout is a
pipe. The last one is the bug that mattered: without it, redirected output
falls back to cp1252 on this machine and printing a result raises
UnicodeEncodeError.
"""

import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import ti                                                    # noqa: E402

BIN = ROOT / "bin"


# --------------------------------------------------------------------------
# The launcher scripts
# --------------------------------------------------------------------------


@pytest.mark.parametrize("name", ["ti.cmd", "ti.ps1"])
def test_launcher_exists(name):
    assert (BIN / name).is_file()


@pytest.mark.parametrize("name", ["ti.cmd", "ti.ps1"])
def test_launcher_finds_the_project_from_its_own_location(name):
    """No absolute paths: moving the project must not need an edit.

    ti.cmd uses %~dp0.. and ti.ps1 uses $PSScriptRoot, both of which are
    where the script itself sits.
    """
    text = (BIN / name).read_text(encoding="utf-8")
    assert "%~dp0" in text or "PSScriptRoot" in text
    # The one path that would break on a move.
    assert "C:\\Users" not in text


@pytest.mark.parametrize("name", ["ti.cmd", "ti.ps1"])
def test_launcher_uses_the_virtual_environment(name):
    text = (BIN / name).read_text(encoding="utf-8")
    assert ".venv" in text
    assert "python.exe" in text
    assert "ti.py" in text


@pytest.mark.parametrize("name", ["ti.cmd", "ti.ps1"])
def test_launcher_sets_utf8(name):
    """Without this, a redirected run dies on the first emoji."""
    text = (BIN / name).read_text(encoding="utf-8")
    assert "PYTHONIOENCODING" in text
    assert "utf-8" in text.lower()


@pytest.mark.parametrize("name", ["ti.cmd", "ti.ps1"])
def test_launcher_explains_a_missing_virtual_environment(name):
    """A missing .venv is the likeliest failure; it should say what to do."""
    text = (BIN / name).read_text(encoding="utf-8")
    assert "python -m venv .venv" in text
    assert "requirements.txt" in text


@pytest.mark.parametrize("name", ["ti.cmd", "ti.ps1"])
def test_launcher_passes_arguments_through(name):
    text = (BIN / name).read_text(encoding="utf-8")
    assert "%*" in text or "@args" in text


# --------------------------------------------------------------------------
# Encoding: the bug these launchers exposed
# --------------------------------------------------------------------------


def test_use_utf8_is_called_at_import():
    """ti.py must fix its own streams, however it was started."""
    assert hasattr(ti, "use_utf8")
    assert sys.stdout.encoding.lower().replace("-", "") == "utf8"


def test_use_utf8_survives_a_stream_that_cannot_reconfigure(monkeypatch):
    class Stubborn:
        encoding = "cp1252"

        def reconfigure(self, **kwargs):
            raise ValueError("not this one")

    monkeypatch.setattr(ti.sys, "stdout", Stubborn())
    monkeypatch.setattr(ti.sys, "stderr", Stubborn())
    ti.use_utf8()                 # must not raise


def test_non_ascii_prints_when_stdout_is_a_pipe(tmp_path):
    """The real check: run ti.py with output redirected and no encoding set.

    The em dash in the banner is enough to trigger it -- under cp1252 this
    used to come out as '?' at best and raise UnicodeEncodeError at worst.
    """
    script = tmp_path / "in.txt"
    script.write_text("/quit\n", encoding="utf-8")
    env = {k: v for k, v in __import__("os").environ.items()
           if k != "PYTHONIOENCODING"}
    with script.open("rb") as stdin:
        done = subprocess.run(
            [sys.executable, str(ROOT / "ti.py")],
            stdin=stdin, capture_output=True, env=env, timeout=300)
    out = done.stdout.decode("utf-8", errors="replace")
    assert "UnicodeEncodeError" not in out + done.stderr.decode(
        "utf-8", errors="replace")
    assert "Tweet Index" in out
    assert "\u2014" in out              # the em dash survived
