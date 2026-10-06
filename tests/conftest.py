"""Shared test fixtures, built from synthetic timeline payloads.

The original suite ran against real captures from the author's account. Those
are personal data and are not in this repository, so `make_fixtures.py`
generates payloads of the same shape with invented tweets, and these fixtures
hand them to the tests.

Building them is cheap, so they are session-scoped only to save the JSON
round-trip, not because anything here is slow.
"""

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from tests import make_fixtures                      # noqa: E402
from tweet_index.parse import iter_entries           # noqa: E402


@pytest.fixture(scope="session")
def bookmark_pages():
    return make_fixtures.sample_bookmarks()


@pytest.fixture(scope="session")
def like_pages():
    return make_fixtures.sample_likes()


@pytest.fixture(scope="session")
def bookmark_entries(bookmark_pages):
    return [e for page in bookmark_pages for e in iter_entries(page)]


@pytest.fixture(scope="session")
def like_entries(like_pages):
    return [e for page in like_pages for e in iter_entries(page)]


@pytest.fixture(scope="session")
def entry_by_id(bookmark_entries, like_entries):
    """Look up a timeline entry by tweet id, preferring bookmarks."""
    index = {}
    for entry in like_entries + bookmark_entries:   # bookmarks win on overlap
        index[entry["entryId"].removeprefix("tweet-")] = entry

    def get(tweet_id):
        assert tweet_id in index, f"tweet {tweet_id} is not in the fixtures"
        return index[tweet_id]

    return get


@pytest.fixture
def captures(tmp_path):
    """A bookmarks and a likes capture file on disk, as the userscript
    writes them. Returns (bookmarks_path, likes_path)."""
    return make_fixtures.write_captures(tmp_path / "raw")
