"""Fetch the title and description of every linked page (Phase 4).

    python fetch_links.py --sample 30     # a few, to look at
    python fetch_links.py                 # every pending row
    python fetch_links.py --retry-failed  # another go at the failures

A saved tweet is often one line plus a link, and the link is where the content
is. The page title and meta description are cheap, small, and usually say what
the page is -- enough to make the tweet findable.

Section 3.2 names httpx and trafilatura for this. trafilatura extracts full
article text, which is a bigger job than titles and is not what this does; and
with no article body to stream, urllib from the standard library is enough. So
no new dependencies. Full article text stays unbuilt.

Politeness, since these are other people's servers:

  * a User-Agent that says what this is,
  * an 8-second timeout, so one slow host cannot hold up the run,
  * at most 8 requests at a time, and never two at once to the same host,
  * at least half a second between requests to the same host,
  * only the first 200 KB of each page is read; the title is in the head,
  * x.com links are skipped -- they point back at tweets we already have.
"""

import argparse
import re
import socket
import sys
import threading
import time
import urllib.error
import urllib.parse
import urllib.request
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from html.parser import HTMLParser
from pathlib import Path

import yaml

from tweet_index.db import connect

ROOT = Path(__file__).resolve().parent

USER_AGENT = ("Mozilla/5.0 (compatible; tweet-index/0.1; personal archive; "
              "+https://github.com/)")
TIMEOUT = 8
MAX_BYTES = 200_000
PER_HOST_GAP = 0.5

LAST_RUN = {}

SKIP_HOSTS = {"x.com", "www.x.com", "twitter.com", "www.twitter.com",
              "mobile.twitter.com", "t.co"}

_host_lock = defaultdict(threading.Lock)
_host_last = defaultdict(float)
_locks_guard = threading.Lock()


def host_of(url):
    try:
        return (urllib.parse.urlparse(url).hostname or "").lower()
    except ValueError:
        return ""


class HeadScraper(HTMLParser):
    """Pull the title and the description meta tags, then stop.

    Written against the head only: as soon as <body> starts there is nothing
    left worth reading, and many pages are megabytes of markup after that.
    """

    def __init__(self):
        super().__init__(convert_charrefs=True)
        self.title = None
        self.description = None
        self.og_title = None
        self.og_description = None
        self._in_title = False
        self.done = False

    def handle_starttag(self, tag, attrs):
        if tag == "title":
            self._in_title = True
        elif tag == "body":
            self.done = True
        elif tag == "meta":
            a = {k.lower(): (v or "") for k, v in attrs}
            name = (a.get("name") or a.get("property") or "").lower()
            content = a.get("content", "").strip()
            if not content:
                return
            if name == "description" and not self.description:
                self.description = content
            elif name == "og:description" and not self.og_description:
                self.og_description = content
            elif name == "og:title" and not self.og_title:
                self.og_title = content

    def handle_endtag(self, tag):
        if tag == "title":
            self._in_title = False
        elif tag == "head":
            self.done = True

    def handle_data(self, data):
        if self._in_title:
            self.title = ((self.title or "") + data)[:400]

    def best(self):
        title = (self.title or self.og_title or "").strip()
        description = (self.description or self.og_description or "").strip()
        return tidy(title), tidy(description)


def tidy(text, limit=400):
    if not text:
        return None
    text = re.sub(r"\s+", " ", text).strip()
    return text[:limit] or None


def wait_for_host(host):
    """Never two requests at once to one host, and never back to back."""
    with _locks_guard:
        lock = _host_lock[host]
    with lock:
        gap = time.monotonic() - _host_last[host]
        if gap < PER_HOST_GAP:
            time.sleep(PER_HOST_GAP - gap)
        _host_last[host] = time.monotonic()


def fetch_one(url):
    """Returns (status, title, description, note)."""
    host = host_of(url)
    if not host:
        return "failed", None, None, "unparseable url"
    if host in SKIP_HOSTS:
        return "skipped", None, None, "points back at x.com"

    wait_for_host(host)
    request = urllib.request.Request(url, headers={
        "User-Agent": USER_AGENT,
        "Accept": "text/html,application/xhtml+xml",
        "Accept-Language": "en",
    })
    try:
        with urllib.request.urlopen(request, timeout=TIMEOUT) as response:
            kind = (response.headers.get_content_type() or "").lower()
            if "html" not in kind and "xml" not in kind:
                return "skipped", None, None, f"not a web page ({kind})"
            charset = response.headers.get_content_charset() or "utf-8"
            raw = response.read(MAX_BYTES)
    except urllib.error.HTTPError as exc:
        return "failed", None, None, f"HTTP {exc.code}"
    except urllib.error.URLError as exc:
        return "failed", None, None, f"{type(exc.reason).__name__}: {exc.reason}"
    except (socket.timeout, TimeoutError):
        return "failed", None, None, "timed out"
    except Exception as exc:                                  # noqa: BLE001
        return "failed", None, None, f"{type(exc).__name__}"

    try:
        text = raw.decode(charset, errors="replace")
    except LookupError:
        text = raw.decode("utf-8", errors="replace")

    scraper = HeadScraper()
    try:
        scraper.feed(text)
    except Exception:                                         # noqa: BLE001
        pass
    title, description = scraper.best()
    if not title and not description:
        return "failed", None, None, "no title or description found"
    return "ok", title, description, None


def load_config(path=None):
    return yaml.safe_load((Path(path) if path else ROOT / "config.yaml").read_text(encoding="utf-8"))


def resolve(p):
    path = Path(p).expanduser()
    return path if path.is_absolute() else ROOT / path


def log(logs_dir, message):
    logs_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).isoformat(timespec="seconds")
    with (logs_dir / "links.log").open("a", encoding="utf-8") as fh:
        fh.write(f"{stamp} {message}\n")


def pending(conn, retry_failed=False):
    wanted = ("pending", "failed") if retry_failed else ("pending",)
    marks = ",".join("?" * len(wanted))
    return conn.execute(
        f"SELECT id, tweet_id, url FROM links"
        f" WHERE fetch_status IN ({marks}) ORDER BY id", wanted).fetchall()


def save(conn, rows):
    with conn:
        conn.executemany(
            "UPDATE links SET title=?, fetched_text=?, fetch_status=?"
            " WHERE id=?", rows)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sample", type=int, metavar="N")
    parser.add_argument("--retry-failed", action="store_true")
    parser.add_argument("--concurrency", type=int, default=8)
    parser.add_argument("--db", default=None)
    parser.add_argument("--config", default=None)
    args = parser.parse_args(argv)

    cfg = load_config(args.config)
    conn = connect(Path(args.db) if args.db else resolve(cfg["paths"]["db"]))
    logs_dir = resolve("data/logs")

    rows = pending(conn, args.retry_failed)
    total = conn.execute("SELECT COUNT(*) FROM links").fetchone()[0]
    print(f"Link rows: {total}")
    print(f"To fetch:  {len(rows)}")
    if args.sample:
        rows = rows[:args.sample]
        print(f"Sample:    {len(rows)}")
    if not rows:
        print("\nNothing to do.")
        conn.close()
        return 0

    hosts = len({host_of(r["url"]) for r in rows})
    print(f"Distinct hosts: {hosts}")
    print(f"Politeness: {args.concurrency} at a time, {TIMEOUT}s timeout, "
          f"{PER_HOST_GAP}s between hits on one host\n")

    t0 = time.perf_counter()
    done = {"ok": 0, "failed": 0, "skipped": 0}
    reasons = {}
    updates = []

    def work(row):
        status, title, description, note = fetch_one(row["url"])
        return row, status, title, description, note

    with ThreadPoolExecutor(max_workers=args.concurrency) as pool:
        for row, status, title, description, note in pool.map(work, rows):
            done[status] += 1
            if note:
                reasons[note] = reasons.get(note, 0) + 1
            if status != "ok":
                log(logs_dir, f"{status} {row['url']}: {note}")
            updates.append((title, description, status, row["id"]))
            if len(updates) >= 100:
                save(conn, updates)
                updates = []
            print(f"\r  {sum(done.values())}/{len(rows)}  "
                  f"ok {done['ok']}  failed {done['failed']}  "
                  f"skipped {done['skipped']}   ", end="", flush=True)
    if updates:
        save(conn, updates)
    seconds = time.perf_counter() - t0
    print()

    print(f"\nFetched {done['ok']}, failed {done['failed']}, "
          f"skipped {done['skipped']}, in {seconds:.1f}s")
    if reasons:
        print("\nWhy they did not yield a title:")
        for reason, n in sorted(reasons.items(), key=lambda kv: -kv[1])[:12]:
            print(f"  {n:5}  {reason}")
    with_title = conn.execute(
        "SELECT COUNT(*) FROM links WHERE TRIM(COALESCE(title,'')) <> ''"
    ).fetchone()[0]
    print(f"\nLink rows with a title: {with_title} of {total}")
    log(logs_dir, f"fetched {done['ok']}, failed {done['failed']}, "
                  f"skipped {done['skipped']}, {seconds:.1f}s")
    LAST_RUN.clear()
    LAST_RUN.update(step="links", done=done["ok"], failed=done["failed"],
                    skipped=done["skipped"], seconds=seconds, cost=0.0,
                    reasons=reasons)
    conn.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
