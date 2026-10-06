"""The Tweet Index shell. One long-lived process, so searching is fast.

    python ti.py

Loading the embedding model takes about 1.6 seconds and reading the vectors
about 25 ms. A one-shot `python ask.py` pays both every time; here they are
paid once at startup, so the second and later searches skip straight to the
search itself.

    /update            fetch and process new captures, then reload
    /ask <topic>       smart search, with the Jev rerank
    /ask <topic> -n 8  show 8 strong matches instead of 5
    /basic <topic>     plain embedding search, free and instant
    <topic>            the same as /ask
    /help              the command list
    /quit              exit

Standard library only: `cmd` gives the prompt loop, history and help. All the
ranking and all seven update stages come from ask.py and update.py; nothing is
reimplemented here.
"""

import cmd
import io
import json
import shlex
import sys
import time
import traceback
import webbrowser
from contextlib import redirect_stdout
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "evals"))

import ask                                                   # noqa: E402
import update                                                # noqa: E402
from tweet_index import spend                               # noqa: E402
from tweet_index.db import connect                           # noqa: E402
from tweet_index.embed import load_matrix, model             # noqa: E402

def use_utf8():
    """Print UTF-8 whatever the console's codepage is.

    Saved tweets are full of emoji, Turkish, Japanese and em dashes. When
    stdout is a real console window Python writes Unicode directly and all of
    that is fine, but as soon as output is redirected -- piped to a file, read
    by a script -- Python falls back to the locale encoding, which on this
    machine is cp1252. Then printing a tweet raises UnicodeEncodeError and
    the search is lost.

    `errors="replace"` matters as much as the encoding: a character the
    terminal cannot draw should cost one glyph, not the whole result.
    """
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8", errors="replace")
        except (AttributeError, ValueError, OSError):
            pass                     # an older Python or an odd stream


use_utf8()

BANNER = "Tweet Index"

# Commands that take an argument: without a slash, the first word is enough
# to recognise them.
ARG_COMMANDS = ("ask", "basic", "open")
# Commands that take none. Without a slash these count only when typed alone,
# because "more cowbell" and "history of rome" are searches, not /more and
# /history with nonsense attached.
BARE_COMMANDS = ("update", "status", "history", "more", "quit", "exit",
                 "eof", "help")


def plural(n, one, many=None):
    return one if n == 1 else (many or one + "s")


class Shell(cmd.Cmd):
    """The prompt loop.

    `cmd` dispatches `/ask foo` to `do_ask` once the leading slash is
    stripped, which `precmd` does. Anything with no slash is treated as a
    search, which is what one actually types most of the time.
    """

    prompt = "ti> "
    doc_header = "Commands (type /help <command> for more)"

    # True means input() does the reading, which is what gives the up and
    # down arrows. On Windows the console itself keeps the per-process line
    # history, so arrows work in PowerShell with no readline and no extra
    # package; on Linux and macOS the stdlib readline provides it. Setting
    # this False would read stdin directly and lose the arrows on every
    # platform, so it stays True.
    use_rawinput = True

    # ---- startup ----------------------------------------------------
    def __init__(self, db=None, config=None):
        super().__init__()
        self.cfg = ask.load_config(config)
        self.config_path = config
        self.db_path = Path(db) if db else ask.resolve(self.cfg["paths"]["db"])
        self.conn = connect(self.db_path)
        self.jev_ready = False
        self.last = None                 # the last search, for /more and /open

        t0 = time.perf_counter()
        model(self.cfg["embedding"]["model"])
        model_s = time.perf_counter() - t0

        t0 = time.perf_counter()
        self.ids, self.matrix = load_matrix(self.conn, "embeddings")
        vectors_s = time.perf_counter() - t0

        self.intro = self.startup_lines(model_s, vectors_s)
        self.load_line_history()

    def load_line_history(self):
        """Keep typed lines across sessions where readline exists.

        Windows has no readline in the standard library -- the arrows work
        there because the console provides them, but only for the life of the
        process. Where readline *is* available the history also survives a
        restart. Either way this is best-effort: no package is required and a
        failure here must not stop the shell.
        """
        self.history_file = None
        try:
            import readline
        except ImportError:
            return
        path = ROOT / "data" / "logs" / "ti_history"
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            if path.exists():
                readline.read_history_file(path)
            readline.set_history_length(1000)
            self.history_file = path
        except OSError:
            self.history_file = None

    def save_line_history(self):
        if not self.history_file:
            return
        try:
            import readline
            readline.write_history_file(self.history_file)
        except (ImportError, OSError):
            pass

    def startup_lines(self, model_s, vectors_s):
        tweets = self.conn.execute("SELECT COUNT(*) FROM tweets").fetchone()[0]
        last = self.conn.execute(
            "SELECT MAX(ingested_at) FROM ingested_files").fetchone()[0]
        waiting = self.waiting_captures()

        lines = [f"\n{BANNER}  —  {tweets:,} tweets"]
        lines.append(f"Last update: {(last or 'never')[:16].replace('T', ' ')}")
        lines.append(f"Ready in {model_s + vectors_s:.1f}s "
                     f"(model {model_s:.1f}s, vectors {vectors_s * 1000:.0f}ms)")
        if waiting:
            lines.append(f"\n{len(waiting)} capture "
                         f"{plural(len(waiting), 'file')} waiting in Downloads"
                         f" — run /update to add {plural(len(waiting), 'it', 'them')}.")
        else:
            lines.append("\nNo new captures waiting. /help for commands.")
        return "\n".join(lines)

    def waiting_captures(self):
        """Capture files sitting in Downloads, not yet ingested."""
        try:
            downloads = ask.resolve(self.cfg["paths"]["downloads"])
            return sorted(downloads.glob("capture_*.json"))
        except OSError:
            return []

    # ---- dispatch ---------------------------------------------------
    ARG_COMMANDS = ARG_COMMANDS
    BARE_COMMANDS = BARE_COMMANDS

    def precmd(self, line):
        """Strip the leading slash; treat bare text as a search.

        A slash always means "this is a command". Without one, the line is a
        search unless it can only be a command.
        """
        # A byte-order mark or a zero-width space at the start would hide the
        # slash and turn a command into a search. PowerShell writes a BOM
        # when piping into a program, so this shows up in scripted use.
        line = line.lstrip("﻿​").strip()
        if not line:
            return ""
        if line.startswith("/"):
            return line[1:]
        words = line.split()
        first = words[0].lower()
        if first in ARG_COMMANDS:
            return line
        if first in BARE_COMMANDS:
            if len(words) == 1:
                return line
            # /help ask is still help; /help me write a cold intro is not.
            if first == "help" and len(words) == 2 and words[1].lower() in (
                    ARG_COMMANDS + BARE_COMMANDS):
                return line
        return f"ask {line}"                 # anything else is a search

    def onecmd(self, line):
        """Run one command, and never let an exception end the session.

        A missing key, a dead network or a bad topic should cost one line of
        explanation and nothing else. The traceback goes to a file so a real
        bug is still diagnosable.
        """
        try:
            return super().onecmd(line)
        except KeyboardInterrupt:
            print("\n(interrupted)")
        except Exception as exc:                        # noqa: BLE001
            self.report(exc)
        return False

    def report(self, exc):
        """One clear sentence, and the detail on disk."""
        text = str(exc)
        low = text.lower()
        if "api_key" in low or "api key" in low or "authentication" in low:
            print("That needed an API key I could not find. Check .env holds "
                  "OPENAI_API_KEY and TYPESAFE_API_KEY.")
        elif any(w in low for w in ("connection", "network", "timed out",
                                    "timeout", "getaddrinfo", "dns",
                                    "unreachable", "ssl")):
            print("The network did not answer. Check the connection and try "
                  "again — nothing was changed.")
        elif "database is locked" in low:
            print("The database is busy. Something else is writing to it; "
                  "wait a moment and try again.")
        elif "no such table" in low:
            print(f"The database is missing a table ({text}). "
                  f"Run python update.py to build it.")
        else:
            print(f"{type(exc).__name__}: {text}")
        logs = ROOT / "data" / "logs"
        try:
            logs.mkdir(parents=True, exist_ok=True)
            with (logs / "ti.log").open("a", encoding="utf-8") as fh:
                fh.write(traceback.format_exc() + "\n")
            print(f"(details in {logs / 'ti.log'})")
        except OSError:
            pass

    def emptyline(self):
        return False                         # Enter does nothing

    # ---- /ask -------------------------------------------------------
    def do_ask(self, arg):
        """Smart search: cosine, then the Jev relevance rerank.

        /ask core workout          the usual 5 strong matches
        /ask core workout -n 8     8 instead
        """
        topic, top_n = self.split_n(arg)
        if not topic:
            print("Give me something to search for, e.g. /ask core workout")
            return
        if not self.jev_ready:
            ask.connect_jev(self.cfg)
            self.jev_ready = True

        t0 = time.perf_counter()
        out = ask.rank_smart(self.conn, self.cfg, topic, self.ids, self.matrix,
                             top_n=top_n)
        elapsed = time.perf_counter() - t0

        print(f'\n{out["corpus"]:,} tweets searched for "{topic}"')
        print(f'{len(out["candidates"])} candidates reranked, '
              f'{len(out["scored"])} scored, {len(out["failures"])} failed\n')
        ask.show(out["tops"], "STRONG MATCHES")
        ask.show(out["maybes"], "MAYBE", start=len(out["tops"]) + 1)
        if not out["tops"] and not out["maybes"]:
            print("Nothing cleared the relevance thresholds.")

        # One canonical order for the whole ranked list: what was printed,
        # then everything else by score. The bands are chosen by band, not by
        # position, so results[:6] is NOT what appeared on screen -- a
        # below-threshold tweet can outscore a maybe and sit between them.
        # Slicing the raw list for /more therefore reprinted a result that
        # had already been shown. Numbering this list once fixes /more and
        # makes /open N mean the same thing throughout.
        printed = out["tops"] + out["maybes"]
        seen = {id(r) for r in printed}
        ordered = printed + [r for r in out["results"] if id(r) not in seen]
        self.last = {"topic": topic, "results": ordered,
                     "shown": len(printed), "kind": "ask"}
        if len(ordered) > len(printed):
            print(f"{len(ordered) - len(printed)} more below the thresholds "
                  f"— /more to see them.")

        cost = out["jev_tokens"] / 1e6 * ask.jev.USD_PER_MILLION_INPUT
        print(f'Cost: Jev {out["jev_tokens"]:,} input tokens ${cost:.5f}')
        t = out["timings"]
        print(f'Time: cosine {t["stage 1 cosine"] * 1000:.0f}ms  '
              f'jev {t["stage 2 jev"] * 1000:.0f}ms  '
              f'blend {t["stage 3 blend"] * 1000:.0f}ms  '
              f'total {elapsed * 1000:.0f}ms   '
              f'(model and vectors already loaded)')

        self.log(topic, out, elapsed)

    def log(self, topic, out, elapsed):
        snapshot = {
            "phase": 2, "ranking": "cosine -> jev relevance -> blend",
            "interface": "ti.py",
            "embedding_model": self.cfg["embedding"]["model"],
            "embeddings_table": "embeddings",
            "jev_model": ask.jev.model(),
            "candidate_cap": self.cfg["ask"]["candidate_cap"],
            "weights": self.cfg["ask"]["weights"],
            "confidence_floor": self.cfg["ask"]["confidence_floor"],
            "shown": {"top": len(out["tops"]), "maybe": len(out["maybes"])},
            "rerank_failures": len(out["failures"]),
        }
        ask.log_query(self.conn, topic, snapshot, out["results"],
                      round(elapsed * 1000), out["jev_tokens"])

    # ---- /basic -----------------------------------------------------
    def do_basic(self, arg):
        """Plain embedding search: cosine order only. Free and instant.

        /basic core workout
        /basic core workout -n 20
        """
        topic, top_n = self.split_n(arg)
        if not topic:
            print("Give me something to search for, e.g. /basic core workout")
            return
        t0 = time.perf_counter()
        out = ask.rank_basic(self.conn, self.cfg, topic, self.ids, self.matrix,
                             top_n)
        elapsed = time.perf_counter() - t0
        print(f'\n{out["corpus"]:,} tweets searched for "{topic}"\n')
        kept = []
        for n, (tid, score) in enumerate(out["candidates"], 1):
            row = out["rows"].get(tid)
            if not row:
                continue
            print(f"{n:2}. {score:.3f}  @{row['author_handle']}  "
                  f"{(row['posted_at'] or '')[:10]}  {ask.saved_as(row)}")
            print(f"    {ask.one_line(row['text']) or '(no text)'}")
            print(f"    {row['url']}")
            print()
            kept.append({"row": row, "cosine": score})
        print(f"Time: {elapsed * 1000:.0f}ms   free, no model calls")
        self.last = {"topic": topic, "results": kept,
                     "shown": len(kept), "kind": "basic"}

    @staticmethod
    def split_n(arg):
        """Pull an optional -n N off the end of a topic."""
        try:
            parts = shlex.split(arg)
        except ValueError:
            parts = arg.split()
        top_n = None
        out = []
        i = 0
        while i < len(parts):
            if parts[i] in ("-n", "--n") and i + 1 < len(parts):
                try:
                    top_n = max(1, int(parts[i + 1]))
                    i += 2
                    continue
                except ValueError:
                    pass
            out.append(parts[i])
            i += 1
        return " ".join(out).strip(), top_n

    # ---- /more, /open -----------------------------------------------
    def do_more(self, arg):
        """Show the next results for the last search. No new API calls.

        A smart search ranks 120 candidates but prints only the ones that
        clear §11.3's thresholds. The rest were already scored and are held
        in memory, so this is free and instant.
        """
        if not self.last:
            print("Nothing searched yet. Try /ask core workout")
            return
        results = self.last["results"]
        shown = self.last["shown"]
        if shown >= len(results):
            print(f'That is all {len(results)} results for '
                  f'"{self.last["topic"]}".')
            return
        batch = results[shown:shown + 10]
        print(f'\nMore for "{self.last["topic"]}" '
              f'({shown + 1}-{shown + len(batch)} of {len(results)}, '
              f'below the thresholds)\n')
        if self.last["kind"] == "basic":
            for n, r in enumerate(batch, shown + 1):
                row = r["row"]
                print(f"{n:2}. {r['cosine']:.3f}  @{row['author_handle']}")
                print(f"    {ask.one_line(row['text']) or '(no text)'}")
                print(f"    {row['url']}\n")
        else:
            ask.show(batch, "", start=shown + 1)
        self.last["shown"] = shown + len(batch)
        left = len(results) - self.last["shown"]
        if left:
            print(f"{left} more — /more again.")

    def do_open(self, arg):
        """Open result N from the last search in the browser.

        /open 3
        """
        if not self.last:
            print("Nothing searched yet. Try /ask core workout")
            return
        results = self.last["results"]
        try:
            n = int(arg.strip())
        except ValueError:
            print(f"Which one? /open 1 to /open {len(results)}")
            return
        if not 1 <= n <= len(results):
            print(f"There is no result {n}. "
                  f"This search has {len(results)}: /open 1 to "
                  f"/open {len(results)}")
            return
        row = results[n - 1]["row"]
        url = row["url"]
        if not url:
            print(f"Result {n} has no link stored.")
            return
        print(f"Opening {n}. @{row['author_handle']}  {url}")
        try:
            webbrowser.open(url)
        except Exception as exc:                        # noqa: BLE001
            print(f"Could not open a browser ({exc}). The link is above.")

    # ---- /status ----------------------------------------------------
    def do_status(self, arg):
        """Counts, last update, captures waiting, and this month's spend."""
        c = self.conn
        tweets, bookmarks, likes, both = c.execute(
            "SELECT COUNT(*), SUM(is_bookmark), SUM(is_like),"
            " SUM(is_like AND is_bookmark) FROM tweets").fetchone()
        last = c.execute(
            "SELECT MAX(ingested_at) FROM ingested_files").fetchone()[0]
        waiting = self.waiting_captures()

        print()
        print(f"Tweets        {tweets or 0:,}")
        print(f"  bookmarks   {bookmarks or 0:,}")
        print(f"  likes       {likes or 0:,}")
        print(f"  in both     {both or 0:,}")
        print(f"Images read   {self.count_images():,} of {self.count_media():,}")
        print(f"Links titled  {self.count_links(True):,} of "
              f"{self.count_links(False):,}")
        print(f"Last update   {(last or 'never')[:16].replace('T', ' ')}")
        print(f"Waiting       {len(waiting)} capture "
              f"{plural(len(waiting), 'file')}"
              f"{' — run /update' if waiting else ''}")

        logs = ROOT / "data" / "logs"
        totals, searches = spend.month(self.conn, logs)
        month_name = datetime.now(timezone.utc).strftime("%B %Y")
        print(f"\nSpend so far in {month_name}")
        if totals:
            for provider in sorted(totals):
                print(f"  {provider:12} ${totals[provider]:.4f}")
            print(f"  {'total':12} ${sum(totals.values()):.4f}")
        else:
            print("  nothing yet")
        print(f"  ({searches} {plural(searches, 'search', 'searches')} "
              f"this month)")
        started = spend.ledger_started(logs)
        if started:
            print(f"  pipeline spend recorded since {started}; "
                  f"searching comes from the queries table")
        else:
            print("  pipeline spend is recorded from the next /update onwards;"
                  " searching comes from the queries table")

    def count_images(self):
        return self.conn.execute(
            "SELECT COUNT(*) FROM media WHERE"
            " TRIM(COALESCE(ocr_text,'')) <> ''").fetchone()[0]

    def count_media(self):
        return self.conn.execute("SELECT COUNT(*) FROM media").fetchone()[0]

    def count_links(self, titled):
        sql = "SELECT COUNT(*) FROM links"
        if titled:
            sql += " WHERE TRIM(COALESCE(title,'')) <> ''"
        return self.conn.execute(sql).fetchone()[0]

    # ---- /history ---------------------------------------------------
    def do_history(self, arg):
        """The last 10 searches, with the date and the top result."""
        rows = self.conn.execute(
            "SELECT asked_at, topic, results FROM queries"
            " ORDER BY id DESC LIMIT 10").fetchall()
        if not rows:
            print("No searches recorded yet.")
            return
        print()
        for r in rows:
            when = (r["asked_at"] or "")[:16].replace("T", " ")
            top = self.top_of(r["results"])
            print(f'  {when}  {r["topic"][:40]:40}  {top}')

    def top_of(self, blob):
        """The best result of a logged search, as a handle and score."""
        try:
            payload = json.loads(blob or "[]")
        except ValueError:
            return "(unreadable)"
        if not payload:
            return "(nothing)"
        best = max(payload, key=lambda p: p.get("final", 0))
        row = self.conn.execute(
            "SELECT author_handle FROM tweets WHERE id = ?",
            (best.get("id"),)).fetchone()
        handle = f"@{row['author_handle']}" if row else "(deleted)"
        return f"{handle} {best.get('final', 0):.3f}"

    # ---- /update ----------------------------------------------------
    def do_update(self, arg):
        """Fetch and process new captures: all seven stages, then reload.

        Prints one line per stage as it runs, then what changed.
        """
        waiting = self.waiting_captures()
        if not waiting:
            print("No capture files waiting in Downloads. Capture in Firefox "
                  "first — see the README.")
            return

        print(f"\n{len(waiting)} capture {plural(len(waiting), 'file')} to "
              f"process. This takes a minute or so.\n")

        before = update.counts(self.db_path)

        # The stages print a lot. Keep each one's output, show a single line
        # while it runs, and summarise from the numbers rather than the text.
        stages = []
        captured = io.StringIO()
        try:
            self.conn.close()               # update.py opens its own
            for label, key, fn in self.stage_list():
                print(f"  {label:34}", end="", flush=True)
                t0 = time.perf_counter()
                with redirect_stdout(captured):
                    result = fn()
                took = time.perf_counter() - t0
                result = dict(result or {})
                result["label"] = label
                result["stage"] = key
                result.setdefault("done", 0)
                result.setdefault("failed", 0)
                result.setdefault("cost", 0.0)
                result["seconds"] = took
                stages.append(result)
                note = f"{result['done']} done"
                if result["failed"]:
                    note += f", {result['failed']} failed"
                print(f"{note:24} {took:5.1f}s")
        finally:
            self.conn = connect(self.db_path)

        after = update.counts(self.db_path)
        spend.record_stages(ROOT / "data" / "logs", stages)
        self.summarise(before, after, stages, captured)
        self.reload_vectors()

    def stage_list(self):
        """The seven stages, each returning its module's LAST_RUN.

        Every stage gets both --db and --config. Passing only --db would
        leave the stages reading the default config while the shell used
        another, so `downloads` and `raw` would point somewhere else than the
        shell reported: it would say a capture was waiting and then ingest a
        different folder entirely.
        """
        db = ["--db", str(self.db_path)]
        if self.config_path:
            db += ["--config", str(self.config_path)]

        def run(module, argv, extra=None):
            def go():
                module.LAST_RUN.clear()
                module.main(argv + db + (extra or []))
                return dict(module.LAST_RUN)
            return go

        def judge():
            ids = update.ids_needing_judgment(self.db_path, self.enriched)
            if not ids:
                return {"done": 0}
            path = update.write_ids(ids, "ti_judge_ids.txt")
            return run(update.tag, ["--ids-file", str(path),
                                    "--concurrency", "16"])()

        def check():
            if not self.enriched:
                return {"done": 0}
            path = update.write_ids(self.enriched, "ti_check_ids.txt")
            return run(update.check_summaries,
                       ["--ids-file", str(path), "--round", "9",
                        "--concurrency", "16"])()

        def enrich():
            out = run(update.enrich, ["--concurrency", "8"])()
            self.enriched = out.get("ids", [])
            return out

        self.enriched = []
        return [
            ("1/7  ingest captures", "ingest", run(update.ingest, [])),
            ("2/7  read images", "images",
             run(update.describe_images, ["--concurrency", "8"])),
            ("3/7  fetch link titles", "links",
             run(update.fetch_links, ["--concurrency", "8"])),
            ("4/7  write summaries", "summaries", enrich),
            ("5/7  embed", "embed", run(update.embed_all, [])),
            ("6/7  judge substance", "substance", judge),
            ("7/7  check summaries", "faithfulness", check),
        ]

    def summarise(self, before, after, stages, captured):
        new = after["tweets"] - before["tweets"]
        ingest = next((s for s in stages
                       if s.get("stage") == "ingest"), {})
        seen = ingest.get("seen", 0)
        already = max(seen - new, 0) if seen else None

        print()
        print(f"New tweets:      {new}")
        print(f"  bookmarks      +{after['bookmarks'] - before['bookmarks']}")
        print(f"  likes          +{after['likes'] - before['likes']}")
        print(f"  in both        +{after['both'] - before['both']}")
        if already is not None:
            print(f"Already saved:   {already} (skipped, nothing to redo)")
        print(f"New total:       {after['tweets']:,} tweets")
        print(f"Images read:     +{after['images_read'] - before['images_read']}")
        print(f"Links titled:    +{after['links_titled'] - before['links_titled']}")
        print(f"Summaries:       +{after['summaries'] - before['summaries']}")
        print(f"Embeddings:      +{after['embeddings'] - before['embeddings']}")

        cost = sum(s["cost"] for s in stages)
        seconds = sum(s["seconds"] for s in stages)
        # Link failures are expected (403s, dead domains), not retryable here.
        failed = sum(s["failed"] for s in stages
                     if s.get("stage") != "links")
        print(f"Cost:            ${cost:.4f}")
        print(f"Time:            {seconds:.0f}s")
        print(f"Failures:        {failed}")

        links = next((s for s in stages
                      if s.get("stage") == "links"), {})
        if links.get("failed"):
            print(f"  {links['failed']} links gave no title (blocked or gone) "
                  f"— expected, not a failure")

        check = next((s for s in stages
                      if s.get("stage") == "faithfulness"), {})
        below = check.get("below") or []
        print(f"Below 0.5 faithfulness: {len(below)}")
        for tid, score in below[:10]:
            print(f"  {score:.2f}  https://x.com/i/status/{tid}")
        if len(below) > 10:
            print(f"  ... and {len(below) - 10} more")
        del captured

    def reload_vectors(self):
        """Re-read the vectors so new tweets are searchable at once."""
        t0 = time.perf_counter()
        self.ids, self.matrix = load_matrix(self.conn, "embeddings")
        print(f"\nReloaded {len(self.ids):,} vectors in "
              f"{(time.perf_counter() - t0) * 1000:.0f}ms — "
              f"new tweets are searchable now.")

    # ---- /help, /quit -----------------------------------------------
    def do_help(self, arg):
        """Show the commands, or help for one of them."""
        if arg:
            return super().do_help(arg)
        print("""
  /ask <topic>        smart search: cosine, then the Jev relevance rerank
                      add -n N for N strong matches (default 5)
  /basic <topic>      plain embedding search — free and instant
  <topic>             anything without a slash is treated as /ask
  /more               the next results for the last search, free
  /open N             open result N in the browser
  /update             process new captures from Downloads, then reload
  /status             counts, last update, captures waiting, spend
  /history            the last 10 searches
  /help [command]     this list, or detail on one command
  /quit               exit

  Up and down arrows walk through what you have typed.
""")

    def do_quit(self, arg):
        """Leave the shell."""
        print("Bye.")
        return True

    do_exit = do_quit

    def do_EOF(self, arg):
        """Ctrl-D or Ctrl-Z also exits."""
        print()
        return self.do_quit(arg)

    def postloop(self):
        self.save_line_history()
        try:
            self.conn.close()
        except Exception:                               # noqa: BLE001
            pass


def main(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    db = config = None
    if "--db" in argv:
        db = argv[argv.index("--db") + 1]
    if "--config" in argv:
        config = argv[argv.index("--config") + 1]
    try:
        shell = Shell(db=db, config=config)
    except Exception as exc:                            # noqa: BLE001
        print(f"Could not start: {type(exc).__name__}: {exc}")
        return 1
    while True:
        try:
            shell.cmdloop()
            break
        except KeyboardInterrupt:
            print("\n(use /quit to exit)")
            shell.intro = None
    return 0


if __name__ == "__main__":
    sys.exit(main())
