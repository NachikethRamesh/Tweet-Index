"""What the paid calls cost, by provider and month.

Nothing in the database records money. The `queries` table stores
`jev_tokens` per search, which covers searching, but the pipeline's OpenAI
spend -- images, summaries -- was only ever printed to the terminal and lost.

A SQLite table would be the tidier home, but the schema is not changed
lightly, so this is a plain append-only JSONL file:

    {"at": "...", "provider": "openai", "usd": 0.0057, "what": "images"}

One line per stage per run. Append-only, so a crash mid-run loses at most
the stage in progress, and the file is readable with `cat` when the numbers
look wrong.

Searching is *not* recorded here: `queries.jev_tokens` already holds it, and
writing it twice would double the total. `month()` adds the two together.
"""

import json
from datetime import datetime, timezone
from pathlib import Path

LEDGER = "spend.jsonl"

# The provider each stage pays.
PROVIDERS = {
    "images": "openai",
    "summaries": "openai",
    "enrich": "openai",
    "substance": "typesafe",
    "faithfulness": "typesafe",
    "judging": "openai",
    "search": "typesafe",
}


def record(logs_dir, provider, usd, what):
    """Append one line. Never raises: a ledger is not worth losing a run for."""
    if not usd:
        return
    try:
        logs_dir = Path(logs_dir)
        logs_dir.mkdir(parents=True, exist_ok=True)
        with (logs_dir / LEDGER).open("a", encoding="utf-8") as fh:
            fh.write(json.dumps({
                "at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
                "provider": provider, "usd": round(float(usd), 6),
                "what": what,
            }) + "\n")
    except (OSError, TypeError, ValueError):
        pass


def record_stages(logs_dir, stages):
    """Record a whole update run, one line per stage that cost something.

    Each stage must carry its own `stage` key naming which step it is. An
    earlier version guessed the provider from the last word of the printed
    label, which billed the faithfulness check ("check summaries") to OpenAI
    because the label ends in "summaries" -- it is a Jev call. Guessing from
    display text was the mistake; the stage names itself now.
    """
    for s in stages:
        usd = s.get("cost") or 0.0
        if not usd:
            continue
        key = str(s.get("stage") or "unknown").lower()
        provider = PROVIDERS.get(key)
        if provider is None:
            provider = "unknown"            # better unattributed than wrong
        record(logs_dir, provider, usd, key)


def read_ledger(logs_dir):
    path = Path(logs_dir) / LEDGER
    if not path.exists():
        return []
    out = []
    try:
        for line in path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                try:
                    out.append(json.loads(line))
                except ValueError:
                    continue            # a torn last line; skip it
    except OSError:
        return []
    return out


def ledger_started(logs_dir):
    rows = read_ledger(logs_dir)
    return rows[0]["at"][:10] if rows else None


def month(conn, logs_dir, when=None):
    """Spend by provider for one month, pipeline plus searching.

    `when` is a 'YYYY-MM' string; the current UTC month by default. Returns
    (totals, searches) where totals maps provider to dollars.
    """
    when = when or datetime.now(timezone.utc).strftime("%Y-%m")
    totals = {}
    for row in read_ledger(logs_dir):
        if str(row.get("at", "")).startswith(when):
            p = row.get("provider", "unknown")
            totals[p] = totals.get(p, 0.0) + float(row.get("usd") or 0.0)

    searches = 0
    if conn is not None:
        from tweet_index import jev
        try:
            n, tokens = conn.execute(
                "SELECT COUNT(*), COALESCE(SUM(jev_tokens), 0) FROM queries"
                " WHERE substr(asked_at, 1, 7) = ?", (when,)).fetchone()
        except Exception:                               # noqa: BLE001
            n, tokens = 0, 0
        searches = n or 0
        if tokens:
            usd = tokens / 1e6 * jev.USD_PER_MILLION_INPUT
            totals["typesafe"] = totals.get("typesafe", 0.0) + usd
    return totals, searches
