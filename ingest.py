"""Move capture files into data/raw, parse them, and fill the database.

This is Section 7.3 of the technical design. Run it after the userscript has
downloaded a capture:

    python ingest.py                 # pick up capture_*.json from ~/Downloads
    python ingest.py path/to/file    # or name files explicitly

Safe to run as often as you like. Files already ingested are skipped by name,
and a tweet seen again simply has its flags OR-ed onto the existing row, so a
second run adds nothing.
"""

import argparse
import json
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path

import yaml

from tweet_index.db import connect, count
from tweet_index.parse import iter_entries, parse_entry

ROOT = Path(__file__).resolve().parent

# The tweets table, in the order the upsert below binds them.
COLS = ["id", "url", "author_handle", "author_name", "text", "lang",
        "quoted_id", "quoted_author", "quoted_text", "in_reply_to_id",
        "initial_tweet_id", "card_title", "card_description", "article_title",
        "article_preview", "translation_en", "community_note", "save_order",
        "posted_at", "first_seen", "is_like", "is_bookmark", "raw_file"]

LAST_RUN = {}

UPSERT = f"""
INSERT INTO tweets ({", ".join(COLS)})
VALUES ({", ".join(":" + c for c in COLS)})
ON CONFLICT(id) DO UPDATE SET
  is_like        = MAX(tweets.is_like,     excluded.is_like),
  is_bookmark    = MAX(tweets.is_bookmark, excluded.is_bookmark),
  community_note = COALESCE(excluded.community_note, tweets.community_note)
  -- first_seen, raw_file, save_order deliberately NOT updated
"""


def load_config(path=None):
    path = Path(path or ROOT / "config.yaml")
    return yaml.safe_load(path.read_text(encoding="utf-8"))


def resolve(cfg_path):
    """A path from config.yaml, absolute and with ~ expanded."""
    p = Path(cfg_path).expanduser()
    return p if p.is_absolute() else ROOT / p


def log(logs_dir, message):
    """Append one timestamped line to data/logs/ingest.log."""
    logs_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).isoformat(timespec="seconds")
    with (logs_dir / "ingest.log").open("a", encoding="utf-8") as fh:
        fh.write(f"{stamp} {message}\n")


def skip_reason(entry):
    """Why parse_entry refused this entry, in words, for the log."""
    result = (((entry.get("content") or {}).get("itemContent") or {})
              .get("tweet_results") or {}).get("result")
    if not isinstance(result, dict):
        return "no tweet_results.result"
    typename = result.get("__typename")
    if typename not in ("Tweet", "TweetWithVisibilityResults", None):
        return typename                      # TweetTombstone, TweetUnavailable
    return "no rest_id"


def back_up_raw(path, backup_dir):
    """Copy one capture to the backup folder, if it is not already there.

    The captures are the only data that cannot be rebuilt, so they get a
    second copy somewhere synced. They are write-once, which makes them safe
    to sync; the database is not -- a sync client writing to an open SQLite
    file is a good way to corrupt it.
    """
    if not backup_dir:
        return None
    backup_dir = Path(backup_dir).expanduser()
    dest = backup_dir / path.name
    if dest.exists() and dest.stat().st_size == path.stat().st_size:
        return None
    backup_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy2(path, dest)
    return dest


def move_new_captures(downloads, raw):
    """Move capture_*.json out of Downloads into data/raw. Returns new paths."""
    raw.mkdir(parents=True, exist_ok=True)
    moved = []
    for src in sorted(downloads.glob("capture_*.json")):
        dest = raw / src.name
        if dest.exists():
            continue
        shutil.move(str(src), str(dest))
        moved.append(dest)
    return moved


def ingest_file(conn, path, logs_dir):
    """Parse one capture file and upsert everything in it.

    Returns a dict of counts. Does nothing if the file was ingested before.
    """
    already = conn.execute("SELECT 1 FROM ingested_files WHERE filename=?",
                           (path.name,)).fetchone()
    if already:
        return {"skipped_file": True, "new": 0, "seen": 0,
                "skipped": 0, "reasons": {}}

    blob = json.loads(path.read_text(encoding="utf-8"))
    kind = blob["kind"]          # 'bookmarks' | 'likes', set by the userscript
    now = datetime.now(timezone.utc).isoformat()
    new = seen = skipped = 0
    reasons = {}

    with conn:                   # one transaction per file
        for page in blob["pages"]:
            for entry in iter_entries(page["json"]):
                t = parse_entry(entry)
                if not t:
                    reason = skip_reason(entry)
                    reasons[reason] = reasons.get(reason, 0) + 1
                    log(logs_dir,
                        f"skip {path.name} {entry.get('entryId')}: {reason}")
                    skipped += 1
                    continue

                seen += 1
                existed = conn.execute("SELECT 1 FROM tweets WHERE id=?",
                                       (t["id"],)).fetchone()

                # The feed decides the flag. x_favorited / x_bookmarked are the
                # payload's own opinion, logged only when they disagree.
                is_like = int(kind == "likes")
                is_bookmark = int(kind == "bookmarks")
                claimed = t["x_favorited"] if kind == "likes" else t["x_bookmarked"]
                if claimed is False:
                    log(logs_dir, f"flag mismatch {path.name} {t['id']}: "
                                  f"from {kind} but favorited="
                                  f"{t['x_favorited']} bookmarked={t['x_bookmarked']}")

                row = {k: t.get(k) for k in COLS} | {
                    "first_seen": now, "raw_file": path.name,
                    "is_like": is_like, "is_bookmark": is_bookmark}
                conn.execute(UPSERT, row)
                new += not existed

                for mtype, murl in t["media"]:
                    conn.execute("INSERT OR IGNORE INTO media (tweet_id, kind,"
                                 " remote_url) VALUES (?,?,?)",
                                 (t["id"], mtype, murl))
                for u in t["links"]:
                    conn.execute("INSERT OR IGNORE INTO links (tweet_id, url)"
                                 " VALUES (?,?)", (t["id"], u))

        conn.execute("INSERT INTO ingested_files (filename, kind, ingested_at,"
                     " entries, skipped) VALUES (?,?,?,?,?)",
                     (path.name, kind, now, seen, skipped))

    log(logs_dir, f"ingested {path.name} ({kind}): {new} new, {seen} entries, "
                  f"{skipped} skipped")
    return {"skipped_file": False, "new": new, "seen": seen,
            "skipped": skipped, "reasons": reasons}


def summarise(conn, results):
    """Print what the run did, and what the database now holds."""
    print("\nFiles")
    for name, r in results:
        if r["skipped_file"]:
            print(f"  {name}: already ingested, skipped")
        else:
            print(f"  {name}: {r['new']} new, {r['seen']} entries parsed, "
                  f"{r['skipped']} skipped")

    reasons = {}
    for _, r in results:
        for k, v in r["reasons"].items():
            reasons[k] = reasons.get(k, 0) + v
    if reasons:
        print("\nSkipped entries")
        for reason, n in sorted(reasons.items()):
            print(f"  {n} x {reason}")

    print("\nRows per table")
    for table in ("tweets", "media", "links", "ingested_files"):
        print(f"  {table:15} {count(conn, table)}")
    for table in ("enrichment", "embeddings", "concepts", "concept_scores",
                  "tweet_judgments", "tweet_tags", "queries"):
        print(f"  {table:15} {count(conn, table)}   (filled in later phases)")

    one = conn.execute(
        "SELECT SUM(is_bookmark), SUM(is_like),"
        " SUM(is_like AND is_bookmark) FROM tweets").fetchone()
    print(f"\nBookmarked: {one[0]}   Liked: {one[1]}   Both: {one[2]}")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("files", nargs="*", type=Path,
                        help="capture files to ingest (default: ~/Downloads)")
    parser.add_argument("--db", default=None,
                        help="use another database (experiments use a copy)")
    parser.add_argument("--config", default=None)
    args = parser.parse_args(argv)

    cfg = load_config(args.config)
    raw = resolve(cfg["paths"]["raw"])
    logs_dir = resolve("data/logs")
    conn = connect(Path(args.db) if args.db else resolve(cfg["paths"]["db"]))

    if args.files:
        raw.mkdir(parents=True, exist_ok=True)
        targets = []
        for src in args.files:
            dest = raw / src.name
            if src.resolve() != dest.resolve() and not dest.exists():
                shutil.move(str(src), str(dest))
            targets.append(dest)
    else:
        downloads = resolve(cfg["paths"]["downloads"])
        moved = move_new_captures(downloads, raw)
        print(f"Moved {len(moved)} capture file(s) to {raw}")
        targets = sorted(raw.glob("capture_*.json"))

    if not targets:
        print("Nothing to ingest.")
        return 0

    results = [(p.name, ingest_file(conn, p, logs_dir)) for p in targets]

    backup_dir = cfg["paths"].get("raw_backup")
    backed_up = [back_up_raw(p, backup_dir) for p in targets]
    backed_up = [b for b in backed_up if b]
    if backup_dir:
        print(f"Backed up {len(backed_up)} capture file(s) to "
              f"{Path(backup_dir).expanduser()}")
        for b in backed_up:
            log(logs_dir, f"backed up {b.name} to {b.parent}")

    summarise(conn, results)
    LAST_RUN.clear()
    LAST_RUN.update(
        step="ingest",
        done=sum(r["new"] for _, r in results),
        # Entries parsed, new or not: `seen - new` is how many were already
        # saved, which is what a weekly run mostly consists of.
        seen=sum(r["seen"] for _, r in results),
        failed=0,
        # Tombstones are expected, not failures: a deleted or withheld tweet
        # will never parse, however many times it is tried.
        skipped=sum(r["skipped"] for _, r in results),
        files=len(targets),
        backed_up=len(backed_up),
        cost=0.0)
    conn.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
