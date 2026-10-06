"""Embed every tweet that needs it, and store the vectors.

    python embed_all.py             # only what is new or changed
    python embed_all.py --all       # re-embed everything

Safe to run repeatedly. A tweet is embedded when it has no vector yet, when the
text fed to the model has changed (a different hash), or when the model name in
config.yaml has changed. A second run with nothing new does no work.

Tweets with no text at all are skipped rather than embedded as an empty string,
which would produce a vector that matches any query. Their ids go to
data/logs/embed.log so Phase 4 (image OCR) knows what to target.
"""

import argparse
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import yaml

from tweet_index.db import connect
from tweet_index.embed import (DOC_PARTS, SOURCES, doc_text, embed_docs, model,
                               text_hash, to_blob)

ROOT = Path(__file__).resolve().parent
BATCH = 64

LAST_RUN = {}


def load_config(path=None):
    return yaml.safe_load(Path(path or ROOT / "config.yaml")
                          .read_text(encoding="utf-8"))


def resolve(p):
    path = Path(p).expanduser()
    return path if path.is_absolute() else ROOT / path


def log(logs_dir, message):
    logs_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).isoformat(timespec="seconds")
    with (logs_dir / "embed.log").open("a", encoding="utf-8") as fh:
        fh.write(f"{stamp} {message}\n")


def plan(conn, model_name, redo_all):
    """Decide what to embed.

    Returns (todo, skipped_ids, unchanged) where todo is a list of
    (tweet_id, text, hash).
    """
    # image_text is one row per image, so it is collapsed to one string per
    # tweet first; the others are plain columns.
    columns = ", ".join(f"{SOURCES[c]}.{c} AS {c}" for c in DOC_PARTS)
    rows = conn.execute(
        f"SELECT t.id, {columns}, e.text_hash AS old_hash,"
        f" e.model_used AS old_model"
        f" FROM tweets t"
        f" LEFT JOIN embeddings e ON e.tweet_id = t.id"
        f" LEFT JOIN enrichment en ON en.tweet_id = t.id"
        f" LEFT JOIN (SELECT tweet_id,"
        f"                   GROUP_CONCAT(ocr_text, char(10)) AS image_text"
        f"            FROM (SELECT tweet_id, ocr_text FROM media"
        f"                  WHERE TRIM(COALESCE(ocr_text,'')) <> ''"
        f"                  ORDER BY id)"
        f"            GROUP BY tweet_id) im ON im.tweet_id = t.id"
        f" LEFT JOIN (SELECT tweet_id,"
        f"                   GROUP_CONCAT(title, char(10)) AS link_titles"
        f"            FROM (SELECT tweet_id, title FROM links"
        f"                  WHERE TRIM(COALESCE(title,'')) <> ''"
        f"                  ORDER BY id)"
        f"            GROUP BY tweet_id) lk ON lk.tweet_id = t.id").fetchall()

    todo, skipped, unchanged = [], [], 0
    for row in rows:
        text = doc_text(row)
        if not text:
            skipped.append(row["id"])
            continue
        digest = text_hash(text)
        fresh = (row["old_hash"] == digest and row["old_model"] == model_name)
        if fresh and not redo_all:
            unchanged += 1
            continue
        todo.append((row["id"], text, digest))
    return todo, skipped, unchanged


def store(conn, batch, vectors, model_name):
    with conn:
        conn.executemany(
            "INSERT INTO embeddings (tweet_id, vector, model_used, text_hash)"
            " VALUES (?,?,?,?)"
            " ON CONFLICT(tweet_id) DO UPDATE SET"
            "   vector = excluded.vector,"
            "   model_used = excluded.model_used,"
            "   text_hash = excluded.text_hash",
            [(tid, to_blob(vec), model_name, digest)
             for (tid, _, digest), vec in zip(batch, vectors)])


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--all", action="store_true",
                        help="re-embed every tweet, not just new or changed")
    parser.add_argument("--db", default=None,
                        help="use another database (experiments use a copy)")
    parser.add_argument("--config", default=None)
    args = parser.parse_args(argv)

    cfg = load_config(args.config)
    model_name = cfg["embedding"]["model"]
    conn = connect(Path(args.db) if args.db else resolve(cfg["paths"]["db"]))
    logs_dir = resolve("data/logs")

    todo, skipped, unchanged = plan(conn, model_name, args.all)

    print(f"Model:      {model_name}")
    print(f"To embed:   {len(todo)}")
    print(f"Unchanged:  {unchanged}")
    print(f"No text:    {len(skipped)} skipped (logged for Phase 4 OCR)")

    if skipped:
        log(logs_dir, f"skipped {len(skipped)} tweets with no embeddable text; "
                      f"ids: {','.join(skipped)}")

    if not todo:
        print("\nNothing to do.")
        conn.close()
        return 0

    t0 = time.perf_counter()
    model(model_name)                      # pay the load cost before timing work
    load_seconds = time.perf_counter() - t0
    print(f"\nModel loaded in {load_seconds:.1f}s")

    t1 = time.perf_counter()
    done = 0
    for start in range(0, len(todo), BATCH):
        batch = todo[start:start + BATCH]
        vectors = embed_docs([text for _, text, _ in batch], model_name, BATCH)
        store(conn, batch, vectors, model_name)
        done += len(batch)
        elapsed = time.perf_counter() - t1
        rate = done / elapsed if elapsed else 0
        left = (len(todo) - done) / rate if rate else 0
        print(f"\r  {done}/{len(todo)}  {rate:.0f}/s  ~{left:.0f}s left   ",
              end="", flush=True)
    embed_seconds = time.perf_counter() - t1
    print()

    total = conn.execute("SELECT COUNT(*) FROM embeddings").fetchone()[0]
    db_bytes = resolve(cfg["paths"]["db"]).stat().st_size

    print(f"\nEmbedded {done} tweets in {embed_seconds:.1f}s "
          f"({done / embed_seconds:.0f}/s)")
    print(f"Vectors in database: {total}")
    print(f"Database size:       {db_bytes / 1_048_576:.1f} MB")
    log(logs_dir, f"embedded {done} tweets in {embed_seconds:.1f}s with "
                  f"{model_name}; {unchanged} unchanged, {len(skipped)} skipped; "
                  f"{total} vectors total")
    LAST_RUN.clear()
    LAST_RUN.update(step="embed", done=done, failed=0,
                    skipped=len(skipped), seconds=embed_seconds, cost=0.0)
    conn.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
