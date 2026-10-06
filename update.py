"""The weekly command. Everything, for new and changed rows only.

    python update.py                 # the whole pipeline
    python update.py --db copy.db    # against a copy, for testing
    python update.py --dry-run       # what each stage would do, no calls

Run it after capturing in Firefox. It picks up the capture files from
Downloads, stores them, backs them up, reads the pictures, fetches the link
titles, writes the summaries, embeds everything, judges how substantive each
tweet is, and checks the new summaries are faithful. Then it prints one
summary.

Every stage selects only what it still needs to do, so this is safe to stop
and rerun: nothing is done twice and nothing is skipped. Each stage also runs
whether or not the one before it found anything, so a week with no new tweets
but a failed image from last week still gets that image read.

The order matters. Images and link titles come before enrichment, so the
summary can draw on them; enrichment comes before embedding and judging, so
those see the summary. That is the dependency order in Section 7.1's diagram.
"""

import argparse
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "evals"))

import check_summaries                                       # noqa: E402
import describe_images                                       # noqa: E402
import embed_all                                             # noqa: E402
import enrich                                                # noqa: E402
import fetch_links                                           # noqa: E402
import ingest                                                # noqa: E402
import tag                                                   # noqa: E402
from tweet_index import spend                                # noqa: E402
from tweet_index.db import connect                           # noqa: E402


def load_config():
    return yaml.safe_load((ROOT / "config.yaml").read_text(encoding="utf-8"))


def resolve(p):
    path = Path(p).expanduser()
    return path if path.is_absolute() else ROOT / path


def log(logs_dir, message):
    logs_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).isoformat(timespec="seconds")
    with (logs_dir / "update.log").open("a", encoding="utf-8") as fh:
        fh.write(f"{stamp} {message}\n")


def counts(db):
    """A snapshot of what exists, for the before-and-after numbers."""
    conn = connect(db)
    one = lambda sql: conn.execute(sql).fetchone()[0]       # noqa: E731
    snapshot = {
        "tweets": one("SELECT COUNT(*) FROM tweets"),
        "bookmarks": one("SELECT COUNT(*) FROM tweets WHERE is_bookmark=1"),
        "likes": one("SELECT COUNT(*) FROM tweets WHERE is_like=1"),
        "both": one("SELECT COUNT(*) FROM tweets"
                    " WHERE is_like=1 AND is_bookmark=1"),
        "media": one("SELECT COUNT(*) FROM media"),
        "images_read": one("SELECT COUNT(*) FROM media"
                           " WHERE TRIM(COALESCE(ocr_text,'')) <> ''"),
        "links": one("SELECT COUNT(*) FROM links"),
        "links_titled": one("SELECT COUNT(*) FROM links"
                            " WHERE TRIM(COALESCE(title,'')) <> ''"),
        "summaries": one("SELECT COUNT(*) FROM enrichment"
                         " WHERE TRIM(COALESCE(context,'')) <> ''"),
        "embeddings": one("SELECT COUNT(*) FROM embeddings"),
        "judgments": one("SELECT COUNT(*) FROM tweet_judgments"),
    }
    conn.close()
    return snapshot


def ids_needing_judgment(db, enriched_ids):
    """Tweets to judge: never judged, or their evidence changed this run."""
    conn = connect(db)
    unjudged = {r[0] for r in conn.execute(
        "SELECT t.id FROM tweets t"
        " LEFT JOIN tweet_judgments j ON j.tweet_id = t.id"
        " WHERE j.tweet_id IS NULL")}
    conn.close()
    return sorted(unjudged | set(enriched_ids))


def write_ids(ids, name):
    """Large id lists go in a file: the command line cannot hold thousands."""
    path = Path(ROOT / "data" / "logs" / name)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(",".join(ids), encoding="utf-8")
    return path


def run(label, fn, module, dry_run=False):
    """Call one stage and collect what it reported."""
    print(f"\n{'=' * 70}\n{label}\n{'=' * 70}")
    if dry_run:
        print("  (dry run: not executed)")
        return {"step": label, "skipped_stage": True}
    module.LAST_RUN.clear()
    started = time.perf_counter()
    code = fn()
    result = dict(module.LAST_RUN)
    # The stage's own name, not the module's, so the table reads the same
    # whether or not the stage found work to do.
    result["stage"] = result.get("step", label)
    result["step"] = label
    result.setdefault("done", 0)
    result.setdefault("failed", 0)
    result.setdefault("cost", 0.0)
    result.setdefault("seconds", time.perf_counter() - started)
    result["exit"] = code
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--db", default=None,
                        help="use another database (test on a copy first)")
    parser.add_argument("--dry-run", action="store_true",
                        help="report what each stage would do, make no calls")
    parser.add_argument("--files", nargs="*", default=None,
                        help="capture files to ingest instead of ~/Downloads")
    parser.add_argument("--concurrency", type=int, default=8)
    args = parser.parse_args(argv)

    cfg = load_config()
    db = Path(args.db) if args.db else resolve(cfg["paths"]["db"])
    logs_dir = resolve("data/logs")
    db_arg = ["--db", str(db)]

    print(f"Tweet Index weekly update")
    print(f"Database: {db}")
    before = counts(db)
    print(f"Starting from {before['tweets']} tweets, "
          f"{before['images_read']}/{before['media']} images read, "
          f"{before['links_titled']}/{before['links']} links titled")

    t0 = time.perf_counter()
    stages = []

    # 1. Ingest. Moves capture files out of Downloads into data/raw, upserts
    #    the tweets, and copies each file to the backup folder.
    ingest_args = (args.files or []) + db_arg
    stages.append(run("1/7  Ingest captures", lambda: ingest.main(ingest_args),
                      ingest, args.dry_run))

    # 2. Images, before enrichment so the summary can use them.
    stages.append(run("2/7  Read images",
                      lambda: describe_images.main(
                          ["--concurrency", str(args.concurrency)] + db_arg),
                      describe_images, args.dry_run))

    # 3. Link titles, likewise.
    stages.append(run("3/7  Fetch link titles",
                      lambda: fetch_links.main(
                          ["--concurrency", str(args.concurrency)] + db_arg),
                      fetch_links, args.dry_run))

    # 4. Summaries. Anything whose payload changed in steps 1-3 is picked up
    #    by its input_hash, so this needs no list of ids.
    enrich_stage = run("4/7  Write summaries",
                       lambda: enrich.main(
                           ["--concurrency", str(args.concurrency)] + db_arg),
                       enrich, args.dry_run)
    stages.append(enrich_stage)
    enriched_ids = enrich_stage.get("ids", [])

    # 5. Embeddings, which see the summaries, image text and link titles.
    stages.append(run("5/7  Embed", lambda: embed_all.main(db_arg),
                      embed_all, args.dry_run))

    # 6. Substance, for tweets never judged or whose evidence changed.
    judge_ids = [] if args.dry_run else ids_needing_judgment(db, enriched_ids)
    if judge_ids:
        path = write_ids(judge_ids, "update_judge_ids.txt")
        stages.append(run(f"6/7  Judge substance ({len(judge_ids)} tweets)",
                          lambda: tag.main(["--ids-file", str(path),
                                            "--concurrency", "16"] + db_arg),
                          tag, args.dry_run))
    else:
        print(f"\n{'=' * 70}\n6/7  Judge substance\n{'=' * 70}")
        print("  nothing new to judge")
        stages.append({"step": "6/7  Judge substance",
                       "stage": "substance", "done": 0, "failed": 0,
                       "cost": 0.0, "seconds": 0.0})

    # 7. Faithfulness, on the summaries written this run.
    if enriched_ids:
        path = write_ids(enriched_ids, "update_check_ids.txt")
        stages.append(run(f"7/7  Check summaries ({len(enriched_ids)})",
                          lambda: check_summaries.main(
                              ["--ids-file", str(path), "--round", "8",
                               "--concurrency", "16"] + db_arg),
                          check_summaries, args.dry_run))
    else:
        print(f"\n{'=' * 70}\n7/7  Check summaries\n{'=' * 70}")
        print("  no new summaries to check")
        stages.append({"step": "7/7  Check summaries",
                       "stage": "faithfulness", "done": 0, "failed": 0,
                       "cost": 0.0, "seconds": 0.0, "below": []})

    seconds = time.perf_counter() - t0
    after = counts(db)
    spend.record_stages(logs_dir, stages)

    # ---- one summary ------------------------------------------------
    print(f"\n{'=' * 70}\nSUMMARY\n{'=' * 70}")
    new_tweets = after["tweets"] - before["tweets"]
    print(f"New tweets: {new_tweets}"
          f"   bookmarks +{after['bookmarks'] - before['bookmarks']}"
          f"   likes +{after['likes'] - before['likes']}"
          f"   in both +{after['both'] - before['both']}")
    print(f"Images read:   +{after['images_read'] - before['images_read']}"
          f"   ({after['images_read']} of {after['media']})")
    print(f"Links titled:  +{after['links_titled'] - before['links_titled']}"
          f"   ({after['links_titled']} of {after['links']})")
    print(f"Summaries:     +{after['summaries'] - before['summaries']}"
          f"   ({after['summaries']})")
    print(f"Embeddings:    +{after['embeddings'] - before['embeddings']}"
          f"   ({after['embeddings']} of {after['tweets']})")

    print(f"\n{'stage':28} {'done':>6} {'failed':>7} {'seconds':>8} {'cost':>9}")
    total_cost = 0.0
    total_failed = 0
    for s in stages:
        total_cost += s.get("cost", 0.0)
        # Link failures are counted under "skipped" below: a 403 or a dead
        # domain is not something the next run will fix by itself.
        if s.get("stage") != "links":
            total_failed += s.get("failed", 0)
        print(f"{s['step'][:28]:28} {s.get('done', 0):6} "
              f"{s.get('failed', 0):7} {s.get('seconds', 0.0):8.1f} "
              f"${s.get('cost', 0.0):8.4f}")
    print(f"{'total':28} {'':6} {total_failed:7} {seconds:8.1f} "
          f"${total_cost:8.4f}")

    skipped = []
    ingest_stage = next((s for s in stages if s.get("stage") == "ingest"), None)
    if ingest_stage and ingest_stage.get("skipped"):
        skipped.append(f"{ingest_stage['skipped']} deleted or withheld tweets "
                       f"(tombstones, will never parse)")
    link_stage = next((s for s in stages if s.get("stage") == "links"), None)
    if link_stage and link_stage.get("skipped"):
        skipped.append(f"{link_stage['skipped']} links skipped "
                       f"(x.com, PDFs, non-pages)")
    if link_stage and link_stage.get("failed"):
        skipped.append(f"{link_stage['failed']} links gave no title "
                       f"(blocked, gone or no metadata); "
                       f"`python fetch_links.py --retry-failed` to try again")
    unreadable = after["media"] - after["images_read"]
    if unreadable:
        skipped.append(f"{unreadable} images unreadable (deleted from X)")
    unembedded = after["tweets"] - after["embeddings"]
    if unembedded:
        skipped.append(f"{unembedded} tweets with nothing to embed")
    if skipped:
        print("\nSkipped:")
        for line in skipped:
            print(f"  {line}")

    check = next((s for s in stages if s.get("stage") == "faithfulness"), None)
    below = (check or {}).get("below") or []
    print(f"\nSummaries below 0.5 faithfulness: {len(below)}")
    for tid, score in below[:10]:
        print(f"  {score:.2f}  https://x.com/i/status/{tid}")
    if len(below) > 10:
        print(f"  ... and {len(below) - 10} more")

    if total_failed:
        print(f"\n{total_failed} failures. They have no row, so the next run "
              f"retries them. See data/logs/ for details.")

    log(logs_dir, f"update: +{new_tweets} tweets, {total_failed} failures, "
                  f"${total_cost:.4f}, {seconds:.0f}s")
    return 0


if __name__ == "__main__":
    sys.exit(main())
