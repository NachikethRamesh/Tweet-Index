"""Ask Jev how substantive each tweet is.

    python tag.py --sample 50        # a spread, to look at before committing
    python tag.py --estimate         # size and cost, no API calls
    python tag.py                    # everything not yet judged
    python tag.py --full             # re-judge everything

This is the `_substance` half of Section 10.3. The concept questions that make
up tagging proper need a vocabulary, which Phase 3 builds; until then each call
asks exactly one Score question and the answer goes to `tweet_judgments`.

Substance is a position on five named levels, not a 0-10 number: Jev's own
documentation says numeric calibration is weak, while descriptive levels are
judged consistently (Section 11.2). A score of 2.4 means "between a useful
pointer and an explanation, nearer the former".
"""

import argparse
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path

import yaml

from tweet_index import jev, ratelimit
from tweet_index.db import connect

ROOT = Path(__file__).resolve().parent

# No vocabulary exists yet. Section 5.2 uses vocab_version as the tagging
# ledger -- a tweet is tagged at the current version when its row matches -- so
# 0 means "judged before any vocabulary existed". Phase 3 will re-judge at
# version 1, which it would have to do anyway to ask the concept questions.
NO_VOCABULARY = 0

LAST_RUN = {}

def read_ids(args):
    """Tweet ids from --ids or --ids-file, or None for all.

    3,562 ids is 71,000 characters, well past the command line's limit, so
    large sets come from a file.
    """
    from pathlib import Path as _P
    text = None
    if getattr(args, "ids_file", None):
        text = _P(args.ids_file).read_text(encoding="utf-8")
    elif args.ids:
        text = args.ids
    if not text:
        return None
    return {p.strip() for p in text.replace("\n", ",").split(",") if p.strip()}


COLUMNS = ["id", "author_handle", "text", "quoted_text", "card_title",
           "card_description", "article_title", "article_preview",
           "translation_en", "community_note"]


def load_config(path=None):
    return yaml.safe_load((Path(path) if path else ROOT / "config.yaml").read_text(encoding="utf-8"))


def resolve(p):
    path = Path(p).expanduser()
    return path if path.is_absolute() else ROOT / path


def load_env():
    env = ROOT / ".env"
    if not env.exists():
        return
    import os
    for line in env.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line and not line.startswith("#") and "=" in line:
            k, v = line.split("=", 1)
            os.environ.setdefault(k.strip(), v.strip().strip('"').strip("'"))


def log(logs_dir, message):
    logs_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).isoformat(timespec="seconds")
    with (logs_dir / "tag.log").open("a", encoding="utf-8") as fh:
        fh.write(f"{stamp} {message}\n")


def candidates(conn, vocab_version=NO_VOCABULARY, full=False):
    """Tweets with no judgment at this vocabulary version."""
    columns = ", ".join("t." + c for c in COLUMNS)
    rows = conn.execute(f"""
        SELECT {columns}, e.context,
               j.vocab_version AS judged_at_version
        FROM tweets t
        LEFT JOIN enrichment e ON e.tweet_id = t.id
        LEFT JOIN tweet_judgments j ON j.tweet_id = t.id
        ORDER BY t.id""").fetchall()
    if full:
        return rows
    return [r for r in rows if r["judged_at_version"] != vocab_version]


def side_data(conn, ids):
    """Media kinds, link titles and image text, keyed by tweet id."""
    media, links, image_text = {}, {}, {}
    if not ids:
        return media, links, image_text
    marks = ",".join("?" * len(ids))
    for r in conn.execute(f"SELECT tweet_id, kind, ocr_text FROM media"
                          f" WHERE tweet_id IN ({marks}) ORDER BY id", ids):
        media.setdefault(r["tweet_id"], []).append(r["kind"])
        if (r["ocr_text"] or "").strip():
            image_text.setdefault(r["tweet_id"], []).append(r["ocr_text"].strip())
    for r in conn.execute(f"SELECT tweet_id, title FROM links"
                          f" WHERE tweet_id IN ({marks})", ids):
        if r["title"]:
            links.setdefault(r["tweet_id"], []).append(r["title"])
    return media, links, {k: "\n".join(v) for k, v in image_text.items()}


def judge_one(row, media, links, image_text=None):
    """One Score question about one tweet."""
    state = jev.tweet_state(row, link_titles=links, media=media,
                            image_text=image_text)
    answers, version, tokens = jev.judge_tweet(
        state, {"_substance": jev.substance_question()})
    answer = answers["_substance"]
    return {
        "tweet_id": row["id"],
        "substance": answer.score,
        "substance_conf": answer.confidence,
        "vocab_version": NO_VOCABULARY,
        "jev_version": version,
        "judged_at": datetime.now(timezone.utc).isoformat(),
    }, tokens


def save(conn, records):
    with conn:
        conn.executemany(
            "INSERT INTO tweet_judgments (tweet_id, substance, substance_conf,"
            " vocab_version, jev_version, judged_at) VALUES"
            " (:tweet_id, :substance, :substance_conf, :vocab_version,"
            "  :jev_version, :judged_at)"
            " ON CONFLICT(tweet_id) DO UPDATE SET"
            "   substance = excluded.substance,"
            "   substance_conf = excluded.substance_conf,"
            "   vocab_version = excluded.vocab_version,"
            "   jev_version = excluded.jev_version,"
            "   judged_at = excluded.judged_at", records)


def pick_sample(rows, size, media, links):
    """A spread across text length and whether a summary exists."""
    buckets = {}
    for r in rows:
        length = len(r["text"] or "")
        band = ("no text" if length == 0 else "terse" if length < 60
                else "ordinary" if length < 300 else "long")
        has_summary = bool((r["context"] or "").strip())
        buckets.setdefault((band, has_summary), []).append(r)
    keys = sorted(buckets, key=lambda k: (k[0], k[1]))
    for k in keys:
        buckets[k].sort(key=lambda r: r["id"])
    chosen, i = [], 0
    while len(chosen) < size and any(buckets[k] for k in keys):
        k = keys[i % len(keys)]
        if buckets[k]:
            chosen.append(buckets[k].pop(0))
        i += 1
    return chosen


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sample", type=int, metavar="N",
                        help="judge a spread of N tweets")
    parser.add_argument("--limit", type=int, metavar="N")
    parser.add_argument("--ids", default=None,
                        help="comma-separated tweet ids to re-judge")
    parser.add_argument("--ids-file", default=None,
                        help="read tweet ids from a file, one per line or comma-separated; use this instead of --ids for more than a few hundred, which overflow the command line")
    parser.add_argument("--full", action="store_true",
                        help="re-judge tweets that already have a row")
    parser.add_argument("--estimate", action="store_true",
                        help="report size and cost without calling Jev")
    parser.add_argument("--concurrency", type=int, default=16)
    parser.add_argument("--db", default=None,
                        help="use another database (experiments use a copy)")
    parser.add_argument("--config", default=None)
    args = parser.parse_args(argv)

    cfg = load_config(args.config)
    conn = connect(Path(args.db) if args.db else resolve(cfg["paths"]["db"]))
    logs_dir = resolve("data/logs")

    rows = candidates(conn, full=args.full)
    wanted = read_ids(args)
    if wanted:
        rows = [r for r in candidates(conn, full=True) if r["id"] in wanted]
    total = conn.execute("SELECT COUNT(*) FROM tweets").fetchone()[0]
    print(f"Tweets in database: {total}")
    print(f"{'To re-judge:' if args.full else 'Needing judgment:':20}{len(rows)}")

    media, links, image_text = side_data(conn, [r["id"] for r in rows])

    if args.estimate:
        import json as _json
        jev.configure(cfg["jev"]["model"])
        sizes = []
        for r in rows:
            state = jev.tweet_state(r, link_titles=links.get(r["id"], []),
                                    media=media.get(r["id"], []),
                                    image_text=image_text.get(r["id"]))
            sizes.append(len(_json.dumps(state, ensure_ascii=False,
                                         default=str)))
        tokens = sum(sizes) / 3.75 + len(rows) * 120   # question + rubric
        print(f"\nState size: median {sorted(sizes)[len(sizes) // 2]} chars, "
              f"max {max(sizes)}")
        print(f"Input tokens: ~{tokens / 1e6:.2f}M")
        print(f"Cost: ${tokens / 1e6 * jev.USD_PER_MILLION_INPUT:.4f}")
        limiter = ratelimit.from_config(cfg, "typesafe")
        if limiter:
            per_min = min(limiter.requests.limit,
                          limiter.tokens.limit / (tokens / len(rows)))
            print(f"Time at the limiter's pace: ~{len(rows) / per_min:.1f} min")
        conn.close()
        return 0

    if args.sample:
        rows = pick_sample(rows, args.sample, media, links)
        print(f"Sample selected:    {len(rows)}")
    elif args.limit:
        rows = rows[:args.limit]

    if not rows:
        print("\nNothing to do.")
        conn.close()
        return 0

    load_env()
    jev.configure(cfg["jev"]["model"])
    limiter = ratelimit.from_config(cfg, "typesafe")
    jev.set_limiter(limiter)
    if limiter:
        print(f"Rate limiter: {limiter.requests.limit:.0f} req/min,"
              f" {limiter.tokens.limit:,.0f} tokens/min")
    print(f"Model: {jev.model()}   concurrency: {args.concurrency}\n")

    cost = jev.CostLog(resolve("data/logs/jev.log"))
    t0 = time.perf_counter()
    records, failed = [], 0

    def work(row):
        try:
            return judge_one(row, media.get(row["id"], []),
                             links.get(row["id"], []),
                             image_text.get(row["id"])), None
        except jev.JevError as exc:
            return None, (row["id"], str(exc))

    with ThreadPoolExecutor(max_workers=args.concurrency) as pool:
        for result, error in pool.map(work, rows):
            if error:
                failed += 1
                cost.record_failure()
                log(logs_dir, f"failed {error[0]}: {error[1]}")
                print(f"  failed {error[0]}: {error[1]}")
                continue
            record, tokens = result
            records.append(record)
            cost.record(tokens)
            print(f"\r  {len(records) + failed}/{len(rows)} judged   ",
                  end="", flush=True)
    seconds = time.perf_counter() - t0
    print()

    save(conn, records)
    print(f"\nJudged {len(records)}, failed {failed}, in {seconds:.1f}s")
    if limiter:
        st = limiter.stats()
        print(f"Pacing: {st['calls_per_second']:.2f} calls/sec")
    print(f"Cost: {cost.summary()}")

    scores = sorted(r["substance"] for r in records)
    if scores:
        print(f"\nSubstance: median {scores[len(scores) // 2]:.2f}, "
              f"min {scores[0]:.2f}, max {scores[-1]:.2f}")
        for low in range(5):
            n = sum(1 for s in scores if low <= s < low + 1)
            print(f"  level {low}: {n:5}  ({n / len(scores) * 100:5.1f}%)")
    cost.write("tag --substance")
    log(logs_dir, f"judged {len(records)}, failed {failed}, {seconds:.1f}s")
    LAST_RUN.clear()
    LAST_RUN.update(step="substance", done=len(records), failed=failed,
                    seconds=seconds, cost=cost.usd)
    conn.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
