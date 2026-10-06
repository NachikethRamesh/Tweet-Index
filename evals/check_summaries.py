"""Ask Jev whether each enrichment summary is faithful to its tweet.

    python evals/check_summaries.py --sample 50     # estimate first
    python evals/check_summaries.py                 # check them all
    python evals/check_summaries.py --below 0.5     # list the failures

A wrong summary is worse than a bland one: a bland summary just fails to help,
while a wrong one sends the tweet to the wrong searches and buries it for the
right ones. Nothing else in the pipeline would notice.

So one Noul question per tweet: does the summary describe what is actually in
the tweet, without introducing topics that are not there? Jev answers with a
probability, which is exactly the shape of question it is built for -- a fixed
judgment against supplied evidence, no text generation.

Results go to the `summary_checks` table so a re-check can be compared against
the first run.
"""

import argparse
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from tweet_index import jev, ratelimit                      # noqa: E402
from tweet_index.db import connect                          # noqa: E402

# The harm being looked for is MISROUTING: a summary that sends a tweet to the
# wrong searches. Correct background the model adds from world knowledge is not
# that -- it is often what makes an oblique tweet findable at all (a tweet about
# "a man from Paris, who played for Paris" became findable as Kingsley Coman).
# So the question asks whether the subject matches, not whether every detail is
# quoted from the text.
QUESTION = (
    "Does `summary` describe the same subject as `tweet`? Answer yes if the "
    "subject matches, even where `summary` adds correct background detail that "
    "is not written in the tweet. "
    "Where the tweet is mainly a reaction to `quoted_tweet` - a short comment, "
    "joke or endorsement above substantial quoted content - answer yes if "
    "`summary` describes the quoted post's subject and indicates that the "
    "tweet reacts to or endorses it; that is the wanted form, because the "
    "quoted post is what the save is about. "
    "Answer no only if `summary` is about a different subject, contradicts the "
    "tweet or the quoted post, or attributes either to the wrong person, work "
    "or event."
)

SCHEMA = """
CREATE TABLE IF NOT EXISTS summary_checks (
  tweet_id    TEXT PRIMARY KEY REFERENCES tweets(id),
  faithful    REAL NOT NULL,          -- Jev Noul probability, 0..1
  jev_version TEXT NOT NULL,
  checked_at  TEXT NOT NULL,
  round       INTEGER NOT NULL DEFAULT 1
);
"""


LAST_RUN = {}


def read_ids(args):
    """Tweet ids from --ids or --ids-file, or None for all.

    3,562 ids is 71,000 characters, well past the command line's limit, so
    large sets come from a file.
    """
    text = None
    if getattr(args, "ids_file", None):
        text = Path(args.ids_file).read_text(encoding="utf-8")
    elif args.ids:
        text = args.ids
    if not text:
        return None
    return {p.strip() for p in text.replace("\n", ",").split(",") if p.strip()}


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


def candidates(conn, only=None):
    """Tweets with a non-empty summary. `only` restricts to a set of ids."""
    rows = conn.execute("""
        SELECT t.id, t.author_handle, t.text, t.quoted_text, t.card_title,
               t.card_description, t.article_title, t.article_preview,
               t.translation_en, t.community_note, e.context
        FROM enrichment e JOIN tweets t ON t.id = e.tweet_id
        WHERE TRIM(COALESCE(e.context,'')) <> ''
        ORDER BY t.id""").fetchall()
    if only is not None:
        rows = [r for r in rows if r["id"] in only]
    return rows


# Evidence is NOT clipped to jev.MAX_FIELD_CHARS here. That 600-character clip
# keeps tagging states small (Section 12.2), but for a faithfulness check Jev
# must see what the summariser saw: enrichment sends the full tweet text. With
# the clip in place, 92.7% of tweets over 2,000 characters were judged
# unfaithful purely because the evidence for their later sentences had been cut
# away. The cap below only bounds the handful of enormous posts.
CHECK_CLIP = 8000


def clip(value, limit=CHECK_CLIP):
    if not value:
        return None
    value = str(value).strip()
    return value[:limit] if value else None


def join(*parts):
    kept = [p.strip() for p in parts if p and p.strip()]
    return " - ".join(kept) or None


def state_for(row, media=()):
    """What Jev sees: the evidence, and the summary to judge against it.

    The evidence must be everything enrichment sent, or the check penalises
    summaries for using signals it cannot see. Omitting community_note alone
    made fact-check summaries look invented.
    """
    return {
        "tweet": {"author": row["author_handle"], "text": clip(row["text"])},
        "quoted_tweet": clip(row["quoted_text"]),
        "card": clip(join(row["card_title"], row["card_description"])),
        "article": clip(join(row["article_title"], row["article_preview"])),
        "translation_en": clip(row["translation_en"]),
        "community_note": clip(row["community_note"]),
        "media": list(media),
        "summary": clip(row["context"]),
    }


def check_one(row, media_by_id):
    questions = {"faithful": jev.sdk().Noul(instructions=QUESTION)}
    state = state_for(row, media_by_id.get(row["id"], []))
    answers, version, tokens = jev.judge_tweet(state, questions)
    return row["id"], answers["faithful"].noul, version, tokens


def save(conn, results, round_number):
    now = datetime.now(timezone.utc).isoformat()
    with conn:
        conn.executemany(
            "INSERT INTO summary_checks (tweet_id, faithful, jev_version,"
            " checked_at, round) VALUES (?,?,?,?,?)"
            " ON CONFLICT(tweet_id) DO UPDATE SET"
            "   faithful = excluded.faithful,"
            "   jev_version = excluded.jev_version,"
            "   checked_at = excluded.checked_at,"
            "   round = excluded.round",
            [(tid, score, version, now, round_number)
             for tid, score, version, _ in results])


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sample", type=int, metavar="N",
                        help="check only the first N, to estimate cost")
    parser.add_argument("--ids", default=None,
                        help="comma-separated tweet ids to check")
    parser.add_argument("--ids-file", default=None,
                        help="read tweet ids from a file, one per line or comma-separated; use this instead of --ids for more than a few hundred, which overflow the command line")
    parser.add_argument("--round", type=int, default=1,
                        help="which pass this is, recorded on each row")
    parser.add_argument("--concurrency", type=int, default=16)
    parser.add_argument("--db", default=None)
    parser.add_argument("--config", default=None)
    args = parser.parse_args(argv)

    cfg = load_config(args.config)
    load_env()
    jev.configure(cfg["jev"]["model"])
    limiter = ratelimit.from_config(cfg, "typesafe")
    jev.set_limiter(limiter)

    conn = connect(Path(args.db) if args.db else resolve(cfg["paths"]["db"]))
    with conn:
        conn.executescript(SCHEMA)

    only = read_ids(args)
    rows = candidates(conn, only)
    if args.sample:
        rows = rows[:args.sample]

    print(f"Summaries to check: {len(rows)}")
    if limiter:
        print(f"Rate limiter: {limiter.requests.limit:.0f} req/min,"
              f" {limiter.tokens.limit:,.0f} tokens/min")

    log = jev.CostLog(resolve("data/logs/jev.log"))
    t0 = time.perf_counter()
    results, failures = [], 0

    media_by_id = {}
    for r in conn.execute("SELECT tweet_id, kind FROM media ORDER BY id"):
        media_by_id.setdefault(r["tweet_id"], []).append(r["kind"])

    def work(row):
        try:
            return check_one(row, media_by_id), None
        except jev.JevError as exc:
            return None, (row["id"], str(exc))

    with ThreadPoolExecutor(max_workers=args.concurrency) as pool:
        for result, error in pool.map(work, rows):
            if error:
                failures += 1
                log.record_failure()
                print(f"  failed {error[0]}: {error[1]}")
                continue
            results.append(result)
            log.record(result[3])
            print(f"\r  {len(results) + failures}/{len(rows)} checked   ",
                  end="", flush=True)
    seconds = time.perf_counter() - t0
    print()

    save(conn, results, args.round)
    scores = [r[1] for r in results]
    print(f"\nChecked {len(results)}, failed {failures}, in {seconds:.1f}s")
    print(f"Cost: {log.summary()}")
    if scores:
        scores_sorted = sorted(scores)
        print(f"Faithfulness: median {scores_sorted[len(scores) // 2]:.3f}, "
              f"min {scores_sorted[0]:.3f}")
        for threshold in (0.5, 0.7, 0.9):
            n = sum(1 for s in scores if s < threshold)
            print(f"  below {threshold}: {n}  ({n / len(scores) * 100:.2f}%)")
        if args.sample:
            per_call = log.input_tokens / len(results)
            total = len(candidates(conn))
            projected = total * per_call / 1e6 * jev.USD_PER_MILLION_INPUT
            print(f"\nProjected for all {total}: "
                  f"{per_call:.0f} tokens/call, ${projected:.3f}")
    log.write("check_summaries")
    LAST_RUN.clear()
    LAST_RUN.update(step="faithfulness", done=len(results), failed=failures,
                    seconds=seconds, cost=log.usd,
                    below=[(tid, round(score, 3)) for tid, score, _v, _t
                           in results if score < 0.5])
    conn.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
