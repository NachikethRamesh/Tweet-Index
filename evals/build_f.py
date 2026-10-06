"""Add version F -- the pipeline as it stands now -- to the candidate pool.

    python evals/build_f.py              # build F, extend the pool
    python evals/build_f.py --floor 0.40 # re-rank F from stored scores only

F runs the same code as version C. The difference is the database underneath
it: since C was built, every image has been read, every link title fetched,
the affected tweets re-enriched, re-embedded and re-judged. So F measures what
Phase 4 bought, with the ranking logic held constant.

What is preserved and what is added:

  * A, B and C keep the rankings they were built with. They are not re-run.
  * The snapshot entries already in the pool are left exactly as they were, so
    every judgment already made still refers to the text it was made against.
  * Only tweets F found that nobody has seen are snapshotted, from the
    database as it is now -- which is right, because that is the text F
    surfaces and the text a judge would be shown.

The one honest wrinkle: a tweet in both C's and F's lists was judged against
its Phase 2 summary, which may since have been rewritten. Re-judging those
would mean throwing away 371 judgments to re-make them, so the judgments
stand and this note records it.
"""

import argparse
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "evals"))

import judge as J                                            # noqa: E402
from tweet_index import jev, ratelimit                        # noqa: E402
from tweet_index.db import connect                            # noqa: E402
from tweet_index.embed import embed_query, load_matrix, search  # noqa: E402

import ask                                                    # noqa: E402

LABEL = ("Phase 4: images and link titles in the index, "
         "same Jev rerank as C")


def snapshot_row(conn, tid):
    row = conn.execute("""
        SELECT t.author_handle, t.text, t.url, t.is_like, t.is_bookmark,
               e.context
        FROM tweets t LEFT JOIN enrichment e ON e.tweet_id = t.id
        WHERE t.id = ?""", (tid,)).fetchone()
    return None if row is None else {k: row[k] for k in row.keys()}


def scored_rows(conn, cfg, topic, workers, floor=None):
    """F's candidates with every component score, so a floor can be varied.

    Returns (rows, elapsed, jev_tokens, failures) where each row is a dict
    with the tweet id, relevance, confidence, substance, cosine and the
    blended final score.
    """
    ask_cfg = dict(cfg["ask"])
    if floor is not None:
        ask_cfg["confidence_floor"] = floor
    local = dict(cfg, ask=ask_cfg)

    ids, matrix = load_matrix(conn, "embeddings")
    t0 = time.perf_counter()
    vector = embed_query(topic, cfg["embedding"]["model"],
                         cfg["embedding"]["query_prefix"])
    candidates = search(vector, ids, matrix, ask_cfg["candidate_cap"])
    rows = ask.fetch(conn, [i for i, _ in candidates])

    media, links = {}, {}
    marks = ",".join("?" * len(rows))
    keys = list(rows)
    for r in conn.execute(f"SELECT tweet_id, kind FROM media"
                          f" WHERE tweet_id IN ({marks}) ORDER BY id", keys):
        media.setdefault(r["tweet_id"], []).append(r["kind"])
    for r in conn.execute(f"SELECT tweet_id, title FROM links"
                          f" WHERE tweet_id IN ({marks})", keys):
        if r["title"]:
            links.setdefault(r["tweet_id"], []).append(r["title"])

    scored, failures = ask.rerank(list(rows.values()), topic, media, links,
                                  workers)
    cosines = dict(candidates)
    out = []
    for tid, relevance, confidence, _v, tokens in scored:
        row = rows[tid]
        substance, note = ask.substance_for(row, local)
        out.append({
            "id": tid,
            "relevance": relevance,
            "relevance_conf": confidence,
            # The raw judgment, not the value the floor turned it into. The
            # floor cannot be varied afterwards from a neutralised number.
            "substance_raw": row["substance"],
            "substance_conf": row["substance_conf"],
            "substance_used": substance,
            "substance_note": note,
            "cosine": cosines[tid],
            "both_saved": int(bool(row["is_like"] and row["is_bookmark"])),
            "jev_tokens": tokens,
            "final": ask.final_score(
                relevance, substance, cosines[tid], 0,
                int(bool(row["is_like"] and row["is_bookmark"])), local),
        })
    out.sort(key=lambda r: -r["final"])
    return (out, time.perf_counter() - t0,
            sum(r["jev_tokens"] for r in out), len(failures))


def with_floor(rows, cfg, floor, n=10):
    """Re-rank and re-band F's candidates under a different floor.

    Makes no model calls. The floor does two separate things, and both are
    recoverable from the stored numbers:

      * in substance_for, a substance judgment below the floor is replaced by
        the corpus median, which changes the blended score and so the order;
      * in band, a result whose *relevance* confidence is below the floor
        cannot be called a strong match however relevant it is.

    Returns (top n ids, how many of those n would be shown as strong).
    """
    ask_cfg = dict(cfg["ask"])
    ask_cfg["confidence_floor"] = floor
    local = dict(cfg, ask=ask_cfg)
    neutral = ask_cfg["substance_neutral"]

    out = []
    for r in rows:
        raw, conf = r["substance_raw"], r["substance_conf"]
        if raw is None or conf is None or conf < floor:
            substance = neutral
        else:
            substance = raw
        out.append((r["id"], ask.final_score(
            r["relevance"], substance, r["cosine"], 0,
            r["both_saved"], local), r["relevance"], r["relevance_conf"]))
    out.sort(key=lambda x: -x[1])
    top = out[:n]
    strong = sum(1 for _i, _f, rel, rconf in top
                 if ask.band(rel, rconf, local) == "top")
    return [i for i, _f, _r, _c in top], strong


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--db", default=None)
    parser.add_argument("--concurrency", type=int, default=8)
    parser.add_argument("--top-n", type=int, default=10)
    args = parser.parse_args(argv)

    if not J.CANDIDATES.exists():
        print("No candidates.json. Build the Phase 2 pool first.")
        return 1
    data = json.loads(J.CANDIDATES.read_text(encoding="utf-8"))
    if "F" in data.get("versions", {}):
        print("F is already in the pool. Delete it from candidates.json "
              "to rebuild.")
        return 1

    cfg = ask.load_config()
    ask.load_env()
    jev.configure(cfg["jev"]["model"])
    jev.set_limiter(ratelimit.from_config(cfg, "typesafe"))
    conn = connect(Path(args.db) if args.db
                   else ask.resolve(cfg["paths"]["db"]))
    print("Loading the embedding model...")
    J.model(cfg["embedding"]["model"])

    snapshot = data.setdefault("snapshot", {})
    before_snapshot = len(snapshot)
    total_tokens = 0
    new_pairs = 0

    for n, topic in enumerate(J.topics(), 1):
        entry = data["topics"].get(topic)
        if not entry:
            continue
        rows, elapsed, tokens, fails = scored_rows(
            conn, cfg, topic, args.concurrency)
        total_tokens += tokens
        f_top = [r["id"] for r in rows[:args.top_n]]

        entry["F"] = f_top
        entry["F_scores"] = rows            # every component, for the floor
        entry["latency_ms"]["F"] = round(elapsed * 1000, 1)
        entry["F_rerank_failures"] = fails

        added = [t for t in f_top if t not in entry["pool"]]
        entry["pool"] = sorted(set(entry["pool"]) | set(f_top))
        new_pairs += len(added)
        for tid in added:
            if tid not in snapshot:
                row = snapshot_row(conn, tid)
                if row:
                    snapshot[tid] = row

        print(f"  {n:2}/{len(J.topics())}  {topic[:40]:40} "
              f"F {elapsed:5.1f}s  +{len(added)} new  {fails} failed")

    data["versions"]["F"] = LABEL
    data["F_built_at"] = datetime.now(timezone.utc).isoformat()
    data["F_jev_tokens"] = total_tokens
    data["F_jev_cost_usd"] = total_tokens / 1e6 * jev.USD_PER_MILLION_INPUT
    J.CANDIDATES.write_text(json.dumps(data, indent=1), encoding="utf-8")

    pools = [len(v["pool"]) for v in data["topics"].values()]
    print(f"\nPool: {sum(pools)} pairs ({min(pools)}-{max(pools)} per topic)")
    print(f"New pairs F introduced: {new_pairs}")
    print(f"Snapshot: {before_snapshot} -> {len(snapshot)} tweets")
    print(f"Jev: {total_tokens:,} input tokens, "
          f"${data['F_jev_cost_usd']:.4f}")
    print("\nNext: python evals/autojudge.py        (judges only new pairs)")
    conn.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
