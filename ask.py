"""Search your saved tweets by topic.

    python ask.py "jev decision models"              # the full three stages
    python ask.py "core workout" --basic             # cosine only
    python ask.py "core workout" --basic --embeddings embeddings_phase1

Three stages, per Section 11:

  1. Candidate generation. Embed the topic, rank every tweet by cosine, keep
     the best 120. Free and about 40 milliseconds. Concept-tag matching joins
     this stage in Phase 3.
  2. Relevance rerank. One Jev question per candidate: how relevant is this
     tweet to the topic, on five named levels. About 120 calls.
  3. Blend and band. Relevance leads, with substance and cosine as support.
     Results above the thresholds are shown as strong matches, the rest as
     maybes, each with its stored enrichment summary underneath.

No model is called at query time except Jev: Section 11.3 ends with an LLM
writing a line per result, which this build does not do: each result
shows its stored summary instead, which costs nothing per query.
The summary written during enrichment says the same kind of thing, costs
nothing, and is already on disk.

Every query is logged to the `queries` table with the configuration that
produced it, so the three versions -- Phase 1 embeddings, Phase 2 embeddings,
and the full rerank -- stay comparable.
"""

import argparse
import json
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path

import yaml

from tweet_index import jev, ratelimit
from tweet_index.db import connect
from tweet_index.embed import embed_query, load_matrix, model, search

ROOT = Path(__file__).resolve().parent

SNIPPET = 150


def load_config(path=None):
    return yaml.safe_load(Path(path or ROOT / "config.yaml")
                          .read_text(encoding="utf-8"))


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


def one_line(text, limit=SNIPPET):
    if not text:
        return ""
    flat = " ".join(text.split())
    return flat if len(flat) <= limit else flat[:limit - 1].rstrip() + "…"


def saved_as(row):
    if row["is_like"] and row["is_bookmark"]:
        return "both"
    return "like" if row["is_like"] else "bmk"


CANDIDATE_COLUMNS = """
    t.id, t.author_handle, t.posted_at, t.is_like, t.is_bookmark, t.text,
    t.url, t.quoted_text, t.card_title, t.card_description, t.article_title,
    t.article_preview, t.translation_en, t.community_note,
    e.context, j.substance, j.substance_conf
"""


def fetch(conn, ids):
    if not ids:
        return {}
    marks = ",".join("?" * len(ids))
    rows = conn.execute(f"""
        SELECT {CANDIDATE_COLUMNS}
        FROM tweets t
        LEFT JOIN enrichment e ON e.tweet_id = t.id
        LEFT JOIN tweet_judgments j ON j.tweet_id = t.id
        WHERE t.id IN ({marks})""", ids).fetchall()
    return {r["id"]: r for r in rows}


# ---------------------------------------------------------------------------
# Stage 3: blend and band
# ---------------------------------------------------------------------------

def substance_for(row, cfg):
    """The substance to use, and whether it was trusted.

    A low-confidence judgment is not evidence of low substance; it is the
    model saying it cannot tell. Scoring it as 0 would push down exactly the
    short, useful tweets that are hardest to judge -- a 159-character formula
    for writing introductions scored 2.02 at confidence 0.18. So below the
    floor it falls back to the corpus median.
    """
    ask = cfg["ask"]
    score = row["substance"]
    conf = row["substance_conf"]
    if score is None or conf is None:
        return ask["substance_neutral"], "unjudged"
    if conf < ask["confidence_floor"]:
        return ask["substance_neutral"], "low confidence"
    return score, "judged"


def final_score(relevance, substance, cosine, tag_hit, both_saved, cfg):
    """Section 11.3, with cosine added and substance normalised honestly."""
    ask = cfg["ask"]
    w = ask["weights"]
    return (w["relevance"] * relevance / ask["relevance_max"]
            + w["substance"] * substance / ask["substance_max"]
            + w["cosine"] * cosine
            + w["tag_hit"] * tag_hit
            + w["both_saved"] * both_saved)


def band(relevance, confidence, cfg):
    """Section 11.3 exactly."""
    ask = cfg["ask"]
    if (confidence >= ask["confidence_floor"]
            and relevance >= ask["top_min_relevance_level"]):
        return "top"
    if relevance >= ask["maybe_min_relevance_level"]:
        return "maybe"
    return "drop"


# ---------------------------------------------------------------------------
# Stage 2: the Jev rerank
# ---------------------------------------------------------------------------

def rerank(rows, topic, media, links, workers):
    """One Jev Score per candidate. Failures are dropped, not fatal."""
    question = {"relevance": jev.relevance_question(topic)}

    def score_one(row):
        try:
            state = jev.tweet_state(row, link_titles=links.get(row["id"], []),
                                    media=media.get(row["id"], []))
            answers, version, tokens = jev.judge_tweet(state, question)
            answer = answers["relevance"]
            return (row["id"], answer.score, answer.confidence, version,
                    tokens), None
        except jev.JevError as exc:
            return None, (row["id"], str(exc))

    scored, failures = [], []
    with ThreadPoolExecutor(max_workers=workers) as pool:
        for result, error in pool.map(score_one, rows):
            if error:
                failures.append(error)
            else:
                scored.append(result)
    return scored, failures


# ---------------------------------------------------------------------------
# Output
# ---------------------------------------------------------------------------

def show(results, header, start=1):
    """Print a band of results, numbered from `start`.

    Each result carries its stored enrichment summary instead of a line
    written at query time. The summary was computed once during enrichment, so
    showing it costs nothing and adds no latency; §11.3's explanation call is
    deliberately not made: it was the slowest stage of the pipeline and the
    stored summary says much the same thing.

    The numbering runs continuously across the bands rather than restarting,
    so that "result 7" means one thing. ti.py's /open takes that number.
    """
    if not results:
        return
    print(header)
    for n, r in enumerate(results, start):
        row = r["row"]
        date = (row["posted_at"] or "")[:10]
        print(f"{n:2}. {r['final']:.3f}  @{row['author_handle']}  {date}  "
              f"{saved_as(row)}")
        print(f"    relevance {r['relevance']:.2f} (conf {r['confidence']:.2f})"
              f"   substance {r['substance']:.2f} [{r['substance_note']}]"
              f"   cosine {r['cosine']:.3f}")
        print(f"    {one_line(row['text']) or '(no text)'}")
        summary = (row["context"] or "").strip()
        if summary:
            print(f"    > {one_line(summary, 220)}")
        print(f"    {row['url']}")
        print()


def log_query(conn, topic, snapshot, results, latency_ms, jev_tokens):
    payload = [{"id": r["row"]["id"], "cosine": round(r["cosine"], 6),
                "rel": round(r["relevance"], 4),
                "rel_conf": round(r["confidence"], 4),
                "sub": round(r["substance"], 4),
                "final": round(r["final"], 6), "band": r["band"]}
               for r in results]
    with conn:
        conn.execute(
            "INSERT INTO queries (asked_at, topic, config, results,"
            " latency_ms, jev_tokens) VALUES (?,?,?,?,?,?)",
            (datetime.now(timezone.utc).isoformat(), topic,
             json.dumps(snapshot), json.dumps(payload), latency_ms,
             jev_tokens))


# ---------------------------------------------------------------------------

def side_tables(conn, ordered):
    """The media kinds and link titles the Jev judge is shown."""
    media, links = {}, {}
    if not ordered:
        return media, links
    marks = ",".join("?" * len(ordered))
    for r in conn.execute(f"SELECT tweet_id, kind FROM media"
                          f" WHERE tweet_id IN ({marks}) ORDER BY id", ordered):
        media.setdefault(r["tweet_id"], []).append(r["kind"])
    for r in conn.execute(f"SELECT tweet_id, title FROM links"
                          f" WHERE tweet_id IN ({marks})", ordered):
        if r["title"]:
            links.setdefault(r["tweet_id"], []).append(r["title"])
    return media, links


def rank_basic(conn, cfg, topic, ids, matrix, k=None):
    """Stage 1 only: cosine order. Free and instant.

    Takes the vectors as arguments rather than loading them, so a long-lived
    process (ti.py) can hold them in memory and a one-shot command can load
    them per run. Both paths then share this one copy of the logic.
    """
    model_name = cfg["embedding"]["model"]
    timings = {}
    t = time.perf_counter()
    vector = embed_query(topic, model_name, cfg["embedding"]["query_prefix"])
    candidates = search(vector, ids, matrix, k or cfg["ask"]["top_k"])
    timings["stage 1 cosine"] = time.perf_counter() - t
    rows = fetch(conn, [i for i, _ in candidates])
    return {"candidates": candidates, "rows": rows, "timings": timings,
            "corpus": len(ids)}


def rank_smart(conn, cfg, topic, ids, matrix, concurrency=24, top_n=None):
    """All three stages: cosine, Jev relevance rerank, blend and band.

    Returns everything both callers need to print or log. Section 11.3's
    ordering, thresholds and weights all live here, so ti.py and the
    command line cannot drift apart.
    """
    ask_cfg = cfg["ask"]
    model_name = cfg["embedding"]["model"]
    timings = {}

    t = time.perf_counter()
    vector = embed_query(topic, model_name, cfg["embedding"]["query_prefix"])
    candidates = search(vector, ids, matrix, ask_cfg["candidate_cap"])
    timings["stage 1 cosine"] = time.perf_counter() - t

    rows = fetch(conn, [i for i, _ in candidates])
    media, links = side_tables(conn, list(rows))

    t = time.perf_counter()
    scored, failures = rerank(list(rows.values()), topic, media, links,
                              concurrency)
    timings["stage 2 jev"] = time.perf_counter() - t
    jev_tokens = sum(s[4] for s in scored)

    t = time.perf_counter()
    cosines = dict(candidates)
    results = []
    for tid, relevance, confidence, _version, _tokens in scored:
        row = rows[tid]
        substance, note = substance_for(row, cfg)
        results.append({
            "row": row,
            "cosine": cosines[tid],
            "relevance": relevance,
            "confidence": confidence,
            "substance": substance,
            "substance_note": note,
            "band": band(relevance, confidence, cfg),
            "final": final_score(relevance, substance, cosines[tid], 0,
                                 int(bool(row["is_like"] and row["is_bookmark"])),
                                 cfg),
        })
    results.sort(key=lambda r: -r["final"])
    tops = [r for r in results if r["band"] == "top"][
        :top_n or ask_cfg["banded_top_k"]]
    maybes = [r for r in results if r["band"] == "maybe"][
        :ask_cfg["max_maybes"]]
    timings["stage 3 blend"] = time.perf_counter() - t

    return {"candidates": candidates, "rows": rows, "results": results,
            "tops": tops, "maybes": maybes, "scored": scored,
            "failures": failures, "jev_tokens": jev_tokens,
            "timings": timings, "corpus": len(ids)}


def connect_jev(cfg):
    """Configure Jev and its limiter. Safe to call more than once."""
    load_env()
    jev.configure(cfg["jev"]["model"])
    jev.set_limiter(ratelimit.from_config(cfg, "typesafe"))


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("topic")
    parser.add_argument("--basic", action="store_true",
                        help="cosine only: no Jev rerank")
    parser.add_argument("--embeddings", default="embeddings",
                        help="which vector table to search; embeddings_phase1"
                             " is the pre-enrichment snapshot")
    parser.add_argument("--k", type=int, default=None)
    parser.add_argument("--concurrency", type=int, default=24)
    parser.add_argument("--db", default=None)
    parser.add_argument("--config", default=None)
    parser.add_argument("--timing", action="store_true")
    args = parser.parse_args(argv)

    cfg = load_config(args.config)
    ask = cfg["ask"]
    model_name = cfg["embedding"]["model"]
    conn = connect(Path(args.db) if args.db else resolve(cfg["paths"]["db"]))

    timings = {}
    t0 = time.perf_counter()
    model(model_name)
    timings["load model"] = time.perf_counter() - t0

    t = time.perf_counter()
    ids, matrix = load_matrix(conn, args.embeddings)
    timings["load vectors"] = time.perf_counter() - t

    label = ("" if args.embeddings == "embeddings"
             else f" [{args.embeddings}]")

    # ---- basic: cosine only -------------------------------------------
    if args.basic:
        out = rank_basic(conn, cfg, args.topic, ids, matrix, args.k)
        candidates, rows = out["candidates"], out["rows"]
        timings.update(out["timings"])
        print(f'\n{len(ids)} tweets searched for "{args.topic}"{label}\n')
        for n, (tid, score) in enumerate(candidates, 1):
            row = rows.get(tid)
            if not row:
                continue
            print(f"{n:2}. {score:.3f}  @{row['author_handle']}  "
                  f"{(row['posted_at'] or '')[:10]}  {saved_as(row)}")
            print(f"    {one_line(row['text']) or '(no text)'}")
            print(f"    {row['url']}")
            print()
        snapshot = {"phase": 1, "ranking": "embedding cosine only",
                    "embedding_model": model_name,
                    "embeddings_table": args.embeddings,
                    "top_k": len(candidates)}
        results = [{"row": rows[t], "cosine": s, "relevance": 0.0,
                    "confidence": 0.0, "substance": 0.0, "final": s,
                    "band": "top", "substance_note": "n/a"}
                   for t, s in candidates if t in rows]
        log_query(conn, args.topic, snapshot, results,
                  round(timings["stage 1 cosine"] * 1000), None)
        if args.timing:
            for k, v in timings.items():
                print(f"{k:16}{v * 1000:7.0f} ms")
        conn.close()
        return 0

    # ---- stages 2 and 3, shared with ti.py ----------------------------
    connect_jev(cfg)
    out = rank_smart(conn, cfg, args.topic, ids, matrix, args.concurrency,
                     top_n=args.k)
    candidates, rows = out["candidates"], out["rows"]
    results, tops, maybes = out["results"], out["tops"], out["maybes"]
    scored, failures = out["scored"], out["failures"]
    jev_tokens = out["jev_tokens"]
    timings.update(out["timings"])

    # Section 11.3 prints the ranked list as soon as banding finishes. There
    # is nothing to follow it now: the per-result line comes from the stored
    # summary, so the list is the whole answer.
    print(f'\n{len(ids)} tweets searched for "{args.topic}"{label}')
    print(f'{len(candidates)} candidates reranked, '
          f'{len(scored)} scored, {len(failures)} failed\n')
    show(tops, "STRONG MATCHES")
    show(maybes, "MAYBE", start=len(tops) + 1)
    if not tops and not maybes:
        print("Nothing cleared the relevance thresholds.")
    timings["to ranked list"] = sum(timings.values())

    snapshot = {
        "phase": 2, "ranking": "cosine -> jev relevance -> blend",
        "embedding_model": model_name, "embeddings_table": args.embeddings,
        "jev_model": jev.model(), "explanations": "none (stored summary shown)",
        "candidate_cap": ask["candidate_cap"], "weights": ask["weights"],
        "top_min_relevance_level": ask["top_min_relevance_level"],
        "maybe_min_relevance_level": ask["maybe_min_relevance_level"],
        "confidence_floor": ask["confidence_floor"],
        "substance_max": ask["substance_max"],
        "substance_neutral": ask["substance_neutral"],
        "shown": {"top": len(tops), "maybe": len(maybes)},
        "rerank_failures": len(failures),
    }
    total_ms = round(timings["to ranked list"] * 1000)
    log_query(conn, args.topic, snapshot, results, total_ms, jev_tokens)

    jev_cost = jev_tokens / 1e6 * jev.USD_PER_MILLION_INPUT
    print(f"Cost: Jev {jev_tokens:,} input tokens ${jev_cost:.5f}"
          f"   (no LLM calls at query time)")
    if args.timing or True:
        order = ["load model", "load vectors", "stage 1 cosine",
                 "stage 2 jev", "stage 3 blend", "to ranked list"]
        print("Time:", "  ".join(f"{k} {timings[k] * 1000:.0f}ms"
                                 for k in order if k in timings),
              f"  total {total_ms}ms")
    conn.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
