"""Judge the three search versions blind, then score them (Section 13).

    python evals/judge.py --build     # run all three versions, save candidates
    python evals/judge.py             # judge the pool, one tweet at a time
    python evals/judge.py --score     # compute the metrics and write them up

The three versions:

    A  Phase 1 embeddings, cosine order        (--basic --embeddings embeddings_phase1)
    B  Phase 2 embeddings, cosine order        (--basic)
    C  Phase 2 embeddings + Jev rerank, blended top 10

Judging is blind and pooled. For each topic the top 10 from all three versions
are combined, de-duplicated and shuffled, and you answer y or n without being
told which version found what. Every answer is written to disk immediately, so
stopping and resuming costs nothing.

An expected set chosen by hand would bake in whatever the ranking already
favoured. Pooling avoids that: for each topic, the expected set is every tweet
you marked relevant, from any version.
"""

import argparse
import json
import random
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import ask                                                  # noqa: E402
from tweet_index import jev, ratelimit                       # noqa: E402
from tweet_index.db import connect                           # noqa: E402
from tweet_index.embed import embed_query, load_matrix, model, search  # noqa: E402

HERE = Path(__file__).resolve().parent
CANDIDATES = HERE / "candidates.json"
JUDGMENTS = HERE / "judgments.jsonl"
RESULTS_JSON = HERE / "results.json"
RESULTS_MD = HERE / "results.md"
HISTORY = HERE / "history"

VERSIONS = ("A", "B", "C")
TOP_N = 10

# The pool as it stood before version F was added, kept so the pairs F
# introduced can be told apart from the ones already judged.
PHASE2_CANDIDATES = HERE / "candidates_phase2.json"


def versions_in(data):
    """The versions this pool actually holds, A B C first, then F."""
    present = [v for v in VERSIONS if v in data.get("versions", {})]
    present += [v for v in data.get("versions", {}) if v not in present]
    return tuple(present)


def new_pairs_only(data):
    """Pairs that version F introduced: in the pool now, absent before."""
    if not PHASE2_CANDIDATES.exists():
        return None
    before = json.loads(PHASE2_CANDIDATES.read_text(encoding="utf-8"))
    pairs = []
    for topic in topics():
        now = data["topics"].get(topic)
        then = before["topics"].get(topic)
        if not now:
            continue
        old = set(then["pool"]) if then else set()
        for tid in shuffled_pool(topic, [t for t in now["pool"]
                                         if t not in old]):
            pairs.append((topic, tid))
    return pairs


def topics():
    data = yaml.safe_load((HERE / "queries.yaml").read_text(encoding="utf-8"))
    return data["topics"]


# ---------------------------------------------------------------------------
# Building the candidate pool
# ---------------------------------------------------------------------------

def cosine_top(conn, cfg, topic, table, n):
    """Versions A and B: embed the topic, take the n closest.

    The timer covers embedding the topic and the cosine scan, which is the
    work a query actually does. It excludes loading the model and the matrix,
    since those happen once per process, not once per query.
    """
    ids, matrix = load_matrix(conn, table)
    t0 = time.perf_counter()
    vector = embed_query(topic, cfg["embedding"]["model"],
                         cfg["embedding"]["query_prefix"])
    ranked = search(vector, ids, matrix, n)
    return [tid for tid, _ in ranked], time.perf_counter() - t0, len(ids)


def blended_top(conn, cfg, topic, n, workers):
    """Version C: the full pipeline, ranked by final score.

    The banded output is deliberately not used here: bands drop results below
    the relevance thresholds, and all three versions must return lists of the
    same length to be comparable.
    """
    ask_cfg = cfg["ask"]
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
    results = []
    for tid, relevance, confidence, _v, _tok in scored:
        row = rows[tid]
        substance, _note = ask.substance_for(row, cfg)
        results.append((
            tid,
            ask.final_score(relevance, substance, cosines[tid], 0,
                            int(bool(row["is_like"] and row["is_bookmark"])),
                            cfg),
        ))
    results.sort(key=lambda r: -r[1])
    elapsed = time.perf_counter() - t0
    return ([tid for tid, _ in results[:n]],
            [tid for tid, _ in candidates],
            elapsed, sum(s[4] for s in scored), len(failures))


def build(args):
    cfg = ask.load_config()
    ask.load_env()
    jev.configure(cfg["jev"]["model"])
    jev.set_limiter(ratelimit.from_config(cfg, "typesafe"))
    conn = connect(Path(args.db) if args.db
                   else ask.resolve(cfg["paths"]["db"]))

    print("Loading the embedding model...")
    model(cfg["embedding"]["model"])

    out = {"built_at": datetime.now(timezone.utc).isoformat(),
           "versions": {
               "A": "Phase 1 embeddings, cosine order",
               "B": "Phase 2 embeddings (with summaries), cosine order",
               "C": "Phase 2 embeddings + Jev rerank, blended top 10"},
           "jev_model": jev.model(),
           "embedding_model": cfg["embedding"]["model"],
           "candidate_cap": cfg["ask"]["candidate_cap"],
           "weights": cfg["ask"]["weights"],
           "topics": {}}

    total_tokens = 0
    for n, topic in enumerate(topics(), 1):
        a, ta, corpus_a = cosine_top(conn, cfg, topic, "embeddings_phase1", TOP_N)
        b, tb, corpus_b = cosine_top(conn, cfg, topic, "embeddings", TOP_N)
        c, cands, tc, tokens, fails = blended_top(conn, cfg, topic, TOP_N,
                                                  args.concurrency)
        total_tokens += tokens
        pool = sorted(set(a) | set(b) | set(c))
        out["topics"][topic] = {
            "A": a, "B": b, "C": c,
            "C_candidates": cands,
            "pool": pool,
            "latency_ms": {"A": round(ta * 1000, 1), "B": round(tb * 1000, 1),
                           "C": round(tc * 1000, 1)},
            "jev_tokens": tokens,
            "rerank_failures": fails,
        }
        print(f"  {n:2}/{len(topics())}  {topic[:44]:44} "
              f"pool {len(pool):2}  C {tc:.1f}s  {fails} failed")

    out["jev_tokens_total"] = total_tokens
    out["jev_cost_usd"] = total_tokens / 1e6 * jev.USD_PER_MILLION_INPUT
    CANDIDATES.write_text(json.dumps(out, indent=1), encoding="utf-8")
    pools = [len(v["pool"]) for v in out["topics"].values()]
    print(f"\nWrote {CANDIDATES}")
    print(f"Pool: {sum(pools)} judgments to make "
          f"({min(pools)}-{max(pools)} per topic)")
    print(f"Jev: {total_tokens:,} input tokens, "
          f"${out['jev_cost_usd']:.4f}")
    conn.close()


# ---------------------------------------------------------------------------
# Judging
# ---------------------------------------------------------------------------

def load_judgments():
    done = {}
    if JUDGMENTS.exists():
        for line in JUDGMENTS.read_text(encoding="utf-8").splitlines():
            if line.strip():
                rec = json.loads(line)
                done[(rec["topic"], rec["tweet_id"])] = rec["relevant"]
    return done


def record(topic, tweet_id, relevant):
    with JUDGMENTS.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps({
            "topic": topic, "tweet_id": tweet_id, "relevant": relevant,
            "judged_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        }) + "\n")


def shuffled_pool(topic, pool):
    """Deterministic shuffle, so a resumed session asks in the same order."""
    order = list(pool)
    random.Random(f"tweet-index::{topic}").shuffle(order)
    return order


def sample_pairs(data, size):
    """`size` topic-tweet pairs spread evenly over the topics.

    370 pairs is more than anyone will sit through by hand. A sample of 60 --
    three per topic across twenty topics -- is enough to check a model's
    judgments against, and the spread matters more than the count: a sample
    drawn at random from the whole pool would miss some topics entirely and
    over-weight the ones with the biggest pools.

    Deterministic, so stopping and resuming asks the same questions.
    """
    per_topic = {}
    for topic in topics():
        entry = data["topics"].get(topic)
        if entry:
            per_topic[topic] = shuffled_pool(topic, entry["pool"])
    chosen, i = [], 0
    names = list(per_topic)
    while len(chosen) < size and any(per_topic[t] for t in names):
        topic = names[i % len(names)]
        if per_topic[topic]:
            chosen.append((topic, per_topic[topic].pop(0)))
        i += 1
    return chosen[:size]


def wrap(text, width=74, indent="   "):
    import textwrap
    return textwrap.fill(" ".join((text or "").split()), width,
                         initial_indent=indent, subsequent_indent=indent)


def judge(args):
    if not CANDIDATES.exists():
        print("No candidates yet. Run: python evals/judge.py --build")
        return 1
    data = json.loads(CANDIDATES.read_text(encoding="utf-8"))
    cfg = ask.load_config()
    conn = connect(Path(args.db) if args.db
                   else ask.resolve(cfg["paths"]["db"]))
    done = load_judgments()
    snapshot = data.get("snapshot", {})
    if snapshot:
        print(f"Showing the {len(snapshot)}-tweet snapshot taken when the pool "
              f"was built,\nso later enrichment cannot change what is judged.")

    # With --sample, judge a spread of that many pairs rather than the whole
    # pool; the rest are judged by a model and validated against these.
    if args.new_only:
        pairs = new_pairs_only(data)
        if pairs is None:
            print("No earlier pool to compare against.")
            return 1
        if args.sample:
            pairs = pairs[:args.sample]
        print(f"Judging {len(pairs)} pairs that version F introduced "
              f"(tweets no earlier version put in its top 10).")
    elif args.sample:
        pairs = sample_pairs(data, args.sample)
        print(f"Judging a sample of {len(pairs)} pairs, "
              f"{args.sample // len(topics())} per topic.")
    else:
        pairs = [(topic, tid) for topic in topics()
                 if data["topics"].get(topic)
                 for tid in shuffled_pool(topic, data["topics"][topic]["pool"])]
    work = [(topic, tid) for topic, tid in pairs if (topic, tid) not in done]

    if not work:
        print(f"All judged: {len(done)} answers in {JUDGMENTS.name}.")
        print("Run: python evals/judge.py --score")
        return 0

    total = sum(len(v["pool"]) for v in data["topics"].values())
    print(f"\n{len(work)} left to judge, {len(done)} done, {total} in total.")
    print("y = relevant to the topic, n = not, s = skip, q = stop and save.\n")

    current = None
    for i, (topic, tid) in enumerate(work, 1):
        if topic != current:
            current = topic
            print("=" * 78)
            print(f"TOPIC: {topic}")
            print("=" * 78)
        # Prefer the snapshot taken when the pool was built. The rankings were
        # frozen then; the text and summaries have to be frozen too, or a
        # later re-enrichment would change what is being judged while the
        # rankings being scored stayed as they were.
        row = snapshot.get(tid)
        if row is None:
            row = conn.execute("""
                SELECT t.author_handle, t.text, t.url, t.is_like,
                       t.is_bookmark, e.context
                FROM tweets t LEFT JOIN enrichment e ON e.tweet_id = t.id
                WHERE t.id = ?""", (tid,)).fetchone()
        if row is None:
            continue
        feed = ("both" if row["is_like"] and row["is_bookmark"]
                else "like" if row["is_like"] else "bookmark")
        print(f"\n[{i}/{len(work)}]  @{row['author_handle']}  ({feed})")
        text = (row["text"] or "").strip()
        print(wrap(text) if text else "   (no text)")
        summary = (row["context"] or "").strip()
        if summary:
            print(wrap("> " + summary))
        print(f"   {row['url']}")
        while True:
            answer = input("   relevant? [y/n/s/q] ").strip().lower()
            if answer in ("y", "n", "s", "q"):
                break
        if answer == "q":
            print(f"\nStopped. {len(done)} judged so far; rerun to continue.")
            conn.close()
            return 0
        if answer == "s":
            continue
        record(topic, tid, answer == "y")
        done[(topic, tid)] = answer == "y"

    print(f"\nDone. {len(done)} judgments saved.")
    print("Run: python evals/judge.py --score")
    conn.close()
    return 0


# ---------------------------------------------------------------------------
# Scoring
# ---------------------------------------------------------------------------

def metrics(ranked, expected):
    """Section 13.2, over one topic."""
    top5, top10 = ranked[:5], ranked[:TOP_N]
    first = next((i + 1 for i, tid in enumerate(top10) if tid in expected), None)
    return {
        "hit5": 1.0 if any(t in expected for t in top5) else 0.0,
        "recall10": (len(expected & set(top10)) / len(expected)
                     if expected else 0.0),
        "mrr": 1.0 / first if first else 0.0,
    }


def merged_judgments():
    """The person's own answers, with a model's standing in for the rest.

    371 pairs is too many to judge by hand, so a person judges a spread of 60
    and gpt-6.1-sol judges all 371. Where both answered, the person wins: the
    model's value is in covering the pairs nobody will ever look at, not in
    overruling the one who owns the library.

    Returns (judgments, provenance).
    """
    human = load_judgments()
    model = {}
    model_file = HERE / "judgments_model.jsonl"
    if model_file.exists():
        for line in model_file.read_text(encoding="utf-8").splitlines():
            if line.strip():
                rec = json.loads(line)
                model[(rec["topic"], rec["tweet_id"])] = rec["relevant"]
    merged = dict(model)
    merged.update(human)              # the person's answers overwrite
    provenance = {
        "human": len(human),
        "model": len(merged) - len(human),
        "model_name": "gpt-6.1-sol",
        "overlap": len([k for k in human if k in model]),
    }
    if provenance["overlap"]:
        agree = len([k for k in human if k in model and human[k] == model[k]])
        provenance["agreement"] = agree / provenance["overlap"]
    return merged, provenance


def score(args):
    if not CANDIDATES.exists():
        print("No candidates. Run --build first.")
        return 1
    data = json.loads(CANDIDATES.read_text(encoding="utf-8"))
    done, provenance = merged_judgments()
    if not done:
        print("No judgments yet. Run: python evals/judge.py --sample 60")
        return 1
    print(f"Judgments: {provenance['human']} from you, "
          f"{provenance['model']} from {provenance['model_name']}")
    if "agreement" in provenance:
        print(f"Agreement on the {provenance['overlap']} you both judged: "
              f"{provenance['agreement'] * 100:.1f}%")

    vers = versions_in(data)
    per_topic, summary = {}, {v: [] for v in vers}
    cand_recall, judged_counts = [], []
    for topic in topics():
        entry = data["topics"].get(topic)
        if not entry:
            continue
        expected = {tid for (tp, tid), rel in done.items()
                    if tp == topic and rel}
        asked = {tid for (tp, tid) in done if tp == topic}
        if not asked:
            continue                       # not judged yet, leave it out
        judged_counts.append(len(asked))
        row = {"expected": len(expected), "judged": len(asked)}
        for v in vers:
            m = metrics(entry[v], expected)
            row[v] = m
            summary[v].append(m)
        if expected:
            cand_recall.append(
                len(expected & set(entry["C_candidates"])) / len(expected))
        per_topic[topic] = row

    def mean(values):
        return sum(values) / len(values) if values else 0.0

    overall = {v: {k: mean([m[k] for m in summary[v]])
                   for k in ("hit5", "recall10", "mrr")} for v in vers}
    latency = {v: sorted(e["latency_ms"][v] for e in data["topics"].values())
               for v in vers}

    def pct(x):
        return f"{x * 100:.1f}"

    out = {
        "scored_at": datetime.now(timezone.utc).isoformat(),
        "topics_judged": len(per_topic),
        "topics_total": len(topics()),
        "judgments": len(done),
        "judgment_provenance": provenance,
        "versions": data["versions"],
        "overall": overall,
        "candidate_recall_C": mean(cand_recall),
        "latency_p50_ms": {v: latency[v][len(latency[v]) // 2]
                           for v in vers},
        "latency_p90_ms": {v: latency[v][int(len(latency[v]) * 0.9)]
                           for v in vers},
        "jev_tokens_total": data.get("jev_tokens_total"),
        "jev_cost_usd": data.get("jev_cost_usd"),
        "per_topic": per_topic,
    }
    RESULTS_JSON.write_text(json.dumps(out, indent=1), encoding="utf-8")

    HISTORY.mkdir(exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    csv = HISTORY / f"{stamp}.csv"
    with csv.open("w", encoding="utf-8") as fh:
        fh.write("version,hit5,recall10,mrr,latency_p50_ms\n")
        for v in vers:
            o = overall[v]
            fh.write(f"{v},{o['hit5']:.4f},{o['recall10']:.4f},"
                     f"{o['mrr']:.4f},{out['latency_p50_ms'][v]}\n")

    lines = []
    add = lines.append
    add(f"# Evaluation results\n")
    add(f"{out['topics_judged']} of {out['topics_total']} topics judged, "
        f"{out['judgments']} blind judgments.\n")

    # Say plainly where the judgments came from. Anyone comparing these
    # numbers with a later run needs to know most of them were model-made,
    # and how far the model was trusted.
    pv = provenance
    if pv["model"]:
        add(f"**How these were judged.** {pv['human']} of the "
            f"{out['judgments']} judgments are the owner's own; the remaining "
            f"{pv['model']} were made by `{pv['model_name']}`, which wrote "
            f"neither the summaries being shown (gpt-6-luna) nor the rankings "
            f"being tested (Jev), and saw exactly what the owner saw.")
        if "agreement" in pv:
            add(f"On the {pv['overlap']} pairs both judged they agreed "
                f"**{pv['agreement'] * 100:.1f}%** of the time. The owner's "
                f"answer is used wherever the two differ.\n")
        else:
            add("")

    add("| Version | What it is | Hit@5 | Recall@10 | MRR | p50 |")
    add("|---|---|---:|---:|---:|---:|")
    for v in vers:
        o = overall[v]
        add(f"| {v} | {data['versions'][v]} | {pct(o['hit5'])}% | "
            f"{pct(o['recall10'])}% | {o['mrr']:.3f} | "
            f"{out['latency_p50_ms'][v]} ms |")
    add("")
    gate_hit = (overall["C"]["hit5"] - overall["A"]["hit5"]) * 100
    gate_rec = (overall["C"]["recall10"] - overall["A"]["recall10"]) * 100
    passed = gate_hit >= 10 or gate_rec >= 10
    add(f"**Section 17 gate (C over A by >= 10 points): "
        f"{'PASS' if passed else 'FAIL'}** — "
        f"Hit@5 {gate_hit:+.1f}, Recall@10 {gate_rec:+.1f}\n")
    add(f"- Summaries (B - A): Hit@5 "
        f"{(overall['B']['hit5'] - overall['A']['hit5']) * 100:+.1f}, "
        f"Recall@10 "
        f"{(overall['B']['recall10'] - overall['A']['recall10']) * 100:+.1f}, "
        f"MRR {overall['B']['mrr'] - overall['A']['mrr']:+.3f}")
    add(f"- Jev rerank (C - B): Hit@5 "
        f"{(overall['C']['hit5'] - overall['B']['hit5']) * 100:+.1f}, "
        f"Recall@10 "
        f"{(overall['C']['recall10'] - overall['B']['recall10']) * 100:+.1f}, "
        f"MRR {overall['C']['mrr'] - overall['B']['mrr']:+.3f}")
    add(f"- Candidate recall into stage 2: {pct(out['candidate_recall_C'])}%")
    add("")
    dead = [t for t, r in per_topic.items()
            if all(r[v]["hit5"] == 0 for v in vers)]
    add(f"Topics no version answered ({len(dead)}): "
        + (", ".join(f"`{t}`" for t in dead) if dead else "none"))
    add("")
    add("## Per topic\n")
    add("| Topic | relevant found | A hit@5 | B hit@5 | C hit@5 "
        "| A rec@10 | B rec@10 | C rec@10 |")
    add("|---|---:|---:|---:|---:|---:|---:|---:|")
    for topic, r in per_topic.items():
        add(f"| {topic} | {r['expected']}/{r['judged']} | "
            + " | ".join(f"{r[v]['hit5']:.0f}" for v in vers) + " | "
            + " | ".join(pct(r[v]["recall10"]) for v in vers) + " |")
    RESULTS_MD.write_text("\n".join(lines) + "\n", encoding="utf-8")

    print("\n".join(lines))
    print(f"\nWrote {RESULTS_JSON.name}, {RESULTS_MD.name}, history/{csv.name}")
    return 0


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--build", action="store_true",
                        help="run all three versions and save the pool")
    parser.add_argument("--score", action="store_true",
                        help="compute the metrics from the judgments so far")
    parser.add_argument("--new-only", action="store_true",
                        help="judge only the pairs version F introduced, "
                             "for spot-checking a new pool")
    parser.add_argument("--sample", type=int, default=None, metavar="N",
                        help="judge a spread of N pairs rather than the whole "
                             "pool; the rest can be judged by a model and "
                             "validated against these (see autojudge.py)")
    parser.add_argument("--concurrency", type=int, default=24)
    parser.add_argument("--db", default=None)
    args = parser.parse_args(argv)
    if args.build:
        return build(args) or 0
    if args.score:
        return score(args)
    return judge(args)


if __name__ == "__main__":
    sys.exit(main())
