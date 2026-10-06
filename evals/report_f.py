"""Compare F with A, B and C, and vary the confidence floor.

    python evals/report_f.py

Reads candidates.json, the judgments and the stored component scores. Makes
no model calls, so the floor can be varied as often as you like.
"""

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "evals"))

import ask                                                   # noqa: E402
import build_f                                               # noqa: E402
import judge as J                                            # noqa: E402
from tweet_index.db import connect                           # noqa: E402


def mean(v):
    return sum(v) / len(v) if v else 0.0


def main():
    data = json.loads(J.CANDIDATES.read_text(encoding="utf-8"))
    judgments, prov = J.merged_judgments()
    cfg = ask.load_config()
    conn = connect(ask.resolve(cfg["paths"]["db"]))
    vers = J.versions_in(data)

    # ---- per topic, every version -----------------------------------
    rows = {}
    for topic in J.topics():
        entry = data["topics"].get(topic)
        if not entry:
            continue
        expected = {t for (tp, t), rel in judgments.items()
                    if tp == topic and rel}
        if not expected:
            rows[topic] = None
            continue
        rows[topic] = {v: J.metrics(entry[v], expected) for v in vers}
        rows[topic]["expected"] = len(expected)

    print("=" * 78)
    print("RECALL@10 AND MRR, ALL FOUR VERSIONS")
    print("=" * 78)
    print(f"{'':42} {'A':>11} {'B':>11} {'C':>11} {'F':>11}")
    for metric in ("recall10", "mrr"):
        print(f"\n{metric}:")
        for topic, r in rows.items():
            if not r:
                print(f"  {topic[:40]:40}   (nothing judged relevant)")
                continue
            cells = "".join(f"{r[v][metric]:11.3f}" for v in vers)
            mark = ""
            if r["F"][metric] > r["C"][metric]:
                mark = "  F better"
            elif r["F"][metric] < r["C"][metric]:
                mark = "  F worse"
            print(f"  {topic[:40]:40}{cells}{mark}")
        vals = {v: mean([r[v][metric] for r in rows.values() if r])
                for v in vers}
        cells = "".join(f"{vals[v]:11.3f}" for v in vers)
        print(f"  {'MEAN':40}{cells}")

    # ---- where F helped, and whether images or links explain it -----
    print("\n" + "=" * 78)
    print("WHERE F BEAT C, AND WHY")
    print("=" * 78)
    better = [t for t, r in rows.items() if r and
              r["F"]["recall10"] > r["C"]["recall10"]]
    worse = [t for t, r in rows.items() if r and
             r["F"]["recall10"] < r["C"]["recall10"]]
    print(f"F better than C on Recall@10: {len(better)} topics")
    print(f"F worse:                      {len(worse)} topics")
    print(f"F tied:                       "
          f"{len([t for t, r in rows.items() if r]) - len(better) - len(worse)}")

    before = json.loads(J.PHASE2_CANDIDATES.read_text(encoding="utf-8"))
    print("\nTweets F put in a top 10 that no earlier version did, and where "
          "their\nsearchable text comes from:")
    for topic in J.topics():
        entry = data["topics"].get(topic)
        then = before["topics"].get(topic)
        if not entry or not then:
            continue
        added = [t for t in entry["F"] if t not in set(then["pool"])]
        if not added:
            continue
        print(f"\n  {topic}")
        for tid in added:
            rel = judgments.get((topic, tid))
            row = conn.execute("""
                SELECT LENGTH(TRIM(COALESCE(t.text,''))) AS tlen,
                       (SELECT COUNT(*) FROM media m WHERE m.tweet_id=t.id
                         AND TRIM(COALESCE(m.ocr_text,'')) <> '') AS imgs,
                       (SELECT COUNT(*) FROM links l WHERE l.tweet_id=t.id
                         AND TRIM(COALESCE(l.title,'')) <> '') AS lnks,
                       t.author_handle
                FROM tweets t WHERE t.id = ?""", (tid,)).fetchone()
            if not row:
                continue
            source = []
            if row["imgs"]:
                source.append(f"{row['imgs']} image(s) read")
            if row["lnks"]:
                source.append(f"{row['lnks']} link title(s)")
            if not source:
                source.append("text only")
            verdict = ("relevant" if rel else "not relevant"
                       if rel is not None else "unjudged")
            print(f"    @{row['author_handle']:20} {row['tlen']:4} chars text"
                  f"  | {', '.join(source):28} | {verdict}")

    # ---- the confidence floor ---------------------------------------
    print("\n" + "=" * 78)
    print("F WITH THE CONFIDENCE FLOOR AT 0.50 AND 0.40")
    print("=" * 78)
    print("Stored scores only; no model calls. The floor does two things:")
    print("  - a substance judgment below it is replaced by the median,")
    print("    which changes the blended order;")
    print("  - a result whose relevance confidence is below it cannot be")
    print("    shown as a strong match, however relevant it is.\n")

    for floor in (0.50, 0.40):
        recalls, mrrs, strongs, unjudged, changed = [], [], [], 0, 0
        for topic in J.topics():
            entry = data["topics"].get(topic)
            if not entry or not rows.get(topic):
                continue
            expected = {t for (tp, t), rel in judgments.items()
                        if tp == topic and rel}
            top, strong = build_f.with_floor(entry["F_scores"], cfg, floor)
            m = J.metrics(top, expected)
            recalls.append(m["recall10"])
            mrrs.append(m["mrr"])
            strongs.append(strong)
            unjudged += len([t for t in top
                             if (topic, t) not in judgments])
            if top != entry["F"]:
                changed += 1
        print(f"floor {floor:.2f}:  Recall@10 {mean(recalls) * 100:5.1f}%   "
              f"MRR {mean(mrrs):.3f}   "
              f"strong results shown {sum(strongs):3} of "
              f"{len(strongs) * 10}")
        print(f"             top-10 order changed on {changed} of "
              f"{len(recalls)} topics, "
              f"{unjudged} of {len(recalls) * 10} results unjudged")

    conn.close()
    del prov
    return 0


if __name__ == "__main__":
    sys.exit(main())
