"""Have a model judge the evaluation pool, validated against a human sample.

    python evals/autojudge.py --estimate      # cost, no calls
    python evals/autojudge.py                 # judge the whole pool
    python evals/autojudge.py --agreement     # compare with the human sample

371 topic-tweet pairs is more than anyone will sit through. So a person judges
a spread of 60 (`judge.py --sample 60`), a model judges all 371, the two are
compared on those 60, and the model's judgments are used for the rest only if
it agrees closely enough.

Which model, and why not the others:

  * not gpt-6-luna -- it wrote the summaries being shown, so it would be
    marking its own work;
  * not Jev -- its relevance scores are what the C ranking is built from, so
    it would be grading itself;
  * gpt-6.1-sol, a larger and more expensive model than either, which has
    seen neither the summaries nor the rankings.

The model is shown exactly what the person is shown: the same snapshot of
author, tweet text and stored summary, with no indication of which search
version found it or where it ranked.
"""

import argparse
import json
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "evals"))

import judge as judge_mod                                   # noqa: E402
from tweet_index import ratelimit                             # noqa: E402
from tweet_index.llm import LLMError, complete_json, set_limiter  # noqa: E402

HERE = Path(__file__).resolve().parent
MODEL_JUDGMENTS = HERE / "judgments_model.jsonl"

MODEL = "gpt-6.1-sol"
PRICE_IN = 2.00
PRICE_OUT = 10.00

# The instruction is the owner's own words, kept verbatim: this is their
# library and their definition of useful, not a general notion of relevance.
SYSTEM_PROMPT = """You are judging search results for one person's private archive of tweets they saved themselves.

For the topic and the tweet below, answer one question: would the owner genuinely want this tweet when searching for this topic?

Answer "y" if they would genuinely want it. Answer "n" otherwise.

Say yes when the tweet is about the topic and carries something worth reading: an explanation, a resource, a specific claim, a concrete example, a pointer to something substantial.

Say no when the tweet is only loosely associated with the topic, shares a word with it but is about something else, is a bare reaction with no content, or is about the topic but says nothing a searcher could use.

Judge the tweet in front of you, not what it links to unless the summary says what is there.

Return only JSON: {"answer": "y" or "n"}"""

SCHEMA = {
    "type": "object",
    "properties": {"answer": {"type": "string", "enum": ["y", "n"]}},
    "required": ["answer"],
    "additionalProperties": False,
}


def shown_text(topic, row):
    """Exactly what judge.py puts on screen for a person."""
    feed = ("both" if row["is_like"] and row["is_bookmark"]
            else "like" if row["is_like"] else "bookmark")
    parts = [f"topic: {topic}", "",
             f"@{row['author_handle']} ({feed})",
             (row["text"] or "").strip() or "(no text)"]
    summary = (row["context"] or "").strip()
    if summary:
        parts += ["", f"summary: {summary}"]
    return "\n".join(parts)


def load_model_judgments():
    done = {}
    if MODEL_JUDGMENTS.exists():
        for line in MODEL_JUDGMENTS.read_text(encoding="utf-8").splitlines():
            if line.strip():
                rec = json.loads(line)
                done[(rec["topic"], rec["tweet_id"])] = rec["relevant"]
    return done


def record(topic, tweet_id, relevant, model):
    with MODEL_JUDGMENTS.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps({
            "topic": topic, "tweet_id": tweet_id, "relevant": relevant,
            "source": "model", "model": model,
            "judged_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        }) + "\n")


def all_pairs(data):
    return [(topic, tid) for topic in judge_mod.topics()
            if data["topics"].get(topic)
            for tid in sorted(data["topics"][topic]["pool"])]


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--estimate", action="store_true")
    parser.add_argument("--agreement", action="store_true")
    parser.add_argument("--model", default=MODEL)
    parser.add_argument("--concurrency", type=int, default=8)
    args = parser.parse_args(argv)

    if not judge_mod.CANDIDATES.exists():
        print("No candidates. Run: python evals/judge.py --build")
        return 1
    data = json.loads(judge_mod.CANDIDATES.read_text(encoding="utf-8"))
    snapshot = data.get("snapshot", {})
    if not snapshot:
        print("The candidate pool has no snapshot; rebuild it.")
        return 1

    pairs = all_pairs(data)
    human = judge_mod.load_judgments()

    # ---- agreement -------------------------------------------------
    if args.agreement:
        model = load_model_judgments()
        overlap = [(t, i) for (t, i) in human if (t, i) in model]
        if not overlap:
            print("No overlap yet: judge a sample, then run the model.")
            return 1
        agree = [k for k in overlap if human[k] == model[k]]
        rate = len(agree) / len(overlap)
        print(f"Compared on {len(overlap)} pairs the person judged.")
        print(f"Agreement: {len(agree)}/{len(overlap)} = {rate * 100:.1f}%")
        hy = sum(1 for k in overlap if human[k])
        my = sum(1 for k in overlap if model[k])
        print(f"  person said yes to {hy}, model said yes to {my}")
        fp = [k for k in overlap if model[k] and not human[k]]
        fn = [k for k in overlap if human[k] and not model[k]]
        print(f"  model too generous: {len(fp)}   too strict: {len(fn)}")
        print()
        if rate >= 0.85:
            print("At or above 85%: the model's judgments can stand in for "
                  "the rest of the pool.")
        else:
            print("Below 85%. Where they differed:")
        if rate < 0.85 or fp or fn:
            conn = None
            for kind, keys in (("model yes, person no", fp),
                               ("person yes, model no", fn)):
                for topic, tid in keys[:12]:
                    row = snapshot.get(tid, {})
                    text = " ".join((row.get("text") or "(no text)").split())
                    print(f"  [{kind}] {topic}")
                    print(f"      @{row.get('author_handle')}: {text[:110]}")
            del conn
        return 0

    # ---- estimate --------------------------------------------------
    todo = [(t, i) for (t, i) in pairs
            if (t, i) not in load_model_judgments()]
    prompts = [shown_text(t, snapshot[i]) for t, i in todo if i in snapshot]
    chars = sum(len(p) for p in prompts) + len(SYSTEM_PROMPT) * len(prompts)
    in_tokens = chars / 3.75
    out_tokens = len(prompts) * 12
    cost = in_tokens / 1e6 * PRICE_IN + out_tokens / 1e6 * PRICE_OUT

    print(f"Pool: {len(pairs)} pairs")
    print(f"Already judged by the model: {len(pairs) - len(todo)}")
    print(f"To judge: {len(todo)}")
    print(f"Model: {args.model}  (${PRICE_IN}/M in, ${PRICE_OUT}/M out)")
    print(f"\nEstimate: {in_tokens / 1e3:.0f}k input + "
          f"{out_tokens / 1e3:.1f}k output tokens")
    print(f"Cost: ${cost:.3f}")
    if args.estimate:
        return 0
    if not todo:
        print("\nNothing to do.")
        return 0

    cfg = judge_mod.ask.load_config()
    judge_mod.ask.load_env()
    set_limiter(ratelimit.from_config(cfg, "openai"))
    print()

    def work(pair):
        topic, tid = pair
        row = snapshot.get(tid)
        if not row:
            return pair, None, "not in the snapshot"
        try:
            out, usage = complete_json(
                SYSTEM_PROMPT, shown_text(topic, row), args.model,
                schema=SCHEMA, max_output_tokens=600)
            return pair, (out["answer"] == "y", usage), None
        except LLMError as exc:
            return pair, None, str(exc)

    t0 = time.perf_counter()
    done = failed = 0
    tin = tout = 0
    with ThreadPoolExecutor(max_workers=args.concurrency) as pool:
        for pair, result, error in pool.map(work, todo):
            if error:
                failed += 1
                print(f"  failed {pair[1]}: {error}")
                continue
            relevant, usage = result
            record(pair[0], pair[1], relevant, args.model)
            done += 1
            tin += usage["input_tokens"]
            tout += usage["output_tokens"]
            print(f"\r  {done + failed}/{len(todo)} judged   ",
                  end="", flush=True)
    seconds = time.perf_counter() - t0
    print()
    spend = tin / 1e6 * PRICE_IN + tout / 1e6 * PRICE_OUT
    print(f"\nJudged {done}, failed {failed}, in {seconds:.1f}s")
    print(f"Tokens: {tin:,} in, {tout:,} out")
    print(f"Cost: ${spend:.4f}")
    model = load_model_judgments()
    yes = sum(1 for v in model.values() if v)
    print(f"\nModel judgments: {len(model)}, of which {yes} relevant "
          f"({yes / max(len(model), 1) * 100:.0f}%)")
    print("\nNext: python evals/autojudge.py --agreement")
    return 0


if __name__ == "__main__":
    sys.exit(main())
