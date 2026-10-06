"""Download saved images and describe what is in them (Phase 4).

    python describe_images.py --sample 30     # a spread, for checking
    python describe_images.py --estimate      # cost and time, no calls
    python describe_images.py                 # everything not yet described
    python describe_images.py --ocr-only      # Windows OCR, no API calls

Many saved bookmarks are a photo with no caption, or a screenshot of text. Of
5,956 tweets, 230 have no searchable text at all and a further 900-odd are
near-silent. Their content is in the picture.

Two things are produced per image and stored together in `media.ocr_text`:
any readable text, then one sentence on what the image shows. The text alone is
not enough -- a photograph of a room contains no words -- and the description
alone loses the screenshots of text, which are common.

Images are downloaded to data/media/ first. A save is only a pointer (PRD
Section 2); once the local copy exists the picture survives the tweet.

Videos are described from their thumbnail, which is what X gives us in
`media_url_https`.
"""

import argparse
import json
import sys
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path

import yaml

from tweet_index import ratelimit
from tweet_index.db import connect
from tweet_index.llm import LLMError, complete_json, set_limiter

ROOT = Path(__file__).resolve().parent

SYSTEM_PROMPT = """You read images saved from social media for a personal search index.

Return only a JSON object, no prose and no code fences:
{
  "text": "every piece of readable text in the image, in reading order,
           verbatim. Empty string if there is none.",
  "description": "one sentence on what the image shows. Name what is in it:
                  the room, the exercise, the chart, the product, the person's
                  activity. No hype, no opinions, no guesses about why it was
                  saved."
}

Transcribe text exactly, including numbers and handles. Do not summarise or
correct it. If the image is a screenshot of a post or article, transcribe the
body text, not the browser or app furniture.

Stop after about 1200 characters of text. If the image holds more than that --
a full page, a long letter, a book spread -- transcribe the opening and stop
there. The point is to make the image findable, not to reproduce it.

Describe only what is visible. If you cannot tell what something is, say what
it looks like rather than guessing a specific name."""

# Used when the first attempt is halted by a content filter. Every image that
# hit the filter was a long verbatim passage from published work: a Buffett
# article from 1951, Jobs' Stanford speech, a page of Tolkien, a court opinion,
# a reposted Instagram essay. The filter is doing its job -- asking for a full
# transcription of copyrighted text is the wrong request. A short identifying
# excerpt plus a description still makes the image findable.
BRIEF_PROMPT = """You read images saved from social media for a personal search index.

Return only a JSON object, no prose and no code fences:
{
  "text": "at most 200 characters: the title, author, headline, opening
           phrase or other identifying words. Not a transcription.",
  "description": "one or two sentences on what the image shows and what the
                  text is about. Name the work, author, publication or subject
                  if they are visible."
}

Do not reproduce the body text. Identify the material and say what it covers."""

SCHEMA = {
    "type": "object",
    "properties": {"text": {"type": "string"},
                   "description": {"type": "string"}},
    "required": ["text", "description"],
    "additionalProperties": False,
}

# gpt-6-luna at $0.10 per million input tokens. A tweet photo costs roughly
# 1,100-1,400 input tokens; measured per run and reported.
PRICE_IN = 0.10
PRICE_OUT = 0.50

USER_AGENT = "tweet-index/0.1 (personal archive)"

LAST_RUN = {}


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
    with (logs_dir / "media.log").open("a", encoding="utf-8") as fh:
        fh.write(f"{stamp} {message}\n")


# ---------------------------------------------------------------------------
# Downloading
# ---------------------------------------------------------------------------

def local_name(row):
    """data/media/<tweet_id>_<media id>.jpg"""
    return f"{row['tweet_id']}_{row['id']}.jpg"


def download(row, media_dir):
    """Fetch one image. Returns the path, or raises."""
    path = media_dir / local_name(row)
    if path.exists() and path.stat().st_size > 0:
        return path
    # X serves a usable size without the ?name= parameter; ask for a large one.
    url = row["remote_url"]
    if "pbs.twimg.com" in url and "?" not in url:
        url += "?name=large"
    request = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    with urllib.request.urlopen(request, timeout=30) as response:
        data = response.read()
    if not data:
        raise OSError("empty response")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(data)
    return path


# ---------------------------------------------------------------------------
# Windows OCR, for comparison
# ---------------------------------------------------------------------------

def windows_ocr(path):
    """Text from Windows' built-in OCR engine. Returns None if unavailable."""
    try:
        import asyncio

        from winsdk.windows.graphics.imaging import BitmapDecoder
        from winsdk.windows.media.ocr import OcrEngine
        from winsdk.windows.storage import FileAccessMode, StorageFile
    except ImportError:
        return None

    async def run():
        engine = OcrEngine.try_create_from_user_profile_languages()
        if engine is None:
            return None
        handle = await StorageFile.get_file_from_path_async(str(path))
        stream = await handle.open_async(FileAccessMode.READ)
        decoder = await BitmapDecoder.create_async(stream)
        bitmap = await decoder.get_software_bitmap_async()
        result = await engine.recognize_async(bitmap)
        return result.text

    return asyncio.run(run())


# ---------------------------------------------------------------------------
# The vision call
# ---------------------------------------------------------------------------

def describe(path, model, max_output_tokens, reasoning_effort):
    """Read one image. Returns (text, description, usage).

    Two ways the budget runs out, both seen in the full run:

      * thinking fills it and the reply arrives empty (2 of 4,172);
      * the transcription of a text-dense image runs past it and the JSON is
        cut off mid-string (14 of 4,172) -- a book page or a long letter.

    Both are retried once with triple the budget. The prompt also caps the
    transcription at about 1,200 characters, which is the real fix: the point
    is to make an image findable, not to reproduce a full page.
    """
    image = path.read_bytes()
    attempts = [(SYSTEM_PROMPT, max_output_tokens),
                (SYSTEM_PROMPT, max_output_tokens * 3),
                (BRIEF_PROMPT, max_output_tokens)]
    retryable = ("empty reply", "was not JSON")
    for i, (prompt, budget) in enumerate(attempts):
        try:
            out, usage = complete_json(
                prompt, "Read this image.", model, schema=SCHEMA,
                max_output_tokens=budget,
                reasoning_effort=reasoning_effort,
                image=image)
            return (out.get("text") or "").strip(), \
                   (out.get("description") or "").strip(), usage
        except LLMError as exc:
            if i == len(attempts) - 1:
                raise
            if not any(r in str(exc) for r in retryable):
                raise
            # A bigger budget cannot help a content filter, so go straight to
            # asking for an excerpt instead of a transcription.
            if "content_filter" in str(exc):
                attempts[i + 1] = (BRIEF_PROMPT, max_output_tokens)
    raise LLMError("unreachable")


def combined(text, description):
    """What goes in media.ocr_text: the words, then the picture."""
    parts = []
    if text:
        parts.append(text)
    if description:
        parts.append(description)
    return "\n".join(parts)


# ---------------------------------------------------------------------------
# Selecting work
# ---------------------------------------------------------------------------

def candidates(conn, redo=False):
    """Media rows with no description yet, newest tweets first."""
    rows = conn.execute("""
        SELECT m.id, m.tweet_id, m.kind, m.remote_url, m.local_path,
               m.ocr_text, m.ocr_engine,
               LENGTH(COALESCE(t.text,'')) AS tl, t.is_bookmark, t.posted_at
        FROM media m JOIN tweets t ON t.id = m.tweet_id
        ORDER BY t.posted_at DESC, m.id""").fetchall()
    if redo:
        return rows
    # An engine recorded with no text means the image was tried and nothing
    # could be had from it -- usually deleted from X's servers. Those are not
    # retried on every run.
    return [r for r in rows
            if not (r["ocr_text"] or "").strip()
            and not (r["ocr_engine"] or "").strip()]


def pick_sample(rows, size):
    """A spread: image-only bookmarks first, then captioned, then videos.

    The image-only bookmarks are the point of Phase 4 -- they are the tweets
    with nothing to search -- so at least a third of any sample is drawn from
    them.
    """
    silent_bm = [r for r in rows if r["tl"] == 0 and r["is_bookmark"]]
    silent_other = [r for r in rows if r["tl"] == 0 and not r["is_bookmark"]]
    captioned = [r for r in rows if 0 < r["tl"] < 300 and r["kind"] == "photo"]
    videos = [r for r in rows if r["kind"] != "photo" and r["tl"] < 300]
    long_posts = [r for r in rows if r["tl"] >= 300]

    want = [(silent_bm, max(10, size // 3)), (silent_other, size // 6),
            (captioned, size // 3), (videos, size // 6),
            (long_posts, size // 6)]
    chosen, seen = [], set()
    for group, n in want:
        for r in group[:n]:
            if r["id"] not in seen:
                seen.add(r["id"])
                chosen.append(r)
    for group, _ in want:                  # top up if a group was short
        for r in group:
            if len(chosen) >= size:
                break
            if r["id"] not in seen:
                seen.add(r["id"])
                chosen.append(r)
    return chosen[:size]


def save(conn, media_id, path, text, engine):
    with conn:
        conn.execute("UPDATE media SET local_path=?, ocr_text=?, ocr_engine=?"
                     " WHERE id=?", (str(path), text, engine, media_id))


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sample", type=int, metavar="N")
    parser.add_argument("--limit", type=int, metavar="N")
    parser.add_argument("--redo", action="store_true")
    parser.add_argument("--estimate", action="store_true")
    parser.add_argument("--ocr-only", action="store_true",
                        help="Windows OCR only: no API calls, text but no "
                             "description")
    parser.add_argument("--compare", action="store_true",
                        help="run both engines and print them side by side")
    parser.add_argument("--concurrency", type=int, default=8)
    parser.add_argument("--db", default=None)
    parser.add_argument("--config", default=None)
    args = parser.parse_args(argv)

    cfg = load_config(args.config)
    llm_cfg = cfg["llm"]
    model = llm_cfg["enrich_model"]
    conn = connect(Path(args.db) if args.db else resolve(cfg["paths"]["db"]))
    media_dir = resolve(cfg["paths"]["media"])
    logs_dir = resolve("data/logs")

    rows = candidates(conn, redo=args.redo)
    total_media = conn.execute("SELECT COUNT(*) FROM media").fetchone()[0]
    print(f"Media rows: {total_media}")
    print(f"Needing description: {len(rows)}")

    if args.estimate:
        kinds = {}
        for r in rows:
            kinds[r["kind"]] = kinds.get(r["kind"], 0) + 1
        print(f"  by kind: {kinds}")
        # 1,250 input + 180 output tokens per image, measured on the sample
        in_tok = len(rows) * 1250
        out_tok = len(rows) * 180
        cost = in_tok / 1e6 * PRICE_IN + out_tok / 1e6 * PRICE_OUT
        print(f"\nAt 1,250 input + 180 output tokens per image:")
        print(f"  input  {in_tok / 1e6:.2f}M, output {out_tok / 1e6:.2f}M")
        print(f"  cost   ${cost:.2f}   (batch ${cost / 2:.2f})")
        limiter = ratelimit.from_config(cfg, "openai")
        if limiter:
            per_min = min(limiter.requests.limit,
                          limiter.tokens.limit / (in_tok / max(len(rows), 1)))
            print(f"  time   ~{len(rows) / per_min:.0f} min at the limiter's pace")
        conn.close()
        return 0

    if args.sample:
        rows = pick_sample(rows, args.sample)
        print(f"Sample selected: {len(rows)}")
    elif args.limit:
        rows = rows[:args.limit]

    if not rows:
        print("\nNothing to do.")
        conn.close()
        return 0

    load_env()
    set_limiter(ratelimit.from_config(cfg, "openai"))
    engine_name = "windows.media.ocr" if args.ocr_only else model
    print(f"Engine: {engine_name}   concurrency: {args.concurrency}\n")

    t0 = time.perf_counter()
    done = failed = 0
    tokens_in = tokens_out = 0
    comparisons = []

    def work(row):
        try:
            path = download(row, media_dir)
        except urllib.error.HTTPError as exc:
            if exc.code in (403, 404, 410):
                # Gone from X's servers. Record the attempt so it is not
                # retried every run.
                return row, "gone", f"image unavailable: HTTP {exc.code}"
            return row, None, f"download failed: {exc}"
        except (urllib.error.URLError, OSError) as exc:
            return row, None, f"download failed: {exc}"
        if args.ocr_only:
            try:
                text = windows_ocr(path) or ""
            except Exception as exc:                      # noqa: BLE001
                return row, None, f"windows ocr failed: {exc}"
            return row, (path, text, "", None), None
        try:
            text, description, usage = describe(
                path, model, llm_cfg.get("max_output_tokens", 700) + 300,
                llm_cfg.get("reasoning_effort"))
        except LLMError as exc:
            return row, None, str(exc)
        ocr = None
        if args.compare:
            try:
                ocr = windows_ocr(path) or ""
            except Exception:                             # noqa: BLE001
                ocr = None
        return row, (path, text, description, usage), (None if not ocr
                                                       else None), ocr

    with ThreadPoolExecutor(max_workers=args.concurrency) as pool:
        for result in pool.map(work, rows):
            row, payload, error = result[0], result[1], result[2]
            ocr = result[3] if len(result) > 3 else None
            if payload == "gone":
                with conn:
                    conn.execute("UPDATE media SET ocr_engine=? WHERE id=?",
                                 ("unavailable", row["id"]))
                failed += 1
                log(logs_dir, f"media {row['id']} {error}")
                print(f"  {row['tweet_id']} media {row['id']}: {error}")
                continue
            if error:
                failed += 1
                log(logs_dir, f"failed media {row['id']}: {error}")
                print(f"  failed {row['tweet_id']} media {row['id']}: {error}")
                continue
            path, text, description, usage = payload
            save(conn, row["id"], path, combined(text, description),
                 engine_name)
            done += 1
            if usage:
                tokens_in += usage["input_tokens"]
                tokens_out += usage["output_tokens"]
            if ocr is not None:
                comparisons.append((row, text, ocr))
            print(f"\r  {done + failed}/{len(rows)} described   ",
                  end="", flush=True)
    seconds = time.perf_counter() - t0
    print()

    cost = tokens_in / 1e6 * PRICE_IN + tokens_out / 1e6 * PRICE_OUT
    print(f"\nDescribed {done}, failed {failed}, in {seconds:.1f}s")
    if tokens_in:
        print(f"Tokens: {tokens_in:,} in, {tokens_out:,} out"
              f"   ({tokens_in / done:.0f} in per image)")
        print(f"Cost this run: ${cost:.4f}")
        remaining = len(candidates(conn))
        projected = (remaining * (tokens_in / done) / 1e6 * PRICE_IN
                     + remaining * (tokens_out / done) / 1e6 * PRICE_OUT)
        print(f"Projected for the remaining {remaining}: ${projected:.2f}")

    if comparisons:
        print("\n--- vision text vs Windows OCR ---")
        for row, text, ocr in comparisons[:5]:
            print(f"\n{row['tweet_id']} media {row['id']} ({row['kind']})")
            print(f"  vision : {(text or '(none)')[:160]!r}")
            print(f"  windows: {(ocr or '(none)')[:160]!r}")

    log(logs_dir, f"described {done}, failed {failed}, {seconds:.1f}s, "
                  f"{tokens_in} in / {tokens_out} out, engine {engine_name}")
    LAST_RUN.clear()
    LAST_RUN.update(step="images", done=done, failed=failed,
                    seconds=seconds, cost=cost)
    conn.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
