# Tweet Index

Search your own X likes and bookmarks, locally.

X gives you a list of everything you ever saved and almost no way to find
anything in it. This builds a private search engine over your own saves: it
reads the pictures, fetches what the links point at, writes a one-line summary
of each post, and ranks results by how relevant *and* how substantial they
are. Everything stays on your machine except the model calls.

It is a personal tool, published because the shape of it may be useful. There
is no hosted version and no account to make.

<img width="4440" height="2790" alt="Tweet Index" src="https://github.com/user-attachments/assets/f3c2c5a2-ea26-44a6-9cfe-2ef936abc5cb" />

```
$ ti
Tweet Index  —  4,812 tweets
Last update: 2026-10-05 09:14
Ready in 1.6s (model 1.6s, vectors 24ms)

ti> strength training for beginners

4,812 tweets searched for "strength training for beginners"
120 candidates reranked, 120 scored, 0 failed

STRONG MATCHES
 1. 0.857  @somecoach  2026-05-22  like
    relevance 3.97 (conf 0.97)   substance 1.12 [judged]   cosine 0.743
    Your weak core is destroying your posture. Seven exercises:
    > Seven core-strengthening exercises, with a claim that a weak core
      harms lower-back health and posture.
    https://x.com/somecoach/status/1234567890123456789
...
```

## How it works

Five stages, each of which you can run on its own.

1. **Capture.** A userscript copies the responses X's own page already
   receives as you scroll your saved posts. It never writes to your account
   and never sends a request X was not already making.
2. **Store.** Those captures are parsed into SQLite. The raw JSON is kept
   forever and is the source of truth; every table except `tweets` can be
   dropped and rebuilt, and `tweets` can be rebuilt from the raw files.
3. **Enrich.** Images are described (text in them *and* what they show), link
   titles are fetched, and each post gets a short summary of what it is
   about. This is what makes a picture with no caption findable at all.
4. **Index.** Everything is embedded locally with a small sentence
   transformer. No API call, no data leaving the machine.
5. **Search.** Cosine similarity picks 120 candidates, a judge model scores
   each one's relevance to your question, and the results are ranked on
   relevance, substance and similarity together, then split into *strong
   matches* and *maybe*.

The last stage is why this is more than a text search. "A thread on
retrieval" and "lol so true" can mention the same words; only one is worth
reading, and the substance score is what separates them.

## Getting started

Needs **Python 3.11+**, **Firefox** with
[Violentmonkey](https://violentmonkey.github.io/), and about 2 GB free for the
images.

```bash
git clone <this repo>
cd tweet-index
python -m venv .venv

# Windows
.venv\Scripts\python.exe -m pip install -r requirements.txt
# macOS, Linux
.venv/bin/python -m pip install -r requirements.txt
```

Then your keys:

```bash
cp .env.example .env      # and fill in the two keys
```

**You can start without any keys.** Capturing, storing and embedding are all
local and free; you only need keys for the summaries, image descriptions and
the relevance rerank. `ti` will tell you which key is missing when it needs
one.

### Capture your saves

1. Install `capture/tweet-index.user.js` in Violentmonkey.
2. Open `https://x.com/i/history` (your likes) or your bookmarks, and reload
   the page.
3. Click **Capture** in the panel, top right. Keep the tab in front — browsers
   throttle background tabs. It scrolls to the end and downloads a JSON file.
4. Do the same for the other feed.

### Index it

```bash
python update.py
```

This runs everything: it picks the capture files out of your Downloads folder,
stores them, reads the images, fetches the link titles, writes the summaries,
embeds it all, scores substance, and checks the summaries actually match
their tweets. Safe to stop and rerun — every stage picks up only what it has
not done.

Expect roughly **$0.05 per 100 new posts**, almost all of it the image
descriptions.

### Search

```bash
python ti.py          # or `ti`, see below
```

```
ti> anything without a slash is a search
ti> /basic <topic>       plain embedding search, free and instant
ti> /ask <topic> -n 8    8 strong matches instead of 5
ti> /more                the next 10 results, free
ti> /open 3              open result 3 in the browser
ti> /update              process new captures, then reload
ti> /status              counts, last update, spend this month
ti> /history             the last 10 searches
ti> /help
```

The shell holds the embedding model in memory, so after the first search each
one costs only the search itself — about 0.7s instead of 2.7s.

One-off searches from the command line work too:

```bash
python ask.py "what did I save about evals"
python ask.py "strength training" --basic     # free, no API call
```

### `ti` from anywhere

Put `bin` on your PATH and the command is just `ti`:

```powershell
# Windows, once. Then open a new window.
[Environment]::SetEnvironmentVariable(
  'PATH',
  [Environment]::GetEnvironmentVariable('PATH','User') + ';C:\path\to\tweet-index\bin',
  'User')
```

`bin/ti.ps1` is what PowerShell finds; `bin/ti.cmd` is what cmd.exe finds.
Both work out where the project is from their own location, so moving it needs
no edit. On macOS and Linux, a one-line shell alias to
`.venv/bin/python ti.py` does the same job.

## Configuration

Everything tunable is in `config.yaml`: which models, the rate limits, and the
ranking weights and thresholds. The numbers in there were tuned against one
person's corpus of ~6,000 posts, so treat them as a starting point. The ones
most worth revisiting:

| Setting | What it does |
|---|---|
| `ask.candidate_cap` | how many posts reach the relevance judge (cost scales with this) |
| `ask.weights` | how relevance, substance and similarity are blended |
| `ask.confidence_floor` | how sure the judge must be to call something a strong match |
| `ask.substance_neutral` | what an unjudgeable post scores — the corpus median |

Changing `confidence_floor` has the largest visible effect. Too high and
good results sit in *maybe*; too low and weak ones get promoted.

## Measuring whether it works

`evals/` compares ranking versions against judgments rather than vibes.

```bash
python evals/judge.py --build          # build a pool from every version
python evals/judge.py --sample 60      # judge 60 yourself, blind
python evals/autojudge.py              # a larger model judges the rest
python evals/autojudge.py --agreement  # how often it agreed with you
python evals/judge.py --score          # Hit@5, Recall@10, MRR per version
```

The pool is the union of every version's top 10, shuffled, with no indication
of which version found what. Judging a sample yourself and checking the model
against it is the only reason to trust the rest.

Two things worth knowing before you read your own numbers. **Hit@5 saturates**
— every version scores near 100% because the pool is built from their top 10
— so Recall@10 and MRR are what separate them. And **agreement must be
measured on the population the claim is about**: 80% overall once hid 67% on
the handful of results that actually decided one version against another.

`evals/queries.yaml` ships with twenty generic example topics. Replace them
with subjects you actually save things about, or the numbers mean nothing.

## Tests

```bash
python -m pytest tests/ -q        # 220
node capture/test_userscript.mjs  # 55
```

The test payloads are synthetic, built by `tests/make_fixtures.py` — the
original suite used real captures, which are personal data and are not in
this repository. `python tests/make_fixtures.py` prints what it generates.

## What is not here

- **No data.** No database, no captures, no images, no summaries. The author's
  are personal; yours will be too. `data/` is git-ignored, and the first run
  creates everything in it.
- **No hosted service.** It runs on your machine, against your own saves.
- **Likes only go back as far as X will serve them.** There is no API behind
  this, only what the page itself loads.

## Costs

Measured on ~6,000 posts, at the prices in `config.yaml`:

| Step | Cost |
|---|---|
| Reading 4,172 images | $0.78 |
| Summaries, all posts | $0.66 |
| Substance scores | $0.15 |
| Embeddings | free, runs locally |
| One search | ~$0.003 |
| A weekly update, ~100 new posts | ~$0.05 |

A full index from nothing is a couple of dollars. Keeping it current is
pennies a month.

## A note on the capture approach

The userscript reads responses the page has already fetched. It makes no
requests of its own, sends nothing anywhere, and cannot post, like, follow or
delete. That is a deliberate limit: the one thing worse than not finding a
saved post is a tool that touches your account. If you would rather read the
code before trusting it, `capture/tweet-index.user.js` is 430 lines and the
network handling is the first 60.

## Licence

MIT. See `LICENSE`.
