"""SQLite connection and schema for Tweet Index.

The schema is the full Section 5.1 DDL, including tables that only later
phases fill (enrichment, embeddings, concepts, concept_scores,
tweet_judgments, tweet_tags, queries). Creating them now means no
migrations later; Phase 0 simply leaves them empty.

Everything is CREATE TABLE IF NOT EXISTS, so connect() is safe to call
on an existing database.
"""

import sqlite3
from pathlib import Path

SCHEMA = """
PRAGMA journal_mode = WAL;
PRAGMA foreign_keys = ON;

CREATE TABLE IF NOT EXISTS tweets (
  id            TEXT PRIMARY KEY,        -- X tweet id; TEXT because it exceeds
                                         -- JavaScript's safe integer range
  url           TEXT NOT NULL,
  author_handle TEXT,
  author_name   TEXT,
  text          TEXT NOT NULL,           -- long-form note text if present
  lang          TEXT,
  quoted_id     TEXT,
  quoted_author TEXT,
  quoted_text   TEXT,                    -- note_tweet or X Article text if present
  in_reply_to_id   TEXT,                 -- set when the saved post is a reply
  initial_tweet_id TEXT,                 -- original id if this is an edited post
  card_title       TEXT,                 -- link preview card, free context
  card_description TEXT,
  article_title    TEXT,                 -- X Article (own or quoted)
  article_preview  TEXT,
  translation_en   TEXT,                 -- X's English translation, if offered
  community_note   TEXT,                 -- Community Note text, if any
  save_order    TEXT,                    -- entry sortIndex; orders saves WITHIN
                                         -- one feed only (see 5.2)
  posted_at     TEXT,                    -- ISO 8601, UTC
  first_seen    TEXT NOT NULL,           -- when WE first captured it
  is_like       INTEGER NOT NULL DEFAULT 0,
  is_bookmark   INTEGER NOT NULL DEFAULT 0,
  raw_file      TEXT NOT NULL            -- provenance: first raw file seen in
);
CREATE INDEX IF NOT EXISTS ix_tweets_first_seen ON tweets(first_seen);

CREATE TABLE IF NOT EXISTS media (
  id          INTEGER PRIMARY KEY AUTOINCREMENT,
  tweet_id    TEXT NOT NULL REFERENCES tweets(id),
  kind        TEXT NOT NULL,             -- photo | video | animated_gif
  remote_url  TEXT NOT NULL,
  local_path  TEXT,
  ocr_text    TEXT,
  ocr_engine  TEXT,
  UNIQUE (tweet_id, remote_url)
);

CREATE TABLE IF NOT EXISTS links (
  id           INTEGER PRIMARY KEY AUTOINCREMENT,
  tweet_id     TEXT NOT NULL REFERENCES tweets(id),
  url          TEXT NOT NULL,            -- expanded URL, never t.co
  title        TEXT,
  fetched_text TEXT,                     -- Phase 4
  fetch_status TEXT NOT NULL DEFAULT 'pending', -- pending|ok|failed|skipped
  UNIQUE (tweet_id, url)
);

CREATE TABLE IF NOT EXISTS enrichment (
  tweet_id          TEXT PRIMARY KEY REFERENCES tweets(id),
  context           TEXT NOT NULL,
  proposed_concepts TEXT NOT NULL,       -- JSON array of strings
  model_used        TEXT NOT NULL,
  input_hash        TEXT NOT NULL,       -- re-enrich when inputs change
  created_at        TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS embeddings (
  tweet_id   TEXT PRIMARY KEY REFERENCES tweets(id),
  vector     BLOB NOT NULL,              -- float32[384], L2-normalized
  model_used TEXT NOT NULL,
  text_hash  TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS concepts (
  id            TEXT PRIMARY KEY,        -- slug, e.g. agentic_security
  label         TEXT NOT NULL,
  question      TEXT,                    -- NULL for inherit-only roots
  parent_ids    TEXT NOT NULL DEFAULT '[]', -- JSON; multiple parents allowed
  vocab_version INTEGER NOT NULL
);

CREATE TABLE IF NOT EXISTS concept_scores (
  -- raw Jev Noul outputs
  tweet_id      TEXT NOT NULL REFERENCES tweets(id),
  concept_id    TEXT NOT NULL REFERENCES concepts(id),
  probability   REAL NOT NULL,
  jev_version   TEXT NOT NULL,
  vocab_version INTEGER NOT NULL,
  PRIMARY KEY (tweet_id, concept_id)
);

CREATE TABLE IF NOT EXISTS tweet_judgments (  -- one row = tweet tagged at version
  tweet_id       TEXT PRIMARY KEY REFERENCES tweets(id),
  substance      REAL NOT NULL,             -- Jev Score position, 0..4
  substance_conf REAL NOT NULL,
  vocab_version  INTEGER NOT NULL,
  jev_version    TEXT NOT NULL,
  judged_at      TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS tweet_tags (       -- derived; rebuilt in code
  tweet_id   TEXT NOT NULL REFERENCES tweets(id),
  concept_id TEXT NOT NULL REFERENCES concepts(id),
  inherited  INTEGER NOT NULL,
  threshold  REAL NOT NULL,
  PRIMARY KEY (tweet_id, concept_id)
);
CREATE INDEX IF NOT EXISTS ix_tags_concept ON tweet_tags(concept_id);

CREATE TABLE IF NOT EXISTS queries (
  id         INTEGER PRIMARY KEY AUTOINCREMENT,
  asked_at   TEXT NOT NULL,
  topic      TEXT NOT NULL,
  config     TEXT NOT NULL,              -- JSON snapshot of ask settings + versions
  results    TEXT NOT NULL,              -- JSON [{id, rel, rel_conf, sub, final, band}]
  latency_ms INTEGER,
  jev_tokens INTEGER
);

-- Not in Section 5.1. Phase 0 addition so ingest can skip files it has
-- already processed (Section 7.4, "Same file ingested twice").
CREATE TABLE IF NOT EXISTS ingested_files (
  filename    TEXT PRIMARY KEY,
  kind        TEXT NOT NULL,
  ingested_at TEXT NOT NULL,
  entries     INTEGER NOT NULL,
  skipped     INTEGER NOT NULL
);
"""


def connect(db_path):
    """Open (creating if needed) the database and make sure the schema exists."""
    path = Path(db_path).expanduser()
    path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(path)
    conn.row_factory = sqlite3.Row
    conn.executescript(SCHEMA)
    return conn


def count(conn, table):
    """Number of rows in a table. Handy in tests and ingest summaries."""
    return conn.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0]
