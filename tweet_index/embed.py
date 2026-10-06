"""Turn text into vectors, and find the closest vectors to a topic.

This is Section 9 of the technical design.

The idea in one paragraph: a small model reads a piece of text and returns 384
numbers. Texts that mean similar things get similar numbers, so "how related
are these two texts?" becomes simple arithmetic — multiply the two lists
together and add up the result. Because every vector is scaled to length 1,
that single number *is* the cosine similarity, between -1 and 1. Searching
5,956 tweets is one matrix multiply, a few milliseconds.

Nothing here touches the database; embed_all.py and ask.py do that.
"""

import hashlib

import numpy as np
from sentence_transformers import SentenceTransformer

# The parts of a tweet that get embedded, in the order Section 9 gives.
# Most come from `tweets`; `context` is the enrichment summary and `image_text`
# is the readable text and description of each attached image, joined. SOURCES
# says where each one lives. Phase 4's link_titles are not built.
DOC_PARTS = ["text", "quoted_text", "translation_en", "card_title",
             "card_description", "article_title", "article_preview",
             "context", "image_text", "link_titles"]

# Which table each part is selected from, for the query in embed_all.plan.
SOURCES = {part: "t" for part in DOC_PARTS}
SOURCES["context"] = "en"
SOURCES["image_text"] = "im"
SOURCES["link_titles"] = "lk"

MAX_CHARS = 2000        # Section 9 truncates the embedded text here

_model = None
_model_name = None


def model(name):
    """Load the model once per process and keep it. Takes a few seconds."""
    global _model, _model_name
    if _model is None or _model_name != name:
        _model = SentenceTransformer(name)
        _model_name = name
    return _model


def doc_text(row):
    """The text embedded for one tweet: its parts joined, empties skipped.

    `row` is anything that can be indexed by column name, such as a
    sqlite3.Row.
    """
    parts = []
    for key in DOC_PARTS:
        value = row[key]
        if value and value.strip():
            parts.append(value.strip())
    return "\n".join(parts)[:MAX_CHARS]


def text_hash(text):
    """Fingerprint of the exact text fed to the model.

    Stored alongside the vector so a rerun can tell which tweets changed and
    re-embed only those.
    """
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def embed_docs(texts, name, batch_size=64):
    """Embed a list of texts. Returns an (N, 384) float32 array, unit length."""
    vectors = model(name).encode(texts, batch_size=batch_size,
                                 normalize_embeddings=True,
                                 show_progress_bar=False)
    return np.asarray(vectors, dtype=np.float32)


def embed_query(topic, name, query_prefix):
    """Embed a search topic.

    bge models are trained to embed short queries with an instruction prefix
    and documents without one; using it measurably improves retrieval for
    two-to-five-word topics (Section 9).
    """
    vector = model(name).encode(query_prefix + topic,
                                normalize_embeddings=True,
                                show_progress_bar=False)
    return np.asarray(vector, dtype=np.float32)


def to_blob(vector):
    """A vector as bytes, for the embeddings.vector BLOB column."""
    return np.asarray(vector, dtype=np.float32).tobytes()


def from_blob(blob):
    return np.frombuffer(blob, dtype=np.float32)


def load_matrix(conn, table="embeddings"):
    """Every stored vector as one matrix, plus the tweet ids in row order.

    `table` exists so an earlier snapshot can be searched instead of the live
    vectors, which keeps a phase's search results reproducible after a
    re-embed. Table names cannot be parameterised in SQL, so it is checked
    against the names actually present.
    """
    allowed = {r[0] for r in conn.execute(
        "SELECT name FROM sqlite_master WHERE type='table'"
        " AND name LIKE 'embeddings%'")}
    if table not in allowed:
        raise ValueError(f"no embeddings table named {table!r}; "
                         f"have {sorted(allowed)}")
    rows = conn.execute(f"SELECT tweet_id, vector FROM {table}"
                        f" ORDER BY tweet_id").fetchall()
    if not rows:
        return [], np.zeros((0, 384), dtype=np.float32)
    ids = [r[0] for r in rows]
    matrix = np.vstack([from_blob(r[1]) for r in rows])
    return ids, matrix


def search(query_vector, ids, matrix, k):
    """The k closest tweets to a query vector, best first.

    Every row of `matrix` and `query_vector` has length 1, so the dot product
    is the cosine similarity and no division is needed.
    """
    if not ids:
        return []
    sims = matrix @ query_vector
    k = min(k, len(ids))
    # argpartition finds the top k without sorting all 5,956, then we sort
    # just those k.
    top = np.argpartition(-sims, k - 1)[:k]
    top = top[np.argsort(-sims[top])]
    return [(ids[i], float(sims[i])) for i in top]
