"""
prose_search.py — semantic retrieval over prose_chunks (pgvector, cosine).

The prose lane is deliberately boring: chunk -> embed -> HNSW cosine top-k. It answers
"what does this field say / which documents mention X"; tabular content belongs to the
tables lane (table_search.py), never here.
"""

from typing import Any, Dict, List, Optional, Sequence

from . import db, embeddings


def available() -> bool:
    """The lane needs Postgres; the embedder is preferred but not required — see
    search(), which falls back to full-text ranking when vectors are unavailable."""
    return db.available()


def semantic_available() -> bool:
    """True when the vector path can actually run (embedder cached AND chunks embedded)."""
    return db.available() and embeddings.available()


def search(query: str, k: int = 8, doc_ids: Optional[Sequence[int]] = None) -> List[Dict[str, Any]]:
    """
    Top-k prose chunks for a query, each with provenance (filename/page) and a score.
    Returns [] when the lane isn't configured — callers degrade, never crash.

    Uses pgvector cosine when embeddings exist, else falls back to Postgres full-text
    ranking over the same chunks. The fallback matters: embeddings are populated by a
    2.3 GB model that may not be cached yet, and chunks ingested while it was missing
    carry NULL embeddings (ingest.py inserts them anyway and backfills later). Without
    this the lane returns nothing at all in that window — the corpus looks empty and
    every question answers "no content", even though the text is sitting right there.
    Results are strictly worse than semantic (no synonym/paraphrase matching), so
    `match` on each hit records which path produced it.

    doc_ids, if given, restricts the search to those documents (e.g. the chat UI's
    document picker) — this lane owns its own SQL, so the filter is applied directly
    rather than relying on the RLS path used by the generated text-to-SQL lane.
    """
    if not db.available() or not (query or "").strip():
        return []
    if not semantic_available() or not _has_embeddings():
        return _keyword_search(query, k, doc_ids)
    qvec = embeddings.to_pgvector(embeddings.embed([query])[0])
    sql = """
        SELECT p.chunk_id, p.doc_id, d.filename, d.line_name, p.page, p.section,
               p.content, 1 - (p.embedding <=> %s::vector) AS score
        FROM prose_chunks p
        JOIN documents d USING (doc_id)
        WHERE p.embedding IS NOT NULL AND d.status = 'ready'
    """
    params: List[Any] = [qvec]
    if doc_ids:
        sql += " AND p.doc_id = ANY(%s)"
        params.append(list(doc_ids))
    sql += " ORDER BY p.embedding <=> %s::vector LIMIT %s"
    params += [qvec, k]
    with db.connect(readonly=True) as conn:
        with conn.cursor() as cur:
            cur.execute(sql, params)
            cols = [c.name for c in (cur.description or [])]
            rows = cur.fetchall()
    return [dict(zip(cols, r), match="semantic") for r in rows]


def _has_embeddings() -> bool:
    """True when at least one chunk carries a vector — the vector query filters on
    `embedding IS NOT NULL`, so an all-NULL table would return zero hits silently."""
    try:
        with db.connect(readonly=True) as conn:
            with conn.cursor() as cur:
                cur.execute("SELECT 1 FROM prose_chunks WHERE embedding IS NOT NULL LIMIT 1")
                return cur.fetchone() is not None
    except Exception:
        return False


_STOPish = {"what", "which", "who", "whom", "whose", "when", "where", "why", "how",
            "the", "and", "for", "are", "was", "were", "does", "did", "has", "have",
            "this", "that", "with", "from", "into", "any", "all", "its", "his", "her",
            "there", "their", "about", "please", "tell", "show", "list", "give"}


def _or_tsquery(query: str) -> str:
    """`a | b | c` from the query's content words, sanitized for to_tsquery.

    to_tsquery takes operator syntax, not free text, so anything non-alphanumeric must
    go or it raises a syntax error. Question words are dropped: they appear in no chunk
    and only dilute the rank.
    """
    import re
    words = [w for w in re.findall(r"[A-Za-z0-9]+", (query or "").lower())
             if len(w) > 2 and w not in _STOPish]
    seen, out = set(), []
    for w in words:
        if w not in seen:
            seen.add(w)
            out.append(w)
    return " | ".join(out[:12])


def _keyword_search(query: str, k: int, doc_ids: Optional[Sequence[int]]) -> List[Dict[str, Any]]:
    """Full-text ranking fallback. Tries websearch_to_tsquery first (handles quoted
    phrases and OR), then a plain ILIKE substring pass for short/odd queries that
    stem to nothing — e.g. an identifier like 'KJDH' or a bare well number."""
    base = ("SELECT p.chunk_id, p.doc_id, d.filename, d.line_name, p.page, p.section, "
            "p.content, {score} AS score "
            "FROM prose_chunks p JOIN documents d USING (doc_id) "
            "WHERE d.status = 'ready' AND {pred}")
    scope, scope_params = ("", [])
    if doc_ids:
        scope, scope_params = " AND p.doc_id = ANY(%s)", [list(doc_ids)]

    # Ordered widest-precision-first. The OR pass is what makes natural-language
    # questions work at all: websearch_to_tsquery ANDs every term, so "who prepared the
    # well completion report" matches only a chunk containing all five words and
    # normally returns nothing. ts_rank still floats the chunks hitting the most terms.
    attempts = [
        (base.format(score="ts_rank(to_tsvector('english', p.content), "
                           "websearch_to_tsquery('english', %s))",
                     pred="to_tsvector('english', p.content) @@ "
                          "websearch_to_tsquery('english', %s)"),
         [query, query]),
    ]
    or_query = _or_tsquery(query)
    if or_query:
        attempts.append(
            (base.format(score="ts_rank(to_tsvector('english', p.content), "
                               "to_tsquery('english', %s))",
                         pred="to_tsvector('english', p.content) @@ "
                              "to_tsquery('english', %s)"),
             [or_query, or_query]))
    attempts.append(
        (base.format(score="0.0", pred="p.content ILIKE %s"), [f"%{query.strip()}%"]))

    for sql, params in attempts:
        sql += scope + " ORDER BY score DESC, p.chunk_id LIMIT %s"
        try:
            with db.connect(readonly=True) as conn:
                with conn.cursor() as cur:
                    cur.execute(sql, params + scope_params + [k])
                    cols = [c.name for c in (cur.description or [])]
                    rows = cur.fetchall()
        except Exception as e:
            print(f"[rag.prose_search] keyword pass failed ({e})")
            continue
        if rows:
            return [dict(zip(cols, r), match="keyword") for r in rows]

    if doc_ids:
        # Document-scoped fallback: if user picked document(s) in Thinker
        # and asks a general question (e.g. "what is this document about", "summarize"),
        # return the document's chunks in page order so the LLM has context to answer.
        fallback_sql = (
            "SELECT p.chunk_id, p.doc_id, d.filename, d.line_name, p.page, p.section, "
            "p.content, 0.1 AS score "
            "FROM prose_chunks p JOIN documents d USING (doc_id) "
            "WHERE d.status = 'ready' AND p.doc_id = ANY(%s) "
            "ORDER BY p.page, p.chunk_id LIMIT %s"
        )
        try:
            with db.connect(readonly=True) as conn:
                with conn.cursor() as cur:
                    cur.execute(fallback_sql, [list(doc_ids), k])
                    cols = [c.name for c in (cur.description or [])]
                    rows = cur.fetchall()
                    if rows:
                        return [dict(zip(cols, r), match="doc_fallback") for r in rows]
        except Exception as e:
            print(f"[rag.prose_search] doc_ids fallback failed ({e})")
    else:
        # Corpus-wide fallback for general questions
        fallback_sql = (
            "SELECT p.chunk_id, p.doc_id, d.filename, d.line_name, p.page, p.section, "
            "p.content, 0.05 AS score "
            "FROM prose_chunks p JOIN documents d USING (doc_id) "
            "WHERE d.status = 'ready' "
            "ORDER BY d.doc_id DESC, p.page, p.chunk_id LIMIT %s"
        )
        try:
            with db.connect(readonly=True) as conn:
                with conn.cursor() as cur:
                    cur.execute(fallback_sql, [k])
                    cols = [c.name for c in (cur.description or [])]
                    rows = cur.fetchall()
                    if rows:
                        return [dict(zip(cols, r), match="corpus_fallback") for r in rows]
        except Exception as e:
            print(f"[rag.prose_search] corpus fallback failed ({e})")

    return []


def backfill(batch: int = 64) -> int:
    """
    Embed any prose_chunks still missing an embedding (e.g. ingested before the prose lane
    existed, or while the embedder was unavailable). Returns how many were embedded.
    Idempotent — safe to run repeatedly.
    """
    if not available():
        return 0
    done = 0
    with db.connect() as conn:
        while True:
            with conn.cursor() as cur:
                cur.execute("SELECT chunk_id, content FROM prose_chunks "
                            "WHERE embedding IS NULL ORDER BY chunk_id LIMIT %s", (batch,))
                rows = cur.fetchall()
            if not rows:
                break
            vecs = embeddings.embed([r[1] for r in rows])
            with conn.cursor() as cur:
                cur.executemany(
                    "UPDATE prose_chunks SET embedding = %s::vector WHERE chunk_id = %s",
                    [(embeddings.to_pgvector(v), rid) for (rid, _), v in zip(rows, vecs)])
            conn.commit()
            done += len(rows)
            print(f"[rag.prose_search] embedded {done} chunk(s)...")
    return done


if __name__ == "__main__":
    n = backfill()
    print(f"backfilled {n} chunk(s)")
