"""
table_search.py — retrieval over page_tables for the HTML-in-context lane.

Deliberately a near-clone of prose_search.py: same available()/semantic_available() split,
same "vector when embeddings exist, full-text when they don't" degradation, same direct
doc_ids WHERE clause (this lane owns its SQL, so it filters directly; the RLS policy on
page_tables is defence in depth for any other reader). One shape to maintain, not two.

The search key is `digest` — a short natural-language description built at ingest — never
`html`. Embedding markup produces a vector dominated by "<td>" and "</tr>", and HTML runs
~1.7 chars/token against this model's tokenizer (vs ~3.9 for prose), so a large table would
also run past the embedder's useful window. `html` is what gets RETRIEVED, never searched.

This lane replaces the retired text-to-SQL lane. Note one property it has that the SQL lane
did not: retrieval here is pure pgvector/Postgres, so it keeps working with vLLM down —
only the narration step needs the LLM.
"""

from typing import Any, Dict, List, Optional, Sequence

from . import db, embeddings

# Columns every caller needs. `html` is the payload; the rest drive the budget allocator
# and the warnings (a dropped or malformed table must be nameable in a message).
_COLS = ("t.table_pk, t.doc_id, d.filename, t.page, t.table_index, t.caption, "
         "t.html, t.n_rows, t.n_cols, t.est_tokens, t.well_formed")


def available() -> bool:
    """The lane needs Postgres; the embedder is preferred but not required — see search()."""
    return db.available()


def semantic_available() -> bool:
    """True when the vector path can actually run (embedder cached AND digests embedded)."""
    return db.available() and embeddings.available()


def _has_embeddings() -> bool:
    """True when at least one table carries a vector. The vector query filters on
    `embedding IS NOT NULL`, so an all-NULL table would silently return zero hits."""
    try:
        with db.connect(readonly=True) as conn:
            with conn.cursor() as cur:
                cur.execute("SELECT 1 FROM page_tables WHERE embedding IS NOT NULL LIMIT 1")
                return cur.fetchone() is not None
    except Exception:
        return False


def _rows(sql: str, params: List[Any], match: str) -> List[Dict[str, Any]]:
    with db.connect(readonly=True) as conn:
        with conn.cursor() as cur:
            cur.execute(sql, params)
            cols = [c.name for c in (cur.description or [])]
            out = cur.fetchall()
    return [dict(zip(cols, r), match=match) for r in out]


def search(query: str, k: int = 4, doc_ids: Optional[Sequence[int]] = None,
           pages: Optional[Sequence[tuple]] = None) -> List[Dict[str, Any]]:
    """
    Top-k tables for a query, each with provenance and a score. Returns [] when the lane
    isn't configured — callers degrade, never crash.

    `pages` is an optional list of (doc_id, page) already known to be relevant — the prose
    lane's hits. Tables on those pages are unioned in, because digest search alone misses
    context-phrased questions ("what did they do about lost circulation") that match the
    surrounding page strongly and the table's own digest weakly. The reverse is also true,
    which is why neither signal is used alone: for documents whose tables live inside prose
    chunks, the prose embedding IS a vector of markup — the weakest possible signal.

    k is small (4) on purpose. Tables are all-or-nothing in the prompt and one real table
    can be ~1,500 tokens of a ~6,200-token budget, so retrieving 10 would guarantee drops.
    """
    if not db.available():
        return []

    hits: List[Dict[str, Any]] = []
    scoped = " AND t.doc_id = ANY(%s)" if doc_ids else ""

    if (query or "").strip():
        if semantic_available() and _has_embeddings():
            qvec = embeddings.to_pgvector(embeddings.embed([query])[0])
            sql = (f"SELECT {_COLS}, 1 - (t.embedding <=> %s::vector) AS score "
                   "FROM page_tables t JOIN documents d USING (doc_id) "
                   "WHERE t.embedding IS NOT NULL AND d.status = 'ready'" + scoped +
                   " ORDER BY t.embedding <=> %s::vector LIMIT %s")
            p: List[Any] = [qvec]
            if doc_ids:
                p.append(list(doc_ids))
            p += [qvec, k]
            hits = _rows(sql, p, "semantic")
        else:
            # Full-text over the digest. OR-of-terms, not AND: websearch_to_tsquery ANDs
            # every word, so a natural-language question matches nothing at all.
            terms = _or_tsquery(query)
            if terms:
                sql = (f"SELECT {_COLS}, "
                       "ts_rank(to_tsvector('english', coalesce(t.digest,'')), "
                       "        to_tsquery('english', %s)) AS score "
                       "FROM page_tables t JOIN documents d USING (doc_id) "
                       "WHERE d.status = 'ready' "
                       "  AND to_tsvector('english', coalesce(t.digest,'')) @@ "
                       "      to_tsquery('english', %s)" + scoped +
                       " ORDER BY score DESC, t.table_pk LIMIT %s")
                p = [terms, terms]
                if doc_ids:
                    p.append(list(doc_ids))
                p.append(k)
                try:
                    hits = _rows(sql, p, "keyword")
                except Exception as e:
                    print(f"[rag.table_search] keyword pass failed ({e})")

    # Page expansion from the prose hits.
    if pages:
        have = {(h["doc_id"], h["page"]) for h in hits}
        want = [(d, p) for (d, p) in dict.fromkeys(pages) if (d, p) not in have]
        if want:
            pair_clauses = " OR ".join(["(t.doc_id = %s AND t.page = %s)"] * len(want))
            sql = (f"SELECT {_COLS}, 0.0 AS score "
                   "FROM page_tables t JOIN documents d USING (doc_id) "
                   f"WHERE d.status = 'ready' AND ({pair_clauses})" + scoped +
                   " ORDER BY t.doc_id, t.page, t.table_index LIMIT %s")
            p = []
            for d, pg in want:
                p.extend([d, pg])
            if doc_ids:
                p.append(list(doc_ids))
            p.append(k)
            try:
                hits += _rows(sql, p, "page_expansion")
            except Exception as e:
                print(f"[rag.table_search] page expansion failed ({e})")

    # Stable de-dup on the natural key; the first (highest-ranked) occurrence wins.
    seen, uniq = set(), []
    for h in hits:
        key = (h["doc_id"], h["page"], h["table_index"])
        if key not in seen:
            seen.add(key)
            uniq.append(h)

    # Document-scoped or corpus-wide fallback: if keyword search matched no table digests,
    # include available tables so tabular evidence reaches the model.
    if not uniq:
        if doc_ids:
            fallback_sql = (
                f"SELECT {_COLS}, 0.1 AS score "
                "FROM page_tables t JOIN documents d USING (doc_id) "
                "WHERE d.status = 'ready' AND t.doc_id = ANY(%s) "
                "ORDER BY t.page, t.table_index LIMIT %s"
            )
            p = [list(doc_ids), k]
        else:
            fallback_sql = (
                f"SELECT {_COLS}, 0.05 AS score "
                "FROM page_tables t JOIN documents d USING (doc_id) "
                "WHERE d.status = 'ready' "
                "ORDER BY t.doc_id DESC, t.page, t.table_index LIMIT %s"
            )
            p = [k]
        try:
            fallback_hits = _rows(fallback_sql, p, "doc_fallback")
            for h in fallback_hits:
                key = (h["doc_id"], h["page"], h["table_index"])
                if key not in seen:
                    seen.add(key)
                    uniq.append(h)
        except Exception as e:
            print(f"[rag.table_search] table fallback failed ({e})")

    return uniq[:k]


_STOPish = {"what", "which", "who", "whom", "whose", "when", "where", "why", "how",
            "the", "and", "for", "are", "was", "were", "does", "did", "has", "have",
            "this", "that", "with", "from", "into", "any", "all", "its", "his", "her",
            "there", "their", "about", "please", "tell", "show", "list", "give"}


def _or_tsquery(query: str) -> str:
    """`a | b | c` from the query's content words, sanitized for to_tsquery.

    to_tsquery takes operator syntax, not free text, so anything non-alphanumeric must go
    or it raises a syntax error. Mirrors prose_search._or_tsquery deliberately.
    """
    import re
    words = [w for w in re.findall(r"[A-Za-z0-9]+", (query or "").lower())
             if len(w) > 2 and w not in _STOPish]
    return " | ".join(dict.fromkeys(words))[:400]


def backfill(batch: int = 32) -> int:
    """Embed any page_tables digest still missing an embedding. Idempotent."""
    if not semantic_available():
        print("[rag.table_search] embedder unavailable — nothing to backfill")
        return 0
    done = 0
    with db.connect() as conn:
        while True:
            with conn.cursor() as cur:
                cur.execute("SELECT table_pk, coalesce(digest, html) FROM page_tables "
                            "WHERE embedding IS NULL ORDER BY table_pk LIMIT %s", (batch,))
                rows = cur.fetchall()
            if not rows:
                break
            vecs = embeddings.embed([(r[1] or "")[:1200] for r in rows])
            with conn.cursor() as cur:
                cur.executemany(
                    "UPDATE page_tables SET embedding = %s::vector WHERE table_pk = %s",
                    [(embeddings.to_pgvector(v), rid) for (rid, _), v in zip(rows, vecs)])
            conn.commit()
            done += len(rows)
            print(f"[rag.table_search] embedded {done} table digest(s)...")
    return done


if __name__ == "__main__":
    print(f"backfilled {backfill()} table digest(s)")
