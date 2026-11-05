"""
db.py — Postgres access for the RAG knowledge base.

Uses psycopg (v3). Imports are lazy so this module loads even when psycopg/Postgres
are not yet installed — callers guard on `available()`.

Env:
  RAG_DATABASE_URL       full DSN, e.g. postgresql://ocr:pass@localhost:5432/ocr_rag  (read/write, ingestion)
  RAG_DATABASE_URL_RO    read-only DSN for generated SQL (SELECT-only role); falls back to RAG_DATABASE_URL
"""

import os
from contextlib import contextmanager
from typing import Any, Iterable, List, Optional, Sequence, Tuple

# Default to the local pgvector container's DSN when the env vars are unset. The
# canonical launcher (run_project.sh) exports these, but the backend has been started
# WITHOUT them (e.g. by an external tool: `RAG_LLM_URL=... python server_...py`), which
# silently made available() False and emptied EVERY RAG surface — the Corpus tab, the
# chat, ingestion — even though all the data sat safe in Postgres. Falling back to the
# same DSN run_project.sh defaults to keeps RAG working no matter how the process was
# launched; an explicit env var still overrides.
_DEFAULT_URL = "postgresql://ocr:ocrpass@localhost:5432/ocr_rag"
_DEFAULT_URL_RO = "postgresql://ocr_ro:ocrpass_ro@localhost:5432/ocr_rag"
DATABASE_URL = os.getenv("RAG_DATABASE_URL") or _DEFAULT_URL
DATABASE_URL_RO = os.getenv("RAG_DATABASE_URL_RO") or os.getenv("RAG_DATABASE_URL") or _DEFAULT_URL_RO

_SCHEMA_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "schema.sql")


def available() -> bool:
    """True when a DSN is configured and psycopg importable — otherwise RAG is a no-op."""
    if not DATABASE_URL:
        return False
    try:
        import psycopg  # noqa: F401
        return True
    except Exception:
        return False


@contextmanager
def connect(readonly: bool = False):
    """Yield a psycopg connection. Caller controls the transaction."""
    import psycopg
    dsn = DATABASE_URL_RO if readonly else DATABASE_URL
    conn = psycopg.connect(dsn)
    try:
        yield conn
    finally:
        conn.close()


def init_schema() -> None:
    """Apply schema.sql (idempotent — all statements are IF NOT EXISTS / OR REPLACE)."""
    with open(_SCHEMA_PATH, encoding="utf-8") as f:
        ddl = f.read()
    with connect() as conn:
        with conn.cursor() as cur:
            cur.execute(ddl)
        conn.commit()


def run_readonly(sql: str, timeout_ms: int = 2000, row_limit: int = 500,
                 doc_ids: Optional[Sequence[int]] = None,
                 params: Optional[Sequence[Any]] = None,
                 ) -> Tuple[List[str], List[Sequence[Any]]]:
    """
    Execute a SELECT under strict guards (originally the text-to-SQL lane; also used
    by hand-written queries elsewhere in rag/*  and the server's /rag/* endpoints).

    Defense in depth: connect via the READ-ONLY DSN (SELECT-only role — the real
    barrier), plus a statement timeout and a hard row cap. Returns (columns, rows).
    Raises on any DB error so the caller can retry once, then fail visibly.

    doc_ids, if given, restricts every table read to those documents via Postgres
    Row-Level Security (see schema.sql: rag_doc_scope policies keyed on the
    `rag.doc_ids` GUC set below) — enforced by the database itself, so it holds even
    if the generated SQL forgets to filter by document. Omit/None == corpus-wide.

    params, if given, are bound with psycopg's normal %s placeholders — use this
    (never f-string/format the value into `sql`) for ANY value that didn't come from
    the LLM-generated-SQL trust boundary, e.g. a filename from a request: user-
    controlled strings must never be interpolated directly into the query text.
    """
    import psycopg
    with psycopg.connect(DATABASE_URL_RO) as conn:
        conn.read_only = True
        with conn.cursor() as cur:
            cur.execute(f"SET statement_timeout = {int(timeout_ms)}")
            if doc_ids:
                # SET does not accept bind parameters (it's a utility statement, parsed
                # before params are bound — passing %s here raises "syntax error at or
                # near $1"). set_config() is a regular function call and takes them fine.
                ids_csv = ",".join(str(int(d)) for d in doc_ids)
                cur.execute("SELECT set_config('rag.doc_ids', %s, false)", (ids_csv,))
            cur.execute(sql, params)
            cols = [d.name for d in (cur.description or [])]
            rows = cur.fetchmany(row_limit)
    return cols, list(rows)


def executemany(conn, sql: str, params: Iterable[Sequence[Any]]) -> None:
    with conn.cursor() as cur:
        cur.executemany(sql, list(params))
