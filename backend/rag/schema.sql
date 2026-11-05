-- RAG knowledge base for the OCR pipeline — flexible long/EAV layout.
-- One database serves both lanes: table_cells (exact SQL) + prose_chunks (pgvector).
-- Run once against the target DB:  psql "$RAG_DATABASE_URL" -f schema.sql

CREATE EXTENSION IF NOT EXISTS vector;

-- ── Documents ───────────────────────────────────────────────────────────────
CREATE TABLE IF NOT EXISTS documents (
    doc_id        SERIAL PRIMARY KEY,
    filename      TEXT NOT NULL,
    line_name     TEXT,                              -- e.g. 'UP67-04' (best-effort)
    form_type     TEXT,                              -- best-effort classification
    content_hash  TEXT UNIQUE NOT NULL,              -- SHA-256(pdf/image bytes): idempotency key
    status        TEXT NOT NULL DEFAULT 'ingesting', -- ingesting | ready | failed
    n_pages       INT,
    ingested_at   TIMESTAMPTZ NOT NULL DEFAULT now()
);

-- ── Table cells (long / EAV) — one row per extracted cell ───────────────────
-- Handles ANY ONGC form layout (static-correction, LINE DATA, …) with no
-- per-form migration. The wide side-by-side G-block presentation is unpivoted;
-- block_index/row_index/col_index preserve exact paper provenance.
--
-- STILL IN USE — do not delete as dead. The text-to-SQL lane that used to query this is
-- gone (answers now read page_tables HTML instead), but two consumers remain:
--   * /rag/document-pages — the Corpus tab editor, which re-renders these into the
--     paper's wide layout via table_extractor._blocks_to_wide_html
--   * /rag/corpus-tables  — deckworks' Maker, which needs the PARSED value_num per cell
--                           so it never re-transcribes a number out of text
-- page_tables is additive and preserves a different thing (layout: rowspan/colspan/<br>);
-- this preserves per-cell numeric parsing and trust flags. Neither replaces the other.
CREATE TABLE IF NOT EXISTS table_cells (
    cell_id       BIGSERIAL PRIMARY KEY,
    doc_id        INT NOT NULL REFERENCES documents(doc_id) ON DELETE CASCADE,
    page          INT NOT NULL,
    block_index   INT NOT NULL,        -- which of the G side-by-side bands
    row_index     INT NOT NULL,        -- position within the band
    col_index     INT NOT NULL,        -- 0..k-1 leaf column
    col_header    TEXT,                -- normalized leaf header, e.g. 'Shot m. secs'
    col_group     TEXT,                -- spanning title if any, e.g. 'Corrections'
    value_text    TEXT,                -- raw cell string
    value_num     NUMERIC,             -- parsed number for numeric columns, else NULL
    -- coarse trust heuristic (no real per-cell OCR confidence exists):
    --   ok | empty | type_mismatch  (extend with 'repaired' later)
    trust         TEXT NOT NULL DEFAULT 'ok',
    verified      BOOLEAN NOT NULL DEFAULT FALSE,
    crop_ref      TEXT                 -- optional key/path to the band crop image
);

CREATE INDEX IF NOT EXISTS idx_cells_doc_page   ON table_cells (doc_id, page);
CREATE INDEX IF NOT EXISTS idx_cells_col_header ON table_cells (col_header);
CREATE INDEX IF NOT EXISTS idx_cells_rowkey     ON table_cells (doc_id, page, block_index, row_index);
CREATE INDEX IF NOT EXISTS idx_cells_trust      ON table_cells (trust) WHERE trust <> 'ok';

-- ── Prose chunks (semantic lane) ────────────────────────────────────────────
CREATE TABLE IF NOT EXISTS prose_chunks (
    chunk_id      BIGSERIAL PRIMARY KEY,
    doc_id        INT NOT NULL REFERENCES documents(doc_id) ON DELETE CASCADE,
    page          INT NOT NULL,
    section       TEXT,                -- form field / heading the chunk came from
    content       TEXT NOT NULL,
    embedding     vector(1024)         -- BGE-M3 dim
);
CREATE INDEX IF NOT EXISTS idx_prose_embedding
    ON prose_chunks USING hnsw (embedding vector_cosine_ops);

-- ── Figures (visual-upgrade plan, Phase 2A) ─────────────────────────────────
-- Real embedded raster figures pulled from the source PDF at ingest time
-- (rag/figures.py), so deckworks' Maker can cite an actual document figure
-- instead of fabricating or going without one. image_ref is a content-addressed
-- key into figure_store/{image_ref}.png (no blob in the DB itself).
CREATE TABLE IF NOT EXISTS figures (
    figure_id     BIGSERIAL PRIMARY KEY,
    doc_id        INT NOT NULL REFERENCES documents(doc_id) ON DELETE CASCADE,
    page          INT NOT NULL,
    bbox          NUMERIC[4],          -- [x0, y0, x1, y1] in PDF points
    caption       TEXT,                -- nearby "Fig 4.2 — ..." caption, if matched
    context       TEXT,                -- nearest text block, for BM25 indexing/gist
    image_ref     TEXT NOT NULL,       -- key into figure_store/{image_ref}.png
    phash         TEXT,                -- perceptual hash (within-doc dedup already applied)
    width_px      INT,
    height_px     INT
);

CREATE INDEX IF NOT EXISTS idx_figures_doc_page ON figures (doc_id, page);

-- ── Page tables (the HTML-in-context lane) ──────────────────────────────────
-- The raw semantic HTML of ONE detected table, exactly as OCR produced it: rowspan,
-- colspan and <br> multi-values preserved, no presentation attributes. This is what the
-- narration model reads. It replaces the retired text-to-SQL lane, which asked the model
-- to write a SELECT over the flattened table_cells store below.
--
-- Deliberately NOT re-rendered from table_cells. That store cannot represent this format:
-- body rowspans are filtered out in _header_spans before `spans` is built, `blocks` is a
-- dense grid by construction, and <br> multi-values never reach it. And table_extractor's
-- _blocks_to_wide_html carries ~77 chars of inline CSS per <td> — ~13,800 chars for a
-- table that is 2,604 chars raw, a 5x regression on a budget measured in tokens.
--
-- `digest` is what gets embedded and full-text searched, never `html`. Markup tokens
-- swamp a bge-m3 vector, and HTML runs ~1.7 chars/token (vs ~3.9 for prose) so a large
-- table would blow past the embedder's useful window. `html` is retrieved, never searched.
CREATE TABLE IF NOT EXISTS page_tables (
    table_pk      BIGSERIAL PRIMARY KEY,
    doc_id        INT NOT NULL REFERENCES documents(doc_id) ON DELETE CASCADE,
    page          INT NOT NULL,
    table_index   INT NOT NULL,        -- 0-based, reading order within the page
    html          TEXT NOT NULL,       -- raw <table>...</table>, unstyled
    digest        TEXT,                -- short NL description: what is embedded/searched
    caption       TEXT,                -- nearby heading/caption line, if one was found
    n_rows        INT,
    n_cols        INT,
    est_tokens    INT,                 -- precomputed so the budget allocator needn't tokenize
    -- FALSE when the stored HTML has unbalanced tags, i.e. OCR stopped mid-table. Kept
    -- and flagged rather than dropped: the rows are permanently lost (only re-OCR
    -- recovers them), so a visible warning beats a silent undercount.
    well_formed   BOOLEAN NOT NULL DEFAULT TRUE,
    source        TEXT,                -- 'tatr' | 'prose_html' | 'backfill_prose_html'
    embedding     vector(1024),        -- bge-m3 over `digest`; NULL until backfilled
    UNIQUE (doc_id, page, table_index)
);

CREATE INDEX IF NOT EXISTS idx_page_tables_doc_page ON page_tables (doc_id, page);
CREATE INDEX IF NOT EXISTS idx_page_tables_embedding
    ON page_tables USING hnsw (embedding vector_cosine_ops);

-- ── Convenience view: cells collapsed back to one logical row ───────────────
-- Row-coherent questions ("Shot & Receiver for surface point 564") stay simple:
-- filter/pivot this view by col_header instead of self-joining table_cells.
-- (A generic long view; the text-to-SQL prompt shows how to pivot with FILTER.)
CREATE OR REPLACE VIEW cell_rows AS
SELECT c.doc_id, d.filename, d.line_name, d.status,
       c.page, c.block_index, c.row_index,
       c.col_header, c.col_group, c.value_text, c.value_num,
       c.trust, c.verified
FROM table_cells c
JOIN documents d USING (doc_id);

-- security_invoker so the view runs with the CALLER's privileges (and RLS), not the
-- view owner's — required for the ocr_ro-scoped policies below to apply through it.
ALTER VIEW cell_rows SET (security_invoker = true);

-- ── Row-Level Security: deterministic chat-context document scoping ─────────
-- The text-to-SQL model writes arbitrary SELECTs; we cannot rely on it to always add
-- a doc filter (it may forget, or a user's question may not mention documents at all).
-- Instead, the app sets a per-connection GUC (rag.doc_ids, a comma-separated int list)
-- before running the generated query, and RLS transparently restricts every table read
-- to those doc_ids. Empty/unset GUC == no restriction (corpus-wide mode, the default).
--
-- This only affects the non-superuser ocr_ro role (RAG_DATABASE_URL_RO). The owning
-- role (ocr, used for ingestion + server-side quality counts) is a superuser and always
-- bypasses RLS, so ingestion/backfill/synthesize's quality-warning queries are unaffected.
ALTER TABLE documents    ENABLE ROW LEVEL SECURITY;
ALTER TABLE table_cells  ENABLE ROW LEVEL SECURITY;
ALTER TABLE prose_chunks ENABLE ROW LEVEL SECURITY;
ALTER TABLE figures      ENABLE ROW LEVEL SECURITY;
ALTER TABLE page_tables  ENABLE ROW LEVEL SECURITY;

DROP POLICY IF EXISTS rag_doc_scope ON documents;
CREATE POLICY rag_doc_scope ON documents FOR SELECT USING (
    coalesce(current_setting('rag.doc_ids', true), '') = ''
    OR doc_id = ANY (string_to_array(current_setting('rag.doc_ids', true), ',')::int[])
);

DROP POLICY IF EXISTS rag_doc_scope ON table_cells;
CREATE POLICY rag_doc_scope ON table_cells FOR SELECT USING (
    coalesce(current_setting('rag.doc_ids', true), '') = ''
    OR doc_id = ANY (string_to_array(current_setting('rag.doc_ids', true), ',')::int[])
);

DROP POLICY IF EXISTS rag_doc_scope ON prose_chunks;
CREATE POLICY rag_doc_scope ON prose_chunks FOR SELECT USING (
    coalesce(current_setting('rag.doc_ids', true), '') = ''
    OR doc_id = ANY (string_to_array(current_setting('rag.doc_ids', true), ',')::int[])
);

DROP POLICY IF EXISTS rag_doc_scope ON figures;
CREATE POLICY rag_doc_scope ON figures FOR SELECT USING (
    coalesce(current_setting('rag.doc_ids', true), '') = ''
    OR doc_id = ANY (string_to_array(current_setting('rag.doc_ids', true), ',')::int[])
);

-- page_tables applies its doc_ids filter directly in table_search.py (that lane owns its
-- own SQL, like prose_search). This policy is defence in depth for any other reader.
DROP POLICY IF EXISTS rag_doc_scope ON page_tables;
CREATE POLICY rag_doc_scope ON page_tables FOR SELECT USING (
    coalesce(current_setting('rag.doc_ids', true), '') = ''
    OR doc_id = ANY (string_to_array(current_setting('rag.doc_ids', true), ',')::int[])
);

-- Guarded: ocr_ro is provisioned out-of-band (see README/run_project.sh), so schema.sql
-- stays idempotent/runnable even before that role exists (e.g. first-time bootstrap).
DO $$
BEGIN
    IF EXISTS (SELECT 1 FROM pg_roles WHERE rolname = 'ocr_ro') THEN
        GRANT SELECT ON documents, table_cells, prose_chunks, figures, page_tables, cell_rows TO ocr_ro;
    END IF;
END
$$;
