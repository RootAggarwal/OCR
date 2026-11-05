"""
ingest.py — load one OCR'd document into the RAG knowledge base.

Called from the WebSocket / upload handlers on document completion, with the per-page
RAG payloads the OCR pipeline already produced (masked-pass prose + structured band
rows). Guarantees:
  * Idempotency  — SHA-256(bytes) is the content_hash; re-uploading REPLACES (delete old
                   doc's cells+chunks via ON DELETE CASCADE, then insert) so aggregates
                   never silently double.
  * Atomicity    — insert under status='ingesting', flip to 'ready', commit; a document
                   is fully queryable or invisible, never half.
  * Never fatal  — any failure raises to the caller, which logs and drops it; OCR delivery
                   to the user is unaffected.

Prose embeddings are left NULL here; Phase 3 (BGE-M3) backfills them.
"""

import hashlib
import os
import re
from typing import Any, Dict, List, Optional

from . import db

# Content-addressed store for original upload bytes, keyed by the SAME content_hash
# already computed below as the idempotency key — no separate file-path column needed
# on `documents`; the read side (server_with_task_management.py's /rag/document-pages
# and /rag/document-page-image) re-derives the same path from a doc's stored hash.
# Lets the Corpus tab render a real original-page image (via local_proc/renderpdf.py)
# for any document ingested after this was added; earlier documents show no preview
# until re-ingested (see PLAN: no backfill from the ambiguous leftover temp_*.pdf files).
# Now lives under OCR_DATA_DIR (see ocr_paths.py) so the repo holds no user data.
# Re-exported from this module under its original name so existing importers — the
# server's /rag/document-pages and /rag/document-page-image, and backfill_pdf_store.py —
# keep working unchanged.
from ocr_paths import PDF_STORE_DIR  # noqa: F401  (re-export)


# Extensions the store keeps verbatim (so a scanned-image upload isn't mislabeled as a
# PDF, which pdftoppm can't render); anything else defaults to .pdf, the overwhelming
# common case and the correct one for every call site that predates this parameter.
_IMAGE_EXTS = {"png", "jpg", "jpeg", "tif", "tiff"}


def pdf_store_path(content_hash: str, filename: str = "") -> str:
    ext = (filename.rsplit(".", 1)[-1].lower() if "." in filename else "")
    ext = ext if ext in _IMAGE_EXTS else "pdf"
    return os.path.join(PDF_STORE_DIR, f"{content_hash}.{ext}")


def _line_name(filename: str) -> Optional[str]:
    """Best-effort survey/line id from the filename, e.g. 'UP67-04-4-7.pdf' -> 'UP67-04'."""
    m = re.search(r"([A-Z]{1,3}\d{1,3}-\d{1,3})", filename or "", re.IGNORECASE)
    return m.group(1).upper() if m else None


def _chunk_prose(prose: str) -> List[str]:
    """Split page prose into section-ish chunks on blank lines. Deliberately simple;
    Phase 3 can refine boundaries. Drops empties and over-long fragments are kept whole
    (BGE-M3 handles 8k)."""
    if not prose or not prose.strip():
        return []
    parts = re.split(r"\n\s*\n", prose.strip())
    return [p.strip() for p in parts if p.strip()]


def ingest_document(doc_bytes: bytes, filename: str,
                    doc_rag_pages: Dict[int, Optional[Dict[str, Any]]]) -> Optional[int]:
    """
    Ingest a whole document. `doc_rag_pages` maps page_num -> {"prose", "structured"} | None
    (structured is the list of {headers,blocks,k,spans,box} from extract_page_html).
    Returns the new doc_id, or None if RAG is not configured (no-op).
    """
    if not db.available():
        return None

    import table_extractor  # for _blocks_to_long_rows (same source of truth as the HTML path)

    # Cross-page header consistency: every page of a multi-page form shares the same
    # column headers, but the per-page VLM header read occasionally hallucinates one
    # page's (e.g. reading a form field label as a column title). Majority-vote the
    # headers/spans across same-shape pages BEFORE unpivoting to cells, so the RAG
    # lane and the corpus editor see one consistent header set. Best-effort/non-fatal.
    try:
        table_extractor.reconcile_doc_headers(doc_rag_pages)
    except Exception as e:
        print(f"[rag.ingest] header reconciliation skipped ({e})")

    content_hash = hashlib.sha256(doc_bytes).hexdigest()
    pages = sorted(p for p in doc_rag_pages if p is not None)
    n_pages = len(pages)

    # Best-effort: retain the original bytes so the Corpus tab can render real page
    # previews later. Never fatal — a disk/IO failure here must not drop the ingest.
    try:
        os.makedirs(PDF_STORE_DIR, exist_ok=True)
        with open(pdf_store_path(content_hash, filename), "wb") as f:
            f.write(doc_bytes)
    except Exception as e:
        print(f"[rag.ingest] pdf_store write skipped ({e}); preview will be unavailable")

    # Best-effort: pull real embedded figures (photos, log panels, cross-sections) out
    # of the PDF for deckworks' Maker to cite later (visual-upgrade plan, Phase 2A).
    # Never fatal, and never runs on a non-PDF store (extract_figures itself no-ops if
    # pymupdf can't open the bytes as a PDF, e.g. a scanned single-image upload).
    extracted_figures: List = []
    try:
        from . import figures as _figures
        extracted_figures = _figures.extract_figures(doc_bytes, content_hash)
    except Exception as e:
        print(f"[rag.ingest] figure extraction skipped ({e})")

    with db.connect() as conn:
        try:
            with conn.cursor() as cur:
                # Idempotent replace: drop any prior doc with the same bytes (CASCADE
                # removes its cells + chunks) in the SAME transaction as the re-insert.
                cur.execute("DELETE FROM documents WHERE content_hash = %s", (content_hash,))
                cur.execute(
                    "INSERT INTO documents (filename, line_name, content_hash, status, n_pages) "
                    "VALUES (%s, %s, %s, 'ingesting', %s) RETURNING doc_id",
                    (filename, _line_name(filename), content_hash, n_pages),
                )
                doc_id = cur.fetchone()[0]

                cell_rows: List[tuple] = []
                prose_rows: List[tuple] = []
                table_rows: List[tuple] = []
                for page in pages:
                    payload = doc_rag_pages.get(page) or {}
                    # Raw per-table HTML for the tables lane (page_tables). This preserves
                    # what table_cells structurally cannot: body rowspan, colspan and <br>
                    # multi-values. The two are complementary, not redundant — cells keep
                    # per-cell numeric parsing, this keeps layout.
                    for ti, t in enumerate(payload.get("page_tables") or []):
                        table_rows.append((
                            doc_id, page, ti, t["html"], t.get("digest"), t.get("caption"),
                            t.get("n_rows"), t.get("n_cols"), t.get("est_tokens"),
                            bool(t.get("well_formed", True)), t.get("source"),
                        ))
                    for structured in (payload.get("structured") or []):
                        if not structured:
                            continue
                        crop_ref = None  # (band-crop persistence is a Phase 5 add)
                        for c in table_extractor._blocks_to_long_rows(structured, page):
                            cell_rows.append((
                                doc_id, c["page"], c["block_index"], c["row_index"], c["col_index"],
                                c["col_header"], c["col_group"], c["value_text"], c["value_num"],
                                c["trust"], crop_ref,
                            ))
                    for chunk in _chunk_prose(payload.get("prose") or ""):
                        prose_rows.append((doc_id, page, None, chunk))

                if cell_rows:
                    cur.executemany(
                        "INSERT INTO table_cells "
                        "(doc_id,page,block_index,row_index,col_index,col_header,col_group,"
                        " value_text,value_num,trust,crop_ref) "
                        "VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)", cell_rows)
                if prose_rows:
                    # Embed inline so the doc is semantically searchable the moment it's
                    # 'ready'. Best-effort: if the embedder is unavailable we insert NULL
                    # embeddings and prose_search.backfill() picks them up later.
                    vectors = None
                    try:
                        from . import embeddings as _emb
                        if _emb.available():
                            vectors = [_emb.to_pgvector(v)
                                       for v in _emb.embed([r[3] for r in prose_rows])]
                    except Exception as e:
                        print(f"[rag.ingest] embedding skipped ({e}); will backfill later")
                    if vectors:
                        cur.executemany(
                            "INSERT INTO prose_chunks (doc_id,page,section,content,embedding) "
                            "VALUES (%s,%s,%s,%s,%s::vector)",
                            [r + (v,) for r, v in zip(prose_rows, vectors)])
                    else:
                        cur.executemany(
                            "INSERT INTO prose_chunks (doc_id,page,section,content) "
                            "VALUES (%s,%s,%s,%s)", prose_rows)

                if table_rows:
                    # Embed the DIGEST, never the html — markup tokens dominate the vector
                    # and HTML runs ~1.7 chars/token, well past the embedder's useful
                    # window on a large table. Same best-effort contract as prose above:
                    # a missing embedder means NULL vectors and a later
                    # table_search.backfill(), not a failed ingest.
                    tvectors = None
                    try:
                        from . import embeddings as _emb
                        if _emb.available():
                            tvectors = [_emb.to_pgvector(v)
                                        for v in _emb.embed([(r[4] or r[3])[:1200]
                                                             for r in table_rows])]
                    except Exception as e:
                        print(f"[rag.ingest] table embedding skipped ({e}); will backfill later")
                    cols = ("doc_id,page,table_index,html,digest,caption,n_rows,n_cols,"
                            "est_tokens,well_formed,source")
                    if tvectors:
                        cur.executemany(
                            f"INSERT INTO page_tables ({cols},embedding) "
                            "VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s::vector)",
                            [r + (v,) for r, v in zip(table_rows, tvectors)])
                    else:
                        cur.executemany(
                            f"INSERT INTO page_tables ({cols}) "
                            "VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)", table_rows)

                figure_rows: List[tuple] = []
                for fig in extracted_figures:
                    try:
                        from . import figures as _figures
                        _figures.persist_figure(fig)
                    except Exception as e:
                        print(f"[rag.ingest] figure {fig.image_ref} not persisted ({e}); skipped")
                        continue
                    figure_rows.append((
                        doc_id, fig.page, list(fig.bbox), fig.caption, fig.context,
                        fig.image_ref, fig.phash, fig.width_px, fig.height_px,
                    ))
                if figure_rows:
                    cur.executemany(
                        "INSERT INTO figures "
                        "(doc_id,page,bbox,caption,context,image_ref,phash,width_px,height_px) "
                        "VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s)", figure_rows)

                # Integrity gate: a multi-page document that produced NOTHING at all
                # (no prose, no cells, no figures) is an extraction/OCR failure — most
                # often an empty per-page VLM read (e.g. a vLLM outage mid-ingest), which
                # previously flipped to 'ready' silently and left an invisible empty
                # document in the corpus (e.g. WCR Kalol-1: 81 pages, 0 output). Record it
                # as 'failed' instead so it is visible and re-ingestable, never a silent
                # empty. A genuinely content-free 1-page upload is not penalized (n>1).
                empty = not cell_rows and not prose_rows and not figure_rows
                if empty and n_pages > 1:
                    cur.execute("UPDATE documents SET status='failed' WHERE doc_id=%s", (doc_id,))
                    conn.commit()
                    print(f"[rag.ingest] {filename}: doc_id={doc_id} produced NOTHING across "
                          f"{n_pages} pages — marked 'failed' (extraction/OCR failure), not 'ready'")
                    return doc_id
                cur.execute("UPDATE documents SET status='ready' WHERE doc_id=%s", (doc_id,))
            conn.commit()
            print(f"[rag.ingest] {filename}: doc_id={doc_id} "
                  f"cells={len(cell_rows)} prose_chunks={len(prose_rows)} "
                  f"tables={len(table_rows)} figures={len(figure_rows)} pages={n_pages}")
            return doc_id
        except Exception:
            conn.rollback()
            raise
