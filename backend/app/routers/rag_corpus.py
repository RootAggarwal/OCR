"""
app/routers/rag_corpus.py — RAG knowledge-base proxy endpoints.

All endpoints here are thin proxies over the rag/ package.  They never contain
business logic — that lives in rag/synthesize.py, rag/prose_search.py, etc.

Endpoints
---------
GET  /rag/documents          — list ingested documents (for the "+" picker)
GET  /rag/document-pages     — reconstruct per-page content from the DB
GET  /rag/document-page-image — render one PDF page as PNG (with disk cache)
GET  /rag/corpus-chunks      — flat text chunks for the Maker tab
GET  /rag/corpus-tables      — structured numeric table export for the Maker
GET  /rag/corpus-figures     — extracted figure metadata for the Maker
GET  /rag/figure-image       — serve a stored figure PNG by content ref
POST /rag-chat/              — RAG-grounded conversational Q&A
POST /rag/prose              — top-k prose retrieval (semantic)
POST /rag/tables             — top-k table retrieval (semantic)
"""

import asyncio
import base64
import functools
import os
import re
import traceback
from datetime import datetime
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, Form
from fastapi.responses import Response

from app.config import UNIT_HINTS
from app.utils.responses import create_error_response
from ocr_paths import PREVIEW_CACHE_DIR

router = APIRouter(tags=["RAG / Corpus"])


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------

def _preview_cache_path(content_hash: str, page: int) -> str:
    """
    Disk-cache path for a rendered page preview PNG, keyed by (content_hash, page).

    A PDF page's pixels never change (content-hash-addressed, immutable), so a
    page is rendered via pdftoppm at most once total.  Lives under
    <OCR_DATA_DIR>/preview_cache — pure cache, safe to delete wholesale since it
    regenerates on next view.
    """
    return os.path.join(PREVIEW_CACHE_DIR, f"{content_hash}_{page}.png")


def _parse_doc_ids(raw: str) -> Optional[List[int]]:
    """
    Parse the comma-separated doc_ids form field from the chat UI's "+" picker
    (e.g. "2,34").

    Returns None (== corpus-wide) for empty/absent input.
    Silently drops any non-integer token rather than failing the whole request.
    """
    if not (raw or "").strip():
        return None
    ids = [int(tok) for tok in raw.split(",") if tok.strip().isdigit()]
    return ids or None


def _guess_unit(header: str) -> str:
    """
    Guess a unit string from a column header keyword.

    Used by /rag/corpus-tables to satisfy deckworks' Quantity.unit (min_length=1)
    with something meaningful rather than a bare placeholder.  The NUMBER is what
    matters for correctness; this is display metadata only.
    """
    lowered = (header or "").lower()
    for needle, unit in UNIT_HINTS:
        if needle in lowered:
            return unit
    return "units"


# ---------------------------------------------------------------------------
# Document listing & page reconstruction
# ---------------------------------------------------------------------------

@router.get("/rag/documents")
async def rag_documents():
    """
    List documents available for the chat UI's "+" document picker.

    Returns one entry per ingested, ready document (newest first).
    doc_id is the stable key to send back in doc_ids.
    """
    try:
        from rag import db

        if not db.available():
            return {"documents": []}
        _, rows = db.run_readonly(
            "SELECT doc_id, filename, line_name, n_pages, ingested_at FROM documents "
            "WHERE status='ready' ORDER BY ingested_at DESC"
        )
        return {
            "documents": [
                {
                    "doc_id": r[0],
                    "filename": r[1],
                    "line_name": r[2],
                    "n_pages": r[3],
                    "ingested_at": r[4].isoformat() if r[4] else None,
                }
                for r in rows
            ]
        }
    except Exception as e:
        return create_error_response(f"rag/documents failed: {e}")


@router.get("/rag/document-pages")
async def rag_document_pages(doc_id: str):
    """
    Reconstruct a document's per-page content (tables as HTML + prose) from the
    RAG knowledge base for the Corpus tab.

    `doc_id` is the document's FILENAME, not the internal integer id.
    Returns pages shaped like the OCR pipeline's own results so the existing
    PageResult component renders them unmodified.
    """
    try:
        from rag import db
        import table_extractor

        if not db.available():
            return {"pages": []}

        _, doc_rows = db.run_readonly(
            "SELECT doc_id, content_hash FROM documents "
            "WHERE filename = %s AND status = 'ready' "
            "ORDER BY ingested_at DESC LIMIT 1",
            params=(doc_id,),
        )
        if not doc_rows:
            return {"pages": []}

        internal_doc_id, content_hash = doc_rows[0]

        from rag.ingest import pdf_store_path

        has_preview = os.path.exists(pdf_store_path(content_hash, doc_id))

        # row_limit is above the default 500 (a text-to-SQL safety cap).
        # A single document's table_cells can exceed 500 rows.
        _, cell_rows = db.run_readonly(
            "SELECT page, block_index, row_index, col_index, col_header, col_group, value_text "
            "FROM table_cells WHERE doc_id = %s "
            "ORDER BY page, block_index, row_index, col_index",
            params=(internal_doc_id,),
            row_limit=100000,
        )
        _, prose_rows = db.run_readonly(
            "SELECT page, content FROM prose_chunks WHERE doc_id = %s "
            "ORDER BY page, chunk_id",
            params=(internal_doc_id,),
            row_limit=100000,
        )

        # Reconstruct each page's table in wide format.
        page_data: Dict[int, Dict[str, Any]] = {}
        for page, block_index, row_index, col_index, col_header, col_group, value_text in cell_rows:
            pd = page_data.setdefault(page, {"cols": {}, "groups": {}, "blocks": {}})
            pd["cols"][col_index] = col_header or ""
            pd["groups"][col_index] = col_group or ""
            pd["blocks"].setdefault(block_index, {}).setdefault(row_index, {})[
                col_index
            ] = value_text or ""

        tables_by_page: Dict[int, List[str]] = {}
        for page, pd in sorted(page_data.items()):
            if not pd["cols"]:
                continue
            k = max(pd["cols"]) + 1
            headers = [pd["cols"].get(i, "") for i in range(k)]
            col_group = [pd["groups"].get(i, "") for i in range(k)]
            spans = table_extractor._col_groups_to_spans(col_group)
            blocks = []
            for bi in sorted(pd["blocks"]):
                brows = pd["blocks"][bi]
                blocks.append(
                    [[brows[r].get(i, "") for i in range(k)] for r in sorted(brows)]
                )
            if not any(any(row) for blk in blocks for row in blk):
                continue
            tables_by_page.setdefault(page, []).append(
                table_extractor._blocks_to_wide_html(headers, blocks, k, spans)
            )

        prose_by_page: Dict[int, List[str]] = {}
        for page, content in prose_rows:
            if content:
                prose_by_page.setdefault(page, []).append(content)

        all_pages = sorted(set(tables_by_page) | set(prose_by_page))
        pages = []
        for page in all_pages:
            prose_html = "".join(
                f"<p>{table_extractor._esc(p)}</p>"
                for p in prose_by_page.get(page, [])
            )
            tables_html = "".join(tables_by_page.get(page, []))
            pages.append(
                {
                    "fileName": doc_id,
                    "pageNumber": page,
                    "extractedText": prose_html + tables_html,
                    "has_table": bool(tables_by_page.get(page)),
                    "has_preview": has_preview,
                }
            )
        return {"pages": pages}

    except Exception as e:
        return create_error_response(f"rag/document-pages failed: {e}")


@router.get("/rag/document-page-image")
async def rag_document_page_image(doc_id: str, page: int = 1):
    """
    Render one page of an uploaded PDF as a PNG for the Corpus tab's "Original" pane.

    Performance notes:
    - render_pdf_to_base64png shells to pdftoppm synchronously; running it
      directly in an async def blocks the event loop.  Moved to asyncio.to_thread
      so concurrent page requests render in parallel.
    - Results are disk-cached by (content_hash, page) — a page is rendered at
      most once; the browser skips the round-trip entirely on repeat views via
      Cache-Control.
    """
    try:
        from rag import db
        from rag.ingest import pdf_store_path
        from local_proc.renderpdf import render_pdf_to_base64png

        if not db.available():
            return create_error_response("rag not configured", status_code=503)

        _, doc_rows = db.run_readonly(
            "SELECT content_hash FROM documents "
            "WHERE filename = %s AND status = 'ready' "
            "ORDER BY ingested_at DESC LIMIT 1",
            params=(doc_id,),
        )
        if not doc_rows:
            return create_error_response("document not found", status_code=404)

        content_hash = doc_rows[0][0]
        store_path = pdf_store_path(content_hash, doc_id)
        if not os.path.exists(store_path):
            return create_error_response(
                "no stored original for this document", status_code=404
            )

        # Non-PDF uploads (scanned images) are served directly.
        ext = store_path.rsplit(".", 1)[-1].lower()
        if ext != "pdf":
            with open(store_path, "rb") as f:
                file_bytes = f.read()
            media_type = (
                "image/jpeg" if ext in ("jpg", "jpeg") else f"image/{ext}"
            )
            return Response(
                content=file_bytes,
                media_type=media_type,
                headers={"Cache-Control": "public, max-age=31536000, immutable"},
            )

        # Serve from disk cache if available.
        cache_path = _preview_cache_path(content_hash, page)
        if os.path.exists(cache_path):
            with open(cache_path, "rb") as f:
                png_bytes = f.read()
        else:
            with open(store_path, "rb") as f:
                pdf_bytes = f.read()
            png_b64 = await asyncio.to_thread(
                render_pdf_to_base64png, pdf_bytes, page_number=page, resolution=1288
            )
            png_bytes = base64.b64decode(png_b64)
            os.makedirs(os.path.dirname(cache_path), exist_ok=True)
            with open(cache_path, "wb") as f:
                f.write(png_bytes)

        return Response(
            content=png_bytes,
            media_type="image/png",
            headers={"Cache-Control": "public, max-age=31536000, immutable"},
        )

    except Exception as e:
        return create_error_response(f"rag/document-page-image failed: {e}")


# ---------------------------------------------------------------------------
# Corpus exports (for the Maker / deckworks tab)
# ---------------------------------------------------------------------------

@router.get("/rag/corpus-chunks")
async def rag_corpus_chunks(doc_ids: str = ""):
    """
    Export the RAG corpus as flat text chunks shaped for deckworks' Chunk model.

    doc_ids: optional comma-separated internal doc_id list; empty = corpus-wide.
    """
    try:
        from rag import db

        if not db.available():
            return {"chunks": []}

        parsed_ids = _parse_doc_ids(doc_ids)

        # row_limit above the default 500: this is a bulk corpus-wide export.
        _, prose_rows = db.run_readonly(
            "SELECT d.filename, p.page, p.content, "
            "       row_number() OVER (PARTITION BY p.doc_id ORDER BY p.page, p.chunk_id) - 1 "
            "FROM prose_chunks p JOIN documents d USING (doc_id) "
            "WHERE d.status='ready' AND p.content IS NOT NULL AND p.content <> ''",
            doc_ids=parsed_ids,
            row_limit=100000,
        )
        chunks = [
            {
                "doc_id": filename,
                "page": int(page),
                "text": content,
                "kind": "body",
                "ordinal": int(ordinal),
                "ingest_version": 1,
            }
            for filename, page, content, ordinal in prose_rows
        ]

        # Tables: rendered as "Header: value, ..." lines — one chunk per
        # (doc, page, block) — so tabular facts stay usable for slide generation.
        _, cell_rows = db.run_readonly(
            "SELECT d.filename, c.page, c.block_index, c.row_index, c.col_header, c.value_text "
            "FROM table_cells c JOIN documents d USING (doc_id) "
            "WHERE d.status='ready' AND c.value_text IS NOT NULL AND c.value_text <> '' "
            "ORDER BY d.filename, c.page, c.block_index, c.row_index, c.col_index",
            doc_ids=parsed_ids,
            row_limit=100000,
        )
        tables: Dict[tuple, Any] = {}
        for filename, page, block_index, row_index, col_header, value_text in cell_rows:
            key = (filename, page, block_index)
            tables.setdefault(key, {}).setdefault(row_index, []).append(
                (col_header or "", value_text)
            )

        next_ordinal: Dict[str, int] = {}
        for c in chunks:
            next_ordinal[c["doc_id"]] = max(
                next_ordinal.get(c["doc_id"], 0), c["ordinal"] + 1
            )

        for (filename, page, _block_index), rows in tables.items():
            lines = [
                ", ".join(f"{h}: {v}" for h, v in cells if h)
                for _, cells in sorted(rows.items())
            ]
            lines = [line for line in lines if line]
            if not lines:
                continue
            ordinal = next_ordinal.get(filename, 0)
            next_ordinal[filename] = ordinal + 1
            chunks.append(
                {
                    "doc_id": filename,
                    "page": int(page),
                    "text": "\n".join(lines),
                    "kind": "body",
                    "ordinal": ordinal,
                    "ingest_version": 1,
                }
            )

        return {"chunks": chunks}

    except Exception as e:
        return create_error_response(f"rag/corpus-chunks failed: {e}")


@router.get("/rag/corpus-tables")
async def rag_corpus_tables(doc_ids: str = ""):
    """
    Structured table export for the Maker's numeric pipeline.

    Keeps each cell's parsed value_num so deckworks can build real numeric
    slots without the LLM re-parsing or transcribing a number from text.

    doc_ids: optional comma-separated internal doc_id list; empty = corpus-wide.
    """
    try:
        from rag import db

        if not db.available():
            return {"tables": []}

        parsed_ids = _parse_doc_ids(doc_ids)

        _, rows = db.run_readonly(
            "SELECT d.filename, c.page, c.block_index, c.col_group, c.row_index, "
            "       c.col_index, c.col_header, c.value_text, c.value_num "
            "FROM table_cells c JOIN documents d USING (doc_id) "
            "WHERE d.status='ready' "
            "ORDER BY d.filename, c.page, c.block_index, c.row_index, c.col_index",
            doc_ids=parsed_ids,
            row_limit=100000,
        )

        tables: Dict[tuple, Dict[str, Any]] = {}
        for (
            filename, page, block_index, col_group,
            row_index, col_index, col_header, value_text, value_num,
        ) in rows:
            key = (filename, page, block_index)
            t = tables.setdefault(key, {"headers": {}, "groups": set(), "rows": {}})
            t["headers"][col_index] = col_header or ""
            if col_group:
                t["groups"].add(col_group)
            t["rows"].setdefault(row_index, {})[col_index] = {
                "text": value_text or "",
                "num": float(value_num) if value_num is not None else None,
            }

        result = []
        for (filename, page, block_index), t in tables.items():
            n_cols = max(t["headers"]) + 1 if t["headers"] else 0
            if n_cols == 0:
                continue
            headers = [t["headers"].get(i, "") for i in range(n_cols)]
            row_list = [t["rows"][r] for r in sorted(t["rows"])]
            if not row_list:
                continue
            numeric_flags = [
                any(row.get(col, {}).get("num") is not None for row in row_list)
                for col in range(n_cols)
            ]
            rows_out = [
                [row.get(col, {"text": "", "num": None}) for col in range(n_cols)]
                for row in row_list
            ]
            table_id = f"p{page}b{block_index}"
            title = " / ".join(sorted(t["groups"]) + [filename])
            result.append(
                {
                    "doc_id": filename,
                    "page": page,
                    "table_id": table_id,
                    "title": title,
                    "columns": [
                        {
                            "header": h,
                            "numeric": numeric_flags[i],
                            "unit": _guess_unit(h),
                        }
                        for i, h in enumerate(headers)
                    ],
                    "rows": rows_out,
                    "provenance": {
                        "doc_id": filename,
                        "page": page,
                        "table_id": table_id,
                        "tier": "B",
                    },
                }
            )

        return {"tables": result}

    except Exception as e:
        return create_error_response(f"rag/corpus-tables failed: {e}")


@router.get("/rag/corpus-figures")
async def rag_corpus_figures(doc_ids: str = ""):
    """
    Real-figure export for the Maker's media pipeline.

    Each entry's caption/context is folded into deckworks' BM25 corpus so a
    slide can retrieve and cite an actual document figure instead of going
    without one.  `image_ref` is fetched via /rag/figure-image.

    doc_ids: optional comma-separated internal doc_id list; empty = corpus-wide.
    """
    try:
        from rag import db

        if not db.available():
            return {"figures": []}

        parsed_ids = _parse_doc_ids(doc_ids)

        _, rows = db.run_readonly(
            "SELECT d.filename, f.page, f.caption, f.context, f.image_ref, "
            "       f.width_px, f.height_px "
            "FROM figures f JOIN documents d USING (doc_id) "
            "WHERE d.status='ready' "
            "ORDER BY d.filename, f.page",
            doc_ids=parsed_ids,
            row_limit=100000,
        )

        return {
            "figures": [
                {
                    "doc_id": filename,
                    "page": page,
                    "caption": caption,
                    "context": context,
                    "image_ref": image_ref,
                    "width_px": width_px,
                    "height_px": height_px,
                }
                for filename, page, caption, context, image_ref, width_px, height_px in rows
            ]
        }

    except Exception as e:
        return create_error_response(f"rag/corpus-figures failed: {e}")


@router.get("/rag/figure-image")
async def rag_figure_image(ref: str):
    """
    Serve one extracted figure PNG by its content-addressed image_ref.

    Cache-Control mirrors /rag/document-page-image: bytes for a given ref
    never change, so the browser can skip the round-trip after the first fetch.
    """
    try:
        from rag.figures import figure_store_path

        # image_ref is DB-sourced but validate the shape anyway — belt & suspenders.
        if not re.fullmatch(r"[0-9a-f]+_\d+_\d+", ref or ""):
            return create_error_response("invalid figure reference", status_code=400)

        path = figure_store_path(ref)
        if not os.path.exists(path):
            return create_error_response("figure not found", status_code=404)

        with open(path, "rb") as f:
            png_bytes = f.read()

        return Response(
            content=png_bytes,
            media_type="image/png",
            headers={"Cache-Control": "public, max-age=31536000, immutable"},
        )

    except Exception as e:
        return create_error_response(f"rag/figure-image failed: {e}")


# ---------------------------------------------------------------------------
# RAG chat & retrieval
# ---------------------------------------------------------------------------

@router.post("/rag-chat/")
async def rag_chat(
    message: str = Form(...),
    session_id: str = Form(default=""),
    doc_ids: str = Form(default=""),
):
    """
    RAG-grounded conversational Q&A over the OCR knowledge base.

    Routes the question to the SQL and/or prose lane, then narrates the
    retrieved evidence.  Unlike /chat/ (which stuffs raw page text into the
    model), every number here comes from a SQL result row.

    doc_ids: optional comma-separated doc_id list from the "+" picker.
    """
    from task_manager import get_task_manager

    task_manager = get_task_manager()
    if session_id and not task_manager.touch_session(session_id):
        pass  # session is advisory here; the KB is not per-session

    if not (message or "").strip():
        return create_error_response("Message cannot be empty", status_code=400)

    try:
        from rag.synthesize import answer

        scoped = functools.partial(answer, message, doc_ids=_parse_doc_ids(doc_ids))
        result = await asyncio.get_event_loop().run_in_executor(None, scoped)
        return result

    except Exception as e:
        traceback.print_exc()
        return create_error_response(f"rag-chat failed: {e}")


@router.post("/rag/prose")
async def rag_prose(
    question: str = Form(...),
    k: int = Form(5),
    doc_ids: str = Form(default=""),
):
    """
    Semantic prose retrieval: top-k prose chunks by cosine similarity
    (BGE-M3 + pgvector), each with provenance and a score.
    """
    try:
        from rag.prose_search import search

        scoped = functools.partial(
            search, question, k=k, doc_ids=_parse_doc_ids(doc_ids)
        )
        hits = await asyncio.get_event_loop().run_in_executor(None, scoped)
        return {
            "question": question,
            "hits": [
                {
                    "score": round(float(h["score"]), 4),
                    "filename": h["filename"],
                    "page": h["page"],
                    "section": h["section"],
                    "content": h["content"],
                }
                for h in hits
            ],
        }

    except Exception as e:
        return create_error_response(f"rag/prose failed: {e}")


@router.post("/rag/tables")
async def rag_tables(
    question: str = Form(...),
    k: int = Form(4),
    doc_ids: str = Form(default=""),
):
    """
    Semantic table retrieval: top-k source tables as their original HTML,
    layout intact (rowspan / colspan / <br> multi-values).

    Mirrors /rag/prose so both lanes can be inspected the same way.
    /rag-chat/ is what actually narrates the results.
    """
    try:
        from rag.table_search import search

        scoped = functools.partial(
            search, question, k=k, doc_ids=_parse_doc_ids(doc_ids)
        )
        hits = await asyncio.get_event_loop().run_in_executor(None, scoped)
        return {
            "question": question,
            "hits": [
                {
                    "score": round(float(h.get("score") or 0.0), 4),
                    "match": h.get("match"),
                    "filename": h["filename"],
                    "page": h["page"],
                    "table_index": h["table_index"],
                    "caption": h.get("caption"),
                    "n_rows": h.get("n_rows"),
                    "n_cols": h.get("n_cols"),
                    "est_tokens": h.get("est_tokens"),
                    "well_formed": h.get("well_formed"),
                    "html": h.get("html"),
                }
                for h in hits
            ],
        }

    except Exception as e:
        return create_error_response(f"rag/tables failed: {e}")
