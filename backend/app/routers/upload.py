"""
app/routers/upload.py — File upload and WebSocket streaming endpoints.

Endpoints
---------
POST /upload/      — batch upload (PDF or image) with task management
WS   /ws/upload/   — chunked binary streaming WebSocket (real-time page results)

Protocol for /ws/upload/ (matches ocrClient.js):
  1. Client sends JSON text: { type: "upload_start", filename, total_chunks, ... }
  2. Client sends N binary ArrayBuffer chunks
  3. Server reassembles → runs OCR → sends page_complete per page
  4. Server sends processing_complete when done
"""

import json
import os
import pathlib
import re
import traceback
from datetime import datetime
from typing import Optional

import asyncio
from fastapi import APIRouter, File, Form, UploadFile, WebSocket, WebSocketDisconnect

from app.config import MAX_UPLOAD_SIZE, RAG_INGEST_ENABLED, TEMP_DIR
from app.utils.responses import create_error_response
from app.utils.text_processing import replace_markdown_tables_with_html, wrap_naked_tr
from ocr_olm import extract_text_from_pdf, stream_ocr_bytes
from task_manager import get_task_manager

router = APIRouter(tags=["Upload"])


# ---------------------------------------------------------------------------
# POST /upload/
# ---------------------------------------------------------------------------

@router.post("/upload/")
async def upload_file(
    file: UploadFile = File(...),
    lang: str = Form(...),
    session_id: str = Form(...),
    mode: str = Form("single"),
):
    """
    Batch upload endpoint with session and task management.

    Required form fields:
        file        — PDF or image file
        lang        — language code (e.g. "eng")
        session_id  — session ID from POST /session/create
    Optional:
        mode        — "single" (default) or "multi"
    """
    temp_file_path: Optional[str] = None
    task_id: Optional[str] = None
    task_manager = get_task_manager()

    try:
        # Validate session.
        if not task_manager.touch_session(session_id):
            return create_error_response(
                "Invalid or expired session. Please create a new session.",
                status_code=401,
            )

        if not file.filename:
            return create_error_response("No filename provided", status_code=400)

        # Create task (cancels any existing tasks for this session).
        task_id = task_manager.create_task(
            session_id=session_id,
            filename=file.filename,
            cancel_existing=True,
        )
        print(f"[UPLOAD] Session {session_id}, Task {task_id}, File: {file.filename}")

        # Build a safe absolute temp path under TEMP_DIR.
        # NOTE: keyed on task_id (not session_id) so it does NOT match the
        # session-scoped sweep in cancel_session — that sweep targets page files
        # and must not delete an in-progress upload.  Cleanup is handled by
        # the `finally` block below.
        safe_name = pathlib.Path(file.filename).name.replace(" ", "_")
        temp_file_path = os.path.join(TEMP_DIR, f"temp_{task_id}_{safe_name}")

        # Read and size-check.
        file_content = await file.read()
        if len(file_content) > MAX_UPLOAD_SIZE:
            task_info = task_manager.get_task(session_id, task_id)
            if task_info:
                task_info.mark_failed("File too large")
            return create_error_response(
                f"File too large. Maximum size is {MAX_UPLOAD_SIZE // (1024 * 1024)}MB",
                status_code=413,
            )

        with open(temp_file_path, "wb") as f:
            f.write(file_content)

        task_info = task_manager.get_task(session_id, task_id)
        if not task_info:
            return create_error_response("Task creation failed", status_code=500)

        # Run OCR.
        print(f"[UPLOAD] Processing task {task_id}...")
        ocr_result = extract_text_from_pdf(temp_file_path, lang, mode)

        # Check cancellation (could have been cancelled while OCR was running).
        if task_info.is_cancelled():
            print(f"[UPLOAD] Task {task_id} was cancelled during processing")
            return {
                "success": False,
                "task_id": task_id,
                "error": "Task was cancelled",
                "timestamp": datetime.now().isoformat(),
            }

        if ocr_result["success"]:
            task_info.mark_completed()

            # Persist each page's OCR text so the export endpoints can find it.
            # The WebSocket path writes these files during streaming; the POST
            # path (used for image uploads) must do the same or exports 404.
            for idx, page_info in enumerate(ocr_result["pages"], start=1):
                pnum = page_info.get("page") or idx
                page_text = page_info.get("text", "") or ""
                try:
                    if re.search(r"\|.*\||<table", page_text, re.IGNORECASE):
                        page_text = replace_markdown_tables_with_html(page_text)
                    page_text = wrap_naked_tr(page_text)
                    temp_filepath = os.path.join(
                        TEMP_DIR, f"temp_{session_id}_page_{pnum}.txt"
                    )
                    with open(temp_filepath, "w", encoding="utf-8") as f:
                        f.write(page_text)
                    print(f"[UPLOAD] Saved page {pnum} export file: {temp_filepath}")
                except Exception as save_err:
                    print(
                        f"[UPLOAD] WARN: could not save page {pnum} export file: {save_err}"
                    )

            # RAG ingestion — guarded so a DB failure never breaks OCR delivery.
            if RAG_INGEST_ENABLED and temp_file_path:
                try:
                    from rag.ingest import ingest_document

                    doc_rag_pages = {
                        (p.get("page") or i): p.get("rag")
                        for i, p in enumerate(ocr_result["pages"], start=1)
                    }
                    with open(temp_file_path, "rb") as _f:
                        _doc_bytes = _f.read()
                    ingest_document(_doc_bytes, file.filename, doc_rag_pages)
                    print(f"[UPLOAD] RAG ingest complete for {file.filename}")
                except Exception as e:
                    print(f"[UPLOAD] RAG ingest skipped/failed (non-fatal): {e}")

            # Format text for response.
            if len(ocr_result["pages"]) == 1:
                extracted_text = ocr_result["pages"][0]["text"]
            else:
                text_parts = []
                for page_info in ocr_result["pages"]:
                    if page_info["error"]:
                        text_parts.append(
                            f"Page {page_info['page']}: ERROR - {page_info['error']}"
                        )
                    else:
                        page_text = page_info["text"]
                        if page_text and page_text.strip():
                            text_parts.append(
                                f"Page {page_info['page']}:\n{page_text}"
                            )
                        else:
                            text_parts.append(
                                f"Page {page_info['page']}: No readable text found"
                            )
                extracted_text = "\n\n".join(text_parts)

            return {
                "success": True,
                "task_id": task_id,
                "session_id": session_id,
                "filename": file.filename,
                "lang": lang,
                "text": extracted_text,
                "pages": ocr_result["pages"],
                "total_pages": ocr_result["total_pages"],
                "timestamp": datetime.now().isoformat(),
            }

        else:
            task_info.mark_failed(ocr_result["error"])
            return create_error_response(
                f"OCR failed: {ocr_result['error']}",
                details={"task_id": task_id, "filename": file.filename},
            )

    except Exception as e:
        print(f"[ERROR] Upload exception: {str(e)}")
        traceback.print_exc()
        if task_id:
            task_manager = get_task_manager()
            task_info = task_manager.get_task(session_id, task_id)
            if task_info:
                task_info.mark_failed(str(e))
        return create_error_response(
            f"Server error: {str(e)}",
            details={
                "task_id": task_id,
                "filename": file.filename if file else "unknown",
            },
        )

    finally:
        if temp_file_path and os.path.exists(temp_file_path):
            try:
                os.remove(temp_file_path)
                print(f"[CLEANUP] Removed temp file: {temp_file_path}")
            except Exception as cleanup_error:
                print(f"[WARN] Failed to remove temp file: {cleanup_error}")


# ---------------------------------------------------------------------------
# WS /ws/upload/
# ---------------------------------------------------------------------------

@router.websocket("/ws/upload/")
async def websocket_upload(websocket: WebSocket):
    """
    Real-time streaming WebSocket with chunked binary file transfer.

    Protocol (matches ocrClient.js):
      1. Client sends JSON text: { type: "upload_start", filename, total_chunks,
                                   chunk_size, file_size, lang, mode, pages,
                                   table_schema }
      2. Client sends N binary ArrayBuffer chunks.
      3. Server reassembles the file, runs OCR, streams page_complete per page.
      4. Server sends processing_complete when done.
    """
    await websocket.accept()
    print("[WebSocket] Client connected")

    session_id = None
    task_id = None
    keep_alive_task = None
    temp_pdf_path = None
    task_manager = get_task_manager()

    async def _send_keep_alive():
        try:
            while True:
                await asyncio.sleep(20)
                await websocket.send_text(
                    json.dumps(
                        {"type": "keep_alive", "timestamp": datetime.now().isoformat()}
                    )
                )
        except asyncio.CancelledError:
            pass
        except Exception:
            pass

    try:
        # ── Step 1: Receive upload metadata (JSON) ──────────────────────────
        raw_meta = await websocket.receive_text()
        metadata = json.loads(raw_meta)

        if metadata.get("type") != "upload_start":
            await websocket.send_text(
                json.dumps(
                    {"type": "error", "error": "Expected upload_start message first"}
                )
            )
            return

        session_id   = metadata.get("session_id")
        filename     = metadata.get("filename", "unknown.pdf")
        lang         = metadata.get("lang", "eng")
        total_bytes  = metadata.get("file_size", 0)
        total_chunks = metadata.get("total_chunks", 1)
        mode         = metadata.get("mode", "single")
        pages_spec   = metadata.get("pages", None)
        user_schema  = metadata.get("table_schema", None)

        print(
            f"[WebSocket] Upload start: {filename} | "
            f"{total_bytes / (1024 * 1024):.1f}MB | {total_chunks} chunks | "
            f"mode: {mode} | user_schema: {'YES' if user_schema else 'NO (default)'}"
        )

        # Validate file size early.
        if total_bytes > MAX_UPLOAD_SIZE:
            await websocket.send_text(
                json.dumps(
                    {
                        "type": "error",
                        "error": (
                            f"File too large ({total_bytes / (1024 * 1024):.1f}MB). "
                            f"Max is {MAX_UPLOAD_SIZE // (1024 * 1024)}MB"
                        ),
                    }
                )
            )
            return

        # ── Step 2: Validate / create session & task ────────────────────────
        if session_id and task_manager.touch_session(session_id):
            print(f"[WebSocket] Using session: {session_id}")
        else:
            session_id = task_manager.create_session()
            print(f"[WebSocket] Created new session: {session_id}")

        task_id = task_manager.create_task(session_id, filename)
        print(f"[WebSocket] Task created: {task_id}")

        await websocket.send_text(
            json.dumps(
                {"type": "task_created", "task_id": task_id, "session_id": session_id}
            )
        )

        # ── Step 3: Receive all binary chunks ───────────────────────────────
        file_buffer = bytearray()
        chunks_received = 0

        print(f"[WebSocket] Receiving {total_chunks} chunks...")
        while chunks_received < total_chunks:
            message = await asyncio.wait_for(
                websocket.receive(), timeout=120  # 2 min per-chunk timeout
            )
            if "bytes" in message:
                file_buffer.extend(message["bytes"])
                chunks_received += 1
                print(
                    f"[WebSocket] Chunk {chunks_received}/{total_chunks} "
                    f"({len(file_buffer) / (1024 * 1024):.1f}MB received)"
                )
            elif "text" in message:
                print(
                    f"[WebSocket] Text during upload (ignoring): "
                    f"{message['text'][:100]}"
                )

        file_bytes = bytes(file_buffer)
        del file_buffer  # free memory immediately
        print(
            f"[WebSocket] \u2705 Full file received: {len(file_bytes) / (1024 * 1024):.1f}MB"
        )

        # Persist the upload for the layout pipeline.
        temp_pdf_path = os.path.join(TEMP_DIR, f"temp_{session_id}_uploaded.pdf")
        with open(temp_pdf_path, "wb") as f:
            f.write(file_bytes)
        print(f"[WebSocket] Saved uploaded file to: {temp_pdf_path}")

        await websocket.send_text(
            json.dumps(
                {
                    "type": "upload_received",
                    "bytes_received": len(file_bytes),
                    "timestamp": datetime.now().isoformat(),
                }
            )
        )

        # ── Step 4: Start keep-alive & OCR streaming ────────────────────────
        keep_alive_task = asyncio.create_task(_send_keep_alive())

        await websocket.send_text(
            json.dumps(
                {
                    "type": "processing_started",
                    "filename": filename,
                    "timestamp": datetime.now().isoformat(),
                }
            )
        )

        # ── Step 5: Stream OCR results page-by-page ─────────────────────────
        page_count = 0
        task_info = (
            task_manager.get_task(session_id, task_id)
            if (task_id and session_id)
            else None
        )
        buffered_pages = {}
        doc_rag_pages = {}  # page_num -> rag payload; fed to the DB on completion

        async for page_result in stream_ocr_bytes(
            file_bytes, mode=mode, pages=pages_spec
        ):
            msg_type = page_result.get("type", "")

            # Skip stream_ocr_bytes's own processing_complete and page_start —
            # the server sends its own completion message below.
            if msg_type in ("processing_complete", "page_start"):
                await asyncio.sleep(0)
                continue

            if msg_type == "page_complete":
                page_count += 1
                current = page_result.get("page", page_count)
                total = page_result.get("total_pages", "?")

                # Extract the RAG payload before sending to the frontend
                # (it is large and the client does not use it).
                doc_rag_pages[current] = page_result.pop("rag", None)

                page_text = page_result.get("text", "")
                if re.search(r"\|.*\||<table", page_text, re.IGNORECASE):
                    try:
                        page_text = replace_markdown_tables_with_html(page_text)
                        page_result["text"] = page_text
                    except Exception as e:
                        print(
                            f"[WebSocket] Error converting markdown tables to HTML: {e}"
                        )

                buffered_pages[current] = {
                    "text": page_text,
                    "preview": page_result.get("preview"),
                }

                # Pre-save so the export endpoint never 404s while remaining
                # pages are still processing.
                temp_filepath = os.path.join(
                    TEMP_DIR, f"temp_{session_id}_page_{current}.txt"
                )
                try:
                    with open(temp_filepath, "w", encoding="utf-8") as f:
                        f.write(page_text)
                    print(
                        f"[WebSocket] Pre-saved OCR temp file for page {current} "
                        f"to: {temp_filepath}"
                    )
                except Exception as fe:
                    print(
                        f"[WebSocket] Error writing pre-save OCR temp file: {fe}"
                    )

                if task_info:
                    task_info.update_progress(current)
                task_manager.touch_session(session_id)

                # Always inject identifiers so the frontend export button
                # has them from the very first page_complete message.
                page_result["session_id"] = session_id
                page_result["task_id"] = task_id

                print(f"[WebSocket] Streamed base page {current}/{total} → client")

            await websocket.send_text(json.dumps(page_result))
            await asyncio.sleep(0)  # yield to event loop

        # ── RAG ingestion ────────────────────────────────────────────────────
        # Runs HERE, right after OCR finishes and doc_rag_pages is fully
        # populated — NOT after the client-facing final streaming below.
        # The client disconnecting (e.g. navigating to the Corpus tab) raises
        # WebSocketDisconnect immediately, which would skip ingestion if it sat
        # behind the streaming loop.  The corpus is the durable artifact; the
        # client stream is best-effort.
        if RAG_INGEST_ENABLED:
            try:
                from rag.ingest import ingest_document

                ingest_document(file_bytes, filename, doc_rag_pages)
                print(f"[WebSocket] RAG ingest complete for {filename}")
            except Exception as e:
                print(f"[WebSocket] RAG ingest skipped/failed (non-fatal): {e}")

        # ── Step 6: Send final structured pages & done ───────────────────────
        print("[WebSocket] Streaming final structured table updates to client...")
        for page_num, info in buffered_pages.items():
            page_text = info["text"]

            # Clean stray *TABLE* markers.
            page_text = re.sub(r"\*TABLE\*", "", page_text, flags=re.IGNORECASE)

            # Wrap any naked <tr> blocks without nesting inside existing tables.
            page_text = wrap_naked_tr(page_text)

            has_table = bool(
                re.search(r"<table[\s\S]*?</table>", page_text, re.IGNORECASE)
            ) or "<tr>" in page_text

            # Always write — even pages without tables get a file so the export
            # endpoint never returns 404.
            temp_filepath = os.path.join(
                TEMP_DIR, f"temp_{session_id}_page_{page_num}.txt"
            )
            try:
                with open(temp_filepath, "w", encoding="utf-8") as f:
                    f.write(page_text)
                status_str = "has tables" if has_table else "text only"
                print(
                    f"[WebSocket] Saved temp file for page {page_num} ({status_str})."
                )
            except Exception as file_err:
                print(
                    f"[WebSocket] Error writing temp file for page {page_num}: {file_err}"
                )

            await websocket.send_text(
                json.dumps(
                    {
                        "type": "page_complete",
                        "page": page_num,
                        "total_pages": page_count,
                        "text": page_text,
                        "preview": info["preview"],
                        "has_table": has_table,
                        "session_id": session_id,
                        "task_id": task_id,
                        "status": "completed",
                    }
                )
            )

        if task_info:
            task_info.mark_completed()

        await websocket.send_text(
            json.dumps(
                {
                    "type": "processing_complete",
                    "total_pages_sent": page_count,
                    "timestamp": datetime.now().isoformat(),
                }
            )
        )
        print(f"[WebSocket] Done. {page_count} pages streamed for {filename}")

    # ── Error handling ───────────────────────────────────────────────────────
    except asyncio.TimeoutError:
        print("[WebSocket] Timeout waiting for chunk")
        try:
            await websocket.send_text(
                json.dumps(
                    {
                        "type": "error",
                        "error": "Upload timeout — no data received within 120 seconds",
                    }
                )
            )
        except Exception:
            pass

    except WebSocketDisconnect:
        print(
            f"[WebSocket] Client disconnected (session: {session_id}, task: {task_id})"
        )

    except json.JSONDecodeError as e:
        print(f"[WebSocket] Bad JSON: {e}")
        try:
            await websocket.send_text(
                json.dumps({"type": "error", "error": f"Invalid JSON: {str(e)}"})
            )
        except Exception:
            pass

    except Exception as e:
        print(f"[WebSocket] Error: {str(e)}")
        traceback.print_exc()
        try:
            await websocket.send_text(
                json.dumps({"type": "error", "error": str(e)})
            )
        except Exception:
            pass

    # ── Cleanup ──────────────────────────────────────────────────────────────
    finally:
        if keep_alive_task and not keep_alive_task.done():
            keep_alive_task.cancel()
            try:
                await keep_alive_task
            except asyncio.CancelledError:
                pass
        if temp_pdf_path and os.path.exists(temp_pdf_path):
            try:
                os.remove(temp_pdf_path)
                print(f"[WebSocket] Cleaned up temp PDF: {temp_pdf_path}")
            except Exception as e:
                print(f"[WebSocket] Failed to remove temp PDF: {e}")
