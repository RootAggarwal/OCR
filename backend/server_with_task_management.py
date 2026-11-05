# FASTAPI: server_with_task_management.py - Enhanced Server with Task Lifecycle Management
"""
Enhanced OCR Server with:
- Session-based task management
- Multi-user support
- Automatic task cancellation on disconnect
- Resource cleanup
"""

from fastapi import FastAPI, File, UploadFile, Form, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, FileResponse, Response
from pydantic import BaseModel
from typing import Optional, List, Dict, Any
from contextlib import asynccontextmanager
import os, traceback, pathlib, json, asyncio, base64, re, glob, functools
import torch
import psutil, subprocess
from datetime import datetime
import openpyxl
from openpyxl.styles import Font, PatternFill, Alignment

class UpdatePageRequest(BaseModel):
    text: str

# Import OCR functions
from ocr_olm import (
    extract_text_from_pdf,
    stream_ocr_bytes,
    run_plain_text_inference,
    get_performance_stats,
)

# Import task management
from task_manager import (
    get_task_manager,
    TaskManager
)

from bs4 import BeautifulSoup


def wrap_naked_tr(text: str) -> str:
    """
    Wrap runs of NAKED <tr>...</tr> (rows the VLM emitted without a <table>) in a
    <table>, WITHOUT touching <tr> that already live inside a well-formed <table>.

    A naive regex that wraps every <tr> run corrupts proper tables: it wraps the rows
    inside <thead>/<tbody> in extra nested <table> tags, so ONE table then exports as
    THREE (outer + header-only + data-only). We first hide complete existing
    <table>...</table> blocks behind placeholders, wrap only the naked <tr> that remain,
    then restore the hidden tables untouched.
    """
    import re as _re
    stash = []

    def _hide(m):
        stash.append(m.group(0))
        return f"\x00TBL{len(stash) - 1}\x00"

    protected = _re.sub(r"<table[\s\S]*?</table>", _hide, text, flags=_re.IGNORECASE)

    def _wrap(m):
        block = m.group(1)
        return ('\n<table class="dataframe table table-hover table-striped '
                'table-bordered ocr-dataframe">\n' + block.strip() + '\n</table>\n')

    protected = _re.sub(r"((?:[ \t\r\n]*<tr>[\s\S]*?</tr>[ \t\r\n]*)+)",
                        _wrap, protected, flags=_re.IGNORECASE)

    for i, tbl in enumerate(stash):
        protected = protected.replace(f"\x00TBL{i}\x00", tbl)
    return protected


def replace_markdown_tables_with_html(text: str) -> str:
    """
    Finds markdown tables AND raw HTML tables in the text and converts/wraps 
    them in a div.table-container.
    """
    import re
    import markdown
    
    # 1. Wrap raw HTML tables first
    # This handles cases where OLM-OCR natively outputs <table>...</table>
    text = re.sub(
        r'(<table.*?>.*?</table>)', 
        r'<div class="table-container">\n\1\n</div>', 
        text, 
        flags=re.IGNORECASE | re.DOTALL
    )
    
    # 2. Convert markdown tables
    lines = text.split('\n')
    new_lines = []
    table_lines = []
    in_table = False
    
    for line in lines:
        # Detect markdown table rows: start and end with |
        if re.match(r'^\s*\|.*\|', line) or (in_table and re.match(r'^\s*[-|:\s]+$', line)):
            in_table = True
            table_lines.append(line)
        else:
            if in_table:
                # Convert the collected table block to HTML
                table_md = "\n".join(table_lines)
                table_html = markdown.markdown(table_md, extensions=['tables'])
                new_lines.append(f'<div class="table-container">\n{table_html}\n</div>')
                table_lines = []
                in_table = False
            new_lines.append(line)
            
    if in_table:
        table_md = "\n".join(table_lines)
        table_html = markdown.markdown(table_md, extensions=['tables'])
        new_lines.append(f'<div class="table-container">\n{table_html}\n</div>')
        
    return "\n".join(new_lines)

class TableParser:
    def __init__(self):
        self.tables = []

    def feed(self, content):
        """
        Parses all HTML tables in content, correctly expanding colspan and rowspan.
        """
        self.tables = []
        soup = BeautifulSoup(content, "html.parser")
        tables = soup.find_all("table")
        for table in tables:
            rows = table.find_all("tr")
            grid = {}  # maps (r, c) -> cell_dict
            
            current_row = 0
            for tr in rows:
                current_col = 0
                cells = tr.find_all(["td", "th"])
                for cell in cells:
                    while (current_row, current_col) in grid:
                        current_col += 1
                        
                    lines = [line.strip() for line in cell.get_text(separator="\n").split("\n")]
                    text = "\n".join(line for line in lines if line)
                    
                    is_header = (cell.name == "th") or (cell.get("class") and "header" in cell.get("class")) or (cell.parent.name == "thead")
                    if not is_header:
                        if cell.find(["strong", "b"]):
                            is_header = True
                            
                    colspan = int(cell.get("colspan", 1))
                    rowspan = int(cell.get("rowspan", 1))
                    
                    for r_offset in range(rowspan):
                        for c_offset in range(colspan):
                            grid[(current_row + r_offset, current_col + c_offset)] = {
                                "text": text,
                                "header": is_header,
                                "colspan": colspan,
                                "rowspan": rowspan,
                                "is_top_left": (r_offset == 0 and c_offset == 0)
                            }
                            
                    current_col += colspan
                current_row += 1

            if not grid:
                continue
                
            max_row = max(r for r, c in grid.keys()) + 1
            max_col = max(c for r, c in grid.keys()) + 1
            
            result_grid = []
            for r in range(max_row):
                row_data = []
                for c in range(max_col):
                    row_data.append(grid.get((r, c), {"text": "", "header": False, "colspan": 1, "rowspan": 1, "is_top_left": True}))
                result_grid.append(row_data)
                
            self.tables.append(result_grid)


# ============================================================================
# CONFIGURATION
# ============================================================================

HOST = os.getenv("OCR_HOST", "0.0.0.0")
PORT = int(os.getenv("OCR_PORT", "8000"))
MAX_UPLOAD_SIZE = int(os.getenv("MAX_UPLOAD_SIZE", str(500 * 1024 * 1024)))  # 500MB
WEBSOCKET_KEEPALIVE_INTERVAL = 30

# All OCR intermediates live in TEMP_DIR (<OCR_DATA_DIR>/temp) — outside the repo, and
# outside the process CWD. Imported rather than recomputed so that this module's writers
# and task_manager.py's cleanup sweep cannot drift apart: they previously agreed only by
# coincidence, and one write site (the POST /upload/ path below) was a bare relative
# filename that landed in the CWD where the sweep never looked.
from ocr_paths import TEMP_DIR

logger = None  # Will use print for now, can add logging

# ============================================================================
# FASTAPI APP SETUP
# ============================================================================

# Global task manager
task_manager: TaskManager = get_task_manager()


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Startup/shutdown lifecycle (replaces deprecated @app.on_event)."""
    print("[STARTUP] Starting task manager cleanup...")
    task_manager.start_cleanup()
    print("[STARTUP] Server ready")
    yield
    print("[SHUTDOWN] Cleaning up task manager...")
    await task_manager.shutdown()
    print("[SHUTDOWN] Server stopped")


app = FastAPI(title="OCR Service with Task Management", version="3.0.0", lifespan=lifespan)

# NOTE: a wildcard origin cannot be combined with allow_credentials=True — browsers
# reject credentialed responses whose Access-Control-Allow-Origin is "*". Since this
# API is token-less and does not rely on cookies, we drop credentials and keep the
# wildcard so any LAN device can reach it. Set OCR_ALLOWED_ORIGINS (comma-separated)
# to lock this down in production.
_allowed_origins_env = os.getenv("OCR_ALLOWED_ORIGINS", "*").strip()
_allowed_origins = ["*"] if _allowed_origins_env == "*" else [
    o.strip() for o in _allowed_origins_env.split(",") if o.strip()
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=_allowed_origins,
    allow_credentials=_allowed_origins != ["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def create_error_response(
    error: str,
    status_code: int = 500,
    details: Optional[Dict[str, Any]] = None
) -> JSONResponse:
    """Create standardized error response."""
    return JSONResponse(
        status_code=status_code,
        content={
            "success": False,
            "error": error,
            "timestamp": datetime.now().isoformat(),
            "details": details or {}
        }
    )


# Session IDs are UUIDs; allow only a safe filename charset so a crafted
# session_id can never traverse out of TEMP_DIR (e.g. "../../etc/passwd").
_SAFE_SESSION_RE = re.compile(r"^[A-Za-z0-9_-]{1,64}$")


def is_safe_session_id(session_id: str) -> bool:
    """Return True only for session IDs safe to embed in a filename."""
    return bool(session_id) and bool(_SAFE_SESSION_RE.match(session_id))


def smart_truncate_text(text: str, max_chars: int = 3000) -> str:
    """Smart text truncation at sentence boundaries."""
    if len(text) <= max_chars:
        return text
    
    truncated = text[:max_chars]
    
    # Find last sentence ending
    last_period = truncated.rfind('.')
    last_question = truncated.rfind('?')
    last_exclamation = truncated.rfind('!')
    last_boundary = max(last_period, last_question, last_exclamation)
    
    if last_boundary > max_chars * 0.8:
        return truncated[:last_boundary + 1]
    
    last_space = truncated.rfind(' ')
    if last_space > 0:
        return truncated[:last_space] + "..."
    
    return truncated + "..."


def validate_base64(data: str) -> bool:
    """Validate base64 string."""
    try:
        base64.b64decode(data, validate=True)
        return True
    except Exception:
        return False


def extract_session_id(websocket: WebSocket) -> Optional[str]:
    """Extract session ID from WebSocket headers or query params."""
    # Try to get from query parameters
    session_id = websocket.query_params.get("session_id")
    if session_id:
        return session_id
    
    # Try to get from headers
    session_id = websocket.headers.get("x-session-id")
    if session_id:
        return session_id
    
    return None


def get_gpu_usage():
    """Get NVIDIA GPU utilization via nvidia-smi."""
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=utilization.gpu,memory.used,memory.total",
             "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=2
        )
        if result.returncode == 0:
            parts = result.stdout.strip().split(", ")
            return {
                "gpu_util": int(parts[0]),
                "gpu_mem_used": int(parts[1]),
                "gpu_mem_total": int(parts[2]),
            }
    except (FileNotFoundError, subprocess.TimeoutExpired, Exception):
        pass
    return {"gpu_util": 0, "gpu_mem_used": 0, "gpu_mem_total": 0}

@app.get("/system-stats")
async def system_stats():
    cpu = psutil.cpu_percent(interval=0.3)
    mem = psutil.virtual_memory()
    gpu = get_gpu_usage()

    return JSONResponse({
        "cpu": round(cpu),
        "gpu": gpu["gpu_util"],
        "mem": round(mem.percent),
        "mem_used_gb": round(mem.used / (1024 ** 3), 1),
        "mem_total_gb": round(mem.total / (1024 ** 3), 1),
        "gpu_mem_used_mb": gpu["gpu_mem_used"],
        "gpu_mem_total_mb": gpu["gpu_mem_total"],
    })

# ============================================================================
# SESSION ENDPOINTS
# ============================================================================

@app.post("/session/create")
async def create_session():
    """
    Create a new user session.
    Returns a session_id that must be used in all subsequent requests.
    """
    try:
        session_id = task_manager.create_session()
        return {
            "success": True,
            "session_id": session_id,
            "timestamp": datetime.now().isoformat(),
            "message": "Session created successfully"
        }
    except Exception as e:
        print(f"[ERROR] Failed to create session: {e}")
        return create_error_response(str(e))


@app.get("/session/{session_id}/tasks")
async def get_session_tasks(session_id: str):
    """Get all tasks for a session."""
    try:
        tasks = task_manager.get_session_tasks(session_id)
        return {
            "success": True,
            "session_id": session_id,
            "tasks": tasks,
            "timestamp": datetime.now().isoformat()
        }
    except Exception as e:
        print(f"[ERROR] Failed to get session tasks: {e}")
        return create_error_response(str(e))


@app.delete("/session/{session_id}")
async def delete_session(session_id: str):
    """Delete a session and cancel all its tasks."""
    try:
        cancelled_count = task_manager.cancel_session(session_id)
        return {
            "success": True,
            "session_id": session_id,
            "cancelled_tasks": cancelled_count,
            "timestamp": datetime.now().isoformat(),
            "message": f"Session deleted, {cancelled_count} tasks cancelled"
        }
    except Exception as e:
        print(f"[ERROR] Failed to delete session: {e}")
        return create_error_response(str(e))


@app.get("/session/{session_id}/task/{task_id}/status")
async def get_task_status(session_id: str, task_id: str):
    """Get status of a specific task."""
    try:
        task_info = task_manager.get_task(session_id, task_id)
        if not task_info:
            return create_error_response(
                f"Task {task_id} not found",
                status_code=404
            )
        
        return {
            "success": True,
            "task": task_info.get_progress(),
            "timestamp": datetime.now().isoformat()
        }
    except Exception as e:
        print(f"[ERROR] Failed to get task status: {e}")
        return create_error_response(str(e))


@app.delete("/session/{session_id}/task/{task_id}")
async def cancel_task(session_id: str, task_id: str):
    """Cancel a specific task."""
    try:
        success = task_manager.cancel_task(session_id, task_id)
        if success:
            return {
                "success": True,
                "message": f"Task {task_id} cancelled",
                "timestamp": datetime.now().isoformat()
            }
        else:
            return create_error_response(
                f"Task {task_id} not found or already completed",
                status_code=404
            )
    except Exception as e:
        print(f"[ERROR] Failed to cancel task: {e}")
        return create_error_response(str(e))


# ============================================================================
# EXPORT AND EDITING ENDPOINTS
# ============================================================================

@app.post("/update-page/{session_id}/{page_num}")
async def update_page_text(session_id: str, page_num: int, req: UpdatePageRequest):
    """Update the parsed text/HTML for a specific page."""
    try:
        if not is_safe_session_id(session_id):
            return create_error_response("Invalid session_id", status_code=400)
        file_path = os.path.join(TEMP_DIR, f"temp_{session_id}_page_{page_num}.txt")
        with open(file_path, "w", encoding="utf-8") as f:
            f.write(req.text)
        return {"status": "success", "message": "Page updated successfully."}
    except Exception as e:
        print(f"[ERROR] Failed to update page text: {e}")
        return create_error_response(str(e))

@app.get("/export-table/{session_id}/{page_num}")
async def export_table(session_id: str, page_num: int, table_index: Optional[int] = None):
    """Export parsed HTML tables from a page to Excel."""
    try:
        if not is_safe_session_id(session_id):
            return create_error_response("Invalid session_id", status_code=400)
        input_file = os.path.join(TEMP_DIR, f"temp_{session_id}_page_{page_num}.txt")
        
        if not os.path.exists(input_file):
            return create_error_response(
                f"No data found for page {page_num}. "
                f"Make sure the document has been processed first (session: {session_id}).",
                status_code=404
            )
            
        with open(input_file, "r", encoding="utf-8") as f:
            content = f.read()
            
        parser = TableParser()
        parser.feed(content)
        
        if not parser.tables:
            return create_error_response("No HTML tables found in the page.", status_code=404)
            
        if table_index is not None:
            if table_index < 0 or table_index >= len(parser.tables):
                return create_error_response(f"Invalid table index {table_index}. Page only has {len(parser.tables)} table(s).", status_code=400)
            tables_to_export = [parser.tables[table_index]]
            output_filename = f"page_{page_num}_table_{table_index + 1}.xlsx"
        else:
            tables_to_export = parser.tables
            output_filename = f"page_{page_num}_tables.xlsx"
        
        output_path = os.path.join(TEMP_DIR, f"temp_{session_id}_page_{page_num}_export.xlsx")
        
        wb = openpyxl.Workbook()
        
        header_fill = PatternFill("solid", fgColor="D9E1F2")
        header_font = Font(bold=True, name="Arial")
        normal_font = Font(name="Arial")
        center = Alignment(horizontal="center", vertical="center", wrap_text=True)

        for i, table_data in enumerate(tables_to_export):
            title_suffix = f" {table_index + 1}" if table_index is not None else f" {i+1}"
            if i == 0:
                ws = wb.active
                ws.title = f"Table{title_suffix}"
            else:
                ws = wb.create_sheet(title=f"Table{title_suffix}")
                
            for row_idx, row in enumerate(table_data, start=1):
                for col_idx, cell in enumerate(row, start=1):
                    if cell.get("is_top_left", True):
                        ws_cell = ws.cell(row=row_idx, column=col_idx, value=cell["text"])
                        ws_cell.font = header_font if cell.get("header") else normal_font
                        ws_cell.alignment = center
                        if cell.get("header"):
                            ws_cell.fill = header_fill
                            
                        colspan = cell.get("colspan", 1)
                        rowspan = cell.get("rowspan", 1)
                        if colspan > 1 or rowspan > 1:
                            ws.merge_cells(
                                start_row=row_idx, start_column=col_idx,
                                end_row=row_idx + rowspan - 1, end_column=col_idx + colspan - 1
                            )

            # Auto-fit column widths
            from openpyxl.utils import get_column_letter
            for col_idx_num, col in enumerate(ws.columns, start=1):
                max_len = max((len(str(c.value or "")) for c in col if hasattr(c, 'value')), default=10)
                col_letter = get_column_letter(col_idx_num)
                ws.column_dimensions[col_letter].width = min(max_len + 4, 50)

        wb.save(output_path)
        
        return FileResponse(
            output_path, 
            filename=output_filename,
            media_type="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            background=None # the file will be left on disk and can be cleaned up later by task_manager
        )
        
    except Exception as e:
        print(f"[ERROR] Failed to export table: {e}")
        return create_error_response(str(e))


@app.get("/export-all/{session_id}")
async def export_all_tables(session_id: str):
    """Export all parsed HTML tables from all pages in the session to a single Excel file."""
    try:
        if not is_safe_session_id(session_id):
            return create_error_response("Invalid session_id", status_code=400)
        import glob
        # Find all temp files for this session using absolute paths
        temp_files = glob.glob(os.path.join(TEMP_DIR, f"temp_{session_id}_page_*.txt"))
        if not temp_files:
            return create_error_response(
                f"No page files found for session '{session_id}'. "
                "Make sure you have processed a document first.",
                status_code=404
            )
        
        # Sort files by page number to maintain order
        # filename format: temp_{session_id}_page_{page_num}.txt
        def get_page_num(filepath):
            try:
                parts = os.path.basename(filepath).split("_")
                return int(parts[-1].replace(".txt", ""))
            except Exception:
                return 9999
        
        temp_files.sort(key=get_page_num)
        
        wb = openpyxl.Workbook()
        default_sheet_removed = False
        
        header_fill = PatternFill("solid", fgColor="D9E1F2")
        header_font = Font(bold=True, name="Arial")
        normal_font = Font(name="Arial")
        center = Alignment(horizontal="center", vertical="center", wrap_text=True)
        
        table_count = 0
        for filepath in temp_files:
            page_num = get_page_num(filepath)
            with open(filepath, "r", encoding="utf-8") as f:
                content = f.read()
            
            parser = TableParser()
            parser.feed(content)
            
            for i, table_data in enumerate(parser.tables):
                table_count += 1
                sheet_title = f"P{page_num}_T{i+1}"
                sheet_title = sheet_title[:31] # limit to 31 chars Excel limit
                
                if not default_sheet_removed:
                    ws = wb.active
                    ws.title = sheet_title
                    default_sheet_removed = True
                else:
                    ws = wb.create_sheet(title=sheet_title)
                
                for row_idx, row in enumerate(table_data, start=1):
                    for col_idx, cell in enumerate(row, start=1):
                        if cell.get("is_top_left", True):
                            ws_cell = ws.cell(row=row_idx, column=col_idx, value=cell["text"])
                            ws_cell.font = header_font if cell.get("header") else normal_font
                            ws_cell.alignment = center
                            if cell.get("header"):
                                ws_cell.fill = header_fill
                                
                            colspan = cell.get("colspan", 1)
                            rowspan = cell.get("rowspan", 1)
                            if colspan > 1 or rowspan > 1:
                                ws.merge_cells(
                                    start_row=row_idx, start_column=col_idx,
                                    end_row=row_idx + rowspan - 1, end_column=col_idx + colspan - 1
                                )
                
                # Auto-fit column widths
                from openpyxl.utils import get_column_letter
                for col_idx_num, col in enumerate(ws.columns, start=1):
                    max_len = max((len(str(c.value or "")) for c in col if hasattr(c, 'value')), default=10)
                    col_letter = get_column_letter(col_idx_num)
                    ws.column_dimensions[col_letter].width = min(max_len + 4, 50)
        
        if table_count == 0:
            return create_error_response("No HTML tables found in any of the session pages.", status_code=404)
            
        output_filename = f"session_{session_id}_all_tables.xlsx"
        output_path = os.path.join(TEMP_DIR, f"temp_{session_id}_all_export.xlsx")
        wb.save(output_path)
        
        return FileResponse(
            output_path,
            filename=output_filename,
            media_type="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            background=None
        )
    except Exception as e:
        print(f"[ERROR] Failed to export all tables: {e}")
        return create_error_response(str(e))


# ============================================================================
# UPLOAD ENDPOINTS (with Task Management)
# ============================================================================

@app.post("/upload/")
async def upload_file(
    file: UploadFile = File(...),
    lang: str = Form(...),
    session_id: str = Form(...),
    mode: str = Form("single")
):
    """
    Batch upload endpoint with session and task management.
    
    Required:
        - file: PDF or image file
        - lang: Language code
        - session_id: Session ID from /session/create
    """
    temp_file_path: Optional[str] = None
    task_id: Optional[str] = None
    
    try:
        # Validate session
        if not task_manager.touch_session(session_id):
            return create_error_response(
                "Invalid or expired session. Please create a new session.",
                status_code=401
            )
        
        # Validate file
        if not file.filename:
            return create_error_response("No filename provided", status_code=400)
        
        # Create task
        task_id = task_manager.create_task(
            session_id=session_id,
            filename=file.filename,
            cancel_existing=True  # Cancel any existing tasks
        )
        
        print(f"[UPLOAD] Session {session_id}, Task {task_id}, File: {file.filename}")
        
        # Create safe temp filename. Absolute, under TEMP_DIR: this used to be a bare
        # relative name, so the upload landed in the process CWD (the project root when
        # launched via run_project.sh) while task_manager's sweep globbed only backend/ —
        # any upload whose `finally` cleanup didn't run was orphaned in the repo forever.
        # NOTE it is still not swept: the sweep patterns key on session_id and this name
        # keys on task_id. Its cleanup remains the `finally` below. That is acceptable now
        # only because TEMP_DIR is a disposable directory outside the repo. Do NOT widen
        # the sweep to temp_{session_id}_* to "fix" this — it would match a live session's
        # in-progress page files and delete them mid-run.
        safe_name = pathlib.Path(file.filename).name.replace(" ", "_")
        temp_file_path = os.path.join(TEMP_DIR, f"temp_{task_id}_{safe_name}")
        
        # Read file
        file_content = await file.read()
        
        # Check file size
        if len(file_content) > MAX_UPLOAD_SIZE:
            if task_id:
                task_info = task_manager.get_task(session_id, task_id)
                if task_info:
                    task_info.mark_failed("File too large")
            
            return create_error_response(
                f"File too large. Maximum size is {MAX_UPLOAD_SIZE // (1024*1024)}MB",
                status_code=413
            )
        
        # Save file
        with open(temp_file_path, "wb") as f:
            f.write(file_content)
        
        # Get task info
        task_info = task_manager.get_task(session_id, task_id)
        if not task_info:
            return create_error_response("Task creation failed", status_code=500)
        
        # Process OCR
        print(f"[UPLOAD] Processing task {task_id}...")
        ocr_result = extract_text_from_pdf(temp_file_path, lang, mode)
        
        # Check if cancelled during processing
        if task_info.is_cancelled():
            print(f"[UPLOAD] Task {task_id} was cancelled during processing")
            return {
                "success": False,
                "task_id": task_id,
                "error": "Task was cancelled",
                "timestamp": datetime.now().isoformat()
            }
        
        if ocr_result["success"]:
            # Mark task as completed
            task_info.mark_completed()

            # Persist each page's OCR text so the export endpoints (/export-table,
            # /export-all) can find it. The WebSocket path writes these files; this
            # POST path (used for IMAGE uploads) previously did not, so exporting an
            # image always failed with "No data found for page N".
            if session_id:
                import re as _re
                for idx, page_info in enumerate(ocr_result["pages"], start=1):
                    pnum = page_info.get("page") or idx
                    page_text = page_info.get("text", "") or ""
                    try:
                        if _re.search(r'\|.*\||<table', page_text, _re.IGNORECASE):
                            page_text = replace_markdown_tables_with_html(page_text)
                        # Wrap any NAKED <tr> in a <table>, without nesting tables inside
                        # already-formed ones (mirrors the WS path).
                        page_text = wrap_naked_tr(page_text)
                        temp_filepath = os.path.join(TEMP_DIR, f"temp_{session_id}_page_{pnum}.txt")
                        with open(temp_filepath, "w", encoding="utf-8") as f:
                            f.write(page_text)
                        print(f"[UPLOAD] Saved page {pnum} export file: {temp_filepath}")
                    except Exception as save_err:
                        print(f"[UPLOAD] WARN: could not save page {pnum} export file: {save_err}")

            # RAG ingestion (POST path — images / non-realtime uploads). Guarded so a
            # RAG/DB failure never breaks OCR delivery; no-op unless RAG_INGEST_ENABLED=1.
            if os.getenv("RAG_INGEST_ENABLED", "0") == "1" and temp_file_path:
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

            # Format text for response
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
                            text_parts.append(f"Page {page_info['page']}:\n{page_text}")
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
                "timestamp": datetime.now().isoformat()
            }
        else:
            # Mark task as failed
            task_info.mark_failed(ocr_result['error'])
            
            return create_error_response(
                f"OCR failed: {ocr_result['error']}",
                details={
                    "task_id": task_id,
                    "filename": file.filename
                }
            )
    
    except Exception as e:
        print(f"[ERROR] Upload exception: {str(e)}")
        traceback.print_exc()
        
        # Mark task as failed
        if task_id:
            task_info = task_manager.get_task(session_id, task_id)
            if task_info:
                task_info.mark_failed(str(e))
        
        return create_error_response(
            f"Server error: {str(e)}",
            details={
                "task_id": task_id,
                "filename": file.filename if file else "unknown"
            }
        )
    
    finally:
        # Cleanup temp file
        if temp_file_path and os.path.exists(temp_file_path):
            try:
                os.remove(temp_file_path)
                print(f"[CLEANUP] Removed temp file: {temp_file_path}")
            except Exception as cleanup_error:
                print(f"[WARN] Failed to remove temp file: {cleanup_error}")


# @app.websocket("/ws/upload/")
# async def websocket_upload(websocket: WebSocket):
#     """
#     Real-time streaming endpoint with task management.
    
#     Client must send session_id in initial message or as query parameter.
#     """
#     session_id: Optional[str] = None
#     task_id: Optional[str] = None
#     keep_alive_task: Optional[asyncio.Task] = None
#     processing_task: Optional[asyncio.Task] = None
    
#     await websocket.accept()
#     print("[WebSocket] Client connected")

#     async def send_keep_alive():
#         """Send periodic keep-alive messages."""
#         try:
#             while True:
#                 await asyncio.sleep(WEBSOCKET_KEEPALIVE_INTERVAL)
#                 await websocket.send_text(json.dumps({
#                     "type": "keep_alive",
#                     "timestamp": datetime.now().isoformat()
#                 }))
                
#                 # Touch session to keep it alive
#                 if session_id:
#                     task_manager.touch_session(session_id)
#         except asyncio.CancelledError:
#             print("[WebSocket] Keep-alive cancelled")
#         except Exception as e:
#             print(f"[WebSocket] Keep-alive error: {e}")

#     async def process_ocr_stream(file_bytes: bytes, filename: str):
#         """Process OCR with task tracking and cancellation support."""
#         try:
#             task_info = task_manager.get_task(session_id, task_id)
#             if not task_info:
#                 raise ValueError("Task not found")
            
#             page_count = 0
            
#             async for result in stream_ocr_bytes(file_bytes):
#                 # Check for cancellation before each result
#                 if task_info.is_cancelled():
#                     print(f"[WebSocket] Task {task_id} cancelled, stopping stream")
#                     await websocket.send_text(json.dumps({
#                         "type": "cancelled",
#                         "task_id": task_id,
#                         "message": "Processing cancelled"
#                     }))
#                     return
                
#                 # Update progress
#                 if result.get("type") == "page_complete":
#                     page_count = result.get("page", page_count + 1)
#                     task_info.update_progress(page_count)
                
#                 # Send result to client
#                 result["task_id"] = task_id
#                 result["session_id"] = session_id
#                 await websocket.send_text(json.dumps(result))
#                 await asyncio.sleep(0.05)
            
#             # Mark as completed
#             task_info.mark_completed()
#             print(f"[WebSocket] Task {task_id} completed successfully")
            
#         except asyncio.CancelledError:
#             print(f"[WebSocket] Task {task_id} processing cancelled")
#             if task_info:
#                 task_info.mark_cancelled()
#             raise
#         except Exception as e:
#             print(f"[WebSocket] Task {task_id} processing error: {e}")
#             if task_info:
#                 task_info.mark_failed(str(e))
#             raise

#     try:
#         # Receive initial data
#         data = await websocket.receive_text()
#         request_data = json.loads(data)

#         # Extract session_id
#         session_id = request_data.get("session_id") or extract_session_id(websocket)
        
#         if not session_id:
#             await websocket.send_text(json.dumps({
#                 "type": "error",
#                 "error": "session_id is required. Create a session first using /session/create"
#             }))
#             return
        
#         # Validate session
#         if not task_manager.touch_session(session_id):
#             await websocket.send_text(json.dumps({
#                 "type": "error",
#                 "error": "Invalid or expired session. Please create a new session."
#             }))
#             return
        
#         # Extract request data
#         file_data = request_data.get("file_data")
#         filename = request_data.get("filename", "unknown")
#         lang = request_data.get("lang", "eng")

#         if not file_data:
#             await websocket.send_text(json.dumps({
#                 "type": "error",
#                 "error": "No file data provided"
#             }))
#             return

#         # Validate base64
#         if not validate_base64(file_data):
#             await websocket.send_text(json.dumps({
#                 "type": "error",
#                 "error": "Invalid base64 data"
#             }))
#             return

#         # Create task (cancels any existing tasks for this session)
#         task_id = task_manager.create_task(
#             session_id=session_id,
#             filename=filename,
#             cancel_existing=True
#         )
        
#         print(f"[WebSocket] Created task {task_id} for session {session_id}")

#         # Send task created notification
#         await websocket.send_text(json.dumps({
#             "type": "task_created",
#             "task_id": task_id,
#             "session_id": session_id,
#             "filename": filename,
#             "timestamp": datetime.now().isoformat()
#         }))

#         # Start keep-alive
#         keep_alive_task = asyncio.create_task(send_keep_alive())

#         # Decode file data
#         try:
#             file_bytes = base64.b64decode(file_data)
#         except Exception as e:
#             await websocket.send_text(json.dumps({
#                 "type": "error",
#                 "error": f"Failed to decode file data: {str(e)}"
#             }))
#             return

#         # Check file size
#         if len(file_bytes) > MAX_UPLOAD_SIZE:
#             task_info = task_manager.get_task(session_id, task_id)
#             if task_info:
#                 task_info.mark_failed("File too large")
            
#             await websocket.send_text(json.dumps({
#                 "type": "error",
#                 "error": f"File too large. Maximum size is {MAX_UPLOAD_SIZE // (1024*1024)}MB"
#             }))
#             return

#         # Start processing
#         await websocket.send_text(json.dumps({
#             "type": "processing_started",
#             "task_id": task_id,
#             "filename": filename,
#             "timestamp": datetime.now().isoformat()
#         }))

#         # Create processing task
#         processing_task = asyncio.create_task(
#             process_ocr_stream(file_bytes, filename)
#         )
        
#         # Associate task with task manager
#         task_manager.set_async_task(session_id, task_id, processing_task)
        
#         # Wait for processing to complete
#         await processing_task
        
#         print("[WebSocket] Processing completed successfully")

#     except WebSocketDisconnect:
#         print(f"[WebSocket] Client disconnected (session: {session_id}, task: {task_id})")
        
#         # Cancel task on disconnect
#         if session_id and task_id:
#             task_manager.cancel_task(session_id, task_id)
#             print(f"[WebSocket] Cancelled task {task_id} due to disconnect")
    
#     except json.JSONDecodeError as e:
#         print(f"[WebSocket] Invalid JSON: {e}")
#         try:
#             await websocket.send_text(json.dumps({
#                 "type": "error",
#                 "error": f"Invalid JSON data: {str(e)}"
#             }))
#         except:
#             pass
    
#     except Exception as e:
#         print(f"[WebSocket] Error: {str(e)}")
#         traceback.print_exc()
        
#         # Mark task as failed
#         if session_id and task_id:
#             task_info = task_manager.get_task(session_id, task_id)
#             if task_info:
#                 task_info.mark_failed(str(e))
        
#         try:
#             await websocket.send_text(json.dumps({
#                 "type": "error",
#                 "task_id": task_id,
#                 "error": str(e)
#             }))
#         except:
#             pass
    
#     finally:
#         # Cleanup
#         if keep_alive_task and not keep_alive_task.done():
#             keep_alive_task.cancel()
#             try:
#                 await keep_alive_task
#             except asyncio.CancelledError:
#                 pass
        
#         if processing_task and not processing_task.done():
#             processing_task.cancel()
#             try:
#                 await processing_task
#             except asyncio.CancelledError:
#                 pass
        
#         print(f"[WebSocket] Connection closed (session: {session_id}, task: {task_id})")




# ============================================================================
# PASTE THIS into your server_with_task_management.py
# Replace the existing @app.websocket("/ws/upload/") handler entirely
# ============================================================================
# Also add this to your uvicorn.run() call:
#   ws_max_size=100 * 1024 * 1024   ← allows 100MB WebSocket messages (safety net)
# ============================================================================

@app.websocket("/ws/upload/")
async def websocket_upload(websocket: WebSocket):
    """
    Real-time streaming WebSocket with chunked binary file transfer.

    Protocol (matches ocrClient.js):
      1. Client sends JSON text: { type: "upload_start", filename, total_chunks, chunk_size, ... }
      2. Client sends N binary ArrayBuffer chunks
      3. Server reassembles → runs OCR → sends page_complete per page
      4. Server sends processing_complete when done
    """
    await websocket.accept()
    print("[WebSocket] Client connected")

    session_id = None
    task_id = None
    keep_alive_task = None
    temp_pdf_path = None

    async def send_keep_alive():
        try:
            while True:
                await asyncio.sleep(20)
                await websocket.send_text(json.dumps({
                    "type": "keep_alive",
                    "timestamp": datetime.now().isoformat()
                }))
        except asyncio.CancelledError:
            pass
        except Exception:
            pass

    try:
        # ── Step 1: Receive metadata (JSON text message) ──────────────────────
        raw_meta = await websocket.receive_text()
        metadata = json.loads(raw_meta)

        if metadata.get("type") != "upload_start":
            # Backwards compat: old single-message base64 format
            # (shouldn't happen with new client but handle gracefully)
            await websocket.send_text(json.dumps({
                "type": "error",
                "error": "Expected upload_start message first"
            }))
            return

        session_id   = metadata.get("session_id")
        filename     = metadata.get("filename", "unknown.pdf")
        lang         = metadata.get("lang", "eng")
        total_bytes  = metadata.get("file_size", 0)
        total_chunks = metadata.get("total_chunks", 1)
        chunk_size   = metadata.get("chunk_size", 1024 * 1024)
        mode         = metadata.get("mode", "single")
        pages_spec   = metadata.get("pages", None)  # ← optional page selection e.g. "3" or "1-5" (blank = all)
        user_schema  = metadata.get("table_schema", None)  # ← user-defined JSON schema from frontend

        print(f"[WebSocket] Upload start: {filename} | "
              f"{total_bytes/(1024*1024):.1f}MB | {total_chunks} chunks | mode: {mode} | "
              f"user_schema: {'YES' if user_schema else 'NO (default)'}")

        # Validate file size
        if total_bytes > MAX_UPLOAD_SIZE:
            await websocket.send_text(json.dumps({
                "type": "error",
                "error": f"File too large ({total_bytes/(1024*1024):.1f}MB). Max is {MAX_UPLOAD_SIZE//(1024*1024)}MB"
            }))
            return

        # ── Step 2: Validate / create session & task ──────────────────────────
        # Validate session
        if session_id and task_manager.touch_session(session_id):
            print(f"[WebSocket] Using session: {session_id}")
        else:
            # Create new session if invalid/missing
            session_id = task_manager.create_session()
            print(f"[WebSocket] Created new session: {session_id}")

        task_id = task_manager.create_task(session_id, filename)
        print(f"[WebSocket] Task created: {task_id}")

        await websocket.send_text(json.dumps({
            "type": "task_created",
            "task_id": task_id,
            "session_id": session_id
        }))

        # ── Step 3: Receive all binary chunks and reassemble ──────────────────
        file_buffer = bytearray()
        chunks_received = 0

        print(f"[WebSocket] Receiving {total_chunks} chunks...")

        while chunks_received < total_chunks:
            message = await asyncio.wait_for(
                websocket.receive(),
                timeout=120  # 2 min per chunk timeout
            )

            if "bytes" in message:
                chunk = message["bytes"]
                file_buffer.extend(chunk)
                chunks_received += 1
                print(f"[WebSocket] Chunk {chunks_received}/{total_chunks} "
                      f"({len(file_buffer)/(1024*1024):.1f}MB received)")
            elif "text" in message:
                # Might be a keep-alive or cancel from client — ignore
                print(f"[WebSocket] Text during upload (ignoring): {message['text'][:100]}")
            else:
                print(f"[WebSocket] Unexpected message format during upload")

        file_bytes = bytes(file_buffer)
        del file_buffer  # Free memory immediately
        print(f"[WebSocket] ✅ Full file received: {len(file_bytes)/(1024*1024):.1f}MB")

        # Save PDF/image to local file system for layout pipeline
        temp_pdf_path = os.path.join(TEMP_DIR, f"temp_{session_id}_uploaded.pdf")
        with open(temp_pdf_path, "wb") as f:
            f.write(file_bytes)
        print(f"[WebSocket] Saved uploaded file to: {temp_pdf_path}")

        await websocket.send_text(json.dumps({
            "type": "upload_received",
            "bytes_received": len(file_bytes),
            "timestamp": datetime.now().isoformat()
        }))

        # ── Step 4: Start keep-alive + OCR streaming ──────────────────────────
        keep_alive_task = asyncio.create_task(send_keep_alive())

        await websocket.send_text(json.dumps({
            "type": "processing_started",
            "filename": filename,
            "timestamp": datetime.now().isoformat()
        }))

        # ── Step 5: Run OCR in thread, stream results per page ────────────────
        page_count = 0
        task_info = task_manager.get_task(session_id, task_id) if (task_id and session_id) else None

        buffered_pages = {}
        doc_rag_pages = {}   # page_num -> {prose, structured}; fed to the RAG DB on completion

        async for page_result in stream_ocr_bytes(file_bytes, mode=mode, pages=pages_spec):
            msg_type = page_result.get("type", "")

            # Skip stream_ocr_bytes's own processing_complete — server sends its own below.
            # Skip page_start — client does not handle this type.
            if msg_type in ("processing_complete", "page_start"):
                await asyncio.sleep(0)
                continue

            if msg_type == "page_complete":
                page_count += 1
                current = page_result.get("page", page_count)
                total   = page_result.get("total_pages", "?")
                # Lift the RAG payload out before the message is sent to the frontend.
                doc_rag_pages[current] = page_result.pop("rag", None)
                page_text = page_result.get("text", "")
                import re
                has_table_marker = bool(re.search(r'\|.*\||<table', page_text, re.IGNORECASE))
                
                if has_table_marker:
                    try:
                        page_text = replace_markdown_tables_with_html(page_text)
                        page_result["text"] = page_text
                    except Exception as e:
                        print(f"[WebSocket] Error converting markdown tables to HTML: {e}")

                buffered_pages[current] = {
                    "text": page_text,
                    "preview": page_result.get("preview")
                }

                # Pre-save the OCR result to a temp file immediately to prevent 404
                # if the user tries to export this page while the remaining pages are still processing.
                if session_id:
                    temp_filepath = os.path.join(TEMP_DIR, f"temp_{session_id}_page_{current}.txt")
                    try:
                        with open(temp_filepath, "w", encoding="utf-8") as f:
                            f.write(page_text)
                        print(f"[WebSocket] Pre-saved OCR temp file for page {current} to: {temp_filepath}")
                    except Exception as fe:
                        print(f"[WebSocket] Error writing pre-save OCR temp file: {fe}")

                # Update task progress and keep session alive
                if task_info:
                    task_info.update_progress(current)
                if session_id:
                    task_manager.touch_session(session_id)

                # Always inject session_id & task_id so the frontend export button
                # has them from the very first page_complete message.
                page_result["session_id"] = session_id
                page_result["task_id"]    = task_id

                print(f"[WebSocket] Streamed base page {current}/{total} → client")

            await websocket.send_text(json.dumps(page_result))
            await asyncio.sleep(0)  # Yield to event loop

        # RAG ingestion — runs HERE, the moment OCR is done and doc_rag_pages is fully
        # populated, NOT after the client-facing streaming below. That streaming raises
        # WebSocketDisconnect the instant the browser closes the socket (navigating to the
        # Corpus tab is enough), which jumped straight to the disconnect handler and
        # skipped ingest — so a fully-OCR'd document silently never reached the knowledge
        # base and the Corpus tab stayed empty. The corpus is the durable artifact; the
        # client stream below is best-effort and must not gate it.
        if os.getenv("RAG_INGEST_ENABLED", "0") == "1":
            try:
                from rag.ingest import ingest_document
                ingest_document(file_bytes, filename, doc_rag_pages)
                print(f"[WebSocket] RAG ingest complete for {filename}")
            except Exception as e:
                print(f"[WebSocket] RAG ingest skipped/failed (non-fatal): {e}")

        # Update client and save temp files for Excel exporter
        print("[WebSocket] Streaming final structured table updates to client...")
        for page_num, info in buffered_pages.items():
            page_text = info["text"]
            
            # Clean *TABLE* markers
            page_text = re.sub(r'\*TABLE\*', '', page_text, flags=re.IGNORECASE)
            
            # Wrap any NAKED <tr> blocks in <table> — without nesting tables inside
            # already-formed ones (that produced the "one table exports as three" bug).
            page_text = wrap_naked_tr(page_text)
            
            has_table = bool(re.search(r'<table[\s\S]*?</table>', page_text, re.IGNORECASE))
            
            # Always write a temp file for every page so the export endpoint
            # never returns 404 — even pages without tables get written so the
            # frontend can always call /export-table/{session_id}/{page_num}.
            if session_id:
                if has_table or '<tr>' in page_text:
                    has_table = True
                temp_filepath = os.path.join(TEMP_DIR, f"temp_{session_id}_page_{page_num}.txt")
                try:
                    with open(temp_filepath, "w", encoding="utf-8") as f:
                        f.write(page_text)
                    if has_table:
                        print(f"[WebSocket] Saved temp file for page {page_num} (has tables).")
                    else:
                        print(f"[WebSocket] Saved temp file for page {page_num} (no tables, text only).")
                except Exception as file_err:
                    print(f"[WebSocket] Error writing temp file for page {page_num}: {file_err}")

            # Send final page_complete with session_id always present
            await websocket.send_text(json.dumps({
                "type": "page_complete",
                "page": page_num,
                "total_pages": page_count,
                "text": page_text,
                "preview": info["preview"],
                "has_table": has_table,
                "session_id": session_id,
                "task_id": task_id,
                "status": "completed"
            }))

        # ── Step 6: Done ──────────────────────────────────────────────────────
        if task_info:
            task_info.mark_completed()

        # (RAG ingestion already ran above, right after the OCR loop — see the comment
        # there for why it must not sit behind the client-facing streaming.)

        await websocket.send_text(json.dumps({
            "type": "processing_complete",
            "total_pages_sent": page_count,
            "timestamp": datetime.now().isoformat()
        }))

        print(f"[WebSocket] Done. {page_count} pages streamed for {filename}")

    except asyncio.TimeoutError:
        print("[WebSocket] Timeout waiting for chunk")
        try:
            await websocket.send_text(json.dumps({
                "type": "error",
                "error": "Upload timeout — no data received within 120 seconds"
            }))
        except Exception:
            pass

    except WebSocketDisconnect:
        print(f"[WebSocket] Client disconnected (session: {session_id}, task: {task_id})")

    except json.JSONDecodeError as e:
        print(f"[WebSocket] Bad JSON: {e}")
        try:
            await websocket.send_text(json.dumps({
                "type": "error", "error": f"Invalid JSON: {str(e)}"
            }))
        except Exception:
            pass

    except Exception as e:
        print(f"[WebSocket] Error: {str(e)}")
        traceback.print_exc()
        try:
            await websocket.send_text(json.dumps({
                "type": "error", "error": str(e)
            }))
        except Exception:
            pass

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


# ============================================================================
# Also update your uvicorn.run() call to add ws_max_size:
# ============================================================================
#
# uvicorn.run(
#     "server_with_task_management:app",
#     host=HOST,
#     port=PORT,
#     reload=True,
#     ws_max_size=100 * 1024 * 1024,   # ← ADD THIS: 100MB WebSocket messages
#     limit_max_request_size=MAX_UPLOAD_SIZE,
#     log_level="info"
# )





# ============================================================================
# CHAT & ANALYSIS ENDPOINTS (Session-aware)
# ============================================================================

@app.post("/chat/")
async def chat_with_document(
    message: str = Form(...),
    extracted_text: str = Form(default=""),
    conversation_history: str = Form(default=""),
    document_name: str = Form(default=""),
    session_id: str = Form(...)
):
    """Chat endpoint with session validation."""
    try:
        # Validate session
        if not task_manager.touch_session(session_id):
            return create_error_response(
                "Invalid or expired session",
                status_code=401
            )
        
        if not message or len(message.strip()) == 0:
            return create_error_response(
                "Message cannot be empty",
                status_code=400
            )
        
        print(f"[CHAT] Session {session_id}: {message[:100]}...")
        
        # Smart truncation of context. Qwen2.5-VL has a large context window, so
        # 3000 chars threw away almost all of a multi-page document. Default to a
        # much larger budget (override with CHAT_CONTEXT_MAX_CHARS).
        chat_context_max = int(os.getenv("CHAT_CONTEXT_MAX_CHARS", "16000"))
        context_text = smart_truncate_text(
            extracted_text,
            max_chars=chat_context_max
        ) if extracted_text else "No document content available."
        
        # Parse conversation history
        history: List[Dict[str, str]] = []
        if conversation_history:
            try:
                parsed_history = json.loads(conversation_history)
                if isinstance(parsed_history, list):
                    history = parsed_history
            except json.JSONDecodeError as e:
                print(f"[CHAT] Warning: Failed to parse conversation_history: {e}")
        
        # Build history text
        history_text = ""
        if history:
            for msg in history[-4:]:
                role = msg.get('role', 'user')
                content = msg.get('content', '')
                history_text += f"{role.capitalize()}: {content}\n"
        
        # Build prompt
        system_prompt = f"""You are an AI Document Assistant.
Document: {document_name if document_name else "Untitled"}
Content: {context_text}

Previous conversation:
{history_text if history_text else "No previous conversation"}

Current question: {message}

Instructions:
1. Answer strictly based on the provided document content.
2. If the question is unrelated to the document, reply: "I can only answer questions related to the provided document."
3. Do not provide coding assistance unless the document is about code.
4. Provide much more detailed, comprehensive answers.
5. If your answer references tabular data or you are extracting information from a table, you MUST format that information as a Markdown table in your response.

Response:"""

        response = run_plain_text_inference(
            prompt=system_prompt,
            max_new_tokens=2000,
            temperature=0.7
        )
        
        print(f"[CHAT] Response generated for session {session_id}")
        
        return {
            "success": True,
            "session_id": session_id,
            "response": response,
            "timestamp": datetime.now().isoformat(),
            "document_context": bool(extracted_text),
            "context_truncated": len(extracted_text) > chat_context_max if extracted_text else False
        }
    
    except Exception as e:
        print(f"[CHAT] Error: {str(e)}")
        traceback.print_exc()
        return create_error_response(str(e))


# ============================================================================
# ADMIN & MONITORING ENDPOINTS
# ============================================================================

@app.get("/")
async def root():
    """Root endpoint with service info."""
    return {
        "service": "OCR Backend with Task Management",
        "version": "3.0.0",
        "model": "qwen2.5-vl",
        "status": "healthy",
        "features": [
            "Session-based task management",
            "Multi-user support",
            "Automatic task cancellation",
            "Resource cleanup"
        ],
        "max_upload_size_mb": MAX_UPLOAD_SIZE // (1024 * 1024)
    }


@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {
        "status": "healthy",
        "device": "cuda" if torch.cuda.is_available() else "cpu",
        "timestamp": datetime.now().isoformat(),
        "gpu_available": torch.cuda.is_available(),
        "gpu_count": torch.cuda.device_count() if torch.cuda.is_available() else 0,
        "task_manager_active": task_manager is not None
    }


@app.get("/admin/tasks")
async def get_all_tasks():
    """Get statistics for all tasks (admin endpoint)."""
    try:
        stats = task_manager.get_stats()
        return {
            "success": True,
            "stats": stats,
            "timestamp": datetime.now().isoformat()
        }
    except Exception as e:
        return create_error_response(str(e))


@app.get("/performance-stats/")
async def performance_stats():
    """Get OCR performance statistics."""
    try:
        stats = get_performance_stats()
        return {
            "success": True,
            "timestamp": datetime.now().isoformat(),
            "stats": stats
        }
    except Exception as e:
        return create_error_response(str(e))


@app.post("/analyze/")
async def analyze_document(
    extracted_text: str = Form(...),
    document_name: str = Form(default=""),
    analysis_type: str = Form(default="summary")
):
    """Quick document analysis endpoint"""
    try:
        analysis_prompts = {
            "summary": "Provide a comprehensive summary of this document.",
            "key_points": "Extract the main key points as a list.",
            "entities": "Identify and list all important entities (names, dates, amounts).",
            "dates": "Find and list all dates and deadlines.",
            "questions": "Suggest 5-7 relevant questions about this document."
        }
        
        prompt_instruction = analysis_prompts.get(analysis_type, analysis_prompts["summary"])
        context_text = extracted_text[:3500] if extracted_text else ""
        
        system_prompt = f"""Analyze the following document and {prompt_instruction}

Document: {document_name}
Content: {context_text}

Analysis:"""

        response = run_plain_text_inference(
            prompt=system_prompt,
            max_new_tokens=600,
            temperature=0.6
        )
        
        return {
            "success": True,
            "analysis": response,
            "analysis_type": analysis_type,
            "timestamp": datetime.now().isoformat(),
            "document_name": document_name
        }
        
    except Exception as e:
        print(f"[ANALYSIS] Error: {str(e)}")
        return create_error_response(str(e))


def _preview_cache_path(content_hash: str, page: int) -> str:
    """Disk cache for rendered page-preview PNGs, keyed on (content_hash, page) — a
    PDF's rendered pixels never change, so a page is rendered via pdftoppm at most
    once total, not once per view. Lives next to rag/pdf_store (same content-addressed
    pattern, no DB column needed). Lives at <OCR_DATA_DIR>/preview_cache — pure cache,
    safe to delete wholesale since it regenerates on next view, unlike its pdf_store
    sibling which holds the only copy of the original upload bytes."""
    from ocr_paths import PREVIEW_CACHE_DIR
    return os.path.join(PREVIEW_CACHE_DIR, f"{content_hash}_{page}.png")


def _parse_doc_ids(raw: str) -> Optional[List[int]]:
    """Parse the comma-separated doc_ids form field sent by the chat UI's "+" document
    picker (e.g. '2,34'). Returns None (== corpus-wide) for empty/absent input; silently
    drops any non-integer token rather than failing the whole request."""
    if not (raw or "").strip():
        return None
    ids = []
    for tok in raw.split(","):
        tok = tok.strip()
        if tok.isdigit():
            ids.append(int(tok))
    return ids or None


@app.get("/rag/documents")
async def rag_documents():
    """
    List documents available for the chat UI's "+" document picker: one entry per
    ingested, ready document, newest first. doc_id is the stable key to send back in
    doc_ids — filenames alone aren't unique (e.g. a re-uploaded/duplicate file).
    """
    try:
        from rag import db
        if not db.available():
            return {"documents": []}
        cols, rows = db.run_readonly(
            "SELECT doc_id, filename, line_name, n_pages, ingested_at FROM documents "
            "WHERE status='ready' ORDER BY ingested_at DESC")
        return {"documents": [
            {"doc_id": r[0], "filename": r[1], "line_name": r[2],
             "n_pages": r[3], "ingested_at": r[4].isoformat() if r[4] else None}
            for r in rows]}
    except Exception as e:
        return create_error_response(f"rag/documents failed: {e}")


@app.get("/rag/document-pages")
async def rag_document_pages(doc_id: str):
    """
    Reconstruct a document's per-page extracted content (tables as HTML + prose text)
    from the RAG knowledge base, so the Corpus tab can show the SAME server-side corpus
    the chatbot/Maker read — not just documents this browser happened to OCR itself.

    `doc_id` here is the document's FILENAME (matches /rag/documents' `filename`), not
    the internal integer id — bound as a query param, never interpolated into SQL,
    since it's user-controlled (an uploaded file's name).

    Returns pages shaped like the OCR pipeline's own results
    ({fileName, pageNumber, extractedText, has_table, has_preview}) so the existing
    PageResult component renders them unmodified. `has_preview` tells the frontend
    whether /rag/document-page-image has a stored PDF to render for this doc — false
    for documents ingested before pdf_store existed (they show CorpusViewer's "no
    preview" state instead).
    """
    try:
        from rag import db
        import table_extractor
        if not db.available():
            return {"pages": []}

        _, doc_rows = db.run_readonly(
            "SELECT doc_id, content_hash FROM documents WHERE filename = %s AND status = 'ready' "
            "ORDER BY ingested_at DESC LIMIT 1",
            params=(doc_id,))
        if not doc_rows:
            return {"pages": []}
        internal_doc_id, content_hash = doc_rows[0]

        from rag.ingest import pdf_store_path
        has_preview = os.path.exists(pdf_store_path(content_hash, doc_id))

        # row_limit above the default 500: that default is a text-to-SQL safety cap,
        # not appropriate for a bulk per-document export — a single document's own
        # table_cells can exceed 500 (e.g. UP67-04-4-7.pdf has 588), which would
        # otherwise silently truncate later pages with no error.
        _, cell_rows = db.run_readonly(
            "SELECT page, block_index, row_index, col_index, col_header, col_group, value_text "
            "FROM table_cells WHERE doc_id = %s "
            "ORDER BY page, block_index, row_index, col_index",
            params=(internal_doc_id,), row_limit=100000)
        _, prose_rows = db.run_readonly(
            "SELECT page, content FROM prose_chunks WHERE doc_id = %s "
            "ORDER BY page, chunk_id",
            params=(internal_doc_id,), row_limit=100000)

        # Reconstruct each page's table the way the paper reads: the side-by-side
        # blocks (block_index 0..g-1) reassembled into ONE wide table with the leaf
        # headers under their spanning title (col_group) — not split into separate
        # flat tables, and not with the spanning header dropped. Uses the reconciled
        # col_header/col_group (consistent across pages after ingest reconciliation)
        # and the same _blocks_to_wide_html the OCR path uses, so the corpus view
        # matches the original layout.
        page_data: Dict[int, Dict[str, Any]] = {}
        for page, block_index, row_index, col_index, col_header, col_group, value_text in cell_rows:
            pd = page_data.setdefault(page, {"cols": {}, "groups": {}, "blocks": {}})
            pd["cols"][col_index] = col_header or ""
            pd["groups"][col_index] = col_group or ""
            pd["blocks"].setdefault(block_index, {}).setdefault(row_index, {})[col_index] = value_text or ""

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
                blocks.append([[brows[r].get(i, "") for i in range(k)] for r in sorted(brows)])
            if not any(any(row) for blk in blocks for row in blk):
                continue
            tables_by_page.setdefault(page, []).append(
                table_extractor._blocks_to_wide_html(headers, blocks, k, spans))

        prose_by_page: Dict[int, List[str]] = {}
        for page, content in prose_rows:
            if content:
                prose_by_page.setdefault(page, []).append(content)

        all_pages = sorted(set(tables_by_page) | set(prose_by_page))
        pages = []
        for page in all_pages:
            prose_html = "".join(
                f"<p>{table_extractor._esc(p)}</p>" for p in prose_by_page.get(page, []))
            tables_html = "".join(tables_by_page.get(page, []))
            pages.append({
                "fileName": doc_id,
                "pageNumber": page,
                "extractedText": prose_html + tables_html,
                "has_table": bool(tables_by_page.get(page)),
                "has_preview": has_preview,
            })
        return {"pages": pages}
    except Exception as e:
        return create_error_response(f"rag/document-pages failed: {e}")


@app.get("/rag/document-page-image")
async def rag_document_page_image(doc_id: str, page: int = 1):
    """
    Render one page of a document's originally-uploaded PDF as a PNG, for the Corpus
    tab's "Original" pane. Looks up the doc's content_hash and reads the bytes
    rag.ingest.ingest_document() persisted under rag/pdf_store/{hash}.pdf — 404s
    cleanly if the document predates that persistence, so an <img> tag falls back
    rather than hanging or throwing in the browser console.

    Two perf fixes (a multi-page document's Corpus view was taking 1+ minute to
    load — 6 concurrent requests measured at 8s total, ~serial, not parallel):
      1. render_pdf_to_base64png shells out to pdftoppm synchronously; called
         directly inside this `async def` it blocks the whole event loop, so
         every other in-flight request (any doc, any page) queues up behind it
         one at a time. Moved onto a worker thread via asyncio.to_thread so
         concurrent page requests actually render in parallel.
      2. A PDF page's rendered pixels never change (content-hash-addressed,
         immutable), but nothing was cached — the same page was re-rendered via
         pdftoppm on every single view. Disk-cached by (content_hash, page) so
         a repeat view is a plain file read; Cache-Control lets the browser
         skip the round-trip entirely on top of that.
    """
    try:
        from rag import db
        from rag.ingest import pdf_store_path
        from local_proc.renderpdf import render_pdf_to_base64png
        if not db.available():
            return create_error_response("rag not configured", status_code=503)

        _, doc_rows = db.run_readonly(
            "SELECT content_hash FROM documents WHERE filename = %s AND status = 'ready' "
            "ORDER BY ingested_at DESC LIMIT 1",
            params=(doc_id,))
        if not doc_rows:
            return create_error_response("document not found", status_code=404)

        content_hash = doc_rows[0][0]
        store_path = pdf_store_path(content_hash, doc_id)
        if not os.path.exists(store_path):
            return create_error_response("no stored original for this document", status_code=404)

        # An image-sourced document (e.g. a scanned .png upload) has no "pages" to
        # render via pdftoppm — it IS the page. Serve it directly rather than routing
        # through render_pdf_to_base64png, which only understands PDF bytes.
        ext = store_path.rsplit(".", 1)[-1].lower()
        if ext != "pdf":
            with open(store_path, "rb") as f:
                file_bytes = f.read()
            media_type = "image/jpeg" if ext in ("jpg", "jpeg") else f"image/{ext}"
            return Response(content=file_bytes, media_type=media_type,
                             headers={"Cache-Control": "public, max-age=31536000, immutable"})

        cache_path = _preview_cache_path(content_hash, page)
        if os.path.exists(cache_path):
            with open(cache_path, "rb") as f:
                png_bytes = f.read()
        else:
            with open(store_path, "rb") as f:
                pdf_bytes = f.read()
            png_b64 = await asyncio.to_thread(
                render_pdf_to_base64png, pdf_bytes, page_number=page, resolution=1288)
            png_bytes = base64.b64decode(png_b64)
            os.makedirs(os.path.dirname(cache_path), exist_ok=True)
            with open(cache_path, "wb") as f:
                f.write(png_bytes)

        return Response(content=png_bytes, media_type="image/png",
                         headers={"Cache-Control": "public, max-age=31536000, immutable"})
    except Exception as e:
        return create_error_response(f"rag/document-page-image failed: {e}")


@app.get("/rag/corpus-chunks")
async def rag_corpus_chunks(doc_ids: str = ""):
    """
    Export the OCR project's RAG corpus (prose + a compact per-row rendering of each
    extracted table) as flat chunks shaped for deckworks' Chunk model — see
    deckworks/src/deckworks/corpus/model.py ({doc_id, page, text, kind, ordinal,
    ingest_version}). This is the bridge that lets the "Maker" tab (deckworks)
    generate from the SAME corpus as the Reader/Thinker, instead of its own
    separate sample corpus — see deckworks' _corpus() in api/app.py.

    doc_id here is the document's FILENAME (a string), not the internal integer
    doc_id — deckworks uses doc_id as a citable, human-meaningful identifier and
    groups/filters by it (e.g. "generate from just this document").

    doc_ids: optional comma-separated internal doc_id list (same scoping as
    /rag-chat/'s "+" picker) to export only selected documents; empty = whole corpus.
    """
    try:
        from rag import db
        if not db.available():
            return {"chunks": []}
        parsed_ids = _parse_doc_ids(doc_ids)

        # Prose: already sentence/paragraph-sized chunks: ordinal = row_number()
        # per document, in read order, so a stale/foreign notion of ordinal
        # doesn't leak in. row_limit above the default 500 (a text-to-SQL safety
        # cap, not appropriate here): the corpus already has 1800+ table cells
        # and grows with every ingest — the default was silently truncating this
        # corpus-wide export to an alphabetical prefix of documents.
        _, prose_rows = db.run_readonly(
            "SELECT d.filename, p.page, p.content, "
            "       row_number() OVER (PARTITION BY p.doc_id ORDER BY p.page, p.chunk_id) - 1 "
            "FROM prose_chunks p JOIN documents d USING (doc_id) "
            "WHERE d.status='ready' AND p.content IS NOT NULL AND p.content <> ''",
            doc_ids=parsed_ids, row_limit=100000)
        chunks = [
            {"doc_id": filename, "page": int(page), "text": content,
             "kind": "body", "ordinal": int(ordinal), "ingest_version": 1}
            for filename, page, content, ordinal in prose_rows
        ]

        # Tables: deckworks only understands text, so render each table's rows as
        # plain "Header: value, Header: value" lines — one chunk per (doc,page,
        # block) — so tabular facts (this project's primary content) stay usable
        # for slide generation instead of being silently dropped.
        _, cell_rows = db.run_readonly(
            "SELECT d.filename, c.page, c.block_index, c.row_index, c.col_header, c.value_text "
            "FROM table_cells c JOIN documents d USING (doc_id) "
            "WHERE d.status='ready' AND c.value_text IS NOT NULL AND c.value_text <> '' "
            "ORDER BY d.filename, c.page, c.block_index, c.row_index, c.col_index",
            doc_ids=parsed_ids, row_limit=100000)
        tables = {}
        for filename, page, block_index, row_index, col_header, value_text in cell_rows:
            key = (filename, page, block_index)
            tables.setdefault(key, {}).setdefault(row_index, []).append((col_header or "", value_text))

        next_ordinal = {}
        for c in chunks:
            next_ordinal[c["doc_id"]] = max(next_ordinal.get(c["doc_id"], 0), c["ordinal"] + 1)

        for (filename, page, _block_index), rows in tables.items():
            lines = [", ".join(f"{h}: {v}" for h, v in cells if h) for _, cells in sorted(rows.items())]
            lines = [line for line in lines if line]
            if not lines:
                continue
            ordinal = next_ordinal.get(filename, 0)
            next_ordinal[filename] = ordinal + 1
            chunks.append({"doc_id": filename, "page": int(page), "text": "\n".join(lines),
                           "kind": "body", "ordinal": ordinal, "ingest_version": 1})

        return {"chunks": chunks}
    except Exception as e:
        return create_error_response(f"rag/corpus-chunks failed: {e}")


# Header-keyword -> plausible unit, purely to satisfy Quantity.unit (min_length=1) with
# something meaningful rather than a bare placeholder. The NUMBER (value_num) is what
# matters for correctness; this is display metadata, not something re-derived from it.
_UNIT_HINTS = (
    ("secs", "ms"), ("sec", "ms"), ("velocity", "m/s"), ("depth", "m"),
    ("pressure", "psi"), ("%", "%"), ("percent", "%"), ("temperature", "°C"),
)


def _guess_unit(header: str) -> str:
    lowered = (header or "").lower()
    for needle, unit in _UNIT_HINTS:
        if needle in lowered:
            return unit
    return "units"


@app.get("/rag/corpus-tables")
async def rag_corpus_tables(doc_ids: str = ""):
    """
    Structured table export for the Maker's numeric pipeline.

    Unlike /rag/corpus-chunks (which flattens every table to plain text for the
    prose-only corpus, discarding the parsed value_num), this keeps each cell's
    parsed number so deckworks can build a real Sourced NumericSlot (schemas/base.py:
    Provenance + Quantity) and chart values FROM value_num directly — the LLM never
    re-parses or transcribes a number from text. See deckworks/src/deckworks/
    corpus/tables.py (the consumer) for how a slide finds and uses a table here.

    doc_id/table_id are strings (filename / "p{page}b{block_index}") so they can
    serve directly as Provenance.doc_id/table_id — citable, human-meaningful keys,
    matching /rag/corpus-chunks' convention.

    doc_ids: optional comma-separated internal doc_id list (same scoping as
    /rag-chat/'s "+" picker) to export only selected documents; empty = whole corpus.
    """
    try:
        from rag import db
        if not db.available():
            return {"tables": []}
        parsed_ids = _parse_doc_ids(doc_ids)

        # row_limit above the default 500 (a text-to-SQL safety cap, not appropriate
        # for a bulk corpus-wide export — see the identical fix on /rag/corpus-chunks).
        _, rows = db.run_readonly(
            "SELECT d.filename, c.page, c.block_index, c.col_group, c.row_index, "
            "       c.col_index, c.col_header, c.value_text, c.value_num "
            "FROM table_cells c JOIN documents d USING (doc_id) "
            "WHERE d.status='ready' "
            "ORDER BY d.filename, c.page, c.block_index, c.row_index, c.col_index",
            doc_ids=parsed_ids, row_limit=100000)

        tables: Dict[tuple, Dict[str, Any]] = {}
        for filename, page, block_index, col_group, row_index, col_index, col_header, value_text, value_num in rows:
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
            result.append({
                "doc_id": filename,
                "page": page,
                "table_id": table_id,
                "title": title,
                "columns": [
                    {"header": h, "numeric": numeric_flags[i], "unit": _guess_unit(h)}
                    for i, h in enumerate(headers)
                ],
                "rows": rows_out,
                "provenance": {"doc_id": filename, "page": page, "table_id": table_id, "tier": "B"},
            })
        return {"tables": result}
    except Exception as e:
        return create_error_response(f"rag/corpus-tables failed: {e}")


@app.get("/rag/corpus-figures")
async def rag_corpus_figures(doc_ids: str = ""):
    """
    Real-figure export for the Maker's media pipeline (visual-upgrade plan, Phase 2A) —
    the deckworks-side counterpart of /rag/corpus-tables. Each entry's caption/context
    is folded into deckworks' BM25 corpus so a slide can retrieve and cite an actual
    document figure (rag/figures.py, extracted at ingest time) instead of fabricating
    or going without one; `image_ref` is fetched via /rag/figure-image.

    doc_id is the filename (string), matching /rag/corpus-tables' and
    /rag/corpus-chunks' convention — citable directly as Provenance.doc_id.
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
            doc_ids=parsed_ids, row_limit=100000)

        return {"figures": [
            {
                "doc_id": filename, "page": page, "caption": caption, "context": context,
                "image_ref": image_ref, "width_px": width_px, "height_px": height_px,
            }
            for filename, page, caption, context, image_ref, width_px, height_px in rows
        ]}
    except Exception as e:
        return create_error_response(f"rag/corpus-figures failed: {e}")


@app.get("/rag/figure-image")
async def rag_figure_image(ref: str):
    """
    Serve one extracted figure's cropped PNG bytes by its content-addressed image_ref
    (see /rag/corpus-figures). Cache-Control mirrors /rag/document-page-image: the
    bytes for a given ref never change, so the browser can skip the round-trip
    entirely after the first fetch.
    """
    try:
        from rag.figures import figure_store_path
        # image_ref is DB-sourced (never user-typed into a path), but validate the
        # shape anyway before touching the filesystem — belt and suspenders.
        if not re.fullmatch(r"[0-9a-f]+_\d+_\d+", ref or ""):
            return create_error_response("invalid figure reference", status_code=400)
        path = figure_store_path(ref)
        if not os.path.exists(path):
            return create_error_response("figure not found", status_code=404)
        with open(path, "rb") as f:
            png_bytes = f.read()
        return Response(content=png_bytes, media_type="image/png",
                         headers={"Cache-Control": "public, max-age=31536000, immutable"})
    except Exception as e:
        return create_error_response(f"rag/figure-image failed: {e}")


@app.post("/rag-chat/")
async def rag_chat(message: str = Form(...), session_id: str = Form(default=""),
                   doc_ids: str = Form(default="")):
    """
    Phase-4 chatbot over the OCR knowledge base: routes the question to the SQL and/or
    prose lane, then narrates the retrieved evidence.

    Unlike the legacy /chat/ (which stuffed raw page text into the model and let it do
    arithmetic), every number here comes from a SQL result row, and the provenance and
    unverified-data warnings are computed from the database — not written by the model.

    doc_ids: optional comma-separated doc_id list from the chat UI's "+" document picker.
    When present, both lanes are scoped to exactly those documents (enforced in the
    database, not just the prompt); empty/absent means corpus-wide, as before.
    """
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


@app.post("/rag/prose")
async def rag_prose(question: str = Form(...), k: int = Form(5), doc_ids: str = Form(default="")):
    """
    Phase-3 semantic lane: top-k prose chunks by cosine similarity (BGE-M3 + pgvector),
    each with provenance (filename/page) and a score. Tabular content belongs to
    /rag/tables, never here.
    """
    try:
        from rag.prose_search import search
        scoped = functools.partial(search, question, k=k, doc_ids=_parse_doc_ids(doc_ids))
        hits = await asyncio.get_event_loop().run_in_executor(None, scoped)
        return {"question": question, "hits": [
            {"score": round(float(h["score"]), 4), "filename": h["filename"],
             "page": h["page"], "section": h["section"], "content": h["content"]}
            for h in hits]}
    except Exception as e:
        return create_error_response(f"rag/prose failed: {e}")


@app.post("/rag/tables")
async def rag_tables(question: str = Form(...), k: int = Form(4), doc_ids: str = Form(default="")):
    """
    Tables lane, retrieval only: the top-k source tables as their original HTML, layout
    intact (rowspan / colspan / <br> multi-values). Mirrors /rag/prose so the two lanes
    can be inspected the same way; /rag-chat/ is what actually narrates them.

    Replaces the retired /rag/sql, which asked the model to write a SELECT over the
    flattened table_cells store. That store cannot represent this shape, and most
    documents here have no structured cells at all. `html` is returned verbatim — it is
    exactly what the narration model is shown.

    doc_ids: optional comma-separated doc_id list to scope retrieval (see /rag-chat/).
    """
    try:
        from rag.table_search import search
        scoped = functools.partial(search, question, k=k, doc_ids=_parse_doc_ids(doc_ids))
        hits = await asyncio.get_event_loop().run_in_executor(None, scoped)
        return {"question": question, "hits": [
            {"score": round(float(h.get("score") or 0.0), 4), "match": h.get("match"),
             "filename": h["filename"], "page": h["page"],
             "table_index": h["table_index"], "caption": h.get("caption"),
             "n_rows": h.get("n_rows"), "n_cols": h.get("n_cols"),
             "est_tokens": h.get("est_tokens"), "well_formed": h.get("well_formed"),
             "html": h.get("html")}
            for h in hits]}
    except Exception as e:
        return create_error_response(f"rag/tables failed: {e}")


# ============================================================================
# SERVER STARTUP
# ============================================================================

if __name__ == "__main__":
    import uvicorn
    
    print("=" * 70)
    print("🚀 OCR Server with Task Management")
    print(f"📍 Host: {HOST}:{PORT}")
    print(f"📊 Max upload size: {MAX_UPLOAD_SIZE // (1024*1024)}MB")
    print(f"🎮 GPU available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"🎮 GPU count: {torch.cuda.device_count()}")
    print(f"🔧 Task timeout: {task_manager.task_timeout}s")
    print(f"🧹 Cleanup interval: {task_manager.cleanup_interval}s")
    print("=" * 70)
    

    uvicorn.run(
        "server_with_task_management:app",
        host=HOST,
        port=PORT,
        reload=os.getenv("OCR_RELOAD", "0") == "1",
        ws_max_size=MAX_UPLOAD_SIZE,   # match the 500MB upload size limit
        log_level="info"
    )
