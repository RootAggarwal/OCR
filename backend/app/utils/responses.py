"""
app/utils/responses.py — Standardized HTTP responses and request validation.
"""

import re
from datetime import datetime
from typing import Any, Dict, Optional

from fastapi import WebSocket
from fastapi.responses import JSONResponse

from app.config import SAFE_SESSION_RE


# ---------------------------------------------------------------------------
# Error responses
# ---------------------------------------------------------------------------

def create_error_response(
    error: str,
    status_code: int = 500,
    details: Optional[Dict[str, Any]] = None,
) -> JSONResponse:
    """Return a standardized JSON error response."""
    return JSONResponse(
        status_code=status_code,
        content={
            "success": False,
            "error": error,
            "timestamp": datetime.now().isoformat(),
            "details": details or {},
        },
    )


# ---------------------------------------------------------------------------
# Session / ID validation
# ---------------------------------------------------------------------------

def is_safe_session_id(session_id: str) -> bool:
    """
    Return True only for session IDs safe to embed in a filesystem path.

    Session IDs are UUIDs.  Restricting to a safe charset prevents a crafted
    session_id from traversing out of TEMP_DIR (e.g. "../../etc/passwd").
    """
    return bool(session_id) and bool(SAFE_SESSION_RE.match(session_id))


def extract_session_id(websocket: WebSocket) -> Optional[str]:
    """
    Extract a session ID from WebSocket query params or headers.

    Checked in order:
    1. ?session_id=<value> query parameter
    2. x-session-id request header
    """
    session_id = websocket.query_params.get("session_id")
    if session_id:
        return session_id

    session_id = websocket.headers.get("x-session-id")
    if session_id:
        return session_id

    return None
