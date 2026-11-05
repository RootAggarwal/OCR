"""
app/config.py — Central configuration for the OCR backend.

All runtime constants and environment-variable reads live here so they
are defined exactly once and can be imported by any module that needs
them.  No business logic belongs here.
"""

import os
import re

# ---------------------------------------------------------------------------
# Server / network
# ---------------------------------------------------------------------------

HOST: str = os.getenv("OCR_HOST", "0.0.0.0")
PORT: int = int(os.getenv("OCR_PORT", "8000"))

# Maximum accepted file payload (default 500 MB).
MAX_UPLOAD_SIZE: int = int(os.getenv("MAX_UPLOAD_SIZE", str(500 * 1024 * 1024)))

# How often the WebSocket keep-alive ping fires (seconds).
WEBSOCKET_KEEPALIVE_INTERVAL: int = 30

# ---------------------------------------------------------------------------
# File-system paths
# ---------------------------------------------------------------------------

# TEMP_DIR is the single, canonical location for all session temp files.
# Imported from ocr_paths so this module and task_manager.py always agree
# on the same directory — a mismatch causes orphaned files that are never
# cleaned up.
from ocr_paths import TEMP_DIR  # noqa: E402 (after stdlib imports)

# ---------------------------------------------------------------------------
# CORS
# ---------------------------------------------------------------------------

# Set OCR_ALLOWED_ORIGINS to a comma-separated list to restrict in production.
# The wildcard ("*") cannot be combined with allow_credentials=True (browsers
# reject that); since this API is token-less we drop credentials and keep "*".
_allowed_origins_env: str = os.getenv("OCR_ALLOWED_ORIGINS", "*").strip()

ALLOWED_ORIGINS: list = (
    ["*"]
    if _allowed_origins_env == "*"
    else [o.strip() for o in _allowed_origins_env.split(",") if o.strip()]
)

# ---------------------------------------------------------------------------
# Security
# ---------------------------------------------------------------------------

# Session IDs are UUIDs.  Allow only a safe filename charset so a crafted
# session_id can never path-traverse out of TEMP_DIR (e.g. "../../etc/passwd").
SAFE_SESSION_RE = re.compile(r"^[A-Za-z0-9_-]{1,64}$")

# ---------------------------------------------------------------------------
# Chat / LLM
# ---------------------------------------------------------------------------

# Default context-window budget for the /chat/ endpoint (chars of OCR text).
# Qwen2.5-VL has a large context window; 3000 chars discarded most of a
# multi-page document.  Override with CHAT_CONTEXT_MAX_CHARS.
CHAT_CONTEXT_MAX_CHARS: int = int(os.getenv("CHAT_CONTEXT_MAX_CHARS", "16000"))

# ---------------------------------------------------------------------------
# RAG / corpus
# ---------------------------------------------------------------------------

# RAG_INGEST_ENABLED=1 activates document ingestion into Postgres after each
# OCR run.  Off by default so the app works without a DB.
RAG_INGEST_ENABLED: bool = os.getenv("RAG_INGEST_ENABLED", "0") == "1"

# Unit-hint table used by /rag/corpus-tables to annotate numeric columns.
# Tuples of (header-keyword, unit-string) — first match wins.
UNIT_HINTS: tuple = (
    ("secs", "ms"),
    ("sec", "ms"),
    ("velocity", "m/s"),
    ("depth", "m"),
    ("pressure", "psi"),
    ("%", "%"),
    ("percent", "%"),
    ("temperature", "°C"),
)
