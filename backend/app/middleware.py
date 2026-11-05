"""
app/middleware.py — Middleware configuration for the OCR backend.

Called once during app assembly in app/__init__.py.
"""

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.config import ALLOWED_ORIGINS


def configure_cors(app: FastAPI) -> None:
    """
    Attach CORS middleware to *app*.

    NOTE: a wildcard origin ("*") cannot be combined with
    allow_credentials=True — browsers reject credentialed responses whose
    Access-Control-Allow-Origin is "*".  Since this API is token-less and
    does not rely on cookies, credentials are disabled when the wildcard is
    active.  Set OCR_ALLOWED_ORIGINS (comma-separated) to lock this down in
    production (which also re-enables credentials automatically).
    """
    app.add_middleware(
        CORSMiddleware,
        allow_origins=ALLOWED_ORIGINS,
        allow_credentials=ALLOWED_ORIGINS != ["*"],
        allow_methods=["*"],
        allow_headers=["*"],
    )
