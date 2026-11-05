"""
app/__init__.py — FastAPI application factory.

This is the single place that assembles the app:
  1. Creates the FastAPI instance with the startup/shutdown lifespan.
  2. Attaches CORS middleware.
  3. Registers all routers.

Nothing else belongs here.  Business logic, config, and utilities each live
in their own dedicated modules.
"""

from fastapi import FastAPI

from app.lifespan import lifespan
from app.middleware import configure_cors
from app.routers import admin, chat, export, rag_corpus, sessions, upload

app = FastAPI(
    title="OCR Service with Task Management",
    version="3.0.0",
    lifespan=lifespan,
)

# Middleware (must be added before routers are registered).
configure_cors(app)

# Routers — order matters only for documentation grouping, not routing.
app.include_router(admin.router)
app.include_router(sessions.router)
app.include_router(upload.router)
app.include_router(export.router)
app.include_router(chat.router)
app.include_router(rag_corpus.router)
