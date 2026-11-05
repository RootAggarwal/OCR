"""
app/lifespan.py — FastAPI startup / shutdown lifecycle handler.

Using the modern @asynccontextmanager lifespan pattern (replaces the
deprecated @app.on_event decorators).
"""

from contextlib import asynccontextmanager

from fastapi import FastAPI

from task_manager import get_task_manager


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Start background cleanup on startup; shut down gracefully on exit."""
    task_manager = get_task_manager()

    print("[STARTUP] Starting task manager cleanup...")
    task_manager.start_cleanup()
    print("[STARTUP] Server ready")

    yield

    print("[SHUTDOWN] Cleaning up task manager...")
    await task_manager.shutdown()
    print("[SHUTDOWN] Server stopped")
