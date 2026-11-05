"""
app/routers/sessions.py — Session and task lifecycle endpoints.

Endpoints
---------
POST   /session/create                          — create a new user session
GET    /session/{session_id}/tasks              — list all tasks in a session
DELETE /session/{session_id}                    — cancel all tasks & delete session
GET    /session/{session_id}/task/{task_id}/status  — query one task's progress
DELETE /session/{session_id}/task/{task_id}     — cancel one task
"""

from datetime import datetime

from fastapi import APIRouter

from app.utils.responses import create_error_response
from task_manager import get_task_manager

router = APIRouter(tags=["Sessions"])


# ---------------------------------------------------------------------------
# Session management
# ---------------------------------------------------------------------------

@router.post("/session/create")
async def create_session():
    """Create a new user session.  Returns a session_id for all future requests."""
    try:
        task_manager = get_task_manager()
        session_id = task_manager.create_session()
        return {
            "success": True,
            "session_id": session_id,
            "timestamp": datetime.now().isoformat(),
            "message": "Session created successfully",
        }
    except Exception as e:
        print(f"[ERROR] Failed to create session: {e}")
        return create_error_response(str(e))


@router.get("/session/{session_id}/tasks")
async def get_session_tasks(session_id: str):
    """Return all tasks associated with *session_id*."""
    try:
        task_manager = get_task_manager()
        tasks = task_manager.get_session_tasks(session_id)
        return {
            "success": True,
            "session_id": session_id,
            "tasks": tasks,
            "timestamp": datetime.now().isoformat(),
        }
    except Exception as e:
        print(f"[ERROR] Failed to get session tasks: {e}")
        return create_error_response(str(e))


@router.delete("/session/{session_id}")
async def delete_session(session_id: str):
    """Cancel all tasks for *session_id* and remove the session."""
    try:
        task_manager = get_task_manager()
        cancelled_count = task_manager.cancel_session(session_id)
        return {
            "success": True,
            "session_id": session_id,
            "cancelled_tasks": cancelled_count,
            "timestamp": datetime.now().isoformat(),
            "message": f"Session deleted, {cancelled_count} tasks cancelled",
        }
    except Exception as e:
        print(f"[ERROR] Failed to delete session: {e}")
        return create_error_response(str(e))


# ---------------------------------------------------------------------------
# Task management
# ---------------------------------------------------------------------------

@router.get("/session/{session_id}/task/{task_id}/status")
async def get_task_status(session_id: str, task_id: str):
    """Return progress information for a single task."""
    try:
        task_manager = get_task_manager()
        task_info = task_manager.get_task(session_id, task_id)
        if not task_info:
            return create_error_response(
                f"Task {task_id} not found", status_code=404
            )
        return {
            "success": True,
            "task": task_info.get_progress(),
            "timestamp": datetime.now().isoformat(),
        }
    except Exception as e:
        print(f"[ERROR] Failed to get task status: {e}")
        return create_error_response(str(e))


@router.delete("/session/{session_id}/task/{task_id}")
async def cancel_task(session_id: str, task_id: str):
    """Cancel a specific task."""
    try:
        task_manager = get_task_manager()
        success = task_manager.cancel_task(session_id, task_id)
        if success:
            return {
                "success": True,
                "message": f"Task {task_id} cancelled",
                "timestamp": datetime.now().isoformat(),
            }
        return create_error_response(
            f"Task {task_id} not found or already completed", status_code=404
        )
    except Exception as e:
        print(f"[ERROR] Failed to cancel task: {e}")
        return create_error_response(str(e))
