# task_manager.py - Task Lifecycle Management System
"""
Task Manager for OCR Processing
- Handles task lifecycle (create, track, cancel, cleanup)
- Session-based isolation for multi-user support
- Automatic cleanup on disconnect
- Thread-safe operations
"""

import asyncio
import os
import glob
import uuid
import time
from typing import Dict, Optional, Set, Any, Callable
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
import threading
import weakref
import logging

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Temp files live in TEMP_DIR (<OCR_DATA_DIR>/temp) — outside the repo and NOT in the
# process CWD (which is the project root when launched via run_project.sh). Imported from
# ocr_paths rather than recomputed so this sweep and the server's writers are guaranteed
# to resolve to the same directory; a mismatch here means files are written but never
# reaped, which shows up only as unbounded disk growth.
from ocr_paths import TEMP_DIR


class TaskStatus(Enum):
    """Task lifecycle states."""
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    CANCELLED = "cancelled"
    FAILED = "failed"


@dataclass
class TaskInfo:
    """
    Information about a single OCR task.
    Thread-safe task tracking with automatic cleanup.
    """
    task_id: str
    session_id: str
    filename: str
    total_pages: Optional[int] = None
    processed_pages: int = 0
    status: TaskStatus = TaskStatus.PENDING
    created_at: datetime = field(default_factory=datetime.now)
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    error: Optional[str] = None
    
    # Async task reference (weakref for auto-cleanup)
    _async_task: Optional[asyncio.Task] = field(default=None, repr=False)
    
    # Cancellation event
    _cancel_event: asyncio.Event = field(default_factory=asyncio.Event, repr=False)
    
    # Lock for thread safety
    _lock: threading.Lock = field(default_factory=threading.Lock, repr=False)
    
    def mark_started(self, total_pages: int):
        """Mark task as started."""
        with self._lock:
            self.status = TaskStatus.RUNNING
            self.started_at = datetime.now()
            self.total_pages = total_pages
    
    def update_progress(self, page_num: int):
        """Update processing progress."""
        with self._lock:
            self.processed_pages = page_num
    
    def mark_completed(self):
        """Mark task as completed."""
        with self._lock:
            self.status = TaskStatus.COMPLETED
            self.completed_at = datetime.now()
    
    def mark_failed(self, error: str):
        """Mark task as failed."""
        with self._lock:
            self.status = TaskStatus.FAILED
            self.error = error
            self.completed_at = datetime.now()
    
    def mark_cancelled(self):
        """Mark task as cancelled."""
        with self._lock:
            self.status = TaskStatus.CANCELLED
            self.completed_at = datetime.now()
            self._cancel_event.set()
    
    def is_cancelled(self) -> bool:
        """Check if task is cancelled."""
        return self._cancel_event.is_set()
    
    def get_progress(self) -> Dict[str, Any]:
        """Get current progress information."""
        with self._lock:
            return {
                "task_id": self.task_id,
                "status": self.status.value,
                "processed_pages": self.processed_pages,
                "total_pages": self.total_pages,
                "progress_percent": (
                    (self.processed_pages / self.total_pages * 100) 
                    if self.total_pages else 0
                ),
                "filename": self.filename,
                "created_at": self.created_at.isoformat(),
                "duration_seconds": (
                    (datetime.now() - self.started_at).total_seconds()
                    if self.started_at else 0
                )
            }


class TaskManager:
    """
    Centralized task management with session isolation.
    
    Features:
    - Session-based task isolation (multi-user support)
    - Automatic cleanup on session disconnect
    - Task cancellation with resource cleanup
    - Thread-safe operations
    - Orphan task detection and cleanup
    """
    
    def __init__(self, cleanup_interval: int = 60, task_timeout: int = 3600):
        """
        Initialize task manager.
        
        Args:
            cleanup_interval: Seconds between cleanup runs (default: 60)
            task_timeout: Seconds before abandoned tasks are cleaned up (default: 3600)
        """
        # Session-based task storage
        # Format: {session_id: {task_id: TaskInfo}}
        self._tasks: Dict[str, Dict[str, TaskInfo]] = {}
        
        # Active sessions with last activity time
        # Format: {session_id: last_activity_timestamp}
        self._sessions: Dict[str, float] = {}
        
        # Lock for thread-safe operations
        self._lock = threading.RLock()
        
        # Configuration
        self.cleanup_interval = cleanup_interval
        self.task_timeout = task_timeout
        
        # Cleanup task
        self._cleanup_task: Optional[asyncio.Task] = None
        self._shutdown = False
        
        logger.info(f"TaskManager initialized (cleanup: {cleanup_interval}s, timeout: {task_timeout}s)")
    
    def create_session(self) -> str:
        """
        Create a new session for a user.
        
        Returns:
            session_id: Unique session identifier
        """
        session_id = str(uuid.uuid4())
        
        with self._lock:
            self._tasks[session_id] = {}
            self._sessions[session_id] = time.time()
        
        logger.info(f"Created session: {session_id}")
        return session_id
    
    def touch_session(self, session_id: str) -> bool:
        """
        Update session activity timestamp.
        
        Args:
            session_id: Session to update
            
        Returns:
            True if session exists, False otherwise
        """
        with self._lock:
            if session_id in self._sessions:
                self._sessions[session_id] = time.time()
                return True
        return False
    
    def create_task(
        self,
        session_id: str,
        filename: str,
        cancel_existing: bool = True
    ) -> str:
        """
        Create a new task for a session.
        
        Args:
            session_id: Session owning this task
            filename: Name of file being processed
            cancel_existing: If True, cancel any existing tasks for this session
            
        Returns:
            task_id: Unique task identifier
        """
        task_id = str(uuid.uuid4())
        
        with self._lock:
            # Ensure session exists
            if session_id not in self._tasks:
                self._tasks[session_id] = {}
                self._sessions[session_id] = time.time()
            
            # Cancel existing tasks if requested
            if cancel_existing:
                self._cancel_session_tasks(session_id)
            
            # Create new task
            task_info = TaskInfo(
                task_id=task_id,
                session_id=session_id,
                filename=filename
            )
            
            self._tasks[session_id][task_id] = task_info
            self._sessions[session_id] = time.time()
        
        logger.info(f"Created task {task_id} for session {session_id} (file: {filename})")
        return task_id
    
    def get_task(self, session_id: str, task_id: str) -> Optional[TaskInfo]:
        """
        Get task information.
        
        Args:
            session_id: Session ID
            task_id: Task ID
            
        Returns:
            TaskInfo if found, None otherwise
        """
        with self._lock:
            session_tasks = self._tasks.get(session_id, {})
            return session_tasks.get(task_id)
    
    def set_async_task(self, session_id: str, task_id: str, async_task: asyncio.Task):
        """
        Associate an asyncio Task with a tracked task.
        
        Args:
            session_id: Session ID
            task_id: Task ID
            async_task: The asyncio.Task to track
        """
        with self._lock:
            task_info = self.get_task(session_id, task_id)
            if task_info:
                task_info._async_task = async_task
                logger.debug(f"Associated async task with {task_id}")
    
    def cancel_task(self, session_id: str, task_id: str) -> bool:
        """
        Cancel a specific task.
        
        Args:
            session_id: Session ID
            task_id: Task ID
            
        Returns:
            True if task was cancelled, False if not found
        """
        with self._lock:
            task_info = self.get_task(session_id, task_id)
            if not task_info:
                logger.warning(f"Task {task_id} not found for cancellation")
                return False
            
            # Mark as cancelled
            task_info.mark_cancelled()
            
            # Cancel async task if it exists
            if task_info._async_task and not task_info._async_task.done():
                task_info._async_task.cancel()
                logger.info(f"Cancelled async task {task_id}")
            
            logger.info(f"Cancelled task {task_id}")
            return True
    
    def _cancel_session_tasks(self, session_id: str):
        """
        Cancel all tasks for a session (internal, already locked).
        
        Args:
            session_id: Session ID
        """
        session_tasks = self._tasks.get(session_id, {})
        
        for task_id, task_info in session_tasks.items():
            if task_info.status in [TaskStatus.PENDING, TaskStatus.RUNNING]:
                task_info.mark_cancelled()
                
                if task_info._async_task and not task_info._async_task.done():
                    task_info._async_task.cancel()
                
                logger.info(f"Cancelled task {task_id} in session cleanup")
    
    def cancel_session(self, session_id: str) -> int:
        """
        Cancel all tasks for a session and remove session.
        
        Args:
            session_id: Session to cancel
            
        Returns:
            Number of tasks cancelled
        """
        with self._lock:
            if session_id not in self._tasks:
                return 0
            
            # Cancel all tasks
            self._cancel_session_tasks(session_id)
            
            task_count = len(self._tasks[session_id])
            
            # Remove session
            del self._tasks[session_id]
            if session_id in self._sessions:
                del self._sessions[session_id]
            
            logger.info(f"Removed session {session_id} ({task_count} tasks)")
            
            # Clean up associated temp files (page text + Excel exports).
            # Glob against TEMP_DIR — the same constant the server writes with.
            try:
                patterns = [
                    f"temp_{session_id}_page_*",      # per-page OCR text + per-page xlsx
                    f"temp_{session_id}_all_export.xlsx",  # combined export
                    f"temp_{session_id}_uploaded.pdf",     # uploaded source (if left behind)
                ]
                for pattern in patterns:
                    for f in glob.glob(os.path.join(TEMP_DIR, pattern)):
                        try:
                            os.remove(f)
                        except Exception as e:
                            logger.warning(f"Failed to remove temp file {f}: {e}")
            except Exception:
                pass
            return task_count
    
    def get_session_tasks(self, session_id: str) -> Dict[str, Dict[str, Any]]:
        """
        Get all tasks for a session.
        
        Args:
            session_id: Session ID
            
        Returns:
            Dictionary of task_id -> task progress info
        """
        with self._lock:
            session_tasks = self._tasks.get(session_id, {})
            return {
                task_id: task_info.get_progress()
                for task_id, task_info in session_tasks.items()
            }
    
    def get_stats(self) -> Dict[str, Any]:
        """
        Get overall task manager statistics.
        
        Returns:
            Statistics dictionary
        """
        with self._lock:
            total_tasks = sum(len(tasks) for tasks in self._tasks.values())
            
            status_counts = {status.value: 0 for status in TaskStatus}
            for session_tasks in self._tasks.values():
                for task_info in session_tasks.values():
                    status_counts[task_info.status.value] += 1
            
            return {
                "active_sessions": len(self._sessions),
                "total_tasks": total_tasks,
                "tasks_by_status": status_counts,
                "sessions": {
                    session_id: {
                        "task_count": len(self._tasks[session_id]),
                        "last_activity": datetime.fromtimestamp(
                            self._sessions[session_id]
                        ).isoformat()
                    }
                    for session_id in self._sessions
                }
            }
    
    async def cleanup_orphaned_tasks(self):
        """
        Cleanup loop that runs periodically to remove abandoned tasks.
        Runs in background, automatically cleaning up:
        - Sessions inactive for > task_timeout
        - Completed/failed tasks older than 1 hour
        """
        logger.info("Starting cleanup task")
        
        while not self._shutdown:
            try:
                await asyncio.sleep(self.cleanup_interval)
                
                now = time.time()
                sessions_to_remove = []
                
                with self._lock:
                    # Find inactive sessions
                    for session_id, last_activity in self._sessions.items():
                        if now - last_activity > self.task_timeout:
                            sessions_to_remove.append(session_id)
                
                # Remove inactive sessions (outside lock)
                for session_id in sessions_to_remove:
                    cancelled = self.cancel_session(session_id)
                    logger.info(f"Cleaned up inactive session {session_id} ({cancelled} tasks)")
                
                # Clean up old completed tasks
                with self._lock:
                    for session_id, session_tasks in list(self._tasks.items()):
                        tasks_to_remove = []
                        
                        for task_id, task_info in session_tasks.items():
                            # Remove completed/failed tasks older than 1 hour
                            if task_info.status in [TaskStatus.COMPLETED, TaskStatus.FAILED]:
                                if task_info.completed_at:
                                    age = datetime.now() - task_info.completed_at
                                    if age > timedelta(hours=1):
                                        tasks_to_remove.append(task_id)
                        
                        # Remove old tasks
                        for task_id in tasks_to_remove:
                            del session_tasks[task_id]
                            logger.debug(f"Removed old task {task_id}")
                
            except asyncio.CancelledError:
                logger.info("Cleanup task cancelled")
                break
            except Exception as e:
                logger.error(f"Error in cleanup task: {e}", exc_info=True)
    
    def start_cleanup(self):
        """Start the background cleanup task."""
        if self._cleanup_task is None or self._cleanup_task.done():
            self._cleanup_task = asyncio.create_task(self.cleanup_orphaned_tasks())
            logger.info("Cleanup task started")
    
    async def shutdown(self):
        """Shutdown task manager and cleanup all resources."""
        logger.info("Shutting down TaskManager")
        self._shutdown = True
        
        # Cancel cleanup task
        if self._cleanup_task and not self._cleanup_task.done():
            self._cleanup_task.cancel()
            try:
                await self._cleanup_task
            except asyncio.CancelledError:
                pass
        
        # Cancel all active tasks
        with self._lock:
            for session_id in list(self._tasks.keys()):
                self.cancel_session(session_id)
        
        logger.info("TaskManager shutdown complete")


# Global task manager instance
_task_manager: Optional[TaskManager] = None
_manager_lock = threading.Lock()


def get_task_manager() -> TaskManager:
    """
    Get or create the global task manager instance.
    Thread-safe singleton pattern.
    
    Returns:
        TaskManager instance
    """
    global _task_manager
    
    if _task_manager is None:
        with _manager_lock:
            if _task_manager is None:
                _task_manager = TaskManager(
                    cleanup_interval=60,  # Cleanup every 60 seconds
                    task_timeout=1800     # Session timeout: 30 minutes
                )
                # Note: cleanup will be started when event loop is available
    
    return _task_manager


def reset_task_manager():
    """Reset the global task manager (for testing)."""
    global _task_manager
    with _manager_lock:
        if _task_manager:
            # Cleanup will happen on next event loop access
            _task_manager._shutdown = True
        _task_manager = None


# Context manager for task lifecycle
class TaskContext:
    """
    Context manager for safe task execution with automatic cleanup.
    
    Usage:
        async with TaskContext(session_id, task_id) as task_info:
            if task_info.is_cancelled():
                return
            # Do work...
    """
    
    def __init__(self, session_id: str, task_id: str):
        self.session_id = session_id
        self.task_id = task_id
        self.task_info: Optional[TaskInfo] = None
        self.manager = get_task_manager()
    
    async def __aenter__(self) -> TaskInfo:
        """Enter context - get task info."""
        self.task_info = self.manager.get_task(self.session_id, self.task_id)
        if not self.task_info:
            raise ValueError(f"Task {self.task_id} not found")
        return self.task_info
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Exit context - cleanup on error."""
        if exc_type is not None and self.task_info:
            # Mark as failed if exception occurred
            if exc_type != asyncio.CancelledError:
                self.task_info.mark_failed(str(exc_val))
            else:
                self.task_info.mark_cancelled()
        return False  # Don't suppress exceptions
