"""
app/routers/admin.py — Health, monitoring, and admin endpoints.

Endpoints
---------
GET /                  — service info
GET /health            — health check (GPU availability, task manager)
GET /system-stats      — live CPU / RAM / GPU utilization
GET /performance-stats/ — OCR model performance counters
GET /admin/tasks       — task manager statistics (all sessions)
"""

from datetime import datetime

import psutil
import torch
from fastapi import APIRouter
from fastapi.responses import JSONResponse

from app.utils.responses import create_error_response
from app.utils.system_info import get_gpu_usage
from app.config import MAX_UPLOAD_SIZE
from ocr_olm import get_performance_stats
from task_manager import get_task_manager

router = APIRouter(tags=["Admin / Monitoring"])


@router.get("/")
async def root():
    """Root endpoint — returns service identity and feature list."""
    return {
        "service": "OCR Backend with Task Management",
        "version": "3.0.0",
        "model": "qwen2.5-vl",
        "status": "healthy",
        "features": [
            "Session-based task management",
            "Multi-user support",
            "Automatic task cancellation",
            "Resource cleanup",
        ],
        "max_upload_size_mb": MAX_UPLOAD_SIZE // (1024 * 1024),
    }


@router.get("/health")
async def health_check():
    """Health check — reports device availability and task manager state."""
    return {
        "status": "healthy",
        "device": "cuda" if torch.cuda.is_available() else "cpu",
        "timestamp": datetime.now().isoformat(),
        "gpu_available": torch.cuda.is_available(),
        "gpu_count": torch.cuda.device_count() if torch.cuda.is_available() else 0,
        "task_manager_active": get_task_manager() is not None,
    }


@router.get("/system-stats")
async def system_stats():
    """Live CPU, RAM, and GPU utilization snapshot."""
    cpu = psutil.cpu_percent(interval=0.3)
    mem = psutil.virtual_memory()
    gpu = get_gpu_usage()

    return JSONResponse(
        {
            "cpu": round(cpu),
            "gpu": gpu["gpu_util"],
            "mem": round(mem.percent),
            "mem_used_gb": round(mem.used / (1024**3), 1),
            "mem_total_gb": round(mem.total / (1024**3), 1),
            "gpu_mem_used_mb": gpu["gpu_mem_used"],
            "gpu_mem_total_mb": gpu["gpu_mem_total"],
        }
    )


@router.get("/performance-stats/")
async def performance_stats():
    """OCR model throughput and timing statistics."""
    try:
        stats = get_performance_stats()
        return {
            "success": True,
            "timestamp": datetime.now().isoformat(),
            "stats": stats,
        }
    except Exception as e:
        return create_error_response(str(e))


@router.get("/admin/tasks")
async def get_all_tasks():
    """Task manager statistics across all active sessions (admin endpoint)."""
    try:
        task_manager = get_task_manager()
        stats = task_manager.get_stats()
        return {
            "success": True,
            "stats": stats,
            "timestamp": datetime.now().isoformat(),
        }
    except Exception as e:
        return create_error_response(str(e))
