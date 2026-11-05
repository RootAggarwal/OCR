"""
app/utils/system_info.py — Hardware utilization queries.

Used by the /system-stats endpoint.  Kept separate so the rest of the app
does not depend on subprocess calls or psutil.
"""

import subprocess
from typing import Any, Dict


def get_gpu_usage() -> Dict[str, Any]:
    """
    Query NVIDIA GPU utilization via nvidia-smi.

    Returns a dict with keys:
        gpu_util        — GPU core utilization (%)
        gpu_mem_used    — Used VRAM (MiB)
        gpu_mem_total   — Total VRAM (MiB)

    Returns zeros for all fields when nvidia-smi is unavailable or times out
    (e.g. no NVIDIA GPU, driver not installed, CPU-only machine).
    """
    try:
        result = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=utilization.gpu,memory.used,memory.total",
                "--format=csv,noheader,nounits",
            ],
            capture_output=True,
            text=True,
            timeout=2,
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
