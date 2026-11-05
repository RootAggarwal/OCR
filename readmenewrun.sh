#!/usr/bin/env bash
# ==============================================================================
# OCR Project - Fast Startup & Management Script
# ==============================================================================
# Usage:
#   ./readmenewrun.sh          # Start all services (Postgres, Frontend container, Backend)
#   ./readmenewrun.sh stop     # Stop all project services & free ports
# ==============================================================================

set -e

PROJECT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd "$PROJECT_DIR"

BACKEND_PORT=8000
FRONTEND_PORT=3001
PG_PORT=5432

PG_CONTAINER="ocr-rag-pg"
FRONTEND_CONTAINER="ocr-frontend-dev"
MODELS_DIR="C:/OCR-models"
DATA_DIR="C:/OCR-data"
LOG_DIR="$DATA_DIR/logs"

# Detect Python in virtual environment
if [ -f "$PROJECT_DIR/.venv/Scripts/python.exe" ]; then
    VENV_PY="$PROJECT_DIR/.venv/Scripts/python.exe"
elif [ -f "$PROJECT_DIR/.venv/bin/python" ]; then
    VENV_PY="$PROJECT_DIR/.venv/bin/python"
else
    echo "❌ Error: Virtual environment Python not found in $PROJECT_DIR/.venv"
    exit 1
fi

mkdir -p "$LOG_DIR" "$DATA_DIR/temp" 2>/dev/null || true

# Function to clear listening ports on Windows
clear_port() {
    local port=$1
    echo "  Checking port :$port..."
    if command -v powershell.exe >/dev/null 2>&1; then
        powershell.exe -NoProfile -Command "
            Get-NetTCPConnection -LocalPort $port -State Listen -ErrorAction SilentlyContinue |
            ForEach-Object {
                \$pid = \$_.OwningProcess
                Stop-Process -Id \$pid -Force -ErrorAction SilentlyContinue
                Write-Host \"    Killed process on :$port (PID \$pid)\"
            }
        " 2>/dev/null || true
    fi
}

# ------------------------------------------------------------------------------
# STOP COMMAND
# ------------------------------------------------------------------------------
if [ "$1" == "stop" ] || [ "$1" == "--stop" ] || [ "$1" == "-Stop" ]; then
    echo "🛑 Stopping OCR stack services..."

    # Stop Python backend processes
    if command -v powershell.exe >/dev/null 2>&1; then
        powershell.exe -NoProfile -Command "
            Get-CimInstance Win32_Process -Filter \"Name='python.exe'\" -ErrorAction SilentlyContinue |
            Where-Object { \$_.CommandLine -like '*server_with_task_management*' } |
            ForEach-Object { Stop-Process -Id \$_.ProcessId -Force -ErrorAction SilentlyContinue; Write-Host \"    Killed backend PID \$(\$_.ProcessId)\" }
        " 2>/dev/null || true
    fi

    # Stop frontend container
    if docker ps -q -f name="^${FRONTEND_CONTAINER}$" | grep -q .; then
        echo "  Stopping container $FRONTEND_CONTAINER..."
        docker stop "$FRONTEND_CONTAINER" >/dev/null 2>&1 || true
    fi

    clear_port "$BACKEND_PORT"
    clear_port "$FRONTEND_PORT"

    echo "✅ Stack stopped. (Postgres container '$PG_CONTAINER' left running to preserve corpus)."
    echo "   To stop Postgres too: docker stop $PG_CONTAINER"
    exit 0
fi

# ------------------------------------------------------------------------------
# START COMMAND
# ------------------------------------------------------------------------------
echo "=================================================="
echo "🚀 Starting OCR Project Stack"
echo "=================================================="
echo "  Project Root : $PROJECT_DIR"
echo "  Models Root  : $MODELS_DIR"
echo "  Data Root    : $DATA_DIR"
echo "=================================================="

# 1. Check Docker Daemon
echo "==> [1/4] Checking Docker..."
if ! docker info >/dev/null 2>&1; then
    echo "  ⚠️ Docker daemon not responding. Attempting to start Docker Desktop..."
    if [ -f "$LOCALAPPDATA/Programs/DockerDesktop/Docker Desktop.exe" ]; then
        "$LOCALAPPDATA/Programs/DockerDesktop/Docker Desktop.exe" &
        echo "  Waiting for Docker daemon to respond..."
        for i in {1..40}; do
            sleep 4
            if docker info >/dev/null 2>&1; then break; fi
        done
    fi
fi

if ! docker info >/dev/null 2>&1; then
    echo "❌ Error: Docker engine is not running. Please start Docker Desktop."
    exit 1
fi
echo "    OK: Docker engine ready"

# 2. Check / Start PostgreSQL Container
echo "==> [2/4] Ensuring PostgreSQL container ($PG_CONTAINER)..."
if ! docker ps -a --format '{{.Names}}' | grep -q "^${PG_CONTAINER}$"; then
    echo "  Creating $PG_CONTAINER container..."
    docker run -d --name "$PG_CONTAINER" -p 5432:5432 \
        -e POSTGRES_DB=ocr_rag \
        -e POSTGRES_USER=ocr \
        -e POSTGRES_PASSWORD=ocrpass \
        pgvector/pgvector:pg16 >/dev/null
    sleep 3
    docker exec -i "$PG_CONTAINER" psql -U ocr -d ocr_rag -c "ALTER ROLE ocr WITH PASSWORD 'ocrpass'; CREATE ROLE ocr_ro WITH LOGIN PASSWORD 'ocrpass_ro'; GRANT CONNECT ON DATABASE ocr_rag TO ocr_ro; GRANT USAGE ON SCHEMA public TO ocr_ro;" >/dev/null 2>&1 || true
    if [ -f "backend/rag/schema.sql" ]; then
        docker exec -i "$PG_CONTAINER" psql -U ocr -d ocr_rag < backend/rag/schema.sql >/dev/null 2>&1 || true
    fi
elif ! docker ps --format '{{.Names}}' | grep -q "^${PG_CONTAINER}$"; then
    echo "  Starting existing $PG_CONTAINER container..."
    docker start "$PG_CONTAINER" >/dev/null
fi
echo "    OK: PostgreSQL running on :$PG_PORT"

# 3. Check / Start Frontend Container
echo "==> [3/4] Ensuring Frontend container ($FRONTEND_CONTAINER)..."
if ! docker ps -a --format '{{.Names}}' | grep -q "^${FRONTEND_CONTAINER}$"; then
    echo "  Building and creating $FRONTEND_CONTAINER..."
    docker build -t ocr-frontend:latest ./frontend >/dev/null
    docker run -d --name "$FRONTEND_CONTAINER" -p 3001:80 ocr-frontend:latest >/dev/null
elif ! docker ps --format '{{.Names}}' | grep -q "^${FRONTEND_CONTAINER}$"; then
    echo "  Starting existing $FRONTEND_CONTAINER container..."
    docker start "$FRONTEND_CONTAINER" >/dev/null
fi
echo "    OK: Frontend running at http://localhost:$FRONTEND_PORT"

# 4. Clear Backend Port & Start Python Backend
echo "==> [4/4] Starting FastAPI Backend on :$BACKEND_PORT..."
clear_port "$BACKEND_PORT"

# Set runtime environment
export OCR_MODELS_DIR="$MODELS_DIR"
export OCR_DATA_DIR="$DATA_DIR"
export OCR_PORT="$BACKEND_PORT"
export PYTHONIOENCODING="utf-8"
export RAG_INGEST_ENABLED="1"
export RAG_LANES="prose"
export RAG_DATABASE_URL="postgresql://ocr:ocrpass@localhost:5432/ocr_rag"
export RAG_DATABASE_URL_RO="postgresql://ocr_ro:ocrpass_ro@localhost:5432/ocr_rag"
export RAG_LLM_URL=""
export RAG_SQL_LOCAL_FALLBACK="1"

BACKEND_LOG="$LOG_DIR/backend_run.log"
BACKEND_ERR="$LOG_DIR/backend_err.log"

echo "  Launching backend process (GPU/CUDA enabled)..."
"$VENV_PY" backend/server_with_task_management.py > "$BACKEND_LOG" 2> "$BACKEND_ERR" &
BACKEND_PID=$!
echo "  Backend PID: $BACKEND_PID"

echo -n "  Waiting for backend health check"
for i in {1..60}; do
    if curl -s -m 2 "http://localhost:$BACKEND_PORT/health" 2>/dev/null | grep -q "healthy"; then
        echo " ✓ Healthy!"
        break
    fi
    if ! kill -0 "$BACKEND_PID" 2>/dev/null; then
        echo ""
        echo "❌ Backend exited unexpectedly. Check logs at: $BACKEND_ERR"
        exit 1
    fi
    echo -n "."
    sleep 3
done

echo ""
echo "=================================================="
echo "🎉 OCR PROJECT IS FULLY RUNNING!"
echo "=================================================="
echo "  👉 Frontend Web UI : http://localhost:$FRONTEND_PORT"
echo "  👉 Backend API     : http://localhost:$BACKEND_PORT"
echo "  👉 Health Endpoint : http://localhost:$BACKEND_PORT/health"
echo "  👉 Stop Stack      : ./readmenewrun.sh stop"
echo "                       or in PowerShell: .\\run_project.ps1 -Stop"
echo "=================================================="
echo "Press Ctrl+C to stop the backend or leave it running."

wait "$BACKEND_PID"
