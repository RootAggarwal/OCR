#!/bin/bash
#
# run_project.sh — start the OCR backend (:8000) + frontend (:3001) for THIS directory.
#
# Self-locating: works from whichever project copy it lives in (uses its own path).
# SAFE process management: only this project's own TCP ports are cleared. It does NOT
# sweep or kill GPU processes, so unrelated GPU jobs on the machine keep running.

PROJECT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd "$PROJECT_DIR"

BACKEND_PORT=8000
FRONTEND_PORT=3001
DECKWORKS_PORT=8002   # deckworks' own backend — :8001 is reserved for vLLM, never this
VENV_PY="$PROJECT_DIR/.venv/bin/python"
if [ -f "$PROJECT_DIR/.venv/Scripts/python.exe" ]; then
    VENV_PY="$PROJECT_DIR/.venv/Scripts/python.exe"
fi

# ── External model + data roots ─────────────────────────────────────────────
# Weights and runtime artifacts live OUTSIDE this directory so the project stays
# copyable (it was 117 GB; it is now ~5 GB). Defaults are siblings of PROJECT_DIR on
# the same volume, which is what made the migration an instant rename rather than a
# multi-hour copy. Derived from $PROJECT_DIR — never a hardcoded D:\ — so a renamed or
# relocated checkout still resolves. See backend/ocr_paths.py for the resolution order.
PROJECT_PARENT="$( dirname "$PROJECT_DIR" )"

# $VENV_PY is a NATIVE Windows python.exe while this script runs under Git Bash. MSYS
# translates argv but NOT arbitrary environment variables, so a POSIX "/d/OCR-data"
# arrives at Python as a relative path and abspath()s to "D:\d\OCR-data" — a phantom
# tree that makedirs() creates without complaint, leaving pdf_store silently empty and
# every Corpus preview 404ing with nothing in the log. Hand Python a native path.
# (backend/ocr_paths.py._norm() also catches this, for anyone exporting these by hand.)
to_native() { if command -v cygpath >/dev/null 2>&1; then cygpath -w "$1"; else printf '%s' "$1"; fi; }

export OCR_MODELS_DIR="${OCR_MODELS_DIR:-$(to_native "$PROJECT_PARENT/OCR-models")}"
export OCR_DATA_DIR="${OCR_DATA_DIR:-$(to_native "$PROJECT_PARENT/OCR-data")}"

# Logs go with the data, not in the repo. POSIX form: only this script redirects here.
OCR_LOG_DIR="${OCR_LOG_DIR:-$PROJECT_PARENT/OCR-data/logs}"
mkdir -p "$OCR_LOG_DIR"

# ── RAG knowledge base (Phase 1) ────────────────────────────────────────────
# Ingestion is OFF unless a Postgres DSN is set. Requires the pgvector container:
#   docker start ocr-rag-pg   (created via: docker run … pgvector/pgvector:pg16)
export RAG_INGEST_ENABLED="${RAG_INGEST_ENABLED:-1}"
export RAG_DATABASE_URL="${RAG_DATABASE_URL:-postgresql://ocr:ocrpass@localhost:5432/ocr_rag}"
export RAG_DATABASE_URL_RO="${RAG_DATABASE_URL_RO:-postgresql://ocr_ro:ocrpass_ro@localhost:5432/ocr_rag}"

# Chat/SQL LLM: Qwen3-14B-AWQ served by vLLM on :8001. If vLLM isn't running the RAG
# lanes fall back to the local Qwen2.5-VL automatically (also ~100% on the SQL eval).
# Start vLLM separately with ./start_vllm.sh — do NOT use the 32B: it starves the OCR
# models of VRAM and silently degrades table extraction to the unconstrained fallback.
export RAG_LLM_URL="${RAG_LLM_URL:-http://localhost:8001/v1}"
export RAG_LLM_MODEL="${RAG_LLM_MODEL:-qwen3-14b-awq}"

# Maker (deckworks): reuses the SAME vLLM server as the RAG chat/SQL lanes above —
# no separate GPU model. This is what removed deckworks' old crash/hang (it used to
# default to loading its own 29 GB Ollama model on the same GPU with no timeout).
export AI_API_KEY="${AI_API_KEY:-local}"
export AI_BASE_URL="${AI_BASE_URL:-http://localhost:8001/v1}"
export AI_MODEL="${AI_MODEL:-qwen3-14b-awq}"
export OCR_API_BASE="${OCR_API_BASE:-http://localhost:$BACKEND_PORT}"

BACKEND_PID=""
FRONTEND_PID=""
DECKWORKS_PID=""

# Kill only the process(es) LISTENING on a given TCP port. Never touches the GPU.
clear_port() {
    local port="$1"
    local pids
    pids=$(lsof -ti ":$port" -sTCP:LISTEN 2>/dev/null)
    if [ -n "$pids" ]; then
        echo "   clearing port $port (PIDs: $pids)"
        kill -9 $pids 2>/dev/null
        sleep 1
    fi
}

cleanup() {
    echo -e "\n🛑 Stopping services (this project only)..."
    [ -n "$BACKEND_PID" ]   && kill -9 "$BACKEND_PID"   2>/dev/null
    [ -n "$FRONTEND_PID" ]  && kill -9 "$FRONTEND_PID"  2>/dev/null
    [ -n "$DECKWORKS_PID" ] && kill -9 "$DECKWORKS_PID" 2>/dev/null
    clear_port "$BACKEND_PORT"
    clear_port "$FRONTEND_PORT"
    clear_port "$DECKWORKS_PORT"
    exit 0
}
# Only on interrupt/terminate — NOT on EXIT, so a normal error never triggers a sweep.
trap cleanup SIGINT SIGTERM

echo "=================================================="
echo "  🚀 OCR PROJECT  —  $(basename "$(dirname "$PROJECT_DIR")")"
echo "=================================================="

# --- Sanity checks ---
if [ ! -x "$VENV_PY" ]; then
    echo "❌ venv Python not found: $VENV_PY"; exit 1
fi
if [ ! -f "$PROJECT_DIR/backend/server_with_task_management.py" ]; then
    echo "❌ backend not found in $PROJECT_DIR/backend"; exit 1
fi
# Test the POSIX form — bash cannot reliably stat the native "D:\..." string above.
if [ ! -d "$PROJECT_PARENT/OCR-models" ]; then
    echo "❌ models dir not found: $OCR_MODELS_DIR"
    echo "   Model weights live outside this repo. Point OCR_MODELS_DIR at them,"
    echo "   or place them at $PROJECT_PARENT/OCR-models (see documentation/README.md)."
    exit 1
fi
echo "   models: $OCR_MODELS_DIR"
echo "   data:   $OCR_DATA_DIR"

echo "🧹 Pre-flight: clearing this project's ports (GPU jobs untouched)..."
clear_port "$BACKEND_PORT"
clear_port "$FRONTEND_PORT"
clear_port "$DECKWORKS_PORT"

# --- Backend ---
echo "🔥 Starting backend on :$BACKEND_PORT ..."
export PYTHONIOENCODING=utf-8
"$VENV_PY" backend/server_with_task_management.py > "$OCR_LOG_DIR/backend_run.log" 2>&1 &
BACKEND_PID=$!
echo "   backend PID $BACKEND_PID  (logs: $OCR_LOG_DIR/backend_run.log)"

echo -n "   waiting for backend to load models"
for i in $(seq 1 40); do
    if curl -s -m 2 "http://localhost:$BACKEND_PORT/health" 2>/dev/null | grep -q healthy; then
        echo " ✓ ready"; break
    fi
    if ! kill -0 "$BACKEND_PID" 2>/dev/null; then
        echo ""; echo "❌ backend exited early — see $OCR_LOG_DIR/backend_run.log"; exit 1
    fi
    echo -n "."; sleep 2
done

# --- Frontend ---
if [ -f "$PROJECT_DIR/frontend/package.json" ]; then
    echo "🔥 Starting frontend on :$FRONTEND_PORT ..."
    PORT=$FRONTEND_PORT BROWSER=none CI=false \
        npm start --prefix "$PROJECT_DIR/frontend" > "$OCR_LOG_DIR/frontend_run.log" 2>&1 &
    FRONTEND_PID=$!
    echo "   frontend PID $FRONTEND_PID  (logs: $OCR_LOG_DIR/frontend_run.log)"
else
    echo "⚠️  no frontend/package.json — skipping frontend"
fi

# --- Maker (deckworks) — optional add-on, best-effort: absence never fails this script ---
DECKWORKS_DIR="$PROJECT_DIR/deckworks"
if [ -d "$DECKWORKS_DIR" ] && command -v uv &> /dev/null; then
    echo "🔥 Preparing Maker (deckworks) on :$DECKWORKS_PORT ..."
    ( cd "$DECKWORKS_DIR" && uv sync ) >> "$OCR_LOG_DIR/deckworks_run.log" 2>&1

    # Build the console once — subsequent runs reuse dist/ (see MakerPanel.js: the
    # OCR frontend embeds this build via iframe, so a dev server isn't needed here).
    if [ ! -f "$DECKWORKS_DIR/frontend/dist/index.html" ]; then
        echo "   building Maker console (first run only)..."
        ( cd "$DECKWORKS_DIR/frontend" && npm install && npm run build ) \
            >> "$OCR_LOG_DIR/deckworks_run.log" 2>&1
    fi

    ( cd "$DECKWORKS_DIR" && uv run uvicorn deckworks.api.app:app \
        --host 0.0.0.0 --port "$DECKWORKS_PORT" ) >> "$OCR_LOG_DIR/deckworks_run.log" 2>&1 &
    DECKWORKS_PID=$!
    echo "   maker PID $DECKWORKS_PID  (logs: $OCR_LOG_DIR/deckworks_run.log, auto backend -> $AI_BASE_URL)"

    echo -n "   waiting for Maker"
    for i in $(seq 1 10); do
        if curl -s -m 2 "http://localhost:$DECKWORKS_PORT/api/health" 2>/dev/null | grep -q '"ok"'; then
            echo " ✓ ready"; break
        fi
        if ! kill -0 "$DECKWORKS_PID" 2>/dev/null; then
            echo ""; echo "⚠️  Maker exited early — see $OCR_LOG_DIR/deckworks_run.log (Reader/Thinker/Corpus still work)"
            DECKWORKS_PID=""
            break
        fi
        echo -n "."; sleep 2
    done
else
    echo "⚠️  deckworks/ not found or 'uv' not installed — skipping Maker (Reader/Thinker/Corpus still work)"
fi

echo "=================================================="
echo "🚀 Services running!"
echo "👉 Frontend: http://localhost:$FRONTEND_PORT   (Reader / Thinker / Corpus / Maker)"
echo "👉 Backend:  http://localhost:$BACKEND_PORT   (health: /health)"
[ -n "$DECKWORKS_PID" ] && echo "👉 Maker:    http://localhost:$DECKWORKS_PORT   (embedded in the Maker tab)"
echo "   Press Ctrl+C to stop (stops only this project)."
echo "=================================================="

# Block until a child exits or Ctrl+C is pressed.
wait
