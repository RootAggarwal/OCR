# OCR-RAG Project — Run Guide & System Documentation

This guide provides instructions on how to start, verify, troubleshoot, and stop the complete OCR stack.

---

## 1. Quick Start (How to Run Next Time)

You have two simple 1-command options to start the entire system:

### Option A: Using PowerShell (Recommended for Windows)

Open **PowerShell** as Administrator or standard user in `C:\OCRproject` and run:

```powershell
.\run_project.ps1 -SkipVllm
```

To stop the entire stack cleanly:
```powershell
.\run_project.ps1 -Stop
```

---

### Option B: Using Git Bash / Shell Script

Open **Git Bash** in `C:\OCRproject` and run:

```bash
./readmenewrun.sh
```

To stop the entire stack cleanly:
```bash
./readmenewrun.sh stop
```

---

## 2. Live Services & Port Architecture

| Service | Mechanism | Host Port | Internal Port | Description / URL |
| :--- | :--- | :--- | :--- | :--- |
| **Frontend UI** | Docker (`ocr-frontend-dev`) | `:3001` | `:80` | [http://localhost:3001](http://localhost:3001) |
| **Backend API** | Python FastAPI (`.venv`) | `:8000` | `:8000` | [http://localhost:8000/health](http://localhost:8000/health) |
| **PostgreSQL + pgvector** | Docker (`ocr-rag-pg`) | `:5432` | `:5432` | `postgresql://ocr:ocrpass@localhost:5432/ocr_rag` |
| **vLLM (Optional)** | Docker (`vllm-qwen3`) | `:8001` | `:8001` | Optional analytical SQL model (`-SkipVllm` uses local VLM) |

---

## 3. Architecture & Components

```
┌────────────────────────────────────────────────────────────────────────┐
│                        BROWSER / WORKSTATION                           │
│                                                                        │
│   Web UI: http://localhost:3001  ──────────┐                          │
└─────────────────────────────────────────────│──────────────────────────┘
                                              │
                      ┌───────────────────────┘
                      ▼
┌──────────────────────────────────────────────┐
│  FRONTEND CONTAINER (ocr-frontend-dev)       │
│  - Image: ocr-frontend:latest (Nginx 1.21)   │
│  - Serves: React SPA bundle                  │
│  - Proxies: /api/ -> host.docker.internal    │
└──────────────────────┬───────────────────────┘
                       │
                       │ Connects to REST & WebSockets
                       ▼
┌──────────────────────────────────────────────┐
│  FASTAPI BACKEND (:8000)                     │
│  - Python 3.12 (.venv) with PyTorch + CUDA   │
│  - ocr_olm.py: olmOCR-2-7B & Qwen2.5-VL-7B   │
│  - ocr_tatr.py & table_extractor.py: TATR    │
│  - RAG Subsystem: BM25 + pgvector + BGE-M3   │
└──────────────────────┬───────────────────────┘
                       │
                       │ Read / Write Transactions
                       ▼
┌──────────────────────────────────────────────┐
│  DATABASE CONTAINER (ocr-rag-pg)             │
│  - Image: pgvector/pgvector:pg16 (:5432)     │
│  - Database: ocr_rag                         │
│  - Users: ocr (read-write), ocr_ro (read-only│
│  - Row-Level Security (RLS) enabled          │
└──────────────────────────────────────────────┘
```

### External Storage Locations (Preserved outside repository)
* **Weights & Models**: `C:\OCR-models`
  * `models--allenai--olmOCR-2-7B-1025`
  * `models--Qwen--Qwen2.5-VL-7B-Instruct`
  * `table-structure-recognition` (TATR structure)
  * `table-transformer-detection` (TATR detection)
* **Data & Runtime Artifacts**: `C:\OCR-data`
  * `C:\OCR-data\temp`: Active processing files
  * `C:\OCR-data\pdf_store`: Content-addressed original PDFs
  * `C:\OCR-data\logs`: Log files (`backend_run.log`, `backend_err.log`)

---

## 4. Docker Management Reference

### 1. PostgreSQL Database Container (`ocr-rag-pg`)
* **Start Container**:
  ```bash
  docker start ocr-rag-pg
  ```
* **Stop Container**:
  ```bash
  docker stop ocr-rag-pg
  ```
* **Re-create from scratch (if needed)**:
  ```bash
  docker run -d --name ocr-rag-pg -p 5432:5432 \
    -e POSTGRES_DB=ocr_rag \
    -e POSTGRES_USER=ocr \
    -e POSTGRES_PASSWORD=ocrpass \
    pgvector/pgvector:pg16
  ```
  Then apply the schema:
  ```powershell
  Get-Content backend\rag\schema.sql | docker exec -i ocr-rag-pg psql -U ocr -d ocr_rag
  ```

### 2. Frontend Docker Container (`ocr-frontend-dev`)
* **Start Container**:
  ```bash
  docker start ocr-frontend-dev
  ```
* **Stop Container**:
  ```bash
  docker stop ocr-frontend-dev
  ```
* **Rebuild Container after modifying frontend code**:
  ```bash
  docker build -t ocr-frontend:latest ./frontend
  docker rm -f ocr-frontend-dev
  docker run -d --name ocr-frontend-dev -p 3001:80 ocr-frontend:latest
  ```

---

## 5. Verifying Health & Functionality

Once started, test these endpoints in PowerShell or browser:

1. **Frontend UI**:
   Open browser at: [http://localhost:3001](http://localhost:3001)

2. **Backend Health Check**:
   ```powershell
   Invoke-RestMethod -Uri "http://localhost:8000/health"
   ```
   **Expected Response**:
   ```json
   {
     "status": "healthy",
     "device": "cuda",
     "gpu_available": true,
     "gpu_count": 1,
     "task_manager_active": true
   }
   ```

3. **Session Creation**:
   ```powershell
   Invoke-RestMethod -Uri "http://localhost:8000/session/create" -Method Post
   ```
   **Expected Response**:
   ```json
   {
     "success": true,
     "session_id": "...",
     "message": "Session created successfully"
   }
   ```

4. **Corpus Documents**:
   ```powershell
   Invoke-RestMethod -Uri "http://localhost:8000/rag/documents"
   ```
   **Expected Response**:
   ```json
   { "documents": [] }
   ```

---

## 6. Common Gotchas & Troubleshooting

1. **Port 8000 Already in Use / Orphaned Socket**:
   * Run `.\run_project.ps1 -Stop` to terminate background processes and free ports.
   * If a socket remains orphaned, check listening processes:
     ```powershell
     Get-NetTCPConnection -LocalPort 8000 -State Listen
     ```

2. **Docker Not Running**:
   * Ensure Docker Desktop is started before running. Both scripts will automatically detect and attempt to launch Docker Desktop if it is not running.

3. **Inspecting Backend Logs**:
   * View runtime log:
     ```powershell
     Get-Content C:\OCR-data\logs\backend_run.log -Tail 50 -Wait
     ```
   * View error log:
     ```powershell
     Get-Content C:\OCR-data\logs\backend_err.log -Tail 50
     ```

4. **Virtual Environment Python**:
   * Python binary is located at `.venv\Scripts\python.exe` and configured for Python 3.12 with PyTorch CUDA 13.

5. **Thinker Tab & LLM Configuration**:
   * When running without vLLM (`-SkipVllm`), the Thinker tab automatically runs inference against the resident local vision-language model (`Qwen2.5-VL-7B-Instruct`).
   * The scripts automatically set `RAG_LLM_URL=""` and `RAG_SQL_LOCAL_FALLBACK=1` so that requests do not attempt to contact an offline vLLM server on port 8001.

6. **Avoiding Duplicate Background Processes**:
   * Always run `.\run_project.ps1 -Stop` before launching a new backend instance. This prevents multiple Python processes from colliding on port 8000.
