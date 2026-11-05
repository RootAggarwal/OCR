# OCR-RAG & Thinker Project — Complete Architecture & Operational Guide

**Author / Reference Document:** `manish.md`  
**Target Audience:** Any developer, engineer, or new user setting up, running, or maintaining the system from scratch.

---

## 1. Project Overview

This project is an **Enterprise Vision-AI OCR & Grounded Conversational RAG (Retrieval-Augmented Generation) Platform** designed for high-precision extraction, reconstruction, and conversational analysis of dense, complex PDF documents, reports, and technical forms (e.g., ONGC geological and seismic well reports).

### Core Capabilities:
1. **Reader (Vision-Language OCR & Geometric Table Parsing):**
   - High-resolution document rendering (up to 2200px) with Poppler (`pdftoppm`).
   - Deep Vision Models: **olmOCR-2-7B** and **Qwen2.5-VL-7B-Instruct** running with hardware acceleration on NVIDIA GPUs (CUDA).
   - Two-stage geometric table extraction pipeline using **Table Transformer (TATR)** detection and structure recognition.
   - Dual-model concurrency: alternating even/odd pages across parallel GPU workers for 2x faster processing.
   - Exports results to Markdown, JSON, and multi-sheet formatted Excel files with preserved table geometry.

2. **Thinker (Grounded RAG Chatbot):**
   - Semantic & tabular question answering over extracted documents.
   - Dual-lane retrieval: **Prose Lane** (full-text search and embeddings) + **Tables Lane** (exact HTML table layout matching from PostgreSQL).
   - Automatic source page citations and figure thumbnail links.
   - Operates fully local on GPU via **Qwen2.5-VL** (or optionally connectable to external vLLM endpoints).

3. **Corpus (Knowledge Base & Quality Review):**
   - Long-term storage of processed documents in **PostgreSQL + pgvector**.
   - Original page image visualizer linked with extracted text and structured tables.
   - Server-wide document catalog shared between Reader, Thinker, and Deckworks (Maker).

---

## 2. Complete Project Directory Structure

```text
C:\OCRproject\
│
├── .env                              # Environment configurations (Neo4j, DB overrides)
├── pyproject.toml                    # Python project packaging metadata
├── README.md                         # Legacy repository overview
├── readmenewrun.sh                   # Bash launcher script for Git Bash / WSL
├── readnewdocumentaion.md            # Quick reference operational documentation
├── run_project.ps1                   # Native Windows PowerShell launcher (PRIMARY SCRIPT)
├── run_project.sh                    # Legacy shell runner
├── manish.md                         # THIS FILE: Complete master guide for new users
│
├── .venv\                            # Python virtual environment (PyTorch, FastAPI, transformers)
│
├── backend\                          # FastAPI Python Backend Service (:8000)
│   ├── requirements.txt              # Backend Python dependencies
│   ├── server_with_task_management.py# Main FastAPI server entry point (REST + WebSockets)
│   ├── task_manager.py               # In-memory session and background task lifecycle manager
│   ├── ocr_olm.py                    # Vision Language Model loader & inference (olmOCR / Qwen2.5-VL)
│   ├── ocr_tatr.py                   # Table Transformer detection & cell structure pipeline
│   ├── ocr_paths.py                  # External models & runtime storage path resolution
│   ├── table_extractor.py            # Geometric table unpivoting, HTML wrapping & validation
│   │
│   ├── app\                          # Modular FastAPI routers and config
│   │   ├── config.py                 # Application settings, unit-hint tables, timeouts
│   │   └── routers\
│   │       ├── rag_corpus.py         # Endpoints for /rag/documents, /rag-chat/, /rag/document-pages
│   │       └── upload.py             # File upload & batch processing endpoints
│   │
│   ├── rag\                          # RAG (Retrieval-Augmented Generation) Knowledge Base
│   │   ├── schema.sql                # PostgreSQL DDL schema (documents, table_cells, page_tables, RLS)
│   │   ├── db.py                     # PostgreSQL connection pool & RLS context manager (psycopg v3)
│   │   ├── ingest.py                 # Multi-pass document ingestion into database on OCR completion
│   │   ├── synthesize.py             # Grounded answer composition, evidence assembly & trust warnings
│   │   ├── router.py                 # Query routing (prose, tables, or both lanes)
│   │   ├── prose_search.py           # Semantic & keyword retrieval over text chunks
│   │   ├── table_search.py           # HTML table layout retrieval from page_tables
│   │   ├── llm_client.py             # LLM gateway (local GPU Qwen2.5-VL fallback / vLLM HTTP)
│   │   ├── prompt_budget.py          # Token reserve management & dynamic evidence budgeting
│   │   ├── aggregate_guard.py        # Guard against hallucinated arithmetic over table rows
│   │   └── figures.py                # PDF embedded figure & chart extractor
│   │
│   ├── local_proc\                   # Local document processing utilities
│   │   └── renderpdf.py              # Poppler wrapper for high-res PDF page-to-PNG rendering
│   │
│   ├── ner\                          # Named Entity Recognition pipelines
│   └── tests\                        # Unit and integration test suites
│
├── frontend\                         # React Frontend Application (Served on :3001)
│   ├── package.json                  # Frontend dependencies and build scripts
│   ├── Dockerfile                    # Multi-stage production build (Node build -> Nginx)
│   ├── nginx.conf                    # Nginx configuration (proxies /api/ and /ws/ to backend)
│   ├── public\                       # Public static assets, index.html, dictionaries
│   └── src\                          # React components & application code
│       ├── App.js                    # Main application navigation (Reader, Thinker, Corpus)
│       ├── features\
│       │   ├── ocr\
│       │   │   ├── FileUpload.js     # Reader interface, file upload, drag-and-drop, options
│       │   │   ├── RealTimeDisplay.js# Live OCR streaming preview & page-by-page rendering
│       │   │   ├── CorpusViewer.js   # Corpus document explorer & page inspector
│       │   │   └── ExportDropdown.js # Markdown / Excel / JSON downloader
│       │   └── chat\
│       │       └── ChatInterface.js  # Thinker chat interface, document picker (+), citations
│       └── infrastructure\
│           └── api\
│               ├── ocrClient.js      # REST API client
│               └── useWebSocketOCR.js# WebSocket streaming hook for chunked uploads
│
├── documentation\                    # Technical architecture diagrams, specs, and flowcharts
└── deckworks\                        # Optional Maker / Deckworks presentation tool (:8002)
```

---

## 3. External Directories (Out-of-Repository Storage)

To keep the git repository lightweight (~50 MB), **AI model weights** and **user runtime data** live in dedicated folders alongside the project directory:

| Folder Path | Purpose | Size / Contents |
| :--- | :--- | :--- |
| `C:\OCR-models\` | AI Weights & Model Snapshots | ~30 GB<br>• `models--allenai--olmOCR-2-7B-1025`<br>• `models--Qwen--Qwen2.5-VL-7B-Instruct`<br>• `table-structure-recognition`<br>• `table-transformer-detection` |
| `C:\OCR-data\` | Dynamic Runtime Data & Store | Grows with usage<br>• `temp\`: Active processing scratch files<br>• `pdf_store\`: Content-addressed stored PDFs (`{sha256}.pdf`)<br>• `logs\`: `backend_run.log`, `backend_err.log` |

---

## 4. Hardware & Software Prerequisites

Before starting, ensure the machine has:
1. **Operating System:** Windows 10 / 11 / Windows Server (64-bit)
2. **GPU & Drivers:** NVIDIA GPU with 16 GB+ VRAM recommended (System verified with NVIDIA RTX A6000 48GB), NVIDIA Driver 550+, CUDA 12.x.
3. **Docker Desktop:** Installed and running (using WSL 2 backend).
4. **Python:** Python 3.10 to 3.12 (64-bit).
5. **Poppler (for PDF rendering):** Installed on system PATH (`pdftoppm`).
   - Quick install via Windows Package Manager:
     ```powershell
     winget install oschwartz10612.Poppler
     ```

---

## 5. Step-by-Step Setup from Scratch (New Machine)

If you are cloning or setting up this project on a brand new machine, follow these steps:

### Step 1: Clone Repository & Create Virtual Environment
```powershell
cd C:\
git clone <repository_url> OCRproject
cd C:\OCRproject

# Create virtual environment
python -m venv .venv

# Activate virtual environment
.\.venv\Scripts\Activate.ps1
```

### Step 2: Install Backend Dependencies
```powershell
# Upgrade pip
python -m pip install --upgrade pip

# Install PyTorch with CUDA support (adjust for your CUDA version):
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121

# Install backend dependencies
pip install -r backend\requirements.txt
```

### Step 3: Verify External Models & Directories
Ensure the following directories exist:
```powershell
New-Item -ItemType Directory -Path "C:\OCR-models", "C:\OCR-data\temp", "C:\OCR-data\pdf_store", "C:\OCR-data\logs" -Force
```
Place your Hugging Face model weight folders inside `C:\OCR-models`:
- `C:\OCR-models\models--allenai--olmOCR-2-7B-1025`
- `C:\OCR-models\models--Qwen--Qwen2.5-VL-7B-Instruct`
- `C:\OCR-models\table-structure-recognition`
- `C:\OCR-models\table-transformer-detection`

### Step 4: Setup Docker Containers

#### 1. PostgreSQL with pgvector (Database):
```powershell
docker run -d `
  --name ocr-rag-pg `
  --restart unless-stopped `
  -e POSTGRES_DB=ocr_rag `
  -e POSTGRES_USER=ocr `
  -e POSTGRES_PASSWORD=ocrpass `
  -p 5432:5432 `
  pgvector/pgvector:pg16
```

Initialize database users and schema:
```powershell
# Create read-only role
docker exec ocr-rag-pg psql -U ocr -d ocr_rag -c "CREATE ROLE ocr_ro WITH LOGIN PASSWORD 'ocrpass_ro';"

# Apply schema
docker exec -i ocr-rag-pg psql -U ocr -d ocr_rag < backend\rag\schema.sql
```

#### 2. Frontend React Container:
```powershell
cd C:\OCRproject\frontend
docker build -t ocr-frontend:latest .

docker run -d `
  --name ocr-frontend-dev `
  --restart unless-stopped `
  -p 3001:80 `
  ocr-frontend:latest
cd C:\OCRproject
```

---

## 6. How to Run the Project (Daily Operations)

The project includes an intelligent native PowerShell script (`run_project.ps1`) that automatically handles port cleanup, Docker container checks, environment variables, model verification, and backend startup.

### 🟢 Starting the Stack (Recommended Command)

Open **PowerShell** in `C:\OCRproject` and run:

```powershell
.\run_project.ps1 -SkipVllm
```

> **Why `-SkipVllm`?**  
> Passing `-SkipVllm` allows the backend to utilize the local **Qwen2.5-VL-7B** model on your GPU for both OCR and Thinker chat. This frees up ~19 GB VRAM and eliminates the need to run an extra heavy vLLM container.

#### What this command does automatically:
1. Verifies `.venv`, model weights, and `pdftoppm`.
2. Starts Docker Desktop if it is closed.
3. Starts the database container (`ocr-rag-pg`) and frontend container (`ocr-frontend-dev`).
4. Cleans up any dead/lingering sockets on port `:8000`.
5. Launches the FastAPI backend process with CUDA acceleration.
6. Waits for `/health` to return `200 OK`.
7. Displays a real-time status summary with clickable URLs.

---

### 🔴 Stopping the Stack

To cleanly stop the entire project:

```powershell
.\run_project.ps1 -Stop
```

This terminates the Python backend process, frees port `:8000`, and stops the frontend container. The PostgreSQL database is kept running so your indexed documents remain safe.

---

## 7. Accessing the Application

Once started, open your web browser:

| Interface | URL | Description |
| :--- | :--- | :--- |
| **OCR Web Application** | **[http://localhost:3001](http://localhost:3001)** | Main user interface (Reader, Thinker, Corpus) |
| **Backend Health Status** | **[http://localhost:8000/health](http://localhost:8000/health)** | API status and GPU availability |
| **API Documentation** | **[http://localhost:8000/docs](http://localhost:8000/docs)** | Interactive Swagger OpenAPI docs |
| **Indexed Documents** | **[http://localhost:8000/rag/documents](http://localhost:8000/rag/documents)** | List of ingested documents in knowledge base |

### 🌐 Accessing from Your Office LAN / Intranet
To access the application from other computers on your local network:
1. Find the host machine's local IP address (e.g., `192.168.1.50`) by running `ipconfig` in PowerShell.
2. Ensure Windows Firewall permits incoming traffic on TCP ports `3001` and `8000`:
   ```powershell
   New-NetFirewallRule -DisplayName "OCR Frontend (3001)" -Direction Inbound -LocalPort 3001 -Protocol TCP -Action Allow
   New-NetFirewallRule -DisplayName "OCR Backend (8000)" -Direction Inbound -LocalPort 8000 -Protocol TCP -Action Allow
   ```
3. Other users on the same WiFi/Ethernet network can then open:  
   👉 **`http://192.168.1.50:3001`**

---

## 8. User Workflow Guide

### Module 1: The Reader Tab (Extracting Documents)
1. **Drop or Browse:** Upload any PDF or scanned document.
2. **Settings:**
   - **Dual Model Mode (Toggle):**  
     - Off: Processes 1 page at a time sequentially.
     - On (Recommended on RTX A6000): Alternates pages between Model 1 & Model 2 simultaneously, cutting total extraction time in half.
   - **Page Selection (Optional):**  
     Enter specific pages (e.g. `1-3`, `5`, `10-15`) if you do not need the entire document processed.
3. **Extract Text:** Click **Extract Text**. A live WebSocket stream will display each page's extracted markdown and rendered tables in real-time.
4. **Exporting:** Use the top-right export button to download as **Markdown**, **JSON**, or formatted **Excel (.xlsx)**.

---

### Module 2: The Thinker Tab (Asking Questions About PDFs)
1. Click the **Thinker** tab in the top navigation bar.
2. **Select Document Context:**
   - Click the **`+`** button on the bottom left of the chat input box.
   - A list of all ingested documents will appear.
   - Check the box for your document. A chip with the document's name will appear above the input box.
3. **Ask Your Questions:**
   - *Example 1:* `"What is this document about?"`
   - *Example 2:* `"Summarize the key drilling findings and mud parameters"`
   - *Example 3:* `"What are the values in the core analysis table?"`
4. **Review Evidence:**
   - Thinker responds using the local Vision-Language model grounded in the database evidence.
   - Citations show the exact source pages (e.g., `[filename.pdf p1]`). Clicking a citation opens that specific page.

---

### Module 3: The Corpus Tab (Catalog & Quality Inspector)
1. Click the **Corpus** tab.
2. View all documents indexed in the database.
3. Select any document to view:
   - High-resolution original page preview images side-by-side with extracted markdown.
   - Reconstructed HTML tables with raw cell values and structure flags.

---

## 9. Troubleshooting & Known Pitfalls

### ⚠️ Issue 1: "No documents in the knowledge base yet" in Thinker
- **Cause:** A native Windows PostgreSQL service (`postgresql-x64-18`) was running on the host machine and hijacked port `5432`, blocking connections to the Docker `ocr-rag-pg` container.
- **Solution:** Stop and set the native Windows PostgreSQL service to Manual:
  ```powershell
  Stop-Service postgresql-x64-18
  Set-Service -Name postgresql-x64-18 -StartupType Manual
  ```
  Then verify port 5432 is owned by Docker:
  ```powershell
  Get-NetTCPConnection -LocalPort 5432 -State Listen
  ```

---

### ⚠️ Issue 2: Uploading PDF takes much longer than before
- **Cause:** Large documents (e.g., 30+ pages). The Vision-Language Model performs geometric table analysis and deep optical inference per page (~28–32s per page).
  - 2 pages × 30s = ~1 minute.
  - 31 pages × 30s = ~15–16 minutes.
- **Speedup Solutions:**
  1. **Enable Dual Model Mode:** Toggle the switch in the Reader tab before uploading (cuts time in half).
  2. **Page Range Selection:** Enter specific pages (e.g. `1-5`) if the entire report is not needed.
  3. **Fast Text Mode:** If tables are simple, set `STRUCTURED_TABLES=0` to skip Table Transformer passes (reduces time to ~6–8s per page).

---

### ⚠️ Issue 3: Port 8000 already in use / Orphaned Sockets
- **Cause:** A previous Python backend process was terminated abnormally, leaving a socket bound to `:8000`.
- **Solution:** Run the built-in stopper:
  ```powershell
  .\run_project.ps1 -Stop
  ```
  If an orphaned socket remains without a PID, restart the machine or run temporarily on an alternate port:
  ```powershell
  .\run_project.ps1 -BackendPort 8010
  ```

---

---

### ⚠️ Issue 4: Thinker says "This document contains no content to analyze"
- **Cause:** Open-ended questions (e.g. "What is this file?") didn't match keyword search terms in pure text mode.
- **Status:** **Resolved.** Fallback handlers are now active in [prose_search.py](file:///c:/OCRproject/backend/rag/prose_search.py) and [table_search.py](file:///c:/OCRproject/backend/rag/table_search.py). When you select a document with the `+` button, Thinker automatically supplies the document's pages and tables directly to the model.

---

### ✨ Feature Highlight: Corpus Thinker Popup & Direct PDF Chat
- **Corpus Direct Chat:** Click **"Ask in Thinker"** directly in the Corpus tab (on any document card in the sidebar or on the main document toolbar).
- **In-Corpus Modal Assistant:** Opens an interactive Thinker AI assistant popup directly over the Corpus view without leaving the document you are examining.
- **Direct PDF Switcher:** In the popup header, click the document dropdown to switch to or add any PDF from the entire corpus, or query across the whole knowledge base.
- **Direct Upload to Corpus:** In the Corpus sidebar header, click the upload icon to directly upload a new PDF into the processing queue without having to switch back to Reader first.

---

## 10. Summary Cheat Sheet

| Action | PowerShell Command |
| :--- | :--- |
| **Start Everything** | `.\run_project.ps1 -SkipVllm` |
| **Stop Everything** | `.\run_project.ps1 -Stop` |
| **Rebuild Frontend Image** | `docker build -t ocr-frontend:latest ./frontend` |
| **Check Port 8000** | `Get-NetTCPConnection -LocalPort 8000` |
| **Check Docker Containers** | `docker ps` |
| **Check GPU Memory** | `nvidia-smi` |
| **View Live Backend Logs** | `Get-Content C:\OCR-data\logs\backend_run.log -Tail 30 -Wait` |
| **Batch Ingest All Stored PDFs** | `.\.venv\Scripts\python.exe backend\rag\ingest_all_pdf_store.py` |

---
*Created and verified for the OCRproject environment.*
