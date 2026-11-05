# Frontend ↔ Backend Integration Guide

> **Purpose**: A comprehensive reference for reconnecting every frontend component to its  
> corresponding backend endpoint when you return to your workstation.

---

## Architecture Overview

```
┌─────────────────────────────────────────────────────────┐
│  FRONTEND  (React, port 3000)                           │
│                                                         │
│  FileUpload ─┬─ useOCRService ──── HTTP batch upload    │
│              ├─ useWebSocketOCR ── WS streaming upload  │
│              └─ SystemStats ────── (simulated / polled)  │
│                                                         │
│  PageResult ──── Save edits via HTTP POST               │
│  ChatInterface ── RAG chat + auto-analysis via HTTP     │
│  ocrClient ───── Session mgmt, file transfer, chat     │
└──────────────────────┬──────────────────────────────────┘
                       │
                       ▼
┌─────────────────────────────────────────────────────────┐
│  BACKEND   (Python, port 8000)                          │
│                                                         │
│  POST   /session/create         → session management    │
│  GET    /session/{id}/tasks     → validate session      │
│  DELETE /session/{id}           → delete session        │
│  POST   /upload/                → batch OCR             │
│  WS     /ws/upload/             → streaming OCR         │
│  POST   /update-page/{sid}/{p}  → save edited text      │
│  POST   /analyze/               → auto doc analysis     │
│  POST   /rag-chat/              → RAG chatbot           │
│  GET    /system-stats           → CPU/GPU/MEM (NEW)     │
│  POST   /export-table/          → Excel table export    │
└─────────────────────────────────────────────────────────┘
```

---

## Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `REACT_APP_OCR_API_URL` | `http://{hostname}:8000/upload/` | OCR backend base URL |
| `REACT_APP_OCR_WS_URL` | `ws://{hostname}:8000/ws/upload/` | WebSocket endpoint |

Set these in `frontend/.env`:
```env
REACT_APP_OCR_API_URL=http://192.168.1.100:8000/upload/
REACT_APP_OCR_WS_URL=ws://192.168.1.100:8000/ws/upload/
```

All components derive their base URL from these two variables.

---

## Component-by-Component Breakdown

---

### 1. `ocrClient.js` — Core Service Layer

**File**: `frontend/src/services/ocrClient.js` (line ~952)

This singleton class is the **central nervous system** — every other component goes through it.

#### Session Management

| Method | Backend Endpoint | Purpose |
|--------|-----------------|---------|
| `ensureSession()` | `POST /session/create` | Creates a new session, returns `{ success, session_id }` |
| `validateSession()` | `GET /session/{id}/tasks` | Checks if an existing session is still valid (200 = valid) |
| `deleteSession()` | `DELETE /session/{id}` | Destroys session on server and clears localStorage |

Sessions are persisted in `localStorage` under key `ocr_session_id`.

#### Batch Upload (HTTP)

| Method | Backend Endpoint | Purpose |
|--------|-----------------|---------|
| `uploadFileBatch(file, lang, mode)` | `POST /upload/` | Sends entire file as `multipart/form-data` |

**Request body** (FormData):
```
file:       <File blob>
lang:       "eng"
session_id: "abc-123"
mode:       "single" | "dual"
```

**Expected response**:
```json
{
  "success": true,
  "task_id": "task-456",
  "pages": [
    {
      "page": 1,
      "text": "<h2>Title</h2><p>Extracted content...</p>",
      "preview": "base64-encoded-image-or-null",
      "error": null
    }
  ]
}
```

#### Streaming Upload (WebSocket)

| Method | Backend Endpoint | Purpose |
|--------|-----------------|---------|
| `uploadFileStreaming(file, callbacks, mode, schema, pagesSpec)` | `WS /ws/upload/` | Chunked binary upload with real-time page results |

**Protocol**:
1. Client opens WebSocket to `/ws/upload/`
2. Client sends JSON metadata message:
   ```json
   {
     "type": "file_metadata",
     "session_id": "abc-123",
     "filename": "report.pdf",
     "total_chunks": 5,
     "chunk_size": 1048576,
     "mode": "single",
     "pages": ""
   }
   ```
3. Client sends N binary `ArrayBuffer` chunks (1MB each)
4. Server responds with progress messages for each page:
   ```json
   {
     "type": "page_complete",
     "page": 3,
     "totalPages": 10,
     "text": "<p>Extracted text...</p>",
     "progress": 30,
     "confidence": 0.95,
     "preview": "base64...",
     "has_table": true,
     "session_id": "abc-123"
   }
   ```
5. Server sends completion:
   ```json
   { "type": "complete", "pages": [...] }
   ```

**Callbacks object**:
```javascript
{
  onTaskCreated:   (data) => {},    // { task_id }
  onTableDetected: (data, reply) => {},  // interactive table schema
  onProgress:      (data) => {},    // page-by-page results
  onComplete:      (data) => {},    // final completion
  onError:         (error) => {},   // errors
  onCancel:        (data) => {},    // task cancelled
}
```

---

### 2. `FileUpload.js` — Main Orchestrator

**File**: `frontend/src/components/FileUpload.js` (line ~127)

This is the main page component. It doesn't call the backend directly — it delegates to hooks.

#### Hooks Used

| Hook | What It Does | Backend Connection |
|------|-------------|-------------------|
| `useFileProcessing()` | State management (files, results, progress) | None (pure state) |
| `useOCRService(apiUrl)` | Batch HTTP upload | `POST /upload/` via `ocrClient.uploadFileBatch()` |
| `useWebSocketOCR(wsUrl)` | Streaming WS upload | `WS /ws/upload/` via `ocrClient.uploadFileStreaming()` |

#### Processing Flow

```
User drops file → handleDrop() → handleUpload() → runProcessing()
                                                      │
                                  ┌───────────────────┤
                                  │ PDF + realtime?    │ Otherwise
                                  ▼                    ▼
                           processFileStream()    processWithStandardAPI()
                           (WebSocket)            (HTTP batch)
                                  │                    │
                                  ▼                    ▼
                           Pages stream in        All pages at once
                           one-by-one             via processFile()
```

#### Key State Variables

| Variable | Type | Purpose |
|----------|------|---------|
| `globalProcessing` | boolean | `true` while OCR is running — drives hero text, TATR glow, SystemStats |
| `pageResults` | array | Accumulated extracted pages |
| `files` | array | Selected File objects |
| `useRealTimeProcessing` | boolean | Toggle between WS streaming and HTTP batch |
| `modelMode` | string | `"single"` or `"dual"` — sent to backend |
| `pageSelection` | string | `""` for all pages, or `"3"` / `"1-5"` for specific pages |
| `appMode` | string | `"ocr"` or `"chat"` — switches between OCR view and Chat view |

#### Simulate OCR Button

The `simulateOCRWithTable()` function (line ~290) is a **frontend-only mock** that bypasses the backend entirely. It sets `globalProcessing = true`, waits 4.5 seconds (to show the typewriter + stats animations), then injects 10 fake pages. **No backend call is made.**

---

### 3. `PageResult.js` — Individual Page Card

**File**: `frontend/src/components/PageResult.js`

Each extracted page is rendered as a card with view/edit/zoom capabilities.

#### Backend Calls

| Action | Endpoint | Method | Purpose |
|--------|----------|--------|---------|
| **Save edited text** | `/update-page/{session_id}/{page_number}` | `POST` | Persists user edits to the backend |
| **Export table to Excel** | `/export-table/` | `POST` | Converts HTML table to downloadable `.xlsx` |

##### Save Edited Text

**Request**:
```javascript
POST /update-page/{session_id}/{page_number}
Content-Type: application/json

{ "text": "<p>User-edited HTML content...</p>" }
```

**How session_id is resolved** (line 24):
```javascript
const sessionId = result.session_id 
  || localStorage.getItem('ocr_session_id') 
  || localStorage.getItem('session_id');
```

##### Export Table to Excel

**Request** (FormData):
```
session_id:  "abc-123"
page_number: 3
table_html:  "<table>...</table>"
```

**Response**: Binary `.xlsx` file download.

---

### 4. `ChatInterface.js` — RAG Chat

**File**: `frontend/src/components/ChatInterface.js` (line ~115)

#### Props Received

| Prop | Type | Source |
|------|------|--------|
| `extractedText` | string | Combined text from all `pageResults` |
| `documentName` | string | Name of the last processed file |
| `isVisible` | boolean | Whether chat panel is showing |
| `onClose` | function | Callback to switch back to OCR mode |

#### Backend Calls

| Action | Endpoint | Method | Purpose |
|--------|----------|--------|---------|
| **Auto-analysis** | `/analyze/` | `POST` | Generates initial document summary on load |
| **Send message** | `/rag-chat/` | `POST` | RAG chatbot query with session context |

##### Auto-Analysis (runs once when chat opens)

**Request** (FormData):
```
extracted_text: "Page 1: ... Page 2: ..."
document_name:  "report.pdf"
analysis_type:  "summary"
```

**Response**:
```json
{
  "success": true,
  "analysis": "This document contains a reservoir report..."
}
```

##### RAG Chat

**Request** (FormData):
```
message:    "What is the porosity value?"
session_id: "abc-123"
```

**Response**:
```json
{
  "answer": "The porosity is 22.5% as stated in...",
  "citations": [
    { "filename": "report.pdf", "page": 3 }
  ],
  "warnings": ["⚠ This value was not found in the SQL database"]
}
```

---

### 5. `SystemStats` — CPU/GPU/MEM Monitor

**File**: `frontend/src/components/FileUpload.js` (line ~72, inline component)

#### Current State: SIMULATED

Right now, this component uses a random-walk algorithm to generate realistic-looking values. **No backend call is made.**

#### To Connect to Backend

See `SYSTEM_STATS_BACKEND_INTEGRATION.md` for the full step-by-step guide. Summary:

| Action | Endpoint | Method | Purpose |
|--------|----------|--------|---------|
| **Poll stats** | `/system-stats` | `GET` | Returns `{ cpu, gpu, mem }` percentages |

Replace the simulation `useEffect` (lines ~96–108) with the polling code documented in that file.

---

### 6. `WorkflowNode.js` — Sidebar Workflow Diagram

**File**: `frontend/src/components/WorkflowNode.js`

**No backend connection.** This is a pure presentational component that renders the workflow pipeline visualization. Its `status` prop (`default` / `active` / `completed`) is driven by `globalProcessing` and `pageResults.length` from `FileUpload.js`.

---

### 7. `FilePreviewCard.js` — File Preview Cards

**File**: `frontend/src/components/FilePreviewCard.js`

**No backend connection.** Displays file metadata (name, type, size) from the local `File` objects. The card shows a thumbnail if the file is an image.

---

### 8. `CrtProgress.js` — Progress Bar

**File**: `frontend/src/components/CrtProgress.js`

**No backend connection.** A pure presentational CRT-styled progress bar driven by the `processingProgress` state.

---

### 9. `BootSequence.js` — Splash Screen

**File**: `frontend/src/components/BootSequence.js`

**No backend connection.** A one-time CRT boot animation that plays when the app first loads.

---

## Complete Backend API Cheatsheet

```
# Session Management
POST   /session/create                    → { success, session_id }
GET    /session/{id}/tasks                → 200 OK (valid) or 404
DELETE /session/{id}                      → 200 OK

# OCR Processing
POST   /upload/                           → { success, pages: [...] }
WS     /ws/upload/                        → Streaming page results

# Page Editing
POST   /update-page/{session_id}/{page}   → { success }

# Document Analysis & Chat
POST   /analyze/                          → { success, analysis }
POST   /rag-chat/                         → { answer, citations, warnings }

# Table Export
POST   /export-table/                     → Binary .xlsx download

# System Stats (NEW — needs to be added)
GET    /system-stats                      → { cpu, gpu, mem }
```

---

## Startup Checklist

1. **Start backend**: `cd backend && python server.py` (or `uvicorn app:app --port 8000`)
2. **Verify CORS** is enabled for `http://localhost:3000`
3. **Check `.env`**: Ensure `REACT_APP_OCR_API_URL` points to the correct backend IP
4. **Start frontend**: `cd frontend && npm start`
5. **Test session**: Open browser DevTools → Network tab → you should see `POST /session/create` on load
6. **Test OCR**: Drop a PDF → verify WebSocket connection to `/ws/upload/`
7. **Test chat**: Switch to RAG Chat → verify `/rag-chat/` calls
8. **Add `/system-stats`** endpoint to backend if you want live CPU/GPU stats (see `SYSTEM_STATS_BACKEND_INTEGRATION.md`)
