# OCR Project

Document OCR, table extraction, chat, and retrieval-augmented generation (RAG) for
PDF workflows. The project includes a FastAPI backend, a React frontend, optional
PostgreSQL/pgvector retrieval, and the optional Deckworks document-making service.

## What it provides

- OCR for scanned and digital PDFs using the configured vision-language models.
- Structured table extraction and downloadable document exports.
- A browser UI for upload, OCR progress, document reading, chat, and corpus search.
- Optional RAG ingestion into PostgreSQL with pgvector for prose, figures, and tables.
- Optional Deckworks integration for generating presentations from extracted content.

## Repository layout

```text
backend/          FastAPI service, OCR pipeline, RAG, entity and table extraction
frontend/         React browser application
deckworks/        Optional presentation-generation service
documentation/    Architecture diagrams and integration notes
notebooks/        Experiments and prompt/similarity-search notebooks
backup/           Small project archives; large generated archives stay local
run_project.ps1   Windows stack launcher
run_project.sh    Linux/Git Bash stack launcher
```

## Requirements

- Python 3.12 or newer
- Node.js and npm
- CUDA-capable PyTorch installation for practical OCR performance
- Docker Desktop for PostgreSQL/pgvector and optional model services
- The model weights and runtime data directories described below

The OCR models are intentionally outside this repository. By default the launcher
expects these sibling directories:

```text
<parent-directory>/OCR-models
<parent-directory>/OCR-data
```

Set `OCR_MODELS_DIR` and `OCR_DATA_DIR` to use different locations. Do not commit
model weights, uploaded documents, databases, logs, or `.env` files.

## Quick start on Windows

Create or activate the project virtual environment, then install the backend
dependencies. PyTorch must come from the CUDA index rather than the default PyPI
CPU wheel:

```powershell
py -3.12 -m venv .venv
\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r backend\requirements.txt --extra-index-url https://download.pytorch.org/whl/cu130
```

Install and start the frontend separately when needed:

```powershell
cd frontend
npm install
npm start
```

For the complete local stack, from the repository root run:

```powershell
.\run_project.ps1
```

Useful launcher options include `-SkipVllm`, `-SkipMaker`, `-ModelsDir`,
`-DataDir`, and `-Stop`. The normal service URLs are:

| Service | URL |
| --- | --- |
| Frontend | http://localhost:3001 |
| OCR/RAG backend | http://localhost:8000 |
| Backend health check | http://localhost:8000/health |
| Deckworks, when enabled | http://localhost:8002 |

## Linux or Git Bash

Install the same Python and Node dependencies, then run:

```bash
./run_project.sh
```

The script expects the virtual environment at `.venv` and the model/data sibling
directories described above. Use `Ctrl+C` to stop the services started by the
script.

## Configuration

Common environment variables are:

| Variable | Default | Purpose |
| --- | --- | --- |
| `OCR_HOST` | `0.0.0.0` | Backend bind address |
| `OCR_PORT` | `8000` | Backend port |
| `OCR_MODELS_DIR` | `../OCR-models` | Model weights directory |
| `OCR_DATA_DIR` | `../OCR-data` | Uploads, temporary data, and logs |
| `OCR_ALLOWED_ORIGINS` | `*` | Comma-separated CORS origins |
| `RAG_INGEST_ENABLED` | `0` | Enable ingestion after OCR |
| `RAG_DATABASE_URL` | launcher default | PostgreSQL/pgvector connection |
| `RAG_LLM_URL` | `http://localhost:8001/v1` | OpenAI-compatible LLM endpoint |

Copy values into a local `.env` file when appropriate. Keep credentials and API
keys out of GitHub.

## Development checks

Run the backend tests from the repository root:

```powershell
pytest backend/tests
```

Build or test the frontend with:

```powershell
cd frontend
npm run build
npm test -- --watchAll=false
```

The architecture diagrams in `documentation/` describe the OCR, table extraction,
retrieval, and frontend/backend integration flows.

## Notes

- The backend can run without RAG services; retrieval features remain unavailable
	until PostgreSQL/pgvector and the configured LLM are running.
- The launcher writes logs under `OCR-data/logs`, outside the repository.
- `backup/frontend.zip` is intentionally ignored because GitHub rejects files over
	100 MB. Keep large local archives outside Git or use Git LFS when required.