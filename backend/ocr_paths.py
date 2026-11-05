"""
ocr_paths.py — the ONE place that knows where model weights and runtime data live.

Everything heavy now lives OUTSIDE the repo so this directory stays copyable: ~112 GB of
weights, dead venvs and runtime artifacts used to sit inside it (117.7 GB -> ~5 GB). Two
roots, each overridable by a bare env var — the same convention as OCR_HOST /
RAG_DATABASE_URL / AI_MODEL. There is deliberately no settings module and no .env, and
this file does not introduce one.

    OCR_MODELS_DIR   read-only weights   default  <repo_parent>/OCR-models
    OCR_DATA_DIR     writable runtime    default  <repo_parent>/OCR-data

OCR_MODELS_DIR preserves the exact layout backend/models/ had, because the migration was a
single directory rename: it IS a HuggingFace cache root (CACHEDIR.TAG + .locks/ +
models--<org>--<repo>/) with two plain non-hub dirs, table-structure-recognition/ and
table-transformer-detection/, sitting alongside. Nothing inside it changed, which is why
nothing here needs to know its internal structure.

DO NOT set HF_HOME / HF_HUB_CACHE to OCR_MODELS_DIR to "unify the caches". BAAI/bge-m3
(~13 GB, the prose lane's embedder) lives in the C: user cache and is deliberately not
moved here. Repointing HF_HOME makes it invisible, rag/embeddings.available() returns
False, and the prose lane silently no-ops — RAG answers quietly get worse with no error
anywhere. See the docstring at rag/embeddings.py:29-36, which documents this exact trap.
"""

import os
import re

_BACKEND_DIR = os.path.dirname(os.path.abspath(__file__))
_REPO_DIR = os.path.dirname(_BACKEND_DIR)
_PARENT_DIR = os.path.dirname(_REPO_DIR)


def _norm(p: str) -> str:
    """Absolutize, accepting a Git-Bash/MSYS '/d/OCR-data' as well as 'D:\\OCR-data'.

    run_project.sh runs under Git Bash but launches a NATIVE Windows python.exe, and MSYS
    translates argv, not arbitrary environment variables. Without this,
    os.path.abspath('/d/OCR-data') resolves against the current drive and silently becomes
    'D:\\d\\OCR-data' — a phantom tree that makedirs() cheerfully creates, so ingest
    "works", pdf_store reads empty, and every existing Corpus preview 404s with nothing in
    the log. run_project.sh also cygpath's these on the way out; this is the net for anyone
    who exports them by hand.
    """
    p = os.path.expanduser(p.strip().strip('"'))
    if os.name == "nt":
        m = re.fullmatch(r"/([A-Za-z])(/.*)?", p)
        if m:
            p = f"{m.group(1).upper()}:{m.group(2) or os.sep}"
    return os.path.abspath(p)


def _env(name: str) -> str:
    """The env override, or "". The emptiness test must happen BEFORE normalization:
    abspath("") returns the current working directory, never ""."""
    raw = os.getenv(name, "").strip()
    return _norm(raw) if raw else ""


# ── Models: env > sibling > legacy in-repo (loud) > sibling as the error target ──
_models_sibling = os.path.join(_PARENT_DIR, "OCR-models")
_models_legacy = os.path.join(_BACKEND_DIR, "models")

MODELS_DIR = _env("OCR_MODELS_DIR")
if not MODELS_DIR:
    if os.path.isdir(_models_sibling):
        MODELS_DIR = _models_sibling
    elif os.path.isdir(_models_legacy):
        print(f"[ocr_paths] WARNING: using pre-migration in-repo models at {_models_legacy}. "
              f"Move it to {_models_sibling} (or set OCR_MODELS_DIR).")
        MODELS_DIR = _models_legacy
    else:
        # Nothing exists. Point at the INTENDED location so the FileNotFoundError raised
        # downstream names the directory you should create, not a legacy one you deleted.
        MODELS_DIR = _models_sibling

# ── Data: env > sibling. No legacy tier, on purpose. ─────────────────────────
#
# The asymmetry with MODELS_DIR above is deliberate. A models fallback is READ-ONLY and
# fails loudly if wrong. A data fallback is a WRITE fallback: it would silently re-pollute
# the repo with the 265 MB store just moved out and split the corpus in two — old documents
# previewable, new ones not — with no error anywhere. There is also no single legacy data
# root to fall back to, since temp_* lived in backend/ while the stores lived in
# backend/rag/. The data move is one-shot by design.
DATA_DIR = _env("OCR_DATA_DIR") or os.path.join(_PARENT_DIR, "OCR-data")

TEMP_DIR = os.path.join(DATA_DIR, "temp")                    # temp_{sid}_page_*.txt, *.xlsx, uploads
PDF_STORE_DIR = os.path.join(DATA_DIR, "pdf_store")          # LIVE: content-addressed originals
FIGURE_STORE_DIR = os.path.join(DATA_DIR, "figure_store")    # LIVE: extracted figure PNGs
PREVIEW_CACHE_DIR = os.path.join(DATA_DIR, "preview_cache")  # pure cache, safe to delete wholesale

# TEMP_DIR is the one write target whose ~11 call sites do NOT makedirs — they relied on
# backend/ always existing. Create it once here rather than touching all of them. The three
# stores keep their own makedirs at write time (rag/ingest.py, rag/figures.py, and the
# server's preview writer), so this import stays a single idempotent syscall.
os.makedirs(TEMP_DIR, exist_ok=True)


def hf_snapshot(repo_id: str) -> str:
    """Resolve <MODELS_DIR>/models--<org>--<repo>/snapshots/<sha> WITHOUT pinning a sha.

    Snapshot shas used to be hardcoded in ocr_olm.py. That is a portability landmine: a
    re-download produces a different sha, and the pinned path then fails with "model dir
    missing", which reads as though the weights vanished.

    Resolution order is refs/main (HuggingFace's own canonical pointer — verified present
    here and byte-identical to the shas that were hardcoded), then "exactly one directory
    under snapshots/", which is the normal state for a cache holding one revision.
    """
    repo_dir = os.path.join(MODELS_DIR, "models--" + repo_id.replace("/", "--"))
    snaps = os.path.join(repo_dir, "snapshots")

    ref = os.path.join(repo_dir, "refs", "main")
    if os.path.isfile(ref):
        try:
            with open(ref, "r", encoding="utf-8") as f:
                sha = f.read().strip()
            cand = os.path.join(snaps, sha)
            if os.path.isdir(cand):
                return cand
        except OSError:
            pass  # fall through to the glob

    entries = sorted(
        d for d in (os.path.join(snaps, e) for e in os.listdir(snaps)) if os.path.isdir(d)
    ) if os.path.isdir(snaps) else []

    if len(entries) == 1:
        return entries[0]

    if not entries:
        raise FileNotFoundError(
            f"[ocr_paths] no snapshot for '{repo_id}' under {snaps}\n"
            f"            MODELS_DIR = {MODELS_DIR}  (override with OCR_MODELS_DIR)")

    # Ambiguous. REFUSE rather than guess. Picking newest-mtime looks helpful and is the
    # worst option available: after a re-download it would silently load a different
    # revision of the OCR model, and the only symptom would be slightly worse extraction —
    # no exception, no log line, HTTP 200 the whole way.
    raise RuntimeError(
        f"[ocr_paths] {len(entries)} snapshots for '{repo_id}' and no usable refs/main:\n"
        + "".join(f"              {e}\n" for e in entries)
        + f"            Pin one explicitly via OCR_OLM_MODEL_PATH / OCR_VL_PROCESSOR_PATH,\n"
          f"            or write the wanted sha into {ref}.")
