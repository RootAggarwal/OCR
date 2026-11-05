"""
backfill_pdf_store.py — one-time recovery of original files for the Corpus preview store.

The PDF preview feature (ingest.py's pdf_store + /rag/document-page-image) only retains a
document's original bytes for documents ingested AFTER it was added. Documents already in the
corpus have no retained original, so the Corpus tab shows "No Preview Available" for them.

This script recovers those originals SAFELY: it scans a bounded set of source directories,
SHA-256-hashes each candidate file, and copies a file into rag/pdf_store/{content_hash}.pdf
ONLY when its hash EXACTLY equals a document's stored `content_hash`. An exact byte-for-byte
hash match is proof the file is that document's original — there is no way to copy the wrong
document's bytes (the whole ambiguity that made the earlier "reuse leftover temp files" idea
unsafe is dissolved by hashing). Idempotent: skips any doc whose store file already exists.

Usage (from the backend/ directory, with the RAG DB env vars set):
    ../.venv/bin/python -m rag.backfill_pdf_store [extra_source_dir ...]

Default source dirs: ~/Downloads and the project root (for its incidental temp_*.pdf files).
"""

import glob
import hashlib
import os
import shutil
import sys
from typing import Dict, List

from . import db
from .ingest import PDF_STORE_DIR, pdf_store_path

# File extensions worth hashing as possible originals. PDFs are the norm; images are included
# so an image-sourced document (e.g. a scanned .png) can also be recovered — the read endpoint
# serves a matched image directly (it can't be rendered via pdftoppm).
_SOURCE_EXTS = ("pdf", "PDF", "png", "PNG", "jpg", "JPG", "jpeg", "JPEG", "tif", "TIF", "tiff", "TIFF")


def _default_source_dirs() -> List[str]:
    home = os.path.expanduser("~")
    # Project root is two levels up from this file (backend/rag/ -> backend/ -> project root).
    project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    return [os.path.join(home, "Downloads"), project_root]


def _candidate_files(source_dirs: List[str]) -> List[str]:
    """Non-recursive glob of each source dir for source-like files. Deliberately shallow:
    recursing into the project root would crawl node_modules/venvs and be pointlessly slow —
    originals live at the top level of Downloads or as temp_*.pdf at the project root."""
    files: List[str] = []
    for d in source_dirs:
        if not os.path.isdir(d):
            continue
        for ext in _SOURCE_EXTS:
            files.extend(glob.glob(os.path.join(d, f"*.{ext}")))
    # De-dup while preserving order.
    seen = set()
    unique = []
    for f in files:
        if f not in seen:
            seen.add(f)
            unique.append(f)
    return unique


def _sha256(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def backfill(source_dirs: List[str]) -> int:
    if not db.available():
        print("[backfill] RAG DB not configured; nothing to do.")
        return 0

    cols, rows = db.run_readonly(
        "SELECT filename, content_hash FROM documents WHERE status='ready'", row_limit=100000)
    # content_hash -> filename (a hash may map to several rows if a file was re-ingested under
    # the same bytes; the store path is keyed by hash alone, so one copy serves all of them).
    wanted: Dict[str, str] = {}
    for filename, content_hash in rows:
        wanted.setdefault(content_hash, filename)
    print(f"[backfill] {len(wanted)} distinct content hashes across ready documents")

    os.makedirs(PDF_STORE_DIR, exist_ok=True)
    already = {h for h, fn in wanted.items() if os.path.exists(pdf_store_path(h, fn))}
    to_find = {h: fn for h, fn in wanted.items() if h not in already}
    if already:
        print(f"[backfill] {len(already)} already have a stored original; skipping those")

    recovered: Dict[str, str] = {}
    for path in _candidate_files(source_dirs):
        if not to_find:
            break
        try:
            h = _sha256(path)
        except Exception as e:
            print(f"[backfill]   (skip unreadable {path}: {e})")
            continue
        if h in to_find and h not in recovered:
            dest = pdf_store_path(h, to_find[h])
            try:
                shutil.copyfile(path, dest)
                recovered[h] = path
                print(f"[backfill]   RECOVERED {to_find[h]}  <-  {path}")
            except Exception as e:
                print(f"[backfill]   FAILED to copy for {to_find[h]}: {e}")

    print("\n[backfill] summary:")
    for h, fn in wanted.items():
        if h in already:
            state = "already-had"
        elif h in recovered:
            state = "RECOVERED"
        else:
            state = "MISSING"
        print(f"  {state:12} {fn}")
    total_with_preview = len(already) + len(recovered)
    print(f"\n[backfill] {total_with_preview}/{len(wanted)} documents now have a stored original.")
    return len(recovered)


def main() -> None:
    extra = sys.argv[1:]
    source_dirs = _default_source_dirs() + [d for d in extra if os.path.isdir(d)]
    print(f"[backfill] scanning: {source_dirs}")
    backfill(source_dirs)


if __name__ == "__main__":
    main()
