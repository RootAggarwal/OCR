"""
embeddings.py — BGE-M3 dense embeddings for the prose (semantic) lane.

Runs on **CPU** by default and deliberately: the GPU is fully committed (vLLM ~26.6 GB +
the OCR VLMs ~16 GB leaves <3 GB), while the embedding volume here is tiny — a handful of
chunks per document. Set RAG_EMBED_DEVICE=cuda to move it if VRAM frees up.

Model files are loaded from the local HF cache only (`local_files_only=True`) — the stale
token in ~/.cache/huggingface/token otherwise fails hub calls with a 401.
"""

import os
import threading
from typing import List

MODEL_NAME = os.getenv("RAG_EMBED_MODEL", "BAAI/bge-m3")
DEVICE = os.getenv("RAG_EMBED_DEVICE", "cpu")
DIM = 1024  # must match prose_chunks.embedding vector(1024)

_model = None
_lock = threading.Lock()


def available() -> bool:
    """
    True when sentence-transformers is importable AND the model is already cached
    locally — otherwise the prose lane no-ops.

    The cache half of that test is load-bearing. This used to return True on the import
    alone, and _resolve_model_path() then fell back to a live snapshot_download() from
    inside the ingest call. Ingest runs in the upload's WebSocket handler, so on a slow
    or stalled link that 2.3 GB fetch blocks the handler indefinitely: OCR finishes, the
    client never receives processing_complete, and the upload sits frozen at 100% while
    the document never reaches the corpus. Never fetch a multi-GB model in a request
    path — prefetch it out-of-band (see _resolve_model_path) and degrade until it lands.
    """
    try:
        import sentence_transformers  # noqa: F401
    except Exception:
        return False
    return _cached_model_path() is not None


# Repo variants SentenceTransformer never loads. bge-m3 ships a ~2.3 GB onnx/ copy of
# the weights; requiring it would mean downloading the model twice.
_UNUSED_VARIANTS = ["onnx/*", "imgs/*", "*.onnx", "*.onnx_data"]
_WEIGHT_FILES = ("model.safetensors", "pytorch_model.bin")


def _cached_model_path():
    """The on-disk snapshot dir for MODEL_NAME, or None when it is not usable.

    Two subtleties, both of which produced a false "not cached" here:
      * snapshot_download(local_files_only=True) with no patterns demands EVERY file in
        the repo, so skipping the unused onnx/ variant made a complete, working cache
        look absent. Pass the same ignore_patterns the prefetch used.
      * A snapshot dir can exist holding only configs and tokenizer while the weights
        are still an unfinished *.incomplete blob, so check for real weights too.
    """
    if os.path.isdir(MODEL_NAME):
        return MODEL_NAME
    from huggingface_hub import snapshot_download
    try:
        # local_files_only => resolves from cache or raises; never touches the network.
        path = snapshot_download(MODEL_NAME, local_files_only=True,
                                 ignore_patterns=_UNUSED_VARIANTS)
    except Exception:
        return None
    if not any(os.path.exists(os.path.join(path, w)) for w in _WEIGHT_FILES):
        return None
    return path


def _resolve_model_path() -> str:
    """
    Resolve to the on-disk snapshot directory so loading never touches the hub.
    (`local_files_only=True` alone isn't enough — transformers then treats the repo id as
    a literal folder and fails; and a plain hub call hits the stale ~/.cache/huggingface
    token with a 401.)

    Raises rather than downloading: callers guard on available(), and ingest treats the
    raise as "insert NULL embeddings, backfill later".
    """
    path = _cached_model_path()
    if path is None:
        raise RuntimeError(
            f"{MODEL_NAME} is not in the local HF cache — refusing to download it inside "
            f"a request. Prefetch it once with:  huggingface-cli download {MODEL_NAME}"
        )
    return path


def _get_model():
    global _model
    if _model is None:
        with _lock:
            if _model is None:
                from sentence_transformers import SentenceTransformer
                path = _resolve_model_path()
                print(f"[rag.embeddings] loading {MODEL_NAME} on {DEVICE} (first call is slow)...")
                _model = SentenceTransformer(path, device=DEVICE)
                print("[rag.embeddings] ready")
    return _model


def embed(texts: List[str], batch_size: int = 8) -> List[List[float]]:
    """One L2-normalized 1024-dim vector per text (normalized => cosine == dot product)."""
    texts = [t for t in texts]
    if not texts:
        return []
    vecs = _get_model().encode(texts, normalize_embeddings=True,
                               batch_size=batch_size, show_progress_bar=False)
    return [list(map(float, v)) for v in vecs]


def to_pgvector(vec: List[float]) -> str:
    """pgvector text literal, e.g. '[0.1,0.2,...]'."""
    return "[" + ",".join(f"{x:.6f}" for x in vec) + "]"
