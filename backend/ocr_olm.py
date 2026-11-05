# ocr_olm.py — Enhanced OCR with format preservation and thread-safe concurrent processing
# (Historical commented-out revisions removed 2026-07-08; see VCS history if restored.)

import os
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

from io import BytesIO
import base64, warnings, json, asyncio, gc, re, threading, time, tempfile, os as _os
from typing import AsyncGenerator, Dict, Any, Optional, Tuple, List
from contextlib import contextmanager
import unicodedata
from PIL import Image, ImageEnhance
Image.MAX_IMAGE_PIXELS = None  # Trusted local PDFs — disable decompression bomb limit
from PyPDF2 import PdfReader
import torch
from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration

try:
    from olmocr.data.renderpdf import render_pdf_to_base64png as _olmocr_render
    _OLMOCR_AVAILABLE = True
except ImportError:
    _OLMOCR_AVAILABLE = False

try:
    from pdf2image import convert_from_bytes
    PDF2IMAGE = True
except ImportError:
    PDF2IMAGE = False

# ============================================================================
# CONSTANTS
# ============================================================================
DPI_SETTING         = 200
MAX_NEW_TOKENS      = 2048  # Match standalone script; 2048 is plenty for a full page
# Tight caps for the small auxiliary calls. Generation time scales with tokens
# emitted, and these answers are tiny (a header name, a small JSON object) — an
# unbounded 2048 budget only leaves room for a runaway ramble.
TOKENS_HEADER_CELL  = 32    # "Receiver m. secs"
TOKENS_CELL         = 48    # one table cell: "-112", "GD-6-1 (Dry)", "Receivers only"
TOKENS_JSON         = 256   # group-detection / spanning-header JSON
MIN_IMAGE_DIMENSION = 1200
PREVIEW_MAX_SIZE    = (400, 600)
MAX_PERF_SAMPLES    = 1000

# When enabled (default), pages are extracted with the geometry-grounded
# structured-table path (TATR detect + per-block VLM read + row repair) instead of
# a single whole-page VLM markdown pass. Set STRUCTURED_TABLES=0 to force the
# legacy unstructured (fast) behavior.
STRUCTURED_TABLES = os.getenv("STRUCTURED_TABLES", "1").lower() not in ("0", "false", "no")

# Page render resolution (longest edge). The structured path slices the page into
# per-block bands and reads each one, so it needs real detail: at 1288 the crops are
# small and the model hallucinates decimal points ("-112" → "-11.2") and garbles
# headers. Higher render costs a little time but markedly improves digit accuracy
# and TATR's column geometry. Stays under _enhance_image_for_ocr's 3000px cap.
RENDER_LONGEST_DIM = int(os.getenv("RENDER_DIM", "2200" if STRUCTURED_TABLES else "1288"))

warnings.filterwarnings("ignore", message=".*preprocessor.json.*")

from ocr_paths import hf_snapshot  # resolves under OCR_MODELS_DIR — see ocr_paths.py

# olmOCR-2 ships weights only: its snapshot carries no tokenizer/processor files, so the
# processor comes from the base Qwen2.5-VL repo (~25 MB of config+tokenizer, no weights).
# That is why these are two separate paths and not one.
#
# Snapshot shas are resolved from refs/main rather than hardcoded. A pinned sha breaks the
# moment anyone re-downloads the model, and it breaks as "model dir missing" — which reads
# as though the weights vanished. Set OCR_OLM_MODEL_PATH / OCR_VL_PROCESSOR_PATH to pin a
# directory explicitly (also the escape hatch if a cache ever holds two revisions).
#
# Raising HERE, at import, is deliberate. server_with_task_management.py imports this
# module at module scope, so a bad path means the process exits before binding :8000 and
# run_project.sh reports "backend exited early". Deferring to first use would instead give
# a healthy /health, a normal-looking UI, and a failure buried in a WebSocket handler on
# the first upload — the silent-degradation pattern this codebase keeps getting bitten by.
MODEL_PATH = os.getenv("OCR_OLM_MODEL_PATH") or hf_snapshot("allenai/olmOCR-2-7B-1025")
PROCESSOR_PATH = os.getenv("OCR_VL_PROCESSOR_PATH") or hf_snapshot("Qwen/Qwen2.5-VL-7B-Instruct")

print(f"Using MODEL_PATH = {MODEL_PATH}")
print(f"Using PROCESSOR_PATH = {PROCESSOR_PATH}")

# ============================================================================
# GLOBAL MODEL STATE  (supports single and dual model)
# ============================================================================
_PROCESSOR:   Any            = None
_MODEL_1_OBJ: Any            = None  # Primary model (always loaded)
_MODEL_2_OBJ: Any            = None  # Secondary model (only in dual mode)
_MODEL_1_LOCK: threading.Lock = threading.Lock()
_MODEL_2_LOCK: threading.Lock = threading.Lock()
_LOAD_LOCK:   threading.Lock = threading.Lock()

# ============================================================================
# PERFORMANCE TRACKING
# ============================================================================
_PAGE_TIMES:            List[float]    = []
_PAGES_PROCESSED:       int            = 0
_PROCESSING_START_TIME: Optional[float] = None
_PERF_LOCK:             threading.Lock = threading.Lock()


# ============================================================================
# MODEL LOADING
# ============================================================================

def _get_model_components(worker_id: int = 0) -> Tuple[Any, Any, threading.Lock]:
    """
    Thread-safe lazy loader. Routes to Model 1 or Model 2 based on worker_id.
    - worker_id % 2 == 0  →  Model 1  (always available)
    - worker_id % 2 == 1  →  Model 2  (only loaded in dual mode)

    Processor is shared between both models.
    Uses device_map='auto' so HuggingFace optimally places layers.
    Model 2 will attempt GPU 1 first, falling back to GPU 0 or CPU.
    """
    global _PROCESSOR, _MODEL_1_OBJ, _MODEL_2_OBJ

    use_model_1 = (worker_id % 2 == 0)

    # Check if already loaded
    if _PROCESSOR is not None:
        if use_model_1 and _MODEL_1_OBJ is not None:
            return _PROCESSOR, _MODEL_1_OBJ, _MODEL_1_LOCK
        if not use_model_1 and _MODEL_2_OBJ is not None:
            return _PROCESSOR, _MODEL_2_OBJ, _MODEL_2_LOCK

    with _LOAD_LOCK:
        # Shared processor (load once)
        if _PROCESSOR is None:
            print("[OCR] 🔄 Loading Processor...")
            _PROCESSOR = AutoProcessor.from_pretrained(
                PROCESSOR_PATH, use_fast=True, local_files_only=True
            )
            print("[OCR] ✓ Processor loaded")

        num_gpus = torch.cuda.device_count() if torch.cuda.is_available() else 0

        # Model 1 (primary)
        if use_model_1 and _MODEL_1_OBJ is None:
            print("[OCR] 🔄 Loading Model 1 (device_map=auto)...")
            _MODEL_1_OBJ = Qwen2_5_VLForConditionalGeneration.from_pretrained(
                MODEL_PATH,
                torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
                device_map="auto",
                low_cpu_mem_usage=True,
                local_files_only=True,
            ).eval()
            if torch.cuda.is_available():
                free_gb = torch.cuda.mem_get_info()[0] / 1024**3
                print(f"[OCR] ✓ Model 1 loaded — GPU free: {free_gb:.1f} GB")
            else:
                print("[OCR] ✓ Model 1 loaded (CPU mode)")

        # Model 2 (secondary — dual mode only)
        if not use_model_1 and _MODEL_2_OBJ is None:
            if _MODEL_1_OBJ is None:
                # Safety: Model 1 must be loaded first; fall back
                print("[OCR] ⚠️  Model 2 requested but Model 1 not loaded — loading Model 1 first")
                _get_model_components(0)  # Recursive load of Model 1

            print("[OCR] 🔄 Loading Model 2 (dual mode)...")
            # Try GPU 1, fall back to GPU 0, then CPU
            if num_gpus > 1:
                device_map = {"": 1}
            elif num_gpus == 1:
                device_map = {"": 0}  # Share GPU 0 (will interleave)
            else:
                device_map = "cpu"
            _MODEL_2_OBJ = Qwen2_5_VLForConditionalGeneration.from_pretrained(
                MODEL_PATH,
                torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
                device_map=device_map,
                low_cpu_mem_usage=True,
                local_files_only=True,
            ).eval()
            print(f"[OCR] ✓ Model 2 loaded (device_map={device_map})")

    if use_model_1:
        if _MODEL_1_OBJ is None:
            raise RuntimeError("[OCR] Model 1 failed to load")
        return _PROCESSOR, _MODEL_1_OBJ, _MODEL_1_LOCK
    else:
        if _MODEL_2_OBJ is None:
            raise RuntimeError("[OCR] Model 2 failed to load")
        return _PROCESSOR, _MODEL_2_OBJ, _MODEL_2_LOCK


def get_performance_stats() -> Dict[str, Any]:
    with _PERF_LOCK:
        times = _PAGE_TIMES.copy()
        pages = _PAGES_PROCESSED
        st    = _PROCESSING_START_TIME

    return {
        "total_pages_processed":  pages,
        "avg_time_per_page":      sum(times) / len(times) if times else 0,
        "min_time":               min(times) if times else 0,
        "max_time":               max(times) if times else 0,
        "total_processing_time":  sum(times),
        "session_info": {
            "model_1_loaded": _MODEL_1_OBJ is not None,
            "model_2_loaded": _MODEL_2_OBJ is not None,
            "uptime_seconds": time.time() - st if st else 0,
        },
    }


def reset_performance_stats():
    global _PAGE_TIMES, _PAGES_PROCESSED, _PROCESSING_START_TIME
    with _PERF_LOCK:
        _PAGE_TIMES.clear()
        _PAGES_PROCESSED = 0
        _PROCESSING_START_TIME = time.time()


@contextmanager
def memory_cleanup():
    try:
        yield
    finally:
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()


# ============================================================================
# HELPERS
# ============================================================================

def _is_pdf(buf: bytes) -> bool:
    return buf.startswith(b"%PDF")


def _detect_script_and_language(text: str) -> Dict[str, Any]:
    if not text.strip():
        return {"script": "unknown", "language": "unknown", "confidence": 0.0}

    scripts = {k: 0 for k in
               ['latin', 'cyrillic', 'arabic', 'chinese', 'japanese', 'korean', 'devanagari', 'other']}

    for char in text:
        if unicodedata.category(char)[0] == 'L':
            n = unicodedata.name(char, '').lower()
            if any(x in n for x in ['latin', 'basic']):  scripts['latin'] += 1
            elif 'cyrillic' in n:                         scripts['cyrillic'] += 1
            elif any(x in n for x in ['arabic', 'persian']): scripts['arabic'] += 1
            elif any(x in n for x in ['cjk', 'chinese']): scripts['chinese'] += 1
            elif 'hiragana' in n or 'katakana' in n:      scripts['japanese'] += 1
            elif 'hangul' in n:                           scripts['korean'] += 1
            elif 'devanagari' in n:                       scripts['devanagari'] += 1
            else:                                         scripts['other'] += 1

    total = sum(scripts.values())
    if total == 0:
        return {"script": "unknown", "language": "unknown", "confidence": 0.0}

    dominant = max(scripts, key=scripts.get)
    mapping = {
        'latin': 'multilingual', 'cyrillic': 'russian/slavic',
        'arabic': 'arabic/persian', 'chinese': 'chinese',
        'japanese': 'japanese', 'korean': 'korean',
        'devanagari': 'hindi/sanskrit', 'other': 'unknown',
    }
    return {"script": dominant, "language": mapping[dominant], "confidence": scripts[dominant] / total}


def _get_format_preserving_prompt(kind: str = "document") -> str:
    """
    Returns a strict prompt that forces the VLM to use Markdown for tables.
    """
    base = (
        "You are an expert OCR and document digitization system. Extract ALL visible text exactly as it appears. "
        "CRITICAL REQUIREMENTS:\n"
        "1. Preserve ALL original formatting: indentation, spacing, line breaks, paragraphs\n"
        "2. Maintain original language - DO NOT translate or convert languages\n"
        "3. Keep exact capitalization, punctuation, and special characters\n"
        "4. **TABLES MUST BE IN STRICT MARKDOWN FORMAT**: When you encounter tabular data or a grid, you MUST use strict Markdown table syntax (e.g. | Column 1 | Column 2 |). "
        "   - You MUST include a header row and a separator row (|---|---|).\n"
        "   - You MUST properly align all columns.\n"
        "   - **CRITICAL**: If a cell is empty or blank, you MUST include it as an empty Markdown cell (e.g. `| |`). Do NOT skip columns or merge them if they are visually distinct.\n"
        "   - If a table is split into multiple visual sub-tables horizontally, merge them logically or extract them as separate Markdown tables.\n"
        "5. Include headers, footers, captions, and annotations\n"
        "6. Maintain reading order (left-to-right, top-to-bottom)\n"
        "7. For multi-column text, preserve column structure\n"
        "8. Include mathematical symbols, equations, formulas as they appear\n"
        "9. DO NOT add explanations, interpretations, or modifications\n"
        "10. Return ONLY the extracted text content in its original format\n\n"
    )
    if kind == "image":
        return base + "Extract from this image maintaining exact visual layout. If handwritten, interpret carefully. Ensure strict markdown tables are used."
    return base + "Extract from this document page maintaining exact layout. Any tables must be converted to strict Markdown tables."


def _get_fallback_html_prompt() -> str:
    """
    Used when TATR fails to find a table grid entirely.
    Directs the VLM to use HTML tables instead of Markdown pipe tables to prevent column shifting on empty cells.
    """
    return (
        "You are an expert OCR system. Extract the table in this image into a single strict HTML <table> element.\n"
        "CRITICAL REQUIREMENTS:\n"
        "1. Output ONLY a single HTML <table>. No prose, no markdown, no ``` fences.\n"
        "2. Put the column names in a <thead><tr> of <th> cells. Include ALL headers, even if they span multiple lines.\n"
        "3. This is likely a vintage, typewritten, or handwritten table without grid lines. Use visual indentation to determine column alignment.\n"
        "4. If a cell is blank in the image, you MUST emit an empty <td></td>. NEVER skip a cell or shift values left.\n"
        "5. Maintain exact text, do not invent values.\n"
    )


def _get_structured_table_prompt(headers: List[str], num_cols: int,
                                 expected_rows: Optional[int] = None) -> str:
    """
    Strict prompt used by table_extractor for reading a SINGLE table block whose
    column count is already known from geometry. Forces exactly `num_cols` cells
    per row so column contents can never shift, and (when known) nudges the model
    to emit every row so trailing rows are not dropped.
    """
    header_list = " | ".join(str(h) for h in headers) if headers else ""
    row_line = (
        f"7. This block has approximately {expected_rows} data rows. Output EVERY row "
        "top-to-bottom — do NOT stop early and do NOT skip the last rows near the bottom border.\n"
        if expected_rows else
        "7. Read rows strictly top-to-bottom. Output one <tr> per visible data row; "
        "do not skip the last rows near the bottom border.\n"
    )
    # HTML, deliberately — NOT a compact pipe format. Pipes are ~4x cheaper to decode,
    # but they were tried and they cost accuracy in two separate ways: bare pipes made
    # the model silently DROP blank cells (shifting every later value left — the exact
    # bug this extractor exists to prevent), and adding an EMPTY placeholder to stop
    # that made it swallow real '-' marks and, once, an entire band. An explicit
    # <td></td> is unambiguous and is what the verified-correct output was built on.
    # Speed is bought elsewhere (masked prose pass, bounded crops, batching).
    return (
        "You are an expert OCR system reading ONE table from an image. "
        f"This table has EXACTLY {num_cols} columns, in this order: {header_list}.\n"
        "STRICT REQUIREMENTS:\n"
        f"1. Output ONLY a single HTML <table>. No prose, no markdown, no ``` fences.\n"
        "2. Put the column names in a <thead><tr> of <th> cells.\n"
        f"3. Every data row MUST be a <tr> with EXACTLY {num_cols} <td> cells, in the same column order.\n"
        "4. If a cell is blank in the image, still emit an empty <td></td> in its position. "
        "NEVER skip a cell, NEVER merge cells, NEVER shift values left to fill a gap.\n"
        "5. Two values separated by a vertical rule are in TWO DIFFERENT columns, even if "
        "they read as one phrase. A narrow number beside a name (e.g. '01' next to "
        "'GGN-11') is its OWN column — put them in separate <td> cells, never combined "
        "into one.\n"
        "6. Read values exactly as written (including handwriting, minus signs, and marks like 'X'). "
        "Do not translate, round, or invent values.\n"
        + row_line +
        "Return ONLY the <table>...</table> markup."
    )


def _get_prose_only_prompt() -> str:
    """
    Used when the geometric extractor has already captured the page's table(s).
    The old behaviour re-read the whole page (including the table, as markdown) and
    then threw the table away — ~3500 wasted output chars, ~70s of decoding. Here we
    ask for only the text OUTSIDE any table, which is a fraction of the tokens.
    """
    return (
        "You are an expert OCR system. Extract ONLY the text that lies OUTSIDE any table "
        "on this page.\n"
        "INCLUDE: titles/headings, form fields and their values (e.g. 'Geophysical Party "
        "No....83'), notes and footnotes, signatures/captions.\n"
        "EXCLUDE COMPLETELY: everything inside a table or ruled grid — its column headers "
        "and every data cell. The tables are already handled separately.\n"
        "Do NOT output a table, a grid, markdown pipes (|), or HTML. Do not describe the "
        "table or say that it was omitted — simply skip it.\n"
        "Preserve the original line breaks and reading order of the text you do return. "
        "Return ONLY that text."
    )


def _get_cell_prompt() -> str:
    """
    Read ONE table cell, cropped to its own bounding box. Used when the model has been
    shown to ignore the column count on a whole-table read — a cell crop makes merging
    impossible, since the neighbouring column's text simply isn't in the image.
    """
    return (
        "This image is a single cell cut out of a table. "
        "Return ONLY the text inside it, exactly as written — a number, a code, a short "
        "phrase, or a mark such as X. Keep minus signs. "
        "If the cell is blank, reply with nothing at all. "
        "No quotes, no explanation, no punctuation you cannot see."
    )


def _get_header_cell_prompt() -> str:
    """Read the header text of ONE column (a single narrow crop)."""
    return (
        "This image is the header cell of a single table column. "
        "Return ONLY the column's own name, exactly as printed. "
        "If it is written on two lines, join them with a space. "
        "Ignore any wider title printed above it that also covers neighbouring columns. "
        "No quotes, no JSON, no explanation — just the name."
    )


def _get_group_detection_prompt() -> str:
    """
    Prompt that asks the VLM to report the repeated-column-group layout of a table
    crop (for side-by-side sub-tables) plus the header names of a single group.
    """
    return (
        "Look at this table image. Some forms repeat the SAME set of columns several "
        "times side-by-side (e.g. three identical blocks placed next to each other), and "
        "some have a two-level header where a title cell spans several columns.\n"
        "NOTE: This may be a vintage or typewritten document without visible grid lines. "
        "Infer the column headers from visual indentation and spacing even if borders are missing.\n"
        "Answer with ONLY a JSON object, no prose, no ``` fences, in this exact shape:\n"
        '{"groups": <int number of identical side-by-side column-blocks>, '
        '"headers": [<bottom-level data-column headers of ONE block, left-to-right>], '
        '"spanning": [{"label": <title text>, "over": [<the exact leaf header names it spans>]}]}\n'
        "Rules:\n"
        "- 'headers' lists ONLY the individual bottom-most data columns of ONE block.\n"
        "- Include ALL headers, even if they span multiple lines. Combine multi-line headers into a single string.\n"
        "- If a header cell spans/merges several of those columns (a title sitting ABOVE a "
        "group of columns, e.g. 'Corrections' above 'Shot m. secs' and 'Receiver m. secs'), "
        "put it in 'spanning' with the exact leaf header names it covers. Use [] if there are none.\n"
        "- One name per actual column in 'headers', left-to-right, for a single block only.\n"
        "If the table is an ordinary single table (columns NOT repeated side-by-side), "
        'return "groups": 1.\n'
        "Count blocks by how many times the SAME leftmost header repeats across the width."
    )


def _extract_and_preserve_formatting(raw: str) -> str:
    try:
        text = raw.strip()
        if text.startswith(('{', '[')):
            try:
                data = json.loads(text)
                if isinstance(data, dict):
                    for key in ['text', 'content', 'output']:
                        if key in data and data[key]:
                            text = str(data[key]); break
            except json.JSONDecodeError:
                pass
        text = text.replace('\\n', '\n').replace('\\t', '\t')
        text = re.sub(r'^["\'`]+|["\'`]+$', '', text.strip())
        return text
    except Exception as e:
        print(f"[WARN] Format preservation error: {e}")
        return raw.strip()


MAX_IMAGE_DIMENSION = 3000  # Cap: downscale any dimension above this before OCR

# Bounds on the "too small → upscale" rule below. Inference cost scales with pixel
# count (vision tokens), and the min-dimension rule alone is unbounded: a wide, short
# crop like a 633x132 header strip was being blown up to 5754x1200 (6.9M px), which
# cost 12.5s to answer with 46 characters. Upscaling a small crop adds no information
# anyway — it only needs enough pixels for the vision encoder.
# These are chosen so a full page and a table band come out UNCHANGED (a 653x875 band
# upscales 1.84x to 1200x1608 = 1.93M px), and only the tiny header crops shrink.
MAX_UPSCALE_FACTOR = 4.0
MAX_TOTAL_PIXELS   = 2_200_000


def _enhance_image_for_ocr(img: Image.Image) -> Image.Image:
    if img.mode != 'RGB':
        img = img.convert('RGB')
    w, h = img.size
    # Downscale if image is too large (e.g. very high-DPI olmOCR renders)
    if max(w, h) > MAX_IMAGE_DIMENSION:
        s = MAX_IMAGE_DIMENSION / max(w, h)
        img = img.resize((int(w * s), int(h * s)), Image.LANCZOS)
        w, h = img.size
    # Upscale if image is too small for the model — bounded by factor, longest edge
    # and total pixels so a thin strip cannot explode into millions of blank pixels.
    if min(w, h) < MIN_IMAGE_DIMENSION:
        s = min(
            MIN_IMAGE_DIMENSION / min(w, h),
            MAX_UPSCALE_FACTOR,
            MAX_IMAGE_DIMENSION / max(w, h),
            (MAX_TOTAL_PIXELS / (w * h)) ** 0.5,
        )
        if s > 1.0:
            img = img.resize((max(1, int(w * s)), max(1, int(h * s))), Image.LANCZOS)
    img = ImageEnhance.Contrast(img).enhance(1.2)
    img = ImageEnhance.Sharpness(img).enhance(1.1)
    return img


def _create_optimized_preview(img: Image.Image) -> str:
    p = img.copy()
    # Use higher resolution for sharper previews (was 400x600)
    p.thumbnail((1200, 1800), Image.LANCZOS)
    if p.mode != 'RGB':
        p = p.convert('RGB')
    buf = BytesIO()
    p.save(buf, format='JPEG', quality=92, optimize=True)
    return f"data:image/jpeg;base64,{base64.b64encode(buf.getvalue()).decode()}"


def _pil_to_b64(img: Image.Image) -> str:
    """Convert a PIL Image to a base64-encoded PNG string."""
    buf = BytesIO()
    img.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode()


# ============================================================================
# CORE INFERENCE  (single model)
# ============================================================================

def _build_chat_text(processor, prompt: str, img: Image.Image) -> str:
    messages = [{
        "role": "user",
        "content": [
            {"type": "text",      "text":      prompt},
            {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{_pil_to_b64(img)}"}},
        ],
    }]
    return processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)


def _generate(processor, model, model_lock, device,
              images: List[Image.Image], prompts: List[str], max_new_tokens: int) -> List[str]:
    """
    One greedy generate() over a batch of (image, prompt) pairs. A batch of 1 is the
    ordinary single-image path.

    Batched decoder-only generation REQUIRES left padding: with the default right
    padding, shorter prompts get pad tokens appended and the model then continues from
    a pad token, silently producing garbage for those samples. With left padding every
    sample's completion starts at the same offset, so slicing off the padded prompt
    length is correct for all of them.
    """
    texts = [_build_chat_text(processor, p, im) for p, im in zip(prompts, images)]

    prev_side = getattr(processor.tokenizer, "padding_side", "right")
    if len(images) > 1:
        processor.tokenizer.padding_side = "left"
    try:
        inputs = processor(text=texts, images=images, return_tensors="pt", padding=True)
        inputs = {k: v.to(device) for k, v in inputs.items()}

        with model_lock:
            with torch.no_grad():
                # Greedy decoding: OCR should be deterministic (reproducible exports)
                # and greedy is marginally faster than sampling.
                output_ids = model.generate(
                    **inputs,
                    do_sample=False,
                    max_new_tokens=max_new_tokens,
                    num_return_sequences=1,
                    pad_token_id=processor.tokenizer.pad_token_id,
                    use_cache=True,
                )

        new_ids = output_ids[:, inputs["input_ids"].shape[1]:]
        decoded = processor.tokenizer.batch_decode(new_ids, skip_special_tokens=True)
        del inputs, output_ids, new_ids
        return decoded
    finally:
        processor.tokenizer.padding_side = prev_side


def _run_enhanced_ocr(
    img: Image.Image,
    prompt: str,
    worker_id: int = 0,
    max_new_tokens: Optional[int] = None,
) -> Tuple[str, Dict[str, Any]]:
    """Run OCR on a PIL image. Thread-safe via per-model lock. worker_id selects which model to use."""
    global _PAGES_PROCESSED, _PROCESSING_START_TIME

    t0 = time.time()
    if _PROCESSING_START_TIME is None:
        with _PERF_LOCK:
            if _PROCESSING_START_TIME is None:
                _PROCESSING_START_TIME = time.time()

    processor, model, model_lock = _get_model_components(worker_id)
    first_device = next(model.parameters()).device
    processed_img = _enhance_image_for_ocr(img)

    with memory_cleanup():
        t1 = time.time()
        decoded = _generate(processor, model, model_lock, first_device,
                            [processed_img], [prompt], max_new_tokens or MAX_NEW_TOKENS)[0]
        inf_time   = time.time() - t1
        text       = _extract_and_preserve_formatting(decoded)
        lang_info  = _detect_script_and_language(text)
        total_time = time.time() - t0

        with _PERF_LOCK:
            _PAGE_TIMES.append(total_time)
            if len(_PAGE_TIMES) > MAX_PERF_SAMPLES:
                _PAGE_TIMES.pop(0)
            _PAGES_PROCESSED += 1

        print(f"[OCR] ⏱️  inf: {inf_time:.2f}s | total: {total_time:.2f}s")
        return text, lang_info


def _run_enhanced_ocr_batch(
    images: List[Image.Image],
    prompts: List[str],
    worker_id: int = 0,
    max_new_tokens: Optional[int] = None,
) -> List[str]:
    """
    Read several independent (image, prompt) pairs in ONE generate() call.

    Used for the per-block band reads and the per-column header reads, which are
    independent of each other — running them sequentially paid the prefill/decode
    launch cost N times. Falls back to sequential calls if the batched path fails,
    so a batching bug degrades to slow rather than wrong.
    """
    if not images:
        return []
    if len(images) == 1:
        return [_run_enhanced_ocr(images[0], prompts[0], worker_id, max_new_tokens)[0]]

    processor, model, model_lock = _get_model_components(worker_id)
    first_device = next(model.parameters()).device
    processed = [_enhance_image_for_ocr(im) for im in images]

    try:
        with memory_cleanup():
            t1 = time.time()
            decoded = _generate(processor, model, model_lock, first_device,
                                processed, prompts, max_new_tokens or MAX_NEW_TOKENS)
            print(f"[OCR] ⏱️  batch x{len(images)}: {time.time() - t1:.2f}s")
            return [_extract_and_preserve_formatting(d) for d in decoded]
    except Exception as e:
        print(f"[OCR] ⚠️  Batched inference failed ({e}) — falling back to sequential")
        return [_run_enhanced_ocr(im, p, worker_id, max_new_tokens)[0]
                for im, p in zip(images, prompts)]


# ============================================================================
# STRUCTURED TABLE PATH  (geometry-grounded, column-shift-proof)
# ============================================================================

def _strip_and_splice_tables(prose: str, tables: List[str]) -> str:
    """
    Remove the whole-page VLM's own (unreliable) table blocks from `prose` and
    splice the geometry-grounded aligned HTML `tables` in at the position of the
    first removed table, preserving surrounding prose order (header fields above,
    footnotes below).
    """
    # 1. Drop any HTML <table>...</table> the VLM emitted.
    had_html_table = bool(re.search(r'<table[\s\S]*?</table>', prose, re.IGNORECASE))
    prose = re.sub(r'<table[\s\S]*?</table>', '\x00TBL\x00', prose, flags=re.IGNORECASE)

    # 2. Walk lines; drop markdown table blocks, remember where the first one was.
    lines = prose.split('\n')
    out: List[str] = []
    inserted = False
    in_tbl = False
    joined = "\n\n".join(tables)

    def _insert():
        nonlocal inserted
        if not inserted:
            out.append(joined)
            inserted = True

    for line in lines:
        is_md_tbl = bool(re.match(r'^\s*\|.*\|', line)) or (in_tbl and bool(re.match(r'^\s*[-|:\s]+$', line)))
        if '\x00TBL\x00' in line:
            # Placeholder for a stripped HTML table → insertion point.
            _insert()
            continue
        if is_md_tbl:
            in_tbl = True
            continue  # drop the markdown table line
        if in_tbl:
            _insert()
            in_tbl = False
        out.append(line)

    if in_tbl:
        _insert()

    if not inserted:
        # No VLM table detected in prose — append aligned tables at the end.
        if had_html_table:
            # Shouldn't happen (placeholder handled above), but be safe.
            out.append(joined)
        else:
            out.append("")
            out.append(joined)

    return "\n".join(out).strip() + "\n"


PROSE_RENDER_DIM = 1288  # prose is printed text — it does not need the table's resolution


def _downscale_for_prose(img: Image.Image) -> Image.Image:
    """Shrink the page for the prose pass. Vision-token prefill scales with pixels,
    and the surrounding text is printed (not handwritten), so it reads fine small."""
    if max(img.size) <= PROSE_RENDER_DIM:
        return img
    s = PROSE_RENDER_DIM / max(img.size)
    return img.resize((max(1, int(img.width * s)), max(1, int(img.height * s))), Image.LANCZOS)


def _mask_table_regions(img: Image.Image, boxes: List[List[float]]) -> Image.Image:
    """
    White out the detected table regions before the prose pass.

    Asking the model to "skip the table" does not work — olmOCR is trained to
    transcribe the whole page and re-emitted the entire table as HTML (2600+ chars,
    ~50s) despite explicit instructions. Removing the table from the *image* is the
    only reliable way: what it cannot see, it cannot transcribe. This also guarantees
    the prose can never duplicate or contradict the geometric table.
    """
    if not boxes:
        return img
    from PIL import ImageDraw
    masked = img.convert("RGB").copy()
    draw = ImageDraw.Draw(masked)
    for x0, y0, x1, y1 in boxes:
        draw.rectangle([int(x0), int(y0), int(x1), int(y1)], fill="white")
    return masked


def _process_page_structured(img: Image.Image, worker_id: int = 0) -> Tuple[str, Dict[str, Any]]:
    """
    Geometry-grounded page extraction:
      1. table_extractor.extract_page_html() for aligned, column-shift-proof tables.
      2. A VLM pass for the surrounding prose / form key-values.
      3. Splice the aligned tables into the prose in place of any table the VLM emitted.

    Tables are extracted FIRST so that, when we have them, the prose pass can be told to
    skip table contents entirely. Previously it re-read the whole table as markdown and
    _strip_and_splice_tables then deleted it — ~3500 wasted output chars per page.

    Falls back to the plain whole-page VLM output on any failure — never a hard error.
    """
    tables: List[str] = []
    boxes: List[List[float]] = []
    structured: List[Any] = []
    # Bound to the module only if the import itself succeeded. The except below swallows
    # everything (this function must never hard-error, see docstring), so a failed import
    # would otherwise leave the name unbound and turn the harvest calls further down into
    # a NameError on every page.
    _te = None
    try:
        import table_extractor as _te
        result = _te.extract_page_html(img)
        tables = result.get("tables", [])
        boxes = result.get("boxes", [])
        structured = result.get("structured", [])
    except Exception as e:
        print(f"[OCR] ⚠️  Structured extraction failed ({e}) — using whole-page VLM output")

    def _harvest(text: str, source: str) -> List[Dict[str, Any]]:
        """Best-effort table harvest for the RAG tables lane — never fatal.

        Same contract as everything else here: a page that yields no tables is a degraded
        knowledge base, not a failed OCR job. The page's text is returned either way.
        """
        if _te is None:
            return []
        try:
            return [dict(t, source=source) for t in _te.harvest_inline_tables(text)]
        except Exception as exc:
            print(f"[OCR] ⚠️  table harvest skipped ({exc})")
            return []

    if not tables:
        # No table on this page: read it in full, exactly as the legacy path does.
        text, lang_info = _run_enhanced_ocr(
            img, _get_format_preserving_prompt("document"), worker_id=worker_id)
        # Whole page is prose for the RAG lane; no structured tables.
        #
        # But the VLM emits HTML tables natively, so a page TATR missed can still CONTAIN
        # tables — inline in this text. Harvest them for the tables lane, otherwise the
        # only tabular pages in the knowledge base would be the ones TATR found, and on
        # this corpus that is the minority (KJ-17 and KJ-12 have zero structured cells,
        # yet 25 tables between them). page_tables holds the layout — rowspan/colspan/<br>
        # — that the structured unpivot cannot represent.
        lang_info["_rag"] = {
            "prose": text,
            "structured": [],
            "page_tables": _harvest(text, "prose_html"),
        }
        return text, lang_info

    # Mask the table out of the page, then read what is left. The table is already
    # captured geometrically; masking is what actually stops the model re-transcribing
    # it (the prompt alone does not).
    prose_img = _downscale_for_prose(_mask_table_regions(img, boxes))
    prose, lang_info = _run_enhanced_ocr(
        prose_img, _get_prose_only_prompt(), worker_id=worker_id
    )
    # _strip_and_splice_tables stays as a safety net: the model may still emit a stray
    # table despite the instruction, and it also decides where the tables get spliced.
    combined = _strip_and_splice_tables(prose, tables)
    # RAG lane payload (internal; the WS handler pops it before sending to the frontend):
    # masked-pass prose + the per-table structured band rows for the DB unpivot.
    #
    # `page_tables` carries the geometry-grounded HTML that used to be computed here and
    # then thrown away — only `combined` (frontend) and `structured` (DB unpivot) survived,
    # so the layout was lost at ingest. Harvest from `combined` rather than the raw
    # `tables` list so the aligned tables are picked up in reading order alongside any
    # stray table the masked pass still emitted, using the same parser as the branch above.
    lang_info["_rag"] = {
        "prose": prose,
        "structured": structured,
        "page_tables": _harvest(combined, "tatr"),
    }
    _n_struct = sum(1 for s in structured if s)
    print(f"[OCR] 🧩 Structured tables: spliced {len(tables)} aligned table(s); "
          f"{_n_struct}/{len(structured)} carry structured rows for the RAG lane")
    return combined, lang_info


def _ocr_page(img: Image.Image, prompt_kind: str = "document", worker_id: int = 0) -> Tuple[str, Dict[str, Any]]:
    """
    Single entry the page wrappers call: routes to the structured path when
    STRUCTURED_TABLES is on, else the legacy single-pass VLM read.
    """
    if STRUCTURED_TABLES:
        return _process_page_structured(img, worker_id=worker_id)
    return _run_enhanced_ocr(img, _get_format_preserving_prompt(prompt_kind), worker_id=worker_id)


# ============================================================================
# CHAT INFERENCE (text-only)
# ============================================================================

def run_plain_text_inference(prompt: str, max_new_tokens: int = 800, temperature: float = 0.7) -> str:
    processor, model, model_lock = _get_model_components()
    first_device = next(model.parameters()).device

    messages = [{"role": "user", "content": [{"type": "text", "text": prompt}]}]
    text_input = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

    with memory_cleanup():
        inputs = processor(text=text_input, images=None, return_tensors="pt")
        inputs = {k: v.to(first_device) for k, v in inputs.items()}

        with model_lock:
            with torch.no_grad():
                output_ids = model.generate(
                    **inputs,
                    max_new_tokens=max_new_tokens, temperature=temperature,
                    do_sample=True, top_p=0.9, repetition_penalty=1.1,
                    pad_token_id=processor.tokenizer.pad_token_id, use_cache=True,
                )

        new_ids  = output_ids[:, inputs["input_ids"].shape[1]:]
        response = processor.tokenizer.batch_decode(new_ids, skip_special_tokens=True)[0].strip()

        for marker in ("Response:", "Analysis:"):
            if marker in response:
                response = response.split(marker)[-1].strip()

        del inputs, output_ids, new_ids
        return response


# ============================================================================
# PAGE WRAPPERS
# ============================================================================

def _process_page_wrapper(
    page_num: int, buf: bytes, total_pages: int,
    forced_worker_id: Optional[int] = None   # kept for API compat, ignored
) -> Dict[str, Any]:
    print(f"[OCR] 🔄 Processing Page {page_num}/{total_pages}...")
    try:
        with memory_cleanup():
            images = convert_from_bytes(buf, dpi=DPI_SETTING, first_page=page_num, last_page=page_num, thread_count=1)
            if not images:
                raise ValueError(f"No image for page {page_num}")
            img     = images[0]
            text, lang_info = _ocr_page(img, "document")
            preview = _create_optimized_preview(img)
            print(f"[OCR] ✓ Page {page_num} done ({len(text)} chars)")
            return {"type": "page_complete", "page": page_num, "text": text,
                    "language_info": lang_info, "error": None,
                    "total_pages": total_pages, "status": "completed", "preview": preview}
    except Exception as e:
        print(f"[OCR] ✗ Page {page_num} FAILED: {e}")
        return {"type": "page_complete", "page": page_num, "text": "",
                "language_info": {"script": "unknown", "language": "unknown", "confidence": 0.0},
                "error": str(e), "total_pages": total_pages, "status": "error", "preview": None}


def _process_page_from_image(
    page_num: int, img: Image.Image, total_pages: int,
    forced_worker_id: Optional[int] = None   # kept for API compat, ignored
) -> Dict[str, Any]:
    print(f"[OCR] 🔄 Processing Page {page_num}/{total_pages}...")
    try:
        with memory_cleanup():
            text, lang_info = _ocr_page(img, "document")
            preview = _create_optimized_preview(img)
            print(f"[OCR] ✓ Page {page_num} done ({len(text)} chars)")
            return {"type": "page_complete", "page": page_num, "text": text,
                    "language_info": lang_info, "error": None,
                    "total_pages": total_pages, "status": "completed", "preview": preview}
    except Exception as e:
        print(f"[OCR] ✗ Page {page_num} FAILED: {e}")
        return {"type": "page_complete", "page": page_num, "text": "",
                "language_info": {"script": "unknown", "language": "unknown", "confidence": 0.0},
                "error": str(e), "total_pages": total_pages, "status": "error", "preview": None}


def _process_single_image_enhanced(buf: bytes) -> Dict[str, Any]:
    """Process a single image upload."""
    print("[OCR] 🔄 Processing single image...")
    try:
        with memory_cleanup():
            img = Image.open(BytesIO(buf))
            text, lang_info = _ocr_page(img, "image")
            preview = _create_optimized_preview(img)
            rag = lang_info.pop("_rag", None) if isinstance(lang_info, dict) else None
            print(f"[OCR] ✓ Image done ({len(text)} chars)")
            return {"success": True,
                    "pages": [{"page": 1, "text": text, "language_info": lang_info,
                                "error": None, "preview": preview, "rag": rag}],
                    "total_pages": 1, "error": None}
    except Exception as e:
        print(f"[OCR] ✗ Image FAILED: {e}")
        return {"success": False, "pages": [], "total_pages": 0, "error": f"Image processing failed: {e}"}


# ============================================================================
# STREAMING ENTRY POINTS
# ============================================================================

def _parse_page_spec(spec: Optional[str], total_pages: int) -> List[int]:
    """
    Parse a 1-indexed page selection ("3", "1-5", "2,4,6-8") into a sorted, unique
    list of page numbers clamped to [1, total_pages]. Empty/None/unparseable → all
    pages, so the default behaviour is unchanged.
    """
    if not spec or not str(spec).strip():
        return list(range(1, total_pages + 1))
    pages: set = set()
    for part in str(spec).replace(" ", "").split(","):
        if not part:
            continue
        if "-" in part:
            a, _, b = part.partition("-")
            try:
                lo, hi = int(a), int(b)
            except ValueError:
                continue
            if lo > hi:
                lo, hi = hi, lo
            pages.update(p for p in range(lo, hi + 1) if 1 <= p <= total_pages)
        else:
            try:
                p = int(part)
            except ValueError:
                continue
            if 1 <= p <= total_pages:
                pages.add(p)
    # A spec WAS given but nothing valid resolved (e.g. all out of range): return
    # empty so the caller can surface an error, rather than silently doing all pages.
    return sorted(pages)


async def stream_ocr_bytes(buf: bytes, mode: str = "single",
                           pages: Optional[str] = None) -> AsyncGenerator[Dict[str, Any], None]:
    if _is_pdf(buf):
        async for result in _stream_pdf_enhanced(buf, mode=mode, pages=pages):
            yield result
    else:
        yield _process_single_image_enhanced(buf)


async def _stream_pdf_enhanced(buf: bytes, mode: str = "single",
                               pages: Optional[str] = None) -> AsyncGenerator[Dict[str, Any], None]:
    """
    Stream PDF pages using olmOCR's native per-page renderer.
    In single mode: all pages use Model 1 (worker_id=0) sequentially.
    In dual mode: preload both models, then process pages concurrently with 2 workers,
    alternating between Model 1 (even pages → worker_id=0) and Model 2 (odd pages → worker_id=1).
    """
    from concurrent.futures import ThreadPoolExecutor
    import asyncio

    tmp_path = None
    try:
        reader      = PdfReader(BytesIO(buf))
        total_pages = len(reader.pages)
        selected    = _parse_page_spec(pages, total_pages)

        _preview = selected if len(selected) <= 12 else selected[:12] + ["..."]
        print(f"[OCR] 📄 {total_pages} page(s) in file; processing {len(selected)}: "
              f"{_preview} (mode={mode}, per-page render)...")

        if not selected:
            yield {"type": "error",
                   "error": f"No valid pages in your selection. This document has {total_pages} page(s)."}
            return

        # Write bytes to a temp file so olmOCR renderer gets a real path
        with tempfile.NamedTemporaryFile(suffix='.pdf', delete=False) as tmp:
            tmp.write(buf)
            tmp_path = tmp.name

        if mode == "dual" and len(selected) > 1:
            # ── DUAL MODE: Eager-load both models before dispatching ──────────
            print("[OCR] 🔀 Dual mode — preloading both models...")
            _get_model_components(0)  # Model 1
            _get_model_components(1)  # Model 2
            print("[OCR] ✓ Both models ready. Starting dual-worker processing...")
            max_workers = 2
        else:
            # ── SINGLE MODE ───────────────────────────────────────────────────
            _get_model_components(0)  # Warm up Model 1
            max_workers = 1

        loop   = asyncio.get_running_loop()

        def _process_page(page_num: int) -> Dict[str, Any]:
            """Synchronous per-page OCR — runs in thread-pool worker."""
            # In dual mode, alternate models per page (1-indexed)
            worker_id = (page_num - 1) % max_workers  # 0 or 1
            try:
                if _OLMOCR_AVAILABLE:
                    b64 = _olmocr_render(tmp_path, page_num, target_longest_image_dim=RENDER_LONGEST_DIM)
                else:
                    from local_proc.renderpdf import render_pdf_to_base64png
                    b64 = render_pdf_to_base64png(BytesIO(buf), page_number=page_num, resolution=RENDER_LONGEST_DIM)

                img             = Image.open(BytesIO(base64.b64decode(b64)))
                text, lang_info = _ocr_page(img, "document", worker_id=worker_id)
                preview         = _create_optimized_preview(img)

                # Lift the internal RAG payload out of language_info so it reaches the
                # server (for DB ingestion) but is NOT bloating the frontend message.
                rag = lang_info.pop("_rag", None) if isinstance(lang_info, dict) else None

                return {
                    "type": "page_complete", "page": page_num, "text": text,
                    "language_info": lang_info, "error": None,
                    "total_pages": total_pages, "status": "completed", "preview": preview,
                    "rag": rag,
                }
            except Exception as e:
                print(f"[OCR] ✗ Page {page_num} FAILED (worker {worker_id}): {e}")
                return {
                    "type": "page_complete", "page": page_num, "text": "", "error": str(e),
                    "total_pages": total_pages, "status": "error", "preview": None,
                    "language_info": {"script": "unknown", "language": "unknown", "confidence": 0.0}
                }

        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            # Submit all pages and preserve order via futures list
            futures = [
                loop.run_in_executor(executor, _process_page, pn)
                for pn in selected
            ]
            for future in futures:
                try:
                    result = await future
                    yield result
                except asyncio.CancelledError:
                    raise
                except Exception as e:
                    print(f"[OCR] ✗ Unexpected error: {e}")

        yield {"type": "processing_complete", "status": "finished", "total_pages": total_pages}

    except asyncio.CancelledError:
        yield {"type": "error", "error": "Stream processing cancelled"}
    except Exception as e:
        import traceback; traceback.print_exc()
        yield {"type": "error", "error": str(e)}
    finally:
        if tmp_path and _os.path.exists(tmp_path):
            try:
                _os.unlink(tmp_path)
            except Exception:
                pass


async def _stream_pdf_fallback(buf: bytes) -> AsyncGenerator[Dict[str, Any], None]:
    """Fallback when pdf2image is unavailable — renders via olmocr or local renderer."""
    try:
        stream      = BytesIO(buf)
        total_pages = len(PdfReader(stream).pages)
        loop        = asyncio.get_running_loop()

        for page_num in range(1, total_pages + 1):
            yield {"type": "page_start", "page": page_num, "total_pages": total_pages, "status": "processing"}
            try:
                with memory_cleanup():
                    stream.seek(0)
                    if _OLMOCR_AVAILABLE:
                        b64 = _olmocr_render(stream, page_num, target_longest_image_dim=RENDER_LONGEST_DIM)
                    else:
                        from local_proc.renderpdf import render_pdf_to_base64png
                        b64 = render_pdf_to_base64png(stream, page_number=page_num, resolution=RENDER_LONGEST_DIM)
                    img = Image.open(BytesIO(base64.b64decode(b64)))
                    text, lang_info = await loop.run_in_executor(
                        None, _run_enhanced_ocr, img, _get_format_preserving_prompt("document")
                    )
                    yield {"type": "page_complete", "page": page_num, "text": text,
                           "language_info": lang_info, "error": None,
                           "total_pages": total_pages, "status": "completed",
                           "preview": _create_optimized_preview(img)}
            except Exception as e:
                yield {"type": "page_complete", "page": page_num, "text": "", "error": str(e),
                       "total_pages": total_pages, "status": "error", "preview": None,
                       "language_info": {"script": "unknown", "language": "unknown", "confidence": 0.0}}

        yield {"type": "processing_complete", "status": "finished", "total_pages": total_pages}
    except Exception as e:
        yield {"type": "error", "error": str(e)}


# ============================================================================
# BATCH ENTRY POINTS
# ============================================================================

def run_ocr_bytes(buf: bytes, mode: str = "single") -> Dict[str, Any]:
    if _is_pdf(buf):
        return _run_pdf_enhanced(buf, mode=mode)
    return _process_single_image_enhanced(buf)


def _run_pdf_enhanced(buf: bytes, mode: str = "single") -> Dict[str, Any]:
    """
    Batch-process all PDF pages using olmOCR's native per-page renderer.
    In single mode: sequential, Model 1 only.
    In dual mode: uses ThreadPoolExecutor(max_workers=2), alternating model per page.
    """
    pages    = []
    tmp_path = None
    try:
        reader      = PdfReader(BytesIO(buf))
        total_pages = len(reader.pages)

        print(f"[OCR] 📄 Processing {total_pages} pages (mode={mode}, per-page render)...")

        # Write to temp file so olmOCR renderer gets a real file path
        with tempfile.NamedTemporaryFile(suffix='.pdf', delete=False) as tmp:
            tmp.write(buf)
            tmp_path = tmp.name

        if mode == "dual" and total_pages > 1:
            print("[OCR] 🔀 Dual mode (batch) — preloading both models...")
            _get_model_components(0)
            _get_model_components(1)
            print("[OCR] ✓ Both models ready")
            max_workers = 2
        else:
            _get_model_components(0)
            max_workers = 1

        def _process_page_batch(page_num: int) -> Dict[str, Any]:
            worker_id = (page_num - 1) % max_workers
            print(f"[OCR] 🔄 Processing Page {page_num}/{total_pages} (worker {worker_id})...")
            try:
                if _OLMOCR_AVAILABLE:
                    b64 = _olmocr_render(tmp_path, page_num, target_longest_image_dim=RENDER_LONGEST_DIM)
                else:
                    from local_proc.renderpdf import render_pdf_to_base64png
                    b64 = render_pdf_to_base64png(BytesIO(buf), page_number=page_num, resolution=RENDER_LONGEST_DIM)

                img             = Image.open(BytesIO(base64.b64decode(b64)))
                text, lang_info = _ocr_page(img, "document", worker_id=worker_id)
                preview         = _create_optimized_preview(img)

                # Lift the internal RAG payload (prose + structured band rows) out of
                # language_info so the POST /upload/ handler can ingest it into the RAG
                # DB — same as the streaming path does. Without this the batch path
                # returned no "rag" key, so POST-path ingests saw 0 cells and the
                # integrity gate marked multi-page docs 'failed' (e.g. WCR_D-1-6).
                rag = lang_info.pop("_rag", None) if isinstance(lang_info, dict) else None

                print(f"[OCR] ✓ Page {page_num} done ({len(text)} chars)")
                return {
                    "type": "page_complete", "page": page_num, "text": text,
                    "language_info": lang_info, "error": None,
                    "total_pages": total_pages, "status": "completed", "preview": preview,
                    "rag": rag,
                }
            except Exception as e:
                print(f"[OCR] ✗ Page {page_num} FAILED: {e}")
                return {
                    "type": "page_complete", "page": page_num, "text": "",
                    "language_info": {"script": "unknown", "language": "unknown", "confidence": 0.0},
                    "error": str(e), "total_pages": total_pages, "status": "error", "preview": None
                }

        from concurrent.futures import ThreadPoolExecutor, as_completed
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            future_map = {executor.submit(_process_page_batch, pn): pn for pn in range(1, total_pages + 1)}
            results_map = {}
            for future in as_completed(future_map):
                pn = future_map[future]
                results_map[pn] = future.result()

        pages = [results_map[pn] for pn in sorted(results_map)]

        return {"success": True, "pages": pages, "total_pages": len(pages), "error": None}
    except Exception as e:
        import traceback; traceback.print_exc()
        return {"success": False, "pages": [], "total_pages": 0, "error": str(e)}
    finally:
        if tmp_path and _os.path.exists(tmp_path):
            try:
                _os.unlink(tmp_path)
            except Exception:
                pass


def extract_text_from_pdf(path: str, lang: Optional[str] = None, mode: str = "single") -> Dict[str, Any]:
    try:
        with open(path, "rb") as fh:
            data = fh.read()
        return run_ocr_bytes(data, mode=mode)
    except Exception as e:
        return {"success": False, "pages": [], "total_pages": 0, "error": f"File processing failed: {e}"}


# Legacy aliases
def _get_enhanced_prompt(kind: str = "document") -> str: return _get_format_preserving_prompt(kind)
def _extract_actual_text(raw: str) -> str: return _extract_and_preserve_formatting(raw)
def _preprocess_image(img: Image.Image) -> Image.Image: return _enhance_image_for_ocr(img)
def _run_ocr_on_image(img: Image.Image, prompt: str) -> str: text, _ = _run_enhanced_ocr(img, prompt); return text