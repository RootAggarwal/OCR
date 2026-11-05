"""
figures.py — extract real figures (photos, logs, cross-sections, maps) embedded in a
source PDF at ingest time, so deckworks' Maker can cite an ACTUAL document figure
instead of either fabricating an image or going without one (visual-upgrade plan,
Phase 2A). This is deliberately the highest-value, ONGC-native media source: a well
completion report's own log panels/photos read as authoritative in a way no generated
or stock image can.

Best-effort, like every other optional lane in this package (embeddings.py's
`available()` pattern): pymupdf (imported as `fitz`) is an optional dependency —
`available()` reports whether it's importable, and every extraction failure (a
malformed PDF, a page pymupdf chokes on) is caught and logged, never raised, so a
missing/broken figure pipeline can never break ingestion itself.

Content-addressed storage, same pattern as ingest.py's pdf_store: the cropped PNG
bytes are written to figure_store/{image_ref}.png, where image_ref is derived from
the document's content_hash (already the ingestion idempotency key) plus page/xref —
no schema migration needed for the blob storage itself, only the `figures` metadata
table (schema.sql) that indexes what was found and where.
"""

from __future__ import annotations

import io
import os
import re
from dataclasses import dataclass
from typing import List, Optional, Tuple

# Now lives under OCR_DATA_DIR (see ocr_paths.py) so the repo holds no user data. The DB
# stores figures.image_ref as a bare key, not a path, so relocating this needs no
# migration — only figure_store_path() below resolves it to disk.
from ocr_paths import FIGURE_STORE_DIR  # noqa: F401  (re-export)

# Filters out logos, bullet glyphs, and decorative rules -- the "repeated tiny image on
# every page" noise a raw page.get_images() dump is otherwise full of. A real figure
# (photo, log panel, map, cross-section) clears this easily; a 40x40 letterhead mark does not.
_MIN_DIM_PX = 120
_MIN_AREA_PX = 30_000

# A SCANNED page (this project's historical WCR/FER well reports are commonly full-page
# photographic scans, not born-digital text+figure layouts) embeds its ENTIRE page as one
# raster -- that single "image" clears the two filters above easily but is the whole page,
# not a figure embedded within it. Excluding anything covering most of the page's own area
# is what actually distinguishes "a real figure on the page" from "the page IS the scan".
_MAX_PAGE_COVERAGE = 0.85

# Matches "Fig 4.2 — ..." / "Figure 1.2.1: ..." / "Figure 4.2." near the image, capturing
# a short trailing caption phrase for both the DB record and downstream BM25 indexing.
# The (?:\.\d+)* tail keeps a dotted multi-part number (e.g. "1.2.1") as ONE group-1
# match instead of splitting at the first digit run and leaving ".2.1" to bleed into
# the caption text itself.
_CAPTION_RE = re.compile(r"\b(Fig(?:ure)?\.?\s*\d+(?:\.\d+)*[A-Za-z]?)\s*[:.\-–—]?\s*(.{0,160})",
                         re.IGNORECASE)


def available() -> bool:
    """True when pymupdf is importable -- otherwise figure extraction no-ops and
    ingestion proceeds exactly as it did before this module existed."""
    try:
        import fitz  # noqa: F401
        return True
    except Exception:
        return False


def figure_store_path(image_ref: str) -> str:
    return os.path.join(FIGURE_STORE_DIR, f"{image_ref}.png")


@dataclass
class ExtractedFigure:
    page: int
    bbox: Tuple[float, float, float, float]
    caption: Optional[str]
    context: Optional[str]
    image_ref: str
    png_bytes: bytes
    width_px: int
    height_px: int
    phash: str


def _phash(png_bytes: bytes) -> str:
    """Cheap perceptual hash for within-document dedup (reports repeat their own
    letterhead/logo across every page): downsample to 8x8 grayscale, threshold each
    pixel against the block's own average into a 64-bit fingerprint. Deliberately not
    the `imagehash` package -- PIL alone (already a project dependency) is enough for
    "is this the exact same repeated image", which is all dedup here needs."""
    from PIL import Image
    img = Image.open(io.BytesIO(png_bytes)).convert("L").resize((8, 8))
    pixels = list(img.getdata())
    avg = sum(pixels) / len(pixels)
    bits = "".join("1" if p >= avg else "0" for p in pixels)
    return f"{int(bits, 2):016x}"


def _nearby_caption(page, bbox: Tuple[float, float, float, float]) -> Tuple[Optional[str], Optional[str]]:
    """Best-effort caption + surrounding context: rank the page's text blocks by vertical
    distance from the image (a caption is conventionally the closest block below or
    above a figure), take the nearest as `context` regardless, and additionally return
    it as `caption` only if it actually matches a "Fig N: ..." pattern -- so a slide
    citing this figure always has something to show, but the pipeline never invents a
    figure NUMBER that was not actually printed next to the image."""
    x0, y0, x1, y1 = bbox
    candidates = []
    for b in page.get_text("blocks"):
        bx0, by0, bx1, by1, text = b[0], b[1], b[2], b[3], b[4]
        if not text.strip():
            continue
        if by0 >= y1:
            vdist = by0 - y1        # block below the image
        elif by1 <= y0:
            vdist = y0 - by1        # block above the image
        else:
            vdist = 0.0             # vertically overlapping (e.g. a side-by-side caption)
        candidates.append((vdist, text.strip()))
    if not candidates:
        return None, None
    candidates.sort(key=lambda t: t[0])
    context = candidates[0][1][:400]
    caption = None
    for _, text in candidates[:3]:
        m = _CAPTION_RE.search(text)
        if m:
            caption = f"{m.group(1)} — {m.group(2)}".strip(" -:–—")
            break
    return caption, context


def extract_figures(doc_bytes: bytes, content_hash: str) -> List[ExtractedFigure]:
    """Pull real embedded raster figures out of a PDF. Never raises -- any failure
    (unreadable PDF, a single bad page/image) is caught and logged; ingestion's caller
    treats an empty list exactly like "no figures found", not an error."""
    if not available():
        return []
    import fitz

    try:
        doc = fitz.open(stream=doc_bytes, filetype="pdf")
    except Exception as e:
        print(f"[rag.figures] open failed ({e}); skipping figure extraction")
        return []

    out: List[ExtractedFigure] = []
    seen_phashes: set = set()
    try:
        for page_index in range(len(doc)):
            page = doc[page_index]
            try:
                images = page.get_images(full=True)
            except Exception:
                continue
            for img in images:
                xref = img[0]
                try:
                    rects = page.get_image_rects(xref)
                    if not rects:
                        continue
                    rect = rects[0]
                    if rect.width < _MIN_DIM_PX or rect.height < _MIN_DIM_PX:
                        continue
                    if rect.width * rect.height < _MIN_AREA_PX:
                        continue
                    page_area = page.rect.width * page.rect.height
                    if page_area > 0 and (rect.width * rect.height) / page_area > _MAX_PAGE_COVERAGE:
                        continue  # a full-page scan, not a figure embedded within the page
                    pix = fitz.Pixmap(doc, xref)
                    if pix.n - pix.alpha >= 4:  # CMYK (or worse) -> RGB for a normal PNG
                        pix = fitz.Pixmap(fitz.csRGB, pix)
                    png_bytes = pix.tobytes("png")
                except Exception as e:
                    print(f"[rag.figures] page {page_index + 1} image xref={xref} skipped ({e})")
                    continue

                phash = _phash(png_bytes)
                if phash in seen_phashes:
                    continue
                seen_phashes.add(phash)

                caption, context = _nearby_caption(page, (rect.x0, rect.y0, rect.x1, rect.y1))
                image_ref = f"{content_hash}_{page_index + 1}_{xref}"
                out.append(ExtractedFigure(
                    page=page_index + 1,
                    bbox=(rect.x0, rect.y0, rect.x1, rect.y1),
                    caption=caption,
                    context=context,
                    image_ref=image_ref,
                    png_bytes=png_bytes,
                    width_px=pix.width,
                    height_px=pix.height,
                    phash=phash,
                ))
    except Exception as e:
        print(f"[rag.figures] extraction aborted early ({e})")
    finally:
        doc.close()
    return out


def persist_figure(fig: ExtractedFigure) -> None:
    """Write the cropped PNG to the content-addressed store. Caller (ingest.py) wraps
    this in the same best-effort try/except as the rest of the figure pipeline."""
    os.makedirs(FIGURE_STORE_DIR, exist_ok=True)
    with open(figure_store_path(fig.image_ref), "wb") as f:
        f.write(fig.png_bytes)
