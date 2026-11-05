# table_extractor.py — Geometry-grounded table extraction.
#
# Fixes the "column shift" failure of the whole-page VLM markdown path: on dense,
# ruled, handwritten forms the VLM drops the occasional blank cell and every later
# value slides one column left. Here we instead:
#   1. Detect the table region(s) with TATR (Table Transformer detection).
#   2. Learn the real column layout — including repeated side-by-side blocks.
#   3. Read each block with the VLM under a FIXED, known column count.
#   4. Repair every row to that width (never silently drop a cell).
#   5. Normalize repeated blocks by stacking them into one clean table.
#
# Emits HTML <table> markup, which the rest of the server (markdown->html pass,
# TableParser, Excel export) and the frontend already consume unchanged.

import json
import re
import difflib
import logging
from typing import Any, Dict, List, Optional, Tuple

from PIL import Image
from bs4 import BeautifulSoup

logger = logging.getLogger("table_extractor")

# TATR detection threshold — low, to catch borderless / handwritten tables
# (mirrors pipeline.py run_step_2). Lowered 0.5 -> 0.4 after faint 1988 WCR core-log
# scans (WCR_D-1-6) had whole tables scored below 0.5 and thus MISSED entirely — the
# page then fell through to the unconstrained whole-page VLM read, which garbled the
# headers. Measured across the threshold sweep: 0.4 recovers those missed tables
# (pages that detected 0 tables now detect them) while UP67 — the verified-good
# reference form — still detects exactly one table per page (no change, no false
# positives); 0.2 was too low (UP67 started splitting into 2). The prose-vs-table
# gate (_is_plausible_table) still rejects any low-confidence PROSE region a looser
# threshold might surface, so this only recovers real tables.
_DETECT_THRESHOLD = 0.5
_STRUCT_THRESHOLD = 0.6
_BOX_PAD = 20          # px padding around detected table boxes (left/right/top)
_BOX_PAD_BOTTOM = 80   # generous bottom pad; trimmed back to the last real row
_ROW_TRIM_MARGIN = 20  # px kept below the last detected row

_TABLE_CLASSES = "dataframe table table-hover table-striped table-bordered ocr-dataframe"

# --- False-positive guard ---------------------------------------------------
# TATR happily reports a numbered prose list as a table: on a page of plain
# "1. … 2. … 3. …" highlights it returned a 2x12 "table" at score 0.83 and a 3x5 one
# at 0.73 — HIGHER confidence than a real handwritten ONGC form scores (0.58). So the
# detection score cannot separate them, and raising the threshold would drop the real
# tables while keeping the fake ones.
# The signal that does separate them is cell CONTENT: a table cell is a short token
# ("636", "-112", "Receivers only"); a prose row is a sentence.
_MAX_WORDS_PER_CELL      = 8     # above this, a cell reads as prose, not data
_MAX_LONG_CELL_FRACTION  = 0.25  # if more than this share of cells are prose, it isn't a table
_MIN_TABLE_COLUMNS       = 2
_MIN_TABLE_DATA_ROWS     = 2

# --- Structure-quality gate (dense-table column-merge guard) ----------------
# When the geometry path forces a wrong block grid onto a table it doesn't fit
# (e.g. a single wide heterogeneous WCR production sheet squeezed into repeating
# k-column blocks), distinct columns collapse and single cells end up holding
# several numbers ("507 246 326 729 815"). That "multiple numbers in one cell"
# rate cleanly separates a good grid from a merged one — measured on real pages:
# UP67 ~0.7%, FER ~0%, garbled WCR ~13%. Above the threshold we re-read the whole
# table with the VLM (no imposed band structure) and keep whichever is cleaner.
_MERGED_CELL_FRACTION_MAX = 0.06  # >6% multi-number cells ⇒ grid looks column-merged
_MERGED_MIN_CELLS         = 20    # don't judge tiny tables on this signal


# ============================================================================
# PUBLIC ENTRY POINT
# ============================================================================

def extract_page_html(img: Image.Image) -> Dict[str, Any]:
    """
    Detect and extract every table on a page image.

    Returns {"tables": [html_str, ...], "boxes": [[x0,y0,x1,y1], ...],
             "structured": [ {headers,blocks,k,spans,box} | None, ... ]} where
    boxes are in full-page pixel coordinates (top table first, reading order) and
    `structured[i]` carries the clean band rows for the RAG unpivot (None for the
    VLM-fallback path, which has no reliable grid). Empty lists when no table found.
    """
    from ocr_tatr import TableTransformerParser

    try:
        parser = TableTransformerParser.get_instance()
    except Exception as e:
        logger.warning("[table_extractor] TATR unavailable (%s) — skipping structured extraction", e)
        return {"tables": [], "boxes": [], "structured": []}

    try:
        boxes = _detect_table_boxes(parser, img)
    except Exception as e:
        logger.warning("[table_extractor] table detection failed: %s", e)
        return {"tables": [], "boxes": [], "structured": []}

    tables_html: List[str] = []
    out_boxes: List[List[float]] = []
    structured_list: List[Optional[Dict[str, Any]]] = []

    for box in boxes:
        crop = img.crop(tuple(box))
        if crop.width < 8 or crop.height < 8:
            continue
        try:
            html, structured = _extract_table_crop(crop, parser, img, box)
        except Exception as e:
            logger.warning("[table_extractor] crop extraction failed (%s) — falling back to plain VLM", e)
            html, structured = _fallback_vlm_html(crop), None

        # Reject regions that are prose, not tables. Done here — the one choke point
        # every extraction path passes through — so fallbacks are covered too. A
        # rejected box is also left OUT of `boxes`, so the prose pass does not mask
        # it and the text is read normally instead of being lost inside a fake table.
        if html and not _is_plausible_table(html):
            continue

        if html:
            tables_html.append(html)
            out_boxes.append(box)
            structured_list.append({**structured, "box": box} if structured else None)

    return {"tables": tables_html, "boxes": out_boxes, "structured": structured_list}


def _is_plausible_table(html: str) -> bool:
    """
    Is this actually a table, or did TATR mistake prose for one?

    Detection score is not usable as a filter (a numbered list scored 0.83 while a real
    handwritten form scored 0.58). Cell content is: table cells are short tokens, prose
    rows are sentences. Also require a real shape — at least 2 columns and 2 data rows,
    which alone rejects the spurious "Column 1 | Column 2 | Column 3" single-row table.
    """
    rows = _html_to_rows(html)
    if not rows:
        return False

    width = max(len(r) for r in rows)
    if width < _MIN_TABLE_COLUMNS:
        logger.info("[table_extractor] rejected region: only %d column(s)", width)
        return False

    body = rows[1:]  # drop the header row (a nested sub-header row is harmless here)
    if len(body) < _MIN_TABLE_DATA_ROWS:
        logger.info("[table_extractor] rejected region: only %d data row(s)", len(body))
        return False

    cells = [c.strip() for r in body for c in r if c and c.strip()]
    if not cells:
        return False

    long_cells = sum(1 for c in cells if len(c.split()) > _MAX_WORDS_PER_CELL)
    fraction = long_cells / len(cells)
    if fraction > _MAX_LONG_CELL_FRACTION:
        logger.info("[table_extractor] rejected region as PROSE: %.0f%% of cells are "
                    "sentences (>%d words)", fraction * 100, _MAX_WORDS_PER_CELL)
        return False

    return True


# ============================================================================
# DETECTION
# ============================================================================

def _merge_stacked_boxes(boxes: List[List[float]]) -> List[List[float]]:
    """
    Merge detections that are really ONE table the detector split into pieces.

    On some skewed scans TATR reports a table's HEADER band as a separate region from
    its BODY (or emits an overlapping duplicate), which then render as two tables — a
    header-only "Table 1" and a data-only "Table 2". Two boxes are merged when they
    share most of their horizontal extent (same columns) AND are vertically flush or
    overlapping. Genuinely separate tables sit apart with a clear vertical gap, so they
    are left alone. Operates on RAW detector boxes (before padding).
    """
    if len(boxes) < 2:
        return [list(b) for b in boxes]
    ordered = sorted((list(b) for b in boxes), key=lambda b: b[1])  # top to bottom
    out = [ordered[0]]
    for b in ordered[1:]:
        prev = out[-1]
        overlap_x = min(prev[2], b[2]) - max(prev[0], b[0])
        min_w = min(prev[2] - prev[0], b[2] - b[0]) or 1.0
        x_frac = overlap_x / min_w
        gap = b[1] - prev[3]                                   # <0 = vertically overlapping
        min_h = min(prev[3] - prev[1], b[3] - b[1]) or 1.0
        MAX_ALLOWED_SPLIT_GAP = 25
        if x_frac > 0.6 and gap < 0.5 * min_h and gap < MAX_ALLOWED_SPLIT_GAP:
            out[-1] = [min(prev[0], b[0]), min(prev[1], b[1]),
                       max(prev[2], b[2]), max(prev[3], b[3])]
            logger.info("[table_extractor] merged a split table region "
                        "(x_overlap=%.2f, gap=%.0fpx) — header+body of one table",
                        x_frac, gap)
        else:
            out.append(b)
    return out


def _detect_table_boxes(parser, img: Image.Image) -> List[List[float]]:
    """
    Detect table regions and return padded [x0,y0,x1,y1] boxes, top-to-bottom.

    The detector's box routinely cuts the LAST data row in half, which silently
    drops that row from the extraction. We therefore pad the bottom generously;
    `_extract_table_crop` then trims back to the last real row using row geometry,
    so the extra padding costs nothing. Split header/body regions are merged first
    (see _merge_stacked_boxes) so one table never renders as two.
    """
    dets = parser.detect_tables(img, threshold=_DETECT_THRESHOLD)
    raw = _merge_stacked_boxes([list(d["box"]) for d in dets])
    boxes = []
    for x0, y0, x1, y1 in raw:
        bottom_pad = max(_BOX_PAD_BOTTOM, int(0.15 * (y1 - y0)))
        top_pad = 10
        boxes.append([
            max(0, x0 - _BOX_PAD),
            max(0, y0 - top_pad),
            min(img.width, x1 + _BOX_PAD),
            min(img.height, y1 + bottom_pad),
        ])
    boxes.sort(key=lambda b: b[1])  # reading order: top to bottom
    return boxes


# ============================================================================
# PER-TABLE EXTRACTION
# ============================================================================

def _merged_cell_fraction(blocks: List[List[List[str]]]) -> Tuple[float, int]:
    """Fraction of non-empty data cells that contain MORE THAN ONE numeric token
    (the column-merge signature — see _MERGED_CELL_FRACTION_MAX), plus the count
    of non-empty cells considered. A clean grid keeps one value per cell."""
    total = 0
    merged = 0
    for block in blocks or []:
        for row in block:
            for cell in row:
                s = str(cell).strip()
                if not s:
                    continue
                total += 1
                if len(re.findall(r"-?\d[\d.,]*\d|\d", s)) > 1:
                    merged += 1
    return (merged / total if total else 0.0), total


def _structured_quality_ok(structured: Dict[str, Any]) -> bool:
    """Does the geometry path's structured grid look trustworthy, or did it force
    a wrong grid and merge columns? Only the column-merge signal is used (it is the
    one that cleanly separated good from garbled on real pages); a small table is
    never judged on it (too few cells to be meaningful)."""
    frac, total = _merged_cell_fraction(structured.get("blocks") or [])
    if total >= _MERGED_MIN_CELLS and frac > _MERGED_CELL_FRACTION_MAX:
        logger.info("[table_extractor] structure-quality gate FAILED: %.0f%% of %d cells hold "
                    "multiple numbers (> %.0f%%) — grid looks column-merged",
                    frac * 100, total, _MERGED_CELL_FRACTION_MAX * 100)
        return False
    return True


def _enhance_for_ocr(crop: Image.Image) -> Image.Image:
    """Make a faint / low-contrast scan legible before a VLM read. The dense 1959
    WCR production tables are washed-out typewriter carbon on aged paper — neither
    the geometry grid nor a plain VLM read resolves their digits, so the fallback
    re-read gets an autocontrast-stretched, upscaled, lightly-sharpened copy. Pure
    image ops (no model); only ever applied on the already-gated hard-table path,
    never to the clean UP67/FER reads."""
    from PIL import ImageEnhance, ImageFilter, ImageOps
    img = ImageOps.autocontrast(crop.convert("L"), cutoff=1)   # stretch faint ink to full range
    # Upscale small/dense crops so thin strokes span enough pixels for the VLM.
    longest = max(img.width, img.height)
    if longest < 2200:
        scale = min(2.0, 2200 / longest)
        img = img.resize((int(img.width * scale), int(img.height * scale)), Image.LANCZOS)
    img = ImageEnhance.Contrast(img).enhance(1.4)
    img = img.filter(ImageFilter.UnsharpMask(radius=2, percent=120, threshold=2))
    return img.convert("RGB")


def _geometry_single_block(crop: Image.Image, col_boxes: List[List[float]],
                           row_boxes: List[List[float]], span_boxes: List[List[float]]
                           ) -> Optional[Tuple[str, Dict[str, Any]]]:
    """Re-read the crop as ONE wide block of all detected columns (groups=1),
    reusing the same band machinery as the main path. This is the fix for a wide
    HETEROGENEOUS table (e.g. a WCR production sheet) whose VLM group-count came
    back >1 and folded its distinct columns via k=round(C/groups): forcing a
    single block keeps every real column. Only invoked from the quality-gate
    fallback (never on a clean grid), so the well-tuned repeating-block path for
    UP67 is untouched."""
    k = len(col_boxes)
    if k < _MIN_TABLE_COLUMNS or not row_boxes:
        return None
    bands, _band_x, grid_cols = _split_bands(crop, 1, k, col_boxes)
    hdr_leaf, hdr_spans, _hbottom = _read_header_robust(
        crop, grid_cols[:k] if grid_cols else [], row_boxes, span_boxes, k)
    hdr_leaf = [_normalize_header(h) for h in hdr_leaf]
    hdr_spans = [(st, ln, _normalize_header(lbl)) for (st, ln, lbl) in hdr_spans]
    headers = _fit_headers(hdr_leaf, k, keep_blanks=True) if hdr_leaf else _fit_headers([], k)
    expected_rows = max(0, len(row_boxes) - 1)
    blocks, _ignored = _read_bands_batched(bands, _prompt_headers(headers, hdr_spans, k), k, expected_rows)
    if not any(blocks):
        return None
    headers, blocks, spans, k = _drop_phantom_trailing_columns(headers, blocks, hdr_spans, k)
    html = _blocks_to_wide_html(headers, blocks, k, spans)
    return html, {"headers": headers, "blocks": blocks, "k": k, "spans": spans}


def _vlm_structured_table(crop: Image.Image) -> Optional[Tuple[str, Dict[str, Any]]]:
    """Whole-table VLM read that returns STRUCTURED rows (not just display HTML),
    so a table the geometry grid mangled still feeds the RAG numeric lane. Imposes
    NO band/block structure — the model reads the table as one grid, which on an
    irregular wide table is more faithful than a wrong repeating-block split.
    The crop is contrast/upscale-enhanced first (_enhance_for_ocr), since the
    tables that reach this path are exactly the faint ones a raw read fails on.
    Returns (html, {headers, blocks, k, spans}) or None."""
    import ocr_olm
    try:
        enhanced = _enhance_for_ocr(crop)
    except Exception as e:
        logger.warning("[table_extractor] enhance-for-OCR failed (%s); using raw crop", e)
        enhanced = crop
    try:
        raw, _ = ocr_olm._run_enhanced_ocr(enhanced, ocr_olm._get_fallback_html_prompt())
    except Exception as e:
        logger.warning("[table_extractor] structured VLM re-read failed: %s", e)
        return None

    rows = _html_to_rows(raw) or _markdown_to_rows(raw)
    if not rows or len(rows) < 2:
        return None
    k = max(len(r) for r in rows)
    if k < _MIN_TABLE_COLUMNS:
        return None
    if _looks_like_header(rows[0], rows[0]):
        headers = _fit_headers(rows[0], k, keep_blanks=True)
        data = rows[1:]
    else:
        headers = [f"Column {i + 1}" for i in range(k)]
        data = rows
    data = _normalize_row_widths(data, k)
    if len(data) < _MIN_TABLE_DATA_ROWS:
        return None
    html = _rows_to_html(headers, data)
    structured = {"headers": headers, "blocks": [data], "k": k, "spans": []}
    return html, structured


def _extract_table_crop(crop: Image.Image, parser, full_img: Optional[Image.Image] = None, box: Optional[List[float]] = None) -> Tuple[Optional[str], Optional[Dict[str, Any]]]:
    """
    Extract a single detected table region.

    Returns ``(html, structured)`` where ``structured`` is
    ``{"headers", "blocks", "k", "spans"}`` for the geometry path (the clean band
    rows the RAG unpivot needs) or ``None`` for the VLM-fallback path (which has no
    reliable column grid). HTML is unchanged from before.
    """
    # 1. Geometry — ONE structure pass gives columns AND rows.
    #    TATR column geometry is far more reliable than the VLM's header count for
    #    determining the true column grid (it isn't fooled by spanning super-headers).
    col_boxes, row_boxes, hdr_boxes, span_boxes = _structure_geometry(parser, crop)

    # 1a. Discard hairline "columns" at the margins and trailing "rows" that the bottom
    #     padding dragged in from the text below the table. Both must happen BEFORE k is
    #     derived, or the artifact becomes a real column of the output.
    col_boxes = _drop_edge_slivers(col_boxes)
    row_boxes = _drop_detached_trailing_rows(row_boxes)

    # 1b. Trim the generous bottom padding back to the last real row. The detector's
    #     box tends to cut the final data row in half, silently dropping it; padding
    #     then trimming on row geometry recovers it. (x is untouched, so the column
    #     boxes stay valid.)
    if row_boxes:
        bottom = int(min(crop.height, max(r[3] for r in row_boxes) + _ROW_TRIM_MARGIN))
        if bottom > 8 and bottom < crop.height:
            crop = crop.crop((0, 0, crop.width, bottom))

    num_struct_cols = len(col_boxes)

    # 2. Ask the VLM how many identical blocks are side-by-side + header names + spans.
    #    NOT cached across pages: measured on 8 pages of the SAME form, TATR returns
    #    12/13/14 columns with centres shifted by several percent (each scan is skewed
    #    differently), so a geometry-keyed cache never hits — and a looser key would
    #    risk applying one form's headers to a different one.
    if full_img and box:
        # Give VLM a crop with an extra 40px of top padding so it can read chopped-off headers,
        # without affecting the strict masking boundaries used for the rest of the document.
        vlm_top = max(0, box[1] - 40)
        vlm_crop = full_img.crop((box[0], vlm_top, box[2], box[3]))
        groups, headers, spanning = _detect_groups_and_headers(vlm_crop)
    else:
        groups, headers, spanning = _detect_groups_and_headers(crop)
    groups = max(1, min(groups, 8))

    # Expected data rows per block (row lines span the full width, so the count
    # applies to every block). Minus one for the header band.
    expected_rows = max(0, len(row_boxes) - 1) if row_boxes else 0

    # 3. Derive columns-per-block (k) from geometry: k = round(C / G). Rounding is
    #    robust to a spurious extra/missing column (e.g. a margin artifact makes
    #    C=13 for a 3x4 form → round(13/3)=4), unlike requiring exact divisibility.
    #    The VLM header count is only a fallback (it miscounts on nested headers).
    if num_struct_cols:
        k = max(1, round(num_struct_cols / groups))
    elif headers:
        k = len(headers)
    else:
        # No geometry and no schema — plain unconstrained read.
        return _fallback_vlm_html(crop), None

    logger.info("[table_extractor] block layout: groups=%d, cols/block=%d, tatr_cols=%d",
                groups, k, num_struct_cols)

    # 4. Split the crop into `groups` vertical bands at the true column boundaries.
    bands, band_x, grid_cols = _split_bands(crop, groups, k, col_boxes)

    # 5. Headers, read from geometry: column ORDER comes from the column boxes and the
    #    two-level structure from TATR's spanning cells. Spans are carried as column
    #    INDICES, never as names — names get respelled, and an unnamed sub-column has
    #    no name to match on at all.
    hdr_leaf, hdr_spans, header_bottom = _read_header_robust(
        crop, grid_cols[:k] if grid_cols else [], row_boxes, span_boxes, k
    )
    # Snap near-miss header spellings ("seca"/"secG"/"m._secs") to their canonical form.
    hdr_leaf = [_normalize_header(h) for h in hdr_leaf]
    hdr_spans = [(st, ln, _normalize_header(lbl)) for (st, ln, lbl) in hdr_spans]
    if hdr_leaf or hdr_spans:
        # Positional: blanks are meaningful (unnamed sub-columns under a title).
        headers = _fit_headers(hdr_leaf, k, keep_blanks=True)
        spans = hdr_spans
        # Rescue spans if TATR found leaf names but missed the spanning lines completely
        if not spans and spanning:
            rescued_spans = []
            for sp in spanning:
                label = sp.get("label", "")
                over = [_normalize_header(str(o).strip()) for o in sp.get("over", [])]
                indices = [i for i, h in enumerate(headers) if _normalize_header(str(h).strip()) in over]
                if indices and len(indices) >= 2 and indices == list(range(indices[0], indices[0] + len(indices))):
                    rescued_spans.append((indices[0], len(indices), label))
            spans = rescued_spans
    else:
        headers = _fit_headers(headers, k)  # VLM guess; no geometry to anchor it
        # Convert VLM spanning to geometric spans
        vlm_spans = []
        for sp in spanning:
            label = sp.get("label", "")
            over = [_normalize_header(str(o).strip()) for o in sp.get("over", [])]
            indices = [i for i, h in enumerate(headers) if _normalize_header(str(h).strip()) in over]
            if indices and len(indices) >= 2 and indices == list(range(indices[0], indices[0] + len(indices))):
                vlm_spans.append((indices[0], len(indices), label))
        spans = vlm_spans

    # 6. Read each band with the VLM constrained to exactly k columns, then repair.
    #    Keep each block's rows separate so the paper's side-by-side layout is
    #    preserved (block g occupies columns [g*k : (g+1)*k]).
    #    The prompt gets NAMED columns even where the printed header is blank — handed
    #    "Sl. No. | Basin | | | | | Success" the model merged the nameless columns and
    #    emitted "01 GGN-11 Released as GNZX" as a single cell.
    blocks, ignored_grid = _read_bands_batched(
        bands, _prompt_headers(headers, spans, k), k, expected_rows)

    # 6b. Escalate any band where the model ignored the column count. Padding cannot
    #     repair fused cells ("01 GGN-11 Released as GNZX"), but the grid can: read
    #     those cells one at a time, where merging is physically impossible.
    data_rows = [r for r in row_boxes if r[1] >= header_bottom - _ROW_TRIM_MARGIN]

    # Cell-by-cell reading is only safe on a PRINTED, RULED table, where the rules give
    # exact cell boundaries. On the handwritten ONGC scan it was catastrophic — each tiny
    # crop of handwriting came back as junk ("G18", "- 1/2", rows of dots) — and there
    # the whole-band read is far better. Ruled lines are the discriminator: the scan has
    # none detectable, the printed reports do.
    can_escalate = len(_horizontal_rules(crop)) >= 2

    if grid_cols and data_rows and can_escalate:
        for g, bad in enumerate(ignored_grid):
            if not bad:
                continue
            block_cols = grid_cols[g * k:(g + 1) * k]
            if len(block_cols) != k:
                continue
            cells = _read_cells_geometric(crop, block_cols, data_rows, k, header_bottom)
            if cells:
                blocks[g] = cells
    elif any(ignored_grid) and not can_escalate:
        logger.info("[table_extractor] band read ignored the grid, but the table has no "
                    "ruled lines — not escalating (cell reads are unreliable there)")

    if not any(blocks):
        logger.warning("[table_extractor] band read produced NO rows — falling back to "
                       "unconstrained VLM (no structured rows for the RAG table lane)")
        return _fallback_vlm_html(crop), None

    # 6c. Remove trailing columns TATR invented (no header, no data anywhere).
    headers, blocks, spans, k = _drop_phantom_trailing_columns(headers, blocks, spans, k)

    structured = {"headers": headers, "blocks": blocks, "k": k, "spans": spans}

    # 6d. Structure-quality gate. If the geometric grid looks column-merged (a wrong
    #     block grid forced onto a table that doesn't fit it — the dense-WCR failure),
    #     try two alternative reads and keep whichever is MEASURABLY cleaner than the
    #     original (by merged-cell fraction):
    #       (a) a groups=1 single-wide-block geometry re-read — the direct fix when a
    #           bad VLM group-count folded distinct columns (k=round(C/groups));
    #       (b) a whole-table VLM read (no band structure) — for when the grid itself
    #           is unreliable.
    #     A clean UP67/FER grid passes the gate untouched, so this never enters here
    #     and the well-tuned band path cannot regress.
    if not _structured_quality_ok(structured):
        geo_frac, _ = _merged_cell_fraction(blocks)
        best_html, best_structured, best_frac = None, None, geo_frac
        for label, cand in (("groups=1 geometry",
                             _geometry_single_block(crop, col_boxes, row_boxes, span_boxes)),
                            ("whole-table VLM", _vlm_structured_table(crop))):
            if cand is None:
                continue
            cand_html, cand_structured = cand
            cand_frac, cand_total = _merged_cell_fraction(cand_structured["blocks"])
            if cand_total > 0 and cand_frac < best_frac:
                best_html, best_structured, best_frac = cand_html, cand_structured, cand_frac
                logger.info("[table_extractor] gate: '%s' is cleaner (merged %.0f%% < %.0f%%)",
                            label, cand_frac * 100, geo_frac * 100)
        if best_structured is not None:
            logger.info("[table_extractor] replaced garbled geometry grid (merged %.0f%% → %.0f%%)",
                        geo_frac * 100, best_frac * 100)
            return best_html, best_structured

    # 7. Assemble a wide table mirroring the paper: the k leaf headers (under any
    #    spanning title) repeated once per block, blocks side by side.
    html = _blocks_to_wide_html(headers, blocks, k, spans)
    return html, structured


def _drop_phantom_trailing_columns(headers: List[str], blocks: List[List[List[str]]],
                                   spans: List[Tuple[int, int, str]], k: int
                                   ) -> Tuple[List[str], List[List[List[str]]], List[Tuple[int, int, str]], int]:
    """
    Drop trailing columns that exist only in TATR's imagination: no header text AND no
    data in any row (the Wikipedia tables produced a "Column 10" and a "Column 15..18").

    The header test is what makes this safe. ONGC's "Remarks" columns are also empty in
    every row, but they are REAL and they have a real header — so they survive. A phantom
    has nothing to read, hence a blank or placeholder name.
    Only trailing columns are considered, so span indices (which start earlier) stay valid.
    """
    def phantom(i: int) -> bool:
        name = str(headers[i]).strip() if i < len(headers) else ""
        if name and not re.fullmatch(r"Column\s*\d+", name, re.IGNORECASE):
            return False                      # has a real header → keep
        if any(st <= i < st + ln for st, ln, _ in (spans or [])):
            return False                      # sits under a title → keep
        for b in blocks:
            for row in b:
                if i < len(row) and str(row[i]).strip():
                    return False              # has data → keep
        return True

    while k > 2 and phantom(k - 1):
        logger.info("[table_extractor] dropping phantom trailing column %d (%r: no header, no data)",
                    k - 1, headers[k - 1] if k - 1 < len(headers) else "")
        k -= 1
        headers = headers[:k]
        blocks = [[row[:k] for row in b] for b in blocks]
        spans = [s for s in (spans or []) if s[0] + s[1] <= k]

    return headers, blocks, spans, k


def _prompt_headers(headers: List[str], spans: List[Tuple[int, int, str]], k: int) -> List[str]:
    """
    Names to describe the columns to the VLM when reading a band.

    The DISPLAY header may legitimately be blank — an unnamed sub-column under a title,
    as printed. But a blank in the prompt ("Sl. No. | Basin | | | | | Success") tells the
    model nothing, and it responds by merging those columns into one cell. So for the
    prompt only, give every column a name: sub-columns are named after their title.
    """
    span_of = {}
    for start, length, label in (spans or []):
        for j in range(length):
            span_of[start + j] = (label, j + 1, length)

    named = []
    for i in range(k):
        name = str(headers[i]).strip() if i < len(headers) else ""
        if not name:
            if i in span_of:
                label, j, length = span_of[i]
                name = f"{label} [part {j} of {length}]"
            else:
                name = f"Column {i + 1}"
        named.append(name)
    return named


def _fit_headers(headers: List[str], k: int, keep_blanks: bool = False) -> List[str]:
    """
    Coerce header names to exactly k entries.

    `keep_blanks` matters: names read from geometry are positional, and a blank is
    meaningful — it is an unnamed sub-column sitting under a spanning title (e.g. the
    two nameless columns under "Drilled Location"). Dropping those blanks would shift
    every later header left. Only names guessed by the VLM (no geometry) get compacted.
    """
    headers = list(headers or [])
    if not keep_blanks:
        headers = [h for h in headers if str(h).strip()]
    if len(headers) >= k:
        return headers[:k]
    if keep_blanks:
        return headers + [""] * (k - len(headers))
    return headers + [f"Column {i + 1}" for i in range(len(headers), k)]


_SLIVER_RATIO = 0.30  # an edge column this much narrower than the median is a margin artifact


def _drop_edge_slivers(cols: List[List[float]]) -> List[List[float]]:
    """
    Drop hairline columns at the left/right edge — TATR reports the page margin or the
    table's own border as a column.

    This matters most for SINGLE-block tables, where k = C: the sliver becomes a real
    column, and the model, asked what its header says, answers with the chat-template
    artifact "system". (`_trim_to_grid` cannot help there — its target equals C.)

    Only EDGE columns are considered, and only when far narrower than the median, so a
    legitimately narrow column is safe. Measured: slivers sit at 0.18-0.25x the median
    while the narrowest genuine column across all test documents is 0.56x. An empty but
    full-width column (ONGC's blank "Remarks") is wide, so it is never touched.
    """
    cols = sorted((list(c) for c in cols), key=lambda b: (b[0] + b[2]) / 2)
    while len(cols) > 2:
        widths = sorted(c[2] - c[0] for c in cols)
        med = widths[len(widths) // 2] or 1.0
        if (cols[0][2] - cols[0][0]) < _SLIVER_RATIO * med:
            logger.info("[table_extractor] dropping left edge sliver column (%.0fpx vs median %.0fpx)",
                        cols[0][2] - cols[0][0], med)
            cols = cols[1:]
        elif (cols[-1][2] - cols[-1][0]) < _SLIVER_RATIO * med:
            logger.info("[table_extractor] dropping right edge sliver column (%.0fpx vs median %.0fpx)",
                        cols[-1][2] - cols[-1][0], med)
            cols = cols[:-1]
        else:
            break
    return cols


def _drop_detached_trailing_rows(rows: List[List[float]]) -> List[List[float]]:
    """
    Drop trailing "rows" that are not actually part of the table.

    The bottom of the detection box is padded generously (it routinely cuts the last
    data row in half), and on a printed page that padding reaches into the paragraph
    BELOW the table — which TATR then reports as another row. Read cell-by-cell, that
    paragraph came back as a junk row of sentence fragments.

    A real table's rows are contiguous: each row starts where the previous one ended.
    Text outside the table is separated by whitespace. So a trailing row preceded by a
    gap far larger than any gap inside the table is not part of it.
    (Measured: GEOPIC gaps 3, -3, 16, 18, then 39 for the paragraph; the scanned ONGC
    form's genuine last row has a gap of -2.)
    """
    while len(rows) > 2:
        gaps = [rows[i + 1][1] - rows[i][3] for i in range(len(rows) - 1)]
        last, others = gaps[-1], gaps[:-1]
        ref = max(others) if others else 0
        if last > max(2 * ref, 20):
            logger.info("[table_extractor] dropping trailing row detached by %.0fpx "
                        "(largest gap inside the table: %.0fpx) — text below the table",
                        last, ref)
            rows = rows[:-1]
        else:
            break
    return rows


def _structure_geometry(parser, crop: Image.Image
                        ) -> Tuple[List[List[float]], List[List[float]], List[List[float]], List[List[float]]]:
    """
    One TATR structure pass → (column boxes L→R, row boxes T→B, column-header boxes,
    spanning-cell boxes). Returns four empty lists if structure recognition fails.

    The spanning-cell boxes are what make a two-level header tractable: they say exactly
    which columns a title sits above, and where it ends vertically.
    """
    try:
        dets = parser.recognize_structure(crop, threshold=_STRUCT_THRESHOLD)
        # Fallback for vintage/faint tables: if it misses rows or columns, lower the threshold
        if not dets or sum(1 for d in dets if d.get("label") in ("table row", "table column")) < 2:
            logger.info("[table_extractor] retrying structure with lower threshold (0.2)")
            dets = parser.recognize_structure(crop, threshold=0.2)
    except Exception as e:
        logger.warning("[table_extractor] structure recognition failed: %s", e)
        return [], [], [], []
    cols = sorted([d["box"] for d in dets if d.get("label") == "table column"],
                  key=lambda b: b[0])
    rows = sorted([d["box"] for d in dets if d.get("label") == "table row"],
                  key=lambda b: b[1])
    hdrs = [d["box"] for d in dets if d.get("label") == "table column header"]
    spans = [d["box"] for d in dets if d.get("label") == "table spanning cell"]
    return cols, rows, hdrs, spans


_TEMPLATE_ARTIFACTS = {"system", "user", "assistant", "none", "empty", "blank", "n/a"}


def _clean_name(raw: str) -> str:
    name = re.sub(r"\s+", " ", (raw or "").strip().strip('"').strip("'"))
    name = name.split("\n")[0][:70]
    if name.strip().lower() in _TEMPLATE_ARTIFACTS:
        return ""
    return name


# Generous cap for a DATA cell — only guards against a runaway model hallucination, never
# clips real content. A column HEADER is a short identifier (the 70-char cap in _clean_name
# is right for it), but a data cell can hold a full free-text sentence — e.g. an "Action
# Taken" remark. Cleaning a value with _clean_name silently truncated such cells at 70 chars
# (doc49 p23: "…reloading driver S/W and" dropped "replaced print head."), which left the RAG
# lane with no way to answer questions about the missing text. Use _clean_value for values.
_VALUE_MAX_CHARS = 2000


def _clean_value(raw: str) -> str:
    val = re.sub(r"\s+", " ", (raw or "").strip().strip('"').strip("'"))
    if val.strip().lower() in _TEMPLATE_ARTIFACTS:
        return ""
    return val[:_VALUE_MAX_CHARS]


# Canonical header spellings for the ONGC forms. The VLM reliably mis-reads a few header
# cells on faint handwritten/printed scans — "secs"→"seca"/"secG", stray underscores
# ("m._secs"). A read header that is a NEAR-MISS of a known one snaps to the canonical
# spelling; anything not clearly a match is left untouched, so other forms are unaffected.
# Extend this list as more form headers are confirmed. Pure string work — no model calls.
KNOWN_HEADERS = [
    # Static-correction form
    "Surface Pkt. No.", "Corrections", "Shot m. secs", "Receiver m. secs", "Remarks",
    # Line-data form
    "Sl. No.", "Shot Point Pkt. No.", "File No.", "Reel No.", "Bearing", "Charge",
    "Channel Nos.", "Reversed", "Channel No.", "Distance", "Depth (mts)",
    "Up-hole Time (m secs)", "Elevation of Shot point (mts)",
]
_HEADER_MATCH_THRESHOLD = 0.72  # min difflib ratio to snap a read header to a known one

# Explicit aliases for header reads that fuzzy matching WON'T catch because they are a
# shorter/variant form of the canonical (too far in edit distance). Keyed by the cleaned,
# lowercased read. This is what unifies e.g. "Surface" (read on some pages) with the full
# "Surface Pkt. No." so one logical column has ONE name everywhere — critical for reliable
# text-to-SQL later.
_HEADER_ALIASES = {
    "surface": "Surface Pkt. No.",
}


def _normalize_header(name: str) -> str:
    """
    Canonicalize a read header. First an explicit alias table (short-form/variant reads),
    then fuzzy-snap to a known header when it is a clear near-miss (fixes VLM OCR errors
    like "seca"/"secG"/"m._secs"). Leaves genuinely different text alone, so headers from
    other forms pass through unchanged. Cannot fix unrelated hallucinations (e.g. an empty
    cell read as "Romeo and Juliet") — those share no similarity with any known header.
    """
    raw = (name or "").strip()
    if not raw:
        return raw
    cleaned = re.sub(r"\s+", " ", raw.replace("_", " ")).strip()
    low = cleaned.lower()
    if low in _HEADER_ALIASES:
        canon = _HEADER_ALIASES[low]
        if canon.lower() != low:
            logger.info("[table_extractor] header aliased: %r -> %r", cleaned, canon)
        return canon
    best, best_ratio = None, 0.0
    for canon in KNOWN_HEADERS:
        r = difflib.SequenceMatcher(None, low, canon.lower()).ratio()
        if r > best_ratio:
            best, best_ratio = canon, r
    if best is not None and best_ratio >= _HEADER_MATCH_THRESHOLD:
        if best.lower() != low:
            logger.info("[table_extractor] header normalized: %r -> %r (%.2f)", cleaned, best, best_ratio)
        return best
    return cleaned


def _header_spans(span_boxes: List[List[float]], block_cols: List[List[float]],
                  row0_bottom: float, k: int) -> List[Tuple[int, int, float, List[float]]]:
    """
    Which TATR spanning cells are HEADER titles sitting above >=2 of this block's columns?
    Returns (start_col, length, span_bottom_y, span_box), left to right.

    Single-column "spans" are just rowspans and are ignored; so are merged cells further
    down the table (they start below the first row).
    """
    out = []
    for s in span_boxes:
        if s[1] > row0_bottom * 0.5:      # must start inside the first row band
            continue
        covered = [i for i, c in enumerate(block_cols)
                   if min(s[2], c[2]) - max(s[0], c[0]) > 0.5 * (c[2] - c[0])]
        if len(covered) < 2 or len(covered) >= k:
            continue                      # 1 column = rowspan; all columns = not a title
        if covered != list(range(covered[0], covered[0] + len(covered))):
            continue                      # must be contiguous
        out.append((covered[0], len(covered), s[3], s))
    out.sort(key=lambda t: t[0])
    return out


def _read_header_by_columns(crop: Image.Image, block_cols: List[List[float]],
                            row_boxes: List[List[float]], span_boxes: List[List[float]],
                            k: int) -> Tuple[List[str], List[Tuple[int, int, str]], float]:
    """
    Read the header from geometry: column ORDER comes from the column boxes, and the
    two-level structure comes from TATR's spanning cells.

    The old version trusted TATR's 'table column header' box for the header's height —
    but that box OVERSHOOTS into the first data row on printed tables (measured: bottom
    at y=277 when the header ends at y=136). Every per-column read then returned
    header+data glued together ("Basin Western Onland"), and those corrupted names were
    handed to the band read as the column schema, which made the model merge columns.

    Header extent is therefore derived from the row bands instead:
      • header = row 0, PLUS row 1 only when multi-column titles exist AND row 1 reads
        as header text (no digits) rather than data. That is exactly what separates the
        ONGC form (title row + sub-header row) from this report (title row, data below).

    Returns (leaf names, spans as (start, length, label)).
    """
    import ocr_olm

    if not row_boxes or len(block_cols) != k:
        return [], [], 0.0

    row0_bottom = row_boxes[0][3]
    spans = _header_spans(span_boxes, block_cols, row0_bottom, k)
    spanned = {j for st, ln, _, _ in spans for j in range(st, st + ln)}

    def col_x(i):
        return int(max(0, block_cols[i][0])), int(min(crop.width, block_cols[i][2]))

    # --- Pass A: the span titles, plus EVERY column's row-1 cell, so we can tell a
    #     sub-header row from the first data row. One batched call.
    #
    #     Row 1's nature must NOT be inferred from whether a span was detected: TATR
    #     misses spans (it found none for the "INTERPRETATION" title), and requiring one
    #     collapsed the header to a 44px sliver, so the cells read as "system" / "ON".
    #     Whether row 1 is a header is a question about its CONTENT — header text has no
    #     digits, a data row does.
    imgs, kinds = [], []
    for st, ln, sb, box in spans:
        x0, x1 = int(max(0, box[0])), int(min(crop.width, box[2]))
        if x1 - x0 >= 4:
            imgs.append(crop.crop((x0, int(box[1]), x1, int(sb))))
            kinds.append(("span", st))

    row1 = row_boxes[1] if len(row_boxes) > 1 else None
    if row1 is not None:
        for i in range(k):
            x0, x1 = col_x(i)
            if x1 - x0 >= 4:
                imgs.append(crop.crop((x0, int(row1[1]), x1, int(row1[3]))))
                kinds.append(("row1", i))

    raws = []
    if imgs:
        try:
            raws = ocr_olm._run_enhanced_ocr_batch(
                imgs, [ocr_olm._get_header_cell_prompt()] * len(imgs),
                max_new_tokens=ocr_olm.TOKENS_HEADER_CELL)
        except Exception as e:
            logger.warning("[table_extractor] header pass A failed: %s", e)
            raws = [""] * len(imgs)

    span_label = {}
    row1_text = {}
    for (kind, i), raw in zip(kinds, raws):
        (span_label if kind == "span" else row1_text)[i] = _clean_name(raw)

    sub_header = bool(row1_text) and _looks_like_header(list(row1_text.values()), [])
    header_bottom = float(row_boxes[1][3]) if (sub_header and row1 is not None) else float(row0_bottom)
    header_bottom = min(crop.height, header_bottom + _ROW_TRIM_MARGIN)

    # --- Pass B: leaf names for the columns NOT under a title. Columns under a title
    #     take their sub-header name from row 1, or stay blank when there isn't one
    #     (an unnamed sub-column, exactly as printed).
    leaf: List[str] = [""] * k
    imgs, slots = [], []
    for i in range(k):
        if i in spanned:
            leaf[i] = row1_text.get(i, "") if sub_header else ""
            continue
        x0, x1 = col_x(i)
        if x1 - x0 >= 4:
            imgs.append(crop.crop((x0, 0, x1, int(header_bottom))))
            slots.append(i)

    if imgs:
        try:
            raws = ocr_olm._run_enhanced_ocr_batch(
                imgs, [ocr_olm._get_header_cell_prompt()] * len(imgs),
                max_new_tokens=ocr_olm.TOKENS_HEADER_CELL)
        except Exception as e:
            logger.warning("[table_extractor] header pass B failed: %s", e)
            raws = [""] * len(imgs)
        for i, raw in zip(slots, raws):
            leaf[i] = _clean_name(raw)

    if not any(leaf) and not spans:
        return [], [], float(header_bottom)

    out_spans = [(st, ln, span_label.get(st, "")) for st, ln, _, _ in spans
                 if span_label.get(st, "")]
    logger.info("[table_extractor] header: %d span(s), sub_header=%s, leaf=%r",
                len(out_spans), sub_header, leaf)
    return leaf, out_spans, float(header_bottom)


def _read_subheader_strip(crop: Image.Image, span_box: List[float], st: int, ln: int,
                          row1_box: List[float]) -> Dict[int, str]:
    """
    Recover the `ln` sub-headers under a spanning title when the per-column reads
    came back blank.

    The per-column path crops each sub-header from its own column box, so when TATR
    miscounts the columns (e.g. 13 vs the real 12 on a skewed scan) those crops are
    misaligned and read empty. Here we instead take the SPAN's own horizontal extent
    — a single, reliably-detected region — and divide it evenly into `ln` cells,
    reading each. That is independent of the individual column boxes, so it survives
    the miscount. Returns {column_index: name} for the cells that read non-empty.
    """
    import ocr_olm

    x0 = int(max(0, span_box[0]))
    x1 = int(min(crop.width, span_box[2]))
    y0 = int(max(0, row1_box[1]))
    y1 = int(min(crop.height, row1_box[3]))
    if ln < 1 or (x1 - x0) < 4 * ln or (y1 - y0) < 4:
        return {}

    cell_w = (x1 - x0) / ln
    imgs, idxs = [], []
    for j in range(ln):
        cx0 = int(x0 + j * cell_w)
        cx1 = int(x0 + (j + 1) * cell_w)
        if cx1 - cx0 >= 4:
            imgs.append(crop.crop((cx0, y0, cx1, y1)))
            idxs.append(st + j)
    if not imgs:
        return {}

    try:
        raws = ocr_olm._run_enhanced_ocr_batch(
            imgs, [ocr_olm._get_header_cell_prompt()] * len(imgs),
            max_new_tokens=ocr_olm.TOKENS_HEADER_CELL)
    except Exception as e:
        logger.warning("[table_extractor] sub-header strip read failed: %s", e)
        return {}

    out: Dict[int, str] = {}
    for i, raw in zip(idxs, raws):
        name = _clean_name(raw)
        if name:
            out[i] = name
    return out


# The single-pass header read is fragile: when a spanning title sits over >=2 sub-columns
# (e.g. "Corrections" over "Shot m. secs" / "Receiver m. secs" on the ONGC form), the VLM
# occasionally misses the sub-header row and those columns come back blank. Observed on one
# form: the SAME image gave sub_header=True on some passes and False on others. Retrying and
# keeping the richest result recovers the sub-headers reliably.
_HEADER_READ_ATTEMPTS = 3


def _read_header_robust(crop: Image.Image, block_cols: List[List[float]],
                        row_boxes: List[List[float]], span_boxes: List[List[float]],
                        k: int, attempts: int = _HEADER_READ_ATTEMPTS
                        ) -> Tuple[List[str], List[Tuple[int, int, str]], float]:
    """
    Retry _read_header_by_columns and keep the result with the most filled sub-columns.

    Whether a column SHOULD carry a sub-header is judged from TATR geometry
    (_header_spans), never from the VLM-labelled spans a pass returns — a bad pass can
    drop the title label itself, which would otherwise make a blank result look
    "complete". Early-exits the moment every geometrically-spanned column has a name, so
    a good first read (the common case) still costs a single pass.
    """
    geo_spanned: set = set()
    if row_boxes and len(block_cols) == k:
        for st, ln, _sb, _box in _header_spans(span_boxes, block_cols, row_boxes[0][3], k):
            geo_spanned.update(range(st, st + ln))

    best: Optional[Tuple[List[str], List[Tuple[int, int, str]], float]] = None
    best_score = -1
    for attempt in range(max(1, attempts)):
        leaf, out_spans, hbottom = _read_header_by_columns(crop, block_cols, row_boxes, span_boxes, k)
        filled_spanned = sum(1 for j in geo_spanned if j < len(leaf) and leaf[j].strip())
        # rank by filled sub-columns first, then by total filled leaf names
        score = filled_spanned * 1000 + sum(1 for v in leaf if v.strip())
        if score > best_score:
            best, best_score = (leaf, out_spans, hbottom), score
        if not geo_spanned or filled_spanned == len(geo_spanned):
            if attempt > 0:
                logger.info("[table_extractor] header sub-columns recovered on attempt %d/%d",
                            attempt + 1, attempts)
            return best

    # Retries exhausted with some geometrically-spanned columns still blank. This is
    # the SYSTEMATIC failure (TATR miscounted the columns, so every per-column read is
    # misaligned) that retrying cannot fix. Fall back to reading each span's sub-headers
    # off the span box directly (see _read_subheader_strip), which is column-count-proof.
    if best is not None and row_boxes and len(row_boxes) > 1 and len(block_cols) == k:
        leaf, out_spans, hbottom = best
        row1 = row_boxes[1]
        recovered = 0
        for st, ln, _sb, span_box in _header_spans(span_boxes, block_cols, row_boxes[0][3], k):
            if all(j < len(leaf) and leaf[j].strip() for j in range(st, st + ln)):
                continue  # this span's sub-columns are already filled
            for i, name in _read_subheader_strip(crop, span_box, st, ln, row1).items():
                if i < len(leaf) and not leaf[i].strip():
                    leaf[i] = name
                    recovered += 1
        if recovered:
            logger.info("[table_extractor] recovered %d sub-header(s) via strip read", recovered)
            best = (leaf, out_spans, hbottom)

    logger.info("[table_extractor] header: kept richest of %d attempts (spanned filled %d/%d)",
                attempts, best_score // 1000, len(geo_spanned))
    return best if best is not None else ([], [], 0.0)


def _detect_groups_and_headers(crop: Image.Image) -> Tuple[int, List[str], List[dict]]:
    """
    One VLM reasoning pass: how many identical side-by-side column-blocks, the leaf
    header names of a single block, and any spanning super-headers. Returns
    (groups, headers, spanning) where spanning is a list of
    {"label": str, "over": [leaf names]}. On failure returns (1, [], []).
    """
    import ocr_olm

    prompt = ocr_olm._get_group_detection_prompt()
    try:
        raw, _ = ocr_olm._run_enhanced_ocr(crop, prompt, max_new_tokens=ocr_olm.TOKENS_JSON)
    except Exception as e:
        logger.warning("[table_extractor] group-detection VLM call failed: %s", e)
        return 1, [], []

    data = _parse_json_object(raw)
    if not isinstance(data, dict):
        return 1, [], []

    headers = data.get("headers") or []
    if not isinstance(headers, list):
        headers = []
    headers = [str(h).strip() for h in headers if str(h).strip()]

    spanning = []
    for sp in (data.get("spanning") or []):
        if not isinstance(sp, dict):
            continue
        label = str(sp.get("label", "")).strip()
        over = [str(o).strip() for o in (sp.get("over") or []) if str(o).strip()]
        if label and len(over) >= 2:
            spanning.append({"label": label, "over": over})

    try:
        groups = int(data.get("groups", 1))
    except (TypeError, ValueError):
        groups = 1
    groups = max(1, groups)

    return groups, headers, spanning


def _trim_to_grid(cols: List[List[float]], target: int) -> List[List[float]]:
    """
    Reconcile TATR's detected columns with the expected grid of `target` columns.

    TATR over-reports in two distinct ways, and they need different treatment:
      1. NEAR-DUPLICATES — it splits one column into two heavily-overlapping boxes
         (e.g. centers 560 and 568). These are merged.
      2. THIN MARGIN ARTIFACTS — a sliver at the far left/right from edge marks or
         a circled annotation. These are dropped, but ONLY when clearly thin
         relative to the median column.

    Anything we cannot confidently identify as an artifact is left alone: dropping a
    REAL edge column (e.g. 'Surface Pkt. No.') rotates every block's data by one and
    is far worse than falling back to a span-based split.
    """
    cols = sorted((list(c) for c in cols), key=lambda b: (b[0] + b[2]) / 2)
    if not cols:
        return []

    widths = sorted(c[2] - c[0] for c in cols)
    med = widths[len(widths) // 2] or 1.0

    # 1. Merge near-duplicate / heavily-overlapping detections.
    merged: List[List[float]] = []
    for c in cols:
        if merged:
            prev = merged[-1]
            centre_gap = ((c[0] + c[2]) / 2) - ((prev[0] + prev[2]) / 2)
            overlap = min(prev[2], c[2]) - max(prev[0], c[0])
            min_w = min(prev[2] - prev[0], c[2] - c[0]) or 1.0
            if centre_gap < 0.5 * med or (overlap / min_w) > 0.6:
                merged[-1] = [min(prev[0], c[0]), min(prev[1], c[1]),
                              max(prev[2], c[2]), max(prev[3], c[3])]
                continue
        merged.append(c)
    cols = merged

    # 2. Drop thin edge artifacts — but never a plausibly-real column.
    while len(cols) > target >= 1 and len(cols) >= 2:
        first_w = cols[0][2] - cols[0][0]
        last_w = cols[-1][2] - cols[-1][0]
        if first_w <= last_w and first_w < 0.5 * med:
            cols = cols[1:]
        elif last_w < first_w and last_w < 0.5 * med:
            cols = cols[:-1]
        else:
            break  # no clear artifact — leave the grid alone

    return cols


def _band_layout(crop: Image.Image, groups: int, k: int, col_boxes: List[List[float]]
                 ) -> Tuple[List[Tuple[int, int]], List[List[float]]]:
    """
    Compute the x-range of each block band, plus the reconciled column boxes.

    Strategy (most→least reliable):
      1. Reconciled column count == groups*k → split at the true midpoints between
         the last column of one block and the first column of the next.
      2. Else split the DETECTED COLUMN SPAN into `groups` equal bands (excludes the
         outer margins that a full-crop split would wrongly include).
      3. Else even-split the whole crop.
    Returns (x_splits, cols) where `cols` is [] when the grid could not be reconciled.
    """
    w = crop.width
    pad = max(2, w // 200)
    target = groups * k
    cols = _trim_to_grid(col_boxes, target)
    x_splits: List[Tuple[int, int]] = []

    if len(cols) == target:
        for g in range(groups):
            left_col = cols[g * k]
            right_col = cols[g * k + k - 1]
            x0 = left_col[0] if g == 0 else (cols[g * k - 1][2] + left_col[0]) / 2
            x1 = right_col[2] if g == groups - 1 else (right_col[2] + cols[(g + 1) * k][0]) / 2
            x_splits.append((int(max(0, x0 - pad)), int(min(w, x1 + pad))))
        return x_splits, cols

    if cols:
        span_l = min(c[0] for c in cols)
        span_r = max(c[2] for c in cols)
        band_w = (span_r - span_l) / groups
        for g in range(groups):
            x0 = span_l + g * band_w
            x1 = span_l + (g + 1) * band_w
            x_splits.append((int(max(0, x0 - pad)), int(min(w, x1 + pad))))
        return x_splits, []

    band_w = w / groups
    for g in range(groups):
        x_splits.append((int(max(0, g * band_w - pad)), int(min(w, (g + 1) * band_w + pad))))
    return x_splits, []


def _split_bands(crop: Image.Image, groups: int, k: int, col_boxes: List[List[float]]
                 ) -> Tuple[List[Image.Image], List[Tuple[int, int]], List[List[float]]]:
    """Split a table crop into `groups` vertical bands (one per repeated block)."""
    if groups <= 1:
        return [crop], [(0, crop.width)], _trim_to_grid(col_boxes, k)

    x_splits, cols = _band_layout(crop, groups, k, col_boxes)
    bands, kept = [], []
    for (x0, x1) in x_splits:
        if x1 - x0 >= 8:
            bands.append(crop.crop((x0, 0, x1, crop.height)))
            kept.append((x0, x1))
    if not bands:
        return [crop], [(0, crop.width)], []
    return bands, kept, cols


def _read_bands_batched(bands: List[Image.Image], header_hint: List[str], k: int,
                        expected_rows: int = 0
                        ) -> Tuple[List[List[List[str]]], List[bool]]:
    """
    Read every block band in ONE batched generate() call. The bands are independent,
    so running them sequentially just paid the launch cost G times.

    Also reports, per band, whether the model IGNORED the column count — i.e. most of
    its rows came back with the wrong number of cells. On the GEOPIC report it kept
    gluing a narrow number column onto its neighbour ("01 GGN-11 Released as GNZX"),
    emitting 5 cells where the grid has 7. Padding cannot repair that: the values are
    already fused. The caller escalates those bands to a cell-by-cell read.
    """
    import ocr_olm

    prompt = ocr_olm._get_structured_table_prompt(header_hint, k, expected_rows or None)
    budget = _token_budget(expected_rows, k)
    try:
        raws = ocr_olm._run_enhanced_ocr_batch(bands, [prompt] * len(bands),
                                               max_new_tokens=budget)
    except Exception as e:
        logger.warning("[table_extractor] batched band read failed (%s) — sequential", e)
        raws = [_read_band_raw(b, header_hint, k, expected_rows) for b in bands]

    blocks, bad = [], []
    for raw in raws:
        rows, mismatched = _parse_band_checked(raw, header_hint, k)
        blocks.append(rows)
        bad.append(mismatched)
    return blocks, bad


_TOKENS_PER_CELL = 14    # "<td>-112</td>" and friends, generously
_TOKENS_MIN      = 2048
_TOKENS_MAX      = 12000


def _token_budget(expected_rows: int, k: int) -> int:
    """
    Output-token budget for a band read, sized to the table.

    A flat 2048 silently TRUNCATED long tables: a 4-column, 38-row list came back with
    32 rows, and an 18-column, 45-row one with 9 — generation simply ran out of budget
    mid-table and the missing rows vanished without any error. HTML is verbose, so the
    budget has to scale with rows x columns.
    """
    if expected_rows <= 0 or k <= 0:
        return _TOKENS_MIN
    need = expected_rows * k * _TOKENS_PER_CELL + 256
    return int(min(_TOKENS_MAX, max(_TOKENS_MIN, need)))


def _read_band_raw(band: Image.Image, header_hint: List[str], k: int, expected_rows: int) -> str:
    import ocr_olm
    try:
        raw, _ = ocr_olm._run_enhanced_ocr(
            band, ocr_olm._get_structured_table_prompt(header_hint, k, expected_rows or None),
            max_new_tokens=_token_budget(expected_rows, k))
        return raw
    except Exception as e:
        logger.warning("[table_extractor] band read failed: %s", e)
        return ""


def _parse_band_checked(raw: str, header_hint: List[str], k: int
                        ) -> Tuple[List[List[str]], bool]:
    """Parse a band, and report whether the model mostly ignored the column count."""
    rows_raw = _html_to_rows(raw) or _pipe_to_rows(raw)
    _, rows = _parse_band(raw, header_hint, k)
    if not rows_raw or not rows:
        return rows, False
    data = [r for r in rows_raw if any(str(c).strip() for c in r)]
    if not data:
        return rows, False
    wrong = sum(1 for r in data if len(r) != k)
    return rows, (wrong / len(data)) > 0.5


_CELL_BATCH = 16      # crops per batched generate() when reading cell-by-cell
_RULE_DARKNESS = 0.60  # a scanline this dark across the table is a ruled line
_RULE_INSET = 3        # px kept clear of the rule itself when cutting a cell


_BLANK_INK = 0.004  # ink fraction below which a cell's interior is empty


def _is_blank_cell(cell: Image.Image) -> bool:
    """
    Is this cell empty? Measured on the cell's INTERIOR, inset far enough to exclude the
    ruled border, so a bordered-but-empty cell still reads as blank.
    """
    try:
        import numpy as np
    except ImportError:
        return False
    w, h = cell.size
    mx, my = max(2, int(w * 0.12)), max(2, int(h * 0.12))
    if w - 2 * mx < 3 or h - 2 * my < 3:
        return False
    inner = np.array(cell.convert("L"))[my:h - my, mx:w - mx]
    return float((inner < 160).mean()) < _BLANK_INK


def _horizontal_rules(crop: Image.Image) -> List[int]:
    """y positions of full-width horizontal rules (printed tables only)."""
    try:
        import numpy as np
    except ImportError:
        return []
    a = np.array(crop.convert("L"))
    prof = (a < 160).mean(axis=1)
    ys = [int(y) for y in range(a.shape[0]) if prof[y] > _RULE_DARKNESS]
    if not ys:
        return []
    groups: List[List[int]] = []
    for y in ys:
        if groups and y - groups[-1][-1] <= 3:
            groups[-1].append(y)
        else:
            groups.append([y])
    return [int(sum(g) / len(g)) for g in groups]


def _row_bands_for_cells(crop: Image.Image, tatr_rows: List[List[float]],
                         header_bottom: float) -> List[Tuple[int, int]]:
    """
    Vertical band of each DATA row, for cutting cells.

    TATR's row boxes are good enough to count rows, but not to cut from: on the GEOPIC
    report the last two were shifted ~40px down, so the cells contained only the bottom
    halves of the glyphs and "TOTAL" read as "TITLE", "04" as "54". A printed table's
    ruled lines give the boundaries exactly, so snap to them when they exist. Scans
    (the ONGC form) have no detectable rules — fall back to the TATR boxes there.
    """
    tatr_bands = [(int(r[1]), int(r[3])) for r in tatr_rows]

    rules = [y for y in _horizontal_rules(crop) if y >= header_bottom - 20]
    if len(rules) >= 2:
        bands = [(rules[i] + _RULE_INSET, rules[i + 1] - _RULE_INSET)
                 for i in range(len(rules) - 1)]
        bands = [b for b in bands if b[1] - b[0] >= 8]
        if bands and abs(len(bands) - len(tatr_bands)) <= 1:
            logger.info("[table_extractor] snapping %d cell rows to ruled lines", len(bands))
            return bands

    return tatr_bands


def _read_cells_geometric(crop: Image.Image, block_cols: List[List[float]],
                          data_rows: List[List[float]], k: int,
                          header_bottom: float = 0.0) -> List[List[str]]:
    """
    Read a block CELL BY CELL, straight off TATR's row x column grid.

    This is the escalation for tables where the model refuses to honour the column
    count — it kept fusing a narrow number column into the wide name beside it. When
    each cell is its own crop, merging is not physically possible: the value read from
    the "01" cell cannot contain "GGN-11", because that text is not in the image.

    Slower (k x rows tiny reads, batched), so it only runs when the band read is shown
    to have ignored the grid.
    """
    import ocr_olm

    bands = _row_bands_for_cells(crop, data_rows, header_bottom)

    imgs, slots = [], []
    blanks = 0
    for r_i, (by0, by1) in enumerate(bands):
        for c_i in range(k):
            x0 = int(max(0, block_cols[c_i][0]))
            x1 = int(min(crop.width, block_cols[c_i][2]))
            y0 = int(max(0, by0))
            y1 = int(min(crop.height, by1))
            if x1 - x0 < 4 or y1 - y0 < 4:
                continue
            cell = crop.crop((x0, y0, x1, y1))
            # An empty cell is decidable from the image — don't ask the model, which
            # invents a mark rather than returning nothing (it answered "X" and "-"
            # for the two blank cells of the TOTAL row). Skipping them is also free.
            if _is_blank_cell(cell):
                blanks += 1
                continue
            imgs.append(cell)
            slots.append((r_i, c_i))

    out = [["" for _ in range(k)] for _ in bands]
    if not imgs:
        return out

    logger.info("[table_extractor] escalating to cell-by-cell: %d cells (%d rows x %d cols)",
                len(imgs), len(bands), k)

    prompt = ocr_olm._get_cell_prompt()
    texts: List[str] = []
    for i in range(0, len(imgs), _CELL_BATCH):
        chunk = imgs[i:i + _CELL_BATCH]
        try:
            texts.extend(ocr_olm._run_enhanced_ocr_batch(
                chunk, [prompt] * len(chunk), max_new_tokens=ocr_olm.TOKENS_CELL))
        except Exception as e:
            logger.warning("[table_extractor] cell batch failed: %s", e)
            texts.extend([""] * len(chunk))

    for (r_i, c_i), t in zip(slots, texts):
        val = _clean_value(t)  # a DATA cell, not a header — must not be capped at 70 chars
        if val.upper() in ("EMPTY", "BLANK", "N/A", "NONE", "-EMPTY-"):
            val = ""
        out[r_i][c_i] = val

    return [r for r in out if any(c.strip() for c in r)]


def _parse_band(raw: str, header_hint: List[str], k: int
                ) -> Tuple[Optional[List[str]], List[List[str]]]:
    """Parse one band's raw model output into (header_row_or_None, repaired data rows)."""
    # HTML is what we ask for (explicit <td></td> keeps blank cells in place); the
    # pipe parser stays as a fallback for when the model answers in that shape anyway.
    rows = _html_to_rows(raw) or _pipe_to_rows(raw)
    if not rows:
        return None, []

    # Strip EVERY leading header-like row, not just the first: on two-level headers
    # the model often emits a <thead> AND a leftover sub-header <tr> ("Shot | Receiver").
    # Leaving it in shifts that block's data down one row relative to the others.
    header_row: Optional[List[str]] = None
    for _ in range(3):
        if not rows or not _looks_like_header(rows[0], header_hint):
            break
        fixed = _normalize_row_widths([rows[0]], k)
        if header_row is None and fixed:
            header_row = fixed[0]
        rows = rows[1:]

    return header_row, _normalize_row_widths(rows, k)


# ============================================================================
# PARSING / REPAIR HELPERS
# ============================================================================

def _parse_json_object(text: str) -> Optional[dict]:
    """Extract the first JSON object from a possibly fenced / chatty string."""
    if not text:
        return None
    t = text.strip()
    t = re.sub(r"^```(?:json)?", "", t).strip()
    t = re.sub(r"```$", "", t).strip()
    try:
        return json.loads(t)
    except json.JSONDecodeError:
        pass
    m = re.search(r"\{.*\}", t, re.DOTALL)
    if m:
        try:
            return json.loads(m.group(0))
        except json.JSONDecodeError:
            return None
    return None


def _html_to_rows(raw: str) -> List[List[str]]:
    """Parse the first <table> out of raw model output into a list of string rows."""
    if not raw:
        return []
    cleaned = re.sub(r"^```(?:html)?", "", raw.strip())
    cleaned = re.sub(r"```$", "", cleaned).strip()

    soup = BeautifulSoup(cleaned, "html.parser")
    table = soup.find("table")
    if table is None:
        return []

    rows: List[List[str]] = []
    for tr in table.find_all("tr"):
        cells = tr.find_all(["td", "th"])
        if not cells:
            continue
        rows.append([c.get_text(" ", strip=True) for c in cells])
    return rows


def _norm_key(s: Any) -> str:
    """Aggressive key for header matching: 'Shot m. secs' == 'Shot m.secs'."""
    return re.sub(r"[^a-z0-9]", "", str(s or "").lower())


def _looks_like_header(row: List[str], headers: List[str]) -> bool:
    """
    True if `row` is a header line rather than data. Two signals:
      1. It restates a known header label (punctuation/case-insensitive, since the
         model spells them inconsistently: 'Shot m. secs' vs 'Shot m.secs').
      2. It is textual with no digits at all — which is what a leftover second-level
         header row ('Shot | Receiver') looks like, while every data row in these
         numeric forms carries digits.
    """
    cells = [c for c in row if str(c).strip()]
    if not cells:
        return False

    hdr_keys = {_norm_key(h) for h in headers if _norm_key(h)}
    row_keys = {_norm_key(c) for c in cells if _norm_key(c)}
    has_digit = any(re.search(r"\d", str(c)) for c in cells)
    has_alpha = any(re.search(r"[A-Za-z]", str(c)) for c in cells)

    if hdr_keys and row_keys & hdr_keys:
        # If the row has digits, it is likely a data row.
        # Only call it a header if a significant portion (>= 50%) of its cells match header names.
        if has_digit:
            if len(row_keys & hdr_keys) >= len(row_keys) * 0.5:
                return True
        else:
            # If it has no digits and shares at least one key with the header, it's a leftover header row
            return True

    return has_alpha and not has_digit


def _normalize_row_widths(rows: List[List[str]], k: int) -> List[List[str]]:
    """
    Force every row to exactly k cells: pad short rows with empty cells, truncate
    overflow. Skips fully-empty rows. Logs any repair — never silently shifts.
    """
    fixed: List[List[str]] = []
    for i, row in enumerate(rows):
        if not any(str(c).strip() for c in row):
            continue
        if len(row) < k:
            logger.info("[table_extractor] padding row %d: %d -> %d cols %r", i, len(row), k, row)
            row = list(row) + [""] * (k - len(row))
        elif len(row) > k:
            logger.info("[table_extractor] truncating row %d: %d -> %d cols %r", i, len(row), k, row)
            row = list(row[:k])
        fixed.append(row)
    return fixed


# ============================================================================
# HTML ASSEMBLY
# ============================================================================

def _esc(s: Any) -> str:
    from html import escape
    return escape("" if s is None else str(s))


# ── Harvesting raw table HTML for the RAG tables lane ───────────────────────
# Matches an OPENING <table> through either its close or (deliberately) the end of the
# string. OCR sometimes stops mid-table, and an unterminated table is still worth keeping:
# flagged well_formed=False, it produces a visible "incomplete read" warning instead of a
# silent undercount. A strict `<table>...</table>` regex would drop those rows entirely
# and nothing downstream would ever know they existed.
_TABLE_OPEN_RE = re.compile(r"<table\b[\s\S]*?(?:</table>|$)", re.IGNORECASE)
_TAG_RE = re.compile(r"<[^>]+>")
_CAPTION_MAX = 160


def _table_digest(html: str, caption: str = "") -> str:
    """A short natural-language description of a table — this is what gets embedded.

    Never embed the HTML itself: markup tokens ("<td>", "</tr>") dominate the vector, and
    HTML runs ~1.7 chars/token against this model's tokenizer (vs ~3.9 for prose), so a
    large table would also blow past the embedder's useful window. The digest keeps the
    header row and a sample of first-column values, which is what questions actually name
    ("the 17 1/2 inch hole", "Depth Interval").
    """
    headers = [_TAG_RE.sub("", h).strip()
               for h in re.findall(r"<th\b[^>]*>([\s\S]*?)</th>", html, re.IGNORECASE)]
    first_col: List[str] = []
    for row in re.findall(r"<tr\b[^>]*>([\s\S]*?)</tr>", html, re.IGNORECASE):
        cells = re.findall(r"<td\b[^>]*>([\s\S]*?)</td>", row, re.IGNORECASE)
        if cells:
            v = _TAG_RE.sub(" ", cells[0]).strip()
            if v and v not in first_col:
                first_col.append(v)
    parts = []
    if caption:
        parts.append(caption)
    if headers:
        parts.append("Columns: " + ", ".join(dict.fromkeys(h for h in headers if h)))
    if first_col:
        parts.append("Rows: " + ", ".join(first_col[:12]))
    return " | ".join(parts)[:1200]


def expand_table_for_prompt(html: str) -> str:
    """Rewrite a table so every row has one cell per column — for LLM consumption only.

    WHY. A rowspan group writes its spanning cell once, in the group's first row, so the
    rows beneath it carry FEWER <td>s than there are columns. A language model then has to
    infer which header each cell belongs to, and a 14B model does this unreliably: asked
    for "Yp" on a rowspan-covered row it returned the neighbouring Pv value, and after the
    prompt was tightened to warn about exactly that, it shifted one column the other way
    instead. Both answers were confident and both were wrong.

    Expanding the span is information-preserving — rowspan="4" MEANS the value applies to
    all four rows, so repeating it states what the markup already said — and it turns
    column alignment from an inference into a count. The raw HTML with its spans intact
    stays in page_tables; only the prompt copy is rectangular.

    Also emits an explicit <th> count and per-cell headers are left to the reader; colspans
    are expanded the same way. Unparseable input is returned unchanged: a table we cannot
    rewrite is still worth showing.
    """
    try:
        rows = re.findall(r"<tr\b[^>]*>([\s\S]*?)(?:</tr>|$)", html, re.IGNORECASE)
        if not rows:
            return html
        grid: List[List[Tuple[str, str]]] = []     # (tag, inner_html)
        carry: Dict[int, Tuple[int, str, str]] = {}  # col -> (rows_left, tag, inner)

        for raw in rows:
            cells = re.findall(r"<(t[hd])\b([^>]*)>([\s\S]*?)</\1>", raw, re.IGNORECASE)
            out: List[Tuple[str, str]] = []
            col = 0
            # Re-emit any cell still spanning down into this row, at its own column.
            def _drain():
                nonlocal col
                while col in carry:
                    left, tag, inner = carry[col]
                    out.append((tag, inner))
                    if left - 1 > 0:
                        carry[col] = (left - 1, tag, inner)
                    else:
                        del carry[col]
                    col += 1
            _drain()
            for tag, attrs, inner in cells:
                rs = re.search(r'rowspan\s*=\s*"?(\d+)', attrs, re.IGNORECASE)
                cs = re.search(r'colspan\s*=\s*"?(\d+)', attrs, re.IGNORECASE)
                nrs, ncs = (int(rs.group(1)) if rs else 1), (int(cs.group(1)) if cs else 1)
                for _ in range(max(1, ncs)):
                    out.append((tag.lower(), inner))
                    if nrs > 1:
                        carry[col] = (nrs - 1, tag.lower(), inner)
                    col += 1
                    _drain()
            grid.append(out)

        if not grid:
            return html
        width = max(len(r) for r in grid)
        parts = ["<table>"]
        for r in grid:
            r = r + [("td", "")] * (width - len(r))
            parts.append("  <tr>" + "".join(f"<{t}>{v}</{t}>" for t, v in r) + "</tr>")
        parts.append("</table>")
        return "\n".join(parts)
    except Exception:
        return html


def harvest_inline_tables(page_text: str) -> List[Dict[str, Any]]:
    """Pull every <table> out of a page's text, with the metadata page_tables needs.

    Used by two callers that must agree exactly: the live capture path (ocr_olm's
    no-table branch, where the whole-page VLM read leaves tables inline in the prose) and
    rag/backfill_page_tables.py (recovering tables already sitting in prose_chunks).

    Operate on PAGE-level text, never a single prose chunk: ingest._chunk_prose splits on
    blank lines, which cuts tables in half, so per-chunk harvesting invents fragments and
    misses tables that straddle a boundary.

    Returns dicts shaped for the page_tables columns. `caption` is the last non-empty,
    non-markup line before the table — usually the section heading that names it.
    """
    out: List[Dict[str, Any]] = []
    for m in _TABLE_OPEN_RE.finditer(page_text or ""):
        html = m.group(0).strip()
        if "<td" not in html.lower() and "<th" not in html.lower():
            continue  # an empty shell, not a table

        preceding = (page_text[:m.start()].rstrip().split("\n") or [""])
        caption = ""
        for line in reversed(preceding[-4:]):
            line = _TAG_RE.sub("", line).strip()
            if line:
                caption = line[:_CAPTION_MAX]
                break

        n_rows = len(re.findall(r"<tr\b", html, re.IGNORECASE))
        header_cells = re.findall(r"<t[hd]\b[^>]*>",
                                  re.search(r"<tr\b[^>]*>[\s\S]*?(?:</tr>|$)", html,
                                            re.IGNORECASE).group(0), re.IGNORECASE) \
            if n_rows else []
        out.append({
            "html": html,
            "digest": _table_digest(html, caption),
            "caption": caption or None,
            "n_rows": max(0, n_rows - 1),          # exclude the header row
            "n_cols": len(header_cells),
            # ~1.7 chars/token measured on real table HTML with this model's tokenizer.
            # Only a precomputed hint — prompt_budget re-counts for real before deciding.
            "est_tokens": max(1, int(len(html) / 1.7)),
            "well_formed": html.lower().rstrip().endswith("</table>"),
            "source": "prose_html",
        })
    return out


# Inline styles so the grid is always visible, independent of any frontend
# CSS (the app defines no table-border styles, and the edit view renders raw
# HTML). Keeps the class list too for anything that does style by class.
_TABLE_STYLE = "border-collapse:collapse;width:100%;font-family:inherit;"
_TH_STYLE = "border:1px solid #334155;padding:6px 10px;background:#1e293b;color:#fff;text-align:left;font-weight:600;"
_TD_STYLE = "border:1px solid #cbd5e1;padding:6px 10px;text-align:left;vertical-align:top;"


# Heavier left border marks a block boundary in the wide (side-by-side) layout.
_TD_STYLE_SEP = "border:1px solid #cbd5e1;border-left:3px solid #334155;padding:6px 10px;text-align:left;vertical-align:top;"
_TH_STYLE_SEP = "border:1px solid #334155;border-left:3px solid #0f172a;padding:6px 10px;background:#1e293b;color:#fff;text-align:left;font-weight:600;"


def _cell_style(col_idx: int, k: int, header: bool = False) -> str:
    """Pick the cell style, adding a heavier left border at each block boundary."""
    boundary = k and col_idx > 0 and col_idx % k == 0
    if header:
        return _TH_STYLE_SEP if boundary else _TH_STYLE
    return _TD_STYLE_SEP if boundary else _TD_STYLE


def _rows_to_html(headers: List[str], rows: List[List[str]], k: Optional[int] = None) -> str:
    """Build an aligned HTML table with a header row and body rows.

    Cells carry inline borders/padding so the table renders as a visible grid
    even though the frontend ships no table CSS (and the edit view injects raw
    HTML with no styling of its own). When `k` is given, a heavier border is
    drawn at every k-th column to separate repeated blocks.
    """
    kk = k or 0
    thead = ("<thead><tr>"
             + "".join(f'<th style="{_cell_style(i, kk, header=True)}">{_esc(h)}</th>'
                       for i, h in enumerate(headers))
             + "</tr></thead>")
    body_rows = []
    for row in rows:
        tds = "".join(f'<td style="{_cell_style(i, kk)}">{_esc(c)}</td>'
                      for i, c in enumerate(row))
        body_rows.append(f"<tr>{tds}</tr>")
    tbody = "<tbody>" + "".join(body_rows) + "</tbody>"
    return f'<table class="{_TABLE_CLASSES}" style="{_TABLE_STYLE}">{thead}{tbody}</table>'


def _build_wide_thead(headers: List[str], k: int, g: int,
                      spans: List[Tuple[int, int, str]]) -> str:
    """
    Build the table header. With no spanning titles this is a single row of the k leaf
    names repeated g times. With titles it is a two-row header: the top row carries the
    titles (colspan) and the standalone columns (rowspan=2), the bottom row carries the
    leaf names sitting under a title — which may legitimately be blank, as on the
    "Drilled Location" columns of the GEOPIC report.

    Spans arrive as (start, length, label) column INDICES straight from geometry.
    """
    spans = [s for s in (spans or []) if 2 <= s[1] <= k and 0 <= s[0] and s[0] + s[1] <= k]
    if not spans:
        full = list(headers) * g
        cells = "".join(f'<th style="{_cell_style(i, k, header=True)}">{_esc(h)}</th>'
                        for i, h in enumerate(full))
        return f"<thead><tr>{cells}</tr></thead>"

    covered = {}  # leaf idx -> (start, length, label)
    for start, length, label in spans:
        for j in range(start, start + length):
            covered[j] = (start, length, label)

    top, bottom = [], []
    for blk in range(g):
        col = 0
        while col < k:
            gcol = blk * k + col
            if col in covered:
                start, length, label = covered[col]
                if col == start:
                    top.append(f'<th colspan="{length}" style="{_cell_style(gcol, k, header=True)}">{_esc(label)}</th>')
                    for j in range(start, start + length):
                        bottom.append(f'<th style="{_cell_style(blk * k + j, k, header=True)}">{_esc(headers[j])}</th>')
                col += length
            else:
                top.append(f'<th rowspan="2" style="{_cell_style(gcol, k, header=True)}">{_esc(headers[col])}</th>')
                col += 1
    return f"<thead><tr>{''.join(top)}</tr><tr>{''.join(bottom)}</tr></thead>"


def _blocks_to_wide_html(headers: List[str], blocks: List[List[List[str]]], k: int,
                         spans: Optional[List[Tuple[int, int, str]]] = None) -> str:
    """
    Assemble repeated side-by-side blocks into one wide table that mirrors the
    paper form. Block g occupies columns [g*k : (g+1)*k]; rows are aligned by
    index and short blocks are padded with empty cells so every output row has
    exactly len(blocks)*k cells. A single block degrades to an ordinary table.
    Two-level headers are reconstructed from `spans` — (start, length, label) column
    indices produced from TATR's spanning cells.
    """
    g = len(blocks)
    num_rows = max((len(b) for b in blocks), default=0)

    combined: List[List[str]] = []
    for i in range(num_rows):
        row: List[str] = []
        for b in blocks:
            cell = b[i] if i < len(b) else []
            cell = (list(cell) + [""] * k)[:k]  # normalize each block-slice to k
            row.extend(cell)
        combined.append(row)

    thead = _build_wide_thead(headers, k, g, spans or [])
    body_rows = []
    for row in combined:
        tds = "".join(f'<td style="{_cell_style(i, k)}">{_esc(c)}</td>' for i, c in enumerate(row))
        body_rows.append(f"<tr>{tds}</tr>")
    tbody = "<tbody>" + "".join(body_rows) + "</tbody>"
    return f'<table class="{_TABLE_CLASSES}" style="{_TABLE_STYLE}">{thead}{tbody}</table>'


# ============================================================================
# LONG-FORM UNPIVOT (RAG table lane)
# ============================================================================

_NUM_RE = re.compile(r"-?\d+(?:\.\d+)?")


def _parse_num(text: str):
    """First signed decimal in a cell, as float, or None. Tolerant of OCR noise
    ("-138(4)" -> -138.0, "(1)" -> 1.0). Used both to fill value_num and to judge
    whether a numeric-expected cell actually parsed."""
    if not text:
        return None
    m = _NUM_RE.search(text)
    if not m:
        return None
    try:
        return float(m.group(0))
    except ValueError:
        return None


def _blocks_to_long_rows(structured: Dict[str, Any], page: int) -> List[Dict[str, Any]]:
    """
    Unpivot the wide side-by-side G-block table into one record per cell for the RAG
    database. The presentation layout (blocks side by side) is stripped; each record
    keeps block_index/row_index so provenance points back to the exact paper cell.

    `structured` is the payload from `_extract_table_crop` ({headers, blocks, k, spans}).
    Returns dicts: page, block_index, row_index, col_index, col_header, col_group,
    value_text, value_num, trust.

    trust (coarse heuristic — there is no real per-cell OCR confidence):
      * empty          — blank cell
      * type_mismatch  — a numeric-expected column whose cell has no parseable number
      * ok             — everything else
    A column is "numeric-expected" when >=60% of its non-empty cells parse as numbers,
    decided empirically so it works across form types without hardcoded schemas.
    """
    headers = structured.get("headers") or []
    blocks = structured.get("blocks") or []
    k = int(structured.get("k") or (len(headers) or 1))
    spans = structured.get("spans") or []

    def cell(b, i, j):
        row = b[i] if i < len(b) else []
        return (row[j] if j < len(row) else "").strip()

    # spanning-title label per column (e.g. col 1,2 -> "Corrections")
    col_group = [""] * k
    for st, ln, label in spans:
        for j in range(st, min(st + ln, k)):
            col_group[j] = label

    # empirically decide which columns are numeric
    numeric_col = [False] * k
    for j in range(k):
        nonempty = [cell(b, i, j) for b in blocks for i in range(len(b)) if cell(b, i, j)]
        if nonempty:
            hits = sum(1 for v in nonempty if _parse_num(v) is not None)
            numeric_col[j] = (hits / len(nonempty)) >= 0.6

    rows: List[Dict[str, Any]] = []
    for g, block in enumerate(blocks):
        for i in range(len(block)):
            for j in range(k):
                text = cell(block, i, j)
                num = _parse_num(text)
                if not text:
                    trust = "empty"
                elif numeric_col[j] and num is None:
                    trust = "type_mismatch"
                else:
                    trust = "ok"
                rows.append({
                    "page": page,
                    "block_index": g,
                    "row_index": i,
                    "col_index": j,
                    "col_header": headers[j] if j < len(headers) else "",
                    "col_group": col_group[j],
                    "value_text": text,
                    "value_num": num if numeric_col[j] else None,
                    "trust": trust,
                })
    return rows


def _majority(values: List[str]) -> Optional[str]:
    """The value held by a STRICT majority (> half) of members, ignoring blanks in
    the tally denominator? No — blanks count as votes too (a column that is blank on
    most pages should stay blank). Returns None when no value exceeds half."""
    from collections import Counter
    n = len(values)
    if n == 0:
        return None
    counts = Counter(v.strip() for v in values)
    val, cnt = counts.most_common(1)[0]
    return val if cnt * 2 > n else None


def _col_groups_to_spans(col_group: List[str]) -> List[Tuple[int, int, str]]:
    """Rebuild (start, length, label) spans from a per-column group array — the
    inverse of _blocks_to_long_rows' span→col_group expansion. Only runs of length
    >= 2 with the same non-empty label become a span (a spanning title needs at
    least two columns, matching _build_wide_thead's own filter)."""
    spans: List[Tuple[int, int, str]] = []
    j = 0
    k = len(col_group)
    while j < k:
        lab = col_group[j].strip()
        if not lab:
            j += 1
            continue
        run = j
        while run + 1 < k and col_group[run + 1].strip() == lab:
            run += 1
        length = run - j + 1
        if length >= 2:
            spans.append((j, length, lab))
        j = run + 1
    return spans


def reconcile_doc_headers(doc_rag_pages: Dict[int, Optional[Dict[str, Any]]]) -> None:
    """Cross-page header consistency for a multi-page form (mutates in place).

    Every page of the same form carries IDENTICAL column headers, but the per-page
    VLM header read occasionally hallucinates one page's — e.g. on a UP67 static-
    correction sheet it read the form's 'Seismic Investigation No' field label as the
    table's leaf header 'seismic' and spanning title 'Investigation No' on page 2,
    while pages 1/3/4 correctly read 'Surface Pkt. No.' and 'Corrections'. Group the
    structured tables across pages by column count and, per column position, replace
    an outlier leaf header / group label with the value a strict majority of pages
    agree on. Purely corrective: when there is no majority (genuinely different tables
    that merely share a column count) nothing changes, so it can't homogenize a
    document of dissimilar tables.
    """
    from collections import defaultdict
    groups: Dict[int, List[Dict[str, Any]]] = defaultdict(list)
    for page in doc_rag_pages:
        payload = doc_rag_pages.get(page) or {}
        for st in (payload.get("structured") or []):
            if st and st.get("headers"):
                groups[int(st.get("k") or len(st["headers"]))].append(st)

    for k, members in groups.items():
        if len(members) < 3:  # need enough pages to out-vote a single bad read
            continue

        # 1) leaf headers, per position
        for j in range(k):
            vals = [(m.get("headers") or [])[j] if j < len(m.get("headers") or []) else ""
                    for m in members]
            canon = _majority(vals)
            if canon is not None:
                for m in members:
                    hs = m.get("headers")
                    if isinstance(hs, list) and j < len(hs) and hs[j].strip() != canon:
                        hs[j] = canon

        # 2) spanning-title groups, per position, then rebuild spans for every member
        def _cg(m: Dict[str, Any]) -> List[str]:
            cg = [""] * k
            for st_, ln_, lab in (m.get("spans") or []):
                for jj in range(st_, min(st_ + ln_, k)):
                    cg[jj] = lab
            return cg

        member_cgs = [_cg(m) for m in members]
        canon_cg = [(_majority([cg[j] for cg in member_cgs]) or "") for j in range(k)]
        # only rewrite spans if the reconciled groups differ from what a member has
        canon_spans = _col_groups_to_spans(canon_cg)
        for m, cg in zip(members, member_cgs):
            if cg != canon_cg:
                m["spans"] = [list(s) for s in canon_spans]


def _fallback_vlm_html(crop: Image.Image) -> str:
    """
    Last resort: run an unconstrained VLM read on the crop and convert whatever
    markdown/HTML table it emits. Better than nothing when structure detection
    fails, but offers no column-shift guarantee.
    """
    import ocr_olm

    try:
        raw, _ = ocr_olm._run_enhanced_ocr(crop, ocr_olm._get_fallback_html_prompt())
    except Exception as e:
        logger.warning("[table_extractor] fallback VLM read failed: %s", e)
        return ""

    rows = _html_to_rows(raw)
    if rows:
        k = max(len(r) for r in rows)
        headers = rows[0] if _looks_like_header(rows[0], rows[0]) else [f"Column {i+1}" for i in range(k)]
        data = rows[1:] if headers is rows[0] else rows
        return _rows_to_html(headers, _normalize_row_widths(data, len(headers)))

    # No HTML table — try markdown pipes.
    md_rows = _markdown_to_rows(raw)
    if md_rows:
        k = max(len(r) for r in md_rows)
        headers = md_rows[0] + [""] * (k - len(md_rows[0]))
        return _rows_to_html(headers, _normalize_row_widths(md_rows[1:], k))
    return ""


def _pipe_to_rows(raw: str) -> List[List[str]]:
    """
    Parse the compact band format: one row per line, cells separated by '|', e.g.
    "636|-112|-137|". Tolerates the model wrapping lines in leading/trailing pipes
    (markdown habit) and skips fences and separator lines. Trailing/spurious cells
    are fixed up afterwards by _normalize_row_widths.
    """
    rows: List[List[str]] = []
    for line in (raw or "").splitlines():
        line = line.strip()
        if not line or "|" not in line:
            continue
        if line.startswith("```"):
            continue
        if re.match(r"^[\s\-|:]+$", line):  # separator row like |---|---|
            continue
        cells = line.split("|")
        if line.startswith("|"):
            cells = cells[1:]  # drop the empty produced by a leading pipe
        # EMPTY is the placeholder the prompt forces for blank cells — without it the
        # model omits blanks entirely and every later value shifts left.
        rows.append(["" if c.strip().upper() == "EMPTY" else c.strip() for c in cells])
    return rows


def _markdown_to_rows(raw: str) -> List[List[str]]:
    """Parse simple pipe-delimited markdown table rows (skips separator lines)."""
    rows = []
    for line in (raw or "").splitlines():
        line = line.strip()
        if not line.startswith("|"):
            continue
        if re.match(r"^\|[\s:\-|]+\|?$", line):  # separator row |---|---|
            continue
        cells = [c.strip() for c in line.strip("|").split("|")]
        rows.append(cells)
    return rows
