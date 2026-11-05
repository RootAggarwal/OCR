"""
backfill_page_tables.py — recover table HTML that predates the page_tables store.

Documents ingested before the tables lane existed never had their table HTML persisted:
ingest consumed only `prose` and `structured`, so the `tables` list that extract_page_html
had already computed was dropped. This recovers what is recoverable, and is explicit about
what is not.

WHAT IT CAN RECOVER
  Tables that ended up inside prose_chunks.content. That happens on any page where TATR
  detected no table: the whole page goes through the VLM, which emits HTML tables natively.
  On this corpus that is the majority of tabular pages.

WHAT IT CANNOT RECOVER — and must not fake
  Pages where TATR DID detect a table. There the table region was masked out of the image
  before the prose pass (ocr_olm._process_page_structured), so the table's text is not in
  prose_chunks at all. The only surviving representation is table_cells, which structurally
  cannot produce the target format: body rowspans are filtered out in _header_spans before
  `spans` is built, `blocks` is a dense grid by construction, and <br> multi-values never
  reach it. Re-rendering those would produce a table that looks authoritative and has
  quietly lost its row groupings — worse than having no row at all. Those pages are SKIPPED
  and listed as RE-INGEST REQUIRED; re-running them through the current capture path fixes
  them properly, and ingest is idempotent on content_hash so it replaces rather than
  duplicates.

  Also unrecoverable: rows the OCR never transcribed (a page whose HTML stops mid-<td>).
  Those are stored with well_formed=False so the gap becomes a visible warning instead of a
  silent undercount.

PAGE-LEVEL, NOT CHUNK-LEVEL
  ingest._chunk_prose splits page prose on blank lines, which cuts tables in half. Harvest
  per chunk and you invent fragments and miss any table that straddles a boundary. This
  reassembles each page's chunks in chunk_id order (== insertion order == reading order)
  before harvesting, so table_index matches what a fresh ingest would assign.

Run:  python -m rag.backfill_page_tables            (from backend/)
      python -m rag.backfill_page_tables --dry-run
"""

import sys
from typing import Dict, List, Tuple

from . import db


def _ready_docs() -> List[Tuple[int, str]]:
    _, rows = db.run_readonly(
        "SELECT doc_id, filename FROM documents WHERE status='ready' ORDER BY doc_id")
    return [(int(r[0]), r[1]) for r in rows]


def _page_texts(doc_id: int) -> List[Tuple[int, str]]:
    """Each page's prose reassembled in reading order. See PAGE-LEVEL note above."""
    _, rows = db.run_readonly(
        "SELECT page, string_agg(content, E'\\n\\n' ORDER BY chunk_id) "
        "FROM prose_chunks WHERE doc_id = %s GROUP BY page ORDER BY page",
        params=(doc_id,))
    return [(int(r[0]), r[1] or "") for r in rows]


def _cell_pages(doc_id: int) -> set:
    """Pages that went through the TATR/masked branch — their table text is not in prose."""
    _, rows = db.run_readonly(
        "SELECT DISTINCT page FROM table_cells WHERE doc_id = %s", params=(doc_id,))
    return {int(r[0]) for r in rows}


def backfill(dry_run: bool = False) -> int:
    if not db.available():
        print("[backfill_page_tables] no database configured")
        return 0

    import table_extractor

    total = 0
    reingest: List[str] = []

    for doc_id, filename in _ready_docs():
        pages = _page_texts(doc_id)
        masked = _cell_pages(doc_id)
        found = 0
        malformed = 0

        for page, text in pages:
            records = table_extractor.harvest_inline_tables(text)
            if not records:
                continue
            for ti, t in enumerate(records):
                if not t.get("well_formed", True):
                    malformed += 1
                found += 1
                if dry_run:
                    continue
                with db.connect() as conn:
                    with conn.cursor() as cur:
                        # ON CONFLICT so the script is idempotent and safe to re-run after
                        # a partial pass; a real re-ingest replaces via the FK cascade.
                        cur.execute(
                            "INSERT INTO page_tables "
                            "(doc_id,page,table_index,html,digest,caption,n_rows,n_cols,"
                            " est_tokens,well_formed,source) "
                            "VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s) "
                            "ON CONFLICT (doc_id,page,table_index) DO UPDATE SET "
                            "  html=EXCLUDED.html, digest=EXCLUDED.digest, "
                            "  caption=EXCLUDED.caption, n_rows=EXCLUDED.n_rows, "
                            "  n_cols=EXCLUDED.n_cols, est_tokens=EXCLUDED.est_tokens, "
                            "  well_formed=EXCLUDED.well_formed, source=EXCLUDED.source, "
                            "  embedding=NULL",
                            (doc_id, page, ti, t["html"], t.get("digest"), t.get("caption"),
                             t.get("n_rows"), t.get("n_cols"), t.get("est_tokens"),
                             bool(t.get("well_formed", True)), "backfill_prose_html"))
                    conn.commit()

        # Pages with structured cells but no prose-recoverable table: the masked branch.
        harvested_pages = {p for p, txt in pages
                           if table_extractor.harvest_inline_tables(txt)}
        for p in sorted(masked - harvested_pages):
            reingest.append(f"{filename} p{p}")

        total += found
        print(f"  {filename:<44} recovered {found:>3} table(s)"
              + (f"  ({malformed} incomplete OCR read(s))" if malformed else ""))

    if reingest:
        print()
        print("RE-INGEST REQUIRED — these pages had their tables masked out of the prose")
        print("pass, so the HTML does not exist anywhere and CANNOT be recovered here.")
        print("Re-upload the document (idempotent by content_hash; the original bytes are")
        print("already in rag/pdf_store) to capture them through the current path:")
        for r in reingest:
            print(f"    {r}")

    if not dry_run and total:
        print()
        try:
            from . import table_search
            n = table_search.backfill()
            print(f"[backfill_page_tables] embedded {n} digest(s)")
        except Exception as e:
            print(f"[backfill_page_tables] embedding skipped ({e}) — "
                  f"run `python -m rag.table_search` once the embedder is available")

    print(f"\n[backfill_page_tables] {'would recover' if dry_run else 'recovered'} "
          f"{total} table(s)")
    return total


if __name__ == "__main__":
    backfill(dry_run="--dry-run" in sys.argv)
