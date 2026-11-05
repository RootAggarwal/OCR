"""
ingest_all_pdf_store.py — Ingest all PDFs from C:/OCR-data/pdf_store into the RAG database.
"""

import os
import re
import sys
import hashlib
import fitz  # PyMuPDF
from typing import Dict, List, Any

# Add backend directory to sys.path
backend_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if backend_dir not in sys.path:
    sys.path.insert(0, backend_dir)

from rag import db, figures
from rag.ingest import _chunk_prose

PDF_STORE_DIR = os.getenv("PDF_STORE_DIR", "C:/OCR-data/pdf_store")


def extract_title_and_line_name(doc, filename):
    line_name = None
    # Check filename
    m = re.search(r"([A-Z]{1,3}\d{1,3}-\d{1,3})", filename or "", re.IGNORECASE)
    if m:
        line_name = m.group(1).upper()

    # Check first 2 pages text
    first_pages_text = ""
    for p in range(min(2, len(doc))):
        first_pages_text += doc[p].get_text() + "\n"

    # Common patterns: KANJIRANGUDI # 17, KARAIKAL # 9, etc.
    well_match = re.search(r"([A-Z\s]{3,20}\s*#\s*[0-9A-Za-z\s\-]+(?:\([A-Z0-9]+\))?)", first_pages_text)
    if well_match and not line_name:
        line_name = well_match.group(1).strip().replace("\n", " ")

    if not line_name:
        # Search for Annual Report or Lecture or Title
        report_match = re.search(r"(GEOPIC Annual Report [0-9\-]+|Computer Networks|OIL\s*[&6]\s*NATURAL GAS|OOPS by Arpit)", first_pages_text, re.IGNORECASE)
        if report_match:
            line_name = report_match.group(1).strip()

    return line_name


def table_to_html(table_data):
    """Convert a 2D list of cells to an HTML table string."""
    if not table_data or len(table_data) < 2:
        return ""
    headers = table_data[0]
    rows = table_data[1:]
    
    html = '<table class="ocr-dataframe">\n<thead>\n<tr>\n'
    for h in headers:
        html += f'  <th>{h or ""}</th>\n'
    html += '</tr>\n</thead>\n<tbody>\n'
    for r in rows:
        html += '<tr>\n'
        for cell in r:
            html += f'  <td>{cell or ""}</td>\n'
        html += '</tr>\n'
    html += '</tbody>\n</table>'
    return html


def ingest_pdf_file(filepath: str):
    filename = os.path.basename(filepath)
    with open(filepath, "rb") as f:
        doc_bytes = f.read()

    content_hash = hashlib.sha256(doc_bytes).hexdigest()
    doc = fitz.open(filepath)
    n_pages = len(doc)
    line_name = extract_title_and_line_name(doc, filename)

    print(f"\nProcessing {filename} ({n_pages} pages)... line_name='{line_name}'")

    with db.connect() as conn:
        with conn.cursor() as cur:
            # Check if document already exists
            cur.execute("SELECT doc_id, status FROM documents WHERE content_hash = %s", (content_hash,))
            existing = cur.fetchone()
            if existing:
                print(f"  -> Already in database as doc_id={existing[0]} (status={existing[1]}). Skipping.")
                return existing[0]

            # Insert document
            cur.execute(
                "INSERT INTO documents (filename, line_name, content_hash, status, n_pages) "
                "VALUES (%s, %s, %s, 'ingesting', %s) RETURNING doc_id",
                (filename, line_name, content_hash, n_pages),
            )
            doc_id = cur.fetchone()[0]

            prose_rows = []
            table_rows = []

            for page_idx in range(n_pages):
                page_num = page_idx + 1
                page = doc[page_idx]

                # 1. Text & Prose
                text = page.get_text() or ""
                chunks = _chunk_prose(text)
                for chunk in chunks:
                    if len(chunk.strip()) > 5:
                        prose_rows.append((doc_id, page_num, None, chunk.strip()))

                # 2. Tables
                try:
                    tables = page.find_tables()
                    for ti, tbl in enumerate(tables.tables):
                        extracted = tbl.extract()
                        if extracted and len(extracted) >= 2:
                            n_rows = len(extracted) - 1
                            n_cols = len(extracted[0]) if extracted else 0
                            html = table_to_html(extracted)
                            headers_str = ", ".join([str(h) for h in extracted[0] if h])
                            digest = f"Table on page {page_num} | Columns: {headers_str[:120]}"
                            est_tokens = len(html) // 4
                            table_rows.append((
                                doc_id, page_num, ti, html, digest, None,
                                n_rows, n_cols, est_tokens, True, "pymupdf"
                            ))
                except Exception as te:
                    pass

            # Insert prose chunks
            if prose_rows:
                cur.executemany(
                    "INSERT INTO prose_chunks (doc_id, page, section, content) "
                    "VALUES (%s, %s, %s, %s)", prose_rows)

            # Insert tables
            if table_rows:
                cols = ("doc_id,page,table_index,html,digest,caption,n_rows,n_cols,"
                        "est_tokens,well_formed,source")
                cur.executemany(
                    f"INSERT INTO page_tables ({cols}) "
                    "VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)", table_rows)

            # Extract figures
            try:
                extracted_figures = figures.extract_figures(doc_bytes, content_hash)
                figure_rows = []
                for fig in extracted_figures:
                    try:
                        figures.persist_figure(fig)
                        figure_rows.append((
                            doc_id, fig.page, list(fig.bbox), fig.caption, fig.context,
                            fig.image_ref, fig.phash, fig.width_px, fig.height_px,
                        ))
                    except Exception:
                        pass
                if figure_rows:
                    cur.executemany(
                        "INSERT INTO figures "
                        "(doc_id,page,bbox,caption,context,image_ref,phash,width_px,height_px) "
                        "VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s)", figure_rows)
            except Exception as fe:
                pass

            # Mark ready
            cur.execute("UPDATE documents SET status='ready' WHERE doc_id=%s", (doc_id,))
            conn.commit()
            print(f"  -> Ingested doc_id={doc_id}: {len(prose_rows)} prose chunks, {len(table_rows)} tables.")
            return doc_id


def main():
    if not os.path.isdir(PDF_STORE_DIR):
        print(f"PDF store dir not found: {PDF_STORE_DIR}")
        return

    files = sorted([f for f in os.listdir(PDF_STORE_DIR) if f.lower().endswith(".pdf")])
    print(f"Found {len(files)} PDFs in {PDF_STORE_DIR}. Starting ingestion...")

    count = 0
    for f in files:
        filepath = os.path.join(PDF_STORE_DIR, f)
        try:
            doc_id = ingest_pdf_file(filepath)
            if doc_id:
                count += 1
        except Exception as e:
            print(f"Error ingesting {f}: {e}")

    print(f"\nAll done! Successfully verified/ingested {count} documents.")


if __name__ == "__main__":
    main()
