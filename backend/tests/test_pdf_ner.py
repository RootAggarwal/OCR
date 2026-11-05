import sys
import os
import tempfile

from PyPDF2 import PdfReader, PdfWriter

from ocr_olm import extract_text_from_pdf
from ner.extractor import extract_entities


pdf_path = sys.argv[1]
MAX_PAGES = 10


# Create temporary PDF containing first 10 pages
reader = PdfReader(pdf_path)
writer = PdfWriter()

for page in reader.pages[:MAX_PAGES]:
    writer.add_page(page)

temp_pdf = tempfile.NamedTemporaryFile(
    suffix=".pdf",
    delete=False
).name

with open(temp_pdf, "wb") as f:
    writer.write(f)

print(f"[TEST] Running OCR + NER on first {MAX_PAGES} pages")


try:
    # Existing OCR pipeline
    result = extract_text_from_pdf(temp_pdf, mode="single")

    if not result.get("success"):
        print("OCR failed:")
        print(result.get("error"))
        sys.exit(1)

    # NER
    for page in result["pages"]:
        page_number = page.get("page", "?")
        text = page.get("text", "")

        if not text.strip():
            continue

        entities = extract_entities(text)

        print(f"\n--- Page {page_number} ---")

        if not entities:
            print("No entities found")
            continue

        for entity in entities:
            print(
                f"{entity['text']} -> "
                f"{entity['label']} "
                f"(score={entity['score']:.3f})"
            )

finally:
    os.remove(temp_pdf)