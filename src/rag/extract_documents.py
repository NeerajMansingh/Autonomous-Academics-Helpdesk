
from pathlib import Path
import csv
import re

from pypdf import PdfReader


PROJECT_ROOT = Path(__file__).resolve().parents[2]

SOURCE_DIR = (
    PROJECT_ROOT
    / "data"
    / "academic-helpdesk-data"
    / "academic-helpdesk"
    / "data"
    / "documents"
)

OUTPUT_DIR = PROJECT_ROOT / "data" / "processed"
REPORT_PATH = PROJECT_ROOT / "reports" / "extraction_report.csv"

MIN_TEXT_CHARACTERS = 100


def clean_extracted_text(text: str) -> str:
    """Normalize whitespace while preserving page markers."""
    text = text.replace("\x00", "")

    # Keep page markers on separate lines.
    pages = re.split(r"(\n?--- Page \d+ ---\n?)", text)
    cleaned_parts = []

    for part in pages:
        if re.fullmatch(r"\n?--- Page \d+ ---\n?", part):
            cleaned_parts.append(part.strip())
            continue

        # Join words separated by layout whitespace.
        part = re.sub(r"[ \t]+", " ", part)
        part = re.sub(r"\s*\n\s*", " ", part)
        part = re.sub(r" {2,}", " ", part)
        cleaned_parts.append(part.strip())

    return "\n".join(p for p in cleaned_parts if p)


def normalize_text(text: str) -> str:
    """Clean whitespace while preserving paragraph boundaries."""
    text = text.replace("\x00", "")
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n[ \t]*\n+", "\n\n", text)
    return text.strip()


def read_existing_text(txt_path: Path) -> str:
    """Read an existing extracted-text file."""
    try:
        return txt_path.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return ""


def extract_pdf_text(pdf_path: Path) -> str:
    """Extract selectable text from a PDF."""
    reader = PdfReader(str(pdf_path))
    pages = []

    for page_number, page in enumerate(reader.pages, start=1):
        page_text = page.extract_text() or ""
        if page_text.strip():
            pages.append(f"--- Page {page_number} ---\n{page_text}")

    return "\n\n".join(pages)


def process_pdf(pdf_path: Path) -> dict:
    """Process one PDF and return a report row."""
    relative_path = pdf_path.relative_to(SOURCE_DIR)
    category = relative_path.parts[0] if len(relative_path.parts) > 1 else "other"

    output_subdir = OUTPUT_DIR / category
    output_subdir.mkdir(parents=True, exist_ok=True)

    # Keep the original filename but change the extension to .txt.
    output_path = output_subdir / f"{pdf_path.stem}.txt"
    existing_txt = Path(str(pdf_path) + ".txt")

    source_used = ""
    error = ""

    try:
        # Prefer the existing sidecar if it contains usable text.
        existing_text = ""
        if existing_txt.is_file():
            existing_text = normalize_text(read_existing_text(existing_txt))

        if len(existing_text) >= MIN_TEXT_CHARACTERS:
            text = clean_extracted_text(existing_text)
            source_used = "existing_text_file"
        else:
            text = clean_extracted_text(extract_pdf_text(pdf_path))
            source_used = "pdf_text_extraction"
        if len(text) >= MIN_TEXT_CHARACTERS:
            output_path.write_text(text, encoding="utf-8")
            status = "success"
        else:
            status = "needs_review"

    except Exception as exc:
        text = ""
        status = "error"
        error = str(exc)

    return {
        "category": category,
        "filename": pdf_path.name,
        "source_path": str(pdf_path),
        "source_used": source_used,
        "characters": len(text),
        "status": status,
        "output_path": str(output_path) if status == "success" else "",
        "error": error,
    }


def main() -> None:
    if not SOURCE_DIR.is_dir():
        raise FileNotFoundError(f"Document source directory not found: {SOURCE_DIR}")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)

    pdf_files = sorted(SOURCE_DIR.rglob("*.pdf"))
    if not pdf_files:
        print(f"No PDFs found under: {SOURCE_DIR}")
        return

    results = [process_pdf(pdf_path) for pdf_path in pdf_files]

    fieldnames = [
        "category",
        "filename",
        "source_path",
        "source_used",
        "characters",
        "status",
        "output_path",
        "error",
    ]

    with REPORT_PATH.open("w", newline="", encoding="utf-8-sig") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(results)

    print(f"PDFs processed: {len(results)}")
    for status in sorted({row['status'] for row in results}):
        count = sum(row["status"] == status for row in results)
        print(f"{status}: {count}")

    print(f"Extracted text directory: {OUTPUT_DIR}")
    print(f"Extraction report: {REPORT_PATH}")


if __name__ == "__main__":
    main()
