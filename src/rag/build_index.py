from pathlib import Path
import os
import re

import chromadb
from sentence_transformers import SentenceTransformer


PROJECT_ROOT = Path(__file__).resolve().parents[2]
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
VECTORSTORE_DIR = PROJECT_ROOT / "vectorstore" / "academic_helpdesk"
COLLECTION_NAME = "academic_documents"
EMBEDDING_MODEL = os.getenv(
    "ACADEMIC_HELPDESK_EMBEDDING_MODEL",
    "sentence-transformers/all-MiniLM-L6-v2",
).strip()

CHUNK_SIZE = 1200
CHUNK_OVERLAP = 200
_PAGE_MARKER_RE = re.compile(r"^--- Page (\d+) ---$", re.MULTILINE)


def split_text_with_pages(text: str) -> list[dict]:
    """Split text into overlapping character chunks while tracking page numbers."""
    text = text.strip()
    if not text:
        return []

    # Map character offsets to page numbers if --- Page N --- markers exist.
    markers = list(_PAGE_MARKER_RE.finditer(text))
    if not markers:
        chunks = []
        start = 0
        while start < len(text):
            end = min(start + CHUNK_SIZE, len(text))
            chunk = text[start:end].strip()
            if chunk:
                chunks.append(
                    {
                        "text": chunk,
                        "page_start": 1,
                        "page_end": 1,
                        "page_label": "Page 1",
                    }
                )
            if end == len(text):
                break
            start = end - CHUNK_OVERLAP
        return chunks

    # Build clean text without raw marker lines while keeping page spans.
    segments = []
    clean_parts = []
    cursor = 0
    for idx, match in enumerate(markers):
        page_num = int(match.group(1))
        seg_start = match.end()
        seg_end = markers[idx + 1].start() if idx + 1 < len(markers) else len(text)
        seg_text = text[seg_start:seg_end].strip()
        if not seg_text:
            continue
        if clean_parts:
            clean_parts.append("\n\n")
            cursor += 2
        start_offset = cursor
        clean_parts.append(seg_text)
        cursor += len(seg_text)
        segments.append((start_offset, cursor, page_num))

    combined = "".join(clean_parts)
    if not combined.strip():
        return []

    chunks = []
    start = 0
    while start < len(combined):
        end = min(start + CHUNK_SIZE, len(combined))
        chunk = combined[start:end].strip()
        if chunk:
            pages = [
                page_num
                for seg_start, seg_end, page_num in segments
                if seg_end > start and seg_start < end
            ]
            p_start = min(pages) if pages else 1
            p_end = max(pages) if pages else p_start
            p_label = f"Page {p_start}" if p_start == p_end else f"Pages {p_start}-{p_end}"
            chunks.append(
                {
                    "text": chunk,
                    "page_start": p_start,
                    "page_end": p_end,
                    "page_label": p_label,
                }
            )
        if end == len(combined):
            break
        start = end - CHUNK_OVERLAP

    return chunks


def split_text(text: str) -> list[str]:
    """Split text into overlapping character chunks."""
    return [item["text"] for item in split_text_with_pages(text)]


def main() -> None:
    if not PROCESSED_DIR.exists():
        raise FileNotFoundError(
            f"Processed text directory not found: {PROCESSED_DIR}"
        )

    text_files = sorted(PROCESSED_DIR.rglob("*.txt"))
    if not text_files:
        raise RuntimeError(f"No extracted text files found in {PROCESSED_DIR}")

    print(f"Text files found: {len(text_files)}")
    print("Loading embedding model...")
    model = SentenceTransformer(EMBEDDING_MODEL)

    client = chromadb.PersistentClient(path=str(VECTORSTORE_DIR))

    # Rebuild this collection so rerunning the script does not add duplicates.
    try:
        client.delete_collection(COLLECTION_NAME)
    except Exception:
        pass

    collection = client.create_collection(
        name=COLLECTION_NAME,
        metadata={"hnsw:space": "cosine"},
    )

    all_ids = []
    all_documents = []
    all_metadatas = []

    for file_path in text_files:
        text = file_path.read_text(encoding="utf-8", errors="replace")
        chunks = split_text_with_pages(text)

        source_file = file_path.name
        document_name = f"{file_path.stem}.pdf"
        category = file_path.parent.name

        for index, chunk_info in enumerate(chunks):
            chunk_id = f"{file_path.relative_to(PROCESSED_DIR).as_posix()}::{index}"
            page_label = chunk_info["page_label"]
            source_label = f"{document_name} ({page_label})"

            all_ids.append(chunk_id)
            all_documents.append(chunk_info["text"])
            all_metadatas.append(
                {
                    "source": source_label,
                    "document_name": document_name,
                    "source_file": source_file,
                    "source_path": str(file_path),
                    "category": category,
                    "page_label": page_label,
                    "page_start": int(chunk_info["page_start"]),
                    "page_end": int(chunk_info["page_end"]),
                    "chunk_index": index,
                }
            )

    if not all_documents:
        raise RuntimeError("No non-empty text chunks were created.")

    batch_size = 64

    for start in range(0, len(all_documents), batch_size):
        end = start + batch_size
        docs_batch = all_documents[start:end]

        embeddings = model.encode(
            docs_batch,
            normalize_embeddings=True,
            show_progress_bar=False,
        ).tolist()

        collection.add(
            ids=all_ids[start:end],
            documents=docs_batch,
            metadatas=all_metadatas[start:end],
            embeddings=embeddings,
        )

        print(f"Indexed {min(end, len(all_documents))}/{len(all_documents)} chunks")

    print("\nIndex build complete.")
    print(f"Chunks indexed: {collection.count()}")
    print(f"Vector database: {VECTORSTORE_DIR}")


if __name__ == "__main__":
    main()
