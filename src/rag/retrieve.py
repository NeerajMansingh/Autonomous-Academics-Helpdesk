from pathlib import Path
import os

import chromadb
from sentence_transformers import SentenceTransformer


PROJECT_ROOT = Path(__file__).resolve().parents[2]
VECTORSTORE_DIR = PROJECT_ROOT / "vectorstore" / "academic_helpdesk"
COLLECTION_NAME = "academic_documents"
EMBEDDING_MODEL = os.getenv(
    "ACADEMIC_HELPDESK_EMBEDDING_MODEL",
    "sentence-transformers/all-MiniLM-L6-v2",
).strip()


def retrieve(query: str, top_k: int = 5, verbose: bool = False) -> list[dict]:
    """Retrieve the top-k matching chunks from the Chroma vector store."""
    model = SentenceTransformer(EMBEDDING_MODEL)

    client = chromadb.PersistentClient(path=str(VECTORSTORE_DIR))
    collection = client.get_collection(COLLECTION_NAME)

    query_embedding = model.encode(
        [query],
        normalize_embeddings=True,
    ).tolist()

    results = collection.query(
        query_embeddings=query_embedding,
        n_results=top_k,
        include=["documents", "metadatas", "distances"],
    )

    passages = []
    for i, (document, metadata, distance) in enumerate(
        zip(
            results["documents"][0],
            results["metadatas"][0],
            results["distances"][0],
        ),
        start=1,
    ):
        metadata = metadata or {}
        source = metadata.get("source") or metadata.get("document_name") or "Unknown source"
        entry = {
            "rank": i,
            "text": str(document),
            "source": str(source),
            "category": str(metadata.get("category", "unknown")),
            "distance": float(distance) if distance is not None else None,
        }
        passages.append(entry)
        if verbose:
            print(f"\n--- Result {i} ---")
            print(f"Source: {entry['source']}")
            print(f"Category: {entry['category']}")
            if entry["distance"] is not None:
                print(f"Distance: {entry['distance']:.4f}")
            print(document[:700])

    return passages


if __name__ == "__main__":
    question = input("Enter an academic question: ").strip()

    if question:
        retrieve(question, verbose=True)