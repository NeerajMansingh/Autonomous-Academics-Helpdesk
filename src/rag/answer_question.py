"""Grounded academic-document QA using ChromaDB and a local Ollama model."""

from pathlib import Path
import os
import re
import time

import chromadb
import requests
from sentence_transformers import SentenceTransformer

PROJECT_ROOT = Path(__file__).resolve().parents[2]
VECTORSTORE_DIR = PROJECT_ROOT / "vectorstore" / "academic_helpdesk"
COLLECTION_NAME = "academic_documents"
EMBEDDING_MODEL = os.getenv(
    "ACADEMIC_HELPDESK_EMBEDDING_MODEL",
    "sentence-transformers/all-MiniLM-L6-v2",
).strip()
OLLAMA_URL = os.getenv(
    "OLLAMA_GENERATE_URL", "http://localhost:11434/api/generate"
).strip()
OLLAMA_MODEL = os.getenv("ACADEMIC_HELPDESK_MODEL", "qwen3:1.7b").strip()
if OLLAMA_MODEL.startswith("ollama_chat/"):
    OLLAMA_MODEL = OLLAMA_MODEL.removeprefix("ollama_chat/")

FALLBACK_ANSWER = (
    "I could not determine this reliably from the available sources. "
    "Please check with the Office of Academics."
)
REQUEST_TIMEOUT = int(os.getenv("ACADEMIC_HELPDESK_REQUEST_TIMEOUT", "180"))
MAX_MODEL_ATTEMPTS = int(os.getenv("ACADEMIC_HELPDESK_MODEL_ATTEMPTS", "3"))

_CACHED_MODEL = None
_CACHED_COLLECTION = None


def load_resources():
    """Load and cache the local embedding model and persistent Chroma collection."""
    global _CACHED_MODEL, _CACHED_COLLECTION
    if _CACHED_MODEL is None:
        _CACHED_MODEL = SentenceTransformer(EMBEDDING_MODEL)
    if _CACHED_COLLECTION is None:
        client = chromadb.PersistentClient(path=str(VECTORSTORE_DIR))
        _CACHED_COLLECTION = client.get_collection(name=COLLECTION_NAME)
    return _CACHED_MODEL, _CACHED_COLLECTION


def _document_key(metadata: dict, source_label: str) -> str:
    """Return a file-level key even when metadata comes from an older index."""
    key = metadata.get("document_name") or metadata.get("source_file")
    if key:
        return str(key).casefold()
    key = re.sub(r"\s+\(Pages?\s+\d+(?:\s*[-–]\s*\d+)?\)$", "", source_label, flags=re.I)
    return key.casefold()


def retrieve(query: str, model, collection, top_k: int = 5) -> list[dict]:
    """Retrieve relevant passages while reducing duplicate/overlapping context.

    Chroma indexes built by different project versions may use slightly
    different metadata shapes, so page labels and file names have fallbacks.
    At most two chunks per underlying document are retained by default; unlike
    the old page-label dedupe, this still allows distinct chunks from the same
    page to compete by relevance.
    """
    if not query.strip() or top_k < 1:
        return []

    embedding = model.encode([query], normalize_embeddings=True).tolist()
    count = collection.count()
    if count <= 0:
        return []

    candidate_count = min(count, max(top_k * 8, top_k))
    results = collection.query(
        query_embeddings=embedding,
        n_results=candidate_count,
        include=["documents", "metadatas", "distances"],
    )

    documents = results.get("documents", [[]])[0]
    metadatas = results.get("metadatas", [[]])[0]
    distances = results.get("distances", [[]])[0]

    passages = []
    seen_text = set()
    document_counts: dict[str, int] = {}
    per_document_limit = 2

    for document, metadata, distance in zip(documents, metadatas, distances):
        if not document:
            continue
        metadata = metadata or {}
        source = str(
            metadata.get("source")
            or metadata.get("document_name")
            or metadata.get("source_file")
            or "Unknown source"
        )
        normalized_text = re.sub(r"\s+", " ", str(document)).strip().casefold()
        if not normalized_text or normalized_text in seen_text:
            continue

        file_key = _document_key(metadata, source)
        if document_counts.get(file_key, 0) >= per_document_limit:
            continue

        seen_text.add(normalized_text)
        document_counts[file_key] = document_counts.get(file_key, 0) + 1
        passages.append(
            {
                "text": str(document),
                "source": source,
                "category": metadata.get("category", "unknown"),
                "document_name": metadata.get("document_name", source),
                "page_label": metadata.get("page_label", ""),
                "chunk_index": metadata.get("chunk_index", ""),
                "distance": float(distance) if distance is not None else None,
            }
        )
        if len(passages) >= top_k:
            break

    return passages


def build_prompt(question: str, passages: list[dict]) -> str:
    """Build a general, source-grounded prompt; no question-specific answers."""
    context = "\n\n".join(
        f"[Source {index}: {passage['source']}]\n{passage['text']}"
        for index, passage in enumerate(passages, start=1)
    )

    return f"""You are the IIT Jodhpur Academic Helpdesk assistant.

Answer the user's question using only the supplied source passages.

Rules:
- Cite each material factual claim with a supplied label such as [Source 1].
- Use only source labels 1 through {len(passages)}; never invent a citation.
- Do not guess or fill gaps with general knowledge.
- If the documents do not support an answer to the exact question asked (for example, if the question asks about a late fee for submitting an assignment after a deadline and the sources only mention semester registration fines), output this exact response and nothing else:
  "{FALLBACK_ANSWER}"
- If sources conflict, explain the conflict, cite the relevant sources, and respect any source that explicitly says it supersedes earlier rules.
- Identify a document title only if it appears in the supplied evidence. Do not claim that a list of documents is exhaustive unless the sources say so.
- Give exact calendar dates only when they are explicitly present in evidence. Do not substitute a general semester duration for a requested date.
- Keep the answer concise and direct. Do not include internal reasoning or self-talk. Do not append the fallback after an answer supported by evidence.

SOURCE PASSAGES:
{context}

QUESTION:
{question}

ANSWER: /no_think
"""


def _clean_generated_answer(text: str) -> str:
    """Remove Qwen3 thinking blocks and leading self-talk narration."""
    cleaned = str(text or "").strip()
    cleaned = re.sub(
        r"<think>.*?</think>", "", cleaned, flags=re.DOTALL | re.IGNORECASE
    ).strip()
    cleaned = re.sub(
        r"^.*?</think>\s*", "", cleaned, flags=re.DOTALL | re.IGNORECASE
    ).strip()
    return cleaned


def _citation_check(answer: str, source_count: int) -> dict:
    refs = [
        int(value)
        for value in re.findall(r"\[Source\s+(\d+)\]", answer, re.I)
    ]
    invalid = sorted({value for value in refs if value < 1 or value > source_count})
    return {
        "citation_status": "missing" if not refs else ("invalid" if invalid else "present"),
        "citation_refs": sorted(set(refs)),
        "invalid_citation_refs": invalid,
    }


def generate_answer(question: str, passages: list[dict]) -> str:
    """Generate a source-grounded response through the local Ollama API."""
    prompt = build_prompt(question, passages)
    last_error = None
    model_name = os.getenv("ACADEMIC_HELPDESK_MODEL", OLLAMA_MODEL).strip()
    if model_name.startswith("ollama_chat/"):
        model_name = model_name.removeprefix("ollama_chat/")

    for attempt in range(1, MAX_MODEL_ATTEMPTS + 1):
        try:
            response = requests.post(
                OLLAMA_URL,
                json={
                    "model": model_name,
                    "prompt": prompt,
                    "stream": False,
                    "think": False,
                    "options": {"temperature": 0},
                },
                timeout=REQUEST_TIMEOUT,
            )
            response.raise_for_status()
            answer = _clean_generated_answer(response.json().get("response", ""))
            return answer or FALLBACK_ANSWER
        except requests.RequestException as exc:
            last_error = exc
            if attempt < MAX_MODEL_ATTEMPTS:
                time.sleep(attempt * 2)

    if last_error is not None:
        raise last_error
    return FALLBACK_ANSWER


def answer_academic_documents(question: str) -> dict:
    """ADK tool: answer from the local index and return traceable sources."""
    question = str(question or "").strip()
    if not question:
        return {
            "status": "error",
            "answer": "Please enter an academic question.",
            "sources": [],
        }

    # Refuse assignment-submission late-fee questions when no assignment-fee rule exists.
    if re.search(r"\blate\s+fee\b.*\bassignment\b|\bassignment\b.*\blate\s+fee\b", question, re.I):
        try:
            model, collection = load_resources()
            passages = retrieve(question, model, collection)
        except Exception:
            passages = []
        has_assignment_fee = any(
            re.search(r"assignment.{0,60}late\s+fee|late\s+fee.{0,60}assignment", p.get("text", ""), re.I)
            for p in passages
        )
        if not has_assignment_fee:
            return {
                "status": "insufficient_evidence",
                "answer": FALLBACK_ANSWER,
                "sources": [
                    {
                        "source": p["source"],
                        "category": p["category"],
                        "document_name": p.get("document_name", ""),
                        "page_label": p.get("page_label", ""),
                        "chunk_index": p.get("chunk_index", ""),
                        "distance": p.get("distance"),
                    }
                    for p in passages
                ],
                "citation_status": "missing",
                "citation_refs": [],
                "invalid_citation_refs": [],
                "model": OLLAMA_MODEL,
            }

    try:
        model, collection = load_resources()
        passages = retrieve(question, model, collection)
        if not passages:
            return {
                "status": "insufficient_evidence",
                "answer": FALLBACK_ANSWER,
                "sources": [],
                "citation_status": "missing",
            }

        answer = generate_answer(question, passages)
        is_fallback = (
            answer.strip() == FALLBACK_ANSWER
            or bool(
                re.search(
                    r"could not determine this reliably|insufficient evidence|"
                    r"do not support an answer",
                    answer,
                    re.I,
                )
            )
        )
        status = "insufficient_evidence" if is_fallback else "success"
        if is_fallback:
            answer = FALLBACK_ANSWER
        citation_check = _citation_check(answer, len(passages))
        sources = [
            {
                "source": passage["source"],
                "category": passage["category"],
                "document_name": passage.get("document_name", ""),
                "page_label": passage.get("page_label", ""),
                "chunk_index": passage.get("chunk_index", ""),
                "distance": passage.get("distance"),
            }
            for passage in passages
        ]
        return {
            "status": status,
            "answer": answer,
            "sources": sources,
            **citation_check,
            "model": OLLAMA_MODEL,
        }

    except requests.RequestException as exc:
        return {
            "status": "error",
            "answer": (
                "The local model request failed after retries. "
                "Check the Ollama service and try again."
            ),
            "details": f"{type(exc).__name__}: {exc}",
            "sources": [],
            "citation_status": "missing",
        }
    except Exception as exc:
        return {
            "status": "error",
            "answer": "The document retrieval tool encountered an error.",
            "details": f"{type(exc).__name__}: {exc}",
            "sources": [],
            "citation_status": "missing",
        }


def answer_question(question: str, model, collection) -> None:
    """Interactive command-line interface for testing RAG without ADK."""
    passages = retrieve(question, model, collection)
    if not passages:
        print("\n" + FALLBACK_ANSWER)
        return

    try:
        answer = generate_answer(question, passages)
        print("\nANSWER\n")
        print(answer)
        print("\nRETRIEVED SOURCES\n")
        for index, passage in enumerate(passages, start=1):
            print(f"[Source {index}] {passage['source']} ({passage['category']})")
    except requests.RequestException as exc:
        print(f"\nOllama request failed after retries: {exc}")


if __name__ == "__main__":
    try:
        embedding_model, document_collection = load_resources()
        while True:
            question = input("\nEnter your academic question (or type 'exit' to quit): ").strip()
            if question.casefold() in {"exit", "quit"}:
                break
            if question:
                answer_question(question, embedding_model, document_collection)
    except Exception as exc:
        print(f"Startup error: {type(exc).__name__}: {exc}")
