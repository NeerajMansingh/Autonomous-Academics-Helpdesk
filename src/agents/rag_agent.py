import os

from google.adk.agents import Agent
from google.adk.models.lite_llm import LiteLlm
from google.adk.tools import ToolContext

try:
    from rag.answer_question import answer_academic_documents
    from agents.response_guards import RAG_RESULT_STATE_KEY, preserve_rag_citations
except ModuleNotFoundError as exc:
    if not (
        exc.name == "rag"
        or str(exc.name).startswith("rag.")
        or exc.name == "agents"
        or str(exc.name).startswith("agents.")
    ):
        raise

    from src.rag.answer_question import answer_academic_documents
    from src.agents.response_guards import RAG_RESULT_STATE_KEY, preserve_rag_citations


def _log_unanswered(question: str, status: str, reason: str) -> None:
    try:
        try:
            from src.utils.unanswered_logger import log_unanswered_question
        except ModuleNotFoundError:
            from utils.unanswered_logger import log_unanswered_question
        log_unanswered_question(
            question,
            route="academic_rag_agent",
            status=status,
            reason=(reason or "Insufficient document evidence.")[:500],
        )
    except (OSError, ImportError):
        pass


def search_academic_documents(
    question: str,
    tool_context: ToolContext | None = None,
) -> dict:
    """Search academic documents and retain source evidence for final citations."""
    result = answer_academic_documents(question)
    status = str(result.get("status", "unknown")).casefold()

    if status in {"insufficient_evidence", "not_found", "error"}:
        _log_unanswered(
            question,
            status=status,
            reason=str(result.get("answer") or result.get("details") or ""),
        )

    # Share the tool's evidence with the response guard when invoked inside ADK
    # so source references are not silently lost during agent summarization.
    if tool_context is not None and getattr(tool_context, "state", None) is not None:
        tool_context.state[RAG_RESULT_STATE_KEY] = {
            "status": result.get("status"),
            "answer": result.get("answer", ""),
            "sources": [
                {
                    "source": item.get("source") or item.get("document_name")
                }
                for item in result.get("sources", [])
                if isinstance(item, dict)
                and (item.get("source") or item.get("document_name"))
            ],
        }
    return result


root_agent = Agent(
    name="academic_rag_agent",
    model=LiteLlm(
        model=f"ollama_chat/{os.getenv('ACADEMIC_HELPDESK_MODEL', 'qwen3:1.7b').strip()}",
        api_base=os.getenv("ACADEMIC_HELPDESK_OLLAMA_API_BASE", "http://localhost:11434"),
    ),
    description=(
        "Specialist for academic regulations, academic calendar and semester dates, "
        "circulars, attendance, examinations, curriculum, degree credits, and academic policies."
    ),
    instruction="""
You are the Academic Policy and Document Specialist for IIT Jodhpur.

- For every document, policy, regulation, academic calendar, curriculum, or degree requirement question, call search_academic_documents first.
- Base your answer strictly on the returned answer and evidence from search_academic_documents.
- Preserve source dates and do not present old rules as current without supporting evidence.
- If the tool reports insufficient_evidence, state clearly: "I could not determine this reliably from the available sources. Please check with the Office of Academics." Do not guess.
- Do not invent facts or citations.
- Keep answers concise and preserve [Source N] labels from the tool result verbatim beside the claims they support.
- Do not include internal reasoning or planning text in your response.
""",
    tools=[search_academic_documents],
    after_model_callback=preserve_rag_citations,
)
