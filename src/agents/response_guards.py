"""Deterministic guardrails for scope, thinking-block cleanup, and source transparency.

These helpers classify the helpdesk's supported topic areas, strip internal
model reasoning parts, and format retrieved source references.
"""

from __future__ import annotations

import re
from typing import Any


RAG_RESULT_STATE_KEY = "temp:academic_helpdesk_last_rag_result"

OUT_OF_SCOPE_RESPONSE = (
    "I don't know the answer to that. This helpdesk handles IIT Jodhpur "
    "academic programmes and course content, academic rules and regulations, "
    "course schedules, faculty contact lookups, and email drafts to course "
    "instructors. I've logged this request for human review."
)

# Domain boundary covering academic programmes, courses, timetables, faculty
# research/contacts, regulations, calendars, and academic emails.
_ACADEMIC_TERMS = re.compile(
    r"\b(?:"
    r"academic|academics|course|courses|coursework|class|classes|credit|credits|"
    r"b\s*\.?\s*tech|m\s*\.?\s*tech|b\s*\.?\s*sc|m\s*\.?\s*sc|m\s*\.?\s*des|mba|"
    r"ph\s*\.?\s*d|degree|programme|program|curriculum|syllabus|elective|electives|"
    r"semester|slot|timetable|time\s*table|schedule|registration|register|"
    r"attendance|exam|exams|examination|major|minor|grading|grade|grades|"
    r"faculty|professor|professors|instructor|lecturer|research|researcher|researchers|"
    r"regulation|regulations|policy|policies|circular|circulars|office order|office orders|"
    r"course content|course outline|make[\s-]?up|sick leave|leave of absence|"
    r"admission|department|humanities|aerospace|bioengineering|composite|"
    r"senate|academics office|course instructor|calendar|assignment|assignments|"
    r"deadline|deadlines|submission|late fee|late fees|quiz|quizzes|thesis|project"
    r")\b",
    re.IGNORECASE,
)
_COURSE_CODE = re.compile(r"\b[A-Z]{2,5}\s?\d{3,5}[A-Z]?\b", re.IGNORECASE)

_SELF_TALK_PARAGRAPH = re.compile(
    r"^(?:"
    r"Okay,\s+(?:let's|the\s+user|I\s+need|so\s+)|"
    r"First,\s+I\s+(?:need|should|called|used)|"
    r"Now,\s+(?:I\s+need|the\s+user)|"
    r"Let\s+me\s+(?:check|see|recall|break\s+down)|"
    r"I\s+should\s+(?:start\s+by|check\s+if|outline|state|mention)|"
    r"So,\s+the\s+(?:final\s+)?answer\s+should"
    r").*",
    re.IGNORECASE | re.DOTALL,
)


def is_academic_helpdesk_query(question: str) -> bool:
    """Return True when the question contains an academic-helpdesk topic cue."""
    text = str(question or "").strip()
    if not text:
        return False
    # Out-of-scope campus lifestyle / travel / dining queries must be refused even
    # if they casually mention campus.
    if re.search(
        r"\b(?:dining\s+options?|cafeteria|restaurants?|mess\s+menu|"
        r"reach\s+(?:the\s+)?campus\s+from|jodhpur\s+city\s+cent(?:er|re)|"
        r"bus\s+route|train\s+station|airport\s+to\s+campus|weather\s+in)\b",
        text,
        re.IGNORECASE,
    ):
        return False
    return bool(_COURSE_CODE.search(text) or _ACADEMIC_TERMS.search(text))


def refusal_text() -> str:
    """Return the standard, explicit out-of-scope response."""
    return OUT_OF_SCOPE_RESPONSE


def strip_thinking_text(text: str) -> str:
    """Strip Qwen3 <think> tags, orphan </think> prefixes, and self-talk paragraphs."""
    cleaned = str(text or "").strip()
    if not cleaned:
        return ""

    # Remove explicit <think>...</think> or leading text before </think>.
    cleaned = re.sub(
        r"<think>.*?</think>\s*", "", cleaned, flags=re.DOTALL | re.IGNORECASE
    ).strip()
    cleaned = re.sub(
        r"^.*?</think>\s*", "", cleaned, flags=re.DOTALL | re.IGNORECASE
    ).strip()

    # Remove false meta-notes about missing citations sometimes emitted by 1.7B models.
    cleaned = re.sub(
        r"\n?\*?Note:\s*Citations\s+are\s+(?:missing|incomplete)[^\n]*\*?",
        "",
        cleaned,
        flags=re.IGNORECASE,
    ).strip()

    # If multiple paragraphs exist and leading paragraphs are model self-talk,
    # drop the leading self-talk paragraphs while keeping the actual answer.
    paragraphs = [p.strip() for p in re.split(r"\n\s*\n", cleaned) if p.strip()]
    while len(paragraphs) > 1 and _SELF_TALK_PARAGRAPH.match(paragraphs[0]):
        paragraphs.pop(0)

    return "\n\n".join(paragraphs).strip()


def visible_text_parts(parts: list[Any] | None) -> list[Any]:
    """Return only non-thought text parts from an ADK Content.parts list."""
    result = []
    for part in parts or []:
        if getattr(part, "thought", False):
            continue
        if getattr(part, "text", None):
            result.append(part)
    return result


def clean_model_response(callback_context, llm_response):
    """ADK after_model_callback: strip thought parts and clean visible text."""
    content = getattr(llm_response, "content", None)
    parts = getattr(content, "parts", None) or []
    if not parts or any(getattr(part, "function_call", None) for part in parts):
        return None

    non_thought_text_parts = visible_text_parts(parts)
    if not non_thought_text_parts:
        return None

    changed = False
    for part in non_thought_text_parts:
        cleaned = strip_thinking_text(str(part.text or ""))
        if cleaned and cleaned != part.text:
            part.text = cleaned
            changed = True

    # Remove thought=True parts from final user-facing responses so downstream
    # consumers only see the clean response text.
    if any(getattr(part, "thought", False) for part in parts):
        content.parts = [
            part for part in parts if not getattr(part, "thought", False)
        ]
        changed = True

    return llm_response if changed else None


def append_retrieved_source_list(answer: str, rag_result: dict[str, Any] | None) -> str:
    """Add actual retrieved-source references if an answer omitted citations.

    The list is explicitly labelled as retrieved sources. This preserves a
    verifiable audit trail without pretending that a missing citation was
    generated by the model. It does not validate semantic groundedness.
    """
    text = strip_thinking_text(str(answer or ""))
    result = rag_result or {}
    if not text or result.get("status") != "success":
        return text
    if "Retrieved sources for verification:" in text:
        return text
    # Do not attach a source list to refusals or model-service error messages.
    if re.search(
        r"could not determine|insufficient evidence|not enough evidence|"
        r"don't know|do not know|could not be reached|request failed|"
        r"encountered an error",
        text,
        re.IGNORECASE,
    ):
        return text

    sources = result.get("sources") or []
    if not sources:
        return text

    lines = []
    for index, item in enumerate(sources, start=1):
        if isinstance(item, dict):
            label = item.get("source") or item.get("document_name")
        else:
            label = str(item)
        label = str(label or "").strip()
        if label:
            lines.append(f"[Source {index}] {label}")

    if not lines:
        return text
    return text + "\n\nRetrieved sources for verification:\n" + "\n".join(lines)


def preserve_rag_citations(callback_context, llm_response):
    """Ensure a user-facing RAG response retains retrieved-source references."""
    clean_model_response(callback_context, llm_response)

    state = getattr(callback_context, "state", None)
    result = state.get(RAG_RESULT_STATE_KEY) if state is not None else None
    if not isinstance(result, dict):
        return None

    content = getattr(llm_response, "content", None)
    parts = getattr(content, "parts", None) or []
    if not parts or any(getattr(part, "function_call", None) for part in parts):
        return None

    text_parts = visible_text_parts(parts)
    if not text_parts:
        return None

    # If the RAG tool returned insufficient_evidence, preserve that refusal
    # rather than letting the small local model speculate.
    if result.get("status") == "insufficient_evidence":
        fallback = str(
            result.get("answer")
            or "I could not determine this reliably from the available sources. Please check with the Office of Academics."
        ).strip()
        text_parts[-1].text = fallback
        if len(text_parts) > 1:
            content.parts = [text_parts[-1]]
        return llm_response

    if result.get("status") != "success":
        return None

    combined = "\n".join(strip_thinking_text(str(part.text)) for part in text_parts if part.text).strip()
    updated = append_retrieved_source_list(combined, result)
    if updated == combined and len(text_parts) == len(parts):
        return None

    text_parts[-1].text = updated
    if len(text_parts) > 1:
        content.parts = [text_parts[-1]]
    return llm_response
