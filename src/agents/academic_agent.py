"""LLM-directed ADK coordinator for the Academic Helpdesk."""

from __future__ import annotations

import os
import re

from google.adk.agents import Agent
from google.adk.models.lite_llm import LiteLlm
from google.genai import types

try:
    from agents.data_agent import root_agent as data_agent
    from agents.email_agent import root_agent as email_agent
    from agents.rag_agent import root_agent as rag_agent
    from agents.schedule_agent import root_agent as schedule_agent
    from agents.intent_router import classify_request
    from agents.response_guards import (
        is_academic_helpdesk_query,
        refusal_text,
        preserve_rag_citations,
        strip_thinking_text,
        visible_text_parts,
    )
except ModuleNotFoundError as exc:
    if not (
        exc.name == "agents"
        or str(exc.name).startswith("agents.")
    ):
        raise

    from src.agents.data_agent import root_agent as data_agent
    from src.agents.email_agent import root_agent as email_agent
    from src.agents.rag_agent import root_agent as rag_agent
    from src.agents.schedule_agent import root_agent as schedule_agent
    from src.agents.intent_router import classify_request
    from src.agents.response_guards import (
        is_academic_helpdesk_query,
        refusal_text,
        preserve_rag_citations,
        strip_thinking_text,
        visible_text_parts,
    )


MODEL_NAME = os.getenv("ACADEMIC_HELPDESK_MODEL", "qwen3:1.7b").strip()
PRIVATE_PHONE_PATTERN = re.compile(
    r"\b(?:private|personal)\s+(?:phone|mobile)\s+number\b",
    re.IGNORECASE,
)

UNRESOLVED_PATTERN = re.compile(
    r"could not determine this reliably|insufficient evidence|"
    r"not enough evidence|no matching (?:course|faculty|timetable|record|records)|"
    r"not found in the collected|could not find|"
    r"email address (?:was )?not found|unable to verify|"
    r"cannot verify from the available|local model request failed|"
    r"document retrieval tool encountered an error",
    re.IGNORECASE,
)

_SPECIALIST_NAMES = {
    "academic_data_agent",
    "academic_schedule_agent",
    "academic_rag_agent",
    "academic_email_agent",
}


def _question_text(context) -> str:
    """Extract the initial user message from ADK context."""
    content = getattr(context, "user_content", None)
    parts = getattr(content, "parts", None) or []

    return " ".join(
        str(getattr(part, "text", "") or "").strip()
        for part in parts
        if getattr(part, "text", None)
    ).strip()


def _log_unanswered(
    question: str,
    *,
    route: str,
    status: str,
    reason: str,
) -> None:
    """Record unresolved or refused requests without breaking the response."""
    try:
        try:
            from src.utils.unanswered_logger import log_unanswered_question
        except ModuleNotFoundError:
            from utils.unanswered_logger import log_unanswered_question

        log_unanswered_question(
            question,
            route=route,
            status=status,
            reason=(reason or "The request could not be resolved.")[:500],
        )
    except (OSError, ImportError):
        # Logging failure must not prevent a user-facing response.
        pass


def _content(text: str) -> types.Content:
    """Create a standard ADK text response."""
    return types.Content(
        role="model",
        parts=[types.Part.from_text(text=text)],
    )


def _guard_request(callback_context):
    """Apply safety and scope checks before the coordinator calls the LLM."""
    question = _question_text(callback_context)

    if PRIVATE_PHONE_PATTERN.search(question):
        message = (
            "I cannot disclose private faculty contact information. Please use "
            "an officially published institutional contact channel."
        )
        _log_unanswered(
            question,
            route="academic_helpdesk_coordinator",
            status="refused_private_contact",
            reason=message,
        )
        return _content(message)

    if not is_academic_helpdesk_query(question):
        message = refusal_text()
        _log_unanswered(
            question,
            route="academic_helpdesk_coordinator",
            status="out_of_scope",
            reason=message,
        )
        return _content(message)

    return None


def _log_unresolved_model_response(callback_context, llm_response):
    """Log an unresolved final coordinator response, excluding thought=True parts."""
    parts = getattr(getattr(llm_response, "content", None), "parts", None) or []

    # Do not interpret an agent transfer or tool call as the final answer.
    if any(getattr(part, "function_call", None) for part in parts):
        return None

    text_parts = visible_text_parts(parts)
    answer = "\n".join(
        strip_thinking_text(str(getattr(part, "text", "") or ""))
        for part in text_parts
        if getattr(part, "text", None)
    ).strip()

    if not answer or not UNRESOLVED_PATTERN.search(answer):
        return None

    question = _question_text(callback_context)
    not_found = re.search(
        r"no matching|not found in the collected|could not find|"
        r"email address (?:was )?not found",
        answer,
        re.IGNORECASE,
    )

    _log_unanswered(
        question,
        route="academic_helpdesk_coordinator",
        status="not_found" if not_found else "insufficient_evidence",
        reason=answer,
    )
    return None


def _ensure_specialist_delegation(callback_context, llm_response):
    """Ensure the 1.7B coordinator delegates to the right specialist sub-agent."""
    content = getattr(llm_response, "content", None)
    parts = getattr(content, "parts", None) or []
    question = _question_text(callback_context)
    expected_specialist = classify_request(question)

    if expected_specialist not in _SPECIALIST_NAMES:
        return None

    # Check if the coordinator emitted a function_call part.
    fc_parts = [p for p in parts if getattr(p, "function_call", None)]
    if fc_parts:
        for p in fc_parts:
            fc = p.function_call
            fc_name = getattr(fc, "name", "")
            fc_args = getattr(fc, "args", None)
            if fc_name == "transfer_to_agent" and isinstance(fc_args, dict):
                if fc_args.get("agent_name") != expected_specialist:
                    fc_args["agent_name"] = expected_specialist
                    return llm_response
            elif fc_name in _SPECIALIST_NAMES:
                fc.name = "transfer_to_agent"
                fc.args = {"agent_name": expected_specialist}
                return llm_response
        return None

    # If the coordinator attempted to answer an in-scope question directly from
    # memory without transferring to a specialist, enforce transfer_to_agent.
    if content is not None:
        content.parts = [
            types.Part.from_function_call(
                name="transfer_to_agent",
                args={"agent_name": expected_specialist},
            )
        ]
        return llm_response

    return None


def _coordinator_after_model(callback_context, llm_response):
    """Verify specialist handoff, log unresolved answers, and preserve RAG citations."""
    delegated = _ensure_specialist_delegation(callback_context, llm_response)
    if delegated is not None:
        return delegated
    _log_unresolved_model_response(callback_context, llm_response)
    return preserve_rag_citations(callback_context, llm_response)


root_agent = Agent(
    name="academic_helpdesk_coordinator",
    model=LiteLlm(
        model=f"ollama_chat/{MODEL_NAME}",
        api_base=os.getenv("ACADEMIC_HELPDESK_OLLAMA_API_BASE", "http://localhost:11434"),
    ),
    description=(
        "LLM coordinator for the IIT Jodhpur Academic Helpdesk. "
        "Transfers in-scope user requests to the appropriate academic specialist."
    ),
    instruction="""
You are the coordinating agent for the IIT Jodhpur Academic Helpdesk.

DELEGATION:
- Delegate every in-scope academic question to the appropriate specialist sub-agent by calling `transfer_to_agent` with `agent_name` set to one of:
  - `academic_data_agent`: for course catalog searches, course counts, floated electives, faculty directory lookups, faculty research areas, and published faculty email addresses.
  - `academic_schedule_agent`: for weekly class timetables, course time slots, class days, and group sections (NOT academic calendar dates).
  - `academic_rag_agent`: for academic regulations, academic calendars and semester dates, circulars, office orders, attendance requirements, examinations, curriculum, Open Elective/degree credit rules, and academic policies (including assignment deadline/late-fee rules).
  - `academic_email_agent`: for drafting academic emails (e.g., sick leave, makeup exams, course registration, faculty meetings) and looking up course instructor contacts for emails.
- Never answer a factual academic question from memory; always transfer to the specialist.

ACCURACY:
- Do not invent dates, course offerings, timetable slots, faculty contacts, policy rules, citations, or email addresses.
- Preserve historical dates and semesters. Never present historical timetable rows as current.
- Preserve source labels and citations returned by the document specialist.
- If the request is outside this helpdesk's academic scope, refuse it rather than guessing.
- Never disclose private faculty contact information.
- An email request means draft only; never claim that an email was sent.

RESPONSE:
- Give a concise answer to the user's actual request.
- Do not expose internal reasoning, routing instructions, or planning text.
""",
    sub_agents=[
        data_agent,
        schedule_agent,
        rag_agent,
        email_agent,
    ],
    before_agent_callback=_guard_request,
    after_model_callback=_coordinator_after_model,
)
