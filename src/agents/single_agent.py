"""Base single-agent Academic Helpdesk combining RAG and all structured tools.

This module implements the initial single-agent version specified in the
assignment before augmenting the architecture to the multi-agent orchestrator
in `src/agents/academic_agent.py`.
"""

from __future__ import annotations

import os
import re

from google.adk.agents import Agent
from google.adk.models.lite_llm import LiteLlm
from google.genai import types

try:
    from agents.data_agent import (
        lookup_courses,
        lookup_faculty,
        count_catalog_courses,
        find_course_instructor_contact,
    )
    from agents.schedule_agent import lookup_schedule
    from agents.rag_agent import search_academic_documents
    from agents.email_agent import lookup_course_instructor_for_email
    from agents.response_guards import (
        is_academic_helpdesk_query,
        refusal_text,
        preserve_rag_citations,
    )
except ModuleNotFoundError as exc:
    if not (exc.name == "agents" or str(exc.name).startswith("agents.")):
        raise
    from src.agents.data_agent import (
        lookup_courses,
        lookup_faculty,
        count_catalog_courses,
        find_course_instructor_contact,
    )
    from src.agents.schedule_agent import lookup_schedule
    from src.agents.rag_agent import search_academic_documents
    from src.agents.email_agent import lookup_course_instructor_for_email
    from src.agents.response_guards import (
        is_academic_helpdesk_query,
        refusal_text,
        preserve_rag_citations,
    )


MODEL_NAME = os.getenv("ACADEMIC_HELPDESK_MODEL", "qwen3:1.7b").strip()
PRIVATE_PHONE_PATTERN = re.compile(
    r"\b(?:private|personal)\s+(?:phone|mobile)\s+number\b",
    re.IGNORECASE,
)


def _single_agent_guard(callback_context):
    """Refuse out-of-scope or private-phone requests and log them for human review."""
    content = getattr(callback_context, "user_content", None)
    parts = getattr(content, "parts", None) or []
    question = " ".join(
        str(getattr(part, "text", "") or "").strip()
        for part in parts
        if getattr(part, "text", None)
    ).strip()

    try:
        from src.utils.unanswered_logger import log_unanswered_question
    except ModuleNotFoundError:
        from utils.unanswered_logger import log_unanswered_question

    if PRIVATE_PHONE_PATTERN.search(question):
        msg = (
            "I cannot disclose private faculty contact information. Please use "
            "an officially published institutional contact channel."
        )
        try:
            log_unanswered_question(
                question,
                route="single_academic_helpdesk_agent",
                status="refused_private_contact",
                reason=msg,
            )
        except OSError:
            pass
        return types.Content(role="model", parts=[types.Part.from_text(text=msg)])

    if not is_academic_helpdesk_query(question):
        msg = refusal_text()
        try:
            log_unanswered_question(
                question,
                route="single_academic_helpdesk_agent",
                status="out_of_scope",
                reason=msg,
            )
        except OSError:
            pass
        return types.Content(role="model", parts=[types.Part.from_text(text=msg)])

    return None


root_agent = Agent(
    name="single_academic_helpdesk_agent",
    model=LiteLlm(
        model=f"ollama_chat/{MODEL_NAME}",
        api_base=os.getenv("ACADEMIC_HELPDESK_OLLAMA_API_BASE", "http://localhost:11434"),
    ),
    description=(
        "Base single-agent IIT Jodhpur Academic Helpdesk with RAG document search, "
        "timetable lookup, course catalog lookup, faculty contact lookup, and email drafting."
    ),
    instruction="""
You are the IIT Jodhpur Academic Helpdesk Assistant (Single-Agent Base Version).

Use your tools to answer academic questions accurately:
- Call `search_academic_documents` for academic regulations, attendance requirements, examinations, academic calendars, circulars, and curriculum credit rules. Preserve `[Source N]` citations.
- Call `lookup_schedule` for course time slots, days, and weekly timetable entries. Preserve the exact semester (e.g., AY 2023-24 Semester-I) and never present historical records as current.
- Call `lookup_courses` or `count_catalog_courses` for course catalog searches and course/elective counts.
- Call `lookup_faculty` or `find_course_instructor_contact` for faculty research areas and published email addresses.
- When asked to write or draft an email to a course instructor (e.g., for sick leave or makeup exams), call `lookup_course_instructor_for_email` if a course code is given, and produce a polite, editable email draft using placeholders for any unknown fields. Never claim the email was sent.
- If a question cannot be answered from the retrieved documents or CSV data, state clearly that you do not know / cannot verify it from the available sources instead of guessing.
""",
    tools=[
        search_academic_documents,
        lookup_schedule,
        lookup_courses,
        count_catalog_courses,
        lookup_faculty,
        find_course_instructor_contact,
        lookup_course_instructor_for_email,
    ],
    before_agent_callback=_single_agent_guard,
    after_model_callback=preserve_rag_citations,
)
