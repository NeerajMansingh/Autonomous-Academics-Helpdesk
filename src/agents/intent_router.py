"""General intent classification for deterministic ADK specialist delegation.

This module classifies request *types*; it does not contain question-specific
answers. Specialist agents still perform retrieval, structured lookups, or
email generation.
"""

from __future__ import annotations

import re

try:
    from agents.response_guards import is_academic_helpdesk_query
except ModuleNotFoundError as exc:
    if not (exc.name == "agents" or str(exc.name).startswith("agents.")):
        raise
    from src.agents.response_guards import is_academic_helpdesk_query


EMAIL_AGENT = "academic_email_agent"
SCHEDULE_AGENT = "academic_schedule_agent"
DATA_AGENT = "academic_data_agent"
RAG_AGENT = "academic_rag_agent"

_COURSE_CODE = re.compile(r"\b[A-Z]{2,5}\s?\d{3,5}[A-Z]?\b", re.IGNORECASE)
_EMAIL_ACTION = re.compile(
    r"\b(?:write|draft|compose|prepare|create)\b.{0,80}\b(?:e-?mail|mail)\b",
    re.IGNORECASE,
)

_CALENDAR_AND_POLICY_TERMS = re.compile(
    r"\b(?:academic calendar|semester dates?|commencement of classes|"
    r"attendance|major exam|major examination|exam weightage|"
    r"examination weightage|weightage|degree requirement|degree requirements|"
    r"open elective credits?|credits? do i need to earn|credits? needed for (?:a )?degree|"
    r"b\s*\.?\s*tech\s*\.? degree|academic regulations?|rules and regulations|"
    r"curriculum rules?|curriculum requirements?|academic policy|academic policies|"
    r"examination rules?|exam rules?|office orders?|circulars?|make[ -]?up exam|"
    r"duration of (?:the )?(?:major )?examination|grading policy|graduation requirements|"
    r"late fees?|late submission|submission deadline|assignment deadline|late penalty|"
    r"deadline for assignments|penalt(?:y|ies) for late|postgraduate programmes?|"
    r"undergraduate programmes?)\b",
    re.IGNORECASE,
)
_SCHEDULE_TERMS = re.compile(
    r"\b(?:timetable|time\s*table|time\s*slot|slot|class schedule|"
    r"schedule for|class time|class timing|class timings|which day|"
    r"which days|when is the class|when does the class|when is .* class|"
    r"scheduled)\b",
    re.IGNORECASE,
)
_COURSE_CONTENT_TERMS = re.compile(
    r"\b(?:course content|course contents|course outline|syllabus|"
    r"what does .* course cover|what is taught in)\b",
    re.IGNORECASE,
)
_FACULTY_TERMS = re.compile(
    r"\b(?:faculty|professor|professors|instructor|lecturer|research area|research areas|"
    r"research interest|research interests|composite[- ]materials?\s+researchers?|"
    r"faculty email|faculty phone|email id for|email address for|email addresses for|"
    r"contact details for|office phone)\b",
    re.IGNORECASE,
)
_COURSE_COUNT_TERMS = re.compile(
    r"\b(?:how many|count|number of)\b.*\b(?:courses?|electives?|floated|offered)\b",
    re.IGNORECASE,
)
_COURSE_CATALOG_TERMS = re.compile(
    r"\b(?:course catalog|course catalogue|course code|course title|"
    r"course credits?|credits? for (?:the )?course|prerequisite|institute core|"
    r"related to bioengineering|find courses?|search for course|"
    r"course category|department for|program level|programme level)\b",
    re.IGNORECASE,
)


def classify_request(question: str) -> str | None:
    """Return a specialist agent name, or ``None`` for out-of-scope requests.

    Precedence is deliberate: email drafting wins over a course code in the
    same request; academic calendar / policy questions use document retrieval;
    timetable intent wins over course-catalog intent.
    """
    text = str(question or "").strip()
    if not text:
        return None

    # Out-of-scope check first (dining, transport from city center, etc.).
    if not is_academic_helpdesk_query(text):
        return None

    # Email drafting must take precedence over a course code or instructor name.
    if _EMAIL_ACTION.search(text):
        return EMAIL_AGENT

    # Protect private contact requests rather than sending them to a general
    # faculty lookup or allowing the language model to disclose private data.
    if re.search(
        r"\b(?:private|personal)\s+(?:phone|mobile)\s+number\b",
        text,
        re.IGNORECASE,
    ):
        return RAG_AGENT

    # Academic calendar and policy questions must route to RAG before schedule
    # checks so "academic calendar ... semester dates" never misroutes to timetable.
    if _CALENDAR_AND_POLICY_TERMS.search(text):
        return RAG_AGENT

    # Course count questions route to data_agent even if they say "this semester".
    if _COURSE_COUNT_TERMS.search(text):
        return DATA_AGENT

    if _SCHEDULE_TERMS.search(text):
        return SCHEDULE_AGENT

    # Content/syllabus questions need document evidence, even if they include
    # a course code; the catalogue is not treated as a syllabus.
    if _COURSE_CONTENT_TERMS.search(text):
        return RAG_AGENT

    if _FACULTY_TERMS.search(text):
        return DATA_AGENT

    if _COURSE_CODE.search(text) or _COURSE_CATALOG_TERMS.search(text):
        return DATA_AGENT

    # An academic question not covered by a narrower route should be sent to
    # RAG, where evidence sufficiency is checked.
    return RAG_AGENT
