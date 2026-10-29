"""Deterministic routing and tool-level evaluation entry point.

The ADK multi-agent coordinator is defined in academic_agent.py. This router
provides predictable routing for unit tests and a non-LLM diagnostic baseline;
its evaluation scores must not be described as end-to-end ADK scores.
"""

import re

from src.agents.email_agent import draft_email
from src.agents.data_agent import (
    lookup_courses,
    lookup_faculty,
    count_catalog_courses,
)
from src.agents.rag_agent import search_academic_documents
from src.agents.schedule_agent import lookup_schedule
from src.agents.response_guards import is_academic_helpdesk_query, refusal_text
from src.utils.unanswered_logger import log_unanswered_question

COURSE_CODE_RE = re.compile(r"\b[A-Z]{2,5}\d{3,5}\b", re.I)


def _strip_prefixes(text: str, prefixes: tuple[str, ...]) -> str:
    """Remove generic request wording so tools receive the search subject."""
    value = text.strip().strip(" ?.,")
    for prefix in sorted(prefixes, key=len, reverse=True):
        value = re.sub(r"^\s*" + re.escape(prefix) + r"\b\s*", "", value, flags=re.I)
    value = re.sub(r"^(?:the|a|an)\s+", "", value, flags=re.I)
    value = re.sub(r"\s+(?:category|courses?|course catalog|course catalogue)\s*$", "", value, flags=re.I)
    value = value.replace("-", " ")
    value = re.sub(r"\b(?:researchers?|faculty members?)\s*$", "", value, flags=re.I)
    return value.strip(" ?.,")


def _status_result(route: str, result: dict) -> dict:
    return {
        "status": result.get("status", "success"),
        "route": route,
        "answer": result,
    }


def _policy_question(lower: str) -> bool:
    """Identify policy/degree requirements that need document evidence."""
    phrases = (
        "b.tech degree",
        "b.tech. degree",
        "degree requirement",
        "degree requirements",
        "open elective credit",
        "credits do i need to earn",
        "curriculum requirement",
        "curriculum requirements",
        "graduation requirement",
        "attendance",
        "major exam",
        "major examination",
        "rules and regulations",
        "academic regulation",
        "examination weightage",
        "exam weightage",
        "makeup exam",
        "make-up exam",
        "academic calendar",
        "semester date",
        "semester dates",
        "office order",
        "circular",
        "programme requirement",
        "program requirement",
        "late fee",
        "late submission",
        "assignment after the deadline",
    )
    return any(phrase in lower for phrase in phrases)


def _parse_course_count(question: str, lower: str) -> dict:
    level_match = re.search(r"\b(\d{3,4})\s*[- ]?level\b", question, re.I)
    dept_match = re.search(
        r"\b(?:by|in|under|from)\s+(?:the\s+)?(?P<department>[A-Za-z][A-Za-z& ]{1,50}?)\s+department\b",
        question,
        re.I,
    )
    if not dept_match:
        dept_match = re.search(
            r"\b(?:the\s+)?(?P<department>[A-Za-z][A-Za-z& ]{1,50}?)\s+department\b",
            question,
            re.I,
        )
    semester_match = re.search(
        r"\bAY\s*20\d{2}[-–]\d{2}\s+Semester\s*[- ]?(?:I|II|III|IV)\b",
        question,
        re.I,
    )
    semester = re.sub(r"\s+", " ", semester_match.group(0)).strip() if semester_match else ""
    asks_term_offering = any(
        word in lower for word in ("this semester", "floated", "offered", "being offered")
    )
    return {
        "department": dept_match.group("department").strip() if dept_match else "",
        "level": level_match.group(1) if level_match else "",
        "category": "elective" if "elective" in lower else "",
        "semester": semester,
        "require_timetable": asks_term_offering,
    }


def _route_question(question: str) -> dict:
    """Route a question without asking a language model to choose a tool."""
    text = str(question or "").strip()
    lower = text.casefold()
    if not text:
        return {"status": "error", "route": "none", "message": "Please enter a question."}

    # Protect private contact details and refuse to invent future schedules.
    private_phone = re.search(
        r"\b(?:private|personal)\s+(?:phone|mobile)\s+number\b", lower
    )
    if private_phone:
        return {
            "status": "insufficient_evidence",
            "route": "academic_helpdesk_coordinator",
            "answer": (
                "I cannot disclose private faculty contact information. "
                "Please use an officially published institutional contact channel."
            ),
        }

    future_schedule = any(
        phrase in lower
        for phrase in ("next academic year", "upcoming academic year", "future timetable", "future schedule")
    )
    schedule_context = any(
        phrase in lower for phrase in ("timetable", "schedule", "class", "time slot", "slot")
    )
    if future_schedule and schedule_context:
        return {
            "status": "insufficient_evidence",
            "route": "academic_schedule_agent",
            "answer": (
                "The exact future timetable cannot be verified from the available dataset. "
                "Please check the latest official timetable published by the institute."
            ),
        }

    # Email intent takes precedence over the course code included in a request.
    asks_to_draft = bool(
        re.search(r"\b(?:draft|write|compose|prepare)\b", lower)
        and re.search(r"\b(?:email|e-mail|mail)\b", lower)
    )
    if asks_to_draft:
        return draft_email(text)

    code_match = COURSE_CODE_RE.search(text)
    course_code = code_match.group(0).upper() if code_match else None

    # A degree/curriculum rule is a policy question even when it mentions credits.
    if _policy_question(lower):
        result = search_academic_documents(text)
        return {
            "status": result.get("status", "success"),
            "route": "academic_rag_agent",
            "answer": result,
        }

    is_count_question = bool(
        re.search(r"\b(?:how many|count|number of)\b", lower)
        and any(term in lower for term in ("course", "elective", "floated", "courses"))
    )
    if is_count_question:
        filters = _parse_course_count(text, lower)
        result = count_catalog_courses(**filters)
        return _status_result("academic_data_agent", result)

    timetable_keywords = (
        "timetable", "time slot", "slot for", "slot of", "class schedule",
        "schedule for", "which day", "which days", "when is the class",
        "when does the class", "class time", "class timing", "scheduled",
    )
    if any(term in lower for term in timetable_keywords):
        query = course_code or _strip_prefixes(
            text,
            (
                "what is the time slot for the course", "what is the slot for the course",
                "what is the time slot for", "what is the slot for", "tell me the slot for",
                "which time slot is", "what slot is the course", "find the timetable for",
                "find timetable for", "show the timetable for", "show timetable for",
                "find the schedule for", "find schedule for", "what is the schedule for",
                "what is the timetable for", "what is the class schedule for",
                "which days is", "which days does", "when is the class for",
                "when does the class", "find the schedule of", "show the slot for",
            ),
        )
        term_match = re.search(
            r"\bAY\s*20\d{2}[-–]\d{2}\s+Semester\s*[- ]?(?:I|II|III|IV)\b",
            text,
            re.I,
        )
        semester = re.sub(r"\s+", " ", term_match.group(0)).strip() if term_match else ""
        return _status_result("academic_schedule_agent", lookup_schedule(query, semester=semester))

    faculty_keywords = (
        "faculty", "professor", "instructor", "research area", "research areas",
        "research interest", "research interests", "faculty email", "faculty phone",
        "email address for a faculty", "email addresses for faculty",
    )
    if any(term in lower for term in faculty_keywords):
        query = _strip_prefixes(
            text,
            (
                "find the research areas of", "find research areas of",
                "find the research area of", "what are the research areas of",
                "find faculty working on", "find faculty who work on",
                "find faculty in", "find faculty from", "find faculty email addresses for",
                "find the faculty email addresses for", "faculty email addresses for",
                "find faculty email for", "find the faculty email for", "find professors in",
                "find professors working on", "show faculty in", "list faculty in",
                "find faculty members working on", "find the faculty members in",
            ),
        )
        return _status_result("academic_data_agent", lookup_faculty(query))

    course_keywords = (
        "course", "courses", "course code", "course title", "course credits",
        "course credit", "credits for course", "credit for course", "course catalog",
        "course catalogue", "prerequisite", "institute core", "program level",
        "programme level", "department for", "course category",
    )
    if course_code or any(term in lower for term in course_keywords):
        query = course_code or _strip_prefixes(
            text,
            (
                "find courses related to", "search courses related to", "find courses in",
                "show courses in", "list courses in", "find course", "search for course",
                "search course", "show course", "find courses for", "find courses by topic",
                "find courses by", "find course code", "find the course title of",
                "show the department and program level for", "tell me the title and credits for",
                "how many credits is", "what are the credits for", "what is the title of",
                "what are the course contents for", "what is the course content for",
                "show the course contents for", "find the course contents for",
                "what does the course cover", "tell me about the course",
            ),
        )
        return _status_result("academic_data_agent", lookup_courses(query))

    # Default to document search. The RAG tool must refuse if evidence is missing.
    result = search_academic_documents(text)
    return {
        "status": result.get("status", "success"),
        "route": "academic_rag_agent",
        "answer": result,
    }


def route_question(question: str) -> dict:
    """Route a question and log unresolved outcomes for human review."""
    result = _route_question(question)
    status = str(result.get("status", "unknown")).casefold()
    nested = result.get("answer")
    if isinstance(nested, dict):
        nested_status = str(nested.get("status", "")).casefold()
        if nested_status in {"not_found", "insufficient_evidence", "error", "ambiguous", "email_not_found"}:
            status = nested_status

    recipient_lookup = result.get("recipient_lookup")
    if isinstance(recipient_lookup, dict) and str(recipient_lookup.get("status", "")).casefold() in {
        "not_found", "insufficient_evidence", "ambiguous", "email_not_found", "error"
    }:
        status = str(recipient_lookup.get("status")).casefold()

    if status in {"not_found", "insufficient_evidence", "error", "ambiguous", "email_not_found"}:
        reason = "The request could not be fully resolved from the returned data."
        if isinstance(nested, dict):
            reason = str(nested.get("message", nested.get("answer", reason)))[:500]
        elif isinstance(nested, str) and nested.strip():
            reason = nested.strip()[:500]
        elif result.get("message"):
            reason = str(result["message"])[:500]
        elif isinstance(result.get("recipient_lookup"), dict):
            reason = str(result["recipient_lookup"].get("message", reason))[:500]
        try:
            log_unanswered_question(
                str(question or ""),
                route=str(result.get("route", "unknown")),
                status=status,
                reason=reason,
            )
        except OSError:
            pass

    return result
