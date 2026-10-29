"""Dedicated timetable/schedule specialist for the Academic Helpdesk."""

import os
import re

from google.adk.agents import Agent
from google.adk.models.lite_llm import LiteLlm

try:
    from tools.academic_lookup import search_timetable
    from agents.response_guards import clean_model_response
except ModuleNotFoundError as exc:
    if not (
        exc.name == "tools"
        or str(exc.name).startswith("tools.")
        or exc.name == "agents"
        or str(exc.name).startswith("agents.")
    ):
        raise
    from src.tools.academic_lookup import search_timetable
    from src.agents.response_guards import clean_model_response

_FUTURE_TERM_RE = re.compile(
    r"\b(?:next\s+academic\s+year|upcoming\s+academic\s+year|future\s+timetable|"
    r"future\s+schedule|next\s+year|2024[-–]25|2025[-–]26|2026[-–]27|2027[-–]28)\b",
    re.IGNORECASE,
)


def _log_unanswered(query: str, status: str, reason: str) -> None:
    try:
        try:
            from src.utils.unanswered_logger import log_unanswered_question
        except ModuleNotFoundError:
            from utils.unanswered_logger import log_unanswered_question
        log_unanswered_question(
            query,
            route="academic_schedule_agent",
            status=status,
            reason=(reason or "No matching timetable records found.")[:500],
        )
    except (OSError, ImportError):
        pass


def lookup_schedule(
    query: str,
    semester: str = "",
    group_section: str = "",
) -> dict:
    """Return schedule rows only from the collected timetable CSV."""
    q_text = str(query or "").strip()
    s_text = str(semester or "").strip()
    g_text = str(group_section or "").strip()

    if not q_text and not s_text and not g_text:
        return {
            "status": "insufficient_evidence",
            "message": "Please specify a course code/title or an exact semester/section before looking up a timetable.",
            "results": [],
        }

    # Refuse to guess future-year timetables when only historical AY 2023-24
    # Semester-I timetable data is in the structured dataset.
    if _FUTURE_TERM_RE.search(q_text) or _FUTURE_TERM_RE.search(s_text):
        msg = (
            "Insufficient evidence: The exact timetable for the next or future academic year "
            "cannot be verified from the available sources (course_timetable.csv only contains "
            "historical AY 2023-24 Semester-I records)."
        )
        _log_unanswered(f"{q_text} {s_text}".strip(), "insufficient_evidence", msg)
        return {
            "status": "insufficient_evidence",
            "message": msg,
            "results": [],
        }

    results = search_timetable(
        q_text,
        semester=s_text,
        group_section=g_text,
        limit=20,
    )
    if not results:
        msg = (
            f"No matching timetable records for '{q_text or s_text}' were found in the "
            "collected timetable data (course_timetable.csv, AY 2023-24 Semester-I)."
        )
        _log_unanswered(q_text or s_text, "not_found", msg)
        return {
            "status": "not_found",
            "message": msg,
            "results": [],
        }

    # Summarize distinct values so small local models report complete details.
    semesters = sorted({str(r.get("Semester", "")).strip() for r in results if str(r.get("Semester", "")).strip()})
    days = []
    for r in results:
        d = str(r.get("Day", "")).strip()
        if d and d not in days:
            days.append(d)
    slots = sorted({str(r.get("Slot_Type", "")).strip() for r in results if str(r.get("Slot_Type", "")).strip()})
    time_slots = sorted({str(r.get("Time_Slot", "")).strip() for r in results if str(r.get("Time_Slot", "")).strip()})
    groups = sorted({str(r.get("Group_Section", "")).strip() for r in results if str(r.get("Group_Section", "")).strip()})
    instructors = sorted({str(r.get("Instructor", "")).strip() for r in results if str(r.get("Instructor", "")).strip()})

    return {
        "status": "success",
        "results": results,
        "summary": {
            "semesters": semesters,
            "days": days,
            "slot_types": slots,
            "time_slots": time_slots,
            "group_sections": groups,
            "instructors": instructors,
        },
        "source": "course_timetable.csv",
        "warning": (
            "Preserve the Semester value exactly as returned. Do not describe a "
            "historical timetable as current or infer a future timetable."
        ),
    }


root_agent = Agent(
    name="academic_schedule_agent",
    model=LiteLlm(
        model=f"ollama_chat/{os.getenv('ACADEMIC_HELPDESK_MODEL', 'qwen3:1.7b').strip()}",
        api_base=os.getenv("ACADEMIC_HELPDESK_OLLAMA_API_BASE", "http://localhost:11434"),
    ),
    description=(
        "Specialist for weekly class timetables, course time slots, scheduled days, "
        "and group sections from course_timetable.csv (not academic calendar dates)."
    ),
    instruction="""
You are the Academic Schedule Specialist for IIT Jodhpur.

- Call lookup_schedule before answering any timetable, slot, class time, day, or section question.
- Pass only the course code or course title in `query`. Leave `semester` and `group_section` empty (`""`) unless the user explicitly names a specific academic year (such as "AY 2023-24 Semester-I") or group (such as "Group A1").
- Use only rows returned by the tool. Do not create a schedule from memory.
- Preserve academic year, semester, time-slot/Slot_Type, days, instructor, and group/section values as returned.
- If the tool returns not_found or insufficient_evidence, state clearly that no matching timetable record was found in the collected data (or that there is insufficient evidence to verify a future timetable).
- Do not describe AY 2023-24 or other historical timetable rows as current; always label them with the exact semester shown (e.g., AY 2023-24 Semester-I).
- Include the timetable source (`course_timetable.csv`) and term in the final answer.
- Do not include internal reasoning or planning text in your response.
""",
    tools=[lookup_schedule],
    after_model_callback=clean_model_response,
)
