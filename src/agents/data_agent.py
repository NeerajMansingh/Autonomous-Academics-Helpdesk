import os

from google.adk.agents import Agent
from google.adk.models.lite_llm import LiteLlm

try:
    from tools.academic_lookup import (
        search_courses,
        search_faculty,
        count_courses,
        lookup_instructor_contact,
    )
    from agents.response_guards import clean_model_response
except ModuleNotFoundError as exc:
    if not (
        exc.name == "tools"
        or str(exc.name).startswith("tools.")
        or exc.name == "agents"
        or str(exc.name).startswith("agents.")
    ):
        raise

    from src.tools.academic_lookup import (
        search_courses,
        search_faculty,
        count_courses,
        lookup_instructor_contact,
    )
    from src.agents.response_guards import clean_model_response


def _log_unanswered(query: str, status: str, reason: str) -> None:
    try:
        try:
            from src.utils.unanswered_logger import log_unanswered_question
        except ModuleNotFoundError:
            from utils.unanswered_logger import log_unanswered_question
        log_unanswered_question(
            query,
            route="academic_data_agent",
            status=status,
            reason=(reason or "No matching catalog or faculty record found.")[:500],
        )
    except (OSError, ImportError):
        pass


def lookup_courses(query: str) -> dict:
    """Search the course catalog by code, title, department, or topic."""
    results = search_courses(query, limit=10)

    if not results:
        msg = "No matching courses were found in the collected catalog."
        _log_unanswered(query, "not_found", msg)
        return {
            "status": "not_found",
            "message": msg,
            "results": [],
        }

    return {
        "status": "success",
        "results": results,
        "source": "course_catalog.csv",
        "note": "Details are from the collected course catalog (course_catalog.csv).",
    }


def count_catalog_courses(
    department: str = "",
    level: str = "",
    category: str = "",
    semester: str = "",
    require_timetable: bool = False,
) -> dict:
    """Count courses using explicit department, level, category and term filters."""
    result = count_courses(
        department=department,
        level=level,
        category=category,
        semester=semester,
        require_timetable=require_timetable,
    )
    if result.get("status") in {"insufficient_evidence", "not_found", "error"}:
        _log_unanswered(
            f"count_courses(department={department}, level={level}, category={category}, semester={semester})",
            str(result.get("status")),
            str(result.get("message", "")),
        )
    return result


def find_course_instructor_contact(course_code: str, semester: str = "") -> dict:
    """Find an instructor and published email for a course, if unambiguous."""
    result = lookup_instructor_contact(course_code=course_code, semester=semester)
    if result.get("status") in {"not_found", "insufficient_evidence", "ambiguous", "email_not_found", "error"}:
        _log_unanswered(
            f"instructor_contact({course_code})",
            str(result.get("status")),
            str(result.get("message", "")),
        )
    return result


def lookup_faculty(query: str) -> dict:
    """Search faculty by name, department, designation, or research area."""
    results = search_faculty(query, limit=10)

    if not results:
        msg = "No matching faculty records were found in faculty_contacts.csv."
        _log_unanswered(query, "not_found", msg)
        return {
            "status": "not_found",
            "message": msg,
            "results": [],
        }

    return {
        "status": "success",
        "results": results,
        "source": "faculty_contacts.csv",
        "note": "Use only returned details from faculty_contacts.csv; never invent missing information.",
    }


root_agent = Agent(
    name="academic_data_agent",
    model=LiteLlm(
        model=f"ollama_chat/{os.getenv('ACADEMIC_HELPDESK_MODEL', 'qwen3:1.7b').strip()}",
        api_base=os.getenv("ACADEMIC_HELPDESK_OLLAMA_API_BASE", "http://localhost:11434"),
    ),
    description=(
        "Specialist for course catalog lookups, course counts, faculty directory, "
        "faculty research areas, and published faculty email contacts."
    ),
    instruction="""
You are the Academic Data Specialist for IIT Jodhpur.

TOOL REQUIREMENTS:
- Course questions: call lookup_courses before answering.
- Faculty questions (including research areas, department faculty lists, and faculty email lookups): call lookup_faculty before answering.
- For numerical course-catalog or floated-elective questions, call count_catalog_courses. If the question asks how many courses/electives have been floated or offered "in this semester", pass `require_timetable=True` and `semester="this semester"`. When the tool returns `insufficient_evidence`, state clearly that there is insufficient evidence in the collected data to verify the current semester's floated courses because the timetable data only covers historical AY 2023-24 Semester-I records.
- For instructor contact requests by course code, call find_course_instructor_contact. Use only returned names/emails and preserve ambiguity or historical-semester information.
- Never answer a lookup question from memory instead of using its tool.

ACCURACY:
- Base factual answers only on actual tool results.
- Never invent course details, schedules, names, email addresses, or URLs.
- If a result has status not_found or insufficient_evidence, report that clearly.
- Treat records from AY 2023-24 or earlier as historical; never describe historical timetable records as current.

RESPONSE:
- Answer only what the user asked and cite the structured data source (`course_catalog.csv` or `faculty_contacts.csv`).
- Keep answers concise and factual.
- Do not display planning text or internal reasoning.
""",
    tools=[
        lookup_courses,
        lookup_faculty,
        count_catalog_courses,
        find_course_instructor_contact,
    ],
    after_model_callback=clean_model_response,
)
