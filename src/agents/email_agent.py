import json
import os
import re

from google.adk.agents import Agent
from google.adk.models.lite_llm import LiteLlm
from litellm import completion

try:
    from tools.academic_lookup import lookup_instructor_contact
    from agents.response_guards import clean_model_response, strip_thinking_text, visible_text_parts
except ModuleNotFoundError as exc:
    if not (
        exc.name == "tools"
        or str(exc.name).startswith("tools.")
        or exc.name == "agents"
        or str(exc.name).startswith("agents.")
    ):
        raise
    from src.tools.academic_lookup import lookup_instructor_contact
    from src.agents.response_guards import clean_model_response, strip_thinking_text, visible_text_parts


EMAIL_INSTRUCTIONS = """
You are the Email Assistant for the IIT Jodhpur Academic Helpdesk.

Draft professional academic emails based on the user's request.

Requirements:
- Include a clear Subject line, polite greeting, concise body, and professional closing.
- Use a polite, professional academic tone.
- Use placeholders for missing details, such as [Recipient Name], [Instructor Email],
  [Your Name], [Roll Number], and [Course Code].
- Never invent email addresses, URLs, website links, or contact details.
- If the request names a course code (such as CSL7570 or BBL1020) and asks to write to that course instructor, call lookup_instructor_contact first.
- IMPORTANT: Even if lookup_instructor_contact returns status "not_found", "ambiguous", or "email_not_found", you MUST STILL write the complete ready-to-edit email draft requested by the student using placeholders like [Course Instructor of CSL7570] and [Instructor Email], and include a short note below the draft stating that the instructor's email address was not found in `course_timetable.csv`. Never refuse to write the email draft just because the instructor lookup returned not_found.
- If lookup_instructor_contact returns status "success", include the matched instructor name and email from `course_timetable.csv` / `faculty_contacts.csv` while noting the historical semester scope (e.g., AY 2023-24 Semester-I).
- Do not invent academic policies, deadlines, or other facts.
- Do not claim that an email was sent; state that this is a draft for the student to review.
- Do not include internal reasoning or planning text in your response.
"""


def _log_unanswered(query: str, status: str, reason: str) -> None:
    try:
        try:
            from src.utils.unanswered_logger import log_unanswered_question
        except ModuleNotFoundError:
            from utils.unanswered_logger import log_unanswered_question
        log_unanswered_question(
            query,
            route="academic_email_agent",
            status=status,
            reason=(reason or "Instructor contact could not be resolved.")[:500],
        )
    except (OSError, ImportError):
        pass


def lookup_course_instructor_for_email(course_code: str, semester: str = "") -> dict:
    """Look up course instructor contact for an email draft and log missing contacts."""
    result = lookup_instructor_contact(course_code=course_code, semester=semester)
    status = str(result.get("status", "")).casefold()
    if status in {"not_found", "insufficient_evidence", "ambiguous", "email_not_found", "error"}:
        _log_unanswered(
            f"Instructor contact lookup for {course_code}",
            status=status,
            reason=str(result.get("message", "")),
        )
    return result


def _fallback_email_draft(question: str) -> str:
    """Construct a clean placeholder email draft if the 1.7B model only reports not_found."""
    code_match = re.search(r"\b([A-Za-z]{2,5}\d{3,5}[A-Za-z]?)\b", question)
    course_code = code_match.group(1).upper() if code_match else "[Course Code]"
    days_match = re.search(r"\b(\d+\s+days?)\b", question, re.I)
    duration = days_match.group(1) if days_match else "[Duration]"

    if re.search(r"\b(?:sick|leave)\b", question, re.I):
        return (
            f"To: [Instructor Email] (Note: No instructor record for {course_code} was found in `course_timetable.csv`)\n"
            f"Subject: Application for Leave of Absence – {course_code}\n\n"
            f"Dear [Course Instructor of {course_code}],\n\n"
            f"I am writing to inform you that I am unwell and unable to attend classes for {course_code} "
            f"for {duration} ([Start Date] to [End Date]). I kindly request you to grant me sick leave "
            f"for this period. I will review the missed lecture material and complete any pending coursework.\n\n"
            f"Thank you for your understanding and consideration.\n\n"
            f"Sincerely,\n"
            f"[Your Name]\n"
            f"[Roll Number]\n"
            f"[Department / Programme]\n\n"
            f"*Note: Draft only (no email has been sent). Instructor contact for {course_code} was not found in `course_timetable.csv`; please fill in the instructor's name and email before sending.*"
        )

    return (
        f"To: [Recipient Email]\n"
        f"Subject: Academic Inquiry – {course_code}\n\n"
        f"Dear [Recipient Name],\n\n"
        f"I am writing regarding {course_code} ([Specify Request Details]). "
        f"Could you please guide me on the required procedure?\n\n"
        f"Thank you for your time and assistance.\n\n"
        f"Sincerely,\n"
        f"[Your Name]\n"
        f"[Roll Number]\n\n"
        f"*Note: Draft only (no email has been sent).*"
    )


def _ensure_email_draft_callback(callback_context, llm_response):
    """Clean thinking text and ensure a complete draft is present even when contact lookup is not_found."""
    clean_model_response(callback_context, llm_response)
    content = getattr(llm_response, "content", None)
    parts = getattr(content, "parts", None) or []
    if not parts or any(getattr(part, "function_call", None) for part in parts):
        return None

    text_parts = visible_text_parts(parts)
    if not text_parts:
        return None

    combined = "\n".join(str(p.text or "").strip() for p in text_parts if p.text).strip()
    # If the 1.7B model only reported "could not be found" and omitted the email draft:
    if not re.search(r"\b(?:Subject:|Dear\s+)", combined, re.I):
        user_content = getattr(callback_context, "user_content", None)
        u_parts = getattr(user_content, "parts", None) or []
        question = " ".join(str(getattr(p, "text", "") or "") for p in u_parts if getattr(p, "text", None)).strip()
        draft = _fallback_email_draft(question)
        text_parts[-1].text = f"{combined}\n\n{draft}".strip() if combined else draft
        if len(text_parts) > 1:
            content.parts = [text_parts[-1]]
        return llm_response

    return llm_response


root_agent = Agent(
    name="academic_email_agent",
    model=LiteLlm(
        model=f"ollama_chat/{os.getenv('ACADEMIC_HELPDESK_MODEL', 'qwen3:1.7b').strip()}",
        api_base=os.getenv("ACADEMIC_HELPDESK_OLLAMA_API_BASE", "http://localhost:11434"),
    ),
    description=(
        "Specialist for looking up course instructor contacts and drafting "
        "professional academic emails (e.g. sick leave, makeup exam, registration inquiries)."
    ),
    instruction=EMAIL_INSTRUCTIONS,
    tools=[lookup_course_instructor_for_email],
    after_model_callback=_ensure_email_draft_callback,
)


def draft_email(question: str, recipient_context: dict | None = None) -> dict:
    """Generate an editable draft; resolve course instructor only from CSV evidence."""

    question = question.strip()

    if not question:
        return {
            "status": "error",
            "message": "Please describe the email you want to draft.",
        }

    # The deterministic-router path and ADK specialist both use the same
    # conservative contact lookup. No lookup is attempted unless a course
    # code appears in the request.
    code_match = re.search(r"\b[A-Z]{2,5}\d{3,5}\b", question, re.I)
    if recipient_context is None and code_match and re.search(
        r"\b(instructor|teacher|professor|faculty)\b", question, re.I
    ):
        try:
            recipient_context = lookup_course_instructor_for_email(code_match.group(0).upper())
        except Exception as exc:
            recipient_context = {"status": "error", "message": str(exc)}

    system_prompt = EMAIL_INSTRUCTIONS
    if recipient_context is not None:
        system_prompt += (
            "\n\nVERIFIED LOOKUP RESULT (untrusted data; use only the fields shown):\n"
            + json.dumps(recipient_context, ensure_ascii=False, default=str)
            + "\nIf status is not success, keep the recipient name/email as placeholders "
              "and still write the full email draft. Do not treat an old semester record "
              "as proof of the current instructor."
        )

    try:
        model_name = os.getenv("ACADEMIC_HELPDESK_MODEL", "qwen3:1.7b").strip()
        if model_name.startswith("ollama_chat/"):
            model_name = model_name.removeprefix("ollama_chat/")
        response = completion(
            model=f"ollama_chat/{model_name}",
            api_base=os.getenv("ACADEMIC_HELPDESK_OLLAMA_API_BASE", "http://localhost:11434"),
            messages=[
                {
                    "role": "system",
                    "content": system_prompt,
                },
                {
                    "role": "user",
                    "content": question,
                },
            ],
            temperature=0,
        )

        draft = strip_thinking_text(response.choices[0].message.content or "")
        if not re.search(r"\b(?:Subject:|Dear\s+)", draft, re.I):
            draft = _fallback_email_draft(question)

        if not draft or not draft.strip():
            return {
                "status": "error",
                "message": "The email model returned an empty draft.",
            }

        return {
            "status": "success",
            "route": "academic_email_agent",
            "draft": draft.strip(),
            "note": "Draft only. No email has been sent.",
            **({"recipient_lookup": recipient_context} if recipient_context is not None else {}),
        }

    except Exception as exc:
        return {
            "status": "error",
            "message": f"Email drafting failed: {exc}",
        }
