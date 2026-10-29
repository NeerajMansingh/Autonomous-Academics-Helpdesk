# Multi-Agent Academic Helpdesk — IIT Jodhpur

A local, 16GB-RAM-friendly multi-agent Academic Helpdesk built with **Google ADK**, **Ollama (`qwen3:1.7b`) via LiteLLM**, **ChromaDB RAG**, and **structured CSV lookup tools**.

The system answers questions about IIT Jodhpur academic programmes, course contents, academic rules and regulations, course slots/timetables, and faculty contacts, and drafts polite emails to course instructors (e.g., for sick leave or makeup exams) for student review without sending them. Out-of-scope queries (such as campus dining or city transport) and unverified/missing records are refused without guessing and logged to `logs/unanswered_questions.csv` for human follow-up.

## Architecture: Base Single-Agent to Multi-Agent Orchestrator

1. **Base Single-Agent Version (`src/agents/single_agent.py`)**:
   Combines the ChromaDB RAG tool (`search_academic_documents`) and all structured CSV / email tools (`lookup_schedule`, `lookup_courses`, `count_catalog_courses`, `lookup_faculty`, `find_course_instructor_contact`, `lookup_course_instructor_for_email`) into a single Google ADK `Agent` (`single_academic_helpdesk_agent`).
2. **Multi-Agent Version (`src/agents/academic_agent.py`)**:
   Augments the base architecture with a top-level ADK orchestrator (`academic_helpdesk_coordinator`) that enforces scope/privacy guardrails and transfers each in-scope question via `transfer_to_agent` to four specialized ADK sub-agents:

```text
Student Request
      |
      v
ADK Helpdesk Orchestrator (`src/agents/academic_agent.py`)
  |-- Scope & Privacy Guard (`src/agents/response_guards.py`)
  |-- Intent Routing Guard (`src/agents/intent_router.py`)
  |
  +---> Policy / RAG Agent (`src/agents/rag_agent.py`)
  |       |-- ChromaDB vector store (`vectorstore/academic_helpdesk`)
  |       |-- Grounded QA with `[Source N]` citations (`src/rag/answer_question.py`)
  |
  +---> Schedule Agent (`src/agents/schedule_agent.py`)
  |       |-- Timetable lookup & slot summaries (`course_timetable.csv`)
  |
  +---> Data Agent (`src/agents/data_agent.py`)
  |       |-- Course catalog search & filtered course counts (`course_catalog.csv`)
  |       |-- Faculty directory, research areas & emails (`faculty_contacts.csv`)
  |
  +---> Email Agent (`src/agents/email_agent.py`)
          |-- Instructor contact resolution (`course_timetable.csv` + `faculty_contacts.csv`)
          |-- Editable student email draft generation (never sends email)

Unresolved / Refused Requests ---> `logs/unanswered_questions.csv` (`src/utils/unanswered_logger.py`)
```

## Dataset & Vector Store Summary

- **Collected Documents (`data/academic-helpdesk-data/academic-helpdesk/data/documents/`)**: 91 official IIT Jodhpur PDFs (11 Academic Regulations & Amendments, 31 Curriculum Booklets, 12 Academic Calendars, 37 Official Circulars/Office Orders) extracted into `data/processed/` and indexed into **3,771 chunks** in `vectorstore/academic_helpdesk` (`academic_documents` collection) with page-level provenance (`Document.pdf (Page N)`).
- **Structured CSVs (`data/academic-helpdesk-data/academic-helpdesk/data/structured/`)**:
  - `course_catalog.csv`: 1,378 course records (`Course_Code`, `Course_Title`, `Department`, `Program_Level`, `LTP`, `Credits`, `Category`, `Focus_Area_Syllabus`, `Source_Booklet`).
  - `course_timetable.csv`: 266 timetable entries (`Semester`, `Group_Section`, `Day`, `Time_Slot`, `Course_Code`, `Course_Name`, `Credits`, `LTP`, `Instructor`, `Instructor_Email`, `Slot_Type`).
  - `faculty_contacts.csv`: 326 faculty profiles (`Name`, `Designation`, `Department`, `Email`, `Phone`, `Education`, `Research_Areas`, `Profile_URL`).
  - `document_manifest.csv`: 157 manifest entries with SHA-256 checksums.

## Setup & Installation (Windows PowerShell, 16GB RAM)

From the project root (`F:\academic-helpdesk`):

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
ollama pull qwen3:1.7b
```

Verify that Ollama is running and `qwen3:1.7b` is available:

```powershell
ollama list
```

To switch models (for example, to compare `qwen3:1.7b` against an 8B model such as `qwen3:8b` or `llama3.1:8b`):

```powershell
ollama pull llama3.1:8b
$env:ACADEMIC_HELPDESK_MODEL = "llama3.1:8b"
```

## Rebuild the ChromaDB Vector Index (Optional)

The repository already includes the extracted text in `data/processed/` and the pre-built ChromaDB vector store in `vectorstore/academic_helpdesk/`. Run these commands only if you add or modify PDF documents:

```powershell
python src\rag\extract_documents.py
python src\rag\build_index.py
```

## Running the Helpdesk Agent

Launch an interactive multi-agent ADK session:

```powershell
adk run src\agents
```

Or launch the ADK web UI:

```powershell
adk web src
```

Example queries to test:

- `What is the attendance requirement under the revised policy?`
- `What is the weightage and duration of the Major examination?`
- `What is the slot for the course Autonomous Systems?`
- `Find the timetable for BBL1020 in AY 2023-24 Semester-I`
- `How many 400 level electives have been floated by the Humanities department in this semester?`
- `How many Open Elective credits do I need to earn to get a B.Tech degree?`
- `Write an email to the course instructor of CSL7570 saying I need sick leave for 2 days`
- `What are the available dining options on the campus?`
- `What is the best way to reach the campus from Jodhpur city center?`

## Running Unit Tests & Evaluations

Run all 37 unit tests:

```powershell
python -m pytest tests -v
```

Run the fast deterministic router/tool-path evaluation across all 30 benchmark prompts:

```powershell
python scripts\run_evaluation.py
```

Run the end-to-end ADK multi-agent evaluation with `qwen3:1.7b` (1 run or 3 runs for consistency tracking):

```powershell
python scripts\run_adk_evaluation.py --model qwen3:1.7b --runs 1
python scripts\run_adk_evaluation.py --model qwen3:1.7b --runs 3
```

Run a 5-prompt smoke test across all specialist routes:

```powershell
python scripts\run_adk_evaluation.py --smoke
```

## Repository Structure

```text
src/
  agents/
    __init__.py          Exports root_agent from academic_agent.py
    single_agent.py      Base single-agent ADK implementation (RAG + all tools)
    academic_agent.py    Multi-agent ADK orchestrator + routing/citation callbacks
    rag_agent.py         Policy/curriculum/calendar RAG specialist agent
    schedule_agent.py    Timetable and slot specialist agent
    data_agent.py        Course catalog, course counts, and faculty specialist agent
    email_agent.py       Instructor lookup and student email drafting specialist agent
    intent_router.py     General domain-intent classifier for specialist routing
    response_guards.py   Scope boundaries, Qwen3 thought-part stripping, and citation guards
    router.py            Deterministic tool-path baseline router
  rag/
    extract_documents.py PDF text extraction with page markers
    build_index.py       Page-aware chunking and ChromaDB indexing
    retrieve.py          Standalone ChromaDB passage retrieval utility
    answer_question.py   Cached ChromaDB retrieval + grounded Ollama QA
  tools/
    academic_lookup.py   Course catalog, timetable, faculty, count, and contact tools
  utils/
    unanswered_logger.py CSV logger for unanswered/out-of-scope questions
scripts/
  run_evaluation.py      Deterministic router/tool evaluation script
  run_adk_evaluation.py  End-to-end ADK multi-agent evaluation & multi-run consistency script
tests/                   37 unit tests covering tools, routing, guards, and logging
evaluation/
  test_questions.csv     30 benchmark prompts across 6 categories
  results/               Saved evaluation CSV outputs
logs/
  unanswered_questions.csv Logged out-of-scope and unresolved queries for human review
reports/
  final_evaluation_report.md Comprehensive architecture, bug-fix audit, and evaluation report
```
