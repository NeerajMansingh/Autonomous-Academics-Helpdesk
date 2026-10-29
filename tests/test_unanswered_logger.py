import csv

from src.utils.unanswered_logger import log_unanswered_question


def test_unanswered_questions_are_appended_to_csv(tmp_path):
    path = tmp_path / "logs" / "unanswered.csv"
    first = log_unanswered_question(
        "What are the dining options?",
        route="academic_rag_agent",
        status="insufficient_evidence",
        reason="No academic source supports this.",
        log_path=path,
    )
    log_unanswered_question(
        "What is the future timetable?",
        route="academic_rag_agent",
        status="insufficient_evidence",
        reason="Future timetable is not in the data.",
        log_path=path,
    )

    with path.open(encoding="utf-8-sig", newline="") as stream:
        rows = list(csv.DictReader(stream))

    assert first == path
    assert len(rows) == 2
    assert rows[0]["question"] == "What are the dining options?"
    assert rows[1]["status"] == "insufficient_evidence"
    assert rows[1]["reason"] == "Future timetable is not in the data."
