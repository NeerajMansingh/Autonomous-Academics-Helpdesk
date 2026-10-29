"""Append questions that the helpdesk could not resolve to a local CSV log."""

from __future__ import annotations

import csv
from datetime import datetime, timezone
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_LOG_PATH = PROJECT_ROOT / "logs" / "unanswered_questions.csv"
FIELDNAMES = ("timestamp_utc", "question", "route", "status", "reason")


def log_unanswered_question(
    question: str,
    *,
    route: str = "unknown",
    status: str = "insufficient_evidence",
    reason: str = "",
    log_path: str | Path | None = None,
) -> Path:
    """Write one unresolved request to CSV and return the log path.

    Logging failures should be handled by the caller so they never turn a
    valid user-facing response into a failure. The question is stored as
    submitted because the assignment requires human-review follow-up; do not
    log credentials or unrelated personal data in this file.
    """
    path = Path(log_path) if log_path is not None else DEFAULT_LOG_PATH
    path.parent.mkdir(parents=True, exist_ok=True)
    is_new = not path.exists() or path.stat().st_size == 0

    with path.open("a", newline="", encoding="utf-8-sig") as stream:
        writer = csv.DictWriter(stream, fieldnames=FIELDNAMES)
        if is_new:
            writer.writeheader()
        writer.writerow(
            {
                "timestamp_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
                "question": str(question).strip(),
                "route": str(route or "unknown"),
                "status": str(status or "unknown"),
                "reason": str(reason or ""),
            }
        )
    return path
