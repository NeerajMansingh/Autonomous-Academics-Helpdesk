"""Evaluate the deterministic router/tool path (not ADK subagent handoffs).

For end-to-end delegation, use scripts/run_adk_evaluation.py. This script is a
fast diagnostic baseline and deliberately separates workflow checks from
citation-presence checks; semantic accuracy/groundedness still needs review.
"""

import json
import re
import sys
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.agents.router import route_question  # noqa: E402

INPUT_FILE = PROJECT_ROOT / "evaluation" / "test_questions.csv"
OUTPUT_FILE = PROJECT_ROOT / "evaluation" / "results" / "evaluation_results.csv"
REPORT_FILE = PROJECT_ROOT / "reports" / "router_evaluation_report.md"
UNRESOLVED_STATUSES = {"insufficient_evidence", "not_found", "error", "ambiguous", "email_not_found"}


def _result_payload(result: dict) -> tuple[str, dict, int]:
    """Extract readable output from text answers and structured tool results."""
    payload = result.get("answer")
    if isinstance(payload, dict):
        answer_text = payload.get("answer", payload.get("draft", payload.get("message", "")))
        if not answer_text:
            # Data and schedule tools return records rather than a prose `answer`.
            # Serialize their real tool result so valid lookup responses are not
            # incorrectly scored as empty responses by the evaluator.
            answer_text = json.dumps(payload, ensure_ascii=False, default=str)
        return str(answer_text or ""), payload, len(payload.get("sources", []) or [])
    if isinstance(payload, str):
        return payload, {}, 0
    draft = result.get("draft", result.get("message", ""))
    return str(draft or ""), {}, 0


def evaluate_question(row: dict) -> dict:
    question = str(row.get("question", "")).strip()
    category = str(row.get("category", "")).strip().lower()
    expected_route = str(row.get("expected_route", "")).strip()
    expected_behavior = str(row.get("expected_behavior", "")).strip()

    try:
        result = route_question(question)
        actual_route = str(result.get("route", "unknown"))
        outer_status = str(result.get("status", "unknown")).lower()
        answer_text, payload, source_count = _result_payload(result)
        inner_status = str(payload.get("status", "")).lower()
        effective_status = inner_status if inner_status in UNRESOLVED_STATUSES else outer_status
        route_correct = actual_route == expected_route or (
            category == "unsupported"
            and expected_route == "academic_helpdesk_coordinator"
            and actual_route in {"academic_helpdesk_coordinator", "academic_rag_agent"}
        )
        citation_refs = sorted({int(n) for n in re.findall(r"\[Source\s+(\d+)\]", answer_text, re.I)})
        citation_present = bool(citation_refs)
        citation_refs_valid = bool(citation_refs) and all(1 <= n <= source_count for n in citation_refs)
        citation_refs_valid = citation_refs_valid if category == "rag" else None

        workflow_passed = route_correct and outer_status == "success" and bool(answer_text.strip())
        refusal_detected = effective_status in {"insufficient_evidence", "not_found"} or bool(
            re.search(
                r"could not determine|insufficient evidence|do not have enough evidence|"
                r"cannot verify|can't verify|cannot be verified|not (?:currently )?available|"
                r"cannot disclose private|don't know|do not know|no matching .* found",
                answer_text,
                re.I,
            )
        )

        # If the rubric explicitly expects a refusal or not-found when evidence
        # is unavailable (e.g., Autonomous Systems slot not in AY 2023-24 first-year
        # timetable, or current-semester Humanities count), that is a valid result.
        expected_insufficient = any(
            phrase in expected_behavior.casefold()
            for phrase in (
                "insufficient evidence",
                "rather than guessing",
                "cannot be established",
                "rather than assuming",
                "preserve the semester/year shown in the data",
            )
        )
        if expected_insufficient and effective_status in {"insufficient_evidence", "not_found"}:
            workflow_passed = route_correct and bool(answer_text.strip())

        expected_not_found = any(
            phrase in expected_behavior.casefold()
            for phrase in ("no matching", "no record", "does not exist", "without inventing")
        )
        if expected_not_found and effective_status == "not_found":
            workflow_passed = route_correct and bool(answer_text.strip())

        # Unsupported prompts must be refused.
        if category == "unsupported":
            workflow_passed = route_correct and refusal_detected

        # Substantive RAG answers are checked separately for citations.
        if category == "rag":
            workflow_passed = route_correct and effective_status == "success" and bool(answer_text.strip())

        timestamp = datetime.now(timezone.utc).isoformat(timespec="seconds")
        return {
            "timestamp_utc": timestamp,
            "evaluation_mode": "deterministic_router_tool_path",
            "id": row.get("id", ""),
            "category": category,
            "question": question,
            "expected_route": expected_route,
            "actual_route": actual_route,
            "route_correct": route_correct,
            "status": effective_status,
            "workflow_passed": bool(workflow_passed),
            "passed": bool(workflow_passed),
            "citation_present": citation_present if category == "rag" else None,
            "citation_refs": ",".join(map(str, citation_refs)),
            "source_count": source_count,
            "citation_refs_valid": citation_refs_valid,
            "refusal_detected": refusal_detected if category == "unsupported" else None,
            "human_accuracy_review": "required" if category == "rag" and effective_status == "success" else "not_scored",
            "response": answer_text[:3000],
            "error": result.get("error", result.get("details", "")),
            "expected_behavior": expected_behavior,
        }
    except Exception as exc:
        return {
            "timestamp_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "evaluation_mode": "deterministic_router_tool_path",
            "id": row.get("id", ""),
            "category": category,
            "question": question,
            "expected_route": expected_route,
            "actual_route": "error",
            "route_correct": False,
            "status": "error",
            "workflow_passed": False,
            "passed": False,
            "citation_present": False if category == "rag" else None,
            "citation_refs": "",
            "source_count": 0,
            "citation_refs_valid": False if category == "rag" else None,
            "refusal_detected": False if category == "unsupported" else None,
            "human_accuracy_review": "required" if category == "rag" else "not_scored",
            "response": "",
            "error": f"{type(exc).__name__}: {exc}",
            "expected_behavior": expected_behavior,
        }


def _write_report(results_df: pd.DataFrame) -> None:
    total = len(results_df)
    passed = int(results_df["workflow_passed"].sum()) if total else 0
    rag = results_df[results_df["category"] == "rag"]
    cited = int(rag["citation_present"].fillna(False).sum()) if len(rag) else 0
    valid = int(rag["citation_refs_valid"].fillna(False).sum()) if len(rag) else 0
    unsupported = results_df[results_df["category"] == "unsupported"]
    refused = int(unsupported["refusal_detected"].fillna(False).sum()) if len(unsupported) else 0

    lines = [
        "# Router/tool-path evaluation report",
        "",
        f"- Run timestamp (UTC): {datetime.now(timezone.utc).isoformat(timespec='seconds')}",
        "- Evaluation mode: deterministic router and tools; **not** end-to-end ADK multiagent execution.",
        f"- Workflow checks passed: {passed}/{total} ({(passed / total * 100) if total else 0:.1f}%).",
        f"- RAG answers with at least one `[Source N]` marker: {cited}/{len(rag)}.",
        f"- RAG answers whose numeric source references are within the returned source count: {valid}/{len(rag)}.",
        f"- Unsupported questions detected as refusals: {refused}/{len(unsupported)}.",
        "",
        "## Category results",
        "",
        "| Category | Passed | Total |", "|---|---:|---:|",
    ]
    for category, group in results_df.groupby("category", dropna=False):
        lines.append(f"| {category} | {int(group['workflow_passed'].sum())} | {len(group)} |")
    lines.extend([
        "",
        "## Interpretation and limitations",
        "",
        "These automatic checks verify routing/status behavior and citation-marker presence. They do **not** establish factual accuracy or semantic groundedness; those require a human reviewer to check each answer against the cited document. Use `scripts/run_adk_evaluation.py` to test the actual ADK coordinator and observe whether the intended specialist agent appears in the event stream.",
        "",
        "RAG source-reference counts are structural checks only. A valid `[Source 1]` marker does not prove that Source 1 supports the claim.",
        "",
    ])
    REPORT_FILE.parent.mkdir(parents=True, exist_ok=True)
    REPORT_FILE.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    if not INPUT_FILE.is_file():
        raise FileNotFoundError(f"Evaluation dataset not found: {INPUT_FILE}")
    questions_df = pd.read_csv(INPUT_FILE).fillna("")
    rows = questions_df.to_dict("records")
    results = []
    for i, row in enumerate(rows, start=1):
        print(f"[{i}/{len(rows)}] {row['question']}")
        result = evaluate_question(row)
        results.append(result)
        print(
            f"  route={result['actual_route']} status={result['status']} "
            f"workflow_passed={result['workflow_passed']}"
        )
        if result["error"]:
            print(f"  error={result['error']}")

    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    result_df = pd.DataFrame(results)
    result_df.to_csv(OUTPUT_FILE, index=False)
    _write_report(result_df)

    total = len(result_df)
    passed = int(result_df["workflow_passed"].sum()) if total else 0
    print("\nRouter/tool-path evaluation summary")
    print(f"Total: {total}; workflow checks passed: {passed}; failed: {total - passed}")
    print(f"Pass rate: {(passed / total * 100) if total else 0:.1f}%")
    print(f"Results: {OUTPUT_FILE}")
    print(f"Report: {REPORT_FILE}")
    print("Reminder: this is not the end-to-end ADK multiagent score.")


if __name__ == "__main__":
    main()
