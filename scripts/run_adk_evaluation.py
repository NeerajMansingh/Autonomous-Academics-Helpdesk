"""Run end-to-end ADK multiagent evaluation and observe specialist handoffs.

This script requires the project dependencies, the local Ollama service, the
configured model, the CSV inputs, and a built Chroma index. Run from the
project root. It does not replace human assessment of factual accuracy.
"""

from __future__ import annotations

import argparse
import asyncio
import os
import re
import sys
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))
INPUT_FILE = PROJECT_ROOT / "evaluation" / "test_questions.csv"

try:
    from src.agents.response_guards import strip_thinking_text
except ModuleNotFoundError:
    def strip_thinking_text(text: str) -> str:
        return str(text or "").strip()


def normalize_answer(text: str) -> str:
    """Normalize an answer for repeated-run consistency comparisons."""
    return re.sub(r"\s+", " ", strip_thinking_text(str(text or ""))).strip().casefold()


def refusal_detected(text: str) -> bool:
    """Detect common refusal and insufficient-evidence statements."""
    return bool(
        re.search(
            r"could not determine|insufficient evidence|not enough evidence|"
            r"cannot verify|can't verify|cannot be verified|unable to verify|"
            r"not (?:currently )?available in the (?:system|sources|collected|available|dataset)|"
            r"not (?:currently )?available|do not know|don't know|cannot disclose private|"
            r"no matching .* found|not found in the (?:available|collected)|"
            r"out of scope|outside the scope|helpdesk handles|"
            r"logged this request for human review",
            text,
            re.I,
        )
    )


async def run_turn(
    runner,
    session_service,
    app_name: str,
    question: str,
    session_key: str,
) -> dict:
    """Run one question and select the final user-facing text event.

    ADK streams intermediate events too, including coordinator text that may
    accompany a transfer or tool call, as well as `thought=True` parts from
    reasoning models. Filter out `thought=True` parts and prefer the final
    response from the active specialist.
    """
    from google.genai import types

    user_id = "academic_helpdesk_evaluation"

    session = await session_service.create_session(
        app_name=app_name,
        user_id=user_id,
        session_id=session_key,
    )

    authors = []
    text_events = []

    async for event in runner.run_async(
        user_id=user_id,
        session_id=session.id,
        new_message=types.Content(
            role="user",
            parts=[types.Part.from_text(text=question)],
        ),
    ):
        author = str(getattr(event, "author", ""))

        if author and author not in authors:
            authors.append(author)

        try:
            is_final = bool(event.is_final_response())
        except (AttributeError, TypeError):
            is_final = False

        content = getattr(event, "content", None)

        for part in getattr(content, "parts", []) or []:
            if getattr(part, "thought", False):
                continue
            part_text = getattr(part, "text", None)

            if part_text:
                cleaned = strip_thinking_text(str(part_text))
                if cleaned:
                    text_events.append(
                        (author, cleaned, is_final)
                    )

    # First preference: the final response marked by ADK.
    final_events = [
        entry for entry in text_events
        if entry[2]
    ]

    if final_events:
        final_text = final_events[-1][1]

    else:
        # Fallback for ADK versions that do not provide reliable final markers.
        # Prefer a specialist response instead of coordinator narration.
        coordinator_name = "academic_helpdesk_coordinator"

        specialist_text = [
            text
            for author, text, _ in text_events
            if author and author != coordinator_name
        ]

        if specialist_text:
            final_text = specialist_text[-1]
        else:
            final_text = (
                text_events[-1][1]
                if text_events
                else ""
            )

    return {
        "authors": authors,
        "response": strip_thinking_text(final_text),
    }


async def main_async(args) -> None:
    # Set the model before importing agent modules so all agents use the
    # same configured local model.
    os.environ["ACADEMIC_HELPDESK_MODEL"] = args.model

    from google.adk.runners import Runner
    from google.adk.sessions import InMemorySessionService
    from src.agents.academic_agent import root_agent

    app_name = "academic_helpdesk_evaluation"

    session_service = InMemorySessionService()

    runner = Runner(
        agent=root_agent,
        app_name=app_name,
        session_service=session_service,
    )

    questions_df = pd.read_csv(INPUT_FILE).fillna("")
    records = questions_df.to_dict("records")

    if args.smoke:
        smoke_ids = {"1", "6", "14", "29", "30"}
        records = [
            row
            for row in records
            if str(row.get("id", "")).strip() in smoke_ids
        ]

    if args.limit:
        records = records[:args.limit]

    rows = []

    for idx, row in enumerate(records, start=1):
        for run_number in range(1, args.runs + 1):
            question = str(row["question"]).strip()
            session_key = (
                f"eval-{row.get('id', idx)}-{run_number}-{idx}"
            )

            print(
                f"[{idx}/{len(records)}] "
                f"run {run_number}/{args.runs}: {question}"
            )

            try:
                result = await run_turn(
                    runner,
                    session_service,
                    app_name,
                    question,
                    session_key,
                )

                authors = result["authors"]
                response = result["response"]

                expected_agent = str(
                    row["expected_route"]
                ).strip()

                expected_handler_seen = expected_agent in authors
                category = str(row["category"]).strip().lower()

                cite_refs = sorted(
                    {
                        int(value)
                        for value in re.findall(
                            r"\[Source\s+(\d+)\]",
                            response,
                            re.I,
                        )
                    }
                )

                refused = refusal_detected(response)

                workflow_passed = (
                    expected_handler_seen
                    and bool(response.strip())
                    and (
                        category != "unsupported"
                        or refused
                    )
                )

                rows.append(
                    {
                        "timestamp_utc": datetime.now(
                            timezone.utc
                        ).isoformat(timespec="seconds"),
                        "model": args.model,
                        "run_number": run_number,
                        "id": row.get("id", ""),
                        "category": category,
                        "question": question,
                        "expected_agent": expected_agent,
                        "observed_authors": " | ".join(authors),
                        "expected_handler_seen": expected_handler_seen,
                        "response_nonempty": bool(response.strip()),
                        "workflow_passed": workflow_passed,
                        "citation_present": (
                            bool(cite_refs)
                            if category == "rag"
                            else None
                        ),
                        "citation_refs": ",".join(
                            map(str, cite_refs)
                        ),
                        "refusal_detected": (
                            refused
                            if category == "unsupported"
                            else None
                        ),
                        "answer_fingerprint": normalize_answer(
                            response
                        ),
                        "response": response[:4000],
                        "expected_behavior": str(
                            row.get("expected_behavior", "")
                        ),
                        "human_accuracy_review": "required",
                        "error": "",
                    }
                )

                print(
                    f"  expected_handler={expected_handler_seen}; "
                    f"authors={authors}"
                )

            except Exception as exc:
                category = str(
                    row.get("category", "")
                ).strip().lower()

                rows.append(
                    {
                        "timestamp_utc": datetime.now(
                            timezone.utc
                        ).isoformat(timespec="seconds"),
                        "model": args.model,
                        "run_number": run_number,
                        "id": row.get("id", ""),
                        "category": category,
                        "question": question,
                        "expected_agent": str(
                            row.get("expected_route", "")
                        ).strip(),
                        "observed_authors": "",
                        "expected_handler_seen": False,
                        "response_nonempty": False,
                        "workflow_passed": False,
                        "citation_present": (
                            False if category == "rag" else None
                        ),
                        "citation_refs": "",
                        "refusal_detected": (
                            False
                            if category == "unsupported"
                            else None
                        ),
                        "answer_fingerprint": "",
                        "response": "",
                        "expected_behavior": str(
                            row.get("expected_behavior", "")
                        ),
                        "human_accuracy_review": "required",
                        "error": f"{type(exc).__name__}: {exc}",
                    }
                )

                print(
                    f"  ERROR: {type(exc).__name__}: {exc}"
                )

    df = pd.DataFrame(rows)

    # Identify questions whose normalized answers differ across runs.
    if not df.empty:
        changed_ids = set()

        for _, group in df.groupby("id"):
            if (
                group["answer_fingerprint"].nunique(
                    dropna=False
                )
                > 1
            ):
                changed_ids.add(str(group.iloc[0]["id"]))

        df["answer_changed_across_runs"] = (
            df["id"].astype(str).isin(changed_ids)
        )

    safe_model = re.sub(
        r"[^A-Za-z0-9_.-]+",
        "_",
        args.model,
    )

    output = (
        Path(args.output)
        if args.output
        else (
            PROJECT_ROOT
            / "evaluation"
            / "results"
            / f"adk_evaluation_{safe_model}.csv"
        )
    )

    output.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output, index=False)

    total = len(df)
    passed = (
        int(df["workflow_passed"].sum())
        if total
        else 0
    )

    rag = df[df["category"] == "rag"]
    unsupported = df[df["category"] == "unsupported"]

    citations = (
        int(rag["citation_present"].fillna(False).sum())
        if len(rag)
        else 0
    )

    refusals = (
        int(unsupported["refusal_detected"].fillna(False).sum())
        if len(unsupported)
        else 0
    )

    changed_questions = (
        int(
            df.loc[
                df["answer_changed_across_runs"],
                "id",
            ].nunique()
        )
        if (
            total
            and "answer_changed_across_runs" in df.columns
        )
        else 0
    )

    report = (
        PROJECT_ROOT
        / "reports"
        / f"adk_evaluation_report_{safe_model}.md"
    )

    report.parent.mkdir(parents=True, exist_ok=True)

    report.write_text(
        "\n".join(
            [
                "# End-to-end ADK multiagent evaluation",
                "",
                (
                    f"- Timestamp (UTC): "
                    f"{datetime.now(timezone.utc).isoformat(timespec='seconds')}"
                ),
                f"- Model: `{args.model}`",
                f"- Runs per question: {args.runs}",
                f"- Questions evaluated per run: {len(records)}",
                (
                    f"- Expected handler / refusal checks passed: "
                    f"{passed}/{total} "
                    f"({(passed / total * 100) if total else 0:.1f}%)."
                ),
                (
                    f"- RAG responses with citation markers: "
                    f"{citations}/{len(rag)} run rows."
                ),
                (
                    f"- Unsupported prompts detected as refusals: "
                    f"{refusals}/{len(unsupported)} run rows."
                ),
                (
                    f"- Questions whose answer changed across repeated runs: "
                    f"{changed_questions}."
                ),
                "",
                "## Limitations",
                "",
                (
                    "Routing checks mean the expected coordinator or specialist "
                    "appeared in the ADK event stream; unsupported-category rows "
                    "also require an explicit refusal. These checks do not prove "
                    "that the final answer is factually correct. Citation-marker "
                    "presence is not semantic groundedness. Manually compare "
                    "each response and cited source with the expected behavior. "
                    "Answer-change fingerprints normalize whitespace and case "
                    "only; stylistic changes are still counted as changes."
                ),
                "",
                (
                    f"Detailed CSV: "
                    f"`{output.relative_to(PROJECT_ROOT).as_posix()}`"
                ),
                "",
            ]
        ),
        encoding="utf-8",
    )

    print("\nADK multiagent evaluation summary")
    print(f"Workflow routing/refusal checks passed: {passed}/{total}")
    print(f"RAG citation markers: {citations}/{len(rag)}")
    print(f"Unsupported refusals detected: {refusals}/{len(unsupported)}")
    print(f"Questions with changed answers across runs: {changed_questions}")
    print(f"CSV: {output}")
    print(f"Report: {report}")


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)

    parser.add_argument(
        "--model",
        default=os.getenv(
            "ACADEMIC_HELPDESK_MODEL",
            "qwen3:1.7b",
        ),
    )
    parser.add_argument(
        "--runs",
        type=int,
        default=1,
        help="Repeat each question this many times (use 3 for consistency analysis).",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=0,
        help="Evaluate only the first N prompts.",
    )
    parser.add_argument(
        "--smoke",
        action="store_true",
        help=(
            "Run representative course, timetable, policy, email, "
            "and out-of-scope prompts."
        ),
    )
    parser.add_argument(
        "--output",
        default="",
        help="Optional CSV output path.",
    )

    args = parser.parse_args()

    if args.runs < 1:
        parser.error("--runs must be at least 1")

    if not INPUT_FILE.is_file():
        parser.error(f"Missing evaluation dataset: {INPUT_FILE}")

    return args


if __name__ == "__main__":
    asyncio.run(main_async(parse_args()))