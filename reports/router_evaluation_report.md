# Router/tool-path evaluation report

- Run timestamp (UTC): 2026-10-29T16:45:00+00:00
- Evaluation mode: deterministic router and tools (`scripts/run_evaluation.py`).
- Workflow checks passed: 30/30 (100.0%).
- RAG answers with at least one `[Source N]` marker: 7/7 (100.0%).
- RAG answers whose numeric source references are within the returned source count: 7/7 (100.0%).
- Unsupported questions detected as refusals: 4/4 (100.0%).

## Category results

| Category | Passed | Total |
|---|---:|---:|
| course | 6 | 6 |
| email | 4 | 4 |
| faculty | 4 | 4 |
| rag | 7 | 7 |
| timetable | 5 | 5 |
| unsupported | 4 | 4 |

## Interpretation and limitations

These checks verify deterministic routing, tool execution, citation-marker presence, and refusal behavior across all 30 evaluation prompts. See `reports/adk_evaluation_report_qwen3_1.7b.md` and `reports/final_evaluation_report.md` for end-to-end Google ADK multi-agent evaluation and human-verified groundedness analysis.
