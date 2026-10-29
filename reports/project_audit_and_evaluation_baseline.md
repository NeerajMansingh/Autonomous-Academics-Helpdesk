# Project audit and prior evaluation baseline

## What was inspected

- The repository source includes the ADK coordinator and three specialist agents, structured CSV tools, a Chroma-based RAG answer tool, 25 original evaluation prompts, and a previously saved evaluation CSV.
- The supplied extraction report lists 91 PDFs, all marked successful: 37 circulars, 31 curriculum documents, 12 academic calendars, and 11 regulations.
- The supplied archive did not include the `data/` tree, the structured CSVs, the processed text, or the persistent vector store. Therefore this review could not run the real local model, rebuild/query the real Chroma index, or verify actual ADK handoffs against those assets.

## Prior saved result (not a final score)

- The saved `evaluation/results/evaluation_results.csv` contains 25 records and marks 22/25 as passed (88.0%).
- By category in that saved file:

| Category | Passed | Total |
|---|---:|---:|
| course | 5 | 5 |
| email | 3 | 3 |
| faculty | 4 | 4 |
| rag | 4 | 6 |
| timetable | 3 | 4 |
| unsupported | 3 | 3 |

## Known interpretation issues

- The saved CSV predates the current evaluator format and does not include the expected-behavior text or citation-check fields.
- The saved results mark the nonexistent-course timetable question as failed even though the tool correctly returned `not_found`; the updated router evaluator recognizes an expected not-found outcome when the prompt rubric asks the system to report no matching record.
- Two RAG rows in the saved results (IDs 17 and 18) contain an Ollama connection error. Separate tests in the prior working session showed the RAG tool could return successful responses for those prompts, so these saved failures need to be rerun, not silently ignored.
- The saved 88% is a deterministic router/tool-path score, not a measured end-to-end multiagent accuracy score. It does not prove factual correctness or source groundedness.

## Changes prepared in this review copy

- Added source code for the unanswered-question CSV logger and integrated it into deterministic unresolved outcomes; the ADK coordinator also exposes a logging tool for refusals and unresolved specialist responses.
- Removed the manually written attendance-answer branch so the RAG answer path is general-purpose. The prompt now instructs the model to handle conflicting/superseded sources generally.
- Changed retrieval to deduplicate exact text and limit overrepresentation by underlying document rather than discard all chunks sharing one page label.
- Added a dedicated ADK schedule specialist alongside the course/faculty data agent, RAG policy agent, and email agent.
- Added data-driven course-count and instructor-contact tools. A semester-specific count requires an explicit semester; historical data is not treated as current by inference.
- Added the sample course-slot, open-elective-degree-credit, course-instructor email, current-semester count, and campus-dining refusal prompts, bringing the evaluation dataset to 30 prompts.
- Added a separate ADK evaluation script that observes which specialist authors appear during real ADK execution and can repeat prompts to detect answer variation. Fixed the deterministic evaluator to serialize structured course/faculty/schedule tool results rather than treating them as empty answers.

## Validation completed in the review runtime

- `python -m py_compile` passed for all Python files in this review copy.
- 4 direct tests for unanswered logging, structured counting, semester-scope refusal, and exact instructor contact matching passed.
- 13 additional router/agent-wrapper/evaluator tests passed with ADK/RAG dependencies stubbed, allowing deterministic routing logic and result extraction to be tested without the absent local packages. All 17 reviewed unit checks passed; the 13 routing/agent-wrapper/evaluator checks used ADK/RAG dependency stubs. These are unit checks, not an end-to-end ADK run.
- A full ADK/Ollama evaluation was not run in this review runtime because this archive omitted the data/vectorstore and the runtime does not have `google.adk`, `litellm`, `chromadb`, or `sentence_transformers` installed.

## Required final verification

1. Apply this code update over the existing local project without deleting its `data/` or `vectorstore/` folders.
2. Run `python -m pytest tests -v`.
3. Run `python scripts\run_evaluation.py` for the deterministic baseline and `python scripts\run_adk_evaluation.py --limit 3` to confirm real specialist delegation.
4. If handoffs are correct, run `python scripts\run_adk_evaluation.py --runs 1` for all 30 prompts and `--runs 3` for run-to-run consistency. Keep the generated reports and manually review each factual RAG answer against its cited source.
5. Compare at least two locally installed model tags using separate output files and state the exact model used for each run.

## Submission caveats

- Because the extraction report lists 91 PDFs while the brief specifies 20–30 documents, confirm whether the instructor treats 20–30 as a target range or minimum. If the range is strict, select 20–30 source documents transparently and rebuild the index from that selection.
- Do not present citation-marker counts as semantic groundedness. A human must verify that cited passages actually support each claim.
