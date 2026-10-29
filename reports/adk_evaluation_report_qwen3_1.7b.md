# End-to-end ADK multiagent evaluation

- Timestamp (UTC): 2026-10-29T19:07:30+00:00
- Model: `qwen3:1.7b`
- Runs per question: 1
- Questions evaluated per run: 30
- Expected handler / refusal checks passed: 30/30 (100.0%).
- RAG responses with citation markers: 3/7 run rows.
- Unsupported prompts detected as refusals: 4/4 run rows.
- Questions whose answer changed across repeated runs: 0.

## Limitations

Routing checks mean the expected coordinator or specialist appeared in the ADK event stream; unsupported-category rows also require an explicit refusal. These checks do not prove that the final answer is factually correct. Citation-marker presence is not semantic groundedness. Manually compare each response and cited source with the expected behavior. Answer-change fingerprints normalize whitespace and case only; stylistic changes are still counted as changes.

Detailed CSV: `evaluation/results/adk_evaluation_qwen3_1.7b.csv`
