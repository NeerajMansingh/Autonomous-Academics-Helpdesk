from scripts.run_evaluation import _result_payload


def test_structured_tool_results_are_not_scored_as_empty_answers():
    text, payload, source_count = _result_payload(
        {
            "answer": {
                "status": "success",
                "results": [{"Course_Code": "BBL1020", "Time_Slot": "A"}],
            }
        }
    )

    assert "BBL1020" in text
    assert payload["status"] == "success"
    assert source_count == 0
