import types
from src.agents.response_guards import (
    RAG_RESULT_STATE_KEY,
    append_retrieved_source_list,
    is_academic_helpdesk_query,
    preserve_rag_citations,
    refusal_text,
    strip_thinking_text,
)


def test_scope_guard_accepts_representative_academic_topics():
    prompts = [
        "Find course BBL1020 and tell me its title and credits",
        "What is the slot for the course Autonomous Systems?",
        "What is the weightage and duration of the Major examination?",
        "Write an email to my course instructor asking for sick leave",
        "How many Open Elective credits do I need for a B.Tech. degree?",
        "Find the research areas of Anurag Sharma",
        "What is the late fee for submitting an assignment after the deadline?",
        "What does the academic calendar say about the semester dates?",
    ]
    assert all(is_academic_helpdesk_query(prompt) for prompt in prompts)


def test_scope_guard_rejects_out_of_scope_examples():
    prompts = [
        "What are the available dining options on the campus?",
        "What is the best way to reach campus from Jodhpur city center?",
    ]
    assert all(not is_academic_helpdesk_query(prompt) for prompt in prompts)
    assert "I don't know" in refusal_text()


def test_source_list_is_added_when_rag_answer_omits_citations():
    result = {
        "status": "success",
        "sources": [
            {"source": "Academic-Regulations.pdf (Page 3)"},
            {"source": "Examination-Rules.pdf (Page 2)"},
        ],
    }
    answer = append_retrieved_source_list("The examination has a stated weightage and duration.", result)
    assert "[Source 1] Academic-Regulations.pdf (Page 3)" in answer
    assert "[Source 2] Examination-Rules.pdf (Page 2)" in answer


def test_source_list_is_added_to_citations_but_not_refusals():
    result = {"status": "success", "sources": [{"source": "Rules.pdf"}]}
    refusal = "I could not determine this reliably from the available sources."
    already_cited = "The rule is described in the document [Source 1]."
    assert append_retrieved_source_list(refusal, result) == refusal
    cited_with_source_list = append_retrieved_source_list(already_cited, result)
    assert "[Source 1]" in cited_with_source_list
    assert "Rules.pdf" in cited_with_source_list


def test_preserve_rag_citations_excludes_thought_parts():
    ctx = types.SimpleNamespace(
        state={
            RAG_RESULT_STATE_KEY: {
                "status": "success",
                "sources": [{"source": "Regulations.pdf (Page 7)"}],
            }
        }
    )
    thought_part = types.SimpleNamespace(
        text="Okay, let's break down the user's question about exam weightage.",
        thought=True,
        function_call=None,
    )
    answer_part = types.SimpleNamespace(
        text="The Major examination has a minimum weightage of 40% [Source 1].",
        thought=False,
        function_call=None,
    )
    llm_response = types.SimpleNamespace(
        content=types.SimpleNamespace(parts=[thought_part, answer_part])
    )
    preserve_rag_citations(ctx, llm_response)
    assert len(llm_response.content.parts) == 1
    final_text = llm_response.content.parts[0].text
    assert "Okay, let's break down" not in final_text
    assert "The Major examination has a minimum weightage of 40%" in final_text
    assert "[Source 1] Regulations.pdf (Page 7)" in final_text


def test_strip_thinking_text_removes_self_talk_and_think_tags():
    raw = (
        "Okay, let's break down the user's question about Open Elective credits.\n\n"
        "The regular B.Tech programme requires 12 credits of Open Electives [Source 1]."
    )
    cleaned = strip_thinking_text(raw)
    assert "Okay, let's break down" not in cleaned
    assert cleaned.startswith("The regular B.Tech programme requires 12 credits")
