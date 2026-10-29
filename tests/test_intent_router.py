from src.agents.intent_router import (
    DATA_AGENT,
    EMAIL_AGENT,
    RAG_AGENT,
    SCHEDULE_AGENT,
    classify_request,
)


def test_course_lookup_routes_to_data_specialist():
    assert classify_request("Find course BBL1020 and tell me its title and credits") == DATA_AGENT


def test_slot_query_routes_to_schedule_even_with_course_name():
    assert classify_request("What is the slot for the course Autonomous Systems?") == SCHEDULE_AGENT


def test_policy_question_routes_to_rag_specialist():
    assert classify_request("What is the weightage and duration of the Major examination?") == RAG_AGENT


def test_academic_calendar_routes_to_rag_specialist():
    assert classify_request("What does the academic calendar say about the semester dates?") == RAG_AGENT


def test_faculty_research_routes_to_data_specialist():
    assert classify_request("Find the research areas of Anurag Sharma") == DATA_AGENT
    assert classify_request("Find faculty email addresses for composite-material researchers") == DATA_AGENT


def test_email_intent_takes_precedence_over_course_code():
    assert classify_request(
        "Write an email to the course instructor of CSL7570 saying I need sick leave for 2 days"
    ) == EMAIL_AGENT


def test_term_specific_course_count_routes_to_data_specialist():
    assert classify_request(
        "How many 400 level electives have been floated by the Humanities department in this semester?"
    ) == DATA_AGENT


def test_course_content_uses_document_retrieval():
    assert classify_request("What course content is covered in BBL1020?") == RAG_AGENT


def test_out_of_scope_question_has_no_specialist_route():
    assert classify_request("What are the available dining options on campus?") is None
    assert classify_request("What is the best way to reach campus from Jodhpur city center?") is None


def test_generic_academic_question_falls_back_to_rag():
    assert classify_request("Can you explain this academic requirement?") == RAG_AGENT


def test_unsupported_assignment_fee_question_still_routes_to_rag():
    assert classify_request(
        "What is the late fee for submitting an assignment after the deadline?"
    ) == RAG_AGENT
