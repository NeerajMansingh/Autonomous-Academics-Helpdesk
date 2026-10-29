from src.agents import router


def test_course_question_routes_to_data_agent(monkeypatch):
    monkeypatch.setattr(router, "lookup_courses", lambda query: {"status": "success", "results": [{"Course_Code": query}]})
    result = router.route_question("Find course BBL1020 and tell me its title and credits")
    assert result["route"] == "academic_data_agent"
    assert result["status"] == "success"


def test_course_title_slot_question_routes_to_timetable(monkeypatch):
    seen = {}
    def fake_lookup(query, semester="", group_section=""):
        seen.update(query=query, semester=semester)
        return {"status": "success", "results": [{"Course_Name": query, "Time_Slot": "A"}]}
    monkeypatch.setattr(router, "lookup_schedule", fake_lookup)
    result = router.route_question("What is the slot for the course Autonomous Systems?")
    assert result["route"] == "academic_schedule_agent"
    assert result["status"] == "success"
    assert seen["query"].casefold() == "autonomous systems"


def test_exact_timetable_semester_is_passed_to_tool(monkeypatch):
    seen = {}
    def fake_lookup(query, semester="", group_section=""):
        seen.update(query=query, semester=semester)
        return {"status": "success", "results": [{"Course_Code": query}]}
    monkeypatch.setattr(router, "lookup_schedule", fake_lookup)
    result = router.route_question("Find the timetable for BBL1020 in AY 2023-24 Semester-I")
    assert result["route"] == "academic_schedule_agent"
    assert seen["query"] == "BBL1020"
    assert seen["semester"].casefold() == "ay 2023-24 semester-i"


def test_degree_open_elective_credits_route_to_rag(monkeypatch):
    monkeypatch.setattr(router, "search_academic_documents", lambda question: {"status": "success", "answer": "Supported answer [Source 1]", "sources": [{"source": "regulations.pdf"}]})
    result = router.route_question("How many Open Elective credits do I need to earn to get a B.Tech degree?")
    assert result["route"] == "academic_rag_agent"
    assert result["status"] == "success"


def test_faculty_query_removes_request_wording(monkeypatch):
    seen = {}
    def fake_lookup(query):
        seen["query"] = query
        return {"status": "success", "results": [{"Name": "Anurag Sharma"}]}
    monkeypatch.setattr(router, "lookup_faculty", fake_lookup)
    result = router.route_question("Find faculty email addresses for composite-material researchers")
    assert result["route"] == "academic_data_agent"
    assert "composite material" in seen["query"].casefold()


def test_course_offering_count_does_not_assume_current_semester(monkeypatch):
    monkeypatch.setattr(router, "count_catalog_courses", lambda **kwargs: {"status": "insufficient_evidence", "message": "Exact term required", "count": None, "filters": kwargs})
    monkeypatch.setattr(router, "log_unanswered_question", lambda *args, **kwargs: None)
    result = router.route_question("How many 400 level electives have been floated by the Humanities department in this semester?")
    assert result["route"] == "academic_data_agent"
    assert result["status"] == "insufficient_evidence"
    assert result["answer"]["filters"]["department"] == "Humanities"
    assert result["answer"]["filters"]["level"] == "400"
    assert result["answer"]["filters"]["require_timetable"] is True


def test_email_question_routes_to_email_agent(monkeypatch):
    monkeypatch.setattr(router, "draft_email", lambda question: {"status": "success", "route": "academic_email_agent", "draft": "Subject: Leave\nDear Instructor,"})
    result = router.route_question("Write an email to the course instructor of CSL7570 that I need leave for 2 days")
    assert result["route"] == "academic_email_agent"
    assert "Subject:" in result["draft"]


def test_unsupported_question_is_logged(monkeypatch):
    calls = []
    monkeypatch.setattr(router, "search_academic_documents", lambda question: {"status": "insufficient_evidence", "answer": "I could not determine this reliably from the available sources.", "sources": []})
    monkeypatch.setattr(router, "log_unanswered_question", lambda *args, **kwargs: calls.append((args, kwargs)))
    result = router.route_question("What are the available dining options on campus?")
    assert result["route"] == "academic_rag_agent"
    assert result["status"] == "insufficient_evidence"
    assert len(calls) == 1
