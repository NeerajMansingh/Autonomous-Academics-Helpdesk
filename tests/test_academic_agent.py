from src.agents import data_agent


def test_lookup_courses_wraps_structured_results(monkeypatch):
    monkeypatch.setattr(
        data_agent,
        "search_courses",
        lambda query, limit=10: [{"Course_Code": query, "Course_Title": "Test Course"}],
    )
    result = data_agent.lookup_courses("BBL1020")
    assert result["status"] == "success"
    assert result["results"][0]["Course_Code"] == "BBL1020"


def test_count_catalog_courses_uses_explicit_filters(monkeypatch):
    monkeypatch.setattr(
        data_agent,
        "count_courses",
        lambda **kwargs: {"status": "success", "count": 2, "filters": kwargs},
    )
    result = data_agent.count_catalog_courses(
        department="Humanities", level="400", category="elective"
    )
    assert result["status"] == "success"
    assert result["filters"]["department"] == "Humanities"
