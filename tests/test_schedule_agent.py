from src.agents import schedule_agent


def test_lookup_schedule_returns_historical_rows_with_warning(monkeypatch):
    monkeypatch.setattr(
        schedule_agent,
        "search_timetable",
        lambda query, semester="", group_section="", limit=20: [
            {"Course_Code": query, "Semester": semester, "Time_Slot": "A"}
        ],
    )
    result = schedule_agent.lookup_schedule("BBL1020", semester="AY 2023-24 Semester-I")
    assert result["status"] == "success"
    assert result["results"][0]["Semester"] == "AY 2023-24 Semester-I"
    assert "historical timetable" in result["warning"]


def test_lookup_schedule_reports_no_match(monkeypatch):
    monkeypatch.setattr(schedule_agent, "search_timetable", lambda *args, **kwargs: [])
    result = schedule_agent.lookup_schedule("XYZ9999")
    assert result["status"] == "not_found"
    assert result["results"] == []
