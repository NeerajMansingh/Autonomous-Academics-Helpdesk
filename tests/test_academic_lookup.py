import pandas as pd

from src.tools import academic_lookup


def fixture_data():
    catalog = pd.DataFrame([
        {"Course_Code": "HSS4010", "Course_Title": "Elective A", "Department": "Humanities", "Program_Level": "UG", "Credits": "3", "Category": "Elective", "Focus_Area_Syllabus": ""},
        {"Course_Code": "HSS4020", "Course_Title": "Elective B", "Department": "Humanities", "Program_Level": "UG", "Credits": "3", "Category": "Elective", "Focus_Area_Syllabus": ""},
        {"Course_Code": "CSL4010", "Course_Title": "Systems", "Department": "Computer Science", "Program_Level": "UG", "Credits": "4", "Category": "Core", "Focus_Area_Syllabus": ""},
    ])
    timetable = pd.DataFrame([
        {"Course_Code": "HSS4010", "Course_Name": "Elective A", "Instructor": "Dr Anurag Sharma", "Day": "Mon", "Time_Slot": "A", "Slot_Type": "A", "Semester": "AY 2023-24 Semester-I", "Group_Section": "Group A1"},
        {"Course_Code": "HSS4020", "Course_Name": "Elective B", "Instructor": "Dr Anurag Sharma", "Day": "Tue", "Time_Slot": "B", "Slot_Type": "B", "Semester": "AY 2023-24 Semester-I", "Group_Section": "Group A1"},
    ])
    faculty = pd.DataFrame([
        {"Name": "Anurag Sharma", "Email": "anurag@example.edu", "Department": "Humanities", "Designation": "Professor", "Research_Areas": "Advanced Composite Materials; Policy", "Profile_URL": ""},
    ])
    return {"course_catalog.csv": catalog, "course_timetable.csv": timetable, "faculty_contacts.csv": faculty}


def test_count_requires_exact_term_for_offering_count(monkeypatch):
    data = fixture_data()
    monkeypatch.setattr(academic_lookup, "_load_csv", lambda filename: data[filename])

    result = academic_lookup.count_courses(
        department="Humanities", level="400", category="elective", require_timetable=True
    )

    assert result["status"] == "insufficient_evidence"
    assert result["count"] is None


def test_count_treats_relative_semester_as_insufficient_evidence(monkeypatch):
    data = fixture_data()
    monkeypatch.setattr(academic_lookup, "_load_csv", lambda filename: data[filename])

    result = academic_lookup.count_courses(
        department="Humanities", level="400", category="elective", semester="this semester", require_timetable=False
    )

    assert result["status"] == "insufficient_evidence"
    assert result["count"] is None


def test_count_filters_catalog_and_explicit_semester(monkeypatch):
    data = fixture_data()
    monkeypatch.setattr(academic_lookup, "_load_csv", lambda filename: data[filename])

    result = academic_lookup.count_courses(
        department="Humanities",
        level="400",
        category="elective",
        semester="AY 2023-24 Semester-I",
        require_timetable=True,
    )

    assert result["status"] == "success"
    assert result["count"] == 2
    assert {row["Course_Code"] for row in result["courses"]} == {"HSS4010", "HSS4020"}


def test_instructor_contact_only_returns_exact_published_email(monkeypatch):
    data = fixture_data()
    monkeypatch.setattr(academic_lookup, "_load_csv", lambda filename: data[filename])

    result = academic_lookup.lookup_instructor_contact("HSS4010")

    assert result["status"] == "success"
    assert result["instructor"] == "Dr Anurag Sharma"
    assert result["email"] == "anurag@example.edu"
    assert result["semester_records"] == ["AY 2023-24 Semester-I"]


def test_search_timetable_ignores_spurious_group_and_relative_semester(monkeypatch):
    data = fixture_data()
    monkeypatch.setattr(academic_lookup, "_load_csv", lambda filename: data[filename])

    rows = academic_lookup.search_timetable("HSS4010", semester="current", group_section="HSS4010")
    assert len(rows) == 1
    assert rows[0]["Course_Code"] == "HSS4010"


def test_search_faculty_handles_hyphenated_researcher_phrases(monkeypatch):
    data = fixture_data()
    monkeypatch.setattr(academic_lookup, "_load_csv", lambda filename: data[filename])

    rows = academic_lookup.search_faculty("composite-material researchers")
    assert len(rows) == 1
    assert rows[0]["Name"] == "Anurag Sharma"
