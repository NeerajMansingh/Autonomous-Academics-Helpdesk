from pathlib import Path
import re
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]

# Use the collected dataset, not a guessed or separately created CSV.
DATA_DIR = (
    PROJECT_ROOT
    / "data"
    / "academic-helpdesk-data"
    / "academic-helpdesk"
    / "data"
    / "structured"
)

_COURSE_CODE_RE = re.compile(r"\b([A-Za-z]{2,5}\d{3,5}[A-Za-z]?)\b")
_PLACEHOLDER_VALUES = {
    "",
    "none",
    "null",
    "nil",
    "all",
    "any",
    "unspecified",
    "n/a",
    "na",
    "unknown",
    "default",
    "current",
    "current semester",
    "this semester",
    "present",
    "now",
}
_GENERIC_QUERY_STOPWORDS = {
    "find",
    "search",
    "show",
    "list",
    "tell",
    "me",
    "the",
    "a",
    "an",
    "for",
    "in",
    "of",
    "on",
    "to",
    "by",
    "from",
    "with",
    "and",
    "or",
    "what",
    "which",
    "who",
    "is",
    "are",
    "course",
    "courses",
    "catalog",
    "catalogue",
    "category",
    "department",
    "program",
    "programme",
    "level",
    "title",
    "credits",
    "credit",
    "related",
    "working",
    "faculty",
    "member",
    "members",
    "professor",
    "professors",
    "instructor",
    "instructors",
    "researcher",
    "researchers",
    "research",
    "area",
    "areas",
    "interest",
    "interests",
    "email",
    "emails",
    "address",
    "addresses",
    "contact",
    "contacts",
    "details",
    "timetable",
    "schedule",
    "slot",
    "time",
    "days",
    "day",
    "scheduled",
}


def _load_csv(filename: str) -> pd.DataFrame:
    """Load a CSV and normalize missing values."""
    path = DATA_DIR / filename

    if not path.is_file():
        raise FileNotFoundError(f"Required data file not found: {path}")

    df = pd.read_csv(path)
    return df.astype(object).where(pd.notna(df), "")


def _extract_meaningful_tokens(query: str) -> list[str]:
    """Extract normalized search tokens and basic singular stems."""
    cleaned = re.sub(r"[-_/,:;()\[\]]+", " ", str(query or "")).strip()
    raw_tokens = [tok.casefold() for tok in re.findall(r"[A-Za-z0-9.]+", cleaned)]
    tokens = [
        tok
        for tok in raw_tokens
        if len(tok) >= 2 and tok not in _GENERIC_QUERY_STOPWORDS
    ]
    if not tokens:
        tokens = [tok for tok in raw_tokens if len(tok) >= 2]
    normalized = []
    for tok in tokens:
        if tok.endswith("ies") and len(tok) > 4:
            tok = tok[:-3] + "y"
        elif tok.endswith("s") and not tok.endswith("ss") and len(tok) > 4:
            tok = tok[:-1]
        if tok not in normalized:
            normalized.append(tok)
    return normalized


def _token_match_mask(df: pd.DataFrame, columns: list[str], tokens: list[str]) -> pd.Series:
    """Return rows where all extracted tokens appear across the given columns."""
    if not tokens or df.empty:
        return pd.Series(False, index=df.index)
    combined = (
        df[columns]
        .astype(str)
        .agg(" ".join, axis=1)
        .str.replace("-", " ", regex=False)
        .str.casefold()
    )
    mask = pd.Series(True, index=df.index)
    for token in tokens:
        mask &= combined.str.contains(token, case=False, regex=False, na=False)
    return mask


def search_courses(query: str, limit: int = 10) -> list[dict]:
    """Search course codes, titles, departments, and syllabus details."""
    df = _load_csv("course_catalog.csv")
    query = str(query or "").strip()

    if not query:
        return []

    columns = [
        col
        for col in (
            "Course_Code",
            "Course_Title",
            "Department",
            "Program_Level",
            "Credits",
            "Category",
            "Focus_Area_Syllabus",
        )
        if col in df.columns
    ]

    # 1. Exact course code match if a course code is present in the query.
    code_match = _COURSE_CODE_RE.search(query)
    if code_match and "Course_Code" in df.columns:
        code = code_match.group(1).upper()
        exact_code = df[
            df["Course_Code"].astype(str).str.strip().str.upper().eq(code)
        ]
        if not exact_code.empty:
            return exact_code[columns].head(max(1, limit)).to_dict("records")

    # 2. Direct substring match.
    mask = (
        df[columns]
        .astype(str)
        .apply(lambda col: col.str.contains(query, case=False, regex=False, na=False))
        .any(axis=1)
    )
    if mask.any():
        return df.loc[mask, columns].head(max(1, limit)).to_dict("records")

    # 3. Hyphen-normalized and token-based fallback match.
    tokens = _extract_meaningful_tokens(query)
    token_mask = _token_match_mask(df, columns, tokens)
    return df.loc[token_mask, columns].head(max(1, limit)).to_dict("records")


def _clean_semester_filter(semester: str) -> str:
    """Normalize semester filter and discard generic placeholders."""
    val = re.sub(r"\s+", " ", str(semester or "")).strip()
    if val.casefold() in _PLACEHOLDER_VALUES:
        return ""
    return val


def _semester_matches(series: pd.Series, semester_filter: str) -> pd.Series:
    """Match semester values flexibly while respecting year/term constraints."""
    raw = semester_filter.strip()
    if not raw:
        return pd.Series(True, index=series.index)

    direct = series.astype(str).str.contains(raw, case=False, regex=False, na=False)
    if direct.any():
        return direct

    norm_series = (
        series.astype(str)
        .str.replace(r"[,–—-]+", " ", regex=True)
        .str.replace(r"\bsemester\s+1\b", "semester i", flags=re.I, regex=True)
        .str.replace(r"\bsemester\s+2\b", "semester ii", flags=re.I, regex=True)
        .str.replace(r"\s+", " ", regex=True)
        .str.casefold()
    )
    norm_filter = (
        re.sub(r"[,–—-]+", " ", raw)
        .replace("Semester 1", "semester i")
        .replace("semester 1", "semester i")
        .replace("Semester 2", "semester ii")
        .replace("semester 2", "semester ii")
    )
    tokens = [
        tok.casefold()
        for tok in re.findall(r"[A-Za-z0-9]+", norm_filter)
        if tok.casefold() not in {"ay", "academic", "year", "in", "for", "the"}
    ]
    if not tokens:
        return direct

    mask = pd.Series(True, index=series.index)
    for tok in tokens:
        if tok in {"i", "ii", "iii", "iv"}:
            mask &= norm_series.str.contains(rf"\b{re.escape(tok)}\b", case=False, regex=True, na=False)
        else:
            mask &= norm_series.str.contains(tok, case=False, regex=False, na=False)
    return mask


def _clean_group_section_filter(group_section: str, query: str) -> str:
    """Discard bogus group_section values invented by small local models."""
    val = re.sub(r"\s+", " ", str(group_section or "")).strip()
    if not val or val.casefold() in _PLACEHOLDER_VALUES:
        return ""
    if query and val.casefold() == str(query).strip().casefold():
        return ""
    if _COURSE_CODE_RE.fullmatch(val):
        return ""
    return val


def search_timetable(
    query: str,
    semester: str = "",
    group_section: str = "",
    limit: int = 20,
) -> list[dict]:
    """Search timetable records by course, instructor, day, or time slot."""
    df = _load_csv("course_timetable.csv")
    raw_query = str(query or "").strip()
    clean_sem = _clean_semester_filter(semester)
    clean_group = _clean_group_section_filter(group_section, raw_query)

    if not raw_query and not clean_sem and not clean_group:
        return []

    searchable_columns = [
        col
        for col in (
            "Course_Code",
            "Course_Name",
            "Instructor",
            "Day",
            "Time_Slot",
            "Slot_Type",
        )
        if col in df.columns
    ]

    filtered = df.copy()

    if raw_query:
        # Prefer an explicit course code if one is embedded in a longer phrase.
        code_match = _COURSE_CODE_RE.search(raw_query)
        search_target = code_match.group(1) if code_match else raw_query

        mask = (
            filtered[searchable_columns]
            .astype(str)
            .apply(lambda col: col.str.contains(search_target, case=False, regex=False, na=False))
            .any(axis=1)
        )
        if not mask.any() and search_target != raw_query:
            mask = (
                filtered[searchable_columns]
                .astype(str)
                .apply(lambda col: col.str.contains(raw_query, case=False, regex=False, na=False))
                .any(axis=1)
            )
        if not mask.any():
            tokens = _extract_meaningful_tokens(raw_query)
            mask = _token_match_mask(filtered, searchable_columns, tokens)

        filtered = filtered.loc[mask]

    if clean_sem and "Semester" in filtered.columns:
        filtered = filtered[_semester_matches(filtered["Semester"], clean_sem)]

    if clean_group and "Group_Section" in filtered.columns:
        group_matches = filtered[
            filtered["Group_Section"].astype(str).str.contains(
                clean_group, case=False, regex=False, na=False
            )
        ]
        # Only apply group_section if it exists in the timetable dataset or
        # matches rows; do not let a hallucinated course title in group_section
        # wipe out valid course matches.
        valid_group_in_dataset = (
            df["Group_Section"]
            .astype(str)
            .str.contains(clean_group, case=False, regex=False, na=False)
            .any()
        )
        if not group_matches.empty or valid_group_in_dataset:
            filtered = group_matches

    return filtered.head(max(1, limit)).to_dict("records")


def search_faculty(query: str, limit: int = 10) -> list[dict]:
    """Search faculty names, departments, emails, and research areas."""
    df = _load_csv("faculty_contacts.csv")
    query = str(query or "").strip()

    if not query:
        return []

    columns = [
        col
        for col in (
            "Name",
            "Designation",
            "Department",
            "Email",
            "Research_Areas",
            "Profile_URL",
        )
        if col in df.columns
    ]

    # 1. Direct substring match.
    mask = (
        df[columns]
        .astype(str)
        .apply(lambda col: col.str.contains(query, case=False, regex=False, na=False))
        .any(axis=1)
    )
    if mask.any():
        return df.loc[mask, columns].head(max(1, limit)).to_dict("records")

    # 2. Hyphen-normalized substring match (e.g. "composite-material" -> "composite material").
    dehyphenated = re.sub(r"[-_]+", " ", query).strip()
    if dehyphenated != query:
        mask = (
            df[columns]
            .astype(str)
            .apply(lambda col: col.str.contains(dehyphenated, case=False, regex=False, na=False))
            .any(axis=1)
        )
        if mask.any():
            return df.loc[mask, columns].head(max(1, limit)).to_dict("records")

    # 3. Token/stem match across faculty fields (e.g. "composite-material researchers").
    tokens = _extract_meaningful_tokens(query)
    token_mask = _token_match_mask(df, columns, tokens)
    return df.loc[token_mask, columns].head(max(1, limit)).to_dict("records")


def _is_relative_or_unspecified_semester(semester: str) -> bool:
    """Return True if semester is empty or a relative phrase like 'this semester'."""
    val = re.sub(r"\s+", " ", str(semester or "")).strip().casefold()
    if not val or val in _PLACEHOLDER_VALUES:
        return True
    if any(phrase in val for phrase in ("this semester", "current", "present", "floated", "offered")):
        return True
    return False


def count_courses(
    department: str = "",
    level: str = "",
    category: str = "",
    semester: str = "",
    require_timetable: bool = False,
) -> dict:
    """Count catalog courses using explicit filters and available records.

    When timetable evidence is requested (or a relative term like 'this semester'
    is passed), an exact semester is required. This prevents historical timetable
    data from being presented as the current semester's offering.
    """
    raw_sem = str(semester or "").strip()
    if raw_sem and _is_relative_or_unspecified_semester(raw_sem):
        require_timetable = True
        raw_sem = ""

    if require_timetable and not raw_sem:
        return {
            "status": "insufficient_evidence",
            "message": (
                "Insufficient evidence: An exact academic year and semester are required "
                "to count courses floated or offered in a specific term. The available "
                "timetable data only covers historical AY 2023-24 Semester-I records and "
                "does not establish which courses are floated in the current semester."
            ),
            "count": None,
            "courses": [],
        }

    df = _load_csv("course_catalog.csv")
    required = {"Course_Code", "Course_Title", "Department", "Category"}
    if not required.issubset(df.columns):
        return {
            "status": "insufficient_evidence",
            "message": "The course catalogue lacks fields needed for this count.",
            "count": None,
            "courses": [],
        }

    filtered = df.copy()
    if department.strip():
        filtered = filtered[
            filtered["Department"].astype(str).str.contains(
                department.strip(), case=False, regex=False, na=False
            )
        ]

    if level.strip():
        level_value = re.sub(r"\D", "", level)
        if not level_value:
            return {
                "status": "error",
                "message": "Level must be a numeric course level, such as 400.",
                "count": None,
                "courses": [],
            }
        level_digit = level_value[0]
        codes = filtered["Course_Code"].astype(str).str.extract(r"[A-Za-z]+(\d+)", expand=False)
        filtered = filtered.loc[codes.str.startswith(level_digit, na=False)]

    if category.strip():
        category_query = category.strip()
        filtered = filtered[
            filtered["Category"].astype(str).str.contains(
                category_query, case=False, regex=False, na=False
            )
        ]

    if require_timetable:
        timetable = _load_csv("course_timetable.csv")
        if not {"Course_Code", "Semester"}.issubset(timetable.columns):
            return {
                "status": "insufficient_evidence",
                "message": "The timetable lacks course-code or semester fields.",
                "count": None,
                "courses": [],
            }
        term_rows = timetable[_semester_matches(timetable["Semester"], raw_sem)]
        offered_codes = set(term_rows["Course_Code"].astype(str).str.strip().str.upper())
        filtered = filtered[
            filtered["Course_Code"].astype(str).str.strip().str.upper().isin(offered_codes)
        ]

    result_columns = [
        col
        for col in ("Course_Code", "Course_Title", "Department", "Program_Level", "Credits", "Category")
        if col in filtered.columns
    ]
    records = filtered[result_columns].to_dict("records")

    # If no catalog records with valid course codes match at all (e.g., when a
    # department's curriculum booklet has unindexed course codes or no semester
    # offering records), report insufficient evidence rather than claiming 0
    # courses are offered this semester.
    if not records and not require_timetable:
        return {
            "status": "insufficient_evidence",
            "message": (
                "Insufficient evidence: No matching courses with the requested department, "
                "level, and category filters could be verified in the structured course "
                "catalog or current-semester timetable records."
            ),
            "count": None,
            "courses": [],
            "filters": {
                "department": department.strip(),
                "level": level.strip(),
                "category": category.strip(),
                "semester": raw_sem,
                "require_timetable": require_timetable,
            },
        }

    return {
        "status": "success",
        "count": len(records),
        "courses": records,
        "filters": {
            "department": department.strip(),
            "level": level.strip(),
            "category": category.strip(),
            "semester": raw_sem,
            "require_timetable": require_timetable,
        },
        "source": "course_catalog.csv" + (" and course_timetable.csv" if require_timetable else ""),
        "note": (
            f"The count is limited to the exact semester filter '{raw_sem}'."
            if require_timetable
            else "This is a catalogue count, not proof that every course is offered this semester."
        ),
    }


def _normalize_person_name(value: str) -> str:
    """Normalize titles and whitespace for conservative name matching."""
    text = re.sub(r"\s+", " ", str(value or "")).strip().casefold()
    text = re.sub(r"^(?:dr|prof|professor|doctor)\.?\s+", "", text)
    return text


def lookup_instructor_contact(course_code: str, semester: str = "") -> dict:
    """Find a course instructor in timetable data and match a published email.

    This function only returns names/emails present in the supplied CSVs.
    It reports ambiguity and keeps the semester evidence visible instead of
    selecting one instructor arbitrarily.
    """
    raw_code = str(course_code or "").strip().upper()
    code_match = _COURSE_CODE_RE.search(raw_code)
    code = code_match.group(1).upper() if code_match else raw_code
    if not code:
        return {"status": "error", "message": "A course code is required."}

    clean_sem = _clean_semester_filter(semester)

    timetable = _load_csv("course_timetable.csv")
    required = {"Course_Code", "Instructor"}
    if not required.issubset(timetable.columns):
        return {
            "status": "insufficient_evidence",
            "message": "The timetable lacks course-code or instructor fields.",
        }

    rows = timetable[
        timetable["Course_Code"].astype(str).str.strip().str.upper().eq(code)
    ].copy()
    if clean_sem and "Semester" in rows.columns:
        rows = rows[_semester_matches(rows["Semester"], clean_sem)]

    if rows.empty:
        return {
            "status": "not_found",
            "course_code": code,
            "message": (
                f"No instructor record for course {code} was found in the collected "
                "timetable data (course_timetable.csv)."
            ),
            "drafting_instruction": (
                f"Still write the complete requested email draft for {code} using "
                f"placeholders such as [Course Instructor of {code}] and [Instructor Email], "
                f"and add a brief note that the instructor contact for {code} was not found "
                "in the collected timetable data."
            ),
            "source": "course_timetable.csv",
        }

    if "Semester" in rows.columns:
        terms = sorted({str(v).strip() for v in rows["Semester"] if str(v).strip()})
    else:
        terms = []

    instructors = sorted(
        {str(v).strip() for v in rows["Instructor"] if str(v).strip()},
        key=str.casefold,
    )
    normalized_to_name = {}
    for name in instructors:
        normalized_to_name.setdefault(_normalize_person_name(name), name)
    normalized_to_name.pop("", None)

    if not normalized_to_name:
        return {
            "status": "not_found",
            "course_code": code,
            "semester_records": terms,
            "message": "The matching timetable records do not name an instructor.",
            "drafting_instruction": (
                f"Still write the complete requested email draft for {code} using "
                "placeholders such as [Instructor Name] and [Instructor Email]."
            ),
            "source": "course_timetable.csv",
        }

    if len(normalized_to_name) > 1:
        return {
            "status": "ambiguous",
            "course_code": code,
            "instructors": list(normalized_to_name.values()),
            "semester_records": terms,
            "message": (
                "Multiple instructors appear in the matching timetable records. "
                "Specify an exact semester/section before selecting a recipient."
            ),
            "drafting_instruction": (
                f"Still write the complete requested email draft for {code} using "
                "placeholders for the recipient name and email."
            ),
            "source": "course_timetable.csv",
        }

    instructor = next(iter(normalized_to_name.values()))
    faculty = _load_csv("faculty_contacts.csv")
    required_faculty = {"Name", "Email"}
    if not required_faculty.issubset(faculty.columns):
        return {
            "status": "email_not_found",
            "course_code": code,
            "instructor": instructor,
            "semester_records": terms,
            "message": "An instructor was found, but the faculty contact file lacks name/email fields.",
            "source": "course_timetable.csv",
        }

    matched = faculty[
        faculty["Name"].astype(str).map(_normalize_person_name).eq(
            _normalize_person_name(instructor)
        )
    ]
    emails = sorted(
        {str(v).strip() for v in matched["Email"] if str(v).strip()},
        key=str.casefold,
    )

    common = {
        "course_code": code,
        "instructor": instructor,
        "semester_records": terms,
        "timetable_source": "course_timetable.csv",
    }
    if len(emails) == 1:
        return {
            "status": "success",
            **common,
            "email": emails[0],
            "faculty_source": "faculty_contacts.csv",
            "note": "Use the listed semester as the evidence scope; do not assume it is current.",
        }
    if len(emails) > 1:
        return {
            "status": "ambiguous",
            **common,
            "emails": emails,
            "message": "Multiple published email records match this instructor name.",
            "faculty_source": "faculty_contacts.csv",
        }
    return {
        "status": "email_not_found",
        **common,
        "message": "The instructor was found, but no exact faculty email match was available.",
        "faculty_source": "faculty_contacts.csv",
    }
