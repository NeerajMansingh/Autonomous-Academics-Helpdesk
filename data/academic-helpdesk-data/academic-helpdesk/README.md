# IIT Jodhpur Academic Helpdesk Dataset

This directory contains complete scraped data, official documents, curriculum booklets, academic regulations, circulars, calendars, and structured catalogs collected from the official Indian Institute of Technology Jodhpur (IITJ) academic portals.

---

## Folder Structure

```
academic-helpdesk/
│
├── data/
│   ├── documents/
│   │   ├── regulations/          # Official Academic Regulations & Amendments PDFs
│   │   ├── curriculum/           # Full UG/PG/MTech/MSc Curriculum Booklets & Syllabi PDFs
│   │   ├── academic_calendar/    # Official Academic Year, Term & Program Calendars PDFs
│   │   ├── circulars/            # Official Academic Circulars resolved from repository
│   │   └── faculty/              # Departmental faculty directories & markdown profiles
│   │
│   ├── webpages/                 # Full HTML & text archives of target portals and department pages
│   │
│   └── structured/               # Normalized structured datasets in CSV format
│       ├── course_timetable.csv
│       ├── course_catalog.csv
│       ├── faculty_contacts.csv
│       └── document_manifest.csv
│
└── structured/                   # Duplicate mirror of structured datasets for direct access
    ├── course_timetable.csv
    ├── course_catalog.csv
    ├── faculty_contacts.csv
    └── document_manifest.csv
```

---

## Dataset Summary & Metrics

| Category | File Count | Description |
|---|---|---|
| **Regulations (`documents/regulations/`)** | 11 PDFs (~5.9 MB) | Core Academic Regulations, PG Regulations, Amendments, Attendance & Exam Policies |
| **Curriculum (`documents/curriculum/`)** | 31 PDFs (~37.6 MB) | Complete curriculum booklets for B.Tech, B.S., M.Tech, M.Des, MBA, and ITEP programs |
| **Academic Calendar (`documents/academic_calendar/`)** | 12 PDFs (~3.3 MB) | Academic calendars for AY 2024-25, 2025-26, 2026-27, Executive, Summer Terms, and AI/ML programs |
| **Circulars (`documents/circulars/`)** | 40 files (37 PDFs) (~17.1 MB) | Official circulars resolved from internal repository handles, plus timetable spreadsheets |
| **Faculty Documents (`documents/faculty/`)** | 23 files (~0.4 MB) | Departmental faculty rosters, complete markdown directory, and JSON data |
| **Archived Webpages (`webpages/`)** | 36 files (30 HTMLs) (~2.4 MB) | Full offline snapshots and text dumps of the main sites and all 24 department people rosters |
| **Structured CSVs (`structured/`)** | 4 CSV files (~1.7 MB) | Machine-readable normalized tables for downstream analytics and RAG |

---

## Structured CSV Files Specification

### 1. `faculty_contacts.csv` (326 Records)
Comprehensive directory of all faculty members across all 24 departments, schools, IDRPs, and centers.
- **Fields:** `Name`, `Designation`, `Department`, `Email`, `Phone`, `Education`, `Research_Areas`, `Profile_URL`, `Photo_URL`

### 2. `course_timetable.csv` (266 Records)
Full timetable schedule mapped across groups and weekly slots.
- **Fields:** `Semester`, `Group_Section`, `Day`, `Time_Slot`, `Course_Code`, `Course_Name`, `Credits`, `LTP`, `Instructor`, `Instructor_Email`, `Slot_Type`

### 3. `course_catalog.csv` (1,378 Records)
Master course catalog extracted from curriculum booklets and institute course structures.
- **Fields:** `Course_Code`, `Course_Title`, `Department`, `Program_Level`, `LTP`, `Credits`, `Category`, `Focus_Area_Syllabus`, `Source_Booklet`

### 4. `document_manifest.csv` (157 Records)
Complete inventory and cryptographic integrity verification of all downloaded documents and assets.
- **Fields:** `Document_ID`, `Filename`, `Category`, `Relative_Path`, `File_Size_Bytes`, `SHA256_Checksum`, `Status`, `Description`

---

## Source URLs Covered
1. **Circulars**: `https://iitj.ac.in/office-of-academics/en/circulars`
2. **Faculty Members**: `https://www.iitj.ac.in/main/en/faculty-members` (covering all 24 department/school rosters)
3. **Academic Regulations**: `https://www.iitj.ac.in/office-of-academics/en/academic-regulations`
4. **Curriculum**: `https://www.iitj.ac.in/office-of-academics/en/curriculum`
5. **Academic Calendar**: `https://www.iitj.ac.in/office-of-academics/en/academic-calendar`
6. **Class Time-Table**: `https://iitj.ac.in/office-of-academics/en/first-year-class-time-table`
