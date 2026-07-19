import re
import json
import os
from dataclasses import dataclass, field
from typing import Optional

@dataclass
class QueryMetadata:
    department: str | None = None
    faculty: str | None = None
    program: str | None = None
    section: str | None = None
    entity: str | None = None
    semester: int | None = None
    course_code: str | None = None
    person_name: str | None = None
    fee_type: str | None = None
    year_level: str | None = None
    filters: dict = field(default_factory=dict)

DEPARTMENTS = {}
FACULTIES = {}
PROGRAMS = {}

SECTIONS = {
    "faculty":[
        "faculty", "chairperson", "chairman", "co-chairperson", "co chairman",
        "dean", "director", "advisor", "coordinator", "professor",
        "associate professor", "assistant professor", "lecturer", "instructor",
    ],
    "curriculum":[
        "curriculum", "scheme of study", "semester", "course", "subjects", "credit hours",
    ],
    "fees":[
        "fee", "fees", "tuition", "security", "challan", "dues",
    ],
    "eligibility":[
        "eligibility", "eligible",
    ],
    "admission":[
        "admission", "apply", "application", "merit", "aggregate", "seat", "seats", "allocation", "quota"
    ],
    "scholarship":[
        "scholarship", "financial assistance",
    ],
    "laboratory":[
        "lab", "laboratory",
    ],
}

ENTITIES = {
    "chairperson":["chairperson", "chairman", "head"],
    "co-chairperson":["co-chairperson", "co chairman"],
    "dean":["dean"],
    "director":["director"],
    "professor":["professor", "professors"],
    "associate professor":["associate professor"],
    "assistant professor":["assistant professor"],
    "lecturer":["lecturer", "lecturers"],
    "course":["course", "subject", "subjects"],
}

_metadata_loaded = False

def load_dynamic_metadata():
    global DEPARTMENTS, FACULTIES, PROGRAMS, _metadata_loaded
    if _metadata_loaded:
        return
        
    DEPARTMENTS.clear()
    FACULTIES.clear()
    PROGRAMS.clear()
    
    files = [
        os.path.join("output_chunks", "UGProspectus_compiled_knowledge.json"),
        os.path.join("output_chunks", "PGProspectus_compiled_knowledge.json")
    ]
    
    unique_depts = set()
    unique_faculties = set()
    unique_programs = set()
    unique_designations = set()
    
    for file_path in files:
        if os.path.exists(file_path):
            try:
                with open(file_path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                    for chunk in data:
                        dept = chunk.get("department")
                        if dept:
                            unique_depts.add(dept.strip())
                            
                        fac = chunk.get("faculty")
                        if fac:
                            unique_faculties.add(fac.strip())
                            
                        prog = chunk.get("program")
                        if prog:
                            unique_programs.add(prog.strip())
                            
                        desig = chunk.get("designation")
                        if desig:
                            unique_designations.add(desig.strip())
            except Exception as e:
                print(f"[WARNING] Failed to load metadata from {file_path}: {e}")
                
    for d in unique_depts:
        if not d or len(d) > 80:
            continue
        d_lower = d.lower()
        aliases = [d_lower]
        clean_d = re.sub(r'^[0-9.\(i\)]+\s*', '', d_lower).strip()
        if clean_d:
            aliases.append(clean_d)
        if "department" in clean_d:
            clean_d2 = clean_d.replace("departments", "").replace("department", "").replace("of", "").strip()
            if len(clean_d2) > 2:
                aliases.append(clean_d2)
            
        # Clean up multiple spaces
        aliases = [re.sub(r'\s+', ' ', a).strip() for a in aliases if a and len(a) > 2]
        DEPARTMENTS[d] = list(set(aliases))
        
    for f in unique_faculties:
        if not f or len(f) > 80 or f.lower() in ("the", "faculty", "departments", "faculty members", "members"):
            continue
        f_lower = f.lower()
        aliases = [f_lower]
        clean_f = re.sub(r'^[0-9.\(i\)]+\s*', '', f_lower).strip()
        if clean_f:
            aliases.append(clean_f)
        if "faculty" in clean_f:
            clean_f2 = clean_f.replace("faculty", "").replace("of", "").strip()
            if len(clean_f2) > 2:
                aliases.append(clean_f2)
            
        aliases = [re.sub(r'\s+', ' ', a).strip() for a in aliases if a and len(a) > 2]
        FACULTIES[f] = list(set(aliases))
        
    for p in unique_programs:
        if not p or len(p) > 80:
            continue
        p_lower = p.lower()
        aliases = [p_lower]
        clean_p = re.sub(r'^[0-9.\(i\)]+\s*', '', p_lower).strip()
        if clean_p:
            aliases.append(clean_p)
            
        no_degree = re.sub(r'^(bs|ms|bachelor of|master of|phd|m\.?s\.?|b\.?s\.?|b\.?e\.?)\s+', '', clean_p).strip()
        if no_degree and len(no_degree) > 2:
            aliases.append(no_degree)
            
        aliases = [re.sub(r'\s+', ' ', a).strip() for a in aliases if a and len(a) > 2]
        PROGRAMS[p] = list(set(aliases))
        
    for desig in unique_designations:
        if not desig or len(desig) > 50:
            continue
        d_lower = desig.lower()
        if d_lower not in SECTIONS["faculty"]:
            SECTIONS["faculty"].append(d_lower)
        if d_lower not in ENTITIES:
            ENTITIES[d_lower] = [d_lower]
        elif d_lower not in ENTITIES[d_lower]:
            ENTITIES[d_lower].append(d_lower)
        
    _metadata_loaded = True


SEMESTER_RE = re.compile(r"semester\s*([1-8])", re.I)
COURSE_RE = re.compile(r"\b[A-Z]{2,5}\s*-?\s*\d{2,4}\b", re.I)
PERSON_RE = re.compile(r"(?:Prof\.?|Professor|Dr\.?|Engr\.?|Mr\.?|Mrs\.?|Ms\.?)\s+[A-Z][A-Za-z\.\s]+", re.I)
YEAR_RE = re.compile(r"\b(first|1st|second|2nd|third|3rd|fourth|4th|final)\s+year\b", re.I)

def detect_program(query: str):
    load_dynamic_metadata()
    q = query.lower()
    for program, aliases in PROGRAMS.items():
        for alias in aliases:
            if alias in q:
                return program
    return None

def detect_faculty(query: str):
    load_dynamic_metadata()
    q = query.lower()
    for faculty, aliases in FACULTIES.items():
        for alias in aliases:
            if alias in q:
                return faculty
    return None

def detect_department(query: str):
    load_dynamic_metadata()
    q = query.lower()
    for key, aliases in DEPARTMENTS.items():
        if key.lower() in q or any(alias in q for alias in aliases):
            return key
    return None

def detect_semester(query):
    # Try digit format: "semester 10" or "10th semester"
    m = re.search(r"(?:semester\s*(\d+)|(\d+)(?:st|nd|rd|th)?\s*semester)", query, re.I)
    if m:
        val = m.group(1) or m.group(2)
        return int(val)
        
    # Try word format: "first semester", "tenth semester", etc.
    word_map = {
        "first": 1, "one": 1,
        "second": 2, "two": 2,
        "third": 3, "three": 3,
        "fourth": 4, "four": 4,
        "fifth": 5, "five": 5,
        "sixth": 6, "six": 6,
        "seventh": 7, "seven": 7,
        "eighth": 8, "eight": 8,
        "ninth": 9, "nine": 9,
        "tenth": 10, "ten": 10,
        "eleventh": 11, "eleven": 11,
        "twelfth": 12, "twelve": 12
    }
    for word, num in word_map.items():
        if re.search(rf"\b{word}\s+semester\b", query, re.I) or re.search(rf"\bsemester\s+{word}\b", query, re.I):
            return num
            
    return None

def detect_year_level(query):
    m = YEAR_RE.search(query)
    if m:
        return m.group(1).lower()
    return None

def detect_course(query):
    m = COURSE_RE.search(query)
    if m:
        return m.group()
    return None

def detect_person(query):
    m = PERSON_RE.search(query)
    if m:
        return m.group().strip()
    return None

def detect_fee_type(query):
    q = query.lower()
    if "tuition" in q: return "tuition"
    if "security" in q: return "security"
    if "admission" in q: return "admission"
    if "hostel" in q: return "hostel"
    return None

def normalize(query: str) -> str:
    return query.lower()

def detect_section(query: str):
    for section, words in SECTIONS.items():
        for word in words:
            if word in query:
                return section
    return None

def detect_entity(query):
    load_dynamic_metadata()
    q = query.lower()
    for key, aliases in ENTITIES.items():
        if any(alias in q for alias in aliases):
            return key
    return None

def build_metadata(query: str):
    query = normalize(query)
    department = detect_department(query)
    section = detect_section(query)
    entity = detect_entity(query)
    faculty = detect_faculty(query)
    program = detect_program(query)
    semester = detect_semester(query)
    year_level = detect_year_level(query)
    course_code = detect_course(query)
    person_name = detect_person(query)
    fee_type = detect_fee_type(query)
    
    filters = {}
    if department:
        filters["department"] = department
    if section:
        if section != "admission":
            filters["section"] = section
    if faculty:
        filters["faculty"] = faculty
    
    # We no longer strictly filter by program because table extraction often appends trailing 
    # table column data to the program name, causing exact-match DB filters to fail.
    # Semantic Search + BM25 handles program routing perfectly on its own.
    # We also do not strictly filter by semester, because previously ingested databases might have 
    # 'null' semester values due to table formatting (e.g. 'SEMESTER-1').
    
    if course_code:
        filters["course_codes"] = {"$in": [course_code]}
    # For fee_type we could map to a specific filter if the DB supports it, or section="fees"
    if fee_type:
        filters["section"] = "fees"

    return QueryMetadata(
        department=department,
        faculty=faculty,
        program=program,
        section=section,
        entity=entity,
        semester=semester,
        year_level=year_level,
        course_code=course_code,
        person_name=person_name,
        fee_type=fee_type,
        filters=filters,
    )