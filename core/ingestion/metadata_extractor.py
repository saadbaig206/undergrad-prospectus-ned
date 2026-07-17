import re
from dataclasses import dataclass
from typing import List
from langchain_core.documents import Document

# Dynamic imports from your entity module
from core.ingestion.entity_extractor import (
    extract_faculty_entities,
    extract_curriculum_entities,
    extract_fee_entities
)

@dataclass
class MetadataContext:
    page: int | None = None
    faculty: str | None = None
    department: str | None = None
    program: str | None = None

# ============================================================
# Generic Regex Patterns (Universal across Prospectuses)
# ============================================================
PAGE_RE = re.compile(r"^##\s*Page\s*(\d+)", re.M)
HEADING_RE = re.compile(r"^(#{1,6})\s+(.+)$", re.M)
SEMESTER_RE = re.compile(r"semester\s*([1-8])", re.I)
COURSE_RE = re.compile(r"[A-Z]{2,5}\s*-?\s*\d{2,4}", re.I)

# Flexible designators that match any department or faculty title format
FACULTY_RE = re.compile(r"(?:Faculty\s+of\s+|School\s+of\s+|College\s+of\s+)([^\n#]+)", re.I)
DEPARTMENT_RE = re.compile(r"(?:Department\s+of\s+|Dept\s*\\.\s*of\s+)([^\n#]+)", re.I)
PROGRAM_RE = re.compile(
    r"\b(Bachelor\s+of\s+[^\n#]+|Master\s+of\s+[^\n#]+|"
    r"BS\s+[^\n#]+|BE\s+[^\n#]+|MS\s+[^\n#]+|ME\s+[^\n#]+|"
    r"M\.?Phil\s+[^\n#]+|Ph\.?D\s+[^\n#]+)", 
    re.I
)

DESIGNATION_RE = re.compile(
    r"\b(Chairperson|Chairman|Chairwoman|Co-?Chairperson|Dean|Director|"
    r"Coordinator|Advisor|Professor|Associate\s+Professor|"
    r"Assistant\s+Professor|Lecturer|Instructor|Faculty\s+Member)\b",
    re.I
)

PERSON_RE = re.compile(
    r"(?:Prof\.?|Professor|Dr\.?|Engr\.?|Mr\.?|Mrs\.?|Ms\.?)\s+[A-Z][A-Za-z\.\s]+",
    re.I
)

# Generic Section Mapping Rules
SECTIONS = {
    "faculty": ["faculty", "chairperson", "chairman", "dean", "professor", "lecturer", "faculty members", "staff"],
    "curriculum": ["curriculum", "course", "subjects", "semester", "credit", "scheme of study", "syllabus"],
    "fees": ["fee", "fees", "tuition", "admission charges", "dues", "payment", "challan", "hostel charges"],
    "eligibility": ["eligibility", "criteria", "requirements", "admission criteria"],
    "scholarships": ["scholarship", "financial aid", "assistance", "bursary"],
    "laboratories": ["laboratory", "lab", "facilities", "workshop"],
    "admission": ["admission", "apply", "application", "prospectus", "intake", "deadline"]
}

# Generic Content Classifier
def detect_content_type(text: str) -> str:
    lower = text.lower()
    if "|" in text:
        if COURSE_RE.search(text):
            return "curriculum_table"
        if "fee" in lower or "tuition" in lower or "charges" in lower or "rs." in lower:
            return "fee_table"
        return "table"
    return "text"

def detect_section(text: str) -> str:
    lower = text.lower()
    for section, keywords in SECTIONS.items():
        if any(keyword in lower for keyword in keywords):
            return section
    return "general"

# ============================================================
# Generic Dynamic Context Tracker
# ============================================================
def update_context_generically(text: str, ctx: MetadataContext):
    """
    Scans headers or lines for dynamic structural cues to deduce the 
    current active contextual tracking parameters safely without lookups.
    """
    page_match = PAGE_RE.search(text)
    if page_match:
        ctx.page = int(page_num := page_match.group(1))

    # Dynamic extraction via structural regex designators
    fac_match = FACULTY_RE.search(text)
    if fac_match:
        ctx.faculty = fac_match.group(1).strip().strip("*:_#")

    dept_match = DEPARTMENT_RE.search(text)
    if dept_match:
        ctx.department = dept_match.group(1).strip().strip("*:_#")

    prog_match = PROGRAM_RE.search(text)
    if prog_match:
        ctx.program = prog_match.group(1).strip().strip("*:_#")

    # Direct Markdown Header Extraction Fallback: 
    # If a line is a Level 1 or Level 2 heading and no dynamic regex was found, 
    # deduce that it is likely a Department or Program block.
    for line in text.splitlines():
        m = HEADING_RE.match(line)
        if m:
            heading_text = m.group(2).strip().lower()
            if "department" in heading_text or "dept" in heading_text:
                ctx.department = m.group(2).replace("Department of", "", re.I).strip()
            elif "faculty" in heading_text:
                ctx.faculty = m.group(2).replace("Faculty of", "", re.I).strip()


def split_into_heading_sections(markdown: str) -> List[str]:
    headings = list(HEADING_RE.finditer(markdown))
    if not headings:
        return [markdown]
    sections = []
    for i, match in enumerate(headings):
        start = match.start()
        end = headings[i + 1].start() if i + 1 < len(headings) else len(markdown)
        sections.append(markdown[start:end].strip())
    return sections


def build_heading_path(section: str) -> List[str]:
    headings = []
    for line in section.splitlines():
        m = HEADING_RE.match(line)
        if m:
            headings.append(m.group(2).strip())
    return headings


def build_metadata(section: str, ctx: MetadataContext, academic_level: str, year: int) -> dict:
    heading_stack = build_heading_path(section)
    heading_path = " > ".join(heading_stack) if heading_stack else None

    # Safe standalone matching
    sem_match = SEMESTER_RE.search(section)
    cc_match = COURSE_RE.search(section)
    p_match = PERSON_RE.search(section)
    des_match = DESIGNATION_RE.search(section)

    metadata = {
        "page": ctx.page,
        "academic_level": academic_level,
        "year": year,
        "faculty": ctx.faculty,
        "department": ctx.department,
        "program": ctx.program,
        "heading": heading_path,
        "heading_path": heading_path,
        "heading_leaf": heading_stack[-1] if heading_stack else None,
        "heading_depth": len(heading_stack),
        "section": detect_section(section),
        "content_type": detect_content_type(section),
        "semester": int(sem_match.group(1)) if sem_match else None,
        "course_code": cc_match.group().strip() if cc_match else None,
        "person_name": p_match.group().strip() if p_match else None,
        "designation": des_match.group().strip() if des_match else None,
        "course_codes": list(set(COURSE_RE.findall(section))),
    }

    # Extract dynamic payload data via entity parsing layers safely
    try:
        f_ent = extract_faculty_entities(section)
        if f_ent: metadata["faculty_entities"] = f_ent
    except: pass

    try:
        c_ent = extract_curriculum_entities(section)
        if c_ent: metadata["curriculum_entities"] = c_ent
    except: pass

    try:
        fee_ent = extract_fee_entities(section)
        if fee_ent: metadata["fee_entities"] = fee_ent
    except: pass

    return metadata


def extract_documents(markdown: str, academic_level: str, year: int) -> List[Document]:
    ctx = MetadataContext()
    documents = []

    # Dynamic boundary sweep across the parsed layout string sheets
    pages = re.split(r"(?=^##\s*Page\s*\d+)", markdown, flags=re.M)

    for page in pages:
        if not page.strip():
            continue

        # Keep state updating globally
        update_context_generically(page, ctx)
        sections = split_into_heading_sections(page)

        for section in sections:
            if not section.strip():
                continue

            # Update state with granular sub-heading metrics
            update_context_generically(section, ctx)
            metadata = build_metadata(section, ctx, academic_level, year)

            documents.append(
                Document(
                    page_content=section.strip(),
                    metadata=metadata
                )
            )

    return documents