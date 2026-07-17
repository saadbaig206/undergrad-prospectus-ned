import re

FACULTY_ROW = re.compile(

    r"""
    (?P<designation>
        Chairperson|
        Chairman|
        Dean|
        Professor|
        Associate\s+Professor|
        Assistant\s+Professor|
        Lecturer|
        Instructor
    )

    \s*

    (?P<name>

        (?:Prof\.?|Professor|Dr\.?|Engr\.?)?

        [A-Z][A-Za-z.\s]{1,40}

    )

    """,

    re.I | re.X,

)

COURSE_ROW = re.compile(
    r"""
    (?P<code>[A-Z]{2,5}\s*-?\s*\d{2,4})            # Matches CS101, CS 101, CS-101, etc.
    .{1,100}?
    (?P<title>.{1,100}?)
    .{1,50}?
    (?P<credits>\d-\d-\d|NC-NC-NC|\d\s*\(\s*\d\s*,\s*\d\s*\)) # Tolerates complex credit definitions
    """,
    re.X | re.I,
)

FEE_ROW = re.compile(
    r"""
    (?P<fee>.{1,100}?)
    (?:Rs\.?|PKR|Rupees|RS)                        # Forgiving currency matching
    \s*
    (?P<amount>[0-9,]{1,20})
    """,
    re.X | re.I,
)
def extract_faculty_entities(text):

    people=[]

    for match in FACULTY_ROW.finditer(text):

        people.append({

            "entity":"faculty",

            "designation":match.group("designation"),

            "person_name":match.group("name").strip(),

        })

    return people

def extract_curriculum_entities(text):

    rows=[]

    for match in COURSE_ROW.finditer(text):

        rows.append({

            "entity":"course",

            "course_code":match.group("code"),

            "course_title":match.group("title").strip(),

            "credit_hours":match.group("credits"),

        })

    return rows

def extract_fee_entities(text):

    fees=[]

    for match in FEE_ROW.finditer(text):

        fees.append({

            "entity":"fee",

            "fee_type":match.group("fee"),

            "amount":match.group("amount"),

        })

    return fees