from core.utils.query_utils import detect_academic_level
from core.config import SEAT_DIST_FILE_LINK


async def handle_seat_query(standalone_query, intent):
    if intent == "SEAT":
        query_lower = standalone_query.lower()
        is_pg, is_ug = detect_academic_level(query_lower)
        if is_pg and not is_ug:
            yield "There is no fixed seat distribution matrix published in the Postgraduate Prospectus. Admissions to postgraduate Master's and Ph.D. programmes are determined based on departmental capacity, eligibility criteria, and academic/faculty resources rather than a predefined category-wise seat matrix. For details on specific programmes, please consult the respective department sections in the Postgraduate Prospectus.\n"
            return
        elif is_ug and not is_pg:
            yield f"For the complete and accurate Undergraduate Seat Distribution Matrix, please refer to the official document: [Undergraduate Seat Distribution PDF]({SEAT_DIST_FILE_LINK}).\n\n"
            return
        else:
            # Ambiguous or both: show undergraduate PDF first, then explain the postgraduate policy
            yield f"### Undergraduate\nFor the complete and accurate Undergraduate Seat Distribution Matrix, please refer to the official document: [Undergraduate Seat Distribution PDF]({SEAT_DIST_FILE_LINK}).\n\n"
            yield "### Postgraduate / Masters\nThere is no fixed seat distribution matrix published in the Postgraduate Prospectus. Admissions to postgraduate Master's and Ph.D. programmes are determined based on departmental capacity, eligibility criteria, and academic/faculty resources rather than a predefined category-wise seat matrix. For details on specific programmes, please consult the respective department sections in the Postgraduate Prospectus.\n"
            return