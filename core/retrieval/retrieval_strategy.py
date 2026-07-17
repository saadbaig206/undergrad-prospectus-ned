from dataclasses import dataclass

from core.retrieval.query_metadata import QueryMetadata


@dataclass
class RetrievalStrategy:

    pinecone_top_k: int

    bm25_top_k: int

    rerank_top_k: int

    expand_section: bool = False

    expand_page: bool = False

    expand_table: bool = False

    use_metadata_filter: bool = False


def choose_strategy(
    metadata: QueryMetadata,
) -> RetrievalStrategy:

    # --------------------------------------------------
    # Faculty / Chairperson / Dean
    # --------------------------------------------------

    if metadata.section == "faculty":

        return RetrievalStrategy(

            pinecone_top_k=10,

            bm25_top_k=10,

            rerank_top_k=10,

            expand_section=True,

            expand_page=True,

            use_metadata_filter=True,

        )

    # --------------------------------------------------
    # Curriculum
    # --------------------------------------------------

    if metadata.section == "curriculum":

        return RetrievalStrategy(

            pinecone_top_k=10,

            bm25_top_k=12,

            rerank_top_k=10,

            expand_section=True,

            expand_table=True,

            use_metadata_filter=True,

        )

    # --------------------------------------------------
    # Fees
    # --------------------------------------------------

    if metadata.section == "fees":

        return RetrievalStrategy(

            pinecone_top_k=10,

            bm25_top_k=12,

            rerank_top_k=10,

            expand_section=True,

            expand_table=True,

            use_metadata_filter=True,

        )

    # --------------------------------------------------
    # Eligibility
    # --------------------------------------------------

    if metadata.section == "eligibility":

        return RetrievalStrategy(

            pinecone_top_k=8,

            bm25_top_k=8,

            rerank_top_k=8,

            use_metadata_filter=True,

        )

    # --------------------------------------------------
    # Admission
    # --------------------------------------------------

    if metadata.section == "admission":

        return RetrievalStrategy(

            pinecone_top_k=10,

            bm25_top_k=10,

            rerank_top_k=8,

            use_metadata_filter=True,

        )

    # --------------------------------------------------
    # Scholarships
    # --------------------------------------------------

    if metadata.section == "scholarship":

        return RetrievalStrategy(

            pinecone_top_k=8,

            bm25_top_k=8,

            rerank_top_k=8,

            expand_section=True,

            use_metadata_filter=True,

        )

    # --------------------------------------------------
    # Laboratory
    # --------------------------------------------------

    if metadata.section == "laboratory":

        return RetrievalStrategy(

            pinecone_top_k=8,

            bm25_top_k=8,

            rerank_top_k=8,

            expand_section=True,

            use_metadata_filter=True,

        )

    # --------------------------------------------------
    # Person queries
    # --------------------------------------------------

    if metadata.person_name:

        return RetrievalStrategy(

            pinecone_top_k=5,

            bm25_top_k=5,

            rerank_top_k=5,

            expand_page=True,
            
            use_metadata_filter=True,

        )

    # --------------------------------------------------
    # Course code
    # --------------------------------------------------

    if metadata.course_code:

        return RetrievalStrategy(

            pinecone_top_k=3,

            bm25_top_k=3,

            rerank_top_k=3,

            expand_table=True,
            
            use_metadata_filter=True,

        )

    # --------------------------------------------------
    # Semester / Year queries
    # --------------------------------------------------

    if metadata.semester or getattr(metadata, "year_level", None):

        return RetrievalStrategy(

            pinecone_top_k=5,

            bm25_top_k=5,

            rerank_top_k=5,

            expand_table=True,

            expand_section=True,
            
            use_metadata_filter=True,

        )

    # --------------------------------------------------
    # Default
    # --------------------------------------------------

    return RetrievalStrategy(

        pinecone_top_k=8,

        bm25_top_k=8,

        rerank_top_k=8,

    )