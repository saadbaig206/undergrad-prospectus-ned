import asyncio
import time

from core.retrieval.bm25 import retrieve_bm25
from core.retrieval.context_builder import build_context
from core.retrieval.expansion import (
    expand_page,
    expand_section,
    expand_table,
)
from core.retrieval.hybrid import hybrid_search_rrf
from core.retrieval.pinecone_retriever import retrieve_from_pinecone
from core.retrieval.query_metadata import build_metadata
from core.retrieval.reranker import rerank_chunks
from core.retrieval.retrieval_strategy import choose_strategy
from core.retrieval.metadata_retriever import retrieve_related_chunks

async def retrieve(
    query: str,
    query_vector: list[float],
    academic_level_filter: dict | None = None,
    is_ug: bool = False,
    is_pg: bool = False,
) -> dict:

    # ----------------------------------------
    # Query Understanding
    # ----------------------------------------

    metadata = build_metadata(query)

    strategy = choose_strategy(metadata)

    filter_dict = {}

    if academic_level_filter:
        filter_dict.update(academic_level_filter)

    if strategy.use_metadata_filter and metadata.filters:
        filter_dict.update(metadata.filters)

    t_start = time.time()
    t_query = time.time()

    # ----------------------------------------
    # Parallel Retrieval
    # ----------------------------------------

    pinecone_tasks = []
    
    if "academic_level" in filter_dict:
        pinecone_tasks.append(
            asyncio.create_task(
                retrieve_from_pinecone(
                    query_vector=query_vector,
                    top_k=strategy.pinecone_top_k,
                    metadata_filter=filter_dict,
                )
            )
        )
    else:
        if is_pg and not is_ug:
            pg_filter = filter_dict.copy()
            pg_filter["academic_level"] = "postgraduate"
            pinecone_tasks.append(
                asyncio.create_task(
                    retrieve_from_pinecone(
                        query_vector=query_vector,
                        top_k=strategy.pinecone_top_k,
                        metadata_filter=pg_filter,
                    )
                )
            )
        elif is_ug and not is_pg:
            ug_filter = filter_dict.copy()
            ug_filter["academic_level"] = "undergraduate"
            pinecone_tasks.append(
                asyncio.create_task(
                    retrieve_from_pinecone(
                        query_vector=query_vector,
                        top_k=strategy.pinecone_top_k,
                        metadata_filter=ug_filter,
                    )
                )
            )
        else:
            # Either both are true, or neither are true. 
            # Make a single combined Pinecone query to avoid doubling the REST API latency.
            pinecone_tasks.append(
                asyncio.create_task(
                    retrieve_from_pinecone(
                        query_vector=query_vector,
                        top_k=strategy.pinecone_top_k * 2,
                        metadata_filter=filter_dict if filter_dict else None,
                    )
                )
            )

    bm25_tasks = []

    if is_pg or (not is_pg and not is_ug):

        bm25_tasks.append(
            asyncio.create_task(
                retrieve_bm25(
                    query,
                    "postgraduate",
                    strategy.bm25_top_k,
                    filter_dict if strategy.use_metadata_filter else None
                )
            )
        )

    if is_ug or (not is_pg and not is_ug):

        bm25_tasks.append(
            asyncio.create_task(
                retrieve_bm25(
                    query,
                    "undergraduate",
                    strategy.bm25_top_k,
                    filter_dict if strategy.use_metadata_filter else None
                )
            )
        )

    pinecone_start = time.time()
    semantic_matches = []
    if pinecone_tasks:
        try:
            pinecone_results = await asyncio.gather(*pinecone_tasks)
            for matches in pinecone_results:
                semantic_matches.extend(matches)
            
            # Sort globally by score descending to prevent one academic level from unfairly dominating ranks
            semantic_matches.sort(key=lambda m: getattr(m, "score", 0.0), reverse=True)
            # Ensure we only pass the top_k best matching chunks overall
            semantic_matches = semantic_matches[:strategy.pinecone_top_k]
        except Exception as e:
            print(f"[WARNING] Pinecone retrieval failed: {e}. Falling back to BM25 only.")
            
    print(f"[DEBUG] Pinecone took {time.time()-pinecone_start:.4f}s")

    bm25_start = time.time()
    keyword_matches = []
    if bm25_tasks:
        try:
            bm25_results = await asyncio.gather(*bm25_tasks)
            for matches in bm25_results:
                keyword_matches.extend(matches)
                
            # Sort globally by bm25_score descending
            keyword_matches.sort(key=lambda m: m.get("bm25_score", 0.0), reverse=True)
            keyword_matches = keyword_matches[:strategy.bm25_top_k]
        except Exception as e:
            print(f"[WARNING] BM25 retrieval failed: {e}.")
            
    print(f"[DEBUG] BM25 took {time.time()-bm25_start:.4f}s")

    print(
        f"[DEBUG] Pinecone returned {len(semantic_matches)} matches"
    )

    print(
        f"[DEBUG] BM25 returned {len(keyword_matches)} matches"
    )

    # ----------------------------------------
    # Hybrid Search
    # ----------------------------------------

    hybrid_start = time.time()
    fused_candidates = hybrid_search_rrf(
        semantic_matches,
        keyword_matches,
        top_k=30,
    )

    print(
        f"[DEBUG] Hybrid fused {len(fused_candidates)} candidates."
    )

    # ----------------------------------------
    # First Reranking
    # ----------------------------------------
    
    first_pass_k = strategy.rerank_top_k
    if not is_pg and not is_ug:
        # If ambiguous, ensure enough chunks are kept for both UG and PG
        first_pass_k = max(first_pass_k, 16)

    rerank_start = time.time()
    top_reranked_chunks = rerank_chunks(
        query,
        query_vector,
        fused_candidates,
        top_k=first_pass_k,
        max_candidates=len(fused_candidates),
    )

    print(
        f"[DEBUG] First reranking completed in {time.time()-rerank_start:.4f}s"
    )
    expanded_chunks = list(top_reranked_chunks)

    # ----------------------------------------
    # Section Expansion
    # ----------------------------------------

    if strategy.expand_section:

        expanded_chunks = expand_section(
            expanded_chunks,
            fused_candidates,
        )

        print(
            f"[DEBUG] Section Expansion -> {len(expanded_chunks)} chunks"
        )

    # ----------------------------------------
    # Page Expansion
    # ----------------------------------------

    if strategy.expand_page:

        expanded_chunks = expand_page(
            expanded_chunks,
            fused_candidates,
        )

        print(
            f"[DEBUG] Page Expansion -> {len(expanded_chunks)} chunks"
        )

    # ----------------------------------------
    # Table Expansion
    # ----------------------------------------

    if strategy.expand_table:

        expanded_chunks = expand_table(
            expanded_chunks,
            fused_candidates,
        )

        print(
            f"[DEBUG] Table Expansion -> {len(expanded_chunks)} chunks"
        )

    # ----------------------------------------
    # Second Reranking
    # ----------------------------------------

    # ----------------------------------------
    # High-Speed Deduplication & Ordering (Replaces Second Rerank)
    # ----------------------------------------
    unique = {}
    for chunk in expanded_chunks:
        # Create a robust composite key handling both flat dicts and Pinecone structure
        page = chunk.get("source_page") or chunk.get("page") or chunk.get("metadata", {}).get("page", "Unknown")
        idx = chunk.get("chunk_index") or chunk.get("metadata", {}).get("chunk_index", 0)
        text_snippet = chunk.get("text", chunk.get("metadata", {}).get("text", ""))[:100]
        
        key = (page, idx, text_snippet)
        unique[key] = chunk

    top_reranked_chunks = list(unique.values())
    print(f"[DEBUG] Deduplicated expansion pool down to {len(top_reranked_chunks)} unique chunks")

    # ----------------------------------------
    # Remove Duplicates
    # ----------------------------------------

    unique = {}

    for chunk in top_reranked_chunks:

        key = (
            chunk.get("source_page"),
            chunk.get("chunk_index"),
            chunk.get("text"),
        )

        unique[key] = chunk

    top_reranked_chunks = list(unique.values())

    # ----------------------------------------
    # Sort by Page
    # ----------------------------------------

    top_reranked_chunks.sort(
        key=lambda c: (
            c.get("source_page", 9999),
            c.get("chunk_index", 0),
        )
    )

    # ----------------------------------------
    # Build Context
    # ----------------------------------------

    context = build_context(
        top_reranked_chunks
    )



    print(
        f"[LATENCY] Retrieval Pipeline: {time.time()-t_start:.4f}s"
    )

    return {
        "context": context
    }