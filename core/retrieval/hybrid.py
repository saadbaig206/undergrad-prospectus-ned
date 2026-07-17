# In hybrid.py

def hybrid_search_rrf(semantic_results: list, keyword_results: list, top_k: int = 15) -> list:
    rrf_scores = {}
    
    # 1. Parse Pinecone semantic matches
    for rank, match in enumerate(semantic_results):
        metadata = dict(match.metadata)
        text = metadata.get("text", "").strip()
        page = metadata.get("source_page", metadata.get("page", "Unknown"))
        
        # Aligned key combination
        text_signature = "".join(text.lower().split())[:100]
        key = (str(page), text_signature)
        
        if key not in rrf_scores:
            rrf_scores[key] = {
                **metadata,
                "text": text,
                "rank_sem": rank,
                "rank_key": None,
            }
        else:
            rrf_scores[key]["rank_sem"] = rank

    # 2. Parse BM25 keyword matches
    for rank, chunk in enumerate(keyword_results):
        metadata = dict(chunk)
        text = metadata.get("text", "").strip()
        page = metadata.get("source_page", metadata.get("page", "Unknown"))
        
        # Match keys exactly with Pinecone's key
        text_signature = "".join(text.lower().split())[:100]
        key = (str(page), text_signature)
        
        if key not in rrf_scores:
            rrf_scores[key] = {
                **metadata,
                "text": text,
                "rank_sem": None,
                "rank_key": rank,
            }
        else:
            rrf_scores[key]["rank_key"] = rank
            
    # 3. Compute RRF Scores
    fused_results = []
    for item in rrf_scores.values():
        score = 0.0
        if item["rank_sem"] is not None:
            score += 1.0 / (50.0 + item["rank_sem"])
        if item["rank_key"] is not None:
            score += 1.0 / (50.0 + item["rank_key"])
            
        item["rrf_score"] = score
        fused_results.append((score, item))
        
    fused_results.sort(key=lambda x: x[0], reverse=True)
    return [item[1] for item in fused_results[:top_k]]