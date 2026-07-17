from core.retrieval.bm25 import get_bm25_instance

def retrieve_related_chunks(academic_level: str, department: str = None, section: str = None):
    bm25 = get_bm25_instance(academic_level)
    if bm25 is None or not hasattr(bm25, "metadata_index"):
        return []
    
    # Query the pre-built index in O(1) time
    lookup_key = (department.strip() if department else None, section.strip() if section else None)
    return bm25.metadata_index.get(lookup_key, [])