import unittest
from unittest.mock import AsyncMock, patch

from core.retrieval.retriever import retrieve


class RetrievalTests(unittest.IsolatedAsyncioTestCase):
    async def test_retrieval_builds_context_after_fusing_results(self):
        semantic_match = type(
            "Match",
            (),
            {"metadata": {"text": "Computer Science eligibility", "source_page": 5, "academic_level": "undergraduate"}},
        )()
        keyword_match = {"text": "Computer Science eligibility", "source_page": 5, "academic_level": "undergraduate"}

        with (
            patch("core.retrieval.retriever.retrieve_from_pinecone", AsyncMock(return_value=[semantic_match])) as pinecone,
            patch("core.retrieval.retriever.retrieve_bm25", AsyncMock(return_value=[keyword_match])),
            patch("core.retrieval.retriever.hybrid_search_rrf", return_value=[keyword_match]),
            patch("core.retrieval.retriever.rerank_chunks", return_value=[keyword_match]),
            patch("core.retrieval.retriever.build_context", return_value="retrieved context"),
        ):
            result = await retrieve(
                "What is the eligibility for Computer Science?",
                [0.1, 0.2],
                academic_level_filter={"academic_level": "undergraduate"},
                is_ug=True,
            )

        self.assertEqual(result, {"context": "retrieved context"})
        self.assertEqual(pinecone.await_args.kwargs["metadata_filter"], {
            "academic_level": "undergraduate",
            "department": "Computer Science",
        })
