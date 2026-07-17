import hashlib
import json
import os
from typing import List

from dotenv import load_dotenv
from langchain_core.documents import Document
from pinecone import Pinecone, ServerlessSpec

from core.ingestion.embedder import embed_documents

load_dotenv()


class PineconeStore:
    def __init__(self,
                 index_name: str = "rag-chatbot-index",
                 dimension: int | None = None):
        self.index_name = index_name
        # 1. Update default to 384 dimensions
        self.dimension = dimension or int(os.getenv("EMBEDDING_DIMENSION", "384"))
        self.pc = Pinecone(api_key=os.environ["PINECONE_API_KEY"])
        self._ensure_index()
        self.index = self.pc.Index(self.index_name)

    def _ensure_index(self):
        names = self.pc.list_indexes().names()
        if self.index_name not in names:
            self.pc.create_index(
                name=self.index_name,
                dimension=self.dimension,
                metric="cosine",
                spec=ServerlessSpec(cloud="aws", region="us-east-1"),
            )

    def clear_academic_level(self, academic_level: str):
        self.index.delete(filter={"academic_level": academic_level})

    @staticmethod
    def _vector_id(doc: Document) -> str:
        base = (
            f"{doc.metadata.get('academic_level')}|"
            f"{doc.metadata.get('year')}|"
            f"{doc.metadata.get('page')}|"
            f"{doc.metadata.get('chunk_index')}|"
            f"{doc.page_content[:200]}"
        )
        return hashlib.sha256(base.encode()).hexdigest()

    @staticmethod
    def _embedding_text(doc: Document) -> str:
        # 2. Rich semantic parent breadcrumbs prepended to the text before vector generation
        parts = []
        if doc.metadata.get("faculty"):
            parts.append(f"Faculty: {doc.metadata.get('faculty')}")
        if doc.metadata.get("department"):
            parts.append(f"Department: {doc.metadata.get('department')}")
        if doc.metadata.get("program"):
            parts.append(f"Program: {doc.metadata.get('program')}")
        if doc.metadata.get("heading_path"):
            parts.append(f"Path: {doc.metadata.get('heading_path')}")
            
        parts.append(doc.page_content)
        return "\n".join([p for p in parts if p])

    def upsert_documents(self,
                         documents: List[Document],
                         batch_size: int = 100):
        texts = [self._embedding_text(d) for d in documents]
        vectors = embed_documents(texts)

        payload = []

        for doc, emb in zip(documents, vectors):
            md = dict(doc.metadata)
            md["text"] = doc.page_content

            # Filter out None values and serialize complex objects
            clean_metadata = {}
            for k, v in md.items():
                if v is not None:
                    if isinstance(v, (dict, list)):
                        # Pinecone only allows list of strings. If it's a list of dicts, or a dict, stringify it
                        if isinstance(v, list) and len(v) > 0 and not isinstance(v[0], str):
                            clean_metadata[k] = json.dumps(v)
                        elif isinstance(v, dict):
                            clean_metadata[k] = json.dumps(v)
                        else:
                            clean_metadata[k] = v
                    else:
                        clean_metadata[k] = v

            payload.append({
                "id": self._vector_id(doc),
                "values": emb,
                "metadata": clean_metadata,
            })

        for i in range(0, len(payload), batch_size):
            self.index.upsert(vectors=payload[i:i + batch_size])

    def replace_documents(self,
                          documents: List[Document],
                          academic_level: str):
        self.clear_academic_level(academic_level)
        self.upsert_documents(documents)
