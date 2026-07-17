"""Small, dependency-free BM25 index over locally generated prospectus chunks."""

import asyncio
import json
import math
import os
import re

from core.utils.abbreviations import normalize_token


class LocalBM25:
    def __init__(self):
        self.chunks = []
        self.doc_freqs = {}
        self.avg_doc_len = 0.0
        self.idf = {}
        self.k1 = 1.5
        self.b = 0.75
        self.loaded = False

    def load_chunks(self, academic_level: str) -> bool:
        prefix = "UG" if academic_level == "undergraduate" else "PG"
        path = os.path.join("output_chunks", f"{prefix}Prospectus_compiled_knowledge.json")
        if not os.path.exists(path):
            return False

        with open(path, "r", encoding="utf-8") as file:
            self.chunks = json.load(file)

        document_lengths = []
        for chunk in self.chunks:
            chunk["academic_level"] = academic_level
            tokens = self.tokenize(chunk.get("text", ""))
            chunk["tokens"] = tokens
            term_freqs = {}
            for token in tokens:
                term_freqs[token] = term_freqs.get(token, 0) + 1
            chunk["term_freqs"] = term_freqs
            for token in term_freqs:
                self.doc_freqs[token] = self.doc_freqs.get(token, 0) + 1
            document_lengths.append(len(tokens))

        document_count = len(self.chunks)
        self.avg_doc_len = sum(document_lengths) / document_count if document_count else 0.0
        self.idf = {
            term: math.log((document_count - frequency + 0.5) / (frequency + 0.5) + 1.0)
            for term, frequency in self.doc_freqs.items()
        }
        self.metadata_index = {}
        for chunk in self.chunks:
            dept = chunk.get("department")
            sect = chunk.get("section")
            if dept or sect:
                # Store by normalized keys for rapid O(1) retrieval
                lookup_key = (dept.strip() if dept else None, sect.strip() if sect else None)
                if lookup_key not in self.metadata_index:
                    self.metadata_index[lookup_key] = []
                self.metadata_index[lookup_key].append(chunk)

        self.loaded = True
        return True

    @staticmethod
    def tokenize(text: str) -> list[str]:
        return [normalize_token(token) for token in re.findall(r"\w+", text.lower())]

    def score(self, query: str, top_k: int = 12, metadata_filter: dict = None) -> list[dict]:
        if not self.loaded or not self.avg_doc_len:
            return []

        scored = []
        for chunk in self.chunks:
            if metadata_filter:
                skip = False
                for k, v in metadata_filter.items():
                    # Handle Pinecone $in filter syntax for arrays
                    if isinstance(v, dict) and "$in" in v:
                        target_list = chunk.get(k) or chunk.get("metadata", {}).get(k, [])
                        if not any(item in target_list for item in v["$in"]):
                            skip = True
                            break
                    elif isinstance(v, dict) and "$eq" in v:
                        val = chunk.get(k) or chunk.get("metadata", {}).get(k)
                        if val != v["$eq"]:
                            skip = True
                            break
                    else:
                        val = chunk.get(k) or chunk.get("metadata", {}).get(k)
                        if val != v:
                            skip = True
                            break
                if skip:
                    continue

            score = 0.0
            document_length = len(chunk["tokens"])
            for token in self.tokenize(query):
                frequency = chunk["term_freqs"].get(token, 0)
                if not frequency:
                    continue
                numerator = frequency * (self.k1 + 1)
                denominator = frequency + self.k1 * (1 - self.b + self.b * document_length / self.avg_doc_len)
                score += self.idf.get(token, 0.0) * numerator / denominator
            if score:
                scored_chunk = dict(chunk)
                scored_chunk["bm25_score"] = score
                scored.append((score, scored_chunk))

        scored.sort(key=lambda item: item[0], reverse=True)
        return [chunk for _, chunk in scored[:top_k]]


_instances: dict[str, LocalBM25] = {}


def get_bm25_instance(academic_level: str) -> LocalBM25 | None:
    if academic_level not in _instances:
        instance = LocalBM25()
        if not instance.load_chunks(academic_level):
            return None
        _instances[academic_level] = instance
    return _instances[academic_level]


async def retrieve_bm25(query: str, academic_level: str, top_k: int, metadata_filter: dict = None) -> list[dict]:
    instance = get_bm25_instance(academic_level)
    if instance is None:
        return []
    return await asyncio.to_thread(instance.score, query, top_k, metadata_filter)
