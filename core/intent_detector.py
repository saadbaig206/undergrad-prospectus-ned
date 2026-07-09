import os
import re
import json
import numpy as np
from core.embeddings import embed_query

class IntentDetector:

    def __init__(self):
        # Load precomputed embeddings
        embeddings_path = os.path.join(os.path.dirname(__file__), "intent_embeddings.json")
        with open(embeddings_path, "r") as f:
            data = json.load(f)
        
        self.seat_embeddings = [np.array(e) for e in data["seat_embeddings"]]
        self.general_embeddings = [np.array(e) for e in data["general_embeddings"]]


        self.seat_keywords = {
            "seat",
            "seats",
            "eats",
            "quota",
            "intake",
            "capacity",
            "strength",
            "vacancy",
            "vacancies",
            "batch size",
            "allocation",
            "distribution",
            "matrix"
        }

        self.seat_patterns = [
            r"how many.*students",
            r"how many.*seat",
            r"seat.*available",
            r"available.*seat",
            r"total.*seat",
            r"seat.*matrix",
            r"seat.*distribution",
            r"intake",
            r"capacity",
            r"quota"
        ]

    def cosine(self, a, b):

        return np.dot(a, b)

    def keyword_score(self, query):

        query = query.lower()

        score = 0

        for word in self.seat_keywords:

            if word in query:

                score += 1

        return min(score / 2, 1)

    def regex_score(self, query):

        query = query.lower()

        for pattern in self.seat_patterns:

            if re.search(pattern, query):

                return 1

        return 0

    def semantic_score(self, query, query_vector=None):
        if query_vector is not None:
            emb = np.array(query_vector)
        else:
            vector = embed_query(query)
            emb = np.array(vector)
            
        # Normalize the query embedding for cosine similarity
        norm = np.linalg.norm(emb)
        if norm > 0:
            emb = emb / norm

        seat = max(
            self.cosine(emb, e)
            for e in self.seat_embeddings
        )

        general = max(
            self.cosine(emb, e)
            for e in self.general_embeddings
        )

        return seat, general

    def classify(self, query, query_vector=None):
        # Direct keyword override for strong indicator words (including common typos like 'eats')
        query_clean = re.sub(r"[^\w\s]", "", query.lower().strip())
        words = set(query_clean.split())
        strong_indicators = {"seat", "seats", "eats", "sats", "seates", "quota", "vacancy", "vacancies", "allocation", "distribution", "matrix"}
        if any(w in words for w in strong_indicators):
            return "SEAT"

        keyword = self.keyword_score(query)

        regex = self.regex_score(query)

        seat_sim, general_sim = self.semantic_score(query, query_vector=query_vector)

        final_seat = (

            keyword * 0.25 +

            regex * 0.25 +

            seat_sim * 0.50

        )
        if final_seat > 0.72:

            return "SEAT"

        if general_sim > 0.82:

            return "GENERAL"

        return "RAG"