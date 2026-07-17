from typing import List


def retrieval_confidence(matches: List) -> float:
    """
    Computes the average similarity score of
    the top Pinecone matches.
    """

    if not matches:
        return 0.0

    scores = []

    for m in matches[:5]:
        score = getattr(m, "score", None)

        if score is None:
            score = m.get("score", 0) if isinstance(m, dict) else 0

        scores.append(score)

    return sum(scores) / len(scores)


def is_low_confidence(matches: List) -> bool:
    """
    Returns True if retrieval confidence is poor.
    """

    confidence = retrieval_confidence(matches)

    return confidence < 0.45
