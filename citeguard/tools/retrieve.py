from __future__ import annotations
from citeguard.retrieval.base import RetrieverBase
from citeguard.tools.session import VerificationSession


def make_retrieve_tools(
    retriever: RetrieverBase | None,
    session: VerificationSession,
    max_requery_rounds: int = 3,
):
    """Return (retrieve_chunks, re_query) tool functions capturing retriever and session."""

    def retrieve_chunks(query: str, top_k: int = 5) -> dict:
        """Retrieve relevant text chunks from the cited paper.

        Args:
            query: Search query derived from the manuscript claim.
            top_k: Number of top-scoring chunks to return.

        Returns:
            dict with 'chunks' list (text, page, score) or 'error' if unavailable.
        """
        if retriever is None:
            return {"error": "No PDF matched for this citation.", "chunks": []}
        chunks = retriever.query(query, top_k)
        session.evidence_chunks.extend(chunks)
        return {
            "chunks": [
                {"text": c.text, "page": c.page, "score": round(c.score, 4)}
                for c in chunks
            ],
            "total": len(chunks),
        }

    def re_query(refined_query: str, reason: str = "", top_k: int = 8) -> dict:
        """Re-query with a refined search term when initial evidence is insufficient.

        Args:
            refined_query: A more specific or differently phrased search query.
            reason: Why the original query was insufficient.
            top_k: Number of chunks to retrieve.

        Returns:
            dict with 'chunks' list or 'error' if max rounds reached.
        """
        if session.re_query_count >= max_requery_rounds:
            return {
                "error": f"Maximum {max_requery_rounds} re-query rounds reached.",
                "chunks": [],
            }
        session.re_query_count += 1
        if retriever is None:
            return {"error": "No PDF matched for this citation.", "chunks": []}
        chunks = retriever.query(refined_query, top_k)
        session.evidence_chunks.extend(chunks)
        return {
            "chunks": [
                {"text": c.text, "page": c.page, "score": round(c.score, 4)}
                for c in chunks
            ],
            "total": len(chunks),
            "round": session.re_query_count,
        }

    return retrieve_chunks, re_query
