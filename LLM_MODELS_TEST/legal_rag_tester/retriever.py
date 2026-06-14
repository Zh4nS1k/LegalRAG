"""Pinecone retriever module with adaptive threshold fallback."""
from pinecone import Pinecone
from typing import List
from logger import pipeline_logger
from config import Settings
from models.schemas import Chunk


class PineconeRetriever:
    """Handles vector search queries to Pinecone."""

    def __init__(self, settings: Settings):
        """Initializes the retriever with Pinecone credentials."""
        self.settings = settings
        self.pc = Pinecone(api_key=settings.pinecone_api_key)
        self.index = self.pc.Index(settings.pinecone_index_name)
        self.namespace = settings.pinecone_namespace
        self.top_k = settings.pinecone_top_k
        self.final_k = settings.pinecone_final_k
        self.score_threshold = settings.pinecone_score_threshold
        self.chunk_text_field = settings.chunk_text_field

        stats = self.index.describe_index_stats()
        self.index_dim = stats.dimension
        namespaces = list(stats.namespaces.keys())
        if self.namespace and self.namespace not in namespaces:
            pipeline_logger.log_warning(
                f"⚠️  Namespace '{self.namespace}' not found. "
                f"Available: {namespaces}. Falling back to default."
            )

    # ── Core retrieval ──────────────────────────────────────────────────

    def _query_pinecone(self, embedding: List[float], threshold: float) -> List[Chunk]:
        """Raw Pinecone query filtered by threshold."""
        response = self.index.query(
            namespace=self.namespace,
            vector=embedding,
            top_k=self.top_k,
            include_values=False,
            include_metadata=True,
        )
        matches = response.get("matches", [])
        chunks = []
        for m in matches:
            score = m.get("score", 0.0)
            if score < threshold:
                continue
            metadata = m.get("metadata", {})
            text = metadata.get(self.chunk_text_field, "")
            if not text.strip():
                continue
            chunks.append(Chunk(
                id=m.get("id", ""),
                text=text,
                score=score,
                metadata=metadata,
            ))
        return sorted(chunks, key=lambda c: c.score, reverse=True)

    def query_raw(self, embedding: List[float], threshold: float = 0.30) -> List[Chunk]:
        """Low-level query with custom threshold — used for coverage diagnostics."""
        return self._query_pinecone(embedding, threshold)

    # ── Ranking helpers ─────────────────────────────────────────────────

    def _text_overlap(self, a: str, b: str) -> float:
        """Jaccard similarity on word sets as diversity proxy."""
        words_a = set(a.lower().split())
        words_b = set(b.lower().split())
        if not words_a or not words_b:
            return 0.0
        return len(words_a & words_b) / len(words_a | words_b)

    def _mmr_rerank(self, chunks: List[Chunk], final_k: int, lambda_param: float = 0.7) -> List[Chunk]:
        if len(chunks) <= final_k:
            return chunks

        selected = []
        remaining = list(chunks)

        while len(selected) < final_k and remaining:
            if not selected:
                selected.append(remaining.pop(0))
                continue
            best_chunk = max(
                remaining,
                key=lambda c: (
                    lambda_param * c.score
                    + (1 - lambda_param) * (1.0 - max(self._text_overlap(c.text, s.text) for s in selected))
                )
            )
            remaining.remove(best_chunk)
            selected.append(best_chunk)

        return selected

    def _keyword_boost(self, query: str, chunks: List[Chunk]) -> List[Chunk]:
        import re
        articles = re.findall(r'[Сс]татья\s+\d+[\-\d]*', query)
        law_keywords = re.findall(
            r'(Гражданск\w+\s+[Кк]одекс|ГК\s+РК|ТК\s+РК|УК\s+РК|'
            r'адвокатск\w+|потребител\w+|трудов\w+|налогов\w+|'
            r'административн\w+|семейн\w+|жилищн\w+)',
            query, re.IGNORECASE
        )
        all_terms = [a.lower() for a in articles] + [k.lower() for k in law_keywords]

        if not all_terms:
            return chunks

        for chunk in chunks:
            text_lower = chunk.text.lower()
            matches = sum(1 for term in all_terms if term in text_lower)
            if matches > 0:
                chunk.score = min(1.0, chunk.score + min(matches * 0.05, 0.15))
                chunk.metadata["keyword_boost"] = matches

        return sorted(chunks, key=lambda c: c.score, reverse=True)

    # ── Public query ────────────────────────────────────────────────────

    def query(self, embedding: List[float], query_text: str = "") -> List[Chunk]:
        """Query Pinecone with adaptive threshold fallback."""
        query_dim = len(embedding)
        assert self.index_dim == query_dim, (
            f"DIMENSION MISMATCH: index={self.index_dim}, embedder={query_dim}. "
            f"Wrong embedding model configured."
        )

        if query_dim > 0 and len(embedding) > 0:
            # Log metadata keys on first call
            probe = self.index.query(
                namespace=self.namespace, vector=embedding, top_k=1,
                include_values=False, include_metadata=True
            )
            probe_matches = probe.get("matches", [])
            if probe_matches:
                pipeline_logger.log_simple_info(
                    f"📄 Pinecone metadata keys: {list(probe_matches[0].metadata.keys())}"
                )

        # ── Adaptive threshold fallback ─────────────────────────────────
        chunks = self._query_pinecone(embedding, threshold=self.score_threshold)

        fallback_thresholds = [0.70, 0.65, 0.60, 0.55, 0.50]
        for threshold in fallback_thresholds:
            if len(chunks) >= 2:
                break
            if threshold >= self.score_threshold:
                continue  # only try lower thresholds
            pipeline_logger.log_warning(
                f"⚠️  Only {len(chunks)} chunks above {self.score_threshold:.2f}. "
                f"Retrying with threshold={threshold:.2f}"
            )
            chunks = self._query_pinecone(embedding, threshold=threshold)

        if chunks:
            actual_min = min(c.score for c in chunks)
            pipeline_logger.log_simple_info(
                f"📄 Retrieved {len(chunks)} chunks, min_score={actual_min:.3f}"
            )

        if query_text:
            chunks = self._keyword_boost(query_text, chunks)

        chunks = self._mmr_rerank(chunks, self.final_k)

        return chunks