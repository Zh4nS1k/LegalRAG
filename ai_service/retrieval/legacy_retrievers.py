from __future__ import annotations

from typing import Any, List

from langchain_core.callbacks import CallbackManagerForRetrieverRun
from langchain_core.documents import Document
from langchain_core.retrievers import BaseRetriever

from ai_service.retrieval.rag_chain import (
    _accumulate_candidate,
    _adaptive_wide_k,
    _collect_bm25_candidates,
    _collect_vector_candidates,
    _detect_target_codes,
    _extract_article_range,
    _extract_query_article_numbers,
    _filter_docs_by_codes,
    _fuse_retrieval_candidates,
    _focus_articles_from_query,
    _get_target_code_profile,
    _is_criminal_query,
    _merge_unique,
    _multi_query_retrieve,
    _multi_query_search_with_code_filters,
    _needs_circumstances_query,
    _normalize_article_number,
    _prioritize_docs,
    _rank_docs_with_legal_scoring,
    _sort_docs_for_coverage,
)


class _FilterByCodeRetriever(BaseRetriever):
    """Compatibility wrapper for code/article post-filtering."""

    retriever: Any
    allowed_code_ru: List[str] | None = None
    article_number: str | None = None

    def _get_relevant_documents(
        self, query: str, *, run_manager: CallbackManagerForRetrieverRun | None = None
    ) -> List[Document]:
        docs = self.retriever.invoke(query)
        filtered = docs

        if self.allowed_code_ru:
            allowed = set(self.allowed_code_ru)
            filtered = [
                d
                for d in filtered
                if (d.metadata.get("code_ru") or "").strip() in allowed
            ]

        if self.article_number:
            filtered = [
                d
                for d in filtered
                if (d.metadata.get("article_number") or "").strip() == self.article_number
            ]

        return filtered if filtered else docs[:5]


class QueryAwareHybridRetriever(BaseRetriever):
    vector_retriever: Any | None = None
    bm25_retriever: Any | None = None
    default_k: int = 24

    def _get_relevant_documents(
        self, query: str, *, run_manager: CallbackManagerForRetrieverRun | None = None
    ) -> List[Document]:
        target_codes, confident_code = _get_target_code_profile(query)
        target_articles = _extract_query_article_numbers(query)
        wide_k = _adaptive_wide_k(query, target_codes, target_articles)
        vector_candidates = _collect_vector_candidates(
            query,
            wide_k=wide_k,
            target_codes=target_codes,
            target_articles=target_articles,
        )
        bm25_candidates = _collect_bm25_candidates(
            query,
            bm25_retriever=self.bm25_retriever,
            wide_k=wide_k,
        )

        if not vector_candidates and self.vector_retriever is not None:
            try:
                fallback_docs = list(self.vector_retriever.invoke(query))
            except Exception:
                fallback_docs = []
            for rank, doc in enumerate(fallback_docs[:wide_k]):
                _accumulate_candidate(
                    vector_candidates,
                    doc,
                    "vector",
                    1.0 - (rank / max(len(fallback_docs), 1)),
                )

        if not vector_candidates and not bm25_candidates:
            return []

        docs = _fuse_retrieval_candidates(
            query,
            vector_candidates=vector_candidates,
            bm25_candidates=bm25_candidates,
            target_codes=target_codes,
            target_articles=target_articles,
            limit=wide_k,
        )
        if not confident_code and target_codes:
            filtered = _filter_docs_by_codes(docs, target_codes)
            if filtered:
                docs = _prioritize_docs(
                    query,
                    _merge_unique(filtered, docs),
                    target_codes=target_codes,
                    target_articles=target_articles,
                    limit=wide_k,
                )
        return docs


class _HeuristicRetriever(BaseRetriever):
    """Compatibility wrapper for the older heuristic expansion path."""

    base_retriever: Any
    vector_store: Any

    def _get_relevant_documents(
        self, query: str, *, run_manager: CallbackManagerForRetrieverRun | None = None
    ) -> List[Document]:
        docs = _multi_query_retrieve(self.base_retriever, query)
        target_codes = _detect_target_codes(query)
        target_articles = _extract_query_article_numbers(query)
        if target_codes:
            filtered = _filter_docs_by_codes(docs, target_codes)
            if filtered:
                docs = _merge_unique(filtered, docs)
            fallback_docs = _multi_query_search_with_code_filters(
                query,
                target_codes,
                k=6,
                article_numbers=target_articles,
            )
            if fallback_docs:
                docs = _merge_unique(docs, fallback_docs)
        elif _is_criminal_query(query):
            filtered = _filter_docs_by_codes(docs, ["Уголовный кодекс РК"])
            if filtered:
                docs = _merge_unique(filtered, docs)
            fallback_docs = _multi_query_search_with_code_filters(
                query,
                ["Уголовный кодекс РК"],
                k=6,
                article_numbers=target_articles,
            )
            if fallback_docs:
                docs = _merge_unique(docs, fallback_docs)
        range_match = _extract_article_range(query)
        if range_match:
            start, end = range_match
            focused = [
                d
                for d in docs
                if _normalize_article_number(d.metadata.get("article_number")).isdigit()
                and start
                <= int(_normalize_article_number(d.metadata.get("article_number")))
                <= end
            ]
            if focused:
                docs = _merge_unique(focused, docs)
        focus = _focus_articles_from_query(query)
        if focus:
            focused = [
                d
                for d in docs
                if _normalize_article_number(d.metadata.get("article_number")) in focus
            ]
            if focused:
                docs = _merge_unique(focused, docs)
        docs = _sort_docs_for_coverage(
            docs,
            target_codes=target_codes,
            target_articles=target_articles,
        )
        return _rank_docs_with_legal_scoring(
            query,
            docs,
            target_codes=target_codes,
            target_articles=target_articles,
        )


class _LawAwareRetriever(BaseRetriever):
    """Compatibility wrapper for the older law-aware path."""

    base_retriever: Any
    vector_store: Any
    min_k_criminal: int = 10

    def _get_relevant_documents(
        self, query: str, *, run_manager: CallbackManagerForRetrieverRun | None = None
    ) -> List[Document]:
        docs = _multi_query_retrieve(self.base_retriever, query)
        target_codes = _detect_target_codes(query)
        target_articles = _extract_query_article_numbers(query)
        article_number = target_articles[0] if target_articles else None
        if target_codes:
            filtered = _filter_docs_by_codes(docs, target_codes)
            if filtered:
                docs = _merge_unique(filtered, docs)
            if len(docs) < min(24, 6):
                extra = _multi_query_search_with_code_filters(
                    query,
                    target_codes,
                    k=max(24, 8),
                    article_number=article_number,
                    article_numbers=target_articles,
                )
                if extra:
                    docs = _merge_unique(docs, extra)
            elif target_articles:
                extra = _multi_query_search_with_code_filters(
                    query,
                    target_codes,
                    k=4,
                    article_numbers=target_articles,
                )
                if extra:
                    docs = _merge_unique(docs, extra)
        elif _is_criminal_query(query):
            filtered = _filter_docs_by_codes(docs, ["Уголовный кодекс РК"])
            if filtered:
                docs = _merge_unique(filtered, docs)
            if len(docs) < self.min_k_criminal:
                extra = _multi_query_search_with_code_filters(
                    query,
                    ["Уголовный кодекс РК"],
                    k=self.min_k_criminal,
                    article_number=article_number,
                    article_numbers=target_articles,
                )
                if extra:
                    docs = _merge_unique(docs, extra)

        if _needs_circumstances_query(query):
            extra_docs: list[Document] = []
            for q in (
                "смягчающие обстоятельства УК РК",
                "отягчающие обстоятельства УК РК",
                "жеңілдететін мән-жайлар Қылмыстық кодекс",
                "ауырлататын мән-жайлар Қылмыстық кодекс",
            ):
                try:
                    extra_docs.extend(
                        _multi_query_search_with_code_filters(
                            q,
                            ["Уголовный кодекс РК"],
                            k=4,
                            article_numbers=target_articles,
                        )
                    )
                except Exception:
                    continue
            if extra_docs:
                docs = _merge_unique(docs, extra_docs)

        docs = _sort_docs_for_coverage(
            docs,
            target_codes=target_codes,
            target_articles=target_articles,
        )
        return _rank_docs_with_legal_scoring(
            query,
            docs,
            target_codes=target_codes,
            target_articles=target_articles,
        )
