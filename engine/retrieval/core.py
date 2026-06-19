from __future__ import annotations

import asyncio
import concurrent.futures
import contextvars
import re
from typing import Any, Callable, Iterable, Sequence

from langchain_core.documents import Document
from langchain_core.retrievers import BaseRetriever
from pydantic import Field

from engine.retrieval.scoring import fuse_retrieval_candidates, normalize_article_number
from engine.retrieval.query_analysis import (
    _detect_target_codes,
    _focus_articles_from_query,
    _rewrite_query_for_retrieval,
)


class MinimalLegalRetriever(BaseRetriever):
    bm25_search: Callable[[str], Sequence[Document]]
    dense_search: Callable[[str], Sequence[tuple[Document, float]]]
    rewrite_query_fn: Callable[[str], str] = Field(default=_rewrite_query_for_retrieval)
    bm25_weight: float = 0.55
    dense_weight: float = 0.45
    candidate_k: int = 40
    final_k: int = 10

    def _get_relevant_documents(
        self, query: str, *, run_manager=None
    ) -> list[Document]:
        rewritten = self.rewrite_query_fn(query) if self.rewrite_query_fn else query

        # Parallel search using threads for synchronous invoke.
        # We manually propagate metrics_ctx (which holds a dict) to threads.
        from engine.utils.latency import metrics_ctx
        current_metrics = metrics_ctx.get()

        def _run_with_ctx(func, *args):
            token = metrics_ctx.set(current_metrics)
            try:
                return func(*args)
            finally:
                metrics_ctx.reset(token)

        with concurrent.futures.ThreadPoolExecutor(max_workers=2) as executor:
            if current_metrics is not None:
                bm25_future = executor.submit(_run_with_ctx, self.bm25_search, rewritten)
                dense_future = executor.submit(_run_with_ctx, self.dense_search, rewritten)
            else:
                bm25_future = executor.submit(self.bm25_search, rewritten)
                dense_future = executor.submit(self.dense_search, rewritten)
            
            bm25_res = bm25_future.result()
            dense_res = dense_future.result()

        bm25_docs = list(bm25_res or [])[: self.candidate_k]
        dense_docs = list(dense_res or [])[: self.candidate_k]

        target_codes = _detect_target_codes(query)
        target_articles = sorted(_focus_articles_from_query(query))
        return fuse_retrieval_candidates(
            query=query,
            bm25_docs=bm25_docs,
            dense_docs=dense_docs,
            bm25_weight=self.bm25_weight,
            dense_weight=self.dense_weight,
            target_codes=target_codes,
            target_articles=target_articles,
            limit=self.final_k,
        )

    async def _aget_relevant_documents(
        self, query: str, *, run_manager=None
    ) -> list[Document]:
        rewritten = self.rewrite_query_fn(query) if self.rewrite_query_fn else query

        # Parallel search
        bm25_task = asyncio.to_thread(self.bm25_search, rewritten)
        dense_task = asyncio.to_thread(self.dense_search, rewritten)

        bm25_res, dense_res = await asyncio.gather(bm25_task, dense_task)

        bm25_docs = list(bm25_res or [])[: self.candidate_k]
        dense_docs = list(dense_res or [])[: self.candidate_k]

        target_codes = _detect_target_codes(query)
        target_articles = sorted(_focus_articles_from_query(query))

        return fuse_retrieval_candidates(
            query=query,
            bm25_docs=bm25_docs,
            dense_docs=dense_docs,
            bm25_weight=self.bm25_weight,
            dense_weight=self.dense_weight,
            target_codes=target_codes,
            target_articles=target_articles,
            limit=self.final_k,
        )
