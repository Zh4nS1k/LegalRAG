from __future__ import annotations

from typing import Any, Callable, Iterable, Sequence

from langchain_core.documents import Document
from langchain_core.retrievers import BaseRetriever
from pydantic import Field

from ai_service.retrieval.query_rewrite import rewrite_query
from ai_service.retrieval.scoring import fuse_retrieval_candidates, normalize_article_number


def _extract_query_article_number(query: str) -> str | None:
    text = str(query or "")
    for token in text.replace("ст.", " ").replace("ст", " ").replace("бап", " ").split():
        article = normalize_article_number(token)
        if article:
            return article
    return None


def _focus_articles_from_query(query: str) -> set[str]:
    q = (query or "").lower()
    focus: set[str] = set()
    if any(token in q for token in ("краж", "украл", "украду", "урл", "ұрлық", "жымқыру")):
        focus.add("188")
    if any(token in q for token in ("мошеннич", "обман", "алаяқ", "fraud")):
        focus.add("190")
    if any(token in q for token in ("шум", "тишин", "тыш", "покой")):
        focus.add("437")
    return focus


def _expand_legal_synonyms(query: str) -> list[str]:
    q = (query or "").lower()
    extras: list[str] = []
    if any(token in q for token in ("краж", "украл", "урл", "ұрлық")):
        extras.extend(["кража", "тайное хищение чужого имущества"])
    if any(token in q for token in ("мошеннич", "обман", "алаяқ")):
        extras.extend(["мошенничество", "хищение путем обмана"])
    if any(token in q for token in ("возврат", "товар", "гарант", "брак")):
        extras.extend(["возврат товара", "защита прав потребителей"])
    return extras


def _normalize_target_codes(query: str) -> list[str]:
    q = (query or "").lower()
    codes: list[str] = []
    if any(token in q for token in ("краж", "мошеннич", "обман", "уголов", "преступ")):
        codes.append("Уголовный кодекс РК")
    if any(token in q for token in ("шум", "тишин", "административ", "штраф")):
        codes.append("Кодекс об административных правонарушениях РК")
    if any(token in q for token in ("товар", "возврат", "гарант", "потребител")):
        codes.append("Закон о защите прав потребителей РК")
    if any(token in q for token in ("труд", "увольн", "зарплат", "работодатель")):
        codes.append("Трудовой кодекс РК")
    if any(token in q for token in ("развод", "алимент", "ребен", "отбасы")):
        codes.append("Кодекс о браке и семье РК")
    return list(dict.fromkeys(codes))


def _rewrite_legal_query(query: str) -> str:
    return rewrite_query(
        query,
        llm=None,
        detect_target_codes=_normalize_target_codes,
        extract_query_article_number=_extract_query_article_number,
        focus_articles_from_query=_focus_articles_from_query,
        expand_legal_synonyms=_expand_legal_synonyms,
    )


class MinimalLegalRetriever(BaseRetriever):
    bm25_search: Callable[[str], Sequence[Document]]
    dense_search: Callable[[str], Sequence[tuple[Document, float]]]
    rewrite_query_fn: Callable[[str], str] = Field(default=_rewrite_legal_query)
    bm25_weight: float = 0.55
    dense_weight: float = 0.45
    candidate_k: int = 40
    final_k: int = 10

    def _get_relevant_documents(
        self, query: str, *, run_manager=None
    ) -> list[Document]:
        rewritten = self.rewrite_query_fn(query) if self.rewrite_query_fn else query
        bm25_docs = list(self.bm25_search(rewritten) or [])[: self.candidate_k]
        dense_docs = list(self.dense_search(rewritten) or [])[: self.candidate_k]
        target_codes = _normalize_target_codes(query)
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
