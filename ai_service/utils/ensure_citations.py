from __future__ import annotations

import re
from typing import Any, Iterable

from ai_service.utils import citations as cit

_NOT_FOUND_MARKERS = (
    "информация не найдена",
    "ақпарат қолжетімді",
)
_CLARIFICATION_MARKERS = (
    "для точного правового заключения",
    "нужны уточнения",
)
_INTERNAL_ADVISORY_TAG = "[INTERNAL_ADVISORY]"
_SKIP_ARTICLE_VALUES = {"", "н/д", "n/d", "-", "неизвестно"}


def _is_kz_text(text: str) -> bool:
    return any(ch in (text or "").lower() for ch in "әғқңөұүһі")


def _normalize_article_value(value: object) -> str:
    return cit.normalize_article(str(value or ""))


def _source_article_numbers(sources: Iterable[Any]) -> set[str]:
    articles: set[str] = set()
    for doc in sources or []:
        meta = getattr(doc, "metadata", None)
        if meta is None and isinstance(doc, dict):
            meta = doc.get("metadata") or {}
        meta = meta or {}
        article = _normalize_article_value(meta.get("article_number"))
        if article and article.lower() not in _SKIP_ARTICLE_VALUES:
            articles.add(article)
    return articles


def format_doc_citation(meta: dict[str, Any]) -> str:
    article = str(meta.get("article_number") or "").strip()
    code_ru = str(meta.get("code_ru") or "").strip()
    normalized_article = _normalize_article_value(article)
    if not normalized_article or normalized_article.lower() in _SKIP_ARTICLE_VALUES:
        return ""
    if not code_ru or _looks_like_internal_code_id(code_ru):
        source = str(meta.get("source") or "").strip()
        if source:
            code_ru = source.split("/")[-1] if "/" in source else source
        else:
            code_ru = "НПА РК"
    return f"ст. {normalized_article} {code_ru}"


def _looks_like_internal_code_id(value: str) -> bool:
    text = str(value or "").strip().lower()
    return bool(text) and bool(re.fullmatch(r"[a-z0-9_]+", text)) and "_" in text


def response_cites_context(response: str, sources: Iterable[Any]) -> bool:
    text = (response or "").strip()
    if not text:
        return False

    if cit.extract_citation_pairs_from_text(text):
        return True

    source_articles = _source_article_numbers(sources)
    if not source_articles:
        return bool(re.search(r"(?:ст\.?|статья|статьи|бап)\s*\d", text.lower()))

    lowered = text.lower()
    for article in source_articles:
        if re.search(
            rf"(?:ст\.?|статья|статьи|бап)\s*{re.escape(article)}\b",
            lowered,
        ):
            return True
    return False


def _should_skip_citation_enforcement(
    response: str,
    *,
    intent: str | None = None,
) -> bool:
    if str(intent or "").upper() == "SOCIAL":
        return True

    text = (response or "").strip()
    if not text:
        return True

    lowered = text.lower()
    if any(marker in lowered for marker in _NOT_FOUND_MARKERS):
        return True
    if any(marker in lowered for marker in _CLARIFICATION_MARKERS):
        return True
    return False


def _build_sources_block(sources: Iterable[Any], *, kz: bool) -> str:
    lines: list[str] = []
    seen: set[str] = set()
    for doc in sources or []:
        meta = getattr(doc, "metadata", None)
        if meta is None and isinstance(doc, dict):
            meta = doc.get("metadata") or {}
        citation = format_doc_citation(dict(meta or {}))
        if not citation or citation in seen:
            continue
        seen.add(citation)
        lines.append(f"- {citation}")
        if len(lines) >= 6:
            break
    if not lines:
        return ""
    label = "Дереккөздер:" if kz else "Источники:"
    return f"{label}\n" + "\n".join(lines)


def ensure_answer_citations(
    response: str,
    sources: Iterable[Any],
    *,
    question: str | None = None,
    intent: str | None = None,
) -> str:
    """Mechanically guarantee article citations in legal answers (Rule 2)."""
    text = (response or "").strip()
    if _should_skip_citation_enforcement(text, intent=intent):
        return text

    source_list = list(sources or [])
    if not source_list:
        if _INTERNAL_ADVISORY_TAG in text:
            return text
        if re.search(r"(?:ст\.?|статья|статьи|бап)\s*\d", text.lower()):
            return f"{_INTERNAL_ADVISORY_TAG}\n\n{text}"
        return text

    if response_cites_context(text, source_list):
        return text

    block = _build_sources_block(
        source_list,
        kz=_is_kz_text(question or text),
    )
    if not block:
        return text

    if re.search(r"(?:источники|дереккөздер)\s*:", text, re.IGNORECASE):
        return f"{text}\n{block.split(chr(10), 1)[1]}"

    return f"{text}\n\n{block}"
