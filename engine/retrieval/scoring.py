from __future__ import annotations

from typing import Any, Iterable, Sequence

from langchain_core.documents import Document

_CODE_PRIOR_MULTIPLIERS: tuple[tuple[tuple[str, ...], float], ...] = (
    (("уголовный кодекс рк", "қылмыстық кодекс", "ук рк"), 1.18),
    (("кодекс об административных правонарушениях рк", "әкімшілік құқық бұзушылық"), 1.12),
    (("закон о защите прав потребителей рк", "тұтынушылардың құқықтарын қорғау"), 1.20),
    (("гражданский кодекс рк", "азаматтық кодекс"), 1.10),
    (("трудовой кодекс рк", "еңбек кодексі"), 1.10),
    (("кодекс о браке и семье рк", "неке және отбасы"), 1.08),
    (("закон о доступе к информации", "ақпаратқа қол жеткізу"), 1.07),
    (("электронном документе и электронной цифровой подписи", "электрондық құжат"), 1.08),
    (("закон о масс-медиа", "масс-медиа"), 1.06),
    (("закон о государственных секретах", "мемлекеттік құпия"), 1.08),
    (("адвокатской деятельности и юридической помощи", "заң көмегі"), 1.07),
    (("об информатизации", "ақпараттандыру"), 1.06),
    (("о миграции населения", "көші-қоны"), 1.06),
    (("о языках в рк", "тілдер туралы"), 1.05),
)


def normalize_article_number(value: Any) -> str:
    text = str(value or "").strip().lower()
    if not text:
        return ""
    text = text.replace("статья", "").replace("ст.", "").replace("ст", "").replace("бап", "")
    cleaned = "".join(ch for ch in text if ch.isdigit() or ch == "-")
    return cleaned.strip("-")


def article_doc_key(doc: Document) -> tuple[str, str]:
    meta = doc.metadata or {}
    code_ru = str(meta.get("code_ru") or "").strip()
    article = normalize_article_number(meta.get("article_number"))
    path = str(meta.get("path") or "").strip()
    if code_ru and article:
        return (code_ru, article)
    if code_ru and path:
        return (code_ru, path)
    return (code_ru, str(doc.page_content or "").strip()[:80])


def dedupe_documents(docs: Sequence[Document]) -> list[Document]:
    seen: set[tuple[str, str]] = set()
    unique: list[Document] = []
    for doc in docs:
        key = article_doc_key(doc)
        if key in seen:
            continue
        seen.add(key)
        unique.append(doc)
    return unique


def normalize_scores(scores: Sequence[float]) -> list[float]:
    values = [float(score) for score in scores]
    if not values:
        return []
    min_score = min(values)
    max_score = max(values)
    if max_score <= min_score:
        return [1.0 if score > 0 else 0.0 for score in values]
    spread = max_score - min_score
    return [(score - min_score) / spread for score in values]


def apply_legal_prior_boost(
    doc: Document,
    base_score: float,
    *,
    query: str = "",
    target_codes: Iterable[str] | None = None,
    target_articles: Iterable[str] | None = None,
) -> float:
    score = float(base_score)
    meta = doc.metadata or {}
    code_ru = str(meta.get("code_ru") or "").strip()
    article_number = normalize_article_number(meta.get("article_number"))
    query_lower = str(query or "").lower()
    target_code_set = {str(code or "").strip() for code in (target_codes or []) if str(code or "").strip()}
    target_article_set = {
        normalize_article_number(article)
        for article in (target_articles or [])
        if normalize_article_number(article)
    }

    if code_ru and target_code_set and code_ru in target_code_set:
        score += 0.25
    if article_number and article_number in target_article_set:
        score += 0.35
        if article_number in query_lower:
            score += 0.10

    code_lower = code_ru.lower()
    for aliases, multiplier in _CODE_PRIOR_MULTIPLIERS:
        if any(alias in code_lower for alias in aliases):
            score *= multiplier
            break

    return score


def fuse_retrieval_candidates(
    *,
    query: str,
    bm25_docs: Sequence[Document],
    dense_docs: Sequence[tuple[Document, float]],
    bm25_weight: float = 0.55,
    dense_weight: float = 0.45,
    target_codes: Iterable[str] | None = None,
    target_articles: Iterable[str] | None = None,
    limit: int = 10,
) -> list[Document]:
    candidate_map: dict[tuple[str, str], dict[str, Any]] = {}

    for rank, doc in enumerate(bm25_docs):
        key = article_doc_key(doc)
        entry = candidate_map.setdefault(
            key,
            {
                "doc": Document(page_content=doc.page_content, metadata=dict(doc.metadata or {})),
                "bm25_raw": 0.0,
                "dense_raw": 0.0,
                "sources": [],
            },
        )
        bm25_raw = 1.0 - (rank / max(len(bm25_docs), 1))
        entry["bm25_raw"] = max(float(entry["bm25_raw"]), float(bm25_raw))
        if "bm25" not in entry["sources"]:
            entry["sources"].append("bm25")
        if bm25_raw >= entry.get("best_raw", -1.0):
            entry["doc"] = Document(page_content=doc.page_content, metadata=dict(doc.metadata or {}))
            entry["best_raw"] = bm25_raw

    for doc, raw_score in dense_docs:
        key = article_doc_key(doc)
        entry = candidate_map.setdefault(
            key,
            {
                "doc": Document(page_content=doc.page_content, metadata=dict(doc.metadata or {})),
                "bm25_raw": 0.0,
                "dense_raw": 0.0,
                "sources": [],
            },
        )
        entry["dense_raw"] = max(float(entry["dense_raw"]), float(raw_score))
        if "dense" not in entry["sources"]:
            entry["sources"].append("dense")
        if raw_score >= entry.get("best_raw", -1.0):
            entry["doc"] = Document(page_content=doc.page_content, metadata=dict(doc.metadata or {}))
            entry["best_raw"] = raw_score

    keys = list(candidate_map.keys())
    bm25_norm = normalize_scores([float(candidate_map[key]["bm25_raw"]) for key in keys])
    dense_norm = normalize_scores([float(candidate_map[key]["dense_raw"]) for key in keys])

    fused: list[tuple[float, int, Document]] = []
    for idx, key in enumerate(keys):
        entry = candidate_map[key]
        doc = entry["doc"]
        lexical = 0.0
        query_lower = (query or "").lower()
        haystack = " ".join(
            str(part)
            for part in (
                doc.metadata.get("code_ru", ""),
                doc.metadata.get("article_title", ""),
                doc.metadata.get("chapter_title", ""),
                doc.page_content[:1200],
            )
            if part
        ).lower()
        if query_lower and haystack:
            query_terms = [token for token in query_lower.split() if len(token) > 3]
            if query_terms:
                overlap = sum(1 for term in query_terms if term in haystack)
                lexical = min(0.20, overlap / max(4.0, float(len(query_terms))))

        base_score = (bm25_weight * bm25_norm[idx]) + (dense_weight * dense_norm[idx]) + lexical
        boosted = apply_legal_prior_boost(
            doc,
            base_score,
            query=query,
            target_codes=target_codes,
            target_articles=target_articles,
        )
        doc.metadata["bm25_score"] = round(bm25_norm[idx], 4)
        doc.metadata["dense_score"] = round(dense_norm[idx], 4)
        doc.metadata["fusion_base_score"] = round(base_score, 4)
        doc.metadata["relevance_score"] = round(boosted, 4)
        doc.metadata["fusion_sources"] = list(entry["sources"])
        fused.append((boosted, idx, doc))

    fused.sort(key=lambda item: (-item[0], item[1]))
    ordered = [doc for _, _, doc in fused]
    return dedupe_documents(ordered)[:limit]
