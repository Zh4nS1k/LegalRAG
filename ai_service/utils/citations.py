from __future__ import annotations

import re
from typing import Any, Iterable

import pandas as pd

from ai_service.core.code_registry import get_code_name


def normalize_gold(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, float) and pd.isna(value):
        return []
    text = str(value).strip()
    if not text:
        return []
    parts = [p.strip() for p in text.split(";") if p.strip()]
    return parts if parts else [text]


def normalize_article(value: str) -> str:
    text = str(value or "").strip().lower().replace("ё", "е")
    if not text:
        return ""
    match = re.search(r"(?:ст\.?|статья|бап)\s*(\d{1,4}(?:-\d+)?)", text)
    if match:
        return match.group(1)
    match = re.search(r"(\d{1,4}(?:-\d+)?)", text)
    return match.group(1) if match else ""


def normalize_code(value: str) -> str:
    text = str(value or "").strip().lower().replace("ё", "е")
    if not text:
        return ""
    text = text.replace("«", '"').replace("»", '"')
    text = re.sub(r'\bреспублики казахстан\b', "рк", text)
    text = re.sub(r"[\"']", "", text)
    text = re.sub(r"\s+", " ", text)

    alias_groups = [
        (
            "гражданский кодекс рк",
            (
                "гк рк",
                "гк",
                "гражданский кодекс",
                "гражданский кодекс рк (общая часть)",
                "гражданский кодекс рк (особенная часть)",
            ),
        ),
        ("уголовный кодекс рк", ("ук рк", "ук", "уголовный кодекс")),
        (
            "гражданский процессуальный кодекс рк",
            ("гпк рк", "гпк", "гражданский процессуальный кодекс"),
        ),
        (
            "уголовно-процессуальный кодекс рк",
            ("упк рк", "упк", "уголовно-процессуальный кодекс"),
        ),
        (
            "кодекс об административных правонарушениях рк",
            ("коап рк", "коап", "кодекс об административных правонарушениях"),
        ),
        (
            "кодекс об административных процедурах рк",
            ("аппк рк", "аппк", "кодекс об административных процедурах"),
        ),
        (
            "трудовой кодекс рк",
            ("тк рк", "тк", "трудовой кодекс"),
        ),
        (
            "налоговый кодекс рк",
            ("нк рк", "нк", "налоговый кодекс"),
        ),
        (
            "закон о защите прав потребителей рк",
            ("закон о защите прав потребителей", "о защите прав потребителей", "зпп"),
        ),
        (
            "закон об адвокатской деятельности и юридической помощи",
            ("об адвокатской деятельности и юридической помощи",),
        ),
        (
            "закон о валютном регулировании и валютном контроле",
            ("о валютном регулировании и валютном контроле",),
        ),
    ]
    for canonical, aliases in alias_groups:
        if canonical in text or any(alias in text for alias in aliases):
            return canonical

    return text.strip(" ,.")


def gold_to_pair(value: str) -> tuple[str, str]:
    raw = str(value or "").strip()
    if not raw:
        return "", ""
    article = normalize_article(raw)
    lower_raw = raw.lower()
    code = ""
    if "закон" in lower_raw:
        code = raw[lower_raw.find("закон") :]
    elif any(
        token in lower_raw
        for token in ("кодекс", "гк", "ук", "гпк", "упк", "коап", "аппк", "тк", "нк")
    ):
        code = raw
    return article, normalize_code(code)


def pair_to_str(article: str, code: str) -> str:
    return f"{normalize_article(article)}::{normalize_code(code)}"


def canonicalize_doc_code(meta: dict[str, Any]) -> str:
    raw_code = str(meta.get("code_ru", "") or "")
    normalized = normalize_code(raw_code)
    if normalized and "_" not in normalized:
        return normalized

    source = str(meta.get("source", "") or "").strip()
    if source:
        code_ru, _ = get_code_name(source)
        return normalize_code(code_ru)
    return normalized


def compute_pair_metrics(
    gold_pairs: list[tuple[str, str]], pred_pairs: list[tuple[str, str]]
) -> dict[str, float]:
    gold_set = {pair for pair in gold_pairs if pair[0]}
    pred_set = {pair for pair in pred_pairs if pair[0]}
    gold_articles = {article for article, _ in gold_set}
    pred_articles = {article for article, _ in pred_set}

    strict_hit = 1.0 if gold_set & pred_set else 0.0
    soft_hit = 1.0 if gold_articles & pred_articles else 0.0
    strict_precision = len(gold_set & pred_set) / len(pred_set) if pred_set else 0.0
    strict_recall = len(gold_set & pred_set) / len(gold_set) if gold_set else 0.0
    soft_precision = (
        len(gold_articles & pred_articles) / len(pred_articles) if pred_articles else 0.0
    )
    soft_recall = (
        len(gold_articles & pred_articles) / len(gold_articles) if gold_articles else 0.0
    )

    strict_mrr = 0.0
    soft_mrr = 0.0
    for rank, pair in enumerate(pred_pairs, start=1):
        if not strict_mrr and pair in gold_set:
            strict_mrr = 1.0 / rank
        if not soft_mrr and pair[0] in gold_articles:
            soft_mrr = 1.0 / rank
        if strict_mrr and soft_mrr:
            break

    return {
        "strict_hit": strict_hit,
        "soft_hit": soft_hit,
        "strict_precision": strict_precision,
        "strict_recall": strict_recall,
        "soft_precision": soft_precision,
        "soft_recall": soft_recall,
        "strict_mrr": strict_mrr,
        "soft_mrr": soft_mrr,
    }


def extract_citation_pairs_from_text(text: str) -> list[tuple[str, str]]:
    if not text:
        return []
    compact = re.sub(r"\s+", " ", str(text)).strip()
    patterns = [
        r"(ст\.?|статья|бап)\s*(\d{1,4}(?:-\d+)?)\s+([^.;:\n]{0,140}?(?:кодекс|закон)[^.;:\n]{0,140})",
        r"(ст\.?|статья|бап)\s*(\d{1,4}(?:-\d+)?)\s+((?:гк|ук|гпк|упк|коап|аппк|тк|нк)\s*рк?)",
    ]
    pairs: list[tuple[str, str]] = []
    for pattern in patterns:
        for match in re.finditer(pattern, compact, re.IGNORECASE):
            article = normalize_article(match.group(2))
            code = normalize_code(match.group(3))
            if article and code and (article, code) not in pairs:
                pairs.append((article, code))
    return pairs


def extract_pairs_from_docs(docs: Iterable[Any]) -> list[tuple[str, str]]:
    pairs: list[tuple[str, str]] = []
    for doc in docs or []:
        meta = getattr(doc, "metadata", {}) or {}
        article = normalize_article(meta.get("article_number", ""))
        code = canonicalize_doc_code(meta)
        if article and code and (article, code) not in pairs:
            pairs.append((article, code))
    return pairs
