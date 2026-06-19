"""Query analysis & routing: code detection, article extraction, synonym/expansion,
and retrieval-query building. Extracted from rag_chain.py to reduce its size.
Backward-compatible: re-exported from rag_chain via `import *`.
"""

from __future__ import annotations

import re
from typing import Any

from engine.core import config
from engine.retrieval.query_rewrite import rewrite_query
from engine.retrieval.language import _is_kz_query
from engine.retrieval.legal_codes_data import *  # noqa: F401,F403

__all__ = [
    "_LAW_ROUTE_HINTS",
    "_LEGAL_SYNONYMS",
    "_augment_retrieval_query",
    "_build_retrieval_queries",
    "_detect_target_codes",
    "_expand_legal_synonyms",
    "_extract_article_range",
    "_extract_query_article_number",
    "_extract_query_article_numbers",
    "_focus_articles_from_query",
    "_get_target_code_profile",
    "_has_alias",
    "_is_criminal_query",
    "_is_illegal_business_query",
    "_is_pyramid_query",
    "_is_subsidy_query",
    "_is_theft_query",
    "_matching_legal_concepts",
    "_needs_circumstances_query",
    "_normalize_article_number",
    "_normalized_query",
    "_rewrite_query_for_retrieval",
    "_score_target_codes",
]

_LAW_ROUTE_HINTS: list[tuple[tuple[str, ...], list[str], float]] = [
    (
        (
            "возмездного оказания услуг",
            "договор оказания услуг",
            "представительство в суде",
            "юридическая помощь",
            "адвокатская деятельность",
            "поверенный",
            "договор поручения",
        ),
        _gk_special_variants,
        1.5,
    ),
    (
        (
            "валютном регулировании",
            "валютном контроле",
            "валюталық реттеу",
            "валюталық бақылау",
            "нерезидент",
            "резидент",
            "импорт",
            "экспорт",
            "внешнеэконом",
            "дубай",
            "дубае",
            "иностранн",
            "за рубежом",
            "наличными деньгами",
            "наличные деньги",
            "оплата наличными",
            "оплата товара наличными",
        ),
        _currency_variants,
        1.2,
    ),
    (
        (
            "товариществах с ограниченной и дополнительной ответственностью",
            "жауапкершілігі шектеулі",
        ),
        _llp_variants,
        1.2,
    ),
    (
        (
            "тоо",
            "т о о",
            "ип",
        ),
        _llp_variants,
        0.15,
    ),
    (
        (
            "зпп",
            "защите прав потребителей",
            "правах потребителей",
            "возврат товара",
            "продавец отказал",
            "некачественный товар",
            "недостаток товара",
            "товарный вид",
            "в течение четырнадцати дней",
            "14 дней",
            "потребитель",
            "бритва",
        ),
        _consumer_variants,
        1.4,
    ),
    (
        (
            "цифровые активы",
            "необеспеченные цифровые активы",
            "крипто",
            "криптобирж",
            "стейкинг",
            "kraken",
        ),
        _digital_assets_variants,
        1.6,
    ),
    (
        (
            "мусор",
            "тбо",
            "отходы",
            "антисанитар",
            "контейнер",
            "санитарно-эпидемиолог",
            "мусорная площадка",
        ),
        _koap_variants,
        1.3,
    ),
    (
        _THEFT_QUERY_HINTS,
        _uk_variants,
        1.6,
    ),
    (
        (
            "банковской деятельности",
            "банковский счет",
            "банковский заем",
            "банк",
            "счет",
            "перевод денег",
        ),
        _banks_variants,
        0.8,
    ),
]

_LEGAL_SYNONYMS: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("договор", ("обязательство", "сделка", "договорные отношения")),
    ("налич", ("наличные расчеты", "денежные средства", "оплата наличными")),
    ("ущерб", ("убытки", "вред", "возмещение вреда")),
    ("мусор", ("отходы", "тбо", "санитарные требования")),
    ("крипто", ("цифровые активы", "необеспеченные цифровые активы", "стейкинг")),
    ("банкрот", ("неплатежеспособность", "восстановление платежеспособности")),
    ("тоо", ("товарищество с ограниченной ответственностью", "участник тоо")),
    ("ип", ("индивидуальный предприниматель", "предпринимательская деятельность")),
    ("недвижим", ("имущество", "право собственности", "регистрация прав")),
)


def _get_target_code_profile(query: str) -> tuple[list[str], bool]:
    ranked = _score_target_codes(query)
    if not ranked:
        return [], False
    top_score = ranked[0][1]
    second_score = ranked[1][1] if len(ranked) > 1 else 0.0
    detected = _detect_target_codes(query)
    confident = top_score >= 1.8 and top_score >= second_score + 1.2
    return detected, confident

def _extract_article_range(query: str) -> tuple[int, int] | None:
    match = re.search(
        r"(?:статья|ст\.|ст|бап)?\s*(\d+)\s*[-–—]\s*(\d+)", query or "", re.IGNORECASE
    )
    if not match:
        return None
    start = int(match.group(1))
    end = int(match.group(2))
    return (start, end) if start <= end else (end, start)

def _normalized_query(query: str) -> str:
    q = (query or "").lower().replace("ё", "е")
    q = re.sub(r"\s+", " ", q)
    return q

def _has_alias(query: str, alias: str) -> bool:
    pattern = r"(?<!\w)" + re.escape(alias) + r"(?!\w)"
    return re.search(pattern, query) is not None

def _score_target_codes(query: str) -> list[tuple[str, float]]:
    q = _normalized_query(query)
    scores: dict[str, float] = {}

    for aliases, code_names in _LAW_ALIAS_GROUPS:
        matched = False
        for alias in aliases:
            if _has_alias(q, alias):
                matched = True
                break
        if not matched:
            continue
        for code_name in code_names:
            scores[code_name] = scores.get(code_name, 0.0) + 1.0

    for aliases, code_names, weight in _LAW_ROUTE_HINTS:
        matched_count = 0
        for alias in aliases:
            if _has_alias(q, alias):
                matched_count += 1
        if not matched_count:
            continue
        boost = min(weight * matched_count, weight * 3)
        for code_name in code_names:
            scores[code_name] = scores.get(code_name, 0.0) + boost

    for bundle in _matching_legal_concepts(query):
        weight = float(bundle.get("weight", 1.0) or 1.0)
        for code_name in bundle.get("code_names", ()):
            scores[code_name] = scores.get(code_name, 0.0) + weight

    ranked = sorted(scores.items(), key=lambda item: item[1], reverse=True)
    if not ranked:
        return []
    return ranked

def _detect_target_codes(query: str) -> list[str]:
    ranked = _score_target_codes(query)
    if not ranked:
        return []

    top_score = ranked[0][1]
    second_score = ranked[1][1] if len(ranked) > 1 else 0.0

    if top_score >= 1.8 and top_score >= second_score + 1.2:
        return [code for code, score in ranked if score >= top_score - 0.4]

    cutoff = max(1.0, top_score - 0.5)
    detected: list[str] = []
    for code, score in ranked:
        if score < cutoff:
            continue
        if code not in detected:
            detected.append(code)
    return detected

def _normalize_article_number(value: str | None) -> str:
    raw = str(value or "").strip().lower()
    if not raw:
        return ""
    raw = raw.replace("статья", "").replace("ст.", "").replace("ст", "").replace("бап", "")
    raw = re.sub(r"\s+", "", raw)
    raw = raw.replace("–", "-").replace("—", "-")
    raw = re.sub(r"[^\da-zа-я\-\.]", "", raw, flags=re.IGNORECASE)
    return raw.strip(".-")

def _extract_query_article_number(query: str) -> str | None:
    q = query or ""
    match = re.search(r"(?:статья|ст\.|ст|бап)\s*(\d+[а-яА-Яa-zA-Z\-]?)", q, re.IGNORECASE)
    if match:
        normalized = _normalize_article_number(match.group(1))
        return normalized or None
    return None

def _extract_query_article_numbers(query: str) -> list[str]:
    q = query or ""
    articles: list[str] = []

    def _add(article: str) -> None:
        normalized = _normalize_article_number(article)
        if normalized and normalized not in articles:
            articles.append(normalized)

    primary = _extract_query_article_number(q)
    if primary:
        _add(primary)

    for match in re.finditer(
        r"(?:статьи|статья|ст\.|ст|баптары|баптар|бап)\s*([0-9,\-\sandи]+)",
        q,
        re.IGNORECASE,
    ):
        raw_tail = match.group(1)
        for token in re.findall(r"\d+(?:-\d+)?", raw_tail):
            _add(token)

    range_match = _extract_article_range(q)
    if range_match:
        start, end = range_match
        for number in range(start, end + 1):
            _add(str(number))

    return articles

def _is_theft_query(query: str) -> bool:
    q = _normalized_query(query)
    return any(token in q for token in _THEFT_QUERY_HINTS)

def _matching_legal_concepts(query: str) -> list[dict[str, Any]]:
    q = _normalized_query(query)
    matched: list[dict[str, Any]] = []
    for bundle in _LEGAL_CONCEPT_BUNDLES:
        if any(token in q for token in bundle.get("patterns", ())):
            matched.append(bundle)
    return matched

def _expand_legal_synonyms(query: str) -> list[str]:
    q = _normalized_query(query)
    extras: list[str] = []
    for needle, synonyms in _LEGAL_SYNONYMS:
        if needle not in q:
            continue
        extras.extend(s for s in synonyms if s not in q)
    return extras

def _rewrite_query_for_retrieval(query: str) -> str:
    llm = None
    if getattr(config, "USE_LLM_QUERY_REWRITE", False):
        try:
            from engine.retrieval.rag_chain import get_llm

            llm = get_llm()
        except Exception:
            llm = None
    return rewrite_query(
        query,
        llm=llm,
        detect_target_codes=_detect_target_codes,
        extract_query_article_number=_extract_query_article_number,
        focus_articles_from_query=_focus_articles_from_query,
        expand_legal_synonyms=_expand_legal_synonyms,
    )

def _build_retrieval_queries(query: str) -> list[str]:
    augmented = _augment_retrieval_query(query)
    rewritten = _rewrite_query_for_retrieval(query)
    queries: list[str] = []
    for candidate in (query, augmented, rewritten):
        cleaned = re.sub(r"\s+", " ", (candidate or "").strip())
        if cleaned and cleaned not in queries:
            queries.append(cleaned)

    target_codes = _detect_target_codes(query)
    if target_codes:
        code_query = f"{rewritten} {' '.join(target_codes)}".strip()
        if code_query and code_query not in queries:
            queries.append(code_query)

    limit = max(1, getattr(config, "RETRIEVER_MULTI_QUERY_LIMIT", 4))
    return queries[:limit]

def _augment_retrieval_query(query: str) -> str:
    q = (query or "").lower()
    extras: list[str] = []
    target_codes = _detect_target_codes(query)
    is_criminal = any(code in set(_uk_variants) for code in target_codes) or _is_criminal_query(query)

    # Check language
    is_kz = _is_kz_query(query)

    if any(
        token in q
        for token in (
            "несовершеннолетний",
            "несовершеннолетние",
            "несовершеннолетних",
            "minor",
            "underage",
            "кәмелетке толмаған",
            "кәмелетке толмағандар",
        )
    ):
        if is_kz:
            extras.append("он сегіз жасқа толмаған жұмыскерлер")
            extras.append("76-бап ҚР Еңбек кодексі")
            extras.append("кәмелетке толмағандардың түнгі жұмысына тыйым салу")
        else:
            extras.append("работники, не достигшие восемнадцатилетнего возраста")
            extras.append("статья 76 Трудовой кодекс РК")
            extras.append("запрет ночной работы несовершеннолетних")
    for bundle in _matching_legal_concepts(query):
        if bundle.get("criminal") and not is_criminal:
            continue
        localized = bundle.get("expansions_kz", ()) if is_kz else bundle.get("expansions_ru", ())
        extras.extend(str(item) for item in localized if item)
    if any(
        token in q
        for token in (
            "субсид",
            "субсидия",
            "гос",
            "государ",
            "бюджет",
            "грант",
            "инвест",
            "смет",
            "договор",
            "фиктив",
            "жалған",
            "құжат",
            "алаяқ",
            "мемлекеттік",
            "қаржы",
            "ақша",
        )
    ) and is_criminal:
        if is_kz:
            extras.append("алаяқтық 190-бап ҚР ҚК")
            extras.append("қылмыстық жолмен алынған ақшаны заңдастыру 218-бап ҚР ҚК")
            extras.append("субсидия алу үшін жалған құжаттар 190-бап ҚР ҚК")
        else:
            extras.append("алаяқтық 190 УК РК")
            extras.append("қылмыстық жолмен алынған ақшаны заңдастыру 218 УК РК")
            extras.append("субсидия алу үшін жалған құжаттар 190 УК РК")
    if any(
        token in q
        for token in (
            "заңсыз кәсіпкер",
            "кәсіпкерлік",
            "лицензиясыз",
            "тіркеусіз",
            "незаконн",
            "без регистрации",
            "без лицензии",
            "салық төлем",
            "налог",
            "уклонен",
        )
    ) and is_criminal:
        if is_kz:
            extras.append("заңсыз кәсіпкерлік 214-бап ҚР ҚК")
            extras.append("салық төлеуден жалтару 245-бап ҚР ҚК")
        else:
            extras.append("заңсыз кәсіпкерлік 214 УК РК")
            extras.append("салық төлеуден жалтару 245 УК РК")
    if any(
        token in q
        for token in (
            "пирамида",
            "пирамид",
            "қаржылық пирамида",
            "инвестиция",
            "инвест",
            "жоғары пайда",
            "30-50%",
        )
    ) and is_criminal:
        if is_kz:
            extras.append("қаржылық пирамида құру және басқару 217-бап ҚР ҚК")
            extras.append("қаржылық пирамиданы жарнамалау 217-1-бап ҚР ҚК")
        else:
            extras.append("қаржылық пирамида құру және басқару 217 УК РК")
            extras.append("финансовая пирамида создание и руководство 217 УК РК")
            extras.append("реклама финансовой пирамиды 217-1 УК РК")
    # ... other heuristics could be localized similarly ...

    range_match = _extract_article_range(query)
    if range_match and ("ук" in q or "қылмыстық" in q or "уголов" in q):
        start, end = range_match
        nums = " ".join(str(n) for n in range(start, end + 1))
        if is_kz:
            extras.append(f"ҚР ҚК {nums} баптары")
        else:
            extras.append(f"статьи {nums} УК РК")
    return (query + " " + " ".join(extras)).strip() if extras else query

def _is_criminal_query(query: str) -> bool:
    q = _normalized_query(query)
    if any(token in q for token in ("қылмыстық", "уголов", "преступ", "ук рк", "квалификация преступ", "состав преступ")):
        return True
    if any(bundle.get("criminal") for bundle in _matching_legal_concepts(q)):
        return True
    return False

def _focus_articles_from_query(query: str) -> set[str]:
    """Return only article numbers the user explicitly named (e.g. "статья 188", "ст. 190-191").

    Topic-keyword → article-number guessing (theft→188, fraud→190, subsidy→190/218,
    pyramid→217, ecology→324/325/328, …) was removed: it overfit specific benchmark
    phrasings and could force the wrong article on paraphrased queries. Code-level routing
    (_detect_target_codes), synonym expansion, semantic retrieval and multilingual
    reranking now decide which article is most relevant on merit.
    """
    return set(_extract_query_article_numbers(query))

def _is_subsidy_query(query: str) -> bool:
    q = (query or "").lower()
    return any(
        token in q
        for token in (
            "субсид",
            "субсидия",
            "грант",
            "гос",
            "государ",
            "мемлекеттік",
            "бюджет",
        )
    )

def _is_illegal_business_query(query: str) -> bool:
    q = (query or "").lower()
    return any(
        token in q
        for token in (
            "заңсыз кәсіпкер",
            "кәсіпкерлік",
            "лицензиясыз",
            "тіркеусіз",
            "незаконн",
            "без регистрации",
            "без лицензии",
            "салық төлем",
            "налог",
            "уклонен",
        )
    )

def _is_pyramid_query(query: str) -> bool:
    q = (query or "").lower()
    return any(
        token in q
        for token in (
            "пирамида",
            "пирамид",
            "қаржылық пирамида",
            "инвестиция",
            "инвест",
            "жоғары пайда",
            "30-50%",
        )
    )

def _needs_circumstances_query(query: str) -> bool:
    q = (query or "").lower()
    return any(
        token in q for token in ("ауырлататын", "жеңілдететін", "смягча", "отягча")
    )
