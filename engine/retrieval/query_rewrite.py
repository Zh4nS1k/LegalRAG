from typing import Any, Callable, Iterable


def _stringify_llm_result(result: Any) -> str:
    if result is None:
        return ""
    if isinstance(result, str):
        return result.strip()
    content = getattr(result, "content", None)
    if isinstance(content, str):
        return content.strip()
    return str(result).strip()


def rewrite_query(
    query: str,
    *,
    llm: Any | None = None,
    detect_target_codes: Callable[[str], Iterable[str]] | None = None,
    extract_query_article_number: Callable[[str], str | None] | None = None,
    focus_articles_from_query: Callable[[str], Iterable[str]] | None = None,
    expand_legal_synonyms: Callable[[str], Iterable[str]] | None = None,
) -> str:
    cleaned_query = (query or "").strip()
    if not cleaned_query:
        return ""

    if llm is not None:
        prompt = f"""
Преобразуй запрос в юридический поисковый формат.
Добавь:
- отрасль права
- юридические термины
- формулировки из законов

Вопрос: {cleaned_query}
"""
        llm_result = _stringify_llm_result(llm.invoke(prompt))
        if llm_result:
            return llm_result

    target_codes = list(detect_target_codes(cleaned_query)) if detect_target_codes else []
    article_number = (
        extract_query_article_number(cleaned_query)
        if extract_query_article_number
        else None
    )
    focus_articles = (
        sorted(set(focus_articles_from_query(cleaned_query)))
        if focus_articles_from_query
        else []
    )
    synonym_tail = (
        list(dict.fromkeys(expand_legal_synonyms(cleaned_query)))
        if expand_legal_synonyms
        else []
    )

    parts = [cleaned_query]
    if target_codes:
        parts.append(" ".join(dict.fromkeys(target_codes)))
    if article_number:
        parts.append(f"статья {article_number}")
    if focus_articles:
        parts.append(" ".join(f"статья {num}" for num in focus_articles[:4]))
    if synonym_tail:
        parts.append(" ".join(synonym_tail[:6]))

    return " ".join(part for part in parts if part).strip()
