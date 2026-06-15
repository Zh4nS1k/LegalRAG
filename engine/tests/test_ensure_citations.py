from langchain_core.documents import Document

from engine.utils.ensure_citations import (
    ensure_answer_citations,
    response_cites_context,
)


def _doc(article: str, code: str) -> Document:
    return Document(
        page_content=f"Статья {article}. Текст нормы.",
        metadata={"article_number": article, "code_ru": code, "source": "test.txt"},
    )


def test_response_cites_context_detects_article_mention():
    sources = [_doc("218", "Закон о государственных закупках РК")]
    response = "По смыслу статьи 218 закон допускает участие поставщика."

    assert response_cites_context(response, sources) is True


def test_ensure_answer_citations_appends_sources_when_missing():
    sources = [_doc("10", "Закон о государственных закупках РК")]
    response = "Для субсидий нужно проверить условия предоставления."

    enriched = ensure_answer_citations(response, sources, question="субсидии")

    assert "Источники:" in enriched
    assert "ст. 10" in enriched
    assert "Закон о государственных закупках РК" in enriched


def test_ensure_answer_citations_skips_social_intent():
    sources = [_doc("1", "Трудовой кодекс РК")]
    response = "Здравствуйте! Чем могу помочь?"

    assert ensure_answer_citations(response, sources, intent="SOCIAL") == response


def test_ensure_answer_citations_tags_internal_knowledge_without_sources():
    response = "По общему правилу ответственность наступает по ст. 917 ГК РК."

    enriched = ensure_answer_citations(response, [], question="ответственность")

    assert enriched.startswith("[INTERNAL_ADVISORY]")
