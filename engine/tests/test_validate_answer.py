from langchain_core.documents import Document

from engine.retrieval.rag_chain import validate_answer


def test_validate_answer_allows_synthesized_article_reference():
    sources = [
        Document(
            page_content="Статья 218. Государственные закупки...",
            metadata={"code_ru": "Закон о государственных закупках РК", "article_number": "218"},
        )
    ]
    response = "По смыслу статьи 218 закон допускает участие поставщика в закупках."

    assert validate_answer("госзакупки", response, sources) == response


def test_validate_answer_does_not_require_subsidy_article_whitelist():
    sources = [
        Document(
            page_content="Статья 10. Общие положения...",
            metadata={"code_ru": "Закон о государственных закупках РК", "article_number": "10"},
        )
    ]
    response = "Для субсидий нужно проверить условия предоставления, но конкретная статья в ответе не указана."

    assert validate_answer("субсидии", response, sources) == response
