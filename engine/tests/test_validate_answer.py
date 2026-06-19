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


def test_validate_answer_does_not_require_illegal_business_or_pyramid_articles():
    sources = [
        Document(
            page_content="Статья 190. Мошенничество...",
            metadata={"code_ru": "Уголовный кодекс РК", "article_number": "190"},
        )
    ]
    # An answer about illegal business / a financial pyramid that cites a relevant article
    # other than the old whitelist (214/245/217) must not be discarded.
    response = "Это может образовывать состав мошенничества по статье 190 УК РК."

    assert validate_answer("незаконное предпринимательство", response, sources) == response
    assert validate_answer("финансовая пирамида обман инвесторов", response, sources) == response
