from langchain_core.documents import Document

from ai_service.processing.prepare_data import _build_indexable_text
from ai_service.retrieval.domain import detect_domain
from ai_service.retrieval.rag_chain import (
    _DedupRetriever,
    _apply_diversity_penalty,
    _apply_legal_score,
    _build_retrieval_queries,
    _normalize_article_number,
    _rewrite_query_for_retrieval,
)


def test_build_indexable_text_includes_legal_structure():
    meta = {
        "code_ru": "Гражданский кодекс РК (Особенная часть)",
        "chapter_number": "33",
        "chapter_title": "Возмездное оказание услуг",
        "article_number": "683",
        "article_title": "Договор возмездного оказания услуг",
        "clause_level": "clause",
        "clause_number": "1",
    }

    result = _build_indexable_text("1. Исполнитель обязан оказать услугу.", meta)

    assert "Кодекс: Гражданский кодекс РК (Особенная часть)" in result
    assert "Глава: 33 Возмездное оказание услуг" in result
    assert "Статья: 683. Договор возмездного оказания услуг" in result
    assert "Пункт: 1" in result
    assert "Текст: 1. Исполнитель обязан оказать услугу." in result


def test_rewrite_query_for_retrieval_adds_legal_terms():
    query = "можно ли платить наличными в дубае за товар для тоо"

    rewritten = _rewrite_query_for_retrieval(query)

    assert "Закон о валютном регулировании и валютном контроле" in rewritten
    assert "наличные расчеты" in rewritten
    assert "товарищество с ограниченной ответственностью" in rewritten


def test_build_retrieval_queries_is_unique_and_limited():
    queries = _build_retrieval_queries("статья 272 договорные обязательства")

    assert 1 <= len(queries) <= 4
    assert len(queries) == len(set(queries))
    assert any("статья 272" in q for q in queries)


def test_normalize_article_number_strips_noise():
    assert _normalize_article_number("ст. 42-4") == "42-4"
    assert _normalize_article_number(" 42. ") == "42"
    assert _normalize_article_number("бап 190") == "190"


def test_detect_domain_picks_expected_code_family():
    assert detect_domain("какая ответственность за административный штраф") == "koap"
    assert detect_domain("договор возмездного оказания услуг") == "gk"


def test_dedup_retriever_collapses_same_code_and_article():
    class _StaticRetriever:
        def invoke(self, query):
            return [
                Document(
                    page_content="A",
                    metadata={"code_ru": "Уголовный кодекс РК", "article_number": "190", "path": "ст. 190 п. 1"},
                ),
                Document(
                    page_content="B",
                    metadata={"code_ru": "Уголовный кодекс РК", "article_number": "190", "path": "ст. 190 п. 2"},
                ),
                Document(
                    page_content="C",
                    metadata={"code_ru": "Уголовный кодекс РК", "article_number": "191", "path": "ст. 191"},
                ),
            ]

    docs = _DedupRetriever(base_retriever=_StaticRetriever()).invoke("мошенничество")

    assert len(docs) == 2
    assert [doc.metadata["article_number"] for doc in docs] == ["190", "191"]


def test_apply_legal_score_boosts_matching_article_and_penalizes_wrong_domain():
    query = "статья 190 уголовное мошенничество"
    matched = Document(
        page_content="",
        metadata={"code_ru": "Уголовный кодекс РК", "article_number": "190"},
    )
    wrong_domain = Document(
        page_content="",
        metadata={"code_ru": "Гражданский кодекс РК (Особенная часть)", "article_number": "190"},
    )

    matched_score = _apply_legal_score(query, matched, 0.8, target_codes=["Уголовный кодекс РК"], target_articles={"190"})
    wrong_score = _apply_legal_score(query, wrong_domain, 0.8, target_codes=["Уголовный кодекс РК"], target_articles={"190"})

    assert matched_score > wrong_score


def test_apply_diversity_penalty_pushes_down_repeated_code():
    docs = [
        (
            Document(page_content="", metadata={"code_ru": "Уголовный кодекс РК", "article_number": "190"}),
            1.0,
        ),
        (
            Document(page_content="", metadata={"code_ru": "Уголовный кодекс РК", "article_number": "191"}),
            0.95,
        ),
        (
            Document(page_content="", metadata={"code_ru": "Гражданский кодекс РК", "article_number": "272"}),
            0.92,
        ),
    ]

    reranked = _apply_diversity_penalty(docs, penalty_step=0.1)

    assert reranked[0][0].metadata["article_number"] == "190"
    assert reranked[1][0].metadata["article_number"] == "272"
