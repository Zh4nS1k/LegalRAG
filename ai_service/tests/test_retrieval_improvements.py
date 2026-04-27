from langchain_core.documents import Document

from ai_service.processing.prepare_data import _build_indexable_text
from ai_service.retrieval.domain import detect_domain
from ai_service.retrieval.rag_chain import (
    _DedupRetriever,
    _apply_diversity_penalty,
    _apply_legal_score,
    _build_retrieval_queries,
    _detect_target_codes,
    _focus_articles_from_query,
    _is_criminal_query,
    _is_noisy_legal_chunk,
    _lexical_overlap_score,
    _looks_like_raw_code_name,
    _normalize_article_number,
    _rank_docs_with_legal_scoring,
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


def test_theft_query_is_treated_as_criminal_and_augmented_for_uk():
    query = "что будет если я украду яблоко из соседского дома"

    queries = _build_retrieval_queries(query)

    assert _is_criminal_query(query)
    assert "188" in _focus_articles_from_query(query)
    assert any("188 УК РК" in candidate or "кража 188" in candidate for candidate in queries)


def test_fraud_query_routes_to_uk_and_focuses_article_190():
    query = "Меня обманом лишили денег, это мошенничество?"

    queries = _build_retrieval_queries(query)
    target_codes = _detect_target_codes(query)

    assert _is_criminal_query(query)
    assert "190" in _focus_articles_from_query(query)
    assert any("190" in candidate and ("мошеннич" in candidate.lower() or "алаяқ" in candidate.lower()) for candidate in queries)
    assert any("Уголовный кодекс" in code or "Қылмыстық кодекс" in code for code in target_codes)


def test_consumer_return_query_routes_to_consumer_law():
    query = "Продавец отказал в возврате некачественного товара, что делать?"

    target_codes = _detect_target_codes(query)
    queries = _build_retrieval_queries(query)

    assert any("защите прав потребителей" in code.lower() for code in target_codes)
    assert any("защита прав потребителей" in candidate.lower() for candidate in queries)


def test_labor_query_routes_to_labor_code():
    query = "Работодатель хочет уволить меня по сокращению штата, какие у меня права?"

    target_codes = _detect_target_codes(query)
    queries = _build_retrieval_queries(query)

    assert any("трудовой кодекс" in code.lower() or "еңбек кодексі" in code.lower() for code in target_codes)
    assert any("трудовой кодекс" in candidate.lower() or "заработная плата" in candidate.lower() or "сокращение" in candidate.lower() for candidate in queries)


def test_family_query_routes_to_family_code():
    query = "После развода как взыскать алименты на ребенка?"

    target_codes = _detect_target_codes(query)
    queries = _build_retrieval_queries(query)

    assert any("браке" in code.lower() or "отбасы" in code.lower() for code in target_codes)
    assert any("алименты" in candidate.lower() or "развод" in candidate.lower() for candidate in queries)


def test_noise_query_routes_to_koap_and_focuses_article_437():
    query = "Можно ли шуметь в субботу в 22:30 в квартире?"

    target_codes = _detect_target_codes(query)
    queries = _build_retrieval_queries(query)

    assert any(
        "административных правонарушениях" in code.lower() or "әкімшілік құқық бұзушылық" in code.lower()
        for code in target_codes
    )
    assert "437" in _focus_articles_from_query(query)
    assert any("437" in candidate or "нарушение тишины" in candidate.lower() for candidate in queries)


def test_bankruptcy_query_routes_to_bankruptcy_laws():
    query = "Я оформила банкротство из-за долгов и хочу понять свои права"

    target_codes = _detect_target_codes(query)
    queries = _build_retrieval_queries(query)

    assert any("банкрот" in code.lower() or "төлем қабілетті" in code.lower() for code in target_codes)
    assert any("банкрот" in candidate.lower() or "платежеспособ" in candidate.lower() for candidate in queries)


def test_normalize_article_number_strips_noise():
    assert _normalize_article_number("ст. 42-4") == "42-4"
    assert _normalize_article_number(" 42. ") == "42"
    assert _normalize_article_number("бап 190") == "190"


def test_looks_like_raw_code_name_detects_slug_metadata():
    assert _looks_like_raw_code_name("law_on_military_service_kz")
    assert not _looks_like_raw_code_name("Закон о воинской службе и статусе военнослужащих РК")


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


def test_is_noisy_legal_chunk_flags_preamble_style_chunk():
    noisy = Document(
        page_content="ЗҚАИ-ның ескертпесі! МАЗМҰНЫ Қолданушылар назарына!",
        metadata={
            "code_ru": "law_on_military_service_kz",
            "article_number": "",
            "clause_level": "article",
            "article_title": "",
        },
    )
    clean = Document(
        page_content="Статья 190. Мошенничество",
        metadata={
            "code_ru": "Уголовный кодекс РК",
            "article_number": "190",
            "clause_level": "article",
            "article_title": "Мошенничество",
        },
    )

    assert _is_noisy_legal_chunk(noisy)
    assert not _is_noisy_legal_chunk(clean)


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


def test_lexical_overlap_score_rewards_shared_terms():
    doc = Document(
        page_content="Договор возмездного оказания услуг регулируется гражданским законодательством.",
        metadata={
            "code_ru": "Гражданский кодекс РК (Особенная часть)",
            "article_title": "Договор возмездного оказания услуг",
        },
    )

    score = _lexical_overlap_score("договор возмездного оказания услуг", doc)

    assert score > 0


def test_rank_docs_with_legal_scoring_prioritizes_code_and_article_match():
    docs = [
        Document(
            page_content="Общие положения об обязательствах",
            metadata={
                "code_ru": "Гражданский кодекс РК (Особенная часть)",
                "article_number": "272",
                "article_title": "Надлежащее исполнение обязательства",
            },
        ),
        Document(
            page_content="Административная ответственность",
            metadata={
                "code_ru": "Кодекс об административных правонарушениях РК",
                "article_number": "272",
                "article_title": "Иная статья",
            },
        ),
    ]

    ranked = _rank_docs_with_legal_scoring(
        "статья 272 договорные обязательства",
        docs,
        target_codes=["Гражданский кодекс РК (Особенная часть)"],
        target_articles=["272"],
    )

    assert ranked[0].metadata["code_ru"] == "Гражданский кодекс РК (Особенная часть)"
