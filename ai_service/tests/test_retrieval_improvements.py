import asyncio

import pytest
from langchain_core.documents import Document

from ai_service.retrieval import agentic_workflow as aw
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
    _fuse_retrieval_candidates,
    _rank_docs_with_legal_scoring,
    _rewrite_query_for_retrieval,
    _get_llm_runnable,
    _TrimRetriever,
)
from ai_service.utils.circuit_breaker import CircuitBreaker, CircuitBreakerProxy


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

    assert "Context: This chunk is from Chapter 33 Возмездное оказание услуг" in result
    assert "Кодекс: Гражданский кодекс РК (Особенная часть)" in result
    assert "Глава: 33 Возмездное оказание услуг" in result
    assert "Статья: 683. Договор возмездного оказания услуг" in result
    assert "Пункт: 1" in result
    assert "Текст: 1. Исполнитель обязан оказать услугу." in result


def test_build_summary_document_uses_summary_metadata():
    from ai_service.processing.prepare_data import _build_summary_document

    source_docs = [
        Document(
            page_content="Context: This chunk is from Article 1 of the Civil Code.",
            metadata={
                "source": "civil_code.txt",
                "code_ru": "Гражданский кодекс РК",
                "code_kz": "Қазақстан Республикасының Азаматтық кодексі",
                "chapter_title": "Общие положения",
                "chapter_number": "1",
                "article_number": "1",
                "article_title": "Предмет регулирования",
                "clause_level": "article",
            },
        ),
        Document(
            page_content="Context: This chunk is from Article 2 of the Civil Code.",
            metadata={
                "source": "civil_code.txt",
                "code_ru": "Гражданский кодекс РК",
                "code_kz": "Қазақстан Республикасының Азаматтық кодексі",
                "chapter_title": "Общие положения",
                "chapter_number": "1",
                "article_number": "2",
                "article_title": "Отношения, регулируемые гражданским законодательством",
                "clause_level": "article",
            },
        ),
    ]

    summary = _build_summary_document(
        "civil_code.txt",
        source_docs[0].metadata,
        source_docs,
        corpus_version="abc123",
    )

    assert summary.metadata["doc_kind"] == "summary"
    assert summary.metadata["summary_level"] == "document"
    assert summary.metadata["summary_source"] == "civil_code.txt"
    assert summary.metadata["jurisdiction"] == "RK"
    assert "Summary index entry for retrieval" in summary.page_content
    assert "Representative articles" in summary.page_content


def test_rewrite_query_for_retrieval_adds_legal_terms():
    query = "можно ли платить наличными в дубае за товар для тоо"

    rewritten = _rewrite_query_for_retrieval(query)

    assert "наличные расчеты" in rewritten
    assert "товарищество с ограниченной ответственностью" in rewritten
    assert "товариществах с ограниченной и дополнительной ответственностью" in rewritten


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


def test_trim_retriever_uses_parent_article_text_for_short_chunks():
    class _StaticRetriever:
        def invoke(self, query):
            return [
                Document(
                    page_content="Пункт 1.",
                    metadata={
                        "code_ru": "Уголовный кодекс РК",
                        "article_number": "190",
                        "parent_article_text": "Статья 190. Мошенничество. Полный текст статьи с контекстом.",
                    },
                )
            ]

    trimmed = _TrimRetriever(base_retriever=_StaticRetriever(), max_docs=1).invoke(
        "мошенничество"
    )

    assert len(trimmed) == 1
    assert "Пункт 1." not in trimmed[0].page_content
    assert "Полный текст статьи с контекстом" in trimmed[0].page_content


def test_article_splitter_preserves_parent_article_text():
    from ai_service.processing.prepare_data import ArticleTextSplitter

    splitter = ArticleTextSplitter()
    docs = splitter.create_documents(
        [
            "Статья 1. Тестовая статья с достаточным количеством текста для разбиения.\n"
            "1. Первый пункт содержит достаточно текста для проверки.\n"
            "2. Второй пункт также содержит достаточно текста для проверки."
        ],
        [{"source": "documents/constitution.txt"}],
    )

    assert docs
    assert all(doc.metadata.get("parent_article_text") for doc in docs)


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


def test_fused_retrieval_candidates_prioritize_docs_supported_by_both_sources():
    query = "какая ответственность за кражу"
    vector_doc = Document(
        page_content="Статья 188. Кража",
        metadata={
            "code_ru": "Уголовный кодекс РК",
            "article_number": "188",
            "article_title": "Кража",
        },
    )
    bm25_doc = Document(
        page_content="Статья 190. Мошенничество",
        metadata={
            "code_ru": "Уголовный кодекс РК",
            "article_number": "190",
            "article_title": "Мошенничество",
        },
    )
    vector_candidates = {
        ("Уголовный кодекс РК", "188"): {
            "doc": vector_doc,
            "vector_raw": 0.95,
            "bm25_raw": 0.0,
        },
        ("Уголовный кодекс РК", "190"): {
            "doc": bm25_doc,
            "vector_raw": 0.10,
            "bm25_raw": 0.0,
        },
    }
    bm25_candidates = {
        ("Уголовный кодекс РК", "188"): {
            "doc": vector_doc,
            "vector_raw": 0.95,
            "bm25_raw": 0.90,
        },
        ("Уголовный кодекс РК", "190"): {
            "doc": bm25_doc,
            "vector_raw": 0.10,
            "bm25_raw": 0.85,
        },
    }

    ranked = _fuse_retrieval_candidates(
        query,
        vector_candidates=vector_candidates,
        bm25_candidates=bm25_candidates,
        target_codes=["Уголовный кодекс РК"],
        target_articles=["188"],
        limit=5,
    )

    assert ranked[0].metadata["article_number"] == "188"
    assert ranked[0].metadata["relevance_score"] >= ranked[1].metadata["relevance_score"]


def test_crag_evaluator_flags_missing_revision_date_for_temporal_query():
    query = "Действует ли статья 50 на 2024 год?"
    doc = Document(
        page_content="Статья 50. Общие положения.",
        metadata={
            "code_ru": "Закон РК о публичной службе",
            "article_number": "50",
            "status": "действует",
        },
    )

    evaluation = aw._evaluate_context_quality(query, [doc], [0.31])

    assert evaluation["action"] == "rewrite"
    assert "missing_revision_date" in evaluation["missing_reasons"]
    assert evaluation["context_score"] < aw.CRAG_MIN_CONTEXT_SCORE


def test_crag_builder_generates_rewrite_and_decomposition_candidates():
    query = "Проверь статью 50 и скажи, действует ли она сейчас"
    top_docs = [
        Document(
            page_content="Статья 50. Общие положения.",
            metadata={
                "code_ru": "Закон РК о публичной службе",
                "article_number": "50",
                "revision_date": "2024-01-01",
            },
        )
    ]
    evaluator = {
        "action": "decompose",
        "missing_reasons": ["missing_revision_date", "missing_status"],
        "contradictions": [],
    }

    corrective_queries = aw._build_corrective_queries(query, top_docs, evaluator, "trace-crag")

    assert corrective_queries[0] == query
    assert any("действующая редакция" in item.lower() for item in corrective_queries)
    assert any("действует утратил силу" in item.lower() for item in corrective_queries)
    assert any(item != query for item in corrective_queries[1:])


def test_corrective_retrieve_rewrites_then_accepts(monkeypatch):
    query = "Действует ли статья 50 на 2024 год?"
    bad_doc = Document(
        page_content="Статья 50. Общие положения.",
        metadata={
            "code_ru": "Закон РК о публичной службе",
            "article_number": "50",
            "status": "действует",
        },
    )
    good_doc = Document(
        page_content="Статья 50. Общие положения.",
        metadata={
            "code_ru": "Закон РК о публичной службе",
            "article_number": "50",
            "status": "действует",
            "revision_date": "2024-01-01",
        },
    )
    seen_queries: list[list[str]] = []

    async def fake_retrieve(queries, hyde_doc, trace_id):
        seen_queries.append(list(queries))
        if any(q.strip() != query for q in queries):
            return [good_doc], [0.93], {"retrieval_candidates_ms": 1, "n_candidates": 1}
        return [bad_doc], [0.22], {"retrieval_candidates_ms": 1, "n_candidates": 1}

    async def fake_rerank(query_arg, candidates, candidate_scores, trace_id):
        return candidates[:5], candidate_scores[:5], {"censor_rerank_ms": 0}

    monkeypatch.setattr(aw, "_retrieve_candidates", fake_retrieve)
    monkeypatch.setattr(aw, "_censor_rerank", fake_rerank)

    top_docs, top_scores, metrics = asyncio.run(
        aw._corrective_retrieve(query, None, "trace-crag", [query], "")
    )

    assert any(
        "действующая редакция" in q.lower() or "дата" in q.lower()
        for batch in seen_queries
        for q in batch
    )
    assert metrics["crag_rounds"] >= 1
    assert metrics["crag_last_action"] in {"rewrite", "accept"}
    assert top_docs[0].metadata["revision_date"] == "2024-01-01"
    assert top_scores[0] == pytest.approx(0.93)


def test_get_llm_runnable_unwraps_circuit_breaker_proxy():
    class DummyLLM:
        def invoke(self, prompt):
            return "ok"

    proxy = CircuitBreakerProxy(DummyLLM(), CircuitBreaker("dummy"))

    from ai_service.retrieval import rag_chain

    original = rag_chain.get_llm
    try:
        rag_chain.get_llm = lambda model_override=None: proxy
        runnable = _get_llm_runnable()
    finally:
        rag_chain.get_llm = original

    assert runnable is proxy._target
