from unittest.mock import MagicMock

import pytest

from ai_service.retrieval.verification_engine import VerificationEngine


@pytest.mark.asyncio
async def test_targeted_fetch_koap_uses_indexed_code_name(monkeypatch):
    engine = VerificationEngine.__new__(VerificationEngine)
    engine.llm = MagicMock()

    vectorstore = MagicMock()
    vectorstore.similarity_search.return_value = [MagicMock(page_content="Art 437")]

    monkeypatch.setattr(
        "ai_service.retrieval.verification_engine.rag_chain.get_vector_store",
        lambda: vectorstore,
    )

    await VerificationEngine.targeted_fetch(engine, "КоАП", "шум в субботу", ["тишина"])

    _, kwargs = vectorstore.similarity_search.call_args
    assert kwargs["filter"]["code_ru"] == "Кодекс об административных правонарушениях РК"


@pytest.mark.asyncio
async def test_targeted_fetch_gk_uses_or_filter_for_both_parts(monkeypatch):
    engine = VerificationEngine.__new__(VerificationEngine)
    engine.llm = MagicMock()

    vectorstore = MagicMock()
    vectorstore.similarity_search.return_value = [MagicMock(page_content="Art 272")]

    monkeypatch.setattr(
        "ai_service.retrieval.verification_engine.rag_chain.get_vector_store",
        lambda: vectorstore,
    )

    await VerificationEngine.targeted_fetch(engine, "ГК", "договор", ["обязательство"])

    _, kwargs = vectorstore.similarity_search.call_args
    assert kwargs["filter"] == {
        "$or": [
            {"code_ru": "Гражданский кодекс РК (Общая часть)"},
            {"code_ru": "Гражданский кодекс РК (Особенная часть)"},
        ]
    }
