import json
from unittest.mock import MagicMock

import pytest

from ai_service.core import config
from ai_service.retrieval import rag_chain
from ai_service.retrieval.sherlock_engine import SherlockEngine


@pytest.fixture
def sherlock(monkeypatch):
    monkeypatch.setattr(config, "LEGAL_RAG_ENABLE_SHERLOCK", False)
    engine = SherlockEngine()
    engine.llm = MagicMock()
    return engine


@pytest.mark.asyncio
async def test_sherlock_disabled_by_default(sherlock):
    result = await sherlock.run_sherlock_loop("Работодатель не платит зарплату")

    assert result["deductive_output"] is None
    assert result["meta"]["enabled"] is False
    assert result["meta"]["mode"] == "disabled"


@pytest.mark.asyncio
async def test_sherlock_uses_canonical_retriever(monkeypatch):
    monkeypatch.setattr(config, "LEGAL_RAG_ENABLE_SHERLOCK", True)
    engine = SherlockEngine()

    docs = [
        MagicMock(
            page_content="Статья 1. Трудовые отношения регулируются настоящим кодексом.",
            metadata={
                "code_ru": "Трудовой кодекс РК",
                "article_number": "1",
            },
        )
    ]
    retriever = MagicMock()
    retriever.invoke.return_value = docs
    monkeypatch.setattr(
        "ai_service.retrieval.sherlock_engine.rag_chain.get_retriever_for_coverage",
        lambda top_k=None: retriever,
    )

    mock_resp = MagicMock()
    mock_resp.content = json.dumps(
        {
            "summary": "Запрос относится к трудовым отношениям.",
            "position": "Работник",
            "applicable_articles": [
                {
                    "code_ru": "Трудовой кодекс РК",
                    "article_number": "1",
                    "reason": "Регулирует трудовые отношения.",
                }
            ],
            "conflicts": [],
            "needs_clarification": False,
            "clarifying_question": "",
            "confidence": 0.82,
        },
        ensure_ascii=False,
    )
    engine.llm = MagicMock()
    engine.llm.invoke.return_value = mock_resp

    result = await engine.run_sherlock_loop("Работодатель не платит зарплату")

    retriever.invoke.assert_called_once_with("Работодатель не платит зарплату")
    assert result["meta"]["enabled"] is True
    assert result["meta"]["mode"] == "canonical_retrieval_audit"
    assert result["meta"]["stack"] == "canonical_retriever"
    assert result["meta"]["docs_count"] == 1
    assert "Sherlock audit" in result["deductive_output"]
    assert "Трудовой кодекс РК" in result["deductive_output"]


@pytest.mark.asyncio
async def test_sherlock_offloads_llm_calls(monkeypatch):
    monkeypatch.setattr(config, "LEGAL_RAG_ENABLE_SHERLOCK", True)
    engine = SherlockEngine()

    retriever = MagicMock()
    retriever.invoke.return_value = []
    monkeypatch.setattr(
        "ai_service.retrieval.sherlock_engine.rag_chain.get_retriever_for_coverage",
        lambda top_k=None: retriever,
    )

    mock_resp = MagicMock()
    mock_resp.content = json.dumps(
        {
            "summary": "ok",
            "position": "Гражданин",
            "applicable_articles": [],
            "conflicts": [],
            "needs_clarification": True,
            "clarifying_question": "Что именно произошло?",
            "confidence": 0.5,
        },
        ensure_ascii=False,
    )
    engine.llm = MagicMock()
    engine.llm.invoke.return_value = mock_resp

    calls = []

    async def fake_to_thread(func, *args, **kwargs):
        calls.append(func)
        return func(*args, **kwargs)

    monkeypatch.setattr(
        "ai_service.retrieval.sherlock_engine.asyncio.to_thread",
        fake_to_thread,
    )

    await engine.run_sherlock_loop("Штраф за парковку")

    assert len(calls) == 2


def test_invoke_qa_offline_mode_uses_extractively_retrieved_docs(monkeypatch):
    monkeypatch.setattr(config, "LEGAL_RAG_OFFLINE_QA", True)
    rag_chain._invoke_qa_impl.cache_clear()

    docs = [
        MagicMock(
            page_content="Статья 188. Кража, то есть тайное хищение чужого имущества.",
            metadata={
                "code_ru": "Уголовный кодекс РК",
                "article_number": "188",
            },
        )
    ]
    retriever = MagicMock()
    retriever.invoke.return_value = docs
    monkeypatch.setattr(
        "ai_service.retrieval.rag_chain._build_offline_bm25_retriever",
        lambda top_k: retriever,
    )
    monkeypatch.setattr(
        "ai_service.retrieval.rag_chain.get_llm",
        lambda: (_ for _ in ()).throw(AssertionError("LLM must not be called in offline QA mode")),
    )

    result = rag_chain.invoke_qa("Что будет если я украду яблоко?")

    assert result["retrieval_method"] == "offline_extractive"
    assert result["source_documents"][0].metadata["article_number"] == "188"
    assert "Офлайн" in result["result"]
    assert "188" in result["result"]
