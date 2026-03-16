import pytest
import asyncio
from unittest.mock import MagicMock, AsyncMock
from ai_service.retrieval.sherlock_engine import SherlockEngine

@pytest.fixture
def sherlock():
    engine = SherlockEngine()
    engine.llm = MagicMock()
    # Mock vectorstore as well
    engine.vectorstore = MagicMock()
    return engine

@pytest.mark.asyncio
async def test_classify_and_validate_labor(sherlock):
    # Mock LLM response for labor query choosing ZK (Land Code) by mistake
    mock_resp = MagicMock()
    mock_resp.content = '{"selected_codes": ["ЗК"], "reasoning": "testing error", "facts": {"action": "salary"}}'
    sherlock.llm.invoke.return_value = mock_resp
    
    query = "Работодатель не платит зарплату"
    result = await sherlock.classify_and_validate(query)
    
    # Validation logic should swap ЗК for ТК
    assert "ТК" in result["selected_codes"]
    assert "ЗК" not in result["selected_codes"]

@pytest.mark.asyncio
async def test_classify_and_validate_penalty(sherlock):
    # Mock LLM response for penalty query choosing GK (Civil Code)
    mock_resp = MagicMock()
    mock_resp.content = '{"selected_codes": ["ГК"], "reasoning": "testing penalty", "facts": {"action": "fine"}}'
    sherlock.llm.invoke.return_value = mock_resp
    
    query = "Штраф за парковку"
    result = await sherlock.classify_and_validate(query)
    
    # Validation logic should add KoAP
    assert "КоАП" in result["selected_codes"]
    assert "ГК" in result["selected_codes"]

@pytest.mark.asyncio
async def test_stage_3_targeted_fetch(sherlock):
    sherlock.vectorstore.similarity_search.return_value = [MagicMock(page_content="Art 1")]
    
    docs = await sherlock.stage_3_targeted_fetch(["ТК"], "увольнение")
    
    sherlock.vectorstore.similarity_search.assert_called_once()
    args, kwargs = sherlock.vectorstore.similarity_search.call_args
    assert kwargs["filter"]["code_ru"] == "Трудовой кодекс Республики Казахстан"

def test_stage_4_fact_check(sherlock):
    doc1 = MagicMock(page_content="Статья 437. Нарушение тишины в ночное время (с 23 до 6 часов утра)")
    doc2 = MagicMock(page_content="Статья 1. Основные понятия")
    
    facts = {"time": "23:00"}
    
    # Test success
    success, matched = sherlock.stage_4_fact_check([doc1, doc2], facts)
    assert success is True
    assert len(matched) == 1
    assert "437" in matched[0].page_content
    
    # Test failure (no match)
    success, matched = sherlock.stage_4_fact_check([doc2], facts)
    assert success is False
