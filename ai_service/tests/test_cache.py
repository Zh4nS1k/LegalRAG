from ai_service.retrieval import rag_chain


def test_cache_skips_history(monkeypatch):
    calls = []

    def fake_impl(query, history, intent):
        calls.append((query, history, intent))
        return {"result": "ok", "source_documents": []}

    monkeypatch.setattr(rag_chain, "_invoke_qa_impl", fake_impl)
    rag_chain.clear_qa_cache()

    rag_chain.invoke_qa("вопрос", history=[{"role": "user", "content": "привет"}], intent="CASE_SPECIFIC")

    assert len(calls) == 1
    assert calls[0][1] == [{"role": "user", "content": "привет"}]


def test_cache_hits_without_history(monkeypatch):
    calls = []

    def fake_impl(query, history, intent):
        calls.append((query, history, intent))
        return {"result": "ok", "source_documents": []}

    monkeypatch.setattr(rag_chain, "_invoke_qa_impl", fake_impl)
    rag_chain.clear_qa_cache()

    first = rag_chain.invoke_qa("вопрос", history=None, intent="CASE_SPECIFIC")
    second = rag_chain.invoke_qa("вопрос", history=[], intent="CASE_SPECIFIC")

    assert first == second
    assert len(calls) == 1
