import pytest
from engine.retrieval import rag_chain


class FakeRedis:
    def __init__(self):
        self.store = {}

    def get(self, key):
        return self.store.get(key)

    def setex(self, key, ttl, value):
        self.store[key] = value

    def delete(self, *names):
        for name in names:
            self.store.pop(name, None)


@pytest.fixture
def fake_redis(monkeypatch):
    r = FakeRedis()
    monkeypatch.setattr(rag_chain, "get_redis", lambda: r)
    return r


def test_cache_skips_history(monkeypatch, fake_redis):
    calls = []

    def fake_impl(query, history, intent, model_override=None):
        calls.append((query, history, intent))
        return {"result": "ok", "source_documents": [], "retrieval_method": "hybrid"}

    monkeypatch.setattr(rag_chain, "_invoke_qa_impl", fake_impl)
    rag_chain.clear_qa_cache()

    rag_chain.invoke_qa("вопрос", history=[{"role": "user", "content": "привет"}], intent="CASE_SPECIFIC")

    assert len(calls) == 1
    assert calls[0][1] == [{"role": "user", "content": "привет"}]


def test_cache_hits_without_history(monkeypatch, fake_redis):
    calls = []

    def fake_impl(query, history, intent, model_override=None):
        calls.append((query, history, intent))
        return {"result": "ok", "source_documents": [], "retrieval_method": "hybrid"}

    monkeypatch.setattr(rag_chain, "_invoke_qa_impl", fake_impl)
    rag_chain.clear_qa_cache()

    first = rag_chain.invoke_qa("вопрос", history=None, intent="CASE_SPECIFIC")
    second = rag_chain.invoke_qa("вопрос", history=[], intent="CASE_SPECIFIC")

    assert first == second
    assert len(calls) == 1
