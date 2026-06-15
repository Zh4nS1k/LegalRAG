import pytest
import ai_service.retrieval.rag_chain as rag_chain

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

r = FakeRedis()
rag_chain.get_redis = lambda: r

calls = []
def fake_impl(query, history, intent, model_override=None):
    calls.append((query, history, intent))
    return {"result": "ok", "source_documents": []}

rag_chain._invoke_qa_impl = fake_impl
rag_chain.config.CACHE_ENABLED = True
rag_chain.clear_qa_cache()

first = rag_chain.invoke_qa("вопрос", history=None, intent="CASE_SPECIFIC")
second = rag_chain.invoke_qa("вопрос", history=[], intent="CASE_SPECIFIC")

print(f"first: {first}")
print(f"second: {second}")
print(f"first == second: {first == second}")
