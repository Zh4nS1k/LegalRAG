from ai_service.retrieval import rag_chain


def test_reset_instances_clears_singletons(monkeypatch):
    monkeypatch.setattr(rag_chain, "_embeddings_instance", object())
    monkeypatch.setattr(rag_chain, "_vector_store_instance", object())
    monkeypatch.setattr(rag_chain, "_retriever_instance", object())
    monkeypatch.setattr(rag_chain, "_llm_instance", object())
    monkeypatch.setattr(rag_chain._ensure_latency_patches, "_done", True, raising=False)

    rag_chain.reset_instances()

    assert rag_chain._embeddings_instance is None
    assert rag_chain._vector_store_instance is None
    assert rag_chain._retriever_instance is None
    assert rag_chain._llm_instance is None
    assert rag_chain._ensure_latency_patches._done is False
