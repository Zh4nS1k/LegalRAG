# LegalRAG Optimization Plan for Faster Response Times

## Current Performance Baseline
Based on architecture analysis, typical response time breakdown:
1. **Embedding Generation**: 120-200ms (multilingual-e5-large)
2. **Vector Search**: 380-500ms (Pinecone + BM25)
3. **Reranking**: 300-800ms (BGE-M3 when enabled)
4. **LLM Inference**: 4900-6000ms (Groq llama-3.1-8b-instant)
5. **Total**: ~5.7-8.5 seconds

**Target**: Reduce to <3 seconds for 80% of queries

## Priority 1: Immediate Config Changes (No Code)

### 1.1 Reduce Context Window
```bash
# In .env file
LEGAL_RAG_CONTEXT_MAX_DOCS=3                    # Was 4
LEGAL_RAG_CONTEXT_MAX_CHARS_PER_DOC=600         # Was 900
LEGAL_RAG_CONTEXT_MAX_TOTAL_TOKENS=1000         # Was 1400
LEGAL_RAG_CONTEXT_MAX_TOKENS_PER_DOC=300        # Was 380
```

**Expected Impact**: 30-40% faster LLM inference

### 1.2 Optimize Retrieval Parameters
```bash
# In .env file
LEGAL_RAG_RETRIEVER_WIDE_K=16                   # Was 24 (fewer initial candidates)
LEGAL_RAG_RETRIEVER_TOP_K_AFTER_RERANK=4        # Was 5
LEGAL_RAG_RETRIEVER_MIN_K_CRIMINAL=6            # Was 8
LEGAL_RAG_AGENTIC_TOP_K_CANDIDATES=30           # Was 50
```

**Expected Impact**: 20-25% faster retrieval

### 1.3 Smart Reranking Toggles
```bash
# In .env file
LEGAL_RAG_USE_RERANKER=1                        # Keep enabled
LEGAL_RAG_RERANK_DYNAMIC_SKIP=1                 # NEW: Skip rerank for simple queries
LEGAL_RAG_RERANK_SKIP_LEXICAL_THRESHOLD=0.4     # Was 0.35 (higher threshold = more skipping)
LEGAL_RAG_RERANK_SKIP_MIN_CODE_MATCHES=1        # Was 2
```

## Priority 2: Caching Implementation

### 2.1 Query Result Cache
Already implemented: `ai_service/utils/query_cache.py`

**Configuration:**
```bash
# In .env file
REDIS_URL=redis://localhost:6379/0              # Optional, Redis for production
LEGAL_RAG_CACHE_TTL_SECONDS=3600                # 1 hour cache
LEGAL_RAG_CACHE_ENABLED=1
```

**Integration Points:**
1. Cache frequent query results (1-hour TTL)
2. Cache embeddings for common legal terms
3. Cache vector search results for article lookups

## Priority 3: Embedding Optimization

### 3.1 Lighter Embedding Model
Switch from `multilingual-e5-large` (1.1GB) to `intfloat/multilingual-e5-base` (480MB):

```bash
# In .env file
LEGAL_RAG_EMBEDDING=intfloat/multilingual-e5-base
```

**Performance Impact:**
- 30-40% faster embedding generation
- 50% less memory usage
- Minimal quality loss for legal text

### 3.2 Embedding Cache
Cache embeddings for frequent legal terms and article references:

```python
# Add to ai_service/retrieval/rag_chain.py
_embedding_cache = {}
def get_cached_embedding(text: str) -> List[float]:
    key = hashlib.md5(text.encode()).hexdigest()
    if key not in _embedding_cache:
        _embedding_cache[key] = embeddings.embed_query(text)
        # Limit cache size
        if len(_embedding_cache) > 10000:
            _embedding_cache.pop(next(iter(_embedding_cache)))
    return _embedding_cache[key]
```

## Priority 4: Parallel Processing

### 4.1 Concurrent Retrieval
Execute vector search and BM25 in parallel:

```python
# Modified retrieval pipeline
async def parallel_retrieval(query: str):
    import asyncio
    
    # Run both retrievers concurrently
    vector_task = asyncio.create_task(vector_retriever.get_relevant_documents(query))
    bm25_task = asyncio.create_task(bm25_retriever.get_relevant_documents(query))
    
    vector_results, bm25_results = await asyncio.gather(vector_task, bm25_task)
    
    # Fusion logic...
    return fused_results
```

**Expected Impact**: 40% faster retrieval (380ms → 230ms)

### 4.2 Streaming LLM Responses
Already implemented: `/api/v1/chat-stream` endpoint

**Optimization**: Send first token as soon as available, don't wait for full response.

## Priority 5: Query Classification & Routing

### 5.1 Fast-Path for Simple Queries
Create classification rules for queries that don't need full RAG:

```python
def should_use_fast_path(query: str) -> bool:
    """Check if query can use simplified response."""
    fast_patterns = [
        r'стать[ьяи]\s+\d+',          # Article lookup
        r'ст\.\s*\d+',                # Article shorthand
        r'что такое\s+[\w\s]+',       # Definition requests
        r'определение\s+[\w\s]+',     # Definition requests
    ]
    
    for pattern in fast_patterns:
        if re.search(pattern, query.lower()):
            return True
    
    return False
```

**Fast Path Flow:**
1. Direct article lookup via BM25 only
2. No reranking, no LLM (return raw article text)
3. Response in <500ms

## Priority 6: Infrastructure Optimization

### 6.1 Pinecone Optimization
```python
# Use faster Pinecone parameters
pinecone_config = {
    "top_k": 10,                      # Reduced from default
    "include_metadata": True,
    "include_values": False,          # Don't need vector values
    "namespace": config.PINECONE_NAMESPACE,
}
```

### 6.2 LLM Provider Tuning
**Groq Optimization:**
```bash
# In .env file
LEGAL_RAG_LLM=llama-3.1-8b-instant           # Already optimal
LEGAL_RAG_LLM_MAX_TOKENS=1024                # Reduce from 2048
LEGAL_RAG_LLM_TEMPERATURE=0.1                # Slight temp for faster convergence
```

**Alternative: Local Ollama** (if network latency > 100ms):
```bash
LEGAL_RAG_LLM_BACKEND=ollama
LEGAL_RAG_LLM=llama3.2:3b                    # Much faster 3B model
OLLAMA_HOST=http://localhost:11434
```

## Priority 7: Monitoring & A/B Testing

### 7.1 Performance Metrics
Track these key metrics:
```python
metrics_to_track = {
    "total_response_time_ms": "end-to-end",
    "embedding_time_ms": "query embedding",
    "retrieval_time_ms": "vector + BM25 search",
    "reranking_time_ms": "BGE-M3 rerank",
    "llm_time_ms": "LLM inference",
    "cache_hit_rate": "query cache effectiveness",
}
```

### 7.2 A/B Testing Framework
```python
class OptimizationExperiment:
    def __init__(self):
        self.variants = {
            "control": {"context_docs": 4, "use_reranker": True},
            "optimized": {"context_docs": 3, "use_reranker": False},
            "fast": {"context_docs": 2, "use_bm25_only": True},
        }
    
    def route_query(self, query: str) -> str:
        # Simple query → fast variant
        # Complex query → optimized variant
        # Critical query → control variant
        pass
```

## Implementation Roadmap

### Phase 1 (Week 1): Config Changes
1. Apply .env optimizations
2. Test with benchmark suite
3. Monitor performance impact

### Phase 2 (Week 2): Caching
1. Deploy query cache
2. Add Redis if available
3. Implement embedding cache

### Phase 3 (Week 3): Parallel Processing
1. Implement concurrent retrieval
2. Optimize streaming responses
3. Add fast-path routing

### Phase 4 (Week 4): Model Optimization
1. Test lighter embedding models
2. Evaluate local Ollama option
3. Fine-tune parameters

## Expected Performance Gains

| Optimization | Time Reduction | Cumulative |
|-------------|---------------|------------|
| Context Reduction | 30% | 5.7s → 4.0s |
| Caching (50% hit rate) | 25% | 4.0s → 3.0s |
| Parallel Retrieval | 20% | 3.0s → 2.4s |
| Lighter Embeddings | 15% | 2.4s → 2.0s |
| Fast-Path Routing | 40% (for 30% queries) | 2.0s → 1.8s avg |

**Final Target**: ~2 seconds average response time

## Risk Mitigation

1. **Quality Impact**: Test each change with benchmark suite
2. **Cache Staleness**: Use appropriate TTLs (1 hour for laws)
3. **Memory Usage**: Monitor cache size, implement eviction policies
4. **Complexity**: Implement gradually, maintain rollback capability

## Success Metrics

1. **P95 Response Time**: <3 seconds
2. **Cache Hit Rate**: >40% for production traffic
3. **LLM Token Usage**: Reduce by 30%
4. **User Satisfaction**: Maintain or improve answer quality scores

---

**Next Step**: Begin with Phase 1 config changes and measure baseline performance before proceeding.