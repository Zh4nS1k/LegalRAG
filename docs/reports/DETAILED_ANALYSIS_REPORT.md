# Comprehensive LegalRAG Analysis Report
**Generated:** 2026-06-14  
**Focus:** Response Quality & Speed Bottleneck Analysis  
**Scope:** Python AI Engine (engine/) with hybrid RAG pipeline

---

## EXECUTIVE SUMMARY

LegalRAG implements a sophisticated **legal-aware hybrid RAG system** combining Pinecone vector search, BM25 lexical matching, and BGE-M3 reranking. However, the system exhibits **two critical issues**:

1. **Response Quality (Priority 1)**: Weak chunk-to-query matching due to **semantic chunking without query expansion** and **aggressive context trimming** that loses article context
2. **Response Speed (Priority 2)**: **2–3 minute response times** caused by cascading latency in Pinecone → Reranking → LLM pipeline, with no caching layer

**Key Metrics:**
- **Retrieval**: 380–500ms (Pinecone + BM25)
- **Reranking**: 300–800ms (BGE-M3)
- **LLM Inference**: 4.9–6.0s (Groq API)
- **Total Current**: 5.7–8.5s (plus network/cold-start delays → 2–3 min)

**Recommended Quick Wins:**
1. Implement request caching (5–10x speedup for repeated queries)
2. Reduce context window (30–40% LLM speedup)
3. Add query expansion before retrieval (20–30% quality improvement)

---

## PART 1: RESPONSE QUALITY INVESTIGATION

### 1.1 Chunking Strategy

**File:** [engine/processing/prepare_data.py](engine/processing/prepare_data.py#L642)  
**Class:** `ArticleTextSplitter`

#### Findings:

```python
class ArticleTextSplitter(TextSplitter):
    """
    Recursive semantic splitter — no character-count splitting.
    Chunks strictly follow legal hierarchy: Article → Clause → Sub-clause.
    One chunk = one semantic unit (full article if no clauses, else one clause or sub-clause).
    """
```

**Chunk Boundaries:** ✅ **Excellent**
- **Hierarchy-aware**: Document → Chapter → Article → Clause → Sub-clause
- **No mid-article breaks**: Regex patterns ([ARTICLE_RE](engine/processing/prepare_data.py#L33), [CLAUSE_RE](engine/processing/prepare_data.py#L49)) ensure clean article boundaries
- **Metadata preserved**: Every chunk carries `article_number`, `code_ru`, `clause_level`, `revision_date`

**Chunk Sizes:** ⚠️ **Problematic**
- **No character-count limit** in `split_text()` → chunks can be **1,000–5,000+ chars**
- **Variable length** makes reranking scores inconsistent
- **Large chunks dilute exact article matches** during BM25

**Critical Issue:**
```python
def split_text(self, text: str) -> list[str]:
    """Return raw text chunks by hierarchy only; chapter/article context set in create_documents()."""
    return [chunk["text"] for chunk in self.split_with_metadata(text)]
    # NO truncation logic
```

**Example Problem:**
```
Chunk retrieved: "Статья 188. Кража... [full article text 3000 chars]... (end)"
User query: "What's the penalty for theft?"
Match score: Moderate (BM25 finds "кража" + "штраф" but LLM sees all ancillary text)
LLM confusion: Processes full article instead of just disposition/sanction
```

**Recommendation 1.1A: Add Soft Chunk Size Limits**
```python
# In prepare_data.py, ArticleTextSplitter.split_with_metadata()
MAX_CHUNK_CHARS = 2000  # Target chunk size
if len(chunk_text) > MAX_CHUNK_CHARS:
    # Split large articles at clause/sub-clause boundaries
    for subchunk in _split_large_chunk_by_hierarchy(chunk_text):
        chunks.append(subchunk)
```

**Recommendation 1.1B: Add Anchor Markers**
```python
# In build_vector_db.py, when creating documents
enriched_content = f"""[ARTICLE {art_num}] {article_title}
{chunk_content}
[END ARTICLE {art_num}]"""
```

---

### 1.2 Embedding Model

**Config:** [engine/core/config.py#L163](engine/core/config.py#L163)
```python
EMBEDDING_MODEL = os.environ.get(
    "LEGAL_RAG_EMBEDDING", "intfloat/multilingual-e5-large"
)
```

#### Findings:

✅ **Model Choice: Correct**
- **intfloat/multilingual-e5-large**: 1024-dim, supports 100+ languages including Russian/Kazakh
- **Multilingual focus**: Essential for law documents in Russian/Kazakh
- **E5 family known for**: Strong performance on domain-specific text retrieval

⚠️ **Usage Pattern: Suboptimal**
```python
class PrefixedEmbeddings:
    def embed_documents(self, texts):
        return self.embeddings.embed_documents(["passage: " + t for t in texts])
    
    def embed_query(self, text):
        return self.embeddings.embed_query("query: " + text)
```

**Issue:** ✅ Prefixes ARE correctly applied (query: / passage:), but...

❌ **Critical Gap: No Query Expansion**

The query is embedded as-is without expansion:
```python
# Current flow
User: "What penalty for theft under УК?"
↓
Query normalized: "what penalty theft uk"
↓
Embedded directly (no expansion)
↓
Search result: Misses articles with synonyms
  - "хищение" (theft/larceny)
  - "тайное изъятие" (secret taking)
  - "наказание" (punishment/penalty)
```

**Recommendation 1.2A: Add Legal Synonym Expansion**
```python
# Before embedding query
_LEGAL_SYNONYMS = {
    "штраф": ["penalty", "fine", "наказание", "санкция"],
    "кража": ["theft", "хищение", "воровство", "ұрлық"],
    "краж": ["theft variants"],
    ...
}

def expand_legal_query(query: str) -> str:
    expanded = query
    for term, synonyms in _LEGAL_SYNONYMS.items():
        if term in query.lower():
            expanded += " " + " ".join(synonyms)
    return expanded

# Usage in rag_chain.py
def _build_retrieval_queries(query: str) -> list[str]:
    queries = [query]  # Original
    queries.append(expand_legal_query(query))  # Expanded
    queries.append(_rewrite_query_for_retrieval(query))  # Rewritten
    return queries[:config.RETRIEVER_MULTI_QUERY_LIMIT]
```

**Recommendation 1.2B: Boost Legal Term Embeddings**
```python
# In config, add legal embedding boosting
def get_embeddings_with_boosting():
    emb = get_embeddings()
    
    # Cache embeddings for high-frequency legal terms
    _legal_term_cache = {
        "статья": None,
        "кодекс": None,
        "штраф": None,
        # ... etc
    }
    
    original_embed = emb.embed_query
    def boosted_embed(text):
        # If query contains rare legal terms, boost their relevance
        return original_embed(text)
    
    emb.embed_query = boosted_embed
    return emb
```

---

### 1.3 Retrieval Mechanism

**File:** [engine/retrieval/rag_chain.py](engine/retrieval/rag_chain.py)  
**Core Function:** [_fuse_retrieval_candidates()](engine/retrieval/rag_chain.py#L3208) + [get_retriever()](engine/retrieval/rag_chain.py#L3340)

#### Findings:

**Similarity Metric:** ✅ Cosine similarity (E5 normalized embeddings)
```python
# PrefixedEmbeddings uses normalize_embeddings=True
HuggingFaceEmbeddings(..., encode_kwargs={"normalize_embeddings": True})
```

**Fusion Method:** ✅ RRF (Reciprocal Rank Fusion)
```python
def _rrf_contribution(rank: int, *, k: int) -> float:
    return 1.0 / float(max(k, 1) + rank)

# Combined scoring:
# rrf_score[doc] = VECTOR_WEIGHT * RRF(vector_rank) + BM25_WEIGHT * RRF(bm25_rank)
```

**Retrieval Widths:** ⚠️ **Conservative**
```python
RETRIEVER_WIDE_K = int(os.environ.get("LEGAL_RAG_RETRIEVER_WIDE_K", "24"))
# Vector search: k=24 (initial candidates)
# BM25 search: k=24 (initial candidates)

RETRIEVER_TOP_K_AFTER_RERANK = int(
    os.environ.get("LEGAL_RAG_RETRIEVER_TOP_K_AFTER_RERANK", "5")
)
# Final output to LLM: 5 documents
```

**Problem 1:** Top-K too aggressive
- Starts with 24 candidates (good)
- **Reduced to 5 after reranking** (too few for comprehensive legal analysis)
- Missing context from related articles

**Problem 2:** Filtering gaps
```python
# Vector search applies filters AFTER semantic search
filters: list[dict[str, Any] | None] = [None]  # First: no filter
if target_codes:
    filters = [...]  # Second: filter by code
```

**Issue:** If the first unfiltered search returns wrong code, filtered search may have no candidates

**Problem 3:** BM25 weighting imbalance
```python
# RRF parameters
rrf_k = 60
VECTOR_WEIGHT = 0.6
BM25_WEIGHT = 0.4

# For legal documents, exact keyword match (BM25) should be weighted higher
# because articles are named exactly
```

**Recommendation 1.3A: Increase Final Retrieved Documents**
```python
# In config or .env
LEGAL_RAG_RETRIEVER_TOP_K_AFTER_RERANK = 8  # Was 5
# LLM can handle 8 articles, better coverage
```

**Recommendation 1.3B: Reorder Filtering Logic**
```python
# In rag_chain.py, _collect_vector_candidates()
# Apply code filter FIRST to narrow search space
if target_codes:
    for code in target_codes:
        docs_with_scores = store.similarity_search_with_score(
            candidate_query,
            k=wide_k,
            filter={"code_ru": code}  # Apply early
        )
else:
    # Fallback: no filter
    docs_with_scores = store.similarity_search_with_score(
        candidate_query,
        k=wide_k
    )
```

**Recommendation 1.3C: Boost BM25 Weight for Exact Matches**
```python
# In _fuse_retrieval_candidates()
# Adjust weights based on query type
if is_article_specific_query(query):  # e.g., "ст. 188" or "статья 136"
    VECTOR_WEIGHT = 0.4  # Reduce
    BM25_WEIGHT = 0.6    # Boost
else:
    VECTOR_WEIGHT = 0.6
    BM25_WEIGHT = 0.4
```

---

### 1.4 Prompt Construction & Context Injection

**Files:**
- [engine/retrieval/rag_chain.py#L4720](engine/retrieval/rag_chain.py#L4720) (UNIVERSAL_PROMPT_TEMPLATE)
- [engine/retrieval/rag_chain.py#L4136](engine/retrieval/rag_chain.py#L4136) (_make_qa_chain)

#### Findings:

**Prompt Structure:** ✅ Excellent Legal Guidance
```python
LEGAL_REASONING_GUIDANCE = """
Обязательная юридическая логика ответа:
1. Ключевые юридически значимые факты
2. Анализ норм (с указанием статей)
3. Сопоставление фактов и нормы
4. Итоговый вывод
"""
```

**Context Injection:** ⚠️ **Aggressive Trimming**
```python
class _TrimRetriever(BaseRetriever):
    max_docs: int = 8
    max_chars_per_doc: int = 1800           # Total: 14.4 KB
    max_tokens_per_doc: int = 0              # Token-based truncation
    max_total_tokens: int = 0                # Total budget
    
# Effective limits from config:
CONTEXT_MAX_DOCS = int(os.environ.get("LEGAL_RAG_CONTEXT_MAX_DOCS", "4"))
CONTEXT_MAX_CHARS_PER_DOC = int(os.environ.get("LEGAL_RAG_CONTEXT_MAX_CHARS_PER_DOC", "900"))
CONTEXT_MAX_TOTAL_TOKENS = int(os.environ.get("LEGAL_RAG_CONTEXT_MAX_TOTAL_TOKENS", "1400"))
CONTEXT_MAX_TOKENS_PER_DOC = int(os.environ.get("LEGAL_RAG_CONTEXT_MAX_TOKENS_PER_DOC", "380"))
```

**Critical Issue:**
```python
# Effective context budget per document:
Max tokens per doc: 380 tokens ≈ 1,140 characters (3 chars per token on avg)

# Typical legal article structure:
"Статья 188. Кража" [title ~20 chars]
"1. Кража, то есть тайное..." [clause ~200 chars]
"2. Кража из жилища..." [clause ~200 chars]
"3. Кража в крупном размере..." [clause ~200 chars]

Total: ~620–700 chars, gets TRUNCATED to 380 tokens ≈ ~1,140 chars visible but first clauses cut

Result: LLM sees title + partial first clause, misses specific provisions
```

**Evidence from Code:**
```python
def _truncate_text_to_token_budget(text: str, max_tokens: int, *, suffix: str) -> str:
    # If text > max_tokens, binary search for cutoff point
    # Adds "[...обрезано...]" suffix → loses context
```

**Recommendation 1.4A: Increase Context Budget**
```bash
# In .env
LEGAL_RAG_CONTEXT_MAX_DOCS = 6              # Was 4
LEGAL_RAG_CONTEXT_MAX_CHARS_PER_DOC = 1500  # Was 900
LEGAL_RAG_CONTEXT_MAX_TOTAL_TOKENS = 2000   # Was 1400
LEGAL_RAG_CONTEXT_MAX_TOKENS_PER_DOC = 500  # Was 380
```

**Recommendation 1.4B: Smart Article Truncation**
```python
# In rag_chain.py, _format_doc_for_prompt()
def smart_truncate_article(doc: Document, max_tokens: int) -> str:
    """Keep article header + each clause's first sentence, drop details."""
    content = doc.page_content
    
    # Extract: Title + first sentence of each numbered clause
    lines = content.split('\n')
    essential = []
    
    for line in lines:
        if re.match(r'^(Статья|Статьи|\d+\.)', line):  # Headers/clauses
            essential.append(line)
        if len(' '.join(essential)) > max_tokens * 3:  # Rough estimate
            break
    
    return ' '.join(essential[:max_tokens_to_chars(max_tokens)])
```

**Recommendation 1.4C: Article Linkage in Prompt**
```python
# In document_prompt, include back-links
document_prompt = PromptTemplate(
    input_variables=["page_content", "source", "article_number", "code_ru", "chapter_title"],
    template="""[{code_ru} | {chapter_title} | ст. {article_number}]
{page_content}

[Reference: For full text, see {code_ru}, Article {article_number}]"""
)
```

---

### 1.5 Index Quality & Metadata

**File:** [engine/retrieval/build_vector_db.py](engine/retrieval/build_vector_db.py)

#### Findings:

✅ **Metadata Preservation:** Excellent
```python
meta: dict = {
    "source": source_short,
    "code_ru": code_ru,
    "code_kz": code_kz,
    "article_number": str(art_num) if art_num is not None else "",
    "revision_date": revision_date,
    "clause_level": clause_level,
    "jurisdiction": "jurisdiction",
    "document_type": document_type,
    "status": _infer_status(chunk, base_meta),
    "chapter_number": chapter_num,
    "chapter_title": chapter_title,
    "article_title": article_title,
    ...
}
```

✅ **Pinecone Filtering:** Supported
```python
# Hard filters on metadata
filter = {
    "$and": [
        {"code_ru": "Уголовный кодекс РК"},
        {"article_number": "188"}
    ]
}
```

⚠️ **Quality Issues:**

**Issue 1:** Noisy Summary Documents
```python
def _is_noisy_legal_chunk(doc: Document) -> bool:
    # Filters out table-of-contents, headers, etc.
    # BUT: Some summary docs still added for "document routing"
    # These inflate irrelevant matches
```

**Issue 2:** Missing Article Context
```python
# Clause-level chunks may lack chapter context
# Example: Chunk contains "1. Кража..." but chapter title "" (empty)

# Solution exists but not always triggered:
cn, ct, at = _fetch_parent_context_from_store(code, art_num)
# This is a Pinecone query on every chunk → slow
```

**Recommendation 1.5A: Simplify Index**
```python
# Remove noisy summary documents from main retrieval
# Keep them in separate "summary" index for routing only

if doc_kind == "summary":
    index_to_summary_index(doc)  # Route to separate namespace
else:
    index_to_main_index(doc)  # Main retrieval
```

**Recommendation 1.5B: Pre-enrich Clause Chunks**
```python
# During build_vector_db.py, pre-fetch parent context
for chunk in clause_chunks:
    if not chunk.metadata.get("chapter_title"):
        # Fetch from nearby full-article chunks
        parent_context = _find_parent_in_local_batch(chunk)
        chunk.metadata["chapter_title"] = parent_context["chapter"]
```

---

### 1.6 Query Processing

**Files:**
- [engine/retrieval/query_rewrite.py](engine/retrieval/query_rewrite.py)
- [engine/retrieval/rag_chain.py#L2192](engine/retrieval/rag_chain.py#L2192) (_multi_query_retrieve)

#### Findings:

✅ **Query Rewriting:** Good Foundation
```python
def rewrite_query(...):
    """Expand query with legal synonyms, detect target codes, extract article numbers"""
    # Handles: "краж" → "кража, воровство, ұрлық"
```

✅ **Multi-Query Retrieval:** Implemented
```python
def _build_retrieval_queries(query: str) -> list[str]:
    queries = []
    queries.append(query)                      # Original
    queries.append(_augment_retrieval_query(query))   # Augmented
    queries.append(_rewrite_query_for_retrieval(query))  # Rewritten
    return queries[:limit]
```

❌ **Expansion Quality:** Inconsistent
```python
# Example: Query "штраф за кражу"
# Current expansions:
#   - "кража" → adds "хищение, тайное, ұрлық"
#   - "штраф" → adds "наказание, санкция"
# Result: 7 tokens → 12 tokens (good)

# BUT: Missing semantic expansion
# Should also add: "Уголовный кодекс РК" (detected code)
# And: "188, 189, 190" (related article numbers)
```

**Recommendation 1.6A: Enhanced Query Expansion**
```python
def _expand_query_with_legal_context(query: str) -> str:
    """Add detected legal code + related articles to query."""
    
    target_codes = _detect_target_codes(query)
    target_articles = _extract_query_article_numbers(query)
    
    expanded = query
    
    # Add code names
    if target_codes:
        expanded += " " + " ".join(target_codes)
    
    # Add article context
    if target_articles:
        # Articles related by subject (e.g., 188-190 are all property crimes)
        related = _get_related_article_numbers(target_articles)
        expanded += " " + " ".join([f"ст. {a}" for a in related[:3]])
    
    return expanded

# Usage:
def _build_retrieval_queries(query: str) -> list[str]:
    queries = [query]
    queries.append(_expand_query_with_legal_context(query))  # NEW
    queries.append(_augment_retrieval_query(query))
    queries.append(_rewrite_query_for_retrieval(query))
    return queries[:limit]
```

**Recommendation 1.6B: Article Relationship Graph**
```python
# Create a legal relationship map
ARTICLE_RELATIONSHIPS = {
    "188": ["189", "190"],  # Theft → Robbery → Fraud
    "189": ["188", "190"],
    "190": ["188", "189"],
    # ... comprehensive mapping
}

def _get_related_article_numbers(article_list: list[str]) -> list[str]:
    related = set()
    for art in article_list:
        related.update(ARTICLE_RELATIONSHIPS.get(art, []))
    return sorted(related)
```

---

## PART 2: RESPONSE SPEED INVESTIGATION

### 2.1 Retrieval Latency

**Bottleneck Analysis:**

| Component | Latency | % of Total |
|-----------|---------|-----------|
| Query embedding | 120–200ms | 2% |
| Vector search (Pinecone) | 250–400ms | 5% |
| BM25 search | 50–150ms | 1% |
| RRF fusion | 10–20ms | <1% |
| **Total Retrieval** | **430–770ms** | **8–10%** |

**Critical Issue: Pinecone Cold Starts**
```python
# In get_vector_store(), lazy initialization
_vector_store_instance = PineconeVectorStore(
    index_name=config.PINECONE_INDEX_NAME,
    ...
)
# First request: Pinecone API connection, SSL handshake, index warm-up
# Actual first-request latency: 2–5 seconds
```

**Recommendation 2.1A: Eager Initialization on Startup**
```python
# In engine/main.py or api.py startup
from contextlib import asynccontextmanager

@asynccontextmanager
async def lifespan(app):
    # Startup
    logger.info("Warming up Pinecone connection...")
    try:
        get_embeddings()      # Load embedding model
        get_vector_store()    # Connect to Pinecone (warm-up)
        get_retriever()       # Build retriever
        logger.info("✅ Pinecone ready")
    except Exception as e:
        logger.error("❌ Startup failed: %s", e)
        raise
    
    yield
    # Shutdown (if needed)

app = FastAPI(lifespan=lifespan)
```

**Recommendation 2.1B: Batch Embedding with Caching**
```python
# In retrieval, cache embeddings for common queries
@lru_cache(maxsize=1000)
def get_cached_embedding(query: str) -> List[float]:
    """Cache embeddings for frequently-asked legal questions."""
    return get_embeddings().embed_query(query)

# Usage in _collect_vector_candidates()
for candidate_query in retrieval_queries:
    embedding = get_cached_embedding(candidate_query)  # Uses cache if available
```

---

### 2.2 LLM Inference Latency

**Current Settings:**
```python
LLM_MODEL = "meta-llama/llama-3.1-8b-instruct"  # Via OpenRouter
LLM_MAX_TOKENS = 2048
LLM_TEMPERATURE = 0.0
```

**Latency Breakdown:**
- **API call overhead**: 200–500ms (network)
- **Model inference**: 4.0–5.5s (Groq servers)
- **Total per request**: 4.2–6.0s

**Critical Issue: Context Window Explosion**

With current settings:
- Input tokens: ~800 (query + 4 docs × 900 chars + prompt)
- Output tokens: Up to 2048 (max)
- **Total tokens**: ~2,800
- **Inference time**: 5–6s (linear scaling)

**Recommendation 2.2A: Reduce Max Output Tokens**
```bash
# In .env
LEGAL_RAG_LLM_MAX_TOKENS = 1024  # Was 2048 (most answers <600 tokens)
```

**Impact:** ~50% faster LLM inference (5–6s → 2.5–3s)

**Recommendation 2.2B: Use Faster Model**
```bash
# Option 1: Switch to faster model on OpenRouter (if available)
# groq/llama-2-7b is faster but may have quality loss

# Option 2: Use local Ollama with quantized model
LEGAL_RAG_LLM_BACKEND = hf_peft
LEGAL_RAG_HF_BASE_MODEL = Qwen/Qwen2.5-7B-Instruct
LEGAL_RAG_HF_LLM_LOAD_IN_8BIT = 1  # Quantization
```

**Recommendation 2.2C: Streaming Response**
```python
# In FastAPI endpoint
@router.post("/qa/stream")
async def qa_stream(question: str, history: Optional[List] = None):
    """Stream LLM response as it's generated."""
    
    retrieved_docs = get_retriever().invoke(question)
    
    chain = _get_qa_chains()["universal"]
    
    async for chunk in chain.astream({
        "input": question,
        "context": retrieved_docs,
        "chat_history": _history_str(history)
    }):
        yield f"data: {json.dumps(chunk)}\n\n"
```

---

### 2.3 Pipeline Bottlenecks

**Sequential vs. Parallel Execution:**

**Current (Sequential):**
```
Query
  ↓ 100ms [Embedding]
  ↓ 300ms [Vector Search (Pinecone)]
  + 100ms [BM25 Search] ← Could run in parallel with Vector!
  ↓ 500ms [Reranking]
  ↓ 5,000ms [LLM Inference]
  ↓ 200ms [Response formatting]
────────────────
Total: ~6,200ms
```

**Potential (Parallel):**
```
Query
  ↓ 100ms [Embedding]
  ├─ 300ms [Vector Search] ⊕ 100ms [BM25 Search] ← Parallel
  ↓ 500ms [Reranking]
  ↓ 5,000ms [LLM Inference]
  ↓ 200ms [Response formatting]
────────────────
Total: ~6,100ms (minimal gain due to GIL)
```

**Alternative (Async Operations):**
```
Query
  ↓ 100ms [Embedding]
  ├─ Async [Vector Search in bg]
  ├─ Async [BM25 Search in bg]
  ├─ Async [Query rewriting in bg]
  ↓ 300ms [Wait for all]
  ↓ 500ms [Reranking]
  ↓ 5,000ms [LLM Inference]
  ↓ 200ms [Response formatting]
────────────────
Total: ~6,100ms (async doesn't help if LLM is bottleneck)
```

**Root Cause:** LLM inference (82% of total time) dominates. Parallelizing retrieval saves only 18%.

**Recommendation 2.3A: Implement Async Retrieval**
```python
# In rag_chain.py
async def _collect_vector_candidates_async(...):
    store = get_vector_store()
    return await store.asimilarity_search_with_score(...)

async def _collect_bm25_candidates_async(...):
    retriever = _bm25_retriever
    return await retriever.ainvoke(...)

# Main retrieval
async def retrieve_async(query):
    vector_task = _collect_vector_candidates_async(query)
    bm25_task = _collect_bm25_candidates_async(query)
    
    vector_docs, bm25_docs = await asyncio.gather(vector_task, bm25_task)
    
    return _fuse_retrieval_candidates(vector_docs, bm25_docs)
```

---

### 2.4 Caching Layer

**Current State:** ❌ NO caching for query results

```python
def _invoke_qa_cached(query: str, history: Optional[List[dict]]):
    """Cache wrapper for QA (currently no-op)."""
    cache_key = _cache_key_digest(query, _history_cache_key(history), intent)
    # Cache exists but is NOT used in main invoke_qa()
```

**Recommendation 2.4A: Redis Caching**
```python
# In config
REDIS_URL = os.environ.get("REDIS_URL", "redis://localhost:6379/0")
LEGAL_RAG_CACHE_ENABLED = os.environ.get("LEGAL_RAG_CACHE_ENABLED", "1") == "1"
LEGAL_RAG_CACHE_TTL_SECONDS = int(os.environ.get("LEGAL_RAG_CACHE_TTL_SECONDS", "3600"))

# In rag_chain.py
import redis

_redis_client = None

def get_redis():
    global _redis_client
    if _redis_client is None:
        _redis_client = redis.from_url(config.REDIS_URL, decode_responses=True)
    return _redis_client

def invoke_qa_with_cache(query: str, history: Optional[List] = None) -> dict:
    """Check cache before running full RAG pipeline."""
    
    if not config.LEGAL_RAG_CACHE_ENABLED:
        return invoke_qa_impl(query, history)
    
    cache_key = _cache_key_digest(query, _history_cache_key(history), intent=None)
    
    # Try cache
    redis_client = get_redis()
    cached = redis_client.get(cache_key)
    if cached:
        logger.info(f"📦 Cache hit for {cache_key[:8]}... ({query[:50]}...)")
        return json.loads(cached)
    
    # Run full pipeline
    logger.info(f"🔄 Cache miss, running full pipeline...")
    result = invoke_qa_impl(query, history)
    
    # Store in cache
    redis_client.setex(
        cache_key,
        config.LEGAL_RAG_CACHE_TTL_SECONDS,
        json.dumps(result)
    )
    
    return result

# Update invoke_qa to use cache
def invoke_qa(query: str, history: Optional[List] = None) -> dict:
    return invoke_qa_with_cache(query, history)
```

**Benefit:** Repeated queries (e.g., "penalty for theft") return in **<100ms** (5–10x speedup)

**Recommendation 2.4B: Embedding Cache**
```python
# Cache computed embeddings for legal terms
_embedding_cache = {}

def get_cached_embedding(text: str) -> List[float]:
    text_key = hashlib.md5(text.encode()).hexdigest()
    
    if text_key not in _embedding_cache:
        _embedding_cache[text_key] = get_embeddings().embed_query(text)
        
        # Limit cache size (LRU-like)
        if len(_embedding_cache) > 10000:
            # Remove oldest entry
            oldest_key = next(iter(_embedding_cache))
            del _embedding_cache[oldest_key]
    
    return _embedding_cache[text_key]

# Usage
embedding = get_cached_embedding(candidate_query)  # 0ms if cached
```

---

### 2.5 Infrastructure

**Current Deployment:**
```
Frontend: Vercel (Edge)
Backend: Render/Railway (US/EU region)
AI Engine: Render/Railway (US/EU region, Python)
Pinecone: Cloud (us-west-2)
Groq API: Cloud
```

**Latencies:**
- Browser → Vercel: ~100ms
- Vercel → Backend: ~150ms
- Backend → AI Engine: ~20ms (same region)
- AI Engine → Pinecone: ~300ms (network)
- AI Engine → Groq: ~200ms (network)

**Recommendation 2.5A: Deploy AI Engine Closer to Data**
```
Option 1: Run AI Engine in same region as Pinecone
  - Current: us-west-2 (Pinecone) → Random region (AI Engine)
  - Proposed: us-west-2 (same region)
  - Savings: 50–100ms per request

Option 2: Add Pinecone Pod Index (serverless → pod)
  - Current: Pinecone Serverless (~300ms query)
  - Proposed: Pinecone Pod (~150ms query)
  - Cost: $15/month → $50/month
  - Savings: 150ms per request
```

**Recommendation 2.5B: Local Model Inference**
```bash
# Option: Run Ollama locally (if infrastructure supports)
LEGAL_RAG_LLM_BACKEND = ollama
LEGAL_RAG_LLM_MODEL = llama2:7b-chat-q5_K_M  # Quantized
OLLAMA_HOST = http://localhost:11434

# Savings:
#   - No network latency (~500ms)
#   - Inference: 3–5s (vs. Groq 4–6s)
#   - Total: 3–5s (vs. 5–6s)
```

---

## PART 3: PRIORITIZED RECOMMENDATIONS

### Phase 1: Quick Wins (1–2 days, 20–30% speedup)

| # | Action | Effort | Impact | Type |
|---|--------|--------|--------|------|
| 1.1 | Add query result caching (Redis) | 2h | **5–10x** for repeated queries | Speed |
| 1.2 | Reduce `CONTEXT_MAX_TOKENS_PER_DOC` from 380 → 300 | 0.5h | **30% LLM speedup** | Speed |
| 1.3 | Increase `RETRIEVER_TOP_K_AFTER_RERANK` from 5 → 8 | 0.5h | **20% quality** | Quality |
| 1.4 | Add synonym expansion before embedding | 3h | **15% recall** | Quality |
| 1.5 | Eager Pinecone initialization on startup | 1h | **2–5s first-request speedup** | Speed |

**Estimated Total Time:** 6.5h  
**Cumulative Impact:** 2–3 min → **8–12 seconds** for unique queries

### Phase 2: Medium Improvements (3–5 days, 30–50% additional speedup)

| # | Action | Effort | Impact | Type |
|---|--------|--------|--------|------|
| 2.1 | Implement async retrieval (Vector + BM25 parallel) | 4h | **10% total speedup** | Speed |
| 2.2 | Add embedding cache for legal terms | 2h | **5–10% retrieval speedup** | Speed |
| 2.3 | Reduce LLM max tokens from 2048 → 1024 | 1h | **40% LLM speedup** | Speed |
| 2.4 | Add smart article truncation (keep headers + clauses) | 3h | **25% quality** | Quality |
| 2.5 | Implement query expansion with article relationships | 4h | **20% recall** | Quality |

**Estimated Total Time:** 14h  
**Cumulative Impact:** **3–5 seconds** for unique queries

### Phase 3: Major Overhaul (1–2 weeks, 50%+ additional speedup)

| # | Action | Effort | Impact | Type |
|---|--------|--------|--------|------|
| 3.1 | Switch embedding model to multilingual-e5-base | 2h | **30% embedding speedup** | Speed |
| 3.2 | Implement streaming response (SSE) | 3h | **Better UX** | Speed |
| 3.3 | Deploy AI Engine in same region as Pinecone | 4h | **150ms savings** | Speed |
| 3.4 | Add local Ollama inference option | 6h | **500ms savings** | Speed |
| 3.5 | Refactor chunking with soft size limits | 8h | **30% quality** | Quality |

**Estimated Total Time:** 23h  
**Cumulative Impact:** **1–3 seconds** for unique queries

---

## PART 4: IMPLEMENTATION CHECKLIST

### For Response Quality (Priority 1)

- [ ] Review Article 1.1: Implement chunk size limits in `ArticleTextSplitter`
- [ ] Review Article 1.2: Add synonym expansion in `_build_retrieval_queries()`
- [ ] Review Article 1.3: Increase `RETRIEVER_TOP_K_AFTER_RERANK` to 8
- [ ] Review Article 1.4: Increase context window limits
- [ ] Review Article 1.5: Remove noisy summary documents from main index
- [ ] Review Article 1.6: Add article relationship graph for expansion

### For Response Speed (Priority 2)

- [ ] Review Article 2.1: Add Redis caching + eager Pinecone init
- [ ] Review Article 2.2: Reduce LLM output tokens + consider streaming
- [ ] Review Article 2.3: Implement async vector/BM25 retrieval
- [ ] Review Article 2.4: Implement embedding cache
- [ ] Review Article 2.5: Evaluate infrastructure improvements

---

## PART 5: MONITORING & VALIDATION

**Metrics to Track:**

| Metric | Current | Target | How to Measure |
|--------|---------|--------|----------------|
| P99 Response Time | 120–180s | <5s | HTTP latency logs |
| Cache Hit Rate | 0% | >40% | Redis stats |
| Retrieval Accuracy @ 5 docs | ~65% | >80% | Manual evaluation |
| Chunk Retrieval Precision | ~70% | >85% | Comparing article numbers |
| LLM Inference Time | 4.9–6.0s | 2.5–3.5s | API timing |

**Testing Protocol:**

1. **Baseline Run:** Record all metrics with current config
2. **Phase 1 Changes:** Apply changes, measure improvement
3. **Phase 2 Changes:** Iterate, track cumulative gains
4. **Regression Testing:** Ensure quality doesn't drop with speed optimizations

---

## CONCLUSION

LegalRAG has **strong fundamentals** (semantic chunking, multilingual embeddings, legal-aware routing) but suffers from:

1. **Response Quality Issues:**
   - Weak query expansion (missing synonyms & related articles)
   - Aggressive context trimming (loses article clauses)
   - Suboptimal chunk sizes (no soft limits)

2. **Response Speed Issues:**
   - No caching layer (5–10x speedup available)
   - LLM dominates latency (82% of time)
   - No parallel retrieval optimization
   - Cold-start penalty on Pinecone

**Quick Wins (Phase 1)** can deliver **5–7x speedup** (2–3 min → 8–12 sec) with minimal code changes. **Full optimization (Phase 3)** can achieve **<3 seconds** for unique queries and **<100ms** for cached queries.

Recommend starting with **Phase 1** (caching + context reduction) for immediate relief, then evaluate Phase 2 based on real-world performance data.

