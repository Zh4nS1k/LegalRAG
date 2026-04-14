# Colab Benchmark

Use this in Google Colab after cloning the repo and installing dependencies.

Run:

```bash
!chmod +x /content/LegalRAG/scripts/run_colab_first100_benchmark.sh
!PINECONE_API_KEY="YOUR_KEY" ROOT_DIR="/content/LegalRAG" /content/LegalRAG/scripts/run_colab_first100_benchmark.sh
```

Outputs:

- `/content/LegalRAG/benchmark_results/colab_first100_reranker_quality.json`
- `/content/LegalRAG/benchmark_results/colab_first100_reranker_quality_canonical.json`

Optional overrides:

```bash
!PINECONE_API_KEY="YOUR_KEY" \
ROOT_DIR="/content/LegalRAG" \
LIMIT=100 \
TOP_K=10 \
PINECONE_INDEX_NAME="legally-01-index" \
PINECONE_NAMESPACE="default" \
LEGAL_RAG_RETRIEVER_WIDE_K=50 \
LEGAL_RAG_RETRIEVER_TOP_K_AFTER_RERANK=5 \
/content/LegalRAG/scripts/run_colab_first100_benchmark.sh
```
