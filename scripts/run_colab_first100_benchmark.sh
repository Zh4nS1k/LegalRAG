#!/usr/bin/env bash
set -euo pipefail

if [[ -z "${PINECONE_API_KEY:-}" ]]; then
  echo "PINECONE_API_KEY is required"
  exit 1
fi

ROOT_DIR="${ROOT_DIR:-/content/LegalRAG}"
XLSX_PATH="${XLSX_PATH:-$ROOT_DIR/tests/benchmarks/642_questions_with_citations.xlsx}"
RAW_OUTPUT="${RAW_OUTPUT:-$ROOT_DIR/benchmark_results/colab_first100_reranker_quality.json}"
CANONICAL_OUTPUT="${CANONICAL_OUTPUT:-$ROOT_DIR/benchmark_results/colab_first100_reranker_quality_canonical.json}"
LIMIT="${LIMIT:-100}"
TOP_K="${TOP_K:-10}"

export PINECONE_INDEX_NAME="${PINECONE_INDEX_NAME:-legally-01-index}"
export PINECONE_NAMESPACE="${PINECONE_NAMESPACE:-default}"
export GROQ_API_KEY="${GROQ_API_KEY:-dummy}"
export LEGAL_RAG_USE_RERANKER="${LEGAL_RAG_USE_RERANKER:-1}"
export LEGAL_RAG_RERANKER_MODEL="${LEGAL_RAG_RERANKER_MODEL:-BAAI/bge-reranker-v2-m3}"
export LEGAL_RAG_RERANKER_FALLBACK_MODEL="${LEGAL_RAG_RERANKER_FALLBACK_MODEL:-BAAI/bge-reranker-base}"
export LEGAL_RAG_RETRIEVER_WIDE_K="${LEGAL_RAG_RETRIEVER_WIDE_K:-50}"
export LEGAL_RAG_RETRIEVER_TOP_K_AFTER_RERANK="${LEGAL_RAG_RETRIEVER_TOP_K_AFTER_RERANK:-5}"
export LEGAL_RAG_RETRIEVER_MULTI_QUERY_LIMIT="${LEGAL_RAG_RETRIEVER_MULTI_QUERY_LIMIT:-4}"

PYTHONPATH="$ROOT_DIR" python -u "$ROOT_DIR/ai_service/utils/gold_citations_retrieval_benchmark.py" \
  --xlsx "$XLSX_PATH" \
  --limit "$LIMIT" \
  --top-k "$TOP_K" \
  --output "$RAW_OUTPUT"

PYTHONPATH="$ROOT_DIR" python -u "$ROOT_DIR/ai_service/utils/recompute_gold_benchmark_with_canonical_codes.py" \
  --input "$RAW_OUTPUT" \
  --output "$CANONICAL_OUTPUT"

python - <<PY
import json
for path in ["$RAW_OUTPUT", "$CANONICAL_OUTPUT"]:
    with open(path, "r", encoding="utf-8") as f:
        payload = json.load(f)
    print(path)
    print(json.dumps(payload["summary"], ensure_ascii=False, indent=2))
    print()
PY
