#!/usr/bin/env python3
"""
Download and cache Hugging Face models into .models_cache before starting the server.
Run once manually (with network access) before starting the ai_service.

Usage:
  python -m ai_service.scripts.download_models
  # or from repo root:
  cd ai_service && python -m scripts.download_models

Models downloaded:
  - intfloat/multilingual-e5-large (embeddings)
  - BAAI/bge-reranker-v2-m3 (reranker, optional)

Environment:
  - LEGAL_RAG_HF_CACHE_DIR: Override cache dir (default: <repo_root>/.models_cache)
  - HF_TOKEN: Optional token for gated models
"""

import os
import sys
from pathlib import Path

# Resolve paths before any HF import
_REPO_ROOT = Path(__file__).resolve().parent.parent.parent
_CACHE_DIR = os.environ.get("LEGAL_RAG_HF_CACHE_DIR", "").strip() or str(
    _REPO_ROOT / ".models_cache"
)


def _setup_cache_env() -> None:
    """Set HF/Transformers cache env vars to project-local .models_cache."""
    os.environ["HF_HOME"] = _CACHE_DIR
    os.environ["HF_HUB_CACHE"] = _CACHE_DIR
    os.environ["TRANSFORMERS_CACHE"] = _CACHE_DIR
    os.environ["SENTENCE_TRANSFORMERS_HOME"] = _CACHE_DIR


def main() -> int:
    _setup_cache_env()
    os.makedirs(_CACHE_DIR, exist_ok=True)
    print(f"[download_models] Cache directory: {_CACHE_DIR}")

    # 1. Embedding model (intfloat/multilingual-e5-large)
    embedding_model = os.environ.get(
        "LEGAL_RAG_EMBEDDING", "intfloat/multilingual-e5-large"
    )
    print(f"[download_models] Downloading embedding model: {embedding_model}")
    try:
        from sentence_transformers import SentenceTransformer

        SentenceTransformer(embedding_model)
        print(f"[download_models] OK: {embedding_model}")
    except Exception as e:
        print(f"[download_models] FAILED: {embedding_model}: {e}", file=sys.stderr)
        return 1

    # 2. Reranker (BAAI/bge-reranker-v2-m3) — used by agentic workflow
    reranker_model = os.environ.get(
        "LEGAL_RAG_RERANKER_MODEL", "BAAI/bge-reranker-v2-m3"
    )
    print(f"[download_models] Downloading reranker model: {reranker_model}")
    try:
        from FlagEmbedding import FlagReranker

        FlagReranker(reranker_model, use_fp16=True)
        print(f"[download_models] OK: {reranker_model}")
    except ImportError:
        print(
            f"[download_models] SKIP: FlagEmbedding not installed, reranker not cached"
        )
    except Exception as e:
        print(f"[download_models] FAILED: {reranker_model}: {e}", file=sys.stderr)
        return 1

    print(
        "[download_models] All models cached. Start server with LEGAL_RAG_HF_LOCAL_ONLY=1 for offline mode."
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
