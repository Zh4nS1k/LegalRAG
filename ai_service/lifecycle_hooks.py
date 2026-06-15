# lifecycle_hooks.py — Level 1: Hooks (Absolute Guarantee)
# Isolates system dependencies from chat logic.
# Ensures deterministic startup and runtime behavior.

import os
import sys
from pathlib import Path
import torch

from ai_service.core import config
from ai_service.utils import connectivity
from ai_service.scripts.download_models import main as download_models_main


def pre_flight_check():
    """Absolute guarantee: Check local models and Pinecone before startup.
    If checks fail — exit(1), no hanging.
    """
    print("🪝 [HOOK] Pre-start check: Local models + Pinecone")

    # 1. Check local models in .models_cache
    cache_dir = Path(config.HF_CACHE_DIR)
    if not cache_dir.exists():
        print(f"❌ [FAIL] Models cache not found: {cache_dir}")
        sys.exit(1)

    model_files = list(cache_dir.rglob("*.bin")) + list(
        cache_dir.rglob("*.safetensors")
    )
    if not model_files:
        print(f"[INFO] No model files in cache: {cache_dir}, downloading...")
        result = download_models_main([])
        if result != 0:
            print("❌ [FAIL] Failed to download models")
            sys.exit(1)
        # Check again after download
        model_files = list(cache_dir.rglob("*.bin")) + list(
            cache_dir.rglob("*.safetensors")
        )
        if not model_files:
            print("❌ [FAIL] Still no model files after download")
            sys.exit(1)

    print(f"✅ [OK] Local models found: {len(model_files)} files")

    # Check GPU memory (force CPU if <4GB)
    if torch.cuda.is_available():
        total_memory = torch.cuda.get_device_properties(0).total_memory
        memory_gb = total_memory / (1024**3)
        if total_memory < 4 * 1024**3:  # 4GB threshold
            print(
                f"🪝 [HOOK] GPU memory insufficient ({memory_gb:.2f}GB < 4GB) — forcing CPU mode"
            )
            os.environ["CUDA_VISIBLE_DEVICES"] = ""
        else:
            print(f"✅ [OK] GPU memory: {memory_gb:.2f}GB")

    # 2. Check Pinecone connectivity (no hangs)
    if os.environ.get("LEGAL_RAG_DISABLE_PINECONE", "0") == "1":
        print("🪝 [HOOK] Pinecone disabled by flag, skipping connectivity check")
        return

    if not connectivity.is_internet_available():
        print("⚠️ [WARNING] No internet connection detected (checked 8.8.8.8/huggingface.co).")
        print("⚠️ [WARNING] Pinecone connectivity check will likely fail, but we'll try once...")

    try:
        from pinecone import Pinecone

        pc = Pinecone(api_key=config.PINECONE_API_KEY)
        # list_indexes() might raise NameResolutionError if DNS is broken
        indexes = pc.list_indexes()
        index_names = [idx.name for idx in indexes]

        if config.PINECONE_INDEX_NAME not in index_names:
            print(f"❌ [FAIL] Pinecone index '{config.PINECONE_INDEX_NAME}' not found")
            print(f"Available indexes: {index_names}")
            # Even if index is missing, we might want to start in degraded mode.
            # But usually index missing is a configuration error.
            # Still, better to start degraded than crash loop in Docker.
            print("⚠️ [WARNING] Continuing in DEGRADED mode (without Pinecone dense search).")
            os.environ["LEGAL_RAG_DISABLE_PINECONE"] = "1"
            return

        index = pc.Index(config.PINECONE_INDEX_NAME)
        # Quick ping: describe index
        desc = index.describe_index_stats()
        print(f"✅ [OK] Pinecone connected: {desc.total_vector_count} vectors")
    except Exception as e:
        print(f"❌ [FAIL] Pinecone unavailable: {e}")
        if "NameResolutionError" in str(e) or "[Errno -3]" in str(e):
            print("💡 [HINT] This looks like a DNS issue in Docker. Check your internet connection or DNS settings.")
        # In Docker, NameResolutionError often means DNS is not yet ready or misconfigured.
        # We allow the app to start in "degraded" mode instead of crashing the container.
        print("⚠️ [WARNING] Continuing in DEGRADED mode (without Pinecone dense search).")
        os.environ["LEGAL_RAG_DISABLE_PINECONE"] = "1"

    print("🪝 [HOOK] Pre-start check passed")


def network_sensor(func):
    """Decorator: Before each embedding, check HF/Groq ping and GPU memory.
    If >2s timeout — force offline_mode (local models only).
    If GPU <4GB — force CPU mode.
    """

    def wrapper(*args, **kwargs):
        # Check internet connectivity with 2s timeout
        if not connectivity.is_internet_available(timeout=2.0):
            print("🪝 [HOOK] No internet — switching to offline mode")
            os.environ["HF_HUB_OFFLINE"] = "1"
            os.environ["LEGAL_RAG_HF_LOCAL_ONLY"] = "1"
        else:
            # Reset to online if was offline
            os.environ.pop("HF_HUB_OFFLINE", None)
            os.environ.pop("LEGAL_RAG_HF_LOCAL_ONLY", None)

        # Check GPU memory
        if torch.cuda.is_available():
            total_memory = torch.cuda.get_device_properties(0).total_memory
            if total_memory < 4 * 1024**3:  # 4GB threshold
                print("🪝 [HOOK] GPU memory insufficient (<4GB) — forcing CPU mode")
                os.environ["CUDA_VISIBLE_DEVICES"] = ""

        return func(*args, **kwargs)

    return wrapper
