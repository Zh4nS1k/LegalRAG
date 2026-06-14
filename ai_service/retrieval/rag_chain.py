# rag_chain.py — Pinecone + BM25, reranker, строгий промпт

import logging
import os
import re
import sys
import threading
import concurrent.futures
import time
import json
import hashlib
from functools import lru_cache
from pathlib import Path
from typing import Any, List, Optional, Sequence

import torch
from langchain_core.callbacks import CallbackManagerForRetrieverRun, Callbacks
from langchain_core.documents import Document
from langchain_core.prompts import PromptTemplate
from langchain_core.retrievers import BaseRetriever
from pydantic import Field

from ai_service.core import config
from ai_service.retrieval.core import MinimalLegalRetriever
from ai_service.utils.circuit_breaker import CircuitBreaker, CircuitBreakerProxy
from ai_service.lifecycle_hooks import network_sensor
from ai_service.retrieval.domain import detect_domain, domain_matches_code
from ai_service.retrieval.query_rewrite import rewrite_query
from ai_service.utils import latency
from ai_service.utils.connectivity import is_cache_populated, is_internet_available

logger = logging.getLogger("ai_service.rag")

_nltk_ready = False
_stemmer = None


def _looks_like_raw_code_name(value: str) -> bool:
    text = str(value or "").strip().lower()
    return bool(text) and bool(re.fullmatch(r"[a-z0-9_]+", text)) and "_" in text


def _is_noisy_legal_chunk(doc: Document) -> bool:
    meta = doc.metadata or {}
    article_number = _normalize_article_number(meta.get("article_number"))
    code_ru = str(meta.get("code_ru") or "").strip()
    clause_level = str(meta.get("clause_level") or "").strip().lower()
    content_head = (doc.page_content or "")[:400].lower()

    if not article_number and _looks_like_raw_code_name(code_ru):
        return True
    if clause_level == "article" and not article_number and not (meta.get("article_title") or "").strip():
        return True
    if any(marker in content_head for marker in ("мазмұны", "содержание", "зқаи-ның ескертпесі", "пользователей назарына")):
        return True
    return False


def _ensure_nltk() -> bool:
    """Lazy NLTK setup (avoid downloading at import/startup)."""
    global _nltk_ready, _stemmer
    if _nltk_ready:
        return True
    try:
        import nltk  # local import: expensive
        from nltk.stem import SnowballStemmer

        # Download required NLTK data quietly if not present
        try:
            nltk.data.find("tokenizers/punkt")
        except LookupError:
            nltk.download("punkt", quiet=True)
        try:
            nltk.data.find("tokenizers/punkt_tab")
        except LookupError:
            nltk.download("punkt_tab", quiet=True)

        _stemmer = SnowballStemmer("russian")
        _nltk_ready = True
        return True
    except Exception:
        return False


def bm25_preprocess_func(text: str) -> List[str] | None:
    if not _ensure_nltk():
        return None
    import nltk

    tokens = nltk.word_tokenize((text or "").lower())
    return [_stemmer.stem(t) for t in tokens if t.isalnum()]


_init_lock = threading.RLock()
_embeddings_instance = None
_vector_store_instance = None
_retriever_instance = None
_llm_instance = None
_disable_pinecone = os.environ.get("LEGAL_RAG_DISABLE_PINECONE", "0") == "1"
_context_tokenizer = None
_groq_breaker = CircuitBreaker(
    "groq",
    failure_threshold=int(os.environ.get("LEGAL_RAG_GROQ_BREAKER_THRESHOLD", "5")),
    reset_timeout=int(os.environ.get("LEGAL_RAG_GROQ_BREAKER_TIMEOUT", "60")),
)
_pinecone_breaker = CircuitBreaker(
    "pinecone",
    failure_threshold=int(os.environ.get("LEGAL_RAG_PINECONE_BREAKER_THRESHOLD", "3")),
    reset_timeout=int(os.environ.get("LEGAL_RAG_PINECONE_BREAKER_TIMEOUT", "30")),
)


def _is_connection_failure(exc: Exception) -> bool:
    name = exc.__class__.__name__
    return isinstance(exc, ConnectionError) or name.endswith("ConnectionError") or name.endswith("ConnectError")


def _similarity_search_with_timeout(
    store: Any,
    query: str,
    *,
    k: int,
    filter: dict[str, Any],
    timeout_sec: float,
):
    executor = concurrent.futures.ThreadPoolExecutor(max_workers=1)
    future = executor.submit(store.similarity_search, query, k=k, filter=filter)
    try:
        return future.result(timeout=timeout_sec)
    finally:
        executor.shutdown(wait=False, cancel_futures=True)


class PrefixedEmbeddings:
    def __init__(self, embeddings):
        self.embeddings = embeddings

    def embed_documents(self, texts):
        return self.embeddings.embed_documents(["passage: " + t for t in texts])

    def embed_query(self, text):
        return self.embeddings.embed_query("query: " + text)


def _get_context_tokenizer():
    global _context_tokenizer
    if _context_tokenizer is not None:
        return _context_tokenizer
    with _init_lock:
        if _context_tokenizer is not None:
            return _context_tokenizer
        try:
            from transformers import AutoTokenizer

            _context_tokenizer = AutoTokenizer.from_pretrained(
                getattr(config, "CONTEXT_TOKENIZER_MODEL", ""),
                local_files_only=True,
            )
        except Exception:
            _context_tokenizer = False
    return _context_tokenizer


def reset_instances() -> None:
    global _embeddings_instance, _vector_store_instance, _retriever_instance, _llm_instance, _context_tokenizer
    with _init_lock:
        _embeddings_instance = None
        _vector_store_instance = None
        _retriever_instance = None
        _llm_instance = None
        _context_tokenizer = None
        if hasattr(_ensure_latency_patches, "_done"):
            _ensure_latency_patches._done = False
    clear_qa_cache()


def get_breaker_states() -> dict[str, str]:
    return {
        "groq": _groq_breaker.state,
        "pinecone": _pinecone_breaker.state,
    }


def _resolve_torch_dtype(value: str):
    normalized = str(value or "auto").strip().lower()
    if normalized in {"", "auto"}:
        return "auto"
    mapping = {
        "float16": torch.float16,
        "fp16": torch.float16,
        "bfloat16": torch.bfloat16,
        "bf16": torch.bfloat16,
        "float32": torch.float32,
        "fp32": torch.float32,
    }
    if normalized not in mapping:
        raise ValueError(f"Unsupported LEGAL_RAG_HF_TORCH_DTYPE: {value}")
    return mapping[normalized]


def _resolve_local_hf_snapshot(model_name_or_path: str) -> str:
    candidate = Path(str(model_name_or_path or "").strip())
    if candidate.exists():
        return str(candidate)

    model_id = str(model_name_or_path or "").strip()
    if not model_id or "/" not in model_id:
        return model_id

    org, name = model_id.split("/", 1)
    cache_root = Path(config.HF_CACHE_DIR)
    model_cache_dir = cache_root / f"models--{org}--{name}"
    snapshots_dir = model_cache_dir / "snapshots"
    refs_main = model_cache_dir / "refs" / "main"

    if refs_main.exists():
        snapshot_name = refs_main.read_text(encoding="utf-8").strip()
        snapshot_dir = snapshots_dir / snapshot_name
        if snapshot_dir.exists():
            return str(snapshot_dir)

    if snapshots_dir.exists():
        snapshot_dirs = sorted((p for p in snapshots_dir.iterdir() if p.is_dir()), key=lambda p: p.stat().st_mtime, reverse=True)
        if snapshot_dirs:
            return str(snapshot_dirs[0])

    return model_id


def _select_hf_runtime() -> str:
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def _make_hf_generation_pipeline():
    logger.info("🚀 [START] HF/PEFT Pipeline Initialization")
    t0 = time.perf_counter()
    config.configure_hf_hub()

    base_model = getattr(config, "HF_LLM_BASE_MODEL", "").strip()
    if not base_model:
        raise RuntimeError(
            "LEGAL_RAG_HF_BASE_MODEL is required when LEGAL_RAG_LLM_BACKEND=hf_peft"
        )

    adapter_path = getattr(config, "HF_LLM_ADAPTER_PATH", "").strip()
    local_only = config.HF_LOCAL_ONLY or not is_internet_available(timeout=2.0)
    model_source = _resolve_local_hf_snapshot(base_model) if local_only else base_model
    runtime_device = _select_hf_runtime()
    common_kwargs: dict[str, Any] = {
        "trust_remote_code": True,
        "cache_dir": config.HF_CACHE_DIR,
        "local_files_only": local_only,
    }
    if config.HF_TOKEN:
        common_kwargs["token"] = config.HF_TOKEN

    from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline

    tokenizer = AutoTokenizer.from_pretrained(model_source, **common_kwargs)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    model_kwargs: dict[str, Any] = {
        "trust_remote_code": True,
        "cache_dir": config.HF_CACHE_DIR,
        "local_files_only": local_only,
    }
    if config.HF_TOKEN:
        model_kwargs["token"] = config.HF_TOKEN

    configured_device_map = getattr(config, "HF_LLM_DEVICE_MAP", "auto")
    if runtime_device == "cuda":
        model_kwargs["device_map"] = configured_device_map
    elif configured_device_map not in {"", "auto"}:
        model_kwargs["device_map"] = configured_device_map

    torch_dtype = _resolve_torch_dtype(getattr(config, "HF_LLM_TORCH_DTYPE", "auto"))
    if torch_dtype != "auto":
        model_kwargs["dtype"] = torch_dtype

    load_in_4bit = getattr(config, "HF_LLM_LOAD_IN_4BIT", False)
    load_in_8bit = getattr(config, "HF_LLM_LOAD_IN_8BIT", False)
    if load_in_4bit and load_in_8bit:
        raise RuntimeError(
            "Choose only one quantization mode: LEGAL_RAG_HF_LOAD_IN_4BIT or LEGAL_RAG_HF_LOAD_IN_8BIT"
        )
    if load_in_4bit or load_in_8bit:
        try:
            from transformers import BitsAndBytesConfig
        except Exception as exc:  # pragma: no cover
            raise RuntimeError(
                "bitsandbytes quantization requested, but BitsAndBytesConfig is unavailable. "
                "Install bitsandbytes and compatible transformers."
            ) from exc
        quant_kwargs = {"load_in_4bit": load_in_4bit, "load_in_8bit": load_in_8bit}
        if load_in_4bit:
            quant_kwargs.update(
                {
                    "bnb_4bit_quant_type": "nf4",
                    "bnb_4bit_use_double_quant": True,
                    "bnb_4bit_compute_dtype": (
                        torch_dtype if torch_dtype != "auto" else torch.float16
                    ),
                }
            )
        model_kwargs["quantization_config"] = BitsAndBytesConfig(**quant_kwargs)

    model = AutoModelForCausalLM.from_pretrained(model_source, **model_kwargs)
    if runtime_device == "mps":
        model = model.to("mps")
    elif runtime_device == "cpu":
        model = model.to("cpu")

    if adapter_path:
        from peft import PeftModel

        model = PeftModel.from_pretrained(
            model,
            adapter_path,
            local_files_only=local_only,
            token=config.HF_TOKEN if config.HF_TOKEN else None,
        )

    text_generation_pipeline = pipeline(
        task="text-generation",
        model=model,
        tokenizer=tokenizer,
        max_new_tokens=config.LLM_MAX_TOKENS,
        temperature=config.LLM_TEMPERATURE,
        do_sample=getattr(config, "HF_LLM_DO_SAMPLE", False),
        top_p=getattr(config, "HF_LLM_TOP_P", 0.95),
        repetition_penalty=getattr(config, "HF_LLM_REPETITION_PENALTY", 1.05),
        return_full_text=False,
        pad_token_id=tokenizer.pad_token_id,
        eos_token_id=tokenizer.eos_token_id,
    )

    from langchain_huggingface import HuggingFacePipeline

    llm = HuggingFacePipeline(pipeline=text_generation_pipeline)
    elapsed = time.perf_counter() - t0
    logger.info(
        "✅ [SUCCESS] HF/PEFT Pipeline Initialization (base=%s, source=%s, adapter=%s) (%.2fs)",
        base_model,
        model_source,
        adapter_path or "<none>",
        elapsed,
    )
    return llm


def _estimate_token_count(text: str) -> int:
    text = str(text or "")
    if not text:
        return 0
    tokenizer = _get_context_tokenizer()
    if tokenizer:
        try:
            return len(tokenizer.encode(text, add_special_tokens=False))
        except Exception:
            pass
    # Fallback heuristic for Cyrillic/Kazakh legal text when no local tokenizer is cached.
    return max(1, (len(text) + 3) // 4)


def _truncate_text_to_token_budget(text: str, max_tokens: int, *, suffix: str) -> str:
    raw = str(text or "").strip()
    if not raw or max_tokens <= 0:
        return suffix.strip()
    if _estimate_token_count(raw) <= max_tokens:
        return raw

    left = 0
    right = len(raw)
    best = ""
    while left <= right:
        mid = (left + right) // 2
        candidate = raw[:mid].rstrip()
        candidate_with_suffix = candidate + suffix
        if _estimate_token_count(candidate_with_suffix) <= max_tokens:
            best = candidate_with_suffix
            left = mid + 1
        else:
            right = mid - 1
    return best or suffix.strip()


def _format_doc_for_prompt(doc: Document, content: str | None = None) -> str:
    meta = doc.metadata or {}
    code_ru = str(meta.get("code_ru") or "").strip() or "Неизвестный источник"
    article_number = str(meta.get("article_number") or "").strip() or "Н/Д"
    source = str(meta.get("source") or "").strip() or "Неизвестно"
    body = str(content if content is not None else doc.page_content or "").strip()
    return f"[{code_ru} | ст. {article_number} | {source}]\n{body}"


def _make_embeddings() -> PrefixedEmbeddings:
    """Hybrid Connectivity Hook: internet first, local fallback,
    fail-safe if neither."""
    logger.info("🚀 [START] Model Initialization (embeddings)")
    t0 = time.perf_counter()
    config.configure_hf_hub()
    logging.getLogger("transformers.modeling_utils").setLevel(logging.ERROR)
    if config.HF_TOKEN:
        os.environ["HF_TOKEN"] = config.HF_TOKEN

    # Deterministic path: project .models_cache (never /app or system-protected)
    cache_folder = config.HF_CACHE_DIR

    # Connectivity check with 2s timeout (no hangs)
    internet_ok = is_internet_available(timeout=2.0)
    cache_ok = is_cache_populated(cache_folder)

    # Fail-safe: no internet + empty cache → exit immediately
    if not internet_ok and not cache_ok:
        logger.error(
            "No local model found and no internet access. Cache: %s", cache_folder
        )
        sys.exit("No local model found and no internet access.")

    # Smart loader: internet → normal (allow cache updates);
    # no internet → local_files_only
    local_only = config.HF_LOCAL_ONLY or not internet_ok
    embedding_source = (
        _resolve_local_hf_snapshot(config.EMBEDDING_MODEL)
        if local_only
        else config.EMBEDDING_MODEL
    )
    model_kwargs: dict = {
        "local_files_only": local_only,
        "trust_remote_code": True,
    }

    # Device selection: force CPU if GPU disabled or insufficient
    device = (
        "cpu"
        if os.environ.get("CUDA_VISIBLE_DEVICES") == ""
        else ("cuda" if torch.cuda.is_available() else "cpu")
    )
    model_kwargs["device"] = device

    try:
        from langchain_huggingface import HuggingFaceEmbeddings

        emb = PrefixedEmbeddings(
            HuggingFaceEmbeddings(
                model_name=embedding_source,
                encode_kwargs={"normalize_embeddings": True},
                model_kwargs=model_kwargs,
                cache_folder=cache_folder,
                show_progress=False,
            )
        )
        elapsed = time.perf_counter() - t0
        mode = "local cache" if local_only else "internet"
        logger.info(
            "✅ [SUCCESS] Model Initialization (embeddings, %s) (%.2fs)", mode, elapsed
        )
        return emb
    except Exception as exc:
        elapsed = time.perf_counter() - t0
        logger.error(
            "❌ [FAIL] Model Initialization (embeddings) (%.2fs): %s",
            elapsed,
            exc,
            exc_info=True,
        )
        raise RuntimeError(
            "Не удалось загрузить эмбеддинги. Проверьте сеть или "
            "запустите: python -m ai_service.scripts.download_models. "
            "Кэш: %s" % cache_folder
        ) from exc


# Apply connectivity hook
_make_embeddings = network_sensor(_make_embeddings)


def get_embeddings() -> PrefixedEmbeddings:
    global _embeddings_instance
    if _embeddings_instance is None:
        with _init_lock:
            if _embeddings_instance is None:
                _embeddings_instance = _make_embeddings()
    return _embeddings_instance


def get_vector_store():
    if _disable_pinecone:
        raise RuntimeError("Pinecone disabled by LEGAL_RAG_DISABLE_PINECONE=1")
    global _vector_store_instance
    if _vector_store_instance is None:
        with _init_lock:
            if _vector_store_instance is None:
                logger.info("🚀 [START] Pinecone Vector Store Initialization")
                t0 = time.perf_counter()
                try:
                    from langchain_pinecone import PineconeVectorStore

                    _vector_store_instance = PineconeVectorStore(
                        index_name=config.PINECONE_INDEX_NAME,
                        embedding=get_embeddings(),
                        namespace=config.PINECONE_NAMESPACE or "default",
                        pinecone_api_key=config.PINECONE_API_KEY,
                    )
                    _vector_store_instance = CircuitBreakerProxy(
                        _vector_store_instance,
                        _pinecone_breaker,
                        sync_methods={
                            "similarity_search",
                            "similarity_search_with_score",
                            "similarity_search_by_vector",
                            "similarity_search_by_vector_with_relevance_scores",
                        },
                    )
                    elapsed = time.perf_counter() - t0
                    logger.info(
                        "✅ [SUCCESS] Pinecone Vector Store Initialization " "(%.2fs)",
                        elapsed,
                    )
                except Exception as e:
                    elapsed = time.perf_counter() - t0
                    logger.error(
                        "❌ [FAIL] Pinecone Vector Store Initialization " "(%.2fs): %s",
                        elapsed,
                        e,
                        exc_info=True,
                    )
                    if _is_connection_failure(e):
                        reset_instances()
                    raise
    return _vector_store_instance


_hybrid_k = getattr(config, "RETRIEVER_WIDE_K", getattr(config, "HYBRID_K", 8))
_vector_kwargs = {"k": _hybrid_k}

# Расширенные фильтры Pinecone:
# - по кодексу: ловит варианты названия (УК РК, Қылмыстық кодекс и т.д.)
# - по номеру статьи: например, 136 для кейса акушерки (баланы ауыстыру)
_filter_code = getattr(config, "RETRIEVER_FILTER_CODE_RU", None)
_filter_article = getattr(config, "RETRIEVER_FILTER_ARTICLE_NUMBER", None)
_uk_variants = [
    "Уголовный кодекс РК",
    "Уголовный кодекс Республики Казахстан",
    "Қылмыстық кодекс",
    "УК РК",
]
_gk_general_variants = [
    "Гражданский кодекс РК (Общая часть)",
    "Азаматтық кодекс (Жалпы бөлім)",
]
_gk_special_variants = [
    "Гражданский кодекс РК (Особенная часть)",
    "Азаматтық кодекс (Ерекше бөлім)",
]
_gk_variants = _gk_general_variants + _gk_special_variants
_koap_variants = [
    "Кодекс об административных правонарушениях РК",
    "Әкімшілік құқық бұзушылық туралы кодекс",
]
_gpk_variants = [
    "Гражданский процессуальный кодекс РК",
    "Азаматтық іс жүргізу кодексі",
]
_upk_variants = [
    "Уголовно-процессуальный кодекс РК",
    "Қылмыстық іс жүргізу кодексі",
]
_tk_variants = [
    "Трудовой кодекс РК",
    "Еңбек кодексі",
]
_nk_variants = [
    "Налоговый кодекс РК",
    "Салық кодексі",
]
_pk_variants = [
    "Предпринимательский кодекс РК",
    "Кәсіпкерлік кодекс",
]
_family_variants = [
    "Кодекс о браке и семье РК",
    "Неке және отбасы туралы кодекс",
]
_admin_proc_variants = [
    "Кодекс об административных процедурах РК",
    "Әкімшілік рәсімдер туралы кодекс",
]
_social_variants = [
    "Социальный кодекс РК",
    "Әлеуметтік кодекс",
]
_procurement_variants = [
    "Закон о государственных закупках РК",
    "Мемлекеттік сатып алу туралы заң",
]
_anti_corruption_variants = [
    "Закон о противодействии коррупции РК",
    "Коррупцияға қарсы күрес туралы заң",
]
_enforcement_variants = [
    "Закон об исполнительном производстве РК",
    "Орындау өндірісі туралы заң",
]
_personal_data_variants = [
    "Закон о персональных данных РК",
    "Жеке деректер туралы заң",
]
_ai_variants = [
    "Закон об искусственном интеллекте РК",
    "Жасанды интеллект туралы заң",
]
_consumer_variants = [
    "Закон о защите прав потребителей РК",
    "Тұтынушылардың құқықтарын қорғау туралы заң",
]
_housing_variants = [
    "Закон о жилищных отношениях РК",
    "Тұрғын үй қатынастары туралы заң",
]
_banks_variants = [
    "Закон о банках и банковской деятельности РК",
    "Банктер және банк қызметі туралы заң",
]
_land_variants = [
    "Земельный кодекс РК",
    "Жер кодексі",
]
_military_variants = [
    "Закон о воинской службе и статусе военнослужащих РК",
    "Әскери қызмет және әскери қызметшілердің мәртебесі туралы заң",
]
_llp_variants = [
    "Закон о товариществах с ограниченной и дополнительной ответственностью РК",
    "Жауапкершілігі шектеулі және қосымша жауапкершілігі бар серіктестіктер туралы заң",
]
_notariat_variants = [
    "Закон о нотариате РК",
    "Нотариат туралы заң",
]
_real_estate_registration_variants = [
    "Закон о государственной регистрации прав на недвижимое имущество РК",
    "Жылжымайтын мүлікке құқықтарды мемлекеттік тіркеу туралы заң",
]
_vehicle_insurance_variants = [
    "Закон об обязательном страховании гражданско-правовой ответственности владельцев транспортных средств РК",
    "Көлік құралдары иелерінің азаматтық-құқықтық жауапкершілігін міндетті сақтандыру туралы заң",
]
_education_variants = [
    "Закон об образовании РК",
    "Білім туралы заң",
]
_public_service_variants = [
    "Закон о государственной службе Республики Казахстан",
    "Қазақстан Республикасының мемлекеттік қызметі туралы заң",
]
_child_rights_variants = [
    "Закон о правах ребенка РК",
    "Баланың құқықтары туралы заң",
]
_advertising_variants = [
    "Закон о рекламе РК",
    "Жарнама туралы заң",
]
_collection_variants = [
    "Закон о коллекторской деятельности РК",
    "Коллекторлық қызмет туралы заң",
]
_road_traffic_variants = [
    "Закон о дорожном движении РК",
    "Жол жүрісі туралы заң",
]
_valuation_variants = [
    "Закон об оценочной деятельности в Республике Казахстан",
    "Қазақстан Республикасындағы бағалау қызметі туралы заң",
]
_legal_entities_registration_variants = [
    "Закон о государственной регистрации юридических лиц и учетной регистрации филиалов и представительств",
    "Заңды тұлғаларды мемлекеттік тіркеу және филиалдар мен өкілдіктерді есептік тіркеу туралы заң",
]
_currency_variants = [
    "Закон о валютном регулировании и валютном контроле",
    "Валюталық реттеу және валюталық бақылау туралы заң",
]
_digital_assets_variants = [
    "Закон о цифровых активах",
    "Цифрлық активтер туралы заң",
]
_personal_data_protection_variants = [
    "Закон о персональных данных и их защите",
    "Дербес деректер және оларды қорғау туралы заң",
]
_credit_bureaus_variants = [
    "Закон о кредитных бюро и формировании кредитных историй",
    "Кредиттік бюролар және кредиттік тарихты қалыптастыру туралы заң",
]
_microfinance_variants = [
    "Закон о микрофинансовой деятельности",
    "Микроқаржылық қызмет туралы заң",
]
_citizen_bankruptcy_variants = [
    "Закон о восстановлении платежеспособности и банкротстве граждан Республики Казахстан",
    "Қазақстан Республикасы азаматтарының төлем қабілеттілігін қалпына келтіру және банкроттығы туралы заң",
]
_rehabilitation_bankruptcy_variants = [
    "Закон о реабилитации и банкротстве",
    "Оңалту және банкроттық туралы заң",
]

_LAW_ALIAS_GROUPS: list[tuple[tuple[str, ...], list[str]]] = [
    (
        (
            "уголовный кодекс",
            "ук рк",
            "ук республики казахстан",
            "қылмыстық кодекс",
            "қр қк",
        ),
        _uk_variants,
    ),
    (
        (
            "гражданский кодекс",
            "гк рк",
            "гк",
            "азаматтық кодекс",
        ),
        _gk_variants,
    ),
    (
        (
            "гражданский кодекс общая часть",
            "гк общая часть",
            "общая часть гк",
            "жалпы бөлім",
        ),
        _gk_general_variants,
    ),
    (
        (
            "гражданский кодекс особенная часть",
            "гк особенная часть",
            "особенная часть гк",
            "ерекше бөлім",
        ),
        _gk_special_variants,
    ),
    (
        (
            "кодекс об административных правонарушениях",
            "коап рк",
            "коап",
            "әкімшілік құқық бұзушылық туралы кодекс",
        ),
        _koap_variants,
    ),
    (
        (
            "гражданский процессуальный кодекс",
            "гпк рк",
            "гпк",
            "азаматтық іс жүргізу кодексі",
        ),
        _gpk_variants,
    ),
    (
        (
            "уголовно-процессуальный кодекс",
            "упк рк",
            "упк",
            "қылмыстық іс жүргізу кодексі",
        ),
        _upk_variants,
    ),
    (
        (
            "трудовой кодекс",
            "тк рк",
            "тк",
            "еңбек кодексі",
        ),
        _tk_variants,
    ),
    (
        (
            "налоговый кодекс",
            "нк рк",
            "нк",
            "салық кодексі",
        ),
        _nk_variants,
    ),
    (
        (
            "предпринимательский кодекс",
            "пк рк",
            "кәсіпкерлік кодекс",
        ),
        _pk_variants,
    ),
    (
        (
            "о браке",
            "о семье",
            "кодекс о браке",
            "кодекс о браке (супружестве) и семье",
            "неке және отбасы туралы кодекс",
        ),
        _family_variants,
    ),
    (
        (
            "административного процедурно-процессуального кодекса",
            "административных процедурах",
            "аппк",
            "аппк рк",
            "әкімшілік рәсімдер туралы кодекс",
        ),
        _admin_proc_variants,
    ),
    (
        (
            "социальный кодекс",
            "әлеуметтік кодекс",
        ),
        _social_variants,
    ),
    (
        (
            "государственных закуп",
            "мемлекеттік сатып алу",
        ),
        _procurement_variants,
    ),
    (
        (
            "противодействии коррупции",
            "коррупцияға қарсы",
        ),
        _anti_corruption_variants,
    ),
    (
        (
            "исполнительном производстве",
            "орындау өндірісі",
        ),
        _enforcement_variants,
    ),
    (
        (
            "защите прав потребителей",
            "правах потребителей",
            "зпп",
            "тұтынушылардың құқықтарын қорғау",
        ),
        _consumer_variants,
    ),
    (
        (
            "жилищных отношениях",
            "жилищные отношения",
            "тұрғын үй қатынастары",
        ),
        _housing_variants,
    ),
    (
        (
            "банках и банковской деятельности",
            "банковской деятельности",
            "банк қызметі",
        ),
        _banks_variants,
    ),
    (
        (
            "земельный кодекс",
            "жер кодексі",
        ),
        _land_variants,
    ),
    (
        (
            "воинской службе",
            "статусе военнослужащих",
            "әскери қызмет",
        ),
        _military_variants,
    ),
    (
        (
            "товариществах с ограниченной и дополнительной ответственностью",
            "тоо",
            "т о о",
            "жауапкершілігі шектеулі",
        ),
        _llp_variants,
    ),
    (
        (
            "нотариате",
            "нотариат туралы",
        ),
        _notariat_variants,
    ),
    (
        (
            "государственной регистрации прав на недвижимое имущество",
            "жылжымайтын мүлікке құқықтарды мемлекеттік тіркеу",
        ),
        _real_estate_registration_variants,
    ),
    (
        (
            "страховании гражданско-правовой ответственности владельцев транспортных средств",
            "көлік құралдары иелерінің азаматтық-құқықтық жауапкершілігін міндетті сақтандыру",
        ),
        _vehicle_insurance_variants,
    ),
    (
        (
            "об образовании",
            "білім туралы",
        ),
        _education_variants,
    ),
    (
        (
            "государственной службе",
            "мемлекеттік қызметі туралы",
        ),
        _public_service_variants,
    ),
    (
        (
            "правах ребенка",
            "баланың құқықтары",
        ),
        _child_rights_variants,
    ),
    (
        (
            "рекламе",
            "жарнама туралы",
        ),
        _advertising_variants,
    ),
    (
        (
            "коллекторской деятельности",
            "коллекторлық қызмет",
        ),
        _collection_variants,
    ),
    (
        (
            "дорожном движении",
            "жол жүрісі",
        ),
        _road_traffic_variants,
    ),
    (
        (
            "оценочной деятельности",
            "бағалау қызметі",
        ),
        _valuation_variants,
    ),
    (
        (
            "государственной регистрации юридических лиц",
            "филиалов и представительств",
            "заңды тұлғаларды мемлекеттік тіркеу",
        ),
        _legal_entities_registration_variants,
    ),
    (
        (
            "валютном регулировании",
            "валютном контроле",
            "валюталық реттеу",
        ),
        _currency_variants,
    ),
    (
        (
            "цифровых активах",
            "цифрлық активтер",
        ),
        _digital_assets_variants,
    ),
    (
        (
            "персональных данных и их защите",
            "дербес деректер және оларды қорғау",
        ),
        _personal_data_protection_variants,
    ),
    (
        (
            "кредитных бюро",
            "кредиттік бюролар",
        ),
        _credit_bureaus_variants,
    ),
    (
        (
            "микрофинансовой деятельности",
            "микроқаржылық қызмет",
        ),
        _microfinance_variants,
    ),
    (
        (
            "банкротстве граждан",
            "восстановлении платежеспособности",
            "азаматтарының төлем қабілеттілігін қалпына келтіру",
        ),
        _citizen_bankruptcy_variants,
    ),
    (
        (
            "реабилитации и банкротстве",
            "оңалту және банкроттық",
        ),
        _rehabilitation_bankruptcy_variants,
    ),
    (
        (
            "персональных данных",
            "жеке деректер",
        ),
        _personal_data_variants,
    ),
    (
        (
            "искусственном интеллекте",
            "жасанды интеллект",
        ),
        _ai_variants,
    ),
]

_THEFT_QUERY_HINTS: tuple[str, ...] = (
    "краж",
    "украл",
    "украду",
    "украсть",
    "воров",
    "похит",
    "тайное хищение",
    "ұрлық",
    "ұрла",
    "жымқыру",
)

_LEGAL_CONCEPT_BUNDLES: tuple[dict[str, Any], ...] = (
    {
        "name": "theft",
        "patterns": _THEFT_QUERY_HINTS,
        "code_names": _uk_variants,
        "focus_articles": ("188",),
        "expansions_ru": (
            "кража 188 УК РК",
            "тайное хищение чужого имущества",
            "кража из жилища",
        ),
        "expansions_kz": (
            "ұрлық 188-бап ҚР ҚК",
            "бөтен мүлікті жасырын жымқыру",
            "тұрғын үйден ұрлау",
        ),
        "doc_terms": (
            "кража",
            "тайное хищение",
            "чужого имущества",
            "ұрлық",
            "жымқыру",
        ),
        "weight": 1.6,
        "criminal": True,
    },
    {
        "name": "fraud",
        "patterns": (
            "мошеннич",
            "алаяқ",
            "обман",
            "обманул",
            "обманным путем",
            "злоупотребление доверием",
            "афер",
        ),
        "code_names": _uk_variants,
        "focus_articles": ("190",),
        "expansions_ru": (
            "мошенничество 190 УК РК",
            "хищение путем обмана или злоупотребления доверием",
        ),
        "expansions_kz": (
            "алаяқтық 190-бап ҚР ҚК",
            "алдау немесе сенімді теріс пайдалану арқылы мүлікті иелену",
        ),
        "doc_terms": (
            "мошенничество",
            "обман",
            "злоупотребление доверием",
            "алаяқтық",
        ),
        "weight": 1.7,
        "criminal": True,
    },
    {
        "name": "violent_property",
        "patterns": (
            "грабеж",
            "грабил",
            "разбой",
            "вымог",
            "шантаж",
            "ограб",
            "тонау",
            "қарақшылық",
            "қорқытып алу",
        ),
        "code_names": _uk_variants,
        "focus_articles": (),
        "expansions_ru": (
            "насильственное хищение чужого имущества",
            "грабеж разбой вымогательство УК РК",
        ),
        "expansions_kz": (
            "бөтен мүлікті күш қолданып иелену",
            "тонау қарақшылық қорқытып алу ҚК",
        ),
        "doc_terms": (
            "грабеж",
            "разбой",
            "вымогательство",
            "тонау",
            "қарақшылық",
        ),
        "weight": 1.4,
        "criminal": True,
    },
    {
        "name": "consumer_return",
        "patterns": (
            "возврат товара",
            "некачественный товар",
            "дефект товара",
            "товарный вид",
            "продавец отказал",
            "каспи магазин",
            "бритва",
            "қайтару",
            "сапасыз тауар",
        ),
        "code_names": _consumer_variants,
        "focus_articles": (),
        "expansions_ru": (
            "защита прав потребителей возврат товара",
            "недостатки товара право потребителя",
        ),
        "expansions_kz": (
            "тұтынушылардың құқықтарын қорғау тауарды қайтару",
            "тауар кемшілігі тұтынушы құқығы",
        ),
        "doc_terms": (
            "защите прав потребителей",
            "возврат товара",
            "недостаток товара",
            "тұтынушылардың құқықтарын қорғау",
        ),
        "weight": 1.4,
        "criminal": False,
    },
    {
        "name": "family_support",
        "patterns": (
            "алимент",
            "развод",
            "расторжение брака",
            "бывш",
            "ребенок",
            "детей",
            "опек",
            "родительских прав",
            "неке",
            "отбасы",
            "қамқоршы",
        ),
        "code_names": _family_variants,
        "focus_articles": (),
        "expansions_ru": (
            "кодекс о браке и семье алименты развод опека",
            "родительские права ребенок содержание",
        ),
        "expansions_kz": (
            "неке отбасы кодексі алимент ажырасу қорғаншылық",
            "ата-ана құқықтары бала асырау",
        ),
        "doc_terms": (
            "алименты",
            "расторжение брака",
            "родительских прав",
            "опека",
            "неке",
            "отбасы",
        ),
        "weight": 1.4,
        "criminal": False,
    },
    {
        "name": "labor_employment",
        "patterns": (
            "увольн",
            "сокращен",
            "зарплат",
            "работодатель",
            "трудовой договор",
            "отпуск",
            "штат",
            "жұмыс",
            "еңбек",
            "жалақы",
        ),
        "code_names": _tk_variants,
        "focus_articles": (),
        "expansions_ru": (
            "трудовой кодекс увольнение сокращение заработная плата",
            "трудовые права работника обязанности работодателя",
        ),
        "expansions_kz": (
            "еңбек кодексі жұмыстан шығару қысқарту жалақы",
            "жұмыскердің құқықтары жұмыс берушінің міндеттері",
        ),
        "doc_terms": (
            "трудовой договор",
            "работодатель",
            "заработная плата",
            "еңбек",
            "жалақы",
        ),
        "weight": 1.4,
        "criminal": False,
    },
    {
        "name": "bank_credit",
        "patterns": (
            "кредит",
            "заем",
            "ипотек",
            "банк",
            "проценты",
            "просроч",
            "кредитор",
            "неси",
            "қарыз",
            "банкрот",
        ),
        "code_names": _banks_variants,
        "focus_articles": (),
        "expansions_ru": (
            "банковский заем кредитный договор просрочка банк",
            "проценты по кредиту реструктуризация задолженности",
        ),
        "expansions_kz": (
            "банктік қарыз кредит шарты мерзімін өткізу банк",
            "кредит бойынша пайыздар берешекті қайта құрылымдау",
        ),
        "doc_terms": (
            "банковский заем",
            "кредит",
            "ипотека",
            "банк",
            "несие",
        ),
        "weight": 1.25,
        "criminal": False,
    },
    {
        "name": "housing_real_estate",
        "patterns": (
            "квартир",
            "жилой дом",
            "дом оформлен",
            "недвижим",
            "собственност",
            "нотариус",
            "жилье",
            "пәтер",
            "үй",
        ),
        "code_names": _housing_variants + _real_estate_registration_variants + _gk_variants,
        "focus_articles": (),
        "expansions_ru": (
            "жилищные отношения право собственности недвижимое имущество",
            "государственная регистрация прав на недвижимое имущество",
        ),
        "expansions_kz": (
            "тұрғын үй қатынастары меншік құқығы жылжымайтын мүлік",
            "жылжымайтын мүлікке құқықтарды мемлекеттік тіркеу",
        ),
        "doc_terms": (
            "жилищные отношения",
            "недвижимое имущество",
            "право собственности",
            "тұрғын үй",
        ),
        "weight": 1.35,
        "criminal": False,
    },
    {
        "name": "tax_business",
        "patterns": (
            "налог",
            "ндс",
            "деклараци",
            "ип",
            "тоо",
            "предприним",
            "салық",
            "деклар",
            "кәсіпкер",
            "тіркеусіз",
        ),
        "code_names": _nk_variants + _llp_variants + _pk_variants,
        "focus_articles": (),
        "expansions_ru": (
            "налоговый кодекс декларация обязательства налогоплательщика",
            "тоо ип предпринимательская деятельность регистрация",
        ),
        "expansions_kz": (
            "салық кодексі декларация салық төлеушінің міндеттері",
            "тоо ип кәсіпкерлік қызмет тіркеу",
        ),
        "doc_terms": (
            "налоговый кодекс",
            "декларация",
            "налогоплательщик",
            "салық",
            "тоо",
        ),
        "weight": 1.45,
        "criminal": False,
    },
    {
        "name": "bankruptcy",
        "patterns": (
            "банкрот",
            "неплатежеспособ",
            "реабилитац",
            "восстановление платежеспособности",
            "төлем қабілетті",
            "оңалту",
            "дәрменсіз",
        ),
        "code_names": _citizen_bankruptcy_variants + _rehabilitation_bankruptcy_variants,
        "focus_articles": (),
        "expansions_ru": (
            "банкротство граждан восстановление платежеспособности",
            "реабилитация и банкротство должника",
        ),
        "expansions_kz": (
            "азаматтардың банкроттығы төлем қабілеттілігін қалпына келтіру",
            "оңалту және банкроттық борышкер",
        ),
        "doc_terms": (
            "банкротство",
            "восстановление платежеспособности",
            "реабилитация",
            "банкроттық",
        ),
        "weight": 1.5,
        "criminal": False,
    },
    {
        "name": "waste_sanitary",
        "patterns": (
            "мусор",
            "тбо",
            "антисанитар",
            "контейнер",
            "отходы",
            "санитарно",
            "қоқыс",
            "санитар",
        ),
        "code_names": _koap_variants,
        "focus_articles": (),
        "expansions_ru": (
            "нарушение санитарных требований отходы коап",
            "мусорная площадка твердые бытовые отходы",
        ),
        "expansions_kz": (
            "санитариялық талаптарды бұзу қалдықтар әкімшілік кодекс",
            "қоқыс алаңы тұрмыстық қатты қалдықтар",
        ),
        "doc_terms": (
            "санитар",
            "отходы",
            "тбо",
            "қоқыс",
        ),
        "weight": 1.3,
        "criminal": False,
    },
    {
        "name": "noise_silence",
        "patterns": (
            "тишин",
            "шум",
            "шумет",
            "шуметь",
            "громк",
            "сосед",
            "суббот",
            "воскресен",
            "выходн",
            "праздничн",
            "ноч",
            "покой",
            "квартира",
            "ремонт",
            "сверл",
            "перфорат",
            "22:00",
            "23:00",
            "09:00",
            "10:00",
            "22 30",
            "22:30",
            "23 30",
            "23:30",
            "тыныш",
            "шу",
            "демалыс",
            "сенбі",
            "жексенбі",
            "мереке",
        ),
        "code_names": _koap_variants,
        "focus_articles": ("437",),
        "expansions_ru": (
            "нарушение тишины коап 437 покой физических лиц",
            "тишина в будние выходные и праздничные дни",
            "шум в квартире ремонтные работы в жилом доме",
        ),
        "expansions_kz": (
            "тыныштықты бұзу әкімшілік кодекс 437 жеке тұлғалардың тыныштығы",
            "жұмыс күндері демалыс және мереке күндеріндегі тыныштық",
            "пәтердегі шу тұрғын үйдегі жөндеу жұмыстары",
        ),
        "doc_terms": (
            "нарушение тишины",
            "покой физических лиц",
            "выходные и праздничные дни",
            "шумом",
            "тыныштық",
            "тыныштығы",
        ),
        "weight": 1.75,
        "criminal": False,
    },
)
_allowed_code_ru_for_filter = None
_filter_clauses: list[dict] = []

if _filter_code:
    # Pinecone не поддерживает $regex — используем $or по известным вариантам
    _variants = list(
        dict.fromkeys([_filter_code] + [v for v in _uk_variants if v != _filter_code])
    )
    _allowed_code_ru_for_filter = _variants
    _filter_clauses.append({"$or": [{"code_ru": v} for v in _variants]})

if _filter_article:
    # Точный номер статьи хранится как строка в metadata["article_number"]
    _filter_clauses.append({"article_number": _filter_article})

if _filter_clauses:
    if len(_filter_clauses) == 1:
        _vector_kwargs["filter"] = _filter_clauses[0]
    else:
        _vector_kwargs["filter"] = {"$and": _filter_clauses}
    # При включённых фильтрах немного увеличиваем k для надёжности
    _vector_kwargs["k"] = min(
        getattr(config, "RETRIEVER_WIDE_K", getattr(config, "HYBRID_K", 8)) + 4, 30
    )
    print(f"Фильтр Pinecone search_kwargs: {_vector_kwargs.get('filter')}")


class LazyPineconeRetriever(BaseRetriever):
    search_kwargs: dict = Field(default_factory=dict)

    def _get_relevant_documents(
        self, query: str, *, run_manager=None
    ) -> List[Document]:
        # Diagnostic: log raw similarity scores
        vs = get_vector_store()
        docs_with_scores = vs.similarity_search_with_score(query, **self.search_kwargs)
        if docs_with_scores:
            scores = [score for _, score in docs_with_scores]
            logger.info(f"[DIAG] Top 5 similarity scores: {scores[:5]}")
            if scores and scores[0] < 0.6:
                logger.critical(
                    "[RETR_FAIL] Low similarity score detected: "
                    f"{scores[0]}. Check embedding model version."
                )
            # Log metadata of first 3 chunks
            for i, (doc, score) in enumerate(docs_with_scores[:3]):
                logger.info(
                    f"[DIAG] Chunk {i+1} metadata: "
                    f"code_ru={doc.metadata.get('code_ru')}, "
                    f"article={doc.metadata.get('article_number')}, "
                    f"score={score}"
                )
        docs = [doc for doc, _ in docs_with_scores]
        return docs


_vector_retriever = LazyPineconeRetriever(search_kwargs=_vector_kwargs)


def _get_target_code_profile(query: str) -> tuple[list[str], bool]:
    ranked = _score_target_codes(query)
    if not ranked:
        return [], False
    top_score = ranked[0][1]
    second_score = ranked[1][1] if len(ranked) > 1 else 0.0
    detected = _detect_target_codes(query)
    confident = top_score >= 1.8 and top_score >= second_score + 1.2
    return detected, confident


def _adaptive_wide_k(query: str, target_codes: list[str], target_articles: list[str]) -> int:
    base_k = int(getattr(config, "RETRIEVER_WIDE_K", getattr(config, "HYBRID_K", 24)))
    query_len = len(str(query or ""))
    if target_codes and target_articles:
        return max(12, min(base_k, 16))
    if target_codes:
        return max(14, min(base_k, 18))
    if query_len > 1200:
        return max(16, min(base_k, 20))
    if query_len > 600:
        return max(18, min(base_k, 22))
    return base_k


def _prioritize_docs(
    query: str,
    docs: Sequence[Document],
    *,
    target_codes: list[str] | None = None,
    target_articles: list[str] | None = None,
    limit: int | None = None,
) -> list[Document]:
    ranked = _rank_docs_with_legal_scoring(
        query,
        list(docs),
        target_codes=target_codes,
        target_articles=target_articles,
    )
    if limit is not None:
        return ranked[:limit]
    return ranked


def _normalize_score_series(scores: Sequence[float]) -> list[float]:
    values = [float(score) for score in scores]
    if not values:
        return []
    min_score = min(values)
    max_score = max(values)
    if max_score <= min_score:
        return [1.0 if score > 0 else 0.0 for score in values]
    spread = max_score - min_score
    return [(score - min_score) / spread for score in values]


def _rank_source_candidates(
    candidates: dict[tuple[str, str], dict[str, Any]],
    source: str,
) -> list[tuple[int, tuple[str, str], dict[str, Any]]]:
    scored: list[tuple[float, int, tuple[str, str], dict[str, Any]]] = []
    score_key = f"{source}_raw"
    for idx, (key, payload) in enumerate(candidates.items()):
        raw_score = float(payload.get(score_key) or 0.0)
        if raw_score <= 0:
            continue
        scored.append((raw_score, idx, key, payload))
    scored.sort(key=lambda item: (-item[0], item[1], item[2]))
    return [(rank + 1, key, payload) for rank, (_, _, key, payload) in enumerate(scored)]


def _rrf_contribution(rank: int, *, k: int) -> float:
    return 1.0 / float(max(k, 1) + rank)


def _candidate_entry(doc: Document, source: str, raw_score: float) -> dict[str, Any]:
    metadata = dict(doc.metadata or {})
    metadata.setdefault("fusion_sources", [])
    metadata["fusion_sources"] = [source]
    doc_copy = Document(page_content=doc.page_content, metadata=metadata)
    return {
        "doc": doc_copy,
        "vector_raw": raw_score if source == "vector" else 0.0,
        "bm25_raw": raw_score if source == "bm25" else 0.0,
    }


def _accumulate_candidate(
    candidates: dict[tuple[str, str], dict[str, Any]],
    doc: Document,
    source: str,
    raw_score: float,
) -> None:
    key = _article_doc_key(doc)
    entry = candidates.get(key)
    if entry is None:
        candidates[key] = _candidate_entry(doc, source, raw_score)
        return

    if source == "vector":
        entry["vector_raw"] = max(float(entry.get("vector_raw") or 0.0), raw_score)
    elif source == "bm25":
        entry["bm25_raw"] = max(float(entry.get("bm25_raw") or 0.0), raw_score)

    fusion_sources = entry["doc"].metadata.setdefault("fusion_sources", [])
    if source not in fusion_sources:
        fusion_sources.append(source)
    current_best = max(float(entry.get("vector_raw") or 0.0), float(entry.get("bm25_raw") or 0.0))
    if raw_score >= current_best:
        entry["doc"] = Document(page_content=doc.page_content, metadata=dict(doc.metadata or {}))


def _collect_vector_candidates(
    query: str,
    *,
    wide_k: int,
    target_codes: list[str],
    target_articles: list[str],
) -> dict[tuple[str, str], dict[str, Any]]:
    try:
        store = get_vector_store()
    except Exception:
        return {}

    candidates: dict[tuple[str, str], dict[str, Any]] = {}
    retrieval_queries = _build_retrieval_queries(query)
    filters: list[dict[str, Any] | None] = [None]
    if target_codes:
        filters = []
        for code in target_codes:
            if target_articles:
                for article in target_articles[:4]:
                    filters.append({"code_ru": code, "article_number": article})
            filters.append({"code_ru": code})

    for candidate_query in retrieval_queries:
        for search_filter in filters:
            try:
                docs_with_scores = store.similarity_search_with_score(
                    candidate_query,
                    k=wide_k,
                    filter=search_filter,
                )
            except Exception:
                continue
            for doc, score in docs_with_scores:
                _accumulate_candidate(candidates, doc, "vector", float(score))
    return candidates


def _collect_bm25_candidates(
    query: str,
    *,
    bm25_retriever: Any,
    wide_k: int,
) -> dict[tuple[str, str], dict[str, Any]]:
    if bm25_retriever is None:
        return {}

    candidates: dict[tuple[str, str], dict[str, Any]] = {}
    retrieval_queries = _build_retrieval_queries(query)
    for candidate_query in retrieval_queries:
        try:
            docs = list(bm25_retriever.invoke(candidate_query))
        except Exception:
            continue
        total = max(len(docs), 1)
        for rank, doc in enumerate(docs[:wide_k]):
            raw_score = 1.0 - (rank / total)
            _accumulate_candidate(candidates, doc, "bm25", float(raw_score))
    return candidates


def _fuse_retrieval_candidates(
    query: str,
    *,
    vector_candidates: dict[tuple[str, str], dict[str, Any]],
    bm25_candidates: dict[tuple[str, str], dict[str, Any]],
    target_codes: list[str],
    target_articles: list[str],
    limit: int,
) -> list[Document]:
    all_keys = list(dict.fromkeys([*vector_candidates.keys(), *bm25_candidates.keys()]))
    if not all_keys:
        return []

    fusion_method = str(getattr(config, "HYBRID_FUSION_METHOD", "rrf") or "rrf").strip().lower()
    target_article_set = {
        _normalize_article_number(article)
        for article in target_articles
        if _normalize_article_number(article)
    }
    vector_scores = _normalize_score_series(
        [float((vector_candidates.get(key) or {}).get("vector_raw") or 0.0) for key in all_keys]
    )
    bm25_scores = _normalize_score_series(
        [float((bm25_candidates.get(key) or {}).get("bm25_raw") or 0.0) for key in all_keys]
    )

    fused: list[tuple[float, int, Document]] = []
    if fusion_method == "rrf":
        rrf_k = int(getattr(config, "HYBRID_RRF_K", 60))
        score_scale = float(getattr(config, "HYBRID_RRF_SCORE_SCALE", 100.0))
        vector_ranked = _rank_source_candidates(vector_candidates, "vector")
        bm25_ranked = _rank_source_candidates(bm25_candidates, "bm25")
        ranked_lookup: dict[tuple[str, str], dict[str, Any]] = {}
        for key in all_keys:
            vector_payload = vector_candidates.get(key)
            bm25_payload = bm25_candidates.get(key)
            if vector_payload and bm25_payload:
                ranked_lookup[key] = (
                    vector_payload
                    if float(vector_payload.get("vector_raw") or 0.0)
                    >= float(bm25_payload.get("bm25_raw") or 0.0)
                    else bm25_payload
                )
            else:
                ranked_lookup[key] = vector_payload or bm25_payload
        rrf_scores: dict[tuple[str, str], float] = {}

        for rank, key, payload in vector_ranked:
            rrf_scores[key] = rrf_scores.get(key, 0.0) + (
                float(getattr(config, "VECTOR_WEIGHT", 0.6))
                * _rrf_contribution(rank, k=rrf_k)
            )
            payload["doc"].metadata["vector_rank"] = rank
        for rank, key, payload in bm25_ranked:
            rrf_scores[key] = rrf_scores.get(key, 0.0) + (
                float(getattr(config, "BM25_WEIGHT", 0.4))
                * _rrf_contribution(rank, k=rrf_k)
            )
            payload["doc"].metadata["bm25_rank"] = rank

        for idx, key in enumerate(all_keys):
            candidate = ranked_lookup.get(key)
            if not candidate:
                continue
            doc = candidate["doc"]
            vector_score = vector_scores[idx] if idx < len(vector_scores) else 0.0
            bm25_score = bm25_scores[idx] if idx < len(bm25_scores) else 0.0
            lexical = _lexical_overlap_score(query, doc)
            fusion_base = (rrf_scores.get(key, 0.0) * score_scale) + (0.20 * lexical)
            doc.metadata["vector_score"] = round(vector_score, 4)
            doc.metadata["bm25_score"] = round(bm25_score, 4)
            doc.metadata["rrf_score"] = round(rrf_scores.get(key, 0.0), 6)
            doc.metadata["fusion_base_score"] = round(fusion_base, 4)
            final_score = _apply_legal_score(
                query,
                doc,
                fusion_base,
                target_codes=target_codes,
                target_articles=target_article_set,
            )
            doc.metadata["relevance_score"] = final_score
            fused.append((final_score, idx, doc))
    else:
        for idx, key in enumerate(all_keys):
            candidate = vector_candidates.get(key) or bm25_candidates.get(key)
            if not candidate:
                continue
            doc = candidate["doc"]
            vector_score = vector_scores[idx] if idx < len(vector_scores) else 0.0
            bm25_score = bm25_scores[idx] if idx < len(bm25_scores) else 0.0
            lexical = _lexical_overlap_score(query, doc)
            # BM25 gets a slightly higher weight to keep exact legal terms dominant.
            fusion_base = (0.30 * vector_score) + (0.50 * bm25_score) + (0.20 * lexical)
            doc.metadata["vector_score"] = round(vector_score, 4)
            doc.metadata["bm25_score"] = round(bm25_score, 4)
            doc.metadata["fusion_base_score"] = round(fusion_base, 4)
            final_score = _apply_legal_score(
                query,
                doc,
                fusion_base,
                target_codes=target_codes,
                target_articles=target_article_set,
            )
            doc.metadata["relevance_score"] = final_score
            fused.append((final_score, idx, doc))

    fused.sort(key=lambda item: (-item[0], item[1]))
    ordered = [doc for _, _, doc in fused]
    ordered = _prioritize_docs(
        query,
        ordered,
        target_codes=target_codes,
        target_articles=target_articles,
        limit=limit,
    )
    return ordered


def _extract_article_range(query: str) -> tuple[int, int] | None:
    match = re.search(
        r"(?:статья|ст\.|ст|бап)?\s*(\d+)\s*[-–—]\s*(\d+)", query or "", re.IGNORECASE
    )
    if not match:
        return None
    start = int(match.group(1))
    end = int(match.group(2))
    return (start, end) if start <= end else (end, start)


def _normalized_query(query: str) -> str:
    q = (query or "").lower().replace("ё", "е")
    q = re.sub(r"\s+", " ", q)
    return q


def _has_alias(query: str, alias: str) -> bool:
    pattern = r"(?<!\w)" + re.escape(alias) + r"(?!\w)"
    return re.search(pattern, query) is not None


_LAW_ROUTE_HINTS: list[tuple[tuple[str, ...], list[str], float]] = [
    (
        (
            "возмездного оказания услуг",
            "договор оказания услуг",
            "представительство в суде",
            "юридическая помощь",
            "адвокатская деятельность",
            "поверенный",
            "договор поручения",
        ),
        _gk_special_variants,
        1.5,
    ),
    (
        (
            "валютном регулировании",
            "валютном контроле",
            "валюталық реттеу",
            "валюталық бақылау",
            "нерезидент",
            "резидент",
            "импорт",
            "экспорт",
            "внешнеэконом",
            "дубай",
            "дубае",
            "иностранн",
            "за рубежом",
            "наличными деньгами",
            "наличные деньги",
            "оплата наличными",
            "оплата товара наличными",
        ),
        _currency_variants,
        1.2,
    ),
    (
        (
            "товариществах с ограниченной и дополнительной ответственностью",
            "жауапкершілігі шектеулі",
        ),
        _llp_variants,
        1.2,
    ),
    (
        (
            "тоо",
            "т о о",
            "ип",
        ),
        _llp_variants,
        0.15,
    ),
    (
        (
            "зпп",
            "защите прав потребителей",
            "правах потребителей",
            "возврат товара",
            "продавец отказал",
            "некачественный товар",
            "недостаток товара",
            "товарный вид",
            "в течение четырнадцати дней",
            "14 дней",
            "потребитель",
            "бритва",
        ),
        _consumer_variants,
        1.4,
    ),
    (
        (
            "цифровые активы",
            "необеспеченные цифровые активы",
            "крипто",
            "криптобирж",
            "стейкинг",
            "kraken",
        ),
        _digital_assets_variants,
        1.6,
    ),
    (
        (
            "мусор",
            "тбо",
            "отходы",
            "антисанитар",
            "контейнер",
            "санитарно-эпидемиолог",
            "мусорная площадка",
        ),
        _koap_variants,
        1.3,
    ),
    (
        _THEFT_QUERY_HINTS,
        _uk_variants,
        1.6,
    ),
    (
        (
            "банковской деятельности",
            "банковский счет",
            "банковский заем",
            "банк",
            "счет",
            "перевод денег",
        ),
        _banks_variants,
        0.8,
    ),
]


def _score_target_codes(query: str) -> list[tuple[str, float]]:
    q = _normalized_query(query)
    scores: dict[str, float] = {}

    for aliases, code_names in _LAW_ALIAS_GROUPS:
        matched = False
        for alias in aliases:
            if _has_alias(q, alias):
                matched = True
                break
        if not matched:
            continue
        for code_name in code_names:
            scores[code_name] = scores.get(code_name, 0.0) + 1.0

    for aliases, code_names, weight in _LAW_ROUTE_HINTS:
        matched_count = 0
        for alias in aliases:
            if _has_alias(q, alias):
                matched_count += 1
        if not matched_count:
            continue
        boost = min(weight * matched_count, weight * 3)
        for code_name in code_names:
            scores[code_name] = scores.get(code_name, 0.0) + boost

    for bundle in _matching_legal_concepts(query):
        weight = float(bundle.get("weight", 1.0) or 1.0)
        for code_name in bundle.get("code_names", ()):
            scores[code_name] = scores.get(code_name, 0.0) + weight

    ranked = sorted(scores.items(), key=lambda item: item[1], reverse=True)
    if not ranked:
        return []
    return ranked


def _detect_target_codes(query: str) -> list[str]:
    ranked = _score_target_codes(query)
    if not ranked:
        return []

    top_score = ranked[0][1]
    second_score = ranked[1][1] if len(ranked) > 1 else 0.0

    if top_score >= 1.8 and top_score >= second_score + 1.2:
        return [code for code, score in ranked if score >= top_score - 0.4]

    cutoff = max(1.0, top_score - 0.5)
    detected: list[str] = []
    for code, score in ranked:
        if score < cutoff:
            continue
        if code not in detected:
            detected.append(code)
    return detected


def _normalize_article_number(value: str | None) -> str:
    raw = str(value or "").strip().lower()
    if not raw:
        return ""
    raw = raw.replace("статья", "").replace("ст.", "").replace("ст", "").replace("бап", "")
    raw = re.sub(r"\s+", "", raw)
    raw = raw.replace("–", "-").replace("—", "-")
    raw = re.sub(r"[^\da-zа-я\-\.]", "", raw, flags=re.IGNORECASE)
    return raw.strip(".-")


def _extract_query_article_number(query: str) -> str | None:
    q = query or ""
    match = re.search(r"(?:статья|ст\.|ст|бап)\s*(\d+[а-яА-Яa-zA-Z\-]?)", q, re.IGNORECASE)
    if match:
        normalized = _normalize_article_number(match.group(1))
        return normalized or None
    return None


def _extract_query_article_numbers(query: str) -> list[str]:
    q = query or ""
    articles: list[str] = []

    def _add(article: str) -> None:
        normalized = _normalize_article_number(article)
        if normalized and normalized not in articles:
            articles.append(normalized)

    primary = _extract_query_article_number(q)
    if primary:
        _add(primary)

    for match in re.finditer(
        r"(?:статьи|статья|ст\.|ст|баптары|баптар|бап)\s*([0-9,\-\sandи]+)",
        q,
        re.IGNORECASE,
    ):
        raw_tail = match.group(1)
        for token in re.findall(r"\d+(?:-\d+)?", raw_tail):
            _add(token)

    range_match = _extract_article_range(q)
    if range_match:
        start, end = range_match
        for number in range(start, end + 1):
            _add(str(number))

    for article in sorted(_focus_articles_from_query(q)):
        _add(article)

    return articles


def _filter_docs_by_codes(docs: List[Document], code_names: list[str]) -> List[Document]:
    if not code_names:
        return docs
    allowed = set(code_names)
    return [d for d in docs if (d.metadata.get("code_ru") or "").strip() in allowed]


def _search_with_code_filters(
    query: str,
    code_names: list[str],
    *,
    k: int,
    article_number: str | None = None,
    article_numbers: list[str] | None = None,
) -> List[Document]:
    if not code_names:
        return []
    try:
        store = get_vector_store()
    except Exception:
        return []
    docs: list[Document] = []
    target_articles = list(article_numbers or [])
    if article_number:
        normalized = _normalize_article_number(article_number)
        if normalized and normalized not in target_articles:
            target_articles.append(normalized)

    for code_name in code_names:
        filters: list[dict[str, Any]] = []
        if target_articles:
            filters.extend(
                {"code_ru": code_name, "article_number": target_article}
                for target_article in target_articles
            )
        filters.append({"code_ru": code_name})

        for search_filter in filters:
            try:
                docs = _merge_unique(
                    docs,
                    store.similarity_search(query, k=k, filter=search_filter),
                )
            except Exception:
                continue
    return docs


_LEGAL_SYNONYMS: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("договор", ("обязательство", "сделка", "договорные отношения")),
    ("налич", ("наличные расчеты", "денежные средства", "оплата наличными")),
    ("ущерб", ("убытки", "вред", "возмещение вреда")),
    ("мусор", ("отходы", "тбо", "санитарные требования")),
    ("крипто", ("цифровые активы", "необеспеченные цифровые активы", "стейкинг")),
    ("банкрот", ("неплатежеспособность", "восстановление платежеспособности")),
    ("тоо", ("товарищество с ограниченной ответственностью", "участник тоо")),
    ("ип", ("индивидуальный предприниматель", "предпринимательская деятельность")),
    ("недвижим", ("имущество", "право собственности", "регистрация прав")),
)

def _is_theft_query(query: str) -> bool:
    q = _normalized_query(query)
    return any(token in q for token in _THEFT_QUERY_HINTS)


def _matching_legal_concepts(query: str) -> list[dict[str, Any]]:
    q = _normalized_query(query)
    matched: list[dict[str, Any]] = []
    for bundle in _LEGAL_CONCEPT_BUNDLES:
        if any(token in q for token in bundle.get("patterns", ())):
            matched.append(bundle)
    return matched


def _expand_legal_synonyms(query: str) -> list[str]:
    q = _normalized_query(query)
    extras: list[str] = []
    for needle, synonyms in _LEGAL_SYNONYMS:
        if needle not in q:
            continue
        extras.extend(s for s in synonyms if s not in q)
    return extras


def _rewrite_query_for_retrieval(query: str) -> str:
    llm = None
    if getattr(config, "USE_LLM_QUERY_REWRITE", False):
        try:
            llm = get_llm()
        except Exception:
            llm = None
    return rewrite_query(
        query,
        llm=llm,
        detect_target_codes=_detect_target_codes,
        extract_query_article_number=_extract_query_article_number,
        focus_articles_from_query=_focus_articles_from_query,
        expand_legal_synonyms=_expand_legal_synonyms,
    )


def _build_retrieval_queries(query: str) -> list[str]:
    augmented = _augment_retrieval_query(query)
    rewritten = _rewrite_query_for_retrieval(query)
    queries: list[str] = []
    for candidate in (query, augmented, rewritten):
        cleaned = re.sub(r"\s+", " ", (candidate or "").strip())
        if cleaned and cleaned not in queries:
            queries.append(cleaned)

    target_codes = _detect_target_codes(query)
    if target_codes:
        code_query = f"{rewritten} {' '.join(target_codes)}".strip()
        if code_query and code_query not in queries:
            queries.append(code_query)

    limit = max(1, getattr(config, "RETRIEVER_MULTI_QUERY_LIMIT", 4))
    return queries[:limit]


def _multi_query_retrieve(base_retriever: Any, query: str) -> List[Document]:
    merged: list[Document] = []
    for candidate in _build_retrieval_queries(query):
        try:
            docs = base_retriever.invoke(candidate)
        except Exception:
            continue
        merged = _merge_unique(merged, list(docs))
    return merged


def _multi_query_search_with_code_filters(
    query: str,
    code_names: list[str],
    *,
    k: int,
    article_number: str | None = None,
    article_numbers: list[str] | None = None,
) -> List[Document]:
    docs: list[Document] = []
    for candidate in _build_retrieval_queries(query):
        docs = _merge_unique(
            docs,
            _search_with_code_filters(
                candidate,
                code_names,
                k=k,
                article_number=article_number,
                article_numbers=article_numbers,
            ),
        )
    return docs


def _sort_docs_for_coverage(
    docs: List[Document],
    *,
    target_codes: list[str] | None = None,
    target_articles: list[str] | None = None,
) -> List[Document]:
    if not docs:
        return []

    target_code_set = set(target_codes or [])
    target_article_set = {_normalize_article_number(a) for a in (target_articles or []) if a}
    scored: list[tuple[float, int, Document]] = []
    for idx, doc in enumerate(docs):
        meta = doc.metadata or {}
        score = 0.0
        doc_code = (meta.get("code_ru") or "").strip()
        doc_article = _normalize_article_number(meta.get("article_number"))

        if target_code_set and doc_code in target_code_set:
            score += 2.0
        if target_article_set and doc_article in target_article_set:
            score += 3.0
        if target_code_set and target_article_set and doc_code in target_code_set and doc_article in target_article_set:
            score += 1.0

        scored.append((score, idx, doc))

    scored.sort(key=lambda item: (-item[0], item[1]))
    return [doc for _, _, doc in scored]


def _augment_retrieval_query(query: str) -> str:
    q = (query or "").lower()
    extras: list[str] = []
    target_codes = _detect_target_codes(query)
    is_criminal = any(code in set(_uk_variants) for code in target_codes) or _is_criminal_query(query)

    # Check language
    is_kz = _is_kz_query(query)

    if any(
        token in q
        for token in (
            "несовершеннолетний",
            "несовершеннолетние",
            "несовершеннолетних",
            "minor",
            "underage",
            "кәмелетке толмаған",
            "кәмелетке толмағандар",
        )
    ):
        if is_kz:
            extras.append("он сегіз жасқа толмаған жұмыскерлер")
            extras.append("76-бап ҚР Еңбек кодексі")
            extras.append("кәмелетке толмағандардың түнгі жұмысына тыйым салу")
        else:
            extras.append("работники, не достигшие восемнадцатилетнего возраста")
            extras.append("статья 76 Трудовой кодекс РК")
            extras.append("запрет ночной работы несовершеннолетних")
    for bundle in _matching_legal_concepts(query):
        if bundle.get("criminal") and not is_criminal:
            continue
        localized = bundle.get("expansions_kz", ()) if is_kz else bundle.get("expansions_ru", ())
        extras.extend(str(item) for item in localized if item)
    if any(
        token in q
        for token in (
            "субсид",
            "субсидия",
            "гос",
            "государ",
            "бюджет",
            "грант",
            "инвест",
            "смет",
            "договор",
            "фиктив",
            "жалған",
            "құжат",
            "алаяқ",
            "мемлекеттік",
            "қаржы",
            "ақша",
        )
    ) and is_criminal:
        if is_kz:
            extras.append("алаяқтық 190-бап ҚР ҚК")
            extras.append("қылмыстық жолмен алынған ақшаны заңдастыру 218-бап ҚР ҚК")
            extras.append("субсидия алу үшін жалған құжаттар 190-бап ҚР ҚК")
        else:
            extras.append("алаяқтық 190 УК РК")
            extras.append("қылмыстық жолмен алынған ақшаны заңдастыру 218 УК РК")
            extras.append("субсидия алу үшін жалған құжаттар 190 УК РК")
    if any(
        token in q
        for token in (
            "заңсыз кәсіпкер",
            "кәсіпкерлік",
            "лицензиясыз",
            "тіркеусіз",
            "незаконн",
            "без регистрации",
            "без лицензии",
            "салық төлем",
            "налог",
            "уклонен",
        )
    ) and is_criminal:
        if is_kz:
            extras.append("заңсыз кәсіпкерлік 214-бап ҚР ҚК")
            extras.append("салық төлеуден жалтару 245-бап ҚР ҚК")
        else:
            extras.append("заңсыз кәсіпкерлік 214 УК РК")
            extras.append("салық төлеуден жалтару 245 УК РК")
    if any(
        token in q
        for token in (
            "пирамида",
            "пирамид",
            "қаржылық пирамида",
            "инвестиция",
            "инвест",
            "жоғары пайда",
            "30-50%",
        )
    ) and is_criminal:
        if is_kz:
            extras.append("қаржылық пирамида құру және басқару 217-бап ҚР ҚК")
            extras.append("қаржылық пирамиданы жарнамалау 217-1-бап ҚР ҚК")
        else:
            extras.append("қаржылық пирамида құру және басқару 217 УК РК")
            extras.append("финансовая пирамида создание и руководство 217 УК РК")
            extras.append("реклама финансовой пирамиды 217-1 УК РК")
    # ... other heuristics could be localized similarly ...

    range_match = _extract_article_range(query)
    if range_match and ("ук" in q or "қылмыстық" in q or "уголов" in q):
        start, end = range_match
        nums = " ".join(str(n) for n in range(start, end + 1))
        if is_kz:
            extras.append(f"ҚР ҚК {nums} баптары")
        else:
            extras.append(f"статьи {nums} УК РК")
    return (query + " " + " ".join(extras)).strip() if extras else query


def _is_criminal_query(query: str) -> bool:
    q = _normalized_query(query)
    if any(token in q for token in ("қылмыстық", "уголов", "преступ", "ук рк", "квалификация преступ", "состав преступ")):
        return True
    if any(bundle.get("criminal") for bundle in _matching_legal_concepts(q)):
        return True
    return False


def _focus_articles_from_query(query: str) -> set[str]:
    q = (query or "").lower()
    focus: set[str] = set()
    for bundle in _matching_legal_concepts(q):
        focus.update(
            _normalize_article_number(article)
            for article in bundle.get("focus_articles", ())
            if _normalize_article_number(article)
        )
    if any(
        token in q
        for token in (
            "субсид",
            "субсидия",
            "гос",
            "государ",
            "бюджет",
            "грант",
            "инвест",
            "смет",
            "договор",
            "фиктив",
            "мемлекеттік",
            "жалған",
            "алаяқ",
            "құжат",
            "қаржы",
            "ақша",
        )
    ) and _is_criminal_query(query):
        focus.update({_normalize_article_number("190"), _normalize_article_number("218")})
    if any(
        token in q
        for token in (
            "заңсыз кәсіпкер",
            "кәсіпкерлік",
            "лицензиясыз",
            "тіркеусіз",
            "незаконн",
            "без регистрации",
            "без лицензии",
            "салық төлем",
            "налог",
            "уклонен",
        )
    ) and _is_criminal_query(query):
        focus.update({_normalize_article_number("214"), _normalize_article_number("245")})
    if any(
        token in q
        for token in (
            "пирамида",
            "пирамид",
            "қаржылық пирамида",
            "инвестиция",
            "инвест",
            "жоғары пайда",
            "30-50%",
        )
    ) and _is_criminal_query(query):
        focus.update({_normalize_article_number("217"), _normalize_article_number("190")})
    if any(
        token in q
        for token in (
            "қалдық су",
            "қалдық сулар",
            "өзен",
            "су ластау",
            "суға төгу",
            "тазарту жүйесі",
            "эколог",
            "өндіріс қалдық",
            "өндірістік қалдық",
            "химия",
            "улы зат",
            "жаппай улану",
            "жаппай ауру",
        )
    ):
        focus.update({_normalize_article_number("328"), _normalize_article_number("325"), _normalize_article_number("324")})
    if any(
        token in q
        for token in (
            "шетел",
            "сырт ел",
            "резидент",
            "жылжымайтын",
            "жарғылық капитал",
            "уставный капитал",
            "капиталға",
            "вклад",
            "взнос",
            "декларация",
            "деклар",
            "имущ",
            "имущественный",
            "прирост стоимости",
        )
    ):
        focus.update({_normalize_article_number("228"), _normalize_article_number("330"), _normalize_article_number("332"), _normalize_article_number("333")})
    return focus


def _is_subsidy_query(query: str) -> bool:
    q = (query or "").lower()
    return any(
        token in q
        for token in (
            "субсид",
            "субсидия",
            "грант",
            "гос",
            "государ",
            "мемлекеттік",
            "бюджет",
        )
    )


def _is_illegal_business_query(query: str) -> bool:
    q = (query or "").lower()
    return any(
        token in q
        for token in (
            "заңсыз кәсіпкер",
            "кәсіпкерлік",
            "лицензиясыз",
            "тіркеусіз",
            "незаконн",
            "без регистрации",
            "без лицензии",
            "салық төлем",
            "налог",
            "уклонен",
        )
    )


def _is_pyramid_query(query: str) -> bool:
    q = (query or "").lower()
    return any(
        token in q
        for token in (
            "пирамида",
            "пирамид",
            "қаржылық пирамида",
            "инвестиция",
            "инвест",
            "жоғары пайда",
            "30-50%",
        )
    )


def _needs_circumstances_query(query: str) -> bool:
    q = (query or "").lower()
    return any(
        token in q for token in ("ауырлататын", "жеңілдететін", "смягча", "отягча")
    )


def _doc_key(doc: Document) -> tuple[str, str]:
    source = str(doc.metadata.get("source", "")).strip()
    article = str(doc.metadata.get("article_number", "")).strip()
    path = str(doc.metadata.get("path", "")).strip()
    clause_level = str(doc.metadata.get("clause_level", "")).strip()
    clause_number = str(doc.metadata.get("clause_number", "")).strip()
    subclause_number = str(doc.metadata.get("subclause_number", "")).strip()
    if path:
        return (source, path)
    if clause_level or clause_number or subclause_number:
        return (
            source,
            "::".join(
                [article, clause_level, clause_number, subclause_number]
            ).strip(":"),
        )
    return (source, article)


def _article_doc_key(doc: Document) -> tuple[str, str]:
    meta = doc.metadata or {}
    code_ru = str(meta.get("code_ru", "")).strip()
    article = _normalize_article_number(meta.get("article_number"))
    path = str(meta.get("path", "")).strip()
    if code_ru and article:
        return (code_ru, article)
    if code_ru and path:
        return (code_ru, path)
    return _doc_key(doc)


def _merge_unique(base: List[Document], extra: List[Document]) -> List[Document]:
    seen = {_doc_key(d) for d in base}
    merged = list(base)
    for d in extra:
        key = _doc_key(d)
        if key not in seen:
            merged.append(d)
            seen.add(key)
    return merged


def _truncate_for_llm_rerank(text: str, limit: int | None = None) -> str:
    if limit is None:
        limit = getattr(config, "LLM_RERANK_MAX_DOC_CHARS", 700)
    text = (text or "").strip()
    if len(text) <= limit:
        return text
    return text[:limit] + "\n[...обрезано...]"


def _parse_llm_rerank_selection(raw: str, total_docs: int) -> list[int]:
    text = (raw or "").strip()
    if not text:
        return []
    try:
        payload = json.loads(text)
        if isinstance(payload, dict):
            candidates = payload.get("selected") or payload.get("indices") or []
            if isinstance(candidates, list):
                result = []
                for item in candidates:
                    try:
                        idx = int(item)
                    except Exception:
                        continue
                    if 1 <= idx <= total_docs and idx not in result:
                        result.append(idx)
                return result
    except Exception:
        pass

    result = []
    for match in re.findall(r"\d+", text):
        idx = int(match)
        if 1 <= idx <= total_docs and idx not in result:
            result.append(idx)
    return result


def _llm_rerank_documents(
    query: str, documents: Sequence[Document], top_n: int
) -> Sequence[Document]:
    if not documents:
        return []
    candidates = list(documents)[: max(top_n, config.LLM_RERANK_CANDIDATES)]
    llm = get_llm()
    query_text = _truncate_for_llm_rerank(query, limit=1200)
    prompt_prefix = (
        "Ты ранжируешь нормы права для юридического поиска.\n"
        "Выбери самые релевантные документы для вопроса.\n"
        "Критерии по приоритету:\n"
        "1. Правильный закон/кодекс.\n"
        "2. Правильная статья.\n"
        "3. Если есть пункты статьи, предпочитай самый точный пункт.\n"
        "4. Не выбирай дубликаты одной и той же нормы без необходимости.\n\n"
        f"Нужно выбрать {top_n} лучших кандидатов.\n"
        'Верни только JSON вида {"selected":[1,2,3]} без пояснений.\n\n'
        f"Вопрос:\n{query_text}\n\n"
        "Кандидаты:\n"
    )
    max_prompt_chars = getattr(config, "LLM_RERANK_MAX_PROMPT_CHARS", 9000)
    candidate_blocks: list[str] = []
    current_len = len(prompt_prefix)
    for i, doc in enumerate(candidates, start=1):
        meta = doc.metadata or {}
        code_ru = (meta.get("code_ru") or "").strip()
        article_number = (meta.get("article_number") or "").strip()
        path = (meta.get("path") or "").strip()
        article_title = (meta.get("article_title") or "").strip()
        header_parts = [f"{i})"]
        if code_ru:
            header_parts.append(code_ru)
        if article_number:
            header_parts.append(f"ст. {article_number}")
        if path and path != f"ст. {article_number}":
            header_parts.append(path)
        if article_title:
            header_parts.append(article_title)
        snippet = _truncate_for_llm_rerank(doc.page_content)
        block = " | ".join(header_parts) + "\n" + snippet
        projected_len = current_len + len(block) + 2
        if candidate_blocks and projected_len > max_prompt_chars:
            break
        candidate_blocks.append(block)
        current_len = projected_len

    if not candidate_blocks:
        return candidates[:top_n]

    prompt = prompt_prefix + "\n\n".join(candidate_blocks)

    try:
        response = llm.invoke(prompt)
        raw = getattr(response, "content", response)
        selected = _parse_llm_rerank_selection(str(raw), len(candidate_blocks))
        visible_candidates = candidates[: len(candidate_blocks)]
        chosen = [visible_candidates[i - 1] for i in selected[:top_n]]
        if chosen:
            seen = {_doc_key(d) for d in chosen}
            for doc in visible_candidates:
                if len(chosen) >= top_n:
                    break
                key = _doc_key(doc)
                if key in seen:
                    continue
                chosen.append(doc)
                seen.add(key)
            return chosen
    except Exception as e:
        logger.error("LLM reranker failed: %s", e, exc_info=True)

    return candidates[:top_n]


def _fetch_parent_context_from_store(
    code_ru: str, article_number: str
) -> tuple[str, str, str]:
    """
    When a clause/subclause chunk is missing Chapter or Article title, pull from Pinecone:
    fetch sibling chunks with same code_ru + article_number; return first that has chapter_title
    and optionally article_title for full legal context.
    Returns (chapter_number, chapter_title, article_title).
    """
    if not code_ru or not article_number:
        return "", "", ""
    try:
        store = get_vector_store()
        siblings = _similarity_search_with_timeout(
            store,
            f"Глава статья {article_number} {code_ru}",
            k=5,
            filter={"code_ru": code_ru, "article_number": article_number},
            timeout_sec=float(getattr(config, "PINECONE_ENRICHMENT_TIMEOUT_SEC", 2.0)),
        )
        best_cn, best_ct, best_at = "", "", ""
        for s in siblings:
            ct = (s.metadata.get("chapter_title") or "").strip()
            cn = (s.metadata.get("chapter_number") or "").strip()
            at = (s.metadata.get("article_title") or "").strip()
            if ct:
                best_cn, best_ct = cn, ct[:150]
            if at:
                best_at = at[:200]
            if best_ct and best_at:
                break
        return best_cn, best_ct, best_at
    except Exception as e:
        logger.debug(
            "Context enrichment skipped for %s ст.%s: %s",
            code_ru,
            article_number,
            e,
            exc_info=True,
        )
    return "", "", ""


def _enrich_with_parent_context(docs: List[Document]) -> List[Document]:
    """
    Context Enrichment: prepend Code -> Chapter -> Article breadcrumb to each doc.
    When a clause/subclause is found and Chapter or Article title is missing,
    pull parent Chapter and Article title from Pinecone for full legal context.
    """
    enriched = []
    for doc in docs:
        m = doc.metadata
        parts: list[str] = []

        code = m.get("code_ru", "").strip()
        code_kz = m.get("code_kz", "").strip()
        source = m.get("source", "").strip()

        if code:
            parts.append(code)

        chapter_num = m.get("chapter_number", "").strip()
        chapter_title = m.get("chapter_title", "").strip()
        article_title = (m.get("article_title") or "").strip()
        art_num = m.get("article_number", "").strip()
        clause_level = (m.get("clause_level") or "").strip().lower()

        # For clause/subclause chunks missing chapter or article title, pull from vector store
        if (clause_level in ("clause", "subclause")) and code and art_num:
            if not chapter_title or not article_title:
                cn, ct, at = _fetch_parent_context_from_store(code, art_num)
                if ct:
                    chapter_num, chapter_title = cn, ct
                if at:
                    article_title = at

        # Determine language block based on source suffix
        is_kz = source.endswith("_kz")

        if is_kz:
            code_label = code_kz if code_kz else code
            chap_label = "Тарау/Бөлім"
            art_label = "бап"
            ed_label = "редакциясы"
            if chapter_num and chapter_title:
                parts.append(f"{chapter_num}-{chap_label}: {chapter_title}")
            elif chapter_title:
                parts.append(f"{chap_label}: {chapter_title}")

            if art_num:
                if article_title:
                    parts.append(f"{art_num}-{art_label}. {article_title}")
                else:
                    parts.append(f"{art_num}-{art_label}")

            rev_date = m.get("revision_date", "").strip()
            if rev_date:
                parts.append(f"{rev_date} {ed_label}")
        else:
            code_label = code
            if chapter_num and chapter_title:
                parts.append(f"Глава {chapter_num}: {chapter_title}")
            elif chapter_title:
                parts.append(f"Глава: {chapter_title}")

            if art_num:
                if article_title:
                    parts.append(f"Статья {art_num}. {article_title}")
                else:
                    parts.append(f"Статья {art_num}")

            rev_date = m.get("revision_date", "").strip()
            if rev_date:
                parts.append(f"ред. от {rev_date}")

        # Insert code label first
        if code_label:
            parts.insert(0, code_label)

        if parts:
            breadcrumb = "[" + " | ".join(parts) + "]\n"
            enriched_content = breadcrumb + doc.page_content
            enriched.append(Document(page_content=enriched_content, metadata=m))
        else:
            enriched.append(doc)

    return enriched


class _TrimRetriever(BaseRetriever):
    """Обрезает количество и длину документов перед LLM, чтобы избежать переполнения контекста."""

    base_retriever: Any
    max_docs: int = 8
    max_chars_per_doc: int = 1800
    max_tokens_per_doc: int = 0
    max_total_tokens: int = 0

    def _get_relevant_documents(
        self,
        query: str | dict,
        *,
        run_manager: CallbackManagerForRetrieverRun | None = None,
    ) -> List[Document]:
        if isinstance(query, dict):
            # If we receive a dict (e.g. from LCEL chain), extract the query string
            # Try common keys
            q = query.get("input") or query.get("query") or query.get("question") or ""
            if not q and "context" not in query:  # If it's not a doc chain input
                print(
                    f"DEBUG: _TrimRetriever received dict without known keys: {query.keys()}"
                )

            # If the dict is just {"input": "..."} which is typical for create_retrieval_chain
            if q:
                query = q
            else:
                # Fallback: convert to string representation if needed or just empty
                pass

        # Ensure query is string for base_retriever
        if not isinstance(query, str):
            query = str(query)

        docs = self.base_retriever.invoke(query)
        trimmed: list[Document] = []
        total_tokens = 0
        doc_suffix = "\n[...текст обрезан...]"
        for d in docs[: self.max_docs]:
            content = str(d.page_content or "")
            parent_article_text = str((d.metadata or {}).get("parent_article_text") or "")
            if parent_article_text and len(content) < 300:
                content = parent_article_text
            if len(content) > self.max_chars_per_doc:
                content = content[: self.max_chars_per_doc].rstrip() + doc_suffix
            if self.max_tokens_per_doc > 0:
                content = _truncate_text_to_token_budget(
                    content,
                    self.max_tokens_per_doc,
                    suffix=doc_suffix,
                )

            candidate = Document(page_content=content, metadata=d.metadata)
            candidate_tokens = _estimate_token_count(_format_doc_for_prompt(candidate))
            if self.max_total_tokens > 0 and total_tokens + candidate_tokens > self.max_total_tokens:
                remaining_tokens = self.max_total_tokens - total_tokens
                if remaining_tokens <= 80:
                    break
                content = _truncate_text_to_token_budget(
                    content,
                    max(40, remaining_tokens - 32),
                    suffix=doc_suffix,
                )
                candidate = Document(page_content=content, metadata=d.metadata)
                candidate_tokens = _estimate_token_count(_format_doc_for_prompt(candidate))
                if total_tokens + candidate_tokens > self.max_total_tokens:
                    break

            trimmed.append(candidate)
            total_tokens += candidate_tokens
        # Inject parent-context breadcrumb so LLM knows Code → Chapter → Article scope
        return _enrich_with_parent_context(trimmed)


class _DedupRetriever(BaseRetriever):
    """Удаляет дубликаты чанков до rerank/trim, чтобы не тратить top-k на повторения."""

    base_retriever: Any

    def _get_relevant_documents(
        self,
        query: str | dict,
        *,
        run_manager: CallbackManagerForRetrieverRun | None = None,
    ) -> List[Document]:
        docs = self.base_retriever.invoke(query)
        unique_docs: list[Document] = []
        seen: set[tuple[str, str]] = set()
        for doc in docs:
            key = _article_doc_key(doc)
            if key in seen:
                continue
            seen.add(key)
            unique_docs.append(doc)
        return unique_docs


def _apply_legal_score(
    query: str,
    doc: Document,
    base_score: float,
    *,
    target_codes: list[str] | None = None,
    target_articles: set[str] | None = None,
) -> float:
    score = float(base_score)
    meta = doc.metadata or {}
    doc_code = (meta.get("code_ru") or "").strip()
    doc_article = _normalize_article_number(meta.get("article_number"))
    query_lower = (query or "").lower()
    domain = detect_domain(query)
    concept_matches = _matching_legal_concepts(query)
    title_haystack = " ".join(
        str(part).lower()
        for part in (
            meta.get("article_title", ""),
            meta.get("chapter_title", ""),
            doc.page_content[:800],
        )
        if part
    )

    if _looks_like_raw_code_name(doc_code):
        score -= 0.25
    if not doc_article:
        score -= 0.20
    if _is_noisy_legal_chunk(doc):
        score -= 0.45

    if target_codes:
        if doc_code in set(target_codes):
            score += 0.45
        else:
            score -= 0.15

    if target_articles and doc_article in target_articles:
        score += 0.30
        if doc_article and doc_article in query_lower:
            score += 0.20

    if domain:
        if domain_matches_code(domain, doc_code):
            score += 0.15
        else:
            score -= 0.20

    if concept_matches:
        for bundle in concept_matches:
            bundle_codes = set(bundle.get("code_names", ()))
            if bundle_codes and doc_code in bundle_codes:
                score += 0.18

            focus_articles = {
                _normalize_article_number(article)
                for article in bundle.get("focus_articles", ())
                if _normalize_article_number(article)
            }
            if focus_articles and doc_article in focus_articles:
                score += 0.30

            doc_terms = tuple(str(term).lower() for term in bundle.get("doc_terms", ()) if term)
            if doc_terms and any(term in title_haystack for term in doc_terms):
                score += 0.16

    return score


def _tokenize_legal_terms(text: str) -> list[str]:
    normalized = _normalized_query(text)
    terms: list[str] = []
    for token in re.findall(r"[a-zа-яёқіңғүұһәө]{4,}", normalized, re.IGNORECASE):
        if token not in terms:
            terms.append(token)
    return terms


def _lexical_overlap_score(query: str, doc: Document) -> float:
    query_terms = _tokenize_legal_terms(query)
    if not query_terms:
        return 0.0

    meta = doc.metadata or {}
    haystack = " ".join(
        str(part)
        for part in (
            meta.get("code_ru", ""),
            meta.get("article_title", ""),
            meta.get("chapter_title", ""),
            doc.page_content[:1200],
        )
        if part
    ).lower()
    if not haystack:
        return 0.0

    overlap = sum(1 for term in query_terms if term in haystack)
    if overlap <= 0:
        return 0.0
    return min(0.35, overlap / max(4.0, float(len(query_terms))))


def _resolve_reranker_backend() -> str:
    raw = (getattr(config, "RERANKER_BACKEND", None) or "auto").strip().lower()
    if raw and raw != "auto":
        return raw
    model = (getattr(config, "RERANKER_MODEL", "") or "").lower()
    if "jina" in model:
        return "jina"
    return "flag_embedding"


def _should_skip_neural_rerank(query: str, documents: Sequence[Document]) -> bool:
    top_k = min(4, len(documents))
    if top_k < 1:
        return False

    if getattr(config, "RERANK_DYNAMIC_SKIP", False):
        target_codes = _detect_target_codes(query)
        if target_codes:
            codes_set = set(target_codes)
            matches = sum(
                1
                for d in documents[:top_k]
                if (d.metadata.get("code_ru") or "").strip() in codes_set
            )
            thresh = float(getattr(config, "RERANK_SKIP_LEXICAL_THRESHOLD", 0.35))
            min_m = int(getattr(config, "RERANK_SKIP_MIN_CODE_MATCHES", 2))
            best_lex = max(_lexical_overlap_score(query, d) for d in documents[:top_k])
            if matches >= min_m and best_lex >= thresh:
                return True

    if not getattr(config, "RERANK_SKIP_LONG_DOCS", False):
        return False

    long_threshold = int(getattr(config, "RERANK_SKIP_LONG_DOC_CHARS", 2400))
    short_threshold = int(getattr(config, "RERANK_SKIP_SHORT_DOC_CHARS", 700))
    lengths = [len((d.page_content or "").strip()) for d in documents[: max(4, top_k)]]
    long_docs = sum(length >= long_threshold for length in lengths)
    short_docs = sum(length <= short_threshold for length in lengths)
    # Large chunks can swamp short but exact statutory chunks in cross-encoder reranking.
    return bool(long_docs and short_docs >= 2)


def _rank_docs_by_legal_only(
    query: str,
    documents: Sequence[Document],
    *,
    target_codes: list[str],
    target_articles: set[str],
) -> list[Document]:
    scored: list[tuple[float, int, Document]] = []
    for idx, doc in enumerate(documents):
        base = 1.0 + _lexical_overlap_score(query, doc)
        s = _apply_legal_score(
            query,
            doc,
            base,
            target_codes=target_codes,
            target_articles=target_articles,
        )
        scored.append((s, idx, doc))
    scored.sort(key=lambda item: (-item[0], item[1]))
    out: list[Document] = []
    for s, _, doc in scored:
        doc.metadata["relevance_score"] = s
        out.append(doc)
    return out


def _rank_docs_with_legal_scoring(
    query: str,
    docs: List[Document],
    *,
    target_codes: list[str] | None = None,
    target_articles: list[str] | None = None,
) -> List[Document]:
    if not docs:
        return []

    target_article_set = {
        _normalize_article_number(article)
        for article in (target_articles or [])
        if _normalize_article_number(article)
    }
    ranked: list[tuple[float, int, Document]] = []
    for idx, doc in enumerate(docs):
        base_score = float(doc.metadata.get("relevance_score", 0.0) or 0.0)
        score = _apply_legal_score(
            query,
            doc,
            base_score,
            target_codes=target_codes,
            target_articles=target_article_set,
        )
        score += _lexical_overlap_score(query, doc)
        ranked.append((score, idx, doc))

    ranked.sort(key=lambda item: (-item[0], item[1]))
    return [doc for _, _, doc in ranked]


def _apply_diversity_penalty(
    scored_docs: Sequence[tuple[Document, float]],
    *,
    penalty_step: float,
) -> list[tuple[Document, float]]:
    if penalty_step <= 0:
        return list(scored_docs)

    code_counts: dict[str, int] = {}
    adjusted: list[tuple[Document, float]] = []
    for doc, score in sorted(scored_docs, key=lambda item: item[1], reverse=True):
        code_ru = (doc.metadata.get("code_ru") or "").strip()
        seen_count = code_counts.get(code_ru, 0)
        adjusted_score = score - (penalty_step * seen_count if code_ru else 0.0)
        adjusted.append((doc, adjusted_score))
        if code_ru:
            code_counts[code_ru] = seen_count + 1

    adjusted.sort(key=lambda item: item[1], reverse=True)
    return adjusted


def _load_bm25_chunks() -> list[Document] | None:
    """Load BM25 chunks lazily from pickle (preferred) or prepare_data."""
    try:
        import pickle  # local import: keep module import fast

        _pkl = getattr(config, "CHUNKS_PICKLE_PATH", None) or (
            config.BASE_DIR / "chunks_for_bm25.pkl"
        )
        if _pkl and _pkl.exists():
            with open(_pkl, "rb") as f:
                chunks = pickle.load(f)
            if chunks:
                print(f"Чанки для BM25: {len(chunks)} из {_pkl.name}")
                return chunks
    except Exception as e:
        print(f"Не удалось загрузить chunks_for_bm25.pkl: {e}")
    try:
        if config.DOCUMENTS_DIR.exists():
            from ai_service.processing import prepare_data as _pd

            chunks = getattr(_pd, "chunks", None)
            if chunks:
                print(f"Чанки для BM25 из prepare_data: {len(chunks)}")
                return chunks
    except Exception as e:
        print(f"prepare_data не загружен: {e}")
    return None


def _load_summary_chunks() -> list[Document] | None:
    """Load summary index chunks from pickle (preferred) or prepare_data."""
    try:
        import pickle  # local import: keep module import fast

        _pkl = getattr(config, "SUMMARY_CHUNKS_PICKLE_PATH", None) or (
            config.BASE_DIR / "summary_chunks_for_bm25.pkl"
        )
        if _pkl and _pkl.exists():
            with open(_pkl, "rb") as f:
                chunks = pickle.load(f)
            if chunks:
                print(f"Summary чанки: {len(chunks)} из {_pkl.name}")
                return chunks
    except Exception as e:
        print(f"Не удалось загрузить summary_chunks_for_bm25.pkl: {e}")
    try:
        if config.DOCUMENTS_DIR.exists():
            from ai_service.processing import prepare_data as _pd

            chunks = getattr(_pd, "summary_chunks", None)
            if chunks:
                print(f"Summary чанки из prepare_data: {len(chunks)}")
                return chunks
    except Exception as e:
        print(f"prepare_data summary не загружен: {e}")
    return None


def _build_minimal_legal_retriever() -> MinimalLegalRetriever:
    bm25_retriever: Any | None = None
    summary_retriever: Any | None = None
    summary_docs: list[Document] = []

    def hybrid_tokenizer(text: str):
        if _is_kz_query(text):
            import nltk

            return [t for t in nltk.word_tokenize(text.lower()) if t.isalnum()]
        return bm25_preprocess_func(text) or []

    try:
        from langchain_community.retrievers import BM25Retriever

        chunks = _load_bm25_chunks()
        if chunks:
            if _ensure_nltk():
                bm25_retriever = BM25Retriever.from_documents(
                    chunks, preprocess_func=hybrid_tokenizer, k=max(40, _hybrid_k)
                )
                print("BM25 инициализирован для MinimalLegalRetriever.")
            else:
                bm25_retriever = BM25Retriever.from_documents(
                    chunks, k=max(40, _hybrid_k)
                )

        summary_chunks = _load_summary_chunks()
        if summary_chunks:
            summary_docs = list(summary_chunks)
            if _ensure_nltk():
                summary_retriever = BM25Retriever.from_documents(
                    summary_docs, preprocess_func=hybrid_tokenizer, k=max(8, _hybrid_k // 4)
                )
            else:
                summary_retriever = BM25Retriever.from_documents(
                    summary_docs, k=max(8, _hybrid_k // 4)
                )
    except Exception as e:
        print(f"BM25 не запустился для MinimalLegalRetriever: {e}")

    def _collect_summary_hints(query: str) -> tuple[list[Document], list[str]]:
        hints: list[Document] = []
        if summary_retriever is not None:
            try:
                hints.extend(list(summary_retriever.invoke(query) or [])[: _hybrid_k])
            except Exception as exc:
                logger.warning("Summary BM25 search failed in MinimalLegalRetriever: %s", exc)

        if not _disable_pinecone and summary_docs:
            try:
                vs = get_vector_store()
                summary_hits = vs.similarity_search_with_score(
                    query,
                    k=min(8, max(4, _hybrid_k // 4)),
                    filter={"doc_kind": "summary"},
                )
                hints.extend([doc for doc, _ in summary_hits or []])
            except Exception as exc:
                logger.warning("Summary dense search failed in MinimalLegalRetriever: %s", exc)

        hints = _merge_unique(hints, [])
        codes: list[str] = []
        for doc in hints:
            code = str(doc.metadata.get("code_ru") or "").strip()
            if code and code not in codes:
                codes.append(code)
        return hints, codes[: int(getattr(config, "SUMMARY_EXPANSION_MAX_CODES", 3))]

    def _bm25_search(query: str) -> Sequence[Document]:
        if bm25_retriever is None:
            return []
        try:
            docs = list(bm25_retriever.invoke(query) or [])
            _, summary_codes = _collect_summary_hints(query)
            if summary_codes:
                preferred = _filter_docs_by_codes(docs, summary_codes)
                if preferred:
                    docs = _merge_unique(preferred, docs)
            return docs
        except Exception as exc:
            logger.warning("BM25 search failed in MinimalLegalRetriever: %s", exc)
            return []

    def _dense_search(query: str) -> Sequence[tuple[Document, float]]:
        if _disable_pinecone:
            return []
        try:
            vs = get_vector_store()
            _, summary_codes = _collect_summary_hints(query)
            search_kwargs = {"k": max(40, _hybrid_k)}
            if summary_codes:
                search_kwargs["filter"] = {
                    "$or": [{"code_ru": code} for code in summary_codes]
                }
            docs_with_scores = vs.similarity_search_with_score(
                query, **search_kwargs
            )
            filtered = [
                (doc, score)
                for doc, score in docs_with_scores or []
                if (doc.metadata.get("doc_kind") or "chunk") != "summary"
            ]
            if not filtered and summary_codes and "filter" in search_kwargs:
                docs_with_scores = vs.similarity_search_with_score(
                    query, k=max(40, _hybrid_k)
                )
                filtered = [
                    (doc, score)
                    for doc, score in docs_with_scores or []
                    if (doc.metadata.get("doc_kind") or "chunk") != "summary"
                ]
            return filtered
        except Exception as exc:
            logger.warning("Dense search failed in MinimalLegalRetriever: %s", exc)
            return []

    if bm25_retriever is None and summary_retriever is None and _disable_pinecone:
        raise RuntimeError(
            "Retriever initialization failed: Pinecone disabled and BM25 chunks unavailable."
        )

    return MinimalLegalRetriever(
        bm25_search=_bm25_search,
        dense_search=_dense_search,
        bm25_weight=0.55,
        dense_weight=0.45,
        candidate_k=max(40, _hybrid_k),
        final_k=max(40, _hybrid_k),
    )


def get_retriever():
    """Build retriever lazily (Pinecone + optional BM25 + optional reranker)."""
    global _retriever_instance
    if _retriever_instance is not None:
        return _retriever_instance

    with _init_lock:
        if _retriever_instance is not None:
            return _retriever_instance

        base_retriever: Any = _build_minimal_legal_retriever()
        print(
            "MinimalLegalRetriever готов: query rewrite -> BM25 -> dense -> fusion -> legal prior."
        )

        retr: Any = _DedupRetriever(base_retriever=base_retriever)
        if getattr(config, "EXPERIMENTAL_DEDUP_RETRIEVAL", False):
            print("Включён experimental dedup retrieval layer.")

        # Optional reranker (very heavy) — build lazily on first request.
        if config.USE_RERANKER or getattr(config, "RERANKER_MANDATORY", True):
            try:
                try:
                    from langchain.retrievers import ContextualCompressionRetriever
                    from langchain.retrievers.document_compressors.base import (
                        BaseDocumentCompressor,
                    )
                except ImportError:
                    from langchain_classic.retrievers import (
                        ContextualCompressionRetriever,
                    )
                    from langchain_classic.retrievers.document_compressors.base import (
                        BaseDocumentCompressor,
                    )

                config.configure_hf_hub()
                logger.info(
                    "🚀 [START] Reranker initialization (%s)", config.RERANKER_MODEL
                )
                t_rerank = time.perf_counter()
                _backend_resolved = _resolve_reranker_backend()
                _reranker_backend: str = _backend_resolved
                _reranker_model: Any = None

                if _backend_resolved == "cross_encoder":
                    from sentence_transformers import CrossEncoder

                    _reranker_model = CrossEncoder(config.RERANKER_MODEL)
                elif _backend_resolved == "jina":
                    try:
                        from transformers import AutoModel

                        _reranker_model = AutoModel.from_pretrained(
                            config.RERANKER_MODEL,
                            dtype="auto",
                            trust_remote_code=True,
                        )
                        _reranker_model.eval()
                    except Exception as jina_err:
                        logger.warning(
                            "Jina reranker load failed (%s). Using CrossEncoder fallback (%s).",
                            jina_err,
                            config.RERANKER_FALLBACK_MODEL,
                        )
                        from sentence_transformers import CrossEncoder

                        _reranker_model = CrossEncoder(config.RERANKER_FALLBACK_MODEL)
                        _reranker_backend = "cross_encoder"

                if _reranker_model is None:
                    try:
                        from FlagEmbedding import FlagReranker

                        # FlagReranker uses trust_remote_code=True by default in some versions.
                        _reranker_model = FlagReranker(
                            config.RERANKER_MODEL, use_fp16=True
                        )
                        _reranker_backend = "flag_embedding"
                        # Some checkpoints do not define pad_token and fail on batched scoring.
                        try:
                            _tok = getattr(_reranker_model, "tokenizer", None)
                            if _tok is not None and getattr(
                                _tok, "pad_token", None
                            ) is None:
                                eos_token = getattr(_tok, "eos_token", None)
                                if eos_token:
                                    _tok.pad_token = eos_token
                                else:
                                    _tok.add_special_tokens({"pad_token": "<pad>"})
                        except Exception:
                            pass
                    except Exception as flag_error:
                        logger.warning(
                            "FlagReranker unavailable (%s). Falling back to CrossEncoder: %s",
                            flag_error,
                            config.RERANKER_FALLBACK_MODEL,
                        )
                        from sentence_transformers import CrossEncoder

                        _reranker_backend = "cross_encoder"
                        _reranker_model = CrossEncoder(config.RERANKER_FALLBACK_MODEL)

                logger.info(
                    "✅ [SUCCESS] Reranker initialized via %s (%.2fs)",
                    _reranker_backend,
                    time.perf_counter() - t_rerank,
                )

                class BGEReranker(BaseDocumentCompressor):
                    top_n: int = 8

                    def compress_documents(
                        self,
                        documents: Sequence[Document],
                        query: str,
                        callbacks: Optional[Callbacks] = None,
                    ) -> Sequence[Document]:
                        if not documents:
                            return []
                        docs_seq = list(documents)
                        target_codes = _detect_target_codes(query)

                        if _should_skip_neural_rerank(query, docs_seq):
                            rerank_docs = list(docs_seq)
                            if target_codes:
                                filtered_docs = _filter_docs_by_codes(
                                    rerank_docs, target_codes
                                )
                                if filtered_docs:
                                    rerank_docs = filtered_docs
                            target_articles_skip: set[str] = set()
                            article_number = _extract_query_article_number(query)
                            if article_number:
                                target_articles_skip.add(
                                    _normalize_article_number(article_number)
                                )
                            range_match = _extract_article_range(query)
                            if range_match:
                                start, end = range_match
                                target_articles_skip.update(
                                    _normalize_article_number(str(n))
                                    for n in range(start, end + 1)
                                )
                            ranked = _rank_docs_by_legal_only(
                                query,
                                rerank_docs,
                                target_codes=target_codes,
                                target_articles=target_articles_skip,
                            )
                            scored_docs = [
                                (d, float(d.metadata.get("relevance_score", 0.0)))
                                for d in ranked
                            ]
                            scored_docs = _apply_diversity_penalty(
                                scored_docs,
                                penalty_step=getattr(
                                    config, "RETRIEVER_SAME_CODE_PENALTY_STEP", 0.1
                                ),
                            )
                            ranked_docs = [d for d, _ in scored_docs]
                            llm_candidate_n = max(
                                self.top_n,
                                getattr(config, "LLM_RERANK_CANDIDATES", self.top_n),
                            )
                            if getattr(config, "USE_LLM_RERANKER", False):
                                return _llm_rerank_documents(
                                    query,
                                    ranked_docs[:llm_candidate_n],
                                    getattr(config, "LLM_RERANK_TOP_N", self.top_n),
                                )
                            return ranked_docs[: self.top_n]

                        rerank_docs = list(documents)
                        if target_codes:
                            filtered_docs = _filter_docs_by_codes(
                                rerank_docs, target_codes
                            )
                            if filtered_docs:
                                rerank_docs = filtered_docs
                        target_articles: set[str] = set()
                        article_number = _extract_query_article_number(query)
                        if article_number:
                            target_articles.add(_normalize_article_number(article_number))
                        range_match = _extract_article_range(query)
                        if range_match:
                            start, end = range_match
                            target_articles.update(
                                _normalize_article_number(str(n))
                                for n in range(start, end + 1)
                            )
                        pairs = [[query, d.page_content] for d in rerank_docs]
                        if _reranker_backend == "jina":
                            texts = [d.page_content for d in rerank_docs]
                            with torch.inference_mode():
                                jina_results = _reranker_model.rerank(
                                    query, texts, top_n=len(texts)
                                )
                            scores = [0.0] * len(rerank_docs)
                            for item in jina_results:
                                ji = int(item.get("index", -1))
                                if 0 <= ji < len(scores):
                                    scores[ji] = float(
                                        item.get("relevance_score", 0.0)
                                    )
                        elif _reranker_backend == "flag_embedding":
                            try:
                                scores = _reranker_model.compute_score(pairs)
                            except ValueError as ve:
                                if "no padding token" in str(ve).lower():
                                    scores = [
                                        float(
                                            _reranker_model.compute_score([pair])[0]
                                        )
                                        for pair in pairs
                                    ]
                                else:
                                    raise
                        else:
                            scores = _reranker_model.predict(pairs)
                        if isinstance(scores, float):
                            scores = [scores]
                        scored_docs = []
                        for i, doc in enumerate(rerank_docs):
                            final_score = _apply_legal_score(
                                query,
                                doc,
                                float(scores[i]),
                                target_codes=target_codes,
                                target_articles=target_articles,
                            )
                            doc.metadata["relevance_score"] = final_score
                            scored_docs.append((doc, final_score))
                        scored_docs = _apply_diversity_penalty(
                            scored_docs,
                            penalty_step=getattr(
                                config, "RETRIEVER_SAME_CODE_PENALTY_STEP", 0.1
                            ),
                        )
                        ranked_docs = [d for d, _ in scored_docs]
                        llm_candidate_n = max(
                            self.top_n,
                            getattr(config, "LLM_RERANK_CANDIDATES", self.top_n),
                        )
                        if getattr(config, "USE_LLM_RERANKER", False):
                            return _llm_rerank_documents(
                                query,
                                ranked_docs[:llm_candidate_n],
                                getattr(config, "LLM_RERANK_TOP_N", self.top_n),
                            )
                        return ranked_docs[: self.top_n]

                compressor = BGEReranker(
                    top_n=getattr(config, "RETRIEVER_TOP_K_AFTER_RERANK", 8)
                )
                retr = ContextualCompressionRetriever(
                    base_compressor=compressor,
                    base_retriever=retr,
                )
            except Exception as e:
                logger.error(
                    "Reranker BGE-M3 failed: %s (using retrieval without reranking)",
                    e,
                    exc_info=True,
                )
                if getattr(config, "RERANKER_MANDATORY", True):
                    raise

        # Trim context before LLM
        retr = _TrimRetriever(
            base_retriever=retr,
            max_docs=getattr(config, "CONTEXT_MAX_DOCS", 8),
            max_chars_per_doc=getattr(config, "CONTEXT_MAX_CHARS_PER_DOC", 1800),
            max_tokens_per_doc=getattr(config, "CONTEXT_MAX_TOKENS_PER_DOC", 0),
            max_total_tokens=getattr(config, "CONTEXT_MAX_TOTAL_TOKENS", 0),
        )

        _retriever_instance = retr

    # Call outside lock — _ensure_latency_patches calls get_llm() which needs _init_lock (avoid deadlock)
    _ensure_latency_patches()
    return _retriever_instance


def get_retriever_for_coverage(top_k: int | None = None):
    """Return retriever with a wider final trim for retrieval benchmarks.

    This keeps the same retrieval stack but avoids cutting results down to the
    LLM context limit before coverage metrics are computed.
    """
    retriever = get_retriever()
    desired_top_k = max(1, int(top_k or getattr(config, "RETRIEVER_WIDE_K", 10)))
    if isinstance(retriever, _TrimRetriever) and retriever.max_docs < desired_top_k:
        return _TrimRetriever(
            base_retriever=retriever.base_retriever,
            max_docs=desired_top_k,
            max_chars_per_doc=retriever.max_chars_per_doc,
            max_tokens_per_doc=retriever.max_tokens_per_doc,
            max_total_tokens=0,
        )
    return retriever


def get_llm(model_override: str | None = None):
    if model_override:
        if "/" in model_override:
            from langchain_openai import ChatOpenAI
            return ChatOpenAI(
                api_key=config.OPENROUTER_API_KEY,
                base_url=config.OPENROUTER_BASE_URL,
                model=model_override,
                temperature=0.1,
                max_tokens=config.LLM_MAX_TOKENS or 2048,
                default_headers=config.build_openrouter_default_headers(),
            )
        else:
            from langchain_groq import ChatGroq
            return ChatGroq(
                api_key=config.GROQ_API_KEY,
                model_name=model_override,
                temperature=0.1,
                max_tokens=config.LLM_MAX_TOKENS,
            )
    global _llm_instance
    if _llm_instance is None:
        with _init_lock:
            if _llm_instance is None:
                logger.info("🚀 [START] LLM Initialization")
                t0 = time.perf_counter()
                try:
                    _llm_backend = getattr(config, "LLM_BACKEND", "groq").lower()
                    if _llm_backend == "groq":
                        try:
                            from langchain_groq import ChatGroq  # type: ignore[import]
                        except Exception as e:  # pragma: no cover
                            raise SystemExit(
                                "Для использования облачного Groq установите пакет 'langchain-groq':\n"
                                "  pip install langchain-groq\n"
                                f"Текущая ошибка импорта: {e}"
                            )
                        groq_api_key = config.GROQ_API_KEY or os.environ.get(
                            "GROQ_API_KEY"
                        )
                        if not groq_api_key:
                            raise SystemExit(
                                "Задайте GROQ_API_KEY для облачного Groq (gsk_...): export GROQ_API_KEY=..."
                            )
                        _llm_instance = ChatGroq(
                            groq_api_key=groq_api_key,
                            model_name=config.LLM_MODEL,
                            temperature=config.LLM_TEMPERATURE,
                            max_tokens=config.LLM_MAX_TOKENS,
                        )
                        logger.info(
                            "✅ [SUCCESS] LLM Initialization (Groq, model=%s) (%.2fs)",
                            config.LLM_MODEL,
                            time.perf_counter() - t0,
                        )
                    elif _llm_backend == "openrouter":
                        try:
                            from langchain_openai import ChatOpenAI
                        except Exception as e:  # pragma: no cover
                            raise SystemExit(
                                "Для использования OpenRouter установите пакет 'langchain-openai':\n"
                                "  pip install langchain-openai openai\n"
                                f"Текущая ошибка импорта: {e}"
                            )
                        openrouter_api_key = os.environ.get("OPENROUTER_API_KEY") or os.environ.get(
                            "OPENAI_API_KEY"
                        )
                        if not openrouter_api_key:
                            raise SystemExit(
                                "Задайте OPENROUTER_API_KEY для OpenRouter: export OPENROUTER_API_KEY=..."
                            )

                        base_url = (
                            config.OPENROUTER_BASE_URL
                            or os.environ.get("OPENROUTER_BASE_URL")
                            or os.environ.get("OPENAI_BASE_URL")
                            or "https://openrouter.ai/api/v1"
                        ).strip()

                        _llm_instance = ChatOpenAI(
                            api_key=openrouter_api_key,
                            base_url=base_url,
                            model=config.LLM_MODEL,
                            temperature=config.LLM_TEMPERATURE,
                            max_tokens=config.LLM_MAX_TOKENS,
                            default_headers=config.build_openrouter_default_headers(),
                        )
                        logger.info(
                            "✅ [SUCCESS] LLM Initialization (OpenRouter, model=%s, base_url=%s) (%.2fs)",
                            config.LLM_MODEL,
                            base_url,
                            time.perf_counter() - t0,
                        )
                    elif _llm_backend == "ollama_cloud":
                        try:
                            from langchain_openai import ChatOpenAI
                        except Exception as e:  # pragma: no cover
                            raise SystemExit(
                                "Для использования Ollama Cloud установите пакет 'langchain-openai':\n"
                                "  pip install langchain-openai openai\n"
                                f"Текущая ошибка импорта: {e}"
                            )
                        ollama_api_key = os.environ.get("OLLAMA_API_KEY")
                        if not ollama_api_key:
                            raise SystemExit(
                                "Задайте OLLAMA_API_KEY для Ollama Cloud: export OLLAMA_API_KEY=..."
                            )
                        _llm_instance = ChatOpenAI(
                            api_key=ollama_api_key,
                            base_url="https://ollama.com/v1",
                            model=config.LLM_MODEL,
                            temperature=config.LLM_TEMPERATURE,
                            max_tokens=config.LLM_MAX_TOKENS,
                        )
                        logger.info(
                            "✅ [SUCCESS] LLM Initialization (Ollama Cloud, model=%s) (%.2fs)",
                            config.LLM_MODEL,
                            time.perf_counter() - t0,
                        )
                    elif _llm_backend == "hf_peft":
                        _llm_instance = _make_hf_generation_pipeline()
                    else:
                        from langchain_ollama import OllamaLLM

                        _llm_instance = OllamaLLM(
                            model=config.LLM_MODEL,
                            temperature=config.LLM_TEMPERATURE,
                            base_url=config.OLLAMA_BASE_URL,
                            num_predict=config.LLM_MAX_TOKENS,
                        )
                        logger.info(
                            "✅ [SUCCESS] LLM Initialization (Ollama, model=%s) (%.2fs)",
                            config.LLM_MODEL,
                            time.perf_counter() - t0,
                        )
                except Exception as e:
                    elapsed = time.perf_counter() - t0
                    logger.error(
                        "❌ [FAIL] LLM Initialization (%.2fs): %s",
                        elapsed,
                        e,
                        exc_info=True,
                    )
                    if _is_connection_failure(e):
                        reset_instances()
                    raise
                if not isinstance(_llm_instance, CircuitBreakerProxy):
                    _llm_instance = CircuitBreakerProxy(
                        _llm_instance,
                        _groq_breaker,
                        sync_methods={"invoke", "batch"},
                        async_methods={"ainvoke", "abatch"},
                        stream_methods={"stream", "astream"},
                    )
    return _llm_instance


def _get_llm_runnable(model_override: str | None = None):
    """Return the underlying Runnable LLM for LCEL chain composition.

    `get_llm()` may return a CircuitBreakerProxy, which is fine for direct
    `invoke()` calls but is not itself a Runnable for `PromptTemplate | llm`.
    """
    llm = get_llm(model_override)
    return llm._target if isinstance(llm, CircuitBreakerProxy) else llm


def _ensure_latency_patches() -> None:
    # Patch once, when heavy components exist. Add diagnostic logging for each step.
    if getattr(_ensure_latency_patches, "_done", False):
        return
    if not _disable_pinecone:
        try:
            emb = get_embeddings()
            original_embed_query = emb.embed_query

            @latency.measure_latency("embedding")
            def wrapped_embed_query(*args, **kwargs):
                logger.info("🚀 [START] Embedding Query")
                t0 = time.perf_counter()
                try:
                    out = original_embed_query(*args, **kwargs)
                    logger.info(
                        "✅ [SUCCESS] Query Embedded (%.2fs)", time.perf_counter() - t0
                    )
                    return out
                except Exception as exc:
                    logger.error("Embedding failed: %s", exc, exc_info=True)
                    raise

            emb.embed_query = wrapped_embed_query
        except Exception:
            pass

    try:
        original_trim_get_docs = _TrimRetriever._get_relevant_documents

        @latency.measure_latency("vector_search")
        def wrapped_trim_get_docs(self, *args, **kwargs):
            logger.info("🚀 [START] Pinecone Vector Search")
            t0 = time.perf_counter()
            try:
                docs = original_trim_get_docs(self, *args, **kwargs)
                elapsed = time.perf_counter() - t0
                logger.info("✅ [SUCCESS] Retrieved %d chunks (%.2fs)", len(docs), elapsed)
                return docs
            except Exception as exc:
                logger.error("Pinecone Vector Search failed: %s", exc, exc_info=True)
                raise

        _TrimRetriever._get_relevant_documents = wrapped_trim_get_docs
    except Exception:
        pass

    try:
        llm = get_llm()
        original_llm_invoke = llm.__class__.invoke

        @latency.measure_latency("llm_inference")
        def wrapped_llm_invoke(self, *args, **kwargs):
            logger.info("🚀 [START] LLM Prompt Construction")
            logger.info("🚀 [START] LLM Inference")
            t0 = time.perf_counter()
            try:
                out = original_llm_invoke(self, *args, **kwargs)
                logger.info(
                    "✅ [SUCCESS] LLM Response Generated (%.2fs)", time.perf_counter() - t0
                )
                return out
            except Exception as exc:
                logger.error("LLM Inference failed: %s", exc, exc_info=True)
                raise

        llm.__class__.invoke = wrapped_llm_invoke
    except (Exception, SystemExit):
        pass

    _ensure_latency_patches._done = True


LEGAL_REASONING_GUIDANCE = """
Обязательная юридическая логика ответа:
1. Сначала выдели ключевые юридически значимые факты из вопроса.
2. Затем проанализируй только те нормы, которые прямо есть в контексте. ОБЯЗАТЕЛЬНО цитируй номера статей, пунктов и названия нормативных актов для каждого довода.
3. После этого явно сопоставь каждый существенный факт с диспозицией соответствующей статьи.
4. Только потом формулируй итоговый вывод.
5. Не пропускай этап сопоставления фактов и нормы, даже если ответ кажется очевидным.
6. Если для любого шага не хватает нормы в контексте, прямо укажи это и не достраивай вывод предположениями.

Обязательные блоки в ответе после краткого прямого вывода:
- "Ключевые юридические факты:"
- "Анализ норм (ОБЯЗАТЕЛЬНО с указанием статей и НПА):"
- "Сопоставление фактов и нормы:"
- "Вывод:"
"""


UNIVERSAL_PROMPT_TEMPLATE = """Ты — сильный Legal AI по законодательству Республики Казахстан.
ОТВЕЧАЙ ИСКЛЮЧИТЕЛЬНО на основе предоставленного контекста.
Если нужной нормы в контексте нет — отвечай ровно:
"Информация не найдена в доступных текстах законов."
Никогда не подставляй нормы из других кодексов и не выдумывай статьи, части статей, исключения или выводы.

Строгие правила качества:
1. Всегда начинай строго с фразы:
   "Это не официальная юридическая консультация. Информация только из базы."
2. Следующей строкой пиши:
   "Здравствуйте!"
3. Если вопрос задан на русском — отвечай только на русском.
   Если вопрос задан на казахском — отвечай только на казахском.
4. Сначала дай прямой вывод по сути вопроса.
   Не пиши расплывчатое "всё зависит", если из контекста можно дать ясный ответ.
5. Затем отвечай в однородном плавном стиле, как опытный юридический ИИ:
   - короткое объяснение сути,
   - применимые нормы,
   - что лучше не использовать,
   - краткая рекомендация,
   - источники.
6. Если называешь конкретный закон, кодекс или статью, кратко объясняй, почему именно эта норма здесь применяется.
7. Не перегружай ответ лишними нормами.
   Лучше 2-6 точных норм с объяснением, чем длинный список без логики.
8. Не смешивай разные отрасли права без опоры в контексте.
9. Обязательно проведи юридический анализ по шагам: факты -> нормы -> сопоставление -> вывод.
10. В КАЖДОМ ответе ОБЯЗАТЕЛЬНО цитируй и указывай номера статей, пунктов и частей применимых законов и кодексов. Ответ без конкретных статей недопустим!

{legal_reasoning_guidance}

Предпочтительный шаблон ответа:
- "Это не официальная юридическая консультация. Информация только из базы."
- "Здравствуйте!"
- Прямой вывод.
- "Ключевые юридические факты:"
- "Анализ норм:"
- "Сопоставление фактов и нормы:"
- "Вывод:"
- "Основу вашей позиции составляют следующие нормы:"
- Плавное объяснение применимых норм.
- "Что лучше не использовать ..."
- "Краткая рекомендация:"
- "Источники:"

{chat_history}
Контекст:
{context}

Вопрос: {input}

Ответ:"""

CRIMINAL_PROMPT_TEMPLATE = """Ты — эксперт по Уголовному кодексу РК.
• Гражданский кодекс РК (Общая и Особенная части)
• Трудовой кодекс РК
• Налоговый кодекс РК
• Кодекс об административных правонарушениях РК (КоАП)
• Уголовный кодекс РК (УК РК)
• Уголовно-процессуальный кодекс РК (УПК РК)
• Гражданский процессуальный кодекс РК (ГПК РК)
• Кодекс о браке (супружестве) и семье РК
• Кодекс о здоровье народа и системе здравоохранения РК
• Предпринимательский кодекс РК
• Социальный кодекс РК
• Кодекс РК об административных процедурах
• Закон РК о государственных закупках
• Закон РК об исполнительном производстве и статусе судебных исполнителей

ОТВЕЧАЙ ИСКЛЮЧИТЕЛЬНО на основе предоставленного ниже контекста.
НИКОГДА НЕ ПРИДУМЫВАЙ номера статей, названия законов, даты, санкции, выводы или факты, которых нет в контексте.
Если в контексте есть релевантные статьи, используй их для ответа.
Если в контексте СОВСЕМ нет релевантной информации — отвечай ровно одной строкой:
"Информация не найдена в доступных текстах законов."

Строгие правила ответа, учитывающие все сферы жизни:
1. Всегда начинай ответ строго с фразы:
   "Это не официальная юридическая консультация. Информация только из базы."
2. Если вопрос на казахском — отвечай ТОЛЬКО на казахском, используя точные формулировки НҚА.
   Если вопрос на русском — отвечай ТОЛЬКО на русском.
3. Цитируй статьи дословно, указывай:
   • точный номер статьи и часть (если есть)
   • название кодекса/закона
   • точную санкцию (если применимо)
   • источник (название файла или кодекса)
4. Если в вопросе диапазон статей (например, ст. 120–135 УК РК) —
   перечисляй ВСЕ релевантные статьи из контекста с кратким описанием и
   санкцией.
5. Для любой сферы всегда разбирай, если применимо (и только если в
   контексте):
   - Конституционное право: права граждан, принципы, нормы Конституции.
   - Гражданское право (ГК): договоры, имущество, обязательства, ущерб, компенсация.
   - Трудовое право (ТК): принципы, права работников/работодателей, нарушения, компенсации.
   - Налоговое право: налоги, нарушения, штрафы, декларации.
   - Административное право (КоАП): нарушения, штрафы, процедуры.
   - Уголовное право (УК): состав преступления (объект, объективная
     сторона, субъект, субъективная сторона), ауырлататын және
     жеңілдететін мән-жайлар, санкция дословно.
   - Процессуальное (УПК/ГПК): процедуры суда, доказательства, сроки.
   - Семейное право: брак, развод, дети, алименты.
   - Здравоохранение: права пациентов, обязанности медработников, нарушения.
   - Предпринимательство: бизнес, регистрация, нарушения.
   - Социальное: пособия, пенсии, социальная защита.
   - Административные процедуры: обращения, сроки, права граждан.
   - Госзакупки: процедуры, нарушения, ответственность.
   - Исполнительное производство: взыскание долгов, действия исполнителей.
6. Никогда не применяй нормы из одной сферы/кодекса к другой.
   Если в контексте нет точной санкции/освобождения/смягчения для данной статьи — пиши:
   "Санкция / освобождение / смягчение в контексте не указаны."
7. Если вопрос касается нескольких сфер — ищи и указывай нормы из каждого релевантного кодекса (например, УК + КоАП).

{chat_history}
Контекст (с источниками, номерами статей и кодексами):
{context}

Вопрос: {input}

Ответ (строго следуй правилам выше, цитируй дословно, указывай статью, часть, кодекс и источник):"""

CRIMINAL_PROMPT_TEMPLATE = """Ты — эксперт по Уголовному кодексу РК / ҚР Қылмыстық кодексінің сарапшысы.
ОТВЕЧАЙ ТОЛЬКО на основе контекста ниже. ТЕК төмендегі контекст негізінде жауап беріңіз.
Если в контексте нет нужной статьи или информации — отвечай/Егер контексте ақпарат болмаса, жауап беріңіз:
"Информация не найдена в доступных текстах законов." или "Ақпарат қолжетімді заң мәтіндерінен табылмады."

Обязательные правила / Міндетті ережелер:
1. Начинай строго с / Қатаң түрде мынадан бастаңыз: "Это не официальная юридическая консультация. Информация только из базы." немесе "Бұл ресми заңдық кеңес емес. Ақпарат тек базадан алынған."
2. Отвечай на казахском, если вопрос на казахском; на русском — если на русском. Сұрақ қазақша болса, қазақша жауап беріңіз; орысша болса, орысша.
3. Для каждого пункта вопроса отвечай по порядку, нумеруя 1), 2), 3) и т.д.
4. Всегда указывай точную статью УК РК, часть и дословную цитату / Әрқашан ҚР ҚК бабын, бөлігін және дәл дәйексөзді көрсетіңіз.
5. Разбирай состав преступления ТОЛЬКО если статья есть в контексте / Қылмыс құрамын ТЕК бап контексте болса ғана талдаңыз:
   - Объект
   - Объективті жағы / Объективная сторона
   - Субъект
   - Субъективті жағы / Субъективная сторона
6. Ауырлататын және жеңілдететін мән-жайлар (Отягчающие и смягчающие обстоятельства) — ТОЛЬКО если они указаны в статье.
7. Санкцию цитируй дословно / Санкцияны дәлме-дәл келтіріңіз.
8. Перед итогом обязательно выполни юридическую последовательность: факты -> анализ нормы -> сопоставление с диспозицией -> вывод.

{legal_reasoning_guidance}

{chat_history}
Контекст (с источниками / дереккөздермен):
{context}

Вопрос (разбери по пунктам / тармақтар бойынша талда):
{input}

Ответ (нумеруй пункты, цитируй дословно, указывай статью и источник / тармақтарды нөмірле, дәл дәйексөз келтір, бапты және дереккөзді көрсет):"""

RANGE_PROMPT_TEMPLATE = """Ты — точный ассистент по УК РК.
ОТВЕЧАЙ ИСКЛЮЧИТЕЛЬНО из контекста.
Если в контексте нет нужных статей — отвечай одной строкой:
"Информация не найдена в доступных текстах законов."

Правила:
• Начинай строго с: "Это не официальная юридическая консультация. Информация только из базы."
• Если в вопросе диапазон статей (например, 120–135) — перечисляй ВСЕ
  релевантные статьи из контекста с номерами и кратким описанием.
• Цитируй санкции и ключевые части дословно.
• Указывай источник (название кодекса) и номер статьи.
• Для каждой статьи соблюдай порядок: факты вопроса -> анализ нормы -> сопоставление -> вывод.

{legal_reasoning_guidance}

{chat_history}
Контекст:
{context}

Вопрос:
{input}

Ответ (перечисли статьи из диапазона, если они есть, цитируй дословно):"""

GENERAL_PROMPT = PromptTemplate.from_template(
    "Дай краткую справку по теории права РК на основе {context}."
)
CASE_PROMPT = PromptTemplate.from_template(
    "Проведи юридический анализ ситуации. Если не хватает данных для "
    "оценки нарушения — уточни их у пользователя. Контекст: {context}"
)

UNIVERSAL_PROMPT = PromptTemplate(
    input_variables=["chat_history", "context", "input"],
    partial_variables={"legal_reasoning_guidance": LEGAL_REASONING_GUIDANCE.strip()},
    template=UNIVERSAL_PROMPT_TEMPLATE,
)
CRIMINAL_PROMPT = PromptTemplate(
    input_variables=["chat_history", "context", "input"],
    partial_variables={"legal_reasoning_guidance": LEGAL_REASONING_GUIDANCE.strip()},
    template=CRIMINAL_PROMPT_TEMPLATE,
)
RANGE_PROMPT = PromptTemplate(
    input_variables=["chat_history", "context", "input"],
    partial_variables={"legal_reasoning_guidance": LEGAL_REASONING_GUIDANCE.strip()},
    template=RANGE_PROMPT_TEMPLATE,
)


@latency.measure_latency("prompt_template_build")
def _select_prompt(question: str, intent: str = None) -> PromptTemplate:
    if _extract_article_range(question):
        return RANGE_PROMPT
    if intent in {"GENERAL_LEGAL", "CASE_SPECIFIC"}:
        # Most legal questions are not criminal-law questions.
        # Route them through the universal legal template instead of the
        # criminal-only template to avoid wrong answer structure and bias.
        return UNIVERSAL_PROMPT
    q = question or ""
    if _is_criminal_query(q) or re.search(
        r"(?:ст\.?\s*\d|статья\s*\d|бап\s*\d)", q, re.IGNORECASE
    ):
        return CRIMINAL_PROMPT
    return UNIVERSAL_PROMPT


try:
    from langchain.chains import create_retrieval_chain
    from langchain.chains.combine_documents import create_stuff_documents_chain
except ImportError:
    from langchain_classic.chains import create_retrieval_chain
    from langchain_classic.chains.combine_documents import create_stuff_documents_chain


def _fill_missing_metadata(docs):
    # docs is a list of Document objects coming from the retriever
    if not isinstance(docs, list):
        return docs

    for d in docs:
        if "article_number" not in d.metadata:
            d.metadata["article_number"] = "Н/Д"
        if "code_ru" not in d.metadata:
            d.metadata["code_ru"] = "Неизвестный источник / Белгісіз дереккөз"
        if "source" not in d.metadata:
            d.metadata["source"] = "Неизвестно / Белгісіз"
    return docs


def _make_qa_chain(prompt: PromptTemplate) -> Any:
    # Define a prompt to format each document including metadata
    document_prompt = PromptTemplate(
        input_variables=["page_content", "source", "article_number", "code_ru"],
        template="[{code_ru} | ст. {article_number} | {source}]\n{page_content}",
    )

    # LCEL pipeline: Retriever -> Document Chain -> Retrieval Chain
    question_answer_chain = create_stuff_documents_chain(
        _get_llm_runnable(), prompt, document_prompt=document_prompt
    )

    # Wrap retriever to ensure metadata exists before docs hit the document chain
    retriever_with_safeguard = get_retriever() | _fill_missing_metadata

    return create_retrieval_chain(retriever_with_safeguard, question_answer_chain)


_QA_CHAINS = None


def _get_qa_chains() -> dict[str, Any]:
    global _QA_CHAINS
    if _QA_CHAINS is None:
        with _init_lock:
            if _QA_CHAINS is None:
                _QA_CHAINS = {
                    "universal": _make_qa_chain(UNIVERSAL_PROMPT),
                    "criminal": _make_qa_chain(CRIMINAL_PROMPT),
                    "range": _make_qa_chain(RANGE_PROMPT),
                }
    return _QA_CHAINS


def _history_str(history: Optional[List[dict]] = None) -> str:
    if not history:
        return ""
    s = "История предыдущих сообщений:\n"
    for msg in history:
        role = "Пользователь" if msg.get("role") == "user" else "Ассистент"
        s += f"{role}: {msg.get('content')}\n"
    return s + "\n"


def _history_cache_key(history: Optional[List[dict]] = None) -> tuple[tuple[str, str], ...]:
    if not history:
        return ()
    return tuple(
        (
            str(msg.get("role") or ""),
            str(msg.get("content") or ""),
        )
        for msg in history
    )


def _cache_key_digest(query: str, history_key: tuple[tuple[str, str], ...], intent: str | None) -> str:
    payload = json.dumps(
        {
            "query": query.strip(),
            "history": history_key,
            "intent": intent or "",
        },
        ensure_ascii=False,
        sort_keys=True,
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _offline_query_terms(query: str) -> list[str]:
    terms = [
        token
        for token in re.findall(r"[0-9A-Za-zА-Яа-яӘәҒғҚқҢңӨөҰұҮүҺһІі]+", str(query or "").lower())
        if len(token) > 2
    ]
    return list(dict.fromkeys(terms))


def _score_offline_doc(query_terms: list[str], doc: Document) -> float:
    meta = doc.metadata or {}
    haystack = " ".join(
        [
            str(meta.get("code_ru") or ""),
            str(meta.get("article_number") or ""),
            str(meta.get("article_title") or ""),
            str(doc.page_content or ""),
        ]
    ).lower()
    score = 0.0
    for term in query_terms:
        if term in haystack:
            score += 1.0
    article_number = str(meta.get("article_number") or "").strip()
    if article_number and article_number in query_terms:
        score += 1.5
    if any(keyword in haystack for keyword in ("ответствен", "наказан", "краж", "штраф", "преступлен")):
        score += 0.25
    return score


def _build_offline_extractive_answer(
    query: str,
    docs: List[Document],
    history: Optional[List[dict]] = None,
    intent: str | None = None,
) -> dict:
    docs = _fill_missing_metadata(list(docs))
    query_terms = _offline_query_terms(query)
    ranked_docs = sorted(
        docs,
        key=lambda doc: _score_offline_doc(query_terms, doc),
        reverse=True,
    )
    top_docs = ranked_docs[: min(3, len(ranked_docs))]

    if not top_docs:
        return _finalize_qa_result(
            {
                "result": (
                    "Офлайн-режим: релевантные статьи не найдены в локальном корпусе. "
                    "Уточните вопрос или проверьте корпус данных."
                ),
                "source_documents": [],
                "retrieval_method": "offline_extractive",
            },
            query,
            intent,
        )

    intent_label = {
        "CRIMINAL": "криминальный",
        "GENERAL_LEGAL": "правовой",
        "PROCEDURAL": "процессуальный",
    }.get(str(intent or "").upper(), "правовой")

    lines = [
        f"Офлайн-{intent_label} ответ на основе локального корпуса:",
        "Наиболее релевантные статьи:",
    ]
    for doc in top_docs:
        meta = doc.metadata or {}
        code_ru = str(meta.get("code_ru") or "Неизвестный источник").strip()
        article_number = str(meta.get("article_number") or "Н/Д").strip()
        snippet = _truncate_text_to_token_budget(
            str(doc.page_content or "").strip().replace("\n", " "),
            90,
            suffix="...",
        )
        lines.append(f"- [{code_ru} | ст. {article_number}] {snippet}")

    if history:
        lines.append("Контекст диалога учтён, но ответ построен без LLM.")

    return _finalize_qa_result(
        {
            "result": "\n".join(lines).strip(),
            "source_documents": top_docs,
            "retrieval_method": "offline_extractive",
        },
        query,
        intent,
    )


def _build_offline_bm25_retriever(top_k: int):
    from langchain_community.retrievers import BM25Retriever

    chunks = _load_bm25_chunks() or []
    if not chunks:
        return None

    if _ensure_nltk():

        def hybrid_tokenizer(text):
            if _is_kz_query(text):
                import nltk

                return [t for t in nltk.word_tokenize(text.lower()) if t.isalnum()]
            return bm25_preprocess_func(text) or []

        return BM25Retriever.from_documents(
            chunks,
            preprocess_func=hybrid_tokenizer,
            k=top_k,
        )

    return BM25Retriever.from_documents(chunks, k=top_k)


def _offline_focus_article_docs(query: str) -> list[Document]:
    focus_articles = {
        _normalize_article_number(article)
        for article in _focus_articles_from_query(query)
        if _normalize_article_number(article)
    }
    target_codes = set(_detect_target_codes(query))
    if not focus_articles:
        return []

    chunks = _load_bm25_chunks() or []
    if not chunks:
        return []

    docs = [
        doc
        for doc in chunks
        if _normalize_article_number((doc.metadata or {}).get("article_number")) in focus_articles
        and (
            not target_codes
            or str((doc.metadata or {}).get("code_ru") or "").strip() in target_codes
        )
    ]
    return docs


def invoke_qa_with_context(
    query: str,
    context_docs: List[Document],
    history: Optional[List[dict]] = None,
    intent: str = None,
    model_override: Optional[str] = None,
) -> dict:
    """Run QA using a pre-retrieved list of documents (no retriever). Used by agentic workflow."""
    _ensure_latency_patches()
    docs = _fill_missing_metadata(list(context_docs))
    prompt = _select_prompt(query, intent=intent)
    if prompt is RANGE_PROMPT:
        _get_qa_chains()["range"]
    elif prompt is CRIMINAL_PROMPT:
        _get_qa_chains()["criminal"]
    else:
        _get_qa_chains()["universal"]
    # Retrieval chain expects "input", "chat_history", and injects "context"
    # from retriever. Our chain was built with retriever; we invoke the
    # inner question_answer_chain by passing context directly.
    # create_retrieval_chain(retriever, question_answer_chain) passes
    # retriever output as "context".
    # So we need to invoke the part that takes context. The retrieval chain
    # does: context = retriever.invoke(input), then
    # question_answer_chain.invoke({**input, "context": context}).
    # So we can't easily invoke just the question_answer_chain without the
    # retriever. Alternative: use the same prompt and
    # create_stuff_documents_chain and invoke with context=docs.
    document_prompt = PromptTemplate(
        input_variables=["page_content", "source", "article_number", "code_ru"],
        template="[{code_ru} | ст. {article_number} | {source}]\n{page_content}",
    )
    question_answer_chain = create_stuff_documents_chain(
        _get_llm_runnable(model_override=model_override), prompt, document_prompt=document_prompt
    )
    res = question_answer_chain.invoke(
        {
            "context": docs,
            "input": query,
            "chat_history": _history_str(history),
        }
    )
    return _finalize_qa_result(
        {"result": res, "source_documents": docs},
        query,
        intent,
    )


def _invoke_qa_impl(
    query: str,
    history: Optional[List[dict]],
    intent: str | None,
    model_override: Optional[str] = None,
) -> dict:
    if config.LEGAL_RAG_OFFLINE_QA:
        try:
            docs = _offline_focus_article_docs(query)
            if docs:
                docs = _prioritize_docs(
                    query,
                    docs,
                    target_articles=list(_focus_articles_from_query(query)),
                    target_codes=_detect_target_codes(query),
                    limit=getattr(config, "RETRIEVER_WIDE_K", 10),
                )
            else:
                retriever = _build_offline_bm25_retriever(
                    getattr(config, "RETRIEVER_WIDE_K", 10)
                )
                docs = (
                    _multi_query_retrieve(retriever, query)
                    if retriever is not None
                    else []
                )
                docs = _prioritize_docs(
                    query,
                    docs,
                    target_articles=list(_focus_articles_from_query(query)),
                    target_codes=_detect_target_codes(query),
                    limit=getattr(config, "RETRIEVER_WIDE_K", 10),
                )
        except Exception as exc:
            logger.warning(
                "[OFFLINE] Retrieval failed in offline QA mode: %s",
                exc,
                exc_info=True,
            )
            docs = []
        return _finalize_qa_result(
            _build_offline_extractive_answer(query, docs, history, intent),
            query,
            intent,
        )

    _ensure_latency_patches()

    if intent == "SOCIAL":
        llm = get_llm()
        s_history = _history_str(history)
        prompt_text = (
            "Ты — дружелюбный юридический ассистент Legally / Сіз Legally мейірімді заң көмекшісісіз. "
            "Ответь на приветствие или общий вопрос / Сәлемдесуге немесе жалпы сұраққа жауап беріңіз.\n"
            "Отвечай на том языке, на котором задан вопрос (қазақша немесе орысша).\n\n"
            f"{s_history}Вопрос/Сұрақ: {query}\nОтвет/Жауап:"
        )
        res = llm.invoke(prompt_text)
        return _finalize_qa_result(
            {
                "result": res.content if hasattr(res, "content") else str(res),
                "source_documents": [],
            },
            query,
            intent,
        )

    prompt = _select_prompt(query, intent=intent)
    if model_override:
        llm = get_llm(model_override)
        document_prompt = PromptTemplate(
            input_variables=["page_content", "source", "article_number", "code_ru"],
            template="[{code_ru} | ст. {article_number} | {source}]\n{page_content}",
        )
        chain = create_retrieval_chain(
            get_retriever(),
            create_stuff_documents_chain(llm, prompt, document_prompt=document_prompt)
        )
    else:
        if prompt is RANGE_PROMPT:
            chain = _get_qa_chains()["range"]
        elif prompt is CRIMINAL_PROMPT:
            chain = _get_qa_chains()["criminal"]
        else:
            chain = _get_qa_chains()["universal"]

    try:
        res = chain.invoke(
            {
                "input": query,
                "chat_history": _history_str(history),
            }
        )
    except Exception as exc:
        if _is_connection_failure(exc):
            logger.warning(
                "[OFFLINE] LLM-backed QA failed, falling back to extractive answer: %s",
                exc,
                exc_info=True,
            )
            try:
                docs = _offline_focus_article_docs(query)
                if not docs:
                    offline_retriever = _build_offline_bm25_retriever(
                        getattr(config, "RETRIEVER_WIDE_K", 10)
                    )
                    docs = (
                        _multi_query_retrieve(offline_retriever, query)
                        if offline_retriever is not None
                        else []
                    )
                    docs = _prioritize_docs(
                        query,
                        docs,
                        target_articles=list(_focus_articles_from_query(query)),
                        target_codes=_detect_target_codes(query),
                        limit=getattr(config, "RETRIEVER_WIDE_K", 10),
                    )
            except Exception as retrieval_exc:
                logger.warning(
                    "[OFFLINE] Fallback retrieval also failed: %s",
                    retrieval_exc,
                    exc_info=True,
                )
                docs = []
            return _finalize_qa_result(
                _build_offline_extractive_answer(query, docs, history, intent),
                query,
                intent,
            )
        raise

    answer = res.get("answer", "")
    source_documents = res.get("context", [])

    if not source_documents:
        logger.warning(
            "[FALLBACK] No documents retrieved, using internal knowledge "
            "for query: %s",
            query,
        )
        fallback_prompt = PromptTemplate.from_template(
            "⚠️ ВНИМАНИЕ: Информация не найдена в текущей базе данных. "
            "Ниже приведен общий правовой анализ на основе НПА РК:\n\n"
            "Вопрос: {input}\n\n"
            "Ответьте на основе вашего общего знания законодательства "
            "Республики Казахстан. Будьте точны и ссылайтесь на "
            "релевантные статьи."
        )
        fallback_chain = fallback_prompt | _get_llm_runnable()
        answer = fallback_chain.invoke({"input": query})
        retrieval_method = "internal_fallback"
    else:
        retrieval_method = "hybrid"

    return _finalize_qa_result(
        {
            "result": answer,
            "source_documents": source_documents,
            "retrieval_method": retrieval_method,
        },
        query,
        intent,
    )


@lru_cache(maxsize=512)
def _invoke_qa_cached(
    query_hash: str,
    query: str,
    intent: str | None,
) -> dict:
    _ = query_hash
    return _invoke_qa_impl(query, None, intent)


def invoke_qa(
    query: str, history: Optional[List[dict]] = None, intent: str = None, model_override: Optional[str] = None
) -> dict:
    if history or model_override:
        return _invoke_qa_impl(query, history, intent, model_override)
    history_key = _history_cache_key(history)
    query_hash = _cache_key_digest(query, history_key, intent)
    return _invoke_qa_cached(query_hash, query, intent)


_KZ_CHARS = set("әғқңөұүһі")
_KZ_COMMON_WORDS = (
    "және",
    "бойынша",
    "қылмыстық",
    "құрамы",
    "қылмысқа",
    "бап",
    "заң",
    "мән-жай",
    "ауырлататын",
    "жеңілдететін",
)


def _is_kz_query(query: str) -> bool:
    return any(ch in (query or "").lower() for ch in _KZ_CHARS)


def _is_kz_response(text: str) -> bool:
    t = (text or "").lower()
    if any(ch in t for ch in _KZ_CHARS):
        return True
    return any(word in t for word in _KZ_COMMON_WORDS)


def _extract_article_numbers_from_docs(docs: List[Document]) -> set[str]:
    nums: set[str] = set()
    for d in docs or []:
        art = (d.metadata.get("article_number") or "").strip()
        if art.isdigit():
            nums.add(art)
    return nums


def _extract_article_numbers_from_text(text: str) -> set[str]:
    nums: set[str] = set()
    if not text:
        return nums
    for m in re.finditer(r"(?:статья|ст\.|ст|бап)\s*(\d{1,4})", text.lower()):
        nums.add(m.group(1))
    return nums


ANALYSIS_PROMPT_TEMPLATE = (
    "Ты — юридический эксперт по законодательству Казахстана. Твоя задача — "
    "проанализировать текст документа (или его часть) и выявить правовые "
    "риски, нарушения и дать рекомендации.\n\n"
    "В ответе придерживайся следующей структуры:\n\n"
    "### Правовые риски\n"
    "1. [Название риска]\n"
    "   - Описание: [подробное описание]\n"
    "   - Нормативный акт: [закон/статья]\n"
    "   - Уровень риска: [высокий/средний/низкий]\n"
    "   - Рекомендация: [предложение по исправлению]\n\n"
    "### Неясные формулировки\n"
    "1. [Формулировка]\n"
    "   - Проблема: [в чем неясность]\n"
    "   - Рекомендация: [как переформулировать]\n"
    "   - Уровень важности: [высокий/средний/низкий]\n\n"
    "### Возможные нарушения\n"
    "1. [Описание нарушения]\n"
    "   - Нормативный акт: [закон/статья]\n"
    "   - Последствия: [возможные санкции]\n"
    "   - Рекомендация: [как избежать]\n\n"
    "### Рекомендации\n"
    "[Список конкретных рекомендаций по исправлению документа]\n\n"
    "### Заключение\n"
    "[Общая сводка по документу с выводами]\n\n"
    "Документ:\n"
    "{text}\n\n"
    "Ответ:"
)

ANALYSIS_PROMPT = PromptTemplate.from_template(ANALYSIS_PROMPT_TEMPLATE)


def _finalize_qa_result(
    payload: dict,
    question: str,
    intent: str | None = None,
) -> dict:
    from ai_service.utils.ensure_citations import ensure_answer_citations

    payload["result"] = ensure_answer_citations(
        payload.get("result", ""),
        payload.get("source_documents") or [],
        question=question,
        intent=intent,
    )
    return payload


def validate_answer(question: str, response: str, sources: List[Document]) -> str:
    fallback = "Информация не найдена в доступных текстах законов."
    if not sources:
        return fallback

    if _is_kz_query(question) and not _is_kz_response(response):
        return fallback

    if _is_criminal_query(question):
        has_uk = any(
            (d.metadata.get("code_ru") or "").strip() in _uk_variants for d in sources
        )
        if not has_uk:
            return fallback

    mentioned = _extract_article_numbers_from_text(response or "")

    if _is_illegal_business_query(question):
        if "214" not in mentioned and "245" not in mentioned:
            return fallback

    if _is_pyramid_query(question):
        if "217" not in mentioned:
            return fallback

    if _needs_circumstances_query(question):
        r = (response or "").lower()
        if not any(
            token in r for token in ("ауырлататын", "жеңілдететін", "отягча", "смягча")
        ):
            return fallback

    return response


def clear_qa_cache() -> None:
    _invoke_qa_cached.cache_clear()


def build_streaming_qa_prompt(
    query: str,
    history: Optional[List[dict]] = None,
    intent: str | None = None,
) -> dict[str, Any]:
    _ensure_latency_patches()

    if intent == "SOCIAL":
        llm = get_llm()
        prompt_text = (
            "Ты — дружелюбный юридический ассистент Legally / Сіз Legally мейірімді заң көмекшісісіз. "
            "Ответь на приветствие или общий вопрос / Сәлемдесуге немесе жалпы сұраққа жауап беріңіз.\n"
            "Отвечай на том языке, на котором задан вопрос (қазақша немесе орысша).\n\n"
            f"{_history_str(history)}Вопрос/Сұрақ: {query}\nОтвет/Жауап:"
        )
        return {"prompt_text": prompt_text, "source_documents": [], "llm": llm}

    prompt = _select_prompt(query, intent=intent)
    if prompt is RANGE_PROMPT:
        _get_qa_chains()["range"]
    elif prompt is CRIMINAL_PROMPT:
        _get_qa_chains()["criminal"]
    else:
        _get_qa_chains()["universal"]

    docs = _fill_missing_metadata(get_retriever().invoke(query))
    context = "\n\n".join(_format_doc_for_prompt(doc) for doc in docs)
    prompt_text = prompt.format(
        input=query,
        chat_history=_history_str(history),
        context=context,
    )
    return {"prompt_text": prompt_text, "source_documents": docs, "llm": get_llm()}


def analyze_text(text: str) -> str:
    """Analyses the provided text using the configured LLM."""
    _ensure_latency_patches()
    chain = ANALYSIS_PROMPT | _get_llm_runnable(model_override=None)
    result = chain.invoke({"text": text})
    # Extract content string if it's an AIMessage
    return result.content if hasattr(result, "content") else str(result)


if __name__ == "__main__":
    question = "Статья 136 УК РК баланы ауыстыру"
    print(f"\nВопрос: {question}\n")
    docs = get_retriever().invoke(question)
    for i, doc in enumerate(docs[:5], 1):
        source = doc.metadata.get("source")
        code = doc.metadata.get("code_ru", "")
        art = doc.metadata.get("article_number", "")
        print(f"{i}. {source} | {code} ст.{art}")
        content = doc.page_content[:250].replace(chr(10), " ")
        print(f"   {content}...\n")
    result = invoke_qa(question)
    print("Ответ:", result["result"][:500])
