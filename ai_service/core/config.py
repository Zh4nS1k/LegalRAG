# config.py — Legal RAG: Pinecone, Adilet, все 20 документов

import os
import sys
from pathlib import Path

# Пути: ai_service/core/config.py -> parent.parent = ai_service
_THIS_DIR = Path(__file__).resolve().parent
AI_SERVICE_DIR = _THIS_DIR.parent
BASE_DIR = AI_SERVICE_DIR.parent  # repo root (LegalRAG)


# Load .env from ai_service first, then repo root (python-dotenv always available in this project)
def _load_dotenv():
    try:
        from dotenv import load_dotenv

        for p in (AI_SERVICE_DIR / ".env", BASE_DIR / ".env"):
            if p.exists():
                # Local service .env must win over inherited shell vars.
                load_dotenv(p, override=True)
                break
    except ImportError:
        pass


_load_dotenv()

# Prefer pydantic_settings when available; else read required vars from os.environ
try:
    from pydantic_settings import BaseSettings, SettingsConfigDict

    class EngineSettings(BaseSettings):
        PINECONE_API_KEY: str
        PINECONE_INDEX_NAME: str = "legally-index"
        PINECONE_NAMESPACE: str = "default"
        GROQ_API_KEY: str | None = None
        OPENROUTER_API_KEY: str | None = None
        HF_TOKEN: str | None = None
        REDIS_URL: str = "redis://localhost:6379/0"
        CACHE_ENABLED: bool = True
        CACHE_TTL_SECONDS: int = 3600
        model_config = SettingsConfigDict(
            env_file=str(AI_SERVICE_DIR / ".env"),
            env_file_encoding="utf-8",
            extra="ignore",
        )

    env_settings = EngineSettings()
except Exception as e:
    # Fallback: no pydantic_settings (e.g. wrong venv); require env vars to be set
    _pk = os.environ.get("PINECONE_API_KEY")
    _gk = os.environ.get("GROQ_API_KEY")
    if not _pk:
        sys.exit(
            f"\n[CRITICAL ERROR] Missing configuration.\n"
            f"Load .env or set PINECONE_API_KEY.\n"
            f"If using a venv, activate the one where you ran: pip install -r requirements.txt\n"
            f"  e.g. source venv/bin/activate  (from LegalRAG) or  .venv/bin/activate  (from ai_service)\n"
            f"Original: {e}\n"
        )
    env_settings = type(
        "Env",
        (),
        {
            "PINECONE_INDEX_NAME": os.environ.get(
                "PINECONE_INDEX_NAME", "legally-index"
            ),
            "PINECONE_NAMESPACE": os.environ.get("PINECONE_NAMESPACE", "default"),
            "PINECONE_API_KEY": _pk,
            "GROQ_API_KEY": _gk,
            "HF_TOKEN": os.environ.get("HF_TOKEN"),
            "REDIS_URL": os.environ.get("REDIS_URL", "redis://localhost:6379/0"),
            "CACHE_ENABLED": os.environ.get("LEGAL_RAG_CACHE_ENABLED", "1") == "1",
            "CACHE_TTL_SECONDS": int(os.environ.get("LEGAL_RAG_CACHE_TTL_SECONDS", "3600")),
        },
    )()

DOCUMENTS_DIR = BASE_DIR / "documents"
BENCHMARK_DIR = BASE_DIR / "benchmark_results"
CORPUS_MANIFEST_DIR = BENCHMARK_DIR / "corpus_manifests"

# Adilet ZAN — источник актуальных кодексов (adilet.zan.kz)
ADILET_BASE_URL = "https://adilet.zan.kz/rus/docs"
# (имя файла в documents/, ID документа на Adilet)
ADILET_SOURCES = [
    ("constitution.txt", "K950001000_"),  # 1. Конституция РК
    ("civil_code.txt", "K940001000_"),  # 2. ГК РК (Общая часть)
    ("civil_code2.txt", "K990000409_"),  # 3. ГК РК (Особенная часть)
    ("labor_code.txt", "K1500000414"),  # 4. Трудовой кодекс РК
    ("tax_code.txt", "K1700000120"),  # 5. Налоговый кодекс РК
    ("code_of_administrative_offenses.txt", "K1400000235"),  # 6. КоАП РК
    ("criminal_code.txt", "K1400000226"),  # 7. Уголовный кодекс РК
    ("code_on_marriage_and_family.txt", "K1100000518"),  # 8. О браке и семье
    ("code_on_public_health.txt", "K2000000360"),  # 9. О здоровье народа
    ("entrepreneurial_code.txt", "K1500000375"),  # 10. Предпринимательский кодекс
    (
        "code_on_administrative_procedures.txt",
        "K2000000350",
    ),  # 11. Об административных процедурах
    ("social_code.txt", "K2300000224"),  # 12. Социальный кодекс РК
    ("civil_procedure_code.txt", "K1500000377"),  # 13. ГПК РК
    ("criminal_procedure_code.txt", "K1400000231"),  # 14. УПК РК
    ("law_on_public_procurement.txt", "Z2400000106"),  # 16. О государственных закупках
    ("law_on_anticorruption.txt", "K1500000410"),  # 17. О противодействии коррупции
    ("law_on_enforcement.txt", "Z100000261_"),  # 18. Об исполнительном производстве
    ("law_on_personal_data.txt", "K130000094_"),  # 19. О персональных данных
    ("law_on_ai.txt", "Z250000230"),  # 20. Об искусственном интеллекте
    ("law_on_consumer_protection.txt", "Z100000274_"),  # О защите прав потребителей
    ("law_on_housing_relations.txt", "Z970000094_"),  # О жилищных отношениях
    ("law_on_banks.txt", "Z950002444_"),  # О банках и банковской деятельности
    ("land_code.txt", "K030000442_"),  # Земельный кодекс РК
    ("law_on_military_service.txt", "Z1200000561"),  # О воинской службе и статусе военнослужащих
    ("law_on_llp.txt", "Z980000220_"),  # О товариществах с ограниченной и дополнительной ответственностью
    ("law_on_notariat.txt", "Z970000155_"),  # О нотариате
    ("law_on_real_estate_registration.txt", "Z070000310_"),  # О государственной регистрации прав на недвижимое имущество
    ("law_on_vehicle_liability_insurance.txt", "Z030000446_"),  # Об обязательном страховании ГПО владельцев ТС
    ("law_on_education.txt", "Z070000319_"),  # Об образовании
    ("law_on_public_service.txt", "Z1500000416"),  # О государственной службе Республики Казахстан
    ("law_on_child_rights.txt", "Z020000345_"),  # О правах ребенка
    ("law_on_advertising.txt", "Z030000461_"),  # О рекламе
    ("law_on_collection_activity.txt", "Z1700000062"),  # О коллекторской деятельности
    ("law_on_road_traffic.txt", "Z1400000194"),  # О дорожном движении
    ("law_on_valuation_activity.txt", "Z1800000133"),  # Об оценочной деятельности
    ("law_on_legal_entities_registration.txt", "Z950002198_"),  # О госрегистрации юрлиц и филиалов
    ("law_on_currency_regulation.txt", "Z1800000167"),  # О валютном регулировании и валютном контроле
    ("law_on_digital_assets.txt", "Z2300000193"),  # О цифровых активах
    ("law_on_personal_data_protection.txt", "Z1300000094"),  # О персональных данных и их защите
    ("law_on_credit_bureaus.txt", "Z040000573_"),  # О кредитных бюро и формировании кредитных историй
    ("law_on_microfinance.txt", "Z1200000056"),  # О микрофинансовой деятельности
    ("law_on_citizen_bankruptcy.txt", "Z2200000178"),  # О восстановлении платежеспособности и банкротстве граждан
    ("law_on_rehabilitation_bankruptcy.txt", "Z1400000176"),  # О реабилитации и банкротстве
    ("law_on_access_to_information.txt", "Z1500000401"),  # Об доступе к информации
    (
        "law_on_electronic_document_signature.txt",
        "Z030000370_",
    ),  # Об электронном документе и электронной цифровой подписи
    ("law_on_mass_media.txt", "Z2400000093"),  # О масс-медиа
    ("law_on_state_secrets.txt", "Z990000349_"),  # О государственных секретах
    (
        "law_on_advocacy_and_legal_assistance.txt",
        "Z1800000176",
    ),  # Об адвокатской деятельности и юридической помощи
    ("law_on_informatization.txt", "Z1500000418"),  # Об информатизации
    ("law_on_migration_of_population.txt", "Z1100000477"),  # О миграции населения
    ("law_on_languages.txt", "Z970000151_"),  # О языках в РК
]

# Pinecone — векторная БД (облако)
PINECONE_INDEX_NAME = env_settings.PINECONE_INDEX_NAME
PINECONE_NAMESPACE = env_settings.PINECONE_NAMESPACE
PINECONE_API_KEY = env_settings.PINECONE_API_KEY
GROQ_API_KEY = env_settings.GROQ_API_KEY
HF_TOKEN = env_settings.HF_TOKEN
OPENROUTER_API_KEY = os.environ.get("OPENROUTER_API_KEY", "")
OPENROUTER_BASE_URL = os.environ.get(
    "OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1"
).strip()
OPENROUTER_SITE_URL = os.environ.get(
    "OPENROUTER_SITE_URL", "https://legalrag.kz"
).strip()
OPENROUTER_APP_NAME = os.environ.get("OPENROUTER_APP_NAME", "LegalRAG").strip()
PINECONE_DIMENSION = 1024  # multilingual-e5-large
PINECONE_ENRICHMENT_TIMEOUT_SEC = float(
    os.environ.get("LEGAL_RAG_PINECONE_ENRICHMENT_TIMEOUT_SEC", "2.0")
)

# Эмбеддинги
EMBEDDING_MODEL = os.environ.get(
    "LEGAL_RAG_EMBEDDING", "intfloat/multilingual-e5-large"
)
HF_READ_TIMEOUT_SEC = int(
    os.environ.get(
        "LEGAL_RAG_HF_READ_TIMEOUT_SEC", os.environ.get("HF_HUB_READ_TIMEOUT", "60")
    )
)
HF_CONNECT_TIMEOUT_SEC = int(
    os.environ.get(
        "LEGAL_RAG_HF_CONNECT_TIMEOUT_SEC",
        os.environ.get("HF_HUB_CONNECT_TIMEOUT", "10"),
    )
)
HF_OFFLINE = (
    os.environ.get("LEGAL_RAG_HF_OFFLINE", os.environ.get("HF_HUB_OFFLINE", "0")) == "1"
)
HF_LOCAL_ONLY = (
    os.environ.get("LEGAL_RAG_HF_LOCAL_ONLY", "0") == "1"
)  # 1=offline-only; 0=internet first, local fallback
# Cache dir: deterministic project .models_cache (never system-protected /app root)
_raw_cache = os.environ.get("LEGAL_RAG_HF_CACHE_DIR", "").strip()
_default_cache = str(BASE_DIR / ".models_cache")
if _raw_cache and _raw_cache.startswith("/app") and not Path("/app").exists():
    HF_CACHE_DIR = _default_cache  # Native run with Docker .env — /app doesn't exist
else:
    HF_CACHE_DIR = _raw_cache if _raw_cache else _default_cache

# Reranker model (used by agentic workflow and optional rag_chain reranker)
RERANKER_MODEL = os.environ.get("LEGAL_RAG_RERANKER_MODEL", "cross-encoder/ms-marco-MiniLM-L-6-v2")
RERANKER_FALLBACK_MODEL = os.environ.get("LEGAL_RAG_RERANKER_FALLBACK", "cross-encoder/ms-marco-MiniLM-L-6-v2")
USE_RERANKER = os.environ.get("LEGAL_RAG_USE_RERANKER", "1") == "1"


def configure_hf_hub() -> None:
    """Set Hugging Face / Transformers cache paths BEFORE any HF library loads.
    Ensures models load from local .models_cache and never hit /app permission issues.
    """
    cache_dir = HF_CACHE_DIR
    os.environ["HF_HOME"] = cache_dir
    os.environ["HF_HUB_CACHE"] = cache_dir
    os.environ.pop("TRANSFORMERS_CACHE", None)
    os.environ["SENTENCE_TRANSFORMERS_HOME"] = cache_dir
    os.environ.setdefault("HF_HUB_READ_TIMEOUT", str(HF_READ_TIMEOUT_SEC))
    os.environ.setdefault("HF_HUB_CONNECT_TIMEOUT", str(HF_CONNECT_TIMEOUT_SEC))
    if HF_OFFLINE:
        os.environ["HF_HUB_OFFLINE"] = "1"


def build_openrouter_default_headers() -> dict[str, str]:
    """Attribution headers for OpenRouter usage dashboard (HTTP-Referer, X-Title)."""
    referer = (
        OPENROUTER_SITE_URL
        or os.environ.get("OPENROUTER_HTTP_REFERER", "").strip()
        or "https://legalrag.kz"
    )
    title = (
        OPENROUTER_APP_NAME
        or os.environ.get("OPENROUTER_APP_TITLE", "").strip()
        or "LegalRAG"
    )
    return {
        "HTTP-Referer": referer.encode("ascii", "ignore").decode("ascii"),
        "X-Title": title.encode("ascii", "ignore").decode("ascii"),
    }


# LLM profile defaults to Groq with a valid Groq model.
# Backend/model can still be overridden via env.
OLLAMA_BASE_URL = os.environ.get("OLLAMA_HOST", "http://localhost:11434")
LLM_BACKEND = "openrouter"
LLM_MODEL = "meta-llama/llama-3.1-8b-instruct"
LLM_TEMPERATURE = 0.0
LLM_MAX_TOKENS = int(os.environ.get("LEGAL_RAG_LLM_MAX_TOKENS", "1024"))
HF_LLM_BASE_MODEL = os.environ.get(
    "LEGAL_RAG_HF_BASE_MODEL", "Qwen/Qwen2.5-7B-Instruct"
)
HF_LLM_ADAPTER_PATH = os.environ.get("LEGAL_RAG_HF_ADAPTER_PATH", "").strip()
HF_LLM_DEVICE_MAP = os.environ.get("LEGAL_RAG_HF_DEVICE_MAP", "auto").strip() or "auto"
HF_LLM_LOAD_IN_4BIT = os.environ.get("LEGAL_RAG_HF_LOAD_IN_4BIT", "0") == "1"
HF_LLM_LOAD_IN_8BIT = os.environ.get("LEGAL_RAG_HF_LOAD_IN_8BIT", "0") == "1"
HF_LLM_TORCH_DTYPE = os.environ.get("LEGAL_RAG_HF_TORCH_DTYPE", "auto").strip().lower()
HF_LLM_TOP_P = float(os.environ.get("LEGAL_RAG_HF_TOP_P", "0.95"))
HF_LLM_REPETITION_PENALTY = float(
    os.environ.get("LEGAL_RAG_HF_REPETITION_PENALTY", "1.05")
)
HF_LLM_DO_SAMPLE = os.environ.get("LEGAL_RAG_HF_DO_SAMPLE", "0") == "1"
# Контекст: conservative defaults to avoid Groq 413/TPM overflow on long legal prompts.
CONTEXT_MAX_DOCS = int(os.environ.get("LEGAL_RAG_CONTEXT_MAX_DOCS", "6"))
CONTEXT_MAX_CHARS_PER_DOC = int(
    os.environ.get("LEGAL_RAG_CONTEXT_MAX_CHARS_PER_DOC", "1500")
)
CONTEXT_MAX_TOTAL_TOKENS = int(
    os.environ.get("LEGAL_RAG_CONTEXT_MAX_TOTAL_TOKENS", "2000")
)
CONTEXT_MAX_TOKENS_PER_DOC = int(
    os.environ.get("LEGAL_RAG_CONTEXT_MAX_TOKENS_PER_DOC", "500")
)
CONTEXT_TOKENIZER_MODEL = os.environ.get(
    "LEGAL_RAG_CONTEXT_TOKENIZER_MODEL", "meta-llama/Meta-Llama-3.1-8B-Instruct"
)

# Retriever (двухэтапный: широкий отбор + rerank)
# Tune via env vars — lower = faster Pinecone (free tier). Raise only if recall drops.
RETRIEVER_WIDE_K = int(os.environ.get("LEGAL_RAG_RETRIEVER_WIDE_K", "24"))
RETRIEVER_TOP_K = RETRIEVER_WIDE_K  # совместимость со старым кодом
RETRIEVER_TOP_K_AFTER_RERANK = int(
    os.environ.get("LEGAL_RAG_RETRIEVER_TOP_K_AFTER_RERANK", "8")
)
RETRIEVER_MIN_K_CRIMINAL = int(
    os.environ.get("LEGAL_RAG_RETRIEVER_MIN_K_CRIMINAL", "8")
)
RETRIEVER_MULTI_QUERY_LIMIT = int(
    os.environ.get("LEGAL_RAG_RETRIEVER_MULTI_QUERY_LIMIT", "2")
)
RETRIEVER_SAME_CODE_PENALTY_STEP = float(
    os.environ.get("LEGAL_RAG_RETRIEVER_SAME_CODE_PENALTY_STEP", "0.1")
)
HYBRID_K = RETRIEVER_WIDE_K
# Pinecone hard filtering (lineage metadata: code_ru, article_number, revision_date from chunks)
# Optional env: restrict retrieval to one code and/or article (e.g. for testing).
# Example (Article 136 only): LEGAL_RAG_FILTER_CODE_RU="Уголовный кодекс РК", LEGAL_RAG_FILTER_ARTICLE_NUMBER="136"
RETRIEVER_FILTER_CODE_RU = os.environ.get("LEGAL_RAG_FILTER_CODE_RU", None)
RETRIEVER_FILTER_ARTICLE_NUMBER = os.environ.get(
    "LEGAL_RAG_FILTER_ARTICLE_NUMBER", None
)

# Hybrid search: BM25 (exact terms e.g. "Article 122") + Dense vectors (semantic). Weights sum to 1.0.
BM25_WEIGHT = float(os.environ.get("LEGAL_RAG_BM25_WEIGHT", "0.4"))
VECTOR_WEIGHT = float(os.environ.get("LEGAL_RAG_VECTOR_WEIGHT", "0.6"))
HYBRID_FUSION_METHOD = os.environ.get("LEGAL_RAG_HYBRID_FUSION_METHOD", "rrf").strip().lower()
HYBRID_RRF_K = int(os.environ.get("LEGAL_RAG_HYBRID_RRF_K", "60"))
HYBRID_RRF_SCORE_SCALE = float(
    os.environ.get("LEGAL_RAG_HYBRID_RRF_SCORE_SCALE", "100.0")
)
CHUNKS_PICKLE_PATH = BASE_DIR / "chunks_for_bm25.pkl"
SUMMARY_CHUNKS_PICKLE_PATH = BASE_DIR / "summary_chunks_for_bm25.pkl"
# Summary index: search summaries first, then expand to full chunks.
ENABLE_SUMMARY_INDEX = os.environ.get("LEGAL_RAG_ENABLE_SUMMARY_INDEX", "1") == "1"
ENABLE_CONTEXTUAL_PREFIX = os.environ.get("LEGAL_RAG_ENABLE_CONTEXTUAL_PREFIX", "1") == "1"
USE_LLM_CONTEXTUAL_PREFIX = os.environ.get("LEGAL_RAG_USE_LLM_CONTEXTUAL_PREFIX", "0") == "1"
USE_LLM_SUMMARIES = os.environ.get("LEGAL_RAG_USE_LLM_SUMMARIES", "0") == "1"
SUMMARY_INDEX_TOP_K = int(os.environ.get("LEGAL_RAG_SUMMARY_INDEX_TOP_K", "4"))
SUMMARY_EXPANSION_MAX_CODES = int(
    os.environ.get("LEGAL_RAG_SUMMARY_EXPANSION_MAX_CODES", "3")
)
# Reranker
USE_RERANKER = os.environ.get("LEGAL_RAG_USE_RERANKER", "1") == "1"
RERANKER_MANDATORY = os.environ.get("LEGAL_RAG_RERANKER_MANDATORY", "1") == "1"
RERANKER_FALLBACK_MODEL = os.environ.get(
    "LEGAL_RAG_RERANKER_FALLBACK_MODEL", "BAAI/bge-reranker-base"
)
# flag_embedding | jina | cross_encoder | auto (auto: jina if "jina" in RERANKER_MODEL else flag_embedding)
RERANKER_BACKEND = os.environ.get("LEGAL_RAG_RERANKER_BACKEND", "auto").strip().lower()
# Skip cross-encoder/Jina when hybrid top docs already match detected code_ru + lexical overlap
RERANK_DYNAMIC_SKIP = os.environ.get("LEGAL_RAG_RERANK_DYNAMIC_SKIP", "0") == "1"
RERANK_SKIP_LEXICAL_THRESHOLD = float(
    os.environ.get("LEGAL_RAG_RERANK_SKIP_LEXICAL_THRESHOLD", "0.35")
)
RERANK_SKIP_MIN_CODE_MATCHES = int(
    os.environ.get("LEGAL_RAG_RERANK_SKIP_MIN_CODE_MATCHES", "2")
)
RERANK_SKIP_LONG_DOCS = os.environ.get("LEGAL_RAG_RERANK_SKIP_LONG_DOCS", "1") == "1"
RERANK_SKIP_LONG_DOC_CHARS = int(
    os.environ.get("LEGAL_RAG_RERANK_SKIP_LONG_DOC_CHARS", "2400")
)
RERANK_SKIP_SHORT_DOC_CHARS = int(
    os.environ.get("LEGAL_RAG_RERANK_SKIP_SHORT_DOC_CHARS", "700")
)
USE_LLM_QUERY_REWRITE = os.environ.get("LEGAL_RAG_USE_LLM_QUERY_REWRITE", "0") == "1"
EXPERIMENTAL_DEDUP_RETRIEVAL = (
    os.environ.get("LEGAL_RAG_EXPERIMENTAL_DEDUP_RETRIEVAL", "0") == "1"
)
USE_LLM_RERANKER = os.environ.get("LEGAL_RAG_USE_LLM_RERANKER", "0") == "1"
LLM_RERANK_CANDIDATES = int(
    os.environ.get("LEGAL_RAG_LLM_RERANK_CANDIDATES", "6")
)
LLM_RERANK_TOP_N = int(os.environ.get("LEGAL_RAG_LLM_RERANK_TOP_N", "5"))
LLM_RERANK_MAX_DOC_CHARS = int(
    os.environ.get("LEGAL_RAG_LLM_RERANK_MAX_DOC_CHARS", "700")
)
LLM_RERANK_MAX_PROMPT_CHARS = int(
    os.environ.get("LEGAL_RAG_LLM_RERANK_MAX_PROMPT_CHARS", "9000")
)
FLASHRANK_MODEL = "ms-marco-MiniLM-L-12-v2"

# Agentic workflow (Board of Directors): Censor = fetch many, rerank to few
AGENTIC_TOP_K_CANDIDATES = int(
    os.environ.get("LEGAL_RAG_AGENTIC_TOP_K_CANDIDATES", "50")
)
AGENTIC_RERANKER_TOP_N = int(os.environ.get("LEGAL_RAG_AGENTIC_RERANKER_TOP_N", "5"))
# Self-RAG: if best reranker score below this, return "Information not found" (no LLM)
AGENTIC_RERANKER_CONFIDENCE_THRESHOLD = float(
    os.environ.get("LEGAL_RAG_AGENTIC_RERANKER_CONFIDENCE_THRESHOLD", "0.35")
)
# HyDE + query expansion (Linguist-Analyst): number of query variations per language
AGENTIC_QUERY_VARIATIONS = int(
    os.environ.get("LEGAL_RAG_AGENTIC_QUERY_VARIATIONS", "4")
)
# CRAG / Self-RAG corrective loop
AGENTIC_CRAG_ENABLED = os.environ.get("LEGAL_RAG_AGENTIC_CRAG_ENABLED", "1") == "1"
AGENTIC_CRAG_MAX_ROUNDS = int(os.environ.get("LEGAL_RAG_AGENTIC_CRAG_MAX_ROUNDS", "1"))
AGENTIC_CRAG_TOP_K = int(os.environ.get("LEGAL_RAG_AGENTIC_CRAG_TOP_K", "10"))
AGENTIC_CRAG_REWRITE_TOP_K = int(
    os.environ.get("LEGAL_RAG_AGENTIC_CRAG_REWRITE_TOP_K", "20")
)
AGENTIC_CRAG_MIN_CONTEXT_SCORE = float(
    os.environ.get("LEGAL_RAG_AGENTIC_CRAG_MIN_CONTEXT_SCORE", "0.10")
)
AGENTIC_CRAG_LLM_EVALUATOR = os.environ.get("LEGAL_RAG_AGENTIC_CRAG_LLM_EVALUATOR", "0") == "1"
AGENTIC_CRAG_DECOMPOSE_QUERY = os.environ.get("LEGAL_RAG_AGENTIC_CRAG_DECOMPOSE_QUERY", "1") == "1"
# CoVe: enable post-response verification against cited articles
AGENTIC_COVE_ENABLED = os.environ.get("LEGAL_RAG_AGENTIC_COVE_ENABLED", "1") == "1"

# Sherlock audit mode: disabled by default, opt-in only.
LEGAL_RAG_ENABLE_SHERLOCK = os.environ.get("LEGAL_RAG_ENABLE_SHERLOCK", "0") == "1"

# Redis Caching
REDIS_URL = env_settings.REDIS_URL
CACHE_ENABLED = env_settings.CACHE_ENABLED
CACHE_TTL_SECONDS = env_settings.CACHE_TTL_SECONDS

# Offline QA mode: deterministic extractive answer from retrieved docs.
LEGAL_RAG_OFFLINE_QA = os.environ.get("LEGAL_RAG_OFFLINE_QA", "0") == "1"

# Detective Mode exit strategy: stop asking after threshold or after one round
DETECTIVE_CONFIDENCE_THRESHOLD = float(
    os.environ.get("LEGAL_RAG_DETECTIVE_CONFIDENCE_THRESHOLD", "0.7")
)
DETECTIVE_MAX_CLARIFYING_QUESTIONS = int(
    os.environ.get("LEGAL_RAG_DETECTIVE_MAX_CLARIFYING_QUESTIONS", "2")
)

# Бенчмарк
BENCHMARK_TIMEOUT_SEC = 300
BENCHMARK_QUESTIONS_MIN = 100

# Безопасность
DISCLAIMER_RU = (
    "Это не официальная юридическая консультация и не заменяет адвоката. "
    "Информация основана исключительно на текстах законов. "
    "Проверяйте актуальные редакции на adilet.zan.kz."
)
DISCLAIMER_KZ = (
    "Бұл ресми заңдық кеңес емес және адвокатты ауыстырмайды. "
    "Ақпарат тек заң мәтініне негізделген. "
    "Актуалды редакцияларды adilet.zan.kz сайтында тексеріңіз."
)
AI_LAW_COMPLIANCE_NOTE = (
    "Ответ сформирован автоматически на основе извлечённых статей; "
    "источники указаны для проверки (требования прозрачности)."
)

# Configure HF/Transformers cache paths at import time (before any HF library loads)
configure_hf_hub()
