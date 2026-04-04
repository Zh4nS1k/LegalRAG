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
                load_dotenv(p, override=False)
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
        GROQ_API_KEY: str
        HF_TOKEN: str | None = None
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
    if not _pk or not _gk:
        sys.exit(
            f"\n[CRITICAL ERROR] Missing configuration.\n"
            f"Load .env or set PINECONE_API_KEY and GROQ_API_KEY.\n"
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
        },
    )()

DOCUMENTS_DIR = BASE_DIR / "documents"
BENCHMARK_DIR = BASE_DIR / "benchmark_results"

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
]

# Pinecone — векторная БД (облако)
PINECONE_INDEX_NAME = env_settings.PINECONE_INDEX_NAME
PINECONE_NAMESPACE = env_settings.PINECONE_NAMESPACE
PINECONE_API_KEY = env_settings.PINECONE_API_KEY
GROQ_API_KEY = env_settings.GROQ_API_KEY
HF_TOKEN = env_settings.HF_TOKEN
PINECONE_DIMENSION = 1024  # multilingual-e5-large

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
RERANKER_MODEL = os.environ.get("LEGAL_RAG_RERANKER_MODEL", "BAAI/bge-reranker-v2-m3")


def configure_hf_hub() -> None:
    """Set Hugging Face / Transformers cache paths BEFORE any HF library loads.
    Ensures models load from local .models_cache and never hit /app permission issues.
    """
    cache_dir = HF_CACHE_DIR
    os.environ["HF_HOME"] = cache_dir
    os.environ["HF_HUB_CACHE"] = cache_dir
    os.environ["TRANSFORMERS_CACHE"] = cache_dir
    os.environ["SENTENCE_TRANSFORMERS_HOME"] = cache_dir
    os.environ.setdefault("HF_HUB_READ_TIMEOUT", str(HF_READ_TIMEOUT_SEC))
    os.environ.setdefault("HF_HUB_CONNECT_TIMEOUT", str(HF_CONNECT_TIMEOUT_SEC))
    if HF_OFFLINE:
        os.environ["HF_HUB_OFFLINE"] = "1"


# LLM (по умолчанию — Groq, можно переключить на Ollama)
OLLAMA_BASE_URL = os.environ.get("OLLAMA_HOST", "http://localhost:11434")
LLM_MODEL = os.environ.get("LEGAL_RAG_LLM", "llama-3.1-8b-instant")
LLM_TEMPERATURE = 0.0
LLM_MAX_TOKENS = int(os.environ.get("LEGAL_RAG_LLM_MAX_TOKENS", "2048"))
# Контекст (ограничение длины для предотвращения 413/TPM)
CONTEXT_MAX_DOCS = int(os.environ.get("LEGAL_RAG_CONTEXT_MAX_DOCS", "5"))
CONTEXT_MAX_CHARS_PER_DOC = int(
    os.environ.get("LEGAL_RAG_CONTEXT_MAX_CHARS_PER_DOC", "1200")
)

# Retriever (двухэтапный: широкий отбор + rerank)
# Tune via env vars — lower = faster Pinecone (free tier). Raise only if recall drops.
RETRIEVER_WIDE_K = int(os.environ.get("LEGAL_RAG_RETRIEVER_WIDE_K", "10"))
RETRIEVER_TOP_K = RETRIEVER_WIDE_K  # совместимость со старым кодом
RETRIEVER_TOP_K_AFTER_RERANK = int(
    os.environ.get("LEGAL_RAG_RETRIEVER_TOP_K_AFTER_RERANK", "6")
)
RETRIEVER_MIN_K_CRIMINAL = int(
    os.environ.get("LEGAL_RAG_RETRIEVER_MIN_K_CRIMINAL", "8")
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
CHUNKS_PICKLE_PATH = BASE_DIR / "chunks_for_bm25.pkl"

# Reranker
USE_RERANKER = os.environ.get("LEGAL_RAG_USE_RERANKER", "1") == "1"
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
# CoVe: enable post-response verification against cited articles
AGENTIC_COVE_ENABLED = os.environ.get("LEGAL_RAG_AGENTIC_COVE_ENABLED", "1") == "1"

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
