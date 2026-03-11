# config.py — Legal RAG: Pinecone, Adilet, все 20 документов

import os
from pathlib import Path

# Автозагрузка переменных из .env
import sys
from pydantic_settings import BaseSettings, SettingsConfigDict

# Пути
BASE_DIR = Path(__file__).resolve().parent.parent.parent

class EngineSettings(BaseSettings):
    PINECONE_API_KEY: str
    PINECONE_INDEX_NAME: str = "legally-index"
    PINECONE_NAMESPACE: str = "default"
    GROQ_API_KEY: str
    HF_TOKEN: str | None = None
    
    model_config = SettingsConfigDict(
        env_file=str(BASE_DIR / ".env"),
        env_file_encoding="utf-8",
        extra="ignore"
    )

try:
    env_settings = EngineSettings()
except Exception as e:
    sys.exit(f"\n[CRITICAL ERROR] Missing Configuration:\n{e}\n")

DOCUMENTS_DIR = BASE_DIR / "documents"
BENCHMARK_DIR = BASE_DIR / "benchmark_results"

# Adilet ZAN — источник актуальных кодексов (adilet.zan.kz)
ADILET_BASE_URL = "https://adilet.zan.kz/rus/docs"
# (имя файла в documents/, ID документа на Adilet)
ADILET_SOURCES = [
    ("constitution.txt", "K950001000_"),           # 1. Конституция РК
    ("civil_code.txt", "K940001000_"),             # 2. ГК РК (Общая часть)
    ("civil_code2.txt", "K990000409_"),            # 3. ГК РК (Особенная часть)
    ("labor_code.txt", "K1500000414"),             # 4. Трудовой кодекс РК
    ("tax_code.txt", "K1700000120"),               # 5. Налоговый кодекс РК
    ("code_of_administrative_offenses.txt", "K1400000235"),  # 6. КоАП РК
    ("criminal_code.txt", "K1400000226"),         # 7. Уголовный кодекс РК
    ("code_on_marriage_and_family.txt", "K1100000518"),  # 8. О браке и семье
    ("code_on_public_health.txt", "K2000000360"),  # 9. О здоровье народа
    ("entrepreneurial_code.txt", "K1500000375"),   # 10. Предпринимательский кодекс
    ("code_on_administrative_procedures.txt", "K2000000350"),  # 11. Об административных процедурах
    ("social_code.txt", "K2300000224"),            # 12. Социальный кодекс РК
    ("civil_procedure_code.txt", "K1500000377"),   # 13. ГПК РК
    ("criminal_procedure_code.txt", "K1400000231"),  # 14. УПК РК
    ("law_on_public_procurement.txt", "Z2400000106"),  # 16. О государственных закупках
    ("law_on_anticorruption.txt", "K1500000410"),  # 17. О противодействии коррупции
    ("law_on_enforcement.txt", "Z100000261_"),    # 18. Об исполнительном производстве
    ("law_on_personal_data.txt", "K130000094_"),  # 19. О персональных данных
    ("law_on_ai.txt", "Z250000230"),              # 20. Об искусственном интеллекте
]

# Pinecone — векторная БД (облако)
PINECONE_INDEX_NAME = env_settings.PINECONE_INDEX_NAME
PINECONE_NAMESPACE = env_settings.PINECONE_NAMESPACE
PINECONE_API_KEY = env_settings.PINECONE_API_KEY
GROQ_API_KEY = env_settings.GROQ_API_KEY
HF_TOKEN = env_settings.HF_TOKEN
PINECONE_DIMENSION = 1024  # multilingual-e5-large

# Эмбеддинги
EMBEDDING_MODEL = os.environ.get("LEGAL_RAG_EMBEDDING", "intfloat/multilingual-e5-large")
HF_READ_TIMEOUT_SEC = int(os.environ.get("LEGAL_RAG_HF_READ_TIMEOUT_SEC", os.environ.get("HF_HUB_READ_TIMEOUT", "60")))
HF_CONNECT_TIMEOUT_SEC = int(os.environ.get("LEGAL_RAG_HF_CONNECT_TIMEOUT_SEC", os.environ.get("HF_HUB_CONNECT_TIMEOUT", "10")))
HF_OFFLINE = os.environ.get("LEGAL_RAG_HF_OFFLINE", os.environ.get("HF_HUB_OFFLINE", "0")) == "1"
HF_LOCAL_ONLY = os.environ.get("LEGAL_RAG_HF_LOCAL_ONLY", "0") == "1"
HF_CACHE_DIR = os.environ.get("LEGAL_RAG_HF_CACHE_DIR", "").strip() or None


def configure_hf_hub() -> None:
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
CONTEXT_MAX_CHARS_PER_DOC = int(os.environ.get("LEGAL_RAG_CONTEXT_MAX_CHARS_PER_DOC", "1200"))

# Retriever (двухэтапный: широкий отбор + rerank)
RETRIEVER_WIDE_K = 20
RETRIEVER_TOP_K = 20  # совместимость со старым кодом
RETRIEVER_TOP_K_AFTER_RERANK = 8
RETRIEVER_MIN_K_CRIMINAL = 12
HYBRID_K = 20
# Опциональные фильтры Pinecone:
# - по кодексу (например, УК РК)
# - по номеру статьи (например, 136 для "подмена ребенка")
# В проде обычно оставьте None; включайте на время тестов/кейсов.
# Пример для ст. 136 УК РК:
#   LEGAL_RAG_FILTER_CODE_RU="Уголовный кодекс РК"
#   LEGAL_RAG_FILTER_ARTICLE_NUMBER="136"
RETRIEVER_FILTER_CODE_RU = os.environ.get("LEGAL_RAG_FILTER_CODE_RU", None)
RETRIEVER_FILTER_ARTICLE_NUMBER = os.environ.get("LEGAL_RAG_FILTER_ARTICLE_NUMBER", None)
BM25_WEIGHT = 0.6
VECTOR_WEIGHT = 0.4
CHUNKS_PICKLE_PATH = BASE_DIR / "chunks_for_bm25.pkl"

# Reranker
USE_RERANKER = os.environ.get("LEGAL_RAG_USE_RERANKER", "1") == "1"
FLASHRANK_MODEL = "ms-marco-MiniLM-L-12-v2"

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
