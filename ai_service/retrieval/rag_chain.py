# rag_chain.py — Pinecone + BM25, reranker, строгий промпт

import logging
import os
import re
import threading
import time
from typing import Any, List, Sequence, Optional
from langchain_core.callbacks import Callbacks

import sys

from ai_service.utils import latency
from ai_service.utils.connectivity import is_internet_available, is_cache_populated
from ai_service.lifecycle_hooks import network_sensor

from ai_service.core import config

logger = logging.getLogger("ai_service.rag")

from langchain_core.prompts import PromptTemplate
from langchain_core.callbacks import CallbackManagerForRetrieverRun
from langchain_core.documents import Document
from langchain_core.retrievers import BaseRetriever

_nltk_ready = False
_stemmer = None


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


_init_lock = threading.Lock()
_embeddings_instance = None
_vector_store_instance = None
_retriever_instance = None
_llm_instance = None


class PrefixedEmbeddings:
    def __init__(self, embeddings):
        self.embeddings = embeddings

    def embed_documents(self, texts):
        return self.embeddings.embed_documents(["passage: " + t for t in texts])

    def embed_query(self, text):
        return self.embeddings.embed_query("query: " + text)


def _make_embeddings() -> PrefixedEmbeddings:
    """Hybrid Connectivity Hook: internet first, local fallback, fail-safe if neither."""
    logger.info("[START] Model Initialization (embeddings)")
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
        logger.error("No local model found and no internet access. Cache: %s", cache_folder)
        sys.exit("No local model found and no internet access.")

    # Smart loader: internet → normal (allow cache updates); no internet → local_files_only
    local_only = config.HF_LOCAL_ONLY or not internet_ok
    model_kwargs: dict = {"local_files_only": local_only}

    # Device selection: force CPU if GPU disabled or insufficient
    device = "cpu" if os.environ.get("CUDA_VISIBLE_DEVICES") == "" else ("cuda" if torch.cuda.is_available() else "cpu")
    model_kwargs["device"] = device

    try:
        from langchain_huggingface import HuggingFaceEmbeddings

        emb = PrefixedEmbeddings(
            HuggingFaceEmbeddings(
                model_name=config.EMBEDDING_MODEL,
                encode_kwargs={"normalize_embeddings": True},
                model_kwargs=model_kwargs,
                cache_folder=cache_folder,
            )
        )
        elapsed = time.perf_counter() - t0
        mode = "local cache" if local_only else "internet"
        logger.info("[SUCCESS] Model Initialization (embeddings, %s) (%.2fs)", mode, elapsed)
        return emb
    except Exception as exc:
        elapsed = time.perf_counter() - t0
        logger.error("[FAIL] Model Initialization (embeddings) (%.2fs): %s", elapsed, exc, exc_info=True)
        raise RuntimeError(
            "Не удалось загрузить эмбеддинги. Проверьте сеть или запустите: python -m ai_service.scripts.download_models. Кэш: %s"
            % cache_folder
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
    global _vector_store_instance
    if _vector_store_instance is None:
        with _init_lock:
            if _vector_store_instance is None:
                logger.info("[START] Pinecone Vector Store Initialization")
                t0 = time.perf_counter()
                try:
                    from langchain_pinecone import PineconeVectorStore
                    _vector_store_instance = PineconeVectorStore(
                        index_name=config.PINECONE_INDEX_NAME,
                        embedding=get_embeddings(),
                        namespace=config.PINECONE_NAMESPACE or "default",
                        pinecone_api_key=config.PINECONE_API_KEY,
                    )
                    elapsed = time.perf_counter() - t0
                    logger.info("[SUCCESS] Pinecone Vector Store Initialization (%.2fs)", elapsed)
                except Exception as e:
                    elapsed = time.perf_counter() - t0
                    logger.error("[FAIL] Pinecone Vector Store Initialization (%.2fs): %s", elapsed, e, exc_info=True)
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
_allowed_code_ru_for_filter = None
_filter_clauses: list[dict] = []

if _filter_code:
    # Pinecone не поддерживает $regex — используем $or по известным вариантам
    _variants = list(dict.fromkeys([_filter_code] + [v for v in _uk_variants if v != _filter_code]))
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
    _vector_kwargs["k"] = min(getattr(config, "RETRIEVER_WIDE_K", getattr(config, "HYBRID_K", 8)) + 4, 30)
    print(f"Фильтр Pinecone search_kwargs: {_vector_kwargs.get('filter')}")

from pydantic import Field
class LazyPineconeRetriever(BaseRetriever):
    search_kwargs: dict = Field(default_factory=dict)
    def _get_relevant_documents(self, query: str, *, run_manager=None) -> List[Document]:
        # Diagnostic: log raw similarity scores
        vs = get_vector_store()
        docs_with_scores = vs.similarity_search_with_score(query, **self.search_kwargs)
        if docs_with_scores:
            scores = [score for _, score in docs_with_scores]
            logger.info(f"[DIAG] Top 5 similarity scores: {scores[:5]}")
            if scores and scores[0] < 0.6:
                logger.critical(f"[RETR_FAIL] Low similarity score detected: {scores[0]}. Check embedding model version.")
            # Log metadata of first 3 chunks
            for i, (doc, score) in enumerate(docs_with_scores[:3]):
                logger.info(f"[DIAG] Chunk {i+1} metadata: code_ru={doc.metadata.get('code_ru')}, article={doc.metadata.get('article_number')}, score={score}")
        docs = [doc for doc, _ in docs_with_scores]
        return docs

_vector_retriever = LazyPineconeRetriever(search_kwargs=_vector_kwargs)


class _FilterByCodeRetriever(BaseRetriever):
    """Оставляет только документы с разрешённым code_ru и/или article_number (убирает шум от BM25)."""
    retriever: Any
    allowed_code_ru: List[str] | None = None
    article_number: str | None = None

    def _get_relevant_documents(
        self, query: str, *, run_manager: CallbackManagerForRetrieverRun | None = None
    ) -> List[Document]:
        docs = self.retriever.invoke(query)
        filtered = docs

        if self.allowed_code_ru:
            allowed = set(self.allowed_code_ru)
            filtered = [d for d in filtered if (d.metadata.get("code_ru") or "").strip() in allowed]

        if self.article_number:
            filtered = [d for d in filtered if (d.metadata.get("article_number") or "").strip() == self.article_number]

        return filtered if filtered else docs[:5]  # fallback: хоть что-то вернуть


def _extract_article_range(query: str) -> tuple[int, int] | None:
    match = re.search(r"(?:статья|ст\.|ст|бап)?\s*(\d+)\s*[-–—]\s*(\d+)", query or "", re.IGNORECASE)
    if not match:
        return None
    start = int(match.group(1))
    end = int(match.group(2))
    return (start, end) if start <= end else (end, start)


def _augment_retrieval_query(query: str) -> str:
    q = (query or "").lower()
    extras: list[str] = []
    if any(token in q for token in ("несовершеннолетний", "несовершеннолетние", "несовершеннолетних", "minor", "underage", "кәмелетке толмаған", "кәмелетке толмағандар")):
        extras.append("работники, не достигшие восемнадцатилетнего возраста")
        extras.append("статья 76 Трудовой кодекс РК")
        extras.append("запрет ночной работы несовершеннолетних")
    if any(token in q for token in ("субсид", "субсидия", "гос", "государ", "бюджет", "грант", "инвест", "смет", "договор", "фиктив", "жалған", "құжат", "алаяқ", "мемлекеттік", "қаржы", "ақша")):
        extras.append("алаяқтық 190 УК РК")
        extras.append("қылмыстық жолмен алынған ақшаны заңдастыру 218 УК РК")
        extras.append("субсидия алу үшін жалған құжаттар 190 УК РК")
    if any(token in q for token in ("заңсыз кәсіпкер", "кәсіпкерлік", "лицензиясыз", "тіркеусіз", "незаконн", "без регистрации", "без лицензии", "салық төлем", "налог", "уклонен")):
        extras.append("заңсыз кәсіпкерлік 214 УК РК")
        extras.append("салық төлеуден жалтару 245 УК РК")
    if any(token in q for token in ("пирамида", "пирамид", "қаржылық пирамида", "инвестиция", "инвест", "жоғары пайда", "30-50%")):
        extras.append("қаржылық пирамида құру және басқару 217 УК РК")
        extras.append("финансовая пирамида создание и руководство 217 УК РК")
        extras.append("реклама финансовой пирамиды 217-1 УК РК")
    if any(token in q for token in ("дәрігер", "жедел жәрдем", "медицин", "медициналық", "фельдшер", "санитар")):
        extras.append("ненадлежащее выполнение профессиональных обязанностей медицинским работником 317 УК РК")
        extras.append("медициналық қызметкердің кәсіби міндеттерін тиісінше орындамауы 317 УК РК")
        extras.append("оставление в опасности 119 УК РК")
    if any(token in q for token in ("қалдық су", "қалдық сулар", "өзен", "су ластау", "суға төгу", "тазарту жүйесі", "эколог", "өндіріс қалдық", "өндірістік қалдық", "химия", "улы зат", "жаппай улану", "жаппай ауру")):
        extras.append("загрязнение вод 328 УК РК")
        extras.append("су ластау 328 УК РК")
        extras.append("нарушение правил охраны окружающей среды 324 УК РК")
        extras.append("опасные химические вещества 325 УК РК")
    if any(token in q for token in ("шетел", "сырт ел", "резидент", "жылжымайтын", "жарғылық капитал", "уставный капитал", "капиталға", "вклад", "взнос", "декларация", "деклар", "имущ", "имущественный", "прирост стоимости")):
        extras.append("доход от прироста стоимости 228 налоговый кодекс")
        extras.append("имущественный доход физического лица 330 налоговый кодекс")
        extras.append("вклад в уставный капитал имущество 333 налоговый кодекс")
        extras.append("иностранные источники дохода 332 налоговый кодекс")
    range_match = _extract_article_range(query)
    if range_match and ("ук" in q or "қылмыстық" in q or "уголов" in q):
        start, end = range_match
        nums = " ".join(str(n) for n in range(start, end + 1))
        extras.append(f"статьи {nums} УК РК")
    return (query + " " + " ".join(extras)).strip() if extras else query


def _is_criminal_query(query: str) -> bool:
    q = (query or "").lower()
    return any(token in q for token in ("қылмыстық", "уголов", "преступ", "ук рк"))


def _focus_articles_from_query(query: str) -> set[str]:
    q = (query or "").lower()
    focus: set[str] = set()
    if any(token in q for token in ("субсид", "субсидия", "гос", "государ", "бюджет", "грант", "инвест", "смет", "договор", "фиктив", "мемлекеттік", "жалған", "алаяқ", "құжат", "қаржы", "ақша")):
        focus.update({"190", "218"})
    if any(token in q for token in ("заңсыз кәсіпкер", "кәсіпкерлік", "лицензиясыз", "тіркеусіз", "незаконн", "без регистрации", "без лицензии", "салық төлем", "налог", "уклонен")):
        focus.update({"214", "245"})
    if any(token in q for token in ("пирамида", "пирамид", "қаржылық пирамида", "инвестиция", "инвест", "жоғары пайда", "30-50%")):
        focus.update({"217", "190"})
    if any(token in q for token in ("қалдық су", "қалдық сулар", "өзен", "су ластау", "суға төгу", "тазарту жүйесі", "эколог", "өндіріс қалдық", "өндірістік қалдық", "химия", "улы зат", "жаппай улану", "жаппай ауру")):
        focus.update({"328", "325", "324"})
    if any(token in q for token in ("шетел", "сырт ел", "резидент", "жылжымайтын", "жарғылық капитал", "уставный капитал", "капиталға", "вклад", "взнос", "декларация", "деклар", "имущ", "имущественный", "прирост стоимости")):
        focus.update({"228", "330", "332", "333"})
    return focus


def _is_subsidy_query(query: str) -> bool:
    q = (query or "").lower()
    return any(token in q for token in ("субсид", "субсидия", "грант", "гос", "государ", "мемлекеттік", "бюджет"))


def _is_illegal_business_query(query: str) -> bool:
    q = (query or "").lower()
    return any(token in q for token in ("заңсыз кәсіпкер", "кәсіпкерлік", "лицензиясыз", "тіркеусіз", "незаконн", "без регистрации", "без лицензии", "салық төлем", "налог", "уклонен"))


def _is_pyramid_query(query: str) -> bool:
    q = (query or "").lower()
    return any(token in q for token in ("пирамида", "пирамид", "қаржылық пирамида", "инвестиция", "инвест", "жоғары пайда", "30-50%"))


def _needs_circumstances_query(query: str) -> bool:
    q = (query or "").lower()
    return any(token in q for token in ("ауырлататын", "жеңілдететін", "смягча", "отягча"))


def _doc_key(doc: Document) -> tuple[str, str]:
    return (
        str(doc.metadata.get("source", "")).strip(),
        str(doc.metadata.get("article_number", "")).strip(),
    )


def _merge_unique(base: List[Document], extra: List[Document]) -> List[Document]:
    seen = {_doc_key(d) for d in base}
    merged = list(base)
    for d in extra:
        key = _doc_key(d)
        if key not in seen:
            merged.append(d)
            seen.add(key)
    return merged


class _HeuristicRetriever(BaseRetriever):
    """Лёгкий эвристический слой: расширяет запрос и мягко фильтрует диапазоны статей."""
    base_retriever: Any
    vector_store: Any

    def _get_relevant_documents(
        self, query: str, *, run_manager: CallbackManagerForRetrieverRun | None = None
    ) -> List[Document]:
        search_query = _augment_retrieval_query(query)
        docs = self.base_retriever.invoke(search_query)
        uk_filter = {"$or": [{"code_ru": v} for v in _uk_variants]}
        if _is_criminal_query(query):
            allowed = set(_uk_variants)
            filtered = [d for d in docs if (d.metadata.get("code_ru") or "").strip() in allowed]
            if filtered:
                docs = filtered
            else:
                fallback_docs: list[Document] = []
                for code_ru in _uk_variants:
                    try:
                        fallback_docs.extend(
                            get_vector_store().similarity_search(
                                search_query,
                                k=6,
                                filter={"code_ru": code_ru},
                            )
                        )
                    except Exception:
                        continue
                if fallback_docs:
                    docs = fallback_docs
        range_match = _extract_article_range(query)
        if range_match:
            start, end = range_match
            filtered = [
                d for d in docs
                if (d.metadata.get("article_number") or "").strip().isdigit()
                and start <= int(d.metadata.get("article_number")) <= end
            ]
            return filtered if filtered else docs
        focus = _focus_articles_from_query(query)
        if focus:
            focused = [
                d for d in docs
                if (d.metadata.get("article_number") or "").strip() in focus
            ]
            return focused if focused else docs
        return docs


class _LawAwareRetriever(BaseRetriever):
    """Жёсткий law-aware слой: для УК — принудительная подвыборка + добор статей."""
    base_retriever: Any
    vector_store: Any
    min_k_criminal: int = 10

    def _get_relevant_documents(
        self, query: str, *, run_manager: CallbackManagerForRetrieverRun | None = None
    ) -> List[Document]:
        search_query = _augment_retrieval_query(query)
        docs = self.base_retriever.invoke(search_query)
        uk_filter = {"$or": [{"code_ru": v} for v in _uk_variants]}

        if _is_criminal_query(query):
            allowed = set(_uk_variants)
            filtered = [d for d in docs if (d.metadata.get("code_ru") or "").strip() in allowed]
            docs = filtered if filtered else docs
            if len(docs) < self.min_k_criminal:
                extra: list[Document] = []
                for code_ru in _uk_variants:
                    try:
                        extra.extend(
                            get_vector_store().similarity_search(
                                search_query,
                                k=self.min_k_criminal,
                                filter={"code_ru": code_ru},
                            )
                        )
                    except Exception:
                        continue
                if extra:
                    docs = _merge_unique(docs, extra)

        if _needs_circumstances_query(query):
            extra_docs: list[Document] = []
            for q in (
                "смягчающие обстоятельства УК РК",
                "отягчающие обстоятельства УК РК",
                "жеңілдететін мән-жайлар Қылмыстық кодекс",
                "ауырлататын мән-жайлар Қылмыстық кодекс",
            ):
                try:
                    extra_docs.extend(
                        get_vector_store().similarity_search(
                            q,
                            k=4,
                            filter=uk_filter,
                        )
                    )
                except Exception:
                    continue
            if extra_docs:
                docs = _merge_unique(docs, extra_docs)

        return docs


def _fetch_parent_context_from_store(code_ru: str, article_number: str) -> tuple[str, str, str]:
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
        siblings = store.similarity_search(
            f"Глава статья {article_number} {code_ru}",
            k=5,
            filter={"code_ru": code_ru, "article_number": article_number},
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
        print(f"Context enrichment: could not fetch parent for {code_ru} ст.{article_number}: {e}")
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

    def _get_relevant_documents(
        self, query: str | dict, *, run_manager: CallbackManagerForRetrieverRun | None = None
    ) -> List[Document]:
        if isinstance(query, dict):
            # If we receive a dict (e.g. from LCEL chain), extract the query string
            # Try common keys
            q = query.get("input") or query.get("query") or query.get("question") or ""
            if not q and "context" not in query: # If it's not a doc chain input
                 print(f"DEBUG: _TrimRetriever received dict without known keys: {query.keys()}")
            
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
        for d in docs[: self.max_docs]:
            content = d.page_content
            if len(content) > self.max_chars_per_doc:
                content = content[: self.max_chars_per_doc] + "\n[...текст обрезан...]"
            trimmed.append(Document(page_content=content, metadata=d.metadata))
        # Inject parent-context breadcrumb so LLM knows Code → Chapter → Article scope
        return _enrich_with_parent_context(trimmed)


def _load_bm25_chunks() -> list[Document] | None:
    """Load BM25 chunks lazily from pickle (preferred) or prepare_data."""
    try:
        import pickle  # local import: keep module import fast
        _pkl = getattr(config, "CHUNKS_PICKLE_PATH", None) or (config.BASE_DIR / "chunks_for_bm25.pkl")
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


def get_retriever():
    """Build retriever lazily (Pinecone + optional BM25 + optional reranker)."""
    global _retriever_instance
    if _retriever_instance is not None:
        return _retriever_instance

    with _init_lock:
        if _retriever_instance is not None:
            return _retriever_instance

        base_retriever: Any = _vector_retriever

        # Optional BM25 hybrid: build only when first request comes in.
        try:
            from langchain_community.retrievers import BM25Retriever
            try:
                from langchain.retrievers import EnsembleRetriever
            except ImportError:
                try:
                    from langchain.retrievers.ensemble import EnsembleRetriever
                except ImportError:
                    from langchain_classic.retrievers import EnsembleRetriever

            chunks = _load_bm25_chunks()
            if chunks:
                if _ensure_nltk():
                    bm25_retriever = BM25Retriever.from_documents(
                        chunks, preprocess_func=lambda t: bm25_preprocess_func(t) or [], k=_hybrid_k
                    )
                    print("BM25 инициализирован со стеммингом (Snowball/Russian).")
                else:
                    bm25_retriever = BM25Retriever.from_documents(chunks, k=_hybrid_k)

                _vector_w = getattr(config, "VECTOR_WEIGHT", 0.6)
                _bm25_w = getattr(config, "BM25_WEIGHT", 0.4)
                base_retriever = EnsembleRetriever(
                    retrievers=[_vector_retriever, bm25_retriever],
                    weights=[_vector_w, _bm25_w],
                )
                print("Гибридный RAG готов! (BM25 + Pinecone, k=%d)" % _hybrid_k)
        except Exception as e:
            # Don't block startup: hybrid is optional.
            print(f"BM25 не запустился: {e}. Используется только Pinecone.")

        # Post-filtering layer if enabled
        if _allowed_code_ru_for_filter or _filter_article:
            base_retriever = _FilterByCodeRetriever(
                retriever=base_retriever,
                allowed_code_ru=_allowed_code_ru_for_filter,
                article_number=_filter_article,
            )
            print("Включён пост-фильтр по кодексу/статье (только разрешённые code_ru/article_number).")

        # Heuristic + law-aware layers
        heuristic_retriever = _HeuristicRetriever(base_retriever=base_retriever, vector_store=None)
        law_aware_retriever = _LawAwareRetriever(
            base_retriever=heuristic_retriever,
            vector_store=None,
            min_k_criminal=getattr(config, "RETRIEVER_MIN_K_CRIMINAL", 10),
        )
        retr: Any = law_aware_retriever

        # Optional reranker (very heavy) — build lazily on first request.
        if config.USE_RERANKER:
            try:
                from FlagEmbedding import FlagReranker
                try:
                    from langchain.retrievers import ContextualCompressionRetriever
                    from langchain.retrievers.document_compressors.base import BaseDocumentCompressor
                except ImportError:
                    from langchain_classic.retrievers import ContextualCompressionRetriever
                    from langchain_classic.retrievers.document_compressors.base import BaseDocumentCompressor

                config.configure_hf_hub()
                logger.info("[START] Reranker initialization (%s)", config.RERANKER_MODEL)
                t_rerank = time.perf_counter()
                _reranker_model = FlagReranker(config.RERANKER_MODEL, use_fp16=True)
                logger.info("[SUCCESS] Reranker initialized (%.2fs)", time.perf_counter() - t_rerank)

                class BGEReranker(BaseDocumentCompressor):
                    top_n: int = 8

                    def compress_documents(
                        self, documents: Sequence[Document], query: str, callbacks: Optional[Callbacks] = None
                    ) -> Sequence[Document]:
                        if not documents:
                            return []
                        pairs = [[query, d.page_content] for d in documents]
                        scores = _reranker_model.compute_score(pairs)
                        if isinstance(scores, float):
                            scores = [scores]
                        scored_docs = []
                        for i, doc in enumerate(documents):
                            doc.metadata["relevance_score"] = scores[i]
                            scored_docs.append((doc, scores[i]))
                        scored_docs.sort(key=lambda x: x[1], reverse=True)
                        return [d for d, _ in scored_docs[: self.top_n]]

                compressor = BGEReranker(top_n=getattr(config, "RETRIEVER_TOP_K_AFTER_RERANK", 8))
                retr = ContextualCompressionRetriever(
                    base_compressor=compressor,
                    base_retriever=law_aware_retriever,
                )
            except Exception as e:
                logger.error("Reranker BGE-M3 failed: %s (using retrieval without reranking)", e, exc_info=True)

        # Trim context before LLM
        retr = _TrimRetriever(
            base_retriever=retr,
            max_docs=getattr(config, "CONTEXT_MAX_DOCS", 8),
            max_chars_per_doc=getattr(config, "CONTEXT_MAX_CHARS_PER_DOC", 1800),
        )

        _retriever_instance = retr

    # Call outside lock — _ensure_latency_patches calls get_llm() which needs _init_lock (avoid deadlock)
    _ensure_latency_patches()
    return _retriever_instance


def get_llm():
    global _llm_instance
    if _llm_instance is None:
        with _init_lock:
            if _llm_instance is None:
                logger.info("[START] LLM Initialization")
                t0 = time.perf_counter()
                try:
                    _llm_backend = os.environ.get("LEGAL_RAG_LLM_BACKEND", "groq").lower()
                    if _llm_backend == "groq":
                        try:
                            from langchain_groq import ChatGroq  # type: ignore[import]
                        except Exception as e:  # pragma: no cover
                            raise SystemExit(
                                "Для использования облачного Groq установите пакет 'langchain-groq':\n"
                                "  pip install langchain-groq\n"
                                f"Текущая ошибка импорта: {e}"
                            )
                        groq_api_key = config.GROQ_API_KEY or os.environ.get("GROQ_API_KEY")
                        if not groq_api_key:
                            raise SystemExit("Задайте GROQ_API_KEY для облачного Groq (gsk_...): export GROQ_API_KEY=...")
                        _llm_instance = ChatGroq(
                            groq_api_key=groq_api_key,
                            model_name=config.LLM_MODEL,
                            temperature=config.LLM_TEMPERATURE,
                            max_tokens=config.LLM_MAX_TOKENS,
                        )
                        logger.info("[SUCCESS] LLM Initialization (Groq, model=%s) (%.2fs)", config.LLM_MODEL, time.perf_counter() - t0)
                    else:
                        from langchain_ollama import OllamaLLM

                        _llm_instance = OllamaLLM(
                            model=config.LLM_MODEL,
                            temperature=config.LLM_TEMPERATURE,
                            base_url=config.OLLAMA_BASE_URL,
                            num_predict=config.LLM_MAX_TOKENS,
                        )
                        logger.info("[SUCCESS] LLM Initialization (Ollama, model=%s) (%.2fs)", config.LLM_MODEL, time.perf_counter() - t0)
                except Exception as e:
                    elapsed = time.perf_counter() - t0
                    logger.error("[FAIL] LLM Initialization (%.2fs): %s", elapsed, e, exc_info=True)
                    raise
    return _llm_instance


def _ensure_latency_patches() -> None:
    # Patch once, when heavy components exist. Add diagnostic logging for each step.
    if getattr(_ensure_latency_patches, "_done", False):
        return
    try:
        emb = get_embeddings()
        original_embed_query = emb.embed_query

        @latency.measure_latency("embedding")
        def wrapped_embed_query(*args, **kwargs):
            logger.info("[START] Embedding Query")
            t0 = time.perf_counter()
            try:
                out = original_embed_query(*args, **kwargs)
                logger.info("[SUCCESS] Query Embedded (%.2fs)", time.perf_counter() - t0)
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
            logger.info("[START] Pinecone Vector Search")
            t0 = time.perf_counter()
            try:
                docs = original_trim_get_docs(self, *args, **kwargs)
                elapsed = time.perf_counter() - t0
                logger.info("[SUCCESS] Retrieved %d chunks (%.2fs)", len(docs), elapsed)
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
            logger.info("[START] LLM Prompt Construction")
            logger.info("[START] LLM Inference")
            t0 = time.perf_counter()
            try:
                out = original_llm_invoke(self, *args, **kwargs)
                logger.info("[SUCCESS] LLM Response Generated (%.2fs)", time.perf_counter() - t0)
                return out
            except Exception as exc:
                logger.error("LLM Inference failed: %s", exc, exc_info=True)
                raise

        llm.__class__.invoke = wrapped_llm_invoke
    except Exception:
        pass

    _ensure_latency_patches._done = True

UNIVERSAL_PROMPT_TEMPLATE = """You are a Legal Synthesizer. Based on the verified context {context} and the analyzed user intent {input}, provide a final legal standing. Mention the 'Flip-Point' (what could change the outcome)."""

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
4. Если в вопросе диапазон статей (например, ст. 120–135 УК РК) — перечисляй ВСЕ релевантные статьи из контекста с кратким описанием и санкцией.
5. Для любой сферы всегда разбирай, если применимо (и только если в контексте):
   - Конституционное право: права граждан, принципы, нормы Конституции.
   - Гражданское право (ГК): договоры, имущество, обязательства, ущерб, компенсация.
   - Трудовое право (ТК): принципы, права работников/работодателей, нарушения, компенсации.
   - Налоговое право: налоги, нарушения, штрафы, декларации.
   - Административное право (КоАП): нарушения, штрафы, процедуры.
   - Уголовное право (УК): состав преступления (объект, объективная сторона, субъект, субъективная сторона), ауырлататын және жеңілдететін мән-жайлар, санкция дословно.
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

CRIMINAL_PROMPT_TEMPLATE = """Ты — эксперт по Уголовному кодексу РК.
ОТВЕЧАЙ ТОЛЬКО на основе контекста ниже.
Если в контексте нет нужной статьи или информации — отвечай ровно одной строкой:
"Информация не найдена в доступных текстах законов."

Обязательные правила:
1. Начинай строго с: "Это не официальная юридическая консультация. Информация только из базы."
2. Отвечай на казахском, если вопрос на казахском; на русском — если на русском.
3. Для каждого пункта вопроса отвечай по порядку, нумеруя 1), 2), 3) и т.д.
4. Всегда указывай точную статью УК РК, часть и дословную цитату.
5. Разбирай состав преступления ТОЛЬКО если статья есть в контексте:
   - Объект
   - Объективтік жағы
   - Субъект
   - Субъективтік жағы
6. Ауырлататын және жеңілдететін мән-жайлар — ТОЛЬКО если они указаны в статье или в контексте.
7. Санкцию цитируй дословно, включая часть статьи.

{chat_history}
Контекст:
{context}

Вопрос (разбери по пунктам):
{input}

Ответ (нумеруй пункты, цитируй дословно, указывай статью и источник):"""

RANGE_PROMPT_TEMPLATE = """Ты — точный ассистент по УК РК.
ОТВЕЧАЙ ИСКЛЮЧИТЕЛЬНО из контекста.
Если в контексте нет нужных статей — отвечай одной строкой:
"Информация не найдена в доступных текстах законов."

Правила:
• Начинай строго с: "Это не официальная юридическая консультация. Информация только из базы."
• Если в вопросе диапазон статей (например, 120–135) — перечисляй ВСЕ релевантные статьи из контекста с номерами и кратким описанием.
• Цитируй санкции и ключевые части дословно.
• Указывай источник (название кодекса) и номер статьи.

{chat_history}
Контекст:
{context}

Вопрос:
{input}

Ответ (перечисли статьи из диапазона, если они есть, цитируй дословно):"""

UNIVERSAL_PROMPT = PromptTemplate.from_template(UNIVERSAL_PROMPT_TEMPLATE)
CRIMINAL_PROMPT = PromptTemplate.from_template(CRIMINAL_PROMPT_TEMPLATE)
RANGE_PROMPT = PromptTemplate.from_template(RANGE_PROMPT_TEMPLATE)


@latency.measure_latency("prompt_template_build")
def _select_prompt(question: str) -> PromptTemplate:
    if _extract_article_range(question):
        return RANGE_PROMPT
    q = question or ""
    if _is_criminal_query(q) or re.search(r"(?:ст\.?\s*\d|статья\s*\d|бап\s*\d)", q, re.IGNORECASE):
        return CRIMINAL_PROMPT
    return UNIVERSAL_PROMPT


try:
    from langchain.chains import create_retrieval_chain
    from langchain.chains.combine_documents import create_stuff_documents_chain
except ImportError:
    from langchain_classic.chains import create_retrieval_chain
    from langchain_classic.chains.combine_documents import create_stuff_documents_chain

from langchain_core.runnables import RunnablePassthrough

def _fill_missing_metadata(docs):
    # docs is a list of Document objects coming from the retriever
    # We must return the list of docs
    if not isinstance(docs, list):
         # If for some reason we get a dict (e.g. from a weird chain state), return it to avoid crashing, 
         # but this shouldn't happen if the chain is correct.
         return docs
         
    for d in docs:
        if "article_number" not in d.metadata:
            d.metadata["article_number"] = "Н/Д"
        if "code_ru" not in d.metadata:
            d.metadata["code_ru"] = "Неизвестный источник"
        if "source" not in d.metadata:
            d.metadata["source"] = "Неизвестно"
    return docs

def _make_qa_chain(prompt: PromptTemplate) -> Any:
    # Define a prompt to format each document including metadata
    document_prompt = PromptTemplate(
        input_variables=["page_content", "source", "article_number", "code_ru"],
        template="Источник: {source}\nКодекс: {code_ru}\nСтатья: {article_number}\nТекст: {page_content}"
    )

    # LCEL pipeline: Retriever -> Document Chain -> Retrieval Chain
    question_answer_chain = create_stuff_documents_chain(get_llm(), prompt, document_prompt=document_prompt)
    
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


def invoke_qa_with_context(
    query: str,
    context_docs: List[Document],
    history: Optional[List[dict]] = None,
) -> dict:
    """Run QA using a pre-retrieved list of documents (no retriever). Used by agentic workflow."""
    _ensure_latency_patches()
    docs = _fill_missing_metadata(list(context_docs))
    prompt = _select_prompt(query)
    if prompt is RANGE_PROMPT:
        chain = _get_qa_chains()["range"]
    elif prompt is CRIMINAL_PROMPT:
        chain = _get_qa_chains()["criminal"]
    else:
        chain = _get_qa_chains()["universal"]
    # Retrieval chain expects "input", "chat_history", and injects "context" from retriever.
    # Our chain was built with retriever; we invoke the inner question_answer_chain by passing context directly.
    # create_retrieval_chain(retriever, question_answer_chain) passes retriever output as "context".
    # So we need to invoke the part that takes context. The retrieval chain does: context = retriever.invoke(input), then question_answer_chain.invoke({**input, "context": context}).
    # So we can't easily invoke just the question_answer_chain without the retriever. Alternative: use the same prompt and create_stuff_documents_chain and invoke with context=docs.
    document_prompt = PromptTemplate(
        input_variables=["page_content", "source", "article_number", "code_ru"],
        template="Источник: {source}\nКодекс: {code_ru}\nСтатья: {article_number}\nТекст: {page_content}",
    )
    question_answer_chain = create_stuff_documents_chain(get_llm(), prompt, document_prompt=document_prompt)
    res = question_answer_chain.invoke({
        "context": docs,
        "input": query,
        "chat_history": _history_str(history),
    })
    return {"result": res, "source_documents": docs}


def invoke_qa(query: str, history: Optional[List[dict]] = None) -> dict:
    _ensure_latency_patches()
    prompt = _select_prompt(query)
    if prompt is RANGE_PROMPT:
        chain = _get_qa_chains()["range"]
    elif prompt is CRIMINAL_PROMPT:
        chain = _get_qa_chains()["criminal"]
    else:
        chain = _get_qa_chains()["universal"]

    res = chain.invoke({
        "input": query,
        "chat_history": _history_str(history),
    })

    answer = res.get("answer", "")
    source_documents = res.get("context", [])

    if not source_documents:
        # Fallback to internal knowledge
        logger.warning("[FALLBACK] No documents retrieved, using internal knowledge for query: %s", query)
        fallback_prompt = PromptTemplate.from_template(
            "⚠️ ВНИМАНИЕ: Информация не найдена в текущей базе данных. Ниже приведен общий правовой анализ на основе НПА РК:\n\n"
            "Вопрос: {input}\n\n"
            "Ответьте на основе вашего общего знания законодательства Республики Казахстан. Будьте точны и ссылайтесь на релевантные статьи."
        )
        fallback_chain = fallback_prompt | get_llm()
        answer = fallback_chain.invoke({"input": query})
        retrieval_method = "internal_fallback"
    else:
        retrieval_method = "hybrid"

    return {
        "result": answer,
        "source_documents": source_documents,
        "retrieval_method": retrieval_method,
    }

_KZ_CHARS = set("әғқңөұүһі")
_KZ_COMMON_WORDS = (
    "және", "бойынша", "қылмыстық", "құрамы", "қылмысқа", "бап", "заң", "мән-жай", "ауырлататын", "жеңілдететін"
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



ANALYSIS_PROMPT_TEMPLATE = """Ты — юридический эксперт по законодательству Казахстана. Твоя задача — проанализировать текст документа (или его часть) и выявить правовые риски, нарушения и дать рекомендации.

В ответе придерживайся следующей структуры:

### Правовые риски
1. [Название риска]
   - Описание: [подробное описание]
   - Нормативный акт: [закон/статья]
   - Уровень риска: [высокий/средний/низкий]
   - Рекомендация: [предложение по исправлению]

### Неясные формулировки
1. [Формулировка]
   - Проблема: [в чем неясность]
   - Рекомендация: [как переформулировать]
   - Уровень важности: [высокий/средний/низкий]

### Возможные нарушения
1. [Описание нарушения]
   - Нормативный акт: [закон/статья]
   - Последствия: [возможные санкции]
   - Рекомендация: [как избежать]

### Рекомендации
[Список конкретных рекомендаций по исправлению документа]

### Заключение
[Общая сводка по документу с выводами]

Документ:
{text}

Ответ:"""

ANALYSIS_PROMPT = PromptTemplate.from_template(ANALYSIS_PROMPT_TEMPLATE)

def validate_answer(question: str, response: str, sources: List[Document]) -> str:
    fallback = "Информация не найдена в доступных текстах законов."
    if not sources:
        return fallback

    if _is_kz_query(question) and not _is_kz_response(response):
        return fallback

    if _is_criminal_query(question):
        has_uk = any((d.metadata.get("code_ru") or "").strip() in _uk_variants for d in sources)
        if not has_uk:
            return fallback

    allowed = _extract_article_numbers_from_docs(sources)
    mentioned = _extract_article_numbers_from_text(response or "")
    if allowed and mentioned and not mentioned.issubset(allowed):
        return fallback

    if _is_subsidy_query(question):
        if "190" not in mentioned and "218" not in mentioned:
            return fallback

    if _is_illegal_business_query(question):
        if "214" not in mentioned and "245" not in mentioned:
            return fallback

    if _is_pyramid_query(question):
        if "217" not in mentioned:
            return fallback

    if _needs_circumstances_query(question):
        r = (response or "").lower()
        if not any(token in r for token in ("ауырлататын", "жеңілдететін", "отягча", "смягча")):
            return fallback

    return response

def analyze_text(text: str) -> str:
    """Analyses the provided text using the configured LLM."""
    _ensure_latency_patches()
    chain = ANALYSIS_PROMPT | get_llm()
    result = chain.invoke({"text": text})
    # Extract content string if it's an AIMessage
    return result.content if hasattr(result, "content") else str(result)

if __name__ == "__main__":
    question = "Статья 136 УК РК баланы ауыстыру"
    print(f"\nВопрос: {question}\n")
    docs = get_retriever().invoke(question)
    for i, doc in enumerate(docs[:5], 1):
        print(f"{i}. {doc.metadata.get('source')} | {doc.metadata.get('code_ru', '')} ст.{doc.metadata.get('article_number', '')}")
        print(f"   {doc.page_content[:250].replace(chr(10), ' ')}...\n")
    result = invoke_qa(question)
    print("Ответ:", result["result"][:500])
