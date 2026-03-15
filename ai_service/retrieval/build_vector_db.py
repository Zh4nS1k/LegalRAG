# build_vector_db.py — Pinecone: чанки из documents/ (парсинг с Adilet)
# Если обновили documents/ (например, перекачали УК РК): очистите namespace в Pinecone
# или удалите индекс и создайте заново, затем запустите этот скрипт (иначе будут дубликаты).

import pickle
import os
import sys
import time
from pathlib import Path

# Allow running as script from ai_service: python retrieval/build_vector_db.py
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

import sys
from ai_service.core import config
from ai_service.utils.connectivity import is_internet_available, is_cache_populated
from langchain_pinecone import PineconeVectorStore
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.documents import Document
from ai_service.processing.prepare_data import chunks, raw_docs

# Проверка до очистки: есть ли ст. 136 УК в распарсенных чанках
uk_136_chunks = [c for c in chunks if "criminal_code" in c.metadata.get("source", "") and "136" in c.page_content]
print(f"[Проверка] Чанков со ст. 136 УК (до очистки): {len(uk_136_chunks)}")
if uk_136_chunks:
    print("Пример ст. 136 УК:", uk_136_chunks[0].page_content[:300].replace("\n", " "), "...")
else:
    print("Ст. 136 УК не найдена в чанках — проверьте criminal_code.txt и чанкинг.")


class PrefixedEmbeddings:
    def __init__(self, embeddings):
        self.embeddings = embeddings

    def embed_documents(self, texts):
        return self.embeddings.embed_documents(["passage: " + t for t in texts])

    def embed_query(self, text):
        return self.embeddings.embed_query("query: " + text)


def _make_embeddings() -> PrefixedEmbeddings:
    """Hybrid Connectivity Hook: internet first, local fallback, fail-safe if neither."""
    config.configure_hf_hub()
    cache_folder = config.HF_CACHE_DIR

    internet_ok = is_internet_available(timeout=2.0)
    cache_ok = is_cache_populated(cache_folder)

    if not internet_ok and not cache_ok:
        print("No local model found and no internet access. Cache:", cache_folder)
        sys.exit("No local model found and no internet access.")

    local_only = config.HF_LOCAL_ONLY or not internet_ok
    model_kwargs: dict = {"local_files_only": local_only}

    try:
        return PrefixedEmbeddings(
            HuggingFaceEmbeddings(
                model_name=config.EMBEDDING_MODEL,
                encode_kwargs={"normalize_embeddings": True},
                model_kwargs=model_kwargs,
                cache_folder=cache_folder,
            )
        )
    except Exception as exc:
        raise RuntimeError(
            "Не удалось загрузить эмбеддинги. Проверьте сеть или запустите download_models. Кэш: %s"
            % cache_folder
        ) from exc


embeddings = _make_embeddings()

api_key = config.PINECONE_API_KEY or os.environ.get("PINECONE_API_KEY")
if not api_key:
    raise SystemExit("Задайте PINECONE_API_KEY в .env или: export PINECONE_API_KEY=...")


from pinecone import Pinecone, ServerlessSpec
pc = Pinecone(api_key=api_key)

index_name = config.PINECONE_INDEX_NAME
namespace = config.PINECONE_NAMESPACE or "default"

existing = [i.name for i in pc.list_indexes()]
if index_name not in existing:
    print(f"Создаём индекс {index_name}...")
    try:
        pc.create_index(
            name=index_name,
            dimension=config.PINECONE_DIMENSION,
            metric="cosine",
            spec=ServerlessSpec(cloud="aws", region="us-east-1")
        )
        print("Индекс создан. Ожидание готовности...")
        while not pc.describe_index(index_name).status.get("ready"):
            time.sleep(2)
        print("Индекс готов.")
    except Exception as e:
        if "already exists" in str(e).lower():
            print("Индекс уже существует.")
        else:
            raise e

print(f"Подключение к индексу {index_name} (namespace: {namespace})")

vector_store = PineconeVectorStore(
    index_name=index_name,
    embedding=embeddings,
    namespace=namespace,
    pinecone_api_key=api_key,  # explicit — pydantic-settings doesn't write back to os.environ
)

import json

# Pinecone limit: 40 KB на метаданные одного вектора.
# langchain_pinecone при add_documents добавляет в метаданные ключ "text" = page_content (полный текст).
# Поэтому обрезаем page_content перед загрузкой, чтобы metadata["text"] + остальные поля < 40 KB.
PINECONE_METADATA_LIMIT_BYTES = 40960
# Резерв на наши поля (source, code_ru, code_kz, article_number) и сериализацию (JSON + экранирование): ~5 KB
MAX_TEXT_IN_METADATA_BYTES = 30000
# Макс. символов для "text" (кириллица 2 байта/символ, JSON экранирование может увеличить размер)
MAX_PAGE_CONTENT_CHARS = 4000

# Whitelist + blacklist: только короткие поля, длинные — удалить или обрезать
def clean_metadata(meta: dict) -> dict:
    """
    Очищает метаданные: whitelist коротких полей + blacklist длинных.
    Гарантирует размер < 2 KB (лимит Pinecone: 40 KB).
    """
    blacklist = [
        'text', 'content', 'full_text', 'raw', 'body', 'snippet', 'notes',
        'chapter_text', 'page_content', 'raw_text', 'chapter_content',
        'article_text', 'snippets', 'full_article', 'raw_content'
    ]
    allowed = [
        'source', 'code_ru', 'code_kz', 'article_number',
        # Legal hierarchy fields (NEW)
        'chapter_title', 'chapter_number', 'clause_level', 'revision_date',
        # Legacy fields
        'chapter', 'section',
    ]

    clean = {}
    for k, v in meta.items():
        if k in blacklist:
            continue  # не добавляем длинные поля вообще
        if k in allowed:
            v_str = str(v)
            clean[k] = v_str[:200] + '...' if len(v_str) > 200 else v_str
        else:
            # неизвестное поле — добавляем только если короткое, иначе обрезаем
            v_str = str(v)
            size_bytes = len(v_str.encode('utf-8'))
            if size_bytes < 1000:
                clean[k] = v_str
            else:
                print(f"Предупреждение: поле '{k}' обрезано (было {size_bytes} байт)")
                clean[k] = v_str[:200] + '...'
    return clean

print("Очистка метаданных (Pinecone limit: 40 KB)...")
print("Отладка метаданных (ищем превышение):")

clean_chunks = []
max_meta_size = 0
bad_chunk_idx = -1
bad_chunks_found = []

for idx, chunk in enumerate(chunks):
    clean_meta = clean_metadata(chunk.metadata)
    # Проверяем размер метаданных в байтах (UTF-8) после сериализации в JSON
    meta_json = json.dumps(clean_meta, ensure_ascii=False)
    meta_size = len(meta_json.encode('utf-8'))
    
    if meta_size > max_meta_size:
        max_meta_size = meta_size
        bad_chunk_idx = idx
    
    # Находим все чанки с метаданными > 38 KB (чуть ниже лимита для отладки)
    if meta_size > 38000:
        bad_chunks_found.append((idx, meta_size, clean_meta, chunk.page_content[:200]))
        print(f"!!! Чанк {idx} — размер метаданных (JSON): {meta_size} байт")
        print(f"   Метаданные: {clean_meta}")
        print(f"   Текст чанка (первые 200): {chunk.page_content[:200]}...\n")
    
    # Обрезаем page_content: библиотека положит его в metadata["text"], лимит 40 KB
    content = chunk.page_content
    # Проверяем размер контента в UTF-8
    if len(content.encode('utf-8')) > MAX_TEXT_IN_METADATA_BYTES:
        # Обрезаем по символам и проверяем байты снова
        content = content[:MAX_PAGE_CONTENT_CHARS]
        while len(content.encode('utf-8')) > MAX_TEXT_IN_METADATA_BYTES and len(content) > 0:
            content = content[:-100]
        content += "\n[... текст обрезан из-за лимита Pinecone 40 KB ...]"
    
    clean_chunk = Document(page_content=content, metadata=clean_meta)
    clean_chunks.append(clean_chunk)

print(f"Максимальный размер метаданных (без text): {max_meta_size} байт (лимит: 40960)")
if bad_chunk_idx >= 0:
    print(f"Самый большой чанк №{bad_chunk_idx} — проверьте его метаданные!")
    if bad_chunk_idx < len(chunks):
        print(f"   Исходные метаданные: {chunks[bad_chunk_idx].metadata}")
        print(f"   Очищенные метаданные: {clean_chunks[bad_chunk_idx].metadata}")

if max_meta_size > 40960:
    print(f"\n⚠️  ОШИБКА: Размер метаданных превышает лимит Pinecone!")
    print(f"   Найдено проблемных чанков: {len(bad_chunks_found)}")
    if bad_chunks_found:
        print("   Первый проблемный чанк:")
        idx, size, meta, content = bad_chunks_found[0]
        print(f"   Индекс: {idx}, Размер: {size} байт")
        print(f"   Метаданные: {meta}")
    raise SystemExit("Исправьте метаданные перед загрузкой в Pinecone!")

# Отладка ПЕРЕД загрузкой: библиотека добавит metadata["text"] = page_content
print("\n=== ОТЛАДКА ПЕРЕД UPLOAD (библиотека добавит 'text' = page_content) ===")
max_meta_only = 0
max_total_est = 0
bad_idx = -1

for idx, chunk in enumerate(clean_chunks):
    # Симулируем что сделает langchain_pinecone
    simulated_meta = chunk.metadata.copy()
    simulated_meta["text"] = chunk.page_content
    
    meta_json = json.dumps(simulated_meta, ensure_ascii=False)
    total_est = len(meta_json.encode('utf-8'))
    
    meta_only_size = len(json.dumps(chunk.metadata, ensure_ascii=False).encode('utf-8'))
    max_meta_only = max(max_meta_only, meta_only_size)
    
    if total_est > max_total_est:
        max_total_est = total_est
        bad_idx = idx
    if total_est > 38000:
        print(f"!!! Чанк №{idx} — оценка итога (JSON): {total_est} байт (meta_only: {meta_only_size})")
        print(f"   Метаданные: {chunk.metadata}")
        print(f"   Первые 200 символов текста: {chunk.page_content[:200]}...\n")

print(f"Метаданные (без text): макс. {max_meta_only} байт")
print(f"Оценка макс. итога (JSON meta + text): {max_total_est} байт (лимит: 40960)")
if bad_idx >= 0 and max_total_est > 40960:
    print(f"Проблемный чанк №{bad_idx} — page_content обрезан до {MAX_PAGE_CONTENT_CHARS} символов")

if max_total_est > 40960:
    print(f"\n⚠️  СТОП: оценка размера > 40 KB. Уменьшите MAX_PAGE_CONTENT_CHARS.")
    raise SystemExit("См. константы MAX_PAGE_CONTENT_CHARS / MAX_TEXT_IN_METADATA_BYTES выше.")

import time as _time

print(f"Загрузка {len(clean_chunks)} чанков (пакетами по 50)...")

BATCH_SIZE = 50
total_chunks = len(clean_chunks)
total_batches = (total_chunks + BATCH_SIZE - 1) // BATCH_SIZE

# Set SKIP_BATCHES=143 to resume from batch 144 (0-indexed skip)
import os as _os
skip_n = int(_os.environ.get("SKIP_BATCHES", "0"))
if skip_n > 0:
    print(f"⏭  Пропускаем первые {skip_n} батчей (резюме с позиции {skip_n * BATCH_SIZE})")

for i in range(0, total_chunks, BATCH_SIZE):
    batch_num = i // BATCH_SIZE + 1
    if batch_num <= skip_n:
        continue

    batch = clean_chunks[i: i + BATCH_SIZE]
    print(f"   Бач {batch_num}/{total_batches}: документы {i} - {i + len(batch)}")

    # Retry with exponential backoff (3 attempts: 5s, 10s, 20s)
    for attempt in range(3):
        try:
            vector_store.add_documents(batch, batch_size=32)
            break  # success
        except Exception as e:
            wait = 5 * (2 ** attempt)
            print(f"   ⚠️  Попытка {attempt + 1}/3 не удалась: {e}")
            if attempt < 2:
                print(f"   ⏳ Ждём {wait}с перед повтором...")
                _time.sleep(wait)
            else:
                print(f"   ❌ Батч {batch_num} пропущен после 3 попыток.")

    # Brief cooldown to avoid free-tier rate limits
    _time.sleep(0.3)


# Сохраняем очищенные чанки для BM25
with open(config.CHUNKS_PICKLE_PATH, "wb") as f:
    pickle.dump(clean_chunks, f)

print("База Pinecone создана!")
print(f"Документов: {len(raw_docs)}")
print(f"Чанков: {len(chunks)}")

# Проверка после очистки: есть ли ст. 136 УК РК (баланы ауыстыру) в загружаемых чанках
uk_136 = [c for c in clean_chunks if "criminal_code" in c.metadata.get("source", "") and "136" in c.page_content]
print(f"[Проверка] Чанков со ст. 136 УК РК (после очистки, в загруженных): {len(uk_136)}")
if uk_136:
    print("Пример ст. 136 УК:", uk_136[0].page_content[:250].replace("\n", " "), "...")
else:
    print("⚠️  Ст. 136 УК не найдена в чанках — проверьте criminal_code.txt и чанкинг.")

# Тест
test_query = "принципы трудового законодательства"
print(f"\nТестовый поиск: '{test_query}'")
test_results = vector_store.similarity_search(test_query, k=3)

for i, doc in enumerate(test_results, 1):
    source = doc.metadata.get('source', 'неизвестно')
    article = doc.metadata.get('article_number', '-')
    code_ru = doc.metadata.get('code_ru', '-')
    content = doc.page_content[:150].replace('\n', ' ')
    print(f"{i}. {code_ru} ст. {article} | {source}")
    print(f"   {content}...\n")
