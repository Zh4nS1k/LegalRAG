# prepare_data.py — метаданные: номер статьи, кодекс (RU/KZ)

import re
from pathlib import Path

from ai_service.core import config
from langchain_community.document_loaders import DirectoryLoader, TextLoader
from langchain_core.documents import Document
from langchain_text_splitters import TextSplitter

# Маппинг имя_файла -> название кодекса (рус/каз)
CODE_NAMES = {
    "constitution.txt": ("Конституция РК", "ҚР Конституциясы"),
    "civil_code.txt": ("Гражданский кодекс РК (Общая часть)", "Азаматтық кодекс (Жалпы бөлім)"),
    "civil_code2.txt": ("Гражданский кодекс РК (Особенная часть)", "Азаматтық кодекс (Ерекше бөлім)"),
    "labor_code.txt": ("Трудовой кодекс РК", "Еңбек кодексі"),
    "tax_code.txt": ("Налоговый кодекс РК", "Салық кодексі"),
    "code_of_administrative_offenses.txt": ("Кодекс об административных правонарушениях РК", "Әкімшілік құқық бұзушылық туралы кодекс"),
    "criminal_code.txt": ("Уголовный кодекс РК", "Қылмыстық кодекс"),
    "code_on_marriage_and_family.txt": ("Кодекс о браке и семье РК", "Неке және отбасы туралы кодекс"),
    "code_on_public_health.txt": ("Кодекс о здоровье народа РК", "Халық денсаулығы туралы кодекс"),
    "entrepreneurial_code.txt": ("Предпринимательский кодекс РК", "Кәсіпкерлік кодекс"),
    "code_on_administrative_procedures.txt": ("Кодекс об административных процедурах РК", "Әкімшілік рәсімдер туралы кодекс"),
    "social_code.txt": ("Социальный кодекс РК", "Әлеуметтік кодекс"),
    "civil_procedure_code.txt": ("Гражданский процессуальный кодекс РК", "Азаматтық іс жүргізу кодексі"),
    "criminal_procedure_code.txt": ("Уголовно-процессуальный кодекс РК", "Қылмыстық іс жүргізу кодексі"),
    "law_on_public_procurement.txt": ("Закон о государственных закупках РК", "Мемлекеттік сатып алу туралы заң"),
    "law_on_anticorruption.txt": ("Закон о противодействии коррупции РК", "Коррупцияға қарсы күрес туралы заң"),
    "law_on_enforcement.txt": ("Закон об исполнительном производстве РК", "Орындау өндірісі туралы заң"),
    "law_on_personal_data.txt": ("Закон о персональных данных РК", "Жеке деректер туралы заң"),
    "law_on_ai.txt": ("Закон об искусственном интеллекте РК", "Жасанды интеллект туралы заң"),
}

# Regex для извлечения номера статьи из начала чанка (^\s* — ведущий пробел в файлах Adilet)
ARTICLE_HEADER = re.compile(
    r'^\s*(?:Статья|Стаття|Мәтін|Article|Section)\s*(\d+[а-яА-Яa-zA-Z]?)\.?\s*',
    re.IGNORECASE
)


def get_article_number(chunk_text: str) -> str | None:
    """Извлекает номер статьи из начала текста чанка."""
    m = ARTICLE_HEADER.match(chunk_text.strip())
    return m.group(1) if m else None


def get_code_name(source_path: str) -> tuple[str, str]:
    """По пути файла возвращает (название_рус, название_каз)."""
    name = Path(source_path).name
    return CODE_NAMES.get(name, (name.replace(".txt", ""), name.replace(".txt", "")))


# Параметры чанкинга по типу текста (см. таблицу рекомендаций)
CHUNK_SHORT_MAX = 1000       # ≤1000: статья целиком
CHUNK_MEDIUM_MAX = 2500      # 1000–2500: статья целиком или по пунктам
CHUNK_MEDIUM_SPLIT = 1500    # если >1500 в средней — делить по пунктам
CHUNK_LONG_MIN = 2500       # >2500: делить по пунктам/подпунктам
PARAGRAPH_OVERLAP = 175     # overlap 150–200 символов между чанками
PREAMBLE_MAX = 800          # преамбула/раздел: один чанк если < 800 символов
MIN_CHUNK_LEN = 50          # минимальная длина чанка

# Паттерн пунктов: 1), 2), 1. 2.
PARAGRAPH_PATTERN = re.compile(
    r'(?m)^(\d+)[\.\)]\s+',
    re.IGNORECASE
)


def _split_by_paragraphs(text: str, overlap: int = PARAGRAPH_OVERLAP) -> list[str]:
    """Делит текст по пунктам 1), 2), 3) с overlap 150–200 символов."""
    parts = PARAGRAPH_PATTERN.split(text)
    if len(parts) <= 1:
        return [text] if text.strip() else []

    # parts[0] — вступление до первого пункта, parts[1]="1", parts[2]="текст после 1)", ...
    chunks = []
    intro = parts[0].strip()
    for i in range(1, len(parts), 2):
        if i + 1 >= len(parts):
            break
        num, content = parts[i], parts[i + 1]
        chunk_text = f"{num}) " + content.strip()
        if intro and i == 1:
            chunk_text = intro + "\n" + chunk_text
            intro = ""
        if chunk_text.strip() and len(chunk_text) > MIN_CHUNK_LEN:
            chunks.append(chunk_text)
    if not chunks:
        return [text] if text.strip() else []
    # Overlap: последние N символов предыдущего чанка в начало следующего
    result = []
    for k, c in enumerate(chunks):
        if k > 0 and overlap > 0 and len(result[-1]) > overlap:
            prefix = result[-1][-overlap:]
            c = prefix + "\n" + c
        result.append(c)
    return result


def _maybe_split_article(article_text: str) -> list[str]:
    """
    Применяет правила из таблицы:
    - Короткая (≤1000): целиком
    - Средняя (1000–2500): целиком или по пунктам если >1500
    - Длинная (>2500): по пунктам
    """
    n = len(article_text)
    if n <= CHUNK_SHORT_MAX:
        return [article_text]
    if n <= CHUNK_MEDIUM_MAX:
        if n <= CHUNK_MEDIUM_SPLIT:
            return [article_text]
        sub = _split_by_paragraphs(article_text)
        return sub if len(sub) > 1 else [article_text]
    # Длинная статья
    sub = _split_by_paragraphs(article_text)
    return sub if len(sub) > 1 else [article_text]


def _maybe_split_preamble(text: str) -> list[str]:
    """Преамбула/раздел: один чанк если < 800 символов."""
    t = text.strip()
    if not t or len(t) < MIN_CHUNK_LEN:
        return []
    if len(t) <= PREAMBLE_MAX:
        return [t]
    # Если длиннее — делим по заголовкам глав/разделов
    section_pattern = re.compile(
        r'(?m)^(?:Глава|Раздел|ГЛАВА|РАЗДЕЛ|Бөлім|Тақырып)\s+[\dIVXLCDM]+[\.\s]',
        re.IGNORECASE
    )
    parts = section_pattern.split(t)
    if len(parts) > 1:
        result = []
        for i in range(1, len(parts), 2):
            header = parts[i] if i < len(parts) else ""
            body = parts[i + 1] if i + 1 < len(parts) else ""
            chunk = (header + body).strip()
            if len(chunk) > MIN_CHUNK_LEN:
                result.append(chunk)
        return result if result else [t]
    return [t]


class ArticleTextSplitter(TextSplitter):
    """
    Splitter по таблице рекомендаций:
    - Короткая статья (≤1000): целиком
    - Средняя (1000–2500): целиком или по пунктам при >1500
    - Длинная (>2500): по пунктам
    - Преамбула/раздел: один чанк если < 800 символов
    """

    def split_text(self, text: str) -> list[str]:
        # ^\s* — в УК РК заголовки часто с ведущим пробелом (" Статья 136.")
        article_pattern = re.compile(
            r'(?m)^\s*(Статья|Стаття|Мәтін|Article|Section)\s*(\d+[а-яА-Яa-zA-Z]?)\.?\s*(.*?$)',
            re.IGNORECASE | re.DOTALL
        )
        matches = list(article_pattern.finditer(text))
        if not matches:
            preamble_chunks = _maybe_split_preamble(text)
            return [c for c in preamble_chunks if c and len(c) > MIN_CHUNK_LEN]

        chunks = []
        prev_end = 0
        for idx, match in enumerate(matches):
            start = match.start()
            if start > prev_end:
                preamble = text[prev_end:start].strip()
                if preamble:
                    for c in _maybe_split_preamble(preamble):
                        if c and len(c) > MIN_CHUNK_LEN:
                            chunks.append(c)
            next_start = matches[idx + 1].start() if idx + 1 < len(matches) else len(text)
            article_text = text[start:next_start].strip()
            for c in _maybe_split_article(article_text):
                if c and len(c) > MIN_CHUNK_LEN:
                    chunks.append(c)
            prev_end = next_start

        return chunks

    def create_documents(self, texts: list[str], metadatas: list[dict] = None) -> list[Document]:
        """
        Тексты в Document с метаданными: source, code_ru, code_kz, article_number.
        Метаданные ограничены до 40 KB (Pinecone limit) — храним только короткие поля.
        """
        _metadatas = metadatas or [{} for _ in texts]
        documents = []
        for i, text in enumerate(texts):
            base_meta = dict(_metadatas[i])
            source = base_meta.get("source", "")
            code_ru, code_kz = get_code_name(source)
            for chunk in self.split_text(text):
                # Только короткие метаданные (Pinecone limit: 40 KB на вектор)
                # source может быть длинным (путь), обрезаем до имени файла
                source_short = Path(source).name if source else ""
                meta = {
                    "source": source_short,  # только имя файла, не полный путь
                    "code_ru": code_ru,
                    "code_kz": code_kz,
                }
                art_num = get_article_number(chunk)
                if art_num is not None:
                    meta["article_number"] = str(art_num)  # гарантируем строку
                # НЕ добавляем другие поля из base_meta — они могут содержать длинный текст
                # (например, page_content, full_text, chapter_content и т.д.)
                new_doc = Document(page_content=chunk, metadata=meta)
                documents.append(new_doc)
        return documents

# Загрузка документов
if not config.DOCUMENTS_DIR.exists():
    raise SystemExit(
        f"Папка {config.DOCUMENTS_DIR} не найдена. Сначала запустите: python fetch_adilet.py"
    )
loader = DirectoryLoader(
    str(config.DOCUMENTS_DIR),
    glob="**/*.txt",
    loader_cls=TextLoader,
    loader_kwargs={"encoding": "utf-8"}
)
raw_docs = loader.load()
if not raw_docs:
    raise SystemExit(
        f"В {config.DOCUMENTS_DIR} нет .txt файлов. Запустите: python fetch_adilet.py"
    )

# Используем кастомный splitter
article_splitter = ArticleTextSplitter()

# Разделяем
chunks = article_splitter.create_documents([doc.page_content for doc in raw_docs], [doc.metadata for doc in raw_docs])

print(f"Всего документов: {len(raw_docs)}")
print(f"Всего чанков (статей): {len(chunks)}")
# Пример: вывести первую чанку
if chunks:
    print("Пример чанка:\n", chunks[0].page_content[:300], "...")