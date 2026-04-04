# prepare_data.py — Legal hierarchy-aware chunking for Kazakhstan RAG
# Recursive semantic splitting only: no character-count limits. Chunks follow strict legal hierarchy:
# Document -> Chapter (Глава) -> Article (Статья) -> Clause (Пункт) -> Sub-clause (Подпункт).
# Lineage metadata on every chunk: code_ru, revision_date, article_number (for Pinecone hard filtering).

import os
import re
import sys
from pathlib import Path
from typing import Optional

# Allow running as script from ai_service: python processing/prepare_data.py
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from ai_service.core import config
from langchain_community.document_loaders import DirectoryLoader, TextLoader
from langchain_core.documents import Document
from langchain_text_splitters import TextSplitter

# ──────────────────────────────────────────────────────────────────────────────
# Code name mapping: filename -> (Russian name, Kazakh name)
# ──────────────────────────────────────────────────────────────────────────────
CODE_NAMES = {
    "constitution.txt": ("Конституция РК", "ҚР Конституциясы"),
    "civil_code.txt": (
        "Гражданский кодекс РК (Общая часть)",
        "Азаматтық кодекс (Жалпы бөлім)",
    ),
    "civil_code2.txt": (
        "Гражданский кодекс РК (Особенная часть)",
        "Азаматтық кодекс (Ерекше бөлім)",
    ),
    "labor_code.txt": ("Трудовой кодекс РК", "Еңбек кодексі"),
    "tax_code.txt": ("Налоговый кодекс РК", "Салық кодексі"),
    "code_of_administrative_offenses.txt": (
        "Кодекс об административных правонарушениях РК",
        "Әкімшілік құқық бұзушылық туралы кодекс",
    ),
    "criminal_code.txt": ("Уголовный кодекс РК", "Қылмыстық кодекс"),
    "code_on_marriage_and_family.txt": (
        "Кодекс о браке и семье РК",
        "Неке және отбасы туралы кодекс",
    ),
    "code_on_public_health.txt": (
        "Кодекс о здоровье народа РК",
        "Халық денсаулығы туралы кодекс",
    ),
    "entrepreneurial_code.txt": ("Предпринимательский кодекс РК", "Кәсіпкерлік кодекс"),
    "code_on_administrative_procedures.txt": (
        "Кодекс об административных процедурах РК",
        "Әкімшілік рәсімдер туралы кодекс",
    ),
    "social_code.txt": ("Социальный кодекс РК", "Әлеуметтік кодекс"),
    "civil_procedure_code.txt": (
        "Гражданский процессуальный кодекс РК",
        "Азаматтық іс жүргізу кодексі",
    ),
    "criminal_procedure_code.txt": (
        "Уголовно-процессуальный кодекс РК",
        "Қылмыстық іс жүргізу кодексі",
    ),
    "law_on_public_procurement.txt": (
        "Закон о государственных закупках РК",
        "Мемлекеттік сатып алу туралы заң",
    ),
    "law_on_anticorruption.txt": (
        "Закон о противодействии коррупции РК",
        "Коррупцияға қарсы күрес туралы заң",
    ),
    "law_on_enforcement.txt": (
        "Закон об исполнительном производстве РК",
        "Орындау өндірісі туралы заң",
    ),
    "law_on_personal_data.txt": (
        "Закон о персональных данных РК",
        "Жеке деректер туралы заң",
    ),
    "law_on_ai.txt": (
        "Закон об искусственном интеллекте РК",
        "Жасанды интеллект туралы заң",
    ),
    "law_on_consumer_protection.txt": (
        "Закон о защите прав потребителей РК",
        "Тұтынушылардың құқықтарын қорғау туралы заң",
    ),
    "law_on_housing_relations.txt": (
        "Закон о жилищных отношениях РК",
        "Тұрғын үй қатынастары туралы заң",
    ),
    "law_on_banks.txt": (
        "Закон о банках и банковской деятельности РК",
        "Банктер және банк қызметі туралы заң",
    ),
    "land_code.txt": (
        "Земельный кодекс РК",
        "Жер кодексі",
    ),
    "law_on_military_service.txt": (
        "Закон о воинской службе и статусе военнослужащих РК",
        "Әскери қызмет және әскери қызметшілердің мәртебесі туралы заң",
    ),
    "law_on_llp.txt": (
        "Закон о товариществах с ограниченной и дополнительной ответственностью РК",
        "Жауапкершілігі шектеулі және қосымша жауапкершілігі бар серіктестіктер туралы заң",
    ),
    "law_on_notariat.txt": (
        "Закон о нотариате РК",
        "Нотариат туралы заң",
    ),
    "law_on_real_estate_registration.txt": (
        "Закон о государственной регистрации прав на недвижимое имущество РК",
        "Жылжымайтын мүлікке құқықтарды мемлекеттік тіркеу туралы заң",
    ),
    "law_on_vehicle_liability_insurance.txt": (
        "Закон об обязательном страховании гражданско-правовой ответственности владельцев транспортных средств РК",
        "Көлік құралдары иелерінің азаматтық-құқықтық жауапкершілігін міндетті сақтандыру туралы заң",
    ),
    "law_on_education.txt": (
        "Закон об образовании РК",
        "Білім туралы заң",
    ),
    "law_on_public_service.txt": (
        "Закон о государственной службе Республики Казахстан",
        "Қазақстан Республикасының мемлекеттік қызметі туралы заң",
    ),
    "law_on_child_rights.txt": (
        "Закон о правах ребенка РК",
        "Баланың құқықтары туралы заң",
    ),
    "law_on_advertising.txt": (
        "Закон о рекламе РК",
        "Жарнама туралы заң",
    ),
    "law_on_collection_activity.txt": (
        "Закон о коллекторской деятельности РК",
        "Коллекторлық қызмет туралы заң",
    ),
    "law_on_road_traffic.txt": (
        "Закон о дорожном движении РК",
        "Жол жүрісі туралы заң",
    ),
    "law_on_valuation_activity.txt": (
        "Закон об оценочной деятельности в Республике Казахстан",
        "Қазақстан Республикасындағы бағалау қызметі туралы заң",
    ),
    "law_on_legal_entities_registration.txt": (
        "Закон о государственной регистрации юридических лиц и учетной регистрации филиалов и представительств",
        "Заңды тұлғаларды мемлекеттік тіркеу және филиалдар мен өкілдіктерді есептік тіркеу туралы заң",
    ),
    "law_on_currency_regulation.txt": (
        "Закон о валютном регулировании и валютном контроле",
        "Валюталық реттеу және валюталық бақылау туралы заң",
    ),
    "law_on_digital_assets.txt": (
        "Закон о цифровых активах",
        "Цифрлық активтер туралы заң",
    ),
    "law_on_personal_data_protection.txt": (
        "Закон о персональных данных и их защите",
        "Дербес деректер және оларды қорғау туралы заң",
    ),
    "law_on_credit_bureaus.txt": (
        "Закон о кредитных бюро и формировании кредитных историй",
        "Кредиттік бюролар және кредиттік тарихты қалыптастыру туралы заң",
    ),
    "law_on_microfinance.txt": (
        "Закон о микрофинансовой деятельности",
        "Микроқаржылық қызмет туралы заң",
    ),
    "law_on_citizen_bankruptcy.txt": (
        "Закон о восстановлении платежеспособности и банкротстве граждан Республики Казахстан",
        "Қазақстан Республикасы азаматтарының төлем қабілеттілігін қалпына келтіру және банкроттығы туралы заң",
    ),
    "law_on_rehabilitation_bankruptcy.txt": (
        "Закон о реабилитации и банкротстве",
        "Оңалту және банкроттық туралы заң",
    ),
}


def _parse_source_allowlist() -> set[str]:
    raw = os.environ.get("SOURCE_ALLOWLIST", "").strip()
    if not raw:
        return set()
    return {part.strip() for part in raw.split(",") if part.strip()}

# ──────────────────────────────────────────────────────────────────────────────
# Regex patterns for legal hierarchy
# ──────────────────────────────────────────────────────────────────────────────

# Article header: "Статья 136.", "Статья 136-1.", " Статья 136 ", "136-бап." (Kazakh)
ARTICLE_RE = re.compile(
    r"^\s*(?:Статья|Стаття|Мәтін|Article|Section)\s*(\d+[а-яА-Яa-zA-Z\-]?)\.*\s*(.*?)$"
    r"|^\s*(\d+[а-яА-Яa-zA-Z\-]?)-бап\.*\s*(.*?)$",
    re.IGNORECASE | re.MULTILINE,
)

# Chapter/Section heading: "Глава 1.", "ГЛАВА 2 ", "Раздел IV", "Бөлім 3" (Kazakh)
CHAPTER_RE = re.compile(
    r"^\s*(?:Глава|ГЛАВА|Раздел|РАЗДЕЛ|Бөлім|Тарау|Chapter|Section)\s+"
    r"([\dIVXLCDMivxlcdm]+)[\.\s]\s*(.*?)$"
    r"|^\s*([\dIVXLCDMivxlcdm]+)-(?:тарау|бөлім)[\.\s]\s*(.*?)$",
    re.IGNORECASE | re.MULTILINE,
)

# Clause (пункт): lines beginning with "1.", "2.", "1) ", "2) "
CLAUSE_RE = re.compile(
    r"^(\d+)[\.\)]\s+",
    re.IGNORECASE | re.MULTILINE,
)

# Sub-clause (подпункт): lines beginning with "1)", "2)", "а)", "б)"
SUBCLAUSE_RE = re.compile(
    r"^([а-яёА-ЯЁa-zA-Z\d]+\))\s+",
    re.IGNORECASE | re.MULTILINE,
)

# Revision date patterns from Adilet file headers:
# "от 3 июля 2014 года", "от 03.07.2014", "N 226-V ЗРК от 3 июля 2014"
# "2014 жылғы 3 шілдедегі", "2014 ж. 03.07."
REVISION_DATE_RE = re.compile(
    r"(?:"
    r"от\s+(\d{1,2})\s+(января|февраля|марта|апреля|мая|июня|июля|августа|сентября|октября|ноября|декабря)\s+(\d{4})"  # от 3 июля 2014
    r"|\b(\d{4})\s+(?:жылғы|ж\.)\s+(\d{1,2})\s+(қаңтардағы|ақпандағы|наурыздағы|сәуірдегі|мамырдағы|маусымдағы|шілдедегі|тамыздағы|қыркүйектегі|қазандағы|қарашадағы|желтоқсандағы)"  # 2014 жылғы 3 шілдедегі
    r"|(?:от|ж\.)\s+(\d{2})\.(\d{2})\.(\d{4})"  # от 03.07.2014
    r")",
    re.IGNORECASE,
)

_MONTH_MAP = {
    "января": "01",
    "февраля": "02",
    "марта": "03",
    "апреля": "04",
    "мая": "05",
    "июня": "06",
    "июля": "07",
    "августа": "08",
    "сентября": "09",
    "октября": "10",
    "ноября": "11",
    "декабря": "12",
    "қаңтардағы": "01",
    "ақпандағы": "02",
    "наурыздағы": "03",
    "сәуірдегі": "04",
    "мамырдағы": "05",
    "маусымдағы": "06",
    "шілдедегі": "07",
    "тамыздағы": "08",
    "қыркүйектегі": "09",
    "қазандағы": "10",
    "қарашадағы": "11",
    "желтоқсандағы": "12",
}

# Minimum chunk length (avoid tiny fragments); no max — hierarchy-only splitting
MIN_CHUNK_LEN = 30


# ──────────────────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────────────────


def get_code_name(source_path: str) -> tuple[str, str]:
    name = Path(source_path).name
    return CODE_NAMES.get(name, (name.replace(".txt", ""), name.replace(".txt", "")))


def get_article_number(chunk_text: str) -> str | None:
    m = ARTICLE_RE.match(chunk_text.strip())
    if m:
        # Check which group matched (Russian vs Kazakh pattern)
        return m.group(1) if m.group(1) else m.group(3)
    return None


def get_article_title(chunk_text: str) -> str:
    """Extract article title from first line if it matches Article header (e.g. 'Статья 136. Подмена ребенка', '136-бап. Баланы ауыстыру')."""
    m = ARTICLE_RE.match(chunk_text.strip())
    if m:
        # Russian title is in group 2, Kazakh title is in group 4
        title = m.group(2) if m.group(1) else m.group(4)
        if title:
            return title.strip()[:200]
    return ""


def _extract_revision_date(text: str) -> str:
    """
    Scans the first 60 lines of a document for a revision/enactment date.
    Returns ISO date string 'YYYY-MM-DD' or '' if not found.
    """
    header = "\n".join(text.splitlines()[:60])
    m = REVISION_DATE_RE.search(header)
    if not m:
        return ""
    # Named word-month form: от DD Month YYYY (Russian)
    if m.group(1):
        day = m.group(1).zfill(2)
        month = _MONTH_MAP.get(m.group(2).lower(), "01")
        year = m.group(3)
        return f"{year}-{month}-{day}"
    # Named word-month form: YYYY жылғы DD Month (Kazakh)
    if m.group(4):
        year = m.group(4)
        day = m.group(5).zfill(2)
        month = _MONTH_MAP.get(m.group(6).lower(), "01")
        return f"{year}-{month}-{day}"
    # Numeric form: от DD.MM.YYYY
    if m.group(7):
        day, month, year = m.group(7), m.group(8), m.group(9)
        return f"{year}-{month}-{day}"
    return ""


def _detect_chapter(line: str) -> tuple[str, str] | None:
    """
    Returns (chapter_number, chapter_title) if the line is a chapter heading, else None.
    """
    m = CHAPTER_RE.match(line)
    if m:
        # Check if Russian or Kazakh pattern matched
        if m.group(1):
            num = m.group(1).strip()
            title_part = m.group(2).strip()
        else:
            num = m.group(3).strip()
            title_part = m.group(4).strip()
        return num, title_part
    return None


def _detect_clause_level(chunk_text: str) -> str:
    """
    Returns 'subclause', 'clause', or 'article' based on chunk content structure.
    """
    stripped = chunk_text.strip()
    # Sub-clause starts with letter or short alpha marker like "а)" "б)" "1)"
    if SUBCLAUSE_RE.match(stripped) and not ARTICLE_RE.match(stripped):
        return "subclause"
    if CLAUSE_RE.match(stripped) and not ARTICLE_RE.match(stripped):
        return "clause"
    return "article"


def _split_article_into_clauses(article_text: str) -> list[str]:
    """
    Recursive semantic split: Article → Clauses (1. 2. 3. or 1) 2)).
    Each returned string is one clause (no character-count splitting).
    """
    parts = CLAUSE_RE.split(article_text)
    if len(parts) <= 1:
        return (
            [article_text]
            if article_text.strip() and len(article_text.strip()) >= MIN_CHUNK_LEN
            else []
        )

    chunks = []
    intro = parts[0].strip()
    for i in range(1, len(parts), 2):
        if i + 1 >= len(parts):
            break
        num, content = parts[i], parts[i + 1]
        clause_text = f"{num}) " + content.strip()
        if intro and i == 1:
            clause_text = intro + "\n" + clause_text
            intro = ""
        if clause_text.strip() and len(clause_text.strip()) >= MIN_CHUNK_LEN:
            chunks.append(clause_text)
    return (
        chunks
        if chunks
        else (
            [article_text]
            if article_text.strip() and len(article_text.strip()) >= MIN_CHUNK_LEN
            else []
        )
    )


def _split_clause_into_subclauses(clause_text: str) -> list[str]:
    """
    Recursive semantic split: Clause → Sub-clauses (а) б) в) or 1) 2)).
    SUBCLAUSE_RE captures marker including parenthesis (e.g. "а)"); split gives [intro, "а)", content, ...].
    """
    parts = SUBCLAUSE_RE.split(clause_text)
    if len(parts) <= 1:
        return (
            [clause_text]
            if clause_text.strip() and len(clause_text.strip()) >= MIN_CHUNK_LEN
            else []
        )

    chunks = []
    intro = parts[0].strip()
    for i in range(1, len(parts), 2):
        if i + 1 >= len(parts):
            break
        marker, content = parts[i], parts[i + 1]  # marker is e.g. "а)" or "1)"
        sub_text = marker + " " + content.strip()
        if intro and i == 1:
            sub_text = intro + "\n" + sub_text
            intro = ""
        if sub_text.strip() and len(sub_text.strip()) >= MIN_CHUNK_LEN:
            chunks.append(sub_text)
    return (
        chunks
        if chunks
        else (
            [clause_text]
            if clause_text.strip() and len(clause_text.strip()) >= MIN_CHUNK_LEN
            else []
        )
    )


def _split_article_by_hierarchy(article_text: str) -> list[str]:
    """
    Strict legal hierarchy: Article → Clause → Sub-clause.
    No character-count splitting. Each chunk is one semantic unit.
    """
    clauses = _split_article_into_clauses(article_text)
    if not clauses:
        return []
    result = []
    for clause in clauses:
        subclauses = _split_clause_into_subclauses(clause)
        for sub in subclauses:
            if sub.strip() and len(sub.strip()) >= MIN_CHUNK_LEN:
                result.append(sub)
    return (
        result
        if result
        else (
            [article_text]
            if article_text.strip() and len(article_text.strip()) >= MIN_CHUNK_LEN
            else []
        )
    )


def _split_preamble_by_hierarchy(text: str) -> list[str]:
    """
    Preamble / intro: split only by Chapter/Раздел/Бөлім/Тарау boundaries.
    No character-count limits.
    """
    t = text.strip()
    if not t or len(t) < MIN_CHUNK_LEN:
        return []
    section_pattern = re.compile(
        r"^(?:Глава|Раздел|ГЛАВА|РАЗДЕЛ|Бөлім|Тақырып)\s+[\dIVXLCDM]+[.\s]\s*(.*?)$"
        r"|^[\dIVXLCDM]+-(?:тарау|бөлім)[.\s]\s*(.*?)$",
        re.IGNORECASE | re.MULTILINE,
    )
    parts = section_pattern.split(t)
    if len(parts) > 1:
        result = []
        for i in range(1, len(parts), 2):
            header = parts[i] if i < len(parts) and parts[i] is not None else ""
            body = (
                parts[i + 1]
                if i + 1 < len(parts) and parts[i + 1] is not None
                else ""
            )
            chunk = (header + body).strip()
            if len(chunk) >= MIN_CHUNK_LEN:
                result.append(chunk)
        return result if result else [t]
    return [t]


# ──────────────────────────────────────────────────────────────────────────────
# Main splitter
# ──────────────────────────────────────────────────────────────────────────────


class ArticleTextSplitter(TextSplitter):
    """
    Recursive semantic splitter — no character-count splitting.
    Chunks strictly follow legal hierarchy: Article (Статья) → Clause (Пункт) → Sub-clause (Подпункт).

    - Preamble: split only by Chapter/Раздел/Бөлім boundaries.
    - Each Article: split into Clauses (1. 2. 3. or 1) 2)); each Clause into Sub-clauses (а) б) в)) if present.
    - One chunk = one semantic unit (full article if no clauses, else one clause or one sub-clause).
    """

    def split_text(self, text: str) -> list[str]:
        """Return raw text chunks by hierarchy only; chapter/article context set in create_documents()."""
        matches = list(ARTICLE_RE.finditer(text))
        if not matches:
            return [
                c
                for c in _split_preamble_by_hierarchy(text)
                if c and len(c.strip()) >= MIN_CHUNK_LEN
            ]

        chunks = []
        prev_end = 0
        for idx, match in enumerate(matches):
            start = match.start()
            if start > prev_end:
                preamble = text[prev_end:start].strip()
                if preamble:
                    for c in _split_preamble_by_hierarchy(preamble):
                        if c and len(c.strip()) >= MIN_CHUNK_LEN:
                            chunks.append(c)
            next_start = (
                matches[idx + 1].start() if idx + 1 < len(matches) else len(text)
            )
            article_text = text[start:next_start].strip()
            for c in _split_article_by_hierarchy(article_text):
                if c and len(c.strip()) >= MIN_CHUNK_LEN:
                    chunks.append(c)
            prev_end = next_start

        return chunks

    def create_documents(
        self, texts: list[str], metadatas: list[dict] = None
    ) -> list[Document]:
        """
        Creates Documents with mandatory lineage metadata for Pinecone hard filtering:
        - code_ru, revision_date, article_number (required on every chunk).
        Plus: source, code_kz, chapter_title, chapter_number, clause_level.
        """
        _metadatas = metadatas or [{} for _ in texts]
        documents = []

        for i, text in enumerate(texts):
            base_meta = dict(_metadatas[i])
            source = base_meta.get("source", "")
            code_ru, code_kz = get_code_name(source)
            source_short = Path(source).name if source else ""

            # Lineage: revision date once per source (mandatory; empty if not found)
            revision_date = _extract_revision_date(text) or ""

            # Track chapter context as we scan the raw text
            current_chapter_number: str = ""
            current_chapter_title: str = ""

            for chunk in self.split_text(text):
                if not chunk or len(chunk.strip()) < MIN_CHUNK_LEN:
                    continue

                # Update chapter context from lines at the start of this chunk
                for line in chunk.splitlines()[:5]:
                    chapter_hit = _detect_chapter(line)
                    if chapter_hit:
                        current_chapter_number, current_chapter_title = chapter_hit
                        break

                art_num = get_article_number(chunk)
                clause_level = _detect_clause_level(chunk)
                article_title = get_article_title(chunk)

                # Mandatory lineage metadata for Pinecone hard filtering (code_ru, revision_date, article_number)
                meta: dict = {
                    "source": source_short,
                    "code_ru": code_ru,
                    "code_kz": code_kz,
                    "article_number": str(art_num) if art_num is not None else "",
                    "revision_date": revision_date,
                    "clause_level": clause_level,
                }
                if current_chapter_title:
                    meta["chapter_title"] = current_chapter_title[:150]
                if current_chapter_number:
                    meta["chapter_number"] = current_chapter_number[:20]
                if article_title:
                    meta["article_title"] = article_title

                documents.append(Document(page_content=chunk, metadata=meta))

        return documents


# ──────────────────────────────────────────────────────────────────────────────
# Module-level execution: load documents and produce chunks
# ──────────────────────────────────────────────────────────────────────────────

if not config.DOCUMENTS_DIR.exists():
    raise SystemExit(
        f"Папка {config.DOCUMENTS_DIR} не найдена. Сначала запустите: python fetch_adilet.py"
    )

loader = DirectoryLoader(
    str(config.DOCUMENTS_DIR),
    glob="**/*.txt",
    loader_cls=TextLoader,
    loader_kwargs={"encoding": "utf-8"},
)
raw_docs = loader.load()
source_allowlist = _parse_source_allowlist()
if source_allowlist:
    raw_docs = [
        doc
        for doc in raw_docs
        if Path(doc.metadata.get("source", "")).name in source_allowlist
    ]
    print(
        f"SOURCE_ALLOWLIST активен: {len(source_allowlist)} файлов, загружено {len(raw_docs)} документов"
    )
if not raw_docs:
    raise SystemExit(
        f"В {config.DOCUMENTS_DIR} нет .txt файлов. Запустите: python fetch_adilet.py"
    )

article_splitter = ArticleTextSplitter()
chunks = article_splitter.create_documents(
    [doc.page_content for doc in raw_docs],
    [doc.metadata for doc in raw_docs],
)

print(f"Всего документов: {len(raw_docs)}")
print(f"Всего чанков (статей): {len(chunks)}")

# Debug: show hierarchy metadata coverage
chapter_chunks = [c for c in chunks if c.metadata.get("chapter_title")]
dated_chunks = [c for c in chunks if c.metadata.get("revision_date")]
clause_chunks = [c for c in chunks if c.metadata.get("clause_level") == "clause"]
print(f"  └─ С chapter_title: {len(chapter_chunks)}")
print(f"  └─ С revision_date: {len(dated_chunks)}")
print(f"  └─ clause-level чанков: {len(clause_chunks)}")

if chunks:
    print("\nПример чанка:")
    print(chunks[0].page_content[:300], "...")
    print("Метаданные:", chunks[0].metadata)
