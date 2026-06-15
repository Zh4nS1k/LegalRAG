# prepare_data.py — Legal hierarchy-aware chunking for Kazakhstan RAG
# Recursive semantic splitting only: no character-count limits. Chunks follow strict legal hierarchy:
# Document -> Chapter (Глава) -> Article (Статья) -> Clause (Пункт) -> Sub-clause (Подпункт).
# Lineage metadata on every chunk: code_ru, revision_date, article_number (for Pinecone hard filtering).

import os
import re
import sys
import pickle
from collections import Counter
from functools import lru_cache
from pathlib import Path

# Allow running as script from engine: python processing/prepare_data.py
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from engine.core import config
from engine.core.code_registry import get_code_name
from engine.processing.corpus_quality import (
    compute_corpus_version,
    sorted_documents_by_source,
)
from langchain_community.document_loaders import DirectoryLoader, TextLoader
from langchain_core.documents import Document
from langchain_text_splitters import TextSplitter

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

# Sub-clause (подпункт): lettered items like "а)", "б)", "a)", "b)".
# Numeric items "1)" / "2)" are treated as clauses at the article level.
SUBCLAUSE_RE = re.compile(
    r"^([а-яёА-ЯЁa-zA-Z]+\))\s+",
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
SOURCE_REVISION_DATE_FALLBACKS: dict[str, str] = {
    "law_on_mass_media.txt": "2024-06-19",
}


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


def _build_chunk_path(
    article_number: str | None,
    clause_number: str = "",
    subclause_number: str = "",
) -> str:
    parts: list[str] = []
    if article_number:
        parts.append(f"ст. {article_number}")
    if clause_number:
        parts.append(f"п. {clause_number}")
    if subclause_number:
        parts.append(f"подп. {subclause_number}")
    return " ".join(parts)


def _extract_article_header(article_text: str) -> str:
    """Keep the legal unit anchored: every clause/subclause should carry the article header."""
    lines = [line.rstrip() for line in article_text.splitlines()]
    header_lines: list[str] = []
    seen_non_empty = False
    for line in lines:
        stripped = line.strip()
        if not stripped:
            if header_lines:
                header_lines.append("")
            continue
        if CLAUSE_RE.match(stripped):
            break
        header_lines.append(line)
        seen_non_empty = True
    if not seen_non_empty:
        return ""
    return "\n".join(header_lines).strip()


def _split_article_into_clauses(article_text: str) -> list[dict[str, str]]:
    """
    Recursive semantic split: Article → Clauses (1. 2. 3. or 1) 2)).
    Each returned string is one clause (no character-count splitting).
    """
    article_number = get_article_number(article_text) or ""
    article_header = _extract_article_header(article_text)
    parts = CLAUSE_RE.split(article_text)
    if len(parts) <= 1:
        if article_text.strip() and len(article_text.strip()) >= MIN_CHUNK_LEN:
            return [
                {
                    "text": article_text,
                    "clause_level": "article",
                    "clause_number": "",
                    "subclause_number": "",
                    "path": _build_chunk_path(article_number),
                }
            ]
        return []

    chunks: list[dict[str, str]] = []
    intro = parts[0].strip()
    for i in range(1, len(parts), 2):
        if i + 1 >= len(parts):
            break
        num, content = parts[i], parts[i + 1]
        clause_text = f"{num}) " + content.strip()
        if article_header:
            clause_text = article_header + "\n" + clause_text
        elif intro and i == 1:
            clause_text = intro + "\n" + clause_text
        if i == 1:
            intro = ""
        if clause_text.strip() and len(clause_text.strip()) >= MIN_CHUNK_LEN:
            chunks.append(
                {
                    "text": clause_text,
                    "clause_level": "clause",
                    "clause_number": num.strip(),
                    "subclause_number": "",
                    "path": _build_chunk_path(article_number, num.strip()),
                }
            )
    if chunks:
        return chunks
    if article_text.strip() and len(article_text.strip()) >= MIN_CHUNK_LEN:
        return [
            {
                "text": article_text,
                "clause_level": "article",
                "clause_number": "",
                "subclause_number": "",
                "path": _build_chunk_path(article_number),
            }
        ]
    return []


def _split_clause_into_subclauses(clause_chunk: dict[str, str]) -> list[dict[str, str]]:
    """
    Recursive semantic split: Clause → Sub-clauses (а) б) в) or 1) 2)).
    SUBCLAUSE_RE captures marker including parenthesis (e.g. "а)"); split gives [intro, "а)", content, ...].
    """
    clause_text = clause_chunk["text"]
    article_number = get_article_number(clause_text) or ""
    clause_number = clause_chunk.get("clause_number", "").strip()
    parts = SUBCLAUSE_RE.split(clause_text)
    if len(parts) <= 1:
        return [clause_chunk] if clause_text.strip() and len(clause_text.strip()) >= MIN_CHUNK_LEN else []

    chunks: list[dict[str, str]] = []
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
            sub_marker = marker.rstrip(")").strip()
            chunks.append(
                {
                    "text": sub_text,
                    "clause_level": "subclause",
                    "clause_number": clause_number,
                    "subclause_number": sub_marker,
                    "path": _build_chunk_path(
                        article_number,
                        clause_number,
                        sub_marker,
                    ),
                }
            )
    return chunks if chunks else [clause_chunk]


def _split_article_by_hierarchy(article_text: str) -> list[dict[str, str]]:
    """
    Strict legal hierarchy: Article → Clause → Sub-clause.
    No character-count splitting. Each chunk is one semantic unit.
    """
    clauses = _split_article_into_clauses(article_text)
    if not clauses:
        return []
    result = []
    for clause_chunk in clauses:
        subclauses = _split_clause_into_subclauses(clause_chunk)
        for sub in subclauses:
            if sub["text"].strip() and len(sub["text"].strip()) >= MIN_CHUNK_LEN:
                result.append(sub)
    if result:
        parent_article_text = article_text[:2000]
        for item in result:
            item["parent_article_text"] = parent_article_text
        return result
    if article_text.strip() and len(article_text.strip()) >= MIN_CHUNK_LEN:
        article_number = get_article_number(article_text) or ""
        return [
            {
                "text": article_text,
                "clause_level": "article",
                "clause_number": "",
                "subclause_number": "",
                "path": _build_chunk_path(article_number),
                "parent_article_text": article_text[:2000],
            }
        ]
    return []


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


def _is_bad_chunk_text(text: str) -> bool:
    normalized = re.sub(r"\s+", " ", (text or "").strip().lower())
    if not normalized:
        return True
    if len(normalized) < 80 and "статья" not in normalized and "-бап" not in normalized:
        return True
    bad_markers = (
        "оглавление",
        "мазмұны",
        "вниманию пользователей",
        "қолданушылар назарына",
        "қолданушыларға ыңғайлы болуы үшін",
        "зқаи-ның ескертпесі",
        "для удобства пользования",
        "сноска.",
        "примечание изпи",
        "примечание рцпи",
        "содержание",
        "осы заңның қолданысқа енгізілу тәртібін",
    )
    return any(marker in normalized[:300] for marker in bad_markers)


def _build_indexable_text(chunk_text: str, meta: dict) -> str:
    prefix = _build_contextual_prefix(meta, chunk_text)
    parts: list[str] = []
    code_ru = str(meta.get("code_ru") or "").strip()
    article_number = str(meta.get("article_number") or "").strip()
    article_title = str(meta.get("article_title") or "").strip()
    chapter_number = str(meta.get("chapter_number") or "").strip()
    chapter_title = str(meta.get("chapter_title") or "").strip()
    clause_level = str(meta.get("clause_level") or "").strip()
    clause_number = str(meta.get("clause_number") or "").strip()
    subclause_number = str(meta.get("subclause_number") or "").strip()

    if code_ru:
        parts.append(f"Кодекс: {code_ru}")
    if chapter_number or chapter_title:
        chapter_label = " ".join(
            part for part in (chapter_number, chapter_title) if part
        ).strip()
        if chapter_label:
            parts.append(f"Глава: {chapter_label}")
    if article_number:
        article_label = article_number
        if article_title:
            article_label += f". {article_title}"
        parts.append(f"Статья: {article_label}")
    if clause_level == "clause" and clause_number:
        parts.append(f"Пункт: {clause_number}")
    elif clause_level == "subclause":
        if clause_number:
            parts.append(f"Пункт: {clause_number}")
        if subclause_number:
            parts.append(f"Подпункт: {subclause_number}")

    if not parts:
        return f"{prefix}\nТекст: {chunk_text.strip()}".strip()
    header = "\n".join(parts).strip()
    return f"{prefix}\n{header}\nТекст: {chunk_text.strip()}".strip()


def _infer_jurisdiction(meta: dict) -> str:
    return str(meta.get("jurisdiction") or "RK").strip() or "RK"


def _infer_document_type(meta: dict) -> str:
    doc_type = str(meta.get("document_type") or "").strip().lower()
    if doc_type:
        return doc_type
    code_ru = str(meta.get("code_ru") or "").lower()
    if "кодекс" in code_ru:
        return "code"
    return "law"


def _infer_status(text: str, meta: dict) -> str:
    status = str(meta.get("status") or "").strip()
    if status:
        return status
    normalized = re.sub(r"\s+", " ", (text or "").lower())
    if any(
        marker in normalized
        for marker in (
            "утратил силу",
            "признать утратившим силу",
            "признан утратившим силу",
            "остановил действие",
        )
    ):
        return "утратил силу"
    return "действует"


def _build_contextual_prefix(meta: dict, chunk_text: str) -> str:
    if not getattr(config, "ENABLE_CONTEXTUAL_PREFIX", True):
        return ""

    code_ru = str(meta.get("code_ru") or "").strip()
    code_kz = str(meta.get("code_kz") or "").strip()
    article_number = str(meta.get("article_number") or "").strip()
    article_title = str(meta.get("article_title") or "").strip()
    chapter_number = str(meta.get("chapter_number") or "").strip()
    chapter_title = str(meta.get("chapter_title") or "").strip()
    clause_level = str(meta.get("clause_level") or "").strip()
    clause_number = str(meta.get("clause_number") or "").strip()
    subclause_number = str(meta.get("subclause_number") or "").strip()
    jurisdiction = _infer_jurisdiction(meta)
    document_type = _infer_document_type(meta)
    status = _infer_status(chunk_text, meta)

    scope_bits: list[str] = []
    if chapter_number or chapter_title:
        scope_bits.append(
            "Chapter "
            + " ".join(part for part in (chapter_number, chapter_title) if part).strip()
        )
    if article_number:
        article_label = f"Article {article_number}"
        if article_title:
            article_label += f" ({article_title})"
        scope_bits.append(article_label)
    if clause_level == "clause" and clause_number:
        scope_bits.append(f"Clause {clause_number}")
    elif clause_level == "subclause":
        if clause_number:
            scope_bits.append(f"Clause {clause_number}")
        if subclause_number:
            scope_bits.append(f"Subclause {subclause_number}")

    scope = ", ".join(scope_bits) if scope_bits else "document-level legal text"
    source_bits = [f"{code_ru or 'unknown legal source'}"]
    if code_kz:
        source_bits.append(code_kz)

    return (
        "Context: This chunk is from "
        f"{scope} of {' / '.join(source_bits)}. "
        f"Jurisdiction: {jurisdiction}. "
        f"Document type: {document_type}. "
        f"Status: {status}."
    )


@lru_cache(maxsize=128)
def _generate_llm_context_summary(summary_seed: str) -> str:
    if not getattr(config, "USE_LLM_SUMMARIES", False):
        return ""
    try:
        from engine.retrieval.rag_chain import get_llm

        llm = get_llm()
        prompt = (
            "You are writing a compact legal retrieval summary for embedding. "
            "Return one concise paragraph in English only. Include the legal document name, jurisdiction, "
            "current status, main chapters or topics, and the main legal scope. "
            "Do not add citations or bullet points. Text:\n"
            f"{summary_seed}"
        )
        response = llm.invoke(prompt)
        text = response.content if hasattr(response, "content") else str(response)
        cleaned = " ".join(str(text).split())
        return cleaned[:900]
    except Exception:
        return ""


def _build_summary_document(
    source_short: str,
    base_meta: dict,
    source_chunks: list[Document],
    *,
    corpus_version: str = "",
) -> Document:
    code_ru = str(base_meta.get("code_ru") or "").strip()
    code_kz = str(base_meta.get("code_kz") or "").strip()
    jurisdiction = _infer_jurisdiction(base_meta)
    document_type = _infer_document_type(base_meta)
    status = _infer_status(" ".join(doc.page_content for doc in source_chunks[:3]), base_meta)
    chapter_titles = [
        str(meta.get("chapter_title") or "").strip()
        for meta in (doc.metadata or {} for doc in source_chunks)
        if str(meta.get("chapter_title") or "").strip()
    ]
    article_pairs = [
        (
            str((doc.metadata or {}).get("article_number") or "").strip(),
            str((doc.metadata or {}).get("article_title") or "").strip(),
        )
        for doc in source_chunks
        if str((doc.metadata or {}).get("article_number") or "").strip()
    ]
    article_counter = Counter(article for article, _ in article_pairs if article)
    top_articles = [article for article, _ in article_counter.most_common(8)]
    top_titles = [title for _, title in article_pairs if title][:8]

    summary_seed = (
        f"Document: {code_ru or source_short}\n"
        f"Source: {source_short}\n"
        f"Jurisdiction: {jurisdiction}\n"
        f"Type: {document_type}\n"
        f"Status: {status}\n"
        f"Chapters: {', '.join(dict.fromkeys(chapter_titles[:8]))}\n"
        f"Articles: {', '.join(dict.fromkeys(top_articles))}\n"
        f"Article titles: {', '.join(dict.fromkeys(top_titles))}\n"
    )
    llm_summary = _generate_llm_context_summary(summary_seed)
    summary_text = llm_summary or (
        "Summary index entry for retrieval.\n"
        f"This document is {code_ru or source_short} and is classified as {document_type} in {jurisdiction}.\n"
        f"Current status: {status}.\n"
        f"Observed chapters: {', '.join(dict.fromkeys(chapter_titles[:5])) or 'n/a'}.\n"
        f"Representative articles: {', '.join(top_articles) or 'n/a'}.\n"
        f"Representative article titles: {', '.join(top_titles) or 'n/a'}.\n"
        "Use this summary only to route retrieval to the full article-level chunks."
    )

    meta = {
        "source": source_short,
        "code_ru": code_ru,
        "code_kz": code_kz,
        "revision_date": str(base_meta.get("revision_date") or ""),
        "jurisdiction": jurisdiction,
        "document_type": document_type,
        "status": status,
        "doc_kind": "summary",
        "summary_level": "document",
        "summary_title": f"{code_ru or source_short} summary",
        "summary_source": source_short,
        "summary_source_count": str(len(source_chunks)),
        "summary_article_count": str(len(article_counter)),
        "clause_level": "summary",
        "article_number": "",
        "corpus_version": corpus_version,
        "contextual_prefix": "Context: Summary index entry for legal document retrieval.",
    }
    meta["page_count"] = str(len(source_chunks))
    meta["summary_articles"] = ", ".join(top_articles[:10])
    return Document(page_content=summary_text.strip(), metadata=meta)


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
        return [chunk["text"] for chunk in self.split_with_metadata(text)]

    def split_with_metadata(self, text: str) -> list[dict[str, str]]:
        """Return hierarchy chunks with explicit clause/subclause metadata."""
        matches = list(ARTICLE_RE.finditer(text))
        if not matches:
            return [
                {
                    "text": c,
                    "clause_level": "article",
                    "clause_number": "",
                    "subclause_number": "",
                    "path": "",
                }
                for c in _split_preamble_by_hierarchy(text)
                if c and len(c.strip()) >= MIN_CHUNK_LEN
            ]

        chunks: list[dict[str, str]] = []
        prev_end = 0
        for idx, match in enumerate(matches):
            start = match.start()
            if start > prev_end:
                preamble = text[prev_end:start].strip()
                if preamble:
                    for c in _split_preamble_by_hierarchy(preamble):
                        if c and len(c.strip()) >= MIN_CHUNK_LEN:
                            chunks.append(
                                {
                                    "text": c,
                                    "clause_level": "article",
                                    "clause_number": "",
                                    "subclause_number": "",
                                    "path": "",
                                }
                            )
            next_start = (
                matches[idx + 1].start() if idx + 1 < len(matches) else len(text)
            )
            article_text = text[start:next_start].strip()
            for c in _split_article_by_hierarchy(article_text):
                if c["text"] and len(c["text"].strip()) >= MIN_CHUNK_LEN:
                    chunks.append(c)
            prev_end = next_start

        return chunks

    def create_documents(
        self,
        texts: list[str],
        metadatas: list[dict] = None,
        *,
        corpus_version: str = "",
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
            revision_date = _extract_revision_date(text) or SOURCE_REVISION_DATE_FALLBACKS.get(
                source_short,
                "",
            )

            # Track chapter context as we scan the raw text
            current_chapter_number: str = ""
            current_chapter_title: str = ""

            for chunk_info in self.split_with_metadata(text):
                chunk = chunk_info["text"]
                if not chunk or len(chunk.strip()) < MIN_CHUNK_LEN:
                    continue
                if _is_bad_chunk_text(chunk):
                    continue

                # Update chapter context from lines at the start of this chunk
                for line in chunk.splitlines()[:5]:
                    chapter_hit = _detect_chapter(line)
                    if chapter_hit:
                        current_chapter_number, current_chapter_title = chapter_hit
                        break

                art_num = get_article_number(chunk)
                clause_level = chunk_info.get("clause_level") or _detect_clause_level(chunk)
                article_title = get_article_title(chunk)

                # Mandatory lineage metadata for Pinecone hard filtering (code_ru, revision_date, article_number)
                meta: dict = {
                    "source": source_short,
                    "code_ru": code_ru,
                    "code_kz": code_kz,
                    "article_number": str(art_num) if art_num is not None else "",
                    "revision_date": revision_date,
                    "clause_level": clause_level,
                    "jurisdiction": "RK",
                    "document_type": _infer_document_type(base_meta | {"code_ru": code_ru}),
                    "status": _infer_status(chunk, base_meta),
                    "doc_kind": "chunk",
                }
                if chunk_info.get("clause_number"):
                    meta["clause_number"] = chunk_info["clause_number"]
                if chunk_info.get("subclause_number"):
                    meta["subclause_number"] = chunk_info["subclause_number"]
                if chunk_info.get("path"):
                    meta["path"] = chunk_info["path"]
                parent_article_text = chunk_info.get("parent_article_text", "")
                if parent_article_text:
                    meta["parent_article_text"] = parent_article_text[:2000]
                if current_chapter_title:
                    meta["chapter_title"] = current_chapter_title[:150]
                if current_chapter_number:
                    meta["chapter_number"] = current_chapter_number[:20]
                if article_title:
                    meta["article_title"] = article_title
                meta["contextual_prefix"] = _build_contextual_prefix(meta, chunk)
                if corpus_version:
                    meta["corpus_version"] = corpus_version

                documents.append(
                    Document(
                        page_content=_build_indexable_text(chunk, meta),
                        metadata=meta,
                    )
                )

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

raw_docs = sorted_documents_by_source(raw_docs)
CORPUS_VERSION = compute_corpus_version(raw_docs)
article_splitter = ArticleTextSplitter()
chunks = article_splitter.create_documents(
    [doc.page_content for doc in raw_docs],
    [doc.metadata for doc in raw_docs],
    corpus_version=CORPUS_VERSION,
)

summary_chunks: list[Document] = []
if getattr(config, "ENABLE_SUMMARY_INDEX", True):
    chunks_by_source: dict[str, list[Document]] = {}
    source_meta_by_source: dict[str, dict] = {}
    for raw_doc in raw_docs:
        source_short = Path(raw_doc.metadata.get("source", "")).name
        source_meta_by_source[source_short] = dict(raw_doc.metadata or {})
    for chunk in chunks:
        source_short = str(chunk.metadata.get("source") or "").strip()
        if not source_short:
            continue
        chunks_by_source.setdefault(source_short, []).append(chunk)

    for source_short, source_docs in chunks_by_source.items():
        if not source_docs:
            continue
        base_meta = source_meta_by_source.get(source_short, {}).copy()
        summary_chunks.append(
            _build_summary_document(
                source_short,
                base_meta,
                source_docs,
                corpus_version=CORPUS_VERSION,
            )
        )

print(f"Всего документов: {len(raw_docs)}")
print(f"Всего чанков (статей): {len(chunks)}")
print(f"Всего summary-чанков: {len(summary_chunks)}")
print(f"Версия корпуса: {CORPUS_VERSION}")

# Debug: show hierarchy metadata coverage
chapter_chunks = [c for c in chunks if c.metadata.get("chapter_title")]
dated_chunks = [c for c in chunks if c.metadata.get("revision_date")]
clause_chunks = [c for c in chunks if c.metadata.get("clause_level") == "clause"]
subclause_chunks = [c for c in chunks if c.metadata.get("clause_level") == "subclause"]
print(f"  └─ С chapter_title: {len(chapter_chunks)}")
print(f"  └─ С revision_date: {len(dated_chunks)}")
print(f"  └─ clause-level чанков: {len(clause_chunks)}")
print(f"  └─ subclause-level чанков: {len(subclause_chunks)}")

if chunks:
    print("\nПример чанка:")
    print(chunks[0].page_content[:300], "...")
    print("Метаданные:", chunks[0].metadata)

if summary_chunks:
    print("\nПример summary-чанка:")
    print(summary_chunks[0].page_content[:300], "...")
    print("Метаданные:", summary_chunks[0].metadata)

if summary_chunks:
    with open(config.SUMMARY_CHUNKS_PICKLE_PATH, "wb") as f:
        pickle.dump(summary_chunks, f)
    print(f"Summary pickle сохранён: {config.SUMMARY_CHUNKS_PICKLE_PATH}")
