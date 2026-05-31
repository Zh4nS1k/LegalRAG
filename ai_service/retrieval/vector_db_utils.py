"""Helpers for preparing Pinecone metadata."""

from __future__ import annotations


def clean_metadata(meta: dict) -> dict:
    """
    Очистка метаданных для Pinecone: удаляем длинные поля и обрезаем слишком
    большие значения, чтобы не превышать лимит metadata.
    """
    blacklist = [
        "text",
        "content",
        "full_text",
        "raw",
        "body",
        "snippet",
        "notes",
        "chapter_text",
        "page_content",
        "raw_text",
        "chapter_content",
        "article_text",
        "snippets",
        "full_article",
        "raw_content",
        "parent_article_text",
    ]
    allowed = [
        "source",
        "code_ru",
        "code_kz",
        "article_number",
        "chapter_title",
        "chapter_number",
        "clause_level",
        "revision_date",
        "chapter",
        "section",
        "jurisdiction",
        "document_type",
        "status",
        "doc_kind",
        "summary_level",
        "summary_title",
        "summary_source_count",
        "summary_article_count",
        "summary_source",
        "contextual_prefix",
    ]

    clean = {}
    for k, v in meta.items():
        if k in blacklist:
            continue
        if k in allowed:
            v_str = str(v)
            clean[k] = v_str[:200] + "..." if len(v_str) > 200 else v_str
        else:
            v_str = str(v)
            size_bytes = len(v_str.encode("utf-8"))
            if size_bytes < 1000:
                clean[k] = v_str
            else:
                clean[k] = v_str[:200] + "..."
    return clean
