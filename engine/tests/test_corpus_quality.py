from langchain_core.documents import Document

from engine.processing.corpus_quality import (
    compute_corpus_version,
    sorted_documents_by_source,
    validate_corpus_documents,
)


def test_corpus_version_is_deterministic():
    docs = [
        Document(page_content="beta", metadata={"source": "b.txt"}),
        Document(page_content="alpha", metadata={"source": "a.txt"}),
    ]

    assert compute_corpus_version(docs) == compute_corpus_version(list(reversed(docs)))


def test_validate_corpus_documents_detects_missing_metadata():
    raw_docs = [Document(page_content="raw", metadata={"source": "a.txt"})]
    chunks = [
        Document(
            page_content="chunk",
            metadata={
                "source": "a.txt",
                "code_ru": "УК",
                "article_number": "1",
                "revision_date": "2020-01-01",
                "clause_level": "article",
                "corpus_version": "abc123",
            },
        )
    ]

    report = validate_corpus_documents(raw_docs, chunks, corpus_version="abc123")
    assert report.errors == []
    assert report.corpus_version == "abc123"


def test_sorted_documents_by_source_is_stable():
    docs = [
        Document(page_content="beta", metadata={"source": "b.txt"}),
        Document(page_content="alpha", metadata={"source": "a.txt"}),
    ]

    sorted_docs = sorted_documents_by_source(docs)
    assert [doc.metadata["source"] for doc in sorted_docs] == ["a.txt", "b.txt"]
