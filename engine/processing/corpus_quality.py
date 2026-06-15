from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable

from langchain_core.documents import Document


MANDATORY_METADATA_KEYS = (
    "source",
    "code_ru",
    "article_number",
    "revision_date",
    "clause_level",
    "corpus_version",
)


@dataclass(frozen=True)
class CorpusQualityIssue:
    level: str
    code: str
    message: str
    source: str = ""
    article_number: str = ""
    path: str = ""


@dataclass
class CorpusQualityReport:
    corpus_version: str
    raw_documents: int
    chunks: int
    issues: list[CorpusQualityIssue] = field(default_factory=list)

    @property
    def errors(self) -> list[CorpusQualityIssue]:
        return [issue for issue in self.issues if issue.level == "error"]

    def to_dict(self) -> dict[str, object]:
        return {
            "corpus_version": self.corpus_version,
            "raw_documents": self.raw_documents,
            "chunks": self.chunks,
            "errors": len(self.errors),
            "warnings": len(self.issues) - len(self.errors),
            "issues": [
                {
                    "level": issue.level,
                    "code": issue.code,
                    "message": issue.message,
                    "source": issue.source,
                    "article_number": issue.article_number,
                    "path": issue.path,
                }
                for issue in self.issues
            ],
        }

    def assert_clean(self) -> None:
        if self.errors:
            first = self.errors[0]
            raise RuntimeError(
                f"Corpus quality gate failed: {first.code} - {first.message}"
            )


def _normalized_source(value: str) -> str:
    return Path(str(value or "").strip()).name.lower()


def compute_corpus_version(raw_docs: Iterable[Document]) -> str:
    digest = hashlib.sha256()
    for doc in sorted(
        raw_docs,
        key=lambda item: _normalized_source(item.metadata.get("source", "")),
    ):
        source = _normalized_source(doc.metadata.get("source", ""))
        content = (doc.page_content or "").strip().replace("\r\n", "\n")
        digest.update(source.encode("utf-8"))
        digest.update(b"\0")
        digest.update(hashlib.sha256(content.encode("utf-8")).hexdigest().encode("utf-8"))
        digest.update(b"\0")
    return digest.hexdigest()[:16]


def _chunk_fingerprint(doc: Document) -> str:
    meta = doc.metadata or {}
    payload = {
        "source": _normalized_source(meta.get("source", "")),
        "code_ru": str(meta.get("code_ru", "")).strip(),
        "article_number": str(meta.get("article_number", "")).strip(),
        "clause_level": str(meta.get("clause_level", "")).strip(),
        "path": str(meta.get("path", "")).strip(),
        "content": (doc.page_content or "").strip().replace("\r\n", "\n"),
    }
    raw = json.dumps(payload, ensure_ascii=False, sort_keys=True)
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def validate_corpus_documents(
    raw_docs: list[Document],
    chunks: list[Document],
    *,
    corpus_version: str,
) -> CorpusQualityReport:
    issues: list[CorpusQualityIssue] = []
    seen_fingerprints: set[str] = set()

    if not corpus_version:
        issues.append(
            CorpusQualityIssue(
                level="error",
                code="MISSING_CORPUS_VERSION",
                message="Corpus version is empty.",
            )
        )

    if not raw_docs:
        issues.append(
            CorpusQualityIssue(
                level="error",
                code="NO_RAW_DOCS",
                message="No raw documents were loaded.",
            )
        )

    for doc in chunks:
        meta = doc.metadata or {}
        source = _normalized_source(meta.get("source", ""))
        article_number = str(meta.get("article_number", "")).strip()
        path = str(meta.get("path", "")).strip()

        for key in MANDATORY_METADATA_KEYS:
            if not str(meta.get(key, "")).strip():
                issues.append(
                    CorpusQualityIssue(
                        level="error",
                        code="MISSING_METADATA",
                        message=f"Missing mandatory metadata field: {key}",
                        source=source,
                        article_number=article_number,
                        path=path,
                    )
                )

        if not (doc.page_content or "").strip():
            issues.append(
                CorpusQualityIssue(
                    level="error",
                    code="EMPTY_CHUNK",
                    message="Chunk text is empty.",
                    source=source,
                    article_number=article_number,
                    path=path,
                )
            )

        fingerprint = _chunk_fingerprint(doc)
        if fingerprint in seen_fingerprints:
            issues.append(
                CorpusQualityIssue(
                    level="error",
                    code="DUPLICATE_CHUNK",
                    message="Duplicate chunk fingerprint detected.",
                    source=source,
                    article_number=article_number,
                    path=path,
                )
            )
        else:
            seen_fingerprints.add(fingerprint)

        content_head = (doc.page_content or "").lower()
        if any(
            marker in content_head
            for marker in (
                "оглавление",
                "мазмұны",
                "вниманию пользователей",
                "содержание",
                "примечание рцпи",
            )
        ):
            issues.append(
                CorpusQualityIssue(
                    level="warning",
                    code="POSSIBLE_NOISE",
                    message="Chunk may contain front-matter or editorial noise.",
                    source=source,
                    article_number=article_number,
                    path=path,
                )
            )

    return CorpusQualityReport(
        corpus_version=corpus_version,
        raw_documents=len(raw_docs),
        chunks=len(chunks),
        issues=issues,
    )


def sorted_documents_by_source(raw_docs: list[Document]) -> list[Document]:
    return sorted(
        raw_docs,
        key=lambda item: (
            _normalized_source(item.metadata.get("source", "")),
            (item.page_content or "")[:80],
        ),
    )
