from __future__ import annotations

import argparse
import asyncio
import json
import logging
from pathlib import Path
from typing import Any

from ai_service.core import config

logger = logging.getLogger("ai_service.single_question_cli")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run one LegalRAG question and print the answer plus top retrieved docs."
    )
    parser.add_argument(
        "--query",
        required=True,
        help="User question to run through the pipeline.",
    )
    parser.add_argument(
        "--mode",
        choices=("offline", "live", "agentic"),
        default="offline",
        help="offline: deterministic extractive answer from local corpus; live: full QA stack; agentic: CRAG + Self-RAG + CoVe.",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=3,
        help="How many top source documents to show.",
    )
    parser.add_argument(
        "--intent",
        default="",
        help="Optional explicit intent label to override routing.",
    )
    parser.add_argument(
        "--expected-article",
        action="append",
        default=[],
        help="Optional expected article number for a quick hit/miss check. Repeatable.",
    )
    parser.add_argument(
        "--expected-code",
        default="",
        help="Optional expected code name for the quick check.",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Print structured JSON instead of a human-readable report.",
    )
    parser.add_argument(
        "--output",
        default="single_question_answer.json",
        help="Path to save the full answer payload as JSON.",
    )
    return parser.parse_args()


def _normalize_article(value: Any) -> str:
    text = str(value or "").strip()
    if not text:
        return ""
    digits = "".join(ch for ch in text if ch.isdigit() or ch == "-")
    return digits or text.lower()


def _format_doc(doc: Any) -> dict[str, Any]:
    meta = getattr(doc, "metadata", {}) or {}
    return {
        "article_number": str(meta.get("article_number") or "").strip(),
        "code_ru": str(meta.get("code_ru") or "").strip(),
        "source": str(meta.get("source") or "").strip(),
        "snippet": str(getattr(doc, "page_content", "") or "").strip()[:500],
    }


def _evaluate_top_docs(
    docs: list[Any],
    expected_articles: list[str],
    expected_code: str,
) -> dict[str, Any]:
    expected = {_normalize_article(article) for article in expected_articles if _normalize_article(article)}
    expected_code_norm = str(expected_code or "").strip().lower()
    top_docs = [_format_doc(doc) for doc in docs]
    found_articles = {
        _normalize_article(doc["article_number"])
        for doc in top_docs
        if _normalize_article(doc["article_number"])
    }
    found_codes = {doc["code_ru"].strip().lower() for doc in top_docs if doc["code_ru"].strip()}

    article_hit = bool(expected & found_articles) if expected else None
    code_hit = bool(expected_code_norm in found_codes) if expected_code_norm else None
    verdict = "n/a"
    if article_hit is True or code_hit is True:
        verdict = "pass"
    elif article_hit is False or code_hit is False:
        verdict = "fail"

    return {
        "expected_articles": sorted(expected),
        "expected_code": expected_code_norm,
        "article_hit": article_hit,
        "code_hit": code_hit,
        "verdict": verdict,
    }


def main() -> None:
    args = _parse_args()
    intent = args.intent.strip() or None

    if args.mode == "agentic":
        from ai_service.retrieval import agentic_workflow

        try:
            config.LEGAL_RAG_OFFLINE_QA = False
            config.USE_RERANKER = True
            result = asyncio.run(
                agentic_workflow.invoke_agentic_qa(args.query, history=None, trace_id=None)
            )
            payload_result = {
                "result": result.get("result", ""),
                "source_documents": result.get("source_documents", []) or [],
                "retrieval_method": "agentic",
                "trace_report": result.get("trace_report", {}),
            }
        except Exception as exc:
            logger.warning("Agentic mode failed, falling back to offline QA: %s", exc)
            config.LEGAL_RAG_OFFLINE_QA = True
            config.USE_RERANKER = False
            from ai_service.retrieval import rag_chain

            rag_chain._invoke_qa_impl.cache_clear()
            fallback = rag_chain.invoke_qa(args.query, intent=intent)
            payload_result = {
                **fallback,
                "retrieval_method": "agentic_fallback_offline",
                "agentic_error": str(exc),
            }
    else:
        config.LEGAL_RAG_OFFLINE_QA = args.mode == "offline"
        config.USE_RERANKER = False if args.mode == "offline" else config.USE_RERANKER

        from ai_service.retrieval import rag_chain

        rag_chain._invoke_qa_impl.cache_clear()
        result = rag_chain.invoke_qa(args.query, intent=intent)
        payload_result = result

    docs = payload_result.get("source_documents", []) or []
    docs = docs[: args.top_k]
    quality = _evaluate_top_docs(docs, args.expected_article, args.expected_code)

    payload = {
        "query": args.query,
        "mode": args.mode,
        "retrieval_method": payload_result.get("retrieval_method", ""),
        "answer": payload_result.get("result", ""),
        "top_documents": [_format_doc(doc) for doc in docs],
        "quality": quality,
    }
    if "trace_report" in payload_result:
        payload["trace_report"] = payload_result["trace_report"]

    output_path = Path(args.output).expanduser()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    if args.json:
        print(json.dumps(payload, ensure_ascii=False, indent=2))
        return

    print(f"Mode: {args.mode}")
    print(f"Retrieval: {payload['retrieval_method']}")
    print()
    print(payload["answer"])
    print()
    print("Top docs:")
    for doc in payload["top_documents"]:
        code = doc["code_ru"] or "Unknown"
        article = doc["article_number"] or "N/A"
        print(f"- [{code} | ст. {article}] {doc['snippet']}")
    if quality["verdict"] != "n/a":
        print()
        print(
            "Quality: "
            f"{quality['verdict']} "
            f"(article_hit={quality['article_hit']}, code_hit={quality['code_hit']})"
        )
    print()
    print(f"Saved JSON: {output_path}")


if __name__ == "__main__":
    main()
