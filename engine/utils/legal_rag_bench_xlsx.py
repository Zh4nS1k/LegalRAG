"""
legal_rag_bench_xlsx.py — метрики для LegalRAG по XLSX бенчмарку с gold-citations.

Поддерживает:
- Context Precision/Recall (по top-k retrieval vs gold_citations)
- Citation Accuracy (по цитатам в ответе vs gold_citations)
- Faithfulness (LLM-judge: ответ vs retrieved context)
- Answer Relevance (LLM-judge: ответ vs вопрос)
- Legal Reasoning (LLM-judge: применение нормы к фактам)

Примеры:
  python -m engine.utils.legal_rag_bench_xlsx --mode offline
  python -m engine.utils.legal_rag_bench_xlsx --mode retrieval --top-k 10
  python -m engine.utils.legal_rag_bench_xlsx --mode full --top-k 10
"""

from __future__ import annotations

import argparse
import json
import os
import time
from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd

from engine.core import config
from engine.utils import citations as cit


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="LegalRAG metrics benchmark from XLSX (queries + gold_citations + optional answers)."
    )
    parser.add_argument(
        "--xlsx",
        default="tests/benchmarks/Полный бенчмарк-3.reviewed8.xlsx",
        help="Path to XLSX with query_id/query/gold_citations and optional answer column.",
    )
    parser.add_argument(
        "--mode",
        choices=("offline", "retrieval", "full"),
        default="offline",
        help="offline: only citations vs gold from XLSX answers; retrieval: add retrieval vs gold; full: add LLM judges.",
    )
    parser.add_argument("--limit", type=int, default=0, help="Limit rows (0 = all).")
    parser.add_argument(
        "--start-from", type=int, default=0, help="Zero-based start offset."
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=10,
        help="How many retrieved docs to evaluate for context precision/faithfulness.",
    )
    parser.add_argument(
        "--output",
        default="benchmark_results/legalrag_metrics_xlsx.json",
        help="Output JSON path.",
    )
    return parser.parse_args()


def _pick_answer_column(columns: list[str]) -> str | None:
    candidates = [
        "Ответ нашего RAG",
        "answer",
        "rag_answer",
        "model_answer",
    ]
    for name in candidates:
        if name in columns:
            return name
    return None


def _safe_float(text: str) -> float | None:
    if text is None:
        return None
    try:
        return float(str(text).strip().replace(",", "."))
    except Exception:
        return None


def _judge_score(llm: Any, prompt: str) -> float | None:
    try:
        response = llm.invoke(prompt)
        content = response.content if hasattr(response, "content") else str(response)
        score = _safe_float(content)
        if score is None:
            return None
        return max(0.0, min(1.0, float(score)))
    except Exception:
        return None


def _judge_faithfulness(llm: Any, *, answer: str, context: str) -> float | None:
    prompt = f"""
Оцени, насколько ответ основан ТОЛЬКО на предоставленном контексте (без выдумок).
Ответь только числом от 0.0 до 1.0, где:
1.0 — все утверждения в ответе имеют прямую поддержку в контексте
0.0 — ответ содержит существенные галлюцинации/вымышленные нормы

Контекст:
{(context or '')[:8000]}

Ответ:
{answer}

Оценка (только число):
""".strip()
    return _judge_score(llm, prompt)


def _judge_answer_relevance(llm: Any, *, query: str, answer: str) -> float | None:
    prompt = f"""
Оцени, насколько ответ действительно отвечает на вопрос пользователя и помогает решить задачу.
Ответь только числом от 0.0 до 1.0, где:
1.0 — по делу, закрывает вопрос, без лишней воды
0.0 — не по теме/уходит от ответа/общие слова

Вопрос:
{query}

Ответ:
{answer}

Оценка (только число):
""".strip()
    return _judge_score(llm, prompt)


def _judge_legal_reasoning(llm: Any, *, query: str, answer: str) -> float | None:
    prompt = f"""
Оцени "юридическую логику" ответа: применяет ли он нормы к фактам, выделяет условия/состав, делает вывод.
Ответь только числом от 0.0 до 1.0, где:
1.0 — есть корректная квалификация/условия применения нормы и вывод
0.0 — просто цитаты без применения или рассуждения не по праву

Вопрос (факты/ситуация):
{query}

Ответ:
{answer}

Оценка (только число):
""".strip()
    return _judge_score(llm, prompt)


def _avg(values: list[float | None]) -> float | None:
    nums = [v for v in values if v is not None]
    return (sum(nums) / len(nums)) if nums else None


def main() -> None:
    args = _parse_args()
    xlsx_path = Path(args.xlsx)
    if not xlsx_path.exists():
        raise FileNotFoundError(f"XLSX not found: {xlsx_path}")

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    df = pd.read_excel(xlsx_path)
    df = df.iloc[args.start_from :].copy()
    if args.limit and args.limit > 0:
        df = df.iloc[: args.limit].copy()

    answer_col = _pick_answer_column(list(df.columns))
    if args.mode == "offline" and not answer_col:
        raise ValueError(
            "offline mode requires an answer column (e.g. 'Ответ нашего RAG')."
        )

    retriever = None
    llm = None
    if args.mode in ("retrieval", "full"):
        from engine.retrieval.rag_chain import get_retriever_for_coverage

        retriever = get_retriever_for_coverage(int(args.top_k))
    if args.mode == "full":
        from engine.retrieval.rag_chain import get_llm

        llm = get_llm()

    results: list[dict[str, Any]] = []
    for row_index, row in df.iterrows():
        qid = str(row.get("query_id") or row_index)
        query = str(row.get("query") or "").strip()
        gold_raw = cit.normalize_gold(row.get("gold_citations"))
        gold_pairs = [cit.gold_to_pair(item) for item in gold_raw]

        answer = str(row.get(answer_col) or "").strip() if answer_col else ""
        answer_pairs = cit.extract_citation_pairs_from_text(answer) if answer else []
        citation_vs_gold = cit.compute_pair_metrics(gold_pairs, answer_pairs)

        row_payload: dict[str, Any] = {
            "row_index": int(row_index),
            "query_id": qid,
            "query": query,
            "gold_citations": gold_raw,
            "answer": answer,
            "answer_citations": [cit.pair_to_str(a, c) for a, c in answer_pairs],
            "citation_vs_gold": citation_vs_gold,
        }

        if retriever is not None:
            started = time.perf_counter()
            error = ""
            try:
                docs = retriever.invoke(query)
            except Exception as exc:
                docs = []
                error = str(exc)
            elapsed = round(time.perf_counter() - started, 3)

            retrieved_pairs = cit.extract_pairs_from_docs((docs or [])[: int(args.top_k)])
            context_vs_gold = cit.compute_pair_metrics(gold_pairs, retrieved_pairs)
            row_payload.update(
                {
                    "retrieval_topk": int(args.top_k),
                    "retrieved_citations": [
                        cit.pair_to_str(a, c) for a, c in retrieved_pairs
                    ],
                    "context_vs_gold": context_vs_gold,
                    "latency_retrieval_sec": elapsed,
                    "retrieval_error": error,
                }
            )

            if llm is not None and answer:
                judge_context = "\n\n".join(
                    getattr(doc, "page_content", "") for doc in (docs or [])[: int(args.top_k)]
                )
                j0 = time.perf_counter()
                faithfulness = _judge_faithfulness(
                    llm, answer=answer, context=judge_context
                )
                answer_relevance = _judge_answer_relevance(
                    llm, query=query, answer=answer
                )
                legal_reasoning = _judge_legal_reasoning(
                    llm, query=query, answer=answer
                )
                row_payload.update(
                    {
                        "faithfulness": faithfulness,
                        "answer_relevance": answer_relevance,
                        "legal_reasoning": legal_reasoning,
                        "latency_judge_sec": round(time.perf_counter() - j0, 3),
                    }
                )

        results.append(row_payload)
        if len(results) % 25 == 0:
            print(f"[{len(results)}/{len(df)}] processed...")

    summary = {
        "timestamp": datetime.now().strftime("%Y-%m-%d_%H-%M"),
        "xlsx": str(xlsx_path),
        "mode": args.mode,
        "rows": int(len(results)),
        "top_k": int(args.top_k),
        "avg_citation_strict_precision": _avg(
            [r["citation_vs_gold"]["strict_precision"] for r in results]
        ),
        "avg_citation_strict_recall": _avg(
            [r["citation_vs_gold"]["strict_recall"] for r in results]
        ),
        "avg_citation_strict_mrr": _avg(
            [r["citation_vs_gold"]["strict_mrr"] for r in results]
        ),
    }
    if args.mode in ("retrieval", "full"):
        summary.update(
            {
                "avg_context_strict_precision": _avg(
                    [
                        (r.get("context_vs_gold") or {}).get("strict_precision")
                        for r in results
                    ]
                ),
                "avg_context_strict_recall": _avg(
                    [
                        (r.get("context_vs_gold") or {}).get("strict_recall")
                        for r in results
                    ]
                ),
                "avg_context_strict_mrr": _avg(
                    [(r.get("context_vs_gold") or {}).get("strict_mrr") for r in results]
                ),
                "avg_latency_retrieval_sec": _avg(
                    [r.get("latency_retrieval_sec") for r in results]
                ),
            }
        )
    if args.mode == "full":
        summary.update(
            {
                "avg_faithfulness": _avg([r.get("faithfulness") for r in results]),
                "avg_answer_relevance": _avg(
                    [r.get("answer_relevance") for r in results]
                ),
                "avg_legal_reasoning": _avg(
                    [r.get("legal_reasoning") for r in results]
                ),
                "avg_latency_judge_sec": _avg(
                    [r.get("latency_judge_sec") for r in results]
                ),
            }
        )

    payload = {
        "summary": summary,
        "results": results,
        "config": {
            "VECTOR_WEIGHT": getattr(config, "VECTOR_WEIGHT", None),
            "BM25_WEIGHT": getattr(config, "BM25_WEIGHT", None),
            "USE_RERANKER": getattr(config, "USE_RERANKER", None),
            "LEGAL_RAG_LLM_BACKEND": os.environ.get("LEGAL_RAG_LLM_BACKEND", ""),
            "LEGAL_RAG_LLM": os.environ.get("LEGAL_RAG_LLM", ""),
        },
    }

    out_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

