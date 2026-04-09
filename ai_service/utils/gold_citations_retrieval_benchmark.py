import argparse
import json
import re
import time
from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Retrieval-only benchmark against gold citations from XLSX."
    )
    parser.add_argument(
        "--xlsx",
        default="tests/benchmarks/642_questions_with_citations.xlsx",
        help="Path to XLSX with query/gold_citations columns.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=642,
        help="How many rows to evaluate.",
    )
    parser.add_argument(
        "--start-from",
        type=int,
        default=0,
        help="Zero-based row offset.",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=10,
        help="How many retrieved docs to evaluate.",
    )
    parser.add_argument(
        "--output",
        default="benchmark_results/gold_citations_retrieval_benchmark.json",
        help="Output JSON path.",
    )
    return parser.parse_args()


def _normalize_gold(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, float) and pd.isna(value):
        return []
    text = str(value).strip()
    if not text:
        return []
    parts = [p.strip() for p in text.split(";") if p.strip()]
    return parts if parts else [text]


def _normalize_article(value: str) -> str:
    text = str(value or "").strip().lower().replace("ё", "е")
    if not text:
        return ""
    match = re.search(r"(?:ст\.?|статья|бап)\s*(\d{1,4}(?:-\d+)?)", text)
    if match:
        return match.group(1)
    match = re.search(r"(\d{1,4}(?:-\d+)?)", text)
    return match.group(1) if match else ""


def _normalize_code(value: str) -> str:
    text = str(value or "").strip().lower().replace("ё", "е")
    if not text:
        return ""
    text = text.replace("«", '"').replace("»", '"')
    text = re.sub(r"\s+", " ", text)

    alias_groups = [
        (
            "гражданский кодекс рк",
            (
                "гк рк",
                "гк",
                "гражданский кодекс",
                "гражданский кодекс рк (общая часть)",
                "гражданский кодекс рк (особенная часть)",
            ),
        ),
        ("уголовный кодекс рк", ("ук рк", "ук", "уголовный кодекс")),
        (
            "гражданский процессуальный кодекс рк",
            ("гпк рк", "гпк", "гражданский процессуальный кодекс"),
        ),
        (
            "уголовно-процессуальный кодекс рк",
            ("упк рк", "упк", "уголовно-процессуальный кодекс"),
        ),
        (
            "кодекс об административных правонарушениях рк",
            ("коап рк", "коап", "кодекс об административных правонарушениях"),
        ),
        (
            "кодекс об административных процедурах рк",
            ("аппк рк", "аппк", "кодекс об административных процедурах"),
        ),
        (
            "закон о защите прав потребителей рк",
            ("закон о защите прав потребителей", "о защите прав потребителей", "зпп"),
        ),
        (
            "закон об адвокатской деятельности и юридической помощи",
            ("об адвокатской деятельности и юридической помощи",),
        ),
        (
            "закон о валютном регулировании и валютном контроле",
            ("о валютном регулировании и валютном контроле",),
        ),
        (
            "закон о товариществах с ограниченной и дополнительной ответственностью рк",
            ("тоо", "товариществах с ограниченной и дополнительной ответственностью"),
        ),
    ]
    for canonical, aliases in alias_groups:
        if canonical in text or any(alias in text for alias in aliases):
            return canonical

    text = re.sub(r'\bреспублики казахстан\b', "рк", text)
    text = re.sub(r"[\"']", "", text)
    return text.strip(" ,.")


def _gold_to_pair(value: str) -> tuple[str, str]:
    raw = str(value or "").strip()
    if not raw:
        return "", ""
    article = _normalize_article(raw)
    lower_raw = raw.lower()
    code = ""
    if "закон" in lower_raw:
        code = raw[lower_raw.find("закон") :]
    elif "кодекс" in lower_raw or "гк" in lower_raw or "ук" in lower_raw or "гпк" in lower_raw:
        code = raw
    return article, _normalize_code(code)


def _pair_to_str(article: str, code: str) -> str:
    return f"{_normalize_article(article)}::{_normalize_code(code)}"


def _compute_metrics(gold_pairs: list[tuple[str, str]], pred_pairs: list[tuple[str, str]]) -> dict[str, float]:
    gold_set = {pair for pair in gold_pairs if pair[0]}
    pred_set = {pair for pair in pred_pairs if pair[0]}
    gold_articles = {article for article, _ in gold_set}
    pred_articles = {article for article, _ in pred_set}

    strict_hit = 1.0 if gold_set & pred_set else 0.0
    soft_hit = 1.0 if gold_articles & pred_articles else 0.0
    strict_precision = len(gold_set & pred_set) / len(pred_set) if pred_set else 0.0
    strict_recall = len(gold_set & pred_set) / len(gold_set) if gold_set else 0.0
    soft_precision = len(gold_articles & pred_articles) / len(pred_articles) if pred_articles else 0.0
    soft_recall = len(gold_articles & pred_articles) / len(gold_articles) if gold_articles else 0.0

    strict_mrr = 0.0
    soft_mrr = 0.0
    for rank, pair in enumerate(pred_pairs, start=1):
        if not strict_mrr and pair in gold_set:
            strict_mrr = 1.0 / rank
        if not soft_mrr and pair[0] in gold_articles:
            soft_mrr = 1.0 / rank
        if strict_mrr and soft_mrr:
            break

    return {
        "strict_hit": strict_hit,
        "soft_hit": soft_hit,
        "strict_precision": strict_precision,
        "strict_recall": strict_recall,
        "soft_precision": soft_precision,
        "soft_recall": soft_recall,
        "strict_mrr": strict_mrr,
        "soft_mrr": soft_mrr,
    }


def main() -> None:
    args = _parse_args()
    from ai_service.retrieval.rag_chain import get_retriever_for_coverage

    xlsx_path = Path(args.xlsx)
    if not xlsx_path.exists():
        raise FileNotFoundError(f"Benchmark XLSX not found: {xlsx_path}")

    retriever = get_retriever_for_coverage(args.top_k)
    df = pd.read_excel(xlsx_path)
    df = df.iloc[args.start_from : args.start_from + args.limit].copy()

    results: list[dict[str, Any]] = []
    for idx, row in df.iterrows():
        qid = str(row.get("query_id") or idx)
        query = str(row.get("query") or "").strip()
        gold_raw = _normalize_gold(row.get("gold_citations"))
        gold_pairs = [_gold_to_pair(item) for item in gold_raw]

        started = time.perf_counter()
        error = ""
        try:
            docs = retriever.invoke(query)
        except Exception as exc:
            docs = []
            error = str(exc)
        elapsed_sec = round(time.perf_counter() - started, 3)

        pred_pairs: list[tuple[str, str]] = []
        for doc in (docs or [])[: args.top_k]:
            meta = getattr(doc, "metadata", {}) or {}
            pair = (
                _normalize_article(meta.get("article_number", "")),
                _normalize_code(meta.get("code_ru", "")),
            )
            if pair[0] and pair not in pred_pairs:
                pred_pairs.append(pair)

        metrics = _compute_metrics(gold_pairs, pred_pairs)
        results.append(
            {
                "row_index": int(idx),
                "query_id": qid,
                "query": query,
                "gold_citations": gold_raw,
                "retrieved_topk": [_pair_to_str(article, code) for article, code in pred_pairs],
                "metrics": metrics,
                "elapsed_sec": elapsed_sec,
                "error": error,
            }
        )

        print(
            f"[{len(results)}/{len(df)}] {qid} "
            f"strict_hit={metrics['strict_hit']:.0f} "
            f"soft_hit={metrics['soft_hit']:.0f} "
            f"strict_mrr={metrics['strict_mrr']:.3f} "
            f"elapsed={elapsed_sec}s"
        )

    def _avg(metric_name: str) -> float:
        values = [float(item["metrics"][metric_name]) for item in results]
        return sum(values) / len(values) if values else 0.0

    payload = {
        "xlsx": str(xlsx_path),
        "evaluated_questions": len(results),
        "top_k": args.top_k,
        "generated_at": datetime.utcnow().isoformat() + "Z",
        "summary": {
            "strict_hit": _avg("strict_hit"),
            "soft_hit": _avg("soft_hit"),
            "strict_precision": _avg("strict_precision"),
            "strict_recall": _avg("strict_recall"),
            "soft_precision": _avg("soft_precision"),
            "soft_recall": _avg("soft_recall"),
            "strict_mrr": _avg("strict_mrr"),
            "soft_mrr": _avg("soft_mrr"),
            "avg_elapsed_sec": sum(item["elapsed_sec"] for item in results) / len(results)
            if results
            else 0.0,
        },
        "results": results,
    }

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)

    print(f"\nSaved: {output_path}")
    print(json.dumps(payload["summary"], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
