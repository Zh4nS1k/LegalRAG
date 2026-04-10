import argparse
import json
import time
from datetime import datetime, UTC
from pathlib import Path
from typing import Any

from ai_service.core import config


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Benchmark retrieval quality on JSON queries with article-level ground truth."
    )
    parser.add_argument(
        "--queries",
        default="test_queries.json",
        help="Path to JSON array with query and relevant_articles fields.",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=5,
        help="How many retrieved docs to score.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=0,
        help="Optional limit for number of queries to evaluate. 0 means all.",
    )
    parser.add_argument(
        "--output",
        default="benchmark_results/retrieval_quality_benchmark.json",
        help="Where to save current benchmark JSON.",
    )
    parser.add_argument(
        "--compare-to",
        default="",
        help="Optional previous benchmark JSON to compare current metrics against.",
    )
    return parser.parse_args()


def _normalize_article(value: Any) -> str:
    text = str(value or "").strip().lower()
    if not text:
        return ""
    text = text.replace("статья", "").replace("ст.", "").replace("ст", "").replace("бап", "")
    text = text.replace("–", "-").replace("—", "-")
    cleaned = "".join(ch for ch in text if ch.isalnum() or ch == "-")
    return cleaned.strip("-")


def _normalize_code(value: Any) -> str:
    text = str(value or "").strip().lower().replace("ё", "е")
    if not text:
        return ""
    text = " ".join(text.split())
    aliases = (
        ("уголовный кодекс рк", ("ук рк", "уголовный кодекс", "қылмыстық кодекс")),
        ("гражданский кодекс рк", ("гк рк", "гражданский кодекс", "азаматтық кодекс")),
        (
            "кодекс об административных правонарушениях рк",
            ("коап рк", "коап", "әкімшілік құқық бұзушылық туралы кодекс"),
        ),
        ("земельный кодекс рк", ("земельный кодекс", "жер кодексі")),
        ("налоговый кодекс рк", ("налоговый кодекс", "салық кодексі")),
        ("трудовой кодекс рк", ("трудовой кодекс", "еңбек кодексі")),
    )
    for canonical, variants in aliases:
        if canonical in text or any(variant in text for variant in variants):
            return canonical
    return text


def _load_relevant_pairs(item: dict[str, Any]) -> list[tuple[str, str]]:
    raw_pairs = item.get("relevant_pairs") or item.get("ground_truth_pairs") or []
    pairs: list[tuple[str, str]] = []
    for raw in raw_pairs:
        if not isinstance(raw, dict):
            continue
        article = _normalize_article(raw.get("article"))
        code = _normalize_code(raw.get("code"))
        if article:
            pairs.append((article, code))
    return pairs


def _load_queries(path: Path, limit: int) -> list[dict[str, Any]]:
    with open(path, "r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, list):
        raise ValueError(f"{path} must contain a JSON array")

    rows: list[dict[str, Any]] = []
    for idx, item in enumerate(payload, start=1):
        if not isinstance(item, dict):
            continue
        query = str(item.get("query") or "").strip()
        if not query:
            continue
        relevant_articles = [
            _normalize_article(value)
            for value in (
                item.get("relevant_articles")
                or item.get("relevant_article_numbers")
                or item.get("ground_truth_articles")
                or []
            )
            if _normalize_article(value)
        ]
        rows.append(
            {
                "id": item.get("id", f"q_{idx:03d}"),
                "query": query,
                "lang": item.get("lang", ""),
                "description": item.get("description", ""),
                "relevant_articles": relevant_articles,
                "relevant_pairs": _load_relevant_pairs(item),
            }
        )

    if limit > 0:
        return rows[:limit]
    return rows


def _extract_predicted_articles(docs: list[Any], top_k: int) -> list[str]:
    predicted: list[str] = []
    for doc in (docs or [])[:top_k]:
        meta = getattr(doc, "metadata", {}) or {}
        article = _normalize_article(meta.get("article_number"))
        if article and article not in predicted:
            predicted.append(article)
    return predicted


def _extract_predicted_pairs(docs: list[Any], top_k: int) -> list[tuple[str, str]]:
    predicted: list[tuple[str, str]] = []
    for doc in (docs or [])[:top_k]:
        meta = getattr(doc, "metadata", {}) or {}
        article = _normalize_article(meta.get("article_number"))
        code = _normalize_code(meta.get("code_ru"))
        pair = (article, code)
        if article and pair not in predicted:
            predicted.append(pair)
    return predicted


def _compute_metrics(
    relevant_articles: list[str],
    predicted_articles: list[str],
    *,
    relevant_pairs: list[tuple[str, str]] | None = None,
    predicted_pairs: list[tuple[str, str]] | None = None,
) -> dict[str, float]:
    relevant = [_normalize_article(item) for item in relevant_articles if _normalize_article(item)]
    predicted = [_normalize_article(item) for item in predicted_articles if _normalize_article(item)]
    relevant_set = set(relevant)
    predicted_set = set(predicted)
    normalized_relevant_pairs = {
        (_normalize_article(article), _normalize_code(code))
        for article, code in (relevant_pairs or [])
        if _normalize_article(article)
    }
    normalized_predicted_pairs = [
        (_normalize_article(article), _normalize_code(code))
        for article, code in (predicted_pairs or [])
        if _normalize_article(article)
    ]

    if normalized_relevant_pairs:
        strict_hit = 1.0 if set(normalized_predicted_pairs) & normalized_relevant_pairs else 0.0
    else:
        strict_hit = 1.0 if relevant_set & predicted_set else 0.0
    soft_hit = 1.0 if relevant_set & predicted_set else 0.0

    reciprocal_rank = 0.0
    for rank, article in enumerate(predicted, start=1):
        if article in relevant_set:
            reciprocal_rank = 1.0 / rank
            break

    return {
        "strict_hit": strict_hit,
        "soft_hit": soft_hit,
        "mrr": reciprocal_rank,
    }


def _average_metric(results: list[dict[str, Any]], metric_name: str) -> float:
    values = [float(row["metrics"][metric_name]) for row in results]
    return sum(values) / len(values) if values else 0.0


def _build_summary(results: list[dict[str, Any]], top_k: int) -> dict[str, float | int]:
    return {
        "queries_evaluated": len(results),
        "top_k": top_k,
        "strict_hit@k": _average_metric(results, "strict_hit"),
        "soft_hit@k": _average_metric(results, "soft_hit"),
        "mrr": _average_metric(results, "mrr"),
    }


def _build_comparison(current_summary: dict[str, Any], previous_summary: dict[str, Any]) -> dict[str, float]:
    comparison: dict[str, float] = {}
    for current_key, previous_key in (
        ("strict_hit@k", "strict_hit@k"),
        ("soft_hit@k", "soft_hit@k"),
        ("mrr", "mrr"),
    ):
        comparison[f"delta_{current_key}"] = float(current_summary.get(current_key, 0.0)) - float(
            previous_summary.get(previous_key, 0.0)
        )
    return comparison


def main() -> None:
    args = _parse_args()
    queries_path = Path(args.queries)
    if not queries_path.exists():
        raise FileNotFoundError(f"Queries JSON not found: {queries_path}")

    from ai_service.retrieval.rag_chain import get_retriever_for_coverage

    queries = _load_queries(queries_path, args.limit)
    retriever = get_retriever_for_coverage(args.top_k)

    results: list[dict[str, Any]] = []
    for index, item in enumerate(queries, start=1):
        started = time.perf_counter()
        error = ""
        try:
            docs = retriever.invoke(item["query"])
        except Exception as exc:
            docs = []
            error = str(exc)
        elapsed = round(time.perf_counter() - started, 3)

        predicted_articles = _extract_predicted_articles(docs, args.top_k)
        predicted_pairs = _extract_predicted_pairs(docs, args.top_k)
        metrics = _compute_metrics(
            item["relevant_articles"],
            predicted_articles,
            relevant_pairs=item.get("relevant_pairs", []),
            predicted_pairs=predicted_pairs,
        )
        results.append(
            {
                "id": item["id"],
                "query": item["query"],
                "lang": item["lang"],
                "description": item["description"],
                "relevant_articles": item["relevant_articles"],
                "relevant_pairs": item.get("relevant_pairs", []),
                "predicted_articles": predicted_articles,
                "predicted_pairs": predicted_pairs,
                "metrics": metrics,
                "elapsed_sec": elapsed,
                "error": error,
            }
        )
        print(
            f"[{index}/{len(queries)}] {item['id']} "
            f"strict_hit={metrics['strict_hit']:.0f} "
            f"soft_hit={metrics['soft_hit']:.0f} "
            f"mrr={metrics['mrr']:.3f} "
            f"elapsed={elapsed}s"
        )

    summary = _build_summary(results, args.top_k)
    payload: dict[str, Any] = {
        "generated_at": datetime.now(UTC).isoformat(),
        "queries_path": str(queries_path),
        "summary": summary,
        "results": results,
    }

    compare_path = Path(args.compare_to).expanduser() if args.compare_to else None
    if compare_path:
        with open(compare_path, "r", encoding="utf-8") as handle:
            previous = json.load(handle)
        previous_summary = previous.get("summary", {})
        payload["comparison"] = {
            "baseline_path": str(compare_path),
            "baseline_summary": previous_summary,
            "delta": _build_comparison(summary, previous_summary),
        }

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)

    print("\nSummary:")
    print(f"  strict_hit@{args.top_k}: {summary['strict_hit@k']:.3f}")
    print(f"  soft_hit@{args.top_k}: {summary['soft_hit@k']:.3f}")
    print(f"  mrr: {summary['mrr']:.3f}")
    print(f"Saved: {output_path}")
    if "comparison" in payload:
        delta = payload["comparison"]["delta"]
        print("Comparison:")
        print(f"  delta_strict_hit@k: {delta['delta_strict_hit@k']:+.3f}")
        print(f"  delta_soft_hit@k: {delta['delta_soft_hit@k']:+.3f}")
        print(f"  delta_mrr: {delta['delta_mrr']:+.3f}")


if __name__ == "__main__":
    main()
