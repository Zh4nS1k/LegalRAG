import argparse
import json
import time
from datetime import datetime, UTC
from pathlib import Path
from statistics import mean
from typing import Any

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


def _classify_query(
    query: str,
    description: str,
    lang: str,
    relevant_articles: list[str],
    relevant_pairs: list[tuple[str, str]],
) -> list[str]:
    text = " ".join(part for part in (query, description) if part).lower()
    tags: list[str] = []

    if lang:
        tags.append(f"lang:{lang}")

    tags.append("multi_article" if len(set(relevant_articles)) > 1 else "single_article")
    if len(set(relevant_articles)) >= 5 or "-" in text:
        tags.append("range_query")

    compound_markers = (
        " и ",
        " және ",
        "еще",
        "ещё",
        "вместе с",
        "одновременно",
        "қоса",
        "бірге",
    )
    if len(set(relevant_articles)) > 1 or any(marker in text for marker in compound_markers):
        tags.append("compound_issue")

    penalty_markers = (
        "что грозит",
        "какое наказание",
        "какая ответственность",
        "какая статья",
        "какие статьи",
        "какой жаза",
        "қандай жаза",
        "қандай жауапкершілік",
        "қандай бап",
    )
    if any(marker in text for marker in penalty_markers):
        tags.append("penalty_focused")

    lookup_markers = ("статья", "статьи", "бап", "ук рк", "қр қк", "кодекс")
    if any(marker in text for marker in lookup_markers) or relevant_pairs:
        tags.append("code_lookup")

    if not tags:
        tags.append("uncategorized")
    return list(dict.fromkeys(tags))


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
        lang = str(item.get("lang", "")).strip()
        description = str(item.get("description", "")).strip()
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
        relevant_pairs = _load_relevant_pairs(item)
        rows.append(
            {
                "id": item.get("id", f"q_{idx:03d}"),
                "query": query,
                "lang": lang,
                "description": description,
                "relevant_articles": relevant_articles,
                "relevant_pairs": relevant_pairs,
                "tags": _classify_query(query, description, lang, relevant_articles, relevant_pairs),
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
    normalized_predicted_pair_set = set(normalized_predicted_pairs)

    if normalized_relevant_pairs:
        strict_relevant_set: set[Any] = normalized_relevant_pairs
        strict_predicted_items: list[Any] = normalized_predicted_pairs
        strict_predicted_set: set[Any] = normalized_predicted_pair_set
    else:
        strict_relevant_set = relevant_set
        strict_predicted_items = predicted
        strict_predicted_set = predicted_set

    if normalized_relevant_pairs:
        strict_hit = 1.0 if normalized_predicted_pair_set & normalized_relevant_pairs else 0.0
    else:
        strict_hit = 1.0 if relevant_set & predicted_set else 0.0
    soft_hit = 1.0 if relevant_set & predicted_set else 0.0

    strict_matches = len(strict_relevant_set & strict_predicted_set)
    soft_matches = len(relevant_set & predicted_set)

    strict_precision = _safe_divide(strict_matches, len(strict_predicted_set))
    strict_recall = _safe_divide(strict_matches, len(strict_relevant_set))
    soft_precision = _safe_divide(soft_matches, len(predicted_set))
    soft_recall = _safe_divide(soft_matches, len(relevant_set))

    reciprocal_rank = 0.0
    for rank, article in enumerate(predicted, start=1):
        if article in relevant_set:
            reciprocal_rank = 1.0 / rank
            break

    return {
        "strict_hit": strict_hit,
        "soft_hit": soft_hit,
        "strict_precision": strict_precision,
        "strict_recall": strict_recall,
        "strict_f1": _f1(strict_precision, strict_recall),
        "soft_precision": soft_precision,
        "soft_recall": soft_recall,
        "soft_f1": _f1(soft_precision, soft_recall),
        "strict_ap": _average_precision(strict_predicted_items, strict_relevant_set),
        "soft_ap": _average_precision(predicted, relevant_set),
        "mrr": reciprocal_rank,
        "relevant_count": float(len(relevant_set)),
        "predicted_count": float(len(predicted_set)),
        "strict_match_count": float(strict_matches),
        "soft_match_count": float(soft_matches),
    }


def _average_metric(results: list[dict[str, Any]], metric_name: str) -> float:
    values = [float(row["metrics"][metric_name]) for row in results]
    return sum(values) / len(values) if values else 0.0


def _safe_divide(numerator: float, denominator: float) -> float:
    return numerator / denominator if denominator else 0.0


def _f1(precision: float, recall: float) -> float:
    return _safe_divide(2 * precision * recall, precision + recall)


def _average_precision(predicted_items: list[Any], relevant_items: set[Any]) -> float:
    if not relevant_items:
        return 0.0

    hits = 0
    precision_sum = 0.0
    seen: set[Any] = set()
    for rank, item in enumerate(predicted_items, start=1):
        if item in seen:
            continue
        seen.add(item)
        if item in relevant_items:
            hits += 1
            precision_sum += hits / rank
    return precision_sum / len(relevant_items)


def _percentile(values: list[float], percentile: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]
    index = round((len(ordered) - 1) * percentile)
    return ordered[index]


def _make_summary(results: list[dict[str, Any]], top_k: int) -> dict[str, float | int]:
    elapsed_values = [float(row.get("elapsed_sec", 0.0)) for row in results]
    return {
        "queries_evaluated": len(results),
        "top_k": top_k,
        "strict_hit@k": _average_metric(results, "strict_hit"),
        "soft_hit@k": _average_metric(results, "soft_hit"),
        "strict_precision@k": _average_metric(results, "strict_precision"),
        "strict_recall@k": _average_metric(results, "strict_recall"),
        "strict_f1@k": _average_metric(results, "strict_f1"),
        "soft_precision@k": _average_metric(results, "soft_precision"),
        "soft_recall@k": _average_metric(results, "soft_recall"),
        "soft_f1@k": _average_metric(results, "soft_f1"),
        "map_strict": _average_metric(results, "strict_ap"),
        "map_soft": _average_metric(results, "soft_ap"),
        "mrr": _average_metric(results, "mrr"),
        "avg_relevant_articles": _average_metric(results, "relevant_count"),
        "avg_predicted_articles": _average_metric(results, "predicted_count"),
        "latency_sec_avg": mean(elapsed_values) if elapsed_values else 0.0,
        "latency_sec_p95": _percentile(elapsed_values, 0.95),
    }


def _group_results(results: list[dict[str, Any]], top_k: int, key: str) -> dict[str, dict[str, float | int]]:
    grouped: dict[str, list[dict[str, Any]]] = {}
    for row in results:
        label = str(row.get(key) or "unknown")
        grouped.setdefault(label, []).append(row)
    return {label: _make_summary(group_rows, top_k) for label, group_rows in sorted(grouped.items())}


def _group_results_by_tags(results: list[dict[str, Any]], top_k: int) -> dict[str, dict[str, float | int]]:
    grouped: dict[str, list[dict[str, Any]]] = {}
    for row in results:
        for tag in row.get("tags", []):
            grouped.setdefault(str(tag), []).append(row)
    return {label: _make_summary(group_rows, top_k) for label, group_rows in sorted(grouped.items())}


def _build_summary(results: list[dict[str, Any]], top_k: int) -> dict[str, Any]:
    by_complexity: dict[str, list[dict[str, Any]]] = {"single_article": [], "multi_article": []}
    for row in results:
        relevant_count = int(float(row["metrics"].get("relevant_count", 0.0)))
        bucket = "multi_article" if relevant_count > 1 else "single_article"
        by_complexity[bucket].append(row)

    summary = _make_summary(results, top_k)
    summary["by_lang"] = _group_results(results, top_k, "lang")
    summary["by_tag"] = _group_results_by_tags(results, top_k)
    summary["by_complexity"] = {
        label: _make_summary(group_rows, top_k)
        for label, group_rows in by_complexity.items()
        if group_rows
    }
    return summary


def _build_comparison(current_summary: dict[str, Any], previous_summary: dict[str, Any]) -> dict[str, float]:
    comparison: dict[str, float] = {}
    metric_names = (
        "strict_hit@k",
        "soft_hit@k",
        "strict_precision@k",
        "strict_recall@k",
        "strict_f1@k",
        "soft_precision@k",
        "soft_recall@k",
        "soft_f1@k",
        "map_strict",
        "map_soft",
        "mrr",
        "avg_relevant_articles",
        "avg_predicted_articles",
        "latency_sec_avg",
        "latency_sec_p95",
    )
    for metric_name in metric_names:
        if metric_name in current_summary or metric_name in previous_summary:
            comparison[f"delta_{metric_name}"] = float(current_summary.get(metric_name, 0.0)) - float(
                previous_summary.get(metric_name, 0.0)
            )
    return comparison


def run_retrieval_quality_benchmark(
    *,
    queries_path: str | Path,
    top_k: int = 5,
    limit: int = 0,
    output_path: str | Path = "benchmark_results/retrieval_quality_benchmark.json",
    compare_to: str | Path | None = None,
) -> dict[str, Any]:
    queries_path = Path(queries_path)
    if not queries_path.exists():
        raise FileNotFoundError(f"Queries JSON not found: {queries_path}")

    from engine.retrieval.rag_chain import get_retriever_for_coverage

    queries = _load_queries(queries_path, limit)
    retriever = get_retriever_for_coverage(top_k)

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

        predicted_articles = _extract_predicted_articles(docs, top_k)
        predicted_pairs = _extract_predicted_pairs(docs, top_k)
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
                "tags": item.get("tags", []),
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

    summary = _build_summary(results, top_k)
    payload: dict[str, Any] = {
        "generated_at": datetime.now(UTC).isoformat(),
        "queries_path": str(queries_path),
        "summary": summary,
        "results": results,
    }

    compare_path = Path(compare_to).expanduser() if compare_to else None
    if compare_path:
        with open(compare_path, "r", encoding="utf-8") as handle:
            previous = json.load(handle)
        previous_summary = previous.get("summary", {})
        payload["comparison"] = {
            "baseline_path": str(compare_path),
            "baseline_summary": previous_summary,
            "delta": _build_comparison(summary, previous_summary),
        }

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)

    return payload


def main() -> None:
    args = _parse_args()
    payload = run_retrieval_quality_benchmark(
        queries_path=args.queries,
        top_k=args.top_k,
        limit=args.limit,
        output_path=args.output,
        compare_to=args.compare_to or None,
    )
    summary = payload["summary"]

    print("\nSummary:")
    print(f"  strict_hit@{args.top_k}: {summary['strict_hit@k']:.3f}")
    print(f"  soft_hit@{args.top_k}: {summary['soft_hit@k']:.3f}")
    print(f"  mrr: {summary['mrr']:.3f}")
    print(f"Saved: {args.output}")
    if "comparison" in payload:
        delta = payload["comparison"]["delta"]
        print("Comparison:")
        print(f"  delta_strict_hit@k: {delta['delta_strict_hit@k']:+.3f}")
        print(f"  delta_soft_hit@k: {delta['delta_soft_hit@k']:+.3f}")
        print(f"  delta_mrr: {delta['delta_mrr']:+.3f}")


if __name__ == "__main__":
    main()
