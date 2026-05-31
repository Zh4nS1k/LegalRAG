import argparse
import json
from pathlib import Path
from typing import Any

from ai_service.core.code_registry import get_code_name
from ai_service.utils.gold_citations_retrieval_benchmark import (
    _compute_metrics,
    _gold_to_pair,
    _normalize_article,
    _normalize_code,
)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Recompute benchmark metrics with canonical code normalization from source metadata."
    )
    parser.add_argument("--input", required=True, help="Path to benchmark JSON.")
    parser.add_argument("--output", required=True, help="Path to output JSON.")
    return parser.parse_args()


def _avg(results: list[dict[str, Any]], metric_name: str) -> float:
    values = [float(item["metrics"][metric_name]) for item in results]
    return sum(values) / len(values) if values else 0.0


def _canonicalize_pred(item: str) -> tuple[str, str]:
    article, _, code = str(item or "").partition("::")
    return _normalize_article(article), _normalize_code(code)


def _canonicalize_code_from_source(source: str, fallback_code: str) -> str:
    normalized = _normalize_code(fallback_code)
    if normalized and "_" not in normalized:
        return normalized
    if source:
        code_ru, _ = get_code_name(source)
        return _normalize_code(code_ru)
    return normalized


def main() -> None:
    args = _parse_args()
    input_path = Path(args.input)
    payload = json.loads(input_path.read_text(encoding="utf-8"))

    results: list[dict[str, Any]] = []
    for row in payload.get("results", []):
        gold_raw = row.get("gold_citations", [])
        gold_pairs = [_gold_to_pair(item) for item in gold_raw]

        pred_pairs: list[tuple[str, str]] = []
        for pred in row.get("retrieved_topk", []):
            article, code = _canonicalize_pred(pred)
            if article and (article, code) not in pred_pairs:
                pred_pairs.append((article, code))

        # If raw results contained source-aware docs, prefer canonical mapping from source.
        docs = row.get("retrieved_docs") or []
        if docs:
            pred_pairs = []
            for doc in docs:
                meta = doc.get("metadata", {}) or {}
                article = _normalize_article(meta.get("article_number", ""))
                code = _canonicalize_code_from_source(
                    str(meta.get("source", "") or ""),
                    str(meta.get("code_ru", "") or ""),
                )
                if article and (article, code) not in pred_pairs:
                    pred_pairs.append((article, code))

        metrics = _compute_metrics(gold_pairs, pred_pairs)
        updated = dict(row)
        updated["retrieved_topk"] = [
            f"{_normalize_article(article)}::{_normalize_code(code)}"
            for article, code in pred_pairs
        ]
        updated["metrics"] = metrics
        results.append(updated)

    payload["results"] = results
    payload["summary"] = {
        "strict_hit": _avg(results, "strict_hit"),
        "soft_hit": _avg(results, "soft_hit"),
        "strict_precision": _avg(results, "strict_precision"),
        "strict_recall": _avg(results, "strict_recall"),
        "soft_precision": _avg(results, "soft_precision"),
        "soft_recall": _avg(results, "soft_recall"),
        "strict_mrr": _avg(results, "strict_mrr"),
        "soft_mrr": _avg(results, "soft_mrr"),
        "avg_elapsed_sec": sum(item.get("elapsed_sec", 0.0) for item in results) / len(results)
        if results
        else 0.0,
    }

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(payload["summary"], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
