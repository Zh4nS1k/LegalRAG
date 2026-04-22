import argparse
import json
import re
from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate article recognition from precomputed RAG XLSX."
    )
    parser.add_argument(
        "--xlsx",
        default="tests/benchmarks/642_questions_with_rag_answers.sample10.xlsx",
        help="XLSX with query, gold_citations, rag_answer, rag_sources.",
    )
    parser.add_argument(
        "--output",
        default="benchmark_results/rag_answers_article_eval.json",
        help="Output JSON path.",
    )
    parser.add_argument("--limit", type=int, default=642, help="Rows to evaluate.")
    parser.add_argument("--start-from", type=int, default=0, help="Zero-based offset.")
    parser.add_argument("--gold-column", default="gold_citations")
    parser.add_argument("--answer-column", default="rag_answer")
    parser.add_argument("--sources-column", default="rag_sources")
    parser.add_argument("--query-column", default="query")
    parser.add_argument("--id-column", default="query_id")
    return parser.parse_args()


def _normalize_article(value: Any) -> str:
    text = str(value or "").strip().lower().replace("ё", "е")
    if not text:
        return ""
    match = re.search(r"(?:ст\.?|статья|бап)\s*(\d{1,4}(?:-\d+)?)", text)
    if match:
        return match.group(1)
    match = re.search(r"(\d{1,4}(?:-\d+)?)", text)
    return match.group(1) if match else ""


def _normalize_code(value: Any) -> str:
    text = str(value or "").strip().lower().replace("ё", "е")
    if not text:
        return ""
    text = text.replace("«", '"').replace("»", '"')
    text = re.sub(r"\s+", " ", text)
    text = re.sub(r'\bреспублики казахстан\b', "рк", text)
    text = re.sub(r"[\"']", "", text)
    return text.strip(" ,.")


def _gold_items(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, float) and pd.isna(value):
        return []
    text = str(value).strip()
    if not text:
        return []
    parts = [part.strip() for part in text.split(";") if part.strip()]
    return parts if parts else [text]


def _gold_to_pair(item: str) -> tuple[str, str]:
    raw = str(item or "").strip()
    if not raw:
        return "", ""
    article = _normalize_article(raw)
    lower_raw = raw.lower()
    code = ""
    if "закон" in lower_raw:
        code = raw[lower_raw.find("закон") :]
    elif "кодекс" in lower_raw or "гк" in lower_raw or "ук" in lower_raw or "коап" in lower_raw:
        code = raw
    return article, _normalize_code(code)


def _extract_pairs_from_answer(answer_text: str) -> list[tuple[str, str]]:
    if not answer_text:
        return []
    text = re.sub(r"\s+", " ", answer_text).strip()
    patterns = [
        r"(?:ст\.?|статья|бап)\s*(\d{1,4}(?:-\d+)?)\s+([^.;:\n]{0,140}?(?:кодекс|закон)[^.;:\n]{0,140})",
        r"(?:ст\.?|статья|бап)\s*(\d{1,4}(?:-\d+)?)\s+((?:гк|ук|гпк|упк|коап)\s*рк?)",
    ]
    pairs: list[tuple[str, str]] = []
    for pattern in patterns:
        for match in re.finditer(pattern, text, re.IGNORECASE):
            article = _normalize_article(match.group(1))
            code = _normalize_code(match.group(2))
            if article and code and (article, code) not in pairs:
                pairs.append((article, code))
    return pairs


def _parse_sources_json(raw_sources: Any) -> list[dict[str, Any]]:
    if raw_sources is None:
        return []
    if isinstance(raw_sources, float) and pd.isna(raw_sources):
        return []
    if isinstance(raw_sources, list):
        return [item for item in raw_sources if isinstance(item, dict)]
    text = str(raw_sources).strip()
    if not text:
        return []
    try:
        payload = json.loads(text)
        if isinstance(payload, list):
            return [item for item in payload if isinstance(item, dict)]
    except Exception:
        return []
    return []


def _extract_pairs_from_sources(raw_sources: Any) -> list[tuple[str, str]]:
    pairs: list[tuple[str, str]] = []
    for item in _parse_sources_json(raw_sources):
        article = _normalize_article(item.get("article_number"))
        code = _normalize_code(item.get("code_ru"))
        if article and code and (article, code) not in pairs:
            pairs.append((article, code))
    return pairs


def _compute_metrics(gold_pairs: list[tuple[str, str]], pred_pairs: list[tuple[str, str]]) -> dict[str, float]:
    gold_pair_set = {pair for pair in gold_pairs if pair[0]}
    pred_pair_set = {pair for pair in pred_pairs if pair[0]}
    gold_articles = {article for article, _ in gold_pair_set}
    pred_articles = {article for article, _ in pred_pair_set}

    strict_hit = 1.0 if gold_pair_set & pred_pair_set else 0.0
    article_hit = 1.0 if gold_articles & pred_articles else 0.0
    strict_precision = len(gold_pair_set & pred_pair_set) / len(pred_pair_set) if pred_pair_set else 0.0
    strict_recall = len(gold_pair_set & pred_pair_set) / len(gold_pair_set) if gold_pair_set else 0.0
    article_precision = len(gold_articles & pred_articles) / len(pred_articles) if pred_articles else 0.0
    article_recall = len(gold_articles & pred_articles) / len(gold_articles) if gold_articles else 0.0

    strict_mrr = 0.0
    article_mrr = 0.0
    for rank, pair in enumerate(pred_pairs, start=1):
        if not strict_mrr and pair in gold_pair_set:
            strict_mrr = 1.0 / rank
        if not article_mrr and pair[0] in gold_articles:
            article_mrr = 1.0 / rank
        if strict_mrr and article_mrr:
            break

    return {
        "strict_hit": strict_hit,
        "article_hit": article_hit,
        "strict_precision": strict_precision,
        "strict_recall": strict_recall,
        "article_precision": article_precision,
        "article_recall": article_recall,
        "strict_mrr": strict_mrr,
        "article_mrr": article_mrr,
    }


def main() -> None:
    args = _parse_args()
    xlsx_path = Path(args.xlsx)
    if not xlsx_path.exists():
        raise FileNotFoundError(f"XLSX not found: {xlsx_path}")

    df = pd.read_excel(xlsx_path)
    df = df.iloc[args.start_from : args.start_from + args.limit].copy()

    results: list[dict[str, Any]] = []
    for row_index, row in df.iterrows():
        query_id = str(row.get(args.id_column) or row_index)
        query = str(row.get(args.query_column) or "").strip()

        gold_raw = _gold_items(row.get(args.gold_column))
        gold_pairs = [_gold_to_pair(item) for item in gold_raw if item]

        answer_text = str(row.get(args.answer_column) or "")
        answer_pairs = _extract_pairs_from_answer(answer_text)
        source_pairs = _extract_pairs_from_sources(row.get(args.sources_column))
        combined_pairs = list(dict.fromkeys(answer_pairs + source_pairs))

        answer_metrics = _compute_metrics(gold_pairs, answer_pairs)
        combined_metrics = _compute_metrics(gold_pairs, combined_pairs)

        results.append(
            {
                "row_index": int(row_index),
                "query_id": query_id,
                "query": query,
                "gold_citations": gold_raw,
                "answer_citations": [f"{a}::{c}" for a, c in answer_pairs],
                "source_citations": [f"{a}::{c}" for a, c in source_pairs],
                "combined_citations": [f"{a}::{c}" for a, c in combined_pairs],
                "answer_metrics": answer_metrics,
                "combined_metrics": combined_metrics,
            }
        )

    def _avg(group_name: str, metric_name: str) -> float:
        values = [float(item[group_name][metric_name]) for item in results]
        return sum(values) / len(values) if values else 0.0

    payload = {
        "xlsx": str(xlsx_path),
        "evaluated_questions": len(results),
        "generated_at": datetime.utcnow().isoformat() + "Z",
        "answer_only": {
            "strict_hit": _avg("answer_metrics", "strict_hit"),
            "article_hit": _avg("answer_metrics", "article_hit"),
            "strict_precision": _avg("answer_metrics", "strict_precision"),
            "strict_recall": _avg("answer_metrics", "strict_recall"),
            "article_precision": _avg("answer_metrics", "article_precision"),
            "article_recall": _avg("answer_metrics", "article_recall"),
            "strict_mrr": _avg("answer_metrics", "strict_mrr"),
            "article_mrr": _avg("answer_metrics", "article_mrr"),
        },
        "answer_plus_sources": {
            "strict_hit": _avg("combined_metrics", "strict_hit"),
            "article_hit": _avg("combined_metrics", "article_hit"),
            "strict_precision": _avg("combined_metrics", "strict_precision"),
            "strict_recall": _avg("combined_metrics", "strict_recall"),
            "article_precision": _avg("combined_metrics", "article_precision"),
            "article_recall": _avg("combined_metrics", "article_recall"),
            "strict_mrr": _avg("combined_metrics", "strict_mrr"),
            "article_mrr": _avg("combined_metrics", "article_mrr"),
        },
        "results": results,
    }

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)

    print(f"Saved: {output_path}")
    print(json.dumps(payload["answer_plus_sources"], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
