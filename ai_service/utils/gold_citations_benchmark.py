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
        description="Benchmark RAG answers against gold citations from XLSX."
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
        "--output",
        default="benchmark_results/gold_citations_answer_benchmark.json",
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
        ("гражданский кодекс рк", ("гк рк", "гк", "гражданский кодекс")),
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
            "закон о валютном регулировании и валютном контроле",
            ("о валютном регулировании и валютном контроле",),
        ),
        (
            "закон об адвокатской деятельности и юридической помощи",
            ("об адвокатской деятельности и юридической помощи",),
        ),
        (
            "закон о защите прав потребителей рк",
            ("закон о защите прав потребителей", "о защите прав потребителей", "зпп"),
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
    elif "кодекс" in lower_raw:
        code = raw
    elif "гк" in lower_raw or "ук" in lower_raw or "гпк" in lower_raw or "коап" in lower_raw:
        code = raw
    return article, _normalize_code(code)


def _extract_citation_pairs_from_text(text: str) -> list[tuple[str, str]]:
    if not text:
        return []
    compact = re.sub(r"\s+", " ", text).strip()
    patterns = [
        r"(ст\.?|статья|бап)\s*(\d{1,4}(?:-\d+)?)\s+([^.;:\n]{0,140}?(?:кодекс|закон)[^.;:\n]{0,140})",
        r"(ст\.?|статья|бап)\s*(\d{1,4}(?:-\d+)?)\s+((?:гк|ук|гпк|упк|коап)\s*рк?)",
    ]
    pairs: list[tuple[str, str]] = []
    for pattern in patterns:
        for match in re.finditer(pattern, compact, re.IGNORECASE):
            article = _normalize_article(match.group(2))
            code = _normalize_code(match.group(3))
            if article and code and (article, code) not in pairs:
                pairs.append((article, code))
    return pairs


def _extract_pairs_from_docs(docs: list[Any]) -> list[tuple[str, str]]:
    pairs: list[tuple[str, str]] = []
    for doc in docs or []:
        meta = getattr(doc, "metadata", {}) or {}
        article = _normalize_article(meta.get("article_number", ""))
        code = _normalize_code(meta.get("code_ru", ""))
        if article and code and (article, code) not in pairs:
            pairs.append((article, code))
    return pairs


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
    from ai_service.retrieval.rag_chain import invoke_qa

    xlsx_path = Path(args.xlsx)
    if not xlsx_path.exists():
        raise FileNotFoundError(f"Benchmark XLSX not found: {xlsx_path}")

    df = pd.read_excel(xlsx_path)
    df = df.iloc[args.start_from : args.start_from + args.limit].copy()

    results: list[dict[str, Any]] = []
    for row_index, row in df.iterrows():
        query = str(row.get("query") or "").strip()
        qid = str(row.get("query_id") or row_index)
        gold_raw = _normalize_gold(row.get("gold_citations"))
        gold_pairs = [_gold_to_pair(item) for item in gold_raw]

        started = time.perf_counter()
        try:
            response = invoke_qa(query)
            answer = response.get("result", "")
            source_documents = response.get("source_documents", [])
            error = ""
        except Exception as exc:
            answer = ""
            source_documents = []
            error = str(exc)
        elapsed_sec = round(time.perf_counter() - started, 3)

        answer_pairs = _extract_citation_pairs_from_text(str(answer))
        source_pairs = _extract_pairs_from_docs(source_documents)
        combined_pairs = list(dict.fromkeys(answer_pairs + source_pairs))

        answer_metrics = _compute_metrics(gold_pairs, answer_pairs)
        combined_metrics = _compute_metrics(gold_pairs, combined_pairs)

        results.append(
            {
                "row_index": int(row_index),
                "query_id": qid,
                "query": query,
                "gold_citations": gold_raw,
                "answer_citations": [f"{a}::{c}" for a, c in answer_pairs],
                "source_citations": [f"{a}::{c}" for a, c in source_pairs],
                "combined_citations": [f"{a}::{c}" for a, c in combined_pairs],
                "answer_metrics": answer_metrics,
                "combined_metrics": combined_metrics,
                "elapsed_sec": elapsed_sec,
                "error": error,
            }
        )
        print(
            f"[{len(results)}/{len(df)}] {qid} "
            f"answer_strict={answer_metrics['strict_hit']:.0f} "
            f"combined_strict={combined_metrics['strict_hit']:.0f} "
            f"elapsed={elapsed_sec}s"
        )

    def _avg(metric_group: str, metric_name: str) -> float:
        values = [float(item[metric_group][metric_name]) for item in results]
        return sum(values) / len(values) if values else 0.0

    payload = {
        "xlsx": str(xlsx_path),
        "evaluated_questions": len(results),
        "generated_at": datetime.utcnow().isoformat() + "Z",
        "answer_only": {
            "strict_hit": _avg("answer_metrics", "strict_hit"),
            "soft_hit": _avg("answer_metrics", "soft_hit"),
            "strict_precision": _avg("answer_metrics", "strict_precision"),
            "strict_recall": _avg("answer_metrics", "strict_recall"),
            "soft_precision": _avg("answer_metrics", "soft_precision"),
            "soft_recall": _avg("answer_metrics", "soft_recall"),
            "strict_mrr": _avg("answer_metrics", "strict_mrr"),
            "soft_mrr": _avg("answer_metrics", "soft_mrr"),
        },
        "answer_plus_sources": {
            "strict_hit": _avg("combined_metrics", "strict_hit"),
            "soft_hit": _avg("combined_metrics", "soft_hit"),
            "strict_precision": _avg("combined_metrics", "strict_precision"),
            "strict_recall": _avg("combined_metrics", "strict_recall"),
            "soft_precision": _avg("combined_metrics", "soft_precision"),
            "soft_recall": _avg("combined_metrics", "soft_recall"),
            "strict_mrr": _avg("combined_metrics", "strict_mrr"),
            "soft_mrr": _avg("combined_metrics", "soft_mrr"),
        },
        "results": results,
    }

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
    print(f"\nSaved: {output_path}")
    print(json.dumps(payload["answer_plus_sources"], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
