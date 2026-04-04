import argparse
import json
import os
import re
import time
from pathlib import Path
from typing import Any

import pandas as pd


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Offline retrieval benchmark with heavy reranker, checkpoints, and resume."
    )
    parser.add_argument(
        "--xlsx",
        default="tests/benchmarks/642_questions_with_citations.xlsx",
        help="Path to benchmark XLSX file.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=100,
        help="How many questions to evaluate from the start of the sheet.",
    )
    parser.add_argument(
        "--start-from",
        type=int,
        default=0,
        help="Zero-based row offset before applying limit.",
    )
    parser.add_argument(
        "--output",
        default="benchmark_results/heavy_reranker_benchmark.json",
        help="Final output JSON path.",
    )
    parser.add_argument(
        "--checkpoint",
        default="benchmark_results/heavy_reranker_benchmark.checkpoint.json",
        help="Checkpoint JSON path for partial progress and resume.",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume from existing checkpoint if present.",
    )
    parser.add_argument(
        "--save-every",
        type=int,
        default=1,
        help="Persist checkpoint every N processed questions.",
    )
    parser.add_argument(
        "--wide-k",
        type=int,
        default=50,
        help="Retrieval candidate pool size before reranking.",
    )
    parser.add_argument(
        "--top-after-rerank",
        type=int,
        default=5,
        help="How many docs reranker keeps.",
    )
    parser.add_argument(
        "--dedup",
        action="store_true",
        help="Enable dedup layer before reranking.",
    )
    return parser.parse_args()


def _configure_env(args: argparse.Namespace) -> None:
    os.environ["LEGAL_RAG_USE_RERANKER"] = "1"
    os.environ["LEGAL_RAG_RETRIEVER_WIDE_K"] = str(args.wide_k)
    os.environ["LEGAL_RAG_RETRIEVER_TOP_K_AFTER_RERANK"] = str(args.top_after_rerank)
    os.environ["LEGAL_RAG_EXPERIMENTAL_DEDUP_RETRIEVAL"] = "1" if args.dedup else "0"


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
    text = str(value or "").strip().lower()
    if not text:
        return ""
    match = re.search(r"(?:ст\.?|статья)\s*(\d{1,4})", text)
    if match:
        return match.group(1)
    match = re.search(r"(\d{1,4})(?!.*\d)", text)
    return match.group(1) if match else text


def _normalize_code(value: str) -> str:
    text = str(value or "").strip().lower().replace("ё", "е")
    if not text:
        return ""
    text = text.replace("«", '"').replace("»", '"')
    text = re.sub(r"\s+", " ", text)

    alias_groups = [
        (
            "гражданский кодекс рк",
            [
                "гражданский кодекс рк (общая часть)",
                "гражданский кодекс рк (особенная часть)",
                "гражданский кодекс рк",
                "гк рк",
                "гк",
            ],
        ),
        (
            "уголовный кодекс рк",
            ["уголовный кодекс рк", "ук рк", "ук"],
        ),
        (
            "гражданский процессуальный кодекс рк",
            ["гражданский процессуальный кодекс рк", "гпк рк", "гпк"],
        ),
        (
            "уголовно-процессуальный кодекс рк",
            ["уголовно-процессуальный кодекс рк", "упк рк", "упк"],
        ),
        (
            "кодекс об административных правонарушениях рк",
            [
                "кодекс об административных правонарушениях рк",
                "коап рк",
                "коап",
            ],
        ),
        (
            "налоговый кодекс рк",
            ["налоговый кодекс рк", "нк рк", "нк"],
        ),
        (
            "кодекс об административных процедурах рк",
            [
                "кодекс об административных процедурах рк",
                "аппк рк",
                "аппк",
            ],
        ),
        (
            "закон о защите прав потребителей рк",
            [
                "закон о защите прав потребителей рк",
                "закон о защите прав потребителей",
                "о защите прав потребителей",
                "зпп",
            ],
        ),
        (
            "закон об адвокатской деятельности и юридической помощи",
            [
                "закон об адвокатской деятельности и юридической помощи",
                "об адвокатской деятельности и юридической помощи",
            ],
        ),
        (
            "закон о валютном регулировании и валютном контроле",
            [
                "закон о валютном регулировании и валютном контроле",
                "о валютном регулировании и валютном контроле",
            ],
        ),
    ]

    for canonical, aliases in alias_groups:
        if any(alias in text for alias in aliases):
            return canonical

    text = re.sub(r'\bреспублики казахстан\b', "рк", text)
    text = re.sub(r"[\"']", "", text)
    text = re.sub(r"\s+", " ", text).strip(" ,.")
    return text


def _gold_to_pair(value: str) -> tuple[str, str]:
    raw = str(value or "").strip()
    if not raw:
        return "", ""
    if "::" in raw:
        article_part, code_part = raw.split("::", 1)
        return _normalize_article(article_part), _normalize_code(code_part)

    article = _normalize_article(raw)
    code_match = ""
    lower_raw = raw.lower()
    if "закон" in lower_raw:
        law_idx = lower_raw.find("закон")
        code_match = raw[law_idx:]
    elif "кодекс" in lower_raw:
        code_match = raw
    elif "," in raw:
        code_match = raw.split(",", 1)[1]
    elif "ст." in lower_raw:
        parts = raw.split(" ", 2)
        code_match = parts[2] if len(parts) > 2 else ""
    return article, _normalize_code(code_match)


def _extract_pairs(docs: list[Any]) -> list[dict[str, str]]:
    pairs: list[dict[str, str]] = []
    for doc in docs:
        metadata = getattr(doc, "metadata", {}) or {}
        article = str(metadata.get("article_number") or "").strip()
        code = str(metadata.get("code_ru") or "").strip()
        source = str(metadata.get("source") or "").strip()
        if article or code or source:
            pairs.append(
                {
                    "article": article,
                    "code": code,
                    "source": source,
                }
            )
    return pairs


def _pair_to_str(pair: dict[str, str]) -> str:
    article = _normalize_article(pair.get("article", ""))
    code = _normalize_code(pair.get("code", ""))
    return f"{article}::{code}"


def _article_to_str(pair: dict[str, str]) -> str:
    return _normalize_article(pair.get("article", ""))


def _compute_summary(results: list[dict[str, Any]]) -> dict[str, Any]:
    total = len(results)
    strict_hits = sum(1 for r in results if r.get("strict_hit5"))
    soft_hits = sum(1 for r in results if r.get("soft_hit5"))
    elapsed_values = [
        float(r.get("elapsed_sec", 0.0)) for r in results if r.get("elapsed_sec") is not None
    ]
    return {
        "questions_evaluated": total,
        "strict_hit5": (strict_hits / total) if total else 0.0,
        "soft_hit5": (soft_hits / total) if total else 0.0,
        "total_elapsed_sec": sum(elapsed_values),
        "avg_elapsed_sec": (sum(elapsed_values) / total) if total else 0.0,
    }


def _load_checkpoint(path: Path) -> dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        payload = json.load(f)
    if not isinstance(payload, dict):
        raise ValueError(f"Checkpoint {path} must contain a JSON object")
    return payload


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def main() -> None:
    args = _parse_args()
    _configure_env(args)

    output_path = Path(args.output)
    checkpoint_path = Path(args.checkpoint)
    xlsx_path = Path(args.xlsx)

    if not xlsx_path.exists():
        raise FileNotFoundError(f"Benchmark XLSX not found: {xlsx_path}")

    df = pd.read_excel(xlsx_path)
    rows = df.iloc[args.start_from : args.start_from + args.limit].copy()

    results: list[dict[str, Any]] = []
    started_at = time.time()
    resumed_from = 0

    if args.resume and checkpoint_path.exists():
        checkpoint = _load_checkpoint(checkpoint_path)
        previous_results = checkpoint.get("results") or []
        if not isinstance(previous_results, list):
            raise ValueError("Checkpoint results must be a list")
        results = previous_results
        resumed_from = len(results)
        started_at = float(checkpoint.get("started_at_epoch", started_at))
        print(f"Resuming from checkpoint: {checkpoint_path} ({resumed_from} completed)")

    from ai_service.retrieval.rag_chain import get_retriever

    retriever = get_retriever()

    for local_idx, (_, row) in enumerate(rows.iterrows()):
        if local_idx < resumed_from:
            continue

        query = str(row["query"])
        gold_items = _normalize_gold(row.get("gold_citations"))
        gold_pairs = [_gold_to_pair(g) for g in gold_items if str(g).strip()]
        gold_strict = {
            f"{article}::{code}"
            for article, code in gold_pairs
            if article or code
        }
        gold_soft = {article for article, _code in gold_pairs if article}

        t0 = time.perf_counter()
        docs = retriever.invoke(query)
        elapsed = time.perf_counter() - t0

        pairs = _extract_pairs(docs[:5])
        pred_strict = [_pair_to_str(pair).lower() for pair in pairs]
        pred_soft = [_article_to_str(pair).lower() for pair in pairs]
        strict_hit = any(value in gold_strict for value in pred_strict)
        soft_hit = any(value in gold_soft for value in pred_soft)

        result = {
            "row_index": int(args.start_from + local_idx),
            "id": row.get("id"),
            "query": query,
            "gold_citations": gold_items,
            "top5": pairs,
            "strict_hit5": strict_hit,
            "soft_hit5": soft_hit,
            "elapsed_sec": round(elapsed, 4),
        }
        results.append(result)

        processed = len(results)
        summary = _compute_summary(results)
        print(
            f"[{processed}/{len(rows)}] "
            f"strict_hit5={summary['strict_hit5']:.3f} "
            f"soft_hit5={summary['soft_hit5']:.3f} "
            f"avg_sec={summary['avg_elapsed_sec']:.2f}",
            flush=True,
        )

        if processed % max(args.save_every, 1) == 0:
            checkpoint_payload = {
                "config": {
                    "xlsx": str(xlsx_path),
                    "limit": args.limit,
                    "start_from": args.start_from,
                    "wide_k": args.wide_k,
                    "top_after_rerank": args.top_after_rerank,
                    "dedup": args.dedup,
                },
                "started_at_epoch": started_at,
                "updated_at_epoch": time.time(),
                "summary": summary,
                "results": results,
            }
            _write_json(checkpoint_path, checkpoint_payload)

    final_summary = _compute_summary(results)
    final_payload = {
        "config": {
            "xlsx": str(xlsx_path),
            "limit": args.limit,
            "start_from": args.start_from,
            "wide_k": args.wide_k,
            "top_after_rerank": args.top_after_rerank,
            "dedup": args.dedup,
            "resume": args.resume,
        },
        "started_at_epoch": started_at,
        "finished_at_epoch": time.time(),
        "wall_clock_sec": round(time.time() - started_at, 3),
        "summary": final_summary,
        "results": results,
    }
    _write_json(output_path, final_payload)
    _write_json(checkpoint_path, final_payload)
    print("FINAL_STRICT_HIT5", final_summary["strict_hit5"])
    print("FINAL_SOFT_HIT5", final_summary["soft_hit5"])
    print("OUTPUT", output_path)


if __name__ == "__main__":
    main()
