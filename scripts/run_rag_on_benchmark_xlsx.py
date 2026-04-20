import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Any

# Repo root must be on sys.path so `import ai_service` works when running
# `python scripts/run_rag_on_benchmark_xlsx.py` (not only `python -m ...`).
_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import pandas as pd


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run LegalRAG over XLSX queries and save model answers."
    )
    parser.add_argument(
        "--xlsx",
        default="tests/benchmarks/642_questions_with_citations.backup.xlsx",
        help="Input XLSX path with a query column.",
    )
    parser.add_argument(
        "--output",
        default="tests/benchmarks/642_questions_with_rag_answers.xlsx",
        help="Output XLSX path.",
    )
    parser.add_argument(
        "--query-column",
        default="query",
        help="Column name containing user questions.",
    )
    parser.add_argument(
        "--id-column",
        default="query_id",
        help="Optional id column used in logs.",
    )
    parser.add_argument(
        "--answer-column",
        default="rag_answer",
        help="Column name for generated answers.",
    )
    parser.add_argument(
        "--sources-column",
        default="rag_sources",
        help="Column name for retrieved source metadata (JSON).",
    )
    parser.add_argument(
        "--method-column",
        default="rag_retrieval_method",
        help="Column name for retrieval method.",
    )
    parser.add_argument(
        "--start-from",
        type=int,
        default=0,
        help="Zero-based row offset for processing.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=642,
        help="How many rows to process from start offset.",
    )
    parser.add_argument(
        "--skip-filled",
        action="store_true",
        help="Skip rows where answer column is already non-empty.",
    )
    parser.add_argument(
        "--backend",
        default="ollama",
        choices=("ollama", "groq", "openrouter", "ollama_cloud"),
        help="LLM backend for rag_chain (env LEGAL_RAG_LLM_BACKEND).",
    )
    parser.add_argument(
        "--model",
        default="qwen2.5:14b-instruct",
        help="LLM model name for selected backend (env LEGAL_RAG_LLM).",
    )
    return parser.parse_args()


def _serialize_sources(source_documents: list[Any]) -> str:
    payload: list[dict[str, Any]] = []
    for doc in source_documents or []:
        if hasattr(doc, "metadata"):
            metadata = dict(doc.metadata or {})
            payload.append(
                {
                    "source": metadata.get("source", ""),
                    "code_ru": metadata.get("code_ru", ""),
                    "article_number": metadata.get("article_number", ""),
                }
            )
        elif isinstance(doc, dict):
            metadata = doc.get("metadata", {}) or {}
            payload.append(
                {
                    "source": metadata.get("source", ""),
                    "code_ru": metadata.get("code_ru", ""),
                    "article_number": metadata.get("article_number", ""),
                }
            )
    return json.dumps(payload, ensure_ascii=False)


def main() -> None:
    args = _parse_args()
    os.environ["LEGAL_RAG_LLM_BACKEND"] = args.backend
    os.environ["LEGAL_RAG_LLM"] = args.model

    from ai_service.retrieval.rag_chain import invoke_qa

    input_path = Path(args.xlsx)
    if not input_path.exists():
        raise FileNotFoundError(f"Input XLSX not found: {input_path}")

    df = pd.read_excel(input_path)
    if args.query_column not in df.columns:
        raise ValueError(f"Column '{args.query_column}' not found in XLSX")

    for col_name in (args.answer_column, args.sources_column, args.method_column):
        if col_name not in df.columns:
            df[col_name] = ""

    total_rows = len(df)
    start = max(0, int(args.start_from))
    end = min(total_rows, start + max(0, int(args.limit)))
    if start >= end:
        raise ValueError(f"Empty processing range: start={start}, end={end}")

    processed = 0
    skipped = 0
    errors = 0
    started_at = time.perf_counter()

    for idx in range(start, end):
        row = df.iloc[idx]
        query = str(row.get(args.query_column) or "").strip()
        row_id = str(row.get(args.id_column) or idx)

        if not query:
            skipped += 1
            continue

        if args.skip_filled and str(row.get(args.answer_column) or "").strip():
            skipped += 1
            continue

        q_start = time.perf_counter()
        try:
            response = invoke_qa(query)
            answer = response.get("result", "")
            sources = response.get("source_documents", [])
            method = response.get("retrieval_method", "")

            if hasattr(answer, "content"):
                answer_text = answer.content
            else:
                answer_text = str(answer)

            df.at[idx, args.answer_column] = answer_text
            df.at[idx, args.sources_column] = _serialize_sources(sources)
            df.at[idx, args.method_column] = str(method)
            processed += 1
            print(
                f"[{idx + 1}/{end}] {row_id} ok "
                f"({time.perf_counter() - q_start:.2f}s)"
            )
        except Exception as exc:
            errors += 1
            df.at[idx, args.answer_column] = f"[ERROR] {exc}"
            df.at[idx, args.sources_column] = "[]"
            df.at[idx, args.method_column] = "error"
            print(f"[{idx + 1}/{end}] {row_id} error: {exc}")

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_excel(output_path, index=False, engine="openpyxl")

    elapsed = time.perf_counter() - started_at
    print(
        f"\nDone. processed={processed}, skipped={skipped}, errors={errors}, "
        f"elapsed_sec={elapsed:.2f}"
    )
    print(f"Saved: {output_path}")


if __name__ == "__main__":
    main()
