from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

try:
    import ujson as json_lib
except ImportError:  # pragma: no cover
    import json as json_lib


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MARKET_JSON = REPO_ROOT / "tests" / "benchmarks" / "llm_budget_estimates.market_2026.json"

GLOBAL_ADVANCED_MODELS: dict[str, str] = {
    "DeepSeek": "deepseek_v3_2",
    "Meta": "llama_4_maverick",
    "OpenAI": "gpt_5_4_pro",
    "Google": "gemini_3_pro",
    "Anthropic": "claude_4_6_opus",
}

KAZAKHSTAN_AI_SOLUTIONS: list[str] = [
    "Zakon AI",
    "BeeFree (AI Legal Assistant)",
    "SmartGov AI (legal module)",
    "Aisulu / Digital Assistant for Entrepreneurs",
    "Astana Hub local legal RAG systems (including niche LegalRAG-like startups)",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Calculate budget scenarios for a commercial-law shortlist: "
            "one flagship model per global vendor + 5 Kazakhstan AI solutions."
        )
    )
    parser.add_argument(
        "--market-json",
        type=Path,
        default=DEFAULT_MARKET_JSON,
        help="Path to JSON with market model prices and avg_projected_per_request_usd.",
    )
    parser.add_argument(
        "--question-counts",
        default="50,100",
        help="Comma-separated question counts (e.g., 50,100).",
    )
    parser.add_argument(
        "--output-json",
        type=Path,
        help="Optional path to save the computed shortlist budget JSON.",
    )
    return parser.parse_args()


def load_market_payload(path: Path) -> dict[str, Any]:
    data = json_lib.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError("market json root must be an object")
    return data


def parse_question_counts(raw: str) -> list[int]:
    values: list[int] = []
    for token in raw.split(","):
        token = token.strip()
        if not token:
            continue
        count = int(token)
        if count <= 0:
            raise ValueError("question counts must be positive integers")
        values.append(count)
    if not values:
        raise ValueError("at least one question count is required")
    return values


def index_market_models(payload: dict[str, Any]) -> dict[str, dict[str, Any]]:
    records = payload.get("market_models")
    if not isinstance(records, list):
        raise ValueError("market_models array is missing")
    indexed: dict[str, dict[str, Any]] = {}
    for item in records:
        if not isinstance(item, dict):
            continue
        model_id = str(item.get("model", "")).strip()
        if model_id:
            indexed[model_id] = item
    return indexed


def build_shortlist_rows(
    indexed_models: dict[str, dict[str, Any]], question_counts: list[int]
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for vendor, model_id in GLOBAL_ADVANCED_MODELS.items():
        item = indexed_models.get(model_id)
        if item is None:
            raise ValueError(f"model '{model_id}' not found in market_models")
        avg_per_request = float(item.get("avg_projected_per_request_usd", 0.0))
        scenario_costs = {
            str(count): round(avg_per_request * count, 6) for count in question_counts
        }
        rows.append(
            {
                "vendor": vendor,
                "model": item.get("label", model_id),
                "model_id": model_id,
                "avg_projected_per_request_usd": avg_per_request,
                "scenario_costs_usd": scenario_costs,
            }
        )
    return rows


def build_totals(rows: list[dict[str, Any]], question_counts: list[int]) -> dict[str, float]:
    totals: dict[str, float] = {}
    for count in question_counts:
        total_value = sum(float(row["scenario_costs_usd"][str(count)]) for row in rows)
        totals[str(count)] = round(total_value, 6)
    return totals


def print_report(rows: list[dict[str, Any]], totals: dict[str, float]) -> None:
    print("Commercial-law shortlist (global flagship models)")
    print("-" * 72)
    for row in rows:
        print(f"{row['vendor']:<10} | {row['model']}")
        print(f"  avg/request: ${row['avg_projected_per_request_usd']:.6f}")
        for key, value in row["scenario_costs_usd"].items():
            print(f"  {key:>3} questions: ${value:.6f}")
    print("-" * 72)
    for key, value in totals.items():
        print(f"Total if one run through all 5 global models for {key} questions: ${value:.6f}")
    print("\nKazakhstan AI solutions for integration-depth review:")
    for solution in KAZAKHSTAN_AI_SOLUTIONS:
        print(f"- {solution}")


def main() -> None:
    args = parse_args()
    question_counts = parse_question_counts(args.question_counts)
    payload = load_market_payload(args.market_json)
    indexed = index_market_models(payload)
    rows = build_shortlist_rows(indexed, question_counts)
    totals = build_totals(rows, question_counts)
    report = {
        "focus": "commercial_law_civil_code",
        "question_counts": question_counts,
        "global_shortlist": rows,
        "global_total_budget_usd": totals,
        "kazakhstan_ai_solutions": KAZAKHSTAN_AI_SOLUTIONS,
    }
    print_report(rows, totals)
    if args.output_json:
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        args.output_json.write_text(
            json_lib.dumps(report, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )


if __name__ == "__main__":
    main()
