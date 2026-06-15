from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from engine.utils.retrieval_quality_benchmark import run_retrieval_quality_benchmark


DEFAULT_BASELINE_PATH = Path("tests/benchmarks/retrieval_quality_baseline.json")
DEFAULT_QUERIES_PATH = Path("tests/benchmarks/test_queries.json")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Retrieval evaluation gate for release checks."
    )
    parser.add_argument(
        "--baseline",
        default=str(DEFAULT_BASELINE_PATH),
        help="Path to baseline contract JSON.",
    )
    parser.add_argument(
        "--candidate",
        default="",
        help="Path to an already generated benchmark JSON. If empty, run a live benchmark.",
    )
    parser.add_argument(
        "--queries",
        default=str(DEFAULT_QUERIES_PATH),
        help="Path to fixed benchmark questions JSON.",
    )
    parser.add_argument("--top-k", type=int, default=5, help="Top-k for retrieval eval.")
    parser.add_argument(
        "--output",
        default="benchmark_results/retrieval_quality_gate_candidate.json",
        help="Where to write the live candidate benchmark if --candidate is not provided.",
    )
    parser.add_argument(
        "--allow-missing-baseline",
        action="store_true",
        help="Do not fail if the baseline file is missing.",
    )
    return parser.parse_args()


def _load_json(path: str | Path) -> dict[str, Any]:
    with open(path, "r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        raise ValueError(f"{path} must contain a JSON object")
    return payload


def _metric_pass(candidate: float, minimum: float, higher_is_better: bool) -> bool:
    return candidate >= minimum if higher_is_better else candidate <= minimum


def _compare_summary(candidate: dict[str, Any], baseline: dict[str, Any]) -> list[str]:
    failures: list[str] = []
    minimum_summary = baseline.get("minimum_summary", {})
    summary = candidate.get("summary", {})
    metric_rules = {
        "strict_hit@k": True,
        "soft_hit@k": True,
        "mrr": True,
        "map_soft": True,
        "latency_sec_avg": False,
        "latency_sec_p95": False,
    }
    for metric_name, higher_is_better in metric_rules.items():
        if metric_name not in minimum_summary:
            continue
        if metric_name not in summary:
            failures.append(f"candidate summary missing metric: {metric_name}")
            continue
        if not _metric_pass(
            float(summary.get(metric_name, 0.0)),
            float(minimum_summary[metric_name]),
            higher_is_better,
        ):
            comparison = ">=" if higher_is_better else "<="
            failures.append(
                f"summary metric {metric_name}={summary.get(metric_name, 0.0):.4f} "
                f"does not satisfy baseline {comparison} {float(minimum_summary[metric_name]):.4f}"
            )
    return failures


def _row_map(payload: dict[str, Any]) -> dict[str, dict[str, Any]]:
    rows: dict[str, dict[str, Any]] = {}
    for row in payload.get("results", []):
        if not isinstance(row, dict):
            continue
        qid = str(row.get("id") or row.get("query_id") or row.get("row_index"))
        rows[qid] = row
    return rows


def _compare_rows(candidate: dict[str, Any], baseline: dict[str, Any]) -> list[str]:
    failures: list[str] = []
    baseline_rows = _row_map(baseline)
    candidate_rows = _row_map(candidate)
    common_ids = sorted(set(baseline_rows) & set(candidate_rows))
    for qid in common_ids:
        base_metrics = baseline_rows[qid].get("metrics", {})
        cand_metrics = candidate_rows[qid].get("metrics", {})
        for metric_name in ("strict_hit", "soft_hit", "mrr"):
            if metric_name not in base_metrics:
                continue
            if float(cand_metrics.get(metric_name, 0.0)) + 1e-9 < float(
                base_metrics.get(metric_name, 0.0)
            ):
                failures.append(
                    f"{qid}: {metric_name} regressed "
                    f"{float(base_metrics.get(metric_name, 0.0)):.4f} -> "
                    f"{float(cand_metrics.get(metric_name, 0.0)):.4f}"
                )
    return failures


def evaluate_gate(candidate: dict[str, Any], baseline: dict[str, Any]) -> list[str]:
    failures = []
    failures.extend(_compare_summary(candidate, baseline))
    if baseline.get("results"):
        failures.extend(_compare_rows(candidate, baseline))
    return failures


def main() -> None:
    args = _parse_args()
    baseline_path = Path(args.baseline)
    if not baseline_path.exists():
        if args.allow_missing_baseline:
            print(f"Baseline not found, skipping gate: {baseline_path}")
            return
        raise FileNotFoundError(
            f"Baseline contract not found: {baseline_path}. "
            f"Generate it from a known-good benchmark run first."
        )

    baseline = _load_json(baseline_path)
    if args.candidate:
        candidate = _load_json(args.candidate)
    else:
        candidate = run_retrieval_quality_benchmark(
            queries_path=args.queries,
            top_k=args.top_k,
            output_path=args.output,
            compare_to=None,
        )

    failures = evaluate_gate(candidate, baseline)
    if failures:
        print("Evaluation gate failed:")
        for failure in failures:
            print(f"- {failure}")
        raise SystemExit(1)

    print("Evaluation gate passed.")


if __name__ == "__main__":
    main()
