import argparse
import json
from pathlib import Path


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare two retrieval benchmark JSON files."
    )
    parser.add_argument("--baseline", required=True, help="Path to baseline JSON.")
    parser.add_argument("--candidate", required=True, help="Path to candidate JSON.")
    parser.add_argument(
        "--top",
        type=int,
        default=30,
        help="How many improved/regressed rows to print.",
    )
    return parser.parse_args()


def _load(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _row_map(payload: dict) -> dict[str, dict]:
    rows = {}
    for row in payload.get("results", []):
        qid = str(row.get("query_id") or row.get("id") or row.get("row_index"))
        rows[qid] = row
    return rows


def _score(row: dict) -> tuple[float, float, float]:
    metrics = row.get("metrics", {})
    return (
        float(metrics.get("strict_hit", 0.0)),
        float(metrics.get("soft_hit", 0.0)),
        float(metrics.get("strict_mrr", 0.0)),
    )


def _format_row(qid: str, base: dict, cand: dict) -> str:
    b = base.get("metrics", {})
    c = cand.get("metrics", {})
    return (
        f"{qid} | "
        f"strict {b.get('strict_hit', 0):.0f}->{c.get('strict_hit', 0):.0f} | "
        f"soft {b.get('soft_hit', 0):.0f}->{c.get('soft_hit', 0):.0f} | "
        f"mrr {b.get('strict_mrr', 0):.3f}->{c.get('strict_mrr', 0):.3f} | "
        f"base={base.get('retrieved_topk', [])[:5]} | "
        f"cand={cand.get('retrieved_topk', [])[:5]}"
    )


def main() -> None:
    args = _parse_args()
    baseline = _load(args.baseline)
    candidate = _load(args.candidate)

    base_rows = _row_map(baseline)
    cand_rows = _row_map(candidate)
    common_ids = sorted(set(base_rows) & set(cand_rows))

    improved: list[tuple[float, str]] = []
    regressed: list[tuple[float, str]] = []
    strict_to_soft_miss: list[str] = []

    for qid in common_ids:
        b = base_rows[qid]
        c = cand_rows[qid]
        b_strict, b_soft, b_mrr = _score(b)
        c_strict, c_soft, c_mrr = _score(c)
        delta = (c_strict - b_strict) * 10 + (c_soft - b_soft) * 3 + (c_mrr - b_mrr)

        if delta > 0:
            improved.append((delta, qid))
        elif delta < 0:
            regressed.append((delta, qid))

        if c_soft == 1.0 and c_strict == 0.0:
            strict_to_soft_miss.append(qid)

    improved.sort(reverse=True)
    regressed.sort()

    print("Baseline summary:")
    print(json.dumps(baseline.get("summary", {}), ensure_ascii=False, indent=2))
    print("\nCandidate summary:")
    print(json.dumps(candidate.get("summary", {}), ensure_ascii=False, indent=2))

    print(f"\nImproved queries: {len(improved)}")
    for _, qid in improved[: args.top]:
        print(_format_row(qid, base_rows[qid], cand_rows[qid]))

    print(f"\nRegressed queries: {len(regressed)}")
    for _, qid in regressed[: args.top]:
        print(_format_row(qid, base_rows[qid], cand_rows[qid]))

    print(f"\nCandidate soft-hit but not strict-hit: {len(strict_to_soft_miss)}")
    for qid in strict_to_soft_miss[: args.top]:
        row = cand_rows[qid]
        print(
            f"{qid} | gold={row.get('gold_citations', [])[:5]} | "
            f"retrieved={row.get('retrieved_topk', [])[:5]}"
        )


if __name__ == "__main__":
    main()
