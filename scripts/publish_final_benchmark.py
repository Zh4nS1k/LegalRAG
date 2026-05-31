from __future__ import annotations

import argparse
import shutil
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Publish the current reviewed benchmark as the canonical final file.")
    parser.add_argument(
        "--source",
        default="tests/benchmarks/Полный бенчмарк-3.reviewed8.xlsx",
        help="Reviewed source workbook.",
    )
    parser.add_argument(
        "--target",
        default="tests/benchmarks/kz_benchmark_gold_final.xlsx",
        help="Canonical published workbook path.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    source = Path(args.source)
    target = Path(args.target)

    if not source.exists():
        raise FileNotFoundError(f"Source workbook not found: {source}")

    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, target)
    print(f"Published final benchmark: {target}")
    print(f"From source: {source}")


if __name__ == "__main__":
    main()
