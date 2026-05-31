from __future__ import annotations

import argparse
import re
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from xml.etree import ElementTree

import tiktoken

try:
    import ujson as json_lib
except ImportError:  # pragma: no cover
    import json as json_lib


DEFAULT_XLSX = Path("tests/benchmarks/kz_benchmark_gold_final.filtered.xlsx")
XML_NS = {"x": "http://schemas.openxmlformats.org/spreadsheetml/2006/main"}


@dataclass(frozen=True)
class ModelPrice:
    label: str
    input_per_million_usd: float
    output_per_million_usd: float
    category: str


QUALITY_MODELS: list[ModelPrice] = [
    ModelPrice("DeepSeek V4 Pro", 1.74, 3.48, "High Quality / Value"),
    ModelPrice("GPT-5.4", 2.50, 15.00, "Production Standard"),
    ModelPrice("Claude Sonnet 4.6", 3.00, 15.00, "Production Standard"),
    ModelPrice("Gemini 3 Pro", 4.00, 18.00, "Extended Context"),
    ModelPrice("Claude Opus 4.7", 5.00, 25.00, "Expert"),
    ModelPrice("GPT-5.5", 5.00, 30.00, "Expert"),
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Precision budget for quality-tier models using question token lengths "
            "+ fixed RAG context and output assumptions."
        )
    )
    parser.add_argument("--xlsx", type=Path, default=DEFAULT_XLSX, help="Input XLSX benchmark.")
    parser.add_argument("--sheet-index", type=int, default=0, help="Zero-based worksheet index.")
    parser.add_argument("--question-column", default="query", help="Question text column name.")
    parser.add_argument("--rag-input-tokens", type=int, default=2500, help="Fixed RAG input tokens.")
    parser.add_argument("--output-tokens", type=int, default=500, help="Fixed output tokens.")
    parser.add_argument(
        "--tokenizer",
        default="cl100k_base",
        help="tiktoken encoding name for question token counting.",
    )
    parser.add_argument("--output-json", type=Path, help="Optional path for JSON report.")
    return parser.parse_args()


def col_ref_to_index(cell_ref: str) -> int:
    match = re.match(r"([A-Z]+)", cell_ref)
    if not match:
        return -1
    letters = match.group(1)
    value = 0
    for ch in letters:
        value = value * 26 + (ord(ch) - ord("A") + 1)
    return value - 1


def _resolve_sheet_path_from_workbook(
    archive: zipfile.ZipFile, sheet_index: int
) -> str:
    workbook_xml = ElementTree.fromstring(archive.read("xl/workbook.xml"))
    rels_xml = ElementTree.fromstring(archive.read("xl/_rels/workbook.xml.rels"))
    sheets = workbook_xml.findall("x:sheets/x:sheet", XML_NS)
    if sheet_index < 0 or sheet_index >= len(sheets):
        raise ValueError(f"sheet-index {sheet_index} is out of range (total sheets: {len(sheets)})")
    sheet = sheets[sheet_index]
    rel_id = sheet.attrib.get("{http://schemas.openxmlformats.org/officeDocument/2006/relationships}id")
    if not rel_id:
        raise ValueError("cannot resolve worksheet relationship id")
    rel_map: dict[str, str] = {}
    for rel in rels_xml.findall("{http://schemas.openxmlformats.org/package/2006/relationships}Relationship"):
        rid = rel.attrib.get("Id")
        target = rel.attrib.get("Target")
        if rid and target:
            rel_map[rid] = target
    if rel_id not in rel_map:
        raise ValueError(f"worksheet relationship '{rel_id}' not found")
    target = rel_map[rel_id]
    if target.startswith("/"):
        return target.lstrip("/")
    if target.startswith("xl/"):
        return target
    return f"xl/{target}"


def _load_shared_strings(archive: zipfile.ZipFile) -> list[str]:
    if "xl/sharedStrings.xml" not in archive.namelist():
        return []
    root = ElementTree.fromstring(archive.read("xl/sharedStrings.xml"))
    values: list[str] = []
    for si in root.findall("x:si", XML_NS):
        text_chunks: list[str] = []
        direct = si.find("x:t", XML_NS)
        if direct is not None and direct.text is not None:
            text_chunks.append(direct.text)
        for run in si.findall("x:r", XML_NS):
            run_text = run.find("x:t", XML_NS)
            if run_text is not None and run_text.text is not None:
                text_chunks.append(run_text.text)
        values.append("".join(text_chunks))
    return values


def _cell_value(cell: ElementTree.Element, shared_strings: list[str]) -> str:
    cell_type = cell.attrib.get("t")
    if cell_type == "inlineStr":
        inline = cell.find("x:is", XML_NS)
        if inline is not None:
            direct = inline.find("x:t", XML_NS)
            if direct is not None and direct.text is not None:
                return direct.text
            chunks: list[str] = []
            for run in inline.findall("x:r", XML_NS):
                run_text = run.find("x:t", XML_NS)
                if run_text is not None and run_text.text is not None:
                    chunks.append(run_text.text)
            return "".join(chunks)
        return ""
    value_node = cell.find("x:v", XML_NS)
    if value_node is None or value_node.text is None:
        return ""
    raw = value_node.text
    if cell_type == "s":
        idx = int(raw)
        if 0 <= idx < len(shared_strings):
            return shared_strings[idx]
        return ""
    return raw


def load_column_texts_from_xlsx(
    xlsx_path: Path, sheet_index: int, column_name: str
) -> list[str]:
    with zipfile.ZipFile(xlsx_path, "r") as archive:
        shared_strings = _load_shared_strings(archive)
        sheet_path = _resolve_sheet_path_from_workbook(archive, sheet_index)
        sheet_root = ElementTree.fromstring(archive.read(sheet_path))
        sheet_data = sheet_root.find("x:sheetData", XML_NS)
        if sheet_data is None:
            raise ValueError("sheetData section is missing")

        rows = sheet_data.findall("x:row", XML_NS)
        if not rows:
            raise ValueError("worksheet is empty")

        header_map: dict[int, str] = {}
        for cell in rows[0].findall("x:c", XML_NS):
            ref = cell.attrib.get("r", "")
            col_idx = col_ref_to_index(ref)
            if col_idx < 0:
                continue
            header_map[col_idx] = _cell_value(cell, shared_strings).strip()

        normalized_target = column_name.strip().casefold()
        aliases = {normalized_target}
        if normalized_target == "query":
            aliases.update({"question", "вопрос"})
        target_col_idx = None
        for col_idx, header in header_map.items():
            if header.strip().casefold() in aliases:
                target_col_idx = col_idx
                break
        if target_col_idx is None:
            raise ValueError(f"column '{column_name}' not found in sheet headers")

        texts: list[str] = []
        for row in rows[1:]:
            row_values: dict[int, str] = {}
            for cell in row.findall("x:c", XML_NS):
                ref = cell.attrib.get("r", "")
                col_idx = col_ref_to_index(ref)
                if col_idx < 0:
                    continue
                row_values[col_idx] = _cell_value(cell, shared_strings)
            value = str(row_values.get(target_col_idx, "")).strip()
            if value:
                texts.append(value)
        return texts


def compute_question_stats(texts: list[str], tokenizer_name: str) -> tuple[int, float]:
    encoding = tiktoken.get_encoding(tokenizer_name)
    token_counts = [len(encoding.encode(text)) for text in texts]
    total = sum(token_counts)
    avg = total / len(token_counts) if token_counts else 0.0
    return total, avg


def compute_model_rows(
    models: list[ModelPrice],
    rows_count: int,
    avg_question_tokens: float,
    total_input_tokens: int,
    total_output_tokens: int,
    rag_input_tokens: int,
    output_tokens: int,
) -> list[dict[str, Any]]:
    results: list[dict[str, Any]] = []
    for model in models:
        per_request = (
            ((avg_question_tokens + rag_input_tokens) / 1_000_000) * model.input_per_million_usd
            + (output_tokens / 1_000_000) * model.output_per_million_usd
        )
        dataset_total = (
            (total_input_tokens / 1_000_000) * model.input_per_million_usd
            + (total_output_tokens / 1_000_000) * model.output_per_million_usd
        )
        results.append(
            {
                "model": model.label,
                "category": model.category,
                "input_per_million_usd": model.input_per_million_usd,
                "output_per_million_usd": model.output_per_million_usd,
                "per_request_usd": per_request,
                "dataset_total_usd": dataset_total,
                "rows": rows_count,
            }
        )
    return results


def main() -> None:
    args = parse_args()
    texts = load_column_texts_from_xlsx(args.xlsx, args.sheet_index, args.question_column)
    rows_count = len(texts)
    total_question_tokens, avg_question_tokens = compute_question_stats(texts, args.tokenizer)
    total_input_tokens = total_question_tokens + (rows_count * args.rag_input_tokens)
    total_output_tokens = rows_count * args.output_tokens

    model_rows = compute_model_rows(
        models=QUALITY_MODELS,
        rows_count=rows_count,
        avg_question_tokens=avg_question_tokens,
        total_input_tokens=total_input_tokens,
        total_output_tokens=total_output_tokens,
        rag_input_tokens=args.rag_input_tokens,
        output_tokens=args.output_tokens,
    )
    model_rows.sort(key=lambda item: item["dataset_total_usd"])

    print(f"Dataset: {args.xlsx}")
    print(f"Rows: {rows_count}")
    print(f"Average question tokens ({args.tokenizer}): {avg_question_tokens:.2f}")
    print(f"Total question tokens: {total_question_tokens:,}")
    print(f"Total input (question + RAG): {total_input_tokens:,}")
    print(f"Total output: {total_output_tokens:,}")
    print("\nQuality-tier model budget:")
    for row in model_rows:
        print(
            f"- {row['model']}: per_request=${row['per_request_usd']:.8f}, "
            f"dataset_total=${row['dataset_total_usd']:.6f}, "
            f"category={row['category']}"
        )

    report: dict[str, Any] = {
        "dataset": str(args.xlsx),
        "rows": rows_count,
        "assumptions": {
            "tokenizer": args.tokenizer,
            "rag_input_tokens": args.rag_input_tokens,
            "output_tokens": args.output_tokens,
        },
        "token_totals": {
            "average_question_tokens": avg_question_tokens,
            "total_question_tokens": total_question_tokens,
            "total_input_tokens": total_input_tokens,
            "total_output_tokens": total_output_tokens,
        },
        "quality_models": model_rows,
    }
    if args.output_json:
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        args.output_json.write_text(
            json_lib.dumps(report, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )


if __name__ == "__main__":
    main()
