from __future__ import annotations

import argparse
import csv
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


XML_NS = {"x": "http://schemas.openxmlformats.org/spreadsheetml/2006/main"}
DEFAULT_INPUT = Path("tests/benchmarks/kz_benchmark_gold_final.filtered.xlsx")
DEFAULT_OUTPUT_JSON = Path("tests/benchmarks/kz_benchmark_commercial_quality100.json")
DEFAULT_OUTPUT_CSV = Path("tests/benchmarks/kz_benchmark_commercial_quality100.csv")

COMMERCIAL_POSITIVE_MARKERS = [
    "гражданского кодекса",
    "предпринимательского кодекса",
    "о товариществах с ограниченной и дополнительной ответственностью",
    "о реабилитации и банкротстве",
    "о банках и банковской деятельности",
    "о рынке ценных бумаг",
    "о государственной регистрации юридических лиц",
    "договор",
    "обязатель",
    "неустойк",
    "убытк",
    "поставка",
    "подряд",
    "аренд",
    "купл",
    "продаж",
    "кредит",
    "займ",
    "поручитель",
    "гарант",
    "тоо",
    "ип ",
    "дивиденд",
    "корпоратив",
    "дол",
]

COMMERCIAL_NEGATIVE_MARKERS = [
    "уголов",
    "коап",
    "административ",
    "трудов",
    "алимент",
    "браке",
    "семь",
    "родительск",
    "жилищн",
    "пенси",
    "миграц",
    "паспорт",
    "воинск",
    "пдд",
]


@dataclass(frozen=True)
class QuestionRow:
    query_id: str
    query: str
    gold_citations: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Build a commercial-law benchmark with top 100 quality questions "
            "from kz_benchmark_gold_final.filtered.xlsx and compute token stats."
        )
    )
    parser.add_argument("--input-xlsx", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--sheet-index", type=int, default=0)
    parser.add_argument("--size", type=int, default=100, help="Number of rows to keep.")
    parser.add_argument("--question-column", default="query")
    parser.add_argument("--id-column", default="query_id")
    parser.add_argument("--citations-column", default="gold_citations")
    parser.add_argument("--tokenizer", default="cl100k_base")
    parser.add_argument("--rag-input-tokens", type=int, default=2500)
    parser.add_argument("--output-tokens", type=int, default=500)
    parser.add_argument("--output-json", type=Path, default=DEFAULT_OUTPUT_JSON)
    parser.add_argument("--output-csv", type=Path, default=DEFAULT_OUTPUT_CSV)
    return parser.parse_args()


def col_ref_to_index(cell_ref: str) -> int:
    match = re.match(r"([A-Z]+)", cell_ref)
    if not match:
        return -1
    value = 0
    for ch in match.group(1):
        value = value * 26 + (ord(ch) - ord("A") + 1)
    return value - 1


def _resolve_sheet_path_from_workbook(archive: zipfile.ZipFile, sheet_index: int) -> str:
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
    target = rel_map.get(rel_id)
    if not target:
        raise ValueError(f"worksheet relationship '{rel_id}' not found")
    if target.startswith("/"):
        return target.lstrip("/")
    if target.startswith("xl/"):
        return target
    return f"xl/{target}"


def _cell_value(cell: ElementTree.Element) -> str:
    cell_type = cell.attrib.get("t")
    if cell_type == "inlineStr":
        inline = cell.find("x:is", XML_NS)
        if inline is None:
            return ""
        direct = inline.find("x:t", XML_NS)
        if direct is not None and direct.text is not None:
            return direct.text
        chunks: list[str] = []
        for run in inline.findall("x:r", XML_NS):
            run_text = run.find("x:t", XML_NS)
            if run_text is not None and run_text.text is not None:
                chunks.append(run_text.text)
        return "".join(chunks)
    value_node = cell.find("x:v", XML_NS)
    return "" if value_node is None or value_node.text is None else value_node.text


def load_rows_from_xlsx(
    xlsx_path: Path,
    sheet_index: int,
    id_column: str,
    question_column: str,
    citations_column: str,
) -> list[QuestionRow]:
    with zipfile.ZipFile(xlsx_path, "r") as archive:
        sheet_path = _resolve_sheet_path_from_workbook(archive, sheet_index)
        sheet_root = ElementTree.fromstring(archive.read(sheet_path))
        sheet_data = sheet_root.find("x:sheetData", XML_NS)
        if sheet_data is None:
            raise ValueError("sheetData section is missing")
        rows = sheet_data.findall("x:row", XML_NS)
        if not rows:
            return []

        headers: dict[int, str] = {}
        for cell in rows[0].findall("x:c", XML_NS):
            idx = col_ref_to_index(cell.attrib.get("r", ""))
            if idx >= 0:
                headers[idx] = _cell_value(cell).strip()

        target_indices: dict[str, int] = {}
        for idx, header_name in headers.items():
            normalized = header_name.strip().casefold()
            if normalized == id_column.strip().casefold():
                target_indices["id"] = idx
            elif normalized == question_column.strip().casefold():
                target_indices["question"] = idx
            elif normalized == citations_column.strip().casefold():
                target_indices["citations"] = idx
        if set(target_indices.keys()) != {"id", "question", "citations"}:
            raise ValueError("Required columns not found: query_id/query/gold_citations")

        out_rows: list[QuestionRow] = []
        for row in rows[1:]:
            values: dict[int, str] = {}
            for cell in row.findall("x:c", XML_NS):
                idx = col_ref_to_index(cell.attrib.get("r", ""))
                if idx >= 0:
                    values[idx] = _cell_value(cell)
            query_id = str(values.get(target_indices["id"], "")).strip()
            query = str(values.get(target_indices["question"], "")).strip()
            citations = str(values.get(target_indices["citations"], "")).strip()
            if not query_id or not query:
                continue
            out_rows.append(QuestionRow(query_id=query_id, query=query, gold_citations=citations))
        return out_rows


def marker_hits(text: str, markers: list[str]) -> int:
    lowered = text.lower()
    return sum(1 for marker in markers if marker in lowered)


def citations_count(citations: str) -> int:
    return len([chunk for chunk in citations.split(";") if chunk.strip()])


def score_quality(row: QuestionRow) -> tuple[int, int, int, int]:
    combined = f"{row.query}\n{row.gold_citations}".lower()
    pos = marker_hits(combined, COMMERCIAL_POSITIVE_MARKERS)
    neg = marker_hits(combined, COMMERCIAL_NEGATIVE_MARKERS)
    cites = citations_count(row.gold_citations)
    q_len = len(row.query)

    # Hard relevance gate for commercial/civil orientation.
    if pos == 0 or pos <= neg:
        return (-999, pos, neg, cites)

    score = 0
    score += pos * 4
    score -= neg * 5
    score += min(cites, 6) * 2

    if "гк рк" in combined or "гражданского кодекса" in combined:
        score += 5
    if "договор" in combined or "обязатель" in combined:
        score += 3
    if 120 <= q_len <= 1200:
        score += 3
    elif 60 <= q_len < 120:
        score += 1
    else:
        score -= 2
    if re.search(r"\d", row.query):
        score += 1
    return (score, pos, neg, cites)


def build_selected_rows(rows: list[QuestionRow], size: int) -> list[dict[str, Any]]:
    scored: list[dict[str, Any]] = []
    for row in rows:
        score, pos, neg, cites = score_quality(row)
        if score < 0:
            continue
        scored.append(
            {
                "query_id": row.query_id,
                "query": row.query,
                "gold_citations": row.gold_citations,
                "quality_score": score,
                "positive_markers": pos,
                "negative_markers": neg,
                "citations_count": cites,
                "question_char_count": len(row.query),
            }
        )
    scored.sort(
        key=lambda item: (
            -int(item["quality_score"]),
            -int(item["citations_count"]),
            -int(item["question_char_count"]),
            str(item["query_id"]),
        )
    )
    return scored[:size]


def attach_token_stats(
    selected: list[dict[str, Any]],
    tokenizer_name: str,
    rag_input_tokens: int,
    output_tokens: int,
) -> dict[str, Any]:
    encoding = tiktoken.get_encoding(tokenizer_name)
    question_token_counts: list[int] = []
    for item in selected:
        token_count = len(encoding.encode(str(item["query"])))
        item["question_tokens"] = token_count
        question_token_counts.append(token_count)

    rows_count = len(selected)
    total_question_tokens = sum(question_token_counts)
    avg_question_tokens = total_question_tokens / rows_count if rows_count else 0.0
    total_input_tokens = total_question_tokens + rows_count * rag_input_tokens
    total_output_tokens = rows_count * output_tokens

    return {
        "rows": rows_count,
        "tokenizer": tokenizer_name,
        "average_question_tokens": avg_question_tokens,
        "total_question_tokens": total_question_tokens,
        "total_input_tokens": total_input_tokens,
        "total_output_tokens": total_output_tokens,
        "rag_input_tokens_per_request": rag_input_tokens,
        "output_tokens_per_request": output_tokens,
    }


def write_csv(path: Path, selected: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "query_id",
        "query",
        "gold_citations",
        "quality_score",
        "positive_markers",
        "negative_markers",
        "citations_count",
        "question_char_count",
        "question_tokens",
    ]
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for item in selected:
            writer.writerow({key: item.get(key) for key in fieldnames})


def main() -> None:
    args = parse_args()
    source_rows = load_rows_from_xlsx(
        xlsx_path=args.input_xlsx,
        sheet_index=args.sheet_index,
        id_column=args.id_column,
        question_column=args.question_column,
        citations_column=args.citations_column,
    )
    selected = build_selected_rows(source_rows, args.size)
    token_summary = attach_token_stats(
        selected=selected,
        tokenizer_name=args.tokenizer,
        rag_input_tokens=args.rag_input_tokens,
        output_tokens=args.output_tokens,
    )

    payload: dict[str, Any] = {
        "source_file": str(args.input_xlsx),
        "selection_size_requested": args.size,
        "selection_size_actual": len(selected),
        "selection_focus": "commercial_sector_civil_law",
        "token_summary": token_summary,
        "questions": selected,
    }

    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(
        json_lib.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    write_csv(args.output_csv, selected)

    print(f"Source rows: {len(source_rows)}")
    print(f"Selected rows: {len(selected)}")
    print(f"Saved JSON: {args.output_json}")
    print(f"Saved CSV: {args.output_csv}")
    print(
        "Token summary: "
        f"avg_question={token_summary['average_question_tokens']:.2f}, "
        f"total_question={token_summary['total_question_tokens']}, "
        f"total_input={token_summary['total_input_tokens']}, "
        f"total_output={token_summary['total_output_tokens']}"
    )


if __name__ == "__main__":
    main()
