import argparse
import functools
import json
import os
import sys
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable


_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import pandas as pd
import requests
import tiktoken
from transformers import AutoTokenizer

try:
    from dotenv import load_dotenv
except ImportError:  # pragma: no cover - optional helper
    load_dotenv = None


DEFAULT_QWEN_TOKENIZER = os.environ.get(
    "LEGAL_RAG_QWEN_TOKENIZER",
    str(_REPO_ROOT / "ai_service" / "models" / "legal-lora-qwen2.5-1.5b"),
)
DEFAULT_LLAMA_TOKENIZER = os.environ.get(
    "LEGAL_RAG_LLAMA_TOKENIZER",
    "Xenova/llama3-tokenizer",
)
DEFAULT_DEEPSEEK_TOKENIZER = os.environ.get(
    "LEGAL_RAG_DEEPSEEK_TOKENIZER", "deepseek-ai/DeepSeek-V3"
)
DEFAULT_GPT_MODEL = os.environ.get("LEGAL_RAG_GPT_TOKENIZER_MODEL", "gpt-4o")
DEFAULT_GEMINI_MODEL = os.environ.get("LEGAL_RAG_GEMINI_MODEL", "gemini-2.5-pro")
DEFAULT_CLAUDE_MODEL = os.environ.get(
    "LEGAL_RAG_CLAUDE_MODEL", "claude-3-5-sonnet-20241022"
)
DEFAULT_OPENROUTER_GPT_MODEL = os.environ.get(
    "LEGAL_RAG_OPENROUTER_GPT_MODEL", "openai/gpt-5.5"
)
DEFAULT_OPENROUTER_GEMINI_MODEL = os.environ.get(
    "LEGAL_RAG_OPENROUTER_GEMINI_MODEL", "google/gemini-2.5-pro"
)
DEFAULT_OPENROUTER_CLAUDE_MODEL = os.environ.get(
    "LEGAL_RAG_OPENROUTER_CLAUDE_MODEL", "anthropic/claude-sonnet-4.5"
)
DEFAULT_OPENROUTER_QWEN_MODEL = os.environ.get(
    "LEGAL_RAG_OPENROUTER_QWEN_MODEL", "qwen/qwen3.5-27b"
)
DEFAULT_OPENROUTER_LLAMA_MODEL = os.environ.get(
    "LEGAL_RAG_OPENROUTER_LLAMA_MODEL", "meta-llama/llama-3.3-70b-instruct"
)
DEFAULT_OPENROUTER_DEEPSEEK_MODEL = os.environ.get(
    "LEGAL_RAG_OPENROUTER_DEEPSEEK_MODEL", "deepseek/deepseek-chat-v3.1"
)
DEFAULT_GROQ_LLAMA_MODEL = os.environ.get(
    "LEGAL_RAG_GROQ_LLAMA_MODEL", "llama-3.1-8b-instant"
)
DEFAULT_GROQ_QWEN_MODEL = os.environ.get(
    "LEGAL_RAG_GROQ_QWEN_MODEL", "qwen/qwen3-32b"
)
DEFAULT_GROQ_DEEPSEEK_MODEL = os.environ.get(
    "LEGAL_RAG_GROQ_DEEPSEEK_MODEL", "deepseek-r1-distill-qwen-32b"
)


@dataclass
class TokenizationResult:
    profile: str
    model_name: str
    token_count: int | None
    token_pieces: list[str] | None
    mode: str
    error: str | None = None


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Compare tokenization across model profiles and compute pairwise "
            "Manhattan/Jaccard distances."
        )
    )
    parser.add_argument(
        "--text",
        action="append",
        default=[],
        help="Inline text to evaluate. Can be passed multiple times.",
    )
    parser.add_argument(
        "--text-file",
        action="append",
        default=[],
        help="UTF-8 text file with one sample per file.",
    )
    parser.add_argument(
        "--xlsx",
        help="XLSX file containing texts to tokenize.",
    )
    parser.add_argument(
        "--column",
        default="query",
        help="Column name for text in --xlsx.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=0,
        help="Optional max number of rows to load from XLSX. 0 means all.",
    )
    parser.add_argument(
        "--profiles",
        default="qwen,gpt,gemini,claude",
        help="Comma-separated profiles to compare.",
    )
    parser.add_argument(
        "--qwen-tokenizer",
        default=DEFAULT_QWEN_TOKENIZER,
        help="HF tokenizer id/path for the qwen profile.",
    )
    parser.add_argument(
        "--llama-tokenizer",
        default=DEFAULT_LLAMA_TOKENIZER,
        help="HF tokenizer id/path for the llama profile.",
    )
    parser.add_argument(
        "--deepseek-tokenizer",
        default=DEFAULT_DEEPSEEK_TOKENIZER,
        help="HF tokenizer id/path for the deepseek profile.",
    )
    parser.add_argument(
        "--gpt-model",
        default=DEFAULT_GPT_MODEL,
        help="OpenAI model name for tiktoken encoding lookup.",
    )
    parser.add_argument(
        "--gemini-model",
        default=DEFAULT_GEMINI_MODEL,
        help="Gemini model name for count_tokens API.",
    )
    parser.add_argument(
        "--claude-model",
        default=DEFAULT_CLAUDE_MODEL,
        help="Claude model name for count_tokens API.",
    )
    parser.add_argument(
        "--allow-remote",
        action="store_true",
        help=(
            "Allow Gemini/Claude API token counting if SDK and API keys are available. "
            "Without this flag, remote-only profiles are reported as unavailable."
        ),
    )
    parser.add_argument(
        "--prefer-openrouter",
        action="store_true",
        help=(
            "Use OpenRouter prompt token usage for gpt/gemini/claude/qwen/llama/deepseek. "
            "This gives count-only comparison and requires OPENROUTER_API_KEY."
        ),
    )
    parser.add_argument(
        "--prefer-groq",
        action="store_true",
        help=(
            "Use Groq prompt token usage for llama/qwen/deepseek. "
            "This gives count-only comparison and requires GROQ_API_KEY."
        ),
    )
    parser.add_argument(
        "--openrouter-gpt-model",
        default=DEFAULT_OPENROUTER_GPT_MODEL,
        help="OpenRouter model id for the gpt family.",
    )
    parser.add_argument(
        "--openrouter-gemini-model",
        default=DEFAULT_OPENROUTER_GEMINI_MODEL,
        help="OpenRouter model id for the gemini family.",
    )
    parser.add_argument(
        "--openrouter-claude-model",
        default=DEFAULT_OPENROUTER_CLAUDE_MODEL,
        help="OpenRouter model id for the claude family.",
    )
    parser.add_argument(
        "--openrouter-qwen-model",
        default=DEFAULT_OPENROUTER_QWEN_MODEL,
        help="OpenRouter model id for the qwen family.",
    )
    parser.add_argument(
        "--openrouter-llama-model",
        default=DEFAULT_OPENROUTER_LLAMA_MODEL,
        help="OpenRouter model id for the llama family.",
    )
    parser.add_argument(
        "--openrouter-deepseek-model",
        default=DEFAULT_OPENROUTER_DEEPSEEK_MODEL,
        help="OpenRouter model id for the deepseek family.",
    )
    parser.add_argument(
        "--groq-llama-model",
        default=DEFAULT_GROQ_LLAMA_MODEL,
        help="Groq model id for the llama family.",
    )
    parser.add_argument(
        "--groq-qwen-model",
        default=DEFAULT_GROQ_QWEN_MODEL,
        help="Groq model id for the qwen family.",
    )
    parser.add_argument(
        "--groq-deepseek-model",
        default=DEFAULT_GROQ_DEEPSEEK_MODEL,
        help="Groq model id for the deepseek family.",
    )
    parser.add_argument(
        "--output-json",
        help="Optional path to save full results JSON.",
    )
    return parser.parse_args()


def _load_texts(args: argparse.Namespace) -> list[dict[str, Any]]:
    items: list[dict[str, Any]] = []

    for idx, text in enumerate(args.text, 1):
        if str(text).strip():
            items.append({"id": f"text:{idx}", "text": str(text)})

    for file_path in args.text_file:
        path = Path(file_path)
        content = path.read_text(encoding="utf-8")
        if content.strip():
            items.append({"id": str(path), "text": content})

    if args.xlsx:
        path = Path(args.xlsx)
        df = pd.read_excel(path)
        if args.column not in df.columns:
            raise ValueError(f"Column '{args.column}' not found in {path}")
        rows = df[[args.column]].fillna("")
        if args.limit > 0:
            rows = rows.head(args.limit)
        for idx, row in rows.iterrows():
            text = str(row[args.column]).strip()
            if text:
                items.append({"id": f"{path.name}#{idx}", "text": text})

    if not items:
        raise ValueError("No input texts provided. Use --text, --text-file, or --xlsx.")

    return items


def _decode_tiktoken_piece(encoding: tiktoken.Encoding, token_id: int) -> str:
    raw = encoding.decode_single_token_bytes(token_id)
    try:
        return raw.decode("utf-8")
    except UnicodeDecodeError:
        return f"<0x{raw.hex()}>"


def _resolve_local_hf_snapshot(model_name_or_path: str) -> str:
    candidate = Path(str(model_name_or_path or "").strip()).expanduser()
    if candidate.exists():
        return str(candidate)

    model_id = str(model_name_or_path or "").strip()
    if not model_id or "/" not in model_id:
        return model_id

    org, name = model_id.split("/", 1)
    cache_root = Path(
        os.environ.get("HF_HOME")
        or os.environ.get("HF_HUB_CACHE")
        or (Path.home() / ".cache" / "huggingface" / "hub")
    )
    model_cache_dir = cache_root / f"models--{org}--{name}"
    snapshots_dir = model_cache_dir / "snapshots"
    refs_main = model_cache_dir / "refs" / "main"

    if refs_main.exists():
        snapshot_name = refs_main.read_text(encoding="utf-8").strip()
        snapshot_dir = snapshots_dir / snapshot_name
        if snapshot_dir.exists():
            return str(snapshot_dir)

    if snapshots_dir.exists():
        snapshot_dirs = sorted(
            (path for path in snapshots_dir.iterdir() if path.is_dir()),
            key=lambda path: path.stat().st_mtime,
            reverse=True,
        )
        if snapshot_dirs:
            return str(snapshot_dirs[0])

    return model_id


@functools.lru_cache(maxsize=None)
def _hf_local_only() -> bool:
    return os.environ.get("LEGAL_RAG_HF_LOCAL_ONLY", "0") == "1" or os.environ.get(
        "HF_HUB_OFFLINE", "0"
    ) == "1"


@functools.lru_cache(maxsize=None)
def _load_hf_tokenizer(model_name: str) -> Any:
    resolved_model_name = _resolve_local_hf_snapshot(model_name)
    tokenizer = AutoTokenizer.from_pretrained(
        resolved_model_name,
        trust_remote_code=True,
        local_files_only=_hf_local_only() or Path(resolved_model_name).exists(),
        token=os.environ.get("HF_TOKEN"),
    )
    return resolved_model_name, tokenizer


@functools.lru_cache(maxsize=None)
def _get_tiktoken_encoding(model_name: str) -> tiktoken.Encoding:
    try:
        return tiktoken.encoding_for_model(model_name)
    except KeyError:
        return tiktoken.get_encoding("o200k_base")


def _tokenize_qwen(text: str, model_name: str) -> TokenizationResult:
    try:
        resolved_model_name, tokenizer = _load_hf_tokenizer(model_name)
        token_ids = tokenizer.encode(text, add_special_tokens=False)
        pieces = tokenizer.convert_ids_to_tokens(token_ids)
        return TokenizationResult(
            profile="qwen",
            model_name=resolved_model_name,
            token_count=len(token_ids),
            token_pieces=[str(piece) for piece in pieces],
            mode="exact_local_hf",
        )
    except Exception as exc:
        return TokenizationResult(
            profile="qwen",
            model_name=model_name,
            token_count=None,
            token_pieces=None,
            mode="error",
            error=str(exc),
        )


def _tokenize_local_hf(text: str, profile: str, model_name: str) -> TokenizationResult:
    try:
        resolved_model_name, tokenizer = _load_hf_tokenizer(model_name)
        token_ids = tokenizer.encode(text, add_special_tokens=False)
        pieces = tokenizer.convert_ids_to_tokens(token_ids)
        return TokenizationResult(
            profile=profile,
            model_name=resolved_model_name,
            token_count=len(token_ids),
            token_pieces=[str(piece) for piece in pieces],
            mode="exact_local_hf",
        )
    except Exception as exc:
        return TokenizationResult(
            profile=profile,
            model_name=model_name,
            token_count=None,
            token_pieces=None,
            mode="error",
            error=str(exc),
        )


def _tokenize_gpt(text: str, model_name: str) -> TokenizationResult:
    try:
        encoding = _get_tiktoken_encoding(model_name)
        token_ids = encoding.encode(text)
        pieces = [_decode_tiktoken_piece(encoding, token_id) for token_id in token_ids]
        return TokenizationResult(
            profile="gpt",
            model_name=model_name,
            token_count=len(token_ids),
            token_pieces=pieces,
            mode="exact_tiktoken",
        )
    except Exception as exc:
        return TokenizationResult(
            profile="gpt",
            model_name=model_name,
            token_count=None,
            token_pieces=None,
            mode="error",
            error=str(exc),
        )


def _tokenize_gemini(text: str, model_name: str, allow_remote: bool) -> TokenizationResult:
    if not allow_remote:
        return TokenizationResult(
            profile="gemini",
            model_name=model_name,
            token_count=None,
            token_pieces=None,
            mode="unavailable",
            error="remote counting disabled; pass --allow-remote",
        )
    api_key = os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
    if not api_key:
        return TokenizationResult(
            profile="gemini",
            model_name=model_name,
            token_count=None,
            token_pieces=None,
            mode="unavailable",
            error="missing GEMINI_API_KEY/GOOGLE_API_KEY",
        )
    try:
        import google.generativeai as genai

        genai.configure(api_key=api_key)
        model = genai.GenerativeModel(model_name)
        response = model.count_tokens(text)
        total_tokens = int(getattr(response, "total_tokens", 0))
        return TokenizationResult(
            profile="gemini",
            model_name=model_name,
            token_count=total_tokens,
            token_pieces=None,
            mode="api_count_only",
        )
    except Exception as exc:
        return TokenizationResult(
            profile="gemini",
            model_name=model_name,
            token_count=None,
            token_pieces=None,
            mode="error",
            error=str(exc),
        )


def _tokenize_claude(text: str, model_name: str, allow_remote: bool) -> TokenizationResult:
    if not allow_remote:
        return TokenizationResult(
            profile="claude",
            model_name=model_name,
            token_count=None,
            token_pieces=None,
            mode="unavailable",
            error="remote counting disabled; pass --allow-remote",
        )
    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        return TokenizationResult(
            profile="claude",
            model_name=model_name,
            token_count=None,
            token_pieces=None,
            mode="unavailable",
            error="missing ANTHROPIC_API_KEY",
        )
    try:
        import anthropic

        client = anthropic.Anthropic(api_key=api_key)
        if not hasattr(client, "beta") or not hasattr(client.beta, "messages"):
            raise RuntimeError("installed anthropic SDK does not expose beta.messages.count_tokens")
        response = client.beta.messages.count_tokens(
            model=model_name,
            messages=[{"role": "user", "content": text}],
        )
        total_tokens = int(getattr(response, "input_tokens", 0))
        return TokenizationResult(
            profile="claude",
            model_name=model_name,
            token_count=total_tokens,
            token_pieces=None,
            mode="api_count_only",
        )
    except Exception as exc:
        return TokenizationResult(
            profile="claude",
            model_name=model_name,
            token_count=None,
            token_pieces=None,
            mode="error",
            error=str(exc),
        )


def _tokenize_openrouter(
    text: str, profile: str, model_name: str, allow_remote: bool
) -> TokenizationResult:
    if not allow_remote:
        return TokenizationResult(
            profile=profile,
            model_name=model_name,
            token_count=None,
            token_pieces=None,
            mode="unavailable",
            error="remote counting disabled; pass --allow-remote",
        )
    api_key = os.environ.get("OPENROUTER_API_KEY")
    if not api_key:
        return TokenizationResult(
            profile=profile,
            model_name=model_name,
            token_count=None,
            token_pieces=None,
            mode="unavailable",
            error="missing OPENROUTER_API_KEY",
        )
    try:
        response = requests.post(
            "https://openrouter.ai/api/v1/chat/completions",
            headers={
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json",
            },
            json={
                "model": model_name,
                "messages": [{"role": "user", "content": text}],
                "max_tokens": 1,
                "temperature": 0,
            },
            timeout=60,
        )
        response.raise_for_status()
        payload = response.json()
        usage = payload.get("usage", {}) or {}
        prompt_tokens = usage.get("prompt_tokens")
        if prompt_tokens is None:
            raise RuntimeError(f"usage.prompt_tokens missing in response: {payload}")
        return TokenizationResult(
            profile=profile,
            model_name=model_name,
            token_count=int(prompt_tokens),
            token_pieces=None,
            mode="openrouter_count_only",
        )
    except Exception as exc:
        return TokenizationResult(
            profile=profile,
            model_name=model_name,
            token_count=None,
            token_pieces=None,
            mode="error",
            error=str(exc),
        )


def _tokenize_groq(
    text: str, profile: str, model_name: str, allow_remote: bool
) -> TokenizationResult:
    if not allow_remote:
        return TokenizationResult(
            profile=profile,
            model_name=model_name,
            token_count=None,
            token_pieces=None,
            mode="unavailable",
            error="remote counting disabled; pass --allow-remote",
        )
    api_key = os.environ.get("GROQ_API_KEY")
    if not api_key:
        return TokenizationResult(
            profile=profile,
            model_name=model_name,
            token_count=None,
            token_pieces=None,
            mode="unavailable",
            error="missing GROQ_API_KEY",
        )
    try:
        response = requests.post(
            "https://api.groq.com/openai/v1/chat/completions",
            headers={
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json",
            },
            json={
                "model": model_name,
                "messages": [{"role": "user", "content": text}],
                "max_tokens": 1,
                "temperature": 0,
            },
            timeout=60,
        )
        response.raise_for_status()
        payload = response.json()
        usage = payload.get("usage", {}) or {}
        prompt_tokens = usage.get("prompt_tokens")
        if prompt_tokens is None:
            raise RuntimeError(f"usage.prompt_tokens missing in response: {payload}")
        return TokenizationResult(
            profile=profile,
            model_name=model_name,
            token_count=int(prompt_tokens),
            token_pieces=None,
            mode="groq_count_only",
        )
    except Exception as exc:
        return TokenizationResult(
            profile=profile,
            model_name=model_name,
            token_count=None,
            token_pieces=None,
            mode="error",
            error=str(exc),
        )


def _build_profile_tokenizer(
    profile: str, args: argparse.Namespace
):
    profile = profile.strip().lower()
    if args.prefer_openrouter:
        openrouter_models = {
            "gpt": args.openrouter_gpt_model,
            "gemini": args.openrouter_gemini_model,
            "claude": args.openrouter_claude_model,
            "qwen": args.openrouter_qwen_model,
            "llama": args.openrouter_llama_model,
            "deepseek": args.openrouter_deepseek_model,
        }
        if profile in openrouter_models:
            return lambda text: _tokenize_openrouter(
                text, profile, openrouter_models[profile], args.allow_remote
            )
    if args.prefer_groq:
        groq_models = {
            "llama": args.groq_llama_model,
            "qwen": args.groq_qwen_model,
            "deepseek": args.groq_deepseek_model,
        }
        if profile in groq_models:
            return lambda text: _tokenize_groq(
                text, profile, groq_models[profile], args.allow_remote
            )
    if profile == "qwen":
        return lambda text: _tokenize_qwen(text, args.qwen_tokenizer)
    if profile == "gpt":
        return lambda text: _tokenize_gpt(text, args.gpt_model)
    if profile == "gemini":
        return lambda text: _tokenize_gemini(text, args.gemini_model, args.allow_remote)
    if profile == "claude":
        return lambda text: _tokenize_claude(text, args.claude_model, args.allow_remote)
    if profile == "llama":
        return lambda text: _tokenize_local_hf(text, "llama", args.llama_tokenizer)
    if profile == "deepseek":
        return lambda text: _tokenize_local_hf(
            text, "deepseek", args.deepseek_tokenizer
        )
    raise ValueError(f"Unsupported profile: {profile}")


def _counter_from_tokens(tokens: Iterable[str]) -> Counter[str]:
    return Counter(str(token) for token in tokens)


def _manhattan_distance(tokens_a: list[str], tokens_b: list[str]) -> int:
    counter_a = _counter_from_tokens(tokens_a)
    counter_b = _counter_from_tokens(tokens_b)
    keys = set(counter_a) | set(counter_b)
    return sum(abs(counter_a.get(key, 0) - counter_b.get(key, 0)) for key in keys)


def _jaccard_distance(tokens_a: list[str], tokens_b: list[str]) -> float:
    set_a = set(tokens_a)
    set_b = set(tokens_b)
    union = set_a | set_b
    if not union:
        return 0.0
    intersection = set_a & set_b
    return 1.0 - (len(intersection) / len(union))


def _pairwise_summary(
    profile_results: dict[str, list[TokenizationResult]]
) -> list[dict[str, Any]]:
    profile_names = list(profile_results.keys())
    summary: list[dict[str, Any]] = []
    for idx, left_name in enumerate(profile_names):
        left_runs = profile_results[left_name]
        for right_name in profile_names[idx + 1 :]:
            right_runs = profile_results[right_name]
            count_diff_sum = 0
            comparable_count_rows = 0
            comparable_piece_rows = 0
            manhattan_sum = 0
            jaccard_values: list[float] = []

            for left, right in zip(left_runs, right_runs, strict=True):
                if left.token_count is not None and right.token_count is not None:
                    comparable_count_rows += 1
                    count_diff_sum += abs(left.token_count - right.token_count)
                if left.token_pieces is not None and right.token_pieces is not None:
                    comparable_piece_rows += 1
                    manhattan_sum += _manhattan_distance(left.token_pieces, right.token_pieces)
                    jaccard_values.append(
                        _jaccard_distance(left.token_pieces, right.token_pieces)
                    )

            summary.append(
                {
                    "left": left_name,
                    "right": right_name,
                    "token_count_rows_compared": comparable_count_rows,
                    "token_count_abs_diff_sum": (
                        count_diff_sum if comparable_count_rows > 0 else None
                    ),
                    "piece_rows_compared": comparable_piece_rows,
                    "manhattan_distance_sum": (
                        manhattan_sum if comparable_piece_rows > 0 else None
                    ),
                    "jaccard_distance_avg": (
                        sum(jaccard_values) / len(jaccard_values)
                        if jaccard_values
                        else None
                    ),
                }
            )
    return summary


def main() -> None:
    if load_dotenv is not None:
        load_dotenv(_REPO_ROOT / ".env")
    args = _parse_args()
    texts = _load_texts(args)
    profiles = [item.strip().lower() for item in args.profiles.split(",") if item.strip()]
    tokenizers = {profile: _build_profile_tokenizer(profile, args) for profile in profiles}

    per_text: list[dict[str, Any]] = []
    profile_results: dict[str, list[TokenizationResult]] = {profile: [] for profile in profiles}

    for index, item in enumerate(texts, 1):
        row_payload: dict[str, Any] = {
            "id": item["id"],
            "char_count": len(item["text"]),
            "results": {},
        }
        for profile, tokenizer_fn in tokenizers.items():
            result = tokenizer_fn(item["text"])
            profile_results[profile].append(result)
            row_payload["results"][profile] = {
                "model_name": result.model_name,
                "mode": result.mode,
                "token_count": result.token_count,
                "error": result.error,
            }
        per_text.append(row_payload)
        print(f"[{index}/{len(texts)}] {item['id']} processed")

    summary = _pairwise_summary(profile_results)
    payload = {
        "inputs": len(texts),
        "profiles": profiles,
        "per_text": per_text,
        "pairwise_summary": summary,
    }

    if args.output_json:
        output_path = Path(args.output_json)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

    print(f"Loaded texts: {len(texts)}")
    for profile in profiles:
        first = profile_results[profile][0]
        print(
            f"- {profile}: model={first.model_name} mode={first.mode}"
            + (f" error={first.error}" if first.error else "")
        )

    print("\nPer-text token counts:")
    for row in per_text:
        counts = []
        for profile in profiles:
            info = row["results"][profile]
            counts.append(f"{profile}={info['token_count']}")
        print(f"- {row['id']}: " + ", ".join(counts))

    print("\nPairwise summary:")
    for item in summary:
        print(
            f"- {item['left']} vs {item['right']}: "
            f"count_abs_diff_sum={item['token_count_abs_diff_sum']}, "
            f"manhattan_sum={item['manhattan_distance_sum']}, "
            f"jaccard_avg={item['jaccard_distance_avg']}, "
            f"piece_rows={item['piece_rows_compared']}"
        )


if __name__ == "__main__":
    main()
