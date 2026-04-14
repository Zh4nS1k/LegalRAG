import argparse
import json
import os
import re
import time
import warnings
from datetime import datetime, UTC
from pathlib import Path
from typing import Any

import pandas as pd
from pinecone import Pinecone

from ai_service.core import config
from ai_service.processing.code_names import get_code_name
from ai_service.retrieval.domain import detect_domain, domain_matches_code


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Clean benchmark retriever: Pinecone + BM25 + reranker against gold citations."
    )
    parser.add_argument("--xlsx", default="tests/benchmarks/642_questions_with_citations.xlsx")
    parser.add_argument("--limit", type=int, default=100)
    parser.add_argument("--start-from", type=int, default=0)
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--pinecone-k", type=int, default=20)
    parser.add_argument("--bm25-k", type=int, default=20)
    parser.add_argument("--candidate-k", type=int, default=30)
    parser.add_argument("--multi-query-limit", type=int, default=4)
    parser.add_argument("--bm25-weight", type=float, default=0.40)
    parser.add_argument("--vector-weight", type=float, default=1.0)
    parser.add_argument("--rerank-weight", type=float, default=0.50)
    parser.add_argument("--diversity-penalty", type=float, default=0.06)
    parser.add_argument("--no-rerank", action="store_true")
    parser.add_argument("--save-every", type=int, default=25)
    parser.add_argument(
        "--reranker-model",
        default=os.environ.get("LEGAL_RAG_RERANKER_FALLBACK_MODEL", "BAAI/bge-reranker-base"),
    )
    parser.add_argument(
        "--output",
        default="benchmark_results/gold_citations_clean_hybrid_rerank_benchmark.json",
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
    parts = [part.strip() for part in text.split(";") if part.strip()]
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
    text = re.sub(r"\bреспублики казахстан\b", "рк", text)
    text = re.sub(r"[\"']", "", text)
    text = re.sub(r"\s+", " ", text)

    alias_groups = [
        (
            "гражданский кодекс рк",
            (
                "гк рк",
                "гк",
                "гражданский кодекс",
                "гражданский кодекс рк (общая часть)",
                "гражданский кодекс рк (особенная часть)",
            ),
        ),
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
            "кодекс об административных процедурах рк",
            ("аппк рк", "аппк", "кодекс об административных процедурах"),
        ),
        ("налоговый кодекс рк", ("налоговый кодекс", "нк рк", "нк")),
        ("трудовой кодекс рк", ("трудовой кодекс", "тк рк", "тк")),
        (
            "закон о защите прав потребителей рк",
            ("закон о защите прав потребителей", "о защите прав потребителей", "зпп"),
        ),
        (
            "закон об адвокатской деятельности и юридической помощи",
            ("об адвокатской деятельности и юридической помощи",),
        ),
        (
            "закон о валютном регулировании и валютном контроле",
            ("о валютном регулировании и валютном контроле",),
        ),
        (
            "закон о товариществах с ограниченной и дополнительной ответственностью рк",
            ("тоо", "товариществах с ограниченной и дополнительной ответственностью"),
        ),
        ("закон о цифровых активах", ("о цифровых активах",)),
        ("закон о дорожном движении рк", ("о дорожном движении",)),
        (
            "закон о воинской службе и статусе военнослужащих рк",
            ("о воинской службе", "статусе военнослужащих"),
        ),
        (
            "закон о восстановлении платежеспособности и банкротстве граждан рк",
            ("о восстановлении платежеспособности и банкротстве граждан", "банкротстве граждан"),
        ),
    ]
    for canonical, aliases in alias_groups:
        if canonical in text or any(alias in text for alias in aliases):
            return canonical

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
    elif any(token in lower_raw for token in ("кодекс", "гк", "ук", "гпк", "упк", "коап", "аппк", "тк", "нк")):
        code = raw
    return article, _normalize_code(code)


def _pair_to_str(article: str, code: str) -> str:
    return f"{_normalize_article(article)}::{_normalize_code(code)}"


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


def _avg(results: list[dict[str, Any]], metric_name: str) -> float:
    values = [float(item["metrics"][metric_name]) for item in results]
    return sum(values) / len(values) if values else 0.0


def _get_embedder():
    config.configure_hf_hub()
    from langchain_huggingface import HuggingFaceEmbeddings

    model_name = config.EMBEDDING_MODEL
    if "/" in model_name:
        org, repo = model_name.split("/", 1)
        snapshot_root = Path(config.HF_CACHE_DIR) / f"models--{org}--{repo}" / "snapshots"
        snapshots = sorted(snapshot_root.glob("*"))
        if snapshots:
            model_name = str(snapshots[-1])
    local_only = bool(config.HF_LOCAL_ONLY)
    model_kwargs = {"local_files_only": True, "trust_remote_code": True, "device": "cpu"}
    try:
        model = HuggingFaceEmbeddings(
            model_name=model_name,
            encode_kwargs={"normalize_embeddings": True},
            model_kwargs=model_kwargs,
            show_progress=False,
        )
    except Exception:
        if local_only or os.environ.get("HF_HUB_OFFLINE", "0") == "1":
            raise RuntimeError(
                "Embedding model is not available in local cache. "
                "Disable offline mode or pre-download model into HF_HOME cache."
            )
        model_kwargs["local_files_only"] = False
        model = HuggingFaceEmbeddings(
            model_name=config.EMBEDDING_MODEL,
            encode_kwargs={"normalize_embeddings": True},
            model_kwargs=model_kwargs,
            show_progress=False,
        )

    def embed_query(text: str) -> list[float]:
        return model.embed_query("query: " + text)

    return embed_query


def _extract_target_article_numbers(query: str) -> set[str]:
    return set(re.findall(r"\b(\d{1,4}(?:-\d+)?)\b", str(query or "").lower()))


def _split_sentences(text: str) -> list[str]:
    compact = re.sub(r"\s+", " ", str(text or "")).strip()
    if not compact:
        return []
    parts = re.split(r"(?<=[\.\?!;])\s+", compact)
    return [part.strip() for part in parts if part.strip()]


def _build_subqueries(query: str) -> list[str]:
    compact = re.sub(r"\s+", " ", str(query or "")).strip()
    if not compact:
        return []
    subqueries = [compact]
    sentences = _split_sentences(compact)
    article_mentions = len(_extract_target_article_numbers(compact))
    if len(compact) >= 220 or len(sentences) >= 4 or article_mentions >= 2:
        legal_markers = ("закон", "кодекс", "гк", "ук", "гпк", "упк", "коап", "аппк", "ст.", "статья", "бап")
        focused = [s for s in sentences if any(marker in s.lower() for marker in legal_markers)]
        subqueries.extend(focused[:2] if focused else sentences[:1])
    deduped = list(dict.fromkeys(item for item in subqueries if item))
    return deduped


def _extract_target_codes(query: str) -> list[str]:
    query_norm = _normalize_code(query)
    candidates = [
        "гражданский кодекс рк",
        "уголовный кодекс рк",
        "гражданский процессуальный кодекс рк",
        "уголовно-процессуальный кодекс рк",
        "кодекс об административных правонарушениях рк",
        "кодекс об административных процедурах рк",
        "налоговый кодекс рк",
        "трудовой кодекс рк",
        "закон о защите прав потребителей рк",
        "закон о валютном регулировании и валютном контроле",
        "закон о воинской службе и статусе военнослужащих рк",
        "закон о товариществах с ограниченной и дополнительной ответственностью рк",
    ]
    found: list[str] = []
    for candidate in candidates:
        if candidate in query_norm and candidate not in found:
            found.append(candidate)
    return found


def _looks_like_raw_code_name(value: str) -> bool:
    text = str(value or "").strip().lower()
    return bool(text) and bool(re.fullmatch(r"[a-z0-9_]+", text)) and "_" in text


def _is_noisy_candidate(meta: dict[str, Any], text: str) -> bool:
    article_number = _normalize_article(str(meta.get("article_number", "")))
    code_ru = str(meta.get("code_ru", "") or "").strip()
    clause_level = str(meta.get("clause_level", "") or "").strip().lower()
    article_title = str(meta.get("article_title", "") or "").strip()
    content_head = str(text or "")[:450].lower()
    if not article_number and _looks_like_raw_code_name(code_ru):
        return True
    if clause_level == "article" and not article_number and not article_title:
        return True
    noisy_markers = ("мазмұны", "содержание", "зқаи-ның ескертпесі", "қолданушылар назарына")
    return any(marker in content_head for marker in noisy_markers)


def _minmax_normalize(values: list[float]) -> list[float]:
    if not values:
        return []
    lo = min(values)
    hi = max(values)
    if hi - lo <= 1e-9:
        return [0.5 for _ in values]
    return [(value - lo) / (hi - lo) for value in values]


def _legal_score(
    *,
    query: str,
    meta: dict[str, Any],
    text: str,
    base_score: float,
    target_codes: list[str],
    target_articles: set[str],
) -> float:
    score = float(base_score)
    doc_code = _canonicalize_code(meta.get("code_ru", ""), meta.get("source", ""))
    doc_article = _normalize_article(meta.get("article_number", ""))
    query_lower = str(query or "").lower()
    domain = detect_domain(query)

    if _looks_like_raw_code_name(str(meta.get("code_ru", ""))):
        score -= 0.25
    if not doc_article:
        score -= 0.20
    if _is_noisy_candidate(meta, text):
        score -= 0.45

    if target_codes:
        if doc_code in set(target_codes):
            score += 0.25
        else:
            score -= 0.15
    if target_articles and doc_article in target_articles:
        score += 0.30
        if doc_article and doc_article in query_lower:
            score += 0.20
    if domain:
        if domain_matches_code(domain, doc_code):
            score += 0.15
        else:
            score -= 0.20
    return score


def _canonicalize_code(code: str, source: str | None = None) -> str:
    normalized = _normalize_code(code)
    if normalized and "_" not in normalized:
        return normalized
    if source:
        code_ru, _ = get_code_name(source)
        return _normalize_code(code_ru)
    return normalized


def _candidate_text(meta: dict[str, Any], fallback_text: str = "") -> str:
    code = _canonicalize_code(str(meta.get("code_ru", "") or ""), str(meta.get("source", "") or ""))
    article = str(meta.get("article_number", "") or "").strip()
    title = str(meta.get("article_title", "") or "").strip()
    chapter = str(meta.get("chapter_title", "") or "").strip()
    path = str(meta.get("path", "") or "").strip()
    text = (
        str(meta.get("text", "") or "")
        or str(meta.get("page_content", "") or "")
        or str(meta.get("chunk_text", "") or "")
        or fallback_text
    )
    head = " | ".join(part for part in (code, f"ст. {article}" if article else "", title, chapter, path) if part)
    return (head + "\n" + text[:1500]).strip()


def _load_reranker(model_name: str):
    try:
        from sentence_transformers import CrossEncoder

        warnings.filterwarnings(
            "ignore",
            message="The Transformer `cache_dir` argument is deprecated",
            category=FutureWarning,
        )
        model = CrossEncoder(model_name)
        return "cross_encoder", model
    except Exception:
        from FlagEmbedding import FlagReranker

        model = FlagReranker(model_name, use_fp16=True)
        return "flag_embedding", model


def _rerank_scores(backend: str, model: Any, query: str, texts: list[str]) -> list[float]:
    pairs = [[query, text] for text in texts]
    if backend == "cross_encoder":
        scores = model.predict(pairs)
    else:
        scores = model.compute_score(pairs)
    if isinstance(scores, float):
        return [float(scores)]
    return [float(score) for score in scores]


def _build_payload(
    *,
    args: argparse.Namespace,
    xlsx_path: Path,
    reranker_backend: str,
    results: list[dict[str, Any]],
) -> dict[str, Any]:
    return {
        "xlsx": str(xlsx_path),
        "evaluated_questions": len(results),
        "top_k": args.top_k,
        "pinecone_k": args.pinecone_k,
        "bm25_k": args.bm25_k,
        "candidate_k": args.candidate_k,
        "reranker_model": args.reranker_model,
        "reranker_backend": reranker_backend,
        "generated_at": datetime.now(UTC).isoformat(),
        "summary": {
            "strict_hit": _avg(results, "strict_hit"),
            "soft_hit": _avg(results, "soft_hit"),
            "strict_precision": _avg(results, "strict_precision"),
            "strict_recall": _avg(results, "strict_recall"),
            "soft_precision": _avg(results, "soft_precision"),
            "soft_recall": _avg(results, "soft_recall"),
            "strict_mrr": _avg(results, "strict_mrr"),
            "soft_mrr": _avg(results, "soft_mrr"),
            "avg_elapsed_sec": sum(item["elapsed_sec"] for item in results) / len(results)
            if results
            else 0.0,
        },
        "results": results,
    }


def _save_payload(output_path: Path, payload: dict[str, Any]) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def main() -> None:
    if "TRANSFORMERS_CACHE" in os.environ and "HF_HOME" not in os.environ:
        os.environ["HF_HOME"] = os.environ["TRANSFORMERS_CACHE"]
    os.environ.pop("TRANSFORMERS_CACHE", None)
    args = _parse_args()
    xlsx_path = Path(args.xlsx)
    if not xlsx_path.exists():
        raise FileNotFoundError(f"Benchmark XLSX not found: {xlsx_path}")

    api_key = os.environ.get("PINECONE_API_KEY") or config.PINECONE_API_KEY
    index_name = os.environ.get("PINECONE_INDEX_NAME") or config.PINECONE_INDEX_NAME
    namespace = os.environ.get("PINECONE_NAMESPACE") or config.PINECONE_NAMESPACE or "default"
    if not api_key or not index_name:
        raise RuntimeError("Pinecone credentials are not set")

    from ai_service.processing import prepare_data
    from langchain_community.retrievers import BM25Retriever

    bm25 = BM25Retriever.from_documents(prepare_data.chunks, k=args.bm25_k)
    embed_query = _get_embedder()
    reranker_backend, reranker = ("disabled", None) if args.no_rerank else _load_reranker(args.reranker_model)
    index = Pinecone(api_key=api_key).Index(index_name)

    df = pd.read_excel(xlsx_path)
    df = df.iloc[args.start_from : args.start_from + args.limit].copy()

    output_path = Path(args.output)
    results: list[dict[str, Any]] = []
    for idx, row in df.iterrows():
        try:
            qid = str(row.get("query_id") or idx)
            query = str(row.get("query") or "").strip()
            gold_raw = _normalize_gold(row.get("gold_citations"))
            gold_pairs = [_gold_to_pair(item) for item in gold_raw]

            started = time.perf_counter()
            error = ""
            pred_pairs: list[tuple[str, str]] = []
            debug_candidates: list[dict[str, Any]] = []
            candidates: dict[tuple[str, str], dict[str, Any]] = {}

            subqueries = _build_subqueries(query)[: max(1, args.multi_query_limit)]
            target_codes = _extract_target_codes(query)
            target_articles = _extract_target_article_numbers(query)
            for query_rank, subquery in enumerate(subqueries, start=1):
                vector = embed_query(subquery)
                response = index.query(
                    vector=vector,
                    top_k=args.pinecone_k,
                    namespace=namespace,
                    include_metadata=True,
                )
                query_weight = args.vector_weight if query_rank == 1 else (args.vector_weight * 0.35)
                for rank, match in enumerate(response.get("matches", []), start=1):
                    meta = match.get("metadata", {}) or {}
                    pair = (
                        _normalize_article(meta.get("article_number", "")),
                        _canonicalize_code(meta.get("code_ru", ""), meta.get("source", "")),
                    )
                    if not pair[0]:
                        continue
                    entry = candidates.setdefault(
                        pair,
                        {
                            "pair": pair,
                            "meta": meta,
                            "text": _candidate_text(meta),
                            "fused_score": 0.0,
                        },
                    )
                    entry["fused_score"] += query_weight * (1.0 / (50 + rank))

            bm25_seen: set[tuple[str, str]] = set()
            for bm25_query in subqueries:
                for rank, doc in enumerate(bm25.invoke(bm25_query), start=1):
                    meta = getattr(doc, "metadata", {}) or {}
                    pair = (
                        _normalize_article(meta.get("article_number", "")),
                        _canonicalize_code(meta.get("code_ru", ""), meta.get("source", "")),
                    )
                    if not pair[0] or pair in bm25_seen:
                        continue
                    bm25_seen.add(pair)
                    entry = candidates.setdefault(
                        pair,
                        {
                            "pair": pair,
                            "meta": meta,
                            "text": _candidate_text(meta, getattr(doc, "page_content", "") or ""),
                            "fused_score": 0.0,
                        },
                    )
                    entry["fused_score"] += args.bm25_weight * (1.0 / (80 + rank))

            rescored = sorted(
                candidates.values(),
                key=lambda item: item["fused_score"],
                reverse=True,
            )[: args.candidate_k]

            if rescored and not args.no_rerank:
                rerank_texts = [item["text"] for item in rescored]
                rerank_scores = _rerank_scores(reranker_backend, reranker, query, rerank_texts)
                fused_norm = _minmax_normalize([float(item["fused_score"]) for item in rescored])
                rerank_norm = _minmax_normalize([float(score) for score in rerank_scores])
                code_counts: dict[str, int] = {}
                for idx, (item, rerank_score) in enumerate(zip(rescored, rerank_scores)):
                    item["rerank_score"] = float(rerank_score)
                    code_key = _canonicalize_code(
                        str((item.get("meta") or {}).get("code_ru", "") or ""),
                        str((item.get("meta") or {}).get("source", "") or ""),
                    )
                    diversity_penalty = args.diversity_penalty * code_counts.get(code_key, 0)
                    item["final_score"] = _legal_score(
                        query=query,
                        meta=item.get("meta") or {},
                        text=item.get("text", ""),
                        base_score=fused_norm[idx] + (args.rerank_weight * rerank_norm[idx]),
                        target_codes=target_codes,
                        target_articles=target_articles,
                    ) - diversity_penalty
                    if code_key:
                        code_counts[code_key] = code_counts.get(code_key, 0) + 1
                rescored.sort(key=lambda item: item["final_score"], reverse=True)
            elif rescored:
                code_counts: dict[str, int] = {}
                for item in rescored:
                    code_key = _canonicalize_code(
                        str((item.get("meta") or {}).get("code_ru", "") or ""),
                        str((item.get("meta") or {}).get("source", "") or ""),
                    )
                    diversity_penalty = args.diversity_penalty * code_counts.get(code_key, 0)
                    item["rerank_score"] = 0.0
                    item["final_score"] = _legal_score(
                        query=query,
                        meta=item.get("meta") or {},
                        text=item.get("text", ""),
                        base_score=float(item.get("fused_score", 0.0)),
                        target_codes=target_codes,
                        target_articles=target_articles,
                    ) - diversity_penalty
                    if code_key:
                        code_counts[code_key] = code_counts.get(code_key, 0) + 1
                rescored.sort(key=lambda item: item["final_score"], reverse=True)

            pred_pairs = [item["pair"] for item in rescored[: args.top_k]]
            debug_candidates = [
                {
                    "pair": _pair_to_str(*item["pair"]),
                    "fused_score": item.get("fused_score", 0.0),
                    "rerank_score": item.get("rerank_score", 0.0),
                    "final_score": item.get("final_score", item.get("fused_score", 0.0)),
                    "source": str((item.get("meta") or {}).get("source", "") or ""),
                }
                for item in rescored[: args.top_k]
            ]
            elapsed_sec = round(time.perf_counter() - started, 3)
            metrics = _compute_metrics(gold_pairs, pred_pairs)
            results.append(
                {
                    "row_index": int(idx),
                    "query_id": qid,
                    "query": query,
                    "gold_citations": gold_raw,
                    "retrieved_topk": [_pair_to_str(article, code) for article, code in pred_pairs],
                    "candidates": debug_candidates,
                    "metrics": metrics,
                    "elapsed_sec": elapsed_sec,
                    "error": error,
                }
            )

            print(
                f"[{len(results)}/{len(df)}] {qid} "
                f"strict_hit={metrics['strict_hit']:.0f} "
                f"soft_hit={metrics['soft_hit']:.0f} "
                f"strict_mrr={metrics['strict_mrr']:.3f} "
                f"elapsed={elapsed_sec}s"
            )
            if args.save_every > 0 and (len(results) % args.save_every == 0):
                payload = _build_payload(
                    args=args,
                    xlsx_path=xlsx_path,
                    reranker_backend=reranker_backend,
                    results=results,
                )
                _save_payload(output_path, payload)
                print(f"Autosaved progress: {output_path} ({len(results)} rows)")
        except KeyboardInterrupt:
            print("\nInterrupted by user. Saving partial results...")
            break

    payload = _build_payload(
        args=args,
        xlsx_path=xlsx_path,
        reranker_backend=reranker_backend,
        results=results,
    )
    _save_payload(output_path, payload)
    print(f"\nSaved: {output_path}")
    print(json.dumps(payload["summary"], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
