from ai_service.utils.retrieval_quality_benchmark import (
    _build_comparison,
    _build_summary,
    _classify_query,
    _compute_metrics,
    _extract_predicted_articles,
    _extract_predicted_pairs,
    _normalize_article,
)


class _Doc:
    def __init__(self, article_number: str, code_ru: str = ""):
        self.metadata = {"article_number": article_number, "code_ru": code_ru}


def test_normalize_article_keeps_article_number():
    assert _normalize_article("ст. 214") == "214"
    assert _normalize_article("214-бап") == "214"
    assert _normalize_article(" 190-1 ") == "190-1"


def test_extract_predicted_articles_is_unique_and_ordered():
    docs = [_Doc("214"), _Doc("214"), _Doc("245")]

    predicted = _extract_predicted_articles(docs, top_k=5)

    assert predicted == ["214", "245"]


def test_extract_predicted_pairs_keeps_code_and_article():
    docs = [_Doc("214", "Уголовный кодекс РК"), _Doc("214", "Уголовный кодекс РК")]

    predicted = _extract_predicted_pairs(docs, top_k=5)

    assert predicted == [("214", "уголовный кодекс рк")]


def test_compute_metrics_uses_top1_for_strict_and_any_for_soft():
    metrics = _compute_metrics(["245"], ["214", "245"])

    assert metrics["strict_hit"] == 1.0
    assert metrics["soft_hit"] == 1.0
    assert metrics["strict_precision"] == 0.5
    assert metrics["strict_recall"] == 1.0
    assert metrics["strict_f1"] == 2 / 3
    assert metrics["soft_ap"] == 0.5
    assert metrics["mrr"] == 0.5


def test_compute_metrics_can_use_strict_code_article_pairs():
    metrics = _compute_metrics(
        ["245"],
        ["245"],
        relevant_pairs=[("245", "Налоговый кодекс РК")],
        predicted_pairs=[("245", "Уголовный кодекс РК")],
    )

    assert metrics["strict_hit"] == 0.0
    assert metrics["soft_hit"] == 1.0
    assert metrics["strict_precision"] == 0.0
    assert metrics["soft_precision"] == 1.0


def test_compute_metrics_handles_partial_multi_article_coverage():
    metrics = _compute_metrics(["214", "245"], ["214", "999"])

    assert metrics["soft_hit"] == 1.0
    assert metrics["soft_recall"] == 0.5
    assert metrics["soft_precision"] == 0.5
    assert metrics["soft_f1"] == 0.5
    assert metrics["soft_ap"] == 0.5
    assert metrics["relevant_count"] == 2.0
    assert metrics["predicted_count"] == 2.0


def test_classify_query_adds_useful_benchmark_tags():
    tags = _classify_query(
        "Если бизнес ведется без регистрации и еще скрываются налоги, какие статьи УК РК применимы?",
        "Комбинированный кейс: незаконный бизнес + налоги",
        "ru",
        ["214", "245"],
        [],
    )

    assert "lang:ru" in tags
    assert "multi_article" in tags
    assert "compound_issue" in tags
    assert "penalty_focused" in tags
    assert "code_lookup" in tags


def test_build_summary_and_comparison():
    results = [
        {
            "lang": "ru",
            "tags": ["lang:ru", "single_article", "penalty_focused"],
            "metrics": {
                "strict_hit": 1.0,
                "soft_hit": 1.0,
                "strict_precision": 1.0,
                "strict_recall": 1.0,
                "strict_f1": 1.0,
                "soft_precision": 1.0,
                "soft_recall": 1.0,
                "soft_f1": 1.0,
                "strict_ap": 1.0,
                "soft_ap": 1.0,
                "mrr": 1.0,
                "relevant_count": 1.0,
                "predicted_count": 1.0,
            },
            "elapsed_sec": 0.2,
        },
        {
            "lang": "kz",
            "tags": ["lang:kz", "multi_article", "compound_issue", "range_query"],
            "metrics": {
                "strict_hit": 0.0,
                "soft_hit": 1.0,
                "strict_precision": 0.0,
                "strict_recall": 0.0,
                "strict_f1": 0.0,
                "soft_precision": 0.5,
                "soft_recall": 1.0,
                "soft_f1": 2 / 3,
                "strict_ap": 0.0,
                "soft_ap": 0.5,
                "mrr": 0.5,
                "relevant_count": 2.0,
                "predicted_count": 2.0,
            },
            "elapsed_sec": 0.4,
        },
    ]

    summary = _build_summary(results, top_k=5)
    comparison = _build_comparison(
        summary,
        {
            "strict_hit@k": 0.25,
            "soft_hit@k": 0.5,
            "strict_precision@k": 0.25,
            "soft_recall@k": 0.25,
            "map_soft": 0.25,
            "mrr": 0.4,
            "latency_sec_avg": 0.1,
        },
    )

    assert summary["strict_hit@k"] == 0.5
    assert summary["soft_hit@k"] == 1.0
    assert summary["mrr"] == 0.75
    assert summary["strict_precision@k"] == 0.5
    assert summary["soft_recall@k"] == 1.0
    assert summary["map_soft"] == 0.75
    assert summary["avg_relevant_articles"] == 1.5
    assert round(summary["latency_sec_avg"], 3) == 0.3
    assert summary["by_lang"]["ru"]["strict_hit@k"] == 1.0
    assert summary["by_tag"]["penalty_focused"]["queries_evaluated"] == 1
    assert summary["by_tag"]["range_query"]["queries_evaluated"] == 1
    assert summary["by_tag"]["compound_issue"]["soft_hit@k"] == 1.0
    assert summary["by_complexity"]["single_article"]["queries_evaluated"] == 1
    assert summary["by_complexity"]["multi_article"]["queries_evaluated"] == 1
    assert comparison["delta_strict_hit@k"] == 0.25
    assert comparison["delta_strict_precision@k"] == 0.25
    assert comparison["delta_map_soft"] == 0.5
    assert round(comparison["delta_latency_sec_avg"], 3) == 0.2
