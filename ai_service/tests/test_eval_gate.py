from ai_service.utils.eval_gate import evaluate_gate


def test_eval_gate_accepts_non_regressing_candidate():
    baseline = {
        "minimum_summary": {
            "strict_hit@k": 0.5,
            "soft_hit@k": 0.8,
            "mrr": 0.4,
            "map_soft": 0.4,
            "latency_sec_avg": 10.0,
            "latency_sec_p95": 15.0,
        },
        "results": [
            {"id": "q1", "metrics": {"strict_hit": 1.0, "soft_hit": 1.0, "mrr": 1.0}}
        ],
    }
    candidate = {
        "summary": {
            "strict_hit@k": 0.6,
            "soft_hit@k": 0.9,
            "mrr": 0.5,
            "map_soft": 0.5,
            "latency_sec_avg": 9.0,
            "latency_sec_p95": 14.0,
        },
        "results": [
            {"id": "q1", "metrics": {"strict_hit": 1.0, "soft_hit": 1.0, "mrr": 1.0}}
        ],
    }

    assert evaluate_gate(candidate, baseline) == []


def test_eval_gate_reports_regressions():
    baseline = {
        "minimum_summary": {"strict_hit@k": 0.5},
        "results": [
            {"id": "q1", "metrics": {"strict_hit": 1.0, "soft_hit": 1.0, "mrr": 1.0}}
        ],
    }
    candidate = {
        "summary": {"strict_hit@k": 0.5},
        "results": [
            {"id": "q1", "metrics": {"strict_hit": 0.0, "soft_hit": 1.0, "mrr": 1.0}}
        ],
    }

    failures = evaluate_gate(candidate, baseline)
    assert failures
    assert any("regressed" in item for item in failures)
