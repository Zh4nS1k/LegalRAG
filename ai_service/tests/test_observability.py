import logging

import pytest

from ai_service.core.logging_config import JsonFormatter, configure_logging
from ai_service.utils.circuit_breaker import CircuitBreaker, CircuitBreakerOpen


def test_json_formatter_includes_core_fields():
    record = logging.LogRecord(
        name="ai_service.test",
        level=logging.INFO,
        pathname=__file__,
        lineno=10,
        msg="hello",
        args=(),
        exc_info=None,
    )
    record.trace_id = "trace-1"
    record.intent = "CASE_SPECIFIC"
    payload = JsonFormatter().format(record)

    assert '"msg": "hello"' in payload
    assert '"trace_id": "trace-1"' in payload
    assert '"intent": "CASE_SPECIFIC"' in payload


def test_configure_logging_switches_root_handlers(monkeypatch):
    monkeypatch.setenv("LOG_FORMAT", "json")
    root = logging.getLogger()
    handler = logging.StreamHandler()
    root.addHandler(handler)

    try:
        configure_logging()
        assert isinstance(root.handlers[-1].formatter, JsonFormatter)
    finally:
        root.removeHandler(handler)


def test_circuit_breaker_opens_after_threshold():
    breaker = CircuitBreaker("groq", failure_threshold=2, reset_timeout=1)

    with pytest.raises(ValueError):
        breaker.call(lambda: (_ for _ in ()).throw(ValueError("boom")))
    assert breaker.state == "closed"

    with pytest.raises(ValueError):
        breaker.call(lambda: (_ for _ in ()).throw(ValueError("boom")))
    assert breaker.state == "open"

    with pytest.raises(CircuitBreakerOpen):
        breaker.call(lambda: "ok")
