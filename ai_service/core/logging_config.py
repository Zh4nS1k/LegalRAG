from __future__ import annotations

import json
import logging
import os
import time


class JsonFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        payload = {
            "ts": time.time(),
            "level": record.levelname,
            "logger": record.name,
            "msg": record.getMessage(),
            "trace_id": getattr(record, "trace_id", None),
            "query_len": getattr(record, "query_len", None),
            "intent": getattr(record, "intent", None),
            "latency_ms": getattr(record, "latency_ms", None),
        }
        if record.exc_info:
            payload["exc_info"] = self.formatException(record.exc_info)
        return json.dumps(payload, ensure_ascii=False)


def configure_logging() -> None:
    if os.environ.get("LOG_FORMAT", "human").lower() != "json":
        return

    root = logging.getLogger()
    formatter = JsonFormatter()
    for handler in root.handlers:
        handler.setFormatter(formatter)
