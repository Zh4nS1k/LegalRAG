import os, time, requests
from models.schemas import LLMResult

class LLMClient:

    def __init__(self):
        self.url     = os.getenv("AI_SERVICE_URL", "http://localhost:8000")
        self.timeout = float(os.getenv("AI_SERVICE_TIMEOUT", "120"))

    def call(self, model: str, question: str,
             q_id: str = "unknown", history: list = None) -> LLMResult:
        if history is None:
            history = []
        start = time.perf_counter()
        try:
            resp = requests.post(
                f"{self.url}/api/v1/internal-chat",
                json={
                    "query": question,
                    "history": history,
                    "model": model,        # ← передаём модель
                },
                timeout=self.timeout,
            )
            resp.raise_for_status()
            elapsed = (time.perf_counter() - start) * 1000
            data = resp.json()

            # engine возвращает "result" согласно документации
            answer = str(data.get("result", "")).strip()

            # Дополнительные метрики если engine их возвращает
            trace = data.get("trace_report", {})
            metrics = trace.get("metrics_ms", {}).get("breakdown", {})

            return LLMResult(
                model=model,
                answer=answer or "Контекстте жауап жоқ.",
                answer_raw=answer,
                error="",
                latency_ms=elapsed,
                retrieve_ms=float(
                    metrics.get("vector_search", 0) or 0
                ),
                embed_ms=float(
                    metrics.get("embedding", 0) or 0
                ),
                chunks_used=len(
                    data.get("source_documents", [])
                ),
            )

        except requests.exceptions.ConnectionError:
            elapsed = (time.perf_counter() - start) * 1000
            return LLMResult(
                model=model, answer="",
                error="engine недоступен на порту 8000",
                latency_ms=elapsed,
            )
        except Exception as e:
            elapsed = (time.perf_counter() - start) * 1000
            return LLMResult(
                model=model, answer="",
                error=f"{type(e).__name__}: {str(e)[:120]}",
                latency_ms=elapsed,
            )