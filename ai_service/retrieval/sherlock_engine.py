import asyncio
import json
import logging
import time
from typing import Any, Dict, List

from langchain_core.documents import Document

from ai_service.core import config
from ai_service.retrieval import rag_chain

logger = logging.getLogger("ai_service.sherlock_engine")

SHERLOCK_AUDIT_PROMPT = """Ты — Sherlock audit mode.
Твоя задача — сделать краткий юридический аудит на основе уже извлечённого canonical retrieval context.

Жёсткие правила:
- Не выполняй отдельный поиск.
- Не придумывай нормы, которых нет в context.
- Если контекста недостаточно, прямо скажи об этом.
- Отвечай строго в JSON.

Формат:
{
  "summary": "краткий вывод",
  "position": "роль пользователя или правовая позиция",
  "applicable_articles": [
    {
      "code_ru": "название кодекса или закона",
      "article_number": "номер статьи или null",
      "reason": "почему статья релевантна"
    }
  ],
  "conflicts": [
    {
      "description": "если есть коллизия, иначе пустой список",
      "resolution": "как её разрешить"
    }
  ],
  "needs_clarification": true,
  "clarifying_question": "вопрос, если данных недостаточно",
  "confidence": 0.0
}

Запрос:
__QUERY__

Контекст:
__CONTEXT__
"""


def _extract_json_block(content: str) -> Dict[str, Any]:
    raw = str(content or "").strip()
    if not raw:
        return {}

    start = raw.find("{")
    end = raw.rfind("}")
    if start < 0 or end < 0 or end <= start:
        return {}

    try:
        return json.loads(raw[start : end + 1])
    except Exception:
        return {}


def _stringify_article(article: Dict[str, Any]) -> str:
    code_ru = str(article.get("code_ru") or "").strip() or "Неизвестный источник"
    article_number = str(article.get("article_number") or "").strip()
    reason = str(article.get("reason") or "").strip()
    header = code_ru
    if article_number and article_number.lower() != "null":
        header = f"{header}, ст. {article_number}"
    return f"- {header}: {reason or 'релевантно к запросу'}"


def _render_report(query: str, analysis: Dict[str, Any], docs: List[Document]) -> str:
    summary = str(analysis.get("summary") or "").strip() or "Недостаточно данных для уверенного вывода."
    position = str(analysis.get("position") or "").strip() or "Позиция не определена."
    clarification = str(analysis.get("clarifying_question") or "").strip()
    confidence = analysis.get("confidence")
    applicable_articles = analysis.get("applicable_articles") or []
    conflicts = analysis.get("conflicts") or []

    lines = [
        "# Sherlock audit",
        f"Query: {query}",
        f"Summary: {summary}",
        f"Position: {position}",
    ]
    if confidence is not None:
        lines.append(f"Confidence: {confidence}")

    if applicable_articles:
        lines.append("Applicable articles:")
        for article in applicable_articles[:5]:
            if isinstance(article, dict):
                lines.append(_stringify_article(article))
    elif docs:
        lines.append("Applicable articles:")
        for doc in docs[:3]:
            meta = doc.metadata or {}
            code_ru = str(meta.get("code_ru") or "Неизвестный источник").strip()
            article_number = str(meta.get("article_number") or "").strip()
            source = code_ru if not article_number else f"{code_ru}, ст. {article_number}"
            lines.append(f"- {source}")

    if conflicts:
        lines.append("Conflicts:")
        for item in conflicts[:3]:
            if isinstance(item, dict):
                description = str(item.get("description") or "").strip() or "Коллизия не описана"
                resolution = str(item.get("resolution") or "").strip() or "Не указано"
                lines.append(f"- {description} -> {resolution}")
    else:
        lines.append("Conflicts: none")

    if analysis.get("needs_clarification"):
        lines.append("Needs clarification: yes")
        if clarification:
            lines.append(f"Clarifying question: {clarification}")
    else:
        lines.append("Needs clarification: no")
        if clarification:
            lines.append(f"Clarifying question: {clarification}")

    return "\n".join(lines).strip()


class SherlockEngine:
    def __init__(self, model_override: str | None = None):
        self.llm = None
        self.enabled = config.LEGAL_RAG_ENABLE_SHERLOCK
        self.retriever_top_k = int(
            getattr(config, "SHERLOCK_RETRIEVER_TOP_K", config.RETRIEVER_WIDE_K)
        )
        self.model_override = model_override

    async def _retrieve_context(self, query: str) -> List[Document]:
        retriever = rag_chain.get_retriever_for_coverage(top_k=self.retriever_top_k)
        docs = await asyncio.to_thread(retriever.invoke, query)
        return list(docs or [])

    async def run_sherlock_loop(self, query: str) -> Dict[str, Any]:
        t0 = time.perf_counter()
        if not self.enabled:
            return {
                "deductive_output": None,
                "meta": {
                    "enabled": False,
                    "mode": "disabled",
                    "stack": "canonical_retriever",
                    "elapsed_ms": round((time.perf_counter() - t0) * 1000),
                },
            }

        docs: List[Document] = []
        analysis: Dict[str, Any] = {}

        try:
            docs = await self._retrieve_context(query)
            llm = self.llm
            if llm is None:
                llm = await asyncio.to_thread(lambda: rag_chain.get_llm(self.model_override))
                self.llm = llm
            context = "\n\n".join(
                [
                    f"[{str((doc.metadata or {}).get('code_ru') or 'Неизвестный источник')} | "
                    f"ст. {str((doc.metadata or {}).get('article_number') or 'Н/Д')}]\n"
                    f"{str(doc.page_content or '').strip()[:1000]}"
                    for doc in docs[: self.retriever_top_k]
                ]
            )
            prompt = (
                SHERLOCK_AUDIT_PROMPT.replace("__QUERY__", query)
                .replace("__CONTEXT__", context or "Контекст не найден.")
            )
            response = await asyncio.to_thread(llm.invoke, prompt)
            content = response.content if hasattr(response, "content") else str(response)
            analysis = _extract_json_block(content)
            if not analysis:
                analysis = {
                    "summary": "Не удалось надёжно распарсить audit output.",
                    "position": "Позиция не определена.",
                    "applicable_articles": [],
                    "conflicts": [],
                    "needs_clarification": True,
                    "clarifying_question": "Уточните фактические обстоятельства запроса.",
                    "confidence": 0.0,
                }
        except Exception as exc:
            logger.error("Sherlock audit failed: %s", exc, exc_info=True)
            analysis = {
                "summary": "Sherlock audit unavailable.",
                "position": "Позиция не определена.",
                "applicable_articles": [],
                "conflicts": [],
                "needs_clarification": True,
                "clarifying_question": "Уточните фактические обстоятельства запроса.",
                "confidence": 0.0,
            }

        return {
            "deductive_output": _render_report(query, analysis, docs),
            "meta": {
                "enabled": True,
                "mode": "canonical_retrieval_audit",
                "stack": "canonical_retriever",
                "retriever_top_k": self.retriever_top_k,
                "docs_count": len(docs),
                "confidence": analysis.get("confidence", 0.0),
                "elapsed_ms": round((time.perf_counter() - t0) * 1000),
            },
        }
