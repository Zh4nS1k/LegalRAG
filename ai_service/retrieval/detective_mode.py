# detective_mode.py — Active Intellectual Subject with exit strategy (no infinite questions)
# Exit: confidence >= 0.7 → full synthesis; < 0.7 and first turn → max 2 questions; else → partial analysis.

import json
import time
from typing import Any, List, Optional, Tuple

from langchain_core.documents import Document

from ai_service.core import config
from ai_service.retrieval import agentic_workflow
from ai_service.retrieval import rag_chain

NOT_FOUND_MSG = "Информация не найдена в доступных текстах законов."

CONFIDENCE_THRESHOLD = getattr(config, "DETECTIVE_CONFIDENCE_THRESHOLD", 0.7)
MAX_CLARIFYING_QUESTIONS = getattr(config, "DETECTIVE_MAX_CLARIFYING_QUESTIONS", 2)

# ─── Missing Info Detection (Critical vs Contextual, with context from history) ───

MISSING_INFO_PROMPT = """Ты — юридический аналитик по НПА РК. Проанализируй запрос пользователя и диалог.

Иерархия права РК: Конституция → кодексы и законы РК. Все выводы только по ним.

Инструкции:
1. Перечисли недостающие переменные из правового чек-листа: даты, суммы, наличие договора, доказательства, стороны, отрасль права.
2. Раздели недостающее на:
   - Critical (критично): без этого нельзя дать заключение — например наличие договора, факт нарушения, сумма иска.
   - Contextual (контекст): уточняет, но можно сформулировать гипотезу — например точное время суток, второстепенные детали. Отметь как "Слепые зоны".
3. Оцени уверенность по имеющимся данным: confidence от 0.0 до 1.0 (сколько процентов картины есть).
4. Если есть Critical и это ПЕРВЫЙ обмен (пользователь ещё не отвечал на уточняющие вопросы) — сформулируй не более 2 коротких уточняющих вопросов.
5. Если это ВТОРОЙ обмен (пользователь уже что-то ответил) или недостаёт только Contextual — не задавай вопросов, укажи proceed_to_search: true.

Уже известное из диалога (контекст):
{context}

Текущий запрос пользователя:
{query}

Ответ СТРОГО в JSON без markdown:
{{
  "confidence": 0.0-1.0,
  "critical_missing": ["элемент1", "элемент2"],
  "contextual_missing": ["слепая зона 1"],
  "clarifying_questions": ["вопрос 1?", "вопрос 2?"],
  "proceed_to_search": true или false
}}

Если данных достаточно для поиска по НПА РК — confidence >= 0.7, списки пустые, proceed_to_search: true.
Максимум 2 вопроса в clarifying_questions.
"""


def _format_history_context(history: Optional[List[dict]]) -> str:
    """Preserve context: already collected answers from the user for the next iteration."""
    if not history:
        return "(Нет предыдущих сообщений.)"
    lines = []
    for m in history[-10:]:  # last 10 turns
        role = "Пользователь" if m.get("role") == "user" else "Ассистент"
        content = (m.get("content") or "").strip()
        if content:
            lines.append(f"{role}: {content}")
    return "\n".join(lines) if lines else "(Нет предыдущих сообщений.)"


def _check_missing_info(
    query: str,
    history: Optional[List[dict]],
    trace_id: str,
) -> Tuple[float, List[str], List[str], List[str], bool, dict]:
    """
    Returns (confidence, critical_missing, contextual_missing, clarifying_questions, proceed_to_search, metrics).
    Preserves context from history.
    """
    t0 = time.perf_counter()
    context_str = _format_history_context(history)
    llm = rag_chain.get_llm()
    try:
        resp = llm.invoke(
            MISSING_INFO_PROMPT.format(context=context_str, query=query.strip())
        )
        text = resp.content if hasattr(resp, "content") else str(resp)
        start, end = text.find("{"), text.rfind("}") + 1
        if start >= 0 and end > start:
            data = json.loads(text[start:end])
            confidence = float(data.get("confidence", 0.7))
            critical = data.get("critical_missing") or []
            contextual = data.get("contextual_missing") or []
            questions = data.get("clarifying_questions") or []
            proceed = data.get("proceed_to_search", confidence >= CONFIDENCE_THRESHOLD)
            if not isinstance(critical, list):
                critical = [str(critical)]
            if not isinstance(contextual, list):
                contextual = [str(contextual)]
            if not isinstance(questions, list):
                questions = [str(questions)]
            questions = questions[:MAX_CLARIFYING_QUESTIONS]
        else:
            confidence, critical, contextual, questions, proceed = 0.7, [], [], [], True
    except Exception as e:
        print(f"[{trace_id}] missing-info check failed: {e}")
        confidence, critical, contextual, questions, proceed = 0.7, [], [], [], True
    ms = round((time.perf_counter() - t0) * 1000)
    return confidence, critical, contextual, questions, proceed, {"detective_completeness_ms": ms}


def _is_first_interaction(history: Optional[List[dict]]) -> bool:
    """True if user has not yet been asked for clarification (no assistant message in history)."""
    if not history:
        return True
    return not any(m.get("role") == "assistant" for m in history)


# ─── Exit strategy: ask only once, then partial analysis ────────────────────

def _should_ask_questions(
    confidence: float,
    critical_missing: List[str],
    is_first: bool,
    proceed_to_search: bool,
) -> bool:
    """Ask max 2 questions only if confidence < 0.7, critical missing, and first interaction."""
    if confidence >= CONFIDENCE_THRESHOLD or proceed_to_search:
        return False
    if not is_first:
        return False  # already asked once → do partial analysis
    return bool(critical_missing)


# ─── Full synthesis (causality + skeptic + flip) ───────────────────────────

CAUSALITY_SYNTHESIS_PROMPT = """На основе приведённого ответа и контекста НПА РК оформи итог в виде строгой правовой логики.

1) Цепочка причинности (обязательно):
   Факт A (что установлено) + Норма права B (статья, кодекс) = Правовое последствие C.
   Дай краткую формулировку вывода, а не только цитату.

2) Анти-мошенничество (Скептик): Проверь по ГК и УК РК — есть ли признаки рисков: кабальные проценты, схемы уклонения от налогов, принуждение к сделке (ст. 159 ГК РК). Если да — добавь блок "Внимание, возможные риски:" с перекрёстной отсылкой к нормам.

3) Раздел "Что может изменить исход дела": Укажи слепые зоны: срок исковой давности, оспоримость сделки (принуждение, ст. 159 ГК РК), истечение сроков, недостающие доказательства.

Исходный ответ и контекст:
---
{answer}

Контекст из НПА:
{context}
---

Ответ оформи так:
[Причинность] ...
[Внимание, возможные риски:] ... (или "Не выявлено.")
[Что может изменить исход дела] ...
"""


def _synthesis_causality_skeptic_flip(
    answer: str,
    source_docs: List[Document],
    trace_id: str,
) -> Tuple[str, dict]:
    """Returns (enriched_answer, metrics)."""
    t0 = time.perf_counter()
    context_parts = [d.page_content[:600] for d in source_docs[:6]]
    context_str = "\n---\n".join(context_parts)
    llm = rag_chain.get_llm()
    try:
        resp = llm.invoke(
            CAUSALITY_SYNTHESIS_PROMPT.format(answer=answer[:3000], context=context_str[:4000])
        )
        enriched = resp.content if hasattr(resp, "content") else str(resp)
    except Exception as e:
        print(f"[{trace_id}] causality/skeptic/flip synthesis failed: {e}")
        enriched = answer
    ms = round((time.perf_counter() - t0) * 1000)
    return enriched.strip(), {"detective_synthesis_ms": ms}


# ─── Partial Analysis (Harvey style: "Based on X%, could flip if..." ) ───────

PARTIAL_ANALYSIS_PROMPT = """Ты — юридический аналитик по НПА РК. По имеющимся данным и контексту законов дай частичный анализ.

Иерархия права РК: Конституция → кодексы и законы РК.

Формат ответа (обязательно):
1) "Исходя из имеющихся данных (примерно {data_pct}% картины), ситуация выглядит так: [X]."
2) "Однако вывод может измениться на [Y], если окажется, что [Недостающий факт / Слепая зона]."
3) "Анализ: " — краткий вывод по нормам РК.
4) "Риски (что мы не знаем): " — перечисли слепые зоны (contextual_missing).
5) "Следующие шаги: " — что должен предоставить пользователь для заключения с точностью 99%.

Недостающие критические факты: {critical_missing}
Слепые зоны (контекст): {contextual_missing}

Исходный ответ по НПА:
---
{answer}
---

Контекст из НПА:
{context}
"""


def _synthesis_partial_analysis(
    answer: str,
    source_docs: List[Document],
    critical_missing: List[str],
    contextual_missing: List[str],
    data_pct: int,
    trace_id: str,
) -> Tuple[str, dict]:
    """Harvey-style partial synthesis: Based on X%, situation looks like [X]. Could flip to [Y] if [Missing]."""
    t0 = time.perf_counter()
    context_str = "\n---\n".join(d.page_content[:600] for d in source_docs[:6])
    llm = rag_chain.get_llm()
    try:
        resp = llm.invoke(
            PARTIAL_ANALYSIS_PROMPT.format(
                answer=answer[:3000],
                context=context_str[:4000],
                data_pct=data_pct,
                critical_missing=", ".join(critical_missing) or "—",
                contextual_missing=", ".join(contextual_missing) or "—",
            )
        )
        enriched = resp.content if hasattr(resp, "content") else str(resp)
    except Exception as e:
        print(f"[{trace_id}] partial analysis synthesis failed: {e}")
        enriched = answer
    ms = round((time.perf_counter() - t0) * 1000)
    return enriched.strip(), {"detective_partial_synthesis_ms": ms}


# ─── Confidence score (final) ──────────────────────────────────────────────

def _compute_confidence(
    best_reranker_score: float,
    cove_passed: bool,
    has_sources: bool,
    completeness_rejected: bool,
) -> float:
    if completeness_rejected:
        return 0.0
    if not has_sources:
        return 0.0
    score = max(0.0, min(1.0, float(best_reranker_score)))
    if not cove_passed:
        score *= 0.5
    return round(score, 2)


# ─── Main entry ─────────────────────────────────────────────────────────────

def invoke_detective_qa(
    query: str,
    history: Optional[List[dict]] = None,
    trace_id: Optional[str] = None,
) -> dict:
    """
    Exit strategy:
    - If confidence >= 0.7 → retrieval + full synthesis.
    - If confidence < 0.7 and first interaction and critical missing → max 2 questions (then stop).
    - If confidence < 0.7 and second interaction OR only contextual missing → retrieval + partial analysis
      ("Based on X% of data... Could flip to [Y] if [Missing]. Analysis / Risks / Next Steps.").
    Context from history is always injected so the model sees already collected answers.
    """
    trace_id = trace_id or f"trace_{int(time.time())}"
    missing_fields: List[str] = []
    clarifying_questions: List[str] = []
    confidence_score = 0.0
    metrics: dict = {}

    # 1. Missing info detection with context (Critical vs Contextual, preserve history)
    confidence_est, critical_missing, contextual_missing, questions, proceed_to_search, m_comp = _check_missing_info(
        query, history, trace_id
    )
    metrics.update(m_comp)
    is_first = _is_first_interaction(history)
    should_ask = _should_ask_questions(
        confidence_est, critical_missing, is_first, proceed_to_search
    )

    # 2. Exit: ask only when allowed (max once, max 2 questions)
    if should_ask and questions:
        result_text = (
            "Для точного правового заключения по НПА РК нужны уточнения:\n\n"
            + "\n".join(f"• {q}" for q in questions[:MAX_CLARIFYING_QUESTIONS])
            + "\n\nОтветьте, пожалуйста, и я дам заключение (или частичный анализ по имеющимся данным)."
        )
        return {
            "result": result_text,
            "source_documents": [],
            "confidence_score": 0.0,
            "missing_fields": critical_missing + contextual_missing,
            "clarifying_questions": questions[:MAX_CLARIFYING_QUESTIONS],
            "trace_report": {
                "metadata": {"id": trace_id},
                "metrics_ms": metrics,
                "detective": {"completeness_rejected": True, "exit": "ask_questions"},
            },
        }

    # 3. Proceed to retrieval (HyDE) and synthesis
    agentic_out = agentic_workflow.invoke_agentic_qa(query, history=history, trace_id=trace_id)
    result = agentic_out.get("result", "")
    source_documents = agentic_out.get("source_documents", [])
    trace_report = agentic_out.get("trace_report") or {}
    metrics.update(trace_report.get("metrics_ms") or {})

    if not result or result.strip() == NOT_FOUND_MSG:
        ms = trace_report.get("metrics_ms") or {}
        best = float((ms.get("agentic") or {}).get("best_reranker_score") or ms.get("best_reranker_score") or 0.0)
        confidence_score = _compute_confidence(best, False, False, False)
        return {
            "result": result,
            "source_documents": source_documents,
            "confidence_score": confidence_score,
            "missing_fields": [],
            "trace_report": {
                **trace_report,
                "metadata": {**(trace_report.get("metadata") or {}), "id": trace_id},
                "metrics_ms": metrics,
            },
        }

    # 4. Synthesis: full (confidence >= 0.7) vs partial (Harvey-style flip)
    if confidence_est >= CONFIDENCE_THRESHOLD:
        result, m_syn = _synthesis_causality_skeptic_flip(result, source_documents, trace_id)
    else:
        data_pct = int(confidence_est * 100) if confidence_est else 40
        result, m_syn = _synthesis_partial_analysis(
            result, source_documents,
            critical_missing, contextual_missing,
            data_pct, trace_id,
        )
    metrics.update(m_syn)

    # 5. Final confidence
    agentic_metrics = (trace_report.get("metrics_ms") or {}).get("agentic") or {}
    best_reranker = float(agentic_metrics.get("best_reranker_score", 0.5))
    cove_ok = not metrics.get("cove_replaced", False)
    confidence_score = _compute_confidence(
        best_reranker, cove_ok, bool(source_documents), False
    )

    trace_report["metrics_ms"] = metrics
    trace_report["metadata"] = trace_report.get("metadata") or {}
    trace_report["metadata"]["id"] = trace_id
    trace_report.setdefault("detective", {})["completeness_rejected"] = False
    trace_report["detective"]["exit"] = "synthesis"
    trace_report["detective"]["partial_analysis"] = confidence_est < CONFIDENCE_THRESHOLD

    return {
        "result": result,
        "source_documents": source_documents,
        "confidence_score": confidence_score,
        "missing_fields": critical_missing + contextual_missing,
        "trace_report": trace_report,
    }
