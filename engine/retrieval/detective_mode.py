# detective_mode.py — Active Intellectual Subject with exit strategy (no infinite questions)
# Exit: confidence >= 0.7 → full synthesis; < 0.7 and first turn → max 2 questions; else → partial analysis.

import json
import time
from typing import List, Optional, Tuple

from langchain_core.documents import Document

from engine.core import config
from engine.retrieval import agentic_workflow
from engine.retrieval import intent_router
from engine.retrieval import rag_chain

NOT_FOUND_MSG = "Информация не найдена в доступных текстах законов."

CONFIDENCE_THRESHOLD = getattr(config, "DETECTIVE_CONFIDENCE_THRESHOLD", 0.7)
MAX_CLARIFYING_QUESTIONS = getattr(config, "DETECTIVE_MAX_CLARIFYING_QUESTIONS", 2)

# ─── STAGE 1: THE LINGUIST (Query Expansion) ───

LINGUIST_EXPANSION_PROMPT = """Ты — лингвист по праву РК. Проанализируй запрос пользователя и переведи неформальные термины в официальный юридический язык РК.

Инструкции:
1. Идентифицируй неформальные/бытовые термины (например, "несовершеннолетний студент", "ночная смена").
2. Переведи в "легализ": официальные термины из НПА РК (например, "несовершеннолетний" -> "работники, не достигшие восемнадцатилетнего возраста").
3. Сгенерируй 3 варианта поиска:
   - Semantic: контекстуальный (описательный).
   - Keyword: с номерами статей, специфическими терминами.
   - Broad: на уровне глав/разделов.

Запрос пользователя:
{query}

Ответ СТРОГО в JSON:
{{
  "expanded_terms": {{"несовершеннолетний": "работники, не достигшие восемнадцатилетнего возраста"}},
  "search_variants": {{
    "semantic": "запрос для семантического поиска",
    "keyword": "запрос с ключевыми словами и статьями",
    "broad": "широкий запрос на уровне кодекса"
  }}
}}
"""


import asyncio

async def _linguist_query_expansion(query: str, trace_id: str, model_override: str | None = None) -> Tuple[dict, dict]:
    """Returns (expanded_terms, search_variants), metrics."""
    t0 = time.perf_counter()
    llm = rag_chain.get_llm(model_override)
    try:
        resp = await asyncio.to_thread(llm.invoke, LINGUIST_EXPANSION_PROMPT.format(query=query.strip()))
        text = resp.content if hasattr(resp, "content") else str(resp)
        start, end = text.find("{"), text.rfind("}") + 1
        if start >= 0 and end > start:
            data = json.loads(text[start:end])
            expanded_terms = data.get("expanded_terms", {})
            search_variants = data.get("search_variants", {})
        else:
            expanded_terms, search_variants = {}, {}
    except Exception as e:
        print(f"[{trace_id}] linguist expansion failed: {e}")
        expanded_terms, search_variants = {}, {}
    ms = round((time.perf_counter() - t0) * 1000)
    return expanded_terms, search_variants, {"linguist_expansion_ms": ms}


MISSING_INFO_PROMPT = """Ты — юридический аналитик по НПА РК. Проанализируй запрос пользователя и диалог.

Иерархия права РК: Конституция → кодексы и законы РК. Все выводы только по ним.

Инструкции:
1. Выяви недостающие ключевые факты, исходя из сути вопроса (например: для драки — степень вреда здоровью; для кражи — сумма; для возврата товара — прошел ли срок 14 дней). Не выдумывай лишние критерии, если они не нужны для ответа.
2. Раздели недостающее на:
   - Critical (критично): без этого нельзя дать точный правовой ответ.
   - Contextual (контекст): второстепенные детали, можно обойтись без них.
3. Оцени уверенность по имеющимся данным: confidence от 0.0 до 1.0 (насколько понятна ситуация).
4. Threshold Logic:
   - Если уверенность < 0.7 и есть Critical факты (и это ПЕРВЫЙ обмен) — сформулируй 1-2 коротких уточняющих вопроса.
   - Если уверенность >= 0.7 или это ВТОРОЙ обмен — proceed_to_search: true.

Уже известное из диалога (контекст):
{context}

Текущий запрос пользователя:
{query}

Ответ СТРОГО в JSON без markdown:
{{
  "confidence": 0.0-1.0,
  "critical_missing": ["факты, без которых нельзя ответить"],
  "contextual_missing": ["второстепенные детали"],
  "assumptions": ["сделанные предположения"],
  "clarifying_questions": ["уточняющий вопрос 1?", "уточняющий вопрос 2?"],
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


async def _check_missing_info(
    query: str,
    history: Optional[List[dict]],
    trace_id: str,
    model_override: str | None = None,
) -> Tuple[float, List[str], List[str], List[str], List[str], List[str], bool, dict]:
    """
    Returns (confidence, critical_missing, contextual_missing, blind_spots, assumptions, clarifying_questions, proceed_to_search, metrics).
    """
    t0 = time.perf_counter()
    context_str = _format_history_context(history)
    llm = rag_chain.get_llm(model_override)
    try:
        resp = await asyncio.to_thread(
            llm.invoke, MISSING_INFO_PROMPT.format(context=context_str, query=query.strip())
        )
        text = resp.content if hasattr(resp, "content") else str(resp)
        start, end = text.find("{"), text.rfind("}") + 1
        if start >= 0 and end > start:
            data = json.loads(text[start:end])
            confidence = float(data.get("confidence", 0.7))
            critical = data.get("critical_missing") or []
            contextual = data.get("contextual_missing") or []
            blind_spots = data.get("blind_spots") or []
            assumptions = data.get("assumptions") or []
            questions = data.get("clarifying_questions") or []
            proceed = data.get("proceed_to_search", confidence >= CONFIDENCE_THRESHOLD)
            if not isinstance(critical, list):
                critical = [str(critical)]
            if not isinstance(contextual, list):
                contextual = [str(contextual)]
            if not isinstance(blind_spots, list):
                blind_spots = [str(blind_spots)]
            if not isinstance(assumptions, list):
                assumptions = [str(assumptions)]
            if not isinstance(questions, list):
                questions = [str(questions)]
            questions = questions[:MAX_CLARIFYING_QUESTIONS]
        else:
            (
                confidence,
                critical,
                contextual,
                blind_spots,
                assumptions,
                questions,
                proceed,
            ) = (0.7, [], [], [], [], [], True)
    except Exception as e:
        print(f"[{trace_id}] missing-info check failed: {e}")
        (
            confidence,
            critical,
            contextual,
            blind_spots,
            assumptions,
            questions,
            proceed,
        ) = (0.7, [], [], [], [], [], True)
    ms = round((time.perf_counter() - t0) * 1000)
    return (
        confidence,
        critical,
        contextual,
        blind_spots,
        assumptions,
        questions,
        proceed,
        {"detective_completeness_ms": ms},
    )


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


# ─── STAGE 3: THE REASONER (Hybrid Synthesis with Internal Fallback) ───

INTERNAL_KNOWLEDGE_FALLBACK_PROMPT = """Роль и цель
Вы — профессиональный юридический консультант, специализирующийся на законодательстве Республики Казахстан (РК). Ваша задача — предоставлять точные, структурированные и обоснованные ответы на основе нормативных правовых актов (НПА) РК.

Внимание: В базе данных не найдено информации по запросу. Используй свои внутренние знания по НПА РК для предварительного ответа.

Пользователь спросил: "{query}"

Источники знаний
Приоритет №1: Общие знания о праве РК, так как в файлах нет специфической информации.
Актуальность: Ссылайтесь на кодексы (Гражданский, Уголовный, Трудовой, Налоговый, Административный и др.) и законы РК.

Правила ответов
Цитирование: КРИТИЧЕСКИ ВАЖНО! ПРИ КАЖДОМ ОТВЕТЕ ВЫ ОБЯЗАНЫ УКАЗАТЬ СТАТЬЮ, ПУНКТ И ЗАКОН (КОДЕКС), которые относятся к вопросу. Ответ без точного номера статьи не принимается! Пример: «Согласно пункту 1 статьи 30 Закона РК "О защите прав потребителей"…».
Структура:
Краткий ответ: Прямой ответ на вопрос пользователя.
Правовое обоснование: Ссылки на конкретные нормы права.
Разъяснение: Простым языком объясните, как эта норма применяется к ситуации пользователя. Обязательно укажите, что ответ основан только на общих знаниях.
Рекомендация: Какие действия необходимо предпринять (например, подать заявление, составить претензию).
Язык: Отвечайте на том языке, на котором задан вопрос (казахский или русский). Если вопрос на казахском, используйте официальную юридическую терминологию.

Ограничения и этика
Отсутствие данных: Если вашей базе нет точного ответа на вопрос, честно сообщите об этом. Не выдумывайте нормы права.
Дисклеймер: В конце каждого сложного или критического ответа добавляйте примечание: «Данная консультация носит информационный характер. Для принятия юридически значимых решений рекомендуется обратиться к лицензированному адвокату или юристу».
Конфиденциальность: Не просите пользователя вводить ИИН, номера удостоверений личности или другие чувствительные персональные данные.

Тон общения
Официально-деловой, сдержанный, объективный и поддерживающий. Избегайте двусмысленных трактовок.
"""


async def _internal_knowledge_fallback(query: str, trace_id: str, model_override: str | None = None) -> Tuple[str, dict]:
    """Returns (answer, metrics)."""
    t0 = time.perf_counter()
    llm = rag_chain.get_llm(model_override)
    try:
        resp = await asyncio.to_thread(
            llm.invoke, INTERNAL_KNOWLEDGE_FALLBACK_PROMPT.format(query=query.strip())
        )
        answer = resp.content if hasattr(resp, "content") else str(resp)
    except Exception as e:
        print(f"[{trace_id}] internal fallback failed: {e}")
        answer = "⚠️ ВНИМАНИЕ: Информация не найдена в текущей базе данных. Рекомендуется консультация юриста."
    ms = round((time.perf_counter() - t0) * 1000)
    return answer.strip(), {"internal_fallback_ms": ms}


CAUSALITY_SYNTHESIS_PROMPT = """Роль и цель
Вы — профессиональный юридический консультант, специализирующийся на законодательстве Республики Казахстан (РК). Ваша задача — предоставлять точные, структурированные и обоснованные ответы на основе нормативных правовых актов (НПА) РК.

Источники знаний
Приоритет №1: Файлы, загруженные в базу знаний (Контекст из НПА). Всегда начинайте поиск ответа с них.
Приоритет №2: Общие знания о праве РК, если в файлах нет специфической информации.
Актуальность: Ссылайтесь на кодексы (Гражданский, Уголовный, Трудовой, Налоговый, Административный и др.) и законы РК.

Правила ответов
Цитирование: КРИТИЧЕСКИ ВАЖНО! ПРИ КАЖДОМ ОТВЕТЕ ВЫ ОБЯЗАНЫ УКАЗАТЬ СТАТЬЮ, ПУНКТ И ЗАКОН (КОДЕКС), которые относятся к вопросу. Ответ без точного номера статьи не принимается! Пример: «Согласно пункту 1 статьи 30 Закона РК "О защите прав потребителей"…».
Структура:
Краткий ответ: Прямой ответ на вопрос пользователя.
Правовое обоснование: Ссылки на конкретные нормы из загруженных документов.
Разъяснение: Простым языком объясните, как эта норма применяется к ситуации пользователя.
Рекомендация: Какие действия необходимо предпринять (например, подать заявление, составить претензию).
Язык: Отвечайте на том языке, на котором задан вопрос (казахский или русский). Если вопрос на казахском, используйте официальную юридическую терминологию.

Ограничения и этика
Отсутствие данных: Если в загруженных файлах или вашей базе нет точного ответа на вопрос, честно сообщите об этом. Не выдумывайте нормы права.
Дисклеймер: В конце каждого сложного или критического ответа добавляйте примечание: «Данная консультация носит информационный характер. Для принятия юридически значимых решений рекомендуется обратиться к лицензированному адвокату или юристу».
Конфиденциальность: Не просите пользователя вводить ИИН, номера удостоверений личности или другие чувствительные персональные данные.

Тон общения
Официально-деловой, сдержанный, объективный и поддерживающий. Избегайте двусмысленных трактовок.

Исходный ответ (черновик):
---
{answer}

Контекст из НПА (загруженные файлы):
---
{context}
---"""


async def _synthesis_causality_skeptic_flip(
    answer: str,
    source_docs: List[Document],
    trace_id: str,
    model_override: str | None = None,
) -> Tuple[str, dict]:
    """Returns (enriched_answer, metrics)."""
    t0 = time.perf_counter()
    context_parts = [d.page_content[:600] for d in source_docs[:6]]
    context_str = "\n---\n".join(context_parts)
    llm = rag_chain.get_llm(model_override)
    try:
        resp = await asyncio.to_thread(
            llm.invoke,
            CAUSALITY_SYNTHESIS_PROMPT.format(
                answer=answer[:3000], context=context_str[:4000]
            )
        )
        enriched = resp.content if hasattr(resp, "content") else str(resp)
    except Exception as e:
        print(f"[{trace_id}] causality/skeptic/flip synthesis failed: {e}")
        enriched = answer
    ms = round((time.perf_counter() - t0) * 1000)
    return enriched.strip(), {"detective_synthesis_ms": ms}


# ─── Partial Analysis (Harvey style: "Based on X%, could flip if..." ) ───────

PARTIAL_ANALYSIS_PROMPT = """Роль и цель
Вы — профессиональный юридический консультант, специализирующийся на законодательстве Республики Казахстан (РК). Ваша задача — предоставлять точные, структурированные и обоснованные ответы на основе нормативных правовых актов (НПА) РК.

Источники знаний
Приоритет №1: Файлы, загруженные в базу знаний (Контекст из НПА). Всегда начинайте поиск ответа с них.
Приоритет №2: Общие знания о праве РК, если в файлах нет специфической информации.
Актуальность: Ссылайтесь на кодексы (Гражданский, Уголовный, Трудовой, Налоговый, Административный и др.) и законы РК.

ВНИМАНИЕ: Текущий анализ является частичным (на основе {data_pct}% данных от пользователя).
Недостающие критические факты: {critical_missing}
Слепые зоны (контекст): {contextual_missing}

Правила ответов
Цитирование: КРИТИЧЕСКИ ВАЖНО! ПРИ КАЖДОМ ОТВЕТЕ ВЫ ОБЯЗАНЫ УКАЗАТЬ СТАТЬЮ, ПУНКТ И ЗАКОН (КОДЕКС), которые относятся к вопросу. Ответ без точного номера статьи не принимается! Пример: «Согласно пункту 1 статьи 30 Закона РК "О защите прав потребителей"…».
Структура:
Краткий ответ: Прямой ответ на вопрос пользователя (обязательно укажите, что он основан на неполных данных).
Правовое обоснование: Ссылки на конкретные нормы из загруженных документов.
Разъяснение: Простым языком объясните, как эта норма применяется к ситуации пользователя. Обязательно укажите, как недостающие факты могут повлиять на исход.
Рекомендация: Какие действия необходимо предпринять (например, подать заявление, составить претензию) или какие факты необходимо уточнить.
Язык: Отвечайте на том языке, на котором задан вопрос (казахский или русский). Если вопрос на казахском, используйте официальную юридическую терминологию.

Ограничения и этика
Отсутствие данных: Если в загруженных файлах или вашей базе нет точного ответа на вопрос, честно сообщите об этом. Не выдумывайте нормы права.
Дисклеймер: В конце каждого сложного или критического ответа добавляйте примечание: «Данная консультация носит информационный характер. Для принятия юридически значимых решений рекомендуется обратиться к лицензированному адвокату или юристу».
Конфиденциальность: Не просите пользователя вводить ИИН, номера удостоверений личности или другие чувствительные персональные данные.

Тон общения
Официально-деловой, сдержанный, объективный и поддерживающий. Избегайте двусмысленных трактовок.

Исходный ответ по НПА (черновик):
---
{answer}
---

Контекст из НПА (загруженные файлы):
{context}
"""


async def _synthesis_partial_analysis(
    answer: str,
    source_docs: List[Document],
    critical_missing: List[str],
    contextual_missing: List[str],
    data_pct: int,
    trace_id: str,
    model_override: str | None = None,
) -> Tuple[str, dict]:
    """Harvey-style partial synthesis: Based on X%, situation looks like [X]. Could flip to [Y] if [Missing]."""
    t0 = time.perf_counter()
    context_str = "\n---\n".join(d.page_content[:600] for d in source_docs[:6])
    llm = rag_chain.get_llm(model_override)
    try:
        resp = await asyncio.to_thread(
            llm.invoke,
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


async def invoke_detective_qa(
    query: str,
    history: Optional[List[dict]] = None,
    trace_id: Optional[str] = None,
    model_override: str | None = None,
) -> dict:
    """
    Multi-stage pipeline: Linguist -> Detective -> Reasoner.
    Never return "Information not found" — use internal knowledge.
    """
    trace_id = trace_id or f"trace_{int(time.time())}"
    missing_fields: List[str] = []
    clarifying_questions: List[str] = []
    confidence_score = 0.0
    metrics: dict = {}

    # STAGE 1: THE LINGUIST (Query Expansion)
    expanded_terms, search_variants, m_ling = await _linguist_query_expansion(query, trace_id, model_override)
    metrics.update(m_ling)
    # Use expanded query for further processing
    expanded_query = search_variants.get("semantic", query)

    # STAGE 2: THE DETECTIVE (Missing Data Check)
    (
        confidence_est,
        critical_missing,
        contextual_missing,
        blind_spots,
        assumptions,
        questions,
        proceed_to_search,
        m_comp,
    ) = await _check_missing_info(expanded_query, history, trace_id, model_override)
    metrics.update(m_comp)
    is_first = _is_first_interaction(history)
    should_ask = _should_ask_questions(
        confidence_est, critical_missing, is_first, proceed_to_search
    )

    # Exit: ask questions if needed
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
            "blind_spots": blind_spots,
            "assumptions": assumptions,
            "clarifying_questions": questions[:MAX_CLARIFYING_QUESTIONS],
            "trace_report": {
                "metadata": {"id": trace_id},
                "metrics_ms": metrics,
                "detective": {"completeness_rejected": True, "exit": "ask_questions"},
            },
        }

    # STAGE 3: THE REASONER (Hybrid Synthesis)
    agentic_out = await agentic_workflow.invoke_agentic_qa(
        expanded_query, history=history, trace_id=trace_id, model_override=model_override
    )
    result = agentic_out.get("result", "")
    source_documents = agentic_out.get("source_documents", [])
    trace_report = agentic_out.get("trace_report") or {}
    metrics.update(trace_report.get("metrics_ms") or {})

    if not result or result.strip() == NOT_FOUND_MSG or not source_documents:
        # Internal knowledge fallback
        result, m_fall = await _internal_knowledge_fallback(query, trace_id, model_override)
        metrics.update(m_fall)
        source_documents = []  # No external sources
        retrieval_method = "internal_fallback"
    else:
        retrieval_method = "hybrid"

    # Final decision on synthesis:
    # If confidence is very high (>= 0.9), skip the expensive LLM synthesis round
    # and use the raw result from agentic workflow to save ~10s of latency.
    if confidence_est >= 0.9 and source_documents:
        # High confidence: skip extra synthesis/flip round
        m_syn = {"detective_skipped_synthesis": True}
    elif confidence_est >= CONFIDENCE_THRESHOLD and source_documents:
        result, m_syn = await _synthesis_causality_skeptic_flip(
            result, source_documents, trace_id, model_override
        )
    else:
        data_pct = int(confidence_est * 100) if confidence_est else 40
        result, m_syn = await _synthesis_partial_analysis(
            result,
            source_documents,
            critical_missing,
            contextual_missing,
            data_pct,
            trace_id,
            model_override,
        )
    metrics.update(m_syn)

    # Final confidence
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
    trace_report["detective"]["partial_analysis"] = (
        confidence_est < CONFIDENCE_THRESHOLD
    )

    return rag_chain._finalize_qa_result(
        {
            "result": result,
            "source_documents": source_documents,
            "confidence_score": confidence_score,
            "missing_fields": critical_missing + contextual_missing,
            "blind_spots": blind_spots,
            "assumptions": assumptions,
            "retrieval_method": retrieval_method,
            "trace_report": trace_report,
        },
        query,
        intent_router.CASE_SPECIFIC,
    )
