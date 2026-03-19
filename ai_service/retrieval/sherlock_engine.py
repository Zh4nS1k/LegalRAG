import json
import logging
import time
from typing import List, Optional, Tuple, Dict, Any
from langchain_core.documents import Document
from ai_service.retrieval import rag_chain

logger = logging.getLogger("ai_service.sherlock_engine")

# Официальные названия 19 кодексов РК для фильтрации в Pinecone
RK_CODES_OFFICIAL = {
    "ГК": "Гражданский кодекс РК",
    "УК": "Уголовный кодекс РК",
    "ТК": "Трудовой кодекс РК",
    "КоАП": "Кодекс РК об административных правонарушениях",
    "УПК": "Уголовно-процессуальный кодекс РК",
    "ГПК": "Гражданский процессуальный кодекс РК",
    "АППК": "Административный процедурно-процессуальный кодекс РК",
    "НК": "Налоговый кодекс РК",
    "БК": "Бюджетный кодекс РК",
    "ЭК": "Экологический кодекс РК",
    "ЗК": "Земельный кодекс РК",
    "Водный кодекс": "Водный кодекс РК",
    "Лесной кодекс": "Лесной кодекс РК",
    "Предпринимательский кодекс": "Предпринимательский кодекс РК",
    "Кодекс о здоровье": "Кодекс РК о здоровье народа и системе здравоохранения",
    "Кодекс о браке и семье": "Кодекс РК о браке (супружестве) и семье",
    "Таможенный кодекс": "Кодекс РК о таможенном регулировании",
    "Кодекс о недрах": "Кодекс РК о недрах и недропользовании",
    "Социальный кодекс": "Социальный кодекс РК",
}

SHERLOCK_CLASSIFIER_PROMPT = """Ты — эксперт по законодательству Республики Казахстан. 
Твоя задача — определить, к каким из 19 кодексов РК относится запрос пользователя.
Выбери от 1 до 3 наиболее релевантных кодексов.

Список кодексов:
{codes_list}

Запрос пользователя: "{query}"

Инструкции:
1. Выбери наиболее подходящие кодексы (макс 3).
2. Обоснуй выбор каждого кодекса.
3. Извлеки ключевые юридические термины (факты) из запроса (например: возраст, время суток, тип договора, статус лица).

Верни ответ СТРОГО в формате JSON:
{{
  "selected_codes": ["Аббревиатура1", "Аббревиатура2"],
  "reasoning": "краткое обоснование выбора",
  "facts": {{
      "age": "значение или null",
      "time": "значение или null",
      "role_hint": "подсказка о роли",
      "action": "основное действие"
  }}
}}
"""

SHERLOCK_SYSTEM_MESSAGE = """Ты — судебный аудитор системы 'Шерлок'. 
Твоя цель — провести независимый дедуктивный аудит ситуации на основе предоставленных статей законов.
Не просто отвечай пользователю, а анализируй логические связи.

Структура твоего анализа:
1. Путь дедукции: [Кодекс] -> [Статья] (объясни, почему выбраны именно они).
2. Анализ позиции: Кто пользователь в этой ситуации (роль) и какие у него права/обязанности.
3. Вердикт по коллизиям: Если нормы разных законов пересекаются или спорят, укажи, какая превалирует (Конституция > Кодекс > Закон). Если спора нет — так и напиши.

Будь сух, точен и детерминирован.
"""


class SherlockEngine:
    def __init__(self):
        self.llm = rag_chain.get_llm()
        self.vectorstore = None

    def _get_vs(self):
        if self.vectorstore is None:
            self.vectorstore = rag_chain.get_vector_store()
        return self.vectorstore

    async def classify_and_validate(self, query: str) -> Dict[str, Any]:
        """Stage 1 & 2: Classify and Validate."""
        codes_text = "\n".join([f"- {k}: {v}" for k, v in RK_CODES_OFFICIAL.items()])
        prompt = SHERLOCK_CLASSIFIER_PROMPT.format(codes_list=codes_text, query=query)

        try:
            resp = self.llm.invoke(prompt)
            content = resp.content if hasattr(resp, "content") else str(resp)
            start = content.find("{")
            end = content.rfind("}") + 1
            data = json.loads(content[start:end])

            # Stage 2: Validation Loop (Internal Logic)
            selected = [c.upper() for c in data.get("selected_codes", [])]
            q_lower = query.lower()

            # Принудительная корректировка типичных ошибок
            validated_codes = []
            for code in selected:
                if (code == "ЗК" or code == "ГК") and (
                    "зарплат" in q_lower
                    or "увольн" in q_lower
                    or "работодател" in q_lower
                ):
                    logger.warning(
                        f"Sherlock Validation: Dropping {code} for labor query, adding ТК"
                    )
                    if "ТК" not in validated_codes:
                        validated_codes.append("ТК")
                elif code == "ГК" and ("штраф" in q_lower or "полиция" in q_lower):
                    logger.info("Sherlock Validation: Adding КоАП for penalty query")
                    if "КоАП" not in validated_codes:
                        validated_codes.append("КоАП")
                    validated_codes.append(code)
                else:
                    validated_codes.append(code)

            if not validated_codes:
                validated_codes = ["ГК"]  # Fallback

            data["selected_codes"] = validated_codes[:3]
            return data
        except Exception as e:
            logger.error(f"Sherlock Classification failed: {e}")
            return {"selected_codes": ["ГК"], "reasoning": "fallback", "facts": {}}

    async def stage_3_targeted_fetch(
        self, codes: List[str], query: str
    ) -> List[Document]:
        """Stage 3: Targeted Fetch within selected codes."""
        vs = self._get_vs()
        target_names = [RK_CODES_OFFICIAL.get(c, c) for c in codes]

        # Pinecone $or filter for codes
        if len(target_names) == 1:
            search_filter = {"code_ru": target_names[0]}
        else:
            search_filter = {"$or": [{"code_ru": name} for name in target_names]}

        docs = vs.similarity_search(query, k=15, filter=search_filter)
        return docs

    def stage_4_fact_check(
        self, docs: List[Document], facts: Dict
    ) -> Tuple[bool, List[Document]]:
        """Stage 4: Fact Check - matching articles with query facts."""
        if not docs:
            return False, []

        important_markers = []
        if facts.get("time"):
            # Если в фактах есть время, ищем статьи про время (например 23:00, ночное время)
            important_markers.extend(["время", "ноч", "23", "22", "8"])

        # Если маркеров нет, полагаемся на семантику (True)
        if not important_markers:
            return True, docs

        matched = []
        for d in docs:
            text = d.page_content.lower()
            if any(m in text for m in important_markers):
                matched.append(d)

        if not matched:
            return False, docs[:5]  # Return some docs but signal failure

        return True, matched

    async def run_sherlock_loop(self, query: str) -> Dict[str, Any]:
        """The core Sherlock Retry Loop (Stages 1-4)."""
        t0 = time.perf_counter()

        # Stage 1 & 2
        classification = await self.classify_and_validate(query)
        selected_codes = classification["selected_codes"]
        facts = classification.get("facts", {})

        # Stage 3 & 4 with Retry
        final_docs = []
        attempt = 0
        current_query = query

        while attempt < 2:
            docs = await self.stage_3_targeted_fetch(selected_codes, current_query)
            success, checked_docs = self.stage_4_fact_check(docs, facts)

            if success:
                final_docs = checked_docs
                break
            else:
                attempt += 1
                logger.info(
                    f"Sherlock Stage 4 failed, retry {attempt} with broader query"
                )
                # Расширяем запрос для второй попытки
                action = facts.get("action", "")
                current_query = f"{query} {action}" if action else query
                final_docs = (
                    docs  # Fallback to original docs if 2nd attempt also 'fails'
                )

        # Skills execution
        skill = SherlockAnalysisSkill(self.llm)
        position_data = skill.identify_position(query)
        conflict_data = skill.detect_conflicts(query, final_docs)

        # Generate Final Deduction Output via LLM
        deduction_report = self._generate_report(
            query, final_docs, position_data, conflict_data
        )

        return {
            "deductive_output": deduction_report,
            "meta": {
                "codes": selected_codes,
                "position": position_data,
                "conflicts": conflict_data,
                "time_ms": round((time.perf_counter() - t0) * 1000),
            },
        }

    def _generate_report(
        self, query: str, docs: List[Document], position: Dict, conflicts: Dict
    ) -> str:
        """Generates the final structured text output for Sherlock block."""
        context_str = "\n".join(
            [
                f"- {d.metadata.get('code_ru')}, ст.{d.metadata.get('article_number')}: {d.page_content[:200]}..."
                for d in docs[:3]
            ]
        )

        prompt = f"{SHERLOCK_SYSTEM_MESSAGE}\n\n"
        prompt += f"Ситуация: {query}\n"
        prompt += f"Выявленная позиция: {position}\n"
        prompt += f"Найденные конфликты: {conflicts}\n"
        prompt += f"Релевантные статьи:\n{context_str}\n\n"
        prompt += "Проведи финальный аудит:"

        try:
            resp = self.llm.invoke(prompt)
            return resp.content if hasattr(resp, "content") else str(resp)
        except Exception:
            return "Ошибка генерации отчета Шерлока."


class SherlockAnalysisSkill:
    def __init__(self, llm):
        self.llm = llm

    def identify_position(self, query: str) -> Dict:
        """Step 5 & 6: Client Position & Interactive Bridge."""
        prompt = f"""Проанализируй ситуацию и определи юридическую роль пользователя.
Ситуация: "{query}"

Верни JSON:
{{
  "role": "Работник/Работодатель/Потерпевший/и др.",
  "needs_clarification": true/false,
  "clarification_question": "вопрос, если роль не ясна"
}}
"""
        try:
            resp = self.llm.invoke(prompt)
            content = resp.content if hasattr(resp, "content") else str(resp)
            start = content.find("{")
            end = content.rfind("}") + 1
            return json.loads(content[start:end])
        except Exception:
            return {"role": "Гражданин", "needs_clarification": False}

    def detect_conflicts(self, query: str, docs: List[Document]) -> Dict:
        """Step 7: Conflict Detection."""
        if not docs:
            return {"has_conflict": False}

        context = "\n".join(
            [
                f"[{d.metadata.get('code_ru')}, ст.{d.metadata.get('article_number')}]: {d.page_content[:300]}"
                for d in docs[:5]
            ]
        )

        prompt = f"""Найди противоречия (коллизии) между следующими нормами для ситуации: "{query}"
Нормы:
{context}

Иерархия: Конституция > Кодекс > Закон.

Верни JSON:
{{
  "has_conflict": true/false,
  "conflict_block": "Описание коллизии и какая норма сильнее",
  "affected_articles": ["ст. X", "ст. Y"]
}}
"""
        try:
            resp = self.llm.invoke(prompt)
            content = resp.content if hasattr(resp, "content") else str(resp)
            start = content.find("{")
            end = content.rfind("}") + 1
            return json.loads(content[start:end])
        except Exception:
            return {"has_conflict": False}
