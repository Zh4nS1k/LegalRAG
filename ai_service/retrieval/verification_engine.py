import json
import logging
import time
import re
from typing import List, Optional, Tuple, Dict
from langchain_core.documents import Document
from ai_service.retrieval import rag_chain

logger = logging.getLogger("ai_service.verification_engine")

# Список кодексов РК (основные 19)
RK_CODES = {
    "ГК": "Гражданский кодекс",
    "УК": "Уголовный кодекс",
    "ТК": "Трудовой кодекс",
    "КоАП": "Кодекс об административных правонарушениях",
    "УПК": "Уголовно-процессуальный кодекс",
    "ГПК": "Гражданский процессуальный кодекс",
    "АППК": "Административный процедурно-процессуальный кодекс",
    "НК": "Налоговый кодекс",
    "БК": "Бюджетный кодекс",
    "ЭК": "Экологический кодекс",
    "ЗК": "Земельный кодекс",
    "Водный кодекс": "Водный кодекс",
    "Лесной кодекс": "Лесной кодекс",
    "Предпринимательский кодекс": "Предпринимательский кодекс",
    "Кодекс о здоровье": "Кодекс о здоровье народа и системе здравоохранения",
    "Кодекс о браке и семье": "Кодекс о браке (супружестве) и семье",
    "Таможенный кодекс": "Таможенный кодекс",
    "Кодекс о недрах": "Кодекс о недрах и недропользовании",
    "Жилищный кодекс": "Закон о жилищных отношениях (часто ищут как кодекс)",  # Технически закон, но важен
}

CLASSIFIER_PROMPT = """Ты — эксперт по законодательству Республики Казахстан. 
Твоя задача — определить, к какому из основных кодексов РК относится запрос пользователя.

Список кодексов:
{codes_list}

Запрос пользователя: "{query}"

Инструкции:
1. Выбери наиболее подходящий кодекс.
2. Если запрос касается штрафов, нарушения общественного порядка, тишины — это КоАП.
3. Если запрос касается преступлений, тюремных сроков — это УК.
4. Если запрос касается работы, увольнения, отпуска — это ТК.
5. Если запрос касается договоров, долгов между физлицами, наследства — это ГК.
6. Если запрос касается налогов — это НК.

Верни ответ СТРОГО в формате JSON:
{{
  "selected_code": "Аббревиатура (ГК/ТК/КоАП и т.д.)",
  "reason": "краткое обоснование",
  "keywords": ["слово1", "слово2"]
}}
"""


class VerificationEngine:
    def __init__(self):
        self.llm = rag_chain.get_llm()

    def classify_code(self, query: str) -> Dict:
        """Step 1 & 2: Classify and Re-verify logic."""
        codes_list = "\n".join([f"- {k}: {v}" for k, v in RK_CODES.items()])
        prompt = CLASSIFIER_PROMPT.format(codes_list=codes_list, query=query)

        try:
            resp = self.llm.invoke(prompt)
            content = resp.content if hasattr(resp, "content") else str(resp)
            # Simple JSON extraction
            start = content.find("{")
            end = content.rfind("}") + 1
            data = json.loads(content[start:end])

            # Step 2: Manual Re-verify for common mistakes
            selected = data.get("selected_code", "").upper()
            query_lower = query.lower()

            if selected == "ГК" and (
                "штраф" in query_lower
                or "полиция" in query_lower
                or "тишин" in query_lower
            ):
                logger.info(
                    "Re-verifying: Switched from ГК to КоАП based on keywords 'штраф/полиция/тишина'"
                )
                data["selected_code"] = "КоАП"

            return data
        except Exception as e:
            logger.error(f"Classification failed: {e}")
            return {"selected_code": "ГК", "reason": "default fallback", "keywords": []}

    async def targeted_fetch(
        self, code_abbr: str, query: str, keywords: List[str]
    ) -> List[Document]:
        """Step 3: Targeted Fetch from Pinecone."""
        enhanced_query = f"{query} {' '.join(keywords)}"
        vectorstore = rag_chain.get_vector_store()

        # Mapping our internal abbreviations to 'code_ru' metadata values
        code_map = {
            "ГК": "Гражданский кодекс РК",
            "УК": "Уголовный кодекс РК",
            "ТК": "Трудовой кодекс РК",
            "КоАП": "Кодекс РК об административных правонарушениях",
            "НК": "Налоговый кодекс РК",
        }

        target_code_ru = code_map.get(code_abbr, code_abbr)

        # Pinecone filter syntax for our metadata
        search_filter = {"code_ru": target_code_ru}

        # Use similarity_search with filter
        docs = vectorstore.similarity_search(enhanced_query, k=10, filter=search_filter)
        return docs

    def article_match(
        self, query: str, docs: List[Document]
    ) -> Tuple[bool, List[Document]]:
        """Step 4: Verify if key objects from query are present in found articles."""
        important_terms = []
        q_lower = query.lower()
        if "тишин" in q_lower or "ноч" in q_lower:
            important_terms.extend(["тишин", "ноч", "покой", "23", "22", "8"])
        if "увольн" in q_lower:
            important_terms.extend(["увольн", "расторж", "трудов", "договор"])
        if "сосед" in q_lower:
            important_terms.extend(["сосед", "жилищ", "обществ"])
        if "штраф" in q_lower:
            important_terms.extend(["штраф", "взыскан", "административ"])

        if not important_terms:
            return True, docs

        matched_docs = []
        for doc in docs:
            text = doc.page_content.lower()
            # If at least one term matches, we consider it a hit
            if any(term in text for term in important_terms):
                matched_docs.append(doc)

        if not matched_docs:
            logger.warning("Article Match failed: No documents contain key terms.")
            return False, []

        return True, matched_docs


POSITION_FINDER_PROMPT = """Ты — юридический аналитик. Определи правовую роль пользователя в ситуации.
Доступные роли: Работник, Работодатель, Арендатор, Арендодатель, Потерпевший, Подозреваемый, Истец, Ответчик, Гражданин.

Ситуация: "{query}"

Инструкции:
1. Если роль однозначно понятна (например, "меня уволили" -> Работник), верни её.
2. Если роль не ясна (например, "просто шум за стеной"), верни "UNKNOWN".
3. Если "UNKNOWN", сформулируй 1 короткий уточняющий вопрос.

Верни СТРОГО JSON:
{{
  "role": "Название роли или UNKNOWN",
  "clarification_needed": true/false,
  "question": "Уточните, вы являетесь тем, кто ..., или тем, кому ...?" (только если UNKNOWN)
}}
"""

CONFLICT_DETECTOR_PROMPT = """Ты — эксперт по правовым коллизиям РК. Проанализируй два набора норм и найди противоречия.
Иерархия: Конституция > Кодекс > Закон.

Запрос: "{query}"
Нормы:
{context}

Задание:
1. Если нормы противоречат друг другу (одна разрешает, другая запрещает), опиши это как "Правовая Коллизия".
2. Объясни, какая норма превалирует согласно иерархии НПА РК.
3. Если коллизий нет, напиши "Коллизий не обнаружено".

Верни СТРОГО JSON:
{{
  "has_conflict": true/false,
  "conflict_description": "описание",
  "prevailing_norm": "какая норма главнее и почему"
}}
"""


class DeductiveReasoningSkill:
    def __init__(self):
        self.llm = rag_chain.get_llm()
        self.engine = VerificationEngine()

    def find_position(self, query: str) -> Dict:
        """Sub-Skill 'Position Finder'."""
        try:
            resp = self.llm.invoke(POSITION_FINDER_PROMPT.format(query=query))
            content = resp.content if hasattr(resp, "content") else str(resp)
            start = content.find("{")
            end = content.rfind("}") + 1
            return json.loads(content[start:end])
        except Exception as e:
            logger.error(f"Position finder failed: {e}")
            return {"role": "Гражданин", "clarification_needed": False}

    def detect_conflicts(self, query: str, docs: List[Document]) -> Dict:
        """Sub-Skill 'Conflict Detector'."""
        context = "\n\n".join(
            [
                f"Источник {d.metadata.get('code_ru')}, ст.{d.metadata.get('article_number')}: {d.page_content[:300]}..."
                for d in docs
            ]
        )
        try:
            resp = self.llm.invoke(
                CONFLICT_DETECTOR_PROMPT.format(query=query, context=context)
            )
            content = resp.content if hasattr(resp, "content") else str(resp)
            start = content.find("{")
            end = content.rfind("}") + 1
            return json.loads(content[start:end])
        except Exception as e:
            logger.error(f"Conflict detector failed: {e}")
            return {"has_conflict": False, "conflict_description": "Ошибка анализа"}

    async def run_deductive_cycle(self, query: str) -> Dict:
        """Runs the full 'Sherlock' verification loop."""
        t0 = time.perf_counter()

        # 1. Classify
        classification = self.engine.classify_code(query)
        selected_code = classification["selected_code"]
        keywords = classification.get("keywords", [])

        # 2 & 3. Targeted Fetch (with loop back)
        attempts = 0
        final_docs = []
        while attempts < 2:
            docs = await self.engine.targeted_fetch(selected_code, query, keywords)

            # 4. Article Match
            success, matched_docs = self.engine.article_match(query, docs)
            if success:
                final_docs = matched_docs
                break
            else:
                attempts += 1
                logger.info(f"Loop back: attempt {attempts} failed to match articles.")
                # Ослабляем поиск для второй попытки - меньше ключевых слов
                keywords = keywords[:1] if keywords else []

        # Skills integration
        position = self.find_position(query)
        conflicts = (
            self.detect_conflicts(query, final_docs)
            if final_docs
            else {"has_conflict": False}
        )

        ms = round((time.perf_counter() - t0) * 1000)
        return {
            "deduction_path": f"{selected_code} -> {[d.metadata.get('article_number') for d in final_docs[:2]]}",
            "selected_code": selected_code,
            "full_code_name": RK_CODES.get(selected_code, selected_code),
            "docs": final_docs,
            "position": position,
            "conflicts": conflicts,
            "classification_reason": classification.get("reason"),
            "verification_ms": ms,
        }
