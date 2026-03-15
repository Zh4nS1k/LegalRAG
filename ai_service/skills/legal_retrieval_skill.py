# legal_retrieval_skill.py — Level 3: Skills (Standardized Processes)
# Formalizes RAG as clear sequence: Linguist -> Hybrid Retriever -> Detective.

from ai_service.retrieval import rag_chain
from ai_service.scripts.confidence_calculator import ConfidenceCalculator

GENERAL_LEGAL_CONTEXT = """
Общий обзор законодательства Республики Казахстан:
- Конституция РК является высшим законом страны.
- Уголовный кодекс регулирует преступления и наказания.
- Гражданский кодекс определяет гражданские права и обязанности.
- Трудовой кодекс регулирует трудовые отношения.
- Налоговый кодекс определяет налоговые обязательства.
- Кодекс об административных правонарушениях устанавливает административную ответственность.
Если конкретные статьи не найдены, рекомендуется обратиться к соответствующему кодексу или закону.
"""

class LegalRetrievalSkill:
    def __init__(self):
        self.confidence_calc = ConfidenceCalculator()

    def execute(self, query, trace_id):
        """Execute the skill: Linguist -> Hybrid Retriever -> Detective."""
        try:
            # Step 1: Linguist - Query Expansion
            expanded_query, search_variants = self._linguist_expansion(query, trace_id)

            # Step 2: Hybrid Retriever - BM25 + Vector Search
            docs, similarity_scores = self._hybrid_retrieval(expanded_query, trace_id)

            # Step 3: Validator - Ensure high-quality context
            confidence = self.confidence_calc.calculate_confidence(docs)
            if confidence < 0.6:
                from langchain_core.documents import Document
                general_doc = Document(page_content=GENERAL_LEGAL_CONTEXT, metadata={"source": "general_context", "score": 0.5})
                docs.append(general_doc)

            # Step 4: Detective - Match facts with sources
            matched_facts = self._detective_matching(query, docs, similarity_scores, trace_id)

            return {
                "success": True,
                "expanded_query": expanded_query,
                "documents": docs,
                "confidence": confidence,
                "matched_facts": matched_facts,
                "trace_id": trace_id
            }
        except Exception as e:
            return {
                "success": False,
                "error": str(e),
                "trace_id": trace_id
            }

    def _linguist_expansion(self, query, trace_id):
        """Step 1: Expand query with legal terminology."""
        # Simple rule-based expansion (mechanical, not LLM)
        expansions = {
            "несовершеннолетний": "работники, не достигшие восемнадцатилетнего возраста",
            "студент": "обучающийся в образовательном учреждении",
            "ночная смена": "работа в ночное время",
            "штраф": "административная ответственность",
            "преступление": "уголовное правонарушение"
        }

        expanded = query
        for informal, formal in expansions.items():
            expanded = expanded.replace(informal, formal)

        search_variants = {
            "semantic": expanded,
            "keyword": f"{expanded} статья кодекс",
            "broad": f"{expanded} законодательство РК"
        }

        return expanded, search_variants

    def _hybrid_retrieval(self, query, trace_id):
        """Step 2: Retrieve using BM25 + Vector Search."""
        retriever = rag_chain.get_retriever()
        docs = retriever.invoke(query)

        # Extract similarity scores (assuming metadata has them)
        scores = []
        for doc in docs:
            score = doc.metadata.get("score", 0.5)
            scores.append(float(score))

        return docs, scores

    def _detective_matching(self, query, docs, scores, trace_id):
        """Step 3: Match query facts with document content."""
        confidence = self.confidence_calc.calculate_confidence(scores)

        matched_facts = []
        for doc in docs:
            content = doc.page_content.lower()
            # Simple keyword matching (mechanical)
            if any(word in content for word in query.lower().split()):
                matched_facts.append(f"Matched in {doc.metadata.get('code_ru', 'Unknown')}")
            if len(matched_facts) >= 3:  # Limit matches
                break

        return confidence, matched_facts
