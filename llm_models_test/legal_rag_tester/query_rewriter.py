"""Query rewriter for legal retrieval optimization."""
from config import settings
from llm_client import LLMClient

class QueryRewriter:
    def __init__(self):
        self.llm_client = LLMClient()
        self.model = settings.query_rewriter_model
        
        self.system_prompt = """You are a legal search query optimizer for Kazakhstani law.
Given a user's legal question (which may be long and conversational),
extract a concise retrieval query of 1-3 sentences that:
1. Identifies the specific legal domain (contract law, consumer protection,
   family law, labor law, administrative law, etc.)
2. Names the specific legal act or code if mentioned or implied
   (ГК РК, ЗРК о защите прав потребителей, Трудовой кодекс, etc.)
3. States the core legal question in precise legal terminology
4. Removes personal narrative, emotional context, and irrelevant details

Output ONLY the optimized query. No explanation. No preamble."""

    def rewrite(self, question: str) -> str:
        """Extracts a short focused retrieval query."""
        if not settings.enable_query_rewriting:
            return question
            
        user_prompt = f"Original question:\n{question}\n\nOptimized retrieval query:"
        try:
            result = self.llm_client.call(self.model, self.system_prompt, user_prompt)
            if result.error:
                return question
            
            # The client might have already stripped think tags if we added that
            return result.answer.strip()
        except Exception:
            return question
