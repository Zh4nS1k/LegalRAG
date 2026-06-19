"""Answer quality scoring via Groq (high RPM, no quota wall)."""
import time
from openai import OpenAI
from config import settings
from logger import pipeline_logger


SCORER_SYSTEM_PROMPT = """You are a legal answer quality evaluator. Score the given answer
on a scale of 0-10 based on these criteria:

10 — Perfect: directly answers the question, cites correct legal
     articles from context, structured clearly, no hallucination
7-9 — Good: answers the question, correct law cited, minor gaps
4-6 — Partial: some correct info but incomplete or missing key articles
1-3 — Poor: vague, misses the point, or cites wrong laws
0   — Refused or completely wrong

Respond with ONLY: <score>|<one sentence reason>
Example: 8|Correctly cites Article 35 and 42-4 but misses Article 634."""

NO_ANSWER = "Контекстте жауап жоқ."


class AnswerScorer:
    """Scores LLM answers using a Groq-hosted judge model."""

    def __init__(self):
        # Use OpenRouter directly
        self.client = OpenAI(
            api_key=settings.openrouter_api_key,
            base_url=settings.openrouter_base_url,
            timeout=20,
            default_headers={
                "HTTP-Referer": settings.openrouter_site_url,
                "X-Title": settings.openrouter_app_name,
            },
        )

    def score(self, question: str, chunks_context: str,
              answer: str, q_id: str = "unknown") -> tuple[int | None, str]:
        """Returns (score 0-10, one-line reason) or (None, error_str) on failure.
        
        IMPORTANT: None means the scorer failed — it does NOT mean the answer is bad.
        0 means the answer was truly wrong/refused.
        """
        # Auto-score no-answer responses
        if answer.strip() == NO_ANSWER:
            return 0, "Model determined context did not contain answer."

        user_prompt = (
            f"Question: {question}\n\n"
            f"Context chunks:\n{chunks_context[:2000]}\n\n"
            f"Answer to evaluate:\n{answer[:1500]}\n\n"
            f"Score:"
        )

        try:
            response = self.client.chat.completions.create(
                model=settings.scorer_model,
                messages=[
                    {"role": "system", "content": SCORER_SYSTEM_PROMPT},
                    {"role": "user", "content": user_prompt},
                ],
                temperature=0.0,
                max_tokens=60,
                timeout=15,
            )
            content = response.choices[0].message.content
            if content is None:
                return None, "Scorer returned None content"
            raw = content.strip()

            # Parse "8|Correctly cites Article 35..."
            parts = raw.split("|", 1)
            try:
                score_val = int(parts[0].strip())
                reason = parts[1].strip() if len(parts) > 1 else raw
                return max(0, min(10, score_val)), reason
            except (ValueError, IndexError):
                return None, f"Invalid scorer output: {raw[:80]}"

        except Exception as e:
            # Return None — not 0 — so 0 in results means "truly bad answer"
            pipeline_logger.log_warning(
                f"[scorer] Q#{q_id} failed: {type(e).__name__}: {str(e)[:80]}"
            )
            return None, f"Scorer error: {str(e)[:100]}"
