"""Prompt builder for constructing strict legal RAG prompts."""
import os
from typing import List, Dict
from config import settings
from models.schemas import Chunk

# Maximum tokens for the entire context block (chunks + question)
_MAX_CONTEXT_TOKENS = int(os.getenv("MAX_CONTEXT_TOKENS", "3000"))
# Hard cap on question length in tokens before truncation kicks in
_MAX_QUESTION_TOKENS = 600

SYSTEM_PROMPT = """You are a precise legal assistant for Kazakhstani law. Answer legal
questions strictly and only from the provided document excerpts.

ABSOLUTE RULES:
1. Use ONLY information explicitly present in the provided chunks.
   Do NOT use your training knowledge, general legal principles, or
   any information not stated in the chunks.
2. If the chunks do not contain enough information to answer: output
   exactly "Контекстте жауап жоқ." — nothing else, no explanation,
   no "however", no partial answer.
3. CHUNK CITATIONS ARE MANDATORY AND MUST BE EXACT.
   Only cite [Chunk N] if the specific article or information
   appears VERBATIM in that chunk's text.
   If you cannot identify the exact chunk, write [Chunk ?] —
   never guess a chunk number.
4. Do NOT speculate, infer, or extrapolate beyond what is written.

REQUIRED ANSWER STRUCTURE (follow exactly):
---
ПРАВОВОЕ ОСНОВАНИЕ:
[List each applicable law/article from the chunks, one per line,
 with chunk citation. Format: • Статья X, Закон Y — [краткое описание] [Chunk N]
 Only list articles that appear word-for-word in the provided
 chunks. If an article is not in any chunk, do not list it,
 even if you know it from your training data.]

ОТВЕТ НА ВОПРОС:
[Direct answer to the user's question based only on the chunks.
 2-5 sentences maximum. Every sentence must end with [Chunk N].]

РЕКОМЕНДУЕМЫЕ ДЕЙСТВИЯ:
[Concrete next steps the person should take, derived ONLY from chunk
 content. Numbered list. If chunks don't specify actions, omit this
 section entirely.]
---

Language: Answer in the same language as the question (Russian or Kazakh)."""


class PromptBuilder:
    """Constructs strict RAG prompts with context chunks."""

    def __init__(self, max_tokens: int = _MAX_CONTEXT_TOKENS):
        """Initializes the PromptBuilder."""
        self.max_tokens = max_tokens
        self.chars_per_token = 4  # ~1 token ≈ 4 chars
        self.system_prompt = SYSTEM_PROMPT
        # Reserve ~25% of budget for system prompt + question overhead
        self._system_chars = len(SYSTEM_PROMPT)

    def _count_tokens_approx(self, text: str) -> int:
        return len(text) // self.chars_per_token

    def _truncate_to_tokens(self, text: str, max_tokens: int) -> str:
        max_chars = max_tokens * self.chars_per_token
        if len(text) <= max_chars:
            return text
        # Truncate to word boundary
        truncated = text[:max_chars]
        last_space = truncated.rfind(" ")
        if last_space > max_chars * 0.8:
            truncated = truncated[:last_space]
        return truncated + "..."

    def _format_chunk_header(self, i: int, chunk: Chunk) -> str:
        source = chunk.metadata.get("source", "")
        law_name = (chunk.metadata.get("law_name", "") or
                    chunk.metadata.get("document", "") or
                    chunk.metadata.get("title", "") or source)
        article = (chunk.metadata.get("article", "") or
                   chunk.metadata.get("article_number", ""))

        header = f"[Chunk {i}]"
        if law_name:
            header += f" | {law_name}"
        if article:
            header += f" | {article}"
        header += f" (relevance: {chunk.score:.3f})"
        if chunk.metadata.get("keyword_boost"):
            header += " ⬆️ keyword match"

        return header + "\n"

    def build(self, question: str, chunks: List[Chunk]) -> Dict[str, str]:
        """Builds system and user prompts with hard token-capped context."""

        # FIX 4: Truncate extremely long questions to prevent HTTP 413
        question_tokens = self._count_tokens_approx(question)
        if question_tokens > _MAX_QUESTION_TOKENS:
            question = self._truncate_to_tokens(question, _MAX_QUESTION_TOKENS)
            question += " [вопрос сокращён]"

        # Budget for chunks: max_tokens minus overhead for question + template
        question_chars = len(question)
        overhead_chars = self._system_chars + question_chars + 200
        max_chunk_chars = (self.max_tokens * self.chars_per_token) - overhead_chars
        max_chunk_chars = max(500, max_chunk_chars)  # always allow at least a bit

        formatted_chunks = []
        chars_used = 0

        for i, chunk in enumerate(chunks, 1):
            header = self._format_chunk_header(i, chunk)
            header_chars = len(header)
            remaining = max_chunk_chars - chars_used - header_chars

            if remaining < 50:
                break  # no budget left

            chunk_text = self._truncate_to_tokens(chunk.text, remaining // self.chars_per_token)
            formatted_chunks.append(f"{header}{chunk_text}")
            chars_used += header_chars + len(chunk_text)

            if chars_used >= max_chunk_chars:
                break

        chunks_str = "\n\n".join(formatted_chunks) if formatted_chunks else "Контекст табылмады."

        user_prompt = (
            f"Context documents:\n{chunks_str}\n\n"
            f"Legal question:\n{question}\n\n"
            "Answer strictly based on the context above:"
        )

        return {
            "system": self.system_prompt,
            "user": user_prompt,
        }
