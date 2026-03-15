# CLAUDE.md — Level 4: Instructions (Project Constitution)
# Fixed architectural rails for reliable AI engineering.

## Rule 1: Determinism first. Never use LLM for math or date calculations.

## Rule 2: Every answer must cite an Article from the RK Code found in the context.

## Rule 3: If RAG context is missing, use internal knowledge but explicitly label it as [INTERNAL_ADVISORY].

## Rule 4: Mechanical Code Only
Write mechanical code, not clever code. Reliability > Intelligence.

## Rule 5: No LLM Hallucinations
If data insufficient, return "Flip-Point: Data incomplete for conclusion."

## Rule 7: Performance Budget
Total processing must not exceed 15s.

## Rule 8: Reranking is a "Premium Skill" — use it only if the primary search is ambiguous (score < 0.7).

## Rule 9: Use ujson instead of standard json for faster serialization of large vector lists.

## Rule 7: Absolute Isolation
Hooks isolate dependencies; Scripts handle logic; Skills standardize processes.</content>
<parameter name="filePath">/home/zhanserik/Desktop/LegalRAG/CLAUDE.md