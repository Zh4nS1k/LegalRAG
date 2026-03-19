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
Hooks isolate dependencies; Scripts handle logic; Skills standardize processes.

## Rule 10: Anti-Bias & Intent Routing
- Do not request contract data if the question concerns general legal theory or administrative offenses.
- If the question is general ("what is..."), answer based on RK NPAs without requesting clarifications.
- **Rule 4 (Deductive Priority)**: Первичный ответ дает базу, «Шерлок-цикл» дает глубину. Они не должны противоречить друг другу.
- **Rule 5 (Validation Loop)**: Запрещено выдавать статью, если она не прошла `semantic_match` с темой запроса.
- **Rule 6 (Cross-Law)**: При обнаружении противоречий в законах, приоритет отдается Кодексу над Законом, и Конституции над Кодексом.
- **Rule 12 (Sherlock Constitution)**:
    - **Isolation**: Данные «Шерлока» не должны смешиваться с основным ответом. Это дополнительный блок `deductive_output`.
    - **No Hallucinated Codes**: Использовать только официальные названия 19 кодексов РК.
    - **Conflict Hierarchy**: Всегда указывать, какая норма сильнее (Конституция > Кодекс > Закон).

## Rule 11: Chat Session Isolation
- Each message in a chat belongs to an isolated context (chat_id).
- Mentioning facts from other sessions is a critical bug. Use only current session history.

## Rule 12: AI-Engineering Architecture Compliance
- **Hooks**: Always use `.hooks/` (pre-commit, post-merge, pre-push) to ensure automated checks run instantly and cannot be bypassed.
- **Scripts**: Delegate complex/deterministic logic to testing scripts in `scripts/` (e.g. `validate_data.py`). Never use LLM reasoning for fixed validations.
- **Skills**: When initiating complex tasks (like creating a PR), use standardized instruction sets stored in `skills/` (e.g. `skills/create_pr/steps.yaml`).
- **Prompts**: Store template prompts in `prompts/`. Prefer executing scripts over generating new instructions.
- **Style Rules**: Always use type hints in Python. Use `camelCase` for JS function names.
- **Security Check**: Never store secrets in code. `scripts/security_scan.py` must pass on every commit.