from ai_service.retrieval.rag_chain import (
    CRIMINAL_PROMPT,
    LEGAL_REASONING_GUIDANCE,
    RANGE_PROMPT,
    UNIVERSAL_PROMPT,
)


def _render_prompt(prompt) -> str:
    return prompt.format(
        chat_history="",
        context="Источник: test\nКодекс: Тестовый кодекс\nСтатья: 1\nТекст: Норма",
        input="Какая статья подходит к ситуации?",
    )


def test_legal_reasoning_guidance_contains_cot_steps():
    guidance = LEGAL_REASONING_GUIDANCE

    assert "ключевые юридически значимые факты" in guidance
    assert "проанализируй только те нормы" in guidance
    assert "сопоставь каждый существенный факт с диспозицией" in guidance
    assert "итоговый вывод" in guidance


def test_universal_prompt_requires_structured_legal_sections():
    rendered = _render_prompt(UNIVERSAL_PROMPT)

    assert "Ключевые юридические факты:" in rendered
    assert "Анализ норм:" in rendered
    assert "Сопоставление фактов и нормы:" in rendered
    assert "Вывод:" in rendered


def test_criminal_and_range_prompts_include_reasoning_sequence():
    criminal_rendered = _render_prompt(CRIMINAL_PROMPT)
    range_rendered = _render_prompt(RANGE_PROMPT)

    assert (
        "факты -> анализ нормы -> сопоставление с диспозицией -> вывод"
        in criminal_rendered
    )
    assert "факты вопроса -> анализ нормы -> сопоставление -> вывод" in range_rendered
