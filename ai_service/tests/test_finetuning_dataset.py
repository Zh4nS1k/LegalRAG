from ai_service.finetuning.dataset import DEFAULT_SYSTEM_PROMPT, build_chat_example


def test_build_chat_example_includes_context_and_sections():
    prompt, response = build_chat_example(
        {
            "instruction": "Какая норма применяется?",
            "context": "Статья 1. Пример нормы.",
            "response": "Нужно применять статью 1.",
        }
    )

    assert DEFAULT_SYSTEM_PROMPT in prompt
    assert "<|system|>" in prompt
    assert "<|user|>" in prompt
    assert "<|assistant|>" in prompt
    assert "Контекст:\nСтатья 1. Пример нормы." in prompt
    assert response == "Нужно применять статью 1."


def test_build_chat_example_requires_instruction_and_response():
    try:
        build_chat_example({"instruction": "", "response": "ok"})
        assert False, "expected ValueError for empty instruction"
    except ValueError as exc:
        assert "instruction" in str(exc)

    try:
        build_chat_example({"instruction": "ok", "response": ""})
        assert False, "expected ValueError for empty response"
    except ValueError as exc:
        assert "response" in str(exc)
