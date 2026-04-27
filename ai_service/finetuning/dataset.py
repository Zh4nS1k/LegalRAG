from __future__ import annotations

from typing import Any


DEFAULT_SYSTEM_PROMPT = (
    "Ты юридический ассистент по праву Республики Казахстан. "
    "Отвечай в профессиональном юридическом стиле, опирайся только на предоставленный контекст, "
    "различай действующую и историческую редакцию нормы, а при нехватке данных прямо укажи это."
)


def build_chat_example(record: dict[str, Any]) -> tuple[str, str]:
    instruction = str(record.get("instruction") or "").strip()
    context = str(record.get("context") or "").strip()
    response = str(record.get("response") or "").strip()
    system_prompt = str(record.get("system") or DEFAULT_SYSTEM_PROMPT).strip()

    if not instruction:
        raise ValueError("Record must contain a non-empty 'instruction'")
    if not response:
        raise ValueError("Record must contain a non-empty 'response'")

    user_parts = [instruction]
    if context:
        user_parts.append("Контекст:\n" + context)

    prompt = (
        "<|system|>\n"
        f"{system_prompt}\n"
        "<|user|>\n"
        f"{chr(10).join(user_parts)}\n"
        "<|assistant|>\n"
    )
    return prompt, response
