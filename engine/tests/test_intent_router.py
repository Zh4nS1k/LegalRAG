import pytest
from engine.retrieval.intent_router import (
    classify_intent_with_confidence,
    classify_intent,
    SOCIAL,
    GENERAL_LEGAL,
    PROCEDURAL,
    CASE_SPECIFIC,
)


@pytest.mark.parametrize(
    "query,expected_intent",
    [
        ("Привет!", SOCIAL),
        ("Как дела?", SOCIAL),
        ("Кто ты такой?", SOCIAL),
        ("Спасибо за помощь", SOCIAL),
        ("Пока", SOCIAL),
        ("Что такое административное правонарушение?", GENERAL_LEGAL),
        ("Дай определение понятию 'налог'", GENERAL_LEGAL),
        ("Какие бывают виды договоров?", GENERAL_LEGAL),
        ("Размер МРП на 2024 год", GENERAL_LEGAL),
        ("Как подать исковое заявление в суд?", PROCEDURAL),
        ("Какие документы нужны для подачи заявления?", PROCEDURAL),
        ("Сосед шумит после 11 вечера, что делать?", CASE_SPECIFIC),
        ("Работодатель не платит зарплату уже два месяца", CASE_SPECIFIC),
        ("Меня уволили без предупреждения, это законно?", CASE_SPECIFIC),
    ],
)
def test_classify_intent(query, expected_intent):
    assert classify_intent(query) == expected_intent


def test_classify_intent_returns_confidence():
    decision = classify_intent_with_confidence("Как подать исковое заявление в суд?")

    assert decision.intent == PROCEDURAL
    assert 0.0 <= decision.confidence <= 0.99
    assert decision.strategy in {"rule_ensemble", "llm_router"}
    assert decision.reason
