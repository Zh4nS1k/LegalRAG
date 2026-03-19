import pytest
from ai_service.retrieval.intent_router import (
    classify_intent,
    SOCIAL,
    GENERAL_LEGAL,
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
        ("Сосед шумит после 11 вечера, что делать?", CASE_SPECIFIC),
        ("Работодатель не платит зарплату уже два месяца", CASE_SPECIFIC),
        ("Меня уволили без предупреждения, это законно?", CASE_SPECIFIC),
    ],
)
def test_classify_intent(query, expected_intent):
    assert classify_intent(query) == expected_intent
