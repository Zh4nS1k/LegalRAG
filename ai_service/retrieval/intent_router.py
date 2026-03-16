import re
import logging

logger = logging.getLogger("ai_service.intent_router")

# Categories
SOCIAL = "SOCIAL"
GENERAL_LEGAL = "GENERAL_LEGAL"
CASE_SPECIFIC = "CASE_SPECIFIC"

# Patterns for SOCIAL intent
SOCIAL_PATTERNS = [
    r"^(привет|здравствуй|добрый день|хай|прив|салам|ассалаумағалейкум)",
    r"(как дела|как поживаешь|что нового)",
    r"(кто ты|что ты умеешь|чем можешь помочь|расскажи о себе)",
    r"^(спасибо|благодарю|рахмет|ок|хорошо|понятно|спасибо за помощь)$",
    r"^(пока|до свидания|сау бол)$"
]

# Patterns for GENERAL_LEGAL (encyclopedic, definitions, general questions)
GENERAL_LEGAL_PATTERNS = [
    r"(что такое|что означает|дай определение|понятие)",
    r"(какие бывают|какие виды|перечисли|основные принципы)",
    r"(сколько налогов|какие налоги|размер мрп|размер зп)",
    r"(краткая сводка|общая информация)",
    r"(когда был принят|дата принятия|кем утвержден)"
]

def classify_intent(query: str) -> str:
    """
    Classifies the user query into SOCIAL, GENERAL_LEGAL, or CASE_SPECIFIC.
    """
    query_lower = query.lower().strip()
    
    # Check SOCIAL
    for pattern in SOCIAL_PATTERNS:
        if re.search(pattern, query_lower):
            logger.info(f"Intent classified as SOCIAL: {query}")
            return SOCIAL

    # Check GENERAL_LEGAL
    for pattern in GENERAL_LEGAL_PATTERNS:
        if re.search(pattern, query_lower):
            logger.info(f"Intent classified as GENERAL_LEGAL: {query}")
            return GENERAL_LEGAL

    # Default to CASE_SPECIFIC (full RAG + Detective)
    logger.info(f"Intent classified as CASE_SPECIFIC: {query}")
    return CASE_SPECIFIC
