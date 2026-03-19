import re
import logging

logger = logging.getLogger("ai_service.intent_router")

# --- CATEGORIES ---
SOCIAL = "SOCIAL"
GENERAL_LEGAL = "GENERAL_LEGAL"  # Definitions, theory, static facts
PROCEDURAL = "PROCEDURAL"  # "How-to", fees, templates, deadlines
CASE_SPECIFIC = "CASE_SPECIFIC"  # Deep RAG analysis (The "Detective")

# --- PATTERNS ---

# 1. SOCIAL (Greetings, Gratitude, Identity) - ~30 variants
SOCIAL_PATTERNS = [
    # Greetings (RU/KZ)
    r"^(привет|здравствуй|добрый день|добрый вечер|хай|прив|салам|ассалаумағалейкум|уалейкум|сәлем|сәлеметсіз бе|кеш жарық|қайырлы күн)",
    # Status/Small Talk
    r"(как дела|как поживаешь|что нового|как жизнь|қалайсың|жағдай қалай|қалай жағдайың|не хабар|амансың ба)",
    # Bot Identity
    r"(кто ты|что ты умеешь|чем можешь помочь|расскажи о себе|сен кімсің|не істей аласың|қандай көмек бересің|функцияларың қандай)",
    # Gratitude
    r"^(спасибо|благодарю|рахмет|көп рахмет|благодарочка|ок|хорошо|понятно|түсінікті|жақсы|келістік|мақұл|болды)",
    # Farewells
    r"^(пока|до свидания|прощай|сау бол|көріскенше|сау болыңыз|қайырлы түн)",
]

# 2. GENERAL_LEGAL (Theory, Statistics, Definitions) - ~45 variants
GENERAL_LEGAL_PATTERNS = [
    # Definitions
    r"(что такое|что означает|дай определение|понятие|термин|мағынасы не|түсінік бер|анықтама|не болып табылады)",
    # Types and Categories
    r"(какие бывают|какие виды|перечисли|түрлері қандай|тізім|қандай бар|классификация|жіктелуі)",
    # Values (MCI/MRP/Salary) - Critical for 2026
    r"(размер мрп|размер аек|размер зп|мзп|минимальная зарплата|етж|айлық есептік көрсеткіш|прожиточный минимум|күнкөріс деңгейі)",
    # History/Adoption (Digital Code 2026 / New Constitution)
    r"(когда был принят|дата принятия|кем утвержден|қашан қабылданды|кім бекітті|цифровой кодекс|жаңа конституция|референдум 2026)",
    # Structure
    r"(структура кодекса|сколько статей|неше бап|қандай бөлім|тарау|глава|параграф)",
    # General Principles
    r"(основные принципы|суть закона|негізгі принциптер|заңның мақсаты)",
]

# 3. PROCEDURAL (Logistics, Fees, Templates, Steps) - ~50 variants
PROCEDURAL_PATTERNS = [
    # Action/How-to
    r"(как подать|как составить|как написать|как оформить|процедура|қалай тапсырады|қалай жазу керек|рәсімдеу|жолы қандай)",
    # Templates
    r"(нужен образец|шаблон|форма|пример заявления|арыз үлгісі|талап қою үлгісі|келісімшарт үлгісі)",
    # Documents
    r"(какие документы нужны|список документов|пакет документов|қандай құжаттар қажет|құжаттар тізімі)",
    # Fees and Duties
    r"(госпошлина|сколько платить|тариф|цена услуги|мемлекеттік баж|қанша төлеймін|төлем мөлшері|салық мөлшері)",
    # Deadlines and Status
    r"(сроки подачи|исковая давность|сколько ждать|мерзімі қандай|уақыты|өтініш қанша уақыт қаралады)",
    # Location/Platform (eGov/eOtinish)
    r"(через егов|на портале|е-өтініш|цон|халыққа қызмет көрсету орталығы|егов арқылы)",
    # Digital Identity (2026 focus)
    r"(как получить эцп|биография через биометрию|цифровой id|электрондық қолтаңба|цифрлық идентификация)",
]


def classify_intent(query: str) -> str:
    """
    Categorizes the user query to route it to the correct RAG logic.
    """
    q = query.lower().strip()

    # Check SOCIAL
    if any(re.search(p, q) for p in SOCIAL_PATTERNS):
        return SOCIAL

    # Check PROCEDURAL (Check this before General because it's more actionable)
    if any(re.search(p, q) for p in PROCEDURAL_PATTERNS):
        return PROCEDURAL

    # Check GENERAL_LEGAL
    if any(re.search(p, q) for p in GENERAL_LEGAL_PATTERNS):
        return GENERAL_LEGAL

    # Default: CASE_SPECIFIC (Complex factual retrieval)
    return CASE_SPECIFIC
