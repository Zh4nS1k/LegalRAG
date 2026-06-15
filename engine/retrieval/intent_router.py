from __future__ import annotations

import json
import logging
import os
import re
from dataclasses import asdict, dataclass
from typing import Iterable, Optional

logger = logging.getLogger("engine.intent_router")

# --- CATEGORIES ---
SOCIAL = "SOCIAL"
GENERAL_LEGAL = "GENERAL_LEGAL"
PROCEDURAL = "PROCEDURAL"
CASE_SPECIFIC = "CASE_SPECIFIC"

ROUTER_LABELS = (SOCIAL, GENERAL_LEGAL, PROCEDURAL, CASE_SPECIFIC)


@dataclass(frozen=True)
class RoutingDecision:
    intent: str
    confidence: float
    strategy: str
    reason: str
    scores: dict[str, float]

    def as_dict(self) -> dict[str, object]:
        payload = asdict(self)
        payload["confidence"] = round(float(self.confidence), 4)
        payload["scores"] = {
            key: round(float(value), 4) for key, value in self.scores.items()
        }
        return payload


# --- PATTERNS ---

SOCIAL_PATTERNS = [
    r"^(привет|здравствуй|добрый день|добрый вечер|хай|прив|салам|ассалаумағалейкум|уалейкүм|сәлем|сәлеметсіз бе|кеш жарық|қайырлы күн)",
    r"(как дела|как поживаешь|что нового|как жизнь|қалайсың|жағдай қалай|қалай жағдайың|не хабар|амансың ба)",
    r"(кто ты|что ты умеешь|чем можешь помочь|расскажи о себе|сен кімсің|не істей аласың|қандай көмек бересің|функцияларың қандай)",
    r"^(спасибо|благодарю|рахмет|көп рахмет|благодарочка|ок|хорошо|понятно|түсінікті|жақсы|келістік|мақұл|болды)",
    r"^(пока|до свидания|прощай|сау бол|көріскенше|сау болыңыз|қайырлы түн)",
]

GENERAL_LEGAL_PATTERNS = [
    r"(что такое|что означает|дай определение|понятие|термин|мағынасы не|түсінік бер|анықтама|не болып табылады)",
    r"(какие бывают|какие виды|перечисли|түрлері қандай|тізім|қандай бар|классификация|жіктелуі)",
    r"(размер мрп|размер аек|размер зп|мзп|минимальная зарплата|етж|айлық есептік көрсеткіш|прожиточный минимум|күнкөріс деңгейі)",
    r"(когда был принят|дата принятия|кем утвержден|қашан қабылданды|кім бекітті|цифровой кодекс|жаңа конституция|референдум 2026)",
    r"(структура кодекса|сколько статей|неше бап|қандай бөлім|тарау|глава|параграф)",
    r"(основные принципы|суть закона|негізгі принциптер|заңның мақсаты)",
]

PROCEDURAL_PATTERNS = [
    r"(как подать|как составить|как написать|как оформить|процедура|қалай тапсырады|қалай жазу керек|рәсімдеу|жолы қандай)",
    r"(нужен образец|шаблон|форма|пример заявления|арыз үлгісі|талап қою үлгісі|келісімшарт үлгісі)",
    r"(какие документы нужны|список документов|пакет документов|қандай құжаттар қажет|құжаттар тізімі)",
    r"(госпошлина|сколько платить|тариф|цена услуги|мемлекеттік баж|қанша төлеймін|төлем мөлшері|салық мөлшері)",
    r"(сроки подачи|исковая давность|сколько ждать|мерзімі қандай|уақыты|өтініш қанша уақыт қаралады)",
    r"(через егов|на портале|е-өтініш|цон|халыққа қызмет көрсету орталығы|егов арқылы)",
    r"(как получить эцп|биометрия|цифровой id|электрондық қолтаңба|цифрлық идентификация)",
]

_SOCIAL_KEYWORDS = (
    "привет",
    "сәлем",
    "спасибо",
    "рахмет",
    "кто ты",
    "сен кімсің",
    "пока",
)

_GENERAL_KEYWORDS = (
    "что такое",
    "понятие",
    "определение",
    "виды",
    "размер мрп",
    "статья",
    "бап",
    "принцип",
)

_PROCEDURAL_KEYWORDS = (
    "как подать",
    "какие документы",
    "шаблон",
    "образец",
    "госпошлина",
    "сроки",
    "цон",
    "е-өтініш",
    "егов",
)

_LLM_ROUTER_ENABLED = os.environ.get("LEGAL_RAG_ROUTER_ENABLE_LLM", "1") != "0"
_LLM_ROUTER_MIN_CONFIDENCE = float(
    os.environ.get("LEGAL_RAG_ROUTER_MIN_CONFIDENCE", "0.72")
)
_LLM_ROUTER_TEMPERATURE = float(
    os.environ.get("LEGAL_RAG_ROUTER_TEMPERATURE", "0.0")
)


def _normalize_query(query: str) -> str:
    return re.sub(r"\s+", " ", (query or "").strip().lower())


def _match_count(query: str, patterns: Iterable[str]) -> int:
    return sum(1 for pattern in patterns if re.search(pattern, query))


def _keyword_hits(query: str, keywords: Iterable[str]) -> int:
    return sum(1 for keyword in keywords if keyword in query)


def _score_rules(query: str) -> dict[str, float]:
    normalized = _normalize_query(query)
    return {
        SOCIAL: _match_count(normalized, SOCIAL_PATTERNS) * 2.2
        + _keyword_hits(normalized, _SOCIAL_KEYWORDS) * 0.7,
        GENERAL_LEGAL: _match_count(normalized, GENERAL_LEGAL_PATTERNS) * 1.8
        + _keyword_hits(normalized, _GENERAL_KEYWORDS) * 0.65,
        PROCEDURAL: _match_count(normalized, PROCEDURAL_PATTERNS) * 1.9
        + _keyword_hits(normalized, _PROCEDURAL_KEYWORDS) * 0.75,
        CASE_SPECIFIC: 0.35,
    }


def _best_label(scores: dict[str, float]) -> tuple[str, float, str]:
    ordered = sorted(scores.items(), key=lambda item: item[1], reverse=True)
    best_label, best_score = ordered[0]
    second_score = ordered[1][1] if len(ordered) > 1 else 0.0
    if best_score <= 0:
        return CASE_SPECIFIC, 0.35, "no_rules_matched"

    if best_label == CASE_SPECIFIC:
        confidence = min(0.92, 0.55 + best_score)
        return CASE_SPECIFIC, confidence, "default_case_specific"

    gap = max(0.0, best_score - second_score)
    confidence = min(0.98, 0.52 + (best_score * 0.12) + (gap * 0.18))
    return best_label, confidence, "rule_ensemble"


def _history_summary(history: Optional[list[dict]]) -> str:
    if not history:
        return ""
    tail = history[-4:]
    pairs = []
    for item in tail:
        role = (item.get("role") or "unknown").strip()
        content = re.sub(r"\s+", " ", str(item.get("content") or "").strip())
        if content:
            pairs.append(f"{role}: {content[:160]}")
    return "\n".join(pairs)


def _llm_route(query: str, history: Optional[list[dict]], heuristic: RoutingDecision) -> RoutingDecision:
    if not _LLM_ROUTER_ENABLED:
        return heuristic

    try:
        from engine.retrieval.rag_chain import get_llm

        llm = get_llm()
    except Exception as exc:
        logger.debug("LLM router unavailable, using heuristic decision: %s", exc)
        return heuristic

    prompt = (
        "Ты классификатор маршрутизации юридических запросов для LegalRAG.\n"
        "Верни только JSON без Markdown с ключами: intent, confidence, reason.\n"
        f"Допустимые intent: {', '.join(ROUTER_LABELS)}.\n"
        "Правила:\n"
        "- SOCIAL: приветствие, благодарность, small talk, идентичность ассистента.\n"
        "- GENERAL_LEGAL: определения, виды, статусы, статические сведения о законах.\n"
        "- PROCEDURAL: как подать, документы, сроки, госпошлина, шаблоны, eGov/eOtinish.\n"
        "- CASE_SPECIFIC: спор, применение закона к фактам, неоднозначная ситуация.\n"
        f"\nВопрос: {query}\n"
    )
    history_block = _history_summary(history)
    if history_block:
        prompt += f"\nИстория:\n{history_block}\n"

    try:
        response = llm.invoke(prompt)
        text = response.content if hasattr(response, "content") else str(response)
        parsed = json.loads(text)
        intent = str(parsed.get("intent") or "").strip().upper()
        confidence = float(parsed.get("confidence", heuristic.confidence))
        reason = str(parsed.get("reason") or "llm_router")
        if intent not in ROUTER_LABELS:
            raise ValueError(f"Unsupported intent from LLM router: {intent}")
        confidence = max(0.0, min(0.99, confidence))
        if confidence < heuristic.confidence and intent == heuristic.intent:
            return heuristic
        return RoutingDecision(
            intent=intent,
            confidence=confidence,
            strategy="llm_router",
            reason=reason,
            scores=heuristic.scores,
        )
    except Exception as exc:
        logger.debug("LLM router parse failed, using heuristic decision: %s", exc)
        return heuristic


def classify_intent_with_confidence(
    query: str, history: Optional[list[dict]] = None
) -> RoutingDecision:
    scores = _score_rules(query)
    intent, confidence, reason = _best_label(scores)
    heuristic = RoutingDecision(
        intent=intent,
        confidence=confidence,
        strategy="rule_ensemble",
        reason=reason,
        scores=scores,
    )

    if heuristic.confidence >= _LLM_ROUTER_MIN_CONFIDENCE:
        logger.info(
            "routing decision: %s",
            json.dumps(heuristic.as_dict(), ensure_ascii=False),
        )
        return heuristic

    routed = _llm_route(query, history, heuristic)
    logger.info("routing decision: %s", json.dumps(routed.as_dict(), ensure_ascii=False))
    return routed


def classify_intent(query: str) -> str:
    return classify_intent_with_confidence(query).intent

# ==========================================
# 21-Intent Rule-Based Prompt Router
# ==========================================

QUESTION_TYPES = {
    "crime_qualification": ["грозит", "уголовн", "преступлен", "состав преступ"],
    "crime_punishment": ["наказан", "штраф", "срок", "лишен", "арест", "краж", "хищен", "ограблен"],
    "crime_aggravating": ["крупн", "групп", "особо опасн", "квалифицирующ"],
    "crime_mitigation": ["смягчающ", "избежат", "освобожден", "прекратить дело"],
    "article_lookup": ["статья", "ст.", "пункт", "часть", "что говорит", "содержание статьи"],
    "contract_dispute": ["договор", "расторжен", "сделк", "неустойк"],
    "consumer_rights": ["товар", "вернут", "потребител", "гарантий"],
    "property_damage": ["ущерб", "ответственн", "возмещен", "вред"],
    "inheritance": ["наследств", "наследник", "завещание"],
    "business_registration": ["регистрац", "тоо", "открыть бизнес"],
    "tax_penalty": ["налог", "ндс", "налогов", "уклонение", "налогоплательщик"],
    "labor_rights": ["увольнен", "трудов", "работник", "работодател"],
    "salary_compensation": ["зарплат", "декретн", "компенсаци", "отпуск"],
    "social_benefits": ["пенси", "социальн", "пособие", "выплат"],
    "labor_discipline": ["прогул", "дисциплин", "выговор"],
    "family_law": ["развод", "алимент", "раздел имуществ", "брак"],
    "filing_complaint": ["жалоб", "подать иск", "куда обращаться", "подача иск"],
    "statute_limitations": ["срок давност", "истек срок"],
    "evidence": ["доказательств", "свидетел"],
    "enforcement": ["взыскать", "исполнение решения", "судебный пристав"],
    "administrative": ["коап", "административн", "протокол"],
}

def detect_question_type(query: str) -> str:
    q = query.lower()
    PRIORITY_ORDER = [
        "article_lookup", "crime_punishment", "crime_aggravating",
        "crime_mitigation", "crime_qualification", "filing_complaint",
        "statute_limitations", "labor_rights", "salary_compensation",
        "labor_discipline", "social_benefits", "family_law",
        "contract_dispute", "consumer_rights", "property_damage",
        "inheritance", "business_registration", "tax_penalty",
        "evidence", "enforcement", "administrative",
    ]
    for qtype in PRIORITY_ORDER:
        keywords = QUESTION_TYPES[qtype]
        if any(kw in q for kw in keywords):
            return qtype
    return "crime_punishment"  # fallback для неоднозначных уголовных вопросов
