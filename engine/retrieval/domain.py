import re

_DOMAIN_KEYWORDS: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("gk", ("договор", "обязатель", "услуг", "сделк", "возмещени", "имуществ", "гражданск")),
    ("koap", ("административ", "штраф", "правонаруш", "взыскани")),
    ("uk", ("уголов", "преступ", "наказан", "санкци", "қылмыстық")),
    ("zemelnyi", ("земля", "земельн", "участок", "жер")),
    ("nalogovyi", ("налог", "салық", "деклараци", "ндс", "кпн")),
    ("trudovoi", ("труд", "работник", "работодател", "еңбек", "зарплат", "отпуск")),
    ("family", ("алименты", "брак", "семь", "ребен", "неке", "отбасы")),
)

_DOMAIN_CODE_HINTS: dict[str, tuple[str, ...]] = {
    "gk": (
        "гражданский кодекс",
        "азаматтық кодекс",
    ),
    "koap": (
        "кодекс об административных правонарушениях",
        "әкімшілік құқық бұзушылық туралы кодекс",
    ),
    "uk": (
        "уголовный кодекс",
        "қылмыстық кодекс",
    ),
    "zemelnyi": (
        "земельный кодекс",
        "жер кодексі",
    ),
    "nalogovyi": (
        "налоговый кодекс",
        "салық кодексі",
    ),
    "trudovoi": (
        "трудовой кодекс",
        "еңбек кодексі",
    ),
    "family": (
        "кодекс о браке",
        "неке және отбасы туралы кодекс",
    ),
}


def detect_domain(query: str) -> str | None:
    normalized = re.sub(r"\s+", " ", (query or "").strip().lower())
    if not normalized:
        return None

    scored: list[tuple[int, str]] = []
    for domain, keywords in _DOMAIN_KEYWORDS:
        hits = sum(1 for keyword in keywords if keyword in normalized)
        if hits:
            scored.append((hits, domain))

    if not scored:
        return None

    scored.sort(key=lambda item: (-item[0], item[1]))
    return scored[0][1]


def domain_matches_code(domain: str | None, code_name: str | None) -> bool:
    if not domain:
        return True

    normalized_code = (code_name or "").strip().lower()
    if not normalized_code:
        return False

    hints = _DOMAIN_CODE_HINTS.get(domain, ())
    return any(hint in normalized_code for hint in hints)
