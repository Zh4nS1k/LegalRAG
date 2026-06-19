"""Kazakh-language detection helpers (extracted from rag_chain.py)."""

from __future__ import annotations

__all__ = ["_KZ_CHARS", "_KZ_COMMON_WORDS", "_is_kz_query", "_is_kz_response"]

_KZ_CHARS = set("әғқңөұүһі")

_KZ_COMMON_WORDS = (
    "және",
    "бойынша",
    "қылмыстық",
    "құрамы",
    "қылмысқа",
    "бап",
    "заң",
    "мән-жай",
    "ауырлататын",
    "жеңілдететін",
)

def _is_kz_query(query: str) -> bool:
    return any(ch in (query or "").lower() for ch in _KZ_CHARS)

def _is_kz_response(text: str) -> bool:
    t = (text or "").lower()
    if any(ch in t for ch in _KZ_CHARS):
        return True
    return any(word in t for word in _KZ_COMMON_WORDS)
