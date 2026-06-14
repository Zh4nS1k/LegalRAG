"""Tests for OpenRouter attribution headers."""

from ai_service.core import config


def test_build_openrouter_default_headers_uses_env(monkeypatch):
    monkeypatch.setenv("OPENROUTER_SITE_URL", "https://legalrag.kz")
    monkeypatch.setenv("OPENROUTER_APP_NAME", "LegalRAG-Tester")
    config.OPENROUTER_SITE_URL = "https://legalrag.kz"
    config.OPENROUTER_APP_NAME = "LegalRAG-Tester"

    headers = config.build_openrouter_default_headers()

    assert headers["HTTP-Referer"] == "https://legalrag.kz"
    assert headers["X-Title"] == "LegalRAG-Tester"
