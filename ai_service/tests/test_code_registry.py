from ai_service.core.code_registry import CODE_NAMES, get_code_name


def test_code_registry_returns_canonical_names():
    assert get_code_name("documents/constitution.txt") == (
        "Конституция РК",
        "ҚР Конституциясы",
    )
    assert "civil_code.txt" in CODE_NAMES

