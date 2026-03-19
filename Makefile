.PHONY: install-hooks lint test security-scan help

help:
	@echo "Available commands:"
	@echo "  make install-hooks  - Configure git to use local .hooks directory"
	@echo "  make lint           - Run linters and formatters"
	@echo "  make test           - Run tests"
	@echo "  make security-scan  - Run security scan script"

install-hooks:
	git config core.hooksPath .hooks
	chmod +x .hooks/* scripts/* 2>/dev/null || true
	@echo "Hooks installed and configured."

lint:
	./scripts/format_code.sh

test:
	pytest tests/

security-scan:
	./scripts/security_scan.py
