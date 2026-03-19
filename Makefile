.PHONY: install-hooks lint test security-scan help deploy backend frontend ai_service build_vectors test-backend test-frontend test-ai_service docker-compose

help:
	@echo "Available commands:"
	@echo "  make install-hooks  - Configure git to use local .hooks directory"
	@echo "  make lint           - Run linters and formatters"
	@echo "  make test           - Run tests"
	@echo "  make security-scan  - Run security scan script"
	@echo "  make deploy         - Run ai_service, backend, and frontend concurrently"
	@echo "  make backend        - Run backend service"
	@echo "  make frontend       - Run frontend application"
	@echo "  make ai_service     - Run AI service"
	@echo "  make build_vectors  - Run vector database build"
	@echo "  make test-backend   - Run backend tests"
	@echo "  make test-frontend  - Run frontend tests"
	@echo "  make test-ai_service- Run ai_service tests"
	@echo "  make docker-compose - Run docker compose"

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

deploy:
	make -j3 backend ai_service frontend

backend:
	cd backend/legally && go run main.go

frontend:
	cd frontend/legally-app && npm start

ai_service:
	cd ai_service && (if [ -f "venv/bin/python" ]; then ./venv/bin/python main.py; elif [ -f ".venv/bin/python" ]; then ./.venv/bin/python main.py; else python3 main.py; fi)

build_vectors:
	cd ai_service && export PYTHONPATH=$$PYTHONPATH:$$(pwd) && (if [ -f "venv/bin/python" ]; then ./venv/bin/python retrieval/build_vector_db.py; elif [ -f ".venv/bin/python" ]; then ./.venv/bin/python retrieval/build_vector_db.py; else python3 retrieval/build_vector_db.py; fi)

test-backend:
	cd backend/legally && go test ./...

test-frontend:
	cd frontend/legally-app && npm test

test-ai_service:
	cd ai_service && export PYTHONPATH=$$PYTHONPATH:$$(pwd) && (if [ -f "venv/bin/pytest" ]; then ./venv/bin/pytest tests/; elif [ -f ".venv/bin/pytest" ]; then ./.venv/bin/pytest tests/; else pytest tests/; fi)

docker-compose:
	docker compose up --build
