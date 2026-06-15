.PHONY: install-hooks lint test security-scan help deploy backend frontend engine build_vectors test-backend test-frontend test-engine docker-compose

help:
	@echo "Available commands:"
	@echo "  make install-hooks  - Configure git to use local .hooks directory"
	@echo "  make lint           - Run linters and formatters"
	@echo "  make test           - Run tests"
	@echo "  make security-scan  - Run security scan script"
	@echo "  make deploy         - Run engine, backend, and frontend concurrently"
	@echo "  make backend        - Run backend service"
	@echo "  make frontend       - Run frontend application"
	@echo "  make engine         - Run AI engine"
	@echo "  make build_vectors  - Run vector database build"
	@echo "  make test-backend   - Run backend tests"
	@echo "  make test-frontend  - Run frontend tests"
	@echo "  make test-engine    - Run engine tests"
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
	make -j3 backend engine frontend

backend:
	cd server && go run main.go

frontend:
	cd client && npm start

engine:
	cd engine && (if [ -f "venv/bin/python" ]; then ./venv/bin/python main.py; elif [ -f ".venv/bin/python" ]; then ./.venv/bin/python main.py; else python3 main.py; fi)

build_vectors:
	cd engine && export PYTHONPATH=$$PYTHONPATH:$$(pwd) && (if [ -f "venv/bin/python" ]; then ./venv/bin/python retrieval/build_vector_db.py; elif [ -f ".venv/bin/python" ]; then ./.venv/bin/python retrieval/build_vector_db.py; else python3 retrieval/build_vector_db.py; fi)

test-backend:
	cd server && go test ./tests/... ./api/controllers/... -v

test-frontend:
	cd client && npm test -- --watchAll=false

test-engine:
	cd engine && export PYTHONPATH=$$PYTHONPATH:$$(pwd) && (if [ -f "venv/bin/pytest" ]; then ./venv/bin/pytest tests/; elif [ -f ".venv/bin/pytest" ]; then ./.venv/bin/pytest tests/; else pytest tests/; fi)

docker-compose:
	docker compose up --build
