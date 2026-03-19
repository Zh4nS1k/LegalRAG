#!/usr/bin/env bash

# format_code.sh
# Deterministic bash script to format Python and Go code in the project.

echo "Running Black (Python formatting)..."
# Assuming virtual environment is at ./venv
if [ -f "./venv/bin/black" ]; then
    ./venv/bin/black --exclude "(\.venv|\.git)" ai_service/ scripts/ tests/
else
    # Fallback to global black if available
    black --exclude "(\.venv|\.git)" ai_service/ scripts/ tests/ || echo "Black is not installed. Skipping Python formatting."
fi

echo "Running Flake8 (Python linting)..."
if [ -f "./venv/bin/flake8" ]; then
    ./venv/bin/flake8 --exclude=".venv,.git" ai_service/ scripts/ tests/
else
    flake8 --exclude=".venv,.git" ai_service/ scripts/ tests/ || echo "Flake8 is not installed. Skipping Python linting."
fi

echo "Running go fmt (Go formatting)..."
if [ -d "backend/legally" ]; then
    cd backend/legally && go fmt ./...
else
    echo "Go backend directory not found. Skipping Go formatting."
fi

echo "Formatting completed."
exit 0
