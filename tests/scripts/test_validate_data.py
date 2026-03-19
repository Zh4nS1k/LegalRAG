import subprocess
import json
import pytest
import os

SCRIPT_PATH = os.path.join(
    os.path.dirname(__file__), "..", "..", "scripts", "validate_data.py"
)


def test_validate_data_success():
    valid_data = {"id": 123, "name": "Test User", "is_active": True}
    result = subprocess.run(
        ["python3", SCRIPT_PATH, "--data", json.dumps(valid_data)],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0
    assert "Validation successful" in result.stdout


def test_validate_data_invalid_json():
    result = subprocess.run(
        ["python3", SCRIPT_PATH, "--data", "{invalid_json"],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 1
    assert "Error: Invalid JSON provided" in result.stderr


def test_validate_data_schema_mismatch():
    invalid_data = {"id": "not-an-int", "name": "", "is_active": 1}
    result = subprocess.run(
        ["python3", SCRIPT_PATH, "--data", json.dumps(invalid_data)],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 2
    assert "Error: 'id' must be an integer" in result.stderr
