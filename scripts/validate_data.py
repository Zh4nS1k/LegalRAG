#!/usr/bin/env python3
"""
validate_data.py

A deterministic script to validate input JSON data against a predefined schema.
Returns 0 on success, non-zero on validation failure.
"""

import argparse
import json
import sys


def validate_schema(data: dict) -> bool:
    """
    Validates that the provided dictionary contains the required keys and correct types.
    For demonstration, we expect:
    {
        "id": int,
        "name": str,
        "is_active": bool
    }
    """
    try:
        if not isinstance(data.get("id"), int):
            print("Error: 'id' must be an integer.", file=sys.stderr)
            return False
        if not isinstance(data.get("name"), str) or not data["name"]:
            print("Error: 'name' must be a non-empty string.", file=sys.stderr)
            return False
        if not isinstance(data.get("is_active"), bool):
            print("Error: 'is_active' must be a boolean.", file=sys.stderr)
            return False
        return True
    except Exception as e:
        print(f"Unexpected error during validation: {e}", file=sys.stderr)
        return False


def main():
    parser = argparse.ArgumentParser(description="Validate input JSON data.")
    parser.add_argument(
        "--data", required=True, type=str, help="JSON string to validate"
    )
    args = parser.parse_args()

    try:
        parsed_data = json.loads(args.data)
    except json.JSONDecodeError as e:
        print(f"Error: Invalid JSON provided. {e}", file=sys.stderr)
        sys.exit(1)

    if validate_schema(parsed_data):
        print("Validation successful.")
        sys.exit(0)
    else:
        print("Validation failed.", file=sys.stderr)
        sys.exit(2)


if __name__ == "__main__":
    main()
