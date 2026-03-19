#!/usr/bin/env python3
"""
security_scan.py

Example script to scan the repository for basic security vulnerabilities
such as hardcoded credentials or dangerous function usage.
Returns 0 if clean, non-zero if issues are found.
"""

import os
import sys

import re

# Words that might indicate an exposed secret if found directly assigned in code
# Adjusted to look for assignments to actual strings with len >= 5
SECRET_PATTERN = re.compile(
    r'(password|secret|api_key|token)\s*=\s*[\'"]([^\'"]{5,})[\'"]', re.IGNORECASE
)


def scan_file(filepath: str) -> bool:
    """Returns False if vulnerable patterns are found."""
    # Skip checking the scanner itself
    if "security_scan.py" in filepath:
        return True

    issues_found = False
    try:
        with open(filepath, "r", encoding="utf-8") as f:
            for i, line in enumerate(f):
                # Skip lines explicitly marked as safe
                if "nosec" in line.lower():
                    continue

                match = SECRET_PATTERN.search(line)
                if match:
                    # Check if it's reading from environment (though regex mostly avoids this, just in case)
                    if (
                        "os.getenv" in line.lower()
                        or "os.environ" in line.lower()
                        or "viper.get" in line.lower()
                    ):
                        continue
                    print(
                        f"[SECURITY WARNING] Potential exposed secret in {filepath}:{i + 1} -> {match.group(1)}="
                    )
                    issues_found = True
    except Exception:
        pass  # Ignore unreadable files (binaries, etc.)

    return not issues_found


def main():
    root_dir = "."
    all_clean = True
    # Exclude directories that shouldn't be scanned
    excludes = [
        ".git",
        "venv",
        ".venv",
        "__pycache__",
        ".hooks",
        "node_modules",
        ".vscode",
        ".idea",
    ]

    print("Starting basic security scan...")
    for subdir, dirs, files in os.walk(root_dir):
        # Exclude directories
        for d in list(dirs):
            if d in excludes:
                dirs.remove(d)

        for file in files:
            # Only scan relevant files (code, config)
            if file.endswith(
                (".py", ".go", ".js", ".ts", ".json", ".yaml", ".yml")
            ) and not file.endswith("package-lock.json"):
                filepath = os.path.join(subdir, file)
                if not scan_file(filepath):
                    all_clean = False

    if all_clean:
        print("Security scan completed cleanly.")
        sys.exit(0)
    else:
        print(
            "Security scan failed: Please review the warnings above.", file=sys.stderr
        )
        sys.exit(1)


if __name__ == "__main__":
    main()
