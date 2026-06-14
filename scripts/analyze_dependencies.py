#!/usr/bin/env python3
"""
Analyze Python dependencies and imports to identify unused libraries.
"""

import ast
import os
import re
import sys
from collections import defaultdict, Counter
from pathlib import Path
from typing import Dict, List, Set, Tuple


def extract_imports_from_file(filepath: str) -> Tuple[Set[str], Set[str]]:
    """Extract imports from a Python file."""
    with open(filepath, 'r', encoding='utf-8', errors='ignore') as f:
        content = f.read()

    imports = set()
    from_imports = set()

    try:
        tree = ast.parse(content)

        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    imports.add(alias.name.split('.')[0])  # Get top-level package
            elif isinstance(node, ast.ImportFrom):
                if node.module:
                    module_name = node.module.split('.')[0]
                    from_imports.add(module_name)
    except (SyntaxError, UnicodeDecodeError):
        # Fallback to regex for problematic files
        lines = content.split('\n')
        for line in lines:
            line = line.strip()
            if line.startswith('import '):
                parts = line[7:].split()
                if parts:
                    package = parts[0].split('.')[0]
                    imports.add(package)
            elif line.startswith('from '):
                parts = line[5:].split(' import ')
                if len(parts) == 2:
                    package = parts[0].split('.')[0]
                    from_imports.add(package)

    return imports, from_imports


def parse_requirements(filepath: str) -> Dict[str, str]:
    """Parse requirements.txt file."""
    requirements = {}

    if not os.path.exists(filepath):
        return requirements

    with open(filepath, 'r') as f:
        for line in f:
            line = line.strip()

            # Skip comments and empty lines
            if not line or line.startswith('#') or line.startswith('-r'):
                continue

            # Parse package name (handle versions, git repos, etc.)
            if '>=' in line:
                package = line.split('>=')[0].strip()
            elif '==' in line:
                package = line.split('==')[0].strip()
            elif '~=' in line:
                package = line.split('~=')[0].strip()
            elif '@' in line:
                # Git or URL dependencies
                continue
            else:
                package = line.strip()

            # Clean up package name
            package = re.sub(r'[\[\]<>=].*', '', package)
            package = package.strip()

            if package:
                requirements[package.lower()] = package

    return requirements


def get_standard_library_modules() -> Set[str]:
    """Get set of Python standard library modules."""
    import sys
    try:
        import sys.stdlib_module_names
        return set(sys.stdlib_module_names)
    except ImportError:
        # Common standard library modules
        common_stdlib = {
            'os', 'sys', 're', 'json', 'time', 'datetime', 'math', 'collections',
            'itertools', 'functools', 'pathlib', 'typing', 'logging', 'hashlib',
            'random', 'string', 'inspect', 'ast', 'threading', 'asyncio',
            'multiprocessing', 'subprocess', 'csv', 'pickle', 'sqlite3',
            'urllib', 'ssl', 'socket', 'email', 'html', 'xml', 'uuid',
            'decimal', 'fractions', 'statistics', 'copy', 'pprint', 'textwrap',
            'base64', 'binascii', 'getpass', 'secrets', 'tempfile', 'io',
            'contextlib', 'dataclasses', 'enum', 'weakref', 'abc', 'types',
            'importlib', 'pkgutil', 'platform', 'sysconfig', 'traceback',
            'warnings', 'argparse', 'configparser', 'getopt', 'optparse',
            'shutil', 'glob', 'fnmatch', 'linecache', 'codecs', 'locale',
            'tokenize', 'keyword', 'symbol', 'token', 'dis', 'opcode',
            'pydoc', 'doctest', 'unittest', 'test', 'builtins', '__future__'
        }
        return common_stdlib


def analyze_project(project_path: str) -> Dict:
    """Analyze dependencies in a project."""
    project_path = Path(project_path)

    # Find requirements files
    requirements_files = list(project_path.glob('**/requirements.txt'))
    if not requirements_files:
        print(f"No requirements.txt found in {project_path}")
        return {}

    # Use main requirements file
    req_file = requirements_files[0]
    if len(requirements_files) > 1:
        print(f"Found multiple requirements.txt files, using: {req_file}")

    requirements = parse_requirements(str(req_file))

    # Find Python files
    python_files = list(project_path.glob('**/*.py'))

    # Collect imports
    all_imports = set()
    all_from_imports = set()
    import_counts = Counter()
    file_imports = defaultdict(set)

    for py_file in python_files:
        if 'venv' in str(py_file) or '.venv' in str(py_file):
            continue

        imports, from_imports = extract_imports_from_file(str(py_file))

        all_imports.update(imports)
        all_from_imports.update(from_imports)

        for imp in imports:
            import_counts[imp] += 1
            file_imports[imp].add(str(py_file.relative_to(project_path)))

        for imp in from_imports:
            import_counts[imp] += 1
            file_imports[imp].add(str(py_file.relative_to(project_path)))

    # Combine imports
    all_packages = all_imports.union(all_from_imports)

    # Get standard library
    stdlib = get_standard_library_modules()

    # Filter out standard library
    third_party_packages = {pkg for pkg in all_packages if pkg not in stdlib}

    # Compare with requirements
    required_packages = set(requirements.keys())
    imported_packages = {pkg.lower() for pkg in third_party_packages}

    # Find unused requirements
    unused_requirements = required_packages - imported_packages

    # Find missing imports (packages used but not in requirements)
    missing_requirements = imported_packages - required_packages

    # Map lowercase back to original names
    unused_original = {requirements.get(pkg, pkg) for pkg in unused_requirements}
    missing_original = {pkg for pkg in third_party_packages
                       if pkg.lower() in missing_requirements}

    return {
        'total_python_files': len(python_files),
        'requirements_file': str(req_file.relative_to(project_path)),
        'total_requirements': len(requirements),
        'total_imports': len(all_packages),
        'third_party_imports': len(third_party_packages),
        'unused_requirements': sorted(unused_original),
        'missing_requirements': sorted(missing_original),
        'import_counts': dict(import_counts.most_common(20)),
        'file_imports': {k: sorted(v)[:3] for k, v in file_imports.items()
                        if k.lower() in unused_requirements or k.lower() in missing_requirements},
        'all_requirements': list(requirements.values()),
        'all_imports': sorted(third_party_packages)
    }


def print_report(report: Dict):
    """Print dependency analysis report."""
    print("=" * 80)
    print("DEPENDENCY ANALYSIS REPORT")
    print("=" * 80)
    print(f"Project: {Path.cwd().name}")
    print(f"Python Files Analyzed: {report['total_python_files']}")
    print(f"Requirements File: {report['requirements_file']}")
    print(f"Total Requirements: {report['total_requirements']}")
    print(f"Third-party Imports Found: {report['third_party_imports']}")
    print()

    print("=" * 80)
    print("🚨 UNUSED REQUIREMENTS (Can potentially be removed):")
    print("=" * 80)
    if report['unused_requirements']:
        for req in report['unused_requirements']:
            print(f"  • {req}")
            if req.lower() in report['file_imports']:
                files = report['file_imports'][req.lower()]
                if files:
                    print(f"    ↳ Imported in: {', '.join(files[:3])}")
    else:
        print("  No unused requirements found!")

    print()
    print("=" * 80)
    print("⚠️  MISSING REQUIREMENTS (Used but not in requirements.txt):")
    print("=" * 80)
    if report['missing_requirements']:
        for req in report['missing_requirements']:
            print(f"  • {req}")
            if req.lower() in report['file_imports']:
                files = report['file_imports'][req.lower()]
                if files:
                    print(f"    ↳ Imported in: {', '.join(files[:3])}")
    else:
        print("  No missing requirements found!")

    print()
    print("=" * 80)
    print("📊 MOST FREQUENT IMPORTS:")
    print("=" * 80)
    for pkg, count in report['import_counts'].items():
        print(f"  {pkg:20s} → {count:3d} files")

    print()
    print("=" * 80)
    print("📋 ALL REQUIREMENTS:")
    print("=" * 80)
    for req in sorted(report['all_requirements']):
        status = "❌ UNUSED" if req in report['unused_requirements'] else "✓ USED"
        print(f"  {status:10s} {req}")


def main():
    """Main function."""
    import argparse

    parser = argparse.ArgumentParser(description='Analyze Python dependencies')
    parser.add_argument('--path', default='.', help='Project path')

    args = parser.parse_args()

    project_path = Path(args.path).resolve()

    if not project_path.exists():
        print(f"Error: Path {project_path} does not exist")
        sys.exit(1)

    print(f"Analyzing project: {project_path}")
    report = analyze_project(str(project_path))

    if report:
        print_report(report)

        # Save report
        import json
        with open('dependency_report.json', 'w') as f:
            json.dump(report, f, indent=2)
        print(f"\n📄 Full report saved to: dependency_report.json")
    else:
        print("No analysis results")


if __name__ == "__main__":
    main()