"""
tests/conftest.py
=================
Pytest shared configuration and fixtures.

This file is discovered automatically by pytest. It ensures the repository
root is on sys.path so `import backend.*` works without manual path hacking
in individual test files.
"""

import sys
from pathlib import Path

# Add the repository root to sys.path so tests can import `backend.*`
# This is the correct approach: configure once here, not in each test file.
_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))
