"""Pytest path setup for direct test execution from the repository root."""

from __future__ import annotations

import sys
from pathlib import Path

PYTHON_PROJECT_ROOT = Path(__file__).resolve().parents[1]

if str(PYTHON_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PYTHON_PROJECT_ROOT))
