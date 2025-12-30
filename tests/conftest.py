"""pytest configuration to ensure the project source code can be imported.

This file adjusts the Python path at runtime so that the `src` package in the
repository root is discoverable when running tests.  Without this adjustment,
invoking pytest from the project root would not add the root itself to
``sys.path``, leading to ``ModuleNotFoundError`` when importing modules from
``src``.  By inserting the repository root at the front of ``sys.path`` we
ensure that imports such as ``from src.config import ModelConfig`` resolve
correctly.
"""

from __future__ import annotations

import sys
from pathlib import Path

# Prepend the repository root to sys.path so that the ``src`` package is
# discoverable when running tests.  The parent of this file's directory is the
# project root.
ROOT_DIR: Path = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
  sys.path.insert(0, str(ROOT_DIR))
