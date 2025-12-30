"""pytest config: ensure repo root is on sys.path for importing ``src``."""

from __future__ import annotations

import sys
from pathlib import Path

# Prepend the repository root to sys.path so that the ``src`` package is
# discoverable when running tests.  The parent of this file's directory is the
# project root.
ROOT_DIR: Path = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
  sys.path.insert(0, str(ROOT_DIR))
