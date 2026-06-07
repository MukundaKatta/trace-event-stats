"""Test package for trace-event-stats.

Ensures the ``src`` layout package is importable when running the suite
with ``python -m unittest discover -s tests`` without installing the
project first (no editable/pip install required).
"""

import sys
from pathlib import Path

_SRC = Path(__file__).resolve().parent.parent / "src"
if _SRC.is_dir() and str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))
