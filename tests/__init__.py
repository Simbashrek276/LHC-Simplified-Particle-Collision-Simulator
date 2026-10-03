"""Test suite for the LHC collision simulator.

Run every test from the project root with:

    python -m unittest

or, for one line per test:

    python -m unittest -v

The tests use only the Python standard library plus numpy, so nothing extra
needs installing beyond requirements.txt.
"""

import sys
from pathlib import Path

# Make the project root importable however the tests are launched (from the
# command line, from inside tests/, or from the VS Code test explorer).
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
