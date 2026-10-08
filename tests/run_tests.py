"""Run every test from any working directory; no temporary artifacts leave tests/."""
from pathlib import Path
import sys
import unittest

sys.dont_write_bytecode = True
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))

if __name__ == "__main__":
    suite = unittest.defaultTestLoader.discover(str(Path(__file__).parent),pattern="test_*.py")
    result = unittest.TextTestRunner(verbosity=2).run(suite)
    raise SystemExit(0 if result.wasSuccessful() else 1)
