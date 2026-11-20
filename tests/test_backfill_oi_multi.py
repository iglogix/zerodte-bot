# tests/test_backfill_oi_multi.py
import importlib.util
import sys
from pathlib import Path

# load the script module by path (it lives in scripts/, not a package)
_spec = importlib.util.spec_from_file_location(
    "backfill_oi_multi",
    str(Path(__file__).resolve().parent.parent / "scripts" / "backfill_oi_multi.py"))
_mod = importlib.util.module_from_spec(_spec)
sys.modules["backfill_oi_multi"] = _mod
_spec.loader.exec_module(_mod)

def test_chunked_splits_under_limit():
    syms = [f"S{i}" for i in range(4100)]
    chunks = list(_mod.chunked(syms, _mod.MAX_SYMBOLS_PER_REQ))
    assert all(len(c) <= _mod.MAX_SYMBOLS_PER_REQ for c in chunks)
    assert sum(len(c) for c in chunks) == 4100
    # 4100 / 1900 -> 3 chunks (1900, 1900, 300)
    assert len(chunks) == 3
    # round-trip preserves order and contents
    flat = [s for c in chunks for s in c]
    assert flat == syms

def test_max_symbols_under_databento_limit():
    assert _mod.MAX_SYMBOLS_PER_REQ <= 2000
