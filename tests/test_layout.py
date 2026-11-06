"""Package layout guards: descriptive package names, resolvable entry points.

The four packages are agentic_ic (formerly the bare top-level package), zerodte,
es_8am and touch_turn. Legacy names are assembled from fragments below so this
file never matches its own scan.
"""
import importlib
import importlib.util
import re
import subprocess
from pathlib import Path

import pytest
import yaml

ROOT = Path(__file__).resolve().parent.parent

_S = "src"
LEGACY_TOKENS = [_S + "_v2", _S + "_8am", _S + "_touch_turn"]
LEGACY_PHRASES = ["from " + _S + ".", "import " + _S + ".", "-m " + _S + "."]
# Quoted dotted module paths into the old bare package, e.g. mock.patch targets.
LEGACY_QUOTED = re.compile(
    r"""["']""" + _S + r"""\.(agent|services|models|strategies|utils|main)\b"""
)
LEGACY_WORD = re.compile(r"\b(" + "|".join(LEGACY_TOKENS) + r")\b")

ENTRY_POINT = re.compile(r"\bpython3?\s+-m\s+([A-Za-z_][\w.]*)")


def _repo_files():
    """Tracked plus untracked-but-not-ignored files (what a commit would carry)."""
    out = subprocess.run(
        ["git", "ls-files", "--cached", "--others", "--exclude-standard"],
        cwd=ROOT, capture_output=True, text=True, check=True,
    ).stdout
    return [ROOT / line for line in out.splitlines() if line and (ROOT / line).is_file()]


def _entry_point_docs():
    docs = [ROOT / "README.md", ROOT / "CLAUDE.md", ROOT / "zerodte" / "README.md"]
    docs += sorted((ROOT / "scripts").glob("*.sh"))
    docs += sorted(p for p in (ROOT / "deploy").rglob("*") if p.is_file())
    return docs


def test_documented_entry_points_import():
    docs = _entry_point_docs()
    assert (ROOT / "zerodte" / "README.md") in docs
    found = {}
    for doc in docs:
        assert doc.is_file(), f"documented file missing: {doc}"
        for mod in ENTRY_POINT.findall(doc.read_text()):
            found.setdefault(mod.rstrip("."), set()).add(doc.relative_to(ROOT).as_posix())
    assert found, "no `python -m` entry points found — the scan is broken"
    unresolved = {m: sorted(d) for m, d in found.items() if importlib.util.find_spec(m) is None}
    assert not unresolved, f"`python -m` targets that do not resolve: {unresolved}"


def _class_values(node):
    if isinstance(node, dict):
        for k, v in node.items():
            if k == "class" and isinstance(v, str):
                yield v
            else:
                yield from _class_values(v)
    elif isinstance(node, list):
        for item in node:
            yield from _class_values(item)


def test_config_class_paths_resolve():
    paths = []
    for cfg in sorted((ROOT / "config").glob("*.yaml")):
        with open(cfg) as f:
            paths += [(cfg.name, v) for v in _class_values(yaml.safe_load(f))]
    assert paths, "no `class:` values found in config/*.yaml — the scan is broken"
    for cfg_name, dotted in paths:
        mod_name, _, attr = dotted.rpartition(".")
        mod = importlib.import_module(mod_name)
        assert hasattr(mod, attr), f"{cfg_name}: {dotted} — module has no attribute {attr!r}"


def test_no_legacy_package_names():
    offenders = []
    for path in _repo_files():
        rel = path.relative_to(ROOT).as_posix()
        if rel.startswith("docs/superpowers/") or path.suffix not in {".py", ".yaml", ".sh", ".md"}:
            continue
        for n, line in enumerate(path.read_text(errors="replace").splitlines(), 1):
            if (LEGACY_WORD.search(line) or LEGACY_QUOTED.search(line)
                    or any(p in line for p in LEGACY_PHRASES)):
                offenders.append(f"{rel}:{n}: {line.strip()[:100]}")
    assert not offenders, f"{len(offenders)} legacy package references:\n" + "\n".join(offenders[:50])


@pytest.mark.parametrize("pkg", ["agentic_ic", "zerodte", "es_8am", "touch_turn"])
def test_descriptive_packages_exist(pkg):
    assert importlib.util.find_spec(pkg) is not None
    assert (ROOT / pkg / "__init__.py").is_file()
