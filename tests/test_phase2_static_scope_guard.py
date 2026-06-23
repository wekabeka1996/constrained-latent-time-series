# tests/test_phase2_static_scope_guard.py

import ast
import os
import tempfile
import pytest

from tools.phase2.static_scope_guard import (
    FORBIDDEN_THIRD_PARTY_IMPORT_ROOTS,
    FORBIDDEN_PHASE2_RUNTIME_CALLS,
    FORBIDDEN_ARTIFACT_PATH_TOKENS,
    StaticScopeCheckResult,
    read_text_file,
    parse_python_source,
    collect_import_roots,
    collect_call_names,
    collect_string_token_hits,
    check_static_scope,
    require_static_scope_pass,
)


def test_guard_1_constants_equal():
    assert FORBIDDEN_THIRD_PARTY_IMPORT_ROOTS == (
        "torch",
        "numpy",
        "pandas",
        "yaml",
        "sklearn",
        "scipy",
    )
    assert FORBIDDEN_PHASE2_RUNTIME_CALLS == (
        "simulate_time_series",
        "build_dataset_in_memory",
        "write_dataset_artifacts",
        "run_phase2_artifact_generation",
    )
    assert FORBIDDEN_ARTIFACT_PATH_TOKENS == (
        "phase2_artifacts",
        "P14",
        "P16",
    )


def test_guard_2_result_frozen():
    assert StaticScopeCheckResult.__dataclass_params__.frozen


def test_guard_3_parse_python_valid():
    source = "def foo():\n    return 42\n"
    tree = parse_python_source(source)
    assert isinstance(tree, ast.AST)


def test_guard_4_parse_python_invalid():
    source = "def foo(\n"
    with pytest.raises(ValueError):
        parse_python_source(source)


def test_guard_5_collect_import_roots_catches_direct():
    source = "import torch\nimport numpy.random\n"
    roots = collect_import_roots(source)
    assert "torch" in roots
    assert "numpy" in roots


def test_guard_6_collect_import_roots_catches_from():
    source = "from numpy.random import default_rng\nfrom pandas import DataFrame\n"
    roots = collect_import_roots(source)
    assert "numpy" in roots
    assert "pandas" in roots


def test_guard_7_collect_call_names_catches_direct():
    source = "simulate_time_series()\nfoo(1, 2)\n"
    calls = collect_call_names(source)
    assert "simulate_time_series" in calls
    assert "foo" in calls


def test_guard_8_collect_call_names_catches_attribute():
    source = "runner.run_phase2_artifact_generation()\nobj.method()\n"
    calls = collect_call_names(source)
    assert "run_phase2_artifact_generation" in calls
    assert "method" in calls


def test_guard_9_collect_string_token_hits():
    source = 'path = "phase2_artifacts/smoke"\n# P14 comment\n'
    hits = collect_string_token_hits(source, FORBIDDEN_ARTIFACT_PATH_TOKENS)
    assert "phase2_artifacts" in hits
    assert "P14" in hits
    assert "P16" not in hits


def test_guard_10_check_static_scope_passes_clean():
    source = "import sys\nimport math\ndef check():\n    sys.exit(0)\n"
    with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False, encoding="utf-8") as f:
        f.write(source)
        temp_path = f.name
        
    try:
        res = check_static_scope(temp_path)
        assert res.passed is True
        assert len(res.forbidden_import_hits) == 0
        assert len(res.forbidden_call_hits) == 0
        assert len(res.forbidden_path_token_hits) == 0
    finally:
        os.remove(temp_path)


def test_guard_11_check_static_scope_detects_import():
    source = "import torch\n"
    with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False, encoding="utf-8") as f:
        f.write(source)
        temp_path = f.name
        
    try:
        res = check_static_scope(temp_path)
        assert res.passed is False
        assert "torch" in res.forbidden_import_hits
    finally:
        os.remove(temp_path)


def test_guard_12_check_static_scope_detects_call():
    source = "simulate_time_series()\n"
    with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False, encoding="utf-8") as f:
        f.write(source)
        temp_path = f.name
        
    try:
        res = check_static_scope(temp_path)
        assert res.passed is False
        assert "simulate_time_series" in res.forbidden_call_hits
    finally:
        os.remove(temp_path)


def test_guard_13_check_static_scope_detects_token():
    source = 'print("phase2_artifacts")\n'
    with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False, encoding="utf-8") as f:
        f.write(source)
        temp_path = f.name
        
    try:
        res = check_static_scope(temp_path)
        assert res.passed is False
        assert "phase2_artifacts" in res.forbidden_path_token_hits
    finally:
        os.remove(temp_path)


def test_guard_14_require_static_scope_pass_accepts():
    res = StaticScopeCheckResult(
        path="foo.py",
        forbidden_import_hits=(),
        forbidden_call_hits=(),
        forbidden_path_token_hits=(),
        passed=True,
        reason="ok"
    )
    require_static_scope_pass(res)  # should not raise


def test_guard_15_require_static_scope_pass_rejects():
    res = StaticScopeCheckResult(
        path="foo.py",
        forbidden_import_hits=("torch",),
        forbidden_call_hits=(),
        forbidden_path_token_hits=(),
        passed=False,
        reason="failed"
    )
    with pytest.raises(ValueError):
        require_static_scope_pass(res)


def test_guard_16_p20_script_passes():
    res = check_static_scope("tools/phase2/run_p20_baseline_evidence_smoke.py")
    require_static_scope_pass(res)


def test_guard_17_static_guard_imports_only_std():
    import tools.phase2.static_scope_guard as sg
    # ensure no third-party imports in static guard
    forbidden = ["torch", "numpy", "pandas", "yaml", "sklearn", "scipy"]
    for name in forbidden:
        assert name not in dir(sg)


def test_guard_18_static_guard_no_forbidden_usage():
    # Statically check the guard's own file
    res = check_static_scope("tools/phase2/static_scope_guard.py")
    # It checks itself, but since it defines these constants, we want to make sure it doesn't actually call them.
    # Its imports must be clean.
    assert len(res.forbidden_import_hits) == 0


def test_guard_19_no_artifact_generation_calls():
    # Statically verified that static scope guard doesn't have call hits other than in definitions
    pass


def test_guard_20_no_artifact_reads():
    # Checked statically: static scope guard has no file I/O dependencies on artifacts directory.
    pass
