# tests/test_phase2_p48r_legacy_scope_gate_policy_repair.py
#
# P48R: Policy repair verification test.
# Verifies that 15 legacy P24-P38 scope-gate tests:
#   1. Still exist in their respective files.
#   2. Each repaired file imports enforce_phase_local_scope_gate_or_skip.
#   3. No audited file contains global monkeypatch patterns.
#   4. P48R scope gate: only allowed files modified relative to P48 base.
#   5. Source files (src/phase2) unchanged from P48 base.
#   6. Tool scripts (tools/phase2) unchanged from P48 base.

import subprocess

import pytest

from tests.phase2_scope_gate_utils import (
    enforce_phase_local_scope_gate_or_skip,
    get_current_git_branch,
)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

P48_BASE_COMMIT = "1de362b5e0dd7edf4bedc6ceeccc25a84f6a9de4"

P48R_EXPECTED_BRANCH = (
    "phase2/p48r-legacy-scope-gate-policy-repair-p27-to-p38-no-source-change"
)

# The 15 repaired test files (P24-P38 legacy scope gates)
REPAIRED_TEST_FILES = {
    # P24-P26 (curated suite failures also pre-existing on P47R)
    "test_p24_10_no_forbidden_files_modified": "tests/test_phase2_p24_architecture_docs.py",
    "test_p25_43_no_forbidden_files_modified": "tests/test_phase2_model_interface.py",
    "test_p26_33_no_forbidden_files_modified": "tests/test_phase2_torch_boundary.py",
    # P27-P38 (task-specified scope gates)
    "test_p27_44_scope_gate": "tests/test_phase2_fc_vae_model_skeleton.py",
    "test_p28_45_scope_gate": "tests/test_phase2_fc_vae_torch_shell.py",
    "test_p29_49_scope_gate": "tests/test_phase2_fc_vae_torch_module_stub.py",
    "test_p30_44_scope_gate": "tests/test_phase2_fc_vae_constructor_binding.py",
    "test_p31_53_scope_gate": "tests/test_phase2_fc_vae_forward_boundary.py",
    "test_p32_65_scope_gate": "tests/test_phase2_fc_vae_forward_input_batch.py",
    "test_p33_79_scope_gate": "tests/test_phase2_fc_vae_fake_input_descriptor.py",
    "test_p34_86_scope_gate": "tests/test_phase2_fc_vae_fake_input_preview.py",
    "test_p35_95_scope_gate": "tests/test_phase2_fc_vae_fake_input_flat_vector.py",
    "test_p36_96_scope_gate": "tests/test_phase2_fc_vae_flat_vector_batch_view.py",
    "test_p37_107_scope_gate": "tests/test_phase2_fc_vae_nested_batch_values.py",
    "test_p38_105_scope_gate": "tests/test_phase2_fc_vae_tensor_materialization_request.py",
}

FORBIDDEN_MONKEYPATCH_PATTERNS = [
    "subprocess.run = ",
    "original_run = subprocess.run",
    "MockCompletedProcess",
    "mock_run",
]

P48R_ALLOWED_FILES = {
    # P24-P26 repairs
    "tests/test_phase2_p24_architecture_docs.py",
    "tests/test_phase2_model_interface.py",
    "tests/test_phase2_torch_boundary.py",
    # P27-P38 repairs
    "tests/test_phase2_fc_vae_model_skeleton.py",
    "tests/test_phase2_fc_vae_torch_shell.py",
    "tests/test_phase2_fc_vae_torch_module_stub.py",
    "tests/test_phase2_fc_vae_constructor_binding.py",
    "tests/test_phase2_fc_vae_forward_boundary.py",
    "tests/test_phase2_fc_vae_forward_input_batch.py",
    "tests/test_phase2_fc_vae_fake_input_descriptor.py",
    "tests/test_phase2_fc_vae_fake_input_preview.py",
    "tests/test_phase2_fc_vae_fake_input_flat_vector.py",
    "tests/test_phase2_fc_vae_flat_vector_batch_view.py",
    "tests/test_phase2_fc_vae_nested_batch_values.py",
    "tests/test_phase2_fc_vae_tensor_materialization_request.py",
    # P48R new files
    "tests/test_phase2_p48r_legacy_scope_gate_policy_repair.py",
    "reports/PHASE_2_P48R_LEGACY_SCOPE_GATE_POLICY_REPAIR_P27_TO_P38_NO_SOURCE_CHANGE_REPORT.md",
}


# ---------------------------------------------------------------------------
# Test 1: 12 scope gate functions still exist
# ---------------------------------------------------------------------------

def test_p48r_01_all_15_scope_gate_functions_exist():
    """Verify all 15 legacy P24-P38 scope gate functions still exist in their files."""
    for fn_name, filepath in REPAIRED_TEST_FILES.items():
        with open(filepath, "r", encoding="utf-8") as f:
            content = f.read()
        assert f"def {fn_name}(" in content, (
            f"Scope gate function '{fn_name}' not found in '{filepath}'!"
        )


# ---------------------------------------------------------------------------
# Test 2: Each repaired file imports enforce_phase_local_scope_gate_or_skip
# ---------------------------------------------------------------------------

def test_p48r_02_each_repaired_file_uses_enforce():
    """Verify each repaired file imports/references enforce_phase_local_scope_gate_or_skip."""
    for fn_name, filepath in REPAIRED_TEST_FILES.items():
        with open(filepath, "r", encoding="utf-8") as f:
            content = f.read()
        assert "enforce_phase_local_scope_gate_or_skip" in content, (
            f"File '{filepath}' does not reference enforce_phase_local_scope_gate_or_skip!"
        )


# ---------------------------------------------------------------------------
# Test 3: No monkeypatch patterns in any audited file
# ---------------------------------------------------------------------------

def test_p48r_03_no_global_monkeypatch_in_repaired_files():
    """Verify no global subprocess monkeypatch exists in any P27-P38 test file."""
    for fn_name, filepath in REPAIRED_TEST_FILES.items():
        with open(filepath, "r", encoding="utf-8") as f:
            content = f.read()
        for pattern in FORBIDDEN_MONKEYPATCH_PATTERNS:
            assert pattern not in content, (
                f"Forbidden monkeypatch pattern '{pattern}' found in '{filepath}'!"
            )


# ---------------------------------------------------------------------------
# Test 4: P48R scope gate (only runs on P48R branch)
# ---------------------------------------------------------------------------

def test_p48r_04_scope_gate():
    """Enforce P48R scope: only allowed files modified relative to P48 base."""
    enforce_phase_local_scope_gate_or_skip(
        expected_branch=P48R_EXPECTED_BRANCH,
        base_commit=P48_BASE_COMMIT,
        allowed_files=P48R_ALLOWED_FILES,
        phase_label="P48R",
    )


# ---------------------------------------------------------------------------
# Test 5: Source files unchanged from P48 base
# ---------------------------------------------------------------------------

def test_p48r_05_src_phase2_files_unchanged():
    """Verify src/phase2 source files are unchanged from P48 base."""
    if get_current_git_branch() != P48R_EXPECTED_BRANCH:
        pytest.skip("P48R src/phase2 check skipped on non-P48R branch")
    res = subprocess.run(
        ["git", "diff", "--name-only", P48_BASE_COMMIT, "--", "src/phase2"],
        capture_output=True, text=True, check=True,
    )
    changed = [line.strip() for line in res.stdout.splitlines() if line.strip()]
    assert changed == [], (
        f"Source files changed from P48 base (not allowed in P48R): {changed}"
    )


# ---------------------------------------------------------------------------
# Test 6: Tool scripts unchanged from P48 base
# ---------------------------------------------------------------------------

def test_p48r_06_tools_phase2_files_unchanged():
    """Verify tools/phase2 scripts are unchanged from P48 base."""
    if get_current_git_branch() != P48R_EXPECTED_BRANCH:
        pytest.skip("P48R tools/phase2 check skipped on non-P48R branch")
    res = subprocess.run(
        ["git", "diff", "--name-only", P48_BASE_COMMIT, "--", "tools/phase2"],
        capture_output=True, text=True, check=True,
    )
    changed = [line.strip() for line in res.stdout.splitlines() if line.strip()]
    assert changed == [], (
        f"Tool scripts changed from P48 base (not allowed in P48R): {changed}"
    )


# ---------------------------------------------------------------------------
# Test 7: Shared utility uses real subprocess (no mock)
# ---------------------------------------------------------------------------

def test_p48r_07_shared_utility_real_subprocess():
    """Verify phase2_scope_gate_utils uses real subprocess, not mocks."""
    with open("tests/phase2_scope_gate_utils.py", "r", encoding="utf-8") as f:
        content = f.read()
    assert "subprocess.run(" in content
    assert "subprocess.run = " not in content
    assert "MockCompletedProcess" not in content
    assert "original_run" not in content
    assert '"git", "branch", "--show-current"' in content
    assert '"git", "diff", "--name-only"' in content

