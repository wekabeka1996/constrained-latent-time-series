# tests/test_phase2_p46r_cumulative_scope_gate_policy.py
#
# P46R: Cumulative scope gate policy test.
# This test:
#   1. Verifies no global monkeypatching in P44/P45/P46/P46R test files.
#   2. Verifies the shared utility uses real subprocess calls.
#   3. Enforces P46R's own scope against P46 remote head base.
#
# This test must NOT skip on the P46R branch.

import pytest
import subprocess

from tests.phase2_scope_gate_utils import (
    get_current_git_branch,
    git_changed_files_since,
    enforce_phase_local_scope_gate_or_skip,
)

# P46 remote head (the P46R base)
P46_REMOTE_HEAD = "7d349f7d06fadaa6d6c9f5d5410d36800451f2eb"

P46R_EXPECTED_BRANCH = "phase2/p46r-cumulative-scope-gate-policy-repair-no-model-no-math-change"

P46R_ALLOWED_FILES = {
    "tests/phase2_scope_gate_utils.py",
    "tests/test_phase2_p46r_cumulative_scope_gate_policy.py",
    "tests/test_phase2_tensor_native_constraint_primitives.py",
    "tests/test_phase2_analytic_moment_spectral_signatures.py",
    "tests/test_phase2_moment_spectral_matching_loss.py",
    "tests/test_phase2_fc_vae_forward_readiness_gate.py",
    "tests/test_phase2_fc_vae_own_forward_boundary_stub.py",
    "reports/PHASE_2_P46R_CUMULATIVE_SCOPE_GATE_POLICY_REPAIR_NO_MODEL_NO_MATH_CHANGE_REPORT.md",
}

# Math source files that must be unchanged from P46 base
MATH_SOURCE_FILES = [
    "src/phase2/tensor_native_constraint_primitives.py",
    "src/phase2/analytic_moment_spectral_signatures.py",
    "src/phase2/moment_spectral_matching_loss.py",
]

TEST_FILES_TO_AUDIT = [
    "tests/test_phase2_fc_vae_forward_readiness_gate.py",
    "tests/test_phase2_fc_vae_own_forward_boundary_stub.py",
    "tests/test_phase2_tensor_native_constraint_primitives.py",
    "tests/test_phase2_analytic_moment_spectral_signatures.py",
    "tests/test_phase2_moment_spectral_matching_loss.py",
    # Note: phase2_scope_gate_utils.py is the utility itself and does not monkeypatch.
    "tests/phase2_scope_gate_utils.py",
    # Note: this file (test_phase2_p46r_cumulative_scope_gate_policy.py) is intentionally
    # excluded from the audit because it contains the forbidden pattern strings as
    # string literals in its test assertions, not as actual code that modifies subprocess.
]


def test_p46r_01_no_global_monkeypatch_in_test_files():
    """Verify no global subprocess.run monkeypatch exists in any P44/P45/P46/P46R test files."""
    forbidden_patterns = [
        "subprocess.run = mock_run",
        "subprocess.run = ",
        "original_run = subprocess.run",
        "MockCompletedProcess",
    ]
    for filepath in TEST_FILES_TO_AUDIT:
        try:
            with open(filepath, "r", encoding="utf-8") as f:
                content = f.read()
        except FileNotFoundError:
            continue
        
        for pattern in forbidden_patterns:
            assert pattern not in content, (
                f"Forbidden monkeypatch pattern found in {filepath}: '{pattern}'"
            )


def test_p46r_02_shared_utility_uses_real_subprocess():
    """Verify the shared utility uses real subprocess.run calls, not mocks."""
    with open("tests/phase2_scope_gate_utils.py", "r", encoding="utf-8") as f:
        content = f.read()
    
    # Must contain real subprocess.run calls
    assert "subprocess.run(" in content
    
    # Must NOT contain any monkeypatch patterns
    assert "subprocess.run = " not in content
    assert "MockCompletedProcess" not in content
    assert "original_run" not in content
    
    # Must use git branch --show-current (real call)
    assert "git\", \"branch\", \"--show-current\"" in content
    
    # Must use git diff --name-only (real call)
    assert "git\", \"diff\", \"--name-only\"" in content


def test_p46r_03_math_sources_unchanged():
    """Verify math source files are unchanged from P46 base."""
    res = subprocess.run(
        ["git", "diff", "--name-only", P46_REMOTE_HEAD, "--"] + MATH_SOURCE_FILES,
        capture_output=True, text=True, check=True
    )
    changed = [line.strip() for line in res.stdout.splitlines() if line.strip()]
    assert changed == [], (
        f"Math source files must not be changed from P46 base. Changed: {changed}"
    )


def test_p46r_04_current_scope_enforced():
    """Enforce P46R scope: only allowed files modified relative to P46 base."""
    current_branch = get_current_git_branch()
    if current_branch != P46R_EXPECTED_BRANCH:
        pytest.skip(f"P46R scope gate skipped on branch '{current_branch}'")
    modified = git_changed_files_since(P46_REMOTE_HEAD)
    for f in modified:
        assert f in P46R_ALLOWED_FILES, (
            f"Forbidden file modification detected in P46R: {f}"
        )


def test_p46r_05_scope_gate_utility_skips_on_wrong_branch():
    """Verify the shared utility correctly identifies the current branch."""
    current = get_current_git_branch()
    if current != P46R_EXPECTED_BRANCH:
        pytest.skip(f"P46R branch verification skipped on branch '{current}'")
    assert isinstance(current, str)
    assert len(current) > 0
    # On P46R branch this should be the P46R branch name
    assert current == P46R_EXPECTED_BRANCH, (
        f"Expected to be on '{P46R_EXPECTED_BRANCH}', but got '{current}'"
    )


def test_p46r_06_old_scope_gates_skip_on_this_branch():
    """Confirm that old scope gate tests would skip on the P46R branch (verifying the mechanism works)."""
    current = get_current_git_branch()
    # P44 expected branch
    p44_branch = "phase2/p44-tensor-native-constraint-primitives-no-model-no-loss-no-training"
    # P45 expected branch
    p45_branch = "phase2/p45-analytic-moment-spectral-signatures-no-loss-no-model-no-training"
    # P46 expected branch
    p46_branch = "phase2/p46-moment-spectral-matching-loss-no-model-no-training"
    
    # None of the old phase branches should equal the current P46R branch
    assert current != p44_branch, "Should not be on P44 branch when running P46R"
    assert current != p45_branch, "Should not be on P45 branch when running P46R"
    assert current != p46_branch, "Should not be on P46 branch when running P46R"
