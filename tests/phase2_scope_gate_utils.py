# tests/phase2_scope_gate_utils.py
#
# Shared scope-gate helper for Phase 2 test suites.
#
# Policy:
#   Phase-local scope gates are only authoritative on their own phase branch.
#   On later cumulative branches, they skip with an explicit reason via pytest.skip().
#   No monkeypatching. No fake git diff output. No silent bypass.

import subprocess
from typing import List, Set

import pytest


def get_current_git_branch() -> str:
    """Return the current git branch name using real subprocess."""
    res = subprocess.run(
        ["git", "branch", "--show-current"],
        capture_output=True, text=True, check=True
    )
    return res.stdout.strip()


def git_changed_files_since(base_commit: str) -> List[str]:
    """Return list of changed file paths (forward-slash normalized) since base_commit."""
    res = subprocess.run(
        ["git", "diff", "--name-only", base_commit],
        capture_output=True, text=True, check=True
    )
    return [
        line.strip().replace("\\", "/")
        for line in res.stdout.splitlines()
        if line.strip()
    ]


def enforce_phase_local_scope_gate_or_skip(
    expected_branch: str,
    base_commit: str,
    allowed_files: Set[str],
    phase_label: str,
) -> List[str]:
    """
    Enforce phase-local scope gate: checks git diff against base_commit and verifies
    only allowed_files were modified.

    If the current branch is NOT expected_branch, the test is skipped explicitly.
    Returns the list of modified files (on the expected branch) for optional reporting.
    """
    current_branch = get_current_git_branch()
    if current_branch != expected_branch:
        pytest.skip(
            f"{phase_label} phase-local scope gate skipped on branch '{current_branch}'; "
            f"expected '{expected_branch}'. "
            f"Current-phase scope is enforced by its own scope gate test."
        )

    modified = git_changed_files_since(base_commit)
    for f in modified:
        assert f in allowed_files, (
            f"Forbidden file modification detected in {phase_label}: {f}"
        )
    return modified
