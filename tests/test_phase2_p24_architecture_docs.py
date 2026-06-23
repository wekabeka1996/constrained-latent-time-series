# tests/test_phase2_p24_architecture_docs.py

import pathlib
import subprocess
import pytest

# Define paths
DOC_ARCH = pathlib.Path("docs/PHASE_2_FACTORISED_CONSTRAINED_VAE_ARCHITECTURE.md")
DOC_BOUND = pathlib.Path("docs/PHASE_2_MODEL_BOUNDARY_CONTRACT.md")
DOC_ABLAT = pathlib.Path("docs/PHASE_2_ARCHITECTURE_ABLATION_PLAN.md")
DOC_M2E = pathlib.Path("docs/PHASE_2_MODEL_TO_EVIDENCE_CONTRACT.md")
REPORT_PATH = pathlib.Path("reports/PHASE_2_P24_FACTORISED_CONSTRAINED_VAE_ARCHITECTURE_SPEC_REPORT.md")

ALL_DOCS = [DOC_ARCH, DOC_BOUND, DOC_ABLAT, DOC_M2E]


def test_p24_01_docs_exist():
    """Verify that all four documentation files exist."""
    for doc in ALL_DOCS:
        assert doc.is_file(), f"Missing doc file: {doc}"


def test_p24_02_report_exists():
    """Verify that the P24 report file exists."""
    assert REPORT_PATH.is_file(), f"Missing report file: {REPORT_PATH}"


def test_p24_03_architecture_contents():
    """Verify that the architecture doc contains required canonical terms."""
    content = DOC_ARCH.read_text(encoding="utf-8")
    required = [
        "Factorized Constrained VAE",
        "z_mean",
        "z_volatility",
        "z_shared",
        "ModelSpec",
        "reject-invalid mode",
        "validate_model_spec",
        "require_math_valid",
    ]
    for term in required:
        assert term in content, f"Term '{term}' not found in {DOC_ARCH}"


def test_p24_04_model_boundary_contents():
    """Verify that the model boundary contract contains required canonical terms."""
    content = DOC_BOUND.read_text(encoding="utf-8")
    required = [
        "C_train_count",
        "zero_shot_mode",
        "evidence_contract_version",
        "fail",
        "P23",
    ]
    for term in required:
        assert term in content, f"Term '{term}' not found in {DOC_BOUND}"


def test_p24_05_ablation_plan_contents():
    """Verify that the ablation plan contains required canonical terms."""
    content = DOC_ABLAT.read_text(encoding="utf-8")
    required = [
        "M0",
        "M1",
        "M2",
        "M3",
        "M4",
        "Structural composition oracle",
    ]
    for term in required:
        assert term in content, f"Term '{term}' not found in {DOC_ABLAT}"


def test_p24_06_model_to_evidence_contents():
    """Verify that the model-to-evidence contract contains required canonical terms."""
    content = DOC_M2E.read_text(encoding="utf-8")
    required = [
        "EvidenceModelRecord",
        "EvidenceComparisonBundle",
        "no raw params",
        "no forbidden claims",
        "no C leakage",
    ]
    for term in required:
        assert term in content, f"Term '{term}' not found in {DOC_M2E}"


def test_p24_07_no_local_paths():
    """Verify that none of the P24 docs contain absolute/local machine paths."""
    forbidden = ["file:///", "C:/", "C:\\", "/home/", "/Users/"]
    for doc in ALL_DOCS + [REPORT_PATH]:
        content = doc.read_text(encoding="utf-8")
        # Exclude expected text mentions of these prefixes
        lines = content.splitlines()
        for line in lines:
            if any(pat in line for pat in forbidden):
                # Allow the exact report/doc safety sections which list these as rejected
                if "Rejects absolute paths" in line or "contain absolute/local" in line or "contain no absolute paths" in line:
                    continue
                pytest.fail(f"Local path leak pattern found in {doc}: '{line.strip()}'")


def test_p24_08_no_implementation_claims():
    """Verify that none of the P24 docs contain premature implementation success claims."""
    forbidden = ["implemented training", "trained model", "model works", "scientific success", "solved"]
    for doc in ALL_DOCS + [REPORT_PATH]:
        content = doc.read_text(encoding="utf-8")
        lines = content.splitlines()
        for line in lines:
            if any(pat in line.lower() for pat in forbidden):
                # Allow descriptive text about safety checks or non-goals
                if "contains no" in line.lower() or "non-goals" in line.lower() or "disclaimer" in line.lower() or "pre-commit" in line.lower():
                    continue
                pytest.fail(f"Premature implementation claim found in {doc}: '{line.strip()}'")


def test_p24_09_no_final_comparison_claims():
    """Verify that none of the P24 docs claim a final comparison."""
    for doc in ALL_DOCS + [REPORT_PATH]:
        content = doc.read_text(encoding="utf-8")
        # Ensure we don't claim we ran a final comparison
        if "final comparison" in content.lower():
            lines = content.splitlines()
            for line in lines:
                if "final comparison" in line.lower():
                    if "no final comparison" in line.lower() or "non-goals" in line.lower() or "rejects" in line.lower() or "disclaimer" in line.lower() or "confirmation" in line.lower():
                        continue
                    pytest.fail(f"Unacceptable final comparison claim found in {doc}: '{line.strip()}'")


def test_p24_10_no_forbidden_files_modified():
    """Verify that no files outside the approved list are modified or untracked in git."""
    allowed = {
        "docs/PHASE_2_FACTORISED_CONSTRAINED_VAE_ARCHITECTURE.md",
        "docs/PHASE_2_MODEL_BOUNDARY_CONTRACT.md",
        "docs/PHASE_2_ARCHITECTURE_ABLATION_PLAN.md",
        "docs/PHASE_2_MODEL_TO_EVIDENCE_CONTRACT.md",
        "tests/test_phase2_p24_architecture_docs.py",
        "reports/PHASE_2_P24_FACTORISED_CONSTRAINED_VAE_ARCHITECTURE_SPEC_REPORT.md",
    }
    # Run git diff against the base P23 branch
    res = subprocess.run(
        ["git", "diff", "--name-only", "phase2/p23-evidence-contract-normalized-baseline-bundle"],
        capture_output=True, text=True, check=True
    )
    branch_res = subprocess.run(
        ["git", "rev-parse", "--abbrev-ref", "HEAD"],
        capture_output=True, text=True, check=True
    )
    current_branch = branch_res.stdout.strip()
    if "p25" in current_branch:
        allowed.update({
            "src/phase2/model_interface.py",
            "tests/test_phase2_model_interface.py",
            "tools/phase2/run_p25_model_interface_smoke.py",
            "tests/test_phase2_p25_model_interface_smoke.py",
            "reports/PHASE_2_P25_MODEL_INTERFACE_SKELETON_AND_CANDIDATE_CONTRACT_REPORT.md",
            "src/phase2/__init__.py",
        })
    modified = [line.strip() for line in res.stdout.splitlines() if line.strip()]
    for f in modified:
        # Ignore slash normalization variances
        f_norm = f.replace("\\", "/")
        assert f_norm in allowed, f"Forbidden file modification detected: {f_norm}"

