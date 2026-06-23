# tests/test_phase2_torch_boundary.py

import dataclasses
import json
import pathlib
import subprocess
import pytest
from dataclasses import is_dataclass

import src.phase2
from src.phase2.torch_boundary import (
    TORCH_BOUNDARY_CONTRACT_VERSION,
    TORCH_BACKEND_NAME,
    TORCH_POLICY_OPTIONAL,
    TORCH_POLICY_REQUIRED_FOR_MODEL_IMPLEMENTATION,
    TORCH_POLICY_FORBIDDEN_IN_CORE,
    SUPPORTED_TORCH_POLICIES,
    FC_VAE_FUTURE_MODULE_NAME,
    FC_VAE_FUTURE_ARCHITECTURE_ID,
    TorchDependencyStatus,
    FutureModelBoundarySpec,
    TorchBoundarySmokeResult,
    validate_non_empty_str,
    validate_bool,
    validate_torch_policy,
    detect_torch_available,
    build_torch_dependency_status,
    validate_torch_dependency_status,
    require_torch_available_for_future_model,
    build_future_model_boundary_spec,
    validate_future_model_boundary_spec,
    build_torch_boundary_smoke_result,
    validate_torch_boundary_smoke_result,
    torch_dependency_status_to_json_dict,
    future_model_boundary_spec_to_json_dict,
    torch_boundary_smoke_result_to_json_dict,
    compact_torch_boundary_json,
)


# A. Constants/dataclasses
def test_p26_01_constants_values():
    assert TORCH_BOUNDARY_CONTRACT_VERSION == "phase2_p26_torch_boundary_contract_v1"
    assert TORCH_BACKEND_NAME == "torch"
    assert TORCH_POLICY_OPTIONAL == "optional"
    assert TORCH_POLICY_REQUIRED_FOR_MODEL_IMPLEMENTATION == "required_for_model_implementation"
    assert TORCH_POLICY_FORBIDDEN_IN_CORE == "forbidden_in_core"
    assert SUPPORTED_TORCH_POLICIES == ("optional", "required_for_model_implementation", "forbidden_in_core")
    assert FC_VAE_FUTURE_MODULE_NAME == "src.phase2.fc_vae_model"
    assert FC_VAE_FUTURE_ARCHITECTURE_ID == "FC-VAE"


def test_p26_02_dataclasses_frozen():
    classes = [TorchDependencyStatus, FutureModelBoundarySpec, TorchBoundarySmokeResult]
    for cls in classes:
        assert is_dataclass(cls)
    status = build_torch_dependency_status()
    with pytest.raises(Exception):
        status.reason = "modified"


# B. Validation helpers
def test_p26_03_validate_non_empty_str_accepts():
    validate_non_empty_str("valid_string", "field")


def test_p26_04_validate_non_empty_str_rejects():
    with pytest.raises(TypeError):
        validate_non_empty_str(123, "field")
    with pytest.raises(ValueError):
        validate_non_empty_str("", "field")
    with pytest.raises(ValueError):
        validate_non_empty_str("  padded  ", "field")


def test_p26_05_validate_bool_accepts():
    validate_bool(True, "field")
    validate_bool(False, "field")
    with pytest.raises(TypeError):
        validate_bool(1, "field")


def test_p26_06_validate_torch_policy_accepts():
    for p in SUPPORTED_TORCH_POLICIES:
        validate_torch_policy(p)


def test_p26_07_validate_torch_policy_rejects():
    with pytest.raises(ValueError):
        validate_torch_policy("unsupported_policy")


# C. Detection / status
def test_p26_08_detect_torch_available_returns_bool():
    result = detect_torch_available()
    assert type(result) is bool


def test_p26_09_build_torch_dependency_status():
    status = build_torch_dependency_status()
    assert status.contract_version == TORCH_BOUNDARY_CONTRACT_VERSION
    assert status.backend_name == TORCH_BACKEND_NAME
    assert type(status.available) is bool
    assert status.top_level_import_required is False


def test_p26_10_validate_torch_dependency_status_accepts():
    status = build_torch_dependency_status(policy=TORCH_POLICY_OPTIONAL)
    validate_torch_dependency_status(status)


def test_p26_11_validate_torch_dependency_status_rejects_wrong_contract():
    status = dataclasses.replace(
        build_torch_dependency_status(), contract_version="wrong_v2"
    )
    with pytest.raises(ValueError, match="Wrong contract version"):
        validate_torch_dependency_status(status)


def test_p26_12_validate_torch_dependency_status_rejects_top_level_import():
    status = dataclasses.replace(
        build_torch_dependency_status(), top_level_import_required=True
    )
    with pytest.raises(ValueError, match="top_level_import_required must be False"):
        validate_torch_dependency_status(status)


def test_p26_13_validate_torch_dependency_status_rejects_import_safe_when_forbidden():
    status = TorchDependencyStatus(
        contract_version=TORCH_BOUNDARY_CONTRACT_VERSION,
        backend_name=TORCH_BACKEND_NAME,
        available=True,
        policy=TORCH_POLICY_FORBIDDEN_IN_CORE,
        import_safe=True,
        top_level_import_required=False,
        reason="test_forbidden_import_safe",
    )
    with pytest.raises(ValueError, match="import_safe must be False"):
        validate_torch_dependency_status(status)


def test_p26_14_require_torch_raises_when_unavailable():
    status = TorchDependencyStatus(
        contract_version=TORCH_BOUNDARY_CONTRACT_VERSION,
        backend_name=TORCH_BACKEND_NAME,
        available=False,
        policy=TORCH_POLICY_OPTIONAL,
        import_safe=False,
        top_level_import_required=False,
        reason="torch_unavailable_optional_boundary",
    )
    with pytest.raises(RuntimeError, match="torch unavailable"):
        require_torch_available_for_future_model(status)


def test_p26_15_require_torch_passes_when_available():
    status = TorchDependencyStatus(
        contract_version=TORCH_BOUNDARY_CONTRACT_VERSION,
        backend_name=TORCH_BACKEND_NAME,
        available=True,
        policy=TORCH_POLICY_OPTIONAL,
        import_safe=True,
        top_level_import_required=False,
        reason="torch_available_optional_boundary",
    )
    require_torch_available_for_future_model(status)


# D. Future model boundary
def test_p26_16_build_future_model_boundary_spec():
    spec = build_future_model_boundary_spec()
    assert spec.architecture_id == FC_VAE_FUTURE_ARCHITECTURE_ID
    assert spec.future_module_name == FC_VAE_FUTURE_MODULE_NAME
    assert spec.allowed_in_p26 is False


def test_p26_17_validate_future_model_boundary_spec_accepts():
    spec = build_future_model_boundary_spec()
    validate_future_model_boundary_spec(spec)


def test_p26_18_validate_future_model_boundary_spec_rejects_allowed():
    spec = dataclasses.replace(build_future_model_boundary_spec(), allowed_in_p26=True)
    with pytest.raises(ValueError, match="allowed_in_p26 must be False"):
        validate_future_model_boundary_spec(spec)


def test_p26_19_validate_future_model_boundary_spec_rejects_wrong_module():
    spec = dataclasses.replace(
        build_future_model_boundary_spec(), future_module_name="wrong.module"
    )
    with pytest.raises(ValueError, match="Wrong future_module_name"):
        validate_future_model_boundary_spec(spec)


def test_p26_20_validate_future_model_boundary_spec_rejects_wrong_arch():
    spec = dataclasses.replace(
        build_future_model_boundary_spec(), architecture_id="WRONG-ARCH"
    )
    with pytest.raises(ValueError, match="Wrong architecture_id"):
        validate_future_model_boundary_spec(spec)


# E. Smoke result / JSON
def test_p26_21_build_torch_boundary_smoke_result():
    result = build_torch_boundary_smoke_result()
    assert result.verdict == "PASS"


def test_p26_22_validate_torch_boundary_smoke_result_accepts():
    result = build_torch_boundary_smoke_result()
    validate_torch_boundary_smoke_result(result)


def test_p26_23_smoke_result_rejects_no_flag_false():
    result = build_torch_boundary_smoke_result()
    bad = dataclasses.replace(result, no_model_implementation=False)
    with pytest.raises(ValueError, match="no_model_implementation must be True"):
        validate_torch_boundary_smoke_result(bad)
    bad2 = dataclasses.replace(result, no_training_loop=False)
    with pytest.raises(ValueError, match="no_training_loop must be True"):
        validate_torch_boundary_smoke_result(bad2)


def test_p26_24_torch_dependency_status_to_json_dict():
    status = build_torch_dependency_status()
    d = torch_dependency_status_to_json_dict(status)
    assert "contract_version" in d
    assert "backend_name" in d
    assert "available" in d
    assert "policy" in d
    assert "import_safe" in d
    assert "top_level_import_required" in d
    assert "reason" in d


def test_p26_25_future_model_boundary_spec_to_json_dict():
    spec = build_future_model_boundary_spec()
    d = future_model_boundary_spec_to_json_dict(spec)
    assert "architecture_id" in d
    assert "future_module_name" in d
    assert "allowed_in_p26" in d


def test_p26_26_torch_boundary_smoke_result_to_json_dict_nested():
    result = build_torch_boundary_smoke_result()
    d = torch_boundary_smoke_result_to_json_dict(result)
    assert type(d["torch_status"]) is dict
    assert type(d["future_model_boundary"]) is dict


def test_p26_27_compact_torch_boundary_json():
    result = build_torch_boundary_smoke_result()
    js = compact_torch_boundary_json(result)
    d = json.loads(js)
    assert d["verdict"] == "PASS"
    assert ": " not in js
    assert ", " not in js
    assert "\n" not in js


# F. Exports / scope
def test_p26_28_exports():
    exports = dir(src.phase2)
    assert "TORCH_BOUNDARY_CONTRACT_VERSION" in exports
    assert "TorchDependencyStatus" in exports
    assert "build_torch_dependency_status" in exports
    assert "detect_torch_available" in exports


def test_p26_29_no_top_level_torch_import():
    p = pathlib.Path("src/phase2/torch_boundary.py").read_text(encoding="utf-8")
    lines = p.splitlines()
    for line in lines:
        stripped = line.strip()
        if stripped.startswith("#"):
            continue
        assert not stripped.startswith("import torch"), (
            f"Top-level 'import torch' found: {stripped}"
        )
        assert not stripped.startswith("from torch"), (
            f"Top-level 'from torch' found: {stripped}"
        )


def test_p26_30_no_forbidden_imports():
    p = pathlib.Path("src/phase2/torch_boundary.py").read_text(encoding="utf-8")
    forbidden = ["numpy", "pandas", "scipy", "sklearn", "yaml", "argparse", "subprocess"]
    for f in forbidden:
        assert f"import {f}" not in p
        assert f"from {f}" not in p


def test_p26_31_no_training_optimizer_checkpoint_classes():
    p = pathlib.Path("src/phase2/torch_boundary.py").read_text(encoding="utf-8")
    forbidden_terms = [
        "class Encoder",
        "class Decoder",
        "class FC_VAE",
        "class Optimizer",
        "class Checkpoint",
        "torch.nn.Module",
    ]
    for term in forbidden_terms:
        assert term not in p


def test_p26_32_init_no_subprocess_monkeypatch():
    p = pathlib.Path("src/phase2/__init__.py").read_text(encoding="utf-8")
    assert "subprocess.run =" not in p
    assert "_patched_run" not in p
    assert "_original_run" not in p


def test_p26_33_no_forbidden_files_modified():
    allowed = {
        "src/phase2/torch_boundary.py",
        "src/phase2/__init__.py",
        "tests/test_phase2_torch_boundary.py",
        "tools/phase2/run_p26_torch_boundary_smoke.py",
        "tests/test_phase2_p26_torch_boundary_smoke.py",
        "reports/PHASE_2_P26_TORCH_GATED_MODEL_DEPENDENCY_BOUNDARY_REPORT.md",
    }
    res = subprocess.run(
        ["git", "diff", "--name-only", "phase2/p25-fix-remove-subprocess-monkeypatch"],
        capture_output=True, text=True, check=True
    )
    modified = [line.strip() for line in res.stdout.splitlines() if line.strip()]
    for f in modified:
        f_norm = f.replace("\\", "/")
        assert f_norm in allowed, f"Forbidden file modification detected in P26: {f_norm}"
