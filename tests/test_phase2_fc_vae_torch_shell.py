# tests/test_phase2_fc_vae_torch_shell.py

import dataclasses
import json
import pathlib
import subprocess
from tests.phase2_scope_gate_utils import enforce_phase_local_scope_gate_or_skip
import re
import pytest
from dataclasses import is_dataclass

import src.phase2
from src.phase2.fc_vae_torch_shell import (
    FC_VAE_TORCH_SHELL_CONTRACT_VERSION,
    FC_VAE_TORCH_SHELL_KIND,
    FC_VAE_TORCH_SHELL_MODULE_NAME,
    FC_VAE_FUTURE_IMPLEMENTATION_MODULE,
    FC_VAE_TORCH_SHELL_STATUS_TORCH_UNAVAILABLE,
    FC_VAE_TORCH_SHELL_STATUS_IMPLEMENTATION_DEFERRED,
    FC_VAE_TORCH_SHELL_STATUS_READY_FOR_FUTURE_IMPLEMENTATION,
    SUPPORTED_FC_VAE_TORCH_SHELL_STATUSES,
    FC_VAE_TORCH_SHELL_NO_MODULE_SENTINEL,
    FCVAETorchShellRequest,
    FCVAETorchShellHandle,
    FCVAETorchShellResult,
    validate_non_empty_str,
    validate_bool,
    validate_shell_status,
    validate_torch_shell_request,
    validate_torch_shell_handle,
    validate_torch_shell_result,
    assert_no_local_path_leakage,
    assert_no_forbidden_claims,
    build_torch_shell_request,
    build_torch_shell_handle,
    build_torch_shell_result,
    require_torch_shell_materialized,
    run_fc_vae_torch_shell_probe,
    torch_shell_request_to_json_dict,
    torch_shell_handle_to_json_dict,
    torch_shell_result_to_json_dict,
    compact_torch_shell_json,
)


# A. Constants/dataclasses
def test_p28_01_constants():
    assert FC_VAE_TORCH_SHELL_CONTRACT_VERSION == "phase2_p28_fc_vae_torch_shell_contract_v1"
    assert FC_VAE_TORCH_SHELL_KIND == "optional_torch_shell_handle"
    assert FC_VAE_TORCH_SHELL_MODULE_NAME == "src.phase2.fc_vae_torch_shell"
    assert FC_VAE_FUTURE_IMPLEMENTATION_MODULE == "src.phase2.fc_vae_model"
    assert FC_VAE_TORCH_SHELL_STATUS_TORCH_UNAVAILABLE == "blocked_torch_unavailable"
    assert FC_VAE_TORCH_SHELL_STATUS_IMPLEMENTATION_DEFERRED == "blocked_implementation_deferred"
    assert FC_VAE_TORCH_SHELL_STATUS_READY_FOR_FUTURE_IMPLEMENTATION == "ready_for_future_implementation"
    assert len(SUPPORTED_FC_VAE_TORCH_SHELL_STATUSES) == 3
    assert "blocked_torch_unavailable" in SUPPORTED_FC_VAE_TORCH_SHELL_STATUSES
    assert "blocked_implementation_deferred" in SUPPORTED_FC_VAE_TORCH_SHELL_STATUSES
    assert "ready_for_future_implementation" in SUPPORTED_FC_VAE_TORCH_SHELL_STATUSES
    assert FC_VAE_TORCH_SHELL_NO_MODULE_SENTINEL == "no_torch_module_created_in_p28"


def test_p28_02_dataclasses_frozen():
    for cls in (FCVAETorchShellRequest, FCVAETorchShellHandle, FCVAETorchShellResult):
        assert is_dataclass(cls)
        # Test freezing
        request = build_torch_shell_request()
        with pytest.raises(Exception):
            request.reason = "mutated"


# B. Basic validators/safety
def test_p28_03_validate_non_empty_str_accepts():
    validate_non_empty_str("hello", "field")


def test_p28_04_validate_non_empty_str_rejects():
    with pytest.raises(TypeError):
        validate_non_empty_str(123, "field")
    with pytest.raises(ValueError):
        validate_non_empty_str("", "field")
    with pytest.raises(ValueError):
        validate_non_empty_str(" padded ", "field")


def test_p28_05_validate_bool():
    validate_bool(True, "field")
    validate_bool(False, "field")
    with pytest.raises(TypeError):
        validate_bool("True", "field")


def test_p28_06_validate_shell_status_accepts():
    for status in SUPPORTED_FC_VAE_TORCH_SHELL_STATUSES:
        validate_shell_status(status)


def test_p28_07_validate_shell_status_rejects():
    with pytest.raises(ValueError):
        validate_shell_status("invalid_status")
    with pytest.raises(TypeError):
        validate_shell_status(True)


def test_p28_08_local_path_leakage():
    assert_no_local_path_leakage("safe_string")
    bad_paths = ["file:///", "C:/", "C:\\", "/home/", "/Users/"]
    for path in bad_paths:
        with pytest.raises(ValueError):
            assert_no_local_path_leakage(f"some data {path} here")


def test_p28_09_forbidden_claims():
    assert_no_forbidden_claims("safe claim")
    assert_no_forbidden_claims("no_scientific_conclusion")
    assert_no_forbidden_claims("no_final_comparison")
    bad_words = ["model works", "scientific success", "solved", "best", "winner"]
    for word in bad_words:
        with pytest.raises(ValueError):
            assert_no_forbidden_claims(f"our {word} is confirmed")


# C. Request/handle/result contracts
def test_p28_10_build_request():
    req = build_torch_shell_request()
    validate_torch_shell_request(req)


def test_p28_11_request_rejects_wrong_contract():
    req = build_torch_shell_request()
    bad_req = dataclasses.replace(req, contract_version="wrong")
    with pytest.raises(ValueError):
        validate_torch_shell_request(bad_req)


def test_p28_12_request_rejects_wrong_shell_kind():
    req = build_torch_shell_request()
    bad_req = dataclasses.replace(req, shell_kind="wrong")
    with pytest.raises(ValueError):
        validate_torch_shell_request(bad_req)


def test_p28_13_request_rejects_wrong_architecture_id():
    req = build_torch_shell_request()
    bad_req = dataclasses.replace(req, architecture_id="wrong")
    with pytest.raises(ValueError):
        validate_torch_shell_request(bad_req)


def test_p28_14_request_rejects_allow_implementation_true():
    req = build_torch_shell_request()
    bad_req = dataclasses.replace(req, allow_implementation_in_p28=True)
    with pytest.raises(ValueError):
        validate_torch_shell_request(bad_req)


def test_p28_15_build_handle():
    handle = build_torch_shell_handle()
    validate_torch_shell_handle(handle)


def test_p28_16_handle_rejects_module_created_true():
    handle = build_torch_shell_handle()
    bad_handle = dataclasses.replace(handle, module_created=True)
    with pytest.raises(ValueError):
        validate_torch_shell_handle(bad_handle)


def test_p28_17_handle_rejects_wrong_sentinel():
    handle = build_torch_shell_handle()
    bad_handle = dataclasses.replace(handle, module_sentinel="wrong")
    with pytest.raises(ValueError):
        validate_torch_shell_handle(bad_handle)


def test_p28_18_build_result():
    req = build_torch_shell_request()
    result = build_torch_shell_result(req)
    validate_torch_shell_result(result)


def test_p28_19_result_rejects_torch_required_true():
    req = build_torch_shell_request()
    result = build_torch_shell_result(req)
    bad_res = dataclasses.replace(result, torch_required_for_p28=True)
    with pytest.raises(ValueError):
        validate_torch_shell_result(bad_res)


def test_p28_20_result_rejects_torch_required_future_false():
    req = build_torch_shell_request()
    result = build_torch_shell_result(req)
    bad_res = dataclasses.replace(result, torch_required_for_future_execution=False)
    with pytest.raises(ValueError):
        validate_torch_shell_result(bad_res)


def test_p28_21_result_rejects_implementation_available_true():
    req = build_torch_shell_request()
    result = build_torch_shell_result(req)
    bad_res = dataclasses.replace(result, implementation_available_in_p28=True)
    with pytest.raises(ValueError):
        validate_torch_shell_result(bad_res)


def test_p28_22_result_rejects_module_created_true():
    req = build_torch_shell_request()
    result = build_torch_shell_result(req)
    bad_res = dataclasses.replace(result, module_created=True)
    with pytest.raises(ValueError):
        validate_torch_shell_result(bad_res)


def test_p28_23_result_rejects_no_model_implementation_false():
    req = build_torch_shell_request()
    result = build_torch_shell_result(req)
    bad_res = dataclasses.replace(result, no_model_implementation=False)
    with pytest.raises(ValueError):
        validate_torch_shell_result(bad_res)


def test_p28_24_result_rejects_no_training_loop_false():
    req = build_torch_shell_request()
    result = build_torch_shell_result(req)
    bad_res = dataclasses.replace(result, no_training_loop=False)
    with pytest.raises(ValueError):
        validate_torch_shell_result(bad_res)


def test_p28_25_result_rejects_no_optimizer_false():
    req = build_torch_shell_request()
    result = build_torch_shell_result(req)
    bad_res = dataclasses.replace(result, no_optimizer=False)
    with pytest.raises(ValueError):
        validate_torch_shell_result(bad_res)


def test_p28_26_result_rejects_no_checkpointing_false():
    req = build_torch_shell_request()
    result = build_torch_shell_result(req)
    bad_res = dataclasses.replace(result, no_checkpointing=False)
    with pytest.raises(ValueError):
        validate_torch_shell_result(bad_res)


def test_p28_27_result_rejects_no_artifact_generation_false():
    req = build_torch_shell_request()
    result = build_torch_shell_result(req)
    bad_res = dataclasses.replace(result, no_artifact_generation=False)
    with pytest.raises(ValueError):
        validate_torch_shell_result(bad_res)


def test_p28_28_result_rejects_inconsistent_torch_status():
    req = build_torch_shell_request()
    result = build_torch_shell_result(req)
    # Force unavailable, but status is deferred
    bad_res = dataclasses.replace(
        result,
        torch_available=False,
        shell_status="blocked_implementation_deferred"
    )
    with pytest.raises(ValueError):
        validate_torch_shell_result(bad_res)


def test_p28_29_require_materialized_raises():
    req = build_torch_shell_request()
    result = build_torch_shell_result(req)
    with pytest.raises(NotImplementedError) as exc_info:
        require_torch_shell_materialized(result)
    assert "FC-VAE torch shell is not materialized in P28" in str(exc_info.value)


def test_p28_30_probe():
    res = run_fc_vae_torch_shell_probe()
    validate_torch_shell_result(res)


# D. Serialization
def test_p28_31_to_json_dict():
    req = build_torch_shell_request()
    res = build_torch_shell_result(req)
    d1 = torch_shell_request_to_json_dict(req)
    d2 = torch_shell_handle_to_json_dict(res.handle)
    d3 = torch_shell_result_to_json_dict(res)
    assert isinstance(d1, dict)
    assert isinstance(d2, dict)
    assert isinstance(d3, dict)


def test_p28_32_compact_json():
    res = run_fc_vae_torch_shell_probe()
    js = compact_torch_shell_json(res)
    assert isinstance(js, str)
    d = json.loads(js)
    assert d["contract_version"] is not None



def test_p28_33_serialized_no_local_paths():
    res = run_fc_vae_torch_shell_probe()
    js = compact_torch_shell_json(res)
    for path in ["file:///", "C:/", "C:\\", "/home/", "/Users/"]:
        assert path not in js


def test_p28_34_serialized_no_forbidden_claims():
    res = run_fc_vae_torch_shell_probe()
    js = compact_torch_shell_json(res).lower()
    forbidden = ["model works", "scientific success", "solved", "best", "winner"]
    for word in forbidden:
        assert word not in js


def test_p28_35_serialized_no_torch_object_representation():
    res = run_fc_vae_torch_shell_probe()
    js = compact_torch_shell_json(res).lower()
    assert "torch.tensor" not in js
    assert "nn.module" not in js
    assert "parameter" not in js


def test_p28_36_serialized_no_raw_parameters():
    res = run_fc_vae_torch_shell_probe()
    js = compact_torch_shell_json(res)
    forbidden_keys = [
        "ar_params", "ma_params", "alpha_params", "beta_params",
        "omega", "model_spec", "generated_spec"
    ]
    for key in forbidden_keys:
        assert f'"{key}"' not in js


# E. Exports/scope
def test_p28_37_exports():
    exports = dir(src.phase2)
    assert "FC_VAE_TORCH_SHELL_CONTRACT_VERSION" in exports
    assert "FCVAETorchShellRequest" in exports
    assert "FCVAETorchShellHandle" in exports
    assert "FCVAETorchShellResult" in exports
    assert "build_torch_shell_request" in exports
    assert "build_torch_shell_handle" in exports
    assert "build_torch_shell_result" in exports
    assert "require_torch_shell_materialized" in exports
    assert "run_fc_vae_torch_shell_probe" in exports
    assert "compact_torch_shell_json" in exports


def test_p28_38_no_top_level_torch_import():
    p = pathlib.Path("src/phase2/fc_vae_torch_shell.py").read_text(encoding="utf-8")
    for line in p.splitlines():
        stripped = line.strip()
        if stripped.startswith("#"):
            continue
        assert not stripped.startswith("import torch"), f"Top-level 'import torch' found: {stripped}"
        assert not stripped.startswith("from torch"), f"Top-level 'from torch' found: {stripped}"


def test_p28_39_no_forbidden_imports():
    p = pathlib.Path("src/phase2/fc_vae_torch_shell.py").read_text(encoding="utf-8")
    forbidden = ["numpy", "pandas", "scipy", "sklearn", "yaml", "argparse", "subprocess"]
    for f in forbidden:
        assert f"import {f}" not in p
        assert f"from {f}" not in p


def test_p28_40_no_torch_nn_module():
    p = pathlib.Path("src/phase2/fc_vae_torch_shell.py").read_text(encoding="utf-8")
    assert "torch.nn.Module" not in p


def test_p28_41_no_forbidden_class_names():
    p = pathlib.Path("src/phase2/fc_vae_torch_shell.py").read_text(encoding="utf-8")
    forbidden_classes = [
        r"class\s+Encoder\b", r"class\s+Decoder\b", r"class\s+FCVAE\s*[:(]",
        r"class\s+FC_VAE\b", r"class\s+Optimizer\b", r"class\s+Checkpoint\b",
        r"class\s+Trainer\b",
    ]
    for pattern in forbidden_classes:
        assert not re.search(pattern, p), f"Forbidden class definition found matching: {pattern}"


def test_p28_42_no_forward_method():
    p = pathlib.Path("src/phase2/fc_vae_torch_shell.py").read_text(encoding="utf-8")
    assert "def forward(" not in p


def test_p28_43_no_training_implementations():
    p = pathlib.Path("src/phase2/fc_vae_torch_shell.py").read_text(encoding="utf-8")
    forbidden_fns = [
        "def train(", "def fit(", "def train_step(", "def training_loop(",
        "def save_checkpoint(", "def load_checkpoint(",
    ]
    for fn in forbidden_fns:
        assert fn not in p, f"Forbidden function found: {fn}"


def test_p28_44_init_no_subprocess_monkeypatch():
    p = pathlib.Path("src/phase2/__init__.py").read_text(encoding="utf-8")
    assert "subprocess.run =" not in p
    assert "_patched_run" not in p
    assert "_original_run" not in p


def test_p28_45_scope_gate():
    """Phase-local scope gate for P28. Skips on non-P28 branches."""
    allowed = {
        "src/phase2/fc_vae_torch_shell.py",
        "src/phase2/__init__.py",
        "tests/test_phase2_fc_vae_torch_shell.py",
        "tools/phase2/run_p28_fc_vae_torch_shell_smoke.py",
        "tests/test_phase2_p28_fc_vae_torch_shell_smoke.py",
        "reports/PHASE_2_P28_OPTIONAL_TORCH_FC_VAE_SHELL_HANDLE_REPORT.md",
    }
    enforce_phase_local_scope_gate_or_skip(
        expected_branch="phase2/p28-optional-torch-fc-vae-shell-handle",
        base_commit="phase2/p27-fc-vae-module-skeleton-shape-contract",
        allowed_files=allowed,
        phase_label="P28",
    )
