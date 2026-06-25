# tests/test_phase2_fc_vae_torch_module_stub.py

import dataclasses
import json
import pathlib
import subprocess
import re
import pytest
from dataclasses import is_dataclass

import src.phase2
from src.phase2.fc_vae_torch_module_stub import (
    FC_VAE_TORCH_MODULE_STUB_CONTRACT_VERSION,
    FC_VAE_TORCH_MODULE_STUB_KIND,
    FC_VAE_TORCH_MODULE_STUB_MODULE_NAME,
    FC_VAE_TORCH_MODULE_STUB_CLASS_NAME,
    FC_VAE_TORCH_MODULE_STUB_STATUS_TORCH_UNAVAILABLE,
    FC_VAE_TORCH_MODULE_STUB_STATUS_STUB_CREATED,
    FC_VAE_TORCH_MODULE_STUB_STATUS_IMPLEMENTATION_DEFERRED,
    SUPPORTED_FC_VAE_TORCH_MODULE_STUB_STATUSES,
    FCVAETorchModuleStubRequest,
    FCVAETorchModuleStubMetadata,
    FCVAETorchModuleStubResult,
    validate_non_empty_str,
    validate_bool,
    validate_non_negative_int,
    validate_stub_status,
    assert_no_local_path_leakage,
    assert_no_forbidden_claims,
    validate_stub_request,
    validate_stub_metadata,
    validate_stub_result,
    build_stub_request,
    build_blocked_stub_metadata,
    materialize_local_torch_stub_metadata,
    build_stub_result,
    run_fc_vae_torch_module_stub_probe,
    stub_request_to_json_dict,
    stub_metadata_to_json_dict,
    stub_result_to_json_dict,
    compact_stub_json,
)


# A. Constants/dataclasses
def test_p29_01_constants():
    assert FC_VAE_TORCH_MODULE_STUB_CONTRACT_VERSION == "phase2_p29_torch_module_stub_contract_v1"
    assert FC_VAE_TORCH_MODULE_STUB_KIND == "gated_torch_nn_module_stub"
    assert FC_VAE_TORCH_MODULE_STUB_MODULE_NAME == "src.phase2.fc_vae_torch_module_stub"
    assert FC_VAE_TORCH_MODULE_STUB_CLASS_NAME == "P29LocalFCVAEModuleStub"
    assert FC_VAE_TORCH_MODULE_STUB_STATUS_TORCH_UNAVAILABLE == "blocked_torch_unavailable"
    assert FC_VAE_TORCH_MODULE_STUB_STATUS_STUB_CREATED == "stub_created_no_forward_no_layers"
    assert FC_VAE_TORCH_MODULE_STUB_STATUS_IMPLEMENTATION_DEFERRED == "blocked_implementation_deferred"
    assert len(SUPPORTED_FC_VAE_TORCH_MODULE_STUB_STATUSES) == 3
    assert "blocked_torch_unavailable" in SUPPORTED_FC_VAE_TORCH_MODULE_STUB_STATUSES
    assert "stub_created_no_forward_no_layers" in SUPPORTED_FC_VAE_TORCH_MODULE_STUB_STATUSES
    assert "blocked_implementation_deferred" in SUPPORTED_FC_VAE_TORCH_MODULE_STUB_STATUSES


def test_p29_02_dataclasses_frozen():
    for cls in (FCVAETorchModuleStubRequest, FCVAETorchModuleStubMetadata, FCVAETorchModuleStubResult):
        assert is_dataclass(cls)
        req = build_stub_request()
        with pytest.raises(Exception):
            req.reason = "modified"


# B. Validators/safety
def test_p29_03_validate_non_empty_str_accepts():
    validate_non_empty_str("valid", "name")


def test_p29_04_validate_non_empty_str_rejects():
    with pytest.raises(TypeError):
        validate_non_empty_str(123, "name")
    with pytest.raises(ValueError):
        validate_non_empty_str("", "name")
    with pytest.raises(ValueError):
        validate_non_empty_str("  padded  ", "name")


def test_p29_05_validate_bool_accepts():
    validate_bool(True, "name")
    validate_bool(False, "name")


def test_p29_06_validate_non_negative_int_rejects():
    validate_non_negative_int(0, "name")
    validate_non_negative_int(10, "name")
    with pytest.raises(TypeError):
        validate_non_negative_int(True, "name")
    with pytest.raises(TypeError):
        validate_non_negative_int(1.5, "name")
    with pytest.raises(ValueError):
        validate_non_negative_int(-1, "name")


def test_p29_07_validate_stub_status_accepts():
    for status in SUPPORTED_FC_VAE_TORCH_MODULE_STUB_STATUSES:
        validate_stub_status(status)


def test_p29_08_validate_stub_status_rejects():
    with pytest.raises(ValueError):
        validate_stub_status("unknown")
    with pytest.raises(TypeError):
        validate_stub_status(123)


def test_p29_09_local_path_leakage_rejects():
    assert_no_local_path_leakage("safe string")
    bad_paths = ["file:///", "C:/", "C:\\", "/home/", "/Users/"]
    for path in bad_paths:
        with pytest.raises(ValueError):
            assert_no_local_path_leakage(f"leak {path}")


def test_p29_10_forbidden_claims_rejects():
    assert_no_forbidden_claims("safe claim")
    assert_no_forbidden_claims("no_scientific_conclusion")
    assert_no_forbidden_claims("no_final_comparison")
    assert_no_forbidden_claims("implementation_available_in_p29=false")
    bad_claims = ["model works", "scientific success", "solved", "best", "winner", "production ready", "state of the art"]
    for claim in bad_claims:
        with pytest.raises(ValueError):
            assert_no_forbidden_claims(f"our {claim} check")


# C. Request/metadata/result
def test_p29_11_default_request_validates():
    req = build_stub_request()
    validate_stub_request(req)


def test_p29_12_request_rejects_wrong_contract():
    req = build_stub_request()
    bad = dataclasses.replace(req, contract_version="wrong")
    with pytest.raises(ValueError):
        validate_stub_request(bad)


def test_p29_13_request_rejects_wrong_stub_kind():
    req = build_stub_request()
    bad = dataclasses.replace(req, stub_kind="wrong")
    with pytest.raises(ValueError):
        validate_stub_request(bad)


def test_p29_14_request_rejects_wrong_architecture():
    req = build_stub_request()
    bad = dataclasses.replace(req, architecture_id="wrong")
    with pytest.raises(ValueError):
        validate_stub_request(bad)


def test_p29_15_request_rejects_allow_forward_true():
    req = build_stub_request()
    bad = dataclasses.replace(req, allow_forward_in_p29=True)
    with pytest.raises(ValueError):
        validate_stub_request(bad)


def test_p29_16_request_rejects_allow_layers_true():
    req = build_stub_request()
    bad = dataclasses.replace(req, allow_layers_in_p29=True)
    with pytest.raises(ValueError):
        validate_stub_request(bad)


def test_p29_17_blocked_metadata_validates():
    meta = build_blocked_stub_metadata(torch_available=False)
    validate_stub_metadata(meta)


def test_p29_18_metadata_rejects_defines_forward_true():
    meta = build_blocked_stub_metadata(torch_available=False)
    bad = dataclasses.replace(meta, defines_forward=True)
    with pytest.raises(ValueError):
        validate_stub_metadata(bad)


def test_p29_19_metadata_rejects_defines_layers_true():
    meta = build_blocked_stub_metadata(torch_available=False)
    bad = dataclasses.replace(meta, defines_layers=True)
    with pytest.raises(ValueError):
        validate_stub_metadata(bad)


def test_p29_20_metadata_rejects_nonzero_parameter_count():
    meta = build_blocked_stub_metadata(torch_available=False)
    bad = dataclasses.replace(meta, parameter_count=1)
    with pytest.raises(ValueError):
        validate_stub_metadata(bad)


def test_p29_21_metadata_rejects_nonzero_buffer_count():
    meta = build_blocked_stub_metadata(torch_available=False)
    bad = dataclasses.replace(meta, buffer_count=1)
    with pytest.raises(ValueError):
        validate_stub_metadata(bad)


def test_p29_22_materialization_metadata_validates():
    meta = materialize_local_torch_stub_metadata()
    validate_stub_metadata(meta)


def test_p29_23_result_validates():
    req = build_stub_request()
    res = build_stub_result(req)
    validate_stub_result(res)


def test_p29_24_result_rejects_module_object_returned_true():
    req = build_stub_request()
    res = build_stub_result(req)
    bad = dataclasses.replace(res, module_object_returned=True)
    with pytest.raises(ValueError):
        validate_stub_result(bad)


def test_p29_25_result_rejects_torch_required_p29_true():
    req = build_stub_request()
    res = build_stub_result(req)
    bad = dataclasses.replace(res, torch_required_for_p29=True)
    with pytest.raises(ValueError):
        validate_stub_result(bad)


def test_p29_26_result_rejects_implementation_available_true():
    req = build_stub_request()
    res = build_stub_result(req)
    bad = dataclasses.replace(res, implementation_available_in_p29=True)
    with pytest.raises(ValueError):
        validate_stub_result(bad)


def test_p29_27_result_rejects_forward_available_true():
    req = build_stub_request()
    res = build_stub_result(req)
    bad = dataclasses.replace(res, forward_available_in_p29=True)
    with pytest.raises(ValueError):
        validate_stub_result(bad)


def test_p29_28_result_rejects_layers_available_true():
    req = build_stub_request()
    res = build_stub_result(req)
    bad = dataclasses.replace(res, layers_available_in_p29=True)
    with pytest.raises(ValueError):
        validate_stub_result(bad)


def test_p29_29_result_rejects_training_available_true():
    req = build_stub_request()
    res = build_stub_result(req)
    bad = dataclasses.replace(res, training_available_in_p29=True)
    with pytest.raises(ValueError):
        validate_stub_result(bad)


def test_p29_30_result_rejects_no_training_loop_false():
    req = build_stub_request()
    res = build_stub_result(req)
    bad = dataclasses.replace(res, no_training_loop=False)
    with pytest.raises(ValueError):
        validate_stub_result(bad)


def test_p29_31_result_rejects_no_optimizer_false():
    req = build_stub_request()
    res = build_stub_result(req)
    bad = dataclasses.replace(res, no_optimizer=False)
    with pytest.raises(ValueError):
        validate_stub_result(bad)


def test_p29_32_result_rejects_no_checkpointing_false():
    req = build_stub_request()
    res = build_stub_result(req)
    bad = dataclasses.replace(res, no_checkpointing=False)
    with pytest.raises(ValueError):
        validate_stub_result(bad)


def test_p29_33_result_rejects_no_artifact_generation_false():
    req = build_stub_request()
    res = build_stub_result(req)
    bad = dataclasses.replace(res, no_artifact_generation=False)
    with pytest.raises(ValueError):
        validate_stub_result(bad)


def test_p29_34_probe():
    res = run_fc_vae_torch_module_stub_probe()
    validate_stub_result(res)


# D. Serialization
def test_p29_35_to_json_dict():
    req = build_stub_request()
    res = build_stub_result(req)
    d1 = stub_request_to_json_dict(req)
    d2 = stub_metadata_to_json_dict(res.metadata)
    d3 = stub_result_to_json_dict(res)
    assert isinstance(d1, dict)
    assert isinstance(d2, dict)
    assert isinstance(d3, dict)


def test_p29_36_compact_json():
    res = run_fc_vae_torch_module_stub_probe()
    js = compact_stub_json(res)
    assert isinstance(js, str)
    d = json.loads(js)
    assert d["contract_version"] == FC_VAE_TORCH_MODULE_STUB_CONTRACT_VERSION


def test_p29_37_serialized_no_local_paths():
    res = run_fc_vae_torch_module_stub_probe()
    js = compact_stub_json(res)
    for path in ["file:///", "C:/", "C:\\", "/home/", "/Users/"]:
        assert path not in js


def test_p29_38_serialized_no_forbidden_claims():
    res = run_fc_vae_torch_module_stub_probe()
    js = compact_stub_json(res).lower()
    forbidden = ["model works", "scientific success", "solved", "best", "winner", "production ready", "state of the art"]
    for claim in forbidden:
        assert claim not in js


def test_p29_39_serialized_no_module_object():
    res = run_fc_vae_torch_module_stub_probe()
    js = compact_stub_json(res)
    # If the module was in the JSON, it would be serialized as some object or string,
    # but module_object_returned is False and there is no module field in dict.
    assert "P29LocalFCVAEModuleStub" not in js or '"class_name":"P29LocalFCVAEModuleStub"' in js
    # Specifically check that the raw module is not present
    d = json.loads(js)
    assert "module" not in d


def test_p29_40_serialized_no_raw_parameters():
    res = run_fc_vae_torch_module_stub_probe()
    js = compact_stub_json(res)
    forbidden_keys = [
        "ar_params", "ma_params", "alpha_params", "beta_params",
        "omega", "model_spec", "generated_spec"
    ]
    for key in forbidden_keys:
        assert f'"{key}"' not in js


# E. Source/static checks
def test_p29_41_no_top_level_torch_import():
    p = pathlib.Path("src/phase2/fc_vae_torch_module_stub.py").read_text(encoding="utf-8")
    for line in p.splitlines():
        if line.strip().startswith("#"):
            continue
        assert not line.startswith("import torch"), f"Top-level 'import torch' found: {line}"
        assert not line.startswith("from torch"), f"Top-level 'from torch' found: {line}"



def test_p29_42_no_top_level_from_torch():
    p = pathlib.Path("src/phase2/fc_vae_torch_module_stub.py").read_text(encoding="utf-8")
    assert "from torch" not in p


def test_p29_43_only_one_local_torch_import():
    p = pathlib.Path("src/phase2/fc_vae_torch_module_stub.py").read_text(encoding="utf-8")
    imports = [m.start() for m in re.finditer(r"\bimport torch\b", p)]
    assert len(imports) <= 1, "More than one local import torch statement found"


def test_p29_44_no_forward_method():
    p = pathlib.Path("src/phase2/fc_vae_torch_module_stub.py").read_text(encoding="utf-8")
    assert "def forward(" not in p


def test_p29_45_no_forbidden_torch_layers():
    p = pathlib.Path("src/phase2/fc_vae_torch_module_stub.py").read_text(encoding="utf-8")
    forbidden = ["nn.Linear", "nn.Conv", "nn.Sequential", "nn.Parameter"]
    for f in forbidden:
        assert f not in p


def test_p29_46_no_training_functions():
    p = pathlib.Path("src/phase2/fc_vae_torch_module_stub.py").read_text(encoding="utf-8")
    forbidden_fns = [
        "def train(", "def fit(", "def train_step(", "def training_loop(",
        "def save_checkpoint(", "def load_checkpoint(",
    ]
    for fn in forbidden_fns:
        assert fn not in p, f"Forbidden function found: {fn}"


def test_p29_47_smoke_script_no_torch():
    p = pathlib.Path("tools/phase2/run_p29_torch_module_stub_smoke.py").read_text(encoding="utf-8")
    assert "import torch" not in p
    assert "from torch" not in p


def test_p29_48_init_no_subprocess_monkeypatch():
    p = pathlib.Path("src/phase2/__init__.py").read_text(encoding="utf-8")
    assert "subprocess.run =" not in p
    assert "_patched_run" not in p
    assert "_original_run" not in p


def test_p29_49_scope_gate():
    allowed = {
        "src/phase2/fc_vae_torch_module_stub.py",
        "src/phase2/__init__.py",
        "tests/test_phase2_fc_vae_torch_module_stub.py",
        "tools/phase2/run_p29_torch_module_stub_smoke.py",
        "tests/test_phase2_p29_torch_module_stub_smoke.py",
        "reports/PHASE_2_P29_GATED_TORCH_NN_MODULE_STUB_REPORT.md",
    }
    # Compare against P28 base branch
    res = subprocess.run(
        ["git", "diff", "--name-only", "phase2/p28-optional-torch-fc-vae-shell-handle"],
        capture_output=True, text=True, check=True
    )
    modified = [line.strip() for line in res.stdout.splitlines() if line.strip()]
    for f in modified:
        f_norm = f.replace("\\", "/")
        assert f_norm in allowed, f"Forbidden file modification detected in P29: {f_norm}"
