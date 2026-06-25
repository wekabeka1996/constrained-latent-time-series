# tests/test_phase2_fc_vae_constructor_binding.py

import dataclasses
import json
import pathlib
import subprocess
import re
import pytest
from dataclasses import is_dataclass

import src.phase2
from src.phase2.fc_vae_constructor_binding import (
    FC_VAE_CONSTRUCTOR_BINDING_CONTRACT_VERSION,
    FC_VAE_CONSTRUCTOR_BINDING_KIND,
    FC_VAE_CONSTRUCTOR_BINDING_MODULE_NAME,
    FC_VAE_CONSTRUCTOR_BINDING_STATUS_TORCH_UNAVAILABLE,
    FC_VAE_CONSTRUCTOR_BINDING_STATUS_BOUND_TO_STUB,
    FC_VAE_CONSTRUCTOR_BINDING_STATUS_STUB_DEFERRED,
    SUPPORTED_FC_VAE_CONSTRUCTOR_BINDING_STATUSES,
    FCVAEConstructorBindingRequest,
    FCVAEConstructorBindingMetadata,
    FCVAEConstructorBindingResult,
    validate_non_empty_str,
    validate_bool,
    validate_positive_int,
    validate_non_negative_int,
    validate_binding_status,
    assert_no_local_path_leakage,
    assert_no_forbidden_claims,
    validate_constructor_binding_request,
    validate_constructor_binding_metadata,
    validate_constructor_binding_result,
    build_constructor_binding_request_from_p27_smoke_contracts,
    build_constructor_binding_metadata,
    build_constructor_binding_result,
    run_constructor_binding_probe,
    constructor_binding_request_to_json_dict,
    constructor_binding_metadata_to_json_dict,
    constructor_binding_result_to_json_dict,
    compact_constructor_binding_json,
)


# ==========================================
# I. Constants & Dataclasses (Tests 1-2)
# ==========================================

def test_p30_01_constants():
    assert FC_VAE_CONSTRUCTOR_BINDING_CONTRACT_VERSION == "phase2_p30_constructor_binding_contract_v1"
    assert FC_VAE_CONSTRUCTOR_BINDING_KIND == "torch_module_constructor_spec_binding"
    assert FC_VAE_CONSTRUCTOR_BINDING_MODULE_NAME == "src.phase2.fc_vae_constructor_binding"
    assert FC_VAE_CONSTRUCTOR_BINDING_STATUS_TORCH_UNAVAILABLE == "blocked_torch_unavailable"
    assert FC_VAE_CONSTRUCTOR_BINDING_STATUS_BOUND_TO_STUB == "bound_to_stub_no_forward_no_layers"
    assert FC_VAE_CONSTRUCTOR_BINDING_STATUS_STUB_DEFERRED == "blocked_stub_deferred"
    assert len(SUPPORTED_FC_VAE_CONSTRUCTOR_BINDING_STATUSES) == 3
    assert "blocked_torch_unavailable" in SUPPORTED_FC_VAE_CONSTRUCTOR_BINDING_STATUSES
    assert "bound_to_stub_no_forward_no_layers" in SUPPORTED_FC_VAE_CONSTRUCTOR_BINDING_STATUSES
    assert "blocked_stub_deferred" in SUPPORTED_FC_VAE_CONSTRUCTOR_BINDING_STATUSES


def test_p30_02_dataclasses_frozen():
    for cls in (FCVAEConstructorBindingRequest, FCVAEConstructorBindingMetadata, FCVAEConstructorBindingResult):
        assert is_dataclass(cls)
        # Check that it is frozen
        req = build_constructor_binding_request_from_p27_smoke_contracts()
        with pytest.raises(Exception):
            req.reason = "modified"


# ==========================================
# II. Validators & Safety Helpers (Tests 3-5)
# ==========================================

def test_p30_03_validate_basic_types():
    validate_non_empty_str("valid_string", "test_field")
    with pytest.raises(TypeError):
        validate_non_empty_str(123, "test_field")
    with pytest.raises(ValueError):
        validate_non_empty_str("", "test_field")
    with pytest.raises(ValueError):
        validate_non_empty_str("  padded  ", "test_field")

    validate_bool(True, "bool_field")
    validate_bool(False, "bool_field")
    with pytest.raises(TypeError):
        validate_bool(1, "bool_field")


def test_p30_04_validate_integers_and_status():
    validate_positive_int(1, "pos_int")
    with pytest.raises(ValueError):
        validate_positive_int(0, "pos_int")
    with pytest.raises(TypeError):
        validate_positive_int(True, "pos_int")

    validate_non_negative_int(0, "non_neg_int")
    validate_non_negative_int(5, "non_neg_int")
    with pytest.raises(ValueError):
        validate_non_negative_int(-1, "non_neg_int")

    for status in SUPPORTED_FC_VAE_CONSTRUCTOR_BINDING_STATUSES:
        validate_binding_status(status)
    with pytest.raises(ValueError):
        validate_binding_status("unknown_status")
    with pytest.raises(TypeError):
        validate_binding_status(123)


def test_p30_05_safety_validators():
    assert_no_local_path_leakage("safe string")
    for leak in ["file:///", "C:/", "C:\\", "/home/", "/Users/"]:
        with pytest.raises(ValueError):
            assert_no_local_path_leakage(f"some string with {leak} inside")

    assert_no_forbidden_claims("safe conclusion")
    # Allowed specifically
    assert_no_forbidden_claims("no_scientific_conclusion")
    assert_no_forbidden_claims("no_final_comparison")
    assert_no_forbidden_claims("constructor_binding_available_in_p30=false")
    # Forbidden
    for claim in ["model works", "scientific success", "solved", "winner", "production ready", "state of the art"]:
        with pytest.raises(ValueError):
            assert_no_forbidden_claims(f"we claim that the {claim} is achieved")


# ==========================================
# III. Request, Metadata, Result Contracts (Tests 6-31)
# ==========================================

def test_p30_06_default_request_validates():
    req = build_constructor_binding_request_from_p27_smoke_contracts()
    validate_constructor_binding_request(req)


def test_p30_07_request_rejects_wrong_contract():
    req = build_constructor_binding_request_from_p27_smoke_contracts()
    bad = dataclasses.replace(req, contract_version="wrong_version")
    with pytest.raises(ValueError):
        validate_constructor_binding_request(bad)


def test_p30_08_request_rejects_wrong_binding_kind():
    req = build_constructor_binding_request_from_p27_smoke_contracts()
    bad = dataclasses.replace(req, binding_kind="wrong_kind")
    with pytest.raises(ValueError):
        validate_constructor_binding_request(bad)


def test_p30_09_request_rejects_wrong_architecture():
    req = build_constructor_binding_request_from_p27_smoke_contracts()
    bad = dataclasses.replace(req, architecture_id="wrong_arch")
    with pytest.raises(ValueError):
        validate_constructor_binding_request(bad)


def test_p30_10_request_rejects_allow_forward():
    req = build_constructor_binding_request_from_p27_smoke_contracts()
    bad = dataclasses.replace(req, allow_forward_in_p30=True)
    with pytest.raises(ValueError):
        validate_constructor_binding_request(bad)


def test_p30_11_request_rejects_allow_layers():
    req = build_constructor_binding_request_from_p27_smoke_contracts()
    bad = dataclasses.replace(req, allow_layers_in_p30=True)
    with pytest.raises(ValueError):
        validate_constructor_binding_request(bad)


def test_p30_12_request_rejects_allow_training():
    req = build_constructor_binding_request_from_p27_smoke_contracts()
    bad = dataclasses.replace(req, allow_training_in_p30=True)
    with pytest.raises(ValueError):
        validate_constructor_binding_request(bad)


def test_p30_13_request_rejects_negative_dims():
    req = build_constructor_binding_request_from_p27_smoke_contracts()
    bad = dataclasses.replace(req, input_flat_dim=-1)
    with pytest.raises(ValueError):
        validate_constructor_binding_request(bad)


def test_p30_14_metadata_default_validates():
    req = build_constructor_binding_request_from_p27_smoke_contracts()
    meta = build_constructor_binding_metadata(req)
    validate_constructor_binding_metadata(meta)


def test_p30_15_metadata_rejects_wrong_contract():
    req = build_constructor_binding_request_from_p27_smoke_contracts()
    meta = build_constructor_binding_metadata(req)
    bad = dataclasses.replace(meta, contract_version="wrong")
    with pytest.raises(ValueError):
        validate_constructor_binding_metadata(bad)


def test_p30_16_metadata_rejects_wrong_binding_kind():
    req = build_constructor_binding_request_from_p27_smoke_contracts()
    meta = build_constructor_binding_metadata(req)
    bad = dataclasses.replace(meta, binding_kind="wrong")
    with pytest.raises(ValueError):
        validate_constructor_binding_metadata(bad)


def test_p30_17_metadata_rejects_wrong_module_name():
    req = build_constructor_binding_request_from_p27_smoke_contracts()
    meta = build_constructor_binding_metadata(req)
    bad = dataclasses.replace(meta, module_name="wrong.module")
    with pytest.raises(ValueError):
        validate_constructor_binding_metadata(bad)


def test_p30_18_metadata_rejects_bound_to_module_object():
    req = build_constructor_binding_request_from_p27_smoke_contracts()
    meta = build_constructor_binding_metadata(req)
    bad = dataclasses.replace(meta, bound_to_module_object=True)
    with pytest.raises(ValueError):
        validate_constructor_binding_metadata(bad)


def test_p30_19_metadata_rejects_defines_forward():
    req = build_constructor_binding_request_from_p27_smoke_contracts()
    meta = build_constructor_binding_metadata(req)
    bad = dataclasses.replace(meta, defines_forward=True)
    with pytest.raises(ValueError):
        validate_constructor_binding_metadata(bad)


def test_p30_20_metadata_rejects_defines_layers():
    req = build_constructor_binding_request_from_p27_smoke_contracts()
    meta = build_constructor_binding_metadata(req)
    bad = dataclasses.replace(meta, defines_layers=True)
    with pytest.raises(ValueError):
        validate_constructor_binding_metadata(bad)


def test_p30_21_metadata_rejects_nonzero_param_count():
    req = build_constructor_binding_request_from_p27_smoke_contracts()
    meta = build_constructor_binding_metadata(req)
    bad = dataclasses.replace(meta, parameter_count=5)
    with pytest.raises(ValueError):
        validate_constructor_binding_metadata(bad)


def test_p30_22_metadata_rejects_nonzero_buffer_count():
    req = build_constructor_binding_request_from_p27_smoke_contracts()
    meta = build_constructor_binding_metadata(req)
    bad = dataclasses.replace(meta, buffer_count=1)
    with pytest.raises(ValueError):
        validate_constructor_binding_metadata(bad)


def test_p30_23_result_default_validates():
    req = build_constructor_binding_request_from_p27_smoke_contracts()
    res = build_constructor_binding_result(req)
    validate_constructor_binding_result(res)


def test_p30_24_result_rejects_torch_required_for_p30():
    req = build_constructor_binding_request_from_p27_smoke_contracts()
    res = build_constructor_binding_result(req)
    bad = dataclasses.replace(res, torch_required_for_p30=True)
    with pytest.raises(ValueError):
        validate_constructor_binding_result(bad)


def test_p30_25_result_rejects_torch_required_for_future_execution_false():
    req = build_constructor_binding_request_from_p27_smoke_contracts()
    res = build_constructor_binding_result(req)
    bad = dataclasses.replace(res, torch_required_for_future_execution=False)
    with pytest.raises(ValueError):
        validate_constructor_binding_result(bad)


def test_p30_26_result_rejects_forward_available():
    req = build_constructor_binding_request_from_p27_smoke_contracts()
    res = build_constructor_binding_result(req)
    bad = dataclasses.replace(res, forward_available_in_p30=True)
    with pytest.raises(ValueError):
        validate_constructor_binding_result(bad)


def test_p30_27_result_rejects_layers_available():
    req = build_constructor_binding_request_from_p27_smoke_contracts()
    res = build_constructor_binding_result(req)
    bad = dataclasses.replace(res, layers_available_in_p30=True)
    with pytest.raises(ValueError):
        validate_constructor_binding_result(bad)


def test_p30_28_result_rejects_training_available():
    req = build_constructor_binding_request_from_p27_smoke_contracts()
    res = build_constructor_binding_result(req)
    bad = dataclasses.replace(res, training_available_in_p30=True)
    with pytest.raises(ValueError):
        validate_constructor_binding_result(bad)


def test_p30_29_result_rejects_module_object_returned():
    req = build_constructor_binding_request_from_p27_smoke_contracts()
    res = build_constructor_binding_result(req)
    bad = dataclasses.replace(res, module_object_returned=True)
    with pytest.raises(ValueError):
        validate_constructor_binding_result(bad)


def test_p30_30_result_consistency_checks():
    req = build_constructor_binding_request_from_p27_smoke_contracts()
    res = build_constructor_binding_result(req)

    # If torch is unavailable, result.status must be FC_VAE_CONSTRUCTOR_BINDING_STATUS_TORCH_UNAVAILABLE
    if not res.metadata.torch_available:
        bad = dataclasses.replace(res, status=FC_VAE_CONSTRUCTOR_BINDING_STATUS_BOUND_TO_STUB)
        with pytest.raises(ValueError):
            validate_constructor_binding_result(bad)
    else:
        # If torch is available, it must be either bound to stub or stub deferred
        bad = dataclasses.replace(res, status=FC_VAE_CONSTRUCTOR_BINDING_STATUS_TORCH_UNAVAILABLE)
        with pytest.raises(ValueError):
            validate_constructor_binding_result(bad)


def test_p30_31_probe_runs():
    res = run_constructor_binding_probe()
    assert isinstance(res, FCVAEConstructorBindingResult)
    validate_constructor_binding_result(res)


# ==========================================
# IV. Serialization Helpers (Tests 32-36)
# ==========================================

def test_p30_32_serialization_helpers():
    req = build_constructor_binding_request_from_p27_smoke_contracts()
    res = build_constructor_binding_result(req)

    d1 = constructor_binding_request_to_json_dict(req)
    d2 = constructor_binding_metadata_to_json_dict(res.metadata)
    d3 = constructor_binding_result_to_json_dict(res)

    assert isinstance(d1, dict)
    assert isinstance(d2, dict)
    assert isinstance(d3, dict)
    assert d1["architecture_id"] == "FC-VAE"
    assert d2["module_name"] == "src.phase2.fc_vae_constructor_binding"
    assert d3["status"] in SUPPORTED_FC_VAE_CONSTRUCTOR_BINDING_STATUSES


def test_p30_33_compact_json_helper():
    res = run_constructor_binding_probe()
    js = compact_constructor_binding_json(res)
    assert isinstance(js, str)
    d = json.loads(js)
    assert d["contract_version"] == FC_VAE_CONSTRUCTOR_BINDING_CONTRACT_VERSION


def test_p30_34_serialized_no_local_paths():
    res = run_constructor_binding_probe()
    js = compact_constructor_binding_json(res)
    for path in ["file:///", "C:/", "C:\\", "/home/", "/Users/"]:
        assert path not in js
        assert path.lower() not in js.lower()


def test_p30_35_serialized_no_forbidden_claims():
    res = run_constructor_binding_probe()
    js = compact_constructor_binding_json(res)
    # Check that forbidden claims are not present (except for allowed ones)
    normalized = js.lower()
    cleaned = (normalized
               .replace("no_final_comparison", "")
               .replace("no_scientific_conclusion", "")
               .replace("constructor_binding_available_in_p30=false", ""))
    for claim in ["model works", "scientific success", "solved", "winner", "production ready", "state of the art"]:
        assert claim not in cleaned


def test_p30_36_serialized_no_raw_parameters():
    res = run_constructor_binding_probe()
    js = compact_constructor_binding_json(res)
    # Check that it doesn't contain raw time-series parameters
    forbidden_keys = ["ar_params", "ma_params", "alpha_params", "beta_params", "omega", "model_spec", "generated_spec"]
    for key in forbidden_keys:
        assert f'"{key}"' not in js


# ==========================================
# V. Source & Static Verification Checks (Tests 37-44)
# ==========================================

def test_p30_37_no_top_level_torch_import():
    p = pathlib.Path("src/phase2/fc_vae_constructor_binding.py").read_text(encoding="utf-8")
    for line in p.splitlines():
        if line.strip().startswith("#"):
            continue
        assert not line.startswith("import torch"), f"Top-level 'import torch' found: {line}"
        assert not line.startswith("from torch"), f"Top-level 'from torch' found: {line}"


def test_p30_38_no_local_torch_import():
    p = pathlib.Path("src/phase2/fc_vae_constructor_binding.py").read_text(encoding="utf-8")
    assert "import torch" not in p
    assert "from torch" not in p


def test_p30_39_smoke_script_has_no_torch():
    p = pathlib.Path("tools/phase2/run_p30_constructor_binding_smoke.py").read_text(encoding="utf-8")
    assert "import torch" not in p
    assert "from torch" not in p


def test_p30_40_no_forward_method():
    p = pathlib.Path("src/phase2/fc_vae_constructor_binding.py").read_text(encoding="utf-8")
    assert "def forward(" not in p


def test_p30_41_no_forbidden_layers():
    p = pathlib.Path("src/phase2/fc_vae_constructor_binding.py").read_text(encoding="utf-8")
    forbidden = ["nn.Linear", "nn.Conv", "nn.Sequential", "nn.Parameter", "nn.Module"]
    for layer in forbidden:
        assert layer not in p


def test_p30_42_no_training_code():
    p = pathlib.Path("src/phase2/fc_vae_constructor_binding.py").read_text(encoding="utf-8")
    forbidden_fns = [
        "def train(", "def fit(", "def train_step(", "def training_loop(",
        "def save_checkpoint(", "def load_checkpoint(",
    ]
    for fn in forbidden_fns:
        assert fn not in p


def test_p30_43_init_no_subprocess_monkeypatch():
    p = pathlib.Path("src/phase2/__init__.py").read_text(encoding="utf-8")
    assert "subprocess.run =" not in p
    assert "_patched_run" not in p
    assert "_original_run" not in p


def test_p30_44_scope_gate():
    allowed = {
        "src/phase2/fc_vae_constructor_binding.py",
        "src/phase2/__init__.py",
        "tests/test_phase2_fc_vae_constructor_binding.py",
        "tools/phase2/run_p30_constructor_binding_smoke.py",
        "tests/test_phase2_p30_constructor_binding_smoke.py",
        "reports/PHASE_2_P30_TORCH_MODULE_CONSTRUCTOR_SPEC_BINDING_REPORT.md",
    }
    # Compare against P29 base branch
    res = subprocess.run(
        ["git", "diff", "--name-only", "phase2/p29-gated-torch-nn-module-stub"],
        capture_output=True, text=True, check=True
    )
    modified = [line.strip() for line in res.stdout.splitlines() if line.strip()]
    for f in modified:
        f_norm = f.replace("\\", "/")
        assert f_norm in allowed, f"Forbidden file modification detected in P30: {f_norm}"
