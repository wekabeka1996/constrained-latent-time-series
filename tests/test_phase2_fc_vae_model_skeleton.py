# tests/test_phase2_fc_vae_model_skeleton.py

import dataclasses
import json
import pathlib
import subprocess
from tests.phase2_scope_gate_utils import enforce_phase_local_scope_gate_or_skip
import pytest
from dataclasses import is_dataclass

import src.phase2
from src.phase2.fc_vae_model import (
    FC_VAE_MODEL_SKELETON_CONTRACT_VERSION,
    FC_VAE_ARCHITECTURE_ID,
    FC_VAE_MODULE_NAME,
    FC_VAE_STATUS_SKELETON_ONLY,
    FC_VAE_STATUS_BLOCKED_TORCH_UNAVAILABLE,
    FC_VAE_STATUS_READY_FOR_FUTURE_IMPLEMENTATION,
    SUPPORTED_FC_VAE_STATUSES,
    FC_VAE_REQUIRED_LATENT_NAMES,
    FC_VAE_DECODER_OUTPUT_KIND_MODELSPEC,
    FCVAEInputShapeContract,
    FCVAELatentLayout,
    FCVAEDecoderOutputContract,
    FCVAEForwardContract,
    FCVAESkeletonStatus,
    validate_non_empty_str,
    validate_bool,
    validate_positive_int,
    validate_exact_tuple_str,
    validate_fc_vae_status,
    validate_input_shape_contract,
    validate_latent_layout,
    validate_decoder_output_contract,
    validate_forward_contract,
    validate_skeleton_status,
    build_smoke_input_shape_contract,
    build_smoke_latent_layout,
    build_smoke_decoder_output_contract,
    build_smoke_forward_contract,
    build_fc_vae_skeleton_status,
    require_fc_vae_implementation_available,
    input_shape_contract_to_json_dict,
    latent_layout_to_json_dict,
    decoder_output_contract_to_json_dict,
    forward_contract_to_json_dict,
    skeleton_status_to_json_dict,
    compact_fc_vae_skeleton_json,
)


# A. Constants/dataclasses
def test_p27_01_constants_values():
    assert FC_VAE_MODEL_SKELETON_CONTRACT_VERSION == "phase2_p27_fc_vae_model_skeleton_contract_v1"
    assert FC_VAE_ARCHITECTURE_ID == "FC-VAE"
    assert FC_VAE_MODULE_NAME == "src.phase2.fc_vae_model"
    assert FC_VAE_STATUS_SKELETON_ONLY == "skeleton_only"
    assert FC_VAE_STATUS_BLOCKED_TORCH_UNAVAILABLE == "blocked_torch_unavailable"
    assert FC_VAE_STATUS_READY_FOR_FUTURE_IMPLEMENTATION == "ready_for_future_implementation"
    assert SUPPORTED_FC_VAE_STATUSES == ("skeleton_only", "blocked_torch_unavailable", "ready_for_future_implementation")
    assert FC_VAE_REQUIRED_LATENT_NAMES == ("z_mean", "z_volatility", "z_shared")
    assert FC_VAE_DECODER_OUTPUT_KIND_MODELSPEC == "typed_modelspec_candidate"


def test_p27_02_dataclasses_frozen():
    classes = [FCVAEInputShapeContract, FCVAELatentLayout, FCVAEDecoderOutputContract,
               FCVAEForwardContract, FCVAESkeletonStatus]
    for cls in classes:
        assert is_dataclass(cls)
    contract = build_smoke_input_shape_contract()
    with pytest.raises(Exception):
        contract.reason = "modified"


# B. Basic validators
def test_p27_03_validate_non_empty_str_accepts():
    validate_non_empty_str("valid_string", "field")


def test_p27_04_validate_non_empty_str_rejects():
    with pytest.raises(TypeError):
        validate_non_empty_str(123, "field")
    with pytest.raises(ValueError):
        validate_non_empty_str("", "field")
    with pytest.raises(ValueError):
        validate_non_empty_str("  padded  ", "field")


def test_p27_05_validate_bool_accepts():
    validate_bool(True, "field")
    validate_bool(False, "field")
    with pytest.raises(TypeError):
        validate_bool(1, "field")


def test_p27_06_validate_positive_int_rejects():
    with pytest.raises(TypeError):
        validate_positive_int(True, "field")
    with pytest.raises(TypeError):
        validate_positive_int("1", "field")
    with pytest.raises(ValueError):
        validate_positive_int(0, "field")
    with pytest.raises(ValueError):
        validate_positive_int(-1, "field")


def test_p27_07_validate_exact_tuple_str_accepts():
    validate_exact_tuple_str(FC_VAE_REQUIRED_LATENT_NAMES, FC_VAE_REQUIRED_LATENT_NAMES, "latent_names")


def test_p27_08_validate_exact_tuple_str_rejects():
    with pytest.raises(TypeError):
        validate_exact_tuple_str(list(FC_VAE_REQUIRED_LATENT_NAMES), FC_VAE_REQUIRED_LATENT_NAMES, "latent_names")
    with pytest.raises(ValueError):
        validate_exact_tuple_str(("z_shared", "z_mean", "z_volatility"), FC_VAE_REQUIRED_LATENT_NAMES, "latent_names")
    with pytest.raises(ValueError):
        validate_exact_tuple_str(("wrong",), FC_VAE_REQUIRED_LATENT_NAMES, "latent_names")


def test_p27_09_validate_fc_vae_status_accepts():
    for s in SUPPORTED_FC_VAE_STATUSES:
        validate_fc_vae_status(s)


def test_p27_10_validate_fc_vae_status_rejects():
    with pytest.raises(ValueError):
        validate_fc_vae_status("unsupported_status")


# C. Shape/contract validators
def test_p27_11_build_smoke_input_shape_contract_validates():
    contract = build_smoke_input_shape_contract()
    validate_input_shape_contract(contract)
    assert contract.flat_dim == 32


def test_p27_12_input_shape_rejects_wrong_contract_version():
    contract = dataclasses.replace(build_smoke_input_shape_contract(), contract_version="wrong_v2")
    with pytest.raises(ValueError, match="Wrong contract version"):
        validate_input_shape_contract(contract)


def test_p27_13_input_shape_rejects_flat_dim_zero():
    contract = dataclasses.replace(build_smoke_input_shape_contract(), flat_dim=0)
    with pytest.raises(ValueError):
        validate_input_shape_contract(contract)


def test_p27_14_build_smoke_latent_layout_validates():
    layout = build_smoke_latent_layout()
    validate_latent_layout(layout)
    assert layout.total_latent_dim == 20


def test_p27_15_latent_layout_rejects_wrong_total():
    layout = dataclasses.replace(build_smoke_latent_layout(), total_latent_dim=99)
    with pytest.raises(ValueError, match="total_latent_dim must equal"):
        validate_latent_layout(layout)


def test_p27_16_latent_layout_rejects_wrong_names():
    layout = dataclasses.replace(build_smoke_latent_layout(), latent_names=("wrong",))
    with pytest.raises(ValueError):
        validate_latent_layout(layout)


def test_p27_17_build_smoke_decoder_output_contract_validates():
    contract = build_smoke_decoder_output_contract()
    validate_decoder_output_contract(contract)
    assert contract.target_boundary == "ModelSpec"


def test_p27_18_decoder_output_rejects_wrong_output_kind():
    contract = dataclasses.replace(build_smoke_decoder_output_contract(), output_kind="wrong_kind")
    with pytest.raises(ValueError, match="output_kind must be"):
        validate_decoder_output_contract(contract)


def test_p27_19_decoder_output_rejects_wrong_target_boundary():
    contract = dataclasses.replace(build_smoke_decoder_output_contract(), target_boundary="WrongBoundary")
    with pytest.raises(ValueError, match="target_boundary must be ModelSpec"):
        validate_decoder_output_contract(contract)


def test_p27_20_build_smoke_forward_contract_validates():
    contract = build_smoke_forward_contract()
    validate_forward_contract(contract)
    assert contract.implemented_in_p27 is False


def test_p27_21_forward_contract_rejects_implemented_true():
    contract = dataclasses.replace(build_smoke_forward_contract(), implemented_in_p27=True)
    with pytest.raises(ValueError, match="implemented_in_p27 must be False"):
        validate_forward_contract(contract)


def test_p27_22_forward_contract_rejects_torch_required_false():
    contract = dataclasses.replace(build_smoke_forward_contract(), torch_required_for_execution=False)
    with pytest.raises(ValueError, match="torch_required_for_execution must be True"):
        validate_forward_contract(contract)


def test_p27_23_forward_contract_rejects_wrong_architecture_id():
    contract = dataclasses.replace(build_smoke_forward_contract(), architecture_id="WRONG-ARCH")
    with pytest.raises(ValueError, match="architecture_id must be"):
        validate_forward_contract(contract)


def test_p27_24_build_fc_vae_skeleton_status_validates():
    status = build_fc_vae_skeleton_status()
    validate_skeleton_status(status)
    assert status.no_model_implementation is True


def test_p27_25_skeleton_status_rejects_no_model_implementation_false():
    status = dataclasses.replace(build_fc_vae_skeleton_status(), no_model_implementation=False)
    with pytest.raises(ValueError, match="no_model_implementation must be True"):
        validate_skeleton_status(status)


def test_p27_26_skeleton_status_rejects_no_training_loop_false():
    status = dataclasses.replace(build_fc_vae_skeleton_status(), no_training_loop=False)
    with pytest.raises(ValueError, match="no_training_loop must be True"):
        validate_skeleton_status(status)


def test_p27_27_skeleton_status_rejects_no_optimizer_false():
    status = dataclasses.replace(build_fc_vae_skeleton_status(), no_optimizer=False)
    with pytest.raises(ValueError, match="no_optimizer must be True"):
        validate_skeleton_status(status)


def test_p27_28_skeleton_status_rejects_no_checkpointing_false():
    status = dataclasses.replace(build_fc_vae_skeleton_status(), no_checkpointing=False)
    with pytest.raises(ValueError, match="no_checkpointing must be True"):
        validate_skeleton_status(status)


def test_p27_29_skeleton_status_rejects_no_artifact_generation_false():
    status = dataclasses.replace(build_fc_vae_skeleton_status(), no_artifact_generation=False)
    with pytest.raises(ValueError, match="no_artifact_generation must be True"):
        validate_skeleton_status(status)


def test_p27_30_skeleton_rejects_torch_false_with_ready():
    status = dataclasses.replace(
        build_fc_vae_skeleton_status(),
        torch_available=False,
        status=FC_VAE_STATUS_READY_FOR_FUTURE_IMPLEMENTATION,
    )
    with pytest.raises(ValueError, match="cannot be ready_for_future_implementation"):
        validate_skeleton_status(status)


def test_p27_31_require_fc_vae_raises_not_implemented():
    status = build_fc_vae_skeleton_status()
    with pytest.raises(NotImplementedError, match="FC-VAE implementation is not available in P27"):
        require_fc_vae_implementation_available(status)


# D. Serialization
def test_p27_32_json_dicts_are_json_safe():
    contract = build_smoke_input_shape_contract()
    d = input_shape_contract_to_json_dict(contract)
    json.dumps(d)

    layout = build_smoke_latent_layout()
    d = latent_layout_to_json_dict(layout)
    json.dumps(d)

    dec = build_smoke_decoder_output_contract()
    d = decoder_output_contract_to_json_dict(dec)
    json.dumps(d)

    fwd = build_smoke_forward_contract()
    d = forward_contract_to_json_dict(fwd)
    json.dumps(d)

    status = build_fc_vae_skeleton_status()
    d = skeleton_status_to_json_dict(status)
    json.dumps(d)


def test_p27_33_compact_json_sorted_compact_parseable():
    status = build_fc_vae_skeleton_status()
    js = compact_fc_vae_skeleton_json(status)
    d = json.loads(js)
    assert d["contract_version"] == FC_VAE_MODEL_SKELETON_CONTRACT_VERSION
    assert ": " not in js
    assert ", " not in js
    assert "\n" not in js


def test_p27_34_serialized_no_raw_params():
    status = build_fc_vae_skeleton_status()
    js = compact_fc_vae_skeleton_json(status)
    forbidden = ["ar_params", "ma_params", "alpha_params", "beta_params", "omega", "model_spec", "generated_spec"]
    for f in forbidden:
        assert f'"' + f + '"' not in js, f"Raw param key found: {f}"


def test_p27_35_serialized_no_local_paths():
    status = build_fc_vae_skeleton_status()
    js = compact_fc_vae_skeleton_json(status)
    for path in ["file:///", "C:/", "C:\\", "/home/", "/Users/"]:
        assert path not in js


def test_p27_36_serialized_no_forbidden_claims():
    status = build_fc_vae_skeleton_status()
    js = compact_fc_vae_skeleton_json(status).lower()
    forbidden = ["scientifically proven", "outperforms", "state of the art", "best model",
                 "final comparison", "conclusive evidence"]
    for f in forbidden:
        assert f not in js


# E. Exports/scope
def test_p27_37_exports():
    exports = dir(src.phase2)
    assert "FC_VAE_MODEL_SKELETON_CONTRACT_VERSION" in exports
    assert "FCVAEInputShapeContract" in exports
    assert "FCVAELatentLayout" in exports
    assert "FCVAEDecoderOutputContract" in exports
    assert "FCVAEForwardContract" in exports
    assert "FCVAESkeletonStatus" in exports
    assert "build_fc_vae_skeleton_status" in exports
    assert "compact_fc_vae_skeleton_json" in exports


def test_p27_38_no_top_level_torch_import():
    p = pathlib.Path("src/phase2/fc_vae_model.py").read_text(encoding="utf-8")
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


def test_p27_39_no_forbidden_imports():
    p = pathlib.Path("src/phase2/fc_vae_model.py").read_text(encoding="utf-8")
    forbidden = ["numpy", "pandas", "scipy", "sklearn", "yaml", "argparse", "subprocess"]
    for f in forbidden:
        assert f"import {f}" not in p
        assert f"from {f}" not in p


def test_p27_40_no_torch_nn_module():
    p = pathlib.Path("src/phase2/fc_vae_model.py").read_text(encoding="utf-8")
    assert "torch.nn.Module" not in p


def test_p27_41_no_forbidden_class_names():
    import re
    p = pathlib.Path("src/phase2/fc_vae_model.py").read_text(encoding="utf-8")
    forbidden_classes = [
        r"class\s+Encoder\b", r"class\s+Decoder\b", r"class\s+FCVAE\s*[:(]",
        r"class\s+FC_VAE\b", r"class\s+Optimizer\b", r"class\s+Checkpoint\b",
        r"class\s+Trainer\b",
    ]
    for pattern in forbidden_classes:
        assert not re.search(pattern, p), f"Forbidden class definition found matching: {pattern}"


def test_p27_42_no_training_functions():
    p = pathlib.Path("src/phase2/fc_vae_model.py").read_text(encoding="utf-8")
    forbidden_fns = [
        "def train(", "def fit(", "def train_step(", "def training_loop(",
        "def save_checkpoint(", "def load_checkpoint(",
    ]
    for fn in forbidden_fns:
        assert fn not in p, f"Forbidden function found: {fn}"


def test_p27_43_init_no_subprocess_monkeypatch():
    p = pathlib.Path("src/phase2/__init__.py").read_text(encoding="utf-8")
    assert "subprocess.run =" not in p
    assert "_patched_run" not in p
    assert "_original_run" not in p


def test_p27_44_scope_gate():
    """Phase-local scope gate for P27. Skips on non-P27 branches."""
    allowed = {
        "src/phase2/fc_vae_model.py",
        "src/phase2/__init__.py",
        "tests/test_phase2_fc_vae_model_skeleton.py",
        "tools/phase2/run_p27_fc_vae_skeleton_smoke.py",
        "tests/test_phase2_p27_fc_vae_skeleton_smoke.py",
        "reports/PHASE_2_P27_FC_VAE_MODULE_SKELETON_AND_SHAPE_CONTRACT_REPORT.md",
    }
    enforce_phase_local_scope_gate_or_skip(
        expected_branch="phase2/p27-fc-vae-module-skeleton-shape-contract",
        base_commit="phase2/p26-torch-gated-model-dependency-boundary",
        allowed_files=allowed,
        phase_label="P27",
    )
