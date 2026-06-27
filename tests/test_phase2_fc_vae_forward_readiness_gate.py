# tests/test_phase2_fc_vae_forward_readiness_gate.py

import dataclasses
import json
import pytest
import subprocess
from typing import Any, Tuple

from src.phase2.torch_boundary import TorchDependencyStatus
from src.phase2.fc_vae_model import FC_VAE_REQUIRED_LATENT_NAMES
from src.phase2.fc_vae_forward_readiness_gate import (
    FC_VAE_FORWARD_READINESS_GATE_CONTRACT_VERSION,
    FC_VAE_FORWARD_READINESS_GATE_KIND,
    FC_VAE_FORWARD_READINESS_GATE_MODULE_NAME,
    FC_VAE_FORWARD_READINESS_STATUS_TORCH_UNAVAILABLE,
    FC_VAE_FORWARD_READINESS_STATUS_TENSOR_UNAVAILABLE,
    FC_VAE_FORWARD_READINESS_STATUS_MODULE_SHELL_UNAVAILABLE,
    FC_VAE_FORWARD_READINESS_STATUS_CONTRACT_MISMATCH,
    FC_VAE_FORWARD_READINESS_STATUS_FORWARD_IMPLEMENTATION_UNAVAILABLE,
    SUPPORTED_FC_VAE_FORWARD_READINESS_STATUSES,
    FCVAEForwardReadinessRequest,
    FCVAEForwardReadinessEvidence,
    FCVAEForwardReadinessResult,
    validate_forward_readiness_status,
    validate_forward_readiness_request,
    validate_forward_readiness_evidence,
    validate_forward_readiness_result,
    build_forward_readiness_request_from_defaults,
    build_forward_readiness_evidence,
    build_forward_readiness_result,
    run_forward_readiness_gate_probe,
    forward_readiness_request_to_json_dict,
    forward_readiness_evidence_to_json_dict,
    forward_readiness_result_to_json_dict,
    compact_forward_readiness_gate_json,
    assert_no_local_path_leakage,
    assert_no_forbidden_claims,
)

# 1-10: Constants & Status checks
def test_p42_01_constants():
    assert FC_VAE_FORWARD_READINESS_GATE_CONTRACT_VERSION == "phase2_p42_forward_readiness_gate_after_module_shell_contract_v1"
    assert FC_VAE_FORWARD_READINESS_GATE_KIND == "forward_readiness_gate_after_module_shell_no_forward_no_output_no_training"
    assert FC_VAE_FORWARD_READINESS_GATE_MODULE_NAME == "src.phase2.fc_vae_forward_readiness_gate"


def test_p42_02_statuses():
    assert FC_VAE_FORWARD_READINESS_STATUS_TORCH_UNAVAILABLE == "blocked_torch_unavailable"
    assert FC_VAE_FORWARD_READINESS_STATUS_TENSOR_UNAVAILABLE == "blocked_by_tensor_materialization_unavailable"
    assert FC_VAE_FORWARD_READINESS_STATUS_MODULE_SHELL_UNAVAILABLE == "blocked_by_module_shell_unavailable"
    assert FC_VAE_FORWARD_READINESS_STATUS_CONTRACT_MISMATCH == "blocked_by_contract_mismatch"
    assert FC_VAE_FORWARD_READINESS_STATUS_FORWARD_IMPLEMENTATION_UNAVAILABLE == "blocked_by_forward_implementation_unavailable"


def test_p42_03_supported_statuses():
    assert len(SUPPORTED_FC_VAE_FORWARD_READINESS_STATUSES) == 5
    for status in SUPPORTED_FC_VAE_FORWARD_READINESS_STATUSES:
        validate_forward_readiness_status(status)


# 11-20: Request Frozen Dataclass
def test_p42_11_request_frozen():
    req = build_forward_readiness_request_from_defaults()
    with pytest.raises(dataclasses.FrozenInstanceError if hasattr(dataclasses, "FrozenInstanceError") else Exception):
        req.reason = "modified"


def test_p42_12_request_fields():
    req = build_forward_readiness_request_from_defaults()
    assert req.contract_version == FC_VAE_FORWARD_READINESS_GATE_CONTRACT_VERSION
    assert req.gate_kind == FC_VAE_FORWARD_READINESS_GATE_KIND
    assert req.expected_batch_size == 2
    assert req.expected_input_flat_dim == 32
    assert req.expected_shape_tuple == (2, 32)
    assert req.expected_tensor_dtype_name == "torch.float32"
    assert req.expected_tensor_device_type == "cpu"
    assert req.expected_latent_total_dim == 20
    assert req.expected_latent_names == ("z_mean", "z_volatility", "z_shared")
    assert not req.allow_forward_execution_in_p42
    assert not req.allow_output_generation_in_p42
    assert not req.allow_training_in_p42


# 21-30: Evidence Frozen Dataclass
def test_p42_21_evidence_frozen():
    req = build_forward_readiness_request_from_defaults()
    ev = build_forward_readiness_evidence(req)
    with pytest.raises(dataclasses.FrozenInstanceError if hasattr(dataclasses, "FrozenInstanceError") else Exception):
        ev.reason = "modified"


# 31-40: Result Frozen Dataclass
def test_p42_31_result_frozen():
    res = run_forward_readiness_gate_probe()
    with pytest.raises(dataclasses.FrozenInstanceError if hasattr(dataclasses, "FrozenInstanceError") else Exception):
        res.reason = "modified"


def test_p42_32_result_no_execution_flags():
    res = run_forward_readiness_gate_probe()
    assert not res.forward_ready_in_p42
    assert not res.forward_available_in_p42
    assert not res.forward_execution_available_in_p42
    assert not res.forward_executed_in_p42
    assert not res.output_generation_available_in_p42
    assert not res.output_generated_in_p42
    assert not res.training_available_in_p42
    assert not res.training_executed_in_p42
    assert res.no_forward_execution
    assert res.no_output_generation
    assert res.no_training_loop
    assert res.no_optimizer
    assert res.no_checkpointing
    assert res.no_artifact_generation
    assert res.no_final_comparison
    assert res.no_scientific_conclusion


# 41-70: Validators Type and Value Checks
def test_p42_41_validate_request_type():
    with pytest.raises(TypeError):
        validate_forward_readiness_request("not_a_request")


def test_p42_42_validate_evidence_type():
    with pytest.raises(TypeError):
        validate_forward_readiness_evidence("not_an_evidence")


def test_p42_43_validate_result_type():
    with pytest.raises(TypeError):
        validate_forward_readiness_result("not_a_result")


def test_p42_44_validate_request_mismatch():
    req = build_forward_readiness_request_from_defaults()
    bad_req = dataclasses.replace(req, contract_version="wrong")
    with pytest.raises(ValueError):
        validate_forward_readiness_request(bad_req)


def test_p42_45_validate_request_bad_gate_kind():
    req = build_forward_readiness_request_from_defaults()
    bad_req = dataclasses.replace(req, gate_kind="wrong")
    with pytest.raises(ValueError):
        validate_forward_readiness_request(bad_req)


def test_p42_46_validate_request_bad_arch_id():
    req = build_forward_readiness_request_from_defaults()
    bad_req = dataclasses.replace(req, architecture_id="wrong")
    with pytest.raises(ValueError):
        validate_forward_readiness_request(bad_req)


def test_p42_47_validate_request_bad_batch_size():
    req = build_forward_readiness_request_from_defaults()
    bad_req = dataclasses.replace(req, expected_batch_size=3)
    with pytest.raises(ValueError):
        validate_forward_readiness_request(bad_req)


def test_p42_48_validate_request_bad_flat_dim():
    req = build_forward_readiness_request_from_defaults()
    bad_req = dataclasses.replace(req, expected_input_flat_dim=12)
    with pytest.raises(ValueError):
        validate_forward_readiness_request(bad_req)


def test_p42_49_validate_request_bad_shape():
    req = build_forward_readiness_request_from_defaults()
    bad_req = dataclasses.replace(req, expected_shape_tuple=(2, 3))
    with pytest.raises(ValueError):
        validate_forward_readiness_request(bad_req)


def test_p42_50_validate_request_bad_dtype():
    req = build_forward_readiness_request_from_defaults()
    bad_req = dataclasses.replace(req, expected_tensor_dtype_name="torch.float64")
    with pytest.raises(ValueError):
        validate_forward_readiness_request(bad_req)


def test_p42_51_validate_request_bad_device():
    req = build_forward_readiness_request_from_defaults()
    bad_req = dataclasses.replace(req, expected_tensor_device_type="cuda")
    with pytest.raises(ValueError):
        validate_forward_readiness_request(bad_req)


def test_p42_52_validate_request_bad_latent_dim():
    req = build_forward_readiness_request_from_defaults()
    bad_req = dataclasses.replace(req, expected_latent_total_dim=10)
    with pytest.raises(ValueError):
        validate_forward_readiness_request(bad_req)


def test_p42_53_validate_request_bad_latent_names():
    req = build_forward_readiness_request_from_defaults()
    bad_req = dataclasses.replace(req, expected_latent_names=("wrong",))
    with pytest.raises(ValueError):
        validate_forward_readiness_request(bad_req)


def test_p42_54_validate_request_allow_forward():
    req = build_forward_readiness_request_from_defaults()
    bad_req = dataclasses.replace(req, allow_forward_execution_in_p42=True)
    with pytest.raises(ValueError):
        validate_forward_readiness_request(bad_req)


def test_p42_55_validate_request_allow_output():
    req = build_forward_readiness_request_from_defaults()
    bad_req = dataclasses.replace(req, allow_output_generation_in_p42=True)
    with pytest.raises(ValueError):
        validate_forward_readiness_request(bad_req)


def test_p42_56_validate_request_allow_training():
    req = build_forward_readiness_request_from_defaults()
    bad_req = dataclasses.replace(req, allow_training_in_p42=True)
    with pytest.raises(ValueError):
        validate_forward_readiness_request(bad_req)


def test_p42_57_validate_request_empty_reason():
    req = build_forward_readiness_request_from_defaults()
    bad_req = dataclasses.replace(req, reason="")
    with pytest.raises(ValueError):
        validate_forward_readiness_request(bad_req)


def test_p42_58_validate_evidence_param_count():
    req = build_forward_readiness_request_from_defaults()
    ev = build_forward_readiness_evidence(req)
    bad_ev = dataclasses.replace(ev, parameter_count=1)
    with pytest.raises(ValueError):
        validate_forward_readiness_evidence(bad_ev)


def test_p42_59_validate_evidence_buffer_count():
    req = build_forward_readiness_request_from_defaults()
    ev = build_forward_readiness_evidence(req)
    bad_ev = dataclasses.replace(ev, buffer_count=1)
    with pytest.raises(ValueError):
        validate_forward_readiness_evidence(bad_ev)


def test_p42_60_validate_evidence_module_object_returned():
    req = build_forward_readiness_request_from_defaults()
    ev = build_forward_readiness_evidence(req)
    bad_ev = dataclasses.replace(ev, module_object_returned=True)
    with pytest.raises(ValueError):
        validate_forward_readiness_evidence(bad_ev)


def test_p42_61_validate_evidence_forward_implementation_available():
    req = build_forward_readiness_request_from_defaults()
    ev = build_forward_readiness_evidence(req)
    bad_ev = dataclasses.replace(ev, forward_implementation_available=True)
    with pytest.raises(ValueError):
        validate_forward_readiness_evidence(bad_ev)


def test_p42_62_validate_result_forward_ready():
    res = run_forward_readiness_gate_probe()
    bad_res = dataclasses.replace(res, forward_ready_in_p42=True)
    with pytest.raises(ValueError):
        validate_forward_readiness_result(bad_res)


def test_p42_63_validate_result_no_forward_execution():
    res = run_forward_readiness_gate_probe()
    bad_res = dataclasses.replace(res, no_forward_execution=False)
    with pytest.raises(ValueError):
        validate_forward_readiness_result(bad_res)


# 71-90: Builders & Happy Blocked Path
def test_p42_71_happy_blocked_path_evidence():
    req = build_forward_readiness_request_from_defaults()
    ev = build_forward_readiness_evidence(req)
    assert ev.torch_available
    assert ev.tensor_materialized_in_p39
    assert ev.module_shell_created_in_p41
    assert ev.implementation_available_in_p41
    assert ev.module_shell_available_in_p41
    assert ev.is_torch_nn_module
    assert not ev.module_object_returned
    assert not ev.defines_forward
    assert not ev.has_own_forward
    assert ev.uses_inherited_unimplemented_forward_only
    assert not ev.defines_layers
    assert ev.parameter_count == 0
    assert ev.buffer_count == 0
    assert ev.tensor_module_shape_compatible
    assert ev.module_contract_compatible
    assert not ev.forward_implementation_available


def test_p42_72_happy_blocked_path_result():
    res = run_forward_readiness_gate_probe()
    assert res.status == FC_VAE_FORWARD_READINESS_STATUS_FORWARD_IMPLEMENTATION_UNAVAILABLE
    assert not res.forward_ready_in_p42
    assert res.module_shell_ready_in_p42
    assert res.implementation_available_in_p41
    assert not res.forward_implementation_available_in_p42


# 91-105: Serializers & Compact JSON
def test_p42_91_request_serializer():
    req = build_forward_readiness_request_from_defaults()
    d = forward_readiness_request_to_json_dict(req)
    assert d["contract_version"] == FC_VAE_FORWARD_READINESS_GATE_CONTRACT_VERSION
    assert d["expected_latent_names"] == list(FC_VAE_REQUIRED_LATENT_NAMES)


def test_p42_92_evidence_serializer():
    req = build_forward_readiness_request_from_defaults()
    ev = build_forward_readiness_evidence(req)
    d = forward_readiness_evidence_to_json_dict(ev)
    assert d["is_torch_nn_module"] is True
    assert "tensor_values" not in d
    assert "torch_object" not in d


def test_p42_93_result_serializer():
    res = run_forward_readiness_gate_probe()
    d = forward_readiness_result_to_json_dict(res)
    assert d["status"] == FC_VAE_FORWARD_READINESS_STATUS_FORWARD_IMPLEMENTATION_UNAVAILABLE
    assert "module_object" not in d


def test_p42_94_compact_json_is_sorted():
    res = run_forward_readiness_gate_probe()
    json_str = compact_forward_readiness_gate_json(res)
    d = json.loads(json_str)
    assert d["contract_version"] == FC_VAE_FORWARD_READINESS_GATE_CONTRACT_VERSION
    # Check that sorting is deterministic
    json_str2 = compact_forward_readiness_gate_json(res)
    assert json_str == json_str2


# 106-115: Local path and claim leakages
def test_p42_106_path_leakage_exception():
    with pytest.raises(ValueError):
        assert_no_local_path_leakage("C:\\Users\\wekab\\project")
    with pytest.raises(ValueError):
        assert_no_local_path_leakage("/home/user/project")


def test_p42_107_claim_leakage_exception():
    with pytest.raises(ValueError):
        assert_no_forbidden_claims("state of the art results")
    with pytest.raises(ValueError):
        assert_no_forbidden_claims("we solved time series prediction")


# 116-130: Static Scope Check on the source module
def test_p42_116_static_checks():
    import re
    filepath = "src/phase2/fc_vae_forward_readiness_gate.py"
    with open(filepath, "r", encoding="utf-8") as f:
        code = f.read()

    # Verify no top-level import torch or from torch
    lines = code.splitlines()
    for line in lines:
        stripped = line.strip()
        if stripped.startswith("import torch") or stripped.startswith("from torch"):
            assert False, f"Forbidden top-level torch import found: {stripped}"

    # Verify no def forward
    assert "def forward(" not in code
    # Verify no .forward
    assert ".forward(" not in code
    # Verify no model(
    assert "model(" not in code
    # Verify no __call__(
    assert "__call__(" not in code

    # Standalone forbidden words check
    forbidden_words = [
        "Linear", "Sequential", "Parameter", "register_buffer",
        "optimizer", "loss", "checkpoint", "train", "eval", "no_grad"
    ]
    for word in forbidden_words:
        pattern = r"\b" + re.escape(word) + r"\b"
        assert not re.search(pattern, code), f"Forbidden standalone word '{word}' found in {filepath}"


# 131-140: Monkeypatch Gated/Blocked statuses
def test_p42_131_mock_torch_unavailable(monkeypatch):
    # Mock build_torch_dependency_status
    class FakeTorchStatus:
        available = False
        import_safe = False
        top_level_import_required = False
        torch_version = ""
        cuda_available = False

    monkeypatch.setattr(
        "src.phase2.fc_vae_forward_readiness_gate.build_torch_dependency_status",
        lambda policy: FakeTorchStatus()
    )

    res = run_forward_readiness_gate_probe()
    assert res.status == FC_VAE_FORWARD_READINESS_STATUS_TORCH_UNAVAILABLE
    assert not res.evidence.torch_available


def test_p42_132_mock_tensor_unavailable(monkeypatch):
    class FakeP39Result:
        status = "blocked_by_tensor_materialization_unavailable"
        tensor_materialized_in_p39 = False
        materialized_tensor = None
        torch_available = True

    monkeypatch.setattr(
        "src.phase2.fc_vae_forward_readiness_gate.run_tensor_materialization_probe",
        lambda: FakeP39Result()
    )

    res = run_forward_readiness_gate_probe()
    assert res.status == FC_VAE_FORWARD_READINESS_STATUS_TENSOR_UNAVAILABLE
    assert not res.evidence.tensor_materialized_in_p39


def test_p42_133_mock_module_shell_unavailable(monkeypatch):
    class FakeP41Result:
        status = "blocked_by_module_shell_unavailable"
        implementation_available_in_p41 = False
        module_shell_available_in_p41 = False
        metadata = None

    monkeypatch.setattr(
        "src.phase2.fc_vae_forward_readiness_gate.run_module_availability_shell_probe",
        lambda: FakeP41Result()
    )

    res = run_forward_readiness_gate_probe()
    assert res.status == FC_VAE_FORWARD_READINESS_STATUS_MODULE_SHELL_UNAVAILABLE
    assert not res.evidence.module_shell_created_in_p41


def test_p42_134_mock_contract_mismatch(monkeypatch):
    class FakeP41Metadata:
        module_shell_created = True
        is_torch_nn_module = True
        module_object_returned = False
        defines_forward = False
        has_own_forward = False
        uses_inherited_unimplemented_forward_only = True
        defines_layers = False
        parameter_count = 0
        buffer_count = 0
        architecture_id = "FC-VAE-WRONG"  # Cause contract mismatch
        input_flat_dim = 32
        latent_total_dim = 20
        latent_names = FC_VAE_REQUIRED_LATENT_NAMES

    class FakeP41Result:
        status = "module_shell_created_no_forward_no_output_no_training"
        implementation_available_in_p41 = True
        module_shell_available_in_p41 = True
        metadata = FakeP41Metadata()

    monkeypatch.setattr(
        "src.phase2.fc_vae_forward_readiness_gate.run_module_availability_shell_probe",
        lambda: FakeP41Result()
    )

    res = run_forward_readiness_gate_probe()
    assert res.status == FC_VAE_FORWARD_READINESS_STATUS_CONTRACT_MISMATCH
    assert not res.evidence.module_contract_compatible


# 141-142: Scope Gate
def test_p42_141_scope_gate():
    allowed = {
        "src/phase2/fc_vae_forward_readiness_gate.py",
        "src/phase2/__init__.py",
        "tests/test_phase2_fc_vae_forward_readiness_gate.py",
        "tools/phase2/run_p42_forward_readiness_gate_smoke.py",
        "tests/test_phase2_p42_forward_readiness_gate_smoke.py",
        "reports/PHASE_2_P42_FORWARD_READINESS_GATE_AFTER_MODULE_SHELL_NO_FORWARD_NO_OUTPUT_NO_TRAINING_REPORT.md",
    }
    # Base branch checkout: 5a6c6fcf0ec966981ecf53e1b87b712fd46b4190
    res = subprocess.run(
        ["git", "diff", "--name-only", "5a6c6fcf0ec966981ecf53e1b87b712fd46b4190"],
        capture_output=True, text=True, check=True
    )
    modified = [line.strip() for line in res.stdout.splitlines() if line.strip()]
    for f in modified:
        f_norm = f.replace("\\", "/")
        assert f_norm in allowed, f"Forbidden file modification detected in P42: {f_norm}"


import dataclasses
