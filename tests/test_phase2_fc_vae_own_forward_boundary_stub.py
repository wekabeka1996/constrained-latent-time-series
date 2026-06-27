# tests/test_phase2_fc_vae_own_forward_boundary_stub.py

import dataclasses
import json
import pytest
import subprocess
from typing import Any, Tuple

from tests.phase2_scope_gate_utils import enforce_phase_local_scope_gate_or_skip

from src.phase2.torch_boundary import TorchDependencyStatus
from src.phase2.fc_vae_model import FC_VAE_REQUIRED_LATENT_NAMES
from src.phase2.fc_vae_own_forward_boundary_stub import (
    FC_VAE_OWN_FORWARD_BOUNDARY_STUB_CONTRACT_VERSION,
    FC_VAE_OWN_FORWARD_BOUNDARY_STUB_KIND,
    FC_VAE_OWN_FORWARD_BOUNDARY_STUB_MODULE_NAME,
    FC_VAE_OWN_FORWARD_BOUNDARY_STUB_CLASS_NAME,
    FC_VAE_OWN_FORWARD_BOUNDARY_STUB_STATUS_TORCH_UNAVAILABLE,
    FC_VAE_OWN_FORWARD_BOUNDARY_STUB_STATUS_MODULE_SHELL_UNAVAILABLE,
    FC_VAE_OWN_FORWARD_BOUNDARY_STUB_STATUS_READINESS_UNAVAILABLE,
    FC_VAE_OWN_FORWARD_BOUNDARY_STUB_STATUS_DECLARED,
    FC_VAE_OWN_FORWARD_BOUNDARY_STUB_STATUS_CONTRACT_MISMATCH,
    SUPPORTED_FC_VAE_OWN_FORWARD_BOUNDARY_STUB_STATUSES,
    FCVAEOwnForwardBoundaryStubRequest,
    FCVAEOwnForwardBoundaryStubMetadata,
    FCVAEOwnForwardBoundaryStubResult,
    validate_own_forward_boundary_stub_status,
    validate_own_forward_boundary_stub_request,
    validate_own_forward_boundary_stub_metadata,
    validate_own_forward_boundary_stub_result,
    build_own_forward_boundary_stub_request_from_defaults,
    build_own_forward_boundary_stub_metadata,
    build_own_forward_boundary_stub_result,
    run_own_forward_boundary_stub_probe,
    own_forward_boundary_stub_request_to_json_dict,
    own_forward_boundary_stub_metadata_to_json_dict,
    own_forward_boundary_stub_result_to_json_dict,
    compact_own_forward_boundary_stub_json,
    assert_no_local_path_leakage,
    assert_no_forbidden_claims,
    get_p43_own_forward_boundary_stub_class,
)

# 1-10: Constants & Status checks
def test_p43_01_constants():
    assert FC_VAE_OWN_FORWARD_BOUNDARY_STUB_CONTRACT_VERSION == "phase2_p43_own_forward_boundary_stub_contract_v1"
    assert FC_VAE_OWN_FORWARD_BOUNDARY_STUB_KIND == "own_forward_boundary_stub_declared_no_execution_no_output_no_training"
    assert FC_VAE_OWN_FORWARD_BOUNDARY_STUB_MODULE_NAME == "src.phase2.fc_vae_own_forward_boundary_stub"
    assert FC_VAE_OWN_FORWARD_BOUNDARY_STUB_CLASS_NAME == "P43FCVAEOwnForwardBoundaryStub"


def test_p43_02_statuses():
    assert FC_VAE_OWN_FORWARD_BOUNDARY_STUB_STATUS_TORCH_UNAVAILABLE == "blocked_torch_unavailable"
    assert FC_VAE_OWN_FORWARD_BOUNDARY_STUB_STATUS_MODULE_SHELL_UNAVAILABLE == "blocked_by_module_shell_unavailable"
    assert FC_VAE_OWN_FORWARD_BOUNDARY_STUB_STATUS_READINESS_UNAVAILABLE == "blocked_by_p42_readiness_unavailable"
    assert FC_VAE_OWN_FORWARD_BOUNDARY_STUB_STATUS_DECLARED == "forward_boundary_declared_no_execution_no_output_no_training"
    assert FC_VAE_OWN_FORWARD_BOUNDARY_STUB_STATUS_CONTRACT_MISMATCH == "blocked_by_contract_mismatch"


def test_p43_03_supported_statuses():
    assert len(SUPPORTED_FC_VAE_OWN_FORWARD_BOUNDARY_STUB_STATUSES) == 5
    for status in SUPPORTED_FC_VAE_OWN_FORWARD_BOUNDARY_STUB_STATUSES:
        validate_own_forward_boundary_stub_status(status)


# 11-20: Request Frozen Dataclass
def test_p43_11_request_frozen():
    req = build_own_forward_boundary_stub_request_from_defaults()
    with pytest.raises(dataclasses.FrozenInstanceError if hasattr(dataclasses, "FrozenInstanceError") else Exception):
        req.reason = "modified"


def test_p43_12_request_fields():
    req = build_own_forward_boundary_stub_request_from_defaults()
    assert req.contract_version == FC_VAE_OWN_FORWARD_BOUNDARY_STUB_CONTRACT_VERSION
    assert req.boundary_kind == FC_VAE_OWN_FORWARD_BOUNDARY_STUB_KIND
    assert req.expected_input_flat_dim == 32
    assert req.expected_latent_total_dim == 20
    assert req.expected_latent_names == ("z_mean", "z_volatility", "z_shared")
    assert req.allow_forward_declaration_in_p43
    assert not req.allow_forward_execution_in_p43
    assert not req.allow_output_generation_in_p43
    assert not req.allow_training_in_p43


# 21-30: Evidence Frozen Dataclass
def test_p43_21_metadata_frozen():
    req = build_own_forward_boundary_stub_request_from_defaults()
    meta = build_own_forward_boundary_stub_metadata(req)
    with pytest.raises(dataclasses.FrozenInstanceError if hasattr(dataclasses, "FrozenInstanceError") else Exception):
        meta.reason = "modified"


# 31-40: Result Frozen Dataclass
def test_p43_31_result_frozen():
    res = run_own_forward_boundary_stub_probe()
    with pytest.raises(dataclasses.FrozenInstanceError if hasattr(dataclasses, "FrozenInstanceError") else Exception):
        res.reason = "modified"


def test_p43_32_result_no_execution_flags():
    res = run_own_forward_boundary_stub_probe()
    assert res.own_forward_boundary_declared_in_p43
    assert not res.forward_execution_available_in_p43
    assert not res.forward_executed_in_p43
    assert not res.output_generation_available_in_p43
    assert not res.output_generated_in_p43
    assert not res.training_available_in_p43
    assert not res.training_executed_in_p43
    assert res.no_forward_execution
    assert res.no_output_generation
    assert res.no_training_loop
    assert res.no_optimizer
    assert res.no_checkpointing
    assert res.no_artifact_generation
    assert res.no_final_comparison
    assert res.no_scientific_conclusion


# 41-70: Validators Type and Value Checks
def test_p43_41_validate_request_type():
    with pytest.raises(TypeError):
        validate_own_forward_boundary_stub_request("not_a_request")


def test_p43_42_validate_metadata_type():
    with pytest.raises(TypeError):
        validate_own_forward_boundary_stub_metadata("not_a_metadata")


def test_p43_43_validate_result_type():
    with pytest.raises(TypeError):
        validate_own_forward_boundary_stub_result("not_a_result")


def test_p43_44_validate_request_mismatch():
    req = build_own_forward_boundary_stub_request_from_defaults()
    bad_req = dataclasses.replace(req, contract_version="wrong")
    with pytest.raises(ValueError):
        validate_own_forward_boundary_stub_request(bad_req)


def test_p43_45_validate_request_bad_kind():
    req = build_own_forward_boundary_stub_request_from_defaults()
    bad_req = dataclasses.replace(req, boundary_kind="wrong")
    with pytest.raises(ValueError):
        validate_own_forward_boundary_stub_request(bad_req)


def test_p43_46_validate_request_bad_arch():
    req = build_own_forward_boundary_stub_request_from_defaults()
    bad_req = dataclasses.replace(req, architecture_id="wrong")
    with pytest.raises(ValueError):
        validate_own_forward_boundary_stub_request(bad_req)


def test_p43_47_validate_request_bad_flat_dim():
    req = build_own_forward_boundary_stub_request_from_defaults()
    bad_req = dataclasses.replace(req, expected_input_flat_dim=12)
    with pytest.raises(ValueError):
        validate_own_forward_boundary_stub_request(bad_req)


def test_p43_48_validate_request_bad_latent_dim():
    req = build_own_forward_boundary_stub_request_from_defaults()
    bad_req = dataclasses.replace(req, expected_latent_total_dim=10)
    with pytest.raises(ValueError):
        validate_own_forward_boundary_stub_request(bad_req)


def test_p43_49_validate_request_bad_latent_names():
    req = build_own_forward_boundary_stub_request_from_defaults()
    bad_req = dataclasses.replace(req, expected_latent_names=("wrong",))
    with pytest.raises(ValueError):
        validate_own_forward_boundary_stub_request(bad_req)


def test_p43_50_validate_request_allow_forward_declaration():
    req = build_own_forward_boundary_stub_request_from_defaults()
    bad_req = dataclasses.replace(req, allow_forward_declaration_in_p43=False)
    with pytest.raises(ValueError):
        validate_own_forward_boundary_stub_request(bad_req)


def test_p43_51_validate_request_allow_forward_execution():
    req = build_own_forward_boundary_stub_request_from_defaults()
    bad_req = dataclasses.replace(req, allow_forward_execution_in_p43=True)
    with pytest.raises(ValueError):
        validate_own_forward_boundary_stub_request(bad_req)


def test_p43_52_validate_request_allow_output():
    req = build_own_forward_boundary_stub_request_from_defaults()
    bad_req = dataclasses.replace(req, allow_output_generation_in_p43=True)
    with pytest.raises(ValueError):
        validate_own_forward_boundary_stub_request(bad_req)


def test_p43_53_validate_request_allow_training():
    req = build_own_forward_boundary_stub_request_from_defaults()
    bad_req = dataclasses.replace(req, allow_training_in_p43=True)
    with pytest.raises(ValueError):
        validate_own_forward_boundary_stub_request(bad_req)


def test_p43_54_validate_request_empty_reason():
    req = build_own_forward_boundary_stub_request_from_defaults()
    bad_req = dataclasses.replace(req, reason="")
    with pytest.raises(ValueError):
        validate_own_forward_boundary_stub_request(bad_req)


def test_p43_55_validate_metadata_param_count():
    req = build_own_forward_boundary_stub_request_from_defaults()
    meta = build_own_forward_boundary_stub_metadata(req)
    bad_meta = dataclasses.replace(meta, parameter_count=1)
    with pytest.raises(ValueError):
        validate_own_forward_boundary_stub_metadata(bad_meta)


def test_p43_56_validate_metadata_buffer_count():
    req = build_own_forward_boundary_stub_request_from_defaults()
    meta = build_own_forward_boundary_stub_metadata(req)
    bad_meta = dataclasses.replace(meta, buffer_count=1)
    with pytest.raises(ValueError):
        validate_own_forward_boundary_stub_metadata(bad_meta)


def test_p43_57_validate_metadata_module_object_returned():
    req = build_own_forward_boundary_stub_request_from_defaults()
    meta = build_own_forward_boundary_stub_metadata(req)
    bad_meta = dataclasses.replace(meta, module_object_returned=True)
    with pytest.raises(ValueError):
        validate_own_forward_boundary_stub_metadata(bad_meta)


def test_p43_58_validate_metadata_p42_forward_ready():
    req = build_own_forward_boundary_stub_request_from_defaults()
    meta = build_own_forward_boundary_stub_metadata(req)
    bad_meta = dataclasses.replace(meta, p42_forward_ready=True)
    with pytest.raises(ValueError):
        validate_own_forward_boundary_stub_metadata(bad_meta)


def test_p43_59_validate_result_no_forward_execution():
    res = run_own_forward_boundary_stub_probe()
    bad_res = dataclasses.replace(res, no_forward_execution=False)
    with pytest.raises(ValueError):
        validate_own_forward_boundary_stub_result(bad_res)


# 71-90: Builders & Happy Blocked Path
def test_p43_71_happy_blocked_path_metadata():
    req = build_own_forward_boundary_stub_request_from_defaults()
    meta = build_own_forward_boundary_stub_metadata(req)
    assert meta.torch_available
    assert meta.class_declared
    assert meta.module_instance_created
    assert not meta.module_object_returned
    assert meta.is_torch_nn_module
    assert meta.own_forward_declared
    assert meta.forward_signature_available
    assert "args" in meta.forward_signature_parameters
    assert "kwargs" in meta.forward_signature_parameters
    assert not meta.forward_execution_attempted
    assert not meta.output_generation_attempted
    assert not meta.training_attempted
    assert not meta.defines_layers
    assert meta.parameter_count == 0
    assert meta.buffer_count == 0
    assert meta.architecture_id == "FC-VAE"
    assert meta.input_flat_dim == 32
    assert meta.latent_total_dim == 20
    assert meta.latent_names == FC_VAE_REQUIRED_LATENT_NAMES
    assert meta.p41_module_shell_available
    assert not meta.p42_forward_ready
    assert meta.p42_forward_blocked_as_expected


def test_p43_72_happy_blocked_path_result():
    res = run_own_forward_boundary_stub_probe()
    assert res.status == FC_VAE_OWN_FORWARD_BOUNDARY_STUB_STATUS_DECLARED
    assert res.own_forward_boundary_declared_in_p43
    assert not res.forward_execution_available_in_p43
    assert not res.forward_executed_in_p43


# 91-105: Serializers & Compact JSON
def test_p43_91_request_serializer():
    req = build_own_forward_boundary_stub_request_from_defaults()
    d = own_forward_boundary_stub_request_to_json_dict(req)
    assert d["contract_version"] == FC_VAE_OWN_FORWARD_BOUNDARY_STUB_CONTRACT_VERSION
    assert d["expected_latent_names"] == list(FC_VAE_REQUIRED_LATENT_NAMES)


def test_p43_92_metadata_serializer():
    req = build_own_forward_boundary_stub_request_from_defaults()
    meta = build_own_forward_boundary_stub_metadata(req)
    d = own_forward_boundary_stub_metadata_to_json_dict(meta)
    assert d["class_declared"] is True
    assert "torch_object" not in d
    assert "module_object" not in d


def test_p43_93_result_serializer():
    res = run_own_forward_boundary_stub_probe()
    d = own_forward_boundary_stub_result_to_json_dict(res)
    assert d["status"] == FC_VAE_OWN_FORWARD_BOUNDARY_STUB_STATUS_DECLARED
    assert "module_object" not in d


def test_p43_94_compact_json_is_sorted():
    res = run_own_forward_boundary_stub_probe()
    json_str = compact_own_forward_boundary_stub_json(res)
    d = json.loads(json_str)
    assert d["contract_version"] == FC_VAE_OWN_FORWARD_BOUNDARY_STUB_CONTRACT_VERSION
    json_str2 = compact_own_forward_boundary_stub_json(res)
    assert json_str == json_str2


# 106-115: Local path and claim leakages
def test_p43_106_path_leakage_exception():
    with pytest.raises(ValueError):
        assert_no_local_path_leakage("C:\\Users\\wekab\\project")
    with pytest.raises(ValueError):
        assert_no_local_path_leakage("/home/user/project")


def test_p43_107_claim_leakage_exception():
    with pytest.raises(ValueError):
        assert_no_forbidden_claims("state of the art results")
    with pytest.raises(ValueError):
        assert_no_forbidden_claims("we solved time series prediction")


# 116-130: Static Scope Check on the source module
def test_p43_116_static_checks():
    import re
    filepath = "src/phase2/fc_vae_own_forward_boundary_stub.py"
    with open(filepath, "r", encoding="utf-8") as f:
        code = f.read()

    # Verify no top-level import torch or from torch
    lines = code.splitlines()
    for line in lines:
        if line.startswith("import torch") or line.startswith("from torch"):
            assert False, f"Forbidden top-level torch import found: {line}"

    # Verify exactly ONE def forward
    forward_declarations = code.count("def forward(")
    assert forward_declarations == 1, f"Expected exactly one 'def forward(' declaration, found {forward_declarations}"

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
def test_p43_131_mock_torch_unavailable(monkeypatch):
    class FakeTorchStatus:
        available = False
        import_safe = False
        top_level_import_required = False
        torch_version = ""
        cuda_available = False

    monkeypatch.setattr(
        "src.phase2.fc_vae_own_forward_boundary_stub.build_torch_dependency_status",
        lambda policy: FakeTorchStatus()
    )

    res = run_own_forward_boundary_stub_probe()
    assert res.status == FC_VAE_OWN_FORWARD_BOUNDARY_STUB_STATUS_TORCH_UNAVAILABLE
    assert not res.metadata.torch_available


def test_p43_132_mock_p41_shell_unavailable(monkeypatch):
    class FakeP41Result:
        status = "blocked_by_module_shell_unavailable"
        implementation_available_in_p41 = False
        module_shell_available_in_p41 = False
        metadata = None

    monkeypatch.setattr(
        "src.phase2.fc_vae_own_forward_boundary_stub.run_module_availability_shell_probe",
        lambda: FakeP41Result()
    )

    res = run_own_forward_boundary_stub_probe()
    assert res.status == FC_VAE_OWN_FORWARD_BOUNDARY_STUB_STATUS_MODULE_SHELL_UNAVAILABLE
    assert not res.metadata.p41_module_shell_available


def test_p43_133_mock_p42_readiness_unavailable(monkeypatch):
    class FakeP42Result:
        status = "blocked_by_contract_mismatch"  # Cause p42 readiness failure
        forward_ready_in_p42 = False
        evidence = None

    monkeypatch.setattr(
        "src.phase2.fc_vae_own_forward_boundary_stub.run_forward_readiness_gate_probe",
        lambda: FakeP42Result()
    )

    res = run_own_forward_boundary_stub_probe()
    assert res.status == FC_VAE_OWN_FORWARD_BOUNDARY_STUB_STATUS_READINESS_UNAVAILABLE
    assert not res.metadata.p42_forward_blocked_as_expected


# 141-142: Scope Gate
def test_p43_141_scope_gate():
    allowed = {
        "src/phase2/fc_vae_own_forward_boundary_stub.py",
        "src/phase2/__init__.py",
        "tests/test_phase2_fc_vae_own_forward_boundary_stub.py",
        "tools/phase2/run_p43_own_forward_boundary_stub_smoke.py",
        "tests/test_phase2_p43_own_forward_boundary_stub_smoke.py",
        "reports/PHASE_2_P43_OWN_FORWARD_BOUNDARY_STUB_DECLARED_NO_FORWARD_EXECUTION_NO_OUTPUT_NO_TRAINING_REPORT.md",
    }
    # Phase-local scope gate: only enforces on the P43 branch; skips on later cumulative branches.
    enforce_phase_local_scope_gate_or_skip(
        expected_branch="phase2/p43-own-forward-boundary-stub-declared-no-forward-execution-no-output-no-training",
        base_commit="edc7fee6a2f66b5264cc05d9ca26f150e37beeb0",
        allowed_files=allowed,
        phase_label="P43",
    )
