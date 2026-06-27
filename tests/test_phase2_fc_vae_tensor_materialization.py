# tests/test_phase2_fc_vae_tensor_materialization.py

import dataclasses
import json
import pathlib
import subprocess
import pytest
from typing import Any, Tuple

from src.phase2.fc_vae_tensor_materialization import (
    FC_VAE_TENSOR_MATERIALIZATION_CONTRACT_VERSION,
    FC_VAE_TENSOR_MATERIALIZATION_KIND,
    FC_VAE_TENSOR_MATERIALIZATION_MODULE_NAME,
    FC_VAE_TENSOR_MATERIALIZATION_STATUS_TORCH_UNAVAILABLE,
    FC_VAE_TENSOR_MATERIALIZATION_STATUS_REQUEST_BLOCKED,
    FC_VAE_TENSOR_MATERIALIZATION_STATUS_MATERIALIZED,
    SUPPORTED_FC_VAE_TENSOR_MATERIALIZATION_STATUSES,
    FC_VAE_TENSOR_MATERIALIZATION_TARGET_FRAMEWORK,
    FC_VAE_TENSOR_MATERIALIZATION_TARGET_DTYPE,
    FC_VAE_TENSOR_MATERIALIZATION_TARGET_DEVICE,
    FC_VAE_TENSOR_MATERIALIZATION_LAYOUT_KIND,
    DEFAULT_P39_BATCH_SIZE,
    DEFAULT_P39_INPUT_FLAT_DIM,
    DEFAULT_P39_FULL_VECTOR_LENGTH,
    DEFAULT_P39_SHAPE_TUPLE,
    DEFAULT_P39_REQUIRES_GRAD,
    FCVAETensorMaterializationRequest as FCVAETensorMaterializationRequestP39,
    FCVAEMaterializedTorchTensor,
    FCVAETensorMaterializationMetadata as FCVAETensorMaterializationMetadataP39,
    FCVAETensorMaterializationResult,
    validate_non_empty_str,
    validate_bool,
    validate_positive_int,
    validate_shape_tuple,
    validate_full_vector_length,
    validate_materialization_status,
    assert_no_local_path_leakage,
    assert_no_forbidden_claims,
    validate_p39_request,
    validate_materialized_torch_tensor,
    validate_p39_metadata,
    validate_p39_result,
    build_tensor_materialization_request_from_p38_default,
    build_materialized_torch_tensor,
    build_tensor_materialization_metadata as build_tensor_materialization_metadata_p39,
    build_tensor_materialization_result,
    run_tensor_materialization_probe,
    p39_tensor_materialization_request_to_json_dict,
    materialized_torch_tensor_to_json_dict,
    p39_tensor_materialization_metadata_to_json_dict,
    p39_tensor_materialization_result_to_json_dict,
    compact_p39_tensor_materialization_json,
    load_torch_for_p39_materialization,
    normalize_torch_dtype_name,
)

# Mock classes for testing the materialized path when torch is not available
class MockDevice:
    def __init__(self, device_type: str):
        self.type = device_type

class MockTensor:
    def __init__(self, data: Any, dtype: str, device: str):
        self.data = data
        self.dtype = dtype
        self.device = MockDevice(device)
        self.requires_grad = False
    def numel(self) -> int:
        return 64
    def is_floating_point(self) -> bool:
        return True
    def detach(self) -> Any:
        return self
    def cpu(self) -> Any:
        return self
    def tolist(self) -> Any:
        return [list(row) for row in self.data]
    @property
    def shape(self) -> Tuple[int, int]:
        return (2, 32)
    def requires_grad_(self, val: bool) -> None:
        self.requires_grad = val

class MockCuda:
    @staticmethod
    def is_available() -> bool:
        return False

class MockTorch:
    __version__ = "2.0.0+mock"
    float32 = "torch.float32"
    cuda = MockCuda
    @staticmethod
    def tensor(data: Any, dtype: str, device: str) -> MockTensor:
        return MockTensor(data, dtype, device)


# 1-14: constants exact
def test_p39_001_contract_version_constant():
    assert FC_VAE_TENSOR_MATERIALIZATION_CONTRACT_VERSION == "phase2_p39_torch_tensor_materialization_contract_v1"

def test_p39_002_materialization_kind_constant():
    assert FC_VAE_TENSOR_MATERIALIZATION_KIND == "optional_torch_cpu_float32_tensor_materialization_no_forward_no_training"

def test_p39_003_module_name_constant():
    assert FC_VAE_TENSOR_MATERIALIZATION_MODULE_NAME == "src.phase2.fc_vae_tensor_materialization"

def test_p39_004_status_constants():
    assert FC_VAE_TENSOR_MATERIALIZATION_STATUS_TORCH_UNAVAILABLE == "blocked_torch_unavailable"
    assert FC_VAE_TENSOR_MATERIALIZATION_STATUS_REQUEST_BLOCKED == "blocked_by_tensor_request_status"
    assert FC_VAE_TENSOR_MATERIALIZATION_STATUS_MATERIALIZED == "torch_tensor_materialized_no_forward_no_training_in_p39"

def test_p39_005_supported_statuses():
    assert SUPPORTED_FC_VAE_TENSOR_MATERIALIZATION_STATUSES == (
        "blocked_torch_unavailable",
        "blocked_by_tensor_request_status",
        "torch_tensor_materialized_no_forward_no_training_in_p39",
    )

def test_p39_006_target_framework_constant():
    assert FC_VAE_TENSOR_MATERIALIZATION_TARGET_FRAMEWORK == "torch"

def test_p39_007_target_dtype_constant():
    assert FC_VAE_TENSOR_MATERIALIZATION_TARGET_DTYPE == "float32"

def test_p39_008_target_device_constant():
    assert FC_VAE_TENSOR_MATERIALIZATION_TARGET_DEVICE == "cpu"

def test_p39_009_layout_kind_constant():
    assert FC_VAE_TENSOR_MATERIALIZATION_LAYOUT_KIND == "row_major_2d_batch_tensor"

def test_p39_010_default_batch_size():
    assert DEFAULT_P39_BATCH_SIZE == 2

def test_p39_011_default_input_flat_dim():
    assert DEFAULT_P39_INPUT_FLAT_DIM == 32

def test_p39_012_default_full_vector_length():
    assert DEFAULT_P39_FULL_VECTOR_LENGTH == 64

def test_p39_013_default_shape_tuple():
    assert DEFAULT_P39_SHAPE_TUPLE == (2, 32)

def test_p39_014_default_requires_grad():
    assert DEFAULT_P39_REQUIRES_GRAD is False


# 15: dataclasses frozen
def test_p39_015_dataclasses_frozen():
    req = build_tensor_materialization_request_from_p38_default()
    with pytest.raises(Exception):
        req.reason = "modified"


# 16-21: primitive validators
def test_p39_016_validate_non_empty_str():
    validate_non_empty_str("valid", "test")
    with pytest.raises(TypeError):
        validate_non_empty_str(123, "test")
    with pytest.raises(ValueError):
        validate_non_empty_str("", "test")
    with pytest.raises(ValueError):
        validate_non_empty_str(" padded ", "test")

def test_p39_017_validate_bool():
    validate_bool(True, "test")
    validate_bool(False, "test")
    with pytest.raises(TypeError):
        validate_bool(1, "test")

def test_p39_018_validate_positive_int():
    validate_positive_int(1, "test")
    with pytest.raises(TypeError):
        validate_positive_int(1.0, "test")
    with pytest.raises(ValueError):
        validate_positive_int(0, "test")

def test_p39_019_validate_shape_tuple():
    validate_shape_tuple((2, 32))
    with pytest.raises(TypeError):
        validate_shape_tuple([2, 32])
    with pytest.raises(ValueError):
        validate_shape_tuple((2, 33))

def test_p39_020_validate_full_vector_length():
    validate_full_vector_length(64)
    with pytest.raises(ValueError):
        validate_full_vector_length(65)

def test_p39_021_validate_materialization_status():
    validate_materialization_status("blocked_torch_unavailable")
    with pytest.raises(ValueError):
        validate_materialization_status("invalid")


# 22-25: path leakage & forbidden claims
def test_p39_022_assert_no_local_path_leakage_safe():
    assert_no_local_path_leakage("safe string")

def test_p39_023_assert_no_local_path_leakage_unsafe():
    with pytest.raises(ValueError):
        assert_no_local_path_leakage("C:\\path\\leak")

def test_p39_024_assert_no_forbidden_claims_safe():
    assert_no_forbidden_claims("no_scientific_conclusion")

def test_p39_025_assert_no_forbidden_claims_unsafe():
    with pytest.raises(ValueError):
        assert_no_forbidden_claims("model works")


# 26-40: request builder properties
def test_p39_026_request_builder_validates():
    req = build_tensor_materialization_request_from_p38_default()
    validate_p39_request(req)

def test_p39_027_request_builder_target_framework():
    req = build_tensor_materialization_request_from_p38_default()
    assert req.target_framework == "torch"

def test_p39_028_request_builder_target_dtype():
    req = build_tensor_materialization_request_from_p38_default()
    assert req.target_dtype == "float32"

def test_p39_029_request_builder_target_device():
    req = build_tensor_materialization_request_from_p38_default()
    assert req.target_device == "cpu"

def test_p39_030_request_builder_target_layout():
    req = build_tensor_materialization_request_from_p38_default()
    assert req.target_layout_kind == "row_major_2d_batch_tensor"

def test_p39_031_request_builder_batch_size():
    req = build_tensor_materialization_request_from_p38_default()
    assert req.batch_size == 2

def test_p39_032_request_builder_input_flat_dim():
    req = build_tensor_materialization_request_from_p38_default()
    assert req.input_flat_dim == 32

def test_p39_033_request_builder_full_vector_length():
    req = build_tensor_materialization_request_from_p38_default()
    assert req.full_vector_length == 64

def test_p39_034_request_builder_shape_tuple():
    req = build_tensor_materialization_request_from_p38_default()
    assert req.shape_tuple == (2, 32)

def test_p39_035_request_builder_requires_grad():
    req = build_tensor_materialization_request_from_p38_default()
    assert req.requires_grad is False

def test_p39_036_request_builder_allow_tensor_materialization():
    req = build_tensor_materialization_request_from_p38_default()
    assert req.allow_tensor_materialization_in_p39 is True

def test_p39_037_request_builder_allow_forward():
    req = build_tensor_materialization_request_from_p38_default()
    assert req.allow_forward_execution_in_p39 is False

def test_p39_038_request_builder_allow_output():
    req = build_tensor_materialization_request_from_p38_default()
    assert req.allow_output_generation_in_p39 is False

def test_p39_039_request_builder_allow_training():
    req = build_tensor_materialization_request_from_p39_default = build_tensor_materialization_request_from_p38_default()
    assert req.allow_training_in_p39 is False

def test_p39_040_request_builder_reason():
    req = build_tensor_materialization_request_from_p38_default()
    assert req.reason == "p39_tensor_materialization_request_from_p38_default"


# 41-43: request builder source contract versions
def test_p39_041_request_builder_source_nested_contract():
    req = build_tensor_materialization_request_from_p38_default()
    assert req.source_nested_values_contract_version == "phase2_p37_nested_batch_values_contract_v1"

def test_p39_042_request_builder_source_tensor_request_contract():
    req = build_tensor_materialization_request_from_p38_default()
    assert req.source_tensor_request_contract_version == "phase2_p38_tensor_materialization_request_contract_v1"

def test_p39_043_request_builder_source_torch_boundary_contract():
    req = build_tensor_materialization_request_from_p38_default()
    assert req.source_torch_boundary_contract_version == "phase2_p26_torch_boundary_contract_v1"


# 44: torch loader raises if unavailable
def test_p39_044_torch_loader_raises_if_unavailable(monkeypatch):
    from src.phase2.torch_boundary import TorchDependencyStatus
    monkeypatch.setattr("src.phase2.fc_vae_tensor_materialization.build_torch_dependency_status", lambda policy: TorchDependencyStatus(
        contract_version="phase2_p26_torch_boundary_contract_v1", backend_name="torch", available=False, policy="optional", import_safe=False, top_level_import_required=False, reason="mocked"
    ))
    with pytest.raises(RuntimeError):
        load_torch_for_p39_materialization()


# 45: normalize_torch_dtype_name
def test_p39_045_normalize_torch_dtype_name():
    assert normalize_torch_dtype_name("torch.float32") == "torch.float32"
    assert normalize_torch_dtype_name("Float32") == "torch.float32"


# 46-66: validate_p39_request variations
def test_p39_046_validate_p39_request_valid():
    req = build_tensor_materialization_request_from_p38_default()
    validate_p39_request(req)

def test_p39_047_validate_p39_request_invalid_contract():
    req = build_tensor_materialization_request_from_p38_default()
    bad = dataclasses.replace(req, contract_version="invalid")
    with pytest.raises(ValueError):
        validate_p39_request(bad)

def test_p39_048_validate_p39_request_invalid_kind():
    req = build_tensor_materialization_request_from_p38_default()
    bad = dataclasses.replace(req, materialization_kind="invalid")
    with pytest.raises(ValueError):
        validate_p39_request(bad)

def test_p39_049_validate_p39_request_invalid_arch():
    req = build_tensor_materialization_request_from_p38_default()
    bad = dataclasses.replace(req, architecture_id="wrong")
    with pytest.raises(ValueError):
        validate_p39_request(bad)

def test_p39_050_validate_p39_request_invalid_source_nested():
    req = build_tensor_materialization_request_from_p38_default()
    bad = dataclasses.replace(req, source_nested_values_contract_version="wrong")
    with pytest.raises(ValueError):
        validate_p39_request(bad)

def test_p39_051_validate_p39_request_invalid_source_request():
    req = build_tensor_materialization_request_from_p38_default()
    bad = dataclasses.replace(req, source_tensor_request_contract_version="wrong")
    with pytest.raises(ValueError):
        validate_p39_request(bad)

def test_p39_052_validate_p39_request_invalid_source_torch():
    req = build_tensor_materialization_request_from_p38_default()
    bad = dataclasses.replace(req, source_torch_boundary_contract_version="wrong")
    with pytest.raises(ValueError):
        validate_p39_request(bad)

def test_p39_053_validate_p39_request_invalid_target_framework():
    req = build_tensor_materialization_request_from_p38_default()
    bad = dataclasses.replace(req, target_framework="tensorflow")
    with pytest.raises(ValueError):
        validate_p39_request(bad)

def test_p39_054_validate_p39_request_invalid_target_dtype():
    req = build_tensor_materialization_request_from_p38_default()
    bad = dataclasses.replace(req, target_dtype="int32")
    with pytest.raises(ValueError):
        validate_p39_request(bad)

def test_p39_055_validate_p39_request_invalid_target_device():
    req = build_tensor_materialization_request_from_p38_default()
    bad = dataclasses.replace(req, target_device="cuda")
    with pytest.raises(ValueError):
        validate_p39_request(bad)

def test_p39_056_validate_p39_request_invalid_target_layout():
    req = build_tensor_materialization_request_from_p38_default()
    bad = dataclasses.replace(req, target_layout_kind="strided")
    with pytest.raises(ValueError):
        validate_p39_request(bad)

def test_p39_057_validate_p39_request_invalid_batch_size():
    req = build_tensor_materialization_request_from_p38_default()
    bad = dataclasses.replace(req, batch_size=3)
    with pytest.raises(ValueError):
        validate_p39_request(bad)

def test_p39_058_validate_p39_request_invalid_input_flat_dim():
    req = build_tensor_materialization_request_from_p38_default()
    bad = dataclasses.replace(req, input_flat_dim=16)
    with pytest.raises(ValueError):
        validate_p39_request(bad)

def test_p39_059_validate_p39_request_invalid_full_vector_length():
    req = build_tensor_materialization_request_from_p38_default()
    bad = dataclasses.replace(req, full_vector_length=32)
    with pytest.raises(ValueError):
        validate_p39_request(bad)

def test_p39_060_validate_p39_request_invalid_shape_tuple():
    req = build_tensor_materialization_request_from_p38_default()
    bad = dataclasses.replace(req, shape_tuple=(3, 32))
    with pytest.raises(ValueError):
        validate_p39_request(bad)

def test_p39_061_validate_p39_request_invalid_requires_grad():
    req = build_tensor_materialization_request_from_p38_default()
    bad = dataclasses.replace(req, requires_grad=True)
    with pytest.raises(ValueError):
        validate_p39_request(bad)

def test_p39_062_validate_p39_request_invalid_allow_tensor():
    req = build_tensor_materialization_request_from_p38_default()
    bad = dataclasses.replace(req, allow_tensor_materialization_in_p39=False)
    with pytest.raises(ValueError):
        validate_p39_request(bad)

def test_p39_063_validate_p39_request_invalid_allow_forward():
    req = build_tensor_materialization_request_from_p38_default()
    bad = dataclasses.replace(req, allow_forward_execution_in_p39=True)
    with pytest.raises(ValueError):
        validate_p39_request(bad)

def test_p39_064_validate_p39_request_invalid_allow_output():
    req = build_tensor_materialization_request_from_p38_default()
    bad = dataclasses.replace(req, allow_output_generation_in_p39=True)
    with pytest.raises(ValueError):
        validate_p39_request(bad)

def test_p39_065_validate_p39_request_invalid_allow_training():
    req = build_tensor_materialization_request_from_p38_default()
    bad = dataclasses.replace(req, allow_training_in_p39=True)
    with pytest.raises(ValueError):
        validate_p39_request(bad)

def test_p39_066_validate_p39_request_invalid_reason():
    req = build_tensor_materialization_request_from_p38_default()
    bad = dataclasses.replace(req, reason="")
    with pytest.raises(ValueError):
        validate_p39_request(bad)


# 67-72: validate_p39_metadata checks
def test_p39_067_validate_p39_metadata_invalid_contract():
    req = build_tensor_materialization_request_from_p38_default()
    meta = build_tensor_materialization_metadata_p39(req, None)
    bad = dataclasses.replace(meta, contract_version="wrong")
    with pytest.raises(ValueError):
        validate_p39_metadata(bad)

def test_p39_068_validate_p39_metadata_invalid_nested():
    req = build_tensor_materialization_request_from_p38_default()
    meta = build_tensor_materialization_metadata_p39(req, None)
    bad = dataclasses.replace(meta, source_nested_values_contract_version="wrong")
    with pytest.raises(ValueError):
        validate_p39_metadata(bad)

def test_p39_069_validate_p39_metadata_invalid_request():
    req = build_tensor_materialization_request_from_p38_default()
    meta = build_tensor_materialization_metadata_p39(req, None)
    bad = dataclasses.replace(meta, source_tensor_request_contract_version="wrong")
    with pytest.raises(ValueError):
        validate_p39_metadata(bad)

def test_p39_070_validate_p39_metadata_invalid_torch():
    req = build_tensor_materialization_request_from_p38_default()
    meta = build_tensor_materialization_metadata_p39(req, None)
    bad = dataclasses.replace(meta, source_torch_boundary_contract_version="wrong")
    with pytest.raises(ValueError):
        validate_p39_metadata(bad)

def test_p39_071_validate_p39_metadata_invalid_flags():
    req = build_tensor_materialization_request_from_p38_default()
    meta = build_tensor_materialization_metadata_p39(req, None)
    bad = dataclasses.replace(meta, training_attempted=True)
    with pytest.raises(ValueError):
        validate_p39_metadata(bad)

def test_p39_072_validate_p39_metadata_invalid_reason():
    req = build_tensor_materialization_request_from_p38_default()
    meta = build_tensor_materialization_metadata_p39(req, None)
    bad = dataclasses.replace(meta, reason="")
    with pytest.raises(ValueError):
        validate_p39_metadata(bad)


# 73-76: validate_p39_result checks
def test_p39_073_validate_p39_result_invalid_contract():
    res = run_tensor_materialization_probe()
    bad = dataclasses.replace(res, contract_version="wrong")
    with pytest.raises(ValueError):
        validate_p39_result(bad)

def test_p39_074_validate_p39_result_invalid_status(monkeypatch):
    from src.phase2.torch_boundary import TorchDependencyStatus
    monkeypatch.setattr("src.phase2.fc_vae_tensor_materialization.build_torch_dependency_status", lambda policy: TorchDependencyStatus(
        contract_version="phase2_p26_torch_boundary_contract_v1", backend_name="torch", available=False, policy="optional", import_safe=False, top_level_import_required=False, reason="mocked"
    ))
    res = run_tensor_materialization_probe()
    bad = dataclasses.replace(res, status="torch_tensor_materialized_no_forward_no_training_in_p39")
    with pytest.raises(ValueError):
        validate_p39_result(bad)

def test_p39_075_validate_p39_result_invalid_flags(monkeypatch):
    from src.phase2.torch_boundary import TorchDependencyStatus
    monkeypatch.setattr("src.phase2.fc_vae_tensor_materialization.build_torch_dependency_status", lambda policy: TorchDependencyStatus(
        contract_version="phase2_p26_torch_boundary_contract_v1", backend_name="torch", available=False, policy="optional", import_safe=False, top_level_import_required=False, reason="mocked"
    ))
    res = run_tensor_materialization_probe()
    bad = dataclasses.replace(res, tensor_materialized_in_p39=True)
    with pytest.raises(ValueError):
        validate_p39_result(bad)

def test_p39_076_validate_p39_result_invalid_reason():
    res = run_tensor_materialization_probe()
    bad = dataclasses.replace(res, reason="")
    with pytest.raises(ValueError):
        validate_p39_result(bad)


# 77-78: metadata & result builders
def test_p39_077_build_tensor_materialization_metadata_valid():
    req = build_tensor_materialization_request_from_p38_default()
    meta = build_tensor_materialization_metadata_p39(req, None)
    validate_p39_metadata(meta)

def test_p39_078_build_tensor_materialization_result_valid():
    req = build_tensor_materialization_request_from_p38_default()
    res = build_tensor_materialization_result(req)
    validate_p39_result(res)


def test_p39_079_result_status_if_torch_unavailable(monkeypatch):
    from src.phase2.torch_boundary import TorchDependencyStatus
    monkeypatch.setattr("src.phase2.fc_vae_tensor_materialization.build_torch_dependency_status", lambda policy: TorchDependencyStatus(
        contract_version="phase2_p26_torch_boundary_contract_v1", backend_name="torch", available=False, policy="optional", import_safe=False, top_level_import_required=False, reason="mocked"
    ))
    res = run_tensor_materialization_probe()
    assert res.status == "blocked_torch_unavailable"
    assert res.materialized_tensor is None


# 80: result status if torch available (mocked)
def test_p39_080_result_status_if_torch_available_mocked(monkeypatch):
    # Patch load_torch_for_p39_materialization to return our MockTorch
    monkeypatch.setattr(
        "src.phase2.fc_vae_tensor_materialization.load_torch_for_p39_materialization",
        lambda: MockTorch
    )
    # Patch build_torch_dependency_status to simulate torch being available
    from src.phase2.torch_boundary import TorchDependencyStatus
    monkeypatch.setattr(
        "src.phase2.fc_vae_tensor_materialization.build_torch_dependency_status",
        lambda policy: TorchDependencyStatus(
            contract_version="phase2_p26_torch_boundary_contract_v1",
            backend_name="torch",
            available=True,
            policy="optional",
            import_safe=True,
            top_level_import_required=False,
            reason="mocked_available"
        )
    )

    res = run_tensor_materialization_probe()
    assert res.status == "torch_tensor_materialized_no_forward_no_training_in_p39"
    assert res.materialized_tensor is not None
    validate_p39_result(res)


# 81-88: result no-op and safety flags
def test_p39_081_result_no_forward_execution():
    res = run_tensor_materialization_probe()
    assert res.no_forward_execution is True

def test_p39_082_result_no_output_generation():
    res = run_tensor_materialization_probe()
    assert res.no_output_generation is True

def test_p39_083_result_no_training_loop():
    res = run_tensor_materialization_probe()
    assert res.no_training_loop is True

def test_p39_084_result_no_optimizer():
    res = run_tensor_materialization_probe()
    assert res.no_optimizer is True

def test_p39_085_result_no_checkpointing():
    res = run_tensor_materialization_probe()
    assert res.no_checkpointing is True

def test_p39_086_result_no_artifact_generation():
    res = run_tensor_materialization_probe()
    assert res.no_artifact_generation is True

def test_p39_087_result_no_final_comparison():
    res = run_tensor_materialization_probe()
    assert res.no_final_comparison is True

def test_p39_088_result_no_scientific_conclusion():
    res = run_tensor_materialization_probe()
    assert res.no_scientific_conclusion is True


# 89-92: serializer to json dicts
def test_p39_089_serializer_request_to_json_dict():
    req = build_tensor_materialization_request_from_p38_default()
    d = p39_tensor_materialization_request_to_json_dict(req)
    assert isinstance(d, dict)
    assert d["materialization_kind"] == FC_VAE_TENSOR_MATERIALIZATION_KIND

def test_p39_090_serializer_metadata_to_json_dict(monkeypatch):
    from src.phase2.torch_boundary import TorchDependencyStatus
    monkeypatch.setattr("src.phase2.fc_vae_tensor_materialization.build_torch_dependency_status", lambda policy: TorchDependencyStatus(
        contract_version="phase2_p26_torch_boundary_contract_v1", backend_name="torch", available=False, policy="optional", import_safe=False, top_level_import_required=False, reason="mocked"
    ))
    req = build_tensor_materialization_request_from_p38_default()
    meta = build_tensor_materialization_metadata_p39(req, None)
    d = p39_tensor_materialization_metadata_to_json_dict(meta)
    assert isinstance(d, dict)
    assert d["torch_available"] is False

def test_p39_091_serializer_result_to_json_dict(monkeypatch):
    from src.phase2.torch_boundary import TorchDependencyStatus
    monkeypatch.setattr("src.phase2.fc_vae_tensor_materialization.build_torch_dependency_status", lambda policy: TorchDependencyStatus(
        contract_version="phase2_p26_torch_boundary_contract_v1", backend_name="torch", available=False, policy="optional", import_safe=False, top_level_import_required=False, reason="mocked"
    ))
    res = run_tensor_materialization_probe()
    d = p39_tensor_materialization_result_to_json_dict(res)
    assert isinstance(d, dict)
    assert d["status"] == "blocked_torch_unavailable"

def test_p39_092_serializer_compact_p39_tensor_materialization_json(monkeypatch):
    from src.phase2.torch_boundary import TorchDependencyStatus
    monkeypatch.setattr("src.phase2.fc_vae_tensor_materialization.build_torch_dependency_status", lambda policy: TorchDependencyStatus(
        contract_version="phase2_p26_torch_boundary_contract_v1", backend_name="torch", available=False, policy="optional", import_safe=False, top_level_import_required=False, reason="mocked"
    ))
    res = run_tensor_materialization_probe()
    js = compact_p39_tensor_materialization_json(res)
    assert isinstance(js, str)
    assert "blocked_torch_unavailable" in js


# 93-99: serialization exclusions & safety
def test_p39_093_serializers_do_not_include_raw_tensor_object(monkeypatch):
    monkeypatch.setattr("src.phase2.fc_vae_tensor_materialization.load_torch_for_p39_materialization", lambda: MockTorch)
    from src.phase2.torch_boundary import TorchDependencyStatus
    monkeypatch.setattr("src.phase2.fc_vae_tensor_materialization.build_torch_dependency_status", lambda policy: TorchDependencyStatus(
        contract_version="phase2_p26_torch_boundary_contract_v1", backend_name="torch", available=True, policy="optional", import_safe=True, top_level_import_required=False, reason="mocked"
    ))
    res = run_tensor_materialization_probe()
    js = compact_p39_tensor_materialization_json(res)
    assert "MockTensor object" not in js
    assert "tensor_object" not in json.loads(js)["materialized_tensor"]

def test_p39_094_serializers_do_not_include_full_nested_batch_values():
    res = run_tensor_materialization_probe()
    js = compact_p39_tensor_materialization_json(res)
    assert '"nested_batch_values":' not in js

def test_p39_095_serializers_do_not_include_flat_values():
    res = run_tensor_materialization_probe()
    js = compact_p39_tensor_materialization_json(res)
    assert "flat_values" not in js

def test_p39_096_json_has_tensor_metadata_only(monkeypatch):
    monkeypatch.setattr("src.phase2.fc_vae_tensor_materialization.load_torch_for_p39_materialization", lambda: MockTorch)
    from src.phase2.torch_boundary import TorchDependencyStatus
    monkeypatch.setattr("src.phase2.fc_vae_tensor_materialization.build_torch_dependency_status", lambda policy: TorchDependencyStatus(
        contract_version="phase2_p26_torch_boundary_contract_v1", backend_name="torch", available=True, policy="optional", import_safe=True, top_level_import_required=False, reason="mocked"
    ))
    res = run_tensor_materialization_probe()
    d = p39_tensor_materialization_result_to_json_dict(res)
    t_dict = d["materialized_tensor"]
    assert "tensor_type_name" in t_dict
    assert "tensor_dtype_name" in t_dict
    assert "tensor_device_type" in t_dict
    assert t_dict["tensor_shape_tuple"] == [2, 32]
    assert t_dict["tensor_numel"] == 64

def test_p39_097_json_has_no_key_with_colon_bool():
    res = run_tensor_materialization_probe()
    js = compact_p39_tensor_materialization_json(res)
    assert ": bool" not in js

def test_p39_098_json_has_no_local_paths():
    res = run_tensor_materialization_probe()
    js = compact_p39_tensor_materialization_json(res)
    assert "file:///" not in js

def test_p39_099_json_has_no_forbidden_success_claims():
    res = run_tensor_materialization_probe()
    js = compact_p39_tensor_materialization_json(res)
    assert "model works" not in js.lower()


# 100-101: smoke script safety
def test_p39_100_smoke_script_imports_clean():
    content = pathlib.Path("tools/phase2/run_p39_tensor_materialization_smoke.py").read_text(encoding="utf-8")
    for pattern in ("import torch", "from torch", "import numpy", "from numpy"):
        assert pattern not in content

def test_p39_101_smoke_script_has_no_torch_numpy_random_secrets_array_argparse_subprocess():
    content = pathlib.Path("tools/phase2/run_p39_tensor_materialization_smoke.py").read_text(encoding="utf-8")
    for pattern in ("import random", "import secrets", "import array", "import argparse", "import subprocess"):
        assert pattern not in content


# 102: init has no monkeypatching
def test_p39_102_src_phase2_init_has_no_monkeypatch_symbols():
    content = pathlib.Path("src/phase2/__init__.py").read_text(encoding="utf-8")
    assert "subprocess.run" not in content
    assert "monkeypatch" not in content


# 103-108: module content safety checks
def test_p39_103_module_has_no_top_level_torch_import():
    content = pathlib.Path("src/phase2/fc_vae_tensor_materialization.py").read_text(encoding="utf-8")
    lines = content.splitlines()
    for line in lines:
        if line.strip().startswith("import torch") or line.strip().startswith("from torch"):
            # Ensure it is not at top-level
            # A simple indentation check
            assert line.startswith("    ") or line.startswith("\t")

def test_p39_104_module_has_no_numpy_import():
    content = pathlib.Path("src/phase2/fc_vae_tensor_materialization.py").read_text(encoding="utf-8")
    assert "import numpy" not in content
    assert "from numpy" not in content

def test_p39_105_module_has_no_random_secrets_import():
    content = pathlib.Path("src/phase2/fc_vae_tensor_materialization.py").read_text(encoding="utf-8")
    assert "import random" not in content
    assert "import secrets" not in content

def test_p39_106_module_has_no_array_import():
    content = pathlib.Path("src/phase2/fc_vae_tensor_materialization.py").read_text(encoding="utf-8")
    assert "import array" not in content

def test_p39_107_module_has_no_train_loss_optimizer_checkpoint_functions():
    content = pathlib.Path("src/phase2/fc_vae_tensor_materialization.py").read_text(encoding="utf-8")
    for pattern in ("def train", "def loss", "def optimizer", "def checkpoint"):
        assert pattern not in content

def test_p39_108_module_has_no_def_forward():
    content = pathlib.Path("src/phase2/fc_vae_tensor_materialization.py").read_text(encoding="utf-8")
    assert "def forward" not in content


# 109-119: mocked tensor properties tests
def test_p39_109_materialized_tensor_mocked_creation(monkeypatch):
    monkeypatch.setattr("src.phase2.fc_vae_tensor_materialization.load_torch_for_p39_materialization", lambda: MockTorch)
    req = build_tensor_materialization_request_from_p38_default()
    t_obj = build_materialized_torch_tensor(req)
    assert t_obj is not None

def test_p39_110_materialized_tensor_mocked_type_name(monkeypatch):
    monkeypatch.setattr("src.phase2.fc_vae_tensor_materialization.load_torch_for_p39_materialization", lambda: MockTorch)
    req = build_tensor_materialization_request_from_p38_default()
    t_obj = build_materialized_torch_tensor(req)
    assert "MockTensor" in t_obj.tensor_type_name

def test_p39_111_materialized_tensor_mocked_dtype_name(monkeypatch):
    monkeypatch.setattr("src.phase2.fc_vae_tensor_materialization.load_torch_for_p39_materialization", lambda: MockTorch)
    req = build_tensor_materialization_request_from_p38_default()
    t_obj = build_materialized_torch_tensor(req)
    assert t_obj.tensor_dtype_name == "torch.float32"

def test_p39_112_materialized_tensor_mocked_device_type(monkeypatch):
    monkeypatch.setattr("src.phase2.fc_vae_tensor_materialization.load_torch_for_p39_materialization", lambda: MockTorch)
    req = build_tensor_materialization_request_from_p38_default()
    t_obj = build_materialized_torch_tensor(req)
    assert t_obj.tensor_device_type == "cpu"

def test_p39_113_materialized_tensor_mocked_shape_tuple(monkeypatch):
    monkeypatch.setattr("src.phase2.fc_vae_tensor_materialization.load_torch_for_p39_materialization", lambda: MockTorch)
    req = build_tensor_materialization_request_from_p38_default()
    t_obj = build_materialized_torch_tensor(req)
    assert t_obj.tensor_shape_tuple == (2, 32)

def test_p39_114_materialized_tensor_mocked_numel(monkeypatch):
    monkeypatch.setattr("src.phase2.fc_vae_tensor_materialization.load_torch_for_p39_materialization", lambda: MockTorch)
    req = build_tensor_materialization_request_from_p38_default()
    t_obj = build_materialized_torch_tensor(req)
    assert t_obj.tensor_numel == 64

def test_p39_115_materialized_tensor_mocked_requires_grad(monkeypatch):
    monkeypatch.setattr("src.phase2.fc_vae_tensor_materialization.load_torch_for_p39_materialization", lambda: MockTorch)
    req = build_tensor_materialization_request_from_p38_default()
    t_obj = build_materialized_torch_tensor(req)
    assert t_obj.tensor_requires_grad is False

def test_p39_116_materialized_tensor_mocked_is_floating_point(monkeypatch):
    monkeypatch.setattr("src.phase2.fc_vae_tensor_materialization.load_torch_for_p39_materialization", lambda: MockTorch)
    req = build_tensor_materialization_request_from_p38_default()
    t_obj = build_materialized_torch_tensor(req)
    assert t_obj.tensor_is_floating_point is True

def test_p39_117_materialized_tensor_mocked_values_match(monkeypatch):
    monkeypatch.setattr("src.phase2.fc_vae_tensor_materialization.load_torch_for_p39_materialization", lambda: MockTorch)
    req = build_tensor_materialization_request_from_p38_default()
    t_obj = build_materialized_torch_tensor(req)
    assert t_obj.tensor_values_match_p37_nested_values is True

def test_p39_118_materialized_tensor_mocked_first_row_first_4_values(monkeypatch):
    monkeypatch.setattr("src.phase2.fc_vae_tensor_materialization.load_torch_for_p39_materialization", lambda: MockTorch)
    req = build_tensor_materialization_request_from_p38_default()
    t_obj = build_materialized_torch_tensor(req)
    expected = (-0.24297189, -0.1686747, -0.09437751, -0.02008032)
    for v, exp in zip(t_obj.first_row_first_4_values, expected):
        assert abs(v - exp) < 1e-6

def test_p39_119_materialized_tensor_mocked_second_row_first_4_values(monkeypatch):
    monkeypatch.setattr("src.phase2.fc_vae_tensor_materialization.load_torch_for_p39_materialization", lambda: MockTorch)
    req = build_tensor_materialization_request_from_p38_default()
    t_obj = build_materialized_torch_tensor(req)
    expected = (0.13253012, 0.20682731, 0.2811245, 0.35542169)
    for v, exp in zip(t_obj.second_row_first_4_values, expected):
        assert abs(v - exp) < 1e-6


# 120: scope gate
def test_p39_120_scope_gate():
    allowed = {
        "src/phase2/fc_vae_tensor_materialization.py",
        "src/phase2/__init__.py",
        "tests/test_phase2_fc_vae_tensor_materialization.py",
        "tools/phase2/run_p39_tensor_materialization_smoke.py",
        "tests/test_phase2_p39_tensor_materialization_smoke.py",
        "reports/PHASE_2_P39_OPTIONAL_TORCH_TENSOR_MATERIALIZATION_NO_FORWARD_NO_TRAINING_REPORT.md",
    }
    # Base branch checkout: phase2/p38-tensor-materialization-request-contract-no-tensor-allocation
    res = subprocess.run(
        ["git", "diff", "--name-only", "phase2/p38-tensor-materialization-request-contract-no-tensor-allocation"],
        capture_output=True, text=True, check=True
    )
    modified = [line.strip() for line in res.stdout.splitlines() if line.strip()]
    for f in modified:
        f_norm = f.replace("\\", "/")
        assert f_norm in allowed, f"Forbidden file modification detected in P39: {f_norm}"
