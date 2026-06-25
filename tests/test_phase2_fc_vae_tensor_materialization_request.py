# tests/test_phase2_fc_vae_tensor_materialization_request.py

import dataclasses
import json
import pathlib
import subprocess
import pytest
from typing import Tuple

from src.phase2.fc_vae_tensor_materialization_request import (
    FC_VAE_TENSOR_MATERIALIZATION_REQUEST_CONTRACT_VERSION,
    FC_VAE_TENSOR_MATERIALIZATION_REQUEST_KIND,
    FC_VAE_TENSOR_MATERIALIZATION_REQUEST_MODULE_NAME,
    FC_VAE_TENSOR_MATERIALIZATION_REQUEST_STATUS_TORCH_UNAVAILABLE,
    FC_VAE_TENSOR_MATERIALIZATION_REQUEST_STATUS_NESTED_VALUES_BLOCKED,
    FC_VAE_TENSOR_MATERIALIZATION_REQUEST_STATUS_REQUEST_ONLY,
    SUPPORTED_FC_VAE_TENSOR_MATERIALIZATION_REQUEST_STATUSES,
    FC_VAE_TENSOR_TARGET_FRAMEWORK,
    FC_VAE_TENSOR_TARGET_DTYPE_INTENT,
    FC_VAE_TENSOR_TARGET_DEVICE_INTENT,
    FC_VAE_TENSOR_LAYOUT_KIND,
    DEFAULT_P38_BATCH_SIZE,
    DEFAULT_P38_INPUT_FLAT_DIM,
    DEFAULT_P38_FULL_VECTOR_LENGTH,
    DEFAULT_P38_SHAPE_TUPLE,
    DEFAULT_P38_REQUIRES_GRAD,
    FCVAETensorMaterializationRequest,
    FCVAETensorMaterializationSpec,
    FCVAETensorMaterializationMetadata,
    FCVAETensorMaterializationRequestResult,
    validate_non_empty_str,
    validate_bool,
    validate_positive_int,
    validate_shape_tuple,
    validate_full_vector_length,
    validate_target_framework,
    validate_target_dtype_intent,
    validate_target_device_intent,
    validate_target_layout_kind,
    validate_tensor_materialization_request_status,
    assert_no_local_path_leakage,
    assert_no_forbidden_claims,
    validate_tensor_materialization_request,
    validate_tensor_materialization_spec,
    validate_tensor_materialization_metadata,
    validate_tensor_materialization_request_result,
    build_tensor_materialization_request_from_p37_default,
    build_tensor_materialization_spec,
    build_tensor_materialization_metadata,
    build_tensor_materialization_request_result,
    run_tensor_materialization_request_probe,
    tensor_materialization_request_to_json_dict,
    tensor_materialization_spec_to_json_dict,
    tensor_materialization_metadata_to_json_dict,
    tensor_materialization_request_result_to_json_dict,
    compact_tensor_materialization_request_json,
)

# 1-16: Constants exact checks
def test_p38_001_contract_version_exact():
    assert FC_VAE_TENSOR_MATERIALIZATION_REQUEST_CONTRACT_VERSION == "phase2_p38_tensor_materialization_request_contract_v1"

def test_p38_002_request_kind_exact():
    assert FC_VAE_TENSOR_MATERIALIZATION_REQUEST_KIND == "tensor_materialization_request_metadata_no_tensor_allocation"

def test_p38_003_module_name_exact():
    assert FC_VAE_TENSOR_MATERIALIZATION_REQUEST_MODULE_NAME == "src.phase2.fc_vae_tensor_materialization_request"

def test_p38_004_status_torch_unavailable():
    assert FC_VAE_TENSOR_MATERIALIZATION_REQUEST_STATUS_TORCH_UNAVAILABLE == "blocked_torch_unavailable"

def test_p38_005_status_nested_values_blocked():
    assert FC_VAE_TENSOR_MATERIALIZATION_REQUEST_STATUS_NESTED_VALUES_BLOCKED == "blocked_by_nested_values_status"

def test_p38_006_status_request_only():
    assert FC_VAE_TENSOR_MATERIALIZATION_REQUEST_STATUS_REQUEST_ONLY == "tensor_request_metadata_only_no_allocation_in_p38"

def test_p38_007_supported_statuses():
    assert SUPPORTED_FC_VAE_TENSOR_MATERIALIZATION_REQUEST_STATUSES == (
        "blocked_torch_unavailable",
        "blocked_by_nested_values_status",
        "tensor_request_metadata_only_no_allocation_in_p38",
    )

def test_p38_008_target_framework():
    assert FC_VAE_TENSOR_TARGET_FRAMEWORK == "torch"

def test_p38_009_target_dtype_intent():
    assert FC_VAE_TENSOR_TARGET_DTYPE_INTENT == "float32"

def test_p38_010_target_device_intent():
    assert FC_VAE_TENSOR_TARGET_DEVICE_INTENT == "cpu"

def test_p38_011_target_layout_kind():
    assert FC_VAE_TENSOR_LAYOUT_KIND == "row_major_2d_batch_tensor"

def test_p38_012_default_batch_size():
    assert DEFAULT_P38_BATCH_SIZE == 2

def test_p38_013_default_input_flat_dim():
    assert DEFAULT_P38_INPUT_FLAT_DIM == 32

def test_p38_014_default_full_vector_length():
    assert DEFAULT_P38_FULL_VECTOR_LENGTH == 64

def test_p38_015_default_shape_tuple():
    assert DEFAULT_P38_SHAPE_TUPLE == (2, 32)

def test_p38_016_default_requires_grad():
    assert DEFAULT_P38_REQUIRES_GRAD is False


# 17-20: Dataclasses frozen checks
def test_p38_017_request_frozen():
    req = build_tensor_materialization_request_from_p37_default()
    with pytest.raises(Exception):
        req.reason = "modified"

def test_p38_018_spec_frozen():
    req = build_tensor_materialization_request_from_p37_default()
    spec = build_tensor_materialization_spec(req)
    with pytest.raises(Exception):
        spec.reason = "modified"

def test_p38_019_metadata_frozen():
    req = build_tensor_materialization_request_from_p37_default()
    meta = build_tensor_materialization_metadata(req)
    with pytest.raises(Exception):
        meta.reason = "modified"

def test_p38_020_result_frozen():
    res = run_tensor_materialization_request_probe()
    with pytest.raises(Exception):
        res.reason = "modified"


# 21-25: validate_non_empty_str
def test_p38_021_validate_non_empty_str_valid():
    validate_non_empty_str("valid", "test")

def test_p38_022_validate_non_empty_str_type_error():
    with pytest.raises(TypeError):
        validate_non_empty_str(123, "test")

def test_p38_023_validate_non_empty_str_empty_error():
    with pytest.raises(ValueError):
        validate_non_empty_str("", "test")

def test_p38_024_validate_non_empty_str_padded_leading():
    with pytest.raises(ValueError):
        validate_non_empty_str(" padded", "test")

def test_p38_025_validate_non_empty_str_padded_trailing():
    with pytest.raises(ValueError):
        validate_non_empty_str("padded ", "test")


# 26-28: validate_bool
def test_p38_026_validate_bool_valid():
    validate_bool(True, "test")
    validate_bool(False, "test")

def test_p38_027_validate_bool_int_error():
    with pytest.raises(TypeError):
        validate_bool(1, "test")

def test_p38_028_validate_bool_str_error():
    with pytest.raises(TypeError):
        validate_bool("True", "test")


# 29-32: validate_positive_int
def test_p38_029_validate_positive_int_valid():
    validate_positive_int(5, "test")

def test_p38_030_validate_positive_int_zero():
    with pytest.raises(ValueError):
        validate_positive_int(0, "test")

def test_p38_031_validate_positive_int_negative():
    with pytest.raises(ValueError):
        validate_positive_int(-5, "test")

def test_p38_032_validate_positive_int_bool_error():
    with pytest.raises(TypeError):
        validate_positive_int(True, "test")


# 33-36: validate_shape_tuple
def test_p38_033_validate_shape_tuple_valid():
    validate_shape_tuple((2, 32))

def test_p38_034_validate_shape_tuple_wrong_values():
    with pytest.raises(ValueError):
        validate_shape_tuple((3, 32))

def test_p38_035_validate_shape_tuple_wrong_length():
    with pytest.raises(ValueError):
        validate_shape_tuple((2, 32, 1))

def test_p38_036_validate_shape_tuple_type_error():
    with pytest.raises(TypeError):
        validate_shape_tuple([2, 32])


# 37-39: validate_full_vector_length
def test_p38_037_validate_full_vector_length_valid():
    validate_full_vector_length(64)

def test_p38_038_validate_full_vector_length_invalid():
    with pytest.raises(ValueError):
        validate_full_vector_length(65)

def test_p38_039_validate_full_vector_length_type_error():
    with pytest.raises(TypeError):
        validate_full_vector_length("64")


# 40-42: validate_target_framework
def test_p38_040_validate_target_framework_valid():
    validate_target_framework("torch")

def test_p38_041_validate_target_framework_invalid():
    with pytest.raises(ValueError):
        validate_target_framework("tensorflow")

def test_p38_042_validate_target_framework_type():
    with pytest.raises(TypeError):
        validate_target_framework(123)


# 43-45: validate_target_dtype_intent
def test_p38_043_validate_target_dtype_intent_valid():
    validate_target_dtype_intent("float32")

def test_p38_044_validate_target_dtype_intent_invalid():
    with pytest.raises(ValueError):
        validate_target_dtype_intent("float64")

def test_p38_045_validate_target_dtype_intent_type():
    with pytest.raises(TypeError):
        validate_target_dtype_intent(32)


# 46-48: validate_target_device_intent
def test_p38_046_validate_target_device_intent_valid():
    validate_target_device_intent("cpu")

def test_p38_047_validate_target_device_intent_invalid():
    with pytest.raises(ValueError):
        validate_target_device_intent("cuda")

def test_p38_048_validate_target_device_intent_type():
    with pytest.raises(TypeError):
        validate_target_device_intent(None)


# 49-51: validate_target_layout_kind
def test_p38_049_validate_target_layout_kind_valid():
    validate_target_layout_kind("row_major_2d_batch_tensor")

def test_p38_050_validate_target_layout_kind_invalid():
    with pytest.raises(ValueError):
        validate_target_layout_kind("col_major")

def test_p38_051_validate_target_layout_kind_type():
    with pytest.raises(TypeError):
        validate_target_layout_kind(False)


# 52-54: validate_tensor_materialization_request_status
def test_p38_052_validate_status_valid():
    validate_tensor_materialization_request_status("blocked_torch_unavailable")
    validate_tensor_materialization_request_status("tensor_request_metadata_only_no_allocation_in_p38")

def test_p38_053_validate_status_invalid():
    with pytest.raises(ValueError):
        validate_tensor_materialization_request_status("invalid_status")

def test_p38_054_validate_status_type():
    with pytest.raises(TypeError):
        validate_tensor_materialization_request_status(1.0)


# 55-58: assert_no_local_path_leakage
def test_p38_055_assert_no_local_path_leakage_valid():
    assert_no_local_path_leakage("safe string")
    assert_no_local_path_leakage({"key": "val", "nested": ["ok"]})

def test_p38_056_assert_no_local_path_leakage_file_url():
    with pytest.raises(ValueError):
        assert_no_local_path_leakage("file:///C:/Users/test")

def test_p38_057_assert_no_local_path_leakage_windows_c():
    with pytest.raises(ValueError):
        assert_no_local_path_leakage("C:\\path\\to\\file")

def test_p38_058_assert_no_local_path_leakage_home():
    with pytest.raises(ValueError):
        assert_no_local_path_leakage("/home/user/project")


# 59-62: assert_no_forbidden_claims
def test_p38_059_assert_no_forbidden_claims_valid():
    assert_no_forbidden_claims("no_scientific_conclusion")
    assert_no_forbidden_claims("no_final_comparison")
    assert_no_forbidden_claims("tensor_request_metadata_only_no_allocation_in_p38")

def test_p38_060_assert_no_forbidden_claims_solved():
    with pytest.raises(ValueError):
        assert_no_forbidden_claims("we solved it")

def test_p38_061_assert_no_forbidden_claims_best():
    with pytest.raises(ValueError):
        assert_no_forbidden_claims("the best model")

def test_p38_062_assert_no_forbidden_claims_sota():
    with pytest.raises(ValueError):
        assert_no_forbidden_claims("state of the art results")


# 63-65: validate_tensor_materialization_request
def test_p38_063_validate_request_valid():
    req = build_tensor_materialization_request_from_p37_default()
    validate_tensor_materialization_request(req)

def test_p38_064_validate_request_type_error():
    with pytest.raises(TypeError):
        validate_tensor_materialization_request("not a request")

def test_p38_065_validate_request_invalid_contract():
    req = build_tensor_materialization_request_from_p37_default()
    bad = dataclasses.replace(req, contract_version="wrong")
    with pytest.raises(ValueError):
        validate_tensor_materialization_request(bad)


# 66-68: validate_tensor_materialization_spec
def test_p38_066_validate_spec_valid():
    req = build_tensor_materialization_request_from_p37_default()
    spec = build_tensor_materialization_spec(req)
    validate_tensor_materialization_spec(spec)

def test_p38_067_validate_spec_type_error():
    with pytest.raises(TypeError):
        validate_tensor_materialization_spec("not a spec")

def test_p38_068_validate_spec_invalid_row_count():
    req = build_tensor_materialization_request_from_p37_default()
    spec = build_tensor_materialization_spec(req)
    bad = dataclasses.replace(spec, source_nested_row_count=3)
    with pytest.raises(ValueError):
        validate_tensor_materialization_spec(bad)


# 69-71: validate_tensor_materialization_metadata
def test_p38_069_validate_metadata_valid():
    req = build_tensor_materialization_request_from_p37_default()
    meta = build_tensor_materialization_metadata(req)
    validate_tensor_materialization_metadata(meta)

def test_p38_070_validate_metadata_type_error():
    with pytest.raises(TypeError):
        validate_tensor_materialization_metadata("not a metadata")

def test_p38_071_validate_metadata_invalid_attempted():
    req = build_tensor_materialization_request_from_p37_default()
    meta = build_tensor_materialization_metadata(req)
    bad = dataclasses.replace(meta, tensor_materialization_attempted=True)
    with pytest.raises(ValueError):
        validate_tensor_materialization_metadata(bad)


# 72-74: validate_tensor_materialization_request_result
def test_p38_072_validate_result_valid():
    res = run_tensor_materialization_request_probe()
    validate_tensor_materialization_request_result(res)

def test_p38_073_validate_result_type_error():
    with pytest.raises(TypeError):
        validate_tensor_materialization_request_result("not a result")

def test_p38_074_validate_result_invalid_no_tensor():
    res = run_tensor_materialization_request_probe()
    bad = dataclasses.replace(res, no_tensor_created=False)
    with pytest.raises(ValueError):
        validate_tensor_materialization_request_result(bad)


# 75-79: Rejects allow flags True
def test_p38_075_request_rejects_allow_tensor():
    req = build_tensor_materialization_request_from_p37_default()
    bad = dataclasses.replace(req, allow_tensor_materialization_in_p38=True)
    with pytest.raises(ValueError):
        validate_tensor_materialization_request(bad)

def test_p38_076_request_rejects_allow_array():
    req = build_tensor_materialization_request_from_p37_default()
    bad = dataclasses.replace(req, allow_array_materialization_in_p38=True)
    with pytest.raises(ValueError):
        validate_tensor_materialization_request(bad)

def test_p38_077_request_rejects_allow_forward():
    req = build_tensor_materialization_request_from_p37_default()
    bad = dataclasses.replace(req, allow_forward_execution_in_p38=True)
    with pytest.raises(ValueError):
        validate_tensor_materialization_request(bad)

def test_p38_078_request_rejects_allow_output():
    req = build_tensor_materialization_request_from_p37_default()
    bad = dataclasses.replace(req, allow_output_generation_in_p38=True)
    with pytest.raises(ValueError):
        validate_tensor_materialization_request(bad)

def test_p38_079_request_rejects_allow_training():
    req = build_tensor_materialization_request_from_p37_default()
    bad = dataclasses.replace(req, allow_training_in_p38=True)
    with pytest.raises(ValueError):
        validate_tensor_materialization_request(bad)


# 80-84: request builder properties
def test_p38_080_request_builder_contract_copies():
    req = build_tensor_materialization_request_from_p37_default()
    assert req.source_nested_values_contract_version == "phase2_p37_nested_batch_values_contract_v1"
    assert req.source_torch_boundary_contract_version == "phase2_p26_torch_boundary_contract_v1"

def test_p38_081_request_builder_shape_exact():
    req = build_tensor_materialization_request_from_p37_default()
    assert req.shape_tuple == (2, 32)
    assert req.batch_size == 2
    assert req.input_flat_dim == 32
    assert req.full_vector_length == 64

def test_p38_082_request_builder_requires_grad_false():
    req = build_tensor_materialization_request_from_p37_default()
    assert req.requires_grad is False

def test_p38_083_request_builder_allow_flags_false():
    req = build_tensor_materialization_request_from_p37_default()
    assert req.allow_tensor_materialization_in_p38 is False
    assert req.allow_array_materialization_in_p38 is False
    assert req.allow_forward_execution_in_p38 is False
    assert req.allow_output_generation_in_p38 is False
    assert req.allow_training_in_p38 is False

def test_p38_084_request_builder_reasons_valid():
    req = build_tensor_materialization_request_from_p37_default()
    assert req.reason == "p38_tensor_materialization_request_from_p37_default"


# 85-89: spec builder properties
def test_p38_085_spec_builder_source_counts():
    req = build_tensor_materialization_request_from_p37_default()
    spec = build_tensor_materialization_spec(req)
    assert spec.source_nested_row_count == 2
    assert spec.source_nested_row_lengths == (32, 32)
    assert spec.source_total_scalar_count == 64
    assert spec.source_flattened_matches_p35 is True

def test_p38_086_spec_builder_flags():
    req = build_tensor_materialization_request_from_p37_default()
    spec = build_tensor_materialization_spec(req)
    assert spec.tensor_request_declared is True
    assert spec.tensor_materialized is False
    assert spec.array_materialized is False
    assert spec.forward_executed is False
    assert spec.output_generated is False
    assert spec.training_executed is False

def test_p38_087_spec_builder_targets():
    req = build_tensor_materialization_request_from_p37_default()
    spec = build_tensor_materialization_spec(req)
    assert spec.target_framework == "torch"
    assert spec.target_dtype_intent == "float32"
    assert spec.target_device_intent == "cpu"
    assert spec.target_layout_kind == "row_major_2d_batch_tensor"

def test_p38_088_spec_builder_reasons():
    req = build_tensor_materialization_request_from_p37_default()
    spec = build_tensor_materialization_spec(req)
    assert spec.reason == "p38_tensor_materialization_spec"

def test_p38_089_spec_builder_requires_grad_false():
    req = build_tensor_materialization_request_from_p37_default()
    spec = build_tensor_materialization_spec(req)
    assert spec.requires_grad is False


# 90-94: metadata builder properties
def test_p38_090_metadata_builder_backend():
    req = build_tensor_materialization_request_from_p37_default()
    meta = build_tensor_materialization_metadata(req)
    assert meta.torch_backend_name == "torch"
    assert meta.torch_policy == "optional"

def test_p38_091_metadata_builder_top_level_import():
    req = build_tensor_materialization_request_from_p37_default()
    meta = build_tensor_materialization_metadata(req)
    assert meta.top_level_torch_import_required is False

def test_p38_092_metadata_builder_nested_p37_available():
    req = build_tensor_materialization_request_from_p37_default()
    meta = build_tensor_materialization_metadata(req)
    assert meta.nested_values_available_in_p37 is True
    assert meta.nested_values_materialized_in_p37 is True

def test_p38_093_metadata_builder_attempted_flags():
    req = build_tensor_materialization_request_from_p37_default()
    meta = build_tensor_materialization_metadata(req)
    assert meta.tensor_materialization_attempted is False
    assert meta.array_materialization_attempted is False
    assert meta.forward_execution_attempted is False
    assert meta.output_generation_attempted is False
    assert meta.training_attempted is False

def test_p38_094_metadata_builder_reasons():
    req = build_tensor_materialization_request_from_p37_default()
    meta = build_tensor_materialization_metadata(req)
    assert meta.reason == "p38_tensor_materialization_metadata"


# 95-99: result builder properties
def test_p38_095_result_builder_availability_flags():
    res = run_tensor_materialization_request_probe()
    assert res.tensor_request_available_in_p38 is True
    assert res.tensor_materialization_available_in_p38 is False
    assert res.array_materialization_available_in_p38 is False
    assert res.forward_execution_available_in_p38 is False
    assert res.output_generation_available_in_p38 is False
    assert res.training_available_in_p38 is False

def test_p38_096_result_builder_no_flags():
    res = run_tensor_materialization_request_probe()
    assert res.no_tensor_created is True
    assert res.no_array_created is True
    assert res.no_forward_execution is True
    assert res.no_output_generation is True
    assert res.no_training_loop is True
    assert res.no_optimizer is True
    assert res.no_checkpointing is True
    assert res.no_artifact_generation is True
    assert res.no_final_comparison is True
    assert res.no_scientific_conclusion is True

def test_p38_097_result_builder_status_uncoherence_torch_unavailable():
    req = build_tensor_materialization_request_from_p37_default()
    res = build_tensor_materialization_request_result(req)
    # mock torch_available is False
    meta_bad = dataclasses.replace(res.metadata, torch_available=False)
    bad_res = dataclasses.replace(res, metadata=meta_bad, status="tensor_request_metadata_only_no_allocation_in_p38")
    with pytest.raises(ValueError):
        validate_tensor_materialization_request_result(bad_res)

def test_p38_098_result_builder_status_uncoherence_nested_blocked():
    req = build_tensor_materialization_request_from_p37_default()
    res = build_tensor_materialization_request_result(req)
    # mock nested_values_available_in_p37 is False
    meta_bad = dataclasses.replace(res.metadata, torch_available=True, nested_values_available_in_p37=False)
    bad_res = dataclasses.replace(res, metadata=meta_bad, status="tensor_request_metadata_only_no_allocation_in_p38")
    with pytest.raises(ValueError):
        validate_tensor_materialization_request_result(bad_res)

def test_p38_099_result_builder_status_uncoherence_ok():
    req = build_tensor_materialization_request_from_p37_default()
    res = build_tensor_materialization_request_result(req)
    # mock torch_available is True
    meta_ok = dataclasses.replace(res.metadata, torch_available=True, nested_values_available_in_p37=True)
    bad_res = dataclasses.replace(res, metadata=meta_ok, status="blocked_torch_unavailable")
    with pytest.raises(ValueError):
        validate_tensor_materialization_request_result(bad_res)


# 100-101: serializers JSON safe
def test_p38_100_serializers_result_to_json_dict():
    res = run_tensor_materialization_request_probe()
    d = tensor_materialization_request_result_to_json_dict(res)
    assert isinstance(d, dict)
    assert d["contract_version"] == "phase2_p38_tensor_materialization_request_contract_v1"
    assert d["request"]["shape_tuple"] == [2, 32]
    assert d["spec"]["source_nested_row_lengths"] == [32, 32]
    assert "nested_batch_values" not in d
    assert "flat_values" not in d
    assert "nested_batch_values" not in d["request"]

def test_p38_101_serializers_compact_json():
    res = run_tensor_materialization_request_probe()
    js = compact_tensor_materialization_request_json(res)
    assert isinstance(js, str)
    assert ": bool" not in js
    d = json.loads(js)
    assert "nested_batch_values" not in d
    assert "flat_values" not in d
    assert d["status"] in SUPPORTED_FC_VAE_TENSOR_MATERIALIZATION_REQUEST_STATUSES


# 102-104: imports and content safety constraints
def test_p38_102_imports_and_content_safety():
    content = pathlib.Path("src/phase2/fc_vae_tensor_materialization_request.py").read_text(encoding="utf-8")
    for pattern in (
        "import torch", "from torch",
        "import numpy", "from numpy",
        "import random", "from random",
        "import secrets", "from secrets",
        "import array", "from array",
        "import argparse", "from argparse",
        "import subprocess", "from subprocess",
        "import yaml",
    ):
        assert pattern not in content

def test_p38_103_no_torch_tensor_allocation():
    content = pathlib.Path("src/phase2/fc_vae_tensor_materialization_request.py").read_text(encoding="utf-8")
    assert "torch.tensor" not in content
    assert "torch.as_tensor" not in content
    assert "torch.from_numpy" not in content
    assert "def forward(" not in content
    assert "def forward " not in content

def test_p38_104_no_training_methods():
    content = pathlib.Path("src/phase2/fc_vae_tensor_materialization_request.py").read_text(encoding="utf-8")
    for pattern in ("def train", "def loss", "def optimizer", "def checkpoint"):
        assert pattern not in content


# 105: P38 scope gate test
def test_p38_105_scope_gate():
    allowed = {
        "src/phase2/fc_vae_tensor_materialization_request.py",
        "src/phase2/__init__.py",
        "tests/test_phase2_fc_vae_tensor_materialization_request.py",
        "tools/phase2/run_p38_tensor_materialization_request_smoke.py",
        "tests/test_phase2_p38_tensor_materialization_request_smoke.py",
        "reports/PHASE_2_P38_TENSOR_MATERIALIZATION_REQUEST_CONTRACT_NO_TENSOR_ALLOCATION_REPORT.md",
    }
    res = subprocess.run(
        ["git", "diff", "--name-only", "phase2/p37-controlled-nested-batch-values-no-tensor-no-array"],
        capture_output=True, text=True, check=True
    )
    modified = [line.strip() for line in res.stdout.splitlines() if line.strip()]
    for f in modified:
        f_norm = f.replace("\\", "/")
        assert f_norm in allowed, f"Forbidden file modification detected in P38: {f_norm}"
