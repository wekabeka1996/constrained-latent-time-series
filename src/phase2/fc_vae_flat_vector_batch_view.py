# src/phase2/fc_vae_flat_vector_batch_view.py

import dataclasses
import json
from typing import Any, Dict, Tuple

from src.phase2.fc_vae_fake_input_flat_vector import (
    FC_VAE_FAKE_INPUT_FLAT_VECTOR_CONTRACT_VERSION,
    run_fake_input_flat_vector_probe,
)

# Constants
FC_VAE_FLAT_VECTOR_BATCH_VIEW_CONTRACT_VERSION = "phase2_p36_flat_vector_2d_batch_view_contract_v1"

FC_VAE_FLAT_VECTOR_BATCH_VIEW_KIND = "flat_vector_2d_batch_view_metadata_no_nested_values"

FC_VAE_FLAT_VECTOR_BATCH_VIEW_MODULE_NAME = "src.phase2.fc_vae_flat_vector_batch_view"

FC_VAE_FLAT_VECTOR_BATCH_VIEW_STATUS_TORCH_UNAVAILABLE = "blocked_torch_unavailable"

FC_VAE_FLAT_VECTOR_BATCH_VIEW_STATUS_FLAT_VECTOR_BLOCKED = "blocked_by_flat_vector_status"

FC_VAE_FLAT_VECTOR_BATCH_VIEW_STATUS_VIEW_ONLY = "view_metadata_only_no_2d_values_in_p36"

SUPPORTED_FC_VAE_FLAT_VECTOR_BATCH_VIEW_STATUSES = (
    "blocked_torch_unavailable",
    "blocked_by_flat_vector_status",
    "view_metadata_only_no_2d_values_in_p36",
)

FC_VAE_FLAT_VECTOR_BATCH_VIEW_ORDER_KIND = "row_major_flat_index_view"

# Defaults
DEFAULT_P36_VIEW_RANK = 2
DEFAULT_P36_BATCH_SIZE = 2
DEFAULT_P36_INPUT_FLAT_DIM = 32
DEFAULT_P36_FULL_VECTOR_LENGTH = 64
DEFAULT_P36_SHAPE_TUPLE = (2, 32)


# Dataclasses
@dataclasses.dataclass(frozen=True)
class FCVAEFlatVectorBatchViewRequest:
    contract_version: str
    view_kind: str
    architecture_id: str
    source_flat_vector_contract_version: str
    batch_size: int
    input_flat_dim: int
    full_vector_length: int
    view_rank: int
    shape_tuple: Tuple[int, int]
    view_order_kind: str
    allow_nested_values_in_p36: bool
    allow_2d_batch_materialization_in_p36: bool
    allow_array_materialization_in_p36: bool
    allow_tensor_materialization_in_p36: bool
    allow_forward_execution_in_p36: bool
    reason: str


@dataclasses.dataclass(frozen=True)
class FCVAEFlatVectorBatchViewShape:
    contract_version: str
    view_order_kind: str
    view_rank: int
    shape_tuple: Tuple[int, int]
    batch_size: int
    input_flat_dim: int
    full_vector_length: int
    row_major_formula: str
    shape_matches_flat_vector: bool
    view_declared: bool
    nested_values_materialized: bool
    two_d_batch_materialized: bool
    array_materialized: bool
    tensor_materialized: bool
    values_copied_from_flat_vector: bool
    reason: str


@dataclasses.dataclass(frozen=True)
class FCVAEFlatVectorBatchViewMetadata:
    contract_version: str
    source_flat_vector_contract_version: str
    flat_vector_status: str
    torch_available: bool
    flat_vector_available_in_p35: bool
    flat_vector_length: int
    batch_size: int
    input_flat_dim: int
    shape_tuple: Tuple[int, int]
    view_rank: int
    flat_vector_reused_without_copy: bool
    nested_values_attempted: bool
    two_d_batch_materialization_attempted: bool
    array_materialization_attempted: bool
    tensor_materialization_attempted: bool
    forward_execution_attempted: bool
    output_generation_attempted: bool
    reason: str


@dataclasses.dataclass(frozen=True)
class FCVAEFlatVectorBatchViewResult:
    contract_version: str
    request: FCVAEFlatVectorBatchViewRequest
    shape: FCVAEFlatVectorBatchViewShape
    metadata: FCVAEFlatVectorBatchViewMetadata
    status: str
    batch_view_available_in_p36: bool
    nested_values_available_in_p36: bool
    two_d_batch_available_in_p36: bool
    array_materialization_available_in_p36: bool
    tensor_materialization_available_in_p36: bool
    forward_execution_available_in_p36: bool
    output_generation_available_in_p36: bool
    no_nested_values: bool
    no_2d_batch_materialized: bool
    no_array_created: bool
    no_tensor_created: bool
    no_forward_execution: bool
    no_output_generation: bool
    no_training_loop: bool
    no_optimizer: bool
    no_checkpointing: bool
    no_artifact_generation: bool
    no_final_comparison: bool
    no_scientific_conclusion: bool
    reason: str


# Validators
def validate_non_empty_str(value: Any, name: str) -> None:
    if type(value) is not str:
        raise TypeError(f"{name} must be exact str, got {type(value).__name__}")
    if not value:
        raise ValueError(f"{name} cannot be empty")
    if value != value.strip():
        raise ValueError(f"{name} cannot have leading/trailing whitespace")


def validate_bool(value: Any, name: str) -> None:
    if type(value) is not bool:
        raise TypeError(f"{name} must be exact bool, got {type(value).__name__}")


def validate_non_negative_int(value: Any, name: str) -> None:
    if type(value) is not int or isinstance(value, bool):
        raise TypeError(f"{name} must be exact int, got {type(value).__name__}")
    if value < 0:
        raise ValueError(f"{name} must be >= 0")


def validate_positive_int(value: Any, name: str) -> None:
    if type(value) is not int or isinstance(value, bool):
        raise TypeError(f"{name} must be exact int, got {type(value).__name__}")
    if value <= 0:
        raise ValueError(f"{name} must be > 0")


def validate_shape_tuple(value: Any) -> None:
    if type(value) is not tuple:
        raise TypeError(f"shape_tuple must be exact tuple, got {type(value).__name__}")
    if len(value) != 2:
        raise ValueError(f"shape_tuple must have length 2, got {len(value)}")
    for i, item in enumerate(value):
        if type(item) is not int or isinstance(item, bool):
            raise TypeError(f"shape_tuple[{i}] must be exact int, got {type(item).__name__}")
        if item <= 0:
            raise ValueError(f"shape_tuple[{i}] must be > 0")


def validate_full_vector_length(value: Any) -> None:
    if type(value) is not int or isinstance(value, bool):
        raise TypeError(f"full_vector_length must be exact int, got {type(value).__name__}")
    if value <= 0:
        raise ValueError("full_vector_length must be > 0")


def validate_view_rank(value: Any) -> None:
    if type(value) is not int or isinstance(value, bool):
        raise TypeError(f"view_rank must be exact int, got {type(value).__name__}")
    if value != 2:
        raise ValueError(f"view_rank must be exactly 2, got {value}")


def validate_row_major_formula(value: Any) -> None:
    expected = "flat_index = batch_index * input_flat_dim + feature_index"
    if type(value) is not str:
        raise TypeError(f"row_major_formula must be exact str, got {type(value).__name__}")
    if value != expected:
        raise ValueError(f"Invalid row_major_formula: '{value}'")


def validate_flat_vector_batch_view_status(value: Any) -> None:
    if type(value) is not str:
        raise TypeError(f"status must be exact str, got {type(value).__name__}")
    if value not in SUPPORTED_FC_VAE_FLAT_VECTOR_BATCH_VIEW_STATUSES:
        raise ValueError(f"Unsupported batch view status: {value}")


def assert_no_local_path_leakage(data: Any) -> None:
    if isinstance(data, str):
        bad_patterns = ("file:///", "C:/", "C:\\", "/home/", "/Users/")
        for pat in bad_patterns:
            if pat in data or pat.lower() in data.lower():
                raise ValueError(f"Local path leakage detected: '{data}' contains '{pat}'")
    elif isinstance(data, dict):
        for k, v in data.items():
            assert_no_local_path_leakage(k)
            assert_no_local_path_leakage(v)
    elif isinstance(data, (list, tuple)):
        for item in data:
            assert_no_local_path_leakage(item)


def assert_no_forbidden_claims(data: Any) -> None:
    if isinstance(data, str):
        normalized = data.lower()
        cleaned = (normalized
                   .replace("no_final_comparison", "")
                   .replace("no_scientific_conclusion", "")
                   .replace("view_metadata_only_no_2d_values_in_p36", ""))
        forbidden = ("model works", "scientific success", "solved", "best", "winner", "production ready", "state of the art")
        for word in forbidden:
            if word in cleaned:
                raise ValueError(f"Forbidden success claim '{word}' found in text: '{data}'")
    elif isinstance(data, dict):
        for k, v in data.items():
            assert_no_forbidden_claims(k)
            assert_no_forbidden_claims(v)
    elif isinstance(data, (list, tuple)):
        for item in data:
            assert_no_forbidden_claims(item)


def validate_flat_vector_batch_view_request(request: FCVAEFlatVectorBatchViewRequest) -> None:
    if type(request) is not FCVAEFlatVectorBatchViewRequest:
        raise TypeError("request must be exact FCVAEFlatVectorBatchViewRequest instance")
    validate_non_empty_str(request.contract_version, "contract_version")
    if request.contract_version != FC_VAE_FLAT_VECTOR_BATCH_VIEW_CONTRACT_VERSION:
        raise ValueError(f"Invalid contract_version: {request.contract_version}")
    validate_non_empty_str(request.view_kind, "view_kind")
    if request.view_kind != FC_VAE_FLAT_VECTOR_BATCH_VIEW_KIND:
        raise ValueError(f"Invalid view_kind: {request.view_kind}")
    validate_non_empty_str(request.architecture_id, "architecture_id")
    if request.architecture_id != "FC-VAE":
        raise ValueError(f"Invalid architecture_id: {request.architecture_id}")
    validate_non_empty_str(request.source_flat_vector_contract_version, "source_flat_vector_contract_version")
    if request.source_flat_vector_contract_version != FC_VAE_FAKE_INPUT_FLAT_VECTOR_CONTRACT_VERSION:
        raise ValueError(f"Invalid source_flat_vector_contract_version: {request.source_flat_vector_contract_version}")
    validate_positive_int(request.batch_size, "batch_size")
    if request.batch_size != 2:
        raise ValueError("batch_size must be 2")
    validate_positive_int(request.input_flat_dim, "input_flat_dim")
    if request.input_flat_dim != 32:
        raise ValueError("input_flat_dim must be 32")
    validate_full_vector_length(request.full_vector_length)
    if request.full_vector_length != 64:
        raise ValueError("full_vector_length must be 64")
    if request.full_vector_length != request.batch_size * request.input_flat_dim:
        raise ValueError("full_vector_length must equal batch_size * input_flat_dim")
    validate_view_rank(request.view_rank)
    validate_shape_tuple(request.shape_tuple)
    if request.shape_tuple != (2, 32):
        raise ValueError("shape_tuple must be (2, 32)")
    validate_non_empty_str(request.view_order_kind, "view_order_kind")
    if request.view_order_kind != FC_VAE_FLAT_VECTOR_BATCH_VIEW_ORDER_KIND:
        raise ValueError(f"Invalid view_order_kind: {request.view_order_kind}")

    for flag_name in (
        "allow_nested_values_in_p36",
        "allow_2d_batch_materialization_in_p36",
        "allow_array_materialization_in_p36",
        "allow_tensor_materialization_in_p36",
        "allow_forward_execution_in_p36",
    ):
        val = getattr(request, flag_name)
        validate_bool(val, flag_name)
        if val:
            raise ValueError(f"{flag_name} must be False")

    validate_non_empty_str(request.reason, "reason")
    assert_no_local_path_leakage(request.reason)
    assert_no_forbidden_claims(request.reason)


def validate_flat_vector_batch_view_shape(shape: FCVAEFlatVectorBatchViewShape) -> None:
    if type(shape) is not FCVAEFlatVectorBatchViewShape:
        raise TypeError("shape must be exact FCVAEFlatVectorBatchViewShape instance")
    validate_non_empty_str(shape.contract_version, "contract_version")
    if shape.contract_version != FC_VAE_FLAT_VECTOR_BATCH_VIEW_CONTRACT_VERSION:
        raise ValueError(f"Invalid contract_version: {shape.contract_version}")
    validate_non_empty_str(shape.view_order_kind, "view_order_kind")
    if shape.view_order_kind != FC_VAE_FLAT_VECTOR_BATCH_VIEW_ORDER_KIND:
        raise ValueError(f"Invalid view_order_kind: {shape.view_order_kind}")
    validate_view_rank(shape.view_rank)
    validate_shape_tuple(shape.shape_tuple)
    if shape.shape_tuple != (2, 32):
        raise ValueError("shape_tuple must be (2, 32)")
    validate_positive_int(shape.batch_size, "batch_size")
    if shape.batch_size != 2:
        raise ValueError("batch_size must be 2")
    validate_positive_int(shape.input_flat_dim, "input_flat_dim")
    if shape.input_flat_dim != 32:
        raise ValueError("input_flat_dim must be 32")
    validate_full_vector_length(shape.full_vector_length)
    if shape.full_vector_length != 64:
        raise ValueError("full_vector_length must be 64")
    if shape.full_vector_length != shape.batch_size * shape.input_flat_dim:
        raise ValueError("full_vector_length must equal batch_size * input_flat_dim")
    validate_row_major_formula(shape.row_major_formula)
    validate_bool(shape.shape_matches_flat_vector, "shape_matches_flat_vector")
    if not shape.shape_matches_flat_vector:
        raise ValueError("shape_matches_flat_vector must be True")
    validate_bool(shape.view_declared, "view_declared")
    if not shape.view_declared:
        raise ValueError("view_declared must be True")

    for flag_name in (
        "nested_values_materialized",
        "two_d_batch_materialized",
        "array_materialized",
        "tensor_materialized",
        "values_copied_from_flat_vector",
    ):
        val = getattr(shape, flag_name)
        validate_bool(val, flag_name)
        if val:
            raise ValueError(f"{flag_name} must be False")

    validate_non_empty_str(shape.reason, "reason")
    assert_no_local_path_leakage(shape.reason)
    assert_no_forbidden_claims(shape.reason)


def validate_flat_vector_batch_view_metadata(metadata: FCVAEFlatVectorBatchViewMetadata) -> None:
    if type(metadata) is not FCVAEFlatVectorBatchViewMetadata:
        raise TypeError("metadata must be exact FCVAEFlatVectorBatchViewMetadata instance")
    validate_non_empty_str(metadata.contract_version, "contract_version")
    if metadata.contract_version != FC_VAE_FLAT_VECTOR_BATCH_VIEW_CONTRACT_VERSION:
        raise ValueError(f"Invalid contract_version: {metadata.contract_version}")
    validate_non_empty_str(metadata.source_flat_vector_contract_version, "source_flat_vector_contract_version")
    if metadata.source_flat_vector_contract_version != FC_VAE_FAKE_INPUT_FLAT_VECTOR_CONTRACT_VERSION:
        raise ValueError(f"Invalid source_flat_vector_contract_version: {metadata.source_flat_vector_contract_version}")
    validate_non_empty_str(metadata.flat_vector_status, "flat_vector_status")
    validate_bool(metadata.torch_available, "torch_available")
    validate_bool(metadata.flat_vector_available_in_p35, "flat_vector_available_in_p35")
    if not metadata.flat_vector_available_in_p35:
        raise ValueError("flat_vector_available_in_p35 must be True")
    validate_full_vector_length(metadata.flat_vector_length)
    if metadata.flat_vector_length != 64:
        raise ValueError("flat_vector_length must be 64")
    validate_positive_int(metadata.batch_size, "batch_size")
    if metadata.batch_size != 2:
        raise ValueError("batch_size must be 2")
    validate_positive_int(metadata.input_flat_dim, "input_flat_dim")
    if metadata.input_flat_dim != 32:
        raise ValueError("input_flat_dim must be 32")
    validate_shape_tuple(metadata.shape_tuple)
    if metadata.shape_tuple != (2, 32):
        raise ValueError("shape_tuple must be (2, 32)")
    validate_view_rank(metadata.view_rank)
    validate_bool(metadata.flat_vector_reused_without_copy, "flat_vector_reused_without_copy")
    if not metadata.flat_vector_reused_without_copy:
        raise ValueError("flat_vector_reused_without_copy must be True")

    for flag_name in (
        "nested_values_attempted",
        "two_d_batch_materialization_attempted",
        "array_materialization_attempted",
        "tensor_materialization_attempted",
        "forward_execution_attempted",
        "output_generation_attempted",
    ):
        val = getattr(metadata, flag_name)
        validate_bool(val, flag_name)
        if val:
            raise ValueError(f"{flag_name} must be False")

    validate_non_empty_str(metadata.reason, "reason")
    assert_no_local_path_leakage(metadata.reason)
    assert_no_forbidden_claims(metadata.reason)


def validate_flat_vector_batch_view_result(result: FCVAEFlatVectorBatchViewResult) -> None:
    if type(result) is not FCVAEFlatVectorBatchViewResult:
        raise TypeError("result must be exact FCVAEFlatVectorBatchViewResult instance")
    validate_non_empty_str(result.contract_version, "contract_version")
    if result.contract_version != FC_VAE_FLAT_VECTOR_BATCH_VIEW_CONTRACT_VERSION:
        raise ValueError(f"Invalid contract_version: {result.contract_version}")
    validate_flat_vector_batch_view_request(result.request)
    validate_flat_vector_batch_view_shape(result.shape)
    validate_flat_vector_batch_view_metadata(result.metadata)
    validate_flat_vector_batch_view_status(result.status)

    validate_bool(result.batch_view_available_in_p36, "batch_view_available_in_p36")
    if not result.batch_view_available_in_p36:
        raise ValueError("batch_view_available_in_p36 must be True")

    for flag_name in (
        "nested_values_available_in_p36",
        "two_d_batch_available_in_p36",
        "array_materialization_available_in_p36",
        "tensor_materialization_available_in_p36",
        "forward_execution_available_in_p36",
        "output_generation_available_in_p36",
    ):
        val = getattr(result, flag_name)
        validate_bool(val, flag_name)
        if val:
            raise ValueError(f"{flag_name} must be False")

    for flag_name in (
        "no_nested_values",
        "no_2d_batch_materialized",
        "no_array_created",
        "no_tensor_created",
        "no_forward_execution",
        "no_output_generation",
        "no_training_loop",
        "no_optimizer",
        "no_checkpointing",
        "no_artifact_generation",
        "no_final_comparison",
        "no_scientific_conclusion",
    ):
        val = getattr(result, flag_name)
        validate_bool(val, flag_name)
        if not val:
            raise ValueError(f"{flag_name} must be True")

    if not result.metadata.torch_available:
        if result.status != FC_VAE_FLAT_VECTOR_BATCH_VIEW_STATUS_TORCH_UNAVAILABLE:
            raise ValueError(f"Inconsistent state: status must be '{FC_VAE_FLAT_VECTOR_BATCH_VIEW_STATUS_TORCH_UNAVAILABLE}' when torch is unavailable")
    elif not result.metadata.flat_vector_available_in_p35:
        if result.status != FC_VAE_FLAT_VECTOR_BATCH_VIEW_STATUS_FLAT_VECTOR_BLOCKED:
            raise ValueError(f"Inconsistent state: status must be '{FC_VAE_FLAT_VECTOR_BATCH_VIEW_STATUS_FLAT_VECTOR_BLOCKED}'")
    else:
        if result.status != FC_VAE_FLAT_VECTOR_BATCH_VIEW_STATUS_VIEW_ONLY:
            raise ValueError(f"Inconsistent state: status must be '{FC_VAE_FLAT_VECTOR_BATCH_VIEW_STATUS_VIEW_ONLY}'")

    validate_non_empty_str(result.reason, "reason")
    assert_no_local_path_leakage(result.reason)
    assert_no_forbidden_claims(result.reason)


# Builders
def compute_row_major_flat_index(batch_index: int, feature_index: int, input_flat_dim: int) -> int:
    if type(batch_index) is not int or isinstance(batch_index, bool):
        raise TypeError(f"batch_index must be exact int, got {type(batch_index).__name__}")
    if type(feature_index) is not int or isinstance(feature_index, bool):
        raise TypeError(f"feature_index must be exact int, got {type(feature_index).__name__}")
    if type(input_flat_dim) is not int or isinstance(input_flat_dim, bool):
        raise TypeError(f"input_flat_dim must be exact int, got {type(input_flat_dim).__name__}")

    if batch_index < 0:
        raise ValueError(f"batch_index must be >= 0, got {batch_index}")
    if feature_index < 0:
        raise ValueError(f"feature_index must be >= 0, got {feature_index}")
    if input_flat_dim <= 0:
        raise ValueError(f"input_flat_dim must be > 0, got {input_flat_dim}")

    return batch_index * input_flat_dim + feature_index


def build_flat_vector_batch_view_request_from_p35_default() -> FCVAEFlatVectorBatchViewRequest:
    p35_result = run_fake_input_flat_vector_probe()

    req = FCVAEFlatVectorBatchViewRequest(
        contract_version=FC_VAE_FLAT_VECTOR_BATCH_VIEW_CONTRACT_VERSION,
        view_kind=FC_VAE_FLAT_VECTOR_BATCH_VIEW_KIND,
        architecture_id="FC-VAE",
        source_flat_vector_contract_version=p35_result.contract_version,
        batch_size=p35_result.request.batch_size,
        input_flat_dim=p35_result.request.input_flat_dim,
        full_vector_length=p35_result.request.full_vector_length,
        view_rank=DEFAULT_P36_VIEW_RANK,
        shape_tuple=DEFAULT_P36_SHAPE_TUPLE,
        view_order_kind=FC_VAE_FLAT_VECTOR_BATCH_VIEW_ORDER_KIND,
        allow_nested_values_in_p36=False,
        allow_2d_batch_materialization_in_p36=False,
        allow_array_materialization_in_p36=False,
        allow_tensor_materialization_in_p36=False,
        allow_forward_execution_in_p36=False,
        reason="p36_flat_vector_batch_view_request_from_p35_default",
    )
    validate_flat_vector_batch_view_request(req)
    return req


def build_flat_vector_batch_view_shape(
    request: FCVAEFlatVectorBatchViewRequest,
) -> FCVAEFlatVectorBatchViewShape:
    validate_flat_vector_batch_view_request(request)

    shape = FCVAEFlatVectorBatchViewShape(
        contract_version=FC_VAE_FLAT_VECTOR_BATCH_VIEW_CONTRACT_VERSION,
        view_order_kind=request.view_order_kind,
        view_rank=request.view_rank,
        shape_tuple=request.shape_tuple,
        batch_size=request.batch_size,
        input_flat_dim=request.input_flat_dim,
        full_vector_length=request.full_vector_length,
        row_major_formula="flat_index = batch_index * input_flat_dim + feature_index",
        shape_matches_flat_vector=True,
        view_declared=True,
        nested_values_materialized=False,
        two_d_batch_materialized=False,
        array_materialized=False,
        tensor_materialized=False,
        values_copied_from_flat_vector=False,
        reason="p36_flat_vector_batch_view_shape",
    )
    validate_flat_vector_batch_view_shape(shape)
    return shape


def build_flat_vector_batch_view_metadata(
    request: FCVAEFlatVectorBatchViewRequest,
) -> FCVAEFlatVectorBatchViewMetadata:
    validate_flat_vector_batch_view_request(request)
    p35_result = run_fake_input_flat_vector_probe()

    meta = FCVAEFlatVectorBatchViewMetadata(
        contract_version=FC_VAE_FLAT_VECTOR_BATCH_VIEW_CONTRACT_VERSION,
        source_flat_vector_contract_version=request.source_flat_vector_contract_version,
        flat_vector_status=p35_result.status,
        torch_available=p35_result.metadata.torch_available,
        flat_vector_available_in_p35=p35_result.flat_vector_available_in_p35,
        flat_vector_length=p35_result.metadata.full_vector_length,
        batch_size=request.batch_size,
        input_flat_dim=request.input_flat_dim,
        shape_tuple=request.shape_tuple,
        view_rank=request.view_rank,
        flat_vector_reused_without_copy=True,
        nested_values_attempted=False,
        two_d_batch_materialization_attempted=False,
        array_materialization_attempted=False,
        tensor_materialization_attempted=False,
        forward_execution_attempted=False,
        output_generation_attempted=False,
        reason="p36_flat_vector_batch_view_metadata",
    )
    validate_flat_vector_batch_view_metadata(meta)
    return meta


def build_flat_vector_batch_view_result(
    request: FCVAEFlatVectorBatchViewRequest,
) -> FCVAEFlatVectorBatchViewResult:
    validate_flat_vector_batch_view_request(request)
    shape = build_flat_vector_batch_view_shape(request)
    metadata = build_flat_vector_batch_view_metadata(request)

    if not metadata.torch_available:
        status = FC_VAE_FLAT_VECTOR_BATCH_VIEW_STATUS_TORCH_UNAVAILABLE
        reason = "batch_view_blocked_torch_unavailable"
    elif not metadata.flat_vector_available_in_p35:
        status = FC_VAE_FLAT_VECTOR_BATCH_VIEW_STATUS_FLAT_VECTOR_BLOCKED
        reason = "batch_view_blocked_by_flat_vector_status"
    else:
        status = FC_VAE_FLAT_VECTOR_BATCH_VIEW_STATUS_VIEW_ONLY
        reason = "batch_view_metadata_only"

    res = FCVAEFlatVectorBatchViewResult(
        contract_version=FC_VAE_FLAT_VECTOR_BATCH_VIEW_CONTRACT_VERSION,
        request=request,
        shape=shape,
        metadata=metadata,
        status=status,
        batch_view_available_in_p36=True,
        nested_values_available_in_p36=False,
        two_d_batch_available_in_p36=False,
        array_materialization_available_in_p36=False,
        tensor_materialization_available_in_p36=False,
        forward_execution_available_in_p36=False,
        output_generation_available_in_p36=False,
        no_nested_values=True,
        no_2d_batch_materialized=True,
        no_array_created=True,
        no_tensor_created=True,
        no_forward_execution=True,
        no_output_generation=True,
        no_training_loop=True,
        no_optimizer=True,
        no_checkpointing=True,
        no_artifact_generation=True,
        no_final_comparison=True,
        no_scientific_conclusion=True,
        reason=reason,
    )
    validate_flat_vector_batch_view_result(res)
    return res


def run_flat_vector_batch_view_probe() -> FCVAEFlatVectorBatchViewResult:
    req = build_flat_vector_batch_view_request_from_p35_default()
    res = build_flat_vector_batch_view_result(req)
    validate_flat_vector_batch_view_result(res)
    return res


# Serialization
def flat_vector_batch_view_request_to_json_dict(
    request: FCVAEFlatVectorBatchViewRequest,
) -> Dict[str, Any]:
    validate_flat_vector_batch_view_request(request)
    d = {
        "contract_version": request.contract_version,
        "view_kind": request.view_kind,
        "architecture_id": request.architecture_id,
        "source_flat_vector_contract_version": request.source_flat_vector_contract_version,
        "batch_size": request.batch_size,
        "input_flat_dim": request.input_flat_dim,
        "full_vector_length": request.full_vector_length,
        "view_rank": request.view_rank,
        "shape_tuple": list(request.shape_tuple),
        "view_order_kind": request.view_order_kind,
        "allow_nested_values_in_p36": request.allow_nested_values_in_p36,
        "allow_2d_batch_materialization_in_p36": request.allow_2d_batch_materialization_in_p36,
        "allow_array_materialization_in_p36": request.allow_array_materialization_in_p36,
        "allow_tensor_materialization_in_p36": request.allow_tensor_materialization_in_p36,
        "allow_forward_execution_in_p36": request.allow_forward_execution_in_p36,
        "reason": request.reason,
    }
    assert_no_local_path_leakage(d)
    assert_no_forbidden_claims(d)
    return d


def flat_vector_batch_view_shape_to_json_dict(
    shape: FCVAEFlatVectorBatchViewShape,
) -> Dict[str, Any]:
    validate_flat_vector_batch_view_shape(shape)
    d = {
        "contract_version": shape.contract_version,
        "view_order_kind": shape.view_order_kind,
        "view_rank": shape.view_rank,
        "shape_tuple": list(shape.shape_tuple),
        "batch_size": shape.batch_size,
        "input_flat_dim": shape.input_flat_dim,
        "full_vector_length": shape.full_vector_length,
        "row_major_formula": shape.row_major_formula,
        "shape_matches_flat_vector": shape.shape_matches_flat_vector,
        "view_declared": shape.view_declared,
        "nested_values_materialized": shape.nested_values_materialized,
        "two_d_batch_materialized": shape.two_d_batch_materialized,
        "array_materialized": shape.array_materialized,
        "tensor_materialized": shape.tensor_materialized,
        "values_copied_from_flat_vector": shape.values_copied_from_flat_vector,
        "reason": shape.reason,
    }
    assert_no_local_path_leakage(d)
    assert_no_forbidden_claims(d)
    return d


def flat_vector_batch_view_metadata_to_json_dict(
    metadata: FCVAEFlatVectorBatchViewMetadata,
) -> Dict[str, Any]:
    validate_flat_vector_batch_view_metadata(metadata)
    d = {
        "contract_version": metadata.contract_version,
        "source_flat_vector_contract_version": metadata.source_flat_vector_contract_version,
        "flat_vector_status": metadata.flat_vector_status,
        "torch_available": metadata.torch_available,
        "flat_vector_available_in_p35": metadata.flat_vector_available_in_p35,
        "flat_vector_length": metadata.flat_vector_length,
        "batch_size": metadata.batch_size,
        "input_flat_dim": metadata.input_flat_dim,
        "shape_tuple": list(metadata.shape_tuple),
        "view_rank": metadata.view_rank,
        "flat_vector_reused_without_copy": metadata.flat_vector_reused_without_copy,
        "nested_values_attempted": metadata.nested_values_attempted,
        "two_d_batch_materialization_attempted": metadata.two_d_batch_materialization_attempted,
        "array_materialization_attempted": metadata.array_materialization_attempted,
        "tensor_materialization_attempted": metadata.tensor_materialization_attempted,
        "forward_execution_attempted": metadata.forward_execution_attempted,
        "output_generation_attempted": metadata.output_generation_attempted,
        "reason": metadata.reason,
    }
    assert_no_local_path_leakage(d)
    assert_no_forbidden_claims(d)
    return d


def flat_vector_batch_view_result_to_json_dict(
    result: FCVAEFlatVectorBatchViewResult,
) -> Dict[str, Any]:
    validate_flat_vector_batch_view_result(result)
    d = {
        "contract_version": result.contract_version,
        "request": flat_vector_batch_view_request_to_json_dict(result.request),
        "shape": flat_vector_batch_view_shape_to_json_dict(result.shape),
        "metadata": flat_vector_batch_view_metadata_to_json_dict(result.metadata),
        "status": result.status,
        "batch_view_available_in_p36": result.batch_view_available_in_p36,
        "nested_values_available_in_p36": result.nested_values_available_in_p36,
        "two_d_batch_available_in_p36": result.two_d_batch_available_in_p36,
        "array_materialization_available_in_p36": result.array_materialization_available_in_p36,
        "tensor_materialization_available_in_p36": result.tensor_materialization_available_in_p36,
        "forward_execution_available_in_p36": result.forward_execution_available_in_p36,
        "output_generation_available_in_p36": result.output_generation_available_in_p36,
        "no_nested_values": result.no_nested_values,
        "no_2d_batch_materialized": result.no_2d_batch_materialized,
        "no_array_created": result.no_array_created,
        "no_tensor_created": result.no_tensor_created,
        "no_forward_execution": result.no_forward_execution,
        "no_output_generation": result.no_output_generation,
        "no_training_loop": result.no_training_loop,
        "no_optimizer": result.no_optimizer,
        "no_checkpointing": result.no_checkpointing,
        "no_artifact_generation": result.no_artifact_generation,
        "no_final_comparison": result.no_final_comparison,
        "no_scientific_conclusion": result.no_scientific_conclusion,
        "reason": result.reason,
    }
    assert_no_local_path_leakage(d)
    assert_no_forbidden_claims(d)
    return d


def compact_flat_vector_batch_view_json(result: FCVAEFlatVectorBatchViewResult) -> str:
    d = flat_vector_batch_view_result_to_json_dict(result)
    return json.dumps(d, sort_keys=True, separators=(",", ":"))
