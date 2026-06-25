# src/phase2/fc_vae_nested_batch_values.py

import dataclasses
import json
from typing import Any, Dict, Tuple

from src.phase2.fc_vae_fake_input_flat_vector import (
    FC_VAE_FAKE_INPUT_FLAT_VECTOR_CONTRACT_VERSION,
    run_fake_input_flat_vector_probe,
)
from src.phase2.fc_vae_flat_vector_batch_view import (
    FC_VAE_FLAT_VECTOR_BATCH_VIEW_CONTRACT_VERSION,
    run_flat_vector_batch_view_probe,
)

# Constants
FC_VAE_NESTED_BATCH_VALUES_CONTRACT_VERSION = "phase2_p37_nested_batch_values_contract_v1"

FC_VAE_NESTED_BATCH_VALUES_KIND = "controlled_nested_python_tuple_batch_values_no_tensor_no_array"

FC_VAE_NESTED_BATCH_VALUES_MODULE_NAME = "src.phase2.fc_vae_nested_batch_values"

FC_VAE_NESTED_BATCH_VALUES_STATUS_TORCH_UNAVAILABLE = "blocked_torch_unavailable"

FC_VAE_NESTED_BATCH_VALUES_STATUS_BATCH_VIEW_BLOCKED = "blocked_by_batch_view_status"

FC_VAE_NESTED_BATCH_VALUES_STATUS_NESTED_VALUES_ONLY = "nested_values_only_no_tensor_no_forward_in_p37"

SUPPORTED_FC_VAE_NESTED_BATCH_VALUES_STATUSES = (
    "blocked_torch_unavailable",
    "blocked_by_batch_view_status",
    "nested_values_only_no_tensor_no_forward_in_p37",
)

FC_VAE_NESTED_BATCH_VALUES_ORDER_KIND = "row_major_flat_to_nested_tuple"

# Defaults
DEFAULT_P37_BATCH_SIZE = 2
DEFAULT_P37_INPUT_FLAT_DIM = 32
DEFAULT_P37_FULL_VECTOR_LENGTH = 64
DEFAULT_P37_SHAPE_TUPLE = (2, 32)


# Dataclasses
@dataclasses.dataclass(frozen=True)
class FCVAENestedBatchValuesRequest:
    contract_version: str
    nested_values_kind: str
    architecture_id: str
    source_flat_vector_contract_version: str
    source_batch_view_contract_version: str
    batch_size: int
    input_flat_dim: int
    full_vector_length: int
    shape_tuple: Tuple[int, int]
    order_kind: str
    allow_nested_values_in_p37: bool
    allow_array_materialization_in_p37: bool
    allow_tensor_materialization_in_p37: bool
    allow_forward_execution_in_p37: bool
    reason: str


@dataclasses.dataclass(frozen=True)
class FCVAENestedBatchValues:
    contract_version: str
    nested_values_kind: str
    order_kind: str
    shape_tuple: Tuple[int, int]
    batch_size: int
    input_flat_dim: int
    full_vector_length: int
    nested_batch_values: Tuple[Tuple[float, ...], ...]
    row_count: int
    row_lengths: Tuple[int, ...]
    total_scalar_count: int
    flattened_matches_source_flat_vector: bool
    all_values_are_exact_float: bool
    all_values_within_source_range: bool
    array_materialized: bool
    tensor_materialized: bool
    forward_executed: bool
    output_generated: bool
    reason: str


@dataclasses.dataclass(frozen=True)
class FCVAENestedBatchValuesMetadata:
    contract_version: str
    source_flat_vector_contract_version: str
    source_batch_view_contract_version: str
    flat_vector_status: str
    batch_view_status: str
    torch_available: bool
    flat_vector_available_in_p35: bool
    batch_view_available_in_p36: bool
    flat_vector_length: int
    shape_tuple: Tuple[int, int]
    batch_size: int
    input_flat_dim: int
    nested_values_materialized: bool
    array_materialization_attempted: bool
    tensor_materialization_attempted: bool
    forward_execution_attempted: bool
    output_generation_attempted: bool
    reason: str


@dataclasses.dataclass(frozen=True)
class FCVAENestedBatchValuesResult:
    contract_version: str
    request: FCVAENestedBatchValuesRequest
    nested_values: FCVAENestedBatchValues
    metadata: FCVAENestedBatchValuesMetadata
    status: str
    nested_values_available_in_p37: bool
    array_materialization_available_in_p37: bool
    tensor_materialization_available_in_p37: bool
    forward_execution_available_in_p37: bool
    output_generation_available_in_p37: bool
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
    if value != (2, 32):
        raise ValueError(f"shape_tuple must be exactly (2, 32), got {value}")


def validate_full_vector_length(value: Any) -> None:
    if type(value) is not int or isinstance(value, bool):
        raise TypeError(f"full_vector_length must be exact int, got {type(value).__name__}")
    if value != 64:
        raise ValueError(f"full_vector_length must be exactly 64, got {value}")


def validate_flat_values_tuple(value: Any) -> None:
    if type(value) is not tuple:
        raise TypeError("flat_values must be exact tuple")
    if len(value) != 64:
        raise ValueError(f"flat_values length must be exactly 64, got {len(value)}")
    for i, x in enumerate(value):
        if type(x) is not float:
            raise TypeError(f"flat_values[{i}] must be exact float, got {type(x).__name__}")


def validate_nested_batch_values_tuple(value: Any) -> None:
    if type(value) is not tuple:
        raise TypeError("nested_batch_values must be exact tuple")
    if len(value) != 2:
        raise ValueError(f"nested_batch_values must have length 2, got {len(value)}")
    for i, row in enumerate(value):
        if type(row) is not tuple:
            raise TypeError(f"nested_batch_values[{i}] must be exact tuple, got {type(row).__name__}")
        if len(row) != 32:
            raise ValueError(f"nested_batch_values[{i}] length must be exactly 32, got {len(row)}")
        for j, val in enumerate(row):
            if type(val) is not float:
                raise TypeError(f"nested_batch_values[{i}][{j}] must be exact float, got {type(val).__name__}")


def validate_nested_batch_values_status(value: Any) -> None:
    if type(value) is not str:
        raise TypeError(f"status must be exact str, got {type(value).__name__}")
    if value not in SUPPORTED_FC_VAE_NESTED_BATCH_VALUES_STATUSES:
        raise ValueError(f"Unsupported status: {value}")


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
                   .replace("nested_values_only_no_tensor_no_forward_in_p37", ""))
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


def validate_nested_batch_values_request(request: FCVAENestedBatchValuesRequest) -> None:
    if type(request) is not FCVAENestedBatchValuesRequest:
        raise TypeError("request must be exact FCVAENestedBatchValuesRequest instance")
    validate_non_empty_str(request.contract_version, "contract_version")
    if request.contract_version != FC_VAE_NESTED_BATCH_VALUES_CONTRACT_VERSION:
        raise ValueError(f"Invalid contract_version: {request.contract_version}")
    validate_non_empty_str(request.nested_values_kind, "nested_values_kind")
    if request.nested_values_kind != FC_VAE_NESTED_BATCH_VALUES_KIND:
        raise ValueError(f"Invalid nested_values_kind: {request.nested_values_kind}")
    validate_non_empty_str(request.architecture_id, "architecture_id")
    if request.architecture_id != "FC-VAE":
        raise ValueError(f"Invalid architecture_id: {request.architecture_id}")
    validate_non_empty_str(request.source_flat_vector_contract_version, "source_flat_vector_contract_version")
    if request.source_flat_vector_contract_version != FC_VAE_FAKE_INPUT_FLAT_VECTOR_CONTRACT_VERSION:
        raise ValueError(f"Invalid source_flat_vector_contract_version: {request.source_flat_vector_contract_version}")
    validate_non_empty_str(request.source_batch_view_contract_version, "source_batch_view_contract_version")
    if request.source_batch_view_contract_version != FC_VAE_FLAT_VECTOR_BATCH_VIEW_CONTRACT_VERSION:
        raise ValueError(f"Invalid source_batch_view_contract_version: {request.source_batch_view_contract_version}")
    validate_positive_int(request.batch_size, "batch_size")
    if request.batch_size != 2:
        raise ValueError("batch_size must be 2")
    validate_positive_int(request.input_flat_dim, "input_flat_dim")
    if request.input_flat_dim != 32:
        raise ValueError("input_flat_dim must be 32")
    validate_full_vector_length(request.full_vector_length)
    validate_shape_tuple(request.shape_tuple)
    validate_non_empty_str(request.order_kind, "order_kind")
    if request.order_kind != FC_VAE_NESTED_BATCH_VALUES_ORDER_KIND:
        raise ValueError(f"Invalid order_kind: {request.order_kind}")

    validate_bool(request.allow_nested_values_in_p37, "allow_nested_values_in_p37")
    if not request.allow_nested_values_in_p37:
        raise ValueError("allow_nested_values_in_p37 must be True")

    for flag_name in (
        "allow_array_materialization_in_p37",
        "allow_tensor_materialization_in_p37",
        "allow_forward_execution_in_p37",
    ):
        val = getattr(request, flag_name)
        validate_bool(val, flag_name)
        if val:
            raise ValueError(f"{flag_name} must be False")

    validate_non_empty_str(request.reason, "reason")
    assert_no_local_path_leakage(request.reason)
    assert_no_forbidden_claims(request.reason)


def validate_nested_batch_values(nested: FCVAENestedBatchValues) -> None:
    if type(nested) is not FCVAENestedBatchValues:
        raise TypeError("nested must be exact FCVAENestedBatchValues instance")
    validate_non_empty_str(nested.contract_version, "contract_version")
    if nested.contract_version != FC_VAE_NESTED_BATCH_VALUES_CONTRACT_VERSION:
        raise ValueError(f"Invalid contract_version: {nested.contract_version}")
    validate_non_empty_str(nested.nested_values_kind, "nested_values_kind")
    if nested.nested_values_kind != FC_VAE_NESTED_BATCH_VALUES_KIND:
        raise ValueError(f"Invalid nested_values_kind: {nested.nested_values_kind}")
    validate_non_empty_str(nested.order_kind, "order_kind")
    if nested.order_kind != FC_VAE_NESTED_BATCH_VALUES_ORDER_KIND:
        raise ValueError(f"Invalid order_kind: {nested.order_kind}")
    validate_shape_tuple(nested.shape_tuple)
    validate_positive_int(nested.batch_size, "batch_size")
    if nested.batch_size != 2:
        raise ValueError("batch_size must be 2")
    validate_positive_int(nested.input_flat_dim, "input_flat_dim")
    if nested.input_flat_dim != 32:
        raise ValueError("input_flat_dim must be 32")
    validate_full_vector_length(nested.full_vector_length)
    validate_nested_batch_values_tuple(nested.nested_batch_values)

    validate_positive_int(nested.row_count, "row_count")
    if nested.row_count != 2:
        raise ValueError("row_count must be 2")

    if type(nested.row_lengths) is not tuple:
        raise TypeError("row_lengths must be exact tuple")
    if nested.row_lengths != (32, 32):
        raise ValueError("row_lengths must be (32, 32)")

    validate_positive_int(nested.total_scalar_count, "total_scalar_count")
    if nested.total_scalar_count != 64:
        raise ValueError("total_scalar_count must be 64")

    validate_bool(nested.flattened_matches_source_flat_vector, "flattened_matches_source_flat_vector")
    if not nested.flattened_matches_source_flat_vector:
        raise ValueError("flattened_matches_source_flat_vector must be True")

    validate_bool(nested.all_values_are_exact_float, "all_values_are_exact_float")
    if not nested.all_values_are_exact_float:
        raise ValueError("all_values_are_exact_float must be True")

    validate_bool(nested.all_values_within_source_range, "all_values_within_source_range")
    if not nested.all_values_within_source_range:
        raise ValueError("all_values_within_source_range must be True")

    for flag_name in (
        "array_materialized",
        "tensor_materialized",
        "forward_executed",
        "output_generated",
    ):
        val = getattr(nested, flag_name)
        validate_bool(val, flag_name)
        if val:
            raise ValueError(f"{flag_name} must be False")

    validate_non_empty_str(nested.reason, "reason")
    assert_no_local_path_leakage(nested.reason)
    assert_no_forbidden_claims(nested.reason)


def validate_nested_batch_values_metadata(metadata: FCVAENestedBatchValuesMetadata) -> None:
    if type(metadata) is not FCVAENestedBatchValuesMetadata:
        raise TypeError("metadata must be exact FCVAENestedBatchValuesMetadata instance")
    validate_non_empty_str(metadata.contract_version, "contract_version")
    if metadata.contract_version != FC_VAE_NESTED_BATCH_VALUES_CONTRACT_VERSION:
        raise ValueError(f"Invalid contract_version: {metadata.contract_version}")
    validate_non_empty_str(metadata.source_flat_vector_contract_version, "source_flat_vector_contract_version")
    if metadata.source_flat_vector_contract_version != FC_VAE_FAKE_INPUT_FLAT_VECTOR_CONTRACT_VERSION:
        raise ValueError(f"Invalid source_flat_vector_contract_version: {metadata.source_flat_vector_contract_version}")
    validate_non_empty_str(metadata.source_batch_view_contract_version, "source_batch_view_contract_version")
    if metadata.source_batch_view_contract_version != FC_VAE_FLAT_VECTOR_BATCH_VIEW_CONTRACT_VERSION:
        raise ValueError(f"Invalid source_batch_view_contract_version: {metadata.source_batch_view_contract_version}")

    validate_non_empty_str(metadata.flat_vector_status, "flat_vector_status")
    validate_non_empty_str(metadata.batch_view_status, "batch_view_status")
    validate_bool(metadata.torch_available, "torch_available")
    validate_bool(metadata.flat_vector_available_in_p35, "flat_vector_available_in_p35")
    if not metadata.flat_vector_available_in_p35:
        raise ValueError("flat_vector_available_in_p35 must be True")
    validate_bool(metadata.batch_view_available_in_p36, "batch_view_available_in_p36")
    if not metadata.batch_view_available_in_p36:
        raise ValueError("batch_view_available_in_p36 must be True")

    validate_full_vector_length(metadata.flat_vector_length)
    validate_shape_tuple(metadata.shape_tuple)
    validate_positive_int(metadata.batch_size, "batch_size")
    if metadata.batch_size != 2:
        raise ValueError("batch_size must be 2")
    validate_positive_int(metadata.input_flat_dim, "input_flat_dim")
    if metadata.input_flat_dim != 32:
        raise ValueError("input_flat_dim must be 32")

    validate_bool(metadata.nested_values_materialized, "nested_values_materialized")
    if not metadata.nested_values_materialized:
        raise ValueError("nested_values_materialized must be True")

    for flag_name in (
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


def validate_nested_batch_values_result(result: FCVAENestedBatchValuesResult) -> None:
    if type(result) is not FCVAENestedBatchValuesResult:
        raise TypeError("result must be exact FCVAENestedBatchValuesResult instance")
    validate_non_empty_str(result.contract_version, "contract_version")
    if result.contract_version != FC_VAE_NESTED_BATCH_VALUES_CONTRACT_VERSION:
        raise ValueError(f"Invalid contract_version: {result.contract_version}")

    validate_nested_batch_values_request(result.request)
    validate_nested_batch_values(result.nested_values)
    validate_nested_batch_values_metadata(result.metadata)
    validate_nested_batch_values_status(result.status)

    validate_bool(result.nested_values_available_in_p37, "nested_values_available_in_p37")
    if not result.nested_values_available_in_p37:
        raise ValueError("nested_values_available_in_p37 must be True")

    for flag_name in (
        "array_materialization_available_in_p37",
        "tensor_materialization_available_in_p37",
        "forward_execution_available_in_p37",
        "output_generation_available_in_p37",
    ):
        val = getattr(result, flag_name)
        validate_bool(val, flag_name)
        if val:
            raise ValueError(f"{flag_name} must be False")

    for flag_name in (
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

    # Coherence check
    if not result.metadata.torch_available:
        if result.status != FC_VAE_NESTED_BATCH_VALUES_STATUS_TORCH_UNAVAILABLE:
            raise ValueError("Inconsistent status: must be blocked_torch_unavailable")
    elif not result.metadata.batch_view_available_in_p36:
        if result.status != FC_VAE_NESTED_BATCH_VALUES_STATUS_BATCH_VIEW_BLOCKED:
            raise ValueError("Inconsistent status: must be blocked_by_batch_view_status")
    else:
        if result.status != FC_VAE_NESTED_BATCH_VALUES_STATUS_NESTED_VALUES_ONLY:
            raise ValueError("Inconsistent status: must be nested_values_only_no_tensor_no_forward_in_p37")

    validate_non_empty_str(result.reason, "reason")
    assert_no_local_path_leakage(result.reason)
    assert_no_forbidden_claims(result.reason)


# Row-major nesting rule helpers
def build_nested_tuple_from_flat_values(
    flat_values: Any,
    batch_size: Any,
    input_flat_dim: Any,
) -> Tuple[Tuple[float, ...], ...]:
    if type(flat_values) is not tuple:
        raise TypeError("flat_values must be exact tuple")
    if type(batch_size) is not int or isinstance(batch_size, bool):
        raise TypeError("batch_size must be exact int")
    if type(input_flat_dim) is not int or isinstance(input_flat_dim, bool):
        raise TypeError("input_flat_dim must be exact int")

    if batch_size != 2:
        raise ValueError("batch_size must be exactly 2")
    if input_flat_dim != 32:
        raise ValueError("input_flat_dim must be exactly 32")
    if len(flat_values) != 64:
        raise ValueError("flat_values length must be exactly 64")

    for i, x in enumerate(flat_values):
        if type(x) is not float:
            raise TypeError(f"flat_values[{i}] must be exact float, got {type(x).__name__}")

    row0 = flat_values[0:32]
    row1 = flat_values[32:64]
    return (row0, row1)


def flatten_nested_batch_values(nested_values: Any) -> Tuple[float, ...]:
    if type(nested_values) is not tuple:
        raise TypeError("nested_values must be exact tuple")
    if len(nested_values) != 2:
        raise ValueError("nested_values must have length 2")

    for i, row in enumerate(nested_values):
        if type(row) is not tuple:
            raise TypeError(f"nested_values[{i}] must be exact tuple, got {type(row).__name__}")
        if len(row) != 32:
            raise ValueError(f"nested_values[{i}] length must be exactly 32, got {len(row)}")
        for j, val in enumerate(row):
            if type(val) is not float:
                raise TypeError(f"nested_values[{i}][{j}] must be exact float, got {type(val).__name__}")

    return nested_values[0] + nested_values[1]


# Builders
def build_nested_batch_values_request_from_p36_default() -> FCVAENestedBatchValuesRequest:
    p35_result = run_fake_input_flat_vector_probe()
    p36_result = run_flat_vector_batch_view_probe()

    req = FCVAENestedBatchValuesRequest(
        contract_version=FC_VAE_NESTED_BATCH_VALUES_CONTRACT_VERSION,
        nested_values_kind=FC_VAE_NESTED_BATCH_VALUES_KIND,
        architecture_id="FC-VAE",
        source_flat_vector_contract_version=p35_result.contract_version,
        source_batch_view_contract_version=p36_result.contract_version,
        batch_size=p36_result.shape.batch_size,
        input_flat_dim=p36_result.shape.input_flat_dim,
        full_vector_length=p36_result.shape.full_vector_length,
        shape_tuple=p36_result.shape.shape_tuple,
        order_kind=FC_VAE_NESTED_BATCH_VALUES_ORDER_KIND,
        allow_nested_values_in_p37=True,
        allow_array_materialization_in_p37=False,
        allow_tensor_materialization_in_p37=False,
        allow_forward_execution_in_p37=False,
        reason="p37_nested_batch_values_request_from_p36_default",
    )
    validate_nested_batch_values_request(req)
    return req


def build_nested_batch_values(
    request: FCVAENestedBatchValuesRequest,
) -> FCVAENestedBatchValues:
    validate_nested_batch_values_request(request)
    p35_result = run_fake_input_flat_vector_probe()

    flat_vector_vals = p35_result.flat_vector.flat_values
    nested_vals = build_nested_tuple_from_flat_values(
        flat_vector_vals, request.batch_size, request.input_flat_dim
    )

    flat_recon = flatten_nested_batch_values(nested_vals)
    flattened_matches = (flat_recon == flat_vector_vals)

    all_exact_float = all(type(x) is float for x in flat_recon)
    min_v = p35_result.request.min_value
    max_v = p35_result.request.max_value
    all_within_range = all(min_v <= x <= max_v for x in flat_recon)

    vals = FCVAENestedBatchValues(
        contract_version=FC_VAE_NESTED_BATCH_VALUES_CONTRACT_VERSION,
        nested_values_kind=FC_VAE_NESTED_BATCH_VALUES_KIND,
        order_kind=request.order_kind,
        shape_tuple=request.shape_tuple,
        batch_size=request.batch_size,
        input_flat_dim=request.input_flat_dim,
        full_vector_length=request.full_vector_length,
        nested_batch_values=nested_vals,
        row_count=len(nested_vals),
        row_lengths=(len(nested_vals[0]), len(nested_vals[1])),
        total_scalar_count=len(flat_recon),
        flattened_matches_source_flat_vector=flattened_matches,
        all_values_are_exact_float=all_exact_float,
        all_values_within_source_range=all_within_range,
        array_materialized=False,
        tensor_materialized=False,
        forward_executed=False,
        output_generated=False,
        reason="p37_nested_batch_values",
    )
    validate_nested_batch_values(vals)
    return vals


def build_nested_batch_values_metadata(
    request: FCVAENestedBatchValuesRequest,
) -> FCVAENestedBatchValuesMetadata:
    validate_nested_batch_values_request(request)
    p35_result = run_fake_input_flat_vector_probe()
    p36_result = run_flat_vector_batch_view_probe()

    meta = FCVAENestedBatchValuesMetadata(
        contract_version=FC_VAE_NESTED_BATCH_VALUES_CONTRACT_VERSION,
        source_flat_vector_contract_version=request.source_flat_vector_contract_version,
        source_batch_view_contract_version=request.source_batch_view_contract_version,
        flat_vector_status=p35_result.status,
        batch_view_status=p36_result.status,
        torch_available=p35_result.metadata.torch_available,
        flat_vector_available_in_p35=p35_result.flat_vector_available_in_p35,
        batch_view_available_in_p36=p36_result.batch_view_available_in_p36,
        flat_vector_length=p35_result.metadata.full_vector_length,
        shape_tuple=request.shape_tuple,
        batch_size=request.batch_size,
        input_flat_dim=request.input_flat_dim,
        nested_values_materialized=True,
        array_materialization_attempted=False,
        tensor_materialization_attempted=False,
        forward_execution_attempted=False,
        output_generation_attempted=False,
        reason="p37_nested_batch_values_metadata",
    )
    validate_nested_batch_values_metadata(meta)
    return meta


def build_nested_batch_values_result(
    request: FCVAENestedBatchValuesRequest,
) -> FCVAENestedBatchValuesResult:
    validate_nested_batch_values_request(request)
    nested_vals = build_nested_batch_values(request)
    metadata = build_nested_batch_values_metadata(request)

    if not metadata.torch_available:
        status = FC_VAE_NESTED_BATCH_VALUES_STATUS_TORCH_UNAVAILABLE
        reason = "nested_values_blocked_torch_unavailable"
    elif not metadata.batch_view_available_in_p36:
        status = FC_VAE_NESTED_BATCH_VALUES_STATUS_BATCH_VIEW_BLOCKED
        reason = "nested_values_blocked_by_batch_view_status"
    else:
        status = FC_VAE_NESTED_BATCH_VALUES_STATUS_NESTED_VALUES_ONLY
        reason = "nested_values_only_no_tensor_no_forward_in_p37"

    res = FCVAENestedBatchValuesResult(
        contract_version=FC_VAE_NESTED_BATCH_VALUES_CONTRACT_VERSION,
        request=request,
        nested_values=nested_vals,
        metadata=metadata,
        status=status,
        nested_values_available_in_p37=True,
        array_materialization_available_in_p37=False,
        tensor_materialization_available_in_p37=False,
        forward_execution_available_in_p37=False,
        output_generation_available_in_p37=False,
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
    validate_nested_batch_values_result(res)
    return res


def run_nested_batch_values_probe() -> FCVAENestedBatchValuesResult:
    req = build_nested_batch_values_request_from_p36_default()
    res = build_nested_batch_values_result(req)
    validate_nested_batch_values_result(res)
    return res


# Serialization
def nested_batch_values_request_to_json_dict(
    request: FCVAENestedBatchValuesRequest,
) -> Dict[str, Any]:
    validate_nested_batch_values_request(request)
    d = {
        "contract_version": request.contract_version,
        "nested_values_kind": request.nested_values_kind,
        "architecture_id": request.architecture_id,
        "source_flat_vector_contract_version": request.source_flat_vector_contract_version,
        "source_batch_view_contract_version": request.source_batch_view_contract_version,
        "batch_size": request.batch_size,
        "input_flat_dim": request.input_flat_dim,
        "full_vector_length": request.full_vector_length,
        "shape_tuple": list(request.shape_tuple),
        "order_kind": request.order_kind,
        "allow_nested_values_in_p37": request.allow_nested_values_in_p37,
        "allow_array_materialization_in_p37": request.allow_array_materialization_in_p37,
        "allow_tensor_materialization_in_p37": request.allow_tensor_materialization_in_p37,
        "allow_forward_execution_in_p37": request.allow_forward_execution_in_p37,
        "reason": request.reason,
    }
    assert_no_local_path_leakage(d)
    assert_no_forbidden_claims(d)
    return d


def nested_batch_values_to_json_dict(
    nested: FCVAENestedBatchValues,
) -> Dict[str, Any]:
    validate_nested_batch_values(nested)
    d = {
        "contract_version": nested.contract_version,
        "nested_values_kind": nested.nested_values_kind,
        "order_kind": nested.order_kind,
        "shape_tuple": list(nested.shape_tuple),
        "batch_size": nested.batch_size,
        "input_flat_dim": nested.input_flat_dim,
        "full_vector_length": nested.full_vector_length,
        "nested_batch_values": [list(row) for row in nested.nested_batch_values],
        "row_count": nested.row_count,
        "row_lengths": list(nested.row_lengths),
        "total_scalar_count": nested.total_scalar_count,
        "flattened_matches_source_flat_vector": nested.flattened_matches_source_flat_vector,
        "all_values_are_exact_float": nested.all_values_are_exact_float,
        "all_values_within_source_range": nested.all_values_within_source_range,
        "array_materialized": nested.array_materialized,
        "tensor_materialized": nested.tensor_materialized,
        "forward_executed": nested.forward_executed,
        "output_generated": nested.output_generated,
        "reason": nested.reason,
    }
    assert_no_local_path_leakage(d)
    assert_no_forbidden_claims(d)
    return d


def nested_batch_values_metadata_to_json_dict(
    metadata: FCVAENestedBatchValuesMetadata,
) -> Dict[str, Any]:
    validate_nested_batch_values_metadata(metadata)
    d = {
        "contract_version": metadata.contract_version,
        "source_flat_vector_contract_version": metadata.source_flat_vector_contract_version,
        "source_batch_view_contract_version": metadata.source_batch_view_contract_version,
        "flat_vector_status": metadata.flat_vector_status,
        "batch_view_status": metadata.batch_view_status,
        "torch_available": metadata.torch_available,
        "flat_vector_available_in_p35": metadata.flat_vector_available_in_p35,
        "batch_view_available_in_p36": metadata.batch_view_available_in_p36,
        "flat_vector_length": metadata.flat_vector_length,
        "shape_tuple": list(metadata.shape_tuple),
        "batch_size": metadata.batch_size,
        "input_flat_dim": metadata.input_flat_dim,
        "nested_values_materialized": metadata.nested_values_materialized,
        "array_materialization_attempted": metadata.array_materialization_attempted,
        "tensor_materialization_attempted": metadata.tensor_materialization_attempted,
        "forward_execution_attempted": metadata.forward_execution_attempted,
        "output_generation_attempted": metadata.output_generation_attempted,
        "reason": metadata.reason,
    }
    assert_no_local_path_leakage(d)
    assert_no_forbidden_claims(d)
    return d


def nested_batch_values_result_to_json_dict(
    result: FCVAENestedBatchValuesResult,
) -> Dict[str, Any]:
    validate_nested_batch_values_result(result)
    d = {
        "contract_version": result.contract_version,
        "request": nested_batch_values_request_to_json_dict(result.request),
        "nested_values": nested_batch_values_to_json_dict(result.nested_values),
        "metadata": nested_batch_values_metadata_to_json_dict(result.metadata),
        "status": result.status,
        "nested_values_available_in_p37": result.nested_values_available_in_p37,
        "array_materialization_available_in_p37": result.array_materialization_available_in_p37,
        "tensor_materialization_available_in_p37": result.tensor_materialization_available_in_p37,
        "forward_execution_available_in_p37": result.forward_execution_available_in_p37,
        "output_generation_available_in_p37": result.output_generation_available_in_p37,
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


def compact_nested_batch_values_json(result: FCVAENestedBatchValuesResult) -> str:
    d = nested_batch_values_result_to_json_dict(result)
    return json.dumps(d, sort_keys=True, separators=(",", ":"))
