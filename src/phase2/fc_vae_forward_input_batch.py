# src/phase2/fc_vae_forward_input_batch.py

import dataclasses
import json
from typing import Any, Dict, Tuple, Union

from src.phase2.fc_vae_forward_boundary import (
    FC_VAE_FORWARD_BOUNDARY_CONTRACT_VERSION,
    run_forward_boundary_probe,
    forward_boundary_result_to_json_dict,
)


# Public Constants
FC_VAE_FORWARD_INPUT_BATCH_CONTRACT_VERSION = "phase2_p32_forward_input_batch_contract_v1"

FC_VAE_FORWARD_INPUT_BATCH_KIND = "forward_input_batch_metadata_no_tensor"

FC_VAE_FORWARD_INPUT_BATCH_MODULE_NAME = "src.phase2.fc_vae_forward_input_batch"

FC_VAE_FORWARD_INPUT_BATCH_STATUS_MATERIALIZATION_BLOCKED = "input_batch_materialization_blocked_in_p32"

FC_VAE_FORWARD_INPUT_BATCH_STATUS_FORWARD_BOUNDARY_BLOCKED = "blocked_by_forward_boundary"

FC_VAE_FORWARD_INPUT_BATCH_STATUS_TORCH_UNAVAILABLE = "blocked_torch_unavailable"

SUPPORTED_FC_VAE_FORWARD_INPUT_BATCH_STATUSES = (
    "input_batch_materialization_blocked_in_p32",
    "blocked_by_forward_boundary",
    "blocked_torch_unavailable",
)

FC_VAE_FORWARD_INPUT_BATCH_SHAPE_KIND = "declared_input_batch_shape_no_tensor"

DEFAULT_P32_BATCH_SIZE = 2
MIN_P32_BATCH_SIZE = 1
MAX_P32_BATCH_SIZE = 16


# Public Dataclasses (All Frozen)
@dataclasses.dataclass(frozen=True)
class FCVAEForwardInputBatchRequest:
    contract_version: str
    batch_kind: str
    architecture_id: str
    batch_size: int
    input_flat_dim: int
    source_forward_boundary_contract_version: str
    allow_tensor_materialization_in_p32: bool
    allow_array_materialization_in_p32: bool
    allow_forward_execution_in_p32: bool
    allow_output_generation_in_p32: bool
    reason: str


@dataclasses.dataclass(frozen=True)
class FCVAEForwardInputBatchShape:
    contract_version: str
    shape_kind: str
    batch_size: int
    input_flat_dim: int
    rank: int
    shape_tuple: Tuple[int, int]
    tensor_materialized: bool
    array_materialized: bool
    values_materialized: bool
    reason: str


@dataclasses.dataclass(frozen=True)
class FCVAEForwardInputBatchMetadata:
    contract_version: str
    batch_kind: str
    forward_boundary_contract_version: str
    forward_boundary_status: str
    torch_available: bool
    forward_execution_available_in_p31: bool
    tensor_allocation_available_in_p31: bool
    output_generation_available_in_p31: bool
    batch_shape_declared: bool
    tensor_materialization_attempted: bool
    array_materialization_attempted: bool
    values_materialization_attempted: bool
    forward_execution_attempted: bool
    output_generation_attempted: bool
    reason: str


@dataclasses.dataclass(frozen=True)
class FCVAEForwardInputBatchResult:
    contract_version: str
    request: FCVAEForwardInputBatchRequest
    shape: FCVAEForwardInputBatchShape
    metadata: FCVAEForwardInputBatchMetadata
    status: str
    input_batch_available_in_p32: bool
    tensor_materialization_available_in_p32: bool
    array_materialization_available_in_p32: bool
    values_materialization_available_in_p32: bool
    forward_execution_available_in_p32: bool
    output_generation_available_in_p32: bool
    no_tensor_created: bool
    no_array_created: bool
    no_values_materialized: bool
    no_forward_execution: bool
    no_output_generation: bool
    no_training_loop: bool
    no_optimizer: bool
    no_checkpointing: bool
    no_artifact_generation: bool
    no_final_comparison: bool
    no_scientific_conclusion: bool
    reason: str


# Public validation & safety helpers
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


def validate_batch_size(value: Any) -> None:
    if type(value) is not int or isinstance(value, bool):
        raise TypeError(f"batch_size must be exact int, got {type(value).__name__}")
    if value < MIN_P32_BATCH_SIZE or value > MAX_P32_BATCH_SIZE:
        raise ValueError(f"batch_size must be between {MIN_P32_BATCH_SIZE} and {MAX_P32_BATCH_SIZE} inclusive")


def validate_shape_tuple(value: Any) -> None:
    if type(value) is not tuple:
        raise TypeError(f"shape_tuple must be exact tuple, got {type(value).__name__}")
    if len(value) != 2:
        raise ValueError("shape_tuple length must be exactly 2")
    for i, dim in enumerate(value):
        if type(dim) is not int or isinstance(dim, bool):
            raise TypeError(f"shape_tuple[{i}] must be exact int, got {type(dim).__name__}")
        if dim <= 0:
            raise ValueError(f"shape_tuple[{i}] must be > 0")


def validate_input_batch_status(value: Any) -> None:
    if type(value) is not str:
        raise TypeError(f"status must be exact str, got {type(value).__name__}")
    if value not in SUPPORTED_FC_VAE_FORWARD_INPUT_BATCH_STATUSES:
        raise ValueError(f"Unsupported forward input batch status: {value}")


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
                   .replace("input_batch_materialization_blocked_in_p32", ""))
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


def validate_forward_input_batch_request(request: FCVAEForwardInputBatchRequest) -> None:
    if type(request) is not FCVAEForwardInputBatchRequest:
        raise TypeError("request must be exact FCVAEForwardInputBatchRequest instance")

    validate_non_empty_str(request.contract_version, "contract_version")
    if request.contract_version != FC_VAE_FORWARD_INPUT_BATCH_CONTRACT_VERSION:
        raise ValueError(f"Invalid contract_version: {request.contract_version}")

    validate_non_empty_str(request.batch_kind, "batch_kind")
    if request.batch_kind != FC_VAE_FORWARD_INPUT_BATCH_KIND:
        raise ValueError(f"Invalid batch_kind: {request.batch_kind}")

    validate_non_empty_str(request.architecture_id, "architecture_id")
    if request.architecture_id != "FC-VAE":
        raise ValueError(f"Invalid architecture_id: {request.architecture_id}")

    validate_batch_size(request.batch_size)
    validate_positive_int(request.input_flat_dim, "input_flat_dim")

    validate_non_empty_str(request.source_forward_boundary_contract_version, "source_forward_boundary_contract_version")
    if request.source_forward_boundary_contract_version != FC_VAE_FORWARD_BOUNDARY_CONTRACT_VERSION:
        raise ValueError(f"Invalid source_forward_boundary_contract_version: {request.source_forward_boundary_contract_version}")

    validate_bool(request.allow_tensor_materialization_in_p32, "allow_tensor_materialization_in_p32")
    if request.allow_tensor_materialization_in_p32:
        raise ValueError("allow_tensor_materialization_in_p32 must be False")

    validate_bool(request.allow_array_materialization_in_p32, "allow_array_materialization_in_p32")
    if request.allow_array_materialization_in_p32:
        raise ValueError("allow_array_materialization_in_p32 must be False")

    validate_bool(request.allow_forward_execution_in_p32, "allow_forward_execution_in_p32")
    if request.allow_forward_execution_in_p32:
        raise ValueError("allow_forward_execution_in_p32 must be False")

    validate_bool(request.allow_output_generation_in_p32, "allow_output_generation_in_p32")
    if request.allow_output_generation_in_p32:
        raise ValueError("allow_output_generation_in_p32 must be False")

    validate_non_empty_str(request.reason, "reason")
    assert_no_local_path_leakage(request.reason)
    assert_no_forbidden_claims(request.reason)


def validate_forward_input_batch_shape(shape: FCVAEForwardInputBatchShape) -> None:
    if type(shape) is not FCVAEForwardInputBatchShape:
        raise TypeError("shape must be exact FCVAEForwardInputBatchShape instance")

    validate_non_empty_str(shape.contract_version, "contract_version")
    if shape.contract_version != FC_VAE_FORWARD_INPUT_BATCH_CONTRACT_VERSION:
        raise ValueError(f"Invalid contract_version: {shape.contract_version}")

    validate_non_empty_str(shape.shape_kind, "shape_kind")
    if shape.shape_kind != FC_VAE_FORWARD_INPUT_BATCH_SHAPE_KIND:
        raise ValueError(f"Invalid shape_kind: {shape.shape_kind}")

    validate_batch_size(shape.batch_size)
    validate_positive_int(shape.input_flat_dim, "input_flat_dim")

    validate_positive_int(shape.rank, "rank")
    if shape.rank != 2:
        raise ValueError("rank must be exactly 2")

    validate_shape_tuple(shape.shape_tuple)
    if shape.shape_tuple != (shape.batch_size, shape.input_flat_dim):
        raise ValueError(f"shape_tuple must equal (batch_size, input_flat_dim), got {shape.shape_tuple}")

    validate_bool(shape.tensor_materialized, "tensor_materialized")
    if shape.tensor_materialized:
        raise ValueError("tensor_materialized must be False")

    validate_bool(shape.array_materialized, "array_materialized")
    if shape.array_materialized:
        raise ValueError("array_materialized must be False")

    validate_bool(shape.values_materialized, "values_materialized")
    if shape.values_materialized:
        raise ValueError("values_materialized must be False")

    validate_non_empty_str(shape.reason, "reason")
    assert_no_local_path_leakage(shape.reason)
    assert_no_forbidden_claims(shape.reason)


def validate_forward_input_batch_metadata(metadata: FCVAEForwardInputBatchMetadata) -> None:
    if type(metadata) is not FCVAEForwardInputBatchMetadata:
        raise TypeError("metadata must be exact FCVAEForwardInputBatchMetadata instance")

    validate_non_empty_str(metadata.contract_version, "contract_version")
    if metadata.contract_version != FC_VAE_FORWARD_INPUT_BATCH_CONTRACT_VERSION:
        raise ValueError(f"Invalid contract_version: {metadata.contract_version}")

    validate_non_empty_str(metadata.batch_kind, "batch_kind")
    if metadata.batch_kind != FC_VAE_FORWARD_INPUT_BATCH_KIND:
        raise ValueError(f"Invalid batch_kind: {metadata.batch_kind}")

    validate_non_empty_str(metadata.forward_boundary_contract_version, "forward_boundary_contract_version")
    if metadata.forward_boundary_contract_version != FC_VAE_FORWARD_BOUNDARY_CONTRACT_VERSION:
        raise ValueError(f"Invalid forward_boundary_contract_version: {metadata.forward_boundary_contract_version}")

    validate_non_empty_str(metadata.forward_boundary_status, "forward_boundary_status")
    validate_bool(metadata.torch_available, "torch_available")
    validate_bool(metadata.forward_execution_available_in_p31, "forward_execution_available_in_p31")
    validate_bool(metadata.tensor_allocation_available_in_p31, "tensor_allocation_available_in_p31")
    validate_bool(metadata.output_generation_available_in_p31, "output_generation_available_in_p31")

    validate_bool(metadata.batch_shape_declared, "batch_shape_declared")
    if not metadata.batch_shape_declared:
        raise ValueError("batch_shape_declared must be True")

    validate_bool(metadata.tensor_materialization_attempted, "tensor_materialization_attempted")
    if metadata.tensor_materialization_attempted:
        raise ValueError("tensor_materialization_attempted must be False")

    validate_bool(metadata.array_materialization_attempted, "array_materialization_attempted")
    if metadata.array_materialization_attempted:
        raise ValueError("array_materialization_attempted must be False")

    validate_bool(metadata.values_materialization_attempted, "values_materialization_attempted")
    if metadata.values_materialization_attempted:
        raise ValueError("values_materialization_attempted must be False")

    validate_bool(metadata.forward_execution_attempted, "forward_execution_attempted")
    if metadata.forward_execution_attempted:
        raise ValueError("forward_execution_attempted must be False")

    validate_bool(metadata.output_generation_attempted, "output_generation_attempted")
    if metadata.output_generation_attempted:
        raise ValueError("output_generation_attempted must be False")

    validate_non_empty_str(metadata.reason, "reason")
    assert_no_local_path_leakage(metadata.reason)
    assert_no_forbidden_claims(metadata.reason)


def validate_forward_input_batch_result(result: FCVAEForwardInputBatchResult) -> None:
    if type(result) is not FCVAEForwardInputBatchResult:
        raise TypeError("result must be exact FCVAEForwardInputBatchResult instance")

    validate_non_empty_str(result.contract_version, "contract_version")
    if result.contract_version != FC_VAE_FORWARD_INPUT_BATCH_CONTRACT_VERSION:
        raise ValueError(f"Invalid contract_version: {result.contract_version}")

    validate_forward_input_batch_request(result.request)
    validate_forward_input_batch_shape(result.shape)
    validate_forward_input_batch_metadata(result.metadata)
    validate_input_batch_status(result.status)

    validate_bool(result.input_batch_available_in_p32, "input_batch_available_in_p32")
    if result.input_batch_available_in_p32:
        raise ValueError("input_batch_available_in_p32 must be False")

    # all materialization/execution/output availability flags False
    for flag_name in (
        "tensor_materialization_available_in_p32",
        "array_materialization_available_in_p32",
        "values_materialization_available_in_p32",
        "forward_execution_available_in_p32",
        "output_generation_available_in_p32",
    ):
        val = getattr(result, flag_name)
        validate_bool(val, flag_name)
        if val:
            raise ValueError(f"{flag_name} must be False")

    # all no_* flags True
    for flag_name in (
        "no_tensor_created",
        "no_array_created",
        "no_values_materialized",
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

    # status logic
    if not result.metadata.torch_available:
        if result.status != FC_VAE_FORWARD_INPUT_BATCH_STATUS_TORCH_UNAVAILABLE:
            raise ValueError(f"Inconsistent state: status must be '{FC_VAE_FORWARD_INPUT_BATCH_STATUS_TORCH_UNAVAILABLE}' when torch is unavailable")
    else:
        if not result.metadata.forward_execution_available_in_p31:
            if result.status != FC_VAE_FORWARD_INPUT_BATCH_STATUS_FORWARD_BOUNDARY_BLOCKED:
                raise ValueError(f"Inconsistent state: status must be '{FC_VAE_FORWARD_INPUT_BATCH_STATUS_FORWARD_BOUNDARY_BLOCKED}' when forward boundary is blocked")
        else:
            if result.status != FC_VAE_FORWARD_INPUT_BATCH_STATUS_MATERIALIZATION_BLOCKED:
                raise ValueError(f"Inconsistent state: status must be '{FC_VAE_FORWARD_INPUT_BATCH_STATUS_MATERIALIZATION_BLOCKED}' when materialization is blocked")

    validate_non_empty_str(result.reason, "reason")
    assert_no_local_path_leakage(result.reason)
    assert_no_forbidden_claims(result.reason)


# Public Builders
def build_forward_input_batch_request_from_p31_default(batch_size: int = DEFAULT_P32_BATCH_SIZE) -> FCVAEForwardInputBatchRequest:
    p31_result = run_forward_boundary_probe()
    input_flat_dim = p31_result.request.input_flat_dim

    req = FCVAEForwardInputBatchRequest(
        contract_version=FC_VAE_FORWARD_INPUT_BATCH_CONTRACT_VERSION,
        batch_kind=FC_VAE_FORWARD_INPUT_BATCH_KIND,
        architecture_id="FC-VAE",
        batch_size=batch_size,
        input_flat_dim=input_flat_dim,
        source_forward_boundary_contract_version=p31_result.contract_version,
        allow_tensor_materialization_in_p32=False,
        allow_array_materialization_in_p32=False,
        allow_forward_execution_in_p32=False,
        allow_output_generation_in_p32=False,
        reason="p32_forward_input_batch_request_from_p31_default",
    )
    validate_forward_input_batch_request(req)
    return req


def build_forward_input_batch_shape(request: FCVAEForwardInputBatchRequest) -> FCVAEForwardInputBatchShape:
    validate_forward_input_batch_request(request)

    shape = FCVAEForwardInputBatchShape(
        contract_version=FC_VAE_FORWARD_INPUT_BATCH_CONTRACT_VERSION,
        shape_kind=FC_VAE_FORWARD_INPUT_BATCH_SHAPE_KIND,
        batch_size=request.batch_size,
        input_flat_dim=request.input_flat_dim,
        rank=2,
        shape_tuple=(request.batch_size, request.input_flat_dim),
        tensor_materialized=False,
        array_materialized=False,
        values_materialized=False,
        reason="p32_forward_input_batch_shape",
    )
    validate_forward_input_batch_shape(shape)
    return shape


def build_forward_input_batch_metadata(request: FCVAEForwardInputBatchRequest) -> FCVAEForwardInputBatchMetadata:
    validate_forward_input_batch_request(request)

    p31_result = run_forward_boundary_probe()

    meta = FCVAEForwardInputBatchMetadata(
        contract_version=FC_VAE_FORWARD_INPUT_BATCH_CONTRACT_VERSION,
        batch_kind=FC_VAE_FORWARD_INPUT_BATCH_KIND,
        forward_boundary_contract_version=p31_result.contract_version,
        forward_boundary_status=p31_result.status,
        torch_available=p31_result.metadata.torch_available,
        forward_execution_available_in_p31=p31_result.forward_execution_available_in_p31,
        tensor_allocation_available_in_p31=p31_result.tensor_allocation_available_in_p31,
        output_generation_available_in_p31=p31_result.output_generation_available_in_p31,
        batch_shape_declared=True,
        tensor_materialization_attempted=False,
        array_materialization_attempted=False,
        values_materialization_attempted=False,
        forward_execution_attempted=False,
        output_generation_attempted=False,
        reason="p32_forward_input_batch_metadata",
    )
    validate_forward_input_batch_metadata(meta)
    return meta


def build_forward_input_batch_result(request: FCVAEForwardInputBatchRequest) -> FCVAEForwardInputBatchResult:
    validate_forward_input_batch_request(request)
    shape = build_forward_input_batch_shape(request)
    metadata = build_forward_input_batch_metadata(request)

    if not metadata.torch_available:
        status = FC_VAE_FORWARD_INPUT_BATCH_STATUS_TORCH_UNAVAILABLE
        reason = "forward_input_batch_blocked_torch_unavailable"
    elif not metadata.forward_execution_available_in_p31:
        status = FC_VAE_FORWARD_INPUT_BATCH_STATUS_FORWARD_BOUNDARY_BLOCKED
        reason = "forward_input_batch_blocked_by_forward_boundary"
    else:
        status = FC_VAE_FORWARD_INPUT_BATCH_STATUS_MATERIALIZATION_BLOCKED
        reason = "forward_input_batch_materialization_blocked"

    res = FCVAEForwardInputBatchResult(
        contract_version=FC_VAE_FORWARD_INPUT_BATCH_CONTRACT_VERSION,
        request=request,
        shape=shape,
        metadata=metadata,
        status=status,
        input_batch_available_in_p32=False,
        tensor_materialization_available_in_p32=False,
        array_materialization_available_in_p32=False,
        values_materialization_available_in_p32=False,
        forward_execution_available_in_p32=False,
        output_generation_available_in_p32=False,
        no_tensor_created=True,
        no_array_created=True,
        no_values_materialized=True,
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
    validate_forward_input_batch_result(res)
    return res


def run_forward_input_batch_probe() -> FCVAEForwardInputBatchResult:
    req = build_forward_input_batch_request_from_p31_default()
    res = build_forward_input_batch_result(req)
    validate_forward_input_batch_result(res)
    return res


# Public Serialization
def forward_input_batch_request_to_json_dict(request: FCVAEForwardInputBatchRequest) -> dict:
    validate_forward_input_batch_request(request)
    d = {
        "contract_version": request.contract_version,
        "batch_kind": request.batch_kind,
        "architecture_id": request.architecture_id,
        "batch_size": request.batch_size,
        "input_flat_dim": request.input_flat_dim,
        "source_forward_boundary_contract_version": request.source_forward_boundary_contract_version,
        "allow_tensor_materialization_in_p32": request.allow_tensor_materialization_in_p32,
        "allow_array_materialization_in_p32": request.allow_array_materialization_in_p32,
        "allow_forward_execution_in_p32": request.allow_forward_execution_in_p32,
        "allow_output_generation_in_p32": request.allow_output_generation_in_p32,
        "reason": request.reason,
    }
    assert_no_local_path_leakage(d)
    assert_no_forbidden_claims(d)
    return d


def forward_input_batch_shape_to_json_dict(shape: FCVAEForwardInputBatchShape) -> dict:
    validate_forward_input_batch_shape(shape)
    d = {
        "contract_version": shape.contract_version,
        "shape_kind": shape.shape_kind,
        "batch_size": shape.batch_size,
        "input_flat_dim": shape.input_flat_dim,
        "rank": shape.rank,
        "shape_tuple": [shape.shape_tuple[0], shape.shape_tuple[1]],
        "tensor_materialized": shape.tensor_materialized,
        "array_materialized": shape.array_materialized,
        "values_materialized": shape.values_materialized,
        "reason": shape.reason,
    }
    assert_no_local_path_leakage(d)
    assert_no_forbidden_claims(d)
    return d


def forward_input_batch_metadata_to_json_dict(metadata: FCVAEForwardInputBatchMetadata) -> dict:
    validate_forward_input_batch_metadata(metadata)
    d = {
        "contract_version": metadata.contract_version,
        "batch_kind": metadata.batch_kind,
        "forward_boundary_contract_version": metadata.forward_boundary_contract_version,
        "forward_boundary_status": metadata.forward_boundary_status,
        "torch_available": metadata.torch_available,
        "forward_execution_available_in_p31": metadata.forward_execution_available_in_p31,
        "tensor_allocation_available_in_p31": metadata.tensor_allocation_available_in_p31,
        "output_generation_available_in_p31": metadata.output_generation_available_in_p31,
        "batch_shape_declared": metadata.batch_shape_declared,
        "tensor_materialization_attempted": metadata.tensor_materialization_attempted,
        "array_materialization_attempted": metadata.array_materialization_attempted,
        "values_materialization_attempted": metadata.values_materialization_attempted,
        "forward_execution_attempted": metadata.forward_execution_attempted,
        "output_generation_attempted": metadata.output_generation_attempted,
        "reason": metadata.reason,
    }
    assert_no_local_path_leakage(d)
    assert_no_forbidden_claims(d)
    return d


def forward_input_batch_result_to_json_dict(result: FCVAEForwardInputBatchResult) -> dict:
    validate_forward_input_batch_result(result)
    d = {
        "contract_version": result.contract_version,
        "request": forward_input_batch_request_to_json_dict(result.request),
        "shape": forward_input_batch_shape_to_json_dict(result.shape),
        "metadata": forward_input_batch_metadata_to_json_dict(result.metadata),
        "status": result.status,
        "input_batch_available_in_p32": result.input_batch_available_in_p32,
        "tensor_materialization_available_in_p32": result.tensor_materialization_available_in_p32,
        "array_materialization_available_in_p32": result.array_materialization_available_in_p32,
        "values_materialization_available_in_p32": result.values_materialization_available_in_p32,
        "forward_execution_available_in_p32": result.forward_execution_available_in_p32,
        "output_generation_available_in_p32": result.output_generation_available_in_p32,
        "no_tensor_created": result.no_tensor_created,
        "no_array_created": result.no_array_created,
        "no_values_materialized": result.no_values_materialized,
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


def compact_forward_input_batch_json(result: FCVAEForwardInputBatchResult) -> str:
    d = forward_input_batch_result_to_json_dict(result)
    return json.dumps(d, sort_keys=True, separators=(",", ":"))
