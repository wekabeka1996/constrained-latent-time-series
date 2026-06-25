# src/phase2/fc_vae_tensor_materialization_request.py

import dataclasses
import json
from typing import Any, Dict, Tuple

from src.phase2.torch_boundary import (
    TORCH_BOUNDARY_CONTRACT_VERSION,
    TORCH_BACKEND_NAME,
    TORCH_POLICY_OPTIONAL,
    build_torch_dependency_status,
)
from src.phase2.fc_vae_nested_batch_values import (
    FC_VAE_NESTED_BATCH_VALUES_CONTRACT_VERSION,
    run_nested_batch_values_probe,
)

# Constants
FC_VAE_TENSOR_MATERIALIZATION_REQUEST_CONTRACT_VERSION = "phase2_p38_tensor_materialization_request_contract_v1"

FC_VAE_TENSOR_MATERIALIZATION_REQUEST_KIND = "tensor_materialization_request_metadata_no_tensor_allocation"

FC_VAE_TENSOR_MATERIALIZATION_REQUEST_MODULE_NAME = "src.phase2.fc_vae_tensor_materialization_request"

FC_VAE_TENSOR_MATERIALIZATION_REQUEST_STATUS_TORCH_UNAVAILABLE = "blocked_torch_unavailable"

FC_VAE_TENSOR_MATERIALIZATION_REQUEST_STATUS_NESTED_VALUES_BLOCKED = "blocked_by_nested_values_status"

FC_VAE_TENSOR_MATERIALIZATION_REQUEST_STATUS_REQUEST_ONLY = "tensor_request_metadata_only_no_allocation_in_p38"

SUPPORTED_FC_VAE_TENSOR_MATERIALIZATION_REQUEST_STATUSES = (
    "blocked_torch_unavailable",
    "blocked_by_nested_values_status",
    "tensor_request_metadata_only_no_allocation_in_p38",
)

FC_VAE_TENSOR_TARGET_FRAMEWORK = "torch"
FC_VAE_TENSOR_TARGET_DTYPE_INTENT = "float32"
FC_VAE_TENSOR_TARGET_DEVICE_INTENT = "cpu"
FC_VAE_TENSOR_LAYOUT_KIND = "row_major_2d_batch_tensor"

# Defaults
DEFAULT_P38_BATCH_SIZE = 2
DEFAULT_P38_INPUT_FLAT_DIM = 32
DEFAULT_P38_FULL_VECTOR_LENGTH = 64
DEFAULT_P38_SHAPE_TUPLE = (2, 32)
DEFAULT_P38_REQUIRES_GRAD = False


# Dataclasses
@dataclasses.dataclass(frozen=True)
class FCVAETensorMaterializationRequest:
    contract_version: str
    request_kind: str
    architecture_id: str
    source_nested_values_contract_version: str
    source_torch_boundary_contract_version: str
    target_framework: str
    target_dtype_intent: str
    target_device_intent: str
    target_layout_kind: str
    batch_size: int
    input_flat_dim: int
    full_vector_length: int
    shape_tuple: Tuple[int, int]
    requires_grad: bool
    allow_tensor_materialization_in_p38: bool
    allow_array_materialization_in_p38: bool
    allow_forward_execution_in_p38: bool
    allow_output_generation_in_p38: bool
    allow_training_in_p38: bool
    reason: str


@dataclasses.dataclass(frozen=True)
class FCVAETensorMaterializationSpec:
    contract_version: str
    target_framework: str
    target_dtype_intent: str
    target_device_intent: str
    target_layout_kind: str
    shape_tuple: Tuple[int, int]
    batch_size: int
    input_flat_dim: int
    full_vector_length: int
    source_nested_row_count: int
    source_nested_row_lengths: Tuple[int, int]
    source_total_scalar_count: int
    source_flattened_matches_p35: bool
    requires_grad: bool
    tensor_request_declared: bool
    tensor_materialized: bool
    array_materialized: bool
    forward_executed: bool
    output_generated: bool
    training_executed: bool
    reason: str


@dataclasses.dataclass(frozen=True)
class FCVAETensorMaterializationMetadata:
    contract_version: str
    source_nested_values_contract_version: str
    source_torch_boundary_contract_version: str
    nested_values_status: str
    torch_backend_name: str
    torch_policy: str
    torch_available: bool
    torch_import_safe: bool
    top_level_torch_import_required: bool
    nested_values_available_in_p37: bool
    nested_values_materialized_in_p37: bool
    shape_tuple: Tuple[int, int]
    batch_size: int
    input_flat_dim: int
    full_vector_length: int
    tensor_materialization_attempted: bool
    array_materialization_attempted: bool
    forward_execution_attempted: bool
    output_generation_attempted: bool
    training_attempted: bool
    reason: str


@dataclasses.dataclass(frozen=True)
class FCVAETensorMaterializationRequestResult:
    contract_version: str
    request: FCVAETensorMaterializationRequest
    spec: FCVAETensorMaterializationSpec
    metadata: FCVAETensorMaterializationMetadata
    status: str
    tensor_request_available_in_p38: bool
    tensor_materialization_available_in_p38: bool
    array_materialization_available_in_p38: bool
    forward_execution_available_in_p38: bool
    output_generation_available_in_p38: bool
    training_available_in_p38: bool
    no_tensor_created: bool
    no_array_created: bool
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


def validate_target_framework(value: Any) -> None:
    if type(value) is not str:
        raise TypeError("target_framework must be exact str")
    if value != FC_VAE_TENSOR_TARGET_FRAMEWORK:
        raise ValueError(f"target_framework must be '{FC_VAE_TENSOR_TARGET_FRAMEWORK}', got '{value}'")


def validate_target_dtype_intent(value: Any) -> None:
    if type(value) is not str:
        raise TypeError("target_dtype_intent must be exact str")
    if value != FC_VAE_TENSOR_TARGET_DTYPE_INTENT:
        raise ValueError(f"target_dtype_intent must be '{FC_VAE_TENSOR_TARGET_DTYPE_INTENT}', got '{value}'")


def validate_target_device_intent(value: Any) -> None:
    if type(value) is not str:
        raise TypeError("target_device_intent must be exact str")
    if value != FC_VAE_TENSOR_TARGET_DEVICE_INTENT:
        raise ValueError(f"target_device_intent must be '{FC_VAE_TENSOR_TARGET_DEVICE_INTENT}', got '{value}'")


def validate_target_layout_kind(value: Any) -> None:
    if type(value) is not str:
        raise TypeError("target_layout_kind must be exact str")
    if value != FC_VAE_TENSOR_LAYOUT_KIND:
        raise ValueError(f"target_layout_kind must be '{FC_VAE_TENSOR_LAYOUT_KIND}', got '{value}'")


def validate_tensor_materialization_request_status(value: Any) -> None:
    if type(value) is not str:
        raise TypeError(f"status must be exact str, got {type(value).__name__}")
    if value not in SUPPORTED_FC_VAE_TENSOR_MATERIALIZATION_REQUEST_STATUSES:
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
                   .replace("tensor_request_metadata_only_no_allocation_in_p38", ""))
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


def validate_tensor_materialization_request(request: FCVAETensorMaterializationRequest) -> None:
    if type(request) is not FCVAETensorMaterializationRequest:
        raise TypeError("request must be exact FCVAETensorMaterializationRequest instance")
    validate_non_empty_str(request.contract_version, "contract_version")
    if request.contract_version != FC_VAE_TENSOR_MATERIALIZATION_REQUEST_CONTRACT_VERSION:
        raise ValueError(f"Invalid contract_version: {request.contract_version}")
    validate_non_empty_str(request.request_kind, "request_kind")
    if request.request_kind != FC_VAE_TENSOR_MATERIALIZATION_REQUEST_KIND:
        raise ValueError(f"Invalid request_kind: {request.request_kind}")
    validate_non_empty_str(request.architecture_id, "architecture_id")
    if request.architecture_id != "FC-VAE":
        raise ValueError(f"Invalid architecture_id: {request.architecture_id}")
    validate_non_empty_str(request.source_nested_values_contract_version, "source_nested_values_contract_version")
    if request.source_nested_values_contract_version != FC_VAE_NESTED_BATCH_VALUES_CONTRACT_VERSION:
        raise ValueError(f"Invalid source_nested_values_contract_version: {request.source_nested_values_contract_version}")
    validate_non_empty_str(request.source_torch_boundary_contract_version, "source_torch_boundary_contract_version")
    if request.source_torch_boundary_contract_version != TORCH_BOUNDARY_CONTRACT_VERSION:
        raise ValueError(f"Invalid source_torch_boundary_contract_version: {request.source_torch_boundary_contract_version}")

    validate_target_framework(request.target_framework)
    validate_target_dtype_intent(request.target_dtype_intent)
    validate_target_device_intent(request.target_device_intent)
    validate_target_layout_kind(request.target_layout_kind)

    validate_positive_int(request.batch_size, "batch_size")
    if request.batch_size != 2:
        raise ValueError("batch_size must be 2")
    validate_positive_int(request.input_flat_dim, "input_flat_dim")
    if request.input_flat_dim != 32:
        raise ValueError("input_flat_dim must be 32")
    validate_full_vector_length(request.full_vector_length)
    validate_shape_tuple(request.shape_tuple)

    validate_bool(request.requires_grad, "requires_grad")
    if request.requires_grad:
        raise ValueError("requires_grad must be False")

    for flag_name in (
        "allow_tensor_materialization_in_p38",
        "allow_array_materialization_in_p38",
        "allow_forward_execution_in_p38",
        "allow_output_generation_in_p38",
        "allow_training_in_p38",
    ):
        val = getattr(request, flag_name)
        validate_bool(val, flag_name)
        if val:
            raise ValueError(f"{flag_name} must be False")

    validate_non_empty_str(request.reason, "reason")
    assert_no_local_path_leakage(request.reason)
    assert_no_forbidden_claims(request.reason)


def validate_tensor_materialization_spec(spec: FCVAETensorMaterializationSpec) -> None:
    if type(spec) is not FCVAETensorMaterializationSpec:
        raise TypeError("spec must be exact FCVAETensorMaterializationSpec instance")
    validate_non_empty_str(spec.contract_version, "contract_version")
    if spec.contract_version != FC_VAE_TENSOR_MATERIALIZATION_REQUEST_CONTRACT_VERSION:
        raise ValueError(f"Invalid contract_version: {spec.contract_version}")

    validate_target_framework(spec.target_framework)
    validate_target_dtype_intent(spec.target_dtype_intent)
    validate_target_device_intent(spec.target_device_intent)
    validate_target_layout_kind(spec.target_layout_kind)
    validate_shape_tuple(spec.shape_tuple)

    validate_positive_int(spec.batch_size, "batch_size")
    if spec.batch_size != 2:
        raise ValueError("batch_size must be 2")
    validate_positive_int(spec.input_flat_dim, "input_flat_dim")
    if spec.input_flat_dim != 32:
        raise ValueError("input_flat_dim must be 32")
    validate_full_vector_length(spec.full_vector_length)

    validate_positive_int(spec.source_nested_row_count, "source_nested_row_count")
    if spec.source_nested_row_count != 2:
        raise ValueError("source_nested_row_count must be 2")

    if type(spec.source_nested_row_lengths) is not tuple:
        raise TypeError("source_nested_row_lengths must be exact tuple")
    if spec.source_nested_row_lengths != (32, 32):
        raise ValueError("source_nested_row_lengths must be (32, 32)")

    validate_positive_int(spec.source_total_scalar_count, "source_total_scalar_count")
    if spec.source_total_scalar_count != 64:
        raise ValueError("source_total_scalar_count must be 64")

    validate_bool(spec.source_flattened_matches_p35, "source_flattened_matches_p35")
    if not spec.source_flattened_matches_p35:
        raise ValueError("source_flattened_matches_p35 must be True")

    validate_bool(spec.requires_grad, "requires_grad")
    if spec.requires_grad:
        raise ValueError("requires_grad must be False")

    validate_bool(spec.tensor_request_declared, "tensor_request_declared")
    if not spec.tensor_request_declared:
        raise ValueError("tensor_request_declared must be True")

    for flag_name in (
        "tensor_materialized",
        "array_materialized",
        "forward_executed",
        "output_generated",
        "training_executed",
    ):
        val = getattr(spec, flag_name)
        validate_bool(val, flag_name)
        if val:
            raise ValueError(f"{flag_name} must be False")

    validate_non_empty_str(spec.reason, "reason")
    assert_no_local_path_leakage(spec.reason)
    assert_no_forbidden_claims(spec.reason)


def validate_tensor_materialization_metadata(metadata: FCVAETensorMaterializationMetadata) -> None:
    if type(metadata) is not FCVAETensorMaterializationMetadata:
        raise TypeError("metadata must be exact FCVAETensorMaterializationMetadata instance")
    validate_non_empty_str(metadata.contract_version, "contract_version")
    if metadata.contract_version != FC_VAE_TENSOR_MATERIALIZATION_REQUEST_CONTRACT_VERSION:
        raise ValueError(f"Invalid contract_version: {metadata.contract_version}")
    validate_non_empty_str(metadata.source_nested_values_contract_version, "source_nested_values_contract_version")
    if metadata.source_nested_values_contract_version != FC_VAE_NESTED_BATCH_VALUES_CONTRACT_VERSION:
        raise ValueError(f"Invalid source_nested_values_contract_version: {metadata.source_nested_values_contract_version}")
    validate_non_empty_str(metadata.source_torch_boundary_contract_version, "source_torch_boundary_contract_version")
    if metadata.source_torch_boundary_contract_version != TORCH_BOUNDARY_CONTRACT_VERSION:
        raise ValueError(f"Invalid source_torch_boundary_contract_version: {metadata.source_torch_boundary_contract_version}")

    validate_non_empty_str(metadata.nested_values_status, "nested_values_status")
    validate_target_framework(metadata.torch_backend_name)

    validate_non_empty_str(metadata.torch_policy, "torch_policy")
    if metadata.torch_policy != TORCH_POLICY_OPTIONAL:
        raise ValueError(f"torch_policy must be '{TORCH_POLICY_OPTIONAL}', got '{metadata.torch_policy}'")

    validate_bool(metadata.torch_available, "torch_available")
    validate_bool(metadata.torch_import_safe, "torch_import_safe")
    validate_bool(metadata.top_level_torch_import_required, "top_level_torch_import_required")
    if metadata.top_level_torch_import_required:
        raise ValueError("top_level_torch_import_required must be False")

    validate_bool(metadata.nested_values_available_in_p37, "nested_values_available_in_p37")
    if not metadata.nested_values_available_in_p37:
        raise ValueError("nested_values_available_in_p37 must be True")

    validate_bool(metadata.nested_values_materialized_in_p37, "nested_values_materialized_in_p37")
    if not metadata.nested_values_materialized_in_p37:
        raise ValueError("nested_values_materialized_in_p37 must be True")

    validate_shape_tuple(metadata.shape_tuple)
    validate_positive_int(metadata.batch_size, "batch_size")
    if metadata.batch_size != 2:
        raise ValueError("batch_size must be 2")
    validate_positive_int(metadata.input_flat_dim, "input_flat_dim")
    if metadata.input_flat_dim != 32:
        raise ValueError("input_flat_dim must be 32")
    validate_full_vector_length(metadata.full_vector_length)

    for flag_name in (
        "tensor_materialization_attempted",
        "array_materialization_attempted",
        "forward_execution_attempted",
        "output_generation_attempted",
        "training_attempted",
    ):
        val = getattr(metadata, flag_name)
        validate_bool(val, flag_name)
        if val:
            raise ValueError(f"{flag_name} must be False")

    validate_non_empty_str(metadata.reason, "reason")
    assert_no_local_path_leakage(metadata.reason)
    assert_no_forbidden_claims(metadata.reason)


def validate_tensor_materialization_request_result(result: FCVAETensorMaterializationRequestResult) -> None:
    if type(result) is not FCVAETensorMaterializationRequestResult:
        raise TypeError("result must be exact FCVAETensorMaterializationRequestResult instance")
    validate_non_empty_str(result.contract_version, "contract_version")
    if result.contract_version != FC_VAE_TENSOR_MATERIALIZATION_REQUEST_CONTRACT_VERSION:
        raise ValueError(f"Invalid contract_version: {result.contract_version}")

    validate_tensor_materialization_request(result.request)
    validate_tensor_materialization_spec(result.spec)
    validate_tensor_materialization_metadata(result.metadata)
    validate_tensor_materialization_request_status(result.status)

    validate_bool(result.tensor_request_available_in_p38, "tensor_request_available_in_p38")
    if not result.tensor_request_available_in_p38:
        raise ValueError("tensor_request_available_in_p38 must be True")

    for flag_name in (
        "tensor_materialization_available_in_p38",
        "array_materialization_available_in_p38",
        "forward_execution_available_in_p38",
        "output_generation_available_in_p38",
        "training_available_in_p38",
    ):
        val = getattr(result, flag_name)
        validate_bool(val, flag_name)
        if val:
            raise ValueError(f"{flag_name} must be False")

    for flag_name in (
        "no_tensor_created",
        "no_array_created",
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
        if result.status != FC_VAE_TENSOR_MATERIALIZATION_REQUEST_STATUS_TORCH_UNAVAILABLE:
            raise ValueError("Inconsistent status: must be blocked_torch_unavailable")
    elif not result.metadata.nested_values_available_in_p37:
        if result.status != FC_VAE_TENSOR_MATERIALIZATION_REQUEST_STATUS_NESTED_VALUES_BLOCKED:
            raise ValueError("Inconsistent status: must be blocked_by_nested_values_status")
    else:
        if result.status != FC_VAE_TENSOR_MATERIALIZATION_REQUEST_STATUS_REQUEST_ONLY:
            raise ValueError("Inconsistent status: must be tensor_request_metadata_only_no_allocation_in_p38")

    validate_non_empty_str(result.reason, "reason")
    assert_no_local_path_leakage(result.reason)
    assert_no_forbidden_claims(result.reason)


# Builders
def build_tensor_materialization_request_from_p37_default() -> FCVAETensorMaterializationRequest:
    p37_res = run_nested_batch_values_probe()

    req = FCVAETensorMaterializationRequest(
        contract_version=FC_VAE_TENSOR_MATERIALIZATION_REQUEST_CONTRACT_VERSION,
        request_kind=FC_VAE_TENSOR_MATERIALIZATION_REQUEST_KIND,
        architecture_id="FC-VAE",
        source_nested_values_contract_version=p37_res.contract_version,
        source_torch_boundary_contract_version=TORCH_BOUNDARY_CONTRACT_VERSION,
        target_framework=FC_VAE_TENSOR_TARGET_FRAMEWORK,
        target_dtype_intent=FC_VAE_TENSOR_TARGET_DTYPE_INTENT,
        target_device_intent=FC_VAE_TENSOR_TARGET_DEVICE_INTENT,
        target_layout_kind=FC_VAE_TENSOR_LAYOUT_KIND,
        batch_size=p37_res.nested_values.batch_size,
        input_flat_dim=p37_res.nested_values.input_flat_dim,
        full_vector_length=p37_res.nested_values.full_vector_length,
        shape_tuple=p37_res.nested_values.shape_tuple,
        requires_grad=DEFAULT_P38_REQUIRES_GRAD,
        allow_tensor_materialization_in_p38=False,
        allow_array_materialization_in_p38=False,
        allow_forward_execution_in_p38=False,
        allow_output_generation_in_p38=False,
        allow_training_in_p38=False,
        reason="p38_tensor_materialization_request_from_p37_default",
    )
    validate_tensor_materialization_request(req)
    return req


def build_tensor_materialization_spec(
    request: FCVAETensorMaterializationRequest,
) -> FCVAETensorMaterializationSpec:
    validate_tensor_materialization_request(request)
    p37_res = run_nested_batch_values_probe()

    spec = FCVAETensorMaterializationSpec(
        contract_version=FC_VAE_TENSOR_MATERIALIZATION_REQUEST_CONTRACT_VERSION,
        target_framework=request.target_framework,
        target_dtype_intent=request.target_dtype_intent,
        target_device_intent=request.target_device_intent,
        target_layout_kind=request.target_layout_kind,
        shape_tuple=request.shape_tuple,
        batch_size=request.batch_size,
        input_flat_dim=request.input_flat_dim,
        full_vector_length=request.full_vector_length,
        source_nested_row_count=p37_res.nested_values.row_count,
        source_nested_row_lengths=p37_res.nested_values.row_lengths,
        source_total_scalar_count=p37_res.nested_values.total_scalar_count,
        source_flattened_matches_p35=p37_res.nested_values.flattened_matches_source_flat_vector,
        requires_grad=request.requires_grad,
        tensor_request_declared=True,
        tensor_materialized=False,
        array_materialized=False,
        forward_executed=False,
        output_generated=False,
        training_executed=False,
        reason="p38_tensor_materialization_spec",
    )
    validate_tensor_materialization_spec(spec)
    return spec


def build_tensor_materialization_metadata(
    request: FCVAETensorMaterializationRequest,
) -> FCVAETensorMaterializationMetadata:
    validate_tensor_materialization_request(request)
    p37_res = run_nested_batch_values_probe()
    torch_status = build_torch_dependency_status(policy=TORCH_POLICY_OPTIONAL)

    meta = FCVAETensorMaterializationMetadata(
        contract_version=FC_VAE_TENSOR_MATERIALIZATION_REQUEST_CONTRACT_VERSION,
        source_nested_values_contract_version=request.source_nested_values_contract_version,
        source_torch_boundary_contract_version=request.source_torch_boundary_contract_version,
        nested_values_status=p37_res.status,
        torch_backend_name=torch_status.backend_name,
        torch_policy=torch_status.policy,
        torch_available=torch_status.available,
        torch_import_safe=torch_status.import_safe,
        top_level_torch_import_required=torch_status.top_level_import_required,
        nested_values_available_in_p37=p37_res.nested_values_available_in_p37,
        nested_values_materialized_in_p37=p37_res.metadata.nested_values_materialized,
        shape_tuple=request.shape_tuple,
        batch_size=request.batch_size,
        input_flat_dim=request.input_flat_dim,
        full_vector_length=request.full_vector_length,
        tensor_materialization_attempted=False,
        array_materialization_attempted=False,
        forward_execution_attempted=False,
        output_generation_attempted=False,
        training_attempted=False,
        reason="p38_tensor_materialization_metadata",
    )
    validate_tensor_materialization_metadata(meta)
    return meta


def build_tensor_materialization_request_result(
    request: FCVAETensorMaterializationRequest,
) -> FCVAETensorMaterializationRequestResult:
    validate_tensor_materialization_request(request)
    spec = build_tensor_materialization_spec(request)
    metadata = build_tensor_materialization_metadata(request)

    if not metadata.torch_available:
        status = FC_VAE_TENSOR_MATERIALIZATION_REQUEST_STATUS_TORCH_UNAVAILABLE
        reason = "tensor_request_blocked_torch_unavailable"
    elif not metadata.nested_values_available_in_p37:
        status = FC_VAE_TENSOR_MATERIALIZATION_REQUEST_STATUS_NESTED_VALUES_BLOCKED
        reason = "tensor_request_blocked_by_nested_values_status"
    else:
        status = FC_VAE_TENSOR_MATERIALIZATION_REQUEST_STATUS_REQUEST_ONLY
        reason = "tensor_request_metadata_only_no_allocation_in_p38"

    res = FCVAETensorMaterializationRequestResult(
        contract_version=FC_VAE_TENSOR_MATERIALIZATION_REQUEST_CONTRACT_VERSION,
        request=request,
        spec=spec,
        metadata=metadata,
        status=status,
        tensor_request_available_in_p38=True,
        tensor_materialization_available_in_p38=False,
        array_materialization_available_in_p38=False,
        forward_execution_available_in_p38=False,
        output_generation_available_in_p38=False,
        training_available_in_p38=False,
        no_tensor_created=True,
        no_array_created=True,
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
    validate_tensor_materialization_request_result(res)
    return res


def run_tensor_materialization_request_probe() -> FCVAETensorMaterializationRequestResult:
    req = build_tensor_materialization_request_from_p37_default()
    res = build_tensor_materialization_request_result(req)
    validate_tensor_materialization_request_result(res)
    return res


# Serialization
def tensor_materialization_request_to_json_dict(
    request: FCVAETensorMaterializationRequest,
) -> Dict[str, Any]:
    validate_tensor_materialization_request(request)
    d = {
        "contract_version": request.contract_version,
        "request_kind": request.request_kind,
        "architecture_id": request.architecture_id,
        "source_nested_values_contract_version": request.source_nested_values_contract_version,
        "source_torch_boundary_contract_version": request.source_torch_boundary_contract_version,
        "target_framework": request.target_framework,
        "target_dtype_intent": request.target_dtype_intent,
        "target_device_intent": request.target_device_intent,
        "target_layout_kind": request.target_layout_kind,
        "batch_size": request.batch_size,
        "input_flat_dim": request.input_flat_dim,
        "full_vector_length": request.full_vector_length,
        "shape_tuple": list(request.shape_tuple),
        "requires_grad": request.requires_grad,
        "allow_tensor_materialization_in_p38": request.allow_tensor_materialization_in_p38,
        "allow_array_materialization_in_p38": request.allow_array_materialization_in_p38,
        "allow_forward_execution_in_p38": request.allow_forward_execution_in_p38,
        "allow_output_generation_in_p38": request.allow_output_generation_in_p38,
        "allow_training_in_p38": request.allow_training_in_p38,
        "reason": request.reason,
    }
    assert_no_local_path_leakage(d)
    assert_no_forbidden_claims(d)
    return d


def tensor_materialization_spec_to_json_dict(
    spec: FCVAETensorMaterializationSpec,
) -> Dict[str, Any]:
    validate_tensor_materialization_spec(spec)
    d = {
        "contract_version": spec.contract_version,
        "target_framework": spec.target_framework,
        "target_dtype_intent": spec.target_dtype_intent,
        "target_device_intent": spec.target_device_intent,
        "target_layout_kind": spec.target_layout_kind,
        "shape_tuple": list(spec.shape_tuple),
        "batch_size": spec.batch_size,
        "input_flat_dim": spec.input_flat_dim,
        "full_vector_length": spec.full_vector_length,
        "source_nested_row_count": spec.source_nested_row_count,
        "source_nested_row_lengths": list(spec.source_nested_row_lengths),
        "source_total_scalar_count": spec.source_total_scalar_count,
        "source_flattened_matches_p35": spec.source_flattened_matches_p35,
        "requires_grad": spec.requires_grad,
        "tensor_request_declared": spec.tensor_request_declared,
        "tensor_materialized": spec.tensor_materialized,
        "array_materialized": spec.array_materialized,
        "forward_executed": spec.forward_executed,
        "output_generated": spec.output_generated,
        "training_executed": spec.training_executed,
        "reason": spec.reason,
    }
    assert_no_local_path_leakage(d)
    assert_no_forbidden_claims(d)
    return d


def tensor_materialization_metadata_to_json_dict(
    metadata: FCVAETensorMaterializationMetadata,
) -> Dict[str, Any]:
    validate_tensor_materialization_metadata(metadata)
    d = {
        "contract_version": metadata.contract_version,
        "source_nested_values_contract_version": metadata.source_nested_values_contract_version,
        "source_torch_boundary_contract_version": metadata.source_torch_boundary_contract_version,
        "nested_values_status": metadata.nested_values_status,
        "torch_backend_name": metadata.torch_backend_name,
        "torch_policy": metadata.torch_policy,
        "torch_available": metadata.torch_available,
        "torch_import_safe": metadata.torch_import_safe,
        "top_level_torch_import_required": metadata.top_level_torch_import_required,
        "nested_values_available_in_p37": metadata.nested_values_available_in_p37,
        "nested_values_materialized_in_p37": metadata.nested_values_materialized_in_p37,
        "shape_tuple": list(metadata.shape_tuple),
        "batch_size": metadata.batch_size,
        "input_flat_dim": metadata.input_flat_dim,
        "full_vector_length": metadata.full_vector_length,
        "tensor_materialization_attempted": metadata.tensor_materialization_attempted,
        "array_materialization_attempted": metadata.array_materialization_attempted,
        "forward_execution_attempted": metadata.forward_execution_attempted,
        "output_generation_attempted": metadata.output_generation_attempted,
        "training_attempted": metadata.training_attempted,
        "reason": metadata.reason,
    }
    assert_no_local_path_leakage(d)
    assert_no_forbidden_claims(d)
    return d


def tensor_materialization_request_result_to_json_dict(
    result: FCVAETensorMaterializationRequestResult,
) -> Dict[str, Any]:
    validate_tensor_materialization_request_result(result)
    d = {
        "contract_version": result.contract_version,
        "request": tensor_materialization_request_to_json_dict(result.request),
        "spec": tensor_materialization_spec_to_json_dict(result.spec),
        "metadata": tensor_materialization_metadata_to_json_dict(result.metadata),
        "status": result.status,
        "tensor_request_available_in_p38": result.tensor_request_available_in_p38,
        "tensor_materialization_available_in_p38": result.tensor_materialization_available_in_p38,
        "array_materialization_available_in_p38": result.array_materialization_available_in_p38,
        "forward_execution_available_in_p38": result.forward_execution_available_in_p38,
        "output_generation_available_in_p38": result.output_generation_available_in_p38,
        "training_available_in_p38": result.training_available_in_p38,
        "no_tensor_created": result.no_tensor_created,
        "no_array_created": result.no_array_created,
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


def compact_tensor_materialization_request_json(
    result: FCVAETensorMaterializationRequestResult,
) -> str:
    d = tensor_materialization_request_result_to_json_dict(result)
    return json.dumps(d, sort_keys=True, separators=(",", ":"))
