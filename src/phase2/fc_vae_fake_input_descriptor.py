# src/phase2/fc_vae_fake_input_descriptor.py

import dataclasses
import json
from typing import Any, Dict, Tuple, Union

from src.phase2.fc_vae_forward_input_batch import (
    FC_VAE_FORWARD_INPUT_BATCH_CONTRACT_VERSION,
    run_forward_input_batch_probe,
    forward_input_batch_result_to_json_dict,
)


# Public Constants
FC_VAE_FAKE_INPUT_DESCRIPTOR_CONTRACT_VERSION = "phase2_p33_fake_input_descriptor_contract_v1"

FC_VAE_FAKE_INPUT_DESCRIPTOR_KIND = "deterministic_fake_input_descriptor_no_values"

FC_VAE_FAKE_INPUT_DESCRIPTOR_MODULE_NAME = "src.phase2.fc_vae_fake_input_descriptor"

FC_VAE_FAKE_INPUT_DESCRIPTOR_STATUS_TORCH_UNAVAILABLE = "blocked_torch_unavailable"

FC_VAE_FAKE_INPUT_DESCRIPTOR_STATUS_BATCH_UNAVAILABLE = "blocked_input_batch_unavailable"

FC_VAE_FAKE_INPUT_DESCRIPTOR_STATUS_DESCRIPTOR_ONLY = "descriptor_only_no_values_in_p33"

SUPPORTED_FC_VAE_FAKE_INPUT_DESCRIPTOR_STATUSES = (
    "blocked_torch_unavailable",
    "blocked_input_batch_unavailable",
    "descriptor_only_no_values_in_p33",
)

FC_VAE_FAKE_INPUT_DESCRIPTOR_SOURCE_KIND = "synthetic_descriptor_only"

FC_VAE_FAKE_INPUT_DESCRIPTOR_DISTRIBUTION_KIND = "bounded_uniform_descriptor"

FC_VAE_FAKE_INPUT_DESCRIPTOR_DTYPE_INTENT = "float32_future_tensor_intent"

# Defaults
DEFAULT_P33_DESCRIPTOR_SEED = 1337
DEFAULT_P33_MIN_VALUE = -1.0
DEFAULT_P33_MAX_VALUE = 1.0


# Public Dataclasses (All Frozen)
@dataclasses.dataclass(frozen=True)
class FCVAEFakeInputDescriptorRequest:
    contract_version: str
    descriptor_kind: str
    architecture_id: str
    source_kind: str
    distribution_kind: str
    dtype_intent: str
    descriptor_seed: int
    min_value: float
    max_value: float
    batch_size: int
    input_flat_dim: int
    source_input_batch_contract_version: str
    allow_rng_execution_in_p33: bool
    allow_value_materialization_in_p33: bool
    allow_array_materialization_in_p33: bool
    allow_tensor_materialization_in_p33: bool
    allow_forward_execution_in_p33: bool
    reason: str


@dataclasses.dataclass(frozen=True)
class FCVAEFakeInputDescriptorShape:
    contract_version: str
    descriptor_kind: str
    shape_kind: str
    batch_size: int
    input_flat_dim: int
    rank: int
    shape_tuple: Tuple[int, int]
    source_batch_shape_tuple: Tuple[int, int]
    shape_matches_p32: bool
    reason: str


@dataclasses.dataclass(frozen=True)
class FCVAEFakeInputDescriptorPolicy:
    contract_version: str
    source_kind: str
    distribution_kind: str
    dtype_intent: str
    descriptor_seed: int
    min_value: float
    max_value: float
    rng_executed: bool
    values_materialized: bool
    array_materialized: bool
    tensor_materialized: bool
    reason: str


@dataclasses.dataclass(frozen=True)
class FCVAEFakeInputDescriptorMetadata:
    contract_version: str
    descriptor_kind: str
    input_batch_contract_version: str
    input_batch_status: str
    torch_available: bool
    input_batch_available_in_p32: bool
    batch_shape_declared: bool
    descriptor_created: bool
    rng_execution_attempted: bool
    value_materialization_attempted: bool
    array_materialization_attempted: bool
    tensor_materialization_attempted: bool
    forward_execution_attempted: bool
    output_generation_attempted: bool
    reason: str


@dataclasses.dataclass(frozen=True)
class FCVAEFakeInputDescriptorResult:
    contract_version: str
    request: FCVAEFakeInputDescriptorRequest
    shape: FCVAEFakeInputDescriptorShape
    policy: FCVAEFakeInputDescriptorPolicy
    metadata: FCVAEFakeInputDescriptorMetadata
    status: str
    descriptor_available_in_p33: bool
    rng_execution_available_in_p33: bool
    value_materialization_available_in_p33: bool
    array_materialization_available_in_p33: bool
    tensor_materialization_available_in_p33: bool
    forward_execution_available_in_p33: bool
    output_generation_available_in_p33: bool
    no_rng_execution: bool
    no_values_materialized: bool
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


def validate_numeric_value(value: Any, name: str) -> None:
    if type(value) not in (int, float) or isinstance(value, bool):
        raise TypeError(f"{name} must be exact int or float, got {type(value).__name__}")


def validate_value_range(min_val: Any, max_val: Any) -> None:
    validate_numeric_value(min_val, "min_val")
    validate_numeric_value(max_val, "max_val")
    if min_val >= max_val:
        raise ValueError(f"min_value ({min_val}) must be < max_value ({max_val})")


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


def validate_fake_input_descriptor_status(value: Any) -> None:
    if type(value) is not str:
        raise TypeError(f"status must be exact str, got {type(value).__name__}")
    if value not in SUPPORTED_FC_VAE_FAKE_INPUT_DESCRIPTOR_STATUSES:
        raise ValueError(f"Unsupported fake input descriptor status: {value}")


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
                   .replace("descriptor_only_no_values_in_p33", ""))
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


def validate_fake_input_descriptor_request(request: FCVAEFakeInputDescriptorRequest) -> None:
    if type(request) is not FCVAEFakeInputDescriptorRequest:
        raise TypeError("request must be exact FCVAEFakeInputDescriptorRequest instance")

    validate_non_empty_str(request.contract_version, "contract_version")
    if request.contract_version != FC_VAE_FAKE_INPUT_DESCRIPTOR_CONTRACT_VERSION:
        raise ValueError(f"Invalid contract_version: {request.contract_version}")

    validate_non_empty_str(request.descriptor_kind, "descriptor_kind")
    if request.descriptor_kind != FC_VAE_FAKE_INPUT_DESCRIPTOR_KIND:
        raise ValueError(f"Invalid descriptor_kind: {request.descriptor_kind}")

    validate_non_empty_str(request.architecture_id, "architecture_id")
    if request.architecture_id != "FC-VAE":
        raise ValueError(f"Invalid architecture_id: {request.architecture_id}")

    validate_non_empty_str(request.source_kind, "source_kind")
    if request.source_kind != FC_VAE_FAKE_INPUT_DESCRIPTOR_SOURCE_KIND:
        raise ValueError(f"Invalid source_kind: {request.source_kind}")

    validate_non_empty_str(request.distribution_kind, "distribution_kind")
    if request.distribution_kind != FC_VAE_FAKE_INPUT_DESCRIPTOR_DISTRIBUTION_KIND:
        raise ValueError(f"Invalid distribution_kind: {request.distribution_kind}")

    validate_non_empty_str(request.dtype_intent, "dtype_intent")
    if request.dtype_intent != FC_VAE_FAKE_INPUT_DESCRIPTOR_DTYPE_INTENT:
        raise ValueError(f"Invalid dtype_intent: {request.dtype_intent}")

    validate_non_negative_int(request.descriptor_seed, "descriptor_seed")
    validate_value_range(request.min_value, request.max_value)
    validate_positive_int(request.batch_size, "batch_size")
    validate_positive_int(request.input_flat_dim, "input_flat_dim")

    validate_non_empty_str(request.source_input_batch_contract_version, "source_input_batch_contract_version")
    if request.source_input_batch_contract_version != FC_VAE_FORWARD_INPUT_BATCH_CONTRACT_VERSION:
        raise ValueError(f"Invalid source_input_batch_contract_version: {request.source_input_batch_contract_version}")

    for flag_name in (
        "allow_rng_execution_in_p33",
        "allow_value_materialization_in_p33",
        "allow_array_materialization_in_p33",
        "allow_tensor_materialization_in_p33",
        "allow_forward_execution_in_p33",
    ):
        val = getattr(request, flag_name)
        validate_bool(val, flag_name)
        if val:
            raise ValueError(f"{flag_name} must be False")

    validate_non_empty_str(request.reason, "reason")
    assert_no_local_path_leakage(request.reason)
    assert_no_forbidden_claims(request.reason)


def validate_fake_input_descriptor_shape(shape: FCVAEFakeInputDescriptorShape) -> None:
    if type(shape) is not FCVAEFakeInputDescriptorShape:
        raise TypeError("shape must be exact FCVAEFakeInputDescriptorShape instance")

    validate_non_empty_str(shape.contract_version, "contract_version")
    if shape.contract_version != FC_VAE_FAKE_INPUT_DESCRIPTOR_CONTRACT_VERSION:
        raise ValueError(f"Invalid contract_version: {shape.contract_version}")

    validate_non_empty_str(shape.descriptor_kind, "descriptor_kind")
    if shape.descriptor_kind != FC_VAE_FAKE_INPUT_DESCRIPTOR_KIND:
        raise ValueError(f"Invalid descriptor_kind: {shape.descriptor_kind}")

    validate_non_empty_str(shape.shape_kind, "shape_kind")
    validate_positive_int(shape.batch_size, "batch_size")
    validate_positive_int(shape.input_flat_dim, "input_flat_dim")

    validate_positive_int(shape.rank, "rank")
    if shape.rank != 2:
        raise ValueError("rank must be exactly 2")

    validate_shape_tuple(shape.shape_tuple)
    if shape.shape_tuple != (shape.batch_size, shape.input_flat_dim):
        raise ValueError(f"shape_tuple must equal (batch_size, input_flat_dim), got {shape.shape_tuple}")

    validate_shape_tuple(shape.source_batch_shape_tuple)
    if shape.source_batch_shape_tuple != shape.shape_tuple:
        raise ValueError(f"source_batch_shape_tuple must equal shape_tuple, got {shape.source_batch_shape_tuple}")

    validate_bool(shape.shape_matches_p32, "shape_matches_p32")
    if not shape.shape_matches_p32:
        raise ValueError("shape_matches_p32 must be True")

    validate_non_empty_str(shape.reason, "reason")
    assert_no_local_path_leakage(shape.reason)
    assert_no_forbidden_claims(shape.reason)


def validate_fake_input_descriptor_policy(policy: FCVAEFakeInputDescriptorPolicy) -> None:
    if type(policy) is not FCVAEFakeInputDescriptorPolicy:
        raise TypeError("policy must be exact FCVAEFakeInputDescriptorPolicy instance")

    validate_non_empty_str(policy.contract_version, "contract_version")
    if policy.contract_version != FC_VAE_FAKE_INPUT_DESCRIPTOR_CONTRACT_VERSION:
        raise ValueError(f"Invalid contract_version: {policy.contract_version}")

    validate_non_empty_str(policy.source_kind, "source_kind")
    if policy.source_kind != FC_VAE_FAKE_INPUT_DESCRIPTOR_SOURCE_KIND:
        raise ValueError(f"Invalid source_kind: {policy.source_kind}")

    validate_non_empty_str(policy.distribution_kind, "distribution_kind")
    if policy.distribution_kind != FC_VAE_FAKE_INPUT_DESCRIPTOR_DISTRIBUTION_KIND:
        raise ValueError(f"Invalid distribution_kind: {policy.distribution_kind}")

    validate_non_empty_str(policy.dtype_intent, "dtype_intent")
    if policy.dtype_intent != FC_VAE_FAKE_INPUT_DESCRIPTOR_DTYPE_INTENT:
        raise ValueError(f"Invalid dtype_intent: {policy.dtype_intent}")

    validate_non_negative_int(policy.descriptor_seed, "descriptor_seed")
    validate_value_range(policy.min_value, policy.max_value)

    for flag_name in (
        "rng_executed",
        "values_materialized",
        "array_materialized",
        "tensor_materialized",
    ):
        val = getattr(policy, flag_name)
        validate_bool(val, flag_name)
        if val:
            raise ValueError(f"{flag_name} must be False")

    validate_non_empty_str(policy.reason, "reason")
    assert_no_local_path_leakage(policy.reason)
    assert_no_forbidden_claims(policy.reason)


def validate_fake_input_descriptor_metadata(metadata: FCVAEFakeInputDescriptorMetadata) -> None:
    if type(metadata) is not FCVAEFakeInputDescriptorMetadata:
        raise TypeError("metadata must be exact FCVAEFakeInputDescriptorMetadata instance")

    validate_non_empty_str(metadata.contract_version, "contract_version")
    if metadata.contract_version != FC_VAE_FAKE_INPUT_DESCRIPTOR_CONTRACT_VERSION:
        raise ValueError(f"Invalid contract_version: {metadata.contract_version}")

    validate_non_empty_str(metadata.descriptor_kind, "descriptor_kind")
    if metadata.descriptor_kind != FC_VAE_FAKE_INPUT_DESCRIPTOR_KIND:
        raise ValueError(f"Invalid descriptor_kind: {metadata.descriptor_kind}")

    validate_non_empty_str(metadata.input_batch_contract_version, "input_batch_contract_version")
    if metadata.input_batch_contract_version != FC_VAE_FORWARD_INPUT_BATCH_CONTRACT_VERSION:
        raise ValueError(f"Invalid input_batch_contract_version: {metadata.input_batch_contract_version}")

    validate_non_empty_str(metadata.input_batch_status, "input_batch_status")
    validate_bool(metadata.torch_available, "torch_available")
    validate_bool(metadata.input_batch_available_in_p32, "input_batch_available_in_p32")
    if metadata.input_batch_available_in_p32:
        raise ValueError("input_batch_available_in_p32 must be False")

    validate_bool(metadata.batch_shape_declared, "batch_shape_declared")
    if not metadata.batch_shape_declared:
        raise ValueError("batch_shape_declared must be True")

    validate_bool(metadata.descriptor_created, "descriptor_created")
    if not metadata.descriptor_created:
        raise ValueError("descriptor_created must be True")

    for flag_name in (
        "rng_execution_attempted",
        "value_materialization_attempted",
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


def validate_fake_input_descriptor_result(result: FCVAEFakeInputDescriptorResult) -> None:
    if type(result) is not FCVAEFakeInputDescriptorResult:
        raise TypeError("result must be exact FCVAEFakeInputDescriptorResult instance")

    validate_non_empty_str(result.contract_version, "contract_version")
    if result.contract_version != FC_VAE_FAKE_INPUT_DESCRIPTOR_CONTRACT_VERSION:
        raise ValueError(f"Invalid contract_version: {result.contract_version}")

    validate_fake_input_descriptor_request(result.request)
    validate_fake_input_descriptor_shape(result.shape)
    validate_fake_input_descriptor_policy(result.policy)
    validate_fake_input_descriptor_metadata(result.metadata)
    validate_fake_input_descriptor_status(result.status)

    validate_bool(result.descriptor_available_in_p33, "descriptor_available_in_p33")
    if not result.descriptor_available_in_p33:
        raise ValueError("descriptor_available_in_p33 must be True")

    for flag_name in (
        "rng_execution_available_in_p33",
        "value_materialization_available_in_p33",
        "array_materialization_available_in_p33",
        "tensor_materialization_available_in_p33",
        "forward_execution_available_in_p33",
        "output_generation_available_in_p33",
    ):
        val = getattr(result, flag_name)
        validate_bool(val, flag_name)
        if val:
            raise ValueError(f"{flag_name} must be False")

    for flag_name in (
        "no_rng_execution",
        "no_values_materialized",
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
        if result.status != FC_VAE_FAKE_INPUT_DESCRIPTOR_STATUS_TORCH_UNAVAILABLE:
            raise ValueError(f"Inconsistent state: status must be '{FC_VAE_FAKE_INPUT_DESCRIPTOR_STATUS_TORCH_UNAVAILABLE}' when torch is unavailable")
    elif not result.metadata.input_batch_available_in_p32:
        if result.status != FC_VAE_FAKE_INPUT_DESCRIPTOR_STATUS_BATCH_UNAVAILABLE:
            raise ValueError(f"Inconsistent state: status must be '{FC_VAE_FAKE_INPUT_DESCRIPTOR_STATUS_BATCH_UNAVAILABLE}' when input batch is unavailable")
    else:
        if result.status != FC_VAE_FAKE_INPUT_DESCRIPTOR_STATUS_DESCRIPTOR_ONLY:
            raise ValueError(f"Inconsistent state: status must be '{FC_VAE_FAKE_INPUT_DESCRIPTOR_STATUS_DESCRIPTOR_ONLY}'")

    validate_non_empty_str(result.reason, "reason")
    assert_no_local_path_leakage(result.reason)
    assert_no_forbidden_claims(result.reason)


# Public Builders
def build_fake_input_descriptor_request_from_p32_default(
    descriptor_seed: int = DEFAULT_P33_DESCRIPTOR_SEED,
    min_value: float = DEFAULT_P33_MIN_VALUE,
    max_value: float = DEFAULT_P33_MAX_VALUE,
) -> FCVAEFakeInputDescriptorRequest:
    p32_result = run_forward_input_batch_probe()

    req = FCVAEFakeInputDescriptorRequest(
        contract_version=FC_VAE_FAKE_INPUT_DESCRIPTOR_CONTRACT_VERSION,
        descriptor_kind=FC_VAE_FAKE_INPUT_DESCRIPTOR_KIND,
        architecture_id="FC-VAE",
        source_kind=FC_VAE_FAKE_INPUT_DESCRIPTOR_SOURCE_KIND,
        distribution_kind=FC_VAE_FAKE_INPUT_DESCRIPTOR_DISTRIBUTION_KIND,
        dtype_intent=FC_VAE_FAKE_INPUT_DESCRIPTOR_DTYPE_INTENT,
        descriptor_seed=descriptor_seed,
        min_value=min_value,
        max_value=max_value,
        batch_size=p32_result.shape.batch_size,
        input_flat_dim=p32_result.shape.input_flat_dim,
        source_input_batch_contract_version=p32_result.contract_version,
        allow_rng_execution_in_p33=False,
        allow_value_materialization_in_p33=False,
        allow_array_materialization_in_p33=False,
        allow_tensor_materialization_in_p33=False,
        allow_forward_execution_in_p33=False,
        reason="p33_fake_input_descriptor_request_from_p32_default",
    )
    validate_fake_input_descriptor_request(req)
    return req


def build_fake_input_descriptor_shape(request: FCVAEFakeInputDescriptorRequest) -> FCVAEFakeInputDescriptorShape:
    validate_fake_input_descriptor_request(request)
    p32_result = run_forward_input_batch_probe()

    if (request.batch_size, request.input_flat_dim) != p32_result.shape.shape_tuple:
        raise ValueError("Request dimensions do not match P32 batch shape")

    shape = FCVAEFakeInputDescriptorShape(
        contract_version=FC_VAE_FAKE_INPUT_DESCRIPTOR_CONTRACT_VERSION,
        descriptor_kind=FC_VAE_FAKE_INPUT_DESCRIPTOR_KIND,
        shape_kind="declared_fake_input_descriptor_shape_no_tensor",
        batch_size=request.batch_size,
        input_flat_dim=request.input_flat_dim,
        rank=2,
        shape_tuple=(request.batch_size, request.input_flat_dim),
        source_batch_shape_tuple=p32_result.shape.shape_tuple,
        shape_matches_p32=True,
        reason="p33_fake_input_descriptor_shape",
    )
    validate_fake_input_descriptor_shape(shape)
    return shape


def build_fake_input_descriptor_policy(request: FCVAEFakeInputDescriptorRequest) -> FCVAEFakeInputDescriptorPolicy:
    validate_fake_input_descriptor_request(request)

    policy = FCVAEFakeInputDescriptorPolicy(
        contract_version=FC_VAE_FAKE_INPUT_DESCRIPTOR_CONTRACT_VERSION,
        source_kind=request.source_kind,
        distribution_kind=request.distribution_kind,
        dtype_intent=request.dtype_intent,
        descriptor_seed=request.descriptor_seed,
        min_value=request.min_value,
        max_value=request.max_value,
        rng_executed=False,
        values_materialized=False,
        array_materialized=False,
        tensor_materialized=False,
        reason="p33_fake_input_descriptor_policy",
    )
    validate_fake_input_descriptor_policy(policy)
    return policy


def build_fake_input_descriptor_metadata(request: FCVAEFakeInputDescriptorRequest) -> FCVAEFakeInputDescriptorMetadata:
    validate_fake_input_descriptor_request(request)
    p32_result = run_forward_input_batch_probe()

    meta = FCVAEFakeInputDescriptorMetadata(
        contract_version=FC_VAE_FAKE_INPUT_DESCRIPTOR_CONTRACT_VERSION,
        descriptor_kind=FC_VAE_FAKE_INPUT_DESCRIPTOR_KIND,
        input_batch_contract_version=p32_result.contract_version,
        input_batch_status=p32_result.status,
        torch_available=p32_result.metadata.torch_available,
        input_batch_available_in_p32=p32_result.input_batch_available_in_p32,
        batch_shape_declared=p32_result.metadata.batch_shape_declared,
        descriptor_created=True,
        rng_execution_attempted=False,
        value_materialization_attempted=False,
        array_materialization_attempted=False,
        tensor_materialization_attempted=False,
        forward_execution_attempted=False,
        output_generation_attempted=False,
        reason="p33_fake_input_descriptor_metadata",
    )
    validate_fake_input_descriptor_metadata(meta)
    return meta


def build_fake_input_descriptor_result(request: FCVAEFakeInputDescriptorRequest) -> FCVAEFakeInputDescriptorResult:
    validate_fake_input_descriptor_request(request)
    shape = build_fake_input_descriptor_shape(request)
    policy = build_fake_input_descriptor_policy(request)
    metadata = build_fake_input_descriptor_metadata(request)

    if not metadata.torch_available:
        status = FC_VAE_FAKE_INPUT_DESCRIPTOR_STATUS_TORCH_UNAVAILABLE
        reason = "fake_input_descriptor_blocked_torch_unavailable"
    elif not metadata.input_batch_available_in_p32:
        status = FC_VAE_FAKE_INPUT_DESCRIPTOR_STATUS_BATCH_UNAVAILABLE
        reason = "fake_input_descriptor_blocked_input_batch_unavailable"
    else:
        status = FC_VAE_FAKE_INPUT_DESCRIPTOR_STATUS_DESCRIPTOR_ONLY
        reason = "fake_input_descriptor_only"

    res = FCVAEFakeInputDescriptorResult(
        contract_version=FC_VAE_FAKE_INPUT_DESCRIPTOR_CONTRACT_VERSION,
        request=request,
        shape=shape,
        policy=policy,
        metadata=metadata,
        status=status,
        descriptor_available_in_p33=True,
        rng_execution_available_in_p33=False,
        value_materialization_available_in_p33=False,
        array_materialization_available_in_p33=False,
        tensor_materialization_available_in_p33=False,
        forward_execution_available_in_p33=False,
        output_generation_available_in_p33=False,
        no_rng_execution=True,
        no_values_materialized=True,
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
    validate_fake_input_descriptor_result(res)
    return res


def run_fake_input_descriptor_probe() -> FCVAEFakeInputDescriptorResult:
    req = build_fake_input_descriptor_request_from_p32_default()
    res = build_fake_input_descriptor_result(req)
    validate_fake_input_descriptor_result(res)
    return res


# Public Serialization
def fake_input_descriptor_request_to_json_dict(request: FCVAEFakeInputDescriptorRequest) -> dict:
    validate_fake_input_descriptor_request(request)
    d = {
        "contract_version": request.contract_version,
        "descriptor_kind": request.descriptor_kind,
        "architecture_id": request.architecture_id,
        "source_kind": request.source_kind,
        "distribution_kind": request.distribution_kind,
        "dtype_intent": request.dtype_intent,
        "descriptor_seed": request.descriptor_seed,
        "min_value": request.min_value,
        "max_value": request.max_value,
        "batch_size": request.batch_size,
        "input_flat_dim": request.input_flat_dim,
        "source_input_batch_contract_version": request.source_input_batch_contract_version,
        "allow_rng_execution_in_p33": request.allow_rng_execution_in_p33,
        "allow_value_materialization_in_p33": request.allow_value_materialization_in_p33,
        "allow_array_materialization_in_p33": request.allow_array_materialization_in_p33,
        "allow_tensor_materialization_in_p33": request.allow_tensor_materialization_in_p33,
        "allow_forward_execution_in_p33": request.allow_forward_execution_in_p33,
        "reason": request.reason,
    }
    assert_no_local_path_leakage(d)
    assert_no_forbidden_claims(d)
    return d


def fake_input_descriptor_shape_to_json_dict(shape: FCVAEFakeInputDescriptorShape) -> dict:
    validate_fake_input_descriptor_shape(shape)
    d = {
        "contract_version": shape.contract_version,
        "descriptor_kind": shape.descriptor_kind,
        "shape_kind": shape.shape_kind,
        "batch_size": shape.batch_size,
        "input_flat_dim": shape.input_flat_dim,
        "rank": shape.rank,
        "shape_tuple": [shape.shape_tuple[0], shape.shape_tuple[1]],
        "source_batch_shape_tuple": [shape.source_batch_shape_tuple[0], shape.source_batch_shape_tuple[1]],
        "shape_matches_p32": shape.shape_matches_p32,
        "reason": shape.reason,
    }
    assert_no_local_path_leakage(d)
    assert_no_forbidden_claims(d)
    return d


def fake_input_descriptor_policy_to_json_dict(policy: FCVAEFakeInputDescriptorPolicy) -> dict:
    validate_fake_input_descriptor_policy(policy)
    d = {
        "contract_version": policy.contract_version,
        "source_kind": policy.source_kind,
        "distribution_kind": policy.distribution_kind,
        "dtype_intent": policy.dtype_intent,
        "descriptor_seed": policy.descriptor_seed,
        "min_value": policy.min_value,
        "max_value": policy.max_value,
        "rng_executed": policy.rng_executed,
        "values_materialized": policy.values_materialized,
        "array_materialized": policy.array_materialized,
        "tensor_materialized": policy.tensor_materialized,
        "reason": policy.reason,
    }
    assert_no_local_path_leakage(d)
    assert_no_forbidden_claims(d)
    return d


def fake_input_descriptor_metadata_to_json_dict(metadata: FCVAEFakeInputDescriptorMetadata) -> dict:
    validate_fake_input_descriptor_metadata(metadata)
    d = {
        "contract_version": metadata.contract_version,
        "descriptor_kind": metadata.descriptor_kind,
        "input_batch_contract_version": metadata.input_batch_contract_version,
        "input_batch_status": metadata.input_batch_status,
        "torch_available": metadata.torch_available,
        "input_batch_available_in_p32": metadata.input_batch_available_in_p32,
        "batch_shape_declared": metadata.batch_shape_declared,
        "descriptor_created": metadata.descriptor_created,
        "rng_execution_attempted": metadata.rng_execution_attempted,
        "value_materialization_attempted": metadata.value_materialization_attempted,
        "array_materialization_attempted": metadata.array_materialization_attempted,
        "tensor_materialization_attempted": metadata.tensor_materialization_attempted,
        "forward_execution_attempted": metadata.forward_execution_attempted,
        "output_generation_attempted": metadata.output_generation_attempted,
        "reason": metadata.reason,
    }
    assert_no_local_path_leakage(d)
    assert_no_forbidden_claims(d)
    return d


def fake_input_descriptor_result_to_json_dict(result: FCVAEFakeInputDescriptorResult) -> dict:
    validate_fake_input_descriptor_result(result)
    d = {
        "contract_version": result.contract_version,
        "request": fake_input_descriptor_request_to_json_dict(result.request),
        "shape": fake_input_descriptor_shape_to_json_dict(result.shape),
        "policy": fake_input_descriptor_policy_to_json_dict(result.policy),
        "metadata": fake_input_descriptor_metadata_to_json_dict(result.metadata),
        "status": result.status,
        "descriptor_available_in_p33": result.descriptor_available_in_p33,
        "rng_execution_available_in_p33": result.rng_execution_available_in_p33,
        "value_materialization_available_in_p33": result.value_materialization_available_in_p33,
        "array_materialization_available_in_p33": result.array_materialization_available_in_p33,
        "tensor_materialization_available_in_p33": result.tensor_materialization_available_in_p33,
        "forward_execution_available_in_p33": result.forward_execution_available_in_p33,
        "output_generation_available_in_p33": result.output_generation_available_in_p33,
        "no_rng_execution": result.no_rng_execution,
        "no_values_materialized": result.no_values_materialized,
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


def compact_fake_input_descriptor_json(result: FCVAEFakeInputDescriptorResult) -> str:
    d = fake_input_descriptor_result_to_json_dict(result)
    return json.dumps(d, sort_keys=True, separators=(",", ":"))
