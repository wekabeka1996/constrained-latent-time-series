# src/phase2/fc_vae_fake_input_flat_vector.py

import dataclasses
import json
from typing import Any, Dict, Tuple, Union

from src.phase2.fc_vae_fake_input_preview import (
    FC_VAE_FAKE_INPUT_PREVIEW_CONTRACT_VERSION,
    run_fake_input_preview_probe,
    fake_input_preview_result_to_json_dict,
)

# Public Constants
FC_VAE_FAKE_INPUT_FLAT_VECTOR_CONTRACT_VERSION = "phase2_p35_fake_input_flat_vector_contract_v1"

FC_VAE_FAKE_INPUT_FLAT_VECTOR_KIND = "deterministic_fake_input_flat_vector_no_tensor_no_array"

FC_VAE_FAKE_INPUT_FLAT_VECTOR_MODULE_NAME = "src.phase2.fc_vae_fake_input_flat_vector"

FC_VAE_FAKE_INPUT_FLAT_VECTOR_STATUS_TORCH_UNAVAILABLE = "blocked_torch_unavailable"

FC_VAE_FAKE_INPUT_FLAT_VECTOR_STATUS_PREVIEW_BLOCKED = "blocked_by_preview_status"

FC_VAE_FAKE_INPUT_FLAT_VECTOR_STATUS_FLAT_VECTOR_ONLY = "flat_vector_only_no_2d_batch_in_p35"

SUPPORTED_FC_VAE_FAKE_INPUT_FLAT_VECTOR_STATUSES = (
    "blocked_torch_unavailable",
    "blocked_by_preview_status",
    "flat_vector_only_no_2d_batch_in_p35",
)

FC_VAE_FAKE_INPUT_FLAT_VECTOR_VALUE_KIND = "bounded_deterministic_flat_scalar_vector"

# Defaults
DEFAULT_P35_EXPECTED_BATCH_SIZE = 2
DEFAULT_P35_EXPECTED_INPUT_FLAT_DIM = 32
DEFAULT_P35_EXPECTED_FULL_VECTOR_LENGTH = 64


# Public Dataclasses (All Frozen)
@dataclasses.dataclass(frozen=True)
class FCVAEFakeInputFlatVectorRequest:
    contract_version: str
    flat_vector_kind: str
    architecture_id: str
    source_preview_contract_version: str
    descriptor_seed: int
    min_value: float
    max_value: float
    batch_size: int
    input_flat_dim: int
    full_vector_length: int
    source_preview_value_count: int
    allow_rng_execution_in_p35: bool
    allow_2d_batch_materialization_in_p35: bool
    allow_array_materialization_in_p35: bool
    allow_tensor_materialization_in_p35: bool
    allow_forward_execution_in_p35: bool
    reason: str


@dataclasses.dataclass(frozen=True)
class FCVAEFakeInputFlatVector:
    contract_version: str
    value_kind: str
    full_vector_length: int
    flat_values: Tuple[float, ...]
    min_value: float
    max_value: float
    all_values_within_range: bool
    is_flat_vector: bool
    is_2d_batch: bool
    rng_executed: bool
    array_materialized: bool
    tensor_materialized: bool
    forward_executed: bool
    reason: str


@dataclasses.dataclass(frozen=True)
class FCVAEFakeInputFlatVectorMetadata:
    contract_version: str
    flat_vector_kind: str
    preview_contract_version: str
    preview_status: str
    torch_available: bool
    preview_available_in_p34: bool
    preview_value_count: int
    full_vector_length: int
    batch_size: int
    input_flat_dim: int
    flat_vector_materialized: bool
    two_d_batch_materialized: bool
    rng_execution_attempted: bool
    array_materialization_attempted: bool
    tensor_materialization_attempted: bool
    forward_execution_attempted: bool
    output_generation_attempted: bool
    reason: str


@dataclasses.dataclass(frozen=True)
class FCVAEFakeInputFlatVectorResult:
    contract_version: str
    request: FCVAEFakeInputFlatVectorRequest
    flat_vector: FCVAEFakeInputFlatVector
    metadata: FCVAEFakeInputFlatVectorMetadata
    status: str
    flat_vector_available_in_p35: bool
    two_d_batch_available_in_p35: bool
    rng_execution_available_in_p35: bool
    array_materialization_available_in_p35: bool
    tensor_materialization_available_in_p35: bool
    forward_execution_available_in_p35: bool
    output_generation_available_in_p35: bool
    no_rng_execution: bool
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


def validate_full_vector_length(value: Any) -> None:
    if type(value) is not int or isinstance(value, bool):
        raise TypeError(f"full_vector_length must be exact int, got {type(value).__name__}")
    if value <= 0:
        raise ValueError("full_vector_length must be > 0")


def validate_flat_values_tuple(values: Any, min_val: float, max_val: float, expected_len: int) -> None:
    if type(values) is not tuple:
        raise TypeError(f"flat_values must be exact tuple, got {type(values).__name__}")
    if len(values) != expected_len:
        raise ValueError(f"flat_values length must match full_vector_length ({expected_len}), got {len(values)}")
    for i, val in enumerate(values):
        if type(val) is not float:
            raise TypeError(f"flat_values[{i}] must be exact float, got {type(val).__name__}")
        if val < min_val or val > max_val:
            raise ValueError(f"flat_values[{i}] value ({val}) is outside range [{min_val}, {max_val}]")


def validate_fake_input_flat_vector_status(value: Any) -> None:
    if type(value) is not str:
        raise TypeError(f"status must be exact str, got {type(value).__name__}")
    if value not in SUPPORTED_FC_VAE_FAKE_INPUT_FLAT_VECTOR_STATUSES:
        raise ValueError(f"Unsupported fake input flat vector status: {value}")


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
                   .replace("flat_vector_only_no_2d_batch_in_p35", ""))
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


def validate_fake_input_flat_vector_request(request: FCVAEFakeInputFlatVectorRequest) -> None:
    if type(request) is not FCVAEFakeInputFlatVectorRequest:
        raise TypeError("request must be exact FCVAEFakeInputFlatVectorRequest instance")

    validate_non_empty_str(request.contract_version, "contract_version")
    if request.contract_version != FC_VAE_FAKE_INPUT_FLAT_VECTOR_CONTRACT_VERSION:
        raise ValueError(f"Invalid contract_version: {request.contract_version}")

    validate_non_empty_str(request.flat_vector_kind, "flat_vector_kind")
    if request.flat_vector_kind != FC_VAE_FAKE_INPUT_FLAT_VECTOR_KIND:
        raise ValueError(f"Invalid flat_vector_kind: {request.flat_vector_kind}")

    validate_non_empty_str(request.architecture_id, "architecture_id")
    if request.architecture_id != "FC-VAE":
        raise ValueError(f"Invalid architecture_id: {request.architecture_id}")

    validate_non_empty_str(request.source_preview_contract_version, "source_preview_contract_version")
    if request.source_preview_contract_version != FC_VAE_FAKE_INPUT_PREVIEW_CONTRACT_VERSION:
        raise ValueError(f"Invalid source_preview_contract_version: {request.source_preview_contract_version}")

    validate_non_negative_int(request.descriptor_seed, "descriptor_seed")
    validate_value_range(request.min_value, request.max_value)
    validate_positive_int(request.batch_size, "batch_size")
    validate_positive_int(request.input_flat_dim, "input_flat_dim")

    validate_full_vector_length(request.full_vector_length)
    if request.full_vector_length != request.batch_size * request.input_flat_dim:
        raise ValueError(f"full_vector_length must equal batch_size * input_flat_dim, got {request.full_vector_length}")

    validate_positive_int(request.source_preview_value_count, "source_preview_value_count")
    if request.source_preview_value_count != 4:
        raise ValueError(f"source_preview_value_count must be exactly 4, got {request.source_preview_value_count}")

    for flag_name in (
        "allow_rng_execution_in_p35",
        "allow_2d_batch_materialization_in_p35",
        "allow_array_materialization_in_p35",
        "allow_tensor_materialization_in_p35",
        "allow_forward_execution_in_p35",
    ):
        val = getattr(request, flag_name)
        validate_bool(val, flag_name)
        if val:
            raise ValueError(f"{flag_name} must be False")

    validate_non_empty_str(request.reason, "reason")
    assert_no_local_path_leakage(request.reason)
    assert_no_forbidden_claims(request.reason)


def validate_fake_input_flat_vector(vector: FCVAEFakeInputFlatVector) -> None:
    if type(vector) is not FCVAEFakeInputFlatVector:
        raise TypeError("vector must be exact FCVAEFakeInputFlatVector instance")

    validate_non_empty_str(vector.contract_version, "contract_version")
    if vector.contract_version != FC_VAE_FAKE_INPUT_FLAT_VECTOR_CONTRACT_VERSION:
        raise ValueError(f"Invalid contract_version: {vector.contract_version}")

    validate_non_empty_str(vector.value_kind, "value_kind")
    if vector.value_kind != FC_VAE_FAKE_INPUT_FLAT_VECTOR_VALUE_KIND:
        raise ValueError(f"Invalid value_kind: {vector.value_kind}")

    validate_full_vector_length(vector.full_vector_length)
    validate_flat_values_tuple(vector.flat_values, vector.min_value, vector.max_value, vector.full_vector_length)

    validate_bool(vector.all_values_within_range, "all_values_within_range")
    if not vector.all_values_within_range:
        raise ValueError("all_values_within_range must be True")

    validate_bool(vector.is_flat_vector, "is_flat_vector")
    if not vector.is_flat_vector:
        raise ValueError("is_flat_vector must be True")

    validate_bool(vector.is_2d_batch, "is_2d_batch")
    if vector.is_2d_batch:
        raise ValueError("is_2d_batch must be False")

    for flag_name in (
        "rng_executed",
        "array_materialized",
        "tensor_materialized",
        "forward_executed",
    ):
        val = getattr(vector, flag_name)
        validate_bool(val, flag_name)
        if val:
            raise ValueError(f"{flag_name} must be False")

    validate_non_empty_str(vector.reason, "reason")
    assert_no_local_path_leakage(vector.reason)
    assert_no_forbidden_claims(vector.reason)


def validate_fake_input_flat_vector_metadata(metadata: FCVAEFakeInputFlatVectorMetadata) -> None:
    if type(metadata) is not FCVAEFakeInputFlatVectorMetadata:
        raise TypeError("metadata must be exact FCVAEFakeInputFlatVectorMetadata instance")

    validate_non_empty_str(metadata.contract_version, "contract_version")
    if metadata.contract_version != FC_VAE_FAKE_INPUT_FLAT_VECTOR_CONTRACT_VERSION:
        raise ValueError(f"Invalid contract_version: {metadata.contract_version}")

    validate_non_empty_str(metadata.flat_vector_kind, "flat_vector_kind")
    if metadata.flat_vector_kind != FC_VAE_FAKE_INPUT_FLAT_VECTOR_KIND:
        raise ValueError(f"Invalid flat_vector_kind: {metadata.flat_vector_kind}")

    validate_non_empty_str(metadata.preview_contract_version, "preview_contract_version")
    if metadata.preview_contract_version != FC_VAE_FAKE_INPUT_PREVIEW_CONTRACT_VERSION:
        raise ValueError(f"Invalid preview_contract_version: {metadata.preview_contract_version}")

    validate_non_empty_str(metadata.preview_status, "preview_status")
    validate_bool(metadata.torch_available, "torch_available")
    validate_bool(metadata.preview_available_in_p34, "preview_available_in_p34")
    validate_positive_int(metadata.preview_value_count, "preview_value_count")
    validate_full_vector_length(metadata.full_vector_length)
    validate_positive_int(metadata.batch_size, "batch_size")
    validate_positive_int(metadata.input_flat_dim, "input_flat_dim")

    if metadata.preview_value_count >= metadata.full_vector_length:
        raise ValueError("preview_value_count must be < full_vector_length")

    validate_bool(metadata.flat_vector_materialized, "flat_vector_materialized")
    if not metadata.flat_vector_materialized:
        raise ValueError("flat_vector_materialized must be True")

    validate_bool(metadata.two_d_batch_materialized, "two_d_batch_materialized")
    if metadata.two_d_batch_materialized:
        raise ValueError("two_d_batch_materialized must be False")

    for flag_name in (
        "rng_execution_attempted",
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


def validate_fake_input_flat_vector_result(result: FCVAEFakeInputFlatVectorResult) -> None:
    if type(result) is not FCVAEFakeInputFlatVectorResult:
        raise TypeError("result must be exact FCVAEFakeInputFlatVectorResult instance")

    validate_non_empty_str(result.contract_version, "contract_version")
    if result.contract_version != FC_VAE_FAKE_INPUT_FLAT_VECTOR_CONTRACT_VERSION:
        raise ValueError(f"Invalid contract_version: {result.contract_version}")

    validate_fake_input_flat_vector_request(result.request)
    validate_fake_input_flat_vector(result.flat_vector)
    validate_fake_input_flat_vector_metadata(result.metadata)
    validate_fake_input_flat_vector_status(result.status)

    validate_bool(result.flat_vector_available_in_p35, "flat_vector_available_in_p35")
    if not result.flat_vector_available_in_p35:
        raise ValueError("flat_vector_available_in_p35 must be True")

    validate_bool(result.two_d_batch_available_in_p35, "two_d_batch_available_in_p35")
    if result.two_d_batch_available_in_p35:
        raise ValueError("two_d_batch_available_in_p35 must be False")

    for flag_name in (
        "rng_execution_available_in_p35",
        "array_materialization_available_in_p35",
        "tensor_materialization_available_in_p35",
        "forward_execution_available_in_p35",
        "output_generation_available_in_p35",
    ):
        val = getattr(result, flag_name)
        validate_bool(val, flag_name)
        if val:
            raise ValueError(f"{flag_name} must be False")

    for flag_name in (
        "no_rng_execution",
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
        if result.status != FC_VAE_FAKE_INPUT_FLAT_VECTOR_STATUS_TORCH_UNAVAILABLE:
            raise ValueError(f"Inconsistent state: status must be '{FC_VAE_FAKE_INPUT_FLAT_VECTOR_STATUS_TORCH_UNAVAILABLE}' when torch is unavailable")
    elif not result.metadata.preview_available_in_p34:
        if result.status != FC_VAE_FAKE_INPUT_FLAT_VECTOR_STATUS_PREVIEW_BLOCKED:
            raise ValueError(f"Inconsistent state: status must be '{FC_VAE_FAKE_INPUT_FLAT_VECTOR_STATUS_PREVIEW_BLOCKED}' when preview is blocked")
    else:
        if result.status != FC_VAE_FAKE_INPUT_FLAT_VECTOR_STATUS_FLAT_VECTOR_ONLY:
            raise ValueError(f"Inconsistent state: status must be '{FC_VAE_FAKE_INPUT_FLAT_VECTOR_STATUS_FLAT_VECTOR_ONLY}'")

    validate_non_empty_str(result.reason, "reason")
    assert_no_local_path_leakage(result.reason)
    assert_no_forbidden_claims(result.reason)


# Public Builders
def compute_deterministic_flat_values(
    descriptor_seed: int,
    min_value: float,
    max_value: float,
    full_vector_length: int,
) -> Tuple[float, ...]:
    validate_non_negative_int(descriptor_seed, "descriptor_seed")
    validate_numeric_value(min_value, "min_value")
    validate_numeric_value(max_value, "max_value")
    validate_value_range(min_value, max_value)
    validate_full_vector_length(full_vector_length)

    values = []
    span = max_value - min_value
    for i in range(full_vector_length):
        phase = ((descriptor_seed % 997) + (i + 1) * 37) % 997
        unit = phase / 996.0
        val = min_value + span * unit
        values.append(round(val, 8))
    return tuple(values)


def build_fake_input_flat_vector_request_from_p34_default() -> FCVAEFakeInputFlatVectorRequest:
    p34_result = run_fake_input_preview_probe()

    req = FCVAEFakeInputFlatVectorRequest(
        contract_version=FC_VAE_FAKE_INPUT_FLAT_VECTOR_CONTRACT_VERSION,
        flat_vector_kind=FC_VAE_FAKE_INPUT_FLAT_VECTOR_KIND,
        architecture_id="FC-VAE",
        source_preview_contract_version=p34_result.contract_version,
        descriptor_seed=p34_result.request.descriptor_seed,
        min_value=p34_result.request.min_value,
        max_value=p34_result.request.max_value,
        batch_size=p34_result.request.batch_size,
        input_flat_dim=p34_result.request.input_flat_dim,
        full_vector_length=p34_result.request.full_batch_scalar_count,
        source_preview_value_count=p34_result.request.preview_value_count,
        allow_rng_execution_in_p35=False,
        allow_2d_batch_materialization_in_p35=False,
        allow_array_materialization_in_p35=False,
        allow_tensor_materialization_in_p35=False,
        allow_forward_execution_in_p35=False,
        reason="p35_fake_input_flat_vector_request_from_p34_default",
    )
    validate_fake_input_flat_vector_request(req)
    return req


def build_fake_input_flat_vector(request: FCVAEFakeInputFlatVectorRequest) -> FCVAEFakeInputFlatVector:
    validate_fake_input_flat_vector_request(request)

    flat_values = compute_deterministic_flat_values(
        request.descriptor_seed,
        request.min_value,
        request.max_value,
        request.full_vector_length,
    )

    vector = FCVAEFakeInputFlatVector(
        contract_version=FC_VAE_FAKE_INPUT_FLAT_VECTOR_CONTRACT_VERSION,
        value_kind=FC_VAE_FAKE_INPUT_FLAT_VECTOR_VALUE_KIND,
        full_vector_length=request.full_vector_length,
        flat_values=flat_values,
        min_value=request.min_value,
        max_value=request.max_value,
        all_values_within_range=True,
        is_flat_vector=True,
        is_2d_batch=False,
        rng_executed=False,
        array_materialized=False,
        tensor_materialized=False,
        forward_executed=False,
        reason="p35_fake_input_flat_vector",
    )
    validate_fake_input_flat_vector(vector)
    return vector


def build_fake_input_flat_vector_metadata(request: FCVAEFakeInputFlatVectorRequest) -> FCVAEFakeInputFlatVectorMetadata:
    validate_fake_input_flat_vector_request(request)
    p34_result = run_fake_input_preview_probe()

    meta = FCVAEFakeInputFlatVectorMetadata(
        contract_version=FC_VAE_FAKE_INPUT_FLAT_VECTOR_CONTRACT_VERSION,
        flat_vector_kind=FC_VAE_FAKE_INPUT_FLAT_VECTOR_KIND,
        preview_contract_version=p34_result.contract_version,
        preview_status=p34_result.status,
        torch_available=p34_result.metadata.torch_available,
        preview_available_in_p34=p34_result.preview_available_in_p34,
        preview_value_count=p34_result.metadata.preview_value_count,
        full_vector_length=request.full_vector_length,
        batch_size=request.batch_size,
        input_flat_dim=request.input_flat_dim,
        flat_vector_materialized=True,
        two_d_batch_materialized=False,
        rng_execution_attempted=False,
        array_materialization_attempted=False,
        tensor_materialization_attempted=False,
        forward_execution_attempted=False,
        output_generation_attempted=False,
        reason="p35_fake_input_flat_vector_metadata",
    )
    validate_fake_input_flat_vector_metadata(meta)
    return meta


def build_fake_input_flat_vector_result(request: FCVAEFakeInputFlatVectorRequest) -> FCVAEFakeInputFlatVectorResult:
    validate_fake_input_flat_vector_request(request)
    flat_vector = build_fake_input_flat_vector(request)
    metadata = build_fake_input_flat_vector_metadata(request)

    if not metadata.torch_available:
        status = FC_VAE_FAKE_INPUT_FLAT_VECTOR_STATUS_TORCH_UNAVAILABLE
        reason = "fake_input_flat_vector_blocked_torch_unavailable"
    elif not metadata.preview_available_in_p34:
        status = FC_VAE_FAKE_INPUT_FLAT_VECTOR_STATUS_PREVIEW_BLOCKED
        reason = "fake_input_flat_vector_blocked_by_preview_status"
    else:
        status = FC_VAE_FAKE_INPUT_FLAT_VECTOR_STATUS_FLAT_VECTOR_ONLY
        reason = "fake_input_flat_vector_only"

    res = FCVAEFakeInputFlatVectorResult(
        contract_version=FC_VAE_FAKE_INPUT_FLAT_VECTOR_CONTRACT_VERSION,
        request=request,
        flat_vector=flat_vector,
        metadata=metadata,
        status=status,
        flat_vector_available_in_p35=True,
        two_d_batch_available_in_p35=False,
        rng_execution_available_in_p35=False,
        array_materialization_available_in_p35=False,
        tensor_materialization_available_in_p35=False,
        forward_execution_available_in_p35=False,
        output_generation_available_in_p35=False,
        no_rng_execution=True,
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
    validate_fake_input_flat_vector_result(res)
    return res


def run_fake_input_flat_vector_probe() -> FCVAEFakeInputFlatVectorResult:
    req = build_fake_input_flat_vector_request_from_p34_default()
    res = build_fake_input_flat_vector_result(req)
    validate_fake_input_flat_vector_result(res)
    return res


# Public Serialization
def fake_input_flat_vector_request_to_json_dict(request: FCVAEFakeInputFlatVectorRequest) -> dict:
    validate_fake_input_flat_vector_request(request)
    d = {
        "contract_version": request.contract_version,
        "flat_vector_kind": request.flat_vector_kind,
        "architecture_id": request.architecture_id,
        "source_preview_contract_version": request.source_preview_contract_version,
        "descriptor_seed": request.descriptor_seed,
        "min_value": request.min_value,
        "max_value": request.max_value,
        "batch_size": request.batch_size,
        "input_flat_dim": request.input_flat_dim,
        "full_vector_length": request.full_vector_length,
        "source_preview_value_count": request.source_preview_value_count,
        "allow_rng_execution_in_p35": request.allow_rng_execution_in_p35,
        "allow_2d_batch_materialization_in_p35": request.allow_2d_batch_materialization_in_p35,
        "allow_array_materialization_in_p35": request.allow_array_materialization_in_p35,
        "allow_tensor_materialization_in_p35": request.allow_tensor_materialization_in_p35,
        "allow_forward_execution_in_p35": request.allow_forward_execution_in_p35,
        "reason": request.reason,
    }
    assert_no_local_path_leakage(d)
    assert_no_forbidden_claims(d)
    return d


def fake_input_flat_vector_to_json_dict(vector: FCVAEFakeInputFlatVector) -> dict:
    validate_fake_input_flat_vector(vector)
    d = {
        "contract_version": vector.contract_version,
        "value_kind": vector.value_kind,
        "full_vector_length": vector.full_vector_length,
        "flat_values": list(vector.flat_values),
        "min_value": vector.min_value,
        "max_value": vector.max_value,
        "all_values_within_range": vector.all_values_within_range,
        "is_flat_vector": vector.is_flat_vector,
        "is_2d_batch": vector.is_2d_batch,
        "rng_executed": vector.rng_executed,
        "array_materialized": vector.array_materialized,
        "tensor_materialized": vector.tensor_materialized,
        "forward_executed": vector.forward_executed,
        "reason": vector.reason,
    }
    assert_no_local_path_leakage(d)
    assert_no_forbidden_claims(d)
    return d


def fake_input_flat_vector_metadata_to_json_dict(metadata: FCVAEFakeInputFlatVectorMetadata) -> dict:
    validate_fake_input_flat_vector_metadata(metadata)
    d = {
        "contract_version": metadata.contract_version,
        "flat_vector_kind": metadata.flat_vector_kind,
        "preview_contract_version": metadata.preview_contract_version,
        "preview_status": metadata.preview_status,
        "torch_available": metadata.torch_available,
        "preview_available_in_p34": metadata.preview_available_in_p34,
        "preview_value_count": metadata.preview_value_count,
        "full_vector_length": metadata.full_vector_length,
        "batch_size": metadata.batch_size,
        "input_flat_dim": metadata.input_flat_dim,
        "flat_vector_materialized": metadata.flat_vector_materialized,
        "two_d_batch_materialized": metadata.two_d_batch_materialized,
        "rng_execution_attempted": metadata.rng_execution_attempted,
        "array_materialization_attempted": metadata.array_materialization_attempted,
        "tensor_materialization_attempted": metadata.tensor_materialization_attempted,
        "forward_execution_attempted": metadata.forward_execution_attempted,
        "output_generation_attempted": metadata.output_generation_attempted,
        "reason": metadata.reason,
    }
    assert_no_local_path_leakage(d)
    assert_no_forbidden_claims(d)
    return d


def fake_input_flat_vector_result_to_json_dict(result: FCVAEFakeInputFlatVectorResult) -> dict:
    validate_fake_input_flat_vector_result(result)
    d = {
        "contract_version": result.contract_version,
        "request": fake_input_flat_vector_request_to_json_dict(result.request),
        "flat_vector": fake_input_flat_vector_to_json_dict(result.flat_vector),
        "metadata": fake_input_flat_vector_metadata_to_json_dict(result.metadata),
        "status": result.status,
        "flat_vector_available_in_p35": result.flat_vector_available_in_p35,
        "two_d_batch_available_in_p35": result.two_d_batch_available_in_p35,
        "rng_execution_available_in_p35": result.rng_execution_available_in_p35,
        "array_materialization_available_in_p35": result.array_materialization_available_in_p35,
        "tensor_materialization_available_in_p35": result.tensor_materialization_available_in_p35,
        "forward_execution_available_in_p35": result.forward_execution_available_in_p35,
        "output_generation_available_in_p35": result.output_generation_available_in_p35,
        "no_rng_execution": result.no_rng_execution,
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


def compact_fake_input_flat_vector_json(result: FCVAEFakeInputFlatVectorResult) -> str:
    d = fake_input_flat_vector_result_to_json_dict(result)
    return json.dumps(d, sort_keys=True, separators=(",", ":"))
