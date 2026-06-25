# src/phase2/fc_vae_fake_input_preview.py

import dataclasses
import json
from typing import Any, Dict, Tuple, Union

from src.phase2.fc_vae_fake_input_descriptor import (
    FC_VAE_FAKE_INPUT_DESCRIPTOR_CONTRACT_VERSION,
    run_fake_input_descriptor_probe,
    fake_input_descriptor_result_to_json_dict,
)


# Public Constants
FC_VAE_FAKE_INPUT_PREVIEW_CONTRACT_VERSION = "phase2_p34_fake_input_scalar_preview_contract_v1"

FC_VAE_FAKE_INPUT_PREVIEW_KIND = "deterministic_fake_input_scalar_preview_no_tensor_no_array"

FC_VAE_FAKE_INPUT_PREVIEW_MODULE_NAME = "src.phase2.fc_vae_fake_input_preview"

FC_VAE_FAKE_INPUT_PREVIEW_STATUS_TORCH_UNAVAILABLE = "blocked_torch_unavailable"

FC_VAE_FAKE_INPUT_PREVIEW_STATUS_DESCRIPTOR_BLOCKED = "blocked_by_descriptor_status"

FC_VAE_FAKE_INPUT_PREVIEW_STATUS_PREVIEW_ONLY = "scalar_preview_only_no_full_batch_in_p34"

SUPPORTED_FC_VAE_FAKE_INPUT_PREVIEW_STATUSES = (
    "blocked_torch_unavailable",
    "blocked_by_descriptor_status",
    "scalar_preview_only_no_full_batch_in_p34",
)

FC_VAE_FAKE_INPUT_PREVIEW_VALUE_KIND = "bounded_deterministic_scalar_preview"

# Defaults
DEFAULT_P34_PREVIEW_VALUE_COUNT = 4
MIN_P34_PREVIEW_VALUE_COUNT = 1
MAX_P34_PREVIEW_VALUE_COUNT = 8


# Public Dataclasses (All Frozen)
@dataclasses.dataclass(frozen=True)
class FCVAEFakeInputPreviewRequest:
    contract_version: str
    preview_kind: str
    architecture_id: str
    source_descriptor_contract_version: str
    descriptor_seed: int
    min_value: float
    max_value: float
    batch_size: int
    input_flat_dim: int
    full_batch_scalar_count: int
    preview_value_count: int
    allow_rng_execution_in_p34: bool
    allow_full_batch_materialization_in_p34: bool
    allow_array_materialization_in_p34: bool
    allow_tensor_materialization_in_p34: bool
    allow_forward_execution_in_p34: bool
    reason: str


@dataclasses.dataclass(frozen=True)
class FCVAEFakeInputScalarPreview:
    contract_version: str
    value_kind: str
    preview_value_count: int
    preview_values: Tuple[float, ...]
    min_value: float
    max_value: float
    all_values_within_range: bool
    rng_executed: bool
    full_batch_materialized: bool
    array_materialized: bool
    tensor_materialized: bool
    reason: str


@dataclasses.dataclass(frozen=True)
class FCVAEFakeInputPreviewMetadata:
    contract_version: str
    preview_kind: str
    descriptor_contract_version: str
    descriptor_status: str
    torch_available: bool
    descriptor_available_in_p33: bool
    full_batch_scalar_count: int
    preview_value_count: int
    preview_is_partial: bool
    rng_execution_attempted: bool
    full_batch_materialization_attempted: bool
    array_materialization_attempted: bool
    tensor_materialization_attempted: bool
    forward_execution_attempted: bool
    output_generation_attempted: bool
    reason: str


@dataclasses.dataclass(frozen=True)
class FCVAEFakeInputPreviewResult:
    contract_version: str
    request: FCVAEFakeInputPreviewRequest
    preview: FCVAEFakeInputScalarPreview
    metadata: FCVAEFakeInputPreviewMetadata
    status: str
    preview_available_in_p34: bool
    full_batch_available_in_p34: bool
    rng_execution_available_in_p34: bool
    array_materialization_available_in_p34: bool
    tensor_materialization_available_in_p34: bool
    forward_execution_available_in_p34: bool
    output_generation_available_in_p34: bool
    no_rng_execution: bool
    no_full_batch_materialized: bool
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


def validate_preview_value_count(value: Any) -> None:
    if type(value) is not int or isinstance(value, bool):
        raise TypeError(f"preview_value_count must be exact int, got {type(value).__name__}")
    if value < MIN_P34_PREVIEW_VALUE_COUNT or value > MAX_P34_PREVIEW_VALUE_COUNT:
        raise ValueError(f"preview_value_count must be between {MIN_P34_PREVIEW_VALUE_COUNT} and {MAX_P34_PREVIEW_VALUE_COUNT} inclusive")


def validate_preview_values_tuple(values: Any, min_val: float, max_val: float, expected_len: int) -> None:
    if type(values) is not tuple:
        raise TypeError(f"preview_values must be exact tuple, got {type(values).__name__}")
    if len(values) != expected_len:
        raise ValueError(f"preview_values length must match preview_value_count ({expected_len}), got {len(values)}")
    for i, val in enumerate(values):
        if type(val) is not float:
            raise TypeError(f"preview_values[{i}] must be exact float, got {type(val).__name__}")
        if val < min_val or val > max_val:
            raise ValueError(f"preview_values[{i}] value ({val}) is outside range [{min_val}, {max_val}]")


def validate_fake_input_preview_status(value: Any) -> None:
    if type(value) is not str:
        raise TypeError(f"status must be exact str, got {type(value).__name__}")
    if value not in SUPPORTED_FC_VAE_FAKE_INPUT_PREVIEW_STATUSES:
        raise ValueError(f"Unsupported fake input preview status: {value}")


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
                   .replace("scalar_preview_only_no_full_batch_in_p34", ""))
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


def validate_fake_input_preview_request(request: FCVAEFakeInputPreviewRequest) -> None:
    if type(request) is not FCVAEFakeInputPreviewRequest:
        raise TypeError("request must be exact FCVAEFakeInputPreviewRequest instance")

    validate_non_empty_str(request.contract_version, "contract_version")
    if request.contract_version != FC_VAE_FAKE_INPUT_PREVIEW_CONTRACT_VERSION:
        raise ValueError(f"Invalid contract_version: {request.contract_version}")

    validate_non_empty_str(request.preview_kind, "preview_kind")
    if request.preview_kind != FC_VAE_FAKE_INPUT_PREVIEW_KIND:
        raise ValueError(f"Invalid preview_kind: {request.preview_kind}")

    validate_non_empty_str(request.architecture_id, "architecture_id")
    if request.architecture_id != "FC-VAE":
        raise ValueError(f"Invalid architecture_id: {request.architecture_id}")

    validate_non_empty_str(request.source_descriptor_contract_version, "source_descriptor_contract_version")
    if request.source_descriptor_contract_version != FC_VAE_FAKE_INPUT_DESCRIPTOR_CONTRACT_VERSION:
        raise ValueError(f"Invalid source_descriptor_contract_version: {request.source_descriptor_contract_version}")

    validate_non_negative_int(request.descriptor_seed, "descriptor_seed")
    validate_value_range(request.min_value, request.max_value)
    validate_positive_int(request.batch_size, "batch_size")
    validate_positive_int(request.input_flat_dim, "input_flat_dim")

    validate_positive_int(request.full_batch_scalar_count, "full_batch_scalar_count")
    if request.full_batch_scalar_count != request.batch_size * request.input_flat_dim:
        raise ValueError(f"full_batch_scalar_count must equal batch_size * input_flat_dim, got {request.full_batch_scalar_count}")

    validate_preview_value_count(request.preview_value_count)
    if request.preview_value_count >= request.full_batch_scalar_count:
        raise ValueError(f"preview_value_count must be < full_batch_scalar_count ({request.full_batch_scalar_count})")

    for flag_name in (
        "allow_rng_execution_in_p34",
        "allow_full_batch_materialization_in_p34",
        "allow_array_materialization_in_p34",
        "allow_tensor_materialization_in_p34",
        "allow_forward_execution_in_p34",
    ):
        val = getattr(request, flag_name)
        validate_bool(val, flag_name)
        if val:
            raise ValueError(f"{flag_name} must be False")

    validate_non_empty_str(request.reason, "reason")
    assert_no_local_path_leakage(request.reason)
    assert_no_forbidden_claims(request.reason)


def validate_fake_input_scalar_preview(preview: FCVAEFakeInputScalarPreview) -> None:
    if type(preview) is not FCVAEFakeInputScalarPreview:
        raise TypeError("preview must be exact FCVAEFakeInputScalarPreview instance")

    validate_non_empty_str(preview.contract_version, "contract_version")
    if preview.contract_version != FC_VAE_FAKE_INPUT_PREVIEW_CONTRACT_VERSION:
        raise ValueError(f"Invalid contract_version: {preview.contract_version}")

    validate_non_empty_str(preview.value_kind, "value_kind")
    if preview.value_kind != FC_VAE_FAKE_INPUT_PREVIEW_VALUE_KIND:
        raise ValueError(f"Invalid value_kind: {preview.value_kind}")

    validate_preview_value_count(preview.preview_value_count)
    validate_preview_values_tuple(preview.preview_values, preview.min_value, preview.max_value, preview.preview_value_count)

    validate_bool(preview.all_values_within_range, "all_values_within_range")
    if not preview.all_values_within_range:
        raise ValueError("all_values_within_range must be True")

    for flag_name in (
        "rng_executed",
        "full_batch_materialized",
        "array_materialized",
        "tensor_materialized",
    ):
        val = getattr(preview, flag_name)
        validate_bool(val, flag_name)
        if val:
            raise ValueError(f"{flag_name} must be False")

    validate_non_empty_str(preview.reason, "reason")
    assert_no_local_path_leakage(preview.reason)
    assert_no_forbidden_claims(preview.reason)


def validate_fake_input_preview_metadata(metadata: FCVAEFakeInputPreviewMetadata) -> None:
    if type(metadata) is not FCVAEFakeInputPreviewMetadata:
        raise TypeError("metadata must be exact FCVAEFakeInputPreviewMetadata instance")

    validate_non_empty_str(metadata.contract_version, "contract_version")
    if metadata.contract_version != FC_VAE_FAKE_INPUT_PREVIEW_CONTRACT_VERSION:
        raise ValueError(f"Invalid contract_version: {metadata.contract_version}")

    validate_non_empty_str(metadata.preview_kind, "preview_kind")
    if metadata.preview_kind != FC_VAE_FAKE_INPUT_PREVIEW_KIND:
        raise ValueError(f"Invalid preview_kind: {metadata.preview_kind}")

    validate_non_empty_str(metadata.descriptor_contract_version, "descriptor_contract_version")
    if metadata.descriptor_contract_version != FC_VAE_FAKE_INPUT_DESCRIPTOR_CONTRACT_VERSION:
        raise ValueError(f"Invalid descriptor_contract_version: {metadata.descriptor_contract_version}")

    validate_non_empty_str(metadata.descriptor_status, "descriptor_status")
    validate_bool(metadata.torch_available, "torch_available")
    validate_bool(metadata.descriptor_available_in_p33, "descriptor_available_in_p33")

    validate_positive_int(metadata.full_batch_scalar_count, "full_batch_scalar_count")
    validate_preview_value_count(metadata.preview_value_count)
    if metadata.preview_value_count >= metadata.full_batch_scalar_count:
        raise ValueError(f"preview_value_count must be < full_batch_scalar_count, got {metadata.preview_value_count}")

    validate_bool(metadata.preview_is_partial, "preview_is_partial")
    if not metadata.preview_is_partial:
        raise ValueError("preview_is_partial must be True")

    for flag_name in (
        "rng_execution_attempted",
        "full_batch_materialization_attempted",
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


def validate_fake_input_preview_result(result: FCVAEFakeInputPreviewResult) -> None:
    if type(result) is not FCVAEFakeInputPreviewResult:
        raise TypeError("result must be exact FCVAEFakeInputPreviewResult instance")

    validate_non_empty_str(result.contract_version, "contract_version")
    if result.contract_version != FC_VAE_FAKE_INPUT_PREVIEW_CONTRACT_VERSION:
        raise ValueError(f"Invalid contract_version: {result.contract_version}")

    validate_fake_input_preview_request(result.request)
    validate_fake_input_scalar_preview(result.preview)
    validate_fake_input_preview_metadata(result.metadata)
    validate_fake_input_preview_status(result.status)

    validate_bool(result.preview_available_in_p34, "preview_available_in_p34")
    if not result.preview_available_in_p34:
        raise ValueError("preview_available_in_p34 must be True")

    for flag_name in (
        "full_batch_available_in_p34",
        "rng_execution_available_in_p34",
        "array_materialization_available_in_p34",
        "tensor_materialization_available_in_p34",
        "forward_execution_available_in_p34",
        "output_generation_available_in_p34",
    ):
        val = getattr(result, flag_name)
        validate_bool(val, flag_name)
        if val:
            raise ValueError(f"{flag_name} must be False")

    for flag_name in (
        "no_rng_execution",
        "no_full_batch_materialized",
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
        if result.status != FC_VAE_FAKE_INPUT_PREVIEW_STATUS_TORCH_UNAVAILABLE:
            raise ValueError(f"Inconsistent state: status must be '{FC_VAE_FAKE_INPUT_PREVIEW_STATUS_TORCH_UNAVAILABLE}' when torch is unavailable")
    elif not result.metadata.descriptor_available_in_p33:
        if result.status != FC_VAE_FAKE_INPUT_PREVIEW_STATUS_DESCRIPTOR_BLOCKED:
            raise ValueError(f"Inconsistent state: status must be '{FC_VAE_FAKE_INPUT_PREVIEW_STATUS_DESCRIPTOR_BLOCKED}' when descriptor is blocked")
    else:
        if result.status != FC_VAE_FAKE_INPUT_PREVIEW_STATUS_PREVIEW_ONLY:
            raise ValueError(f"Inconsistent state: status must be '{FC_VAE_FAKE_INPUT_PREVIEW_STATUS_PREVIEW_ONLY}'")

    validate_non_empty_str(result.reason, "reason")
    assert_no_local_path_leakage(result.reason)
    assert_no_forbidden_claims(result.reason)


# Public Builders
def compute_deterministic_preview_values(
    descriptor_seed: int,
    min_value: float,
    max_value: float,
    preview_value_count: int,
) -> Tuple[float, ...]:
    validate_non_negative_int(descriptor_seed, "descriptor_seed")
    validate_numeric_value(min_value, "min_value")
    validate_numeric_value(max_value, "max_value")
    validate_value_range(min_value, max_value)
    validate_preview_value_count(preview_value_count)

    values = []
    span = max_value - min_value
    for i in range(preview_value_count):
        phase = ((descriptor_seed % 997) + (i + 1) * 37) % 997
        unit = phase / 996.0
        val = min_value + span * unit
        values.append(round(val, 8))
    return tuple(values)


def build_fake_input_preview_request_from_p33_default(
    preview_value_count: int = DEFAULT_P34_PREVIEW_VALUE_COUNT,
) -> FCVAEFakeInputPreviewRequest:
    p33_result = run_fake_input_descriptor_probe()

    req = FCVAEFakeInputPreviewRequest(
        contract_version=FC_VAE_FAKE_INPUT_PREVIEW_CONTRACT_VERSION,
        preview_kind=FC_VAE_FAKE_INPUT_PREVIEW_KIND,
        architecture_id="FC-VAE",
        source_descriptor_contract_version=p33_result.contract_version,
        descriptor_seed=p33_result.request.descriptor_seed,
        min_value=p33_result.request.min_value,
        max_value=p33_result.request.max_value,
        batch_size=p33_result.request.batch_size,
        input_flat_dim=p33_result.request.input_flat_dim,
        full_batch_scalar_count=p33_result.request.batch_size * p33_result.request.input_flat_dim,
        preview_value_count=preview_value_count,
        allow_rng_execution_in_p34=False,
        allow_full_batch_materialization_in_p34=False,
        allow_array_materialization_in_p34=False,
        allow_tensor_materialization_in_p34=False,
        allow_forward_execution_in_p34=False,
        reason="p34_fake_input_preview_request_from_p33_default",
    )
    validate_fake_input_preview_request(req)
    return req


def build_fake_input_scalar_preview(request: FCVAEFakeInputPreviewRequest) -> FCVAEFakeInputScalarPreview:
    validate_fake_input_preview_request(request)

    preview_values = compute_deterministic_preview_values(
        request.descriptor_seed,
        request.min_value,
        request.max_value,
        request.preview_value_count,
    )

    preview = FCVAEFakeInputScalarPreview(
        contract_version=FC_VAE_FAKE_INPUT_PREVIEW_CONTRACT_VERSION,
        value_kind=FC_VAE_FAKE_INPUT_PREVIEW_VALUE_KIND,
        preview_value_count=request.preview_value_count,
        preview_values=preview_values,
        min_value=request.min_value,
        max_value=request.max_value,
        all_values_within_range=True,
        rng_executed=False,
        full_batch_materialized=False,
        array_materialized=False,
        tensor_materialized=False,
        reason="p34_fake_input_scalar_preview",
    )
    validate_fake_input_scalar_preview(preview)
    return preview


def build_fake_input_preview_metadata(request: FCVAEFakeInputPreviewRequest) -> FCVAEFakeInputPreviewMetadata:
    validate_fake_input_preview_request(request)
    p33_result = run_fake_input_descriptor_probe()

    meta = FCVAEFakeInputPreviewMetadata(
        contract_version=FC_VAE_FAKE_INPUT_PREVIEW_CONTRACT_VERSION,
        preview_kind=FC_VAE_FAKE_INPUT_PREVIEW_KIND,
        descriptor_contract_version=p33_result.contract_version,
        descriptor_status=p33_result.status,
        torch_available=p33_result.metadata.torch_available,
        descriptor_available_in_p33=p33_result.descriptor_available_in_p33,
        full_batch_scalar_count=request.full_batch_scalar_count,
        preview_value_count=request.preview_value_count,
        preview_is_partial=True,
        rng_execution_attempted=False,
        full_batch_materialization_attempted=False,
        array_materialization_attempted=False,
        tensor_materialization_attempted=False,
        forward_execution_attempted=False,
        output_generation_attempted=False,
        reason="p34_fake_input_preview_metadata",
    )
    validate_fake_input_preview_metadata(meta)
    return meta


def build_fake_input_preview_result(request: FCVAEFakeInputPreviewRequest) -> FCVAEFakeInputPreviewResult:
    validate_fake_input_preview_request(request)
    preview = build_fake_input_scalar_preview(request)
    metadata = build_fake_input_preview_metadata(request)

    if not metadata.torch_available:
        status = FC_VAE_FAKE_INPUT_PREVIEW_STATUS_TORCH_UNAVAILABLE
        reason = "fake_input_preview_blocked_torch_unavailable"
    elif not metadata.descriptor_available_in_p33:
        status = FC_VAE_FAKE_INPUT_PREVIEW_STATUS_DESCRIPTOR_BLOCKED
        reason = "fake_input_preview_blocked_by_descriptor_status"
    else:
        status = FC_VAE_FAKE_INPUT_PREVIEW_STATUS_PREVIEW_ONLY
        reason = "fake_input_preview_only"

    res = FCVAEFakeInputPreviewResult(
        contract_version=FC_VAE_FAKE_INPUT_PREVIEW_CONTRACT_VERSION,
        request=request,
        preview=preview,
        metadata=metadata,
        status=status,
        preview_available_in_p34=True,
        full_batch_available_in_p34=False,
        rng_execution_available_in_p34=False,
        array_materialization_available_in_p34=False,
        tensor_materialization_available_in_p34=False,
        forward_execution_available_in_p34=False,
        output_generation_available_in_p34=False,
        no_rng_execution=True,
        no_full_batch_materialized=True,
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
    validate_fake_input_preview_result(res)
    return res


def run_fake_input_preview_probe() -> FCVAEFakeInputPreviewResult:
    req = build_fake_input_preview_request_from_p33_default()
    res = build_fake_input_preview_result(req)
    validate_fake_input_preview_result(res)
    return res


# Public Serialization
def fake_input_preview_request_to_json_dict(request: FCVAEFakeInputPreviewRequest) -> dict:
    validate_fake_input_preview_request(request)
    d = {
        "contract_version": request.contract_version,
        "preview_kind": request.preview_kind,
        "architecture_id": request.architecture_id,
        "source_descriptor_contract_version": request.source_descriptor_contract_version,
        "descriptor_seed": request.descriptor_seed,
        "min_value": request.min_value,
        "max_value": request.max_value,
        "batch_size": request.batch_size,
        "input_flat_dim": request.input_flat_dim,
        "full_batch_scalar_count": request.full_batch_scalar_count,
        "preview_value_count": request.preview_value_count,
        "allow_rng_execution_in_p34": request.allow_rng_execution_in_p34,
        "allow_full_batch_materialization_in_p34": request.allow_full_batch_materialization_in_p34,
        "allow_array_materialization_in_p34": request.allow_array_materialization_in_p34,
        "allow_tensor_materialization_in_p34": request.allow_tensor_materialization_in_p34,
        "allow_forward_execution_in_p34": request.allow_forward_execution_in_p34,
        "reason": request.reason,
    }
    assert_no_local_path_leakage(d)
    assert_no_forbidden_claims(d)
    return d


def fake_input_scalar_preview_to_json_dict(preview: FCVAEFakeInputScalarPreview) -> dict:
    validate_fake_input_scalar_preview(preview)
    d = {
        "contract_version": preview.contract_version,
        "value_kind": preview.value_kind,
        "preview_value_count": preview.preview_value_count,
        "preview_values": list(preview.preview_values),
        "min_value": preview.min_value,
        "max_value": preview.max_value,
        "all_values_within_range": preview.all_values_within_range,
        "rng_executed": preview.rng_executed,
        "full_batch_materialized": preview.full_batch_materialized,
        "array_materialized": preview.array_materialized,
        "tensor_materialized": preview.tensor_materialized,
        "reason": preview.reason,
    }
    assert_no_local_path_leakage(d)
    assert_no_forbidden_claims(d)
    return d


def fake_input_preview_metadata_to_json_dict(metadata: FCVAEFakeInputPreviewMetadata) -> dict:
    validate_fake_input_preview_metadata(metadata)
    d = {
        "contract_version": metadata.contract_version,
        "preview_kind": metadata.preview_kind,
        "descriptor_contract_version": metadata.descriptor_contract_version,
        "descriptor_status": metadata.descriptor_status,
        "torch_available": metadata.torch_available,
        "descriptor_available_in_p33": metadata.descriptor_available_in_p33,
        "full_batch_scalar_count": metadata.full_batch_scalar_count,
        "preview_value_count": metadata.preview_value_count,
        "preview_is_partial": metadata.preview_is_partial,
        "rng_execution_attempted": metadata.rng_execution_attempted,
        "full_batch_materialization_attempted": metadata.full_batch_materialization_attempted,
        "array_materialization_attempted": metadata.array_materialization_attempted,
        "tensor_materialization_attempted": metadata.tensor_materialization_attempted,
        "forward_execution_attempted": metadata.forward_execution_attempted,
        "output_generation_attempted": metadata.output_generation_attempted,
        "reason": metadata.reason,
    }
    assert_no_local_path_leakage(d)
    assert_no_forbidden_claims(d)
    return d


def fake_input_preview_result_to_json_dict(result: FCVAEFakeInputPreviewResult) -> dict:
    validate_fake_input_preview_result(result)
    d = {
        "contract_version": result.contract_version,
        "request": fake_input_preview_request_to_json_dict(result.request),
        "preview": fake_input_scalar_preview_to_json_dict(result.preview),
        "metadata": fake_input_preview_metadata_to_json_dict(result.metadata),
        "status": result.status,
        "preview_available_in_p34": result.preview_available_in_p34,
        "full_batch_available_in_p34": result.full_batch_available_in_p34,
        "rng_execution_available_in_p34": result.rng_execution_available_in_p34,
        "array_materialization_available_in_p34": result.array_materialization_available_in_p34,
        "tensor_materialization_available_in_p34": result.tensor_materialization_available_in_p34,
        "forward_execution_available_in_p34": result.forward_execution_available_in_p34,
        "output_generation_available_in_p34": result.output_generation_available_in_p34,
        "no_rng_execution": result.no_rng_execution,
        "no_full_batch_materialized": result.no_full_batch_materialized,
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


def compact_fake_input_preview_json(result: FCVAEFakeInputPreviewResult) -> str:
    d = fake_input_preview_result_to_json_dict(result)
    return json.dumps(d, sort_keys=True, separators=(",", ":"))
