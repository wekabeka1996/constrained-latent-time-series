# src/phase2/fc_vae_forward_boundary.py

import dataclasses
import json
from typing import Any, Dict, Tuple, Union

from src.phase2.fc_vae_model import (
    FC_VAE_MODEL_SKELETON_CONTRACT_VERSION,
    build_smoke_input_shape_contract,
    build_smoke_latent_layout,
    build_smoke_decoder_output_contract,
)
from src.phase2.fc_vae_constructor_binding import (
    FC_VAE_CONSTRUCTOR_BINDING_CONTRACT_VERSION,
    run_constructor_binding_probe,
    constructor_binding_result_to_json_dict,
)


# Public Constants
FC_VAE_FORWARD_BOUNDARY_CONTRACT_VERSION = "phase2_p31_noop_forward_boundary_contract_v1"

FC_VAE_FORWARD_BOUNDARY_KIND = "noop_forward_boundary_contract"

FC_VAE_FORWARD_BOUNDARY_MODULE_NAME = "src.phase2.fc_vae_forward_boundary"

FC_VAE_FORWARD_BOUNDARY_STATUS_TORCH_UNAVAILABLE = "blocked_torch_unavailable"

FC_VAE_FORWARD_BOUNDARY_STATUS_CONSTRUCTOR_NOT_BOUND = "blocked_constructor_not_bound"

FC_VAE_FORWARD_BOUNDARY_STATUS_NOOP_BLOCKED = "noop_forward_blocked_in_p31"

SUPPORTED_FC_VAE_FORWARD_BOUNDARY_STATUSES = (
    "blocked_torch_unavailable",
    "blocked_constructor_not_bound",
    "noop_forward_blocked_in_p31",
)

FC_VAE_FORWARD_BOUNDARY_OUTPUT_KIND = "declared_noop_modelspec_logits_shape"


# Public Dataclasses (All Frozen)
@dataclasses.dataclass(frozen=True)
class FCVAEForwardBoundaryRequest:
    contract_version: str
    boundary_kind: str
    architecture_id: str
    input_flat_dim: int
    z_mean_dim: int
    z_volatility_dim: int
    z_shared_dim: int
    declared_family_head_dim: int
    declared_mean_head_dim: int
    declared_volatility_head_dim: int
    declared_diagnostic_head_dim: int
    output_kind: str
    target_boundary: str
    allow_execution_in_p31: bool
    allow_tensor_allocation_in_p31: bool
    allow_output_generation_in_p31: bool
    allow_training_in_p31: bool
    reason: str


@dataclasses.dataclass(frozen=True)
class FCVAEForwardBoundaryOutputShape:
    contract_version: str
    output_kind: str
    family_head_dim: int
    mean_head_dim: int
    volatility_head_dim: int
    diagnostic_head_dim: int
    total_declared_output_dim: int
    target_boundary: str
    generated_output: bool
    reason: str


@dataclasses.dataclass(frozen=True)
class FCVAEForwardBoundaryMetadata:
    contract_version: str
    boundary_kind: str
    module_name: str
    architecture_id: str
    constructor_binding_contract_version: str
    constructor_binding_status: str
    constructor_binding_available_in_p30: bool
    torch_available: bool
    module_created: bool
    binding_created: bool
    execution_attempted: bool
    tensor_allocation_attempted: bool
    output_generation_attempted: bool
    module_object_returned: bool
    defines_forward: bool
    defines_layers: bool
    parameter_count: int
    buffer_count: int
    reason: str


@dataclasses.dataclass(frozen=True)
class FCVAEForwardBoundaryResult:
    contract_version: str
    request: FCVAEForwardBoundaryRequest
    declared_output_shape: FCVAEForwardBoundaryOutputShape
    metadata: FCVAEForwardBoundaryMetadata
    status: str
    torch_required_for_p31: bool
    torch_required_for_future_execution: bool
    forward_execution_available_in_p31: bool
    tensor_allocation_available_in_p31: bool
    output_generation_available_in_p31: bool
    training_available_in_p31: bool
    loss_available_in_p31: bool
    optimizer_available_in_p31: bool
    checkpointing_available_in_p31: bool
    artifact_generation_available_in_p31: bool
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


def validate_non_negative_int(value: Any, name: str) -> None:
    if type(value) is not int or isinstance(value, bool):
        raise TypeError(f"{name} must be exact int, got {type(value).__name__}")
    if value < 0:
        raise ValueError(f"{name} must be >= 0")


def validate_forward_boundary_status(value: Any) -> None:
    if type(value) is not str:
        raise TypeError(f"status must be exact str, got {type(value).__name__}")
    if value not in SUPPORTED_FC_VAE_FORWARD_BOUNDARY_STATUSES:
        raise ValueError(f"Unsupported forward boundary status: {value}")


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
                   .replace("noop_forward_blocked_in_p31", "")
                   .replace("forward_execution_available_in_p31=false", ""))
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


def validate_forward_boundary_request(request: FCVAEForwardBoundaryRequest) -> None:
    if type(request) is not FCVAEForwardBoundaryRequest:
        raise TypeError("request must be exact FCVAEForwardBoundaryRequest instance")

    validate_non_empty_str(request.contract_version, "contract_version")
    if request.contract_version != FC_VAE_FORWARD_BOUNDARY_CONTRACT_VERSION:
        raise ValueError(f"Invalid contract_version: {request.contract_version}")

    validate_non_empty_str(request.boundary_kind, "boundary_kind")
    if request.boundary_kind != FC_VAE_FORWARD_BOUNDARY_KIND:
        raise ValueError(f"Invalid boundary_kind: {request.boundary_kind}")

    validate_non_empty_str(request.architecture_id, "architecture_id")
    if request.architecture_id != "FC-VAE":
        raise ValueError(f"Invalid architecture_id: {request.architecture_id}")

    validate_positive_int(request.input_flat_dim, "input_flat_dim")
    validate_positive_int(request.z_mean_dim, "z_mean_dim")
    validate_positive_int(request.z_volatility_dim, "z_volatility_dim")
    validate_positive_int(request.z_shared_dim, "z_shared_dim")

    validate_positive_int(request.declared_family_head_dim, "declared_family_head_dim")
    validate_positive_int(request.declared_mean_head_dim, "declared_mean_head_dim")
    validate_positive_int(request.declared_volatility_head_dim, "declared_volatility_head_dim")
    validate_positive_int(request.declared_diagnostic_head_dim, "declared_diagnostic_head_dim")

    validate_non_empty_str(request.output_kind, "output_kind")
    if request.output_kind != FC_VAE_FORWARD_BOUNDARY_OUTPUT_KIND:
        raise ValueError(f"Invalid output_kind: {request.output_kind}")

    validate_non_empty_str(request.target_boundary, "target_boundary")
    if request.target_boundary != "ModelSpec":
        raise ValueError(f"Invalid target_boundary: {request.target_boundary}")

    validate_bool(request.allow_execution_in_p31, "allow_execution_in_p31")
    if request.allow_execution_in_p31:
        raise ValueError("allow_execution_in_p31 must be False")

    validate_bool(request.allow_tensor_allocation_in_p31, "allow_tensor_allocation_in_p31")
    if request.allow_tensor_allocation_in_p31:
        raise ValueError("allow_tensor_allocation_in_p31 must be False")

    validate_bool(request.allow_output_generation_in_p31, "allow_output_generation_in_p31")
    if request.allow_output_generation_in_p31:
        raise ValueError("allow_output_generation_in_p31 must be False")

    validate_bool(request.allow_training_in_p31, "allow_training_in_p31")
    if request.allow_training_in_p31:
        raise ValueError("allow_training_in_p31 must be False")

    validate_non_empty_str(request.reason, "reason")
    assert_no_local_path_leakage(request.reason)
    assert_no_forbidden_claims(request.reason)


def validate_forward_boundary_output_shape(output_shape: FCVAEForwardBoundaryOutputShape) -> None:
    if type(output_shape) is not FCVAEForwardBoundaryOutputShape:
        raise TypeError("output_shape must be exact FCVAEForwardBoundaryOutputShape instance")

    validate_non_empty_str(output_shape.contract_version, "contract_version")
    if output_shape.contract_version != FC_VAE_FORWARD_BOUNDARY_CONTRACT_VERSION:
        raise ValueError(f"Invalid contract_version: {output_shape.contract_version}")

    validate_non_empty_str(output_shape.output_kind, "output_kind")
    if output_shape.output_kind != FC_VAE_FORWARD_BOUNDARY_OUTPUT_KIND:
        raise ValueError(f"Invalid output_kind: {output_shape.output_kind}")

    validate_positive_int(output_shape.family_head_dim, "family_head_dim")
    validate_positive_int(output_shape.mean_head_dim, "mean_head_dim")
    validate_positive_int(output_shape.volatility_head_dim, "volatility_head_dim")
    validate_positive_int(output_shape.diagnostic_head_dim, "diagnostic_head_dim")

    validate_positive_int(output_shape.total_declared_output_dim, "total_declared_output_dim")
    expected_total = (output_shape.family_head_dim +
                      output_shape.mean_head_dim +
                      output_shape.volatility_head_dim +
                      output_shape.diagnostic_head_dim)
    if output_shape.total_declared_output_dim != expected_total:
        raise ValueError(f"total_declared_output_dim must equal sum of head dimensions: {expected_total}")

    validate_non_empty_str(output_shape.target_boundary, "target_boundary")
    if output_shape.target_boundary != "ModelSpec":
        raise ValueError(f"Invalid target_boundary: {output_shape.target_boundary}")

    validate_bool(output_shape.generated_output, "generated_output")
    if output_shape.generated_output:
        raise ValueError("generated_output must be False")

    validate_non_empty_str(output_shape.reason, "reason")
    assert_no_local_path_leakage(output_shape.reason)
    assert_no_forbidden_claims(output_shape.reason)


def validate_forward_boundary_metadata(metadata: FCVAEForwardBoundaryMetadata) -> None:
    if type(metadata) is not FCVAEForwardBoundaryMetadata:
        raise TypeError("metadata must be exact FCVAEForwardBoundaryMetadata instance")

    validate_non_empty_str(metadata.contract_version, "contract_version")
    if metadata.contract_version != FC_VAE_FORWARD_BOUNDARY_CONTRACT_VERSION:
        raise ValueError(f"Invalid contract_version: {metadata.contract_version}")

    validate_non_empty_str(metadata.boundary_kind, "boundary_kind")
    if metadata.boundary_kind != FC_VAE_FORWARD_BOUNDARY_KIND:
        raise ValueError(f"Invalid boundary_kind: {metadata.boundary_kind}")

    validate_non_empty_str(metadata.module_name, "module_name")
    if metadata.module_name != FC_VAE_FORWARD_BOUNDARY_MODULE_NAME:
        raise ValueError(f"Invalid module_name: {metadata.module_name}")

    validate_non_empty_str(metadata.architecture_id, "architecture_id")
    if metadata.architecture_id != "FC-VAE":
        raise ValueError(f"Invalid architecture_id: {metadata.architecture_id}")

    validate_non_empty_str(metadata.constructor_binding_contract_version, "constructor_binding_contract_version")
    if metadata.constructor_binding_contract_version != FC_VAE_CONSTRUCTOR_BINDING_CONTRACT_VERSION:
        raise ValueError(f"Invalid constructor_binding_contract_version: {metadata.constructor_binding_contract_version}")

    validate_non_empty_str(metadata.constructor_binding_status, "constructor_binding_status")
    validate_bool(metadata.constructor_binding_available_in_p30, "constructor_binding_available_in_p30")
    validate_bool(metadata.torch_available, "torch_available")
    validate_bool(metadata.module_created, "module_created")
    validate_bool(metadata.binding_created, "binding_created")

    validate_bool(metadata.execution_attempted, "execution_attempted")
    if metadata.execution_attempted:
        raise ValueError("execution_attempted must be False")

    validate_bool(metadata.tensor_allocation_attempted, "tensor_allocation_attempted")
    if metadata.tensor_allocation_attempted:
        raise ValueError("tensor_allocation_attempted must be False")

    validate_bool(metadata.output_generation_attempted, "output_generation_attempted")
    if metadata.output_generation_attempted:
        raise ValueError("output_generation_attempted must be False")

    validate_bool(metadata.module_object_returned, "module_object_returned")
    if metadata.module_object_returned:
        raise ValueError("module_object_returned must be False")

    validate_bool(metadata.defines_forward, "defines_forward")
    if metadata.defines_forward:
        raise ValueError("defines_forward must be False")

    validate_bool(metadata.defines_layers, "defines_layers")
    if metadata.defines_layers:
        raise ValueError("defines_layers must be False")

    validate_non_negative_int(metadata.parameter_count, "parameter_count")
    if metadata.parameter_count != 0:
        raise ValueError("parameter_count must be 0")

    validate_non_negative_int(metadata.buffer_count, "buffer_count")
    if metadata.buffer_count != 0:
        raise ValueError("buffer_count must be 0")

    validate_non_empty_str(metadata.reason, "reason")
    assert_no_local_path_leakage(metadata.reason)
    assert_no_forbidden_claims(metadata.reason)


def validate_forward_boundary_result(result: FCVAEForwardBoundaryResult) -> None:
    if type(result) is not FCVAEForwardBoundaryResult:
        raise TypeError("result must be exact FCVAEForwardBoundaryResult instance")

    validate_non_empty_str(result.contract_version, "contract_version")
    if result.contract_version != FC_VAE_FORWARD_BOUNDARY_CONTRACT_VERSION:
        raise ValueError(f"Invalid contract_version: {result.contract_version}")

    validate_forward_boundary_request(result.request)
    validate_forward_boundary_output_shape(result.declared_output_shape)
    validate_forward_boundary_metadata(result.metadata)
    validate_forward_boundary_status(result.status)

    validate_bool(result.torch_required_for_p31, "torch_required_for_p31")
    if result.torch_required_for_p31:
        raise ValueError("torch_required_for_p31 must be False")

    validate_bool(result.torch_required_for_future_execution, "torch_required_for_future_execution")
    if not result.torch_required_for_future_execution:
        raise ValueError("torch_required_for_future_execution must be True")

    # all execution/training/loss/optimizer/checkpoint/artifact availability flags False
    for flag_name in (
        "forward_execution_available_in_p31",
        "tensor_allocation_available_in_p31",
        "output_generation_available_in_p31",
        "training_available_in_p31",
        "loss_available_in_p31",
        "optimizer_available_in_p31",
        "checkpointing_available_in_p31",
        "artifact_generation_available_in_p31",
    ):
        val = getattr(result, flag_name)
        validate_bool(val, flag_name)
        if val:
            raise ValueError(f"{flag_name} must be False")

    validate_bool(result.no_final_comparison, "no_final_comparison")
    if not result.no_final_comparison:
        raise ValueError("no_final_comparison must be True")

    validate_bool(result.no_scientific_conclusion, "no_scientific_conclusion")
    if not result.no_scientific_conclusion:
        raise ValueError("no_scientific_conclusion must be True")

    # Consistent check status
    if not result.metadata.torch_available:
        if result.status != FC_VAE_FORWARD_BOUNDARY_STATUS_TORCH_UNAVAILABLE:
            raise ValueError(f"Inconsistent state: status must be '{FC_VAE_FORWARD_BOUNDARY_STATUS_TORCH_UNAVAILABLE}' when torch is unavailable")
    else:
        if not result.metadata.constructor_binding_available_in_p30:
            if result.status != FC_VAE_FORWARD_BOUNDARY_STATUS_CONSTRUCTOR_NOT_BOUND:
                raise ValueError(f"Inconsistent state: status must be '{FC_VAE_FORWARD_BOUNDARY_STATUS_CONSTRUCTOR_NOT_BOUND}' when constructor is not bound")
        else:
            if result.status != FC_VAE_FORWARD_BOUNDARY_STATUS_NOOP_BLOCKED:
                raise ValueError(f"Inconsistent state: status must be '{FC_VAE_FORWARD_BOUNDARY_STATUS_NOOP_BLOCKED}' when constructor is bound")

    validate_non_empty_str(result.reason, "reason")
    assert_no_local_path_leakage(result.reason)
    assert_no_forbidden_claims(result.reason)


# Public Builders
def build_forward_boundary_request_from_p27_smoke_contracts() -> FCVAEForwardBoundaryRequest:
    input_shape = build_smoke_input_shape_contract()
    latent_layout = build_smoke_latent_layout()
    decoder_output = build_smoke_decoder_output_contract()

    req = FCVAEForwardBoundaryRequest(
        contract_version=FC_VAE_FORWARD_BOUNDARY_CONTRACT_VERSION,
        boundary_kind=FC_VAE_FORWARD_BOUNDARY_KIND,
        architecture_id="FC-VAE",
        input_flat_dim=input_shape.flat_dim,
        z_mean_dim=latent_layout.z_mean_dim,
        z_volatility_dim=latent_layout.z_volatility_dim,
        z_shared_dim=latent_layout.z_shared_dim,
        declared_family_head_dim=decoder_output.family_head_dim,
        declared_mean_head_dim=decoder_output.mean_head_dim,
        declared_volatility_head_dim=decoder_output.volatility_head_dim,
        declared_diagnostic_head_dim=decoder_output.diagnostic_head_dim,
        output_kind=FC_VAE_FORWARD_BOUNDARY_OUTPUT_KIND,
        target_boundary=decoder_output.target_boundary,
        allow_execution_in_p31=False,
        allow_tensor_allocation_in_p31=False,
        allow_output_generation_in_p31=False,
        allow_training_in_p31=False,
        reason="p31_forward_boundary_request_from_p27_smoke_contracts",
    )
    validate_forward_boundary_request(req)
    return req


def build_forward_boundary_output_shape(request: FCVAEForwardBoundaryRequest) -> FCVAEForwardBoundaryOutputShape:
    validate_forward_boundary_request(request)

    total_dim = (request.declared_family_head_dim +
                 request.declared_mean_head_dim +
                 request.declared_volatility_head_dim +
                 request.declared_diagnostic_head_dim)

    shape = FCVAEForwardBoundaryOutputShape(
        contract_version=FC_VAE_FORWARD_BOUNDARY_CONTRACT_VERSION,
        output_kind=FC_VAE_FORWARD_BOUNDARY_OUTPUT_KIND,
        family_head_dim=request.declared_family_head_dim,
        mean_head_dim=request.declared_mean_head_dim,
        volatility_head_dim=request.declared_volatility_head_dim,
        diagnostic_head_dim=request.declared_diagnostic_head_dim,
        total_declared_output_dim=total_dim,
        target_boundary="ModelSpec",
        generated_output=False,
        reason="p31_forward_boundary_output_shape",
    )
    validate_forward_boundary_output_shape(shape)
    return shape


def build_forward_boundary_metadata(request: FCVAEForwardBoundaryRequest) -> FCVAEForwardBoundaryMetadata:
    validate_forward_boundary_request(request)

    p30_result = run_constructor_binding_probe()
    torch_available = p30_result.metadata.torch_available
    module_created = p30_result.metadata.module_created
    binding_created = p30_result.metadata.binding_created

    meta = FCVAEForwardBoundaryMetadata(
        contract_version=FC_VAE_FORWARD_BOUNDARY_CONTRACT_VERSION,
        boundary_kind=FC_VAE_FORWARD_BOUNDARY_KIND,
        module_name=FC_VAE_FORWARD_BOUNDARY_MODULE_NAME,
        architecture_id="FC-VAE",
        constructor_binding_contract_version=p30_result.contract_version,
        constructor_binding_status=p30_result.status,
        constructor_binding_available_in_p30=p30_result.constructor_binding_available_in_p30,
        torch_available=torch_available,
        module_created=module_created,
        binding_created=binding_created,
        execution_attempted=False,
        tensor_allocation_attempted=False,
        output_generation_attempted=False,
        module_object_returned=False,
        defines_forward=False,
        defines_layers=False,
        parameter_count=0,
        buffer_count=0,
        reason="p31_forward_boundary_metadata",
    )
    validate_forward_boundary_metadata(meta)
    return meta


def build_forward_boundary_result(request: FCVAEForwardBoundaryRequest) -> FCVAEForwardBoundaryResult:
    validate_forward_boundary_request(request)
    output_shape = build_forward_boundary_output_shape(request)
    meta = build_forward_boundary_metadata(request)

    if not meta.torch_available:
        status = FC_VAE_FORWARD_BOUNDARY_STATUS_TORCH_UNAVAILABLE
        reason = "forward_boundary_blocked_torch_unavailable"
    elif not meta.constructor_binding_available_in_p30:
        status = FC_VAE_FORWARD_BOUNDARY_STATUS_CONSTRUCTOR_NOT_BOUND
        reason = "forward_boundary_blocked_constructor_not_bound"
    else:
        status = FC_VAE_FORWARD_BOUNDARY_STATUS_NOOP_BLOCKED
        reason = "forward_boundary_noop_blocked"

    res = FCVAEForwardBoundaryResult(
        contract_version=FC_VAE_FORWARD_BOUNDARY_CONTRACT_VERSION,
        request=request,
        declared_output_shape=output_shape,
        metadata=meta,
        status=status,
        torch_required_for_p31=False,
        torch_required_for_future_execution=True,
        forward_execution_available_in_p31=False,
        tensor_allocation_available_in_p31=False,
        output_generation_available_in_p31=False,
        training_available_in_p31=False,
        loss_available_in_p31=False,
        optimizer_available_in_p31=False,
        checkpointing_available_in_p31=False,
        artifact_generation_available_in_p31=False,
        no_final_comparison=True,
        no_scientific_conclusion=True,
        reason=reason,
    )
    validate_forward_boundary_result(res)
    return res


def run_forward_boundary_probe() -> FCVAEForwardBoundaryResult:
    req = build_forward_boundary_request_from_p27_smoke_contracts()
    res = build_forward_boundary_result(req)
    validate_forward_boundary_result(res)
    return res


# Public Serialization
def forward_boundary_request_to_json_dict(request: FCVAEForwardBoundaryRequest) -> dict:
    validate_forward_boundary_request(request)
    d = {
        "contract_version": request.contract_version,
        "boundary_kind": request.boundary_kind,
        "architecture_id": request.architecture_id,
        "input_flat_dim": request.input_flat_dim,
        "z_mean_dim": request.z_mean_dim,
        "z_volatility_dim": request.z_volatility_dim,
        "z_shared_dim": request.z_shared_dim,
        "declared_family_head_dim": request.declared_family_head_dim,
        "declared_mean_head_dim": request.declared_mean_head_dim,
        "declared_volatility_head_dim": request.declared_volatility_head_dim,
        "declared_diagnostic_head_dim": request.declared_diagnostic_head_dim,
        "output_kind": request.output_kind,
        "target_boundary": request.target_boundary,
        "allow_execution_in_p31": request.allow_execution_in_p31,
        "allow_tensor_allocation_in_p31": request.allow_tensor_allocation_in_p31,
        "allow_output_generation_in_p31": request.allow_output_generation_in_p31,
        "allow_training_in_p31": request.allow_training_in_p31,
        "reason": request.reason,
    }
    assert_no_local_path_leakage(d)
    assert_no_forbidden_claims(d)
    return d


def forward_boundary_output_shape_to_json_dict(output_shape: FCVAEForwardBoundaryOutputShape) -> dict:
    validate_forward_boundary_output_shape(output_shape)
    d = {
        "contract_version": output_shape.contract_version,
        "output_kind": output_shape.output_kind,
        "family_head_dim": output_shape.family_head_dim,
        "mean_head_dim": output_shape.mean_head_dim,
        "volatility_head_dim": output_shape.volatility_head_dim,
        "diagnostic_head_dim": output_shape.diagnostic_head_dim,
        "total_declared_output_dim": output_shape.total_declared_output_dim,
        "target_boundary": output_shape.target_boundary,
        "generated_output": output_shape.generated_output,
        "reason": output_shape.reason,
    }
    assert_no_local_path_leakage(d)
    assert_no_forbidden_claims(d)
    return d


def forward_boundary_metadata_to_json_dict(metadata: FCVAEForwardBoundaryMetadata) -> dict:
    validate_forward_boundary_metadata(metadata)
    d = {
        "contract_version": metadata.contract_version,
        "boundary_kind": metadata.boundary_kind,
        "module_name": metadata.module_name,
        "architecture_id": metadata.architecture_id,
        "constructor_binding_contract_version": metadata.constructor_binding_contract_version,
        "constructor_binding_status": metadata.constructor_binding_status,
        "constructor_binding_available_in_p30": metadata.constructor_binding_available_in_p30,
        "torch_available": metadata.torch_available,
        "module_created": metadata.module_created,
        "binding_created": metadata.binding_created,
        "execution_attempted": metadata.execution_attempted,
        "tensor_allocation_attempted": metadata.tensor_allocation_attempted,
        "output_generation_attempted": metadata.output_generation_attempted,
        "module_object_returned": metadata.module_object_returned,
        "defines_forward": metadata.defines_forward,
        "defines_layers": metadata.defines_layers,
        "parameter_count": metadata.parameter_count,
        "buffer_count": metadata.buffer_count,
        "reason": metadata.reason,
    }
    assert_no_local_path_leakage(d)
    assert_no_forbidden_claims(d)
    return d


def forward_boundary_result_to_json_dict(result: FCVAEForwardBoundaryResult) -> dict:
    validate_forward_boundary_result(result)
    d = {
        "contract_version": result.contract_version,
        "request": forward_boundary_request_to_json_dict(result.request),
        "declared_output_shape": forward_boundary_output_shape_to_json_dict(result.declared_output_shape),
        "metadata": forward_boundary_metadata_to_json_dict(result.metadata),
        "status": result.status,
        "torch_required_for_p31": result.torch_required_for_p31,
        "torch_required_for_future_execution": result.torch_required_for_future_execution,
        "forward_execution_available_in_p31": result.forward_execution_available_in_p31,
        "tensor_allocation_available_in_p31": result.tensor_allocation_available_in_p31,
        "output_generation_available_in_p31": result.output_generation_available_in_p31,
        "training_available_in_p31": result.training_available_in_p31,
        "loss_available_in_p31": result.loss_available_in_p31,
        "optimizer_available_in_p31": result.optimizer_available_in_p31,
        "checkpointing_available_in_p31": result.checkpointing_available_in_p31,
        "artifact_generation_available_in_p31": result.artifact_generation_available_in_p31,
        "no_final_comparison": result.no_final_comparison,
        "no_scientific_conclusion": result.no_scientific_conclusion,
        "reason": result.reason,
    }
    assert_no_local_path_leakage(d)
    assert_no_forbidden_claims(d)
    return d


def compact_forward_boundary_json(result: FCVAEForwardBoundaryResult) -> str:
    d = forward_boundary_result_to_json_dict(result)
    return json.dumps(d, sort_keys=True, separators=(",", ":"))
