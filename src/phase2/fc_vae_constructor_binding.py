# src/phase2/fc_vae_constructor_binding.py

import dataclasses
import json
from typing import Any, Dict, Tuple, Union

from src.phase2.fc_vae_model import (
    FC_VAE_MODEL_SKELETON_CONTRACT_VERSION,
    build_smoke_input_shape_contract,
    build_smoke_latent_layout,
    build_smoke_decoder_output_contract,
    build_smoke_forward_contract,
)
from src.phase2.fc_vae_torch_module_stub import (
    FC_VAE_TORCH_MODULE_STUB_CONTRACT_VERSION,
    run_fc_vae_torch_module_stub_probe,
    stub_result_to_json_dict,
)


# Public constants
FC_VAE_CONSTRUCTOR_BINDING_CONTRACT_VERSION = "phase2_p30_constructor_binding_contract_v1"

FC_VAE_CONSTRUCTOR_BINDING_KIND = "torch_module_constructor_spec_binding"

FC_VAE_CONSTRUCTOR_BINDING_MODULE_NAME = "src.phase2.fc_vae_constructor_binding"

FC_VAE_CONSTRUCTOR_BINDING_STATUS_TORCH_UNAVAILABLE = "blocked_torch_unavailable"

FC_VAE_CONSTRUCTOR_BINDING_STATUS_BOUND_TO_STUB = "bound_to_stub_no_forward_no_layers"

FC_VAE_CONSTRUCTOR_BINDING_STATUS_STUB_DEFERRED = "blocked_stub_deferred"

SUPPORTED_FC_VAE_CONSTRUCTOR_BINDING_STATUSES = (
    "blocked_torch_unavailable",
    "bound_to_stub_no_forward_no_layers",
    "blocked_stub_deferred",
)


# Public Dataclasses
@dataclasses.dataclass(frozen=True)
class FCVAEConstructorBindingRequest:
    contract_version: str
    binding_kind: str
    architecture_id: str
    input_flat_dim: int
    z_mean_dim: int
    z_volatility_dim: int
    z_shared_dim: int
    decoder_output_kind: str
    target_boundary: str
    allow_forward_in_p30: bool
    allow_layers_in_p30: bool
    allow_training_in_p30: bool
    reason: str


@dataclasses.dataclass(frozen=True)
class FCVAEConstructorBindingMetadata:
    contract_version: str
    binding_kind: str
    module_name: str
    architecture_id: str
    input_shape_contract_version: str
    latent_layout_contract_version: str
    decoder_output_contract_version: str
    forward_contract_version: str
    stub_contract_version: str
    stub_status: str
    torch_available: bool
    module_created: bool
    binding_created: bool
    bound_to_module_object: bool
    defines_forward: bool
    defines_layers: bool
    parameter_count: int
    buffer_count: int
    reason: str


@dataclasses.dataclass(frozen=True)
class FCVAEConstructorBindingResult:
    contract_version: str
    request: FCVAEConstructorBindingRequest
    metadata: FCVAEConstructorBindingMetadata
    status: str
    torch_required_for_p30: bool
    torch_required_for_future_execution: bool
    constructor_binding_available_in_p30: bool
    forward_available_in_p30: bool
    layers_available_in_p30: bool
    training_available_in_p30: bool
    module_object_returned: bool
    no_training_loop: bool
    no_optimizer: bool
    no_checkpointing: bool
    no_artifact_generation: bool
    no_final_comparison: bool
    no_scientific_conclusion: bool
    reason: str


# Public validation / safety functions
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


def validate_binding_status(value: Any) -> None:
    if type(value) is not str:
        raise TypeError(f"status must be exact str, got {type(value).__name__}")
    if value not in SUPPORTED_FC_VAE_CONSTRUCTOR_BINDING_STATUSES:
        raise ValueError(f"Unsupported constructor binding status: {value}")


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
                   .replace("constructor_binding_available_in_p30=false", ""))
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


def validate_constructor_binding_request(request: FCVAEConstructorBindingRequest) -> None:
    if type(request) is not FCVAEConstructorBindingRequest:
        raise TypeError("request must be exact FCVAEConstructorBindingRequest instance")

    validate_non_empty_str(request.contract_version, "contract_version")
    if request.contract_version != FC_VAE_CONSTRUCTOR_BINDING_CONTRACT_VERSION:
        raise ValueError(f"Invalid contract_version: {request.contract_version}")

    validate_non_empty_str(request.binding_kind, "binding_kind")
    if request.binding_kind != FC_VAE_CONSTRUCTOR_BINDING_KIND:
        raise ValueError(f"Invalid binding_kind: {request.binding_kind}")

    validate_non_empty_str(request.architecture_id, "architecture_id")
    if request.architecture_id != "FC-VAE":
        raise ValueError(f"Invalid architecture_id: {request.architecture_id}")

    validate_positive_int(request.input_flat_dim, "input_flat_dim")
    validate_positive_int(request.z_mean_dim, "z_mean_dim")
    validate_positive_int(request.z_volatility_dim, "z_volatility_dim")
    validate_positive_int(request.z_shared_dim, "z_shared_dim")

    validate_non_empty_str(request.decoder_output_kind, "decoder_output_kind")
    if request.decoder_output_kind != "typed_modelspec_candidate":
        raise ValueError(f"Invalid decoder_output_kind: {request.decoder_output_kind}")

    validate_non_empty_str(request.target_boundary, "target_boundary")
    if request.target_boundary != "ModelSpec":
        raise ValueError(f"Invalid target_boundary: {request.target_boundary}")

    validate_bool(request.allow_forward_in_p30, "allow_forward_in_p30")
    if request.allow_forward_in_p30:
        raise ValueError("allow_forward_in_p30 must be False")

    validate_bool(request.allow_layers_in_p30, "allow_layers_in_p30")
    if request.allow_layers_in_p30:
        raise ValueError("allow_layers_in_p30 must be False")

    validate_bool(request.allow_training_in_p30, "allow_training_in_p30")
    if request.allow_training_in_p30:
        raise ValueError("allow_training_in_p30 must be False")

    validate_non_empty_str(request.reason, "reason")
    assert_no_local_path_leakage(request.reason)
    assert_no_forbidden_claims(request.reason)


def validate_constructor_binding_metadata(metadata: FCVAEConstructorBindingMetadata) -> None:
    if type(metadata) is not FCVAEConstructorBindingMetadata:
        raise TypeError("metadata must be exact FCVAEConstructorBindingMetadata instance")

    validate_non_empty_str(metadata.contract_version, "contract_version")
    if metadata.contract_version != FC_VAE_CONSTRUCTOR_BINDING_CONTRACT_VERSION:
        raise ValueError(f"Invalid contract_version: {metadata.contract_version}")

    validate_non_empty_str(metadata.binding_kind, "binding_kind")
    if metadata.binding_kind != FC_VAE_CONSTRUCTOR_BINDING_KIND:
        raise ValueError(f"Invalid binding_kind: {metadata.binding_kind}")

    validate_non_empty_str(metadata.module_name, "module_name")
    if metadata.module_name != FC_VAE_CONSTRUCTOR_BINDING_MODULE_NAME:
        raise ValueError(f"Invalid module_name: {metadata.module_name}")

    validate_non_empty_str(metadata.architecture_id, "architecture_id")
    if metadata.architecture_id != "FC-VAE":
        raise ValueError(f"Invalid architecture_id: {metadata.architecture_id}")

    validate_non_empty_str(metadata.input_shape_contract_version, "input_shape_contract_version")
    if metadata.input_shape_contract_version != FC_VAE_MODEL_SKELETON_CONTRACT_VERSION:
        raise ValueError(f"Invalid input_shape_contract_version: {metadata.input_shape_contract_version}")

    validate_non_empty_str(metadata.latent_layout_contract_version, "latent_layout_contract_version")
    if metadata.latent_layout_contract_version != FC_VAE_MODEL_SKELETON_CONTRACT_VERSION:
        raise ValueError(f"Invalid latent_layout_contract_version: {metadata.latent_layout_contract_version}")

    validate_non_empty_str(metadata.decoder_output_contract_version, "decoder_output_contract_version")
    if metadata.decoder_output_contract_version != FC_VAE_MODEL_SKELETON_CONTRACT_VERSION:
        raise ValueError(f"Invalid decoder_output_contract_version: {metadata.decoder_output_contract_version}")

    validate_non_empty_str(metadata.forward_contract_version, "forward_contract_version")
    if metadata.forward_contract_version != FC_VAE_MODEL_SKELETON_CONTRACT_VERSION:
        raise ValueError(f"Invalid forward_contract_version: {metadata.forward_contract_version}")

    validate_non_empty_str(metadata.stub_contract_version, "stub_contract_version")
    if metadata.stub_contract_version != FC_VAE_TORCH_MODULE_STUB_CONTRACT_VERSION:
        raise ValueError(f"Invalid stub_contract_version: {metadata.stub_contract_version}")

    validate_non_empty_str(metadata.stub_status, "stub_status")
    validate_bool(metadata.torch_available, "torch_available")
    validate_bool(metadata.module_created, "module_created")
    validate_bool(metadata.binding_created, "binding_created")

    validate_bool(metadata.bound_to_module_object, "bound_to_module_object")
    if metadata.bound_to_module_object:
        raise ValueError("bound_to_module_object must be False")

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


def validate_constructor_binding_result(result: FCVAEConstructorBindingResult) -> None:
    if type(result) is not FCVAEConstructorBindingResult:
        raise TypeError("result must be exact FCVAEConstructorBindingResult instance")

    validate_non_empty_str(result.contract_version, "contract_version")
    if result.contract_version != FC_VAE_CONSTRUCTOR_BINDING_CONTRACT_VERSION:
        raise ValueError(f"Invalid contract_version: {result.contract_version}")

    validate_constructor_binding_request(result.request)
    validate_constructor_binding_metadata(result.metadata)
    validate_binding_status(result.status)

    validate_bool(result.torch_required_for_p30, "torch_required_for_p30")
    if result.torch_required_for_p30:
        raise ValueError("torch_required_for_p30 must be False")

    validate_bool(result.torch_required_for_future_execution, "torch_required_for_future_execution")
    if not result.torch_required_for_future_execution:
        raise ValueError("torch_required_for_future_execution must be True")

    validate_bool(result.constructor_binding_available_in_p30, "constructor_binding_available_in_p30")
    if result.constructor_binding_available_in_p30 != result.metadata.binding_created:
        raise ValueError("constructor_binding_available_in_p30 must equal binding_created")

    validate_bool(result.forward_available_in_p30, "forward_available_in_p30")
    if result.forward_available_in_p30:
        raise ValueError("forward_available_in_p30 must be False")

    validate_bool(result.layers_available_in_p30, "layers_available_in_p30")
    if result.layers_available_in_p30:
        raise ValueError("layers_available_in_p30 must be False")

    validate_bool(result.training_available_in_p30, "training_available_in_p30")
    if result.training_available_in_p30:
        raise ValueError("training_available_in_p30 must be False")

    validate_bool(result.module_object_returned, "module_object_returned")
    if result.module_object_returned:
        raise ValueError("module_object_returned must be False")

    validate_bool(result.no_training_loop, "no_training_loop")
    if not result.no_training_loop:
        raise ValueError("no_training_loop must be True")

    validate_bool(result.no_optimizer, "no_optimizer")
    if not result.no_optimizer:
        raise ValueError("no_optimizer must be True")

    validate_bool(result.no_checkpointing, "no_checkpointing")
    if not result.no_checkpointing:
        raise ValueError("no_checkpointing must be True")

    validate_bool(result.no_artifact_generation, "no_artifact_generation")
    if not result.no_artifact_generation:
        raise ValueError("no_artifact_generation must be True")

    validate_bool(result.no_final_comparison, "no_final_comparison")
    if not result.no_final_comparison:
        raise ValueError("no_final_comparison must be True")

    validate_bool(result.no_scientific_conclusion, "no_scientific_conclusion")
    if not result.no_scientific_conclusion:
        raise ValueError("no_scientific_conclusion must be True")

    # Consistent check status
    if not result.metadata.torch_available:
        if result.status != FC_VAE_CONSTRUCTOR_BINDING_STATUS_TORCH_UNAVAILABLE:
            raise ValueError(f"Inconsistent state: status must be '{FC_VAE_CONSTRUCTOR_BINDING_STATUS_TORCH_UNAVAILABLE}' when torch is unavailable")
    else:
        if result.metadata.binding_created:
            if result.status != FC_VAE_CONSTRUCTOR_BINDING_STATUS_BOUND_TO_STUB:
                raise ValueError(f"Inconsistent state: status must be '{FC_VAE_CONSTRUCTOR_BINDING_STATUS_BOUND_TO_STUB}' when binding is created")
        else:
            if result.status != FC_VAE_CONSTRUCTOR_BINDING_STATUS_STUB_DEFERRED:
                raise ValueError(f"Inconsistent state: status must be '{FC_VAE_CONSTRUCTOR_BINDING_STATUS_STUB_DEFERRED}' when binding is deferred")

    validate_non_empty_str(result.reason, "reason")
    assert_no_local_path_leakage(result.reason)
    assert_no_forbidden_claims(result.reason)


# Public Builders
def build_constructor_binding_request_from_p27_smoke_contracts() -> FCVAEConstructorBindingRequest:
    input_shape = build_smoke_input_shape_contract()
    latent_layout = build_smoke_latent_layout()
    decoder_output = build_smoke_decoder_output_contract()
    forward_contract = build_smoke_forward_contract()

    req = FCVAEConstructorBindingRequest(
        contract_version=FC_VAE_CONSTRUCTOR_BINDING_CONTRACT_VERSION,
        binding_kind=FC_VAE_CONSTRUCTOR_BINDING_KIND,
        architecture_id="FC-VAE",
        input_flat_dim=input_shape.flat_dim,
        z_mean_dim=latent_layout.z_mean_dim,
        z_volatility_dim=latent_layout.z_volatility_dim,
        z_shared_dim=latent_layout.z_shared_dim,
        decoder_output_kind=decoder_output.output_kind,
        target_boundary=decoder_output.target_boundary,
        allow_forward_in_p30=False,
        allow_layers_in_p30=False,
        allow_training_in_p30=False,
        reason="p30_binding_request_from_p27_smoke_contracts",
    )
    validate_constructor_binding_request(req)
    return req


def build_constructor_binding_metadata(request: FCVAEConstructorBindingRequest) -> FCVAEConstructorBindingMetadata:
    validate_constructor_binding_request(request)

    stub_result = run_fc_vae_torch_module_stub_probe()
    torch_available = stub_result.torch_available
    module_created = stub_result.metadata.module_created

    binding_created = False
    reason = "binding_not_created_stub_deferred"

    if torch_available and module_created and stub_result.status == "stub_created_no_forward_no_layers":
        binding_created = True
        reason = "binding_successfully_created_over_stub"
    elif not torch_available:
        reason = "binding_blocked_torch_unavailable"

    meta = FCVAEConstructorBindingMetadata(
        contract_version=FC_VAE_CONSTRUCTOR_BINDING_CONTRACT_VERSION,
        binding_kind=FC_VAE_CONSTRUCTOR_BINDING_KIND,
        module_name=FC_VAE_CONSTRUCTOR_BINDING_MODULE_NAME,
        architecture_id="FC-VAE",
        input_shape_contract_version=FC_VAE_MODEL_SKELETON_CONTRACT_VERSION,
        latent_layout_contract_version=FC_VAE_MODEL_SKELETON_CONTRACT_VERSION,
        decoder_output_contract_version=FC_VAE_MODEL_SKELETON_CONTRACT_VERSION,
        forward_contract_version=FC_VAE_MODEL_SKELETON_CONTRACT_VERSION,
        stub_contract_version=FC_VAE_TORCH_MODULE_STUB_CONTRACT_VERSION,
        stub_status=stub_result.status,
        torch_available=torch_available,
        module_created=module_created,
        binding_created=binding_created,
        bound_to_module_object=False,
        defines_forward=False,
        defines_layers=False,
        parameter_count=0,
        buffer_count=0,
        reason=reason,
    )
    validate_constructor_binding_metadata(meta)
    return meta


def build_constructor_binding_result(request: FCVAEConstructorBindingRequest) -> FCVAEConstructorBindingResult:
    validate_constructor_binding_request(request)
    meta = build_constructor_binding_metadata(request)

    if not meta.torch_available:
        status = FC_VAE_CONSTRUCTOR_BINDING_STATUS_TORCH_UNAVAILABLE
        reason = "constructor_binding_blocked_torch_unavailable"
    elif meta.binding_created:
        status = FC_VAE_CONSTRUCTOR_BINDING_STATUS_BOUND_TO_STUB
        reason = "constructor_binding_bound_to_stub"
    else:
        status = FC_VAE_CONSTRUCTOR_BINDING_STATUS_STUB_DEFERRED
        reason = "constructor_binding_blocked_stub_deferred"

    res = FCVAEConstructorBindingResult(
        contract_version=FC_VAE_CONSTRUCTOR_BINDING_CONTRACT_VERSION,
        request=request,
        metadata=meta,
        status=status,
        torch_required_for_p30=False,
        torch_required_for_future_execution=True,
        constructor_binding_available_in_p30=meta.binding_created,
        forward_available_in_p30=False,
        layers_available_in_p30=False,
        training_available_in_p30=False,
        module_object_returned=False,
        no_training_loop=True,
        no_optimizer=True,
        no_checkpointing=True,
        no_artifact_generation=True,
        no_final_comparison=True,
        no_scientific_conclusion=True,
        reason=reason,
    )
    validate_constructor_binding_result(res)
    return res


def run_constructor_binding_probe() -> FCVAEConstructorBindingResult:
    req = build_constructor_binding_request_from_p27_smoke_contracts()
    res = build_constructor_binding_result(req)
    validate_constructor_binding_result(res)
    return res


# Public Serialization
def constructor_binding_request_to_json_dict(request: FCVAEConstructorBindingRequest) -> dict:
    validate_constructor_binding_request(request)
    d = {
        "contract_version": request.contract_version,
        "binding_kind": request.binding_kind,
        "architecture_id": request.architecture_id,
        "input_flat_dim": request.input_flat_dim,
        "z_mean_dim": request.z_mean_dim,
        "z_volatility_dim": request.z_volatility_dim,
        "z_shared_dim": request.z_shared_dim,
        "decoder_output_kind": request.decoder_output_kind,
        "target_boundary": request.target_boundary,
        "allow_forward_in_p30": request.allow_forward_in_p30,
        "allow_layers_in_p30": request.allow_layers_in_p30,
        "allow_training_in_p30": request.allow_training_in_p30,
        "reason": request.reason,
    }
    assert_no_local_path_leakage(d)
    assert_no_forbidden_claims(d)
    return d


def constructor_binding_metadata_to_json_dict(metadata: FCVAEConstructorBindingMetadata) -> dict:
    validate_constructor_binding_metadata(metadata)
    d = {
        "contract_version": metadata.contract_version,
        "binding_kind": metadata.binding_kind,
        "module_name": metadata.module_name,
        "architecture_id": metadata.architecture_id,
        "input_shape_contract_version": metadata.input_shape_contract_version,
        "latent_layout_contract_version": metadata.latent_layout_contract_version,
        "decoder_output_contract_version": metadata.decoder_output_contract_version,
        "forward_contract_version": metadata.forward_contract_version,
        "stub_contract_version": metadata.stub_contract_version,
        "stub_status": metadata.stub_status,
        "torch_available": metadata.torch_available,
        "module_created": metadata.module_created,
        "binding_created": metadata.binding_created,
        "bound_to_module_object": metadata.bound_to_module_object,
        "defines_forward": metadata.defines_forward,
        "defines_layers": metadata.defines_layers,
        "parameter_count": metadata.parameter_count,
        "buffer_count": metadata.buffer_count,
        "reason": metadata.reason,
    }
    assert_no_local_path_leakage(d)
    assert_no_forbidden_claims(d)
    return d


def constructor_binding_result_to_json_dict(result: FCVAEConstructorBindingResult) -> dict:
    validate_constructor_binding_result(result)
    d = {
        "contract_version": result.contract_version,
        "request": constructor_binding_request_to_json_dict(result.request),
        "metadata": constructor_binding_metadata_to_json_dict(result.metadata),
        "status": result.status,
        "torch_required_for_p30": result.torch_required_for_p30,
        "torch_required_for_future_execution": result.torch_required_for_future_execution,
        "constructor_binding_available_in_p30": result.constructor_binding_available_in_p30,
        "forward_available_in_p30": result.forward_available_in_p30,
        "layers_available_in_p30": result.layers_available_in_p30,
        "training_available_in_p30": result.training_available_in_p30,
        "module_object_returned": result.module_object_returned,
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


def compact_constructor_binding_json(result: FCVAEConstructorBindingResult) -> str:
    d = constructor_binding_result_to_json_dict(result)
    return json.dumps(d, sort_keys=True, separators=(",", ":"))
