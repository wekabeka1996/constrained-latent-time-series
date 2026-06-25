# src/phase2/fc_vae_torch_module_stub.py

import dataclasses
import json
from typing import Any, Dict, Tuple, Union

from src.phase2.torch_boundary import (
    build_torch_dependency_status,
)
from src.phase2.fc_vae_model import (
    FC_VAE_MODEL_SKELETON_CONTRACT_VERSION,
)
from src.phase2.fc_vae_torch_shell import (
    FC_VAE_TORCH_SHELL_CONTRACT_VERSION,
    run_fc_vae_torch_shell_probe,
)


# Public constants
FC_VAE_TORCH_MODULE_STUB_CONTRACT_VERSION = "phase2_p29_torch_module_stub_contract_v1"

FC_VAE_TORCH_MODULE_STUB_KIND = "gated_torch_nn_module_stub"

FC_VAE_TORCH_MODULE_STUB_MODULE_NAME = "src.phase2.fc_vae_torch_module_stub"

FC_VAE_TORCH_MODULE_STUB_CLASS_NAME = "P29LocalFCVAEModuleStub"

FC_VAE_TORCH_MODULE_STUB_STATUS_TORCH_UNAVAILABLE = "blocked_torch_unavailable"

FC_VAE_TORCH_MODULE_STUB_STATUS_STUB_CREATED = "stub_created_no_forward_no_layers"

FC_VAE_TORCH_MODULE_STUB_STATUS_IMPLEMENTATION_DEFERRED = "blocked_implementation_deferred"

SUPPORTED_FC_VAE_TORCH_MODULE_STUB_STATUSES = (
    "blocked_torch_unavailable",
    "stub_created_no_forward_no_layers",
    "blocked_implementation_deferred",
)


# Public dataclasses
@dataclasses.dataclass(frozen=True)
class FCVAETorchModuleStubRequest:
    contract_version: str
    stub_kind: str
    architecture_id: str
    shell_contract_version: str
    require_torch_available: bool
    allow_stub_creation_if_torch_available: bool
    allow_forward_in_p29: bool
    allow_layers_in_p29: bool
    reason: str


@dataclasses.dataclass(frozen=True)
class FCVAETorchModuleStubMetadata:
    contract_version: str
    stub_kind: str
    module_name: str
    class_name: str
    torch_available: bool
    module_created: bool
    is_torch_nn_module: bool
    defines_forward: bool
    defines_layers: bool
    parameter_count: int
    buffer_count: int
    reason: str


@dataclasses.dataclass(frozen=True)
class FCVAETorchModuleStubResult:
    contract_version: str
    request: FCVAETorchModuleStubRequest
    shell_status: str
    torch_available: bool
    status: str
    metadata: FCVAETorchModuleStubMetadata
    module_object_returned: bool
    torch_required_for_p29: bool
    torch_required_for_future_execution: bool
    implementation_available_in_p29: bool
    forward_available_in_p29: bool
    layers_available_in_p29: bool
    training_available_in_p29: bool
    no_training_loop: bool
    no_optimizer: bool
    no_checkpointing: bool
    no_artifact_generation: bool
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


def validate_non_negative_int(value: Any, name: str) -> None:
    if type(value) is not int or isinstance(value, bool):
        raise TypeError(f"{name} must be exact int, got {type(value).__name__}")
    if value < 0:
        raise ValueError(f"{name} must be >= 0")


def validate_stub_status(value: Any) -> None:
    if type(value) is not str:
        raise TypeError(f"status must be exact str, got {type(value).__name__}")
    if value not in SUPPORTED_FC_VAE_TORCH_MODULE_STUB_STATUSES:
        raise ValueError(f"Unsupported stub status: {value}")


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
        cleaned = normalized.replace("no_final_comparison", "").replace("no_scientific_conclusion", "").replace("implementation_available_in_p29=false", "")
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


def validate_stub_request(request: FCVAETorchModuleStubRequest) -> None:
    if type(request) is not FCVAETorchModuleStubRequest:
        raise TypeError("request must be exact FCVAETorchModuleStubRequest instance")

    validate_non_empty_str(request.contract_version, "contract_version")
    if request.contract_version != FC_VAE_TORCH_MODULE_STUB_CONTRACT_VERSION:
        raise ValueError(f"Invalid contract_version: {request.contract_version}")

    validate_non_empty_str(request.stub_kind, "stub_kind")
    if request.stub_kind != FC_VAE_TORCH_MODULE_STUB_KIND:
        raise ValueError(f"Invalid stub_kind: {request.stub_kind}")

    validate_non_empty_str(request.architecture_id, "architecture_id")
    if request.architecture_id != "FC-VAE":
        raise ValueError(f"Invalid architecture_id: {request.architecture_id}")

    validate_non_empty_str(request.shell_contract_version, "shell_contract_version")
    if request.shell_contract_version != FC_VAE_TORCH_SHELL_CONTRACT_VERSION:
        raise ValueError(f"Invalid shell_contract_version: {request.shell_contract_version}")

    validate_bool(request.require_torch_available, "require_torch_available")
    validate_bool(request.allow_stub_creation_if_torch_available, "allow_stub_creation_if_torch_available")
    validate_bool(request.allow_forward_in_p29, "allow_forward_in_p29")
    if request.allow_forward_in_p29:
        raise ValueError("allow_forward_in_p29 must be False in P29")

    validate_bool(request.allow_layers_in_p29, "allow_layers_in_p29")
    if request.allow_layers_in_p29:
        raise ValueError("allow_layers_in_p29 must be False in P29")

    validate_non_empty_str(request.reason, "reason")
    assert_no_local_path_leakage(request.reason)
    assert_no_forbidden_claims(request.reason)


def validate_stub_metadata(metadata: FCVAETorchModuleStubMetadata) -> None:
    if type(metadata) is not FCVAETorchModuleStubMetadata:
        raise TypeError("metadata must be exact FCVAETorchModuleStubMetadata instance")

    validate_non_empty_str(metadata.contract_version, "contract_version")
    if metadata.contract_version != FC_VAE_TORCH_MODULE_STUB_CONTRACT_VERSION:
        raise ValueError(f"Invalid contract_version: {metadata.contract_version}")

    validate_non_empty_str(metadata.stub_kind, "stub_kind")
    if metadata.stub_kind != FC_VAE_TORCH_MODULE_STUB_KIND:
        raise ValueError(f"Invalid stub_kind: {metadata.stub_kind}")

    validate_non_empty_str(metadata.module_name, "module_name")
    if metadata.module_name != FC_VAE_TORCH_MODULE_STUB_MODULE_NAME:
        raise ValueError(f"Invalid module_name: {metadata.module_name}")

    validate_non_empty_str(metadata.class_name, "class_name")
    if metadata.class_name != FC_VAE_TORCH_MODULE_STUB_CLASS_NAME:
        raise ValueError(f"Invalid class_name: {metadata.class_name}")

    validate_bool(metadata.torch_available, "torch_available")
    validate_bool(metadata.module_created, "module_created")
    validate_bool(metadata.is_torch_nn_module, "is_torch_nn_module")
    validate_bool(metadata.defines_forward, "defines_forward")
    if metadata.defines_forward:
        raise ValueError("defines_forward must be False in P29")

    validate_bool(metadata.defines_layers, "defines_layers")
    if metadata.defines_layers:
        raise ValueError("defines_layers must be False in P29")

    validate_non_negative_int(metadata.parameter_count, "parameter_count")
    if metadata.parameter_count != 0:
        raise ValueError("parameter_count must be 0 in P29")

    validate_non_negative_int(metadata.buffer_count, "buffer_count")
    if metadata.buffer_count != 0:
        raise ValueError("buffer_count must be 0 in P29")

    validate_non_empty_str(metadata.reason, "reason")
    assert_no_local_path_leakage(metadata.reason)
    assert_no_forbidden_claims(metadata.reason)


def validate_stub_result(result: FCVAETorchModuleStubResult) -> None:
    if type(result) is not FCVAETorchModuleStubResult:
        raise TypeError("result must be exact FCVAETorchModuleStubResult instance")

    validate_non_empty_str(result.contract_version, "contract_version")
    if result.contract_version != FC_VAE_TORCH_MODULE_STUB_CONTRACT_VERSION:
        raise ValueError(f"Invalid contract_version: {result.contract_version}")

    validate_stub_request(result.request)
    validate_non_empty_str(result.shell_status, "shell_status")
    validate_bool(result.torch_available, "torch_available")
    validate_stub_status(result.status)
    validate_stub_metadata(result.metadata)

    validate_bool(result.module_object_returned, "module_object_returned")
    if result.module_object_returned:
        raise ValueError("module_object_returned must be False in P29")

    validate_bool(result.torch_required_for_p29, "torch_required_for_p29")
    if result.torch_required_for_p29:
        raise ValueError("torch_required_for_p29 must be False")

    validate_bool(result.torch_required_for_future_execution, "torch_required_for_future_execution")
    if not result.torch_required_for_future_execution:
        raise ValueError("torch_required_for_future_execution must be True")

    validate_bool(result.implementation_available_in_p29, "implementation_available_in_p29")
    if result.implementation_available_in_p29:
        raise ValueError("implementation_available_in_p29 must be False")

    validate_bool(result.forward_available_in_p29, "forward_available_in_p29")
    if result.forward_available_in_p29:
        raise ValueError("forward_available_in_p29 must be False")

    validate_bool(result.layers_available_in_p29, "layers_available_in_p29")
    if result.layers_available_in_p29:
        raise ValueError("layers_available_in_p29 must be False")

    validate_bool(result.training_available_in_p29, "training_available_in_p29")
    if result.training_available_in_p29:
        raise ValueError("training_available_in_p29 must be False")

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

    # Consistent check
    if not result.torch_available:
        if result.status != FC_VAE_TORCH_MODULE_STUB_STATUS_TORCH_UNAVAILABLE:
            raise ValueError(f"Inconsistent state: status must be '{FC_VAE_TORCH_MODULE_STUB_STATUS_TORCH_UNAVAILABLE}' when torch is unavailable")
        if result.metadata.module_created:
            raise ValueError("Inconsistent state: module_created must be False when torch is unavailable")
    else:
        if result.status == FC_VAE_TORCH_MODULE_STUB_STATUS_TORCH_UNAVAILABLE:
            raise ValueError(f"Inconsistent state: status cannot be '{FC_VAE_TORCH_MODULE_STUB_STATUS_TORCH_UNAVAILABLE}' when torch is available")

    validate_non_empty_str(result.reason, "reason")
    assert_no_local_path_leakage(result.reason)
    assert_no_forbidden_claims(result.reason)


# Public Builders / Gating Materialization
def build_stub_request(
    require_torch_available: bool = False,
    allow_stub_creation_if_torch_available: bool = True
) -> FCVAETorchModuleStubRequest:
    req = FCVAETorchModuleStubRequest(
        contract_version=FC_VAE_TORCH_MODULE_STUB_CONTRACT_VERSION,
        stub_kind=FC_VAE_TORCH_MODULE_STUB_KIND,
        architecture_id="FC-VAE",
        shell_contract_version=FC_VAE_TORCH_SHELL_CONTRACT_VERSION,
        require_torch_available=require_torch_available,
        allow_stub_creation_if_torch_available=allow_stub_creation_if_torch_available,
        allow_forward_in_p29=False,
        allow_layers_in_p29=False,
        reason="p29_torch_module_stub_request_instantiation",
    )
    validate_stub_request(req)
    return req


def build_blocked_stub_metadata(torch_available: bool) -> FCVAETorchModuleStubMetadata:
    meta = FCVAETorchModuleStubMetadata(
        contract_version=FC_VAE_TORCH_MODULE_STUB_CONTRACT_VERSION,
        stub_kind=FC_VAE_TORCH_MODULE_STUB_KIND,
        module_name=FC_VAE_TORCH_MODULE_STUB_MODULE_NAME,
        class_name=FC_VAE_TORCH_MODULE_STUB_CLASS_NAME,
        torch_available=torch_available,
        module_created=False,
        is_torch_nn_module=False,
        defines_forward=False,
        defines_layers=False,
        parameter_count=0,
        buffer_count=0,
        reason="stub_blocked_instantiation",
    )
    validate_stub_metadata(meta)
    return meta


def materialize_local_torch_stub_metadata() -> FCVAETorchModuleStubMetadata:
    torch_status = build_torch_dependency_status()
    if not torch_status.available:
        return build_blocked_stub_metadata(torch_available=False)

    # Local import of torch inside this function only
    import torch

    class P29LocalFCVAEModuleStub(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()

    stub = P29LocalFCVAEModuleStub()
    if not isinstance(stub, torch.nn.Module):
        raise TypeError("Stub object is not a subclass of torch.nn.Module")

    param_count = sum(1 for _ in stub.parameters())
    buffer_count = sum(1 for _ in stub.buffers())

    meta = FCVAETorchModuleStubMetadata(
        contract_version=FC_VAE_TORCH_MODULE_STUB_CONTRACT_VERSION,
        stub_kind=FC_VAE_TORCH_MODULE_STUB_KIND,
        module_name=FC_VAE_TORCH_MODULE_STUB_MODULE_NAME,
        class_name=FC_VAE_TORCH_MODULE_STUB_CLASS_NAME,
        torch_available=True,
        module_created=True,
        is_torch_nn_module=True,
        defines_forward=False,
        defines_layers=False,
        parameter_count=param_count,
        buffer_count=buffer_count,
        reason="stub_successfully_materialized",
    )
    validate_stub_metadata(meta)
    return meta


def build_stub_result(request: FCVAETorchModuleStubRequest) -> FCVAETorchModuleStubResult:
    validate_stub_request(request)

    shell_result = run_fc_vae_torch_shell_probe()
    torch_status = build_torch_dependency_status()
    torch_available = torch_status.available

    # Materialize stub metadata depending on request allowance
    if request.allow_stub_creation_if_torch_available:
        metadata = materialize_local_torch_stub_metadata()
    else:
        metadata = build_blocked_stub_metadata(torch_available=torch_available)

    # Determine status
    if not torch_available:
        status = FC_VAE_TORCH_MODULE_STUB_STATUS_TORCH_UNAVAILABLE
    elif metadata.module_created:
        status = FC_VAE_TORCH_MODULE_STUB_STATUS_STUB_CREATED
    else:
        status = FC_VAE_TORCH_MODULE_STUB_STATUS_IMPLEMENTATION_DEFERRED

    result = FCVAETorchModuleStubResult(
        contract_version=FC_VAE_TORCH_MODULE_STUB_CONTRACT_VERSION,
        request=request,
        shell_status=shell_result.shell_status,
        torch_available=torch_available,
        status=status,
        metadata=metadata,
        module_object_returned=False,
        torch_required_for_p29=False,
        torch_required_for_future_execution=True,
        implementation_available_in_p29=False,
        forward_available_in_p29=False,
        layers_available_in_p29=False,
        training_available_in_p29=False,
        no_training_loop=True,
        no_optimizer=True,
        no_checkpointing=True,
        no_artifact_generation=True,
        reason="p29_torch_module_stub_result_generation",
    )
    validate_stub_result(result)
    return result


def run_fc_vae_torch_module_stub_probe() -> FCVAETorchModuleStubResult:
    req = build_stub_request(require_torch_available=False, allow_stub_creation_if_torch_available=True)
    res = build_stub_result(req)
    validate_stub_result(res)
    return res


# Public JSON Serialization functions
def stub_request_to_json_dict(request: FCVAETorchModuleStubRequest) -> dict:
    validate_stub_request(request)
    d = {
        "contract_version": request.contract_version,
        "stub_kind": request.stub_kind,
        "architecture_id": request.architecture_id,
        "shell_contract_version": request.shell_contract_version,
        "require_torch_available": request.require_torch_available,
        "allow_stub_creation_if_torch_available": request.allow_stub_creation_if_torch_available,
        "allow_forward_in_p29": request.allow_forward_in_p29,
        "allow_layers_in_p29": request.allow_layers_in_p29,
        "reason": request.reason,
    }
    assert_no_local_path_leakage(d)
    assert_no_forbidden_claims(d)
    return d


def stub_metadata_to_json_dict(metadata: FCVAETorchModuleStubMetadata) -> dict:
    validate_stub_metadata(metadata)
    d = {
        "contract_version": metadata.contract_version,
        "stub_kind": metadata.stub_kind,
        "module_name": metadata.module_name,
        "class_name": metadata.class_name,
        "torch_available": metadata.torch_available,
        "module_created": metadata.module_created,
        "is_torch_nn_module": metadata.is_torch_nn_module,
        "defines_forward": metadata.defines_forward,
        "defines_layers": metadata.defines_layers,
        "parameter_count": metadata.parameter_count,
        "buffer_count": metadata.buffer_count,
        "reason": metadata.reason,
    }
    assert_no_local_path_leakage(d)
    assert_no_forbidden_claims(d)
    return d


def stub_result_to_json_dict(result: FCVAETorchModuleStubResult) -> dict:
    validate_stub_result(result)
    d = {
        "contract_version": result.contract_version,
        "request": stub_request_to_json_dict(result.request),
        "shell_status": result.shell_status,
        "torch_available": result.torch_available,
        "status": result.status,
        "metadata": stub_metadata_to_json_dict(result.metadata),
        "module_object_returned": result.module_object_returned,
        "torch_required_for_p29": result.torch_required_for_p29,
        "torch_required_for_future_execution": result.torch_required_for_future_execution,
        "implementation_available_in_p29": result.implementation_available_in_p29,
        "forward_available_in_p29": result.forward_available_in_p29,
        "layers_available_in_p29": result.layers_available_in_p29,
        "training_available_in_p29": result.training_available_in_p29,
        "no_training_loop": result.no_training_loop,
        "no_optimizer": result.no_optimizer,
        "no_checkpointing": result.no_checkpointing,
        "no_artifact_generation": result.no_artifact_generation,
        "reason": result.reason,
    }
    assert_no_local_path_leakage(d)
    assert_no_forbidden_claims(d)
    return d


def compact_stub_json(result: FCVAETorchModuleStubResult) -> str:
    d = stub_result_to_json_dict(result)
    return json.dumps(d, sort_keys=True, separators=(",", ":"))
