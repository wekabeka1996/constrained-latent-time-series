# src/phase2/fc_vae_torch_shell.py

import dataclasses
import json
from typing import Any, Dict, Tuple, Union

from src.phase2.torch_boundary import (
    TORCH_BOUNDARY_CONTRACT_VERSION,
    build_torch_dependency_status,
)
from src.phase2.fc_vae_model import (
    FC_VAE_MODEL_SKELETON_CONTRACT_VERSION,
    build_fc_vae_skeleton_status,
)


# Public constants
FC_VAE_TORCH_SHELL_CONTRACT_VERSION = "phase2_p28_fc_vae_torch_shell_contract_v1"

FC_VAE_TORCH_SHELL_KIND = "optional_torch_shell_handle"

FC_VAE_TORCH_SHELL_MODULE_NAME = "src.phase2.fc_vae_torch_shell"

FC_VAE_FUTURE_IMPLEMENTATION_MODULE = "src.phase2.fc_vae_model"

FC_VAE_TORCH_SHELL_STATUS_TORCH_UNAVAILABLE = "blocked_torch_unavailable"

FC_VAE_TORCH_SHELL_STATUS_IMPLEMENTATION_DEFERRED = "blocked_implementation_deferred"

FC_VAE_TORCH_SHELL_STATUS_READY_FOR_FUTURE_IMPLEMENTATION = "ready_for_future_implementation"

SUPPORTED_FC_VAE_TORCH_SHELL_STATUSES = (
    "blocked_torch_unavailable",
    "blocked_implementation_deferred",
    "ready_for_future_implementation",
)

FC_VAE_TORCH_SHELL_NO_MODULE_SENTINEL = "no_torch_module_created_in_p28"


# Public Dataclasses
@dataclasses.dataclass(frozen=True)
class FCVAETorchShellRequest:
    contract_version: str
    shell_kind: str
    architecture_id: str
    requested_module_name: str
    require_torch_available: bool
    allow_implementation_in_p28: bool
    reason: str


@dataclasses.dataclass(frozen=True)
class FCVAETorchShellHandle:
    contract_version: str
    shell_kind: str
    architecture_id: str
    shell_module_name: str
    future_implementation_module: str
    skeleton_contract_version: str
    torch_boundary_contract_version: str
    module_created: bool
    module_sentinel: str
    reason: str


@dataclasses.dataclass(frozen=True)
class FCVAETorchShellResult:
    contract_version: str
    request: FCVAETorchShellRequest
    handle: FCVAETorchShellHandle
    torch_available: bool
    skeleton_status: str
    shell_status: str
    torch_required_for_p28: bool
    torch_required_for_future_execution: bool
    implementation_available_in_p28: bool
    module_created: bool
    no_model_implementation: bool
    no_training_loop: bool
    no_optimizer: bool
    no_checkpointing: bool
    no_artifact_generation: bool
    reason: str


# Public validation functions
def validate_non_empty_str(value: Any, name: str) -> None:
    if type(value) is not str:
        raise TypeError(f"{name} must be exact str, got {type(value).__name__}")
    if not value:
        raise ValueError(f"{name} cannot be empty")
    if value != value.strip():
        raise ValueError(f"{name} cannot have leading or trailing whitespace")


def validate_bool(value: Any, name: str) -> None:
    if type(value) is not bool:
        raise TypeError(f"{name} must be exact bool, got {type(value).__name__}")


def validate_shell_status(value: Any) -> None:
    if type(value) is not str:
        raise TypeError(f"shell_status must be exact str, got {type(value).__name__}")
    if value not in SUPPORTED_FC_VAE_TORCH_SHELL_STATUSES:
        raise ValueError(f"Unsupported shell status: {value}")


# Safety helpers
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
        # Clean allowed negative claims to avoid false positives
        cleaned = normalized.replace("no_final_comparison", "").replace("no_scientific_conclusion", "")
        forbidden = ("model works", "scientific success", "solved", "best", "winner")
        for word in forbidden:
            if word in cleaned:
                raise ValueError(f"Forbidden claim word '{word}' found in: '{data}'")
    elif isinstance(data, dict):
        for k, v in data.items():
            assert_no_forbidden_claims(k)
            assert_no_forbidden_claims(v)
    elif isinstance(data, (list, tuple)):
        for item in data:
            assert_no_forbidden_claims(item)


def validate_torch_shell_request(request: FCVAETorchShellRequest) -> None:
    if type(request) is not FCVAETorchShellRequest:
        raise TypeError("request must be exact FCVAETorchShellRequest instance")

    validate_non_empty_str(request.contract_version, "contract_version")
    if request.contract_version != FC_VAE_TORCH_SHELL_CONTRACT_VERSION:
        raise ValueError(f"Invalid contract_version: {request.contract_version}")

    validate_non_empty_str(request.shell_kind, "shell_kind")
    if request.shell_kind != FC_VAE_TORCH_SHELL_KIND:
        raise ValueError(f"Invalid shell_kind: {request.shell_kind}")

    validate_non_empty_str(request.architecture_id, "architecture_id")
    if request.architecture_id != "FC-VAE":
        raise ValueError(f"Invalid architecture_id: {request.architecture_id}")

    validate_non_empty_str(request.requested_module_name, "requested_module_name")
    if request.requested_module_name != FC_VAE_FUTURE_IMPLEMENTATION_MODULE:
        raise ValueError(f"Invalid requested_module_name: {request.requested_module_name}")

    validate_bool(request.require_torch_available, "require_torch_available")
    validate_bool(request.allow_implementation_in_p28, "allow_implementation_in_p28")
    if request.allow_implementation_in_p28:
        raise ValueError("allow_implementation_in_p28 must be False in P28")

    validate_non_empty_str(request.reason, "reason")

    assert_no_local_path_leakage(request.reason)
    assert_no_forbidden_claims(request.reason)


def validate_torch_shell_handle(handle: FCVAETorchShellHandle) -> None:
    if type(handle) is not FCVAETorchShellHandle:
        raise TypeError("handle must be exact FCVAETorchShellHandle instance")

    validate_non_empty_str(handle.contract_version, "contract_version")
    if handle.contract_version != FC_VAE_TORCH_SHELL_CONTRACT_VERSION:
        raise ValueError(f"Invalid contract_version: {handle.contract_version}")

    validate_non_empty_str(handle.shell_kind, "shell_kind")
    if handle.shell_kind != FC_VAE_TORCH_SHELL_KIND:
        raise ValueError(f"Invalid shell_kind: {handle.shell_kind}")

    validate_non_empty_str(handle.architecture_id, "architecture_id")
    if handle.architecture_id != "FC-VAE":
        raise ValueError(f"Invalid architecture_id: {handle.architecture_id}")

    validate_non_empty_str(handle.shell_module_name, "shell_module_name")
    if handle.shell_module_name != FC_VAE_TORCH_SHELL_MODULE_NAME:
        raise ValueError(f"Invalid shell_module_name: {handle.shell_module_name}")

    validate_non_empty_str(handle.future_implementation_module, "future_implementation_module")
    if handle.future_implementation_module != FC_VAE_FUTURE_IMPLEMENTATION_MODULE:
        raise ValueError(f"Invalid future_implementation_module: {handle.future_implementation_module}")

    validate_non_empty_str(handle.skeleton_contract_version, "skeleton_contract_version")
    if handle.skeleton_contract_version != FC_VAE_MODEL_SKELETON_CONTRACT_VERSION:
        raise ValueError(f"Invalid skeleton_contract_version: {handle.skeleton_contract_version}")

    validate_non_empty_str(handle.torch_boundary_contract_version, "torch_boundary_contract_version")
    if handle.torch_boundary_contract_version != TORCH_BOUNDARY_CONTRACT_VERSION:
        raise ValueError(f"Invalid torch_boundary_contract_version: {handle.torch_boundary_contract_version}")

    validate_bool(handle.module_created, "module_created")
    if handle.module_created:
        raise ValueError("module_created must be False in P28")

    validate_non_empty_str(handle.module_sentinel, "module_sentinel")
    if handle.module_sentinel != FC_VAE_TORCH_SHELL_NO_MODULE_SENTINEL:
        raise ValueError(f"Invalid module_sentinel: {handle.module_sentinel}")

    validate_non_empty_str(handle.reason, "reason")

    assert_no_local_path_leakage(handle.reason)
    assert_no_forbidden_claims(handle.reason)


def validate_torch_shell_result(result: FCVAETorchShellResult) -> None:
    if type(result) is not FCVAETorchShellResult:
        raise TypeError("result must be exact FCVAETorchShellResult instance")

    validate_non_empty_str(result.contract_version, "contract_version")
    if result.contract_version != FC_VAE_TORCH_SHELL_CONTRACT_VERSION:
        raise ValueError(f"Invalid contract_version: {result.contract_version}")

    validate_torch_shell_request(result.request)
    validate_torch_shell_handle(result.handle)

    validate_bool(result.torch_available, "torch_available")
    validate_non_empty_str(result.skeleton_status, "skeleton_status")
    validate_shell_status(result.shell_status)

    validate_bool(result.torch_required_for_p28, "torch_required_for_p28")
    if result.torch_required_for_p28:
        raise ValueError("torch_required_for_p28 must be False")

    validate_bool(result.torch_required_for_future_execution, "torch_required_for_future_execution")
    if not result.torch_required_for_future_execution:
        raise ValueError("torch_required_for_future_execution must be True")

    validate_bool(result.implementation_available_in_p28, "implementation_available_in_p28")
    if result.implementation_available_in_p28:
        raise ValueError("implementation_available_in_p28 must be False")

    validate_bool(result.module_created, "module_created")
    if result.module_created:
        raise ValueError("module_created must be False")

    # all no_* flags must be True
    validate_bool(result.no_model_implementation, "no_model_implementation")
    if not result.no_model_implementation:
        raise ValueError("no_model_implementation must be True")

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

    # Check status-torch consistency
    if not result.torch_available:
        if result.shell_status != FC_VAE_TORCH_SHELL_STATUS_TORCH_UNAVAILABLE:
            raise ValueError(
                f"Inconsistent state: shell_status must be '{FC_VAE_TORCH_SHELL_STATUS_TORCH_UNAVAILABLE}' "
                f"when torch is unavailable"
            )
    else:
        if result.shell_status not in (
            FC_VAE_TORCH_SHELL_STATUS_IMPLEMENTATION_DEFERRED,
            FC_VAE_TORCH_SHELL_STATUS_READY_FOR_FUTURE_IMPLEMENTATION,
        ):
            raise ValueError(
                f"Inconsistent state: shell_status must be deferred or ready "
                f"when torch is available, got: {result.shell_status}"
            )

    validate_non_empty_str(result.reason, "reason")

    assert_no_local_path_leakage(result.reason)
    assert_no_forbidden_claims(result.reason)


# Public Builder / Gating Functions
def build_torch_shell_request(require_torch_available: bool = False) -> FCVAETorchShellRequest:
    request = FCVAETorchShellRequest(
        contract_version=FC_VAE_TORCH_SHELL_CONTRACT_VERSION,
        shell_kind=FC_VAE_TORCH_SHELL_KIND,
        architecture_id="FC-VAE",
        requested_module_name=FC_VAE_FUTURE_IMPLEMENTATION_MODULE,
        require_torch_available=require_torch_available,
        allow_implementation_in_p28=False,
        reason="p28_torch_shell_request_instantiation",
    )
    validate_torch_shell_request(request)
    return request


def build_torch_shell_handle() -> FCVAETorchShellHandle:
    handle = FCVAETorchShellHandle(
        contract_version=FC_VAE_TORCH_SHELL_CONTRACT_VERSION,
        shell_kind=FC_VAE_TORCH_SHELL_KIND,
        architecture_id="FC-VAE",
        shell_module_name=FC_VAE_TORCH_SHELL_MODULE_NAME,
        future_implementation_module=FC_VAE_FUTURE_IMPLEMENTATION_MODULE,
        skeleton_contract_version=FC_VAE_MODEL_SKELETON_CONTRACT_VERSION,
        torch_boundary_contract_version=TORCH_BOUNDARY_CONTRACT_VERSION,
        module_created=False,
        module_sentinel=FC_VAE_TORCH_SHELL_NO_MODULE_SENTINEL,
        reason="p28_torch_shell_handle_instantiation",
    )
    validate_torch_shell_handle(handle)
    return handle


def build_torch_shell_result(request: FCVAETorchShellRequest) -> FCVAETorchShellResult:
    validate_torch_shell_request(request)

    skeleton_stat = build_fc_vae_skeleton_status()
    torch_stat = build_torch_dependency_status()

    # Apply gating rules
    torch_available = torch_stat.available
    if not torch_available:
        shell_status = FC_VAE_TORCH_SHELL_STATUS_TORCH_UNAVAILABLE
    else:
        # Prefer deferred in P28
        shell_status = FC_VAE_TORCH_SHELL_STATUS_IMPLEMENTATION_DEFERRED

    result = FCVAETorchShellResult(
        contract_version=FC_VAE_TORCH_SHELL_CONTRACT_VERSION,
        request=request,
        handle=build_torch_shell_handle(),
        torch_available=torch_available,
        skeleton_status=skeleton_stat.status,
        shell_status=shell_status,
        torch_required_for_p28=False,
        torch_required_for_future_execution=True,
        implementation_available_in_p28=False,
        module_created=False,
        no_model_implementation=True,
        no_training_loop=True,
        no_optimizer=True,
        no_checkpointing=True,
        no_artifact_generation=True,
        reason="p28_torch_shell_result_generation",
    )
    validate_torch_shell_result(result)
    return result


def require_torch_shell_materialized(result: FCVAETorchShellResult) -> None:
    validate_torch_shell_result(result)
    raise NotImplementedError(
        "FC-VAE torch shell is not materialized in P28; this phase defines only an optional shell handle"
    )


def run_fc_vae_torch_shell_probe() -> FCVAETorchShellResult:
    request = build_torch_shell_request(require_torch_available=False)
    result = build_torch_shell_result(request)
    validate_torch_shell_result(result)
    return result


# JSON Serialization functions
def torch_shell_request_to_json_dict(request: FCVAETorchShellRequest) -> dict:
    validate_torch_shell_request(request)
    d = {
        "contract_version": request.contract_version,
        "shell_kind": request.shell_kind,
        "architecture_id": request.architecture_id,
        "requested_module_name": request.requested_module_name,
        "require_torch_available": request.require_torch_available,
        "allow_implementation_in_p28": request.allow_implementation_in_p28,
        "reason": request.reason,
    }
    assert_no_local_path_leakage(d)
    assert_no_forbidden_claims(d)
    return d


def torch_shell_handle_to_json_dict(handle: FCVAETorchShellHandle) -> dict:
    validate_torch_shell_handle(handle)
    d = {
        "contract_version": handle.contract_version,
        "shell_kind": handle.shell_kind,
        "architecture_id": handle.architecture_id,
        "shell_module_name": handle.shell_module_name,
        "future_implementation_module": handle.future_implementation_module,
        "skeleton_contract_version": handle.skeleton_contract_version,
        "torch_boundary_contract_version": handle.torch_boundary_contract_version,
        "module_created": handle.module_created,
        "module_sentinel": handle.module_sentinel,
        "reason": handle.reason,
    }
    assert_no_local_path_leakage(d)
    assert_no_forbidden_claims(d)
    return d


def torch_shell_result_to_json_dict(result: FCVAETorchShellResult) -> dict:
    validate_torch_shell_result(result)
    d = {
        "contract_version": result.contract_version,
        "request": torch_shell_request_to_json_dict(result.request),
        "handle": torch_shell_handle_to_json_dict(result.handle),
        "torch_available": result.torch_available,
        "skeleton_status": result.skeleton_status,
        "shell_status": result.shell_status,
        "torch_required_for_p28": result.torch_required_for_p28,
        "torch_required_for_future_execution": result.torch_required_for_future_execution,
        "implementation_available_in_p28": result.implementation_available_in_p28,
        "module_created": result.module_created,
        "no_model_implementation": result.no_model_implementation,
        "no_training_loop": result.no_training_loop,
        "no_optimizer": result.no_optimizer,
        "no_checkpointing": result.no_checkpointing,
        "no_artifact_generation": result.no_artifact_generation,
        "reason": result.reason,
    }
    assert_no_local_path_leakage(d)
    assert_no_forbidden_claims(d)
    return d


def compact_torch_shell_json(result: FCVAETorchShellResult) -> str:
    d = torch_shell_result_to_json_dict(result)
    return json.dumps(d, sort_keys=True, separators=(",", ":"))
