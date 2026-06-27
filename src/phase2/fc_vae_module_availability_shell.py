# src/phase2/fc_vae_module_availability_shell.py

import dataclasses
import json
from typing import Any, Tuple

from src.phase2.torch_boundary import (
    TORCH_BOUNDARY_CONTRACT_VERSION,
    TORCH_POLICY_OPTIONAL,
    build_torch_dependency_status,
)
from src.phase2.fc_vae_model import (
    FC_VAE_MODEL_SKELETON_CONTRACT_VERSION,
    FC_VAE_ARCHITECTURE_ID,
    FC_VAE_REQUIRED_LATENT_NAMES,
    build_fc_vae_skeleton_status,
    build_smoke_forward_contract,
)
from src.phase2.fc_vae_torch_module_stub import (
    FC_VAE_TORCH_MODULE_STUB_CONTRACT_VERSION,
)
from src.phase2.fc_vae_forward_eligibility_gate import (
    FC_VAE_FORWARD_ELIGIBILITY_GATE_CONTRACT_VERSION,
    run_forward_eligibility_gate_probe,
)

# Public constants
FC_VAE_MODULE_AVAILABILITY_SHELL_CONTRACT_VERSION = "phase2_p41_fc_vae_module_availability_shell_contract_v1"

FC_VAE_MODULE_AVAILABILITY_SHELL_KIND = "fc_vae_torch_module_availability_shell_no_forward_no_output_no_training"

FC_VAE_MODULE_AVAILABILITY_SHELL_MODULE_NAME = "src.phase2.fc_vae_module_availability_shell"

FC_VAE_MODULE_AVAILABILITY_SHELL_CLASS_NAME = "P41FCVAEModuleAvailabilityShell"

FC_VAE_MODULE_AVAILABILITY_STATUS_TORCH_UNAVAILABLE = "blocked_torch_unavailable"

FC_VAE_MODULE_AVAILABILITY_STATUS_SHELL_CREATED = "module_shell_created_no_forward_no_output_no_training"

FC_VAE_MODULE_AVAILABILITY_STATUS_CONTRACT_MISMATCH = "blocked_by_contract_mismatch"

SUPPORTED_FC_VAE_MODULE_AVAILABILITY_STATUSES = (
    "blocked_torch_unavailable",
    "module_shell_created_no_forward_no_output_no_training",
    "blocked_by_contract_mismatch",
)


# Public dataclasses
@dataclasses.dataclass(frozen=True)
class FCVAEModuleAvailabilityShellRequest:
    contract_version: str
    shell_kind: str
    architecture_id: str
    source_torch_boundary_contract_version: str
    source_model_skeleton_contract_version: str
    source_torch_module_stub_contract_version: str
    source_forward_eligibility_gate_contract_version: str
    expected_input_flat_dim: int
    expected_latent_total_dim: int
    expected_latent_names: Tuple[str, ...]
    target_device_type: str
    allow_module_shell_creation_in_p41: bool
    allow_forward_execution_in_p41: bool
    allow_output_generation_in_p41: bool
    allow_training_in_p41: bool
    reason: str


@dataclasses.dataclass(frozen=True)
class FCVAEModuleAvailabilityShellMetadata:
    contract_version: str
    shell_kind: str
    module_name: str
    class_name: str
    torch_available: bool
    torch_import_safe: bool
    top_level_torch_import_required: bool
    module_shell_created: bool
    module_object_returned: bool
    is_torch_nn_module: bool
    defines_forward: bool
    has_own_forward: bool
    uses_inherited_unimplemented_forward_only: bool
    defines_layers: bool
    parameter_count: int
    buffer_count: int
    training_mode_after_creation: bool
    module_device_type: str
    architecture_id: str
    input_flat_dim: int
    latent_total_dim: int
    latent_names: Tuple[str, ...]
    forward_execution_attempted: bool
    output_generation_attempted: bool
    training_attempted: bool
    reason: str


@dataclasses.dataclass(frozen=True)
class FCVAEModuleAvailabilityShellResult:
    contract_version: str
    request: FCVAEModuleAvailabilityShellRequest
    metadata: FCVAEModuleAvailabilityShellMetadata
    status: str
    implementation_available_in_p41: bool
    module_shell_available_in_p41: bool
    forward_available_in_p41: bool
    forward_execution_available_in_p41: bool
    forward_executed_in_p41: bool
    output_generation_available_in_p41: bool
    output_generated_in_p41: bool
    training_available_in_p41: bool
    training_executed_in_p41: bool
    no_forward_execution: bool
    no_output_generation: bool
    no_training_loop: bool
    no_optimizer: bool
    no_checkpointing: bool
    no_artifact_generation: bool
    no_final_comparison: bool
    no_scientific_conclusion: bool
    reason: str


# Public validation functions
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
    if type(value) is bool:
        raise TypeError(f"{name} must be exact int, got bool")
    if type(value) is not int:
        raise TypeError(f"{name} must be exact int, got {type(value).__name__}")
    if value <= 0:
        raise ValueError(f"{name} must be > 0, got {value}")


def validate_non_negative_int(value: Any, name: str) -> None:
    if type(value) is bool:
        raise TypeError(f"{name} must be exact int, got bool")
    if type(value) is not int:
        raise TypeError(f"{name} must be exact int, got {type(value).__name__}")
    if value < 0:
        raise ValueError(f"{name} must be >= 0, got {value}")


def validate_exact_tuple_str(value: Any, expected: Tuple[str, ...], name: str) -> None:
    if type(value) is not tuple:
        raise TypeError(f"{name} must be exact tuple, got {type(value).__name__}")
    for i, v in enumerate(value):
        if type(v) is not str:
            raise TypeError(f"{name}[{i}] must be exact str, got {type(v).__name__}")
    if value != expected:
        raise ValueError(f"{name} must equal {expected}, got {value}")


def validate_module_availability_status(value: Any) -> None:
    if type(value) is not str:
        raise TypeError(f"status must be exact str, got {type(value).__name__}")
    if value not in SUPPORTED_FC_VAE_MODULE_AVAILABILITY_STATUSES:
        raise ValueError(f"Unsupported module availability status: {value}")


def assert_no_local_path_leakage(value: str, name: str) -> None:
    forbidden = ["file:///", "C:/", "C:\\", "/home/", "/Users/"]
    for pattern in forbidden:
        if pattern in value:
            raise ValueError(f"Local path pattern '{pattern}' detected in {name}")


def assert_no_forbidden_claims(value: str, name: str) -> None:
    forbidden = [
        "model " + "works",
        "scientific " + "success",
        "sol" + "ved",
        "be" + "st",
        "win" + "ner",
        "production " + "ready",
        "state of " + "the art",
    ]
    value_lower = value.lower()
    for pattern in forbidden:
        if pattern in value_lower:
            raise ValueError(f"Forbidden claim '{pattern}' detected in {name}")


def validate_module_availability_request(request: FCVAEModuleAvailabilityShellRequest) -> None:
    if type(request) is not FCVAEModuleAvailabilityShellRequest:
        raise TypeError("request must be exact FCVAEModuleAvailabilityShellRequest instance")
    validate_non_empty_str(request.contract_version, "contract_version")
    if request.contract_version != FC_VAE_MODULE_AVAILABILITY_SHELL_CONTRACT_VERSION:
        raise ValueError(f"Wrong contract version: expected {FC_VAE_MODULE_AVAILABILITY_SHELL_CONTRACT_VERSION}")
    validate_non_empty_str(request.shell_kind, "shell_kind")
    if request.shell_kind != FC_VAE_MODULE_AVAILABILITY_SHELL_KIND:
        raise ValueError(f"Wrong shell kind: expected {FC_VAE_MODULE_AVAILABILITY_SHELL_KIND}")
    validate_non_empty_str(request.architecture_id, "architecture_id")
    if request.architecture_id != FC_VAE_ARCHITECTURE_ID:
        raise ValueError(f"Wrong architecture_id: expected {FC_VAE_ARCHITECTURE_ID}")
    validate_non_empty_str(request.source_torch_boundary_contract_version, "source_torch_boundary_contract_version")
    if request.source_torch_boundary_contract_version != TORCH_BOUNDARY_CONTRACT_VERSION:
        raise ValueError(f"Wrong source torch boundary contract: expected {TORCH_BOUNDARY_CONTRACT_VERSION}")
    validate_non_empty_str(request.source_model_skeleton_contract_version, "source_model_skeleton_contract_version")
    if request.source_model_skeleton_contract_version != FC_VAE_MODEL_SKELETON_CONTRACT_VERSION:
        raise ValueError(f"Wrong source skeleton contract: expected {FC_VAE_MODEL_SKELETON_CONTRACT_VERSION}")
    validate_non_empty_str(request.source_torch_module_stub_contract_version, "source_torch_module_stub_contract_version")
    if request.source_torch_module_stub_contract_version != FC_VAE_TORCH_MODULE_STUB_CONTRACT_VERSION:
        raise ValueError(f"Wrong source stub contract: expected {FC_VAE_TORCH_MODULE_STUB_CONTRACT_VERSION}")
    validate_non_empty_str(request.source_forward_eligibility_gate_contract_version, "source_forward_eligibility_gate_contract_version")
    if request.source_forward_eligibility_gate_contract_version != FC_VAE_FORWARD_ELIGIBILITY_GATE_CONTRACT_VERSION:
        raise ValueError(f"Wrong source gate contract: expected {FC_VAE_FORWARD_ELIGIBILITY_GATE_CONTRACT_VERSION}")
    validate_positive_int(request.expected_input_flat_dim, "expected_input_flat_dim")
    if request.expected_input_flat_dim != 32:
        raise ValueError(f"expected_input_flat_dim must be 32, got {request.expected_input_flat_dim}")
    validate_positive_int(request.expected_latent_total_dim, "expected_latent_total_dim")
    if request.expected_latent_total_dim != 20:
        raise ValueError(f"expected_latent_total_dim must be 20, got {request.expected_latent_total_dim}")
    validate_exact_tuple_str(request.expected_latent_names, FC_VAE_REQUIRED_LATENT_NAMES, "expected_latent_names")
    validate_non_empty_str(request.target_device_type, "target_device_type")
    if request.target_device_type != "cpu":
        raise ValueError(f"target_device_type must be cpu, got {request.target_device_type}")
    validate_bool(request.allow_module_shell_creation_in_p41, "allow_module_shell_creation_in_p41")
    if not request.allow_module_shell_creation_in_p41:
        raise ValueError("allow_module_shell_creation_in_p41 must be True")
    validate_bool(request.allow_forward_execution_in_p41, "allow_forward_execution_in_p41")
    if request.allow_forward_execution_in_p41:
        raise ValueError("allow_forward_execution_in_p41 must be False")
    validate_bool(request.allow_output_generation_in_p41, "allow_output_generation_in_p41")
    if request.allow_output_generation_in_p41:
        raise ValueError("allow_output_generation_in_p41 must be False")
    validate_bool(request.allow_training_in_p41, "allow_training_in_p41")
    if request.allow_training_in_p41:
        raise ValueError("allow_training_in_p41 must be False")
    validate_non_empty_str(request.reason, "reason")
    assert_no_local_path_leakage(request.reason, "reason")
    assert_no_forbidden_claims(request.reason, "reason")


def validate_module_availability_metadata(metadata: FCVAEModuleAvailabilityShellMetadata) -> None:
    if type(metadata) is not FCVAEModuleAvailabilityShellMetadata:
        raise TypeError("metadata must be exact FCVAEModuleAvailabilityShellMetadata instance")
    validate_non_empty_str(metadata.contract_version, "contract_version")
    if metadata.contract_version != FC_VAE_MODULE_AVAILABILITY_SHELL_CONTRACT_VERSION:
        raise ValueError(f"Wrong contract version: expected {FC_VAE_MODULE_AVAILABILITY_SHELL_CONTRACT_VERSION}")
    validate_non_empty_str(metadata.shell_kind, "shell_kind")
    if metadata.shell_kind != FC_VAE_MODULE_AVAILABILITY_SHELL_KIND:
        raise ValueError(f"Wrong shell kind: expected {FC_VAE_MODULE_AVAILABILITY_SHELL_KIND}")
    validate_non_empty_str(metadata.module_name, "module_name")
    if metadata.module_name != FC_VAE_MODULE_AVAILABILITY_SHELL_MODULE_NAME:
        raise ValueError(f"Wrong module name: expected {FC_VAE_MODULE_AVAILABILITY_SHELL_MODULE_NAME}")
    validate_non_empty_str(metadata.class_name, "class_name")
    if metadata.class_name != FC_VAE_MODULE_AVAILABILITY_SHELL_CLASS_NAME:
        raise ValueError(f"Wrong class name: expected {FC_VAE_MODULE_AVAILABILITY_SHELL_CLASS_NAME}")
    validate_bool(metadata.torch_available, "torch_available")
    validate_bool(metadata.torch_import_safe, "torch_import_safe")
    validate_bool(metadata.top_level_torch_import_required, "top_level_torch_import_required")
    validate_bool(metadata.module_shell_created, "module_shell_created")
    validate_bool(metadata.module_object_returned, "module_object_returned")
    if metadata.module_object_returned:
        raise ValueError("module_object_returned must be False")
    validate_bool(metadata.is_torch_nn_module, "is_torch_nn_module")
    if metadata.module_shell_created:
        if not metadata.is_torch_nn_module:
            raise ValueError("is_torch_nn_module must be True if shell was created")
    else:
        if metadata.is_torch_nn_module:
            raise ValueError("is_torch_nn_module must be False if shell was not created")
    validate_bool(metadata.defines_forward, "defines_forward")
    if metadata.defines_forward:
        raise ValueError("defines_forward must be False")
    validate_bool(metadata.has_own_forward, "has_own_forward")
    if metadata.has_own_forward:
        raise ValueError("has_own_forward must be False")
    validate_bool(metadata.uses_inherited_unimplemented_forward_only, "uses_inherited_unimplemented_forward_only")
    if metadata.module_shell_created:
        if not metadata.uses_inherited_unimplemented_forward_only:
            raise ValueError("uses_inherited_unimplemented_forward_only must be True if shell created")
    validate_bool(metadata.defines_layers, "defines_layers")
    if metadata.defines_layers:
        raise ValueError("defines_layers must be False")
    validate_non_negative_int(metadata.parameter_count, "parameter_count")
    if metadata.parameter_count != 0:
        raise ValueError("parameter_count must be 0")
    validate_non_negative_int(metadata.buffer_count, "buffer_count")
    if metadata.buffer_count != 0:
        raise ValueError("buffer_count must be 0")
    validate_bool(metadata.training_mode_after_creation, "training_mode_after_creation")
    validate_non_empty_str(metadata.module_device_type, "module_device_type")
    if metadata.module_device_type != "cpu":
        raise ValueError("module_device_type must be cpu")
    validate_non_empty_str(metadata.architecture_id, "architecture_id")
    if metadata.architecture_id != FC_VAE_ARCHITECTURE_ID:
        raise ValueError(f"architecture_id must be {FC_VAE_ARCHITECTURE_ID}")
    validate_positive_int(metadata.input_flat_dim, "input_flat_dim")
    if metadata.input_flat_dim != 32:
        raise ValueError("input_flat_dim must be 32")
    validate_positive_int(metadata.latent_total_dim, "latent_total_dim")
    if metadata.latent_total_dim != 20:
        raise ValueError("latent_total_dim must be 20")
    validate_exact_tuple_str(metadata.latent_names, FC_VAE_REQUIRED_LATENT_NAMES, "latent_names")
    validate_bool(metadata.forward_execution_attempted, "forward_execution_attempted")
    if metadata.forward_execution_attempted:
        raise ValueError("forward_execution_attempted must be False")
    validate_bool(metadata.output_generation_attempted, "output_generation_attempted")
    if metadata.output_generation_attempted:
        raise ValueError("output_generation_attempted must be False")
    validate_bool(metadata.training_attempted, "training_attempted")
    if metadata.training_attempted:
        raise ValueError("training_attempted must be False")
    validate_non_empty_str(metadata.reason, "reason")
    assert_no_local_path_leakage(metadata.reason, "reason")
    assert_no_forbidden_claims(metadata.reason, "reason")


def validate_module_availability_result(result: FCVAEModuleAvailabilityShellResult) -> None:
    if type(result) is not FCVAEModuleAvailabilityShellResult:
        raise TypeError("result must be exact FCVAEModuleAvailabilityShellResult instance")
    validate_non_empty_str(result.contract_version, "contract_version")
    if result.contract_version != FC_VAE_MODULE_AVAILABILITY_SHELL_CONTRACT_VERSION:
        raise ValueError(f"Wrong contract version: expected {FC_VAE_MODULE_AVAILABILITY_SHELL_CONTRACT_VERSION}")
    validate_module_availability_request(result.request)
    validate_module_availability_metadata(result.metadata)
    validate_module_availability_status(result.status)
    validate_bool(result.implementation_available_in_p41, "implementation_available_in_p41")
    if result.status == FC_VAE_MODULE_AVAILABILITY_STATUS_SHELL_CREATED:
        if not result.implementation_available_in_p41:
            raise ValueError("implementation_available_in_p41 must be True if shell created")
    else:
        if result.implementation_available_in_p41:
            raise ValueError("implementation_available_in_p41 must be False if shell blocked")
    validate_bool(result.module_shell_available_in_p41, "module_shell_available_in_p41")
    if result.status == FC_VAE_MODULE_AVAILABILITY_STATUS_SHELL_CREATED:
        if not result.module_shell_available_in_p41:
            raise ValueError("module_shell_available_in_p41 must be True if shell created")
    else:
        if result.module_shell_available_in_p41:
            raise ValueError("module_shell_available_in_p41 must be False if shell blocked")
    validate_bool(result.forward_available_in_p41, "forward_available_in_p41")
    if result.forward_available_in_p41:
        raise ValueError("forward_available_in_p41 must be False")
    validate_bool(result.forward_execution_available_in_p41, "forward_execution_available_in_p41")
    if result.forward_execution_available_in_p41:
        raise ValueError("forward_execution_available_in_p41 must be False")
    validate_bool(result.forward_executed_in_p41, "forward_executed_in_p41")
    if result.forward_executed_in_p41:
        raise ValueError("forward_executed_in_p41 must be False")
    validate_bool(result.output_generation_available_in_p41, "output_generation_available_in_p41")
    if result.output_generation_available_in_p41:
        raise ValueError("output_generation_available_in_p41 must be False")
    validate_bool(result.output_generated_in_p41, "output_generated_in_p41")
    if result.output_generated_in_p41:
        raise ValueError("output_generated_in_p41 must be False")
    validate_bool(result.training_available_in_p41, "training_available_in_p41")
    if result.training_available_in_p41:
        raise ValueError("training_available_in_p41 must be False")
    validate_bool(result.training_executed_in_p41, "training_executed_in_p41")
    if result.training_executed_in_p41:
        raise ValueError("training_executed_in_p41 must be False")
    validate_bool(result.no_forward_execution, "no_forward_execution")
    if not result.no_forward_execution:
        raise ValueError("no_forward_execution must be True")
    validate_bool(result.no_output_generation, "no_output_generation")
    if not result.no_output_generation:
        raise ValueError("no_output_generation must be True")
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
    validate_non_empty_str(result.reason, "reason")
    assert_no_local_path_leakage(result.reason, "reason")
    assert_no_forbidden_claims(result.reason, "reason")


# Guarded loader helper
def load_torch_for_p41_module_shell() -> Any:
    status = build_torch_dependency_status(policy=TORCH_POLICY_OPTIONAL)
    if not status.available:
        raise RuntimeError("PyTorch is not available under the P26 optional policy boundary")
    
    # Gated local import only
    import torch
    return torch


# Cached persistent class definition
_CachedFCVAEModuleClass = None


def get_module_shell_class() -> type:
    global _CachedFCVAEModuleClass
    if _CachedFCVAEModuleClass is not None:
        return _CachedFCVAEModuleClass
    
    torch = load_torch_for_p41_module_shell()
    
    class P41FCVAEModuleAvailabilityShell(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.architecture_id = "FC-VAE"
            self.input_flat_dim = 32
            self.latent_total_dim = 20
            self.latent_names = ("z_mean", "z_volatility", "z_shared")

    # Override class name representation explicitly
    P41FCVAEModuleAvailabilityShell.__name__ = FC_VAE_MODULE_AVAILABILITY_SHELL_CLASS_NAME
    P41FCVAEModuleAvailabilityShell.__qualname__ = FC_VAE_MODULE_AVAILABILITY_SHELL_CLASS_NAME

    _CachedFCVAEModuleClass = P41FCVAEModuleAvailabilityShell
    return _CachedFCVAEModuleClass


def materialize_p41_module_shell_metadata(
    request: FCVAEModuleAvailabilityShellRequest
) -> FCVAEModuleAvailabilityShellMetadata:
    validate_module_availability_request(request)
    
    torch_status = build_torch_dependency_status(policy=TORCH_POLICY_OPTIONAL)
    if not torch_status.available:
        meta = FCVAEModuleAvailabilityShellMetadata(
            contract_version=FC_VAE_MODULE_AVAILABILITY_SHELL_CONTRACT_VERSION,
            shell_kind=FC_VAE_MODULE_AVAILABILITY_SHELL_KIND,
            module_name=FC_VAE_MODULE_AVAILABILITY_SHELL_MODULE_NAME,
            class_name=FC_VAE_MODULE_AVAILABILITY_SHELL_CLASS_NAME,
            torch_available=False,
            torch_import_safe=bool(torch_status.import_safe),
            top_level_torch_import_required=bool(torch_status.top_level_import_required),
            module_shell_created=False,
            module_object_returned=False,
            is_torch_nn_module=False,
            defines_forward=False,
            has_own_forward=False,
            uses_inherited_unimplemented_forward_only=False,
            defines_layers=False,
            parameter_count=0,
            buffer_count=0,
            training_mode_after_creation=False,
            module_device_type="cpu",
            architecture_id=FC_VAE_ARCHITECTURE_ID,
            input_flat_dim=32,
            latent_total_dim=20,
            latent_names=FC_VAE_REQUIRED_LATENT_NAMES,
            forward_execution_attempted=False,
            output_generation_attempted=False,
            training_attempted=False,
            reason="torch_unavailable_metadata",
        )
        validate_module_availability_metadata(meta)
        return meta
        
    torch = load_torch_for_p41_module_shell()
    cls = get_module_shell_class()
    instance = cls()
    
    is_nn_module = isinstance(instance, torch.nn.Module)
    has_forward_in_dict = ("forward" in cls.__dict__)
    
    # Check parameters/buffers
    param_count = sum(1 for _ in instance.parameters())
    buf_count = sum(1 for _ in instance.buffers())
    
    # uses_inherited_unimplemented_forward_only = True if the parent class forward is not overridden
    uses_inherited_unimplemented = (not has_forward_in_dict and hasattr(instance, "forward"))
    
    meta = FCVAEModuleAvailabilityShellMetadata(
        contract_version=FC_VAE_MODULE_AVAILABILITY_SHELL_CONTRACT_VERSION,
        shell_kind=FC_VAE_MODULE_AVAILABILITY_SHELL_KIND,
        module_name=FC_VAE_MODULE_AVAILABILITY_SHELL_MODULE_NAME,
        class_name=FC_VAE_MODULE_AVAILABILITY_SHELL_CLASS_NAME,
        torch_available=True,
        torch_import_safe=bool(torch_status.import_safe),
        top_level_torch_import_required=bool(torch_status.top_level_import_required),
        module_shell_created=True,
        module_object_returned=False,
        is_torch_nn_module=is_nn_module,
        defines_forward=False,
        has_own_forward=False,
        uses_inherited_unimplemented_forward_only=uses_inherited_unimplemented,
        defines_layers=False,
        parameter_count=param_count,
        buffer_count=buf_count,
        training_mode_after_creation=bool(instance.training),
        module_device_type="cpu",
        architecture_id=instance.architecture_id,
        input_flat_dim=instance.input_flat_dim,
        latent_total_dim=instance.latent_total_dim,
        latent_names=instance.latent_names,
        forward_execution_attempted=False,
        output_generation_attempted=False,
        training_attempted=False,
        reason="module_shell_successfully_instantiated",
    )
    validate_module_availability_metadata(meta)
    return meta


# Public Builders
def build_module_availability_shell_request_from_defaults() -> FCVAEModuleAvailabilityShellRequest:
    req = FCVAEModuleAvailabilityShellRequest(
        contract_version=FC_VAE_MODULE_AVAILABILITY_SHELL_CONTRACT_VERSION,
        shell_kind=FC_VAE_MODULE_AVAILABILITY_SHELL_KIND,
        architecture_id=FC_VAE_ARCHITECTURE_ID,
        source_torch_boundary_contract_version=TORCH_BOUNDARY_CONTRACT_VERSION,
        source_model_skeleton_contract_version=FC_VAE_MODEL_SKELETON_CONTRACT_VERSION,
        source_torch_module_stub_contract_version=FC_VAE_TORCH_MODULE_STUB_CONTRACT_VERSION,
        source_forward_eligibility_gate_contract_version=FC_VAE_FORWARD_ELIGIBILITY_GATE_CONTRACT_VERSION,
        expected_input_flat_dim=32,
        expected_latent_total_dim=20,
        expected_latent_names=FC_VAE_REQUIRED_LATENT_NAMES,
        target_device_type="cpu",
        allow_module_shell_creation_in_p41=True,
        allow_forward_execution_in_p41=False,
        allow_output_generation_in_p41=False,
        allow_training_in_p41=False,
        reason="p41_module_availability_shell_request_defaults",
    )
    validate_module_availability_request(req)
    return req


def build_module_availability_shell_metadata(
    request: FCVAEModuleAvailabilityShellRequest
) -> FCVAEModuleAvailabilityShellMetadata:
    validate_module_availability_request(request)
    meta = materialize_p41_module_shell_metadata(request)
    validate_module_availability_metadata(meta)
    return meta


def build_module_availability_shell_result(
    request: FCVAEModuleAvailabilityShellRequest
) -> FCVAEModuleAvailabilityShellResult:
    validate_module_availability_request(request)
    
    # Contract validation check against P27 smoke specs
    skeleton = build_fc_vae_skeleton_status()
    forward_contract = skeleton.forward_contract
    
    contract_match = (
        request.expected_input_flat_dim == forward_contract.input_shape.flat_dim == 32
        and request.expected_latent_total_dim == forward_contract.latent_layout.total_latent_dim == 20
        and request.expected_latent_names == forward_contract.latent_layout.latent_names == FC_VAE_REQUIRED_LATENT_NAMES
    )
    
    metadata = build_module_availability_shell_metadata(request)
    
    if not metadata.torch_available:
        status = FC_VAE_MODULE_AVAILABILITY_STATUS_TORCH_UNAVAILABLE
    elif not contract_match:
        status = FC_VAE_MODULE_AVAILABILITY_STATUS_CONTRACT_MISMATCH
    elif metadata.module_shell_created:
        status = FC_VAE_MODULE_AVAILABILITY_STATUS_SHELL_CREATED
    else:
        status = FC_VAE_MODULE_AVAILABILITY_STATUS_TORCH_UNAVAILABLE # fallback
        
    shell_available = (status == FC_VAE_MODULE_AVAILABILITY_STATUS_SHELL_CREATED)
    
    res = FCVAEModuleAvailabilityShellResult(
        contract_version=FC_VAE_MODULE_AVAILABILITY_SHELL_CONTRACT_VERSION,
        request=request,
        metadata=metadata,
        status=status,
        implementation_available_in_p41=shell_available,
        module_shell_available_in_p41=shell_available,
        forward_available_in_p41=False,
        forward_execution_available_in_p41=False,
        forward_executed_in_p41=False,
        output_generation_available_in_p41=False,
        output_generated_in_p41=False,
        training_available_in_p41=False,
        training_executed_in_p41=False,
        no_forward_execution=True,
        no_output_generation=True,
        no_training_loop=True,
        no_optimizer=True,
        no_checkpointing=True,
        no_artifact_generation=True,
        no_final_comparison=True,
        no_scientific_conclusion=True,
        reason="p41_module_availability_shell_result_built",
    )
    validate_module_availability_result(res)
    return res


def run_module_availability_shell_probe() -> FCVAEModuleAvailabilityShellResult:
    req = build_module_availability_shell_request_from_defaults()
    res = build_module_availability_shell_result(req)
    validate_module_availability_result(res)
    return res


# Serialization functions
def module_availability_shell_request_to_json_dict(request: FCVAEModuleAvailabilityShellRequest) -> dict:
    validate_module_availability_request(request)
    d = {
        "contract_version": request.contract_version,
        "shell_kind": request.shell_kind,
        "architecture_id": request.architecture_id,
        "source_torch_boundary_contract_version": request.source_torch_boundary_contract_version,
        "source_model_skeleton_contract_version": request.source_model_skeleton_contract_version,
        "source_torch_module_stub_contract_version": request.source_torch_module_stub_contract_version,
        "source_forward_eligibility_gate_contract_version": request.source_forward_eligibility_gate_contract_version,
        "expected_input_flat_dim": request.expected_input_flat_dim,
        "expected_latent_total_dim": request.expected_latent_total_dim,
        "expected_latent_names": list(request.expected_latent_names),
        "target_device_type": request.target_device_type,
        "allow_module_shell_creation_in_p41": request.allow_module_shell_creation_in_p41,
        "allow_forward_execution_in_p41": request.allow_forward_execution_in_p41,
        "allow_output_generation_in_p41": request.allow_output_generation_in_p41,
        "allow_training_in_p41": request.allow_training_in_p41,
        "reason": request.reason,
    }
    assert_no_local_path_leakage(request.reason, "reason")
    assert_no_forbidden_claims(request.reason, "reason")
    return d


def module_availability_shell_metadata_to_json_dict(metadata: FCVAEModuleAvailabilityShellMetadata) -> dict:
    validate_module_availability_metadata(metadata)
    d = {
        "contract_version": metadata.contract_version,
        "shell_kind": metadata.shell_kind,
        "module_name": metadata.module_name,
        "class_name": metadata.class_name,
        "torch_available": metadata.torch_available,
        "torch_import_safe": metadata.torch_import_safe,
        "top_level_torch_import_required": metadata.top_level_torch_import_required,
        "module_shell_created": metadata.module_shell_created,
        "module_object_returned": metadata.module_object_returned,
        "is_torch_nn_module": metadata.is_torch_nn_module,
        "defines_forward": metadata.defines_forward,
        "has_own_forward": metadata.has_own_forward,
        "uses_inherited_unimplemented_forward_only": metadata.uses_inherited_unimplemented_forward_only,
        "defines_layers": metadata.defines_layers,
        "parameter_count": metadata.parameter_count,
        "buffer_count": metadata.buffer_count,
        "training_mode_after_creation": metadata.training_mode_after_creation,
        "module_device_type": metadata.module_device_type,
        "architecture_id": metadata.architecture_id,
        "input_flat_dim": metadata.input_flat_dim,
        "latent_total_dim": metadata.latent_total_dim,
        "latent_names": list(metadata.latent_names),
        "forward_execution_attempted": metadata.forward_execution_attempted,
        "output_generation_attempted": metadata.output_generation_attempted,
        "training_attempted": metadata.training_attempted,
        "reason": metadata.reason,
    }
    assert_no_local_path_leakage(metadata.reason, "reason")
    assert_no_forbidden_claims(metadata.reason, "reason")
    return d


def module_availability_shell_result_to_json_dict(result: FCVAEModuleAvailabilityShellResult) -> dict:
    validate_module_availability_result(result)
    d = {
        "contract_version": result.contract_version,
        "request": module_availability_shell_request_to_json_dict(result.request),
        "metadata": module_availability_shell_metadata_to_json_dict(result.metadata),
        "status": result.status,
        "implementation_available_in_p41": result.implementation_available_in_p41,
        "module_shell_available_in_p41": result.module_shell_available_in_p41,
        "forward_available_in_p41": result.forward_available_in_p41,
        "forward_execution_available_in_p41": result.forward_execution_available_in_p41,
        "forward_executed_in_p41": result.forward_executed_in_p41,
        "output_generation_available_in_p41": result.output_generation_available_in_p41,
        "output_generated_in_p41": result.output_generated_in_p41,
        "training_available_in_p41": result.training_available_in_p41,
        "training_executed_in_p41": result.training_executed_in_p41,
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
    assert_no_local_path_leakage(result.reason, "reason")
    assert_no_forbidden_claims(result.reason, "reason")
    return d


def compact_module_availability_shell_json(result: FCVAEModuleAvailabilityShellResult) -> str:
    return json.dumps(
        module_availability_shell_result_to_json_dict(result),
        sort_keys=True,
        separators=(",", ":"),
    )
