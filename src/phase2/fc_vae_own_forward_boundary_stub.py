# src/phase2/fc_vae_own_forward_boundary_stub.py

import dataclasses
import inspect
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
    FC_VAE_MODULE_NAME,
    FC_VAE_REQUIRED_LATENT_NAMES,
)
from src.phase2.fc_vae_module_availability_shell import (
    FC_VAE_MODULE_AVAILABILITY_SHELL_CONTRACT_VERSION,
    run_module_availability_shell_probe,
)
from src.phase2.fc_vae_forward_readiness_gate import (
    FC_VAE_FORWARD_READINESS_GATE_CONTRACT_VERSION,
    run_forward_readiness_gate_probe,
)

# Public constants
FC_VAE_OWN_FORWARD_BOUNDARY_STUB_CONTRACT_VERSION = "phase2_p43_own_forward_boundary_stub_contract_v1"

FC_VAE_OWN_FORWARD_BOUNDARY_STUB_KIND = "own_forward_boundary_stub_declared_no_execution_no_output_no_training"

FC_VAE_OWN_FORWARD_BOUNDARY_STUB_MODULE_NAME = "src.phase2.fc_vae_own_forward_boundary_stub"

FC_VAE_OWN_FORWARD_BOUNDARY_STUB_CLASS_NAME = "P43FCVAEOwnForwardBoundaryStub"

# Statuses
FC_VAE_OWN_FORWARD_BOUNDARY_STUB_STATUS_TORCH_UNAVAILABLE = "blocked_torch_unavailable"
FC_VAE_OWN_FORWARD_BOUNDARY_STUB_STATUS_MODULE_SHELL_UNAVAILABLE = "blocked_by_module_shell_unavailable"
FC_VAE_OWN_FORWARD_BOUNDARY_STUB_STATUS_READINESS_UNAVAILABLE = "blocked_by_p42_readiness_unavailable"
FC_VAE_OWN_FORWARD_BOUNDARY_STUB_STATUS_DECLARED = "forward_boundary_declared_no_execution_no_output_no_training"
FC_VAE_OWN_FORWARD_BOUNDARY_STUB_STATUS_CONTRACT_MISMATCH = "blocked_by_contract_mismatch"

SUPPORTED_FC_VAE_OWN_FORWARD_BOUNDARY_STUB_STATUSES = (
    FC_VAE_OWN_FORWARD_BOUNDARY_STUB_STATUS_TORCH_UNAVAILABLE,
    FC_VAE_OWN_FORWARD_BOUNDARY_STUB_STATUS_MODULE_SHELL_UNAVAILABLE,
    FC_VAE_OWN_FORWARD_BOUNDARY_STUB_STATUS_READINESS_UNAVAILABLE,
    FC_VAE_OWN_FORWARD_BOUNDARY_STUB_STATUS_DECLARED,
    FC_VAE_OWN_FORWARD_BOUNDARY_STUB_STATUS_CONTRACT_MISMATCH,
)


# Dataclasses
@dataclasses.dataclass(frozen=True)
class FCVAEOwnForwardBoundaryStubRequest:
    contract_version: str
    boundary_kind: str
    architecture_id: str
    source_module_availability_shell_contract_version: str
    source_forward_readiness_gate_contract_version: str
    expected_input_flat_dim: int
    expected_latent_total_dim: int
    expected_latent_names: Tuple[str, ...]
    allow_forward_declaration_in_p43: bool
    allow_forward_execution_in_p43: bool
    allow_output_generation_in_p43: bool
    allow_training_in_p43: bool
    reason: str


@dataclasses.dataclass(frozen=True)
class FCVAEOwnForwardBoundaryStubMetadata:
    contract_version: str
    boundary_kind: str
    module_name: str
    class_name: str
    torch_available: bool
    torch_import_safe: bool
    top_level_torch_import_required: bool
    class_declared: bool
    module_instance_created: bool
    module_object_returned: bool
    is_torch_nn_module: bool
    own_forward_declared: bool
    forward_signature_available: bool
    forward_signature_parameters: Tuple[str, ...]
    forward_execution_attempted: bool
    output_generation_attempted: bool
    training_attempted: bool
    defines_layers: bool
    parameter_count: int
    buffer_count: int
    architecture_id: str
    input_flat_dim: int
    latent_total_dim: int
    latent_names: Tuple[str, ...]
    p41_module_shell_available: bool
    p42_forward_ready: bool
    p42_forward_blocked_as_expected: bool
    reason: str


@dataclasses.dataclass(frozen=True)
class FCVAEOwnForwardBoundaryStubResult:
    contract_version: str
    request: FCVAEOwnForwardBoundaryStubRequest
    metadata: FCVAEOwnForwardBoundaryStubMetadata
    status: str
    own_forward_boundary_declared_in_p43: bool
    forward_execution_available_in_p43: bool
    forward_executed_in_p43: bool
    output_generation_available_in_p43: bool
    output_generated_in_p43: bool
    training_available_in_p43: bool
    training_executed_in_p43: bool
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
def validate_non_empty_str(val: Any, name: str) -> None:
    if type(val) is not str:
        raise TypeError(f"{name} must be exact str instance")
    if not val.strip():
        raise ValueError(f"{name} cannot be empty or whitespace only")


def validate_bool(val: Any, name: str) -> None:
    if type(val) is not bool:
        raise TypeError(f"{name} must be exact bool instance")


def validate_positive_int(val: Any, name: str) -> None:
    if type(val) is not int:
        raise TypeError(f"{name} must be exact int instance")
    if val <= 0:
        raise ValueError(f"{name} must be positive")


def validate_non_negative_int(val: Any, name: str) -> None:
    if type(val) is not int:
        raise TypeError(f"{name} must be exact int instance")
    if val < 0:
        raise ValueError(f"{name} must be non-negative")


def validate_exact_tuple_str(val: Any, expected: Tuple[str, ...], name: str) -> None:
    if type(val) is not tuple:
        raise TypeError(f"{name} must be exact tuple instance")
    if val != expected:
        raise ValueError(f"{name} mismatch, expected {expected}")


def validate_own_forward_boundary_stub_status(status: Any) -> None:
    validate_non_empty_str(status, "status")
    if status not in SUPPORTED_FC_VAE_OWN_FORWARD_BOUNDARY_STUB_STATUSES:
        raise ValueError(f"Unsupported own forward boundary stub status: {status}")


def assert_no_local_path_leakage(val: str, name: str = "field") -> None:
    forbidden = [":\\", "Users", "home", "/Users", "/home", ".git"]
    val_lower = val.lower()
    for item in forbidden:
        if item.lower() in val_lower:
            raise ValueError(f"Local path leak detected in {name}")


def assert_no_forbidden_claims(val: str, name: str = "field") -> None:
    forbidden = ["scientific success", "solved", "best", "winner", "production ready", "state of the art"]
    val_lower = val.lower()
    for item in forbidden:
        if item.lower() in val_lower:
            raise ValueError(f"Forbidden claim detected in {name}")


def validate_own_forward_boundary_stub_request(request: FCVAEOwnForwardBoundaryStubRequest) -> None:
    if type(request) is not FCVAEOwnForwardBoundaryStubRequest:
        raise TypeError("request must be exact FCVAEOwnForwardBoundaryStubRequest instance")
    validate_non_empty_str(request.contract_version, "contract_version")
    if request.contract_version != FC_VAE_OWN_FORWARD_BOUNDARY_STUB_CONTRACT_VERSION:
        raise ValueError("Invalid contract_version")
    validate_non_empty_str(request.boundary_kind, "boundary_kind")
    if request.boundary_kind != FC_VAE_OWN_FORWARD_BOUNDARY_STUB_KIND:
        raise ValueError("Invalid boundary_kind")
    validate_non_empty_str(request.architecture_id, "architecture_id")
    if request.architecture_id != FC_VAE_ARCHITECTURE_ID:
        raise ValueError("Invalid architecture_id")

    validate_non_empty_str(request.source_module_availability_shell_contract_version, "source_module_availability_shell_contract_version")
    if request.source_module_availability_shell_contract_version != FC_VAE_MODULE_AVAILABILITY_SHELL_CONTRACT_VERSION:
        raise ValueError("Invalid source_module_availability_shell_contract_version")
    validate_non_empty_str(request.source_forward_readiness_gate_contract_version, "source_forward_readiness_gate_contract_version")
    if request.source_forward_readiness_gate_contract_version != FC_VAE_FORWARD_READINESS_GATE_CONTRACT_VERSION:
        raise ValueError("Invalid source_forward_readiness_gate_contract_version")

    validate_positive_int(request.expected_input_flat_dim, "expected_input_flat_dim")
    if request.expected_input_flat_dim != 32:
        raise ValueError("expected_input_flat_dim must be 32")
    validate_positive_int(request.expected_latent_total_dim, "expected_latent_total_dim")
    if request.expected_latent_total_dim != 20:
        raise ValueError("expected_latent_total_dim must be 20")
    validate_exact_tuple_str(request.expected_latent_names, FC_VAE_REQUIRED_LATENT_NAMES, "expected_latent_names")

    validate_bool(request.allow_forward_declaration_in_p43, "allow_forward_declaration_in_p43")
    if not request.allow_forward_declaration_in_p43:
        raise ValueError("allow_forward_declaration_in_p43 must be True")

    for flag_name in ("allow_forward_execution_in_p43", "allow_output_generation_in_p43", "allow_training_in_p43"):
        val = getattr(request, flag_name)
        validate_bool(val, flag_name)
        if val:
            raise ValueError(f"{flag_name} must be False")

    validate_non_empty_str(request.reason, "reason")
    assert_no_local_path_leakage(request.reason, "reason")
    assert_no_forbidden_claims(request.reason, "reason")


def validate_own_forward_boundary_stub_metadata(metadata: FCVAEOwnForwardBoundaryStubMetadata) -> None:
    if type(metadata) is not FCVAEOwnForwardBoundaryStubMetadata:
        raise TypeError("metadata must be exact FCVAEOwnForwardBoundaryStubMetadata instance")
    validate_non_empty_str(metadata.contract_version, "contract_version")
    if metadata.contract_version != FC_VAE_OWN_FORWARD_BOUNDARY_STUB_CONTRACT_VERSION:
        raise ValueError("Invalid contract_version")
    validate_non_empty_str(metadata.boundary_kind, "boundary_kind")
    if metadata.boundary_kind != FC_VAE_OWN_FORWARD_BOUNDARY_STUB_KIND:
        raise ValueError("Invalid boundary_kind")
    validate_non_empty_str(metadata.module_name, "module_name")
    if metadata.module_name != FC_VAE_OWN_FORWARD_BOUNDARY_STUB_MODULE_NAME:
        raise ValueError("Invalid module_name")
    validate_non_empty_str(metadata.class_name, "class_name")
    if metadata.class_name != FC_VAE_OWN_FORWARD_BOUNDARY_STUB_CLASS_NAME:
        raise ValueError("Invalid class_name")

    validate_bool(metadata.torch_available, "torch_available")
    validate_bool(metadata.torch_import_safe, "torch_import_safe")
    validate_bool(metadata.top_level_torch_import_required, "top_level_torch_import_required")
    validate_bool(metadata.class_declared, "class_declared")
    validate_bool(metadata.module_instance_created, "module_instance_created")
    validate_bool(metadata.module_object_returned, "module_object_returned")
    if metadata.module_object_returned:
        raise ValueError("module_object_returned must be False")

    validate_bool(metadata.is_torch_nn_module, "is_torch_nn_module")
    validate_bool(metadata.own_forward_declared, "own_forward_declared")
    validate_bool(metadata.forward_signature_available, "forward_signature_available")
    if type(metadata.forward_signature_parameters) is not tuple:
        raise TypeError("forward_signature_parameters must be tuple")

    for flag_name in ("forward_execution_attempted", "output_generation_attempted", "training_attempted"):
        val = getattr(metadata, flag_name)
        validate_bool(val, flag_name)
        if val:
            raise ValueError(f"{flag_name} must be False")

    validate_bool(metadata.defines_layers, "defines_layers")
    if metadata.defines_layers:
        raise ValueError("defines_layers must be False")
    validate_non_negative_int(metadata.parameter_count, "parameter_count")
    if metadata.parameter_count != 0:
        raise ValueError("parameter_count must be 0")
    validate_non_negative_int(metadata.buffer_count, "buffer_count")
    if metadata.buffer_count != 0:
        raise ValueError("buffer_count must be 0")

    if metadata.class_declared:
        validate_non_empty_str(metadata.architecture_id, "architecture_id")
        validate_positive_int(metadata.input_flat_dim, "input_flat_dim")
        validate_positive_int(metadata.latent_total_dim, "latent_total_dim")
        if type(metadata.latent_names) is not tuple:
            raise TypeError("latent_names must be tuple")
    else:
        if metadata.latent_names != ():
            raise ValueError("latent_names must be empty tuple if class not declared")

    validate_bool(metadata.p41_module_shell_available, "p41_module_shell_available")
    validate_bool(metadata.p42_forward_ready, "p42_forward_ready")
    if metadata.p42_forward_ready:
        raise ValueError("p42_forward_ready must be False")
    validate_bool(metadata.p42_forward_blocked_as_expected, "p42_forward_blocked_as_expected")

    validate_non_empty_str(metadata.reason, "reason")
    assert_no_local_path_leakage(metadata.reason, "reason")
    assert_no_forbidden_claims(metadata.reason, "reason")


def validate_own_forward_boundary_stub_result(result: FCVAEOwnForwardBoundaryStubResult) -> None:
    if type(result) is not FCVAEOwnForwardBoundaryStubResult:
        raise TypeError("result must be exact FCVAEOwnForwardBoundaryStubResult instance")
    validate_non_empty_str(result.contract_version, "contract_version")
    if result.contract_version != FC_VAE_OWN_FORWARD_BOUNDARY_STUB_CONTRACT_VERSION:
        raise ValueError("Invalid contract_version")

    validate_own_forward_boundary_stub_request(result.request)
    validate_own_forward_boundary_stub_metadata(result.metadata)
    validate_own_forward_boundary_stub_status(result.status)

    validate_bool(result.own_forward_boundary_declared_in_p43, "own_forward_boundary_declared_in_p43")
    if result.metadata.own_forward_declared != result.own_forward_boundary_declared_in_p43:
        raise ValueError("own_forward_boundary_declared_in_p43 mismatch against metadata")

    for flag_name in (
        "forward_execution_available_in_p43",
        "forward_executed_in_p43",
        "output_generation_available_in_p43",
        "output_generated_in_p43",
        "training_available_in_p43",
        "training_executed_in_p43",
    ):
        val = getattr(result, flag_name)
        validate_bool(val, flag_name)
        if val:
            raise ValueError(f"{flag_name} must be False")

    for flag_name in (
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

    validate_non_empty_str(result.reason, "reason")
    assert_no_local_path_leakage(result.reason, "reason")
    assert_no_forbidden_claims(result.reason, "reason")


# Torch Loader and dynamic class Definition
def load_torch_for_p43_forward_boundary_stub() -> Any:
    status = build_torch_dependency_status(policy=TORCH_POLICY_OPTIONAL)
    if not status.available or not status.import_safe:
        raise RuntimeError("PyTorch is not available or safe to import in P43 loader.")
    import torch
    return torch


def get_p43_own_forward_boundary_stub_class() -> type:
    torch = load_torch_for_p43_forward_boundary_stub()

    class P43FCVAEOwnForwardBoundaryStub(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.architecture_id = "FC-VAE"
            self.input_flat_dim = 32
            self.latent_total_dim = 20
            self.latent_names = ("z_mean", "z_volatility", "z_shared")

        def forward(self, *args: Any, **kwargs: Any) -> Any:
            raise NotImplementedError("P43 declares the FC-VAE forward boundary but does not allow forward execution.")

    return P43FCVAEOwnForwardBoundaryStub


# Builders
def build_own_forward_boundary_stub_request_from_defaults() -> FCVAEOwnForwardBoundaryStubRequest:
    return FCVAEOwnForwardBoundaryStubRequest(
        contract_version=FC_VAE_OWN_FORWARD_BOUNDARY_STUB_CONTRACT_VERSION,
        boundary_kind=FC_VAE_OWN_FORWARD_BOUNDARY_STUB_KIND,
        architecture_id=FC_VAE_ARCHITECTURE_ID,
        source_module_availability_shell_contract_version=FC_VAE_MODULE_AVAILABILITY_SHELL_CONTRACT_VERSION,
        source_forward_readiness_gate_contract_version=FC_VAE_FORWARD_READINESS_GATE_CONTRACT_VERSION,
        expected_input_flat_dim=32,
        expected_latent_total_dim=20,
        expected_latent_names=FC_VAE_REQUIRED_LATENT_NAMES,
        allow_forward_declaration_in_p43=True,
        allow_forward_execution_in_p43=False,
        allow_output_generation_in_p43=False,
        allow_training_in_p43=False,
        reason="p43_own_forward_boundary_stub_request_defaults",
    )


def build_own_forward_boundary_stub_metadata(
    request: FCVAEOwnForwardBoundaryStubRequest
) -> FCVAEOwnForwardBoundaryStubMetadata:
    validate_own_forward_boundary_stub_request(request)

    # Query reference modules
    torch_status = build_torch_dependency_status(policy=TORCH_POLICY_OPTIONAL)
    p41_res = run_module_availability_shell_probe()
    p42_res = run_forward_readiness_gate_probe()

    torch_available = bool(torch_status.available)
    torch_import_safe = bool(torch_status.import_safe)
    top_level_torch_import_required = bool(torch_status.top_level_import_required)

    p41_module_shell_available = bool(p41_res.implementation_available_in_p41)
    p42_forward_ready = bool(p42_res.forward_ready_in_p42)
    p42_forward_blocked_as_expected = (p42_res.status == "blocked_by_forward_implementation_unavailable")

    class_declared = False
    module_instance_created = False
    is_torch_nn_module = False
    own_forward_declared = False
    forward_signature_available = False
    forward_signature_parameters = ()
    defines_layers = False
    parameter_count = 0
    buffer_count = 0
    architecture_id = ""
    input_flat_dim = 0
    latent_total_dim = 0
    latent_names = ()
    reason = "p43_own_forward_boundary_stub_metadata_built"

    if torch_available and torch_import_safe:
        try:
            cls = get_p43_own_forward_boundary_stub_class()
            class_declared = True

            # Instantiate class to check properties
            instance = cls()
            module_instance_created = True

            torch_lib = load_torch_for_p43_forward_boundary_stub()
            is_torch_nn_module = isinstance(instance, torch_lib.nn.Module)

            own_forward_declared = "forward" in cls.__dict__

            if own_forward_declared:
                sig = inspect.signature(cls.forward)
                forward_signature_available = True
                forward_signature_parameters = tuple(sig.parameters.keys())

            defines_layers = False
            parameter_count = len(list(instance.parameters()))
            buffer_count = len(list(instance.buffers()))

            architecture_id = str(getattr(instance, "architecture_id", ""))
            input_flat_dim = int(getattr(instance, "input_flat_dim", 0))
            latent_total_dim = int(getattr(instance, "latent_total_dim", 0))
            latent_names = tuple(getattr(instance, "latent_names", ()))
        except Exception as e:
            reason = f"p43_own_forward_boundary_stub_metadata_failed: {str(e)}"

    metadata = FCVAEOwnForwardBoundaryStubMetadata(
        contract_version=FC_VAE_OWN_FORWARD_BOUNDARY_STUB_CONTRACT_VERSION,
        boundary_kind=FC_VAE_OWN_FORWARD_BOUNDARY_STUB_KIND,
        module_name=FC_VAE_OWN_FORWARD_BOUNDARY_STUB_MODULE_NAME,
        class_name=FC_VAE_OWN_FORWARD_BOUNDARY_STUB_CLASS_NAME,
        torch_available=torch_available,
        torch_import_safe=torch_import_safe,
        top_level_torch_import_required=top_level_torch_import_required,
        class_declared=class_declared,
        module_instance_created=module_instance_created,
        module_object_returned=False,
        is_torch_nn_module=is_torch_nn_module,
        own_forward_declared=own_forward_declared,
        forward_signature_available=forward_signature_available,
        forward_signature_parameters=forward_signature_parameters,
        forward_execution_attempted=False,
        output_generation_attempted=False,
        training_attempted=False,
        defines_layers=defines_layers,
        parameter_count=parameter_count,
        buffer_count=buffer_count,
        architecture_id=architecture_id,
        input_flat_dim=input_flat_dim,
        latent_total_dim=latent_total_dim,
        latent_names=latent_names,
        p41_module_shell_available=p41_module_shell_available,
        p42_forward_ready=p42_forward_ready,
        p42_forward_blocked_as_expected=p42_forward_blocked_as_expected,
        reason=reason,
    )
    validate_own_forward_boundary_stub_metadata(metadata)
    return metadata


def build_own_forward_boundary_stub_result(
    request: FCVAEOwnForwardBoundaryStubRequest
) -> FCVAEOwnForwardBoundaryStubResult:
    validate_own_forward_boundary_stub_request(request)
    metadata = build_own_forward_boundary_stub_metadata(request)

    # Route status
    if not metadata.torch_available:
        status = FC_VAE_OWN_FORWARD_BOUNDARY_STUB_STATUS_TORCH_UNAVAILABLE
    elif not metadata.p41_module_shell_available:
        status = FC_VAE_OWN_FORWARD_BOUNDARY_STUB_STATUS_MODULE_SHELL_UNAVAILABLE
    elif not metadata.p42_forward_blocked_as_expected:
        status = FC_VAE_OWN_FORWARD_BOUNDARY_STUB_STATUS_READINESS_UNAVAILABLE
    elif (
        metadata.class_declared
        and metadata.own_forward_declared
        and not metadata.forward_execution_attempted
    ):
        contract_match = (
            metadata.architecture_id == request.architecture_id
            and metadata.input_flat_dim == request.expected_input_flat_dim
            and metadata.latent_total_dim == request.expected_latent_total_dim
            and metadata.latent_names == request.expected_latent_names
        )
        if contract_match:
            status = FC_VAE_OWN_FORWARD_BOUNDARY_STUB_STATUS_DECLARED
        else:
            status = FC_VAE_OWN_FORWARD_BOUNDARY_STUB_STATUS_CONTRACT_MISMATCH
    else:
        status = FC_VAE_OWN_FORWARD_BOUNDARY_STUB_STATUS_CONTRACT_MISMATCH

    res = FCVAEOwnForwardBoundaryStubResult(
        contract_version=FC_VAE_OWN_FORWARD_BOUNDARY_STUB_CONTRACT_VERSION,
        request=request,
        metadata=metadata,
        status=status,
        own_forward_boundary_declared_in_p43=bool(metadata.own_forward_declared),
        forward_execution_available_in_p43=False,
        forward_executed_in_p43=False,
        output_generation_available_in_p43=False,
        output_generated_in_p43=False,
        training_available_in_p43=False,
        training_executed_in_p43=False,
        no_forward_execution=True,
        no_output_generation=True,
        no_training_loop=True,
        no_optimizer=True,
        no_checkpointing=True,
        no_artifact_generation=True,
        no_final_comparison=True,
        no_scientific_conclusion=True,
        reason="p43_own_forward_boundary_stub_result_built",
    )
    validate_own_forward_boundary_stub_result(res)
    return res


def run_own_forward_boundary_stub_probe() -> FCVAEOwnForwardBoundaryStubResult:
    req = build_own_forward_boundary_stub_request_from_defaults()
    res = build_own_forward_boundary_stub_result(req)
    validate_own_forward_boundary_stub_result(res)
    return res


# Serialization
def own_forward_boundary_stub_request_to_json_dict(request: FCVAEOwnForwardBoundaryStubRequest) -> dict:
    validate_own_forward_boundary_stub_request(request)
    d = {
        "contract_version": request.contract_version,
        "boundary_kind": request.boundary_kind,
        "architecture_id": request.architecture_id,
        "source_module_availability_shell_contract_version": request.source_module_availability_shell_contract_version,
        "source_forward_readiness_gate_contract_version": request.source_forward_readiness_gate_contract_version,
        "expected_input_flat_dim": request.expected_input_flat_dim,
        "expected_latent_total_dim": request.expected_latent_total_dim,
        "expected_latent_names": list(request.expected_latent_names),
        "allow_forward_declaration_in_p43": request.allow_forward_declaration_in_p43,
        "allow_forward_execution_in_p43": request.allow_forward_execution_in_p43,
        "allow_output_generation_in_p43": request.allow_output_generation_in_p43,
        "allow_training_in_p43": request.allow_training_in_p43,
        "reason": request.reason,
    }
    assert_no_local_path_leakage(request.reason, "reason")
    assert_no_forbidden_claims(request.reason, "reason")
    return d


def own_forward_boundary_stub_metadata_to_json_dict(metadata: FCVAEOwnForwardBoundaryStubMetadata) -> dict:
    validate_own_forward_boundary_stub_metadata(metadata)
    d = {
        "contract_version": metadata.contract_version,
        "boundary_kind": metadata.boundary_kind,
        "module_name": metadata.module_name,
        "class_name": metadata.class_name,
        "torch_available": metadata.torch_available,
        "torch_import_safe": metadata.torch_import_safe,
        "top_level_torch_import_required": metadata.top_level_torch_import_required,
        "class_declared": metadata.class_declared,
        "module_instance_created": metadata.module_instance_created,
        "module_object_returned": metadata.module_object_returned,
        "is_torch_nn_module": metadata.is_torch_nn_module,
        "own_forward_declared": metadata.own_forward_declared,
        "forward_signature_available": metadata.forward_signature_available,
        "forward_signature_parameters": list(metadata.forward_signature_parameters),
        "forward_execution_attempted": metadata.forward_execution_attempted,
        "output_generation_attempted": metadata.output_generation_attempted,
        "training_attempted": metadata.training_attempted,
        "defines_layers": metadata.defines_layers,
        "parameter_count": metadata.parameter_count,
        "buffer_count": metadata.buffer_count,
        "architecture_id": metadata.architecture_id,
        "input_flat_dim": metadata.input_flat_dim,
        "latent_total_dim": metadata.latent_total_dim,
        "latent_names": list(metadata.latent_names),
        "p41_module_shell_available": metadata.p41_module_shell_available,
        "p42_forward_ready": metadata.p42_forward_ready,
        "p42_forward_blocked_as_expected": metadata.p42_forward_blocked_as_expected,
        "reason": metadata.reason,
    }
    assert_no_local_path_leakage(metadata.reason, "reason")
    assert_no_forbidden_claims(metadata.reason, "reason")
    return d


def own_forward_boundary_stub_result_to_json_dict(result: FCVAEOwnForwardBoundaryStubResult) -> dict:
    validate_own_forward_boundary_stub_result(result)
    d = {
        "contract_version": result.contract_version,
        "request": own_forward_boundary_stub_request_to_json_dict(result.request),
        "metadata": own_forward_boundary_stub_metadata_to_json_dict(result.metadata),
        "status": result.status,
        "own_forward_boundary_declared_in_p43": result.own_forward_boundary_declared_in_p43,
        "forward_execution_available_in_p43": result.forward_execution_available_in_p43,
        "forward_executed_in_p43": result.forward_executed_in_p43,
        "output_generation_available_in_p43": result.output_generation_available_in_p43,
        "output_generated_in_p43": result.output_generated_in_p43,
        "training_available_in_p43": result.training_available_in_p43,
        "training_executed_in_p43": result.training_executed_in_p43,
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


def compact_own_forward_boundary_stub_json(result: FCVAEOwnForwardBoundaryStubResult) -> str:
    d = own_forward_boundary_stub_result_to_json_dict(result)
    return json.dumps(d, sort_keys=True, separators=(",", ":"))
