# src/phase2/fc_vae_forward_readiness_gate.py

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
    FC_VAE_MODULE_NAME,
    FC_VAE_REQUIRED_LATENT_NAMES,
)
from src.phase2.fc_vae_tensor_materialization import (
    FC_VAE_TENSOR_MATERIALIZATION_CONTRACT_VERSION,
    run_tensor_materialization_probe,
)
from src.phase2.fc_vae_forward_eligibility_gate import (
    FC_VAE_FORWARD_ELIGIBILITY_GATE_CONTRACT_VERSION,
    run_forward_eligibility_gate_probe,
)
from src.phase2.fc_vae_module_availability_shell import (
    FC_VAE_MODULE_AVAILABILITY_SHELL_CONTRACT_VERSION,
    run_module_availability_shell_probe,
)

# Public constants
FC_VAE_FORWARD_READINESS_GATE_CONTRACT_VERSION = "phase2_p42_forward_readiness_gate_after_module_shell_contract_v1"

FC_VAE_FORWARD_READINESS_GATE_KIND = "forward_readiness_gate_after_module_shell_no_forward_no_output_no_training"

FC_VAE_FORWARD_READINESS_GATE_MODULE_NAME = "src.phase2.fc_vae_forward_readiness_gate"

FC_VAE_FORWARD_READINESS_STATUS_TORCH_UNAVAILABLE = "blocked_torch_unavailable"

FC_VAE_FORWARD_READINESS_STATUS_TENSOR_UNAVAILABLE = "blocked_by_tensor_materialization_unavailable"

FC_VAE_FORWARD_READINESS_STATUS_MODULE_SHELL_UNAVAILABLE = "blocked_by_module_shell_unavailable"

FC_VAE_FORWARD_READINESS_STATUS_CONTRACT_MISMATCH = "blocked_by_contract_mismatch"

FC_VAE_FORWARD_READINESS_STATUS_FORWARD_IMPLEMENTATION_UNAVAILABLE = "blocked_by_forward_implementation_unavailable"

SUPPORTED_FC_VAE_FORWARD_READINESS_STATUSES = (
    FC_VAE_FORWARD_READINESS_STATUS_TORCH_UNAVAILABLE,
    FC_VAE_FORWARD_READINESS_STATUS_TENSOR_UNAVAILABLE,
    FC_VAE_FORWARD_READINESS_STATUS_MODULE_SHELL_UNAVAILABLE,
    FC_VAE_FORWARD_READINESS_STATUS_CONTRACT_MISMATCH,
    FC_VAE_FORWARD_READINESS_STATUS_FORWARD_IMPLEMENTATION_UNAVAILABLE,
)


# Dataclasses
@dataclasses.dataclass(frozen=True)
class FCVAEForwardReadinessRequest:
    contract_version: str
    gate_kind: str
    architecture_id: str
    source_tensor_materialization_contract_version: str
    source_forward_eligibility_gate_contract_version: str
    source_module_availability_shell_contract_version: str
    expected_batch_size: int
    expected_input_flat_dim: int
    expected_shape_tuple: Tuple[int, int]
    expected_tensor_dtype_name: str
    expected_tensor_device_type: str
    expected_latent_total_dim: int
    expected_latent_names: Tuple[str, ...]
    allow_forward_execution_in_p42: bool
    allow_output_generation_in_p42: bool
    allow_training_in_p42: bool
    reason: str


@dataclasses.dataclass(frozen=True)
class FCVAEForwardReadinessEvidence:
    contract_version: str
    torch_available: bool
    tensor_materialized_in_p39: bool
    tensor_materialization_status: str
    tensor_shape_tuple: Tuple[int, int]
    tensor_dtype_name: str
    tensor_device_type: str
    tensor_numel: int
    tensor_requires_grad: bool
    tensor_values_match_p37_nested_values: bool
    p40_gate_status: str
    p40_shape_matches: bool
    p40_dtype_matches: bool
    p40_device_matches: bool
    p40_latent_layout_matches: bool
    module_shell_created_in_p41: bool
    module_shell_status: str
    implementation_available_in_p41: bool
    module_shell_available_in_p41: bool
    is_torch_nn_module: bool
    module_object_returned: bool
    defines_forward: bool
    has_own_forward: bool
    uses_inherited_unimplemented_forward_only: bool
    defines_layers: bool
    parameter_count: int
    buffer_count: int
    module_architecture_id: str
    module_input_flat_dim: int
    module_latent_total_dim: int
    module_latent_names: Tuple[str, ...]
    tensor_module_shape_compatible: bool
    module_contract_compatible: bool
    forward_implementation_available: bool
    forward_execution_attempted: bool
    output_generation_attempted: bool
    training_attempted: bool
    reason: str


@dataclasses.dataclass(frozen=True)
class FCVAEForwardReadinessResult:
    contract_version: str
    request: FCVAEForwardReadinessRequest
    evidence: FCVAEForwardReadinessEvidence
    status: str
    forward_ready_in_p42: bool
    module_shell_ready_in_p42: bool
    implementation_available_in_p41: bool
    forward_implementation_available_in_p42: bool
    forward_available_in_p42: bool
    forward_execution_available_in_p42: bool
    forward_executed_in_p42: bool
    output_generation_available_in_p42: bool
    output_generated_in_p42: bool
    training_available_in_p42: bool
    training_executed_in_p42: bool
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


def validate_shape_tuple(val: Any) -> None:
    if type(val) is not tuple or len(val) != 2:
        raise TypeError("shape_tuple must be exact tuple of length 2")
    validate_positive_int(val[0], "shape_tuple[0]")
    validate_positive_int(val[1], "shape_tuple[1]")


def validate_exact_tuple_str(val: Any, expected: Tuple[str, ...], name: str) -> None:
    if type(val) is not tuple:
        raise TypeError(f"{name} must be exact tuple instance")
    if val != expected:
        raise ValueError(f"{name} mismatch, expected {expected}")


def validate_forward_readiness_status(status: Any) -> None:
    validate_non_empty_str(status, "status")
    if status not in SUPPORTED_FC_VAE_FORWARD_READINESS_STATUSES:
        raise ValueError(f"Unsupported forward readiness status: {status}")


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


def validate_forward_readiness_request(request: FCVAEForwardReadinessRequest) -> None:
    if type(request) is not FCVAEForwardReadinessRequest:
        raise TypeError("request must be exact FCVAEForwardReadinessRequest instance")
    validate_non_empty_str(request.contract_version, "contract_version")
    if request.contract_version != FC_VAE_FORWARD_READINESS_GATE_CONTRACT_VERSION:
        raise ValueError("Invalid contract_version")
    validate_non_empty_str(request.gate_kind, "gate_kind")
    if request.gate_kind != FC_VAE_FORWARD_READINESS_GATE_KIND:
        raise ValueError("Invalid gate_kind")
    validate_non_empty_str(request.architecture_id, "architecture_id")
    if request.architecture_id != FC_VAE_ARCHITECTURE_ID:
        raise ValueError("Invalid architecture_id")

    validate_non_empty_str(request.source_tensor_materialization_contract_version, "source_tensor_materialization_contract_version")
    if request.source_tensor_materialization_contract_version != FC_VAE_TENSOR_MATERIALIZATION_CONTRACT_VERSION:
        raise ValueError("Invalid source_tensor_materialization_contract_version")
    validate_non_empty_str(request.source_forward_eligibility_gate_contract_version, "source_forward_eligibility_gate_contract_version")
    if request.source_forward_eligibility_gate_contract_version != FC_VAE_FORWARD_ELIGIBILITY_GATE_CONTRACT_VERSION:
        raise ValueError("Invalid source_forward_eligibility_gate_contract_version")
    validate_non_empty_str(request.source_module_availability_shell_contract_version, "source_module_availability_shell_contract_version")
    if request.source_module_availability_shell_contract_version != FC_VAE_MODULE_AVAILABILITY_SHELL_CONTRACT_VERSION:
        raise ValueError("Invalid source_module_availability_shell_contract_version")

    validate_positive_int(request.expected_batch_size, "expected_batch_size")
    if request.expected_batch_size != 2:
        raise ValueError("expected_batch_size must be 2")
    validate_positive_int(request.expected_input_flat_dim, "expected_input_flat_dim")
    if request.expected_input_flat_dim != 32:
        raise ValueError("expected_input_flat_dim must be 32")
    validate_shape_tuple(request.expected_shape_tuple)
    if request.expected_shape_tuple != (2, 32):
        raise ValueError("expected_shape_tuple must be (2, 32)")

    validate_non_empty_str(request.expected_tensor_dtype_name, "expected_tensor_dtype_name")
    if "float32" not in request.expected_tensor_dtype_name:
        raise ValueError("expected_tensor_dtype_name must be float32")
    validate_non_empty_str(request.expected_tensor_device_type, "expected_tensor_device_type")
    if request.expected_tensor_device_type != "cpu":
        raise ValueError("expected_tensor_device_type must be cpu")

    validate_positive_int(request.expected_latent_total_dim, "expected_latent_total_dim")
    if request.expected_latent_total_dim != 20:
        raise ValueError("expected_latent_total_dim must be 20")
    validate_exact_tuple_str(request.expected_latent_names, FC_VAE_REQUIRED_LATENT_NAMES, "expected_latent_names")

    for flag_name in ("allow_forward_execution_in_p42", "allow_output_generation_in_p42", "allow_training_in_p42"):
        val = getattr(request, flag_name)
        validate_bool(val, flag_name)
        if val:
            raise ValueError(f"{flag_name} must be False")

    validate_non_empty_str(request.reason, "reason")
    assert_no_local_path_leakage(request.reason, "reason")
    assert_no_forbidden_claims(request.reason, "reason")


def validate_forward_readiness_evidence(evidence: FCVAEForwardReadinessEvidence) -> None:
    if type(evidence) is not FCVAEForwardReadinessEvidence:
        raise TypeError("evidence must be exact FCVAEForwardReadinessEvidence instance")
    validate_non_empty_str(evidence.contract_version, "contract_version")
    if evidence.contract_version != FC_VAE_FORWARD_READINESS_GATE_CONTRACT_VERSION:
        raise ValueError("Invalid contract_version")

    validate_bool(evidence.torch_available, "torch_available")
    validate_bool(evidence.tensor_materialized_in_p39, "tensor_materialized_in_p39")
    validate_non_empty_str(evidence.tensor_materialization_status, "tensor_materialization_status")

    if evidence.tensor_materialized_in_p39:
        validate_shape_tuple(evidence.tensor_shape_tuple)
        validate_non_empty_str(evidence.tensor_dtype_name, "tensor_dtype_name")
        validate_non_empty_str(evidence.tensor_device_type, "tensor_device_type")
        validate_positive_int(evidence.tensor_numel, "tensor_numel")
        validate_bool(evidence.tensor_requires_grad, "tensor_requires_grad")
        validate_bool(evidence.tensor_values_match_p37_nested_values, "tensor_values_match_p37_nested_values")
    else:
        if evidence.tensor_shape_tuple != (0, 0):
            raise ValueError("tensor_shape_tuple must be (0, 0) if not materialized")
        if evidence.tensor_numel != 0:
            raise ValueError("tensor_numel must be 0 if not materialized")
        validate_bool(evidence.tensor_requires_grad, "tensor_requires_grad")
        validate_bool(evidence.tensor_values_match_p37_nested_values, "tensor_values_match_p37_nested_values")

    validate_non_empty_str(evidence.p40_gate_status, "p40_gate_status")
    validate_bool(evidence.p40_shape_matches, "p40_shape_matches")
    validate_bool(evidence.p40_dtype_matches, "p40_dtype_matches")
    validate_bool(evidence.p40_device_matches, "p40_device_matches")
    validate_bool(evidence.p40_latent_layout_matches, "p40_latent_layout_matches")

    validate_bool(evidence.module_shell_created_in_p41, "module_shell_created_in_p41")
    validate_non_empty_str(evidence.module_shell_status, "module_shell_status")
    validate_bool(evidence.implementation_available_in_p41, "implementation_available_in_p41")
    validate_bool(evidence.module_shell_available_in_p41, "module_shell_available_in_p41")
    validate_bool(evidence.is_torch_nn_module, "is_torch_nn_module")
    validate_bool(evidence.module_object_returned, "module_object_returned")
    if evidence.module_object_returned:
        raise ValueError("module_object_returned must be False")

    validate_bool(evidence.defines_forward, "defines_forward")
    validate_bool(evidence.has_own_forward, "has_own_forward")
    validate_bool(evidence.uses_inherited_unimplemented_forward_only, "uses_inherited_unimplemented_forward_only")
    validate_bool(evidence.defines_layers, "defines_layers")

    validate_non_negative_int(evidence.parameter_count, "parameter_count")
    validate_non_negative_int(evidence.buffer_count, "buffer_count")
    if evidence.parameter_count != 0:
        raise ValueError("parameter_count must be 0")
    if evidence.buffer_count != 0:
        raise ValueError("buffer_count must be 0")

    if evidence.module_shell_created_in_p41:
        validate_non_empty_str(evidence.module_architecture_id, "module_architecture_id")
        validate_positive_int(evidence.module_input_flat_dim, "module_input_flat_dim")
        validate_positive_int(evidence.module_latent_total_dim, "module_latent_total_dim")
        if type(evidence.module_latent_names) is not tuple:
            raise TypeError("module_latent_names must be tuple")
    else:
        if evidence.module_latent_names != ():
            raise ValueError("module_latent_names must be empty tuple if not created")

    validate_bool(evidence.tensor_module_shape_compatible, "tensor_module_shape_compatible")
    validate_bool(evidence.module_contract_compatible, "module_contract_compatible")
    validate_bool(evidence.forward_implementation_available, "forward_implementation_available")
    if evidence.forward_implementation_available:
        raise ValueError("forward_implementation_available must be False in P42")

    for flag_name in ("forward_execution_attempted", "output_generation_attempted", "training_attempted"):
        val = getattr(evidence, flag_name)
        validate_bool(val, flag_name)
        if val:
            raise ValueError(f"{flag_name} must be False")

    validate_non_empty_str(evidence.reason, "reason")
    assert_no_local_path_leakage(evidence.reason, "reason")
    assert_no_forbidden_claims(evidence.reason, "reason")


def validate_forward_readiness_result(result: FCVAEForwardReadinessResult) -> None:
    if type(result) is not FCVAEForwardReadinessResult:
        raise TypeError("result must be exact FCVAEForwardReadinessResult instance")
    validate_non_empty_str(result.contract_version, "contract_version")
    if result.contract_version != FC_VAE_FORWARD_READINESS_GATE_CONTRACT_VERSION:
        raise ValueError("Invalid contract_version")

    validate_forward_readiness_request(result.request)
    validate_forward_readiness_evidence(result.evidence)
    validate_forward_readiness_status(result.status)

    validate_bool(result.forward_ready_in_p42, "forward_ready_in_p42")
    if result.forward_ready_in_p42:
        raise ValueError("forward_ready_in_p42 must be False")

    validate_bool(result.module_shell_ready_in_p42, "module_shell_ready_in_p42")
    if result.evidence.module_shell_created_in_p41 and result.evidence.module_contract_compatible:
        if not result.module_shell_ready_in_p42:
            raise ValueError("module_shell_ready_in_p42 must be True when shell is available and compatible")
    else:
        if result.module_shell_ready_in_p42:
            raise ValueError("module_shell_ready_in_p42 must be False when shell is unavailable or incompatible")

    validate_bool(result.implementation_available_in_p41, "implementation_available_in_p41")
    if result.implementation_available_in_p41 != result.evidence.implementation_available_in_p41:
        raise ValueError("implementation_available_in_p41 mismatch against evidence")

    validate_bool(result.forward_implementation_available_in_p42, "forward_implementation_available_in_p42")
    if result.forward_implementation_available_in_p42 != result.evidence.forward_implementation_available:
        raise ValueError("forward_implementation_available_in_p42 mismatch against evidence")

    for flag_name in (
        "forward_available_in_p42",
        "forward_execution_available_in_p42",
        "forward_executed_in_p42",
        "output_generation_available_in_p42",
        "output_generated_in_p42",
        "training_available_in_p42",
        "training_executed_in_p42",
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


# Builders
def build_forward_readiness_request_from_defaults() -> FCVAEForwardReadinessRequest:
    return FCVAEForwardReadinessRequest(
        contract_version=FC_VAE_FORWARD_READINESS_GATE_CONTRACT_VERSION,
        gate_kind=FC_VAE_FORWARD_READINESS_GATE_KIND,
        architecture_id=FC_VAE_ARCHITECTURE_ID,
        source_tensor_materialization_contract_version=FC_VAE_TENSOR_MATERIALIZATION_CONTRACT_VERSION,
        source_forward_eligibility_gate_contract_version=FC_VAE_FORWARD_ELIGIBILITY_GATE_CONTRACT_VERSION,
        source_module_availability_shell_contract_version=FC_VAE_MODULE_AVAILABILITY_SHELL_CONTRACT_VERSION,
        expected_batch_size=2,
        expected_input_flat_dim=32,
        expected_shape_tuple=(2, 32),
        expected_tensor_dtype_name="torch.float32",
        expected_tensor_device_type="cpu",
        expected_latent_total_dim=20,
        expected_latent_names=FC_VAE_REQUIRED_LATENT_NAMES,
        allow_forward_execution_in_p42=False,
        allow_output_generation_in_p42=False,
        allow_training_in_p42=False,
        reason="p42_forward_readiness_request_defaults",
    )


def build_forward_readiness_evidence(
    request: FCVAEForwardReadinessRequest
) -> FCVAEForwardReadinessEvidence:
    validate_forward_readiness_request(request)

    # Query dependencies
    torch_status = build_torch_dependency_status(policy=TORCH_POLICY_OPTIONAL)
    p39_res = run_tensor_materialization_probe()
    p40_res = run_forward_eligibility_gate_probe()
    p41_res = run_module_availability_shell_probe()

    torch_available = bool(torch_status.available)
    tensor_materialized_in_p39 = bool(p39_res.tensor_materialized_in_p39)
    tensor_materialization_status = str(p39_res.status)

    if tensor_materialized_in_p39 and p39_res.materialized_tensor is not None:
        t_info = p39_res.materialized_tensor
        tensor_shape_tuple = t_info.tensor_shape_tuple
        tensor_dtype_name = str(t_info.tensor_dtype_name)
        tensor_device_type = str(t_info.tensor_device_type)
        tensor_numel = int(t_info.tensor_numel)
        tensor_requires_grad = bool(t_info.tensor_requires_grad)
        tensor_values_match_p37_nested_values = bool(t_info.tensor_values_match_p37_nested_values)
    else:
        tensor_shape_tuple = (0, 0)
        tensor_dtype_name = ""
        tensor_device_type = ""
        tensor_numel = 0
        tensor_requires_grad = False
        tensor_values_match_p37_nested_values = False

    p40_gate_status = str(p40_res.status)
    p40_shape_matches = bool(p40_res.evidence.shape_matches) if p40_res.evidence is not None else False
    p40_dtype_matches = bool(p40_res.evidence.dtype_matches) if p40_res.evidence is not None else False
    p40_device_matches = bool(p40_res.evidence.device_matches) if p40_res.evidence is not None else False
    p40_latent_layout_matches = bool(p40_res.evidence.latent_layout_matches) if p40_res.evidence is not None else False

    module_shell_created_in_p41 = bool(p41_res.metadata.module_shell_created) if p41_res.metadata is not None else False
    module_shell_status = str(p41_res.status)
    implementation_available_in_p41 = bool(p41_res.implementation_available_in_p41)
    module_shell_available_in_p41 = bool(p41_res.module_shell_available_in_p41)

    if p41_res.metadata is not None:
        m_meta = p41_res.metadata
        is_torch_nn_module = bool(m_meta.is_torch_nn_module)
        module_object_returned = bool(m_meta.module_object_returned)
        defines_forward = bool(m_meta.defines_forward)
        has_own_forward = bool(m_meta.has_own_forward)
        uses_inherited_unimplemented_forward_only = bool(m_meta.uses_inherited_unimplemented_forward_only)
        defines_layers = bool(m_meta.defines_layers)
        parameter_count = int(m_meta.parameter_count)
        buffer_count = int(m_meta.buffer_count)
        module_architecture_id = str(m_meta.architecture_id)
        module_input_flat_dim = int(m_meta.input_flat_dim)
        module_latent_total_dim = int(m_meta.latent_total_dim)
        module_latent_names = tuple(m_meta.latent_names)
    else:
        is_torch_nn_module = False
        module_object_returned = False
        defines_forward = False
        has_own_forward = False
        uses_inherited_unimplemented_forward_only = True
        defines_layers = False
        parameter_count = 0
        buffer_count = 0
        module_architecture_id = ""
        module_input_flat_dim = 0
        module_latent_total_dim = 0
        module_latent_names = ()

    # Compatibility calculations
    tensor_module_shape_compatible = (
        tensor_materialized_in_p39
        and module_shell_created_in_p41
        and tensor_shape_tuple[1] == module_input_flat_dim == request.expected_input_flat_dim
    )

    module_contract_compatible = (
        module_shell_created_in_p41
        and module_architecture_id == request.architecture_id
        and module_input_flat_dim == request.expected_input_flat_dim
        and module_latent_total_dim == request.expected_latent_total_dim
        and module_latent_names == request.expected_latent_names
    )

    # Forward availability is always False in P42 because the shell contains no own forward path
    forward_implementation_available = (
        module_shell_created_in_p41
        and (defines_forward or has_own_forward or not uses_inherited_unimplemented_forward_only)
    )

    evidence = FCVAEForwardReadinessEvidence(
        contract_version=FC_VAE_FORWARD_READINESS_GATE_CONTRACT_VERSION,
        torch_available=torch_available,
        tensor_materialized_in_p39=tensor_materialized_in_p39,
        tensor_materialization_status=tensor_materialization_status,
        tensor_shape_tuple=tensor_shape_tuple,
        tensor_dtype_name=tensor_dtype_name,
        tensor_device_type=tensor_device_type,
        tensor_numel=tensor_numel,
        tensor_requires_grad=tensor_requires_grad,
        tensor_values_match_p37_nested_values=tensor_values_match_p37_nested_values,
        p40_gate_status=p40_gate_status,
        p40_shape_matches=p40_shape_matches,
        p40_dtype_matches=p40_dtype_matches,
        p40_device_matches=p40_device_matches,
        p40_latent_layout_matches=p40_latent_layout_matches,
        module_shell_created_in_p41=module_shell_created_in_p41,
        module_shell_status=module_shell_status,
        implementation_available_in_p41=implementation_available_in_p41,
        module_shell_available_in_p41=module_shell_available_in_p41,
        is_torch_nn_module=is_torch_nn_module,
        module_object_returned=module_object_returned,
        defines_forward=defines_forward,
        has_own_forward=has_own_forward,
        uses_inherited_unimplemented_forward_only=uses_inherited_unimplemented_forward_only,
        defines_layers=defines_layers,
        parameter_count=parameter_count,
        buffer_count=buffer_count,
        module_architecture_id=module_architecture_id,
        module_input_flat_dim=module_input_flat_dim,
        module_latent_total_dim=module_latent_total_dim,
        module_latent_names=module_latent_names,
        tensor_module_shape_compatible=tensor_module_shape_compatible,
        module_contract_compatible=module_contract_compatible,
        forward_implementation_available=forward_implementation_available,
        forward_execution_attempted=False,
        output_generation_attempted=False,
        training_attempted=False,
        reason="p42_forward_readiness_evidence_built",
    )
    validate_forward_readiness_evidence(evidence)
    return evidence


def build_forward_readiness_result(
    request: FCVAEForwardReadinessRequest
) -> FCVAEForwardReadinessResult:
    validate_forward_readiness_request(request)
    evidence = build_forward_readiness_evidence(request)

    # Route status
    if not evidence.torch_available:
        status = FC_VAE_FORWARD_READINESS_STATUS_TORCH_UNAVAILABLE
    elif not evidence.tensor_materialized_in_p39:
        status = FC_VAE_FORWARD_READINESS_STATUS_TENSOR_UNAVAILABLE
    elif not evidence.module_shell_created_in_p41:
        status = FC_VAE_FORWARD_READINESS_STATUS_MODULE_SHELL_UNAVAILABLE
    elif not evidence.module_contract_compatible or not evidence.tensor_module_shape_compatible:
        status = FC_VAE_FORWARD_READINESS_STATUS_CONTRACT_MISMATCH
    elif not evidence.forward_implementation_available:
        status = FC_VAE_FORWARD_READINESS_STATUS_FORWARD_IMPLEMENTATION_UNAVAILABLE
    else:
        status = FC_VAE_FORWARD_READINESS_STATUS_FORWARD_IMPLEMENTATION_UNAVAILABLE

    module_ready = (
        evidence.module_shell_created_in_p41
        and evidence.module_contract_compatible
    )

    res = FCVAEForwardReadinessResult(
        contract_version=FC_VAE_FORWARD_READINESS_GATE_CONTRACT_VERSION,
        request=request,
        evidence=evidence,
        status=status,
        forward_ready_in_p42=False,
        module_shell_ready_in_p42=module_ready,
        implementation_available_in_p41=evidence.implementation_available_in_p41,
        forward_implementation_available_in_p42=evidence.forward_implementation_available,
        forward_available_in_p42=False,
        forward_execution_available_in_p42=False,
        forward_executed_in_p42=False,
        output_generation_available_in_p42=False,
        output_generated_in_p42=False,
        training_available_in_p42=False,
        training_executed_in_p42=False,
        no_forward_execution=True,
        no_output_generation=True,
        no_training_loop=True,
        no_optimizer=True,
        no_checkpointing=True,
        no_artifact_generation=True,
        no_final_comparison=True,
        no_scientific_conclusion=True,
        reason="p42_forward_readiness_result_built",
    )
    validate_forward_readiness_result(res)
    return res


def run_forward_readiness_gate_probe() -> FCVAEForwardReadinessResult:
    req = build_forward_readiness_request_from_defaults()
    res = build_forward_readiness_result(req)
    validate_forward_readiness_result(res)
    return res


# Serialization
def forward_readiness_request_to_json_dict(request: FCVAEForwardReadinessRequest) -> dict:
    validate_forward_readiness_request(request)
    d = {
        "contract_version": request.contract_version,
        "gate_kind": request.gate_kind,
        "architecture_id": request.architecture_id,
        "source_tensor_materialization_contract_version": request.source_tensor_materialization_contract_version,
        "source_forward_eligibility_gate_contract_version": request.source_forward_eligibility_gate_contract_version,
        "source_module_availability_shell_contract_version": request.source_module_availability_shell_contract_version,
        "expected_batch_size": request.expected_batch_size,
        "expected_input_flat_dim": request.expected_input_flat_dim,
        "expected_shape_tuple": list(request.expected_shape_tuple),
        "expected_tensor_dtype_name": request.expected_tensor_dtype_name,
        "expected_tensor_device_type": request.expected_tensor_device_type,
        "expected_latent_total_dim": request.expected_latent_total_dim,
        "expected_latent_names": list(request.expected_latent_names),
        "allow_forward_execution_in_p42": request.allow_forward_execution_in_p42,
        "allow_output_generation_in_p42": request.allow_output_generation_in_p42,
        "allow_training_in_p42": request.allow_training_in_p42,
        "reason": request.reason,
    }
    assert_no_local_path_leakage(request.reason, "reason")
    assert_no_forbidden_claims(request.reason, "reason")
    return d


def forward_readiness_evidence_to_json_dict(evidence: FCVAEForwardReadinessEvidence) -> dict:
    validate_forward_readiness_evidence(evidence)
    d = {
        "contract_version": evidence.contract_version,
        "torch_available": evidence.torch_available,
        "tensor_materialized_in_p39": evidence.tensor_materialized_in_p39,
        "tensor_materialization_status": evidence.tensor_materialization_status,
        "tensor_shape_tuple": list(evidence.tensor_shape_tuple),
        "tensor_dtype_name": evidence.tensor_dtype_name,
        "tensor_device_type": evidence.tensor_device_type,
        "tensor_numel": evidence.tensor_numel,
        "tensor_requires_grad": evidence.tensor_requires_grad,
        "tensor_values_match_p37_nested_values": evidence.tensor_values_match_p37_nested_values,
        "p40_gate_status": evidence.p40_gate_status,
        "p40_shape_matches": evidence.p40_shape_matches,
        "p40_dtype_matches": evidence.p40_dtype_matches,
        "p40_device_matches": evidence.p40_device_matches,
        "p40_latent_layout_matches": evidence.p40_latent_layout_matches,
        "module_shell_created_in_p41": evidence.module_shell_created_in_p41,
        "module_shell_status": evidence.module_shell_status,
        "implementation_available_in_p41": evidence.implementation_available_in_p41,
        "module_shell_available_in_p41": evidence.module_shell_available_in_p41,
        "is_torch_nn_module": evidence.is_torch_nn_module,
        "module_object_returned": evidence.module_object_returned,
        "defines_forward": evidence.defines_forward,
        "has_own_forward": evidence.has_own_forward,
        "uses_inherited_unimplemented_forward_only": evidence.uses_inherited_unimplemented_forward_only,
        "defines_layers": evidence.defines_layers,
        "parameter_count": evidence.parameter_count,
        "buffer_count": evidence.buffer_count,
        "module_architecture_id": evidence.module_architecture_id,
        "module_input_flat_dim": evidence.module_input_flat_dim,
        "module_latent_total_dim": evidence.module_latent_total_dim,
        "module_latent_names": list(evidence.module_latent_names),
        "tensor_module_shape_compatible": evidence.tensor_module_shape_compatible,
        "module_contract_compatible": evidence.module_contract_compatible,
        "forward_implementation_available": evidence.forward_implementation_available,
        "forward_execution_attempted": evidence.forward_execution_attempted,
        "output_generation_attempted": evidence.output_generation_attempted,
        "training_attempted": evidence.training_attempted,
        "reason": evidence.reason,
    }
    assert_no_local_path_leakage(evidence.reason, "reason")
    assert_no_forbidden_claims(evidence.reason, "reason")
    return d


def forward_readiness_result_to_json_dict(result: FCVAEForwardReadinessResult) -> dict:
    validate_forward_readiness_result(result)
    d = {
        "contract_version": result.contract_version,
        "request": forward_readiness_request_to_json_dict(result.request),
        "evidence": forward_readiness_evidence_to_json_dict(result.evidence),
        "status": result.status,
        "forward_ready_in_p42": result.forward_ready_in_p42,
        "module_shell_ready_in_p42": result.module_shell_ready_in_p42,
        "implementation_available_in_p41": result.implementation_available_in_p41,
        "forward_implementation_available_in_p42": result.forward_implementation_available_in_p42,
        "forward_available_in_p42": result.forward_available_in_p42,
        "forward_execution_available_in_p42": result.forward_execution_available_in_p42,
        "forward_executed_in_p42": result.forward_executed_in_p42,
        "output_generation_available_in_p42": result.output_generation_available_in_p42,
        "output_generated_in_p42": result.output_generated_in_p42,
        "training_available_in_p42": result.training_available_in_p42,
        "training_executed_in_p42": result.training_executed_in_p42,
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


def compact_forward_readiness_gate_json(result: FCVAEForwardReadinessResult) -> str:
    d = forward_readiness_result_to_json_dict(result)
    return json.dumps(d, sort_keys=True, separators=(",", ":"))
