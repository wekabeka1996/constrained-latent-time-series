# src/phase2/fc_vae_forward_eligibility_gate.py

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
    build_fc_vae_skeleton_status,
    build_smoke_forward_contract,
    require_fc_vae_implementation_available,
)
from src.phase2.fc_vae_tensor_materialization import (
    FC_VAE_TENSOR_MATERIALIZATION_CONTRACT_VERSION,
    run_tensor_materialization_probe,
)

# Public constants
FC_VAE_FORWARD_ELIGIBILITY_GATE_CONTRACT_VERSION = "phase2_p40_forward_eligibility_gate_contract_v1"

FC_VAE_FORWARD_ELIGIBILITY_GATE_KIND = "forward_eligibility_gate_no_forward_no_output_no_training"

FC_VAE_FORWARD_ELIGIBILITY_GATE_MODULE_NAME = "src.phase2.fc_vae_forward_eligibility_gate"

FC_VAE_FORWARD_ELIGIBILITY_STATUS_TORCH_UNAVAILABLE = "blocked_torch_unavailable"

FC_VAE_FORWARD_ELIGIBILITY_STATUS_TENSOR_UNAVAILABLE = "blocked_by_tensor_materialization_unavailable"

FC_VAE_FORWARD_ELIGIBILITY_STATUS_MODEL_IMPLEMENTATION_UNAVAILABLE = "blocked_by_model_implementation_unavailable"

FC_VAE_FORWARD_ELIGIBILITY_STATUS_ELIGIBLE_BUT_NOT_EXECUTED = "forward_eligible_but_not_executed_in_p40"

SUPPORTED_FC_VAE_FORWARD_ELIGIBILITY_STATUSES = (
    "blocked_torch_unavailable",
    "blocked_by_tensor_materialization_unavailable",
    "blocked_by_model_implementation_unavailable",
    "forward_eligible_but_not_executed_in_p40",
)


# Public dataclasses
@dataclasses.dataclass(frozen=True)
class FCVAEForwardEligibilityRequest:
    contract_version: str
    gate_kind: str
    architecture_id: str
    module_name: str
    source_tensor_materialization_contract_version: str
    source_model_skeleton_contract_version: str
    source_torch_boundary_contract_version: str
    expected_batch_size: int
    expected_input_flat_dim: int
    expected_shape_tuple: Tuple[int, int]
    expected_tensor_dtype_name: str
    expected_tensor_device_type: str
    allow_forward_execution_in_p40: bool
    allow_output_generation_in_p40: bool
    allow_training_in_p40: bool
    reason: str


@dataclasses.dataclass(frozen=True)
class FCVAEForwardEligibilityEvidence:
    contract_version: str
    torch_available: bool
    torch_import_safe: bool
    top_level_torch_import_required: bool
    tensor_materialized_in_p39: bool
    tensor_materialization_status: str
    tensor_dtype_name: str
    tensor_device_type: str
    tensor_shape_tuple: Tuple[int, int]
    tensor_numel: int
    tensor_requires_grad: bool
    tensor_is_floating_point: bool
    tensor_values_match_p37_nested_values: bool
    model_skeleton_status: str
    model_no_implementation: bool
    model_forward_contract_declared: bool
    model_forward_implemented_in_p27: bool
    input_flat_dim_matches: bool
    shape_matches: bool
    dtype_matches: bool
    device_matches: bool
    latent_layout_matches: bool
    decoder_contract_available: bool
    model_implementation_available: bool
    forward_execution_attempted: bool
    output_generation_attempted: bool
    training_attempted: bool
    reason: str


@dataclasses.dataclass(frozen=True)
class FCVAEForwardEligibilityResult:
    contract_version: str
    request: FCVAEForwardEligibilityRequest
    evidence: FCVAEForwardEligibilityEvidence
    status: str
    forward_eligible_in_p40: bool
    forward_execution_available_in_p40: bool
    forward_executed_in_p40: bool
    output_generation_available_in_p40: bool
    output_generated_in_p40: bool
    training_available_in_p40: bool
    training_executed_in_p40: bool
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


def validate_shape_tuple(value: Any, expected: Tuple[int, int], name: str) -> None:
    if type(value) is not tuple:
        raise TypeError(f"{name} must be exact tuple, got {type(value).__name__}")
    if len(value) != 2:
        raise ValueError(f"{name} must have length 2, got {len(value)}")
    for i, v in enumerate(value):
        if type(v) is bool:
            raise TypeError(f"{name}[{i}] must be exact int, got bool")
        if type(v) is not int:
            raise TypeError(f"{name}[{i}] must be exact int, got {type(v).__name__}")
    if value != expected:
        raise ValueError(f"{name} must be {expected}, got {value}")


def validate_forward_eligibility_status(value: Any) -> None:
    if type(value) is not str:
        raise TypeError(f"status must be exact str, got {type(value).__name__}")
    if value not in SUPPORTED_FC_VAE_FORWARD_ELIGIBILITY_STATUSES:
        raise ValueError(f"Unsupported forward eligibility status: {value}")


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


def validate_forward_eligibility_request(request: FCVAEForwardEligibilityRequest) -> None:
    if type(request) is not FCVAEForwardEligibilityRequest:
        raise TypeError("request must be exact FCVAEForwardEligibilityRequest instance")
    validate_non_empty_str(request.contract_version, "contract_version")
    if request.contract_version != FC_VAE_FORWARD_ELIGIBILITY_GATE_CONTRACT_VERSION:
        raise ValueError(f"Wrong contract version: expected {FC_VAE_FORWARD_ELIGIBILITY_GATE_CONTRACT_VERSION}")
    validate_non_empty_str(request.gate_kind, "gate_kind")
    if request.gate_kind != FC_VAE_FORWARD_ELIGIBILITY_GATE_KIND:
        raise ValueError(f"Wrong gate kind: expected {FC_VAE_FORWARD_ELIGIBILITY_GATE_KIND}")
    validate_non_empty_str(request.architecture_id, "architecture_id")
    if request.architecture_id != FC_VAE_ARCHITECTURE_ID:
        raise ValueError(f"Wrong architecture_id: expected {FC_VAE_ARCHITECTURE_ID}")
    validate_non_empty_str(request.module_name, "module_name")
    if request.module_name != FC_VAE_FORWARD_ELIGIBILITY_GATE_MODULE_NAME:
        raise ValueError(f"Wrong module name: expected {FC_VAE_FORWARD_ELIGIBILITY_GATE_MODULE_NAME}")
    validate_non_empty_str(request.source_tensor_materialization_contract_version, "source_tensor_materialization_contract_version")
    if request.source_tensor_materialization_contract_version != FC_VAE_TENSOR_MATERIALIZATION_CONTRACT_VERSION:
        raise ValueError(f"Wrong source tensor contract: expected {FC_VAE_TENSOR_MATERIALIZATION_CONTRACT_VERSION}")
    validate_non_empty_str(request.source_model_skeleton_contract_version, "source_model_skeleton_contract_version")
    if request.source_model_skeleton_contract_version != FC_VAE_MODEL_SKELETON_CONTRACT_VERSION:
        raise ValueError(f"Wrong source skeleton contract: expected {FC_VAE_MODEL_SKELETON_CONTRACT_VERSION}")
    validate_non_empty_str(request.source_torch_boundary_contract_version, "source_torch_boundary_contract_version")
    if request.source_torch_boundary_contract_version != TORCH_BOUNDARY_CONTRACT_VERSION:
        raise ValueError(f"Wrong source torch boundary contract: expected {TORCH_BOUNDARY_CONTRACT_VERSION}")
    validate_positive_int(request.expected_batch_size, "expected_batch_size")
    if request.expected_batch_size != 2:
        raise ValueError(f"expected_batch_size must be 2, got {request.expected_batch_size}")
    validate_positive_int(request.expected_input_flat_dim, "expected_input_flat_dim")
    if request.expected_input_flat_dim != 32:
        raise ValueError(f"expected_input_flat_dim must be 32, got {request.expected_input_flat_dim}")
    validate_shape_tuple(request.expected_shape_tuple, (2, 32), "expected_shape_tuple")
    validate_non_empty_str(request.expected_tensor_dtype_name, "expected_tensor_dtype_name")
    if request.expected_tensor_dtype_name != "torch.float32":
        raise ValueError(f"expected_tensor_dtype_name must be torch.float32, got {request.expected_tensor_dtype_name}")
    validate_non_empty_str(request.expected_tensor_device_type, "expected_tensor_device_type")
    if request.expected_tensor_device_type != "cpu":
        raise ValueError(f"expected_tensor_device_type must be cpu, got {request.expected_tensor_device_type}")
    validate_bool(request.allow_forward_execution_in_p40, "allow_forward_execution_in_p40")
    if request.allow_forward_execution_in_p40:
        raise ValueError("allow_forward_execution_in_p40 must be False")
    validate_bool(request.allow_output_generation_in_p40, "allow_output_generation_in_p40")
    if request.allow_output_generation_in_p40:
        raise ValueError("allow_output_generation_in_p40 must be False")
    validate_bool(request.allow_training_in_p40, "allow_training_in_p40")
    if request.allow_training_in_p40:
        raise ValueError("allow_training_in_p40 must be False")
    validate_non_empty_str(request.reason, "reason")
    assert_no_local_path_leakage(request.reason, "reason")
    assert_no_forbidden_claims(request.reason, "reason")


def validate_forward_eligibility_evidence(evidence: FCVAEForwardEligibilityEvidence) -> None:
    if type(evidence) is not FCVAEForwardEligibilityEvidence:
        raise TypeError("evidence must be exact FCVAEForwardEligibilityEvidence instance")
    validate_non_empty_str(evidence.contract_version, "contract_version")
    if evidence.contract_version != FC_VAE_FORWARD_ELIGIBILITY_GATE_CONTRACT_VERSION:
        raise ValueError(f"Wrong contract version: expected {FC_VAE_FORWARD_ELIGIBILITY_GATE_CONTRACT_VERSION}")
    validate_bool(evidence.torch_available, "torch_available")
    validate_bool(evidence.torch_import_safe, "torch_import_safe")
    validate_bool(evidence.top_level_torch_import_required, "top_level_torch_import_required")
    validate_bool(evidence.tensor_materialized_in_p39, "tensor_materialized_in_p39")
    validate_non_empty_str(evidence.tensor_materialization_status, "tensor_materialization_status")
    
    if evidence.tensor_materialized_in_p39:
        validate_non_empty_str(evidence.tensor_dtype_name, "tensor_dtype_name")
        validate_non_empty_str(evidence.tensor_device_type, "tensor_device_type")
        if type(evidence.tensor_shape_tuple) is not tuple or len(evidence.tensor_shape_tuple) != 2:
            raise TypeError("tensor_shape_tuple must be Tuple[int, int]")
        validate_positive_int(evidence.tensor_numel, "tensor_numel")
        validate_bool(evidence.tensor_requires_grad, "tensor_requires_grad")
        validate_bool(evidence.tensor_is_floating_point, "tensor_is_floating_point")
        validate_bool(evidence.tensor_values_match_p37_nested_values, "tensor_values_match_p37_nested_values")
    
    validate_non_empty_str(evidence.model_skeleton_status, "model_skeleton_status")
    validate_bool(evidence.model_no_implementation, "model_no_implementation")
    validate_bool(evidence.model_forward_contract_declared, "model_forward_contract_declared")
    validate_bool(evidence.model_forward_implemented_in_p27, "model_forward_implemented_in_p27")
    validate_bool(evidence.input_flat_dim_matches, "input_flat_dim_matches")
    validate_bool(evidence.shape_matches, "shape_matches")
    validate_bool(evidence.dtype_matches, "dtype_matches")
    validate_bool(evidence.device_matches, "device_matches")
    validate_bool(evidence.latent_layout_matches, "latent_layout_matches")
    validate_bool(evidence.decoder_contract_available, "decoder_contract_available")
    validate_bool(evidence.model_implementation_available, "model_implementation_available")
    validate_bool(evidence.forward_execution_attempted, "forward_execution_attempted")
    if evidence.forward_execution_attempted:
        raise ValueError("forward_execution_attempted must be False")
    validate_bool(evidence.output_generation_attempted, "output_generation_attempted")
    if evidence.output_generation_attempted:
        raise ValueError("output_generation_attempted must be False")
    validate_bool(evidence.training_attempted, "training_attempted")
    if evidence.training_attempted:
        raise ValueError("training_attempted must be False")
    validate_non_empty_str(evidence.reason, "reason")
    assert_no_local_path_leakage(evidence.reason, "reason")
    assert_no_forbidden_claims(evidence.reason, "reason")


def validate_forward_eligibility_result(result: FCVAEForwardEligibilityResult) -> None:
    if type(result) is not FCVAEForwardEligibilityResult:
        raise TypeError("result must be exact FCVAEForwardEligibilityResult instance")
    validate_non_empty_str(result.contract_version, "contract_version")
    if result.contract_version != FC_VAE_FORWARD_ELIGIBILITY_GATE_CONTRACT_VERSION:
        raise ValueError(f"Wrong contract version: expected {FC_VAE_FORWARD_ELIGIBILITY_GATE_CONTRACT_VERSION}")
    validate_forward_eligibility_request(result.request)
    validate_forward_eligibility_evidence(result.evidence)
    validate_forward_eligibility_status(result.status)
    validate_bool(result.forward_eligible_in_p40, "forward_eligible_in_p40")
    if result.status == FC_VAE_FORWARD_ELIGIBILITY_STATUS_ELIGIBLE_BUT_NOT_EXECUTED:
        if not result.forward_eligible_in_p40:
            raise ValueError("forward_eligible_in_p40 must be True if status is eligible")
    else:
        if result.forward_eligible_in_p40:
            raise ValueError("forward_eligible_in_p40 must be False if status is blocked")
    validate_bool(result.forward_execution_available_in_p40, "forward_execution_available_in_p40")
    if result.forward_execution_available_in_p40:
        raise ValueError("forward_execution_available_in_p40 must be False")
    validate_bool(result.forward_executed_in_p40, "forward_executed_in_p40")
    if result.forward_executed_in_p40:
        raise ValueError("forward_executed_in_p40 must be False")
    validate_bool(result.output_generation_available_in_p40, "output_generation_available_in_p40")
    if result.output_generation_available_in_p40:
        raise ValueError("output_generation_available_in_p40 must be False")
    validate_bool(result.output_generated_in_p40, "output_generated_in_p40")
    if result.output_generated_in_p40:
        raise ValueError("output_generated_in_p40 must be False")
    validate_bool(result.training_available_in_p40, "training_available_in_p40")
    if result.training_available_in_p40:
        raise ValueError("training_available_in_p40 must be False")
    validate_bool(result.training_executed_in_p40, "training_executed_in_p40")
    if result.training_executed_in_p40:
        raise ValueError("training_executed_in_p40 must be False")
        
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


# Public builders
def build_forward_eligibility_request_from_defaults() -> FCVAEForwardEligibilityRequest:
    req = FCVAEForwardEligibilityRequest(
        contract_version=FC_VAE_FORWARD_ELIGIBILITY_GATE_CONTRACT_VERSION,
        gate_kind=FC_VAE_FORWARD_ELIGIBILITY_GATE_KIND,
        architecture_id=FC_VAE_ARCHITECTURE_ID,
        module_name=FC_VAE_FORWARD_ELIGIBILITY_GATE_MODULE_NAME,
        source_tensor_materialization_contract_version=FC_VAE_TENSOR_MATERIALIZATION_CONTRACT_VERSION,
        source_model_skeleton_contract_version=FC_VAE_MODEL_SKELETON_CONTRACT_VERSION,
        source_torch_boundary_contract_version=TORCH_BOUNDARY_CONTRACT_VERSION,
        expected_batch_size=2,
        expected_input_flat_dim=32,
        expected_shape_tuple=(2, 32),
        expected_tensor_dtype_name="torch.float32",
        expected_tensor_device_type="cpu",
        allow_forward_execution_in_p40=False,
        allow_output_generation_in_p40=False,
        allow_training_in_p40=False,
        reason="p40_forward_eligibility_request_from_defaults",
    )
    validate_forward_eligibility_request(req)
    return req


def build_forward_eligibility_evidence(
    request: FCVAEForwardEligibilityRequest
) -> FCVAEForwardEligibilityEvidence:
    validate_forward_eligibility_request(request)
    
    # Query reference modules
    torch_status = build_torch_dependency_status(policy=TORCH_POLICY_OPTIONAL)
    tensor_res = run_tensor_materialization_probe()
    skeleton = build_fc_vae_skeleton_status()
    forward_contract = build_smoke_forward_contract()
    
    tensor_mat = bool(tensor_res.tensor_materialized_in_p39)
    if tensor_mat and tensor_res.materialized_tensor is not None:
        t_info = tensor_res.materialized_tensor
        tensor_dtype_name = str(t_info.tensor_dtype_name)
        tensor_device_type = str(t_info.tensor_device_type)
        tensor_shape_tuple = t_info.tensor_shape_tuple
        tensor_numel = int(t_info.tensor_numel)
        tensor_requires_grad = bool(t_info.tensor_requires_grad)
        tensor_is_floating_point = bool(t_info.tensor_is_floating_point)
        tensor_values_match = bool(t_info.tensor_values_match_p37_nested_values)
    else:
        tensor_dtype_name = ""
        tensor_device_type = ""
        tensor_shape_tuple = (0, 0)
        tensor_numel = 0
        tensor_requires_grad = False
        tensor_is_floating_point = False
        tensor_values_match = False
        
    # Match tests
    input_flat_dim_matches = False
    shape_matches = False
    dtype_matches = False
    device_matches = False
    if tensor_mat:
        input_flat_dim_matches = (
            tensor_shape_tuple[1] == request.expected_input_flat_dim == forward_contract.input_shape.flat_dim
        )
        shape_matches = (tensor_shape_tuple == request.expected_shape_tuple)
        dtype_matches = (tensor_dtype_name == request.expected_tensor_dtype_name)
        device_matches = (tensor_device_type == request.expected_tensor_device_type)
        
    latent_layout_matches = (
        forward_contract.latent_layout.z_mean_dim == 8
        and forward_contract.latent_layout.z_volatility_dim == 8
        and forward_contract.latent_layout.z_shared_dim == 4
        and forward_contract.latent_layout.total_latent_dim == 20
        and forward_contract.latent_layout.latent_names == FC_VAE_REQUIRED_LATENT_NAMES
    )
    
    decoder_contract_available = (
        forward_contract.decoder_output is not None
        and forward_contract.decoder_output.output_kind == "typed_modelspec_candidate"
        and forward_contract.decoder_output.family_head_dim == 3
        and forward_contract.decoder_output.mean_head_dim == 3
        and forward_contract.decoder_output.volatility_head_dim == 3
        and forward_contract.decoder_output.diagnostic_head_dim == 4
        and forward_contract.decoder_output.target_boundary == "ModelSpec"
    )
    
    model_implementation_available = True
    try:
        require_fc_vae_implementation_available(skeleton)
    except NotImplementedError:
        model_implementation_available = False
    except Exception:
        model_implementation_available = False
        
    evidence = FCVAEForwardEligibilityEvidence(
        contract_version=FC_VAE_FORWARD_ELIGIBILITY_GATE_CONTRACT_VERSION,
        torch_available=bool(torch_status.available),
        torch_import_safe=bool(torch_status.import_safe),
        top_level_torch_import_required=bool(torch_status.top_level_import_required),
        tensor_materialized_in_p39=tensor_mat,
        tensor_materialization_status=str(tensor_res.status),
        tensor_dtype_name=tensor_dtype_name,
        tensor_device_type=tensor_device_type,
        tensor_shape_tuple=tensor_shape_tuple,
        tensor_numel=tensor_numel,
        tensor_requires_grad=tensor_requires_grad,
        tensor_is_floating_point=tensor_is_floating_point,
        tensor_values_match_p37_nested_values=tensor_values_match,
        model_skeleton_status=str(skeleton.status),
        model_no_implementation=bool(skeleton.no_model_implementation),
        model_forward_contract_declared=True,
        model_forward_implemented_in_p27=bool(forward_contract.implemented_in_p27),
        input_flat_dim_matches=input_flat_dim_matches,
        shape_matches=shape_matches,
        dtype_matches=dtype_matches,
        device_matches=device_matches,
        latent_layout_matches=latent_layout_matches,
        decoder_contract_available=decoder_contract_available,
        model_implementation_available=model_implementation_available,
        forward_execution_attempted=False,
        output_generation_attempted=False,
        training_attempted=False,
        reason="p40_forward_eligibility_evidence_built",
    )
    validate_forward_eligibility_evidence(evidence)
    return evidence


def build_forward_eligibility_result(
    request: FCVAEForwardEligibilityRequest
) -> FCVAEForwardEligibilityResult:
    validate_forward_eligibility_request(request)
    evidence = build_forward_eligibility_evidence(request)
    
    # Status routing logic
    if not evidence.torch_available:
        status = FC_VAE_FORWARD_ELIGIBILITY_STATUS_TORCH_UNAVAILABLE
    elif not evidence.tensor_materialized_in_p39:
        status = FC_VAE_FORWARD_ELIGIBILITY_STATUS_TENSOR_UNAVAILABLE
    elif not evidence.model_implementation_available:
        status = FC_VAE_FORWARD_ELIGIBILITY_STATUS_MODEL_IMPLEMENTATION_UNAVAILABLE
    else:
        status = FC_VAE_FORWARD_ELIGIBILITY_STATUS_ELIGIBLE_BUT_NOT_EXECUTED
        
    forward_eligible = (status == FC_VAE_FORWARD_ELIGIBILITY_STATUS_ELIGIBLE_BUT_NOT_EXECUTED)
    
    result = FCVAEForwardEligibilityResult(
        contract_version=FC_VAE_FORWARD_ELIGIBILITY_GATE_CONTRACT_VERSION,
        request=request,
        evidence=evidence,
        status=status,
        forward_eligible_in_p40=forward_eligible,
        forward_execution_available_in_p40=False,
        forward_executed_in_p40=False,
        output_generation_available_in_p40=False,
        output_generated_in_p40=False,
        training_available_in_p40=False,
        training_executed_in_p40=False,
        no_forward_execution=True,
        no_output_generation=True,
        no_training_loop=True,
        no_optimizer=True,
        no_checkpointing=True,
        no_artifact_generation=True,
        no_final_comparison=True,
        no_scientific_conclusion=True,
        reason="p40_forward_eligibility_result_built",
    )
    validate_forward_eligibility_result(result)
    return result


def run_forward_eligibility_gate_probe() -> FCVAEForwardEligibilityResult:
    req = build_forward_eligibility_request_from_defaults()
    res = build_forward_eligibility_result(req)
    validate_forward_eligibility_result(res)
    return res


# Serialization functions
def forward_eligibility_request_to_json_dict(request: FCVAEForwardEligibilityRequest) -> dict:
    validate_forward_eligibility_request(request)
    return {
        "contract_version": request.contract_version,
        "gate_kind": request.gate_kind,
        "architecture_id": request.architecture_id,
        "module_name": request.module_name,
        "source_tensor_materialization_contract_version": request.source_tensor_materialization_contract_version,
        "source_model_skeleton_contract_version": request.source_model_skeleton_contract_version,
        "source_torch_boundary_contract_version": request.source_torch_boundary_contract_version,
        "expected_batch_size": request.expected_batch_size,
        "expected_input_flat_dim": request.expected_input_flat_dim,
        "expected_shape_tuple": list(request.expected_shape_tuple),
        "expected_tensor_dtype_name": request.expected_tensor_dtype_name,
        "expected_tensor_device_type": request.expected_tensor_device_type,
        "allow_forward_execution_in_p40": request.allow_forward_execution_in_p40,
        "allow_output_generation_in_p40": request.allow_output_generation_in_p40,
        "allow_training_in_p40": request.allow_training_in_p40,
        "reason": request.reason,
    }


def forward_eligibility_evidence_to_json_dict(evidence: FCVAEForwardEligibilityEvidence) -> dict:
    validate_forward_eligibility_evidence(evidence)
    return {
        "contract_version": evidence.contract_version,
        "torch_available": evidence.torch_available,
        "torch_import_safe": evidence.torch_import_safe,
        "top_level_torch_import_required": evidence.top_level_torch_import_required,
        "tensor_materialized_in_p39": evidence.tensor_materialized_in_p39,
        "tensor_materialization_status": evidence.tensor_materialization_status,
        "tensor_dtype_name": evidence.tensor_dtype_name,
        "tensor_device_type": evidence.tensor_device_type,
        "tensor_shape_tuple": list(evidence.tensor_shape_tuple) if evidence.tensor_shape_tuple else [0, 0],
        "tensor_numel": evidence.tensor_numel,
        "tensor_requires_grad": evidence.tensor_requires_grad,
        "tensor_is_floating_point": evidence.tensor_is_floating_point,
        "tensor_values_match_p37_nested_values": evidence.tensor_values_match_p37_nested_values,
        "model_skeleton_status": evidence.model_skeleton_status,
        "model_no_implementation": evidence.model_no_implementation,
        "model_forward_contract_declared": evidence.model_forward_contract_declared,
        "model_forward_implemented_in_p27": evidence.model_forward_implemented_in_p27,
        "input_flat_dim_matches": evidence.input_flat_dim_matches,
        "shape_matches": evidence.shape_matches,
        "dtype_matches": evidence.dtype_matches,
        "device_matches": evidence.device_matches,
        "latent_layout_matches": evidence.latent_layout_matches,
        "decoder_contract_available": evidence.decoder_contract_available,
        "model_implementation_available": evidence.model_implementation_available,
        "forward_execution_attempted": evidence.forward_execution_attempted,
        "output_generation_attempted": evidence.output_generation_attempted,
        "training_attempted": evidence.training_attempted,
        "reason": evidence.reason,
    }


def forward_eligibility_result_to_json_dict(result: FCVAEForwardEligibilityResult) -> dict:
    validate_forward_eligibility_result(result)
    return {
        "contract_version": result.contract_version,
        "request": forward_eligibility_request_to_json_dict(result.request),
        "evidence": forward_eligibility_evidence_to_json_dict(result.evidence),
        "status": result.status,
        "forward_eligible_in_p40": result.forward_eligible_in_p40,
        "forward_execution_available_in_p40": result.forward_execution_available_in_p40,
        "forward_executed_in_p40": result.forward_executed_in_p40,
        "output_generation_available_in_p40": result.output_generation_available_in_p40,
        "output_generated_in_p40": result.output_generated_in_p40,
        "training_available_in_p40": result.training_available_in_p40,
        "training_executed_in_p40": result.training_executed_in_p40,
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


def compact_forward_eligibility_gate_json(result: FCVAEForwardEligibilityResult) -> str:
    return json.dumps(
        forward_eligibility_result_to_json_dict(result),
        sort_keys=True,
        separators=(",", ":"),
    )
