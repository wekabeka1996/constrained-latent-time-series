# src/phase2/fc_vae_tensor_materialization.py

import dataclasses
import json
from typing import Any, Dict, Tuple

from src.phase2.torch_boundary import (
    TORCH_BOUNDARY_CONTRACT_VERSION,
    TORCH_POLICY_OPTIONAL,
    build_torch_dependency_status,
)
from src.phase2.fc_vae_nested_batch_values import (
    FC_VAE_NESTED_BATCH_VALUES_CONTRACT_VERSION,
    run_nested_batch_values_probe,
)
from src.phase2.fc_vae_tensor_materialization_request import (
    FC_VAE_TENSOR_MATERIALIZATION_REQUEST_CONTRACT_VERSION,
    run_tensor_materialization_request_probe,
)

# Constants
FC_VAE_TENSOR_MATERIALIZATION_CONTRACT_VERSION = "phase2_p39_torch_tensor_materialization_contract_v1"

FC_VAE_TENSOR_MATERIALIZATION_KIND = "optional_torch_cpu_float32_tensor_materialization_no_forward_no_training"

FC_VAE_TENSOR_MATERIALIZATION_MODULE_NAME = "src.phase2.fc_vae_tensor_materialization"

FC_VAE_TENSOR_MATERIALIZATION_STATUS_TORCH_UNAVAILABLE = "blocked_torch_unavailable"

FC_VAE_TENSOR_MATERIALIZATION_STATUS_REQUEST_BLOCKED = "blocked_by_tensor_request_status"

FC_VAE_TENSOR_MATERIALIZATION_STATUS_MATERIALIZED = "torch_tensor_materialized_no_forward_no_training_in_p39"

SUPPORTED_FC_VAE_TENSOR_MATERIALIZATION_STATUSES = (
    "blocked_torch_unavailable",
    "blocked_by_tensor_request_status",
    "torch_tensor_materialized_no_forward_no_training_in_p39",
)

FC_VAE_TENSOR_MATERIALIZATION_TARGET_FRAMEWORK = "torch"
FC_VAE_TENSOR_MATERIALIZATION_TARGET_DTYPE = "float32"
FC_VAE_TENSOR_MATERIALIZATION_TARGET_DEVICE = "cpu"
FC_VAE_TENSOR_MATERIALIZATION_LAYOUT_KIND = "row_major_2d_batch_tensor"

DEFAULT_P39_BATCH_SIZE = 2
DEFAULT_P39_INPUT_FLAT_DIM = 32
DEFAULT_P39_FULL_VECTOR_LENGTH = 64
DEFAULT_P39_SHAPE_TUPLE = (2, 32)
DEFAULT_P39_REQUIRES_GRAD = False


# Dataclasses
@dataclasses.dataclass(frozen=True)
class FCVAETensorMaterializationRequest:
    contract_version: str
    materialization_kind: str
    architecture_id: str
    source_nested_values_contract_version: str
    source_tensor_request_contract_version: str
    source_torch_boundary_contract_version: str
    target_framework: str
    target_dtype: str
    target_device: str
    target_layout_kind: str
    batch_size: int
    input_flat_dim: int
    full_vector_length: int
    shape_tuple: Tuple[int, int]
    requires_grad: bool
    allow_tensor_materialization_in_p39: bool
    allow_forward_execution_in_p39: bool
    allow_output_generation_in_p39: bool
    allow_training_in_p39: bool
    reason: str


@dataclasses.dataclass(frozen=True)
class FCVAEMaterializedTorchTensor:
    contract_version: str
    materialization_kind: str
    target_framework: str
    target_dtype: str
    target_device: str
    target_layout_kind: str
    shape_tuple: Tuple[int, int]
    batch_size: int
    input_flat_dim: int
    full_vector_length: int
    tensor_object: Any
    tensor_type_name: str
    tensor_dtype_name: str
    tensor_device_type: str
    tensor_shape_tuple: Tuple[int, int]
    tensor_numel: int
    tensor_requires_grad: bool
    tensor_is_floating_point: bool
    tensor_values_match_p37_nested_values: bool
    first_row_first_4_values: Tuple[float, ...]
    second_row_first_4_values: Tuple[float, ...]
    forward_executed: bool
    output_generated: bool
    training_executed: bool
    reason: str


@dataclasses.dataclass(frozen=True)
class FCVAETensorMaterializationMetadata:
    contract_version: str
    source_nested_values_contract_version: str
    source_tensor_request_contract_version: str
    source_torch_boundary_contract_version: str
    torch_available: bool
    torch_import_safe: bool
    top_level_torch_import_required: bool
    torch_version: str
    cuda_available: bool
    nested_values_status: str
    tensor_request_status: str
    nested_values_available_in_p37: bool
    tensor_request_available_in_p38: bool
    tensor_materialization_attempted: bool
    tensor_materialized: bool
    forward_execution_attempted: bool
    output_generation_attempted: bool
    training_attempted: bool
    reason: str


@dataclasses.dataclass(frozen=True)
class FCVAETensorMaterializationResult:
    contract_version: str
    request: FCVAETensorMaterializationRequest
    materialized_tensor: FCVAEMaterializedTorchTensor | None
    metadata: FCVAETensorMaterializationMetadata
    status: str
    tensor_materialization_available_in_p39: bool
    tensor_materialized_in_p39: bool
    forward_execution_available_in_p39: bool
    output_generation_available_in_p39: bool
    training_available_in_p39: bool
    no_forward_execution: bool
    no_output_generation: bool
    no_training_loop: bool
    no_optimizer: bool
    no_checkpointing: bool
    no_artifact_generation: bool
    no_final_comparison: bool
    no_scientific_conclusion: bool
    reason: str


# Helpers
def load_torch_for_p39_materialization() -> Any:
    status = build_torch_dependency_status(policy=TORCH_POLICY_OPTIONAL)
    if not status.available:
        raise RuntimeError("torch is unavailable according to P26 status")
    
    # local guarded import
    import torch
    return torch


def normalize_torch_dtype_name(dtype: Any) -> str:
    s = str(dtype)
    if "float32" in s or "Float" in s:
        return "torch.float32"
    return s


def tensor_to_nested_tuple_preview(tensor: Any) -> Tuple[Tuple[float, ...], Tuple[float, ...]]:
    lst = tensor.detach().cpu().tolist()
    if not isinstance(lst, list) or len(lst) != 2:
        raise ValueError("Tensor list representation must have length 2")
    return (tuple(float(x) for x in lst[0]), tuple(float(x) for x in lst[1]))


def build_torch_tensor_from_p37_nested_values(
    torch_module: Any,
    nested_values: Tuple[Tuple[float, ...], Tuple[float, ...]],
    dtype_name: str,
    device_name: str,
    requires_grad: bool,
) -> Any:
    # Target CPU Float32 Tensor only in P39
    if device_name != "cpu":
        raise ValueError("Only cpu device is supported in P39")
    if dtype_name != "float32":
        raise ValueError("Only float32 dtype is supported in P39")
        
    t = torch_module.tensor(
        nested_values,
        dtype=torch_module.float32,
        device="cpu"
    )
    if t.requires_grad != requires_grad:
        # Should be False by default, but set explicitly if needed
        t.requires_grad_(requires_grad)
    return t


# Validators
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


def validate_shape_tuple(value: Any) -> None:
    if type(value) is not tuple:
        raise TypeError(f"shape_tuple must be exact tuple, got {type(value).__name__}")
    if len(value) != 2:
        raise ValueError(f"shape_tuple must have length 2, got {len(value)}")
    for i, item in enumerate(value):
        if type(item) is not int or isinstance(item, bool):
            raise TypeError(f"shape_tuple[{i}] must be exact int, got {type(item).__name__}")
        if item <= 0:
            raise ValueError(f"shape_tuple[{i}] must be > 0")
    if value != (2, 32):
        raise ValueError(f"shape_tuple must be exactly (2, 32), got {value}")


def validate_full_vector_length(value: Any) -> None:
    if type(value) is not int or isinstance(value, bool):
        raise TypeError(f"full_vector_length must be exact int, got {type(value).__name__}")
    if value != 64:
        raise ValueError(f"full_vector_length must be exactly 64, got {value}")


def validate_materialization_status(value: Any) -> None:
    if type(value) is not str:
        raise TypeError("status must be exact str")
    if value not in SUPPORTED_FC_VAE_TENSOR_MATERIALIZATION_STATUSES:
        raise ValueError(f"Unsupported status: {value}")


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
                   .replace("torch_tensor_materialized_no_forward_no_training_in_p39", ""))
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


def validate_p39_request(request: FCVAETensorMaterializationRequest) -> None:
    if type(request) is not FCVAETensorMaterializationRequest:
        raise TypeError("request must be exact FCVAETensorMaterializationRequest instance")
    validate_non_empty_str(request.contract_version, "contract_version")
    if request.contract_version != FC_VAE_TENSOR_MATERIALIZATION_CONTRACT_VERSION:
        raise ValueError(f"Invalid contract_version: {request.contract_version}")
    validate_non_empty_str(request.materialization_kind, "materialization_kind")
    if request.materialization_kind != FC_VAE_TENSOR_MATERIALIZATION_KIND:
        raise ValueError(f"Invalid materialization_kind: {request.materialization_kind}")
    validate_non_empty_str(request.architecture_id, "architecture_id")
    if request.architecture_id != "FC-VAE":
        raise ValueError(f"Invalid architecture_id: {request.architecture_id}")
    validate_non_empty_str(request.source_nested_values_contract_version, "source_nested_values_contract_version")
    if request.source_nested_values_contract_version != FC_VAE_NESTED_BATCH_VALUES_CONTRACT_VERSION:
        raise ValueError(f"Invalid source_nested_values_contract_version: {request.source_nested_values_contract_version}")
    validate_non_empty_str(request.source_tensor_request_contract_version, "source_tensor_request_contract_version")
    if request.source_tensor_request_contract_version != FC_VAE_TENSOR_MATERIALIZATION_REQUEST_CONTRACT_VERSION:
        raise ValueError(f"Invalid source_tensor_request_contract_version: {request.source_tensor_request_contract_version}")
    validate_non_empty_str(request.source_torch_boundary_contract_version, "source_torch_boundary_contract_version")
    if request.source_torch_boundary_contract_version != TORCH_BOUNDARY_CONTRACT_VERSION:
        raise ValueError(f"Invalid source_torch_boundary_contract_version: {request.source_torch_boundary_contract_version}")

    if request.target_framework != FC_VAE_TENSOR_MATERIALIZATION_TARGET_FRAMEWORK:
        raise ValueError("Invalid target_framework")
    if request.target_dtype != FC_VAE_TENSOR_MATERIALIZATION_TARGET_DTYPE:
        raise ValueError("Invalid target_dtype")
    if request.target_device != FC_VAE_TENSOR_MATERIALIZATION_TARGET_DEVICE:
        raise ValueError("Invalid target_device")
    if request.target_layout_kind != FC_VAE_TENSOR_MATERIALIZATION_LAYOUT_KIND:
        raise ValueError("Invalid target_layout_kind")

    validate_positive_int(request.batch_size, "batch_size")
    if request.batch_size != 2:
        raise ValueError("batch_size must be 2")
    validate_positive_int(request.input_flat_dim, "input_flat_dim")
    if request.input_flat_dim != 32:
        raise ValueError("input_flat_dim must be 32")
    validate_full_vector_length(request.full_vector_length)
    validate_shape_tuple(request.shape_tuple)

    validate_bool(request.requires_grad, "requires_grad")
    if request.requires_grad:
        raise ValueError("requires_grad must be False")

    validate_bool(request.allow_tensor_materialization_in_p39, "allow_tensor_materialization_in_p39")
    if not request.allow_tensor_materialization_in_p39:
        raise ValueError("allow_tensor_materialization_in_p39 must be True")

    for flag_name in (
        "allow_forward_execution_in_p39",
        "allow_output_generation_in_p39",
        "allow_training_in_p39",
    ):
        val = getattr(request, flag_name)
        validate_bool(val, flag_name)
        if val:
            raise ValueError(f"{flag_name} must be False")

    validate_non_empty_str(request.reason, "reason")
    assert_no_local_path_leakage(request.reason)
    assert_no_forbidden_claims(request.reason)


def validate_materialized_torch_tensor(tensor: FCVAEMaterializedTorchTensor) -> None:
    if type(tensor) is not FCVAEMaterializedTorchTensor:
        raise TypeError("tensor must be exact FCVAEMaterializedTorchTensor instance")
    validate_non_empty_str(tensor.contract_version, "contract_version")
    if tensor.contract_version != FC_VAE_TENSOR_MATERIALIZATION_CONTRACT_VERSION:
        raise ValueError("Invalid contract_version")
    validate_non_empty_str(tensor.materialization_kind, "materialization_kind")
    if tensor.materialization_kind != FC_VAE_TENSOR_MATERIALIZATION_KIND:
        raise ValueError("Invalid materialization_kind")

    if tensor.target_framework != FC_VAE_TENSOR_MATERIALIZATION_TARGET_FRAMEWORK:
        raise ValueError("Invalid target_framework")
    if tensor.target_dtype != FC_VAE_TENSOR_MATERIALIZATION_TARGET_DTYPE:
        raise ValueError("Invalid target_dtype")
    if tensor.target_device != FC_VAE_TENSOR_MATERIALIZATION_TARGET_DEVICE:
        raise ValueError("Invalid target_device")
    if tensor.target_layout_kind != FC_VAE_TENSOR_MATERIALIZATION_LAYOUT_KIND:
        raise ValueError("Invalid target_layout_kind")

    validate_shape_tuple(tensor.shape_tuple)
    validate_positive_int(tensor.batch_size, "batch_size")
    if tensor.batch_size != 2:
        raise ValueError("batch_size must be 2")
    validate_positive_int(tensor.input_flat_dim, "input_flat_dim")
    if tensor.input_flat_dim != 32:
        raise ValueError("input_flat_dim must be 32")
    validate_full_vector_length(tensor.full_vector_length)

    # Validate actual tensor object
    if tensor.tensor_object is None:
        raise ValueError("tensor_object cannot be None")
    
    validate_non_empty_str(tensor.tensor_type_name, "tensor_type_name")
    if "Tensor" not in tensor.tensor_type_name:
        raise ValueError("tensor_type_name must include Tensor")

    validate_non_empty_str(tensor.tensor_dtype_name, "tensor_dtype_name")
    if "float32" not in tensor.tensor_dtype_name:
        raise ValueError("tensor_dtype_name must represent float32")

    validate_non_empty_str(tensor.tensor_device_type, "tensor_device_type")
    if tensor.tensor_device_type != "cpu":
        raise ValueError("tensor_device_type must be cpu")

    validate_shape_tuple(tensor.tensor_shape_tuple)
    validate_positive_int(tensor.tensor_numel, "tensor_numel")
    if tensor.tensor_numel != 64:
        raise ValueError("tensor_numel must be 64")

    validate_bool(tensor.tensor_requires_grad, "tensor_requires_grad")
    if tensor.tensor_requires_grad:
        raise ValueError("tensor_requires_grad must be False")

    validate_bool(tensor.tensor_is_floating_point, "tensor_is_floating_point")
    if not tensor.tensor_is_floating_point:
        raise ValueError("tensor_is_floating_point must be True")

    validate_bool(tensor.tensor_values_match_p37_nested_values, "tensor_values_match_p37_nested_values")
    if not tensor.tensor_values_match_p37_nested_values:
        raise ValueError("tensor_values_match_p37_nested_values must be True")

    # Values validation
    if type(tensor.first_row_first_4_values) is not tuple or len(tensor.first_row_first_4_values) != 4:
        raise TypeError("first_row_first_4_values must be exact tuple of length 4")
    if type(tensor.second_row_first_4_values) is not tuple or len(tensor.second_row_first_4_values) != 4:
        raise TypeError("second_row_first_4_values must be exact tuple of length 4")

    for flag_name in ("forward_executed", "output_generated", "training_executed"):
        val = getattr(tensor, flag_name)
        validate_bool(val, flag_name)
        if val:
            raise ValueError(f"{flag_name} must be False")

    validate_non_empty_str(tensor.reason, "reason")
    assert_no_local_path_leakage(tensor.reason)
    assert_no_forbidden_claims(tensor.reason)


def validate_p39_metadata(metadata: FCVAETensorMaterializationMetadata) -> None:
    if type(metadata) is not FCVAETensorMaterializationMetadata:
        raise TypeError("metadata must be exact FCVAETensorMaterializationMetadata instance")
    validate_non_empty_str(metadata.contract_version, "contract_version")
    if metadata.contract_version != FC_VAE_TENSOR_MATERIALIZATION_CONTRACT_VERSION:
        raise ValueError("Invalid contract_version")
    validate_non_empty_str(metadata.source_nested_values_contract_version, "source_nested_values_contract_version")
    if metadata.source_nested_values_contract_version != FC_VAE_NESTED_BATCH_VALUES_CONTRACT_VERSION:
        raise ValueError("Invalid source_nested_values_contract_version")
    validate_non_empty_str(metadata.source_tensor_request_contract_version, "source_tensor_request_contract_version")
    if metadata.source_tensor_request_contract_version != FC_VAE_TENSOR_MATERIALIZATION_REQUEST_CONTRACT_VERSION:
        raise ValueError("Invalid source_tensor_request_contract_version")
    validate_non_empty_str(metadata.source_torch_boundary_contract_version, "source_torch_boundary_contract_version")
    if metadata.source_torch_boundary_contract_version != TORCH_BOUNDARY_CONTRACT_VERSION:
        raise ValueError("Invalid source_torch_boundary_contract_version")

    validate_bool(metadata.torch_available, "torch_available")
    validate_bool(metadata.torch_import_safe, "torch_import_safe")
    validate_bool(metadata.top_level_torch_import_required, "top_level_torch_import_required")
    if metadata.top_level_torch_import_required:
        raise ValueError("top_level_torch_import_required must be False")

    if metadata.torch_available:
        validate_non_empty_str(metadata.torch_version, "torch_version")
        validate_bool(metadata.cuda_available, "cuda_available")

    validate_non_empty_str(metadata.nested_values_status, "nested_values_status")
    validate_non_empty_str(metadata.tensor_request_status, "tensor_request_status")
    validate_bool(metadata.nested_values_available_in_p37, "nested_values_available_in_p37")
    validate_bool(metadata.tensor_request_available_in_p38, "tensor_request_available_in_p38")
    validate_bool(metadata.tensor_materialization_attempted, "tensor_materialization_attempted")
    validate_bool(metadata.tensor_materialized, "tensor_materialized")

    for flag_name in (
        "forward_execution_attempted",
        "output_generation_attempted",
        "training_attempted",
    ):
        val = getattr(metadata, flag_name)
        validate_bool(val, flag_name)
        if val:
            raise ValueError(f"{flag_name} must be False")

    validate_non_empty_str(metadata.reason, "reason")
    assert_no_local_path_leakage(metadata.reason)
    assert_no_forbidden_claims(metadata.reason)


def validate_p39_result(result: FCVAETensorMaterializationResult) -> None:
    if type(result) is not FCVAETensorMaterializationResult:
        raise TypeError("result must be exact FCVAETensorMaterializationResult instance")
    validate_non_empty_str(result.contract_version, "contract_version")
    if result.contract_version != FC_VAE_TENSOR_MATERIALIZATION_CONTRACT_VERSION:
        raise ValueError("Invalid contract_version")

    validate_p39_request(result.request)
    validate_p39_metadata(result.metadata)
    validate_materialization_status(result.status)

    if result.materialized_tensor is not None:
        validate_materialized_torch_tensor(result.materialized_tensor)
        if result.status != FC_VAE_TENSOR_MATERIALIZATION_STATUS_MATERIALIZED:
            raise ValueError("Status mismatch: must be materialized")
        if not result.metadata.tensor_materialized:
            raise ValueError("Metadata shows not materialized but tensor object exists")
    else:
        if result.status == FC_VAE_TENSOR_MATERIALIZATION_STATUS_MATERIALIZED:
            raise ValueError("Status cannot be materialized if tensor object is None")
        if result.metadata.tensor_materialized:
            raise ValueError("Metadata shows materialized but tensor object is None")

    validate_bool(result.tensor_materialization_available_in_p39, "tensor_materialization_available_in_p39")
    validate_bool(result.tensor_materialized_in_p39, "tensor_materialized_in_p39")

    if result.status == FC_VAE_TENSOR_MATERIALIZATION_STATUS_MATERIALIZED:
        if not result.tensor_materialization_available_in_p39:
            raise ValueError("tensor_materialization_available_in_p39 must be True when status is materialized")
        if not result.tensor_materialized_in_p39:
            raise ValueError("tensor_materialized_in_p39 must be True when status is materialized")
    else:
        if result.tensor_materialization_available_in_p39:
            raise ValueError("tensor_materialization_available_in_p39 must be False when status is not materialized")
        if result.tensor_materialized_in_p39:
            raise ValueError("tensor_materialized_in_p39 must be False when status is not materialized")

    for flag_name in (
        "forward_execution_available_in_p39",
        "output_generation_available_in_p39",
        "training_available_in_p39",
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
    assert_no_local_path_leakage(result.reason)
    assert_no_forbidden_claims(result.reason)


# Builders
def build_tensor_materialization_request_from_p38_default() -> FCVAETensorMaterializationRequest:
    p37_res = run_nested_batch_values_probe()
    p38_res = run_tensor_materialization_request_probe()

    req = FCVAETensorMaterializationRequest(
        contract_version=FC_VAE_TENSOR_MATERIALIZATION_CONTRACT_VERSION,
        materialization_kind=FC_VAE_TENSOR_MATERIALIZATION_KIND,
        architecture_id="FC-VAE",
        source_nested_values_contract_version=p37_res.contract_version,
        source_tensor_request_contract_version=p38_res.contract_version,
        source_torch_boundary_contract_version=TORCH_BOUNDARY_CONTRACT_VERSION,
        target_framework=FC_VAE_TENSOR_MATERIALIZATION_TARGET_FRAMEWORK,
        target_dtype=FC_VAE_TENSOR_MATERIALIZATION_TARGET_DTYPE,
        target_device=FC_VAE_TENSOR_MATERIALIZATION_TARGET_DEVICE,
        target_layout_kind=FC_VAE_TENSOR_MATERIALIZATION_LAYOUT_KIND,
        batch_size=p38_res.request.batch_size,
        input_flat_dim=p38_res.request.input_flat_dim,
        full_vector_length=p38_res.request.full_vector_length,
        shape_tuple=p38_res.request.shape_tuple,
        requires_grad=p38_res.request.requires_grad,
        allow_tensor_materialization_in_p39=True,
        allow_forward_execution_in_p39=False,
        allow_output_generation_in_p39=False,
        allow_training_in_p39=False,
        reason="p39_tensor_materialization_request_from_p38_default",
    )
    validate_p39_request(req)
    return req


def build_materialized_torch_tensor(
    request: FCVAETensorMaterializationRequest,
) -> FCVAEMaterializedTorchTensor:
    validate_p39_request(request)
    p37_res = run_nested_batch_values_probe()
    
    torch_module = load_torch_for_p39_materialization()
    t = build_torch_tensor_from_p37_nested_values(
        torch_module=torch_module,
        nested_values=p37_res.nested_values.nested_batch_values,
        dtype_name=request.target_dtype,
        device_name=request.target_device,
        requires_grad=request.requires_grad
    )

    t_type_name = str(type(t))
    t_dtype_name = normalize_torch_dtype_name(t.dtype)
    t_device_type = str(t.device.type)
    t_shape_tuple = tuple(t.shape)
    t_numel = t.numel()
    t_requires_grad = t.requires_grad
    t_is_floating_point = bool(t.is_floating_point())

    # Preview conversion
    preview = tensor_to_nested_tuple_preview(t)
    values_match = (preview == p37_res.nested_values.nested_batch_values)

    first_row = preview[0][:4]
    second_row = preview[1][:4]

    materialized = FCVAEMaterializedTorchTensor(
        contract_version=FC_VAE_TENSOR_MATERIALIZATION_CONTRACT_VERSION,
        materialization_kind=FC_VAE_TENSOR_MATERIALIZATION_KIND,
        target_framework=request.target_framework,
        target_dtype=request.target_dtype,
        target_device=request.target_device,
        target_layout_kind=request.target_layout_kind,
        shape_tuple=request.shape_tuple,
        batch_size=request.batch_size,
        input_flat_dim=request.input_flat_dim,
        full_vector_length=request.full_vector_length,
        tensor_object=t,
        tensor_type_name=t_type_name,
        tensor_dtype_name=t_dtype_name,
        tensor_device_type=t_device_type,
        tensor_shape_tuple=t_shape_tuple,
        tensor_numel=t_numel,
        tensor_requires_grad=t_requires_grad,
        tensor_is_floating_point=t_is_floating_point,
        tensor_values_match_p37_nested_values=values_match,
        first_row_first_4_values=first_row,
        second_row_first_4_values=second_row,
        forward_executed=False,
        output_generated=False,
        training_executed=False,
        reason="p39_materialized_torch_tensor",
    )
    validate_materialized_torch_tensor(materialized)
    return materialized


def build_tensor_materialization_metadata(
    request: FCVAETensorMaterializationRequest,
    materialized_tensor_or_none: FCVAEMaterializedTorchTensor | None,
) -> FCVAETensorMaterializationMetadata:
    validate_p39_request(request)
    
    torch_status = build_torch_dependency_status(policy=TORCH_POLICY_OPTIONAL)
    p37_res = run_nested_batch_values_probe()
    p38_res = run_tensor_materialization_request_probe()

    torch_version = ""
    cuda_available = False
    
    if torch_status.available:
        try:
            torch_module = load_torch_for_p39_materialization()
            torch_version = getattr(torch_module, "__version__", "")
            cuda_available = bool(torch_module.cuda.is_available())
        except Exception:
            pass

    attempted = (torch_status.available and p37_res.nested_values_available_in_p37)
    materialized = (materialized_tensor_or_none is not None)

    meta = FCVAETensorMaterializationMetadata(
        contract_version=FC_VAE_TENSOR_MATERIALIZATION_CONTRACT_VERSION,
        source_nested_values_contract_version=request.source_nested_values_contract_version,
        source_tensor_request_contract_version=request.source_tensor_request_contract_version,
        source_torch_boundary_contract_version=request.source_torch_boundary_contract_version,
        torch_available=torch_status.available,
        torch_import_safe=torch_status.import_safe,
        top_level_torch_import_required=torch_status.top_level_import_required,
        torch_version=torch_version,
        cuda_available=cuda_available,
        nested_values_status=p37_res.status,
        tensor_request_status=p38_res.status,
        nested_values_available_in_p37=p37_res.nested_values_available_in_p37,
        tensor_request_available_in_p38=p38_res.tensor_request_available_in_p38,
        tensor_materialization_attempted=attempted,
        tensor_materialized=materialized,
        forward_execution_attempted=False,
        output_generation_attempted=False,
        training_attempted=False,
        reason="p39_tensor_materialization_metadata",
    )
    validate_p39_metadata(meta)
    return meta


def build_tensor_materialization_result(
    request: FCVAETensorMaterializationRequest,
) -> FCVAETensorMaterializationResult:
    validate_p39_request(request)
    
    torch_status = build_torch_dependency_status(policy=TORCH_POLICY_OPTIONAL)
    p38_res = run_tensor_materialization_request_probe()

    if not torch_status.available:
        # blocked torch unavailable
        meta = build_tensor_materialization_metadata(request, None)
        res = FCVAETensorMaterializationResult(
            contract_version=FC_VAE_TENSOR_MATERIALIZATION_CONTRACT_VERSION,
            request=request,
            materialized_tensor=None,
            metadata=meta,
            status=FC_VAE_TENSOR_MATERIALIZATION_STATUS_TORCH_UNAVAILABLE,
            tensor_materialization_available_in_p39=False,
            tensor_materialized_in_p39=False,
            forward_execution_available_in_p39=False,
            output_generation_available_in_p39=False,
            training_available_in_p39=False,
            no_forward_execution=True,
            no_output_generation=True,
            no_training_loop=True,
            no_optimizer=True,
            no_checkpointing=True,
            no_artifact_generation=True,
            no_final_comparison=True,
            no_scientific_conclusion=True,
            reason="tensor_materialization_blocked_torch_unavailable",
        )
        validate_p39_result(res)
        return res

    if not p38_res.tensor_request_available_in_p38:
        # blocked by tensor request status
        meta = build_tensor_materialization_metadata(request, None)
        res = FCVAETensorMaterializationResult(
            contract_version=FC_VAE_TENSOR_MATERIALIZATION_CONTRACT_VERSION,
            request=request,
            materialized_tensor=None,
            metadata=meta,
            status=FC_VAE_TENSOR_MATERIALIZATION_STATUS_REQUEST_BLOCKED,
            tensor_materialization_available_in_p39=False,
            tensor_materialized_in_p39=False,
            forward_execution_available_in_p39=False,
            output_generation_available_in_p39=False,
            training_available_in_p39=False,
            no_forward_execution=True,
            no_output_generation=True,
            no_training_loop=True,
            no_optimizer=True,
            no_checkpointing=True,
            no_artifact_generation=True,
            no_final_comparison=True,
            no_scientific_conclusion=True,
            reason="tensor_materialization_blocked_by_tensor_request_status",
        )
        validate_p39_result(res)
        return res

    # Materialize actual tensor
    t_obj = build_materialized_torch_tensor(request)
    meta = build_tensor_materialization_metadata(request, t_obj)
    
    res = FCVAETensorMaterializationResult(
        contract_version=FC_VAE_TENSOR_MATERIALIZATION_CONTRACT_VERSION,
        request=request,
        materialized_tensor=t_obj,
        metadata=meta,
        status=FC_VAE_TENSOR_MATERIALIZATION_STATUS_MATERIALIZED,
        tensor_materialization_available_in_p39=True,
        tensor_materialized_in_p39=True,
        forward_execution_available_in_p39=False,
        output_generation_available_in_p39=False,
        training_available_in_p39=False,
        no_forward_execution=True,
        no_output_generation=True,
        no_training_loop=True,
        no_optimizer=True,
        no_checkpointing=True,
        no_artifact_generation=True,
        no_final_comparison=True,
        no_scientific_conclusion=True,
        reason="tensor_materialization_completed_successfully",
    )
    validate_p39_result(res)
    return res


def run_tensor_materialization_probe() -> FCVAETensorMaterializationResult:
    req = build_tensor_materialization_request_from_p38_default()
    res = build_tensor_materialization_result(req)
    validate_p39_result(res)
    return res


# Serialization
def p39_tensor_materialization_request_to_json_dict(
    request: FCVAETensorMaterializationRequest,
) -> Dict[str, Any]:
    validate_p39_request(request)
    d = {
        "contract_version": request.contract_version,
        "materialization_kind": request.materialization_kind,
        "architecture_id": request.architecture_id,
        "source_nested_values_contract_version": request.source_nested_values_contract_version,
        "source_tensor_request_contract_version": request.source_tensor_request_contract_version,
        "source_torch_boundary_contract_version": request.source_torch_boundary_contract_version,
        "target_framework": request.target_framework,
        "target_dtype": request.target_dtype,
        "target_device": request.target_device,
        "target_layout_kind": request.target_layout_kind,
        "batch_size": request.batch_size,
        "input_flat_dim": request.input_flat_dim,
        "full_vector_length": request.full_vector_length,
        "shape_tuple": list(request.shape_tuple),
        "requires_grad": request.requires_grad,
        "allow_tensor_materialization_in_p39": request.allow_tensor_materialization_in_p39,
        "allow_forward_execution_in_p39": request.allow_forward_execution_in_p39,
        "allow_output_generation_in_p39": request.allow_output_generation_in_p39,
        "allow_training_in_p39": request.allow_training_in_p39,
        "reason": request.reason,
    }
    assert_no_local_path_leakage(d)
    assert_no_forbidden_claims(d)
    return d


def materialized_torch_tensor_to_json_dict(
    tensor: FCVAEMaterializedTorchTensor,
) -> Dict[str, Any]:
    validate_materialized_torch_tensor(tensor)
    d = {
        "contract_version": tensor.contract_version,
        "materialization_kind": tensor.materialization_kind,
        "target_framework": tensor.target_framework,
        "target_dtype": tensor.target_dtype,
        "target_device": tensor.target_device,
        "target_layout_kind": tensor.target_layout_kind,
        "shape_tuple": list(tensor.shape_tuple),
        "batch_size": tensor.batch_size,
        "input_flat_dim": tensor.input_flat_dim,
        "full_vector_length": tensor.full_vector_length,
        "tensor_type_name": tensor.tensor_type_name,
        "tensor_dtype_name": tensor.tensor_dtype_name,
        "tensor_device_type": tensor.tensor_device_type,
        "tensor_shape_tuple": list(tensor.tensor_shape_tuple),
        "tensor_numel": tensor.tensor_numel,
        "tensor_requires_grad": tensor.tensor_requires_grad,
        "tensor_is_floating_point": tensor.tensor_is_floating_point,
        "tensor_values_match_p37_nested_values": tensor.tensor_values_match_p37_nested_values,
        "first_row_first_4_values": list(tensor.first_row_first_4_values),
        "second_row_first_4_values": list(tensor.second_row_first_4_values),
        "forward_executed": tensor.forward_executed,
        "output_generated": tensor.output_generated,
        "training_executed": tensor.training_executed,
        "reason": tensor.reason,
    }
    assert_no_local_path_leakage(d)
    assert_no_forbidden_claims(d)
    return d


def p39_tensor_materialization_metadata_to_json_dict(
    metadata: FCVAETensorMaterializationMetadata,
) -> Dict[str, Any]:
    validate_p39_metadata(metadata)
    d = {
        "contract_version": metadata.contract_version,
        "source_nested_values_contract_version": metadata.source_nested_values_contract_version,
        "source_tensor_request_contract_version": metadata.source_tensor_request_contract_version,
        "source_torch_boundary_contract_version": metadata.source_torch_boundary_contract_version,
        "torch_available": metadata.torch_available,
        "torch_import_safe": metadata.torch_import_safe,
        "top_level_torch_import_required": metadata.top_level_torch_import_required,
        "torch_version": metadata.torch_version,
        "cuda_available": metadata.cuda_available,
        "nested_values_status": metadata.nested_values_status,
        "tensor_request_status": metadata.tensor_request_status,
        "nested_values_available_in_p37": metadata.nested_values_available_in_p37,
        "tensor_request_available_in_p38": metadata.tensor_request_available_in_p38,
        "tensor_materialization_attempted": metadata.tensor_materialization_attempted,
        "tensor_materialized": metadata.tensor_materialized,
        "forward_execution_attempted": metadata.forward_execution_attempted,
        "output_generation_attempted": metadata.output_generation_attempted,
        "training_attempted": metadata.training_attempted,
        "reason": metadata.reason,
    }
    assert_no_local_path_leakage(d)
    assert_no_forbidden_claims(d)
    return d


def p39_tensor_materialization_result_to_json_dict(
    result: FCVAETensorMaterializationResult,
) -> Dict[str, Any]:
    validate_p39_result(result)
    
    t_dict = None
    if result.materialized_tensor is not None:
        t_dict = materialized_torch_tensor_to_json_dict(result.materialized_tensor)

    d = {
        "contract_version": result.contract_version,
        "request": p39_tensor_materialization_request_to_json_dict(result.request),
        "materialized_tensor": t_dict,
        "metadata": p39_tensor_materialization_metadata_to_json_dict(result.metadata),
        "status": result.status,
        "tensor_materialization_available_in_p39": result.tensor_materialization_available_in_p39,
        "tensor_materialized_in_p39": result.tensor_materialized_in_p39,
        "forward_execution_available_in_p39": result.forward_execution_available_in_p39,
        "output_generation_available_in_p39": result.output_generation_available_in_p39,
        "training_available_in_p39": result.training_available_in_p39,
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


def compact_p39_tensor_materialization_json(
    result: FCVAETensorMaterializationResult,
) -> str:
    d = p39_tensor_materialization_result_to_json_dict(result)
    return json.dumps(d, sort_keys=True, separators=(",", ":"))
