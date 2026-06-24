# src/phase2/fc_vae_model.py

import dataclasses
import json
from typing import Any, Tuple

from src.phase2.torch_boundary import (
    build_torch_dependency_status,
)


# Public constants
FC_VAE_MODEL_SKELETON_CONTRACT_VERSION = "phase2_p27_fc_vae_model_skeleton_contract_v1"

FC_VAE_ARCHITECTURE_ID = "FC-VAE"

FC_VAE_MODULE_NAME = "src.phase2.fc_vae_model"

FC_VAE_STATUS_SKELETON_ONLY = "skeleton_only"

FC_VAE_STATUS_BLOCKED_TORCH_UNAVAILABLE = "blocked_torch_unavailable"

FC_VAE_STATUS_READY_FOR_FUTURE_IMPLEMENTATION = "ready_for_future_implementation"

SUPPORTED_FC_VAE_STATUSES = (
    "skeleton_only",
    "blocked_torch_unavailable",
    "ready_for_future_implementation",
)

FC_VAE_REQUIRED_LATENT_NAMES = (
    "z_mean",
    "z_volatility",
    "z_shared",
)

FC_VAE_DECODER_OUTPUT_KIND_MODELSPEC = "typed_modelspec_candidate"


# Public dataclasses
@dataclasses.dataclass(frozen=True)
class FCVAEInputShapeContract:
    contract_version: str
    flat_dim: int
    source: str
    reason: str


@dataclasses.dataclass(frozen=True)
class FCVAELatentLayout:
    contract_version: str
    z_mean_dim: int
    z_volatility_dim: int
    z_shared_dim: int
    latent_names: Tuple[str, ...]
    total_latent_dim: int
    reason: str


@dataclasses.dataclass(frozen=True)
class FCVAEDecoderOutputContract:
    contract_version: str
    output_kind: str
    family_head_dim: int
    mean_head_dim: int
    volatility_head_dim: int
    diagnostic_head_dim: int
    target_boundary: str
    reason: str


@dataclasses.dataclass(frozen=True)
class FCVAEForwardContract:
    contract_version: str
    architecture_id: str
    module_name: str
    input_shape: FCVAEInputShapeContract
    latent_layout: FCVAELatentLayout
    decoder_output: FCVAEDecoderOutputContract
    torch_required_for_execution: bool
    implemented_in_p27: bool
    reason: str


@dataclasses.dataclass(frozen=True)
class FCVAESkeletonStatus:
    contract_version: str
    architecture_id: str
    module_name: str
    torch_available: bool
    status: str
    forward_contract: FCVAEForwardContract
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


def validate_exact_tuple_str(value: Any, expected: Tuple[str, ...], name: str) -> None:
    if type(value) is not tuple:
        raise TypeError(f"{name} must be exact tuple, got {type(value).__name__}")
    for i, v in enumerate(value):
        if type(v) is not str:
            raise TypeError(f"{name}[{i}] must be exact str, got {type(v).__name__}")
    if value != expected:
        raise ValueError(f"{name} must equal {expected}, got {value}")


def validate_fc_vae_status(value: Any) -> None:
    if type(value) is not str:
        raise TypeError(f"status must be exact str, got {type(value).__name__}")
    if value not in SUPPORTED_FC_VAE_STATUSES:
        raise ValueError(f"Unsupported FC-VAE status: {value}")


def validate_input_shape_contract(contract: FCVAEInputShapeContract) -> None:
    if type(contract) is not FCVAEInputShapeContract:
        raise TypeError("contract must be exact FCVAEInputShapeContract instance")
    validate_non_empty_str(contract.contract_version, "contract_version")
    if contract.contract_version != FC_VAE_MODEL_SKELETON_CONTRACT_VERSION:
        raise ValueError(
            f"Wrong contract version: expected {FC_VAE_MODEL_SKELETON_CONTRACT_VERSION}"
        )
    validate_positive_int(contract.flat_dim, "flat_dim")
    validate_non_empty_str(contract.source, "source")
    validate_non_empty_str(contract.reason, "reason")


def validate_latent_layout(layout: FCVAELatentLayout) -> None:
    if type(layout) is not FCVAELatentLayout:
        raise TypeError("layout must be exact FCVAELatentLayout instance")
    validate_non_empty_str(layout.contract_version, "contract_version")
    if layout.contract_version != FC_VAE_MODEL_SKELETON_CONTRACT_VERSION:
        raise ValueError(
            f"Wrong contract version: expected {FC_VAE_MODEL_SKELETON_CONTRACT_VERSION}"
        )
    validate_positive_int(layout.z_mean_dim, "z_mean_dim")
    validate_positive_int(layout.z_volatility_dim, "z_volatility_dim")
    validate_positive_int(layout.z_shared_dim, "z_shared_dim")
    validate_exact_tuple_str(layout.latent_names, FC_VAE_REQUIRED_LATENT_NAMES, "latent_names")
    validate_positive_int(layout.total_latent_dim, "total_latent_dim")
    expected_total = layout.z_mean_dim + layout.z_volatility_dim + layout.z_shared_dim
    if layout.total_latent_dim != expected_total:
        raise ValueError(
            f"total_latent_dim must equal z_mean_dim + z_volatility_dim + z_shared_dim "
            f"({expected_total}), got {layout.total_latent_dim}"
        )
    validate_non_empty_str(layout.reason, "reason")


def validate_decoder_output_contract(contract: FCVAEDecoderOutputContract) -> None:
    if type(contract) is not FCVAEDecoderOutputContract:
        raise TypeError("contract must be exact FCVAEDecoderOutputContract instance")
    validate_non_empty_str(contract.contract_version, "contract_version")
    if contract.contract_version != FC_VAE_MODEL_SKELETON_CONTRACT_VERSION:
        raise ValueError(
            f"Wrong contract version: expected {FC_VAE_MODEL_SKELETON_CONTRACT_VERSION}"
        )
    validate_non_empty_str(contract.output_kind, "output_kind")
    if contract.output_kind != FC_VAE_DECODER_OUTPUT_KIND_MODELSPEC:
        raise ValueError(
            f"output_kind must be {FC_VAE_DECODER_OUTPUT_KIND_MODELSPEC}, got {contract.output_kind}"
        )
    validate_positive_int(contract.family_head_dim, "family_head_dim")
    validate_positive_int(contract.mean_head_dim, "mean_head_dim")
    validate_positive_int(contract.volatility_head_dim, "volatility_head_dim")
    validate_positive_int(contract.diagnostic_head_dim, "diagnostic_head_dim")
    validate_non_empty_str(contract.target_boundary, "target_boundary")
    if contract.target_boundary != "ModelSpec":
        raise ValueError(
            f"target_boundary must be ModelSpec, got {contract.target_boundary}"
        )
    validate_non_empty_str(contract.reason, "reason")


def validate_forward_contract(contract: FCVAEForwardContract) -> None:
    if type(contract) is not FCVAEForwardContract:
        raise TypeError("contract must be exact FCVAEForwardContract instance")
    validate_non_empty_str(contract.contract_version, "contract_version")
    if contract.contract_version != FC_VAE_MODEL_SKELETON_CONTRACT_VERSION:
        raise ValueError(
            f"Wrong contract version: expected {FC_VAE_MODEL_SKELETON_CONTRACT_VERSION}"
        )
    validate_non_empty_str(contract.architecture_id, "architecture_id")
    if contract.architecture_id != FC_VAE_ARCHITECTURE_ID:
        raise ValueError(
            f"architecture_id must be {FC_VAE_ARCHITECTURE_ID}, got {contract.architecture_id}"
        )
    validate_non_empty_str(contract.module_name, "module_name")
    if contract.module_name != FC_VAE_MODULE_NAME:
        raise ValueError(
            f"module_name must be {FC_VAE_MODULE_NAME}, got {contract.module_name}"
        )
    validate_input_shape_contract(contract.input_shape)
    validate_latent_layout(contract.latent_layout)
    validate_decoder_output_contract(contract.decoder_output)
    validate_bool(contract.torch_required_for_execution, "torch_required_for_execution")
    if not contract.torch_required_for_execution:
        raise ValueError("torch_required_for_execution must be True")
    validate_bool(contract.implemented_in_p27, "implemented_in_p27")
    if contract.implemented_in_p27:
        raise ValueError("implemented_in_p27 must be False in P27")
    validate_non_empty_str(contract.reason, "reason")


def validate_skeleton_status(status: FCVAESkeletonStatus) -> None:
    if type(status) is not FCVAESkeletonStatus:
        raise TypeError("status must be exact FCVAESkeletonStatus instance")
    validate_non_empty_str(status.contract_version, "contract_version")
    if status.contract_version != FC_VAE_MODEL_SKELETON_CONTRACT_VERSION:
        raise ValueError(
            f"Wrong contract version: expected {FC_VAE_MODEL_SKELETON_CONTRACT_VERSION}"
        )
    validate_non_empty_str(status.architecture_id, "architecture_id")
    if status.architecture_id != FC_VAE_ARCHITECTURE_ID:
        raise ValueError(
            f"architecture_id must be {FC_VAE_ARCHITECTURE_ID}, got {status.architecture_id}"
        )
    validate_non_empty_str(status.module_name, "module_name")
    if status.module_name != FC_VAE_MODULE_NAME:
        raise ValueError(
            f"module_name must be {FC_VAE_MODULE_NAME}, got {status.module_name}"
        )
    validate_bool(status.torch_available, "torch_available")
    validate_fc_vae_status(status.status)
    if not status.torch_available and status.status == FC_VAE_STATUS_READY_FOR_FUTURE_IMPLEMENTATION:
        raise ValueError(
            "status cannot be ready_for_future_implementation when torch_available is False"
        )
    validate_forward_contract(status.forward_contract)
    validate_bool(status.no_model_implementation, "no_model_implementation")
    if not status.no_model_implementation:
        raise ValueError("no_model_implementation must be True")
    validate_bool(status.no_training_loop, "no_training_loop")
    if not status.no_training_loop:
        raise ValueError("no_training_loop must be True")
    validate_bool(status.no_optimizer, "no_optimizer")
    if not status.no_optimizer:
        raise ValueError("no_optimizer must be True")
    validate_bool(status.no_checkpointing, "no_checkpointing")
    if not status.no_checkpointing:
        raise ValueError("no_checkpointing must be True")
    validate_bool(status.no_artifact_generation, "no_artifact_generation")
    if not status.no_artifact_generation:
        raise ValueError("no_artifact_generation must be True")
    validate_non_empty_str(status.reason, "reason")


# Public builder functions
def build_smoke_input_shape_contract() -> FCVAEInputShapeContract:
    contract = FCVAEInputShapeContract(
        contract_version=FC_VAE_MODEL_SKELETON_CONTRACT_VERSION,
        flat_dim=32,
        source="schema_v2_flat_boundary_vector",
        reason="p27_smoke_input_shape_contract",
    )
    validate_input_shape_contract(contract)
    return contract


def build_smoke_latent_layout() -> FCVAELatentLayout:
    layout = FCVAELatentLayout(
        contract_version=FC_VAE_MODEL_SKELETON_CONTRACT_VERSION,
        z_mean_dim=8,
        z_volatility_dim=8,
        z_shared_dim=4,
        latent_names=FC_VAE_REQUIRED_LATENT_NAMES,
        total_latent_dim=20,
        reason="p27_smoke_latent_layout",
    )
    validate_latent_layout(layout)
    return layout


def build_smoke_decoder_output_contract() -> FCVAEDecoderOutputContract:
    contract = FCVAEDecoderOutputContract(
        contract_version=FC_VAE_MODEL_SKELETON_CONTRACT_VERSION,
        output_kind=FC_VAE_DECODER_OUTPUT_KIND_MODELSPEC,
        family_head_dim=3,
        mean_head_dim=3,
        volatility_head_dim=3,
        diagnostic_head_dim=4,
        target_boundary="ModelSpec",
        reason="p27_smoke_decoder_output_contract",
    )
    validate_decoder_output_contract(contract)
    return contract


def build_smoke_forward_contract() -> FCVAEForwardContract:
    contract = FCVAEForwardContract(
        contract_version=FC_VAE_MODEL_SKELETON_CONTRACT_VERSION,
        architecture_id=FC_VAE_ARCHITECTURE_ID,
        module_name=FC_VAE_MODULE_NAME,
        input_shape=build_smoke_input_shape_contract(),
        latent_layout=build_smoke_latent_layout(),
        decoder_output=build_smoke_decoder_output_contract(),
        torch_required_for_execution=True,
        implemented_in_p27=False,
        reason="p27_smoke_forward_contract",
    )
    validate_forward_contract(contract)
    return contract


def build_fc_vae_skeleton_status() -> FCVAESkeletonStatus:
    torch_status = build_torch_dependency_status()
    torch_available = torch_status.available
    if torch_available:
        current_status = FC_VAE_STATUS_READY_FOR_FUTURE_IMPLEMENTATION
    else:
        current_status = FC_VAE_STATUS_BLOCKED_TORCH_UNAVAILABLE
    status = FCVAESkeletonStatus(
        contract_version=FC_VAE_MODEL_SKELETON_CONTRACT_VERSION,
        architecture_id=FC_VAE_ARCHITECTURE_ID,
        module_name=FC_VAE_MODULE_NAME,
        torch_available=torch_available,
        status=current_status,
        forward_contract=build_smoke_forward_contract(),
        no_model_implementation=True,
        no_training_loop=True,
        no_optimizer=True,
        no_checkpointing=True,
        no_artifact_generation=True,
        reason="p27_fc_vae_skeleton_status",
    )
    validate_skeleton_status(status)
    return status


def require_fc_vae_implementation_available(status: FCVAESkeletonStatus) -> None:
    validate_skeleton_status(status)
    raise NotImplementedError(
        "FC-VAE implementation is not available in P27; "
        "this phase defines only the skeleton and shape contract"
    )


# JSON serialization functions
def input_shape_contract_to_json_dict(contract: FCVAEInputShapeContract) -> dict:
    validate_input_shape_contract(contract)
    return {
        "contract_version": contract.contract_version,
        "flat_dim": contract.flat_dim,
        "source": contract.source,
        "reason": contract.reason,
    }


def latent_layout_to_json_dict(layout: FCVAELatentLayout) -> dict:
    validate_latent_layout(layout)
    return {
        "contract_version": layout.contract_version,
        "z_mean_dim": layout.z_mean_dim,
        "z_volatility_dim": layout.z_volatility_dim,
        "z_shared_dim": layout.z_shared_dim,
        "latent_names": list(layout.latent_names),
        "total_latent_dim": layout.total_latent_dim,
        "reason": layout.reason,
    }


def decoder_output_contract_to_json_dict(contract: FCVAEDecoderOutputContract) -> dict:
    validate_decoder_output_contract(contract)
    return {
        "contract_version": contract.contract_version,
        "output_kind": contract.output_kind,
        "family_head_dim": contract.family_head_dim,
        "mean_head_dim": contract.mean_head_dim,
        "volatility_head_dim": contract.volatility_head_dim,
        "diagnostic_head_dim": contract.diagnostic_head_dim,
        "target_boundary": contract.target_boundary,
        "reason": contract.reason,
    }


def forward_contract_to_json_dict(contract: FCVAEForwardContract) -> dict:
    validate_forward_contract(contract)
    return {
        "contract_version": contract.contract_version,
        "architecture_id": contract.architecture_id,
        "module_name": contract.module_name,
        "input_shape": input_shape_contract_to_json_dict(contract.input_shape),
        "latent_layout": latent_layout_to_json_dict(contract.latent_layout),
        "decoder_output": decoder_output_contract_to_json_dict(contract.decoder_output),
        "torch_required_for_execution": contract.torch_required_for_execution,
        "implemented_in_p27": contract.implemented_in_p27,
        "reason": contract.reason,
    }


def skeleton_status_to_json_dict(status: FCVAESkeletonStatus) -> dict:
    validate_skeleton_status(status)
    return {
        "contract_version": status.contract_version,
        "architecture_id": status.architecture_id,
        "module_name": status.module_name,
        "torch_available": status.torch_available,
        "status": status.status,
        "forward_contract": forward_contract_to_json_dict(status.forward_contract),
        "no_model_implementation": status.no_model_implementation,
        "no_training_loop": status.no_training_loop,
        "no_optimizer": status.no_optimizer,
        "no_checkpointing": status.no_checkpointing,
        "no_artifact_generation": status.no_artifact_generation,
        "reason": status.reason,
    }


def compact_fc_vae_skeleton_json(status: FCVAESkeletonStatus) -> str:
    return json.dumps(
        skeleton_status_to_json_dict(status),
        sort_keys=True,
        separators=(",", ":"),
    )
