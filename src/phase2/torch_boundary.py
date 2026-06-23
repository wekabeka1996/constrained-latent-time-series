# src/phase2/torch_boundary.py

import dataclasses
import importlib.util
import json
from typing import Any


# Public constants
TORCH_BOUNDARY_CONTRACT_VERSION = "phase2_p26_torch_boundary_contract_v1"

TORCH_BACKEND_NAME = "torch"

TORCH_POLICY_OPTIONAL = "optional"
TORCH_POLICY_REQUIRED_FOR_MODEL_IMPLEMENTATION = "required_for_model_implementation"
TORCH_POLICY_FORBIDDEN_IN_CORE = "forbidden_in_core"

SUPPORTED_TORCH_POLICIES = (
    "optional",
    "required_for_model_implementation",
    "forbidden_in_core",
)

FC_VAE_FUTURE_MODULE_NAME = "src.phase2.fc_vae_model"

FC_VAE_FUTURE_ARCHITECTURE_ID = "FC-VAE"


# Public Dataclasses
@dataclasses.dataclass(frozen=True)
class TorchDependencyStatus:
    contract_version: str
    backend_name: str
    available: bool
    policy: str
    import_safe: bool
    top_level_import_required: bool
    reason: str


@dataclasses.dataclass(frozen=True)
class FutureModelBoundarySpec:
    contract_version: str
    architecture_id: str
    future_module_name: str
    torch_policy: str
    requires_training_loop: bool
    requires_optimizer: bool
    requires_checkpointing: bool
    allowed_in_p26: bool
    reason: str


@dataclasses.dataclass(frozen=True)
class TorchBoundarySmokeResult:
    contract_version: str
    torch_status: TorchDependencyStatus
    future_model_boundary: FutureModelBoundarySpec
    no_model_implementation: bool
    no_training_loop: bool
    no_optimizer: bool
    no_checkpointing: bool
    no_artifact_generation: bool
    verdict: str
    reason: str


# Public Validation Functions
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


def validate_torch_policy(value: Any) -> None:
    if type(value) is not str:
        raise TypeError(f"torch_policy must be exact str, got {type(value).__name__}")
    if value not in SUPPORTED_TORCH_POLICIES:
        raise ValueError(f"Unsupported torch policy: {value}")


# Detection
def detect_torch_available() -> bool:
    return importlib.util.find_spec("torch") is not None


# Status builder
def build_torch_dependency_status(
    policy: str = TORCH_POLICY_OPTIONAL,
) -> TorchDependencyStatus:
    validate_torch_policy(policy)
    available = detect_torch_available()
    import_safe = available
    if policy == TORCH_POLICY_FORBIDDEN_IN_CORE:
        import_safe = False
    status = TorchDependencyStatus(
        contract_version=TORCH_BOUNDARY_CONTRACT_VERSION,
        backend_name=TORCH_BACKEND_NAME,
        available=available,
        policy=policy,
        import_safe=import_safe,
        top_level_import_required=False,
        reason="torch_available_optional_boundary" if available else "torch_unavailable_optional_boundary",
    )
    validate_torch_dependency_status(status)
    return status


def validate_torch_dependency_status(status: TorchDependencyStatus) -> None:
    if type(status) is not TorchDependencyStatus:
        raise TypeError("status must be exact TorchDependencyStatus instance")

    validate_non_empty_str(status.contract_version, "contract_version")
    if status.contract_version != TORCH_BOUNDARY_CONTRACT_VERSION:
        raise ValueError(
            f"Wrong contract version: expected {TORCH_BOUNDARY_CONTRACT_VERSION}"
        )

    validate_non_empty_str(status.backend_name, "backend_name")
    if status.backend_name != TORCH_BACKEND_NAME:
        raise ValueError(f"Wrong backend name: expected {TORCH_BACKEND_NAME}")

    validate_bool(status.available, "available")
    validate_torch_policy(status.policy)
    validate_bool(status.import_safe, "import_safe")
    validate_bool(status.top_level_import_required, "top_level_import_required")

    if status.top_level_import_required:
        raise ValueError("top_level_import_required must be False in P26")

    if status.policy == TORCH_POLICY_FORBIDDEN_IN_CORE:
        if status.import_safe:
            raise ValueError(
                "import_safe must be False when policy is forbidden_in_core"
            )

    validate_non_empty_str(status.reason, "reason")


def require_torch_available_for_future_model(
    status: TorchDependencyStatus,
) -> None:
    validate_torch_dependency_status(status)
    if not status.available:
        raise RuntimeError(
            "torch unavailable; future model implementation must be skipped or blocked"
        )


# Future model boundary
def build_future_model_boundary_spec() -> FutureModelBoundarySpec:
    spec = FutureModelBoundarySpec(
        contract_version=TORCH_BOUNDARY_CONTRACT_VERSION,
        architecture_id=FC_VAE_FUTURE_ARCHITECTURE_ID,
        future_module_name=FC_VAE_FUTURE_MODULE_NAME,
        torch_policy=TORCH_POLICY_REQUIRED_FOR_MODEL_IMPLEMENTATION,
        requires_training_loop=True,
        requires_optimizer=True,
        requires_checkpointing=True,
        allowed_in_p26=False,
        reason="FC-VAE model implementation deferred to future phase",
    )
    validate_future_model_boundary_spec(spec)
    return spec


def validate_future_model_boundary_spec(spec: FutureModelBoundarySpec) -> None:
    if type(spec) is not FutureModelBoundarySpec:
        raise TypeError("spec must be exact FutureModelBoundarySpec instance")

    validate_non_empty_str(spec.contract_version, "contract_version")
    if spec.contract_version != TORCH_BOUNDARY_CONTRACT_VERSION:
        raise ValueError(
            f"Wrong contract version: expected {TORCH_BOUNDARY_CONTRACT_VERSION}"
        )

    validate_non_empty_str(spec.architecture_id, "architecture_id")
    if spec.architecture_id != FC_VAE_FUTURE_ARCHITECTURE_ID:
        raise ValueError(
            f"Wrong architecture_id: expected {FC_VAE_FUTURE_ARCHITECTURE_ID}"
        )

    validate_non_empty_str(spec.future_module_name, "future_module_name")
    if spec.future_module_name != FC_VAE_FUTURE_MODULE_NAME:
        raise ValueError(
            f"Wrong future_module_name: expected {FC_VAE_FUTURE_MODULE_NAME}"
        )

    validate_torch_policy(spec.torch_policy)
    validate_bool(spec.requires_training_loop, "requires_training_loop")
    validate_bool(spec.requires_optimizer, "requires_optimizer")
    validate_bool(spec.requires_checkpointing, "requires_checkpointing")
    validate_bool(spec.allowed_in_p26, "allowed_in_p26")

    if spec.allowed_in_p26:
        raise ValueError("allowed_in_p26 must be False in P26")

    validate_non_empty_str(spec.reason, "reason")


# Smoke result
def build_torch_boundary_smoke_result() -> TorchBoundarySmokeResult:
    status = build_torch_dependency_status(policy=TORCH_POLICY_OPTIONAL)
    boundary = build_future_model_boundary_spec()
    result = TorchBoundarySmokeResult(
        contract_version=TORCH_BOUNDARY_CONTRACT_VERSION,
        torch_status=status,
        future_model_boundary=boundary,
        no_model_implementation=True,
        no_training_loop=True,
        no_optimizer=True,
        no_checkpointing=True,
        no_artifact_generation=True,
        verdict="PASS",
        reason="p26_torch_boundary_smoke_completed",
    )
    validate_torch_boundary_smoke_result(result)
    return result


def validate_torch_boundary_smoke_result(
    result: TorchBoundarySmokeResult,
) -> None:
    if type(result) is not TorchBoundarySmokeResult:
        raise TypeError("result must be exact TorchBoundarySmokeResult instance")

    validate_non_empty_str(result.contract_version, "contract_version")
    if result.contract_version != TORCH_BOUNDARY_CONTRACT_VERSION:
        raise ValueError(
            f"Wrong contract version: expected {TORCH_BOUNDARY_CONTRACT_VERSION}"
        )

    validate_torch_dependency_status(result.torch_status)
    validate_future_model_boundary_spec(result.future_model_boundary)

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

    validate_non_empty_str(result.verdict, "verdict")
    if result.verdict != "PASS":
        raise ValueError(f"verdict must be PASS, got {result.verdict}")

    validate_non_empty_str(result.reason, "reason")


# JSON serialization
def torch_dependency_status_to_json_dict(
    status: TorchDependencyStatus,
) -> dict:
    validate_torch_dependency_status(status)
    return {
        "contract_version": status.contract_version,
        "backend_name": status.backend_name,
        "available": status.available,
        "policy": status.policy,
        "import_safe": status.import_safe,
        "top_level_import_required": status.top_level_import_required,
        "reason": status.reason,
    }


def future_model_boundary_spec_to_json_dict(
    spec: FutureModelBoundarySpec,
) -> dict:
    validate_future_model_boundary_spec(spec)
    return {
        "contract_version": spec.contract_version,
        "architecture_id": spec.architecture_id,
        "future_module_name": spec.future_module_name,
        "torch_policy": spec.torch_policy,
        "requires_training_loop": spec.requires_training_loop,
        "requires_optimizer": spec.requires_optimizer,
        "requires_checkpointing": spec.requires_checkpointing,
        "allowed_in_p26": spec.allowed_in_p26,
        "reason": spec.reason,
    }


def torch_boundary_smoke_result_to_json_dict(
    result: TorchBoundarySmokeResult,
) -> dict:
    validate_torch_boundary_smoke_result(result)
    return {
        "contract_version": result.contract_version,
        "torch_status": torch_dependency_status_to_json_dict(result.torch_status),
        "future_model_boundary": future_model_boundary_spec_to_json_dict(
            result.future_model_boundary
        ),
        "no_model_implementation": result.no_model_implementation,
        "no_training_loop": result.no_training_loop,
        "no_optimizer": result.no_optimizer,
        "no_checkpointing": result.no_checkpointing,
        "no_artifact_generation": result.no_artifact_generation,
        "verdict": result.verdict,
        "reason": result.reason,
    }


def compact_torch_boundary_json(result: TorchBoundarySmokeResult) -> str:
    return json.dumps(
        torch_boundary_smoke_result_to_json_dict(result),
        sort_keys=True,
        separators=(",", ":"),
    )
