# src/phase2/model_interface.py

import dataclasses
import json
import math
from typing import Any, Tuple, Optional

from src.phase2.schema import ModelSpec, validate_model_spec
from src.phase2.constraints import require_math_valid
from src.phase2.evidence import (
    assert_no_local_path_leakage,
    assert_no_raw_params,
    assert_no_forbidden_claims,
)

# Public constants
MODEL_INTERFACE_CONTRACT_VERSION = "phase2_p25_model_interface_contract_v1"
FC_VAE_ARCHITECTURE_ID = "FC-VAE"
SUPPORTED_MODEL_ARCHITECTURE_IDS = (
    "FC-VAE",
)

MODEL_RUN_KIND_ZERO_SHOT = "zero_shot"
MODEL_RUN_KIND_FEWSHOT = "fewshot"
SUPPORTED_MODEL_RUN_KINDS = (
    "zero_shot",
    "fewshot",
)

MODEL_CANDIDATE_SUMMARY_CONTRACT_VERSION = "phase2_p25_model_candidate_summary_v1"

FORBIDDEN_MODEL_INTERFACE_RAW_PARAM_KEYS = (
    "ar_params",
    "ma_params",
    "alpha_params",
    "beta_params",
    "omega",
    "constraint_flags",
    "model_spec",
    "generated_model_spec",
    "candidate_specs",
    "reference_specs",
)

FORBIDDEN_MODEL_INTERFACE_CLAIM_WORDS = (
    "best",
    "winner",
    "solved",
    "scientific success",
    "model works",
)


# Public Dataclasses
@dataclasses.dataclass(frozen=True)
class ModelRunMetadata:
    contract_version: str
    model_architecture_id: str
    model_run_kind: str
    model_code_git_commit: str
    training_artifact_root: str
    training_split_contract: str
    model_repeat_seed: int
    zero_shot_mode: bool
    C_train_count: int
    config_hash: str
    evidence_contract_version: str
    reason: str


@dataclasses.dataclass(frozen=True)
class ModelCandidateRecord:
    candidate_id: str
    metadata: ModelRunMetadata
    generated_spec: ModelSpec
    validity_pass: bool
    math_validity_pass: bool
    generated_family_id: str
    generated_mean_family: str
    generated_volatility_family: str
    source_split: str
    reason: str


@dataclasses.dataclass(frozen=True)
class ModelCandidateBatch:
    contract_version: str
    metadata: ModelRunMetadata
    candidates: Tuple[ModelCandidateRecord, ...]
    candidate_count: int
    reason: str


@dataclasses.dataclass(frozen=True)
class ModelCandidateSummary:
    contract_version: str
    model_architecture_id: str
    model_run_kind: str
    model_repeat_seed: int
    zero_shot_mode: bool
    C_train_count: int
    candidate_count: int
    generated_family_ids: Tuple[str, ...]
    generated_mean_families: Tuple[str, ...]
    generated_volatility_families: Tuple[str, ...]
    validity_pass_count: int
    math_validity_pass_count: int
    source_split: str
    evidence_contract_version: str
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


def validate_non_negative_int(value: Any, name: str) -> None:
    if type(value) is not int or isinstance(value, bool):
        raise TypeError(f"{name} must be exact int, got {type(value).__name__}")
    if value < 0:
        raise ValueError(f"{name} must be >= 0")


def validate_positive_int(value: Any, name: str) -> None:
    if type(value) is not int or isinstance(value, bool):
        raise TypeError(f"{name} must be exact int, got {type(value).__name__}")
    if value <= 0:
        raise ValueError(f"{name} must be > 0")


def validate_model_architecture_id(value: Any) -> None:
    if type(value) is not str:
        raise TypeError(f"model_architecture_id must be exact str, got {type(value).__name__}")
    if value not in SUPPORTED_MODEL_ARCHITECTURE_IDS:
        raise ValueError(f"Unsupported model architecture: {value}")


def validate_model_run_kind(value: Any) -> None:
    if type(value) is not str:
        raise TypeError(f"model_run_kind must be exact str, got {type(value).__name__}")
    if value not in SUPPORTED_MODEL_RUN_KINDS:
        raise ValueError(f"Unsupported model run kind: {value}")


def assert_no_model_interface_local_path_leakage(data: Any) -> None:
    # Use existing P23 helper
    assert_no_local_path_leakage(data)


def assert_no_model_interface_raw_params(data: Any) -> None:
    # Check for forbidden raw param keys in keys/values
    if isinstance(data, str):
        for forbidden in FORBIDDEN_MODEL_INTERFACE_RAW_PARAM_KEYS:
            if forbidden == data or f"'{forbidden}'" in data or f'"{forbidden}"' in data or f"{forbidden}:" in data or f"{forbidden}=" in data:
                raise ValueError(f"Forbidden raw parameter leakage key '{forbidden}' found in: '{data}'")
    elif isinstance(data, dict):
        for k, v in data.items():
            if k in FORBIDDEN_MODEL_INTERFACE_RAW_PARAM_KEYS:
                raise ValueError(f"Forbidden raw parameter leakage key '{k}' found in dictionary keys")
            assert_no_model_interface_raw_params(k)
            assert_no_model_interface_raw_params(v)
    elif isinstance(data, (list, tuple)):
        for item in data:
            assert_no_model_interface_raw_params(item)


def assert_no_model_interface_forbidden_claims(data: Any) -> None:
    if isinstance(data, str):
        normalized = data.lower()
        # Clean allowed flags first
        cleaned = normalized.replace("no_scientific_conclusion", "").replace("no_final_comparison", "")
        for word in FORBIDDEN_MODEL_INTERFACE_CLAIM_WORDS:
            if word in cleaned:
                raise ValueError(f"Forbidden claim word '{word}' found in: '{data}'")
    elif isinstance(data, dict):
        for k, v in data.items():
            assert_no_model_interface_forbidden_claims(k)
            assert_no_model_interface_forbidden_claims(v)
    elif isinstance(data, (list, tuple)):
        for item in data:
            assert_no_model_interface_forbidden_claims(item)


def validate_model_run_metadata(metadata: ModelRunMetadata) -> None:
    if type(metadata) is not ModelRunMetadata:
        raise TypeError("metadata must be exact ModelRunMetadata instance")

    validate_non_empty_str(metadata.contract_version, "contract_version")
    if metadata.contract_version != MODEL_INTERFACE_CONTRACT_VERSION:
        raise ValueError(f"Wrong contract version: expected {MODEL_INTERFACE_CONTRACT_VERSION}")

    validate_model_architecture_id(metadata.model_architecture_id)
    validate_model_run_kind(metadata.model_run_kind)

    validate_non_empty_str(metadata.model_code_git_commit, "model_code_git_commit")
    validate_non_empty_str(metadata.training_artifact_root, "training_artifact_root")
    # Verify repo-relative path has no leaks
    assert_no_model_interface_local_path_leakage(metadata.training_artifact_root)

    validate_non_empty_str(metadata.training_split_contract, "training_split_contract")
    validate_non_negative_int(metadata.model_repeat_seed, "model_repeat_seed")
    validate_bool(metadata.zero_shot_mode, "zero_shot_mode")
    validate_non_negative_int(metadata.C_train_count, "C_train_count")

    # Enforce zero-shot C leakage guard
    if metadata.zero_shot_mode:
        if metadata.model_run_kind != "zero_shot":
            raise ValueError("model_run_kind must be 'zero_shot' when zero_shot_mode is True")
        if metadata.C_train_count != 0:
            raise ValueError("C_train_count must be 0 when zero_shot_mode is True")

    if metadata.model_run_kind == "zero_shot":
        if not metadata.zero_shot_mode:
            raise ValueError("zero_shot_mode must be True when model_run_kind is 'zero_shot'")
        if metadata.C_train_count != 0:
            raise ValueError("C_train_count must be 0 when model_run_kind is 'zero_shot'")

    validate_non_empty_str(metadata.config_hash, "config_hash")
    validate_non_empty_str(metadata.evidence_contract_version, "evidence_contract_version")
    validate_non_empty_str(metadata.reason, "reason")

    # Metadata record safety gates (no local paths or claims)
    meta_dict = {
        "contract_version": metadata.contract_version,
        "model_architecture_id": metadata.model_architecture_id,
        "model_run_kind": metadata.model_run_kind,
        "model_code_git_commit": metadata.model_code_git_commit,
        "training_artifact_root": metadata.training_artifact_root,
        "training_split_contract": metadata.training_split_contract,
        "model_repeat_seed": metadata.model_repeat_seed,
        "zero_shot_mode": metadata.zero_shot_mode,
        "C_train_count": metadata.C_train_count,
        "config_hash": metadata.config_hash,
        "evidence_contract_version": metadata.evidence_contract_version,
        "reason": metadata.reason,
    }
    assert_no_model_interface_local_path_leakage(meta_dict)
    assert_no_model_interface_forbidden_claims(meta_dict)
    assert_no_model_interface_raw_params(meta_dict)


def validate_model_candidate_record(record: ModelCandidateRecord) -> None:
    if type(record) is not ModelCandidateRecord:
        raise TypeError("record must be exact ModelCandidateRecord instance")

    validate_non_empty_str(record.candidate_id, "candidate_id")
    validate_model_run_metadata(record.metadata)

    if type(record.generated_spec) is not ModelSpec:
        raise TypeError("generated_spec must be exact ModelSpec instance")

    # Enforce validity pass validation gates
    validate_model_spec(record.generated_spec)
    validate_bool(record.validity_pass, "validity_pass")
    if not record.validity_pass:
        raise ValueError("validity_pass must be True since validate_model_spec passed")

    # Math validity gate check
    actual_math_valid = True
    try:
        require_math_valid(record.generated_spec)
    except (ValueError, TypeError):
        actual_math_valid = False

    validate_bool(record.math_validity_pass, "math_validity_pass")
    if record.math_validity_pass != actual_math_valid:
        raise ValueError(
            f"math_validity_pass mismatch: expected {actual_math_valid}, got {record.math_validity_pass}"
        )

    # Validate family classifications
    if record.generated_family_id != record.generated_spec.family_id.value:
        raise ValueError(
            f"generated_family_id mismatch: expected {record.generated_spec.family_id.value}, got {record.generated_family_id}"
        )
    if record.generated_mean_family != record.generated_spec.mean_family.value:
        raise ValueError(
            f"generated_mean_family mismatch: expected {record.generated_spec.mean_family.value}, got {record.generated_mean_family}"
        )
    if record.generated_volatility_family != record.generated_spec.volatility_family.value:
        raise ValueError(
            f"generated_volatility_family mismatch: expected {record.generated_spec.volatility_family.value}, got {record.generated_volatility_family}"
        )

    validate_non_empty_str(record.source_split, "source_split")
    validate_non_empty_str(record.reason, "reason")

    # Candidate safety gates (no local paths or claims)
    # MANDATORY CORRECTION: Do not check raw parameter leakage on ModelCandidateRecord because it contains the ModelSpec.
    candidate_meta = {
        "candidate_id": record.candidate_id,
        "generated_family_id": record.generated_family_id,
        "generated_mean_family": record.generated_mean_family,
        "generated_volatility_family": record.generated_volatility_family,
        "source_split": record.source_split,
        "reason": record.reason,
    }
    assert_no_model_interface_local_path_leakage(candidate_meta)
    assert_no_model_interface_forbidden_claims(candidate_meta)


def validate_model_candidate_batch(batch: ModelCandidateBatch) -> None:
    if type(batch) is not ModelCandidateBatch:
        raise TypeError("batch must be exact ModelCandidateBatch instance")

    validate_non_empty_str(batch.contract_version, "contract_version")
    if batch.contract_version != MODEL_INTERFACE_CONTRACT_VERSION:
        raise ValueError(f"Wrong contract version: expected {MODEL_INTERFACE_CONTRACT_VERSION}")

    validate_model_run_metadata(batch.metadata)

    if type(batch.candidates) is not tuple:
        raise TypeError("candidates must be a tuple")

    validate_positive_int(batch.candidate_count, "candidate_count")
    if len(batch.candidates) != batch.candidate_count:
        raise ValueError(
            f"candidates tuple length {len(batch.candidates)} does not match candidate_count {batch.candidate_count}"
        )

    seen_ids = set()
    for idx, cand in enumerate(batch.candidates):
        validate_model_candidate_record(cand)
        if cand.metadata != batch.metadata:
            raise ValueError(f"Candidate {idx} metadata does not match batch metadata")
        if cand.candidate_id in seen_ids:
            raise ValueError(f"Duplicate candidate_id found: {cand.candidate_id}")
        seen_ids.add(cand.candidate_id)

    validate_non_empty_str(batch.reason, "reason")
    assert_no_model_interface_forbidden_claims(batch.reason)


def validate_model_candidate_summary(summary: ModelCandidateSummary) -> None:
    if type(summary) is not ModelCandidateSummary:
        raise TypeError("summary must be exact ModelCandidateSummary instance")

    validate_non_empty_str(summary.contract_version, "contract_version")
    if summary.contract_version != MODEL_CANDIDATE_SUMMARY_CONTRACT_VERSION:
        raise ValueError(f"Wrong contract version: expected {MODEL_CANDIDATE_SUMMARY_CONTRACT_VERSION}")

    validate_model_architecture_id(summary.model_architecture_id)
    validate_model_run_kind(summary.model_run_kind)
    validate_non_negative_int(summary.model_repeat_seed, "model_repeat_seed")
    validate_bool(summary.zero_shot_mode, "zero_shot_mode")
    validate_non_negative_int(summary.C_train_count, "C_train_count")

    # Enforce metadata-equivalent guards
    if summary.zero_shot_mode:
        if summary.model_run_kind != "zero_shot":
            raise ValueError("model_run_kind must be 'zero_shot' when zero_shot_mode is True")
        if summary.C_train_count != 0:
            raise ValueError("C_train_count must be 0 when zero_shot_mode is True")

    if summary.model_run_kind == "zero_shot":
        if not summary.zero_shot_mode:
            raise ValueError("zero_shot_mode must be True when model_run_kind is 'zero_shot'")
        if summary.C_train_count != 0:
            raise ValueError("C_train_count must be 0 when model_run_kind is 'zero_shot'")

    validate_positive_int(summary.candidate_count, "candidate_count")

    # Check generated specification lists
    if type(summary.generated_family_ids) is not tuple:
        raise TypeError("generated_family_ids must be a tuple")
    if len(summary.generated_family_ids) != summary.candidate_count:
        raise ValueError("generated_family_ids length mismatch")
    for idx, val in enumerate(summary.generated_family_ids):
        validate_non_empty_str(val, f"generated_family_ids[{idx}]")

    if type(summary.generated_mean_families) is not tuple:
        raise TypeError("generated_mean_families must be a tuple")
    if len(summary.generated_mean_families) != summary.candidate_count:
        raise ValueError("generated_mean_families length mismatch")
    for idx, val in enumerate(summary.generated_mean_families):
        validate_non_empty_str(val, f"generated_mean_families[{idx}]")

    if type(summary.generated_volatility_families) is not tuple:
        raise TypeError("generated_volatility_families must be a tuple")
    if len(summary.generated_volatility_families) != summary.candidate_count:
        raise ValueError("generated_volatility_families length mismatch")
    for idx, val in enumerate(summary.generated_volatility_families):
        validate_non_empty_str(val, f"generated_volatility_families[{idx}]")

    validate_non_negative_int(summary.validity_pass_count, "validity_pass_count")
    if summary.validity_pass_count > summary.candidate_count:
        raise ValueError("validity_pass_count cannot exceed candidate_count")

    validate_non_negative_int(summary.math_validity_pass_count, "math_validity_pass_count")
    if summary.math_validity_pass_count > summary.candidate_count:
        raise ValueError("math_validity_pass_count cannot exceed candidate_count")

    validate_non_empty_str(summary.source_split, "source_split")
    validate_non_empty_str(summary.evidence_contract_version, "evidence_contract_version")
    validate_non_empty_str(summary.reason, "reason")

    # Enforce safety check boundaries on evidence summary fields (including raw params checks)
    sum_dict = {
        "contract_version": summary.contract_version,
        "model_architecture_id": summary.model_architecture_id,
        "model_run_kind": summary.model_run_kind,
        "model_repeat_seed": summary.model_repeat_seed,
        "zero_shot_mode": summary.zero_shot_mode,
        "C_train_count": summary.C_train_count,
        "candidate_count": summary.candidate_count,
        "generated_family_ids": list(summary.generated_family_ids),
        "generated_mean_families": list(summary.generated_mean_families),
        "generated_volatility_families": list(summary.generated_volatility_families),
        "validity_pass_count": summary.validity_pass_count,
        "math_validity_pass_count": summary.math_validity_pass_count,
        "source_split": summary.source_split,
        "evidence_contract_version": summary.evidence_contract_version,
        "reason": summary.reason,
    }
    assert_no_model_interface_local_path_leakage(sum_dict)
    assert_no_model_interface_forbidden_claims(sum_dict)
    assert_no_model_interface_raw_params(sum_dict)


# Conversion and Serialization Functions
def model_candidate_record_to_summary_fields(record: ModelCandidateRecord) -> dict:
    validate_model_candidate_record(record)

    summary_fields = {
        "candidate_id": record.candidate_id,
        "generated_family_id": record.generated_family_id,
        "generated_mean_family": record.generated_mean_family,
        "generated_volatility_family": record.generated_volatility_family,
        "validity_pass": record.validity_pass,
        "math_validity_pass": record.math_validity_pass,
        "source_split": record.source_split,
    }

    # Run safety checks on summary fields
    assert_no_model_interface_local_path_leakage(summary_fields)
    assert_no_model_interface_forbidden_claims(summary_fields)
    assert_no_model_interface_raw_params(summary_fields)

    return summary_fields


def summarize_model_candidate_batch(batch: ModelCandidateBatch) -> ModelCandidateSummary:
    validate_model_candidate_batch(batch)

    # Extract dynamic listings and pass counts
    family_ids = []
    mean_families = []
    volatility_families = []
    validity_count = 0
    math_validity_count = 0

    first_split = None
    for cand in batch.candidates:
        if first_split is None:
            first_split = cand.source_split
        elif cand.source_split != first_split:
            raise ValueError(
                f"Mixed source splits found in batch: '{first_split}' and '{cand.source_split}'"
            )

        family_ids.append(cand.generated_family_id)
        mean_families.append(cand.generated_mean_family)
        volatility_families.append(cand.generated_volatility_family)
        if cand.validity_pass:
            validity_count += 1
        if cand.math_validity_pass:
            math_validity_count += 1

    summary = ModelCandidateSummary(
        contract_version=MODEL_CANDIDATE_SUMMARY_CONTRACT_VERSION,
        model_architecture_id=batch.metadata.model_architecture_id,
        model_run_kind=batch.metadata.model_run_kind,
        model_repeat_seed=batch.metadata.model_repeat_seed,
        zero_shot_mode=batch.metadata.zero_shot_mode,
        C_train_count=batch.metadata.C_train_count,
        candidate_count=batch.candidate_count,
        generated_family_ids=tuple(family_ids),
        generated_mean_families=tuple(mean_families),
        generated_volatility_families=tuple(volatility_families),
        validity_pass_count=validity_count,
        math_validity_pass_count=math_validity_count,
        source_split=first_split if first_split is not None else "empty_batch",
        evidence_contract_version=batch.metadata.evidence_contract_version,
        reason=batch.reason,
    )

    validate_model_candidate_summary(summary)
    return summary


def model_candidate_summary_to_json_dict(summary: ModelCandidateSummary) -> dict:
    if type(summary) is not ModelCandidateSummary:
        raise TypeError("summary must be exact ModelCandidateSummary instance")
    validate_model_candidate_summary(summary)

    json_dict = {
        "contract_version": summary.contract_version,
        "model_architecture_id": summary.model_architecture_id,
        "model_run_kind": summary.model_run_kind,
        "model_repeat_seed": summary.model_repeat_seed,
        "zero_shot_mode": summary.zero_shot_mode,
        "C_train_count": summary.C_train_count,
        "candidate_count": summary.candidate_count,
        "generated_family_ids": list(summary.generated_family_ids),
        "generated_mean_families": list(summary.generated_mean_families),
        "generated_volatility_families": list(summary.generated_volatility_families),
        "validity_pass_count": summary.validity_pass_count,
        "math_validity_pass_count": summary.math_validity_pass_count,
        "source_split": summary.source_split,
        "evidence_contract_version": summary.evidence_contract_version,
        "reason": summary.reason,
    }

    # Verify safety filters prior to returning dict
    assert_no_model_interface_local_path_leakage(json_dict)
    assert_no_model_interface_forbidden_claims(json_dict)
    assert_no_model_interface_raw_params(json_dict)

    return json_dict


def compact_model_candidate_summary_json(summary: ModelCandidateSummary) -> str:
    return json.dumps(
        model_candidate_summary_to_json_dict(summary),
        sort_keys=True,
        separators=(",", ":"),
    )
