# src/phase2/evidence.py

import dataclasses
import json
import math
from typing import Any, Tuple, Optional, Union

# Public constants
EVIDENCE_CONTRACT_VERSION = "phase2_p23_evidence_contract_v1"
EVIDENCE_RUN_KIND_BASELINE_ARTIFACT_SMOKE = "baseline_artifact_smoke"

APPROVED_EVIDENCE_RUN_KINDS = (
    "baseline_artifact_smoke",
)

APPROVED_EVIDENCE_BASELINE_NAMES = (
    "copy_reference",
    "random_valid",
    "structural_composition_oracle",
)

FORBIDDEN_EVIDENCE_RAW_PARAM_KEYS = (
    "ar_params",
    "ma_params",
    "alpha_params",
    "beta_params",
    "omega",
    "constraint_flags",
    "model_spec",
    "candidate_specs",
    "reference_specs",
)

FORBIDDEN_EVIDENCE_CLAIM_WORDS = (
    "best",
    "winner",
    "solved",
    "final comparison",
    "scientific success",
    "model works",
)


# Public Dataclasses
@dataclasses.dataclass(frozen=True)
class EvidenceRunMetadata:
    contract_version: str
    run_kind: str
    source_phase: str
    artifact_root: str
    p14_audit_verified: bool
    no_artifact_generation: bool
    no_model_training: bool
    no_final_comparison: bool
    no_scientific_conclusion: bool
    reason: str


@dataclasses.dataclass(frozen=True)
class EvidenceReferenceRecord:
    role: str
    sample_id: str
    line_number: int
    family_id: str
    mean_family: str
    volatility_family: str


@dataclasses.dataclass(frozen=True)
class EvidenceMetricRecord:
    validity_valid_count: int
    validity_total_count: int
    validity_valid_rate: float
    composition_pass_count: int
    composition_pass_rate: float
    novelty_pass_count: int
    novelty_pass_rate: float
    distribution_mmd_rbf: Optional[float]
    seed_stability_present: bool
    candidate_record_count: int


@dataclasses.dataclass(frozen=True)
class EvidenceGenerationRecord:
    candidate_count: int
    candidate_family_ids: Tuple[str, ...]
    source_reference_indices: Tuple[Tuple[int, ...], ...]
    seed_sequence: Tuple[Optional[int], ...]


@dataclasses.dataclass(frozen=True)
class EvidenceBaselineRecord:
    baseline_name: str
    generation: EvidenceGenerationRecord
    metrics: EvidenceMetricRecord
    metric_bundle_bridge_verified: bool
    reason: str


@dataclasses.dataclass(frozen=True)
class EvidenceBundle:
    metadata: EvidenceRunMetadata
    loaded_reference_count: int
    references: Tuple[EvidenceReferenceRecord, ...]
    baseline_count: int
    candidate_count_per_baseline: int
    baselines: Tuple[EvidenceBaselineRecord, ...]
    verdict: str


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


def validate_probability(value: Any, name: str) -> None:
    if type(value) not in (int, float) or isinstance(value, bool):
        raise TypeError(f"{name} must be numeric, got {type(value).__name__}")
    if not math.isfinite(value):
        raise ValueError(f"{name} must be finite")
    if not (0.0 <= value <= 1.0):
        raise ValueError(f"{name} must be in [0.0, 1.0]")


def validate_optional_finite_float(value: Any, name: str) -> None:
    if value is None:
        return
    if type(value) not in (int, float) or isinstance(value, bool):
        raise TypeError(f"{name} must be numeric or None, got {type(value).__name__}")
    if not math.isfinite(value):
        raise ValueError(f"{name} must be finite")


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


def assert_no_raw_params(data: Any) -> None:
    if isinstance(data, str):
        for forbidden in FORBIDDEN_EVIDENCE_RAW_PARAM_KEYS:
            if forbidden == data or f"'{forbidden}'" in data or f'"{forbidden}"' in data or f"{forbidden}:" in data or f"{forbidden}=" in data:
                raise ValueError(f"Raw parameter leakage key '{forbidden}' found in text: '{data}'")
    elif isinstance(data, dict):
        for k, v in data.items():
            if k in FORBIDDEN_EVIDENCE_RAW_PARAM_KEYS:
                raise ValueError(f"Forbidden raw parameter key '{k}' found in dictionary keys")
            assert_no_raw_params(k)
            assert_no_raw_params(v)
    elif isinstance(data, (list, tuple)):
        for item in data:
            assert_no_raw_params(item)


def assert_no_forbidden_claims(data: Any) -> None:
    if isinstance(data, str):
        normalized = data.lower()
        # Clean allowed flags first to avoid false positives
        cleaned = normalized.replace("no_final_comparison", "").replace("no_scientific_conclusion", "")
        for word in FORBIDDEN_EVIDENCE_CLAIM_WORDS:
            if word in cleaned:
                raise ValueError(f"Forbidden claim word '{word}' found in: '{data}'")
    elif isinstance(data, dict):
        for k, v in data.items():
            assert_no_forbidden_claims(k)
            assert_no_forbidden_claims(v)
    elif isinstance(data, (list, tuple)):
        for item in data:
            assert_no_forbidden_claims(item)


def validate_evidence_run_metadata(metadata: EvidenceRunMetadata) -> None:
    if type(metadata) is not EvidenceRunMetadata:
        raise TypeError("metadata must be exact EvidenceRunMetadata instance")
        
    validate_non_empty_str(metadata.contract_version, "contract_version")
    if metadata.contract_version != EVIDENCE_CONTRACT_VERSION:
        raise ValueError(f"Wrong contract version: expected {EVIDENCE_CONTRACT_VERSION}")
        
    validate_non_empty_str(metadata.run_kind, "run_kind")
    if metadata.run_kind not in APPROVED_EVIDENCE_RUN_KINDS:
        raise ValueError(f"Unsupported run kind: {metadata.run_kind}")
        
    validate_non_empty_str(metadata.source_phase, "source_phase")
    validate_non_empty_str(metadata.artifact_root, "artifact_root")
    # Verify repo-relative path has no leaks
    assert_no_local_path_leakage(metadata.artifact_root)
    
    validate_bool(metadata.p14_audit_verified, "p14_audit_verified")
    
    validate_bool(metadata.no_artifact_generation, "no_artifact_generation")
    if not metadata.no_artifact_generation:
        raise ValueError("no_artifact_generation must be True")
        
    validate_bool(metadata.no_model_training, "no_model_training")
    if not metadata.no_model_training:
        raise ValueError("no_model_training must be True")
        
    validate_bool(metadata.no_final_comparison, "no_final_comparison")
    if not metadata.no_final_comparison:
        raise ValueError("no_final_comparison must be True")
        
    validate_bool(metadata.no_scientific_conclusion, "no_scientific_conclusion")
    if not metadata.no_scientific_conclusion:
        raise ValueError("no_scientific_conclusion must be True")
        
    validate_non_empty_str(metadata.reason, "reason")


def validate_evidence_reference_record(record: EvidenceReferenceRecord) -> None:
    if type(record) is not EvidenceReferenceRecord:
        raise TypeError("record must be exact EvidenceReferenceRecord instance")
        
    validate_non_empty_str(record.role, "role")
    if record.role not in ("arma_mean_source", "garch_volatility_source"):
        raise ValueError(f"Invalid reference role: {record.role}")
        
    validate_non_empty_str(record.sample_id, "sample_id")
    validate_positive_int(record.line_number, "line_number")
    validate_non_empty_str(record.family_id, "family_id")
    validate_non_empty_str(record.mean_family, "mean_family")
    validate_non_empty_str(record.volatility_family, "volatility_family")
    
    # Record serialization safety
    ref_dict = {
        "role": record.role,
        "sample_id": record.sample_id,
        "line_number": record.line_number,
        "family_id": record.family_id,
        "mean_family": record.mean_family,
        "volatility_family": record.volatility_family
    }
    assert_no_raw_params(ref_dict)


def validate_evidence_metric_record(record: EvidenceMetricRecord) -> None:
    if type(record) is not EvidenceMetricRecord:
        raise TypeError("record must be exact EvidenceMetricRecord instance")
        
    validate_non_negative_int(record.validity_valid_count, "validity_valid_count")
    validate_positive_int(record.validity_total_count, "validity_total_count")
    validate_probability(record.validity_valid_rate, "validity_valid_rate")
    
    validate_non_negative_int(record.composition_pass_count, "composition_pass_count")
    validate_probability(record.composition_pass_rate, "composition_pass_rate")
    
    validate_non_negative_int(record.novelty_pass_count, "novelty_pass_count")
    validate_probability(record.novelty_pass_rate, "novelty_pass_rate")
    
    validate_optional_finite_float(record.distribution_mmd_rbf, "distribution_mmd_rbf")
    validate_bool(record.seed_stability_present, "seed_stability_present")
    validate_positive_int(record.candidate_record_count, "candidate_record_count")


def validate_evidence_generation_record(record: EvidenceGenerationRecord) -> None:
    if type(record) is not EvidenceGenerationRecord:
        raise TypeError("record must be exact EvidenceGenerationRecord instance")
        
    validate_positive_int(record.candidate_count, "candidate_count")
    
    if type(record.candidate_family_ids) is not tuple:
        raise TypeError("candidate_family_ids must be a tuple")
    if len(record.candidate_family_ids) != record.candidate_count:
        raise ValueError("candidate_family_ids length mismatch")
    for idx, item in enumerate(record.candidate_family_ids):
        validate_non_empty_str(item, f"candidate_family_ids[{idx}]")
        
    if type(record.source_reference_indices) is not tuple:
        raise TypeError("source_reference_indices must be a tuple")
    if len(record.source_reference_indices) != record.candidate_count:
        raise ValueError("source_reference_indices length mismatch")
    for idx, item in enumerate(record.source_reference_indices):
        if type(item) is not tuple:
            raise TypeError(f"source_reference_indices[{idx}] must be a tuple")
        for val_idx, val in enumerate(item):
            validate_non_negative_int(val, f"source_reference_indices[{idx}][{val_idx}]")
            
    if type(record.seed_sequence) is not tuple:
        raise TypeError("seed_sequence must be a tuple")
    if len(record.seed_sequence) != record.candidate_count:
        raise ValueError("seed_sequence length mismatch")
    for idx, item in enumerate(record.seed_sequence):
        if item is not None:
            validate_non_negative_int(item, f"seed_sequence[{idx}]")


def validate_evidence_baseline_record(record: EvidenceBaselineRecord) -> None:
    if type(record) is not EvidenceBaselineRecord:
        raise TypeError("record must be exact EvidenceBaselineRecord instance")
        
    validate_non_empty_str(record.baseline_name, "baseline_name")
    if record.baseline_name not in APPROVED_EVIDENCE_BASELINE_NAMES:
        raise ValueError(f"Unsupported baseline: {record.baseline_name}")
        
    validate_evidence_generation_record(record.generation)
    validate_evidence_metric_record(record.metrics)
    
    if record.generation.candidate_count != record.metrics.candidate_record_count:
        raise ValueError("Candidate count mismatch between generation and metrics")
        
    validate_bool(record.metric_bundle_bridge_verified, "metric_bundle_bridge_verified")
    if not record.metric_bundle_bridge_verified:
        raise ValueError("metric_bundle_bridge_verified must be True")
        
    validate_non_empty_str(record.reason, "reason")
    assert_no_forbidden_claims(record.reason)


def validate_evidence_bundle(bundle: EvidenceBundle) -> None:
    if type(bundle) is not EvidenceBundle:
        raise TypeError("bundle must be exact EvidenceBundle instance")
        
    validate_evidence_run_metadata(bundle.metadata)
    validate_positive_int(bundle.loaded_reference_count, "loaded_reference_count")
    
    if type(bundle.references) is not tuple:
        raise TypeError("references must be a tuple")
    if len(bundle.references) != bundle.loaded_reference_count:
        raise ValueError("references length mismatch")
    for idx, ref in enumerate(bundle.references):
        validate_evidence_reference_record(ref)
        
    validate_positive_int(bundle.baseline_count, "baseline_count")
    if type(bundle.baselines) is not tuple:
        raise TypeError("baselines must be a tuple")
    if len(bundle.baselines) != bundle.baseline_count:
        raise ValueError("baselines length mismatch")
        
    validate_positive_int(bundle.candidate_count_per_baseline, "candidate_count_per_baseline")
    
    for idx, b in enumerate(bundle.baselines):
        validate_evidence_baseline_record(b)
        if b.generation.candidate_count != bundle.candidate_count_per_baseline:
            raise ValueError(f"Baseline {b.baseline_name} candidate count does not match bundle expectation")
            
    baseline_names = tuple(b.baseline_name for b in bundle.baselines)
    expected_names = ("copy_reference", "random_valid", "structural_composition_oracle")
    if baseline_names != expected_names:
        raise ValueError(f"Baseline names/order mismatch: expected {expected_names}, got {baseline_names}")
        
    validate_non_empty_str(bundle.verdict, "verdict")
    if bundle.verdict != "PASS":
        raise ValueError("Verdict must be PASS")
        
    # Serialize unchecked to avoid infinite loops, then run static safety checks
    serialized = _evidence_bundle_to_json_dict_unchecked(bundle)
    assert_no_local_path_leakage(serialized)
    assert_no_raw_params(serialized)
    assert_no_forbidden_claims(serialized)


# Conversion and Serialization Functions
def evidence_metric_record_from_summary(summary: dict) -> EvidenceMetricRecord:
    if type(summary) is not dict:
        raise TypeError("summary must be a dict")
        
    expected_keys = (
        "validity_valid_count",
        "validity_total_count",
        "validity_valid_rate",
        "composition_pass_count",
        "composition_pass_rate",
        "novelty_pass_count",
        "novelty_pass_rate",
        "distribution_mmd_rbf",
        "seed_stability_present",
        "candidate_record_count"
    )
    for k in expected_keys:
        if k not in summary:
            raise KeyError(f"Missing expected metric key: {k}")
            
    # Protect against raw parameter leakage keys
    for forbidden in FORBIDDEN_EVIDENCE_RAW_PARAM_KEYS:
        if forbidden in summary:
            raise ValueError(f"Leakage of raw candidate param key '{forbidden}' found in metric summary")
            
    return EvidenceMetricRecord(
        validity_valid_count=summary["validity_valid_count"],
        validity_total_count=summary["validity_total_count"],
        validity_valid_rate=summary["validity_valid_rate"],
        composition_pass_count=summary["composition_pass_count"],
        composition_pass_rate=summary["composition_pass_rate"],
        novelty_pass_count=summary["novelty_pass_count"],
        novelty_pass_rate=summary["novelty_pass_rate"],
        distribution_mmd_rbf=summary["distribution_mmd_rbf"],
        seed_stability_present=summary["seed_stability_present"],
        candidate_record_count=summary["candidate_record_count"]
    )


def evidence_generation_record_from_summary(summary: dict) -> EvidenceGenerationRecord:
    if type(summary) is not dict:
        raise TypeError("summary must be a dict")
        
    expected_keys = (
        "candidate_count",
        "candidate_family_ids",
        "source_records_summary"
    )
    for k in expected_keys:
        if k not in summary:
            raise KeyError(f"Missing expected generation key: {k}")
            
    candidate_count = summary["candidate_count"]
    candidate_family_ids = tuple(summary["candidate_family_ids"])
    
    source_records = summary["source_records_summary"]
    if type(source_records) is not list:
        raise TypeError("source_records_summary must be a list")
    if len(source_records) != candidate_count:
        raise ValueError("source_records_summary length mismatch")
        
    ref_indices = []
    seeds = []
    
    for idx, rec in enumerate(source_records):
        if type(rec) is not dict:
            raise TypeError(f"source_records_summary[{idx}] must be a dict")
        if "source_reference_indices" not in rec or "seed" not in rec:
            raise KeyError(f"Missing indices/seed in record {idx}")
        ref_indices.append(tuple(rec["source_reference_indices"]))
        seeds.append(rec["seed"])
        
    return EvidenceGenerationRecord(
        candidate_count=candidate_count,
        candidate_family_ids=candidate_family_ids,
        source_reference_indices=tuple(ref_indices),
        seed_sequence=tuple(seeds)
    )


def evidence_reference_records_from_p22_summary(summary: dict) -> Tuple[EvidenceReferenceRecord, ...]:
    if type(summary) is not dict:
        raise TypeError("summary must be a dict")
    if "references" not in summary:
        raise KeyError("Missing 'references' in reference summary")
        
    refs_list = summary["references"]
    if type(refs_list) is not list:
        raise TypeError("references must be a list")
        
    records = []
    for idx, rec in enumerate(refs_list):
        if type(rec) is not dict:
            raise TypeError(f"reference[{idx}] must be a dict")
        expected_keys = ("role", "sample_id", "line_number", "family_id", "mean_family", "volatility_family")
        for k in expected_keys:
            if k not in rec:
                raise KeyError(f"Missing reference key: {k}")
                
        # Guard against raw parameter leakage keys
        for forbidden in FORBIDDEN_EVIDENCE_RAW_PARAM_KEYS:
            if forbidden in rec:
                raise ValueError(f"Leakage of raw candidate param key '{forbidden}' found in reference record")
                
        records.append(
            EvidenceReferenceRecord(
                role=rec["role"],
                sample_id=rec["sample_id"],
                line_number=rec["line_number"],
                family_id=rec["family_id"],
                mean_family=rec["mean_family"],
                volatility_family=rec["volatility_family"]
            )
        )
    return tuple(records)


def normalize_p22_smoke_summary_to_evidence_bundle(summary: dict) -> EvidenceBundle:
    if type(summary) is not dict:
        raise TypeError("summary must be a dict")
        
    # Validate P22 summary context
    if summary.get("contract") != "phase2_p22_artifact_backed_baseline_smoke_v1":
        raise ValueError(f"Unsupported summary contract: {summary.get('contract')}")
    if summary.get("verdict") != "PASS":
        raise ValueError(f"P22 verdict was not PASS: {summary.get('verdict')}")
    if summary.get("p14_audit_verified") is not True:
        raise ValueError("P22 manifest audit was not verified")
    if summary.get("no_artifact_generation") is not True:
        raise ValueError("no_artifact_generation was not True")
    if summary.get("no_model_training") is not True:
        raise ValueError("no_model_training was not True")
    if summary.get("no_final_comparison") is not True:
        raise ValueError("no_final_comparison was not True")
    if summary.get("no_scientific_conclusion") is not True:
        raise ValueError("no_scientific_conclusion was not True")
        
    # Build Metadata
    metadata = EvidenceRunMetadata(
        contract_version=EVIDENCE_CONTRACT_VERSION,
        run_kind=EVIDENCE_RUN_KIND_BASELINE_ARTIFACT_SMOKE,
        source_phase="P22",
        artifact_root=summary["artifact_root"],
        p14_audit_verified=summary["p14_audit_verified"],
        no_artifact_generation=summary["no_artifact_generation"],
        no_model_training=summary["no_model_training"],
        no_final_comparison=summary["no_final_comparison"],
        no_scientific_conclusion=summary["no_scientific_conclusion"],
        reason="normalized_from_p22_artifact_backed_baseline_smoke"
    )
    
    # Normalize References
    references = evidence_reference_records_from_p22_summary(summary["reference_summary"])
    
    # Normalize Baselines
    baselines_summary = summary["baselines"]
    if type(baselines_summary) is not list:
        raise TypeError("baselines must be a list")
        
    baselines = []
    for b_summary in baselines_summary:
        if type(b_summary) is not dict:
            raise TypeError("each baseline entry must be a dict")
        baseline_name = b_summary["baseline_name"]
        
        gen = evidence_generation_record_from_summary(b_summary["generation_summary"])
        metrics = evidence_metric_record_from_summary(b_summary["evaluation_summary"])
        
        baselines.append(
            EvidenceBaselineRecord(
                baseline_name=baseline_name,
                generation=gen,
                metrics=metrics,
                metric_bundle_bridge_verified=b_summary["metric_bundle_bridge_verified"],
                reason=b_summary.get("reason", f"normalized_baseline_{baseline_name}")
            )
        )
        
    bundle = EvidenceBundle(
        metadata=metadata,
        loaded_reference_count=summary["loaded_reference_count"],
        references=references,
        baseline_count=summary["baseline_count"],
        candidate_count_per_baseline=summary["candidate_count_per_baseline"],
        baselines=tuple(baselines),
        verdict="PASS"
    )
    
    # Validate before returning
    validate_evidence_bundle(bundle)
    return bundle


def _evidence_bundle_to_json_dict_unchecked(bundle: EvidenceBundle) -> dict:
    meta = bundle.metadata
    meta_dict = {
        "contract_version": meta.contract_version,
        "run_kind": meta.run_kind,
        "source_phase": meta.source_phase,
        "artifact_root": meta.artifact_root,
        "p14_audit_verified": meta.p14_audit_verified,
        "no_artifact_generation": meta.no_artifact_generation,
        "no_model_training": meta.no_model_training,
        "no_final_comparison": meta.no_final_comparison,
        "no_scientific_conclusion": meta.no_scientific_conclusion,
        "reason": meta.reason
    }
    
    refs_list = []
    for ref in bundle.references:
        refs_list.append({
            "role": ref.role,
            "sample_id": ref.sample_id,
            "line_number": ref.line_number,
            "family_id": ref.family_id,
            "mean_family": ref.mean_family,
            "volatility_family": ref.volatility_family
        })
        
    baselines_list = []
    for b in bundle.baselines:
        gen = b.generation
        gen_dict = {
            "candidate_count": gen.candidate_count,
            "candidate_family_ids": list(gen.candidate_family_ids),
            "source_reference_indices": [list(item) for item in gen.source_reference_indices],
            "seed_sequence": list(gen.seed_sequence)
        }
        metrics = b.metrics
        metrics_dict = {
            "validity_valid_count": metrics.validity_valid_count,
            "validity_total_count": metrics.validity_total_count,
            "validity_valid_rate": metrics.validity_valid_rate,
            "composition_pass_count": metrics.composition_pass_count,
            "composition_pass_rate": metrics.composition_pass_rate,
            "novelty_pass_count": metrics.novelty_pass_count,
            "novelty_pass_rate": metrics.novelty_pass_rate,
            "distribution_mmd_rbf": metrics.distribution_mmd_rbf,
            "seed_stability_present": metrics.seed_stability_present,
            "candidate_record_count": metrics.candidate_record_count
        }
        baselines_list.append({
            "baseline_name": b.baseline_name,
            "generation": gen_dict,
            "metrics": metrics_dict,
            "metric_bundle_bridge_verified": b.metric_bundle_bridge_verified,
            "reason": b.reason
        })
        
    return {
        "metadata": meta_dict,
        "loaded_reference_count": bundle.loaded_reference_count,
        "references": refs_list,
        "baseline_count": bundle.baseline_count,
        "candidate_count_per_baseline": bundle.candidate_count_per_baseline,
        "baselines": baselines_list,
        "verdict": bundle.verdict
    }


def evidence_bundle_to_json_dict(bundle: EvidenceBundle) -> dict:
    if type(bundle) is not EvidenceBundle:
        raise TypeError("bundle must be exact EvidenceBundle instance")
    # Field-level validation first to check correctness
    validate_evidence_bundle(bundle)
    return _evidence_bundle_to_json_dict_unchecked(bundle)


def compact_evidence_json(bundle: EvidenceBundle) -> str:
    return json.dumps(evidence_bundle_to_json_dict(bundle), sort_keys=True, separators=(",", ":"))
