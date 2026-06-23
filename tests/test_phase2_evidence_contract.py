# tests/test_phase2_evidence_contract.py

import dataclasses
import json
import math
import pathlib
import pytest
from dataclasses import is_dataclass

import src.phase2
from src.phase2.evidence import (
    EVIDENCE_CONTRACT_VERSION,
    EVIDENCE_RUN_KIND_BASELINE_ARTIFACT_SMOKE,
    APPROVED_EVIDENCE_RUN_KINDS,
    APPROVED_EVIDENCE_BASELINE_NAMES,
    FORBIDDEN_EVIDENCE_RAW_PARAM_KEYS,
    FORBIDDEN_EVIDENCE_CLAIM_WORDS,
    EvidenceRunMetadata,
    EvidenceReferenceRecord,
    EvidenceMetricRecord,
    EvidenceGenerationRecord,
    EvidenceBaselineRecord,
    EvidenceBundle,
    validate_non_empty_str,
    validate_bool,
    validate_non_negative_int,
    validate_positive_int,
    validate_probability,
    validate_optional_finite_float,
    assert_no_local_path_leakage,
    assert_no_raw_params,
    assert_no_forbidden_claims,
    validate_evidence_run_metadata,
    validate_evidence_reference_record,
    validate_evidence_metric_record,
    validate_evidence_generation_record,
    validate_evidence_baseline_record,
    validate_evidence_bundle,
    evidence_metric_record_from_summary,
    evidence_generation_record_from_summary,
    evidence_reference_records_from_p22_summary,
    normalize_p22_smoke_summary_to_evidence_bundle,
    evidence_bundle_to_json_dict,
    compact_evidence_json,
)


# Helper fixtures
@pytest.fixture
def valid_metadata():
    return EvidenceRunMetadata(
        contract_version=EVIDENCE_CONTRACT_VERSION,
        run_kind=EVIDENCE_RUN_KIND_BASELINE_ARTIFACT_SMOKE,
        source_phase="P22",
        artifact_root="phase2_artifacts/p14_smoke_dry_run",
        p14_audit_verified=True,
        no_artifact_generation=True,
        no_model_training=True,
        no_final_comparison=True,
        no_scientific_conclusion=True,
        reason="p22_smoke_normalized"
    )


@pytest.fixture
def valid_references():
    ref1 = EvidenceReferenceRecord(
        role="arma_mean_source",
        sample_id="ARMA_0001",
        line_number=10,
        family_id="ARMA",
        mean_family="ARMA",
        volatility_family="NONE"
    )
    ref2 = EvidenceReferenceRecord(
        role="garch_volatility_source",
        sample_id="GARCH_0001",
        line_number=20,
        family_id="GARCH",
        mean_family="NONE",
        volatility_family="GARCH"
    )
    return (ref1, ref2)


@pytest.fixture
def valid_generation():
    return EvidenceGenerationRecord(
        candidate_count=6,
        candidate_family_ids=("ARMA", "GARCH", "ARMA", "GARCH", "ARMA", "GARCH"),
        source_reference_indices=((0,), (1,), (0,), (1,), (0,), (1,)),
        seed_sequence=(None, None, None, None, None, None)
    )


@pytest.fixture
def valid_metrics():
    return EvidenceMetricRecord(
        validity_valid_count=6,
        validity_total_count=6,
        validity_valid_rate=1.0,
        composition_pass_count=0,
        composition_pass_rate=0.0,
        novelty_pass_count=0,
        novelty_pass_rate=0.0,
        distribution_mmd_rbf=0.0,
        seed_stability_present=False,
        candidate_record_count=6
    )


@pytest.fixture
def valid_bundle(valid_metadata, valid_references, valid_generation, valid_metrics):
    b1 = EvidenceBaselineRecord(
        baseline_name="copy_reference",
        generation=valid_generation,
        metrics=valid_metrics,
        metric_bundle_bridge_verified=True,
        reason="clean copy references"
    )
    b2 = EvidenceBaselineRecord(
        baseline_name="random_valid",
        generation=valid_generation,
        metrics=valid_metrics,
        metric_bundle_bridge_verified=True,
        reason="clean random valid"
    )
    b3 = EvidenceBaselineRecord(
        baseline_name="structural_composition_oracle",
        generation=valid_generation,
        metrics=valid_metrics,
        metric_bundle_bridge_verified=True,
        reason="clean oracle composition"
    )
    return EvidenceBundle(
        metadata=valid_metadata,
        loaded_reference_count=2,
        references=valid_references,
        baseline_count=3,
        candidate_count_per_baseline=6,
        baselines=(b1, b2, b3),
        verdict="PASS"
    )


# A. Constants/dataclasses
def test_p23_01_constants_values():
    assert EVIDENCE_CONTRACT_VERSION == "phase2_p23_evidence_contract_v1"
    assert APPROVED_EVIDENCE_RUN_KINDS == ("baseline_artifact_smoke",)
    assert APPROVED_EVIDENCE_BASELINE_NAMES == ("copy_reference", "random_valid", "structural_composition_oracle")


def test_p23_02_dataclasses_frozen():
    classes = [
        EvidenceRunMetadata,
        EvidenceReferenceRecord,
        EvidenceMetricRecord,
        EvidenceGenerationRecord,
        EvidenceBaselineRecord,
        EvidenceBundle
    ]
    for cls in classes:
        assert is_dataclass(cls)
        # Check that we cannot set attribute to verify it is frozen
        inst = None
        if cls is EvidenceRunMetadata:
            inst = EvidenceRunMetadata(
                contract_version="", run_kind="", source_phase="", artifact_root="",
                p14_audit_verified=False, no_artifact_generation=False,
                no_model_training=False, no_final_comparison=False,
                no_scientific_conclusion=False, reason=""
            )
        if inst is not None:
            with pytest.raises(Exception):
                inst.reason = "new"


# B. Validation helpers
def test_p23_03_validate_non_empty_str_accepts():
    validate_non_empty_str("valid_string", "test_field")


def test_p23_04_validate_non_empty_str_rejects():
    with pytest.raises(TypeError):
        validate_non_empty_str(123, "test_field")
    with pytest.raises(ValueError):
        validate_non_empty_str("", "test_field")
    with pytest.raises(ValueError):
        validate_non_empty_str("  padded  ", "test_field")


def test_p23_05_validate_bool_accepts_exact_bool():
    validate_bool(True, "field")
    validate_bool(False, "field")
    with pytest.raises(TypeError):
        validate_bool(1, "field")


def test_p23_06_validate_non_negative_int_rejects():
    validate_non_negative_int(0, "field")
    validate_non_negative_int(10, "field")
    with pytest.raises(TypeError):
        validate_non_negative_int(True, "field")
    with pytest.raises(TypeError):
        validate_non_negative_int(2.5, "field")
    with pytest.raises(ValueError):
        validate_non_negative_int(-1, "field")


def test_p23_07_validate_positive_int_rejects():
    validate_positive_int(1, "field")
    with pytest.raises(TypeError):
        validate_positive_int(True, "field")
    with pytest.raises(ValueError):
        validate_positive_int(0, "field")
    with pytest.raises(ValueError):
        validate_positive_int(-5, "field")


def test_p23_08_validate_probability_accepts():
    validate_probability(0.0, "field")
    validate_probability(1.0, "field")
    validate_probability(0.5, "field")


def test_p23_09_validate_probability_rejects():
    with pytest.raises(TypeError):
        validate_probability(True, "field")
    with pytest.raises(ValueError):
        validate_probability(float("nan"), "field")
    with pytest.raises(ValueError):
        validate_probability(float("inf"), "field")
    with pytest.raises(ValueError):
        validate_probability(-0.01, "field")
    with pytest.raises(ValueError):
        validate_probability(1.01, "field")


def test_p23_10_validate_optional_finite_float_accepts():
    validate_optional_finite_float(None, "field")
    validate_optional_finite_float(0.123, "field")


def test_p23_11_validate_optional_finite_float_rejects():
    with pytest.raises(TypeError):
        validate_optional_finite_float(True, "field")
    with pytest.raises(ValueError):
        validate_optional_finite_float(float("nan"), "field")
    with pytest.raises(ValueError):
        validate_optional_finite_float(float("-inf"), "field")


def test_p23_12_assert_no_local_path_leakage():
    # Accepts repo-relative
    assert_no_local_path_leakage("phase2_artifacts/p14_smoke_dry_run/samples.jsonl")
    assert_no_local_path_leakage({"path": "src/phase2/evidence.py"})
    
    # Rejects local paths
    with pytest.raises(ValueError):
        assert_no_local_path_leakage("C:/Users/name/repo")
    with pytest.raises(ValueError):
        assert_no_local_path_leakage("/home/user/repo")
    with pytest.raises(ValueError):
        assert_no_local_path_leakage("file:///C:/Users/name")


def test_p23_13_assert_no_raw_params():
    # Accepts safe data
    assert_no_raw_params({"loaded_reference_count": 2, "sample_id": "AR_sample"})
    
    # Rejects raw param keys
    with pytest.raises(ValueError):
        assert_no_raw_params({"ar_params": [1.0]})
    with pytest.raises(ValueError):
        assert_no_raw_params({"some_dict": {"omega": 0.5}})
    with pytest.raises(ValueError):
        assert_no_raw_params("Forbidden raw key: ar_params=something")


def test_p23_14_assert_no_forbidden_claims():
    # Accepts allowed scope flags
    assert_no_forbidden_claims("no_final_comparison")
    assert_no_forbidden_claims("no_scientific_conclusion")
    assert_no_forbidden_claims({"key": "no_final_comparison", "val": "no_scientific_conclusion"})
    
    # Rejects forbidden words
    with pytest.raises(ValueError):
        assert_no_forbidden_claims("This baseline is the best one")
    with pytest.raises(ValueError):
        assert_no_forbidden_claims("We solved the task successfully")


# C. Record validation
def test_p23_15_validate_evidence_run_metadata_accepts(valid_metadata):
    validate_evidence_run_metadata(valid_metadata)


def test_p23_16_metadata_rejects_wrong_contract_version(valid_metadata):
    meta = dataclasses.replace(valid_metadata, contract_version="wrong_v2")
    with pytest.raises(ValueError, match="Wrong contract version"):
        validate_evidence_run_metadata(meta)


def test_p23_17_metadata_rejects_no_artifact_generation_false(valid_metadata):
    meta = dataclasses.replace(valid_metadata, no_artifact_generation=False)
    with pytest.raises(ValueError, match="no_artifact_generation must be True"):
        validate_evidence_run_metadata(meta)


def test_p23_18_validate_evidence_reference_record_accepts(valid_references):
    validate_evidence_reference_record(valid_references[0])
    validate_evidence_reference_record(valid_references[1])


def test_p23_19_reference_record_rejects_bad_role(valid_references):
    ref = dataclasses.replace(valid_references[0], role="unknown_role")
    with pytest.raises(ValueError, match="Invalid reference role"):
        validate_evidence_reference_record(ref)


def test_p23_20_validate_evidence_metric_record_accepts(valid_metrics):
    validate_evidence_metric_record(valid_metrics)


def test_p23_21_metric_record_rejects_invalid_probabilities_counts(valid_metrics):
    with pytest.raises(ValueError):
        validate_evidence_metric_record(dataclasses.replace(valid_metrics, validity_valid_rate=1.05))
    with pytest.raises(ValueError):
        validate_evidence_metric_record(dataclasses.replace(valid_metrics, validity_valid_count=-1))


def test_p23_22_validate_evidence_generation_record_accepts(valid_generation):
    validate_evidence_generation_record(valid_generation)


def test_p23_23_generation_record_rejects_length_mismatch(valid_generation):
    # Short family ids tuple
    gen = dataclasses.replace(valid_generation, candidate_family_ids=("ARMA", "GARCH"))
    with pytest.raises(ValueError, match="candidate_family_ids length mismatch"):
        validate_evidence_generation_record(gen)


def test_p23_24_validate_evidence_baseline_record_accepts(valid_generation, valid_metrics):
    rec = EvidenceBaselineRecord(
        baseline_name="random_valid",
        generation=valid_generation,
        metrics=valid_metrics,
        metric_bundle_bridge_verified=True,
        reason="valid baseline reason"
    )
    validate_evidence_baseline_record(rec)


def test_p23_25_baseline_record_rejects_unsupported_name(valid_generation, valid_metrics):
    rec = EvidenceBaselineRecord(
        baseline_name="unsupported_baseline",
        generation=valid_generation,
        metrics=valid_metrics,
        metric_bundle_bridge_verified=True,
        reason="reason"
    )
    with pytest.raises(ValueError, match="Unsupported baseline"):
        validate_evidence_baseline_record(rec)


def test_p23_26_baseline_record_rejects_bridge_verified_false(valid_generation, valid_metrics):
    rec = EvidenceBaselineRecord(
        baseline_name="copy_reference",
        generation=valid_generation,
        metrics=valid_metrics,
        metric_bundle_bridge_verified=False,
        reason="reason"
    )
    with pytest.raises(ValueError, match="metric_bundle_bridge_verified must be True"):
        validate_evidence_baseline_record(rec)


def test_p23_27_validate_evidence_bundle_accepts(valid_bundle):
    validate_evidence_bundle(valid_bundle)


def test_p23_28_evidence_bundle_rejects_baseline_order(valid_bundle):
    # Swap baselines to corrupt expected order
    baselines = (valid_bundle.baselines[1], valid_bundle.baselines[0], valid_bundle.baselines[2])
    bundle = dataclasses.replace(valid_bundle, baselines=baselines)
    with pytest.raises(ValueError, match="Baseline names/order mismatch"):
        validate_evidence_bundle(bundle)


def test_p23_29_evidence_bundle_rejects_raw_param_leakage(valid_bundle):
    # Insert forbidden raw key into reason
    b = dataclasses.replace(valid_bundle.baselines[0], reason="ar_params=leaked")
    baselines = (b, valid_bundle.baselines[1], valid_bundle.baselines[2])
    bundle = dataclasses.replace(valid_bundle, baselines=baselines)
    with pytest.raises(ValueError, match="Raw parameter leakage key"):
        validate_evidence_bundle(bundle)


# D. Conversion
def test_p23_30_evidence_metric_record_from_summary():
    summary = {
        "validity_valid_count": 6,
        "validity_total_count": 6,
        "validity_valid_rate": 1.0,
        "composition_pass_count": 2,
        "composition_pass_rate": 0.33,
        "novelty_pass_count": 0,
        "novelty_pass_rate": 0.0,
        "distribution_mmd_rbf": 0.45,
        "seed_stability_present": False,
        "candidate_record_count": 6
    }
    rec = evidence_metric_record_from_summary(summary)
    assert rec.validity_valid_count == 6
    assert rec.distribution_mmd_rbf == 0.45


def test_p23_31_evidence_generation_record_from_summary():
    summary = {
        "candidate_count": 6,
        "candidate_family_ids": ["AR", "GARCH", "ARMA_GARCH", "AR", "GARCH", "ARMA_GARCH"],
        "source_records_summary": [
            {"candidate_index": 0, "source_reference_indices": [0], "seed": 22001},
            {"candidate_index": 1, "source_reference_indices": [1], "seed": 22002},
            {"candidate_index": 2, "source_reference_indices": [0, 1], "seed": 22003},
            {"candidate_index": 3, "source_reference_indices": [0], "seed": 22004},
            {"candidate_index": 4, "source_reference_indices": [1], "seed": 22005},
            {"candidate_index": 5, "source_reference_indices": [0, 1], "seed": 22006}
        ]
    }
    rec = evidence_generation_record_from_summary(summary)
    assert rec.candidate_count == 6
    assert rec.seed_sequence == (22001, 22002, 22003, 22004, 22005, 22006)
    assert rec.source_reference_indices[2] == (0, 1)


def test_p23_32_evidence_reference_records_from_p22_summary():
    summary = {
        "references": [
            {
                "role": "arma_mean_source",
                "sample_id": "ARMA_001",
                "line_number": 100,
                "family_id": "ARMA",
                "mean_family": "ARMA",
                "volatility_family": "NONE"
            }
        ]
    }
    recs = evidence_reference_records_from_p22_summary(summary)
    assert len(recs) == 1
    assert recs[0].sample_id == "ARMA_001"


def test_p23_33_normalize_p22_smoke_summary_to_evidence_bundle():
    # Construct a valid P22 summary dict
    p22_summary = {
        "contract": "phase2_p22_artifact_backed_baseline_smoke_v1",
        "verdict": "PASS",
        "p14_audit_verified": True,
        "no_artifact_generation": True,
        "no_model_training": True,
        "no_final_comparison": True,
        "no_scientific_conclusion": True,
        "artifact_root": "phase2_artifacts/p14_smoke_dry_run",
        "loaded_reference_count": 2,
        "baseline_count": 3,
        "candidate_count_per_baseline": 6,
        "reference_summary": {
            "references": [
                {"role": "arma_mean_source", "sample_id": "ARMA_1", "line_number": 1, "family_id": "ARMA", "mean_family": "ARMA", "volatility_family": "NONE"},
                {"role": "garch_volatility_source", "sample_id": "GARCH_1", "line_number": 2, "family_id": "GARCH", "mean_family": "NONE", "volatility_family": "GARCH"}
            ]
        },
        "baselines": [
            {
                "baseline_name": "copy_reference",
                "metric_bundle_bridge_verified": True,
                "reason": "completed copy_reference",
                "generation_summary": {
                    "candidate_count": 6, "candidate_family_ids": ["ARMA"] * 6,
                    "source_records_summary": [{"source_reference_indices": [0], "seed": None}] * 6
                },
                "evaluation_summary": {
                    "validity_valid_count": 6, "validity_total_count": 6, "validity_valid_rate": 1.0,
                    "composition_pass_count": 0, "composition_pass_rate": 0.0,
                    "novelty_pass_count": 0, "novelty_pass_rate": 0.0, "distribution_mmd_rbf": 0.0,
                    "seed_stability_present": False, "candidate_record_count": 6
                }
            },
            {
                "baseline_name": "random_valid",
                "metric_bundle_bridge_verified": True,
                "reason": "completed random_valid",
                "generation_summary": {
                    "candidate_count": 6, "candidate_family_ids": ["ARMA"] * 6,
                    "source_records_summary": [{"source_reference_indices": [], "seed": 22001}] * 6
                },
                "evaluation_summary": {
                    "validity_valid_count": 6, "validity_total_count": 6, "validity_valid_rate": 1.0,
                    "composition_pass_count": 0, "composition_pass_rate": 0.0,
                    "novelty_pass_count": 0, "novelty_pass_rate": 0.0, "distribution_mmd_rbf": 0.1,
                    "seed_stability_present": False, "candidate_record_count": 6
                }
            },
            {
                "baseline_name": "structural_composition_oracle",
                "metric_bundle_bridge_verified": True,
                "reason": "completed structural_composition_oracle",
                "generation_summary": {
                    "candidate_count": 6, "candidate_family_ids": ["ARMA_GARCH"] * 6,
                    "source_records_summary": [{"source_reference_indices": [0, 1], "seed": None}] * 6
                },
                "evaluation_summary": {
                    "validity_valid_count": 6, "validity_total_count": 6, "validity_valid_rate": 1.0,
                    "composition_pass_count": 6, "composition_pass_rate": 1.0,
                    "novelty_pass_count": 0, "novelty_pass_rate": 0.0, "distribution_mmd_rbf": 0.2,
                    "seed_stability_present": False, "candidate_record_count": 6
                }
            }
        ]
    }
    bundle = normalize_p22_smoke_summary_to_evidence_bundle(p22_summary)
    assert bundle.metadata.contract_version == EVIDENCE_CONTRACT_VERSION
    assert len(bundle.baselines) == 3
    assert bundle.baselines[0].baseline_name == "copy_reference"
    assert bundle.baselines[2].metrics.composition_pass_rate == 1.0


def test_p23_34_normalization_rejects_wrong_p22_contract():
    wrong_summary = {"contract": "wrong_contract_v1"}
    with pytest.raises(ValueError, match="Unsupported summary contract"):
        normalize_p22_smoke_summary_to_evidence_bundle(wrong_summary)


def test_p23_35_normalization_rejects_no_artifact_generation_false():
    wrong_summary = {
        "contract": "phase2_p22_artifact_backed_baseline_smoke_v1",
        "verdict": "PASS",
        "p14_audit_verified": True,
        "no_artifact_generation": False  # Violates scope
    }
    with pytest.raises(ValueError):
        normalize_p22_smoke_summary_to_evidence_bundle(wrong_summary)


def test_p23_36_evidence_bundle_to_json_dict_keys(valid_bundle):
    d = evidence_bundle_to_json_dict(valid_bundle)
    assert list(d.keys()) == [
        "metadata",
        "loaded_reference_count",
        "references",
        "baseline_count",
        "candidate_count_per_baseline",
        "baselines",
        "verdict"
    ]
    assert d["verdict"] == "PASS"


def test_p23_37_evidence_bundle_to_json_dict_contains_no_raw_params(valid_bundle):
    d = evidence_bundle_to_json_dict(valid_bundle)
    assert_no_raw_params(d)


def test_p23_38_compact_evidence_json(valid_bundle):
    js = compact_evidence_json(valid_bundle)
    d = json.loads(js)
    assert d["verdict"] == "PASS"
    # Compact format has no whitespace formatting spacers
    assert ": " not in js
    assert ", " not in js
    assert "\n" not in js


# E. Exports/scope
def test_p23_39_exports():
    exports = dir(src.phase2)
    assert "EVIDENCE_CONTRACT_VERSION" in exports
    assert "EvidenceBundle" in exports
    assert "normalize_p22_smoke_summary_to_evidence_bundle" in exports


def test_p23_40_evidence_script_does_not_import_forbidden():
    p = pathlib.Path("src/phase2/evidence.py").read_text(encoding="utf-8")
    forbidden = ["torch", "numpy", "pandas", "yaml", "argparse", "sklearn", "scipy"]
    for f in forbidden:
        assert f"import {f}" not in p
        assert f"from {f}" not in p


def test_p23_41_evidence_script_does_not_call_forbidden():
    p = pathlib.Path("src/phase2/evidence.py").read_text(encoding="utf-8")
    forbidden_calls = [
        "simulate_time_series",
        "build_dataset_in_memory",
        "write_dataset_artifacts",
        "run_phase2_artifact_generation"
    ]
    for c in forbidden_calls:
        assert c not in p
