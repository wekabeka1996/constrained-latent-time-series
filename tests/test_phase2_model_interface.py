# tests/test_phase2_model_interface.py

import dataclasses
import json
import pathlib
import pytest
from dataclasses import is_dataclass

from tests.phase2_scope_gate_utils import enforce_phase_local_scope_gate_or_skip

import src.phase2
from src.phase2.schema import FamilyId, MeanFamily, VolatilityFamily, ModelSpec
from src.phase2.model_interface import (
    MODEL_INTERFACE_CONTRACT_VERSION,
    FC_VAE_ARCHITECTURE_ID,
    SUPPORTED_MODEL_ARCHITECTURE_IDS,
    MODEL_RUN_KIND_ZERO_SHOT,
    MODEL_RUN_KIND_FEWSHOT,
    SUPPORTED_MODEL_RUN_KINDS,
    MODEL_CANDIDATE_SUMMARY_CONTRACT_VERSION,
    FORBIDDEN_MODEL_INTERFACE_RAW_PARAM_KEYS,
    FORBIDDEN_MODEL_INTERFACE_CLAIM_WORDS,
    ModelRunMetadata,
    ModelCandidateRecord,
    ModelCandidateBatch,
    ModelCandidateSummary,
    validate_non_empty_str,
    validate_bool,
    validate_non_negative_int,
    validate_positive_int,
    validate_model_architecture_id,
    validate_model_run_kind,
    assert_no_model_interface_local_path_leakage,
    assert_no_model_interface_raw_params,
    assert_no_model_interface_forbidden_claims,
    validate_model_run_metadata,
    validate_model_candidate_record,
    validate_model_candidate_batch,
    validate_model_candidate_summary,
    model_candidate_record_to_summary_fields,
    summarize_model_candidate_batch,
    model_candidate_summary_to_json_dict,
    compact_model_candidate_summary_json,
)


# Reusable valid structures for testing
@pytest.fixture
def valid_metadata():
    return ModelRunMetadata(
        contract_version=MODEL_INTERFACE_CONTRACT_VERSION,
        model_architecture_id=FC_VAE_ARCHITECTURE_ID,
        model_run_kind=MODEL_RUN_KIND_ZERO_SHOT,
        model_code_git_commit="p25_test_commit_hash",
        training_artifact_root="phase2_artifacts/p14_smoke_dry_run",
        training_split_contract="p14_smoke_zero_shot_train",
        model_repeat_seed=25001,
        zero_shot_mode=True,
        C_train_count=0,
        config_hash="p25_test_config_hash",
        evidence_contract_version="phase2_p23_evidence_contract_v1",
        reason="valid metadata for P25 testing"
    )


@pytest.fixture
def valid_spec_arma():
    return ModelSpec(
        family_id=FamilyId.ARMA,
        mean_family=MeanFamily.ARMA,
        volatility_family=VolatilityFamily.NONE,
        p=1,
        q=1,
        r=0,
        s=0,
        ar_params=(0.5,),
        ma_params=(-0.2,),
        alpha_params=(),
        beta_params=(),
        omega=None,
        constraint_flags=(1.0, 1.0, 0.0, 0.0),
        provenance=(("generator", "test"),)
    )


@pytest.fixture
def valid_candidate_arma(valid_metadata, valid_spec_arma):
    return ModelCandidateRecord(
        candidate_id="cand_arma_001",
        metadata=valid_metadata,
        generated_spec=valid_spec_arma,
        validity_pass=True,
        math_validity_pass=True,
        generated_family_id="ARMA",
        generated_mean_family="ARMA",
        generated_volatility_family="NONE",
        source_split="train",
        reason="valid candidate record reason"
    )


# A. Constants/dataclasses
def test_p25_01_constants_values():
    assert MODEL_INTERFACE_CONTRACT_VERSION == "phase2_p25_model_interface_contract_v1"
    assert FC_VAE_ARCHITECTURE_ID == "FC-VAE"
    assert SUPPORTED_MODEL_ARCHITECTURE_IDS == ("FC-VAE",)
    assert MODEL_RUN_KIND_ZERO_SHOT == "zero_shot"
    assert MODEL_RUN_KIND_FEWSHOT == "fewshot"
    assert SUPPORTED_MODEL_RUN_KINDS == ("zero_shot", "fewshot")
    assert MODEL_CANDIDATE_SUMMARY_CONTRACT_VERSION == "phase2_p25_model_candidate_summary_v1"


def test_p25_02_dataclasses_frozen(valid_metadata):
    classes = [
        ModelRunMetadata,
        ModelCandidateRecord,
        ModelCandidateBatch,
        ModelCandidateSummary
    ]
    for cls in classes:
        assert is_dataclass(cls)
    
    with pytest.raises(Exception):
        valid_metadata.reason = "modified"


# B. Validation helpers
def test_p25_03_validate_non_empty_str_accepts():
    validate_non_empty_str("valid_string", "field")


def test_p25_04_validate_non_empty_str_rejects():
    with pytest.raises(TypeError):
        validate_non_empty_str(123, "field")
    with pytest.raises(ValueError):
        validate_non_empty_str("", "field")
    with pytest.raises(ValueError):
        validate_non_empty_str("  whitespace_padded  ", "field")


def test_p25_05_validate_bool_accepts():
    validate_bool(True, "field")
    validate_bool(False, "field")
    with pytest.raises(TypeError):
        validate_bool(1, "field")


def test_p25_06_validate_non_negative_int_rejects():
    validate_non_negative_int(0, "field")
    validate_non_negative_int(123, "field")
    with pytest.raises(TypeError):
        validate_non_negative_int(True, "field")
    with pytest.raises(TypeError):
        validate_non_negative_int(2.5, "field")
    with pytest.raises(ValueError):
        validate_non_negative_int(-1, "field")


def test_p25_07_validate_positive_int_rejects():
    validate_positive_int(1, "field")
    with pytest.raises(TypeError):
        validate_positive_int(True, "field")
    with pytest.raises(ValueError):
        validate_positive_int(0, "field")
    with pytest.raises(ValueError):
        validate_positive_int(-5, "field")


def test_p25_08_validate_model_architecture_id_accepts():
    validate_model_architecture_id("FC-VAE")


def test_p25_09_validate_model_architecture_id_rejects():
    with pytest.raises(ValueError):
        validate_model_architecture_id("unsupported_model")


def test_p25_10_validate_model_run_kind_accepts():
    validate_model_run_kind("zero_shot")
    validate_model_run_kind("fewshot")


def test_p25_11_validate_model_run_kind_rejects():
    with pytest.raises(ValueError):
        validate_model_run_kind("invalid_run_kind")


def test_p25_12_local_path_leakage_rejects():
    assert_no_model_interface_local_path_leakage("phase2_artifacts/p14_smoke_dry_run")
    with pytest.raises(ValueError):
        assert_no_model_interface_local_path_leakage("C:/Users/name/repo")
    with pytest.raises(ValueError):
        assert_no_model_interface_local_path_leakage("file:///C:/Users/name")


def test_p25_13_raw_param_check_rejects():
    assert_no_model_interface_raw_params({"candidate_id": "cand_01", "generated_family_id": "ARMA"})
    with pytest.raises(ValueError):
        assert_no_model_interface_raw_params({"ar_params": [0.5]})
    with pytest.raises(ValueError):
        assert_no_model_interface_raw_params("Forbidden raw key: ar_params=something")


def test_p25_14_forbidden_claim_check_rejects():
    assert_no_model_interface_forbidden_claims("no_scientific_conclusion")
    assert_no_model_interface_forbidden_claims("no_final_comparison")
    with pytest.raises(ValueError):
        assert_no_model_interface_forbidden_claims("This model is the best one")
    with pytest.raises(ValueError):
        assert_no_model_interface_forbidden_claims("We solved the task successfully")


# C. Metadata validation
def test_p25_15_validate_model_run_metadata_accepts(valid_metadata):
    validate_model_run_metadata(valid_metadata)


def test_p25_16_metadata_rejects_wrong_contract_version(valid_metadata):
    meta = dataclasses.replace(valid_metadata, contract_version="wrong_v2")
    with pytest.raises(ValueError, match="Wrong contract version"):
        validate_model_run_metadata(meta)


def test_p25_17_metadata_rejects_unsupported_architecture_id(valid_metadata):
    meta = dataclasses.replace(valid_metadata, model_architecture_id="unknown_model")
    with pytest.raises(ValueError, match="Unsupported model architecture"):
        validate_model_run_metadata(meta)


def test_p25_18_metadata_rejects_zero_shot_mode_with_C_train_count(valid_metadata):
    meta = dataclasses.replace(valid_metadata, zero_shot_mode=True, C_train_count=2)
    with pytest.raises(ValueError, match="C_train_count must be 0"):
        validate_model_run_metadata(meta)


def test_p25_19_metadata_rejects_model_run_kind_zero_shot_with_zero_shot_mode_false(valid_metadata):
    meta = dataclasses.replace(valid_metadata, model_run_kind="zero_shot", zero_shot_mode=False)
    with pytest.raises(ValueError, match="zero_shot_mode must be True"):
        validate_model_run_metadata(meta)


def test_p25_20_metadata_rejects_local_path_leakage(valid_metadata):
    meta = dataclasses.replace(valid_metadata, training_artifact_root="C:/Users/name/repo")
    with pytest.raises(ValueError, match="Local path leakage detected"):
        validate_model_run_metadata(meta)


# D. Candidate validation
def test_p25_21_validate_model_candidate_record_accepts(valid_candidate_arma):
    validate_model_candidate_record(valid_candidate_arma)


def test_p25_22_candidate_rejects_mismatched_family_id(valid_candidate_arma):
    cand = dataclasses.replace(valid_candidate_arma, generated_family_id="GARCH")
    with pytest.raises(ValueError, match="generated_family_id mismatch"):
        validate_model_candidate_record(cand)


def test_p25_23_candidate_rejects_mismatched_mean_family(valid_candidate_arma):
    cand = dataclasses.replace(valid_candidate_arma, generated_mean_family="NONE")
    with pytest.raises(ValueError, match="generated_mean_family mismatch"):
        validate_model_candidate_record(cand)


def test_p25_24_candidate_rejects_mismatched_volatility_family(valid_candidate_arma):
    cand = dataclasses.replace(valid_candidate_arma, generated_volatility_family="GARCH")
    with pytest.raises(ValueError, match="generated_volatility_family mismatch"):
        validate_model_candidate_record(cand)


def test_p25_25_candidate_rejects_invalid_ModelSpec(valid_candidate_arma, valid_spec_arma):
    invalid_spec = dataclasses.replace(valid_spec_arma, p=10) # Out of bounds
    cand = dataclasses.replace(valid_candidate_arma, generated_spec=invalid_spec)
    with pytest.raises(ValueError):
        validate_model_candidate_record(cand)


def test_p25_26_candidate_handles_math_invalid_ModelSpec(valid_candidate_arma, valid_spec_arma):
    # Set AR params out of stationarity bounds (e.g. 1.5)
    invalid_spec = dataclasses.replace(valid_spec_arma, ar_params=(1.5,))
    
    # Expected: math_validity_pass must be False
    cand_fail = ModelCandidateRecord(
        candidate_id="cand_01",
        metadata=valid_candidate_arma.metadata,
        generated_spec=invalid_spec,
        validity_pass=True,
        math_validity_pass=False, # Should match actual math_validity_pass
        generated_family_id="ARMA",
        generated_mean_family="ARMA",
        generated_volatility_family="NONE",
        source_split="train",
        reason="math invalid spec"
    )
    validate_model_candidate_record(cand_fail)
    
    # If math_validity_pass is incorrectly set to True, it must fail
    cand_wrong = dataclasses.replace(cand_fail, math_validity_pass=True)
    with pytest.raises(ValueError, match="math_validity_pass mismatch"):
        validate_model_candidate_record(cand_wrong)


def test_p25_27_candidate_rejects_forbidden_claim_in_reason(valid_candidate_arma):
    cand = dataclasses.replace(valid_candidate_arma, reason="This model is the winner")
    with pytest.raises(ValueError, match="Forbidden claim word"):
        validate_model_candidate_record(cand)


# E. Batch validation
@pytest.fixture
def valid_batch(valid_metadata, valid_candidate_arma):
    return ModelCandidateBatch(
        contract_version=MODEL_INTERFACE_CONTRACT_VERSION,
        metadata=valid_metadata,
        candidates=(valid_candidate_arma,),
        candidate_count=1,
        reason="valid batch reason"
    )


def test_p25_28_validate_model_candidate_batch_accepts(valid_batch):
    validate_model_candidate_batch(valid_batch)


def test_p25_29_batch_rejects_wrong_contract_version(valid_batch):
    batch = dataclasses.replace(valid_batch, contract_version="wrong_v2")
    with pytest.raises(ValueError, match="Wrong contract version"):
        validate_model_candidate_batch(batch)


def test_p25_30_batch_rejects_duplicate_candidate_ids(valid_batch, valid_candidate_arma):
    c1 = dataclasses.replace(valid_candidate_arma, candidate_id="cand_dup")
    c2 = dataclasses.replace(valid_candidate_arma, candidate_id="cand_dup")
    batch = dataclasses.replace(valid_batch, candidates=(c1, c2), candidate_count=2)
    with pytest.raises(ValueError, match="Duplicate candidate_id found"):
        validate_model_candidate_batch(batch)


def test_p25_31_batch_rejects_candidate_count_mismatch(valid_batch):
    batch = dataclasses.replace(valid_batch, candidate_count=10)
    with pytest.raises(ValueError, match="does not match candidate_count"):
        validate_model_candidate_batch(batch)


def test_p25_32_batch_rejects_candidate_metadata_mismatch(valid_batch, valid_candidate_arma, valid_metadata):
    meta2 = dataclasses.replace(valid_metadata, model_repeat_seed=999)
    c2 = dataclasses.replace(valid_candidate_arma, candidate_id="cand_02", metadata=meta2)
    batch = dataclasses.replace(valid_batch, candidates=(valid_candidate_arma, c2), candidate_count=2)
    with pytest.raises(ValueError, match="metadata does not match batch metadata"):
        validate_model_candidate_batch(batch)


# F. Summary / serialization
def test_p25_33_model_candidate_record_to_summary_fields(valid_candidate_arma):
    fields = model_candidate_record_to_summary_fields(valid_candidate_arma)
    assert "candidate_id" in fields
    assert "generated_family_id" in fields
    assert "validity_pass" in fields
    assert "generated_spec" not in fields
    assert "ar_params" not in fields


def test_p25_34_summarize_model_candidate_batch(valid_batch):
    summary = summarize_model_candidate_batch(valid_batch)
    assert summary.candidate_count == 1
    assert summary.validity_pass_count == 1
    assert summary.generated_family_ids == ("ARMA",)


def test_p25_35_summarize_model_candidate_batch_rejects_mixed_source_split(valid_batch, valid_candidate_arma):
    c2 = dataclasses.replace(valid_candidate_arma, candidate_id="cand_02", source_split="validation")
    batch = dataclasses.replace(valid_batch, candidates=(valid_candidate_arma, c2), candidate_count=2)
    with pytest.raises(ValueError, match="Mixed source splits found"):
        summarize_model_candidate_batch(batch)


@pytest.fixture
def valid_summary():
    return ModelCandidateSummary(
        contract_version=MODEL_CANDIDATE_SUMMARY_CONTRACT_VERSION,
        model_architecture_id=FC_VAE_ARCHITECTURE_ID,
        model_run_kind=MODEL_RUN_KIND_ZERO_SHOT,
        model_repeat_seed=25001,
        zero_shot_mode=True,
        C_train_count=0,
        candidate_count=1,
        generated_family_ids=("ARMA",),
        generated_mean_families=("ARMA",),
        generated_volatility_families=("NONE",),
        validity_pass_count=1,
        math_validity_pass_count=1,
        source_split="train",
        evidence_contract_version="phase2_p23_evidence_contract_v1",
        reason="valid summary reason"
    )


def test_p25_36_validate_model_candidate_summary_accepts(valid_summary):
    validate_model_candidate_summary(valid_summary)


def test_p25_37_summary_rejects_raw_param_leakage(valid_summary):
    sum_leak = dataclasses.replace(valid_summary, reason="ar_params=leaked_here")
    with pytest.raises(ValueError, match="Forbidden raw parameter leakage key"):
        validate_model_candidate_summary(sum_leak)


def test_p25_38_model_candidate_summary_to_json_dict(valid_summary):
    d = model_candidate_summary_to_json_dict(valid_summary)
    assert d["contract_version"] == MODEL_CANDIDATE_SUMMARY_CONTRACT_VERSION
    assert d["candidate_count"] == 1
    assert type(d["generated_family_ids"]) is list


def test_p25_39_compact_model_candidate_summary_json(valid_summary):
    js = compact_model_candidate_summary_json(valid_summary)
    d = json.loads(js)
    assert d["candidate_count"] == 1
    assert ":" in js
    assert "," in js
    assert ": " not in js
    assert ", " not in js
    assert "\n" not in js


# G. Exports/scope
def test_p25_40_exports():
    exports = dir(src.phase2)
    assert "MODEL_INTERFACE_CONTRACT_VERSION" in exports
    assert "ModelCandidateRecord" in exports
    assert "summarize_model_candidate_batch" in exports


def test_p25_41_evidence_script_does_not_import_forbidden():
    p = pathlib.Path("src/phase2/model_interface.py").read_text(encoding="utf-8")
    forbidden = ["torch", "numpy", "pandas", "yaml", "argparse", "sklearn", "scipy"]
    for f in forbidden:
        assert f"import {f}" not in p
        assert f"from {f}" not in p


def test_p25_42_evidence_script_does_not_call_forbidden():
    p = pathlib.Path("src/phase2/model_interface.py").read_text(encoding="utf-8")
    forbidden_calls = [
        "simulate_time_series",
        "build_dataset_in_memory",
        "write_dataset_artifacts",
        "run_phase2_artifact_generation"
    ]
    for c in forbidden_calls:
        assert c not in p


def test_p25_43_no_forbidden_files_modified():
    """Phase-local scope gate for P25. Skips on non-P25 branches."""
    allowed = {
        "src/phase2/model_interface.py",
        "src/phase2/__init__.py",
        "tests/test_phase2_model_interface.py",
        "tools/phase2/run_p25_model_interface_smoke.py",
        "tests/test_phase2_p25_model_interface_smoke.py",
        "reports/PHASE_2_P25_MODEL_INTERFACE_SKELETON_AND_CANDIDATE_CONTRACT_REPORT.md",
    }
    enforce_phase_local_scope_gate_or_skip(
        expected_branch="phase2/p25-model-interface-skeleton-candidate-contract",
        base_commit="phase2/p24-factorised-constrained-vae-architecture-spec",
        allowed_files=allowed,
        phase_label="P25",
    )
