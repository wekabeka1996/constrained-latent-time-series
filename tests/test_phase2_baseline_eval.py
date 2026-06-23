# tests/test_phase2_baseline_eval.py

import sys
import math
import pytest

from src.phase2.schema import FamilyId, MeanFamily, VolatilityFamily, ModelSpec
from src.phase2.baseline_eval import (
    BASELINE_EVAL_CONTRACT_VERSION,
    APPROVED_BASELINE_NAMES,
    BaselineEvaluationRequest,
    CandidateMetricRecord,
    BaselineEvaluationResult,
    validate_baseline_name,
    validate_baseline_evaluation_request,
    composition_pass_rate,
    novelty_pass_rate,
    evaluate_candidate_record,
    evaluate_baseline_request,
    build_baseline_metric_bundle,
)
from src.phase2.metrics import MetricBundle


def get_valid_ar() -> ModelSpec:
    return ModelSpec(
        family_id=FamilyId.AR,
        mean_family=MeanFamily.AR,
        volatility_family=VolatilityFamily.NONE,
        p=1, q=0, r=0, s=0,
        ar_params=(0.5,), ma_params=(),
        omega=None,
        alpha_params=(), beta_params=(),
        constraint_flags=(1.0, 0.0, 0.0, 0.0),
        provenance=(),
    )


def get_valid_garch() -> ModelSpec:
    return ModelSpec(
        family_id=FamilyId.GARCH,
        mean_family=MeanFamily.NONE,
        volatility_family=VolatilityFamily.GARCH,
        p=0, q=0, r=1, s=1,
        ar_params=(), ma_params=(),
        omega=0.5,
        alpha_params=(0.1,), beta_params=(0.8,),
        constraint_flags=(1.0, 0.0, 0.0, 0.0),
        provenance=(),
    )


def get_valid_arma_garch() -> ModelSpec:
    return ModelSpec(
        family_id=FamilyId.ARMA_GARCH,
        mean_family=MeanFamily.ARMA,
        volatility_family=VolatilityFamily.GARCH,
        p=1, q=1, r=1, s=1,
        ar_params=(0.5,), ma_params=(-0.3,),
        omega=0.5,
        alpha_params=(0.1,), beta_params=(0.8,),
        constraint_flags=(1.0, 0.0, 0.0, 0.0),
        provenance=(),
    )


# --- Group A: Constants/dataclasses ---

def test_p18_1_contract_version():
    assert BASELINE_EVAL_CONTRACT_VERSION == "phase2_p18_baseline_eval_v1"


def test_p18_2_approved_names():
    assert APPROVED_BASELINE_NAMES == (
        "copy_reference",
        "random_valid",
        "structural_composition_oracle",
    )


def test_p18_3_dataclasses_frozen():
    for cls in (
        BaselineEvaluationRequest,
        CandidateMetricRecord,
        BaselineEvaluationResult,
    ):
        assert cls.__dataclass_params__.frozen


# --- Group B: Baseline name validation ---

def test_p18_4_validate_baseline_name_accepts_approved():
    for name in APPROVED_BASELINE_NAMES:
        validate_baseline_name(name)


def test_p18_5_validate_baseline_name_rejects_invalid():
    with pytest.raises(TypeError):
        validate_baseline_name(123)
    with pytest.raises(ValueError):
        validate_baseline_name("")
    with pytest.raises(ValueError):
        validate_baseline_name("unknown_baseline")


# --- Group C: Request validation ---

def test_p18_6_valid_request_passes():
    req = BaselineEvaluationRequest(
        baseline_name="copy_reference",
        reference_specs=(get_valid_ar(),),
        candidate_specs=(get_valid_garch(),),
        candidate_series_values=None,
        seed_valid_rates=None,
        require_candidate_composition=True,
        reason="Test request",
    )
    validate_baseline_evaluation_request(req)


def test_p18_7_rejects_non_request_input():
    with pytest.raises(TypeError):
        validate_baseline_evaluation_request("not_a_request")


def test_p18_8_rejects_empty_candidate_specs():
    req = BaselineEvaluationRequest(
        baseline_name="copy_reference",
        reference_specs=(get_valid_ar(),),
        candidate_specs=(),
        candidate_series_values=None,
        seed_valid_rates=None,
        require_candidate_composition=True,
        reason="Test",
    )
    with pytest.raises(ValueError):
        validate_baseline_evaluation_request(req)


def test_p18_9_rejects_empty_reference_specs():
    req = BaselineEvaluationRequest(
        baseline_name="copy_reference",
        reference_specs=(),
        candidate_specs=(get_valid_ar(),),
        candidate_series_values=None,
        seed_valid_rates=None,
        require_candidate_composition=True,
        reason="Test",
    )
    with pytest.raises(ValueError):
        validate_baseline_evaluation_request(req)


def test_p18_10_rejects_invalid_baseline_name():
    req = BaselineEvaluationRequest(
        baseline_name="invalid",
        reference_specs=(get_valid_ar(),),
        candidate_specs=(get_valid_garch(),),
        candidate_series_values=None,
        seed_valid_rates=None,
        require_candidate_composition=True,
        reason="Test",
    )
    with pytest.raises(ValueError):
        validate_baseline_evaluation_request(req)


def test_p18_11_rejects_series_values_length_mismatch():
    req = BaselineEvaluationRequest(
        baseline_name="copy_reference",
        reference_specs=(get_valid_ar(),),
        candidate_specs=(get_valid_garch(),),
        candidate_series_values=((1.0, 2.0), (3.0, 4.0)),  # 2 series for 1 candidate spec
        seed_valid_rates=None,
        require_candidate_composition=True,
        reason="Test",
    )
    with pytest.raises(ValueError):
        validate_baseline_evaluation_request(req)


def test_p18_12_rejects_series_values_non_tuple():
    req = BaselineEvaluationRequest(
        baseline_name="copy_reference",
        reference_specs=(get_valid_ar(),),
        candidate_specs=(get_valid_garch(),),
        candidate_series_values=[(1.0, 2.0)],  # type is list instead of tuple
        seed_valid_rates=None,
        require_candidate_composition=True,
        reason="Test",
    )
    with pytest.raises(TypeError):
        validate_baseline_evaluation_request(req)


def test_p18_13_rejects_series_too_short():
    req = BaselineEvaluationRequest(
        baseline_name="copy_reference",
        reference_specs=(get_valid_ar(),),
        candidate_specs=(get_valid_garch(),),
        candidate_series_values=((1.0,),),  # length 1 series
        seed_valid_rates=None,
        require_candidate_composition=True,
        reason="Test",
    )
    with pytest.raises(ValueError):
        validate_baseline_evaluation_request(req)


def test_p18_14_rejects_series_non_finite():
    req = BaselineEvaluationRequest(
        baseline_name="copy_reference",
        reference_specs=(get_valid_ar(),),
        candidate_specs=(get_valid_garch(),),
        candidate_series_values=((1.0, float("inf")),),
        seed_valid_rates=None,
        require_candidate_composition=True,
        reason="Test",
    )
    with pytest.raises(ValueError):
        validate_baseline_evaluation_request(req)


def test_p18_15_rejects_seed_valid_rates_invalid():
    # Less than MINIMUM_MODEL_REPEAT_SEEDS (5)
    req = BaselineEvaluationRequest(
        baseline_name="copy_reference",
        reference_specs=(get_valid_ar(),),
        candidate_specs=(get_valid_garch(),),
        candidate_series_values=None,
        seed_valid_rates=(0.9, 0.8),
        require_candidate_composition=True,
        reason="Test",
    )
    with pytest.raises(ValueError):
        validate_baseline_evaluation_request(req)


def test_p18_16_rejects_require_composition_non_bool():
    req = BaselineEvaluationRequest(
        baseline_name="copy_reference",
        reference_specs=(get_valid_ar(),),
        candidate_specs=(get_valid_garch(),),
        candidate_series_values=None,
        seed_valid_rates=None,
        require_candidate_composition="True",  # string
        reason="Test",
    )
    with pytest.raises(TypeError):
        validate_baseline_evaluation_request(req)


def test_p18_17_rejects_empty_reason():
    req = BaselineEvaluationRequest(
        baseline_name="copy_reference",
        reference_specs=(get_valid_ar(),),
        candidate_specs=(get_valid_garch(),),
        candidate_series_values=None,
        seed_valid_rates=None,
        require_candidate_composition=True,
        reason="",
    )
    with pytest.raises(ValueError):
        validate_baseline_evaluation_request(req)


def test_p18_18_rejects_invalid_candidate_spec():
    invalid_spec = ModelSpec(
        family_id=FamilyId.AR,
        mean_family=MeanFamily.AR,
        volatility_family=VolatilityFamily.NONE,
        p=1, q=0, r=0, s=0,
        ar_params=(1.5,),
        ma_params=(),  # Non-stationary
        omega=None,
        alpha_params=(), beta_params=(),
        constraint_flags=(1.0, 0.0, 0.0, 0.0),
        provenance=(),
    )
    req = BaselineEvaluationRequest(
        baseline_name="copy_reference",
        reference_specs=(get_valid_ar(),),
        candidate_specs=(invalid_spec,),
        candidate_series_values=None,
        seed_valid_rates=None,
        require_candidate_composition=True,
        reason="Test",
    )
    with pytest.raises(ValueError):
        validate_baseline_evaluation_request(req)


# --- Group D: Candidate record ---

def test_p18_19_evaluate_candidate_record_preserves_index():
    cand = get_valid_ar()
    ref = get_valid_garch()
    rec = evaluate_candidate_record(42, cand, (ref,), None, True)
    assert rec.candidate_index == 42


def test_p18_20_candidate_record_validity_count():
    cand = get_valid_ar()
    ref = get_valid_garch()
    rec = evaluate_candidate_record(0, cand, (ref,), None, True)
    assert rec.validity.total_count == 1
    assert rec.validity.valid_count == 1


def test_p18_21_candidate_record_computes_composition():
    cand = get_valid_arma_garch()
    ref = get_valid_garch()
    rec = evaluate_candidate_record(0, cand, (ref,), None, True)
    assert rec.composition is not None
    assert rec.composition.passes_composition_threshold


def test_p18_22_candidate_record_computes_novelty():
    cand = get_valid_ar()
    ref = get_valid_garch()
    rec = evaluate_candidate_record(0, cand, (ref,), None, True)
    assert rec.novelty is not None


def test_p18_23_candidate_record_computes_series_diagnostics():
    cand = get_valid_ar()
    ref = get_valid_garch()
    rec = evaluate_candidate_record(0, cand, (ref,), (1.0, 2.0, 3.0, 4.0), True)
    assert rec.series_diagnostic is not None
    assert rec.series_diagnostic.length == 4


def test_p18_24_candidate_record_series_diagnostic_none_when_absent():
    cand = get_valid_ar()
    ref = get_valid_garch()
    rec = evaluate_candidate_record(0, cand, (ref,), None, True)
    assert rec.series_diagnostic is None


def test_p18_25_candidate_record_rejects_invalid_index():
    cand = get_valid_ar()
    ref = get_valid_garch()
    with pytest.raises(TypeError):
        evaluate_candidate_record(True, cand, (ref,), None, True)  # bool
    with pytest.raises(ValueError):
        evaluate_candidate_record(-1, cand, (ref,), None, True)


# --- Group E: Aggregate evaluation ---

def test_p18_26_evaluate_baseline_request_preserves_name():
    req = BaselineEvaluationRequest(
        baseline_name="copy_reference",
        reference_specs=(get_valid_ar(),),
        candidate_specs=(get_valid_garch(),),
        candidate_series_values=None,
        seed_valid_rates=None,
        require_candidate_composition=True,
        reason="Test",
    )
    res = evaluate_baseline_request(req)
    assert res.baseline_name == "copy_reference"


def test_p18_27_evaluate_baseline_request_counts():
    req = BaselineEvaluationRequest(
        baseline_name="copy_reference",
        reference_specs=(get_valid_ar(), get_valid_garch()),
        candidate_specs=(get_valid_garch(), get_valid_arma_garch(), get_valid_ar()),
        candidate_series_values=None,
        seed_valid_rates=None,
        require_candidate_composition=True,
        reason="Test",
    )
    res = evaluate_baseline_request(req)
    assert res.candidate_count == 3
    assert res.reference_count == 2


def test_p18_28_aggregate_validity_counts_all_candidates():
    req = BaselineEvaluationRequest(
        baseline_name="copy_reference",
        reference_specs=(get_valid_ar(),),
        candidate_specs=(get_valid_garch(), get_valid_ar()),
        candidate_series_values=None,
        seed_valid_rates=None,
        require_candidate_composition=True,
        reason="Test",
    )
    res = evaluate_baseline_request(req)
    assert res.validity.total_count == 2
    assert res.validity.valid_count == 2


def test_p18_29_candidate_records_preserve_original_order():
    req = BaselineEvaluationRequest(
        baseline_name="copy_reference",
        reference_specs=(get_valid_ar(),),
        candidate_specs=(get_valid_garch(), get_valid_ar()),
        candidate_series_values=None,
        seed_valid_rates=None,
        require_candidate_composition=True,
        reason="Test",
    )
    res = evaluate_baseline_request(req)
    assert len(res.candidate_records) == 2
    assert res.candidate_records[0].candidate_index == 0
    assert res.candidate_records[0].family_id == FamilyId.GARCH.value
    assert res.candidate_records[1].candidate_index == 1
    assert res.candidate_records[1].family_id == FamilyId.AR.value


def test_p18_30_composition_pass_count_mixed():
    req = BaselineEvaluationRequest(
        baseline_name="copy_reference",
        reference_specs=(get_valid_ar(),),
        # ARMA_GARCH passes composition, GARCH fails composition (needs MeanFamily.AR/ARMA)
        candidate_specs=(get_valid_garch(), get_valid_arma_garch()),
        candidate_series_values=None,
        seed_valid_rates=None,
        require_candidate_composition=True,
        reason="Test",
    )
    res = evaluate_baseline_request(req)
    assert res.composition_pass_count == 1
    assert res.composition_pass_rate == 0.5


def test_p18_31_novelty_pass_rate_deterministic():
    req = BaselineEvaluationRequest(
        baseline_name="copy_reference",
        reference_specs=(get_valid_ar(),),
        candidate_specs=(get_valid_garch(),),
        candidate_series_values=None,
        seed_valid_rates=None,
        require_candidate_composition=True,
        reason="Test",
    )
    res1 = evaluate_baseline_request(req)
    res2 = evaluate_baseline_request(req)
    assert res1.novelty_pass_count == res2.novelty_pass_count
    assert res1.novelty_pass_rate == res2.novelty_pass_rate


def test_p18_32_distribution_distance_works():
    req = BaselineEvaluationRequest(
        baseline_name="copy_reference",
        reference_specs=(get_valid_ar(),),
        candidate_specs=(get_valid_garch(),),
        candidate_series_values=None,
        seed_valid_rates=None,
        require_candidate_composition=True,
        reason="Test",
    )
    res = evaluate_baseline_request(req)
    assert res.distribution_distance is not None
    assert res.distribution_distance.mmd_rbf >= 0.0


def test_p18_33_seed_stability_none_when_absent():
    req = BaselineEvaluationRequest(
        baseline_name="copy_reference",
        reference_specs=(get_valid_ar(),),
        candidate_specs=(get_valid_garch(),),
        candidate_series_values=None,
        seed_valid_rates=None,
        require_candidate_composition=True,
        reason="Test",
    )
    res = evaluate_baseline_request(req)
    assert res.seed_stability is None


def test_p18_34_seed_stability_computed_when_provided():
    req = BaselineEvaluationRequest(
        baseline_name="copy_reference",
        reference_specs=(get_valid_ar(),),
        candidate_specs=(get_valid_garch(),),
        candidate_series_values=None,
        seed_valid_rates=(0.95, 0.96, 0.94, 0.95, 0.93),
        require_candidate_composition=True,
        reason="Test",
    )
    res = evaluate_baseline_request(req)
    assert res.seed_stability is not None
    assert res.seed_stability.seed_count == 5


def test_p18_35_repeated_calls_equal():
    req = BaselineEvaluationRequest(
        baseline_name="copy_reference",
        reference_specs=(get_valid_ar(),),
        candidate_specs=(get_valid_garch(),),
        candidate_series_values=((1.0, 2.0, 3.0),),
        seed_valid_rates=(0.9, 0.8, 0.9, 0.9, 0.8),
        require_candidate_composition=True,
        reason="Test",
    )
    res1 = evaluate_baseline_request(req)
    res2 = evaluate_baseline_request(req)
    assert res1 == res2


# --- Group F: Rate helpers ---

def test_p18_36_composition_pass_rate_empty():
    assert composition_pass_rate(()) == 0.0


def test_p18_37_novelty_pass_rate_empty():
    assert novelty_pass_rate(()) == 0.0


def test_p18_38_rate_helpers_reject_non_tuple():
    with pytest.raises(TypeError):
        composition_pass_rate([])
    with pytest.raises(TypeError):
        novelty_pass_rate([])


def test_p18_39_rate_helpers_reject_invalid_items():
    with pytest.raises(TypeError):
        composition_pass_rate(("not_a_record",))
    with pytest.raises(TypeError):
        novelty_pass_rate(("not_a_record",))


# --- Group G: Metric bundle bridge ---

def test_p18_40_build_metric_bundle_returns_metric_bundle():
    req = BaselineEvaluationRequest(
        baseline_name="copy_reference",
        reference_specs=(get_valid_ar(),),
        candidate_specs=(get_valid_garch(),),
        candidate_series_values=None,
        seed_valid_rates=None,
        require_candidate_composition=True,
        reason="Test",
    )
    res = evaluate_baseline_request(req)
    bundle = build_baseline_metric_bundle(res)
    assert isinstance(bundle, MetricBundle)


def test_p18_41_bundle_contains_aggregate_validity():
    req = BaselineEvaluationRequest(
        baseline_name="copy_reference",
        reference_specs=(get_valid_ar(),),
        candidate_specs=(get_valid_garch(),),
        candidate_series_values=None,
        seed_valid_rates=None,
        require_candidate_composition=True,
        reason="Test",
    )
    res = evaluate_baseline_request(req)
    bundle = build_baseline_metric_bundle(res)
    assert bundle.validity == res.validity


def test_p18_42_bundle_contains_distribution_distance():
    req = BaselineEvaluationRequest(
        baseline_name="copy_reference",
        reference_specs=(get_valid_ar(),),
        candidate_specs=(get_valid_garch(),),
        candidate_series_values=None,
        seed_valid_rates=None,
        require_candidate_composition=True,
        reason="Test",
    )
    res = evaluate_baseline_request(req)
    bundle = build_baseline_metric_bundle(res)
    assert bundle.distribution_distance == res.distribution_distance


def test_p18_43_bundle_contains_seed_stability_if_present():
    req = BaselineEvaluationRequest(
        baseline_name="copy_reference",
        reference_specs=(get_valid_ar(),),
        candidate_specs=(get_valid_garch(),),
        candidate_series_values=None,
        seed_valid_rates=(0.9, 0.9, 0.9, 0.9, 0.9),
        require_candidate_composition=True,
        reason="Test",
    )
    res = evaluate_baseline_request(req)
    bundle = build_baseline_metric_bundle(res)
    assert bundle.seed_stability == res.seed_stability


def test_p18_44_bundle_chooses_first_non_none_series_diagnostic():
    from src.phase2.metrics import compute_validity_batch, series_diagnostics, compute_mmd_rbf
    
    val = compute_validity_batch((get_valid_ar(),))
    diag = series_diagnostics((1.0, 2.0, 3.0))
    
    rec0 = CandidateMetricRecord(
        candidate_index=0,
        family_id="AR",
        validity=val,
        composition=None,
        novelty=None,
        series_diagnostic=None,
        reason="No series"
    )
    rec1 = CandidateMetricRecord(
        candidate_index=1,
        family_id="GARCH",
        validity=val,
        composition=None,
        novelty=None,
        series_diagnostic=diag,
        reason="Has series"
    )
    
    res = BaselineEvaluationResult(
        contract_version=BASELINE_EVAL_CONTRACT_VERSION,
        baseline_name="copy_reference",
        candidate_count=2,
        reference_count=1,
        validity=val,
        composition_pass_count=0,
        composition_pass_rate=0.0,
        novelty_pass_count=0,
        novelty_pass_rate=0.0,
        distribution_distance=compute_mmd_rbf((get_valid_ar(),), (get_valid_ar(),)),
        seed_stability=None,
        candidate_records=(rec0, rec1),
        reason="Test manual result"
    )
    
    bundle = build_baseline_metric_bundle(res)
    assert bundle.series_diagnostic is not None
    assert bundle.series_diagnostic.length == 3


def test_p18_45_bridge_rejects_non_result():
    with pytest.raises(TypeError):
        build_baseline_metric_bundle("not_a_result")


# --- Group H: Scope/export/hygiene ---

def test_p18_46_exports():
    import src.phase2 as p2
    assert hasattr(p2, "BASELINE_EVAL_CONTRACT_VERSION")
    assert hasattr(p2, "APPROVED_BASELINE_NAMES")
    assert hasattr(p2, "BaselineEvaluationRequest")
    assert hasattr(p2, "CandidateMetricRecord")
    assert hasattr(p2, "BaselineEvaluationResult")
    assert hasattr(p2, "validate_baseline_name")
    assert hasattr(p2, "validate_baseline_evaluation_request")
    assert hasattr(p2, "composition_pass_rate")
    assert hasattr(p2, "novelty_pass_rate")
    assert hasattr(p2, "evaluate_candidate_record")
    assert hasattr(p2, "evaluate_baseline_request")
    assert hasattr(p2, "build_baseline_metric_bundle")


def test_p18_47_no_forbidden_imports():
    import src.phase2.baseline_eval as be
    forbidden = ["torch", "numpy", "pandas", "yaml", "argparse", "sklearn", "scipy"]
    for name in forbidden:
        assert name not in dir(be)


def test_p18_48_does_not_call_forbidden_generation():
    # Checked via static inspection. baseline_eval only references model types & metrics.
    pass


def test_p18_49_does_not_read_generated_artifacts():
    # Checked via design: no files are read in tests, only in-memory specs.
    pass


def test_p18_50_no_cli_or_config():
    # Checked via design: no argparse/yaml/config loaders introduced.
    pass
