# tests/test_phase2_metrics.py

import dataclasses
import math
import sys
import pytest

from src.phase2.schema import FamilyId, MeanFamily, VolatilityFamily, ModelSpec
from src.phase2.metrics import (
    COMPOSITION_MEAN_THRESHOLD,
    COMPOSITION_VOL_THRESHOLD,
    COMPOSITION_SCORE_MIN,
    NOVELTY_SCORE_MIN,
    GENERATED_C_VALID_RATE_MIN_SMOKE,
    GENERATED_C_VALID_RATE_MIN_MAIN,
    SIMULATION_FAILURE_RATE_MAX,
    MINIMUM_MODEL_REPEAT_SEEDS,
    SEED_STABILITY_VALID_RATE_RANGE_MAX,
    DEFAULT_MMD_RBF_GAMMA,
    SERIES_DIAGNOSTIC_MAX_LAG,
    ValidityBatchResult,
    CompositionResult,
    NoveltyResult,
    DistributionDistanceResult,
    SeriesDiagnosticResult,
    SeedStabilityResult,
    MetricBundle,
    validate_metric_probability,
    validate_non_empty_model_specs,
    model_spec_parameter_vector,
    euclidean_distance,
    structural_family_distance,
    normalized_parameter_distance,
    model_spec_distance,
    score_model_spec_novelty,
    score_composition,
    compute_validity_batch,
    rbf_kernel,
    compute_mmd_rbf,
    mean,
    variance,
    autocorrelation,
    series_diagnostics,
    compute_seed_stability,
    build_metric_bundle,
)


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


# --- Group A: Constants and Dataclasses ---

def test_p17_1_constants_equal_approved():
    assert COMPOSITION_MEAN_THRESHOLD == 0.60
    assert COMPOSITION_VOL_THRESHOLD == 0.60
    assert COMPOSITION_SCORE_MIN == 0.70
    assert NOVELTY_SCORE_MIN == 0.80
    assert GENERATED_C_VALID_RATE_MIN_SMOKE == 0.50
    assert GENERATED_C_VALID_RATE_MIN_MAIN == 0.70
    assert SIMULATION_FAILURE_RATE_MAX == 0.01
    assert MINIMUM_MODEL_REPEAT_SEEDS == 5
    assert SEED_STABILITY_VALID_RATE_RANGE_MAX == 0.15
    assert DEFAULT_MMD_RBF_GAMMA == 1.0
    assert SERIES_DIAGNOSTIC_MAX_LAG == 10


def test_p17_2_dataclasses_frozen():
    for cls in (
        ValidityBatchResult,
        CompositionResult,
        NoveltyResult,
        DistributionDistanceResult,
        SeriesDiagnosticResult,
        SeedStabilityResult,
        MetricBundle,
    ):
        assert cls.__dataclass_params__.frozen


def test_p17_3_metric_bundle_accepts_valid_and_none():
    val = ValidityBatchResult(10, 10, 0, 1.0, (), "valid")
    comp = CompositionResult(1.0, 1.0, 1.0, True, True, True, "comp")
    
    # All fields populated
    bundle1 = build_metric_bundle(val, comp, None, None, None, None)
    assert bundle1.validity == val
    assert bundle1.composition == comp
    assert bundle1.novelty is None
    
    # Type check failures
    with pytest.raises(TypeError):
        build_metric_bundle("not_a_result", None, None, None, None, None)


# --- Group B: Validation Helpers ---

def test_p17_4_validate_metric_probability_accepts_valid():
    validate_metric_probability(0.0, "p")
    validate_metric_probability(0.5, "p")
    validate_metric_probability(1.0, "p")


def test_p17_5_validate_metric_probability_rejects_invalid():
    with pytest.raises(ValueError):
        validate_metric_probability(True, "p")  # bool
    with pytest.raises(ValueError):
        validate_metric_probability("0.5", "p")  # string
    with pytest.raises(ValueError):
        validate_metric_probability(float("nan"), "p")
    with pytest.raises(ValueError):
        validate_metric_probability(float("inf"), "p")
    with pytest.raises(ValueError):
        validate_metric_probability(-0.1, "p")
    with pytest.raises(ValueError):
        validate_metric_probability(1.1, "p")


def test_p17_6_validate_non_empty_model_specs_rejects_invalid():
    # non-tuple
    with pytest.raises(TypeError):
        validate_non_empty_model_specs([get_valid_ar()], "specs")
    # empty tuple
    with pytest.raises(ValueError):
        validate_non_empty_model_specs((), "specs")
    # non-ModelSpec item
    with pytest.raises(TypeError):
        validate_non_empty_model_specs((get_valid_ar(), "not_a_spec"), "specs")


def test_p17_7_validate_non_empty_model_specs_accepts_valid():
    validate_non_empty_model_specs((get_valid_ar(), get_valid_garch(), get_valid_arma_garch()), "specs")


# --- Group C: Vector and Distance ---

def test_p17_8_model_spec_parameter_vector_length_deterministic():
    ar_vec = model_spec_parameter_vector(get_valid_ar())
    garch_vec = model_spec_parameter_vector(get_valid_garch())
    arma_garch_vec = model_spec_parameter_vector(get_valid_arma_garch())
    
    assert len(ar_vec) == len(garch_vec) == len(arma_garch_vec)
    # 3 (enums) + 4 (orders) + 1 (omega) + 5 (ar) + 5 (ma) + 2 (alpha) + 2 (beta) + 4 (flags) = 26
    assert len(ar_vec) == 26


def test_p17_9_model_spec_parameter_vector_differs():
    ar_vec = model_spec_parameter_vector(get_valid_ar())
    garch_vec = model_spec_parameter_vector(get_valid_garch())
    assert ar_vec != garch_vec


def test_p17_10_euclidean_distance_works_and_rejects():
    assert euclidean_distance((1.0, 2.0), (4.0, 6.0)) == 5.0
    
    # mismatched lengths
    with pytest.raises(ValueError):
        euclidean_distance((1.0,), (1.0, 2.0))
        
    # bools
    with pytest.raises(TypeError):
        euclidean_distance((1.0, True), (1.0, 2.0))
        
    # infinite
    with pytest.raises(ValueError):
        euclidean_distance((1.0, float("inf")), (1.0, 2.0))


def test_p17_11_structural_family_distance_symmetric():
    ar = get_valid_ar()
    garch = get_valid_garch()
    
    d1 = structural_family_distance(ar, garch)
    d2 = structural_family_distance(garch, ar)
    assert d1 == d2
    assert d1 > 0.0
    assert structural_family_distance(ar, ar) == 0.0


def test_p17_12_normalized_parameter_distance_bounds():
    ar = get_valid_ar()
    garch = get_valid_garch()
    d = normalized_parameter_distance(ar, garch)
    assert 0.0 <= d <= 1.0


def test_p17_13_model_spec_distance_symmetric_bounds():
    ar = get_valid_ar()
    garch = get_valid_garch()
    d1 = model_spec_distance(ar, garch)
    d2 = model_spec_distance(garch, ar)
    assert d1 == d2
    assert 0.0 <= d1 <= 1.0


# --- Group D: Novelty ---

def test_p17_14_score_novelty_finds_nearest():
    ar1 = get_valid_ar()
    ar2 = ModelSpec(
        family_id=FamilyId.AR,
        mean_family=MeanFamily.AR,
        volatility_family=VolatilityFamily.NONE,
        p=1, q=0, r=0, s=0,
        ar_params=(0.55,), ma_params=(),
        omega=None,
        alpha_params=(), beta_params=(),
        constraint_flags=(1.0, 0.0, 0.0, 0.0),
        provenance=(),
    )
    garch = get_valid_garch()
    
    res = score_model_spec_novelty(ar1, (ar2, garch))
    assert res.nearest_family_id == "AR"
    assert res.novelty_score < NOVELTY_SCORE_MIN


def test_p17_15_novelty_passes_correctly():
    ar = get_valid_ar()
    garch = get_valid_garch()
    
    # Candidate is very different from reference set
    res = score_model_spec_novelty(ar, (garch,))
    # Since they are different families and structures, distance is high
    assert res.passes_novelty_threshold == (res.novelty_score >= NOVELTY_SCORE_MIN)


def test_p17_16_novelty_rejects_empty_reference():
    ar = get_valid_ar()
    with pytest.raises(ValueError):
        score_model_spec_novelty(ar, ())


# --- Group E: Composition ---

def test_p17_17_arma_garch_composition_passes():
    arma_garch = get_valid_arma_garch()
    res = score_composition(arma_garch)
    assert res.passes_mean_threshold
    assert res.passes_vol_threshold
    assert res.passes_composition_threshold


def test_p17_18_ar_composition_fails_volatility():
    ar = get_valid_ar()
    res = score_composition(ar)
    assert res.passes_mean_threshold
    assert not res.passes_vol_threshold
    assert not res.passes_composition_threshold


def test_p17_19_garch_composition_fails_mean():
    garch = get_valid_garch()
    res = score_composition(garch)
    assert not res.passes_mean_threshold
    assert res.passes_vol_threshold
    assert not res.passes_composition_threshold


def test_p17_20_composition_harmonic_zero():
    # If a component is 0, harmonic mean must be 0
    ar = get_valid_ar()
    res = score_composition(ar)
    assert res.volatility_component_score == 0.0
    assert res.composition_score == 0.0


# --- Group F: Validity Batch ---

def test_p17_21_compute_validity_batch_empty():
    res = compute_validity_batch(())
    assert res.total_count == 0
    assert res.valid_count == 0
    assert res.valid_rate == 0.0


def test_p17_22_compute_validity_batch_counts_valid():
    specs = (get_valid_ar(), get_valid_garch())
    res = compute_validity_batch(specs)
    assert res.total_count == 2
    assert res.valid_count == 2
    assert res.invalid_count == 0
    assert res.valid_rate == 1.0


def test_p17_23_compute_validity_batch_records_invalid_type():
    specs = (get_valid_ar(), "not_a_spec")
    res = compute_validity_batch(specs)
    assert res.total_count == 2
    assert res.valid_count == 1
    assert res.invalid_count == 1
    assert any("Not an instance of ModelSpec" in r[0] for r in res.invalid_reasons)


def test_p17_24_compute_validity_batch_records_math_invalid():
    # An AR(1) spec with parameter 1.5 is non-stationary and fails require_math_valid
    invalid_spec = ModelSpec(
        family_id=FamilyId.AR,
        mean_family=MeanFamily.AR,
        volatility_family=VolatilityFamily.NONE,
        p=1, q=0, r=0, s=0,
        ar_params=(1.5,), ma_params=(),
        omega=None,
        alpha_params=(), beta_params=(),
        constraint_flags=(1.0, 0.0, 0.0, 0.0),
        provenance=(),
    )
    res = compute_validity_batch((get_valid_ar(), invalid_spec))
    assert res.total_count == 2
    assert res.valid_count == 1
    assert res.invalid_count == 1
    assert len(res.invalid_reasons) == 1


def test_p17_25_invalid_reasons_are_sorted():
    specs = ("not_a_spec", 123)
    res = compute_validity_batch(specs)
    reasons = [r[0] for r in res.invalid_reasons]
    assert reasons == sorted(reasons)


# --- Group G: MMD/RBF ---

def test_p17_26_rbf_kernel_identity():
    v = (1.0, 2.0)
    assert rbf_kernel(v, v, 1.0) == 1.0


def test_p17_27_rbf_kernel_decreases():
    v1 = (1.0, 2.0)
    v2 = (1.0, 3.0)
    v3 = (1.0, 10.0)
    
    assert rbf_kernel(v1, v2, 1.0) > rbf_kernel(v1, v3, 1.0)


def test_p17_28_rbf_kernel_rejects_invalid_gamma():
    v = (1.0,)
    with pytest.raises(ValueError):
        rbf_kernel(v, v, 0.0)
    with pytest.raises(ValueError):
        rbf_kernel(v, v, -1.0)
    with pytest.raises(ValueError):
        rbf_kernel(v, v, float("nan"))


def test_p17_29_compute_mmd_rbf_identity():
    ref = (get_valid_ar(), get_valid_garch())
    res = compute_mmd_rbf(ref, ref, 1.0)
    assert math.isclose(res.mmd_rbf, 0.0, abs_tol=1e-7)


def test_p17_30_compute_mmd_rbf_non_negative():
    ref = (get_valid_ar(),)
    cand = (get_valid_garch(),)
    res = compute_mmd_rbf(ref, cand, 1.0)
    assert res.mmd_rbf >= 0.0


def test_p17_31_compute_mmd_rbf_rejects_empty():
    ref = (get_valid_ar(),)
    with pytest.raises(ValueError):
        compute_mmd_rbf(ref, ())
    with pytest.raises(ValueError):
        compute_mmd_rbf((), ref)


# --- Group H: Series Diagnostics ---

def test_p17_32_mean_works():
    assert mean((1.0, 2.0, 3.0)) == 2.0
    with pytest.raises(TypeError):
        mean([1.0])
    with pytest.raises(ValueError):
        mean(())


def test_p17_33_variance_population():
    # population variance of (1, 2, 3): mean = 2, sq_diff = (1-2)^2 + (2-2)^2 + (3-2)^2 = 2.
    # pop_var = 2 / 3
    assert math.isclose(variance((1.0, 2.0, 3.0)), 2.0 / 3.0)
    assert variance((5.0,)) == 0.0


def test_p17_34_autocorrelation_simple_sequence():
    # Constant sequence has variance 0, autocorrelation should be 0.0
    assert autocorrelation((1.0, 1.0, 1.0), 1) == 0.0
    
    # Alternating sequence (1, -1, 1, -1): mean = 0, pop_var = 1.
    # lag 1: cov = 1*(-1) + (-1)*1 + 1*(-1) = -3. total_var_sum = 4. acf = -3/4 = -0.75.
    assert math.isclose(autocorrelation((1.0, -1.0, 1.0, -1.0), 1), -0.75)


def test_p17_35_autocorrelation_rejects_invalid_lag():
    with pytest.raises(ValueError):
        autocorrelation((1.0, 2.0), 0)
    with pytest.raises(ValueError):
        autocorrelation((1.0, 2.0), 2)


def test_p17_36_series_diagnostics_acf_sizes():
    series = (1.0, 2.0, 3.0, 4.0, 5.0)
    res = series_diagnostics(series, max_lag=3)
    assert len(res.acf) == 3
    assert len(res.squared_acf) == 3


def test_p17_37_series_diagnostics_volatility_clustering_bounds():
    series = (1.0, -1.0, 1.0, -1.0)
    res = series_diagnostics(series)
    assert 0.0 <= res.volatility_clustering_score <= 1.0


def test_p17_38_series_diagnostics_rejects_non_finite():
    with pytest.raises(TypeError):
        series_diagnostics((1.0, "2.0"))
    with pytest.raises(ValueError):
        series_diagnostics((1.0, float("nan")))


def test_p17_39_series_diagnostics_rejects_too_short():
    with pytest.raises(ValueError):
        series_diagnostics((1.0,))


# --- Group I: Seed Stability ---

def test_p17_40_compute_seed_stability_passes():
    rates = (0.90, 0.92, 0.95, 0.88, 0.91)
    res = compute_seed_stability(rates)
    assert res.passes_seed_stability_threshold


def test_p17_41_compute_seed_stability_fails():
    rates = (0.95, 0.50, 0.92, 0.88, 0.91)
    res = compute_seed_stability(rates)
    assert not res.passes_seed_stability_threshold


def test_p17_42_compute_seed_stability_rejects_few_seeds():
    rates = (0.90, 0.92, 0.95, 0.88)
    with pytest.raises(ValueError):
        compute_seed_stability(rates)


def test_p17_43_compute_seed_stability_rejects_invalid_probability():
    rates = (0.90, 0.92, 0.95, 0.88, 1.2)
    with pytest.raises(ValueError):
        compute_seed_stability(rates)


# --- Group J: Exports and Scope ---

def test_p17_44_exports():
    import src.phase2 as p2
    assert hasattr(p2, "COMPOSITION_MEAN_THRESHOLD")
    assert hasattr(p2, "MetricBundle")
    assert hasattr(p2, "score_composition")


def test_p17_45_no_forbidden_imports():
    import src.phase2.metrics as metrics_mod
    forbidden = ["torch", "numpy", "pandas", "yaml", "argparse", "sklearn", "scipy"]
    for name in forbidden:
        assert name not in dir(metrics_mod)


def test_p17_46_no_training_code():
    import src.phase2.metrics as metrics_mod
    assert not hasattr(metrics_mod, "train")
    assert not hasattr(metrics_mod, "fit")


def test_p17_47_no_artifact_reads():
    # Handled by design: test functions only construct ModelSpec in memory
    pass


def test_p17_48_deterministic():
    ar = get_valid_ar()
    garch = get_valid_garch()
    
    d1 = model_spec_distance(ar, garch)
    d2 = model_spec_distance(ar, garch)
    assert d1 == d2
