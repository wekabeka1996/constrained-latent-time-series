# src/phase2/metrics.py

import math
from dataclasses import dataclass
from typing import Any, Tuple, Optional

from src.phase2.schema import ModelSpec


# Public constants
COMPOSITION_MEAN_THRESHOLD = 0.60
COMPOSITION_VOL_THRESHOLD = 0.60
COMPOSITION_SCORE_MIN = 0.70
NOVELTY_SCORE_MIN = 0.80
GENERATED_C_VALID_RATE_MIN_SMOKE = 0.50
GENERATED_C_VALID_RATE_MIN_MAIN = 0.70
SIMULATION_FAILURE_RATE_MAX = 0.01
MINIMUM_MODEL_REPEAT_SEEDS = 5
SEED_STABILITY_VALID_RATE_RANGE_MAX = 0.15
DEFAULT_MMD_RBF_GAMMA = 1.0
SERIES_DIAGNOSTIC_MAX_LAG = 10


# Public dataclasses
@dataclass(frozen=True)
class ValidityBatchResult:
    total_count: int
    valid_count: int
    invalid_count: int
    valid_rate: float
    invalid_reasons: Tuple[Tuple[str, int], ...]
    reason: str


@dataclass(frozen=True)
class CompositionResult:
    mean_component_score: float
    volatility_component_score: float
    composition_score: float
    passes_mean_threshold: bool
    passes_vol_threshold: bool
    passes_composition_threshold: bool
    reason: str


@dataclass(frozen=True)
class NoveltyResult:
    nearest_distance: float
    normalized_parameter_distance: float
    structural_distance: float
    novelty_score: float
    passes_novelty_threshold: bool
    nearest_family_id: str
    reason: str


@dataclass(frozen=True)
class DistributionDistanceResult:
    mmd_rbf: float
    gamma: float
    reference_count: int
    candidate_count: int
    reason: str


@dataclass(frozen=True)
class SeriesDiagnosticResult:
    length: int
    mean: float
    variance: float
    acf: Tuple[float, ...]
    squared_acf: Tuple[float, ...]
    volatility_clustering_score: float
    reason: str


@dataclass(frozen=True)
class SeedStabilityResult:
    seed_count: int
    min_valid_rate: float
    max_valid_rate: float
    valid_rate_range: float
    passes_seed_stability_threshold: bool
    reason: str


@dataclass(frozen=True)
class MetricBundle:
    validity: ValidityBatchResult
    composition: Optional[CompositionResult]
    novelty: Optional[NoveltyResult]
    distribution_distance: Optional[DistributionDistanceResult]
    series_diagnostic: Optional[SeriesDiagnosticResult]
    seed_stability: Optional[SeedStabilityResult]
    reason: str


# Public functions
def validate_metric_probability(value: float, name: str) -> None:
    if isinstance(value, bool):
        raise ValueError(f"{name} must not be a bool")
    if not isinstance(value, (int, float)):
        raise ValueError(f"{name} must be float or int, got {type(value)}")
    if not math.isfinite(value):
        raise ValueError(f"{name} must be finite")
    if not (0.0 <= value <= 1.0):
        raise ValueError(f"{name} must be between 0.0 and 1.0, got {value}")


def validate_non_empty_model_specs(specs: Tuple[ModelSpec, ...], name: str) -> None:
    if type(specs) is not tuple:
        raise TypeError(f"{name} must be exactly a tuple")
    if len(specs) == 0:
        raise ValueError(f"{name} must not be empty")
    from src.phase2.schema import validate_model_spec
    from src.phase2.constraints import require_math_valid
    for i, spec in enumerate(specs):
        if not isinstance(spec, ModelSpec):
            raise TypeError(f"Item {i} in {name} is not a ModelSpec instance")
        validate_model_spec(spec)
        require_math_valid(spec)


def model_spec_parameter_vector(spec: ModelSpec) -> Tuple[float, ...]:
    from src.phase2.schema import (
        validate_model_spec,
        FAMILY_ID_MAP,
        MEAN_FAMILY_MAP,
        VOLATILITY_FAMILY_MAP,
        APPROVED_MAX_P,
        APPROVED_MAX_Q,
        APPROVED_MAX_R,
        APPROVED_MAX_S,
    )
    from src.phase2.constraints import require_math_valid
    
    validate_model_spec(spec)
    require_math_valid(spec)
    
    vec = []
    vec.append(FAMILY_ID_MAP[spec.family_id])
    vec.append(MEAN_FAMILY_MAP[spec.mean_family])
    vec.append(VOLATILITY_FAMILY_MAP[spec.volatility_family])
    vec.append(float(spec.p))
    vec.append(float(spec.q))
    vec.append(float(spec.r))
    vec.append(float(spec.s))
    vec.append(spec.omega if spec.omega is not None else 0.0)
    
    # padded ar_params
    for i in range(APPROVED_MAX_P):
        val = spec.ar_params[i] if i < len(spec.ar_params) else 0.0
        vec.append(float(val))
        
    # padded ma_params
    for i in range(APPROVED_MAX_Q):
        val = spec.ma_params[i] if i < len(spec.ma_params) else 0.0
        vec.append(float(val))
        
    # padded alpha_params
    for i in range(APPROVED_MAX_R):
        val = spec.alpha_params[i] if i < len(spec.alpha_params) else 0.0
        vec.append(float(val))
        
    # padded beta_params
    for i in range(APPROVED_MAX_S):
        val = spec.beta_params[i] if i < len(spec.beta_params) else 0.0
        vec.append(float(val))
        
    # constraint_flags
    for val in spec.constraint_flags:
        vec.append(float(val))
        
    return tuple(vec)


def euclidean_distance(left: Tuple[float, ...], right: Tuple[float, ...]) -> float:
    if type(left) is not tuple or type(right) is not tuple:
        raise TypeError("Inputs must be exactly tuples")
    if len(left) != len(right):
        raise ValueError("Mismatched dimensions")
        
    sq_sum = 0.0
    for a, b in zip(left, right):
        if isinstance(a, bool) or isinstance(b, bool):
            raise TypeError("Elements must not be bool")
        if not isinstance(a, (int, float)) or not isinstance(b, (int, float)):
            raise TypeError("Elements must be numeric")
        if not math.isfinite(a) or not math.isfinite(b):
            raise ValueError("Elements must be finite")
        sq_sum += (a - b) ** 2
        
    return math.sqrt(sq_sum)


def structural_family_distance(left: ModelSpec, right: ModelSpec) -> float:
    from src.phase2.schema import validate_model_spec
    from src.phase2.constraints import require_math_valid
    validate_model_spec(left)
    require_math_valid(left)
    validate_model_spec(right)
    require_math_valid(right)
    
    d = 0.0
    if left.family_id != right.family_id:
        d += 0.25
    if left.mean_family != right.mean_family:
        d += 0.20
    if left.volatility_family != right.volatility_family:
        d += 0.20
        
    mismatch_orders = 0
    if left.p != right.p:
        mismatch_orders += 1
    if left.q != right.q:
        mismatch_orders += 1
    if left.r != right.r:
        mismatch_orders += 1
    if left.s != right.s:
        mismatch_orders += 1
        
    d += min(0.20, mismatch_orders * 0.05)
    
    return min(1.0, max(0.0, d))


def normalized_parameter_distance(left: ModelSpec, right: ModelSpec) -> float:
    v_left = model_spec_parameter_vector(left)
    v_right = model_spec_parameter_vector(right)
    d = euclidean_distance(v_left, v_right)
    return d / (1.0 + d)


def model_spec_distance(left: ModelSpec, right: ModelSpec) -> float:
    return 0.5 * normalized_parameter_distance(left, right) + 0.5 * structural_family_distance(left, right)


def score_model_spec_novelty(candidate: ModelSpec, reference_specs: Tuple[ModelSpec, ...]) -> NoveltyResult:
    from src.phase2.schema import validate_model_spec
    from src.phase2.constraints import require_math_valid
    validate_model_spec(candidate)
    require_math_valid(candidate)
    
    validate_non_empty_model_specs(reference_specs, "reference_specs")
    
    nearest_dist = float("inf")
    nearest_spec = None
    
    for ref in reference_specs:
        dist = model_spec_distance(candidate, ref)
        if dist < nearest_dist:
            nearest_dist = dist
            nearest_spec = ref
            
    novelty_score = nearest_dist
    passes = novelty_score >= NOVELTY_SCORE_MIN
    
    norm_p_dist = normalized_parameter_distance(candidate, nearest_spec)
    struct_dist = structural_family_distance(candidate, nearest_spec)
    
    reason = f"nearest_distance={nearest_dist:.6f}"
    
    return NoveltyResult(
        nearest_distance=nearest_dist,
        normalized_parameter_distance=norm_p_dist,
        structural_distance=struct_dist,
        novelty_score=novelty_score,
        passes_novelty_threshold=passes,
        nearest_family_id=nearest_spec.family_id.value,
        reason=reason
    )


def score_composition(candidate: ModelSpec) -> CompositionResult:
    from src.phase2.schema import validate_model_spec, MeanFamily, VolatilityFamily
    from src.phase2.constraints import require_math_valid
    validate_model_spec(candidate)
    require_math_valid(candidate)
    
    # mean_component_score
    if candidate.mean_family == MeanFamily.ARMA:
        mean_component_score = 1.0
    elif candidate.mean_family == MeanFamily.AR:
        mean_component_score = 0.8
    else:
        mean_component_score = 0.0
        
    # volatility_component_score
    if candidate.volatility_family == VolatilityFamily.GARCH:
        volatility_component_score = 1.0
    else:
        volatility_component_score = 0.0
        
    # harmonic mean
    if mean_component_score == 0.0 or volatility_component_score == 0.0:
        composition_score = 0.0
    else:
        composition_score = (2.0 * mean_component_score * volatility_component_score) / (mean_component_score + volatility_component_score)
        
    passes_mean = mean_component_score >= COMPOSITION_MEAN_THRESHOLD
    passes_vol = volatility_component_score >= COMPOSITION_VOL_THRESHOLD
    passes_comp = composition_score >= COMPOSITION_SCORE_MIN
    
    reason = f"mean={mean_component_score:.2f}, vol={volatility_component_score:.2f}, comp={composition_score:.2f}"
    
    return CompositionResult(
        mean_component_score=mean_component_score,
        volatility_component_score=volatility_component_score,
        composition_score=composition_score,
        passes_mean_threshold=passes_mean,
        passes_vol_threshold=passes_vol,
        passes_composition_threshold=passes_comp,
        reason=reason
    )


def compute_validity_batch(specs: Tuple[ModelSpec, ...]) -> ValidityBatchResult:
    if type(specs) is not tuple:
        raise TypeError("specs must be exactly a tuple")
        
    from src.phase2.schema import validate_model_spec
    from src.phase2.constraints import require_math_valid
    
    total_count = len(specs)
    valid_count = 0
    invalid_count = 0
    reasons_dict = {}
    
    for item in specs:
        if not isinstance(item, ModelSpec):
            reason = "Not an instance of ModelSpec"
            reasons_dict[reason] = reasons_dict.get(reason, 0) + 1
            invalid_count += 1
            continue
        try:
            validate_model_spec(item)
            require_math_valid(item)
            valid_count += 1
        except Exception as e:
            reason = str(e)
            reasons_dict[reason] = reasons_dict.get(reason, 0) + 1
            invalid_count += 1
            
    sorted_reasons = tuple(sorted(reasons_dict.items()))
    valid_rate = float(valid_count) / total_count if total_count > 0 else 0.0
    
    reason = f"valid_rate={valid_rate:.4f}"
    
    return ValidityBatchResult(
        total_count=total_count,
        valid_count=valid_count,
        invalid_count=invalid_count,
        valid_rate=valid_rate,
        invalid_reasons=sorted_reasons,
        reason=reason
    )


def rbf_kernel(left: Tuple[float, ...], right: Tuple[float, ...], gamma: float) -> float:
    if isinstance(gamma, bool) or not isinstance(gamma, (int, float)) or not math.isfinite(gamma) or gamma <= 0:
        raise ValueError("gamma must be a positive finite float/int")
    if len(left) != len(right):
        raise ValueError("Mismatched dimensions")
        
    sq_dist = 0.0
    for a, b in zip(left, right):
        if isinstance(a, bool) or isinstance(b, bool):
            raise TypeError("Bools not allowed in vector")
        if not isinstance(a, (int, float)) or not isinstance(b, (int, float)):
            raise TypeError("Elements must be numeric")
        if not math.isfinite(a) or not math.isfinite(b):
            raise ValueError("Elements must be finite")
        sq_dist += (a - b) ** 2
        
    return math.exp(-gamma * sq_dist)


def compute_mmd_rbf(
    reference_specs: Tuple[ModelSpec, ...],
    candidate_specs: Tuple[ModelSpec, ...],
    gamma: float = DEFAULT_MMD_RBF_GAMMA,
) -> DistributionDistanceResult:
    validate_non_empty_model_specs(reference_specs, "reference_specs")
    validate_non_empty_model_specs(candidate_specs, "candidate_specs")
    
    ref_vectors = [model_spec_parameter_vector(s) for s in reference_specs]
    cand_vectors = [model_spec_parameter_vector(s) for s in candidate_specs]
    
    n = len(ref_vectors)
    m = len(cand_vectors)
    
    # mean(k(x, x'))
    sum_xx = 0.0
    for x in ref_vectors:
        for xp in ref_vectors:
            sum_xx += rbf_kernel(x, xp, gamma)
    mean_xx = sum_xx / (n * n)
    
    # mean(k(y, y'))
    sum_yy = 0.0
    for y in cand_vectors:
        for yp in cand_vectors:
            sum_yy += rbf_kernel(y, yp, gamma)
    mean_yy = sum_yy / (m * m)
    
    # mean(k(x, y))
    sum_xy = 0.0
    for x in ref_vectors:
        for y in cand_vectors:
            sum_xy += rbf_kernel(x, y, gamma)
    mean_xy = sum_xy / (n * m)
    
    mmd_val = mean_xx + mean_yy - 2.0 * mean_xy
    if mmd_val < 0.0:
        mmd_val = 0.0
        
    return DistributionDistanceResult(
        mmd_rbf=mmd_val,
        gamma=gamma,
        reference_count=n,
        candidate_count=m,
        reason=f"mmd_rbf={mmd_val:.6f}"
    )


def mean(values: Tuple[float, ...]) -> float:
    if type(values) is not tuple:
        raise TypeError("values must be exactly a tuple")
    if len(values) == 0:
        raise ValueError("values cannot be empty")
    s = 0.0
    for val in values:
        if isinstance(val, bool):
            raise TypeError("values must not contain bool")
        if not isinstance(val, (int, float)):
            raise TypeError("values must contain float or int")
        if not math.isfinite(val):
            raise ValueError("values must be finite")
        s += val
    return s / len(values)


def variance(values: Tuple[float, ...]) -> float:
    if type(values) is not tuple:
        raise TypeError("values must be exactly a tuple")
    if len(values) == 0:
        raise ValueError("values cannot be empty")
    for val in values:
        if isinstance(val, bool):
            raise TypeError("values must not contain bool")
        if not isinstance(val, (int, float)):
            raise TypeError("values must contain float or int")
        if not math.isfinite(val):
            raise ValueError("values must be finite")
            
    if len(values) == 1:
        return 0.0
        
    m = mean(values)
    sq_diff_sum = sum((x - m) ** 2 for x in values)
    return sq_diff_sum / len(values)


def autocorrelation(values: Tuple[float, ...], lag: int) -> float:
    if type(values) is not tuple:
        raise TypeError("values must be exactly a tuple")
    if isinstance(lag, bool) or type(lag) is not int:
        raise TypeError("lag must be an int, not bool")
    if lag <= 0:
        raise ValueError("lag must be positive")
    if lag >= len(values):
        raise ValueError("lag must be less than the length of values")
        
    v = variance(values)
    if v == 0.0:
        return 0.0
        
    m = mean(values)
    cov_sum = 0.0
    for t in range(len(values) - lag):
        cov_sum += (values[t] - m) * (values[t + lag] - m)
        
    total_var_sum = sum((x - m) ** 2 for x in values)
    return cov_sum / total_var_sum


def series_diagnostics(values: Tuple[float, ...], max_lag: int = SERIES_DIAGNOSTIC_MAX_LAG) -> SeriesDiagnosticResult:
    if type(values) is not tuple:
        raise TypeError("values must be exactly a tuple")
    if len(values) < 2:
        raise ValueError("values length must be >= 2")
    for val in values:
        if isinstance(val, bool):
            raise TypeError("values must not contain bool")
        if not isinstance(val, (int, float)):
            raise TypeError("values must contain float or int")
        if not math.isfinite(val):
            raise ValueError("values must contain finite float or int")
            
    if isinstance(max_lag, bool) or type(max_lag) is not int:
        raise TypeError("max_lag must be an int, not bool")
    if max_lag < 1:
        raise ValueError("max_lag must be >= 1")
        
    n = len(values)
    m = mean(values)
    v = variance(values)
    
    actual_max_lag = min(max_lag, n - 1)
    
    acf_list = []
    for lag in range(1, actual_max_lag + 1):
        acf_list.append(autocorrelation(values, lag))
        
    # Compute squared centered values: (x - mean(values))**2
    sq_centered = tuple((x - m) ** 2 for x in values)
    
    squared_acf_list = []
    for lag in range(1, actual_max_lag + 1):
        squared_acf_list.append(autocorrelation(sq_centered, lag))
        
    volatility_clustering_score = 0.0
    if squared_acf_list:
        volatility_clustering_score = max(abs(r) for r in squared_acf_list)
        
    reason = f"squared_acf computed on squared centered values (x - mean(x))^2; volatility_clustering_score={volatility_clustering_score:.6f}"
    
    return SeriesDiagnosticResult(
        length=n,
        mean=m,
        variance=v,
        acf=tuple(acf_list),
        squared_acf=tuple(squared_acf_list),
        volatility_clustering_score=volatility_clustering_score,
        reason=reason
    )


def compute_seed_stability(valid_rates: Tuple[float, ...]) -> SeedStabilityResult:
    if type(valid_rates) is not tuple:
        raise TypeError("valid_rates must be exactly a tuple")
    if len(valid_rates) < MINIMUM_MODEL_REPEAT_SEEDS:
        raise ValueError(f"valid_rates count must be >= {MINIMUM_MODEL_REPEAT_SEEDS}")
    for r in valid_rates:
        validate_metric_probability(r, "valid_rate")
        
    mn = min(valid_rates)
    mx = max(valid_rates)
    r_range = mx - mn
    passes = r_range <= SEED_STABILITY_VALID_RATE_RANGE_MAX
    
    reason = f"range={r_range:.4f} (min={mn:.4f}, max={mx:.4f})"
    
    return SeedStabilityResult(
        seed_count=len(valid_rates),
        min_valid_rate=mn,
        max_valid_rate=mx,
        valid_rate_range=r_range,
        passes_seed_stability_threshold=passes,
        reason=reason
    )


def build_metric_bundle(
    validity: ValidityBatchResult,
    composition: Optional[CompositionResult],
    novelty: Optional[NoveltyResult],
    distribution_distance: Optional[DistributionDistanceResult],
    series_diagnostic: Optional[SeriesDiagnosticResult],
    seed_stability: Optional[SeedStabilityResult],
) -> MetricBundle:
    if not isinstance(validity, ValidityBatchResult):
        raise TypeError("validity must be a ValidityBatchResult")
    if composition is not None and not isinstance(composition, CompositionResult):
        raise TypeError("composition must be a CompositionResult or None")
    if novelty is not None and not isinstance(novelty, NoveltyResult):
        raise TypeError("novelty must be a NoveltyResult or None")
    if distribution_distance is not None and not isinstance(distribution_distance, DistributionDistanceResult):
        raise TypeError("distribution_distance must be a DistributionDistanceResult or None")
    if series_diagnostic is not None and not isinstance(series_diagnostic, SeriesDiagnosticResult):
        raise TypeError("series_diagnostic must be a SeriesDiagnosticResult or None")
    if seed_stability is not None and not isinstance(seed_stability, SeedStabilityResult):
        raise TypeError("seed_stability must be a SeedStabilityResult or None")
        
    reason = "MetricBundle built successfully"
    
    return MetricBundle(
        validity=validity,
        composition=composition,
        novelty=novelty,
        distribution_distance=distribution_distance,
        series_diagnostic=series_diagnostic,
        seed_stability=seed_stability,
        reason=reason
    )
