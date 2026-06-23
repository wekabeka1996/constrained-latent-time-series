# src/phase2/baseline_eval.py

from dataclasses import dataclass
from typing import Tuple, Optional
import math

from src.phase2.schema import ModelSpec, validate_model_spec
from src.phase2.constraints import require_math_valid
from src.phase2.metrics import (
    ValidityBatchResult,
    CompositionResult,
    NoveltyResult,
    DistributionDistanceResult,
    SeriesDiagnosticResult,
    SeedStabilityResult,
    MetricBundle,
    validate_metric_probability,
    score_composition,
    score_model_spec_novelty,
    compute_validity_batch,
    compute_mmd_rbf,
    series_diagnostics,
    compute_seed_stability,
)

# Public constants
BASELINE_EVAL_CONTRACT_VERSION = "phase2_p18_baseline_eval_v1"

APPROVED_BASELINE_NAMES = (
    "copy_reference",
    "random_valid",
    "structural_composition_oracle",
)


# Public dataclasses
@dataclass(frozen=True)
class BaselineEvaluationRequest:
    baseline_name: str
    reference_specs: Tuple[ModelSpec, ...]
    candidate_specs: Tuple[ModelSpec, ...]
    candidate_series_values: Optional[Tuple[Tuple[float, ...], ...]]
    seed_valid_rates: Optional[Tuple[float, ...]]
    require_candidate_composition: bool
    reason: str


@dataclass(frozen=True)
class CandidateMetricRecord:
    candidate_index: int
    family_id: str
    validity: ValidityBatchResult
    composition: Optional[CompositionResult]
    novelty: Optional[NoveltyResult]
    series_diagnostic: Optional[SeriesDiagnosticResult]
    reason: str


@dataclass(frozen=True)
class BaselineEvaluationResult:
    contract_version: str
    baseline_name: str
    candidate_count: int
    reference_count: int
    validity: ValidityBatchResult
    composition_pass_count: int
    composition_pass_rate: float
    novelty_pass_count: int
    novelty_pass_rate: float
    distribution_distance: DistributionDistanceResult
    seed_stability: Optional[SeedStabilityResult]
    candidate_records: Tuple[CandidateMetricRecord, ...]
    reason: str


# Public functions
def validate_baseline_name(baseline_name: str) -> None:
    if type(baseline_name) is not str:
        raise TypeError("baseline_name must be exactly a str")
    if len(baseline_name) == 0:
        raise ValueError("baseline_name must not be empty")
    if baseline_name not in APPROVED_BASELINE_NAMES:
        raise ValueError(f"Unknown baseline name: {baseline_name}")


def validate_baseline_evaluation_request(request: BaselineEvaluationRequest) -> None:
    if not isinstance(request, BaselineEvaluationRequest):
        raise TypeError("request must be exactly a BaselineEvaluationRequest instance")
        
    validate_baseline_name(request.baseline_name)
    
    # reference_specs
    if type(request.reference_specs) is not tuple:
        raise TypeError("reference_specs must be exactly a tuple")
    if len(request.reference_specs) == 0:
        raise ValueError("reference_specs must not be empty")
    for i, spec in enumerate(request.reference_specs):
        if not isinstance(spec, ModelSpec):
            raise TypeError(f"reference_specs[{i}] is not a ModelSpec instance")
        validate_model_spec(spec)
        require_math_valid(spec)
        
    # candidate_specs
    if type(request.candidate_specs) is not tuple:
        raise TypeError("candidate_specs must be exactly a tuple")
    if len(request.candidate_specs) == 0:
        raise ValueError("candidate_specs must not be empty")
    for i, spec in enumerate(request.candidate_specs):
        if not isinstance(spec, ModelSpec):
            raise TypeError(f"candidate_specs[{i}] is not a ModelSpec instance")
        validate_model_spec(spec)
        require_math_valid(spec)
        
    # candidate_series_values
    if request.candidate_series_values is not None:
        if type(request.candidate_series_values) is not tuple:
            raise TypeError("candidate_series_values must be exactly a tuple")
        if len(request.candidate_series_values) != len(request.candidate_specs):
            raise ValueError("candidate_series_values length must match candidate_specs")
        for i, series in enumerate(request.candidate_series_values):
            if type(series) is not tuple:
                raise TypeError(f"candidate_series_values[{i}] must be exactly a tuple")
            if len(series) < 2:
                raise ValueError(f"candidate_series_values[{i}] length must be >= 2")
            for j, val in enumerate(series):
                if isinstance(val, bool):
                    raise TypeError(f"candidate_series_values[{i}][{j}] must not be a bool")
                if not isinstance(val, (int, float)):
                    raise TypeError(f"candidate_series_values[{i}][{j}] must be numeric")
                if not math.isfinite(val):
                    raise ValueError(f"candidate_series_values[{i}][{j}] must be finite")
                    
    # seed_valid_rates
    if request.seed_valid_rates is not None:
        if type(request.seed_valid_rates) is not tuple:
            raise TypeError("seed_valid_rates must be exactly a tuple")
        from src.phase2.metrics import MINIMUM_MODEL_REPEAT_SEEDS
        if len(request.seed_valid_rates) < MINIMUM_MODEL_REPEAT_SEEDS:
            raise ValueError(f"seed_valid_rates count must be >= {MINIMUM_MODEL_REPEAT_SEEDS}")
        for i, rate in enumerate(request.seed_valid_rates):
            validate_metric_probability(rate, f"seed_valid_rates[{i}]")
            
    # require_candidate_composition
    if type(request.require_candidate_composition) is not bool:
        raise TypeError("require_candidate_composition must be exactly a bool")
        
    # reason
    if type(request.reason) is not str:
        raise TypeError("reason must be exactly a str")
    if len(request.reason) == 0:
        raise ValueError("reason must not be empty")


def composition_pass_rate(records: Tuple[CandidateMetricRecord, ...]) -> float:
    if type(records) is not tuple:
        raise TypeError("records must be exactly a tuple")
    if len(records) == 0:
        return 0.0
    pass_count = 0
    for i, r in enumerate(records):
        if not isinstance(r, CandidateMetricRecord):
            raise TypeError(f"Item {i} in records is not a CandidateMetricRecord instance")
        if r.composition is not None and r.composition.passes_composition_threshold:
            pass_count += 1
    return float(pass_count) / len(records)


def novelty_pass_rate(records: Tuple[CandidateMetricRecord, ...]) -> float:
    if type(records) is not tuple:
        raise TypeError("records must be exactly a tuple")
    if len(records) == 0:
        return 0.0
    pass_count = 0
    for i, r in enumerate(records):
        if not isinstance(r, CandidateMetricRecord):
            raise TypeError(f"Item {i} in records is not a CandidateMetricRecord instance")
        if r.novelty is not None and r.novelty.passes_novelty_threshold:
            pass_count += 1
    return float(pass_count) / len(records)


def evaluate_candidate_record(
    candidate_index: int,
    candidate_spec: ModelSpec,
    reference_specs: Tuple[ModelSpec, ...],
    candidate_series_values: Optional[Tuple[float, ...]] = None,
    require_candidate_composition: bool = True,
) -> CandidateMetricRecord:
    if isinstance(candidate_index, bool) or type(candidate_index) is not int:
        raise TypeError("candidate_index must be exactly an int, not bool")
    if candidate_index < 0:
        raise ValueError("candidate_index must be >= 0")
        
    validate_model_spec(candidate_spec)
    require_math_valid(candidate_spec)
    
    if type(reference_specs) is not tuple:
        raise TypeError("reference_specs must be exactly a tuple")
    if len(reference_specs) == 0:
        raise ValueError("reference_specs must not be empty")
    for r in reference_specs:
        validate_model_spec(r)
        require_math_valid(r)
        
    if type(require_candidate_composition) is not bool:
        raise TypeError("require_candidate_composition must be exactly a bool")
        
    # validity
    validity = compute_validity_batch((candidate_spec,))
    
    # composition
    composition = score_composition(candidate_spec)
    
    # novelty
    novelty = score_model_spec_novelty(candidate_spec, reference_specs)
    
    # series_diagnostic
    series_diag = None
    if candidate_series_values is not None:
        if type(candidate_series_values) is not tuple:
            raise TypeError("candidate_series_values must be exactly a tuple")
        series_diag = series_diagnostics(candidate_series_values)
        
    family_id = candidate_spec.family_id.value
    reason = f"candidate_index={candidate_index} evaluated successfully"
    
    return CandidateMetricRecord(
        candidate_index=candidate_index,
        family_id=family_id,
        validity=validity,
        composition=composition,
        novelty=novelty,
        series_diagnostic=series_diag,
        reason=reason
    )


def evaluate_baseline_request(request: BaselineEvaluationRequest) -> BaselineEvaluationResult:
    validate_baseline_evaluation_request(request)
    
    # compute aggregate validity
    aggregate_validity = compute_validity_batch(request.candidate_specs)
    
    # compute per-candidate records
    records = []
    for i, candidate in enumerate(request.candidate_specs):
        series = None
        if request.candidate_series_values is not None:
            series = request.candidate_series_values[i]
            
        record = evaluate_candidate_record(
            candidate_index=i,
            candidate_spec=candidate,
            reference_specs=request.reference_specs,
            candidate_series_values=series,
            require_candidate_composition=request.require_candidate_composition
        )
        records.append(record)
        
    records_tuple = tuple(records)
    
    # composition pass count and rate
    composition_passes = 0
    for r in records_tuple:
        if r.composition is not None and r.composition.passes_composition_threshold:
            composition_passes += 1
            
    comp_pass_rate = composition_pass_rate(records_tuple)
    
    # novelty pass count and rate
    novelty_passes = 0
    for r in records_tuple:
        if r.novelty is not None and r.novelty.passes_novelty_threshold:
            novelty_passes += 1
            
    nov_pass_rate = novelty_pass_rate(records_tuple)
    
    # distribution distance
    dist_distance = compute_mmd_rbf(request.reference_specs, request.candidate_specs)
    
    # seed stability
    seed_stability_res = None
    if request.seed_valid_rates is not None:
        seed_stability_res = compute_seed_stability(request.seed_valid_rates)
        
    reason = f"Baseline '{request.baseline_name}' evaluated successfully with {len(request.candidate_specs)} candidates"
    
    return BaselineEvaluationResult(
        contract_version=BASELINE_EVAL_CONTRACT_VERSION,
        baseline_name=request.baseline_name,
        candidate_count=len(request.candidate_specs),
        reference_count=len(request.reference_specs),
        validity=aggregate_validity,
        composition_pass_count=composition_passes,
        composition_pass_rate=comp_pass_rate,
        novelty_pass_count=novelty_passes,
        novelty_pass_rate=nov_pass_rate,
        distribution_distance=dist_distance,
        seed_stability=seed_stability_res,
        candidate_records=records_tuple,
        reason=reason
    )


def build_baseline_metric_bundle(result: BaselineEvaluationResult) -> MetricBundle:
    if not isinstance(result, BaselineEvaluationResult):
        raise TypeError("result must be a BaselineEvaluationResult instance")
        
    validity = result.validity
    
    composition = None
    novelty = None
    if len(result.candidate_records) > 0:
        composition = result.candidate_records[0].composition
        novelty = result.candidate_records[0].novelty
        
    distribution_distance = result.distribution_distance
    
    series_diagnostic = None
    for r in result.candidate_records:
        if r.series_diagnostic is not None:
            series_diagnostic = r.series_diagnostic
            break
            
    seed_stability = result.seed_stability
    reason = "baseline_metric_bundle_from_p18"
    
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
        
    return MetricBundle(
        validity=validity,
        composition=composition,
        novelty=novelty,
        distribution_distance=distribution_distance,
        series_diagnostic=series_diagnostic,
        seed_stability=seed_stability,
        reason=reason
    )
