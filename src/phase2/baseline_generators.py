# src/phase2/baseline_generators.py

from dataclasses import dataclass, replace
from typing import Tuple, Optional
import math

from src.phase2.schema import FamilyId, MeanFamily, VolatilityFamily, ModelSpec, validate_model_spec
from src.phase2.constraints import require_math_valid
from src.phase2.sampler import GenerationRequest, generate_model_spec, validate_generation_request
from src.phase2.baseline_eval import (
    APPROVED_BASELINE_NAMES,
    BaselineEvaluationRequest,
    validate_baseline_evaluation_request,
)

# Public constants
BASELINE_GENERATOR_CONTRACT_VERSION = "phase2_p19_baseline_generators_v1"

SUPPORTED_BASELINE_GENERATOR_NAMES = (
    "copy_reference",
    "random_valid",
    "structural_composition_oracle",
)

COPY_REFERENCE_BASELINE_NAME = "copy_reference"
RANDOM_VALID_BASELINE_NAME = "random_valid"
STRUCTURAL_COMPOSITION_ORACLE_BASELINE_NAME = "structural_composition_oracle"


# Public dataclasses
@dataclass(frozen=True)
class BaselineCandidateSourceRecord:
    candidate_index: int
    generator_name: str
    family_id: str
    source_reference_indices: Tuple[int, ...]
    seed: Optional[int]
    reason: str


@dataclass(frozen=True)
class BaselineGenerationRequest:
    baseline_name: str
    reference_specs: Tuple[ModelSpec, ...]
    candidate_count: int
    seed: int
    random_valid_generation_templates: Optional[Tuple[Tuple[FamilyId, GenerationRequest], ...]]
    random_valid_family_schedule: Optional[Tuple[FamilyId, ...]]
    oracle_constraint_flags: Tuple[float, float, float, float]
    reason: str


@dataclass(frozen=True)
class BaselineGenerationResult:
    contract_version: str
    baseline_name: str
    candidate_count: int
    candidates: Tuple[ModelSpec, ...]
    source_records: Tuple[BaselineCandidateSourceRecord, ...]
    reason: str


# Public functions
def validate_baseline_generator_name(baseline_name: str) -> None:
    if type(baseline_name) is not str:
        raise TypeError("baseline_name must be exactly a str")
    if len(baseline_name) == 0:
        raise ValueError("baseline_name must not be empty")
    if baseline_name not in SUPPORTED_BASELINE_GENERATOR_NAMES:
        raise ValueError(f"Unknown baseline generator name: {baseline_name}")
    if baseline_name not in APPROVED_BASELINE_NAMES:
        raise ValueError(f"Name not in P18 APPROVED_BASELINE_NAMES: {baseline_name}")


def validate_positive_int(value: int, name: str) -> None:
    if isinstance(value, bool) or type(value) is not int:
        raise TypeError(f"{name} must be exactly an int, not bool")
    if value <= 0:
        raise ValueError(f"{name} must be positive, got {value}")


def validate_seed(value: int, name: str) -> None:
    if isinstance(value, bool) or type(value) is not int:
        raise TypeError(f"{name} must be exactly an int, not bool")
    if value < 0:
        raise ValueError(f"{name} must be non-negative, got {value}")


def validate_reference_specs(reference_specs: Tuple[ModelSpec, ...]) -> None:
    if type(reference_specs) is not tuple:
        raise TypeError("reference_specs must be exactly a tuple")
    if len(reference_specs) == 0:
        raise ValueError("reference_specs must not be empty")
    for i, spec in enumerate(reference_specs):
        if not isinstance(spec, ModelSpec):
            raise TypeError(f"reference_specs[{i}] is not a ModelSpec instance")
        validate_model_spec(spec)
        require_math_valid(spec)


def validate_oracle_constraint_flags(flags: Tuple[float, float, float, float]) -> None:
    if type(flags) is not tuple:
        raise TypeError("oracle_constraint_flags must be exactly a tuple")
    if len(flags) != 4:
        raise ValueError(f"oracle_constraint_flags must have length 4, got {len(flags)}")
    for i, val in enumerate(flags):
        if isinstance(val, bool) or not isinstance(val, (int, float)):
            raise TypeError(f"oracle_constraint_flags[{i}] must be numeric, got {type(val)}")
        if not math.isfinite(val):
            raise ValueError(f"oracle_constraint_flags[{i}] must be finite")


def validate_random_valid_templates(
    templates: Tuple[Tuple[FamilyId, GenerationRequest], ...],
    family_schedule: Tuple[FamilyId, ...],
) -> None:
    if type(templates) is not tuple:
        raise TypeError("templates must be exactly a tuple")
    if len(templates) == 0:
        raise ValueError("templates must not be empty")
        
    seen_families = set()
    template_families = set()
    
    for i, item in enumerate(templates):
        if type(item) is not tuple or len(item) != 2:
            raise TypeError(f"templates[{i}] must be a tuple of length 2")
        fam, req = item
        if not isinstance(fam, FamilyId):
            raise TypeError(f"templates[{i}][0] must be a FamilyId, got {type(fam)}")
        if not isinstance(req, GenerationRequest):
            raise TypeError(f"templates[{i}][1] must be a GenerationRequest, got {type(req)}")
        
        # validate generation request
        validate_generation_request(req)
        
        if fam in seen_families:
            raise ValueError(f"Duplicate FamilyId in templates: {fam}")
        seen_families.add(fam)
        template_families.add(fam)
        
    if type(family_schedule) is not tuple:
        raise TypeError("family_schedule must be exactly a tuple")
    if len(family_schedule) == 0:
        raise ValueError("family_schedule must not be empty")
        
    for i, fam in enumerate(family_schedule):
        if not isinstance(fam, FamilyId):
            raise TypeError(f"family_schedule[{i}] must be a FamilyId, got {type(fam)}")
        if fam not in template_families:
            raise ValueError(f"family_schedule[{i}] ({fam}) is missing from templates")


def validate_baseline_generation_request(request: BaselineGenerationRequest) -> None:
    if type(request) is not BaselineGenerationRequest:
        raise TypeError("request must be exactly a BaselineGenerationRequest instance")
        
    validate_baseline_generator_name(request.baseline_name)
    validate_reference_specs(request.reference_specs)
    validate_positive_int(request.candidate_count, "candidate_count")
    validate_seed(request.seed, "seed")
    validate_oracle_constraint_flags(request.oracle_constraint_flags)
    
    if type(request.reason) is not str:
        raise TypeError("reason must be exactly a str")
    if len(request.reason) == 0:
        raise ValueError("reason must not be empty")
        
    if request.baseline_name == RANDOM_VALID_BASELINE_NAME:
        if request.random_valid_generation_templates is None:
            raise ValueError("random_valid_generation_templates must not be None for random_valid baseline")
        if request.random_valid_family_schedule is None:
            raise ValueError("random_valid_family_schedule must not be None for random_valid baseline")
        validate_random_valid_templates(
            request.random_valid_generation_templates,
            request.random_valid_family_schedule
        )
    else:
        # If provided, validate them anyway
        if request.random_valid_generation_templates is not None and request.random_valid_family_schedule is not None:
            validate_random_valid_templates(
                request.random_valid_generation_templates,
                request.random_valid_family_schedule
            )
        elif request.random_valid_generation_templates is not None or request.random_valid_family_schedule is not None:
            raise ValueError("Both random_valid templates and schedule must be provided or both None")


def clone_model_spec_with_provenance(
    spec: ModelSpec,
    provenance: Tuple[Tuple[str, str], ...],
) -> ModelSpec:
    validate_model_spec(spec)
    
    if type(provenance) is not tuple:
        raise TypeError("provenance must be exactly a tuple")
    for i, pair in enumerate(provenance):
        if type(pair) is not tuple or len(pair) != 2:
            raise TypeError(f"provenance[{i}] must be a tuple of length 2")
        if type(pair[0]) is not str or type(pair[1]) is not str:
            raise TypeError(f"provenance[{i}] elements must be strings")
            
    cloned = replace(spec, provenance=provenance)
    validate_model_spec(cloned)
    require_math_valid(cloned)
    return cloned


def generate_copy_reference_candidates(request: BaselineGenerationRequest) -> BaselineGenerationResult:
    validate_baseline_generation_request(request)
    if request.baseline_name != COPY_REFERENCE_BASELINE_NAME:
        raise ValueError(f"Expected baseline_name='copy_reference', got '{request.baseline_name}'")
        
    candidates = []
    source_records = []
    
    n_ref = len(request.reference_specs)
    for i in range(request.candidate_count):
        source_index = i % n_ref
        ref_spec = request.reference_specs[source_index]
        
        provenance = (
            ("baseline_generator", COPY_REFERENCE_BASELINE_NAME),
            ("source_reference_index", str(source_index)),
            ("candidate_index", str(i)),
        )
        cloned = clone_model_spec_with_provenance(ref_spec, provenance)
        candidates.append(cloned)
        
        source_record = BaselineCandidateSourceRecord(
            candidate_index=i,
            generator_name=COPY_REFERENCE_BASELINE_NAME,
            family_id=ref_spec.family_id.value,
            source_reference_indices=(source_index,),
            seed=None,
            reason=f"Candidate {i} copied from reference index {source_index}"
        )
        source_records.append(source_record)
        
    reason = f"Generated {request.candidate_count} copy_reference candidates from {n_ref} references"
    return BaselineGenerationResult(
        contract_version=BASELINE_GENERATOR_CONTRACT_VERSION,
        baseline_name=COPY_REFERENCE_BASELINE_NAME,
        candidate_count=request.candidate_count,
        candidates=tuple(candidates),
        source_records=tuple(source_records),
        reason=reason
    )


def generate_random_valid_candidates(request: BaselineGenerationRequest) -> BaselineGenerationResult:
    validate_baseline_generation_request(request)
    if request.baseline_name != RANDOM_VALID_BASELINE_NAME:
        raise ValueError(f"Expected baseline_name='random_valid', got '{request.baseline_name}'")
        
    templates_dict = dict(request.random_valid_generation_templates)
    schedule = request.random_valid_family_schedule
    
    candidates = []
    source_records = []
    
    for i in range(request.candidate_count):
        family = schedule[i % len(schedule)]
        template_req = templates_dict[family]
        candidate_seed = request.seed + i
        
        # extend provenance in request
        prov_list = list(template_req.provenance) if template_req.provenance is not None else []
        prov_list.extend([
            ("baseline_generator", RANDOM_VALID_BASELINE_NAME),
            ("candidate_index", str(i)),
            ("seed", str(candidate_seed)),
        ])
        
        new_req = replace(
            template_req,
            seed=candidate_seed,
            provenance=tuple(prov_list)
        )
        
        # call P7 generate_model_spec
        gen_res = generate_model_spec(new_req)
        cand_spec = gen_res.spec
        
        validate_model_spec(cand_spec)
        require_math_valid(cand_spec)
        
        candidates.append(cand_spec)
        
        source_record = BaselineCandidateSourceRecord(
            candidate_index=i,
            generator_name=RANDOM_VALID_BASELINE_NAME,
            family_id=cand_spec.family_id.value,
            source_reference_indices=(),
            seed=candidate_seed,
            reason=f"Candidate {i} generated randomly with seed {candidate_seed}"
        )
        source_records.append(source_record)
        
    reason = f"Generated {request.candidate_count} random_valid candidates using templates"
    return BaselineGenerationResult(
        contract_version=BASELINE_GENERATOR_CONTRACT_VERSION,
        baseline_name=RANDOM_VALID_BASELINE_NAME,
        candidate_count=request.candidate_count,
        candidates=tuple(candidates),
        source_records=tuple(source_records),
        reason=reason
    )


def find_oracle_mean_sources(reference_specs: Tuple[ModelSpec, ...]) -> Tuple[Tuple[int, ModelSpec], ...]:
    validate_reference_specs(reference_specs)
    
    results = []
    for i, spec in enumerate(reference_specs):
        if (
            spec.family_id == FamilyId.ARMA and
            spec.mean_family == MeanFamily.ARMA and
            spec.volatility_family == VolatilityFamily.NONE and
            spec.p > 0 and
            spec.q > 0
        ):
            validate_model_spec(spec)
            require_math_valid(spec)
            results.append((i, spec))
            
    return tuple(results)


def find_oracle_volatility_sources(reference_specs: Tuple[ModelSpec, ...]) -> Tuple[Tuple[int, ModelSpec], ...]:
    validate_reference_specs(reference_specs)
    
    results = []
    for i, spec in enumerate(reference_specs):
        if (
            spec.family_id == FamilyId.GARCH and
            spec.mean_family == MeanFamily.NONE and
            spec.volatility_family == VolatilityFamily.GARCH and
            spec.r > 0 and
            spec.s > 0 and
            spec.omega is not None
        ):
            validate_model_spec(spec)
            require_math_valid(spec)
            results.append((i, spec))
            
    return tuple(results)


def compose_oracle_candidate(
    candidate_index: int,
    mean_source_index: int,
    mean_source: ModelSpec,
    volatility_source_index: int,
    volatility_source: ModelSpec,
    oracle_constraint_flags: Tuple[float, float, float, float],
) -> ModelSpec:
    if isinstance(candidate_index, bool) or type(candidate_index) is not int:
        raise TypeError("candidate_index must be exactly an int, not bool")
    if candidate_index < 0:
        raise ValueError("candidate_index must be >= 0")
        
    if isinstance(mean_source_index, bool) or type(mean_source_index) is not int:
        raise TypeError("mean_source_index must be exactly an int, not bool")
    if mean_source_index < 0:
        raise ValueError("mean_source_index must be >= 0")
        
    if isinstance(volatility_source_index, bool) or type(volatility_source_index) is not int:
        raise TypeError("volatility_source_index must be exactly an int, not bool")
    if volatility_source_index < 0:
        raise ValueError("volatility_source_index must be >= 0")
        
    # validate mean source is valid ARMA mean-only
    validate_model_spec(mean_source)
    require_math_valid(mean_source)
    if (
        mean_source.family_id != FamilyId.ARMA or
        mean_source.mean_family != MeanFamily.ARMA or
        mean_source.volatility_family != VolatilityFamily.NONE or
        mean_source.p <= 0 or
        mean_source.q <= 0
    ):
        raise ValueError("mean_source must be a valid ARMA mean-only source spec")
        
    # validate volatility source is valid GARCH volatility-only
    validate_model_spec(volatility_source)
    require_math_valid(volatility_source)
    if (
        volatility_source.family_id != FamilyId.GARCH or
        volatility_source.mean_family != MeanFamily.NONE or
        volatility_source.volatility_family != VolatilityFamily.GARCH or
        volatility_source.r <= 0 or
        volatility_source.s <= 0 or
        volatility_source.omega is None
    ):
        raise ValueError("volatility_source must be a valid GARCH volatility-only source spec")
        
    validate_oracle_constraint_flags(oracle_constraint_flags)
    
    provenance = (
        ("baseline_generator", STRUCTURAL_COMPOSITION_ORACLE_BASELINE_NAME),
        ("candidate_index", str(candidate_index)),
        ("mean_source_index", str(mean_source_index)),
        ("volatility_source_index", str(volatility_source_index)),
    )
    
    composed = ModelSpec(
        family_id=FamilyId.ARMA_GARCH,
        mean_family=MeanFamily.ARMA,
        volatility_family=VolatilityFamily.GARCH,
        p=mean_source.p,
        q=mean_source.q,
        r=volatility_source.r,
        s=volatility_source.s,
        ar_params=mean_source.ar_params,
        ma_params=mean_source.ma_params,
        omega=volatility_source.omega,
        alpha_params=volatility_source.alpha_params,
        beta_params=volatility_source.beta_params,
        constraint_flags=oracle_constraint_flags,
        provenance=provenance
    )
    
    validate_model_spec(composed)
    require_math_valid(composed)
    
    return composed


def generate_structural_composition_oracle_candidates(
    request: BaselineGenerationRequest,
) -> BaselineGenerationResult:
    validate_baseline_generation_request(request)
    if request.baseline_name != STRUCTURAL_COMPOSITION_ORACLE_BASELINE_NAME:
        raise ValueError(f"Expected baseline_name='structural_composition_oracle', got '{request.baseline_name}'")
        
    mean_sources = find_oracle_mean_sources(request.reference_specs)
    volatility_sources = find_oracle_volatility_sources(request.reference_specs)
    
    if len(mean_sources) == 0:
        raise ValueError("No valid ARMA mean-only sources found in reference specs")
    if len(volatility_sources) == 0:
        raise ValueError("No valid GARCH volatility-only sources found in reference specs")
        
    candidates = []
    source_records = []
    
    for i in range(request.candidate_count):
        mean_index, mean_spec = mean_sources[i % len(mean_sources)]
        vol_index, vol_spec = volatility_sources[i % len(volatility_sources)]
        
        composed = compose_oracle_candidate(
            candidate_index=i,
            mean_source_index=mean_index,
            mean_source=mean_spec,
            volatility_source_index=vol_index,
            volatility_source=vol_spec,
            oracle_constraint_flags=request.oracle_constraint_flags
        )
        candidates.append(composed)
        
        source_record = BaselineCandidateSourceRecord(
            candidate_index=i,
            generator_name=STRUCTURAL_COMPOSITION_ORACLE_BASELINE_NAME,
            family_id=composed.family_id.value,
            source_reference_indices=(mean_index, vol_index),
            seed=None,
            reason=f"Candidate {i} composed from mean source index {mean_index} and volatility source index {vol_index}"
        )
        source_records.append(source_record)
        
    reason = f"Generated {request.candidate_count} structural_composition_oracle candidates from {len(mean_sources)} mean and {len(volatility_sources)} volatility sources"
    return BaselineGenerationResult(
        contract_version=BASELINE_GENERATOR_CONTRACT_VERSION,
        baseline_name=STRUCTURAL_COMPOSITION_ORACLE_BASELINE_NAME,
        candidate_count=request.candidate_count,
        candidates=tuple(candidates),
        source_records=tuple(source_records),
        reason=reason
    )


def generate_baseline_candidates(request: BaselineGenerationRequest) -> BaselineGenerationResult:
    validate_baseline_generation_request(request)
    
    if request.baseline_name == COPY_REFERENCE_BASELINE_NAME:
        return generate_copy_reference_candidates(request)
    elif request.baseline_name == RANDOM_VALID_BASELINE_NAME:
        return generate_random_valid_candidates(request)
    elif request.baseline_name == STRUCTURAL_COMPOSITION_ORACLE_BASELINE_NAME:
        return generate_structural_composition_oracle_candidates(request)
    else:
        raise ValueError(f"Unknown baseline name: {request.baseline_name}")


def baseline_generation_result_to_evaluation_request(
    result: BaselineGenerationResult,
    reference_specs: Tuple[ModelSpec, ...],
    candidate_series_values: Optional[Tuple[Tuple[float, ...], ...]],
    seed_valid_rates: Optional[Tuple[float, ...]],
    require_candidate_composition: bool,
    reason: str,
) -> BaselineEvaluationRequest:
    if type(result) is not BaselineGenerationResult:
        raise TypeError("result must be exactly a BaselineGenerationResult instance")
        
    validate_reference_specs(reference_specs)
    
    if candidate_series_values is not None:
        if type(candidate_series_values) is not tuple:
            raise TypeError("candidate_series_values must be exactly a tuple")
        if len(candidate_series_values) != result.candidate_count:
            raise ValueError(f"candidate_series_values length must match result.candidate_count ({result.candidate_count}), got {len(candidate_series_values)}")
            
    if seed_valid_rates is not None:
        if type(seed_valid_rates) is not tuple:
            raise TypeError("seed_valid_rates must be exactly a tuple")
            
    if type(require_candidate_composition) is not bool:
        raise TypeError("require_candidate_composition must be exactly a bool")
        
    if type(reason) is not str:
        raise TypeError("reason must be exactly a str")
    if len(reason) == 0:
        raise ValueError("reason must not be empty")
        
    eval_req = BaselineEvaluationRequest(
        baseline_name=result.baseline_name,
        reference_specs=reference_specs,
        candidate_specs=result.candidates,
        candidate_series_values=candidate_series_values,
        seed_valid_rates=seed_valid_rates,
        require_candidate_composition=require_candidate_composition,
        reason=reason
    )
    
    validate_baseline_evaluation_request(eval_req)
    return eval_req
