# tools/phase2/run_p20_baseline_evidence_smoke.py

import json
import math
import sys
from typing import Tuple, Dict, Any, Optional

from src.phase2.schema import FamilyId, MeanFamily, VolatilityFamily, ModelSpec, validate_model_spec
from src.phase2.constraints import require_math_valid
from src.phase2.sampler import GenerationRequest
from src.phase2.baseline_generators import (
    BaselineGenerationRequest,
    BaselineGenerationResult,
    generate_baseline_candidates,
    baseline_generation_result_to_evaluation_request,
)
from src.phase2.baseline_eval import (
    BaselineEvaluationResult,
    evaluate_baseline_request,
    build_baseline_metric_bundle,
)


def compact_json(data: dict) -> str:
    return json.dumps(data, sort_keys=True, separators=(",", ":"))


def assert_no_path_leakage(data: Any) -> None:
    if isinstance(data, dict):
        for k, v in data.items():
            assert_no_path_leakage(k)
            assert_no_path_leakage(v)
    elif isinstance(data, (list, tuple)):
        for item in data:
            assert_no_path_leakage(item)
    elif isinstance(data, str):
        bad_patterns = ["file:///", "C:/", "C:\\", "/home/", "/Users/"]
        for pat in bad_patterns:
            if pat in data:
                raise ValueError(f"Path leakage detected in: '{data}'")


def build_p20_reference_specs() -> Tuple[ModelSpec, ...]:
    arma_spec = ModelSpec(
        family_id=FamilyId.ARMA,
        mean_family=MeanFamily.ARMA,
        volatility_family=VolatilityFamily.NONE,
        p=1,
        q=1,
        r=0,
        s=0,
        ar_params=(0.35,),
        ma_params=(-0.25,),
        omega=None,
        alpha_params=(),
        beta_params=(),
        constraint_flags=(1.0, 0.0, 0.0, 0.0),
        provenance=(("fixture", "p20_reference_arma"),)
    )
    
    garch_spec = ModelSpec(
        family_id=FamilyId.GARCH,
        mean_family=MeanFamily.NONE,
        volatility_family=VolatilityFamily.GARCH,
        p=0,
        q=0,
        r=1,
        s=1,
        ar_params=(),
        ma_params=(),
        omega=0.40,
        alpha_params=(0.08,),
        beta_params=(0.75,),
        constraint_flags=(1.0, 0.0, 0.0, 0.0),
        provenance=(("fixture", "p20_reference_garch"),)
    )
    
    validate_model_spec(arma_spec)
    require_math_valid(arma_spec)
    validate_model_spec(garch_spec)
    require_math_valid(garch_spec)
    
    return (arma_spec, garch_spec)


def build_p20_random_valid_templates() -> Tuple[Tuple[FamilyId, GenerationRequest], ...]:
    template_ar = GenerationRequest(
        family_id=FamilyId.AR,
        seed=0,
        p=1,
        q=0,
        r=0,
        s=0,
        ar_range=(-0.40, 0.40),
        ma_range=(0.0, 0.0),
        omega_range=(0.1, 0.5),
        alpha_range=(0.01, 0.10),
        beta_range=(0.30, 0.70),
        constraint_flags=(1.0, 0.0, 0.0, 0.0),
        provenance=(("fixture", "p20_random_template_ar"),),
        max_attempts=100
    )
    
    template_garch = GenerationRequest(
        family_id=FamilyId.GARCH,
        seed=0,
        p=0,
        q=0,
        r=1,
        s=1,
        ar_range=(0.0, 0.0),
        ma_range=(0.0, 0.0),
        omega_range=(0.10, 0.50),
        alpha_range=(0.01, 0.10),
        beta_range=(0.30, 0.70),
        constraint_flags=(1.0, 0.0, 0.0, 0.0),
        provenance=(("fixture", "p20_random_template_garch"),),
        max_attempts=100
    )
    
    template_arma_garch = GenerationRequest(
        family_id=FamilyId.ARMA_GARCH,
        seed=0,
        p=1,
        q=1,
        r=1,
        s=1,
        ar_range=(-0.35, 0.35),
        ma_range=(-0.35, 0.35),
        omega_range=(0.10, 0.50),
        alpha_range=(0.01, 0.10),
        beta_range=(0.30, 0.70),
        constraint_flags=(1.0, 0.0, 0.0, 0.0),
        provenance=(("fixture", "p20_random_template_arma_garch"),),
        max_attempts=100
    )
    
    return (
        (FamilyId.AR, template_ar),
        (FamilyId.GARCH, template_garch),
        (FamilyId.ARMA_GARCH, template_arma_garch),
    )


def build_p20_random_valid_family_schedule() -> Tuple[FamilyId, ...]:
    return (
        FamilyId.AR,
        FamilyId.GARCH,
        FamilyId.ARMA_GARCH,
    )


def build_p20_generation_request(
    baseline_name: str,
    reference_specs: Tuple[ModelSpec, ...],
) -> BaselineGenerationRequest:
    from src.phase2.baseline_generators import validate_baseline_generation_request
    
    templates = None
    schedule = None
    if baseline_name == "random_valid":
        templates = build_p20_random_valid_templates()
        schedule = build_p20_random_valid_family_schedule()
        
    req = BaselineGenerationRequest(
        baseline_name=baseline_name,
        reference_specs=reference_specs,
        candidate_count=6,
        seed=19001,
        random_valid_generation_templates=templates,
        random_valid_family_schedule=schedule,
        oracle_constraint_flags=(1.0, 0.0, 0.0, 0.0),
        reason="p20_bounded_baseline_evidence_smoke_generation"
    )
    
    validate_baseline_generation_request(req)
    return req


def summarize_generation_result(result: BaselineGenerationResult) -> dict:
    source_records_summary = []
    for rec in result.source_records:
        source_records_summary.append({
            "candidate_index": rec.candidate_index,
            "generator_name": rec.generator_name,
            "family_id": rec.family_id,
            "source_reference_indices": rec.source_reference_indices,
            "seed": rec.seed
        })
        
    return {
        "baseline_name": result.baseline_name,
        "candidate_count": result.candidate_count,
        "candidate_family_ids": [cand.family_id.value for cand in result.candidates],
        "source_records_summary": source_records_summary
    }


def summarize_evaluation_result(result: BaselineEvaluationResult) -> dict:
    return {
        "baseline_name": result.baseline_name,
        "candidate_count": result.candidate_count,
        "reference_count": result.reference_count,
        "validity_valid_count": result.validity.valid_count,
        "validity_total_count": result.validity.total_count,
        "validity_valid_rate": result.validity.valid_rate,
        "composition_pass_count": result.composition_pass_count,
        "composition_pass_rate": result.composition_pass_rate,
        "novelty_pass_count": result.novelty_pass_count,
        "novelty_pass_rate": result.novelty_pass_rate,
        "distribution_mmd_rbf": result.distribution_distance.mmd_rbf,
        "seed_stability_present": result.seed_stability is not None,
        "candidate_record_count": len(result.candidate_records)
    }


def run_single_p20_baseline(
    baseline_name: str,
    reference_specs: Tuple[ModelSpec, ...],
) -> dict:
    gen_req = build_p20_generation_request(baseline_name, reference_specs)
    gen_res = generate_baseline_candidates(gen_req)
    
    eval_req = baseline_generation_result_to_evaluation_request(
        result=gen_res,
        reference_specs=reference_specs,
        candidate_series_values=None,
        seed_valid_rates=None,
        require_candidate_composition=True,
        reason="p20_bounded_baseline_evidence_smoke_evaluation"
    )
    
    eval_res = evaluate_baseline_request(eval_req)
    
    # verify bundle bridge availability
    bundle = build_baseline_metric_bundle(eval_res)
    metric_bundle_bridge_verified = (bundle is not None)
    
    return {
        "baseline_name": baseline_name,
        "generation_summary": summarize_generation_result(gen_res),
        "evaluation_summary": summarize_evaluation_result(eval_res),
        "metric_bundle_bridge_verified": metric_bundle_bridge_verified,
        "reason": f"p20_bounded_baseline_evidence_smoke_single_baseline_completed: {baseline_name}"
    }


def run_p20_baseline_evidence_smoke() -> dict:
    ref_specs = build_p20_reference_specs()
    
    baseline_summaries = []
    for name in ["copy_reference", "random_valid", "structural_composition_oracle"]:
        res = run_single_p20_baseline(name, ref_specs)
        baseline_summaries.append(res)
        
    summary = {
        "verdict": "PASS",
        "contract": "phase2_p20_bounded_baseline_evidence_smoke_v1",
        "baseline_count": 3,
        "reference_count": 2,
        "candidate_count_per_baseline": 6,
        "baselines": baseline_summaries,
        "no_artifact_dependency": True,
        "no_model_training": True,
        "no_scientific_conclusion": True,
        "reason": "p20_bounded_baseline_evidence_smoke_completed"
    }
    
    assert_no_path_leakage(summary)
    return summary


def main() -> int:
    if len(sys.argv) > 1:
        sys.stderr.write("Error: run_p20_baseline_evidence_smoke.py does not accept arguments\n")
        return 1
        
    try:
        summary = run_p20_baseline_evidence_smoke()
        sys.stdout.write(compact_json(summary) + "\n")
        return 0
    except Exception as e:
        sys.stderr.write(f"Error executing P20 smoke run: {str(e)}\n")
        return 1


if __name__ == "__main__":
    sys.exit(main())
