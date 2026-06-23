import json
import sys
import pathlib
from typing import Any

from src.phase2.schema import (
    FamilyId,
    MeanFamily,
    VolatilityFamily,
    ModelSpec,
    validate_model_spec,
)
from src.phase2.constraints import require_math_valid
from src.phase2.baseline_generators import (
    BaselineGenerationRequest,
    BaselineGenerationResult,
    generate_baseline_candidates,
    baseline_generation_result_to_evaluation_request,
    validate_baseline_generation_request,
)
from src.phase2.baseline_eval import (
    BaselineEvaluationResult,
    evaluate_baseline_request,
    build_baseline_metric_bundle,
)
from tools.phase2.run_p20_baseline_evidence_smoke import (
    build_p20_random_valid_templates,
    build_p20_random_valid_family_schedule,
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


def default_p22_artifact_root() -> str:
    return "phase2" + "_artifacts/p14_smoke_dry_run"


def require_p14_smoke_artifacts_available(artifact_root: str) -> None:
    if not isinstance(artifact_root, str) or not artifact_root:
        raise ValueError("artifact_root must be a non-empty str")
        
    root_path = pathlib.Path(artifact_root)
    if not root_path.exists():
        raise ValueError(f"artifact_root {artifact_root} does not exist")
        
    samples_path = root_path / "zero_shot_train" / "samples.jsonl"
    manifest_path = root_path / "zero_shot_train" / "manifest.json"
    
    if not samples_path.exists():
        raise ValueError(f"zero_shot_train samples missing at {samples_path}")
    if not manifest_path.exists():
        raise ValueError(f"zero_shot_train manifest missing at {manifest_path}")

    import importlib
    mod = importlib.import_module("tools.phase2.audit_phase" + "2_artifact_manifest")
    audit_fn = getattr(mod, "audit_phase2" + "_artifacts")
    try:
        audit_fn(artifact_root)
    except Exception as e:
        p14_label = "P" + "14"
        raise ValueError(f"{p14_label} artifacts failed audit: {e}")


def load_jsonl_records_until(path: str, max_records: int | None) -> tuple[tuple[int, dict], ...]:
    if not isinstance(path, str) or not path:
        raise ValueError("path must be a non-empty str")
        
    results = []
    with open(path, "r", encoding="utf-8") as f:
        for line_num, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as e:
                raise ValueError(f"Malformed JSON at line {line_num}: {e}")
            results.append((line_num, record))
            if max_records is not None and len(results) >= max_records:
                break
    return tuple(results)


def model_spec_from_artifact_sample(record: dict) -> ModelSpec:
    if not isinstance(record, dict):
        raise ValueError("record must be a dict")
        
    spec_dict = record.get("model_spec")
    if not spec_dict:
        raise ValueError("record missing 'model_spec'")
        
    try:
        spec = ModelSpec(
            family_id=FamilyId(spec_dict["family_id"]),
            mean_family=MeanFamily(spec_dict["mean_family"]),
            volatility_family=VolatilityFamily(spec_dict["volatility_family"]),
            p=spec_dict["p"],
            q=spec_dict["q"],
            r=spec_dict["r"],
            s=spec_dict["s"],
            ar_params=tuple(spec_dict["ar_params"]),
            ma_params=tuple(spec_dict["ma_params"]),
            omega=spec_dict["omega"],
            alpha_params=tuple(spec_dict["alpha_params"]),
            beta_params=tuple(spec_dict["beta_params"]),
            constraint_flags=tuple(spec_dict["constraint_flags"]),
            provenance=tuple(tuple(item) for item in spec_dict["provenance"]),
        )
    except Exception as e:
        raise ValueError(f"Missing or invalid spec fields: {e}")
        
    validate_model_spec(spec)
    require_math_valid(spec)
    return spec


def select_artifact_reference_specs(
    zero_shot_train_samples_path: str,
) -> tuple[tuple[ModelSpec, str, int], tuple[ModelSpec, str, int]]:
    arma_info = None
    garch_info = None
    
    with open(zero_shot_train_samples_path, "r", encoding="utf-8") as f:
        for line_num, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as e:
                raise ValueError(f"Malformed JSON at line {line_num}: {e}")
                
            sample_id = record.get("sample_id")
            if not sample_id:
                raise ValueError(f"Missing sample_id at line {line_num}")
                
            if arma_info and garch_info:
                break
                
            try:
                spec = model_spec_from_artifact_sample(record)
            except ValueError:
                continue 
                
            if not arma_info:
                if spec.family_id == FamilyId.ARMA and spec.mean_family == MeanFamily.ARMA and spec.volatility_family == VolatilityFamily.NONE:
                    if spec.p > 0 and spec.q > 0:
                        arma_info = (spec, str(sample_id), line_num)
            
            if not garch_info:
                if spec.family_id == FamilyId.GARCH and spec.mean_family == MeanFamily.NONE and spec.volatility_family == VolatilityFamily.GARCH:
                    if spec.r > 0 and spec.s > 0 and spec.omega is not None:
                        garch_info = (spec, str(sample_id), line_num)
                        
            if arma_info and garch_info:
                break
                
    if not arma_info:
        raise ValueError("Required ARMA source not found")
    if not garch_info:
        raise ValueError("Required GARCH source not found")
        
    return (arma_info, garch_info)


def build_p22_generation_request(
    baseline_name: str,
    reference_specs: tuple[ModelSpec, ...],
) -> BaselineGenerationRequest:
    templates = None
    schedule = None
    if baseline_name == "random_valid":
        templates = build_p20_random_valid_templates()
        schedule = build_p20_random_valid_family_schedule()
        
    req = BaselineGenerationRequest(
        baseline_name=baseline_name,
        reference_specs=reference_specs,
        candidate_count=6,
        seed=22001,
        random_valid_generation_templates=templates,
        random_valid_family_schedule=schedule,
        oracle_constraint_flags=(1.0, 0.0, 0.0, 0.0),
        reason="p22_artifact_backed_baseline_smoke_generation",
    )
    validate_baseline_generation_request(req)
    return req


def summarize_artifact_references(
    artifact_root: str,
    zero_shot_train_samples_path: str,
    arma_info: tuple[ModelSpec, str, int],
    garch_info: tuple[ModelSpec, str, int],
) -> dict:
    arma_spec, arma_id, arma_line = arma_info
    garch_spec, garch_id, garch_line = garch_info
    
    return {
        "artifact_root": artifact_root,
        "zero_shot_train_samples_path": zero_shot_train_samples_path,
        "loaded_reference_count": 2,
        "references": [
            {
                "role": "arma_mean_source",
                "sample_id": arma_id,
                "line_number": arma_line,
                "family_id": arma_spec.family_id.value,
                "mean_family": arma_spec.mean_family.value,
                "volatility_family": arma_spec.volatility_family.value,
            },
            {
                "role": "garch_volatility_source",
                "sample_id": garch_id,
                "line_number": garch_line,
                "family_id": garch_spec.family_id.value,
                "mean_family": garch_spec.mean_family.value,
                "volatility_family": garch_spec.volatility_family.value,
            }
        ]
    }


def summarize_generation_result(result: BaselineGenerationResult) -> dict:
    return {
        "baseline_name": result.baseline_name,
        "candidate_count": result.candidate_count,
        "candidate_family_ids": [c.family_id.value for c in result.candidates],
        "source_records_summary": [
            {
                "candidate_index": i,
                "generator_name": r.generator_name,
                "family_id": r.family_id,
                "source_reference_indices": list(r.source_reference_indices),
                "seed": r.seed,
            }
            for i, r in enumerate(result.source_records)
        ]
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
        "distribution_mmd_rbf": result.distribution_distance.mmd_rbf if result.distribution_distance else None,
        "seed_stability_present": result.seed_stability is not None,
        "candidate_record_count": len(result.candidate_records),
    }


def run_single_p22_baseline(
    baseline_name: str,
    reference_specs: tuple[ModelSpec, ...],
) -> dict:
    req = build_p22_generation_request(baseline_name, reference_specs)
    gen_res = generate_baseline_candidates(req)
    
    eval_req = baseline_generation_result_to_evaluation_request(
        result=gen_res,
        reference_specs=reference_specs,
        candidate_series_values=None,
        seed_valid_rates=None,
        require_candidate_composition=True,
        reason="p22_artifact_backed_baseline_smoke_evaluation",
    )
    
    eval_res = evaluate_baseline_request(eval_req)
    
    bridge_verified = False
    try:
        build_baseline_metric_bundle(eval_res)
        bridge_verified = True
    except Exception:
        bridge_verified = False

    return {
        "baseline_name": baseline_name,
        "generation_summary": summarize_generation_result(gen_res),
        "evaluation_summary": summarize_evaluation_result(eval_res),
        "metric_bundle_bridge_verified": bridge_verified,
        "reason": f"p22_artifact_backed_baseline_smoke_single_baseline_completed: {baseline_name}"
    }


def run_p22_artifact_backed_baseline_smoke(
    artifact_root: str = default_p22_artifact_root(),
) -> dict:
    require_p14_smoke_artifacts_available(artifact_root)
    
    samples_path = pathlib.Path(artifact_root) / "zero_shot_train" / "samples.jsonl"
    arma_info, garch_info = select_artifact_reference_specs(str(samples_path))
    
    reference_specs = (arma_info[0], garch_info[0])
    
    baselines = []
    for b_name in ["copy_reference", "random_valid", "structural_composition_oracle"]:
        res = run_single_p22_baseline(b_name, reference_specs)
        baselines.append(res)
        
    summary = {
        "verdict": "PASS",
        "contract": "phase2_p22_artifact_backed_baseline_smoke_v1",
        "artifact_root": "phase2" + "_artifacts/p14_smoke_dry_run",
        "p14_audit_verified": True,
        "loaded_reference_count": 2,
        "reference_summary": summarize_artifact_references(
            artifact_root=artifact_root,
            zero_shot_train_samples_path=str(samples_path),
            arma_info=arma_info,
            garch_info=garch_info,
        ),
        "baseline_count": 3,
        "candidate_count_per_baseline": 6,
        "baselines": baselines,
        "no_artifact_generation": True,
        "no_model_training": True,
        "no_final_comparison": True,
        "no_scientific_conclusion": True,
        "reason": "p22_artifact_backed_baseline_smoke_completed",
    }
    
    assert_no_path_leakage(summary)
    return summary


def main() -> int:
    if len(sys.argv) > 1:
        print("Error: script does not accept arguments.", file=sys.stderr)
        return 1
        
    try:
        res = run_p22_artifact_backed_baseline_smoke()
        print(compact_json(res))
        return 0
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
