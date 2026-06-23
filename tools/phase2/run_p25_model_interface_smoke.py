# tools/phase2/run_p25_model_interface_smoke.py

import json
import sys

from src.phase2.schema import FamilyId, MeanFamily, VolatilityFamily, ModelSpec
from src.phase2.model_interface import (
    MODEL_INTERFACE_CONTRACT_VERSION,
    MODEL_CANDIDATE_SUMMARY_CONTRACT_VERSION,
    FC_VAE_ARCHITECTURE_ID,
    MODEL_RUN_KIND_ZERO_SHOT,
    ModelRunMetadata,
    ModelCandidateRecord,
    ModelCandidateBatch,
    validate_model_candidate_batch,
    summarize_model_candidate_batch,
    model_candidate_summary_to_json_dict,
)


def build_valid_arma_spec_for_p25_smoke() -> ModelSpec:
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


def build_valid_garch_spec_for_p25_smoke() -> ModelSpec:
    return ModelSpec(
        family_id=FamilyId.GARCH,
        mean_family=MeanFamily.NONE,
        volatility_family=VolatilityFamily.GARCH,
        p=0,
        q=0,
        r=1,
        s=1,
        ar_params=(),
        ma_params=(),
        alpha_params=(0.1,),
        beta_params=(0.8,),
        omega=0.1,
        constraint_flags=(1.0, 1.0, 0.0, 0.0),
        provenance=(("generator", "test"),)
    )


def build_p25_smoke_metadata() -> ModelRunMetadata:
    return ModelRunMetadata(
        contract_version=MODEL_INTERFACE_CONTRACT_VERSION,
        model_architecture_id=FC_VAE_ARCHITECTURE_ID,
        model_run_kind=MODEL_RUN_KIND_ZERO_SHOT,
        model_code_git_commit="p25_smoke_no_model_code",
        training_artifact_root="phase2_artifacts/p14_smoke_dry_run",
        training_split_contract="p14_smoke_zero_shot_train",
        model_repeat_seed=25001,
        zero_shot_mode=True,
        C_train_count=0,
        config_hash="p25_smoke_no_config",
        evidence_contract_version="phase2_p23_evidence_contract_v1",
        reason="p25_model_interface_smoke_metadata"
    )


def build_p25_smoke_batch() -> ModelCandidateBatch:
    meta = build_p25_smoke_metadata()
    arma_spec = build_valid_arma_spec_for_p25_smoke()
    garch_spec = build_valid_garch_spec_for_p25_smoke()

    c1 = ModelCandidateRecord(
        candidate_id="cand_arma_001",
        metadata=meta,
        generated_spec=arma_spec,
        validity_pass=True,
        math_validity_pass=True,
        generated_family_id="ARMA",
        generated_mean_family="ARMA",
        generated_volatility_family="NONE",
        source_split="train",
        reason="ARMA smoke candidate"
    )

    c2 = ModelCandidateRecord(
        candidate_id="cand_garch_002",
        metadata=meta,
        generated_spec=garch_spec,
        validity_pass=True,
        math_validity_pass=True,
        generated_family_id="GARCH",
        generated_mean_family="NONE",
        generated_volatility_family="GARCH",
        source_split="train",
        reason="GARCH smoke candidate"
    )

    return ModelCandidateBatch(
        contract_version=MODEL_INTERFACE_CONTRACT_VERSION,
        metadata=meta,
        candidates=(c1, c2),
        candidate_count=2,
        reason="P25 smoke batch completed"
    )


def run_p25_model_interface_smoke() -> dict:
    # 1. Build and validate batch
    batch = build_p25_smoke_batch()
    validate_model_candidate_batch(batch)

    # 2. Summarize
    summary = summarize_model_candidate_batch(batch)

    # 3. Convert to json dict
    sum_dict = model_candidate_summary_to_json_dict(summary)

    # 4. Build output dict
    return {
        "verdict": "PASS",
        "contract": MODEL_INTERFACE_CONTRACT_VERSION,
        "summary_contract": MODEL_CANDIDATE_SUMMARY_CONTRACT_VERSION,
        "model_architecture_id": FC_VAE_ARCHITECTURE_ID,
        "source_phase": "P25",
        "candidate_count": 2,
        "summary": sum_dict,
        "no_artifact_generation": True,
        "no_model_training": True,
        "no_model_implementation": True,
        "no_final_comparison": True,
        "no_scientific_conclusion": True,
        "reason": "p25_model_interface_smoke_completed"
    }


def compact_json(data: dict) -> str:
    return json.dumps(data, sort_keys=True, separators=(",", ":"))


def main() -> int:
    if len(sys.argv) > 1:
        print("Error: script does not accept arguments.", file=sys.stderr)
        return 1

    try:
        res = run_p25_model_interface_smoke()
        print(compact_json(res))
        return 0
    except Exception as e:
        print(f"Error executing smoke: {e}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
