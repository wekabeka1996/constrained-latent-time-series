# tools/phase2/run_p23_evidence_contract_smoke.py

import json
import sys

from src.phase2.evidence import (
    EVIDENCE_CONTRACT_VERSION,
    normalize_p22_smoke_summary_to_evidence_bundle,
    evidence_bundle_to_json_dict,
    assert_no_local_path_leakage,
    assert_no_raw_params,
    assert_no_forbidden_claims,
)
from tools.phase2.run_p22_artifact_backed_baseline_smoke import run_p22_artifact_backed_baseline_smoke


def run_p23_evidence_contract_smoke() -> dict:
    # 1. Run P22 artifact backed baseline smoke
    p22_summary = run_p22_artifact_backed_baseline_smoke()
    
    # 2. Normalize to evidence bundle
    bundle = normalize_p22_smoke_summary_to_evidence_bundle(p22_summary)
    
    # 3. Convert to json dict
    bundle_dict = evidence_bundle_to_json_dict(bundle)
    
    # 4. Additional safety checks on final dict
    assert_no_local_path_leakage(bundle_dict)
    assert_no_raw_params(bundle_dict)
    assert_no_forbidden_claims(bundle_dict)
    
    # 5. Build final smoke summary dict
    return {
        "verdict": "PASS",
        "contract": EVIDENCE_CONTRACT_VERSION,
        "source_phase": "P22",
        "evidence_bundle": bundle_dict,
        "no_artifact_generation": True,
        "no_model_training": True,
        "no_final_comparison": True,
        "no_scientific_conclusion": True,
        "reason": "p23_evidence_contract_smoke_completed"
    }


def compact_json(data: dict) -> str:
    return json.dumps(data, sort_keys=True, separators=(",", ":"))


def main() -> int:
    if len(sys.argv) > 1:
        print("Error: script does not accept arguments.", file=sys.stderr)
        return 1
        
    try:
        res = run_p23_evidence_contract_smoke()
        print(compact_json(res))
        return 0
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        print("VERDICT: P23_BLOCKED_BY_REQUIRED_P14_ARTIFACTS_MISSING_OR_INVALID", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
