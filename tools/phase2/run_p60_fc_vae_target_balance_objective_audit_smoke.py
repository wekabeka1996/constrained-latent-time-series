# tools/phase2/run_p60_fc_vae_target_balance_objective_audit_smoke.py

import sys
import json

# Crucial: No direct torch import at top level or anywhere in this file.

def main():
    if len(sys.argv) > 1:
        print("Error: run_p60_fc_vae_target_balance_objective_audit_smoke.py does not accept command line arguments.", file=sys.stderr)
        sys.exit(1)
        
    try:
        from src.phase2.fc_vae_target_balance_objective_audit import (
            run_fc_vae_target_balance_objective_audit_probe,
            fc_vae_target_balance_objective_audit_probe_to_json_dict
        )
    except ImportError as e:
        err_res = {
            "verdict": "FAIL",
            "source_phase": "P60",
            "source_evidence_phase": "P59",
            "error": str(e),
            "status": "blocked_by_import_failure"
        }
        print(json.dumps(err_res, sort_keys=True))
        sys.exit(1)
        
    probe_res = run_fc_vae_target_balance_objective_audit_probe()
    
    final_dict = fc_vae_target_balance_objective_audit_probe_to_json_dict(probe_res)
    
    final_dict["verdict"] = "PASS" if probe_res.get("verdict") == "PASS" else "FAIL"
    final_dict["source_phase"] = "P60"
    final_dict["source_evidence_phase"] = "P59"
    
    compact_json = json.dumps(final_dict, sort_keys=True, separators=(",", ":"))
    print(compact_json)
    
    if final_dict["verdict"] == "PASS":
        sys.exit(0)
    else:
        sys.exit(1)


if __name__ == "__main__":
    main()
