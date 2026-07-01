# tools/phase2/run_p66_fc_vae_target_identity_numeric_separation_audit_smoke.py

import sys
import json

def main():
    if len(sys.argv) > 1:
        print("Error: run_p66_fc_vae_target_identity_numeric_separation_audit_smoke.py does not accept command line arguments.", file=sys.stderr)
        sys.exit(1)
        
    try:
        from src.phase2.fc_vae_target_identity_numeric_separation_audit import (
            run_p66_target_identity_numeric_separation_audit_probe,
            fc_vae_target_identity_numeric_separation_audit_probe_to_json_dict
        )
    except ImportError as e:
        err_res = {
            "verdict": "FAIL",
            "source_phase": "P66",
            "error": str(e),
            "status": "blocked_by_import_failure"
        }
        print(json.dumps(err_res, sort_keys=True))
        sys.exit(1)
        
    probe_res = run_p66_target_identity_numeric_separation_audit_probe()
    
    final_dict = fc_vae_target_identity_numeric_separation_audit_probe_to_json_dict(probe_res)
    
    final_dict["verdict"] = "PASS" if probe_res.get("verdict") == "PASS" else "FAIL"
    final_dict["source_phase"] = "P66"
    
    compact_json = json.dumps(final_dict, sort_keys=True, separators=(",", ":"))
    print(compact_json)
    
    if final_dict["verdict"] == "PASS":
        sys.exit(0)
    else:
        sys.exit(1)


if __name__ == "__main__":
    main()
