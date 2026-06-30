# tools/phase2/run_p54_fc_vae_architecture_forward_smoke.py

import sys
import json

# Crucial: No direct torch import at top level or anywhere in this file.

def main():
    if len(sys.argv) > 1:
        print("Error: run_p54_fc_vae_architecture_forward_smoke.py does not accept command line arguments.", file=sys.stderr)
        sys.exit(1)
        
    try:
        from src.phase2.fc_vae_architecture_forward import (
            run_fc_vae_architecture_forward_probe,
            compact_fc_vae_architecture_forward_json,
            fc_vae_architecture_forward_probe_to_json_dict
        )
    except ImportError as e:
        err_res = {
            "verdict": "FAIL",
            "source_phase": "P54",
            "error": str(e),
            "status": "blocked_by_import_failure"
        }
        print(json.dumps(err_res, sort_keys=True))
        sys.exit(1)
        
    probe_res = run_fc_vae_architecture_forward_probe()
    
    final_dict = fc_vae_architecture_forward_probe_to_json_dict(probe_res)
    
    passed = (probe_res.get("status") == "fc_vae_architecture_and_forward_pass_available_no_training")
    final_dict["verdict"] = "PASS" if passed else "FAIL"
    final_dict["source_phase"] = "P54"
    
    compact_json = json.dumps(final_dict, sort_keys=True, separators=(",", ":"))
    print(compact_json)
    
    if final_dict["verdict"] == "PASS":
        sys.exit(0)
    else:
        sys.exit(1)


if __name__ == "__main__":
    main()
