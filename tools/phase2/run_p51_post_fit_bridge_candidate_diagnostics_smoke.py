# tools/phase2/run_p51_post_fit_bridge_candidate_diagnostics_smoke.py

import sys
import json

# Crucial: No direct torch import at top level or anywhere in this file.
# We import from src.phase2.post_fit_bridge_candidate_diagnostics which loads torch dynamically.

def main():
    # Reject command line arguments
    if len(sys.argv) > 1:
        print("Error: run_p51_post_fit_bridge_candidate_diagnostics_smoke.py does not accept command line arguments.", file=sys.stderr)
        sys.exit(1)
        
    try:
        from src.phase2.post_fit_bridge_candidate_diagnostics import (
            run_post_fit_bridge_candidate_diagnostics_probe,
            compact_post_fit_bridge_candidate_diagnostics_json,
            post_fit_bridge_candidate_diagnostics_probe_to_json_dict
        )
    except ImportError as e:
        err_res = {
            "verdict": "FAIL",
            "source_phase": "P51",
            "error": str(e),
            "status": "blocked_by_import_failure"
        }
        print(json.dumps(err_res, sort_keys=True))
        sys.exit(1)
        
    # Run the probe
    probe_res = run_post_fit_bridge_candidate_diagnostics_probe()
    
    # Structure the result as specified
    final_dict = post_fit_bridge_candidate_diagnostics_probe_to_json_dict(probe_res)
    
    # Append smoke fields
    final_dict["verdict"] = "PASS" if probe_res.get("status") == "post_fit_bridge_candidate_diagnostics_available_no_model_no_vae_no_gsb_no_science" else "FAIL"
    final_dict["source_phase"] = "P51"
    
    # Print compact sorted JSON to stdout
    compact_json = json.dumps(final_dict, sort_keys=True, separators=(",", ":"))
    print(compact_json)
    
    if final_dict["verdict"] == "PASS":
        sys.exit(0)
    else:
        sys.exit(1)


if __name__ == "__main__":
    main()
