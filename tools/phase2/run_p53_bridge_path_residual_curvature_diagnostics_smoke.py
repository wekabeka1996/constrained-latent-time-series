# tools/phase2/run_p53_bridge_path_residual_curvature_diagnostics_smoke.py

import sys
import json

# Crucial: No direct torch import at top level or anywhere in this file.

def main():
    if len(sys.argv) > 1:
        print("Error: run_p53_bridge_path_residual_curvature_diagnostics_smoke.py does not accept command line arguments.", file=sys.stderr)
        sys.exit(1)
        
    try:
        from src.phase2.bridge_path_residual_curvature_diagnostics import (
            run_bridge_path_residual_curvature_diagnostics_probe,
            compact_bridge_path_residual_curvature_diagnostics_json,
            bridge_path_residual_curvature_diagnostics_probe_to_json_dict
        )
    except ImportError as e:
        err_res = {
            "verdict": "FAIL",
            "source_phase": "P53",
            "error": str(e),
            "status": "blocked_by_import_failure"
        }
        print(json.dumps(err_res, sort_keys=True))
        sys.exit(1)
        
    probe_res = run_bridge_path_residual_curvature_diagnostics_probe()
    
    final_dict = bridge_path_residual_curvature_diagnostics_probe_to_json_dict(probe_res)
    
    passed = bool(probe_res.get("path_shape_diagnostics_passed", False))
    final_dict["verdict"] = "PASS" if passed else "FAIL"
    final_dict["source_phase"] = "P53"
    
    compact_json = json.dumps(final_dict, sort_keys=True, separators=(",", ":"))
    print(compact_json)
    
    if final_dict["verdict"] == "PASS":
        sys.exit(0)
    else:
        sys.exit(1)


if __name__ == "__main__":
    main()
