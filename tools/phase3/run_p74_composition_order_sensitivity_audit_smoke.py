# tools/phase3/run_p74_composition_order_sensitivity_audit_smoke.py

import sys
import json

def main():
    if len(sys.argv) > 1:
        print("Error: run_p74_composition_order_sensitivity_audit_smoke.py does not accept command line arguments.", file=sys.stderr)
        sys.exit(1)
        
    try:
        from src.phase3.composition_order_sensitivity_audit import (
            run_p74_composition_order_sensitivity_audit_probe
        )
    except ImportError as e:
        err_res = {
            "verdict": "FAIL",
            "source_phase": "P74",
            "error": str(e),
            "status": "blocked_by_import_failure"
        }
        print(json.dumps(err_res, sort_keys=True))
        sys.exit(1)
        
    probe_res = run_p74_composition_order_sensitivity_audit_probe()
    
    compact_json = json.dumps(probe_res, sort_keys=True, separators=(",", ":"))
    print(compact_json)
    
    if probe_res.get("verdict") == "P74_READY_FOR_REVIEW":
        sys.exit(0)
    else:
        sys.exit(1)


if __name__ == "__main__":
    main()
