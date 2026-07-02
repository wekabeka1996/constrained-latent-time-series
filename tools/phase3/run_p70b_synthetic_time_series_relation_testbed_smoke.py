# tools/phase3/run_p70b_synthetic_time_series_relation_testbed_smoke.py

import sys
import json

def main():
    if len(sys.argv) > 1:
        print("Error: run_p70b_synthetic_time_series_relation_testbed_smoke.py does not accept command line arguments.", file=sys.stderr)
        sys.exit(1)
        
    try:
        from src.phase3.synthetic_time_series_relation_testbed import (
            run_p70b_synthetic_time_series_relation_testbed_probe
        )
    except ImportError as e:
        err_res = {
            "verdict": "FAIL",
            "source_phase": "P70B",
            "error": str(e),
            "status": "blocked_by_import_failure"
        }
        print(json.dumps(err_res, sort_keys=True))
        sys.exit(1)
        
    probe_res = run_p70b_synthetic_time_series_relation_testbed_probe()
    
    compact_json = json.dumps(probe_res, sort_keys=True, separators=(",", ":"))
    print(compact_json)
    
    if probe_res.get("verdict") == "P70B_READY_FOR_REVIEW":
        sys.exit(0)
    else:
        sys.exit(1)


if __name__ == "__main__":
    main()
