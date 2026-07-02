# tools/phase3/run_p71_relation_contrastive_signal_smoke.py

import sys
import json

def main():
    if len(sys.argv) > 1:
        print("Error: run_p71_relation_contrastive_signal_smoke.py does not accept command line arguments.", file=sys.stderr)
        sys.exit(1)
        
    try:
        from src.phase3.relation_contrastive_signal_smoke import (
            run_p71_relation_contrastive_signal_smoke_probe
        )
    except ImportError as e:
        err_res = {
            "verdict": "FAIL",
            "source_phase": "P71",
            "error": str(e),
            "status": "blocked_by_import_failure"
        }
        print(json.dumps(err_res, sort_keys=True))
        sys.exit(1)
        
    probe_res = run_p71_relation_contrastive_signal_smoke_probe()
    
    compact_json = json.dumps(probe_res, sort_keys=True, separators=(",", ":"))
    print(compact_json)
    
    if probe_res.get("verdict") == "P71_READY_FOR_REVIEW":
        sys.exit(0)
    else:
        sys.exit(1)


if __name__ == "__main__":
    main()
