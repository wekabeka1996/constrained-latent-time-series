# tools/phase4/run_p88_nonlearned_support_delta_metric_baseline_smoke.py

import sys
import json

def main():
    if len(sys.argv) > 1:
        sys.stderr.write("Error: run_p88_nonlearned_support_delta_metric_baseline_smoke.py does not accept arguments.\n")
        sys.exit(1)
        
    try:
        from src.phase4.nonlearned_support_delta_metric_baseline import (
            run_p88_nonlearned_support_delta_metric_baseline_probe,
        )
        res = run_p88_nonlearned_support_delta_metric_baseline_probe()
        print(json.dumps(res, separators=(',', ':')))
        
        if res.get("verdict") == "P88_READY_FOR_REVIEW":
            sys.exit(0)
        else:
            sys.exit(1)
    except Exception as e:
        sys.stderr.write(f"Exception raised: {str(e)}\n")
        sys.exit(1)

if __name__ == '__main__':
    main()
