# tools/phase4/run_p90_external_support_context_builder_smoke.py

import sys
import json
from src.phase4.external_support_context_builder import (
    run_p90_external_support_context_builder_probe
)

def main():
    if len(sys.argv) > 1:
        print("Error: CLI arguments are rejected.", file=sys.stderr)
        sys.exit(1)
        
    try:
        res = run_p90_external_support_context_builder_probe()
        compact_json = json.dumps(res, separators=(',', ':'))
        print(compact_json)
        
        if res.get("verdict") == "P90_READY_FOR_REVIEW":
            sys.exit(0)
        else:
            sys.exit(1)
    except Exception as e:
        print(f"Execution Error: {e}", file=sys.stderr)
        sys.exit(1)

if __name__ == '__main__':
    main()
