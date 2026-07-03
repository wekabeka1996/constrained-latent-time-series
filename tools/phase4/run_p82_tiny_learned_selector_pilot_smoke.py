# tools/phase4/run_p82_tiny_learned_selector_pilot_smoke.py

import sys
import json

def main():
    if len(sys.argv) > 1:
        sys.stderr.write("Error: run_p82_tiny_learned_selector_pilot_smoke.py does not accept arguments.\n")
        sys.exit(1)
        
    try:
        from src.phase4.tiny_learned_selector_pilot import (
            run_p82_tiny_learned_selector_pilot_probe,
        )
        res = run_p82_tiny_learned_selector_pilot_probe()
        print(json.dumps(res, separators=(',', ':')))
        
        if res.get("verdict") == "P82_READY_FOR_REVIEW":
            sys.exit(0)
        else:
            sys.exit(1)
    except Exception as e:
        sys.stderr.write(f"Exception raised: {str(e)}\n")
        sys.exit(1)

if __name__ == '__main__':
    main()
