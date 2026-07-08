# tools/phase4/run_p96_real_task_card_support_retrieval_pilot_smoke.py

import sys
import json
from src.phase4.real_task_card_support_retrieval_pilot import (
    run_p96_real_task_card_support_retrieval_pilot_probe,
)


def main():
    if len(sys.argv) > 1:
        print("Error: CLI arguments are rejected.", file=sys.stderr)
        sys.exit(1)

    try:
        res = run_p96_real_task_card_support_retrieval_pilot_probe()
        print(json.dumps(res, separators=(",", ":")))
        sys.exit(0 if res.get("verdict") == "P96_READY_FOR_REVIEW" else 1)
    except Exception as e:
        print(f"Execution Error: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
