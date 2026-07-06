# tools/phase4/run_p91_external_context_metric_evaluation_smoke.py

import sys
import json
from src.phase4.external_context_metric_evaluation import (
    run_p91_external_context_metric_evaluation_probe,
)


def main():
    if len(sys.argv) > 1:
        print("Error: CLI arguments are rejected.", file=sys.stderr)
        sys.exit(1)

    try:
        res = run_p91_external_context_metric_evaluation_probe()
        print(json.dumps(res, separators=(",", ":")))
        sys.exit(0 if res.get("verdict") == "P91_READY_FOR_REVIEW" else 1)
    except Exception as e:
        print(f"Execution Error: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
