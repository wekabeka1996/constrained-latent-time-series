# tools/phase4/run_p94_external_task_context_enrichment_smoke.py

import sys
import json
from src.phase4.external_task_context_enrichment import (
    run_p94_external_task_context_enrichment_probe,
)


def main():
    if len(sys.argv) > 1:
        print("Error: CLI arguments are rejected.", file=sys.stderr)
        sys.exit(1)

    try:
        res = run_p94_external_task_context_enrichment_probe()
        print(json.dumps(res, separators=(",", ":")))
        sys.exit(0 if res.get("verdict") == "P94_READY_FOR_REVIEW" else 1)
    except Exception as e:
        print(f"Execution Error: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
