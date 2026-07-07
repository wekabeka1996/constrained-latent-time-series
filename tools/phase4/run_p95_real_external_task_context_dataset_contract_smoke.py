# tools/phase4/run_p95_real_external_task_context_dataset_contract_smoke.py

import sys
import json
from src.phase4.real_external_task_context_dataset_contract import (
    run_p95_real_external_task_context_dataset_contract_probe,
)


def main():
    if len(sys.argv) > 1:
        print("Error: CLI arguments are rejected.", file=sys.stderr)
        sys.exit(1)

    try:
        res = run_p95_real_external_task_context_dataset_contract_probe()
        print(json.dumps(res, separators=(",", ":")))
        sys.exit(0 if res.get("verdict") == "P95_READY_FOR_REVIEW" else 1)
    except Exception as e:
        print(f"Execution Error: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
