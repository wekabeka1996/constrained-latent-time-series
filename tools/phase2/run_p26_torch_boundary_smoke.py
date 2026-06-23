# tools/phase2/run_p26_torch_boundary_smoke.py

import json
import sys

from src.phase2.torch_boundary import (
    TORCH_BOUNDARY_CONTRACT_VERSION,
    build_torch_boundary_smoke_result,
    torch_boundary_smoke_result_to_json_dict,
)


def run_p26_torch_boundary_smoke() -> dict:
    result = build_torch_boundary_smoke_result()
    result_dict = torch_boundary_smoke_result_to_json_dict(result)

    return {
        "verdict": "PASS",
        "contract": TORCH_BOUNDARY_CONTRACT_VERSION,
        "source_phase": "P26",
        "torch_boundary": result_dict,
        "torch_available": result.torch_status.available,
        "torch_required_for_p26": False,
        "future_model_allowed_in_p26": False,
        "no_model_implementation": True,
        "no_training_loop": True,
        "no_optimizer": True,
        "no_checkpointing": True,
        "no_artifact_generation": True,
        "no_final_comparison": True,
        "no_scientific_conclusion": True,
        "reason": "p26_torch_boundary_smoke_completed",
    }


def compact_json(data: dict) -> str:
    return json.dumps(data, sort_keys=True, separators=(",", ":"))


def main() -> int:
    if len(sys.argv) > 1:
        print("Error: script does not accept arguments.", file=sys.stderr)
        return 1

    try:
        res = run_p26_torch_boundary_smoke()
        print(compact_json(res))
        return 0
    except Exception as e:
        print(f"Error executing smoke: {e}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
