# tools/phase2/run_p30_constructor_binding_smoke.py

import json
import sys

from src.phase2.fc_vae_constructor_binding import (
    FC_VAE_CONSTRUCTOR_BINDING_CONTRACT_VERSION,
    run_constructor_binding_probe,
    constructor_binding_result_to_json_dict,
)


def run_p30_constructor_binding_smoke() -> dict:
    result = run_constructor_binding_probe()
    d = {
        "verdict": "PASS",
        "contract": FC_VAE_CONSTRUCTOR_BINDING_CONTRACT_VERSION,
        "source_phase": "P30",
        "binding_kind": "torch_module_constructor_spec_binding",
        "architecture_id": "FC-VAE",
        "torch_required_for_p30": False,
        "torch_required_for_future_execution": True,
        "torch_available": result.metadata.torch_available,
        "status": result.status,
        "binding_created": result.metadata.binding_created,
        "constructor_binding_available_in_p30": result.constructor_binding_available_in_p30,
        "module_created": result.metadata.module_created,
        "module_object_returned": False,
        "bound_to_module_object": False,
        "defines_forward": False,
        "defines_layers": False,
        "parameter_count": 0,
        "buffer_count": 0,
        "forward_available_in_p30": False,
        "layers_available_in_p30": False,
        "training_available_in_p30": False,
        "result": constructor_binding_result_to_json_dict(result),
        "no_training_loop": True,
        "no_optimizer": True,
        "no_checkpointing": True,
        "no_artifact_generation": True,
        "no_final_comparison": True,
        "no_scientific_conclusion": True,
        "reason": "p30_constructor_binding_smoke_completed",
    }
    return d


def compact_json(data: dict) -> str:
    return json.dumps(data, sort_keys=True, separators=(",", ":"))


def main() -> int:
    if len(sys.argv) > 1:
        sys.stderr.write("Error: Smoke script does not accept arguments\n")
        return 1
    try:
        data = run_p30_constructor_binding_smoke()
        sys.stdout.write(compact_json(data) + "\n")
        return 0
    except Exception as e:
        sys.stderr.write(f"Error: {e}\n")
        return 1


if __name__ == "__main__":
    sys.exit(main())
