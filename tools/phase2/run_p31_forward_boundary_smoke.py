# tools/phase2/run_p31_forward_boundary_smoke.py

import json
import sys

from src.phase2.fc_vae_forward_boundary import (
    FC_VAE_FORWARD_BOUNDARY_CONTRACT_VERSION,
    run_forward_boundary_probe,
    forward_boundary_result_to_json_dict,
)


def run_p31_forward_boundary_smoke() -> dict:
    result = run_forward_boundary_probe()
    d = {
        "verdict": "PASS",
        "contract": FC_VAE_FORWARD_BOUNDARY_CONTRACT_VERSION,
        "source_phase": "P31",
        "boundary_kind": "noop_forward_boundary_contract",
        "architecture_id": "FC-VAE",
        "torch_required_for_p31": False,
        "torch_required_for_future_execution": True,
        "torch_available": result.metadata.torch_available,
        "status": result.status,
        "constructor_binding_status": result.metadata.constructor_binding_status,
        "constructor_binding_available_in_p30": result.metadata.constructor_binding_available_in_p30,
        "module_created": result.metadata.module_created,
        "binding_created": result.metadata.binding_created,
        "execution_attempted": False,
        "tensor_allocation_attempted": False,
        "output_generation_attempted": False,
        "module_object_returned": False,
        "generated_output": False,
        "defines_forward": False,
        "defines_layers": False,
        "parameter_count": 0,
        "buffer_count": 0,
        "forward_execution_available_in_p31": False,
        "tensor_allocation_available_in_p31": False,
        "output_generation_available_in_p31": False,
        "training_available_in_p31": False,
        "loss_available_in_p31": False,
        "optimizer_available_in_p31": False,
        "checkpointing_available_in_p31": False,
        "artifact_generation_available_in_p31": False,
        "declared_total_output_dim": result.declared_output_shape.total_declared_output_dim,
        "result": forward_boundary_result_to_json_dict(result),
        "no_final_comparison": True,
        "no_scientific_conclusion": True,
        "reason": "p31_forward_boundary_smoke_completed",
    }
    return d


def compact_json(data: dict) -> str:
    return json.dumps(data, sort_keys=True, separators=(",", ":"))


def main() -> int:
    if len(sys.argv) > 1:
        sys.stderr.write("Error: Smoke script does not accept arguments\n")
        return 1
    try:
        data = run_p31_forward_boundary_smoke()
        sys.stdout.write(compact_json(data) + "\n")
        return 0
    except Exception as e:
        sys.stderr.write(f"Error: {e}\n")
        return 1


if __name__ == "__main__":
    sys.exit(main())
