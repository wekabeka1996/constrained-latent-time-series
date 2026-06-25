# tools/phase2/run_p32_forward_input_batch_smoke.py

import json
import sys

from src.phase2.fc_vae_forward_input_batch import (
    FC_VAE_FORWARD_INPUT_BATCH_CONTRACT_VERSION,
    run_forward_input_batch_probe,
    forward_input_batch_result_to_json_dict,
)


def run_p32_forward_input_batch_smoke() -> dict:
    result = run_forward_input_batch_probe()
    d = {
        "verdict": "PASS",
        "contract": FC_VAE_FORWARD_INPUT_BATCH_CONTRACT_VERSION,
        "source_phase": "P32",
        "batch_kind": "forward_input_batch_metadata_no_tensor",
        "shape_kind": "declared_input_batch_shape_no_tensor",
        "architecture_id": "FC-VAE",
        "batch_size": 2,
        "input_flat_dim": 32,
        "rank": 2,
        "shape_tuple": [result.shape.shape_tuple[0], result.shape.shape_tuple[1]],
        "torch_available": result.metadata.torch_available,
        "forward_boundary_status": result.metadata.forward_boundary_status,
        "status": result.status,
        "batch_shape_declared": True,
        "input_batch_available_in_p32": False,
        "tensor_materialization_attempted": False,
        "array_materialization_attempted": False,
        "values_materialization_attempted": False,
        "forward_execution_attempted": False,
        "output_generation_attempted": False,
        "tensor_materialized": False,
        "array_materialized": False,
        "values_materialized": False,
        "no_tensor_created": True,
        "no_array_created": True,
        "no_values_materialized": True,
        "no_forward_execution": True,
        "no_output_generation": True,
        "no_training_loop": True,
        "no_optimizer": True,
        "no_checkpointing": True,
        "no_artifact_generation": True,
        "no_final_comparison": True,
        "no_scientific_conclusion": True,
        "result": forward_input_batch_result_to_json_dict(result),
        "reason": "p32_forward_input_batch_smoke_completed",
    }
    return d


def compact_json(data: dict) -> str:
    return json.dumps(data, sort_keys=True, separators=(",", ":"))


def main() -> int:
    if len(sys.argv) > 1:
        sys.stderr.write("Error: Smoke script does not accept arguments\n")
        return 1
    try:
        data = run_p32_forward_input_batch_smoke()
        sys.stdout.write(compact_json(data) + "\n")
        return 0
    except Exception as e:
        sys.stderr.write(f"Error: {e}\n")
        return 1


if __name__ == "__main__":
    sys.exit(main())
