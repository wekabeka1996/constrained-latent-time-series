# tools/phase2/run_p33_fake_input_descriptor_smoke.py

import json
import sys

from src.phase2.fc_vae_fake_input_descriptor import (
    FC_VAE_FAKE_INPUT_DESCRIPTOR_CONTRACT_VERSION,
    run_fake_input_descriptor_probe,
    fake_input_descriptor_result_to_json_dict,
)


def run_p33_fake_input_descriptor_smoke() -> dict:
    result = run_fake_input_descriptor_probe()
    d = {
        "verdict": "PASS",
        "contract": FC_VAE_FAKE_INPUT_DESCRIPTOR_CONTRACT_VERSION,
        "source_phase": "P33",
        "descriptor_kind": "deterministic_fake_input_descriptor_no_values",
        "source_kind": "synthetic_descriptor_only",
        "distribution_kind": "bounded_uniform_descriptor",
        "dtype_intent": "float32_future_tensor_intent",
        "descriptor_seed": 1337,
        "min_value": -1.0,
        "max_value": 1.0,
        "batch_size": 2,
        "input_flat_dim": 32,
        "rank": 2,
        "shape_tuple": [result.shape.shape_tuple[0], result.shape.shape_tuple[1]],
        "shape_matches_p32": True,
        "torch_available": result.metadata.torch_available,
        "input_batch_status": result.metadata.input_batch_status,
        "status": result.status,
        "descriptor_available_in_p33": True,
        "rng_execution_attempted": False,
        "value_materialization_attempted": False,
        "array_materialization_attempted": False,
        "tensor_materialization_attempted": False,
        "forward_execution_attempted": False,
        "output_generation_attempted": False,
        "rng_executed": False,
        "values_materialized": False,
        "array_materialized": False,
        "tensor_materialized": False,
        "no_rng_execution": True,
        "no_values_materialized": True,
        "no_array_created": True,
        "no_tensor_created": True,
        "no_forward_execution": True,
        "no_output_generation": True,
        "no_training_loop": True,
        "no_optimizer": True,
        "no_checkpointing": True,
        "no_artifact_generation": True,
        "no_final_comparison": True,
        "no_scientific_conclusion": True,
        "result": fake_input_descriptor_result_to_json_dict(result),
        "reason": "p33_fake_input_descriptor_smoke_completed",
    }
    return d


def compact_json(data: dict) -> str:
    return json.dumps(data, sort_keys=True, separators=(",", ":"))


def main() -> int:
    if len(sys.argv) > 1:
        sys.stderr.write("Error: Smoke script does not accept arguments\n")
        return 1
    try:
        data = run_p33_fake_input_descriptor_smoke()
        sys.stdout.write(compact_json(data) + "\n")
        return 0
    except Exception as e:
        sys.stderr.write(f"Error: {e}\n")
        return 1


if __name__ == "__main__":
    sys.exit(main())
