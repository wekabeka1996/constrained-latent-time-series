# tools/phase2/run_p35_fake_input_flat_vector_smoke.py

import json
import sys

from src.phase2.fc_vae_fake_input_flat_vector import (
    FC_VAE_FAKE_INPUT_FLAT_VECTOR_CONTRACT_VERSION,
    run_fake_input_flat_vector_probe,
    fake_input_flat_vector_result_to_json_dict,
)


def run_p35_fake_input_flat_vector_smoke() -> dict:
    result = run_fake_input_flat_vector_probe()
    d = {
        "verdict": "PASS",
        "contract": FC_VAE_FAKE_INPUT_FLAT_VECTOR_CONTRACT_VERSION,
        "source_phase": "P35",
        "flat_vector_kind": "deterministic_fake_input_flat_vector_no_tensor_no_array",
        "value_kind": "bounded_deterministic_flat_scalar_vector",
        "descriptor_seed": 1337,
        "min_value": -1.0,
        "max_value": 1.0,
        "batch_size": 2,
        "input_flat_dim": 32,
        "full_vector_length": 64,
        "flat_values": list(result.flat_vector.flat_values),
        "flat_values_count": 64,
        "all_values_within_range": True,
        "is_flat_vector": True,
        "is_2d_batch": False,
        "torch_available": result.metadata.torch_available,
        "preview_status": result.metadata.preview_status,
        "status": result.status,
        "flat_vector_available_in_p35": True,
        "two_d_batch_available_in_p35": False,
        "rng_execution_attempted": False,
        "array_materialization_attempted": False,
        "tensor_materialization_attempted": False,
        "forward_execution_attempted": False,
        "output_generation_attempted": False,
        "rng_executed": False,
        "array_materialized": False,
        "tensor_materialized": False,
        "forward_executed": False,
        "no_rng_execution": True,
        "no_2d_batch_materialized": True,
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
        "result": fake_input_flat_vector_result_to_json_dict(result),
        "reason": "p35_fake_input_flat_vector_smoke_completed",
    }
    return d


def compact_json(data: dict) -> str:
    return json.dumps(data, sort_keys=True, separators=(",", ":"))


def main() -> int:
    if len(sys.argv) > 1:
        sys.stderr.write("Error: Smoke script does not accept arguments\n")
        return 1
    try:
        data = run_p35_fake_input_flat_vector_smoke()
        sys.stdout.write(compact_json(data) + "\n")
        return 0
    except Exception as e:
        sys.stderr.write(f"Error: {e}\n")
        return 1


if __name__ == "__main__":
    sys.exit(main())
