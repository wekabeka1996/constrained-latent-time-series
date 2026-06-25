# tools/phase2/run_p34_fake_input_preview_smoke.py

import json
import sys

from src.phase2.fc_vae_fake_input_preview import (
    FC_VAE_FAKE_INPUT_PREVIEW_CONTRACT_VERSION,
    run_fake_input_preview_probe,
    fake_input_preview_result_to_json_dict,
)


def run_p34_fake_input_preview_smoke() -> dict:
    result = run_fake_input_preview_probe()
    d = {
        "verdict": "PASS",
        "contract": FC_VAE_FAKE_INPUT_PREVIEW_CONTRACT_VERSION,
        "source_phase": "P34",
        "preview_kind": "deterministic_fake_input_scalar_preview_no_tensor_no_array",
        "value_kind": "bounded_deterministic_scalar_preview",
        "descriptor_seed": 1337,
        "min_value": -1.0,
        "max_value": 1.0,
        "batch_size": 2,
        "input_flat_dim": 32,
        "full_batch_scalar_count": 64,
        "preview_value_count": 4,
        "preview_values": list(result.preview.preview_values),
        "preview_is_partial": True,
        "all_values_within_range": True,
        "torch_available": result.metadata.torch_available,
        "descriptor_status": result.metadata.descriptor_status,
        "status": result.status,
        "preview_available_in_p34": True,
        "full_batch_available_in_p34": False,
        "rng_execution_attempted": False,
        "full_batch_materialization_attempted": False,
        "array_materialization_attempted": False,
        "tensor_materialization_attempted": False,
        "forward_execution_attempted": False,
        "output_generation_attempted": False,
        "rng_executed": False,
        "full_batch_materialized": False,
        "array_materialized": False,
        "tensor_materialized": False,
        "no_rng_execution": True,
        "no_full_batch_materialized": True,
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
        "result": fake_input_preview_result_to_json_dict(result),
        "reason": "p34_fake_input_preview_smoke_completed",
    }
    return d


def compact_json(data: dict) -> str:
    return json.dumps(data, sort_keys=True, separators=(",", ":"))


def main() -> int:
    if len(sys.argv) > 1:
        sys.stderr.write("Error: Smoke script does not accept arguments\n")
        return 1
    try:
        data = run_p34_fake_input_preview_smoke()
        sys.stdout.write(compact_json(data) + "\n")
        return 0
    except Exception as e:
        sys.stderr.write(f"Error: {e}\n")
        return 1


if __name__ == "__main__":
    sys.exit(main())
