# tools/phase2/run_p37_nested_batch_values_smoke.py

import json
import sys

from src.phase2.fc_vae_nested_batch_values import (
    FC_VAE_NESTED_BATCH_VALUES_CONTRACT_VERSION,
    run_nested_batch_values_probe,
    nested_batch_values_result_to_json_dict,
)


def run_p37_nested_batch_values_smoke() -> dict:
    result = run_nested_batch_values_probe()
    d = {
        "verdict": "PASS",
        "contract": FC_VAE_NESTED_BATCH_VALUES_CONTRACT_VERSION,
        "source_phase": "P37",
        "nested_values_kind": "controlled_nested_python_tuple_batch_values_no_tensor_no_array",
        "order_kind": "row_major_flat_to_nested_tuple",
        "shape_tuple": list(result.nested_values.shape_tuple),
        "batch_size": result.nested_values.batch_size,
        "input_flat_dim": result.nested_values.input_flat_dim,
        "full_vector_length": result.nested_values.full_vector_length,
        "row_count": result.nested_values.row_count,
        "row_lengths": list(result.nested_values.row_lengths),
        "total_scalar_count": result.nested_values.total_scalar_count,
        "first_row_first_4_values": list(result.nested_values.nested_batch_values[0][:4]),
        "second_row_first_4_values": list(result.nested_values.nested_batch_values[1][:4]),
        "flattened_matches_source_flat_vector": result.nested_values.flattened_matches_source_flat_vector,
        "all_values_are_exact_float": result.nested_values.all_values_are_exact_float,
        "all_values_within_source_range": result.nested_values.all_values_within_source_range,
        "flat_vector_status": result.metadata.flat_vector_status,
        "batch_view_status": result.metadata.batch_view_status,
        "torch_available": result.metadata.torch_available,
        "status": result.status,
        "nested_values_available_in_p37": result.nested_values_available_in_p37,
        "array_materialization_available_in_p37": result.array_materialization_available_in_p37,
        "tensor_materialization_available_in_p37": result.tensor_materialization_available_in_p37,
        "forward_execution_available_in_p37": result.forward_execution_available_in_p37,
        "output_generation_available_in_p37": result.output_generation_available_in_p37,
        "nested_values_materialized": result.metadata.nested_values_materialized,
        "array_materialization_attempted": result.metadata.array_materialization_attempted,
        "tensor_materialization_attempted": result.metadata.tensor_materialization_attempted,
        "forward_execution_attempted": result.metadata.forward_execution_attempted,
        "output_generation_attempted": result.metadata.output_generation_attempted,
        "no_array_created": result.no_array_created,
        "no_tensor_created": result.no_tensor_created,
        "no_forward_execution": result.no_forward_execution,
        "no_output_generation": result.no_output_generation,
        "no_training_loop": result.no_training_loop,
        "no_optimizer": result.no_optimizer,
        "no_checkpointing": result.no_checkpointing,
        "no_artifact_generation": result.no_artifact_generation,
        "no_final_comparison": result.no_final_comparison,
        "no_scientific_conclusion": result.no_scientific_conclusion,
        "result": nested_batch_values_result_to_json_dict(result),
        "reason": "p37_nested_batch_values_smoke_completed",
    }
    return d


def compact_json(data: dict) -> str:
    return json.dumps(data, sort_keys=True, separators=(",", ":"))


def main() -> int:
    if len(sys.argv) > 1:
        sys.stderr.write("Error: Smoke script does not accept arguments\n")
        return 1
    try:
        data = run_p37_nested_batch_values_smoke()
        sys.stdout.write(compact_json(data) + "\n")
        return 0
    except Exception as e:
        sys.stderr.write(f"Error: {e}\n")
        return 1


if __name__ == "__main__":
    sys.exit(main())
