# tools/phase2/run_p36_flat_vector_batch_view_smoke.py

import json
import sys

from src.phase2.fc_vae_flat_vector_batch_view import (
    FC_VAE_FLAT_VECTOR_BATCH_VIEW_CONTRACT_VERSION,
    run_flat_vector_batch_view_probe,
    flat_vector_batch_view_result_to_json_dict,
    compute_row_major_flat_index,
)


def run_p36_flat_vector_batch_view_smoke() -> dict:
    result = run_flat_vector_batch_view_probe()
    d = {
        "verdict": "PASS",
        "contract": FC_VAE_FLAT_VECTOR_BATCH_VIEW_CONTRACT_VERSION,
        "source_phase": "P36",
        "view_kind": "flat_vector_2d_batch_view_metadata_no_nested_values",
        "view_order_kind": "row_major_flat_index_view",
        "view_rank": result.shape.view_rank,
        "shape_tuple": list(result.shape.shape_tuple),
        "batch_size": result.shape.batch_size,
        "input_flat_dim": result.shape.input_flat_dim,
        "full_vector_length": result.shape.full_vector_length,
        "row_major_formula": result.shape.row_major_formula,
        "example_flat_index_0_0": compute_row_major_flat_index(0, 0, result.shape.input_flat_dim),
        "example_flat_index_0_31": compute_row_major_flat_index(0, 31, result.shape.input_flat_dim),
        "example_flat_index_1_0": compute_row_major_flat_index(1, 0, result.shape.input_flat_dim),
        "example_flat_index_1_31": compute_row_major_flat_index(1, 31, result.shape.input_flat_dim),
        "flat_vector_status": result.metadata.flat_vector_status,
        "torch_available": result.metadata.torch_available,
        "status": result.status,
        "batch_view_available_in_p36": result.batch_view_available_in_p36,
        "nested_values_available_in_p36": result.nested_values_available_in_p36,
        "two_d_batch_available_in_p36": result.two_d_batch_available_in_p36,
        "array_materialization_available_in_p36": result.array_materialization_available_in_p36,
        "tensor_materialization_available_in_p36": result.tensor_materialization_available_in_p36,
        "forward_execution_available_in_p36": result.forward_execution_available_in_p36,
        "output_generation_available_in_p36": result.output_generation_available_in_p36,
        "flat_vector_reused_without_copy": result.metadata.flat_vector_reused_without_copy,
        "nested_values_attempted": result.metadata.nested_values_attempted,
        "two_d_batch_materialization_attempted": result.metadata.two_d_batch_materialization_attempted,
        "array_materialization_attempted": result.metadata.array_materialization_attempted,
        "tensor_materialization_attempted": result.metadata.tensor_materialization_attempted,
        "forward_execution_attempted": result.metadata.forward_execution_attempted,
        "output_generation_attempted": result.metadata.output_generation_attempted,
        "no_nested_values": result.no_nested_values,
        "no_2d_batch_materialized": result.no_2d_batch_materialized,
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
        "result": flat_vector_batch_view_result_to_json_dict(result),
        "reason": "p36_flat_vector_batch_view_smoke_completed",
    }
    return d


def compact_json(data: dict) -> str:
    return json.dumps(data, sort_keys=True, separators=(",", ":"))


def main() -> int:
    if len(sys.argv) > 1:
        sys.stderr.write("Error: Smoke script does not accept arguments\n")
        return 1
    try:
        data = run_p36_flat_vector_batch_view_smoke()
        sys.stdout.write(compact_json(data) + "\n")
        return 0
    except Exception as e:
        sys.stderr.write(f"Error: {e}\n")
        return 1


if __name__ == "__main__":
    sys.exit(main())
