# tools/phase2/run_p38_tensor_materialization_request_smoke.py

import json
import sys
from src.phase2.fc_vae_tensor_materialization_request import (
    run_tensor_materialization_request_probe,
    tensor_materialization_request_result_to_json_dict,
)

def compact_json(data: dict) -> str:
    return json.dumps(data, sort_keys=True, separators=(",", ":"))

def run_p38_tensor_materialization_request_smoke() -> dict:
    result = run_tensor_materialization_request_probe()
    
    out = {
        "verdict": "PASS",
        "contract": result.contract_version,
        "source_phase": "P38",
        "request_kind": result.request.request_kind,
        "target_framework": result.request.target_framework,
        "target_dtype_intent": result.request.target_dtype_intent,
        "target_device_intent": result.request.target_device_intent,
        "target_layout_kind": result.request.target_layout_kind,
        "shape_tuple": list(result.request.shape_tuple),
        "batch_size": result.request.batch_size,
        "input_flat_dim": result.request.input_flat_dim,
        "full_vector_length": result.request.full_vector_length,
        "source_nested_row_count": result.spec.source_nested_row_count,
        "source_nested_row_lengths": list(result.spec.source_nested_row_lengths),
        "source_total_scalar_count": result.spec.source_total_scalar_count,
        "source_flattened_matches_p35": result.spec.source_flattened_matches_p35,
        "requires_grad": result.request.requires_grad,
        "torch_backend_name": result.metadata.torch_backend_name,
        "torch_policy": result.metadata.torch_policy,
        "torch_available": result.metadata.torch_available,
        "torch_import_safe": result.metadata.torch_import_safe,
        "top_level_torch_import_required": result.metadata.top_level_torch_import_required,
        "nested_values_status": result.metadata.nested_values_status,
        "nested_values_available_in_p37": result.metadata.nested_values_available_in_p37,
        "nested_values_materialized_in_p37": result.metadata.nested_values_materialized_in_p37,
        "status": result.status,
        "tensor_request_available_in_p38": result.tensor_request_available_in_p38,
        "tensor_materialization_available_in_p38": result.tensor_materialization_available_in_p38,
        "array_materialization_available_in_p38": result.array_materialization_available_in_p38,
        "forward_execution_available_in_p38": result.forward_execution_available_in_p38,
        "output_generation_available_in_p38": result.output_generation_available_in_p38,
        "training_available_in_p38": result.training_available_in_p38,
        "tensor_materialization_attempted": result.metadata.tensor_materialization_attempted,
        "array_materialization_attempted": result.metadata.array_materialization_attempted,
        "forward_execution_attempted": result.metadata.forward_execution_attempted,
        "output_generation_attempted": result.metadata.output_generation_attempted,
        "training_attempted": result.metadata.training_attempted,
        "no_tensor_created": result.no_tensor_created,
        "no_array_created": result.no_array_created,
        "no_forward_execution": result.no_forward_execution,
        "no_output_generation": result.no_output_generation,
        "no_training_loop": result.no_training_loop,
        "no_optimizer": result.no_optimizer,
        "no_checkpointing": result.no_checkpointing,
        "no_artifact_generation": result.no_artifact_generation,
        "no_final_comparison": result.no_final_comparison,
        "no_scientific_conclusion": result.no_scientific_conclusion,
        "result": tensor_materialization_request_result_to_json_dict(result),
        "reason": "p38_tensor_materialization_request_smoke_completed",
    }
    return out

def main() -> int:
    if len(sys.argv) > 1:
        print("Error: smoke script does not accept arguments.", file=sys.stderr)
        return 1
    data = run_p38_tensor_materialization_request_smoke()
    print(compact_json(data))
    return 0

if __name__ == "__main__":
    sys.exit(main())
