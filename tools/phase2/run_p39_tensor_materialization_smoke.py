# tools/phase2/run_p39_tensor_materialization_smoke.py

import json
import sys
from src.phase2.fc_vae_tensor_materialization import (
    run_tensor_materialization_probe,
    p39_tensor_materialization_result_to_json_dict,
)

def compact_json(data: dict) -> str:
    return json.dumps(data, sort_keys=True, separators=(",", ":"))

def run_p39_tensor_materialization_smoke() -> dict:
    result = run_tensor_materialization_probe()
    
    if not result.metadata.torch_available:
        out = {
            "verdict": "BLOCKED",
            "contract": result.contract_version,
            "source_phase": "P39",
            "status": result.status,
            "torch_available": False,
            "torch_import_safe": result.metadata.torch_import_safe,
            "top_level_torch_import_required": result.metadata.top_level_torch_import_required,
            "torch_version": result.metadata.torch_version,
            "cuda_available": result.metadata.cuda_available,
            "target_framework": result.request.target_framework,
            "target_dtype": result.request.target_dtype,
            "target_device": result.request.target_device,
            "target_layout_kind": result.request.target_layout_kind,
            "shape_tuple": list(result.request.shape_tuple),
            "batch_size": result.request.batch_size,
            "input_flat_dim": result.request.input_flat_dim,
            "full_vector_length": result.request.full_vector_length,
            "tensor_materialization_available_in_p39": result.tensor_materialization_available_in_p39,
            "tensor_materialized_in_p39": result.tensor_materialized_in_p39,
            "forward_execution_available_in_p39": result.forward_execution_available_in_p39,
            "output_generation_available_in_p39": result.output_generation_available_in_p39,
            "training_available_in_p39": result.training_available_in_p39,
            "no_forward_execution": result.no_forward_execution,
            "no_output_generation": result.no_output_generation,
            "no_training_loop": result.no_training_loop,
            "no_optimizer": result.no_optimizer,
            "no_checkpointing": result.no_checkpointing,
            "no_artifact_generation": result.no_artifact_generation,
            "no_final_comparison": result.no_final_comparison,
            "no_scientific_conclusion": result.no_scientific_conclusion,
            "result": p39_tensor_materialization_result_to_json_dict(result),
            "reason": "p39_tensor_materialization_smoke_completed",
        }
        return out
        
    t = result.materialized_tensor
    if t is None:
        raise ValueError("Materialized tensor cannot be None if torch is available")

    out = {
        "verdict": "PASS",
        "contract": result.contract_version,
        "source_phase": "P39",
        "status": result.status,
        "torch_available": True,
        "torch_import_safe": result.metadata.torch_import_safe,
        "top_level_torch_import_required": result.metadata.top_level_torch_import_required,
        "torch_version": result.metadata.torch_version,
        "cuda_available": result.metadata.cuda_available,
        "target_framework": result.request.target_framework,
        "target_dtype": result.request.target_dtype,
        "target_device": result.request.target_device,
        "target_layout_kind": result.request.target_layout_kind,
        "shape_tuple": list(result.request.shape_tuple),
        "batch_size": result.request.batch_size,
        "input_flat_dim": result.request.input_flat_dim,
        "full_vector_length": result.request.full_vector_length,
        "tensor_type_name": t.tensor_type_name,
        "tensor_dtype_name": t.tensor_dtype_name,
        "tensor_device_type": t.tensor_device_type,
        "tensor_shape_tuple": list(t.tensor_shape_tuple),
        "tensor_numel": t.tensor_numel,
        "tensor_requires_grad": t.tensor_requires_grad,
        "tensor_is_floating_point": t.tensor_is_floating_point,
        "tensor_values_match_p37_nested_values": t.tensor_values_match_p37_nested_values,
        "first_row_first_4_values": list(t.first_row_first_4_values),
        "second_row_first_4_values": list(t.second_row_first_4_values),
        "tensor_materialization_available_in_p39": result.tensor_materialization_available_in_p39,
        "tensor_materialized_in_p39": result.tensor_materialized_in_p39,
        "forward_execution_available_in_p39": result.forward_execution_available_in_p39,
        "output_generation_available_in_p39": result.output_generation_available_in_p39,
        "training_available_in_p39": result.training_available_in_p39,
        "no_forward_execution": result.no_forward_execution,
        "no_output_generation": result.no_output_generation,
        "no_training_loop": result.no_training_loop,
        "no_optimizer": result.no_optimizer,
        "no_checkpointing": result.no_checkpointing,
        "no_artifact_generation": result.no_artifact_generation,
        "no_final_comparison": result.no_final_comparison,
        "no_scientific_conclusion": result.no_scientific_conclusion,
        "result": p39_tensor_materialization_result_to_json_dict(result),
        "reason": "p39_tensor_materialization_smoke_completed",
    }
    return out

def main() -> int:
    if len(sys.argv) > 1:
        print("Error: smoke script does not accept arguments.", file=sys.stderr)
        return 1
    data = run_p39_tensor_materialization_smoke()
    print(compact_json(data))
    return 0

if __name__ == "__main__":
    sys.exit(main())
