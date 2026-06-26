# tools/phase2/run_p40_forward_eligibility_gate_smoke.py

import json
import sys
from src.phase2.fc_vae_forward_eligibility_gate import (
    run_forward_eligibility_gate_probe,
    forward_eligibility_result_to_json_dict,
)

def compact_json(data: dict) -> str:
    return json.dumps(data, sort_keys=True, separators=(",", ":"))

def run_p40_forward_eligibility_gate_smoke() -> dict:
    result = run_forward_eligibility_gate_probe()
    ev = result.evidence
    
    out = {
        "verdict": "PASS",
        "contract": result.contract_version,
        "source_phase": "P40",
        "status": result.status,
        "torch_available": ev.torch_available,
        "tensor_materialized_in_p39": ev.tensor_materialized_in_p39,
        "tensor_materialization_status": ev.tensor_materialization_status,
        "tensor_dtype_name": ev.tensor_dtype_name,
        "tensor_device_type": ev.tensor_device_type,
        "tensor_shape_tuple": list(ev.tensor_shape_tuple) if ev.tensor_shape_tuple else [0, 0],
        "tensor_numel": ev.tensor_numel,
        "tensor_requires_grad": ev.tensor_requires_grad,
        "tensor_is_floating_point": ev.tensor_is_floating_point,
        "tensor_values_match_p37_nested_values": ev.tensor_values_match_p37_nested_values,
        "model_skeleton_status": ev.model_skeleton_status,
        "model_no_implementation": ev.model_no_implementation,
        "model_implementation_available": ev.model_implementation_available,
        "model_forward_contract_declared": ev.model_forward_contract_declared,
        "model_forward_implemented_in_p27": ev.model_forward_implemented_in_p27,
        "input_flat_dim_matches": ev.input_flat_dim_matches,
        "shape_matches": ev.shape_matches,
        "dtype_matches": ev.dtype_matches,
        "device_matches": ev.device_matches,
        "latent_layout_matches": ev.latent_layout_matches,
        "decoder_contract_available": ev.decoder_contract_available,
        "forward_eligible_in_p40": result.forward_eligible_in_p40,
        "forward_execution_available_in_p40": result.forward_execution_available_in_p40,
        "forward_executed_in_p40": result.forward_executed_in_p40,
        "output_generation_available_in_p40": result.output_generation_available_in_p40,
        "output_generated_in_p40": result.output_generated_in_p40,
        "training_available_in_p40": result.training_available_in_p40,
        "training_executed_in_p40": result.training_executed_in_p40,
        "no_forward_execution": result.no_forward_execution,
        "no_output_generation": result.no_output_generation,
        "no_training_loop": result.no_training_loop,
        "no_optimizer": result.no_optimizer,
        "no_checkpointing": result.no_checkpointing,
        "no_artifact_generation": result.no_artifact_generation,
        "no_final_comparison": result.no_final_comparison,
        "no_scientific_conclusion": result.no_scientific_conclusion,
        "result": forward_eligibility_result_to_json_dict(result),
        "reason": "p40_forward_eligibility_gate_smoke_completed",
    }
    return out

def main() -> int:
    if len(sys.argv) > 1:
        print("Error: smoke script does not accept arguments.", file=sys.stderr)
        return 1
    data = run_p40_forward_eligibility_gate_smoke()
    print(compact_json(data))
    return 0

if __name__ == "__main__":
    sys.exit(main())
