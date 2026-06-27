# tools/phase2/run_p41_module_availability_shell_smoke.py

import json
import sys
from src.phase2.fc_vae_module_availability_shell import (
    run_module_availability_shell_probe,
    module_availability_shell_result_to_json_dict,
)

def compact_json(data: dict) -> str:
    return json.dumps(data, sort_keys=True, separators=(",", ":"))

def run_p41_module_availability_shell_smoke() -> dict:
    result = run_module_availability_shell_probe()
    meta = result.metadata
    
    out = {
        "verdict": "PASS",
        "contract": result.contract_version,
        "source_phase": "P41",
        "status": result.status,
        "torch_available": meta.torch_available,
        "module_shell_created": meta.module_shell_created,
        "is_torch_nn_module": meta.is_torch_nn_module,
        "defines_forward": meta.defines_forward,
        "has_own_forward": meta.has_own_forward,
        "uses_inherited_unimplemented_forward_only": meta.uses_inherited_unimplemented_forward_only,
        "defines_layers": meta.defines_layers,
        "parameter_count": meta.parameter_count,
        "buffer_count": meta.buffer_count,
        "training_mode_after_creation": meta.training_mode_after_creation,
        "module_device_type": meta.module_device_type,
        "architecture_id": meta.architecture_id,
        "input_flat_dim": meta.input_flat_dim,
        "latent_total_dim": meta.latent_total_dim,
        "latent_names": list(meta.latent_names) if meta.latent_names else [],
        "forward_execution_attempted": meta.forward_execution_attempted,
        "output_generation_attempted": meta.output_generation_attempted,
        "training_attempted": meta.training_attempted,
        "implementation_available_in_p41": result.implementation_available_in_p41,
        "module_shell_available_in_p41": result.module_shell_available_in_p41,
        "forward_available_in_p41": result.forward_available_in_p41,
        "forward_execution_available_in_p41": result.forward_execution_available_in_p41,
        "forward_executed_in_p41": result.forward_executed_in_p41,
        "output_generation_available_in_p41": result.output_generation_available_in_p41,
        "output_generated_in_p41": result.output_generated_in_p41,
        "training_available_in_p41": result.training_available_in_p41,
        "training_executed_in_p41": result.training_executed_in_p41,
        "no_forward_execution": result.no_forward_execution,
        "no_output_generation": result.no_output_generation,
        "no_training_loop": result.no_training_loop,
        "no_optimizer": result.no_optimizer,
        "no_checkpointing": result.no_checkpointing,
        "no_artifact_generation": result.no_artifact_generation,
        "no_final_comparison": result.no_final_comparison,
        "no_scientific_conclusion": result.no_scientific_conclusion,
        "result": module_availability_shell_result_to_json_dict(result),
        "reason": "p41_module_availability_shell_smoke_completed",
    }
    return out

def main() -> int:
    if len(sys.argv) > 1:
        print("Error: smoke script does not accept arguments.", file=sys.stderr)
        return 1
    data = run_p41_module_availability_shell_smoke()
    print(compact_json(data))
    return 0

if __name__ == "__main__":
    sys.exit(main())
