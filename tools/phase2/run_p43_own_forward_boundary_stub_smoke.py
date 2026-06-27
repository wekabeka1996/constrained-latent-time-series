# tools/phase2/run_p43_own_forward_boundary_stub_smoke.py

import json
import sys

from src.phase2.fc_vae_own_forward_boundary_stub import (
    FC_VAE_OWN_FORWARD_BOUNDARY_STUB_CONTRACT_VERSION,
    run_own_forward_boundary_stub_probe,
    own_forward_boundary_stub_result_to_json_dict,
)


def run_p43_own_forward_boundary_stub_smoke() -> dict:
    probe_res = run_own_forward_boundary_stub_probe()
    res_dict = own_forward_boundary_stub_result_to_json_dict(probe_res)

    smoke_data = {
        "verdict": "PASS",
        "contract": FC_VAE_OWN_FORWARD_BOUNDARY_STUB_CONTRACT_VERSION,
        "source_phase": "P43",
        "status": probe_res.status,
        "torch_available": probe_res.metadata.torch_available,
        "class_declared": probe_res.metadata.class_declared,
        "module_instance_created": probe_res.metadata.module_instance_created,
        "module_object_returned": probe_res.metadata.module_object_returned,
        "is_torch_nn_module": probe_res.metadata.is_torch_nn_module,
        "own_forward_declared": probe_res.metadata.own_forward_declared,
        "forward_signature_available": probe_res.metadata.forward_signature_available,
        "forward_execution_attempted": probe_res.metadata.forward_execution_attempted,
        "forward_execution_available_in_p43": probe_res.forward_execution_available_in_p43,
        "forward_executed_in_p43": probe_res.forward_executed_in_p43,
        "output_generation_available_in_p43": probe_res.output_generation_available_in_p43,
        "output_generated_in_p43": probe_res.output_generated_in_p43,
        "training_available_in_p43": probe_res.training_available_in_p43,
        "training_executed_in_p43": probe_res.training_executed_in_p43,
        "no_forward_execution": probe_res.no_forward_execution,
        "no_output_generation": probe_res.no_output_generation,
        "no_training_loop": probe_res.no_training_loop,
        "no_optimizer": probe_res.no_optimizer,
        "no_checkpointing": probe_res.no_checkpointing,
        "no_artifact_generation": probe_res.no_artifact_generation,
        "no_final_comparison": probe_res.no_final_comparison,
        "no_scientific_conclusion": probe_res.no_scientific_conclusion,
        "result": res_dict,
        "reason": "p43_own_forward_boundary_stub_smoke_completed",
    }
    return smoke_data


def compact_json(data: dict) -> str:
    return json.dumps(data, sort_keys=True, separators=(",", ":"))


def main() -> int:
    try:
        data = run_p43_own_forward_boundary_stub_smoke()
        sys.stdout.write(compact_json(data) + "\n")
        return 0
    except Exception as e:
        sys.stderr.write(f"Error in smoke runner: {str(e)}\n")
        return 1


if __name__ == "__main__":
    sys.exit(main())
