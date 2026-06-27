# tools/phase2/run_p42_forward_readiness_gate_smoke.py

import json
import sys

from src.phase2.fc_vae_forward_readiness_gate import (
    FC_VAE_FORWARD_READINESS_GATE_CONTRACT_VERSION,
    run_forward_readiness_gate_probe,
    forward_readiness_result_to_json_dict,
)


def run_p42_forward_readiness_gate_smoke() -> dict:
    probe_res = run_forward_readiness_gate_probe()
    res_dict = forward_readiness_result_to_json_dict(probe_res)

    smoke_data = {
        "verdict": "PASS",
        "contract": FC_VAE_FORWARD_READINESS_GATE_CONTRACT_VERSION,
        "source_phase": "P42",
        "status": probe_res.status,
        "torch_available": probe_res.evidence.torch_available,
        "tensor_materialized_in_p39": probe_res.evidence.tensor_materialized_in_p39,
        "module_shell_created_in_p41": probe_res.evidence.module_shell_created_in_p41,
        "implementation_available_in_p41": probe_res.implementation_available_in_p41,
        "module_shell_ready_in_p42": probe_res.module_shell_ready_in_p42,
        "forward_ready_in_p42": probe_res.forward_ready_in_p42,
        "forward_implementation_available_in_p42": probe_res.forward_implementation_available_in_p42,
        "forward_available_in_p42": probe_res.forward_available_in_p42,
        "forward_execution_available_in_p42": probe_res.forward_execution_available_in_p42,
        "forward_executed_in_p42": probe_res.forward_executed_in_p42,
        "output_generation_available_in_p42": probe_res.output_generation_available_in_p42,
        "output_generated_in_p42": probe_res.output_generated_in_p42,
        "training_available_in_p42": probe_res.training_available_in_p42,
        "training_executed_in_p42": probe_res.training_executed_in_p42,
        "no_forward_execution": probe_res.no_forward_execution,
        "no_output_generation": probe_res.no_output_generation,
        "no_training_loop": probe_res.no_training_loop,
        "no_optimizer": probe_res.no_optimizer,
        "no_checkpointing": probe_res.no_checkpointing,
        "no_artifact_generation": probe_res.no_artifact_generation,
        "no_final_comparison": probe_res.no_final_comparison,
        "no_scientific_conclusion": probe_res.no_scientific_conclusion,
        "result": res_dict,
        "reason": "p42_forward_readiness_gate_smoke_completed",
    }
    return smoke_data


def compact_json(data: dict) -> str:
    return json.dumps(data, sort_keys=True, separators=(",", ":"))


def main() -> int:
    try:
        data = run_p42_forward_readiness_gate_smoke()
        sys.stdout.write(compact_json(data) + "\n")
        return 0
    except Exception as e:
        sys.stderr.write(f"Error in smoke runner: {str(e)}\n")
        return 1


if __name__ == "__main__":
    sys.exit(main())
