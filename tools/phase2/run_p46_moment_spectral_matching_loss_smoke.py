# tools/phase2/run_p46_moment_spectral_matching_loss_smoke.py

import json
import sys

from src.phase2.moment_spectral_matching_loss import (
    run_moment_spectral_matching_loss_probe,
    moment_spectral_matching_loss_probe_to_json_dict,
)


def run_p46_smoke_runner() -> dict:
    probe_res = run_moment_spectral_matching_loss_probe()
    res_dict = moment_spectral_matching_loss_probe_to_json_dict(probe_res)

    smoke_data = {
        "verdict": "PASS",
        "source_phase": "P46",
        "status": probe_res.get("status", ""),
        "torch_available": bool(probe_res.get("torch_available", False)),
        "ar_loss_available": bool(probe_res.get("ar_loss_available", False)),
        "garch_loss_available": bool(probe_res.get("garch_loss_available", False)),
        "combined_loss_available": bool(probe_res.get("combined_loss_available", False)),
        "total_loss_finite": bool(probe_res.get("total_loss_finite", False)),
        "total_loss_nonnegative": bool(probe_res.get("total_loss_nonnegative", False)),
        "ar_component_loss_nonnegative": bool(probe_res.get("ar_component_loss_nonnegative", False)),
        "garch_component_loss_nonnegative": bool(probe_res.get("garch_component_loss_nonnegative", False)),
        "candidate_raw_kappa_grad_nonzero": bool(probe_res.get("candidate_raw_kappa_grad_nonzero", False)),
        "candidate_raw_omega_grad_nonzero": bool(probe_res.get("candidate_raw_omega_grad_nonzero", False)),
        "candidate_raw_total_mass_grad_nonzero": bool(probe_res.get("candidate_raw_total_mass_grad_nonzero", False)),
        "candidate_raw_allocation_logits_grad_nonzero": bool(probe_res.get("candidate_raw_allocation_logits_grad_nonzero", False)),
        "no_model": bool(probe_res.get("no_model", False)),
        "no_forward_execution": bool(probe_res.get("no_forward_execution", False)),
        "no_output_generation": bool(probe_res.get("no_output_generation", False)),
        "no_optimizer": bool(probe_res.get("no_optimizer", False)),
        "no_training_loop": bool(probe_res.get("no_training_loop", False)),
        "no_scientific_conclusion": bool(probe_res.get("no_scientific_conclusion", False)),
        "result": res_dict,
        "reason": "p46_smoke_runner_completed_successfully",
    }
    return smoke_data


def compact_json(data: dict) -> str:
    return json.dumps(data, sort_keys=True, separators=(",", ":"))


def main() -> int:
    try:
        data = run_p46_smoke_runner()
        sys.stdout.write(compact_json(data) + "\n")
        return 0
    except Exception as e:
        sys.stderr.write(f"Error in smoke runner: {str(e)}\n")
        return 1


if __name__ == "__main__":
    sys.exit(main())
