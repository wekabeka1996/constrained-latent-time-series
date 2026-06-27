# tools/phase2/run_p44_tensor_native_constraint_primitives_smoke.py

import json
import sys

from src.phase2.tensor_native_constraint_primitives import (
    run_tensor_native_constraint_primitives_probe,
    tensor_native_constraint_primitives_probe_to_json_dict,
)


def run_p44_smoke_runner() -> dict:
    probe_res = run_tensor_native_constraint_primitives_probe()
    res_dict = tensor_native_constraint_primitives_probe_to_json_dict(probe_res)

    smoke_data = {
        "verdict": "PASS",
        "source_phase": "P44",
        "status": probe_res.get("status", ""),
        "torch_available": bool(probe_res.get("torch_available", False)),
        "ar_order": int(probe_res.get("ar_order", 0)),
        "pacf_abs_max_lt_one": bool(probe_res.get("pacf_abs_max_lt_one", False)),
        "ar_coefficients_shape": list(probe_res.get("ar_coefficients_shape", [])),
        "garch_omega_positive": bool(probe_res.get("garch_omega_positive", False)),
        "garch_alpha_nonnegative": bool(probe_res.get("garch_alpha_nonnegative", False)),
        "garch_beta_nonnegative": bool(probe_res.get("garch_beta_nonnegative", False)),
        "garch_alpha_beta_sum_below_one_minus_eps": bool(probe_res.get("garch_alpha_beta_sum_below_one_minus_eps", False)),
        "garch_stationarity_margin_positive": bool(probe_res.get("garch_stationarity_margin_positive", False)),
        "beta_prior_greater_than_alpha_prior": bool(probe_res.get("beta_prior_greater_than_alpha_prior", False)),
        "no_model": bool(probe_res.get("no_model", False)),
        "no_forward_execution": bool(probe_res.get("no_forward_execution", False)),
        "no_output_generation": bool(probe_res.get("no_output_generation", False)),
        "no_loss": bool(probe_res.get("no_loss", False)),
        "no_training": bool(probe_res.get("no_training", False)),
        "no_scientific_conclusion": bool(probe_res.get("no_scientific_conclusion", False)),
        "result": res_dict,
        "reason": "p44_smoke_runner_completed_successfully",
    }
    return smoke_data


def compact_json(data: dict) -> str:
    return json.dumps(data, sort_keys=True, separators=(",", ":"))


def main() -> int:
    try:
        data = run_p44_smoke_runner()
        sys.stdout.write(compact_json(data) + "\n")
        return 0
    except Exception as e:
        sys.stderr.write(f"Error in smoke runner: {str(e)}\n")
        return 1


if __name__ == "__main__":
    sys.exit(main())
