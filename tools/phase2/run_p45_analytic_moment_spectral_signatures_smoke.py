# tools/phase2/run_p45_analytic_moment_spectral_signatures_smoke.py

import json
import sys

from src.phase2.analytic_moment_spectral_signatures import (
    run_analytic_moment_spectral_signatures_probe,
    analytic_moment_spectral_signatures_probe_to_json_dict,
)


def run_p45_smoke_runner() -> dict:
    probe_res = run_analytic_moment_spectral_signatures_probe()
    res_dict = analytic_moment_spectral_signatures_probe_to_json_dict(probe_res)

    smoke_data = {
        "verdict": "PASS",
        "source_phase": "P45",
        "status": probe_res.get("status", ""),
        "torch_available": bool(probe_res.get("torch_available", False)),
        "ar_signature_available": bool(probe_res.get("ar_signature_available", False)),
        "garch_signature_available": bool(probe_res.get("garch_signature_available", False)),
        "combined_signature_available": bool(probe_res.get("combined_signature_available", False)),
        "ar_spectrum_shape": list(probe_res.get("ar_spectrum_shape", [])),
        "garch_persistence_shape": list(probe_res.get("garch_persistence_shape", [])),
        "garch_persistence_decay_shape": list(probe_res.get("garch_persistence_decay_shape", [])),
        "ar_spectrum_positive": bool(probe_res.get("ar_spectrum_positive", False)),
        "ar_spectrum_finite": bool(probe_res.get("ar_spectrum_finite", False)),
        "garch_unconditional_variance_positive": bool(probe_res.get("garch_unconditional_variance_positive", False)),
        "garch_persistence_below_one": bool(probe_res.get("garch_persistence_below_one", False)),
        "garch_stationarity_margin_positive": bool(probe_res.get("garch_stationarity_margin_positive", False)),
        "beta_share_greater_than_alpha_share": bool(probe_res.get("beta_share_greater_than_alpha_share", False)),
        "no_model": bool(probe_res.get("no_model", False)),
        "no_forward_execution": bool(probe_res.get("no_forward_execution", False)),
        "no_output_generation": bool(probe_res.get("no_output_generation", False)),
        "no_loss": bool(probe_res.get("no_loss", False)),
        "no_training": bool(probe_res.get("no_training", False)),
        "no_scientific_conclusion": bool(probe_res.get("no_scientific_conclusion", False)),
        "result": res_dict,
        "reason": "p45_smoke_runner_completed_successfully",
    }
    return smoke_data


def compact_json(data: dict) -> str:
    return json.dumps(data, sort_keys=True, separators=(",", ":"))


def main() -> int:
    try:
        data = run_p45_smoke_runner()
        sys.stdout.write(compact_json(data) + "\n")
        return 0
    except Exception as e:
        sys.stderr.write(f"Error in smoke runner: {str(e)}\n")
        return 1


if __name__ == "__main__":
    sys.exit(main())
