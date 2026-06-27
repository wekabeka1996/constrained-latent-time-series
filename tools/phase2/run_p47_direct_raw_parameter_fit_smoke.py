# tools/phase2/run_p47_direct_raw_parameter_fit_smoke.py
#
# P47 smoke script: prints compact sorted JSON.
# No argparse. Rejects command-line arguments.
# No subprocess. No direct torch import.

import json
import sys

_USAGE_NOTE = (
    "run_p47_direct_raw_parameter_fit_smoke.py takes no arguments. "
    "Run as: python tools/phase2/run_p47_direct_raw_parameter_fit_smoke.py"
)

if len(sys.argv) != 1:
    print(
        json.dumps(
            {"verdict": "REJECTED", "reason": _USAGE_NOTE},
            sort_keys=True,
        )
    )
    sys.exit(1)

from src.phase2.direct_raw_parameter_fit_smoke import (
    run_direct_raw_parameter_fit_probe,
    compact_direct_raw_parameter_fit_json,
    DIRECT_RAW_PARAMETER_FIT_KIND,
    P47_STATUS_AVAILABLE,
)

probe_res = run_direct_raw_parameter_fit_probe()

all_checks = [
    probe_res.get("torch_available", False),
    probe_res.get("initial_loss_finite", False),
    probe_res.get("final_loss_finite", False),
    probe_res.get("initial_loss_nonnegative", False),
    probe_res.get("final_loss_nonnegative", False),
    probe_res.get("loss_decreased", False),
    probe_res.get("loss_decrease_positive", False),
    probe_res.get("no_model", False),
    probe_res.get("no_vae", False),
    probe_res.get("no_encoder", False),
    probe_res.get("no_decoder", False),
    probe_res.get("no_dataset", False),
    probe_res.get("no_dataloader", False),
    probe_res.get("no_torch_optimizer", False),
    probe_res.get("no_scientific_conclusion", False),
    probe_res.get("status") == P47_STATUS_AVAILABLE,
]

verdict = "PASS" if all(all_checks) else "FAIL"

output = {
    "verdict": verdict,
    "source_phase": "P47",
    "kind": DIRECT_RAW_PARAMETER_FIT_KIND,
    "status": probe_res.get("status", ""),
    "torch_available": probe_res.get("torch_available", False),
    "initial_loss_finite": probe_res.get("initial_loss_finite", False),
    "final_loss_finite": probe_res.get("final_loss_finite", False),
    "initial_loss_nonnegative": probe_res.get("initial_loss_nonnegative", False),
    "final_loss_nonnegative": probe_res.get("final_loss_nonnegative", False),
    "loss_decreased": probe_res.get("loss_decreased", False),
    "loss_decrease_positive": probe_res.get("loss_decrease_positive", False),
    "step_count": probe_res.get("step_count", 0),
    "initial_loss_value": probe_res.get("initial_loss_value", 0.0),
    "final_loss_value": probe_res.get("final_loss_value", 0.0),
    "loss_delta_value": probe_res.get("loss_delta_value", 0.0),
    "no_model": probe_res.get("no_model", False),
    "no_vae": probe_res.get("no_vae", False),
    "no_encoder": probe_res.get("no_encoder", False),
    "no_decoder": probe_res.get("no_decoder", False),
    "no_dataset": probe_res.get("no_dataset", False),
    "no_dataloader": probe_res.get("no_dataloader", False),
    "no_torch_optimizer": probe_res.get("no_torch_optimizer", False),
    "no_scientific_conclusion": probe_res.get("no_scientific_conclusion", False),
    "reason": probe_res.get("reason", ""),
}

print(json.dumps(output, sort_keys=True, separators=(",", ":")))
