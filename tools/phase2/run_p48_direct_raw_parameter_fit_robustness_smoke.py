# tools/phase2/run_p48_direct_raw_parameter_fit_robustness_smoke.py
#
# P48 smoke script: prints compact sorted JSON.
# No argparse. Rejects command-line arguments.
# No subprocess. No direct torch import.

import json
import sys

_USAGE_NOTE = (
    "run_p48_direct_raw_parameter_fit_robustness_smoke.py takes no arguments. "
    "Run as: python tools/phase2/run_p48_direct_raw_parameter_fit_robustness_smoke.py"
)

if len(sys.argv) != 1:
    print(
        json.dumps(
            {"verdict": "REJECTED", "reason": _USAGE_NOTE},
            sort_keys=True,
        )
    )
    sys.exit(1)

from src.phase2.direct_raw_parameter_fit_robustness_smoke import (
    run_direct_fit_robustness_probe,
    direct_fit_robustness_probe_to_json_dict,
    DIRECT_FIT_ROBUSTNESS_KIND,
    P48_STATUS_AVAILABLE,
    P48_SCENARIO_COUNT,
)

probe_res = run_direct_fit_robustness_probe()

all_checks = [
    probe_res.get("torch_available", False),
    probe_res.get("all_scenarios_loss_decreased", False),
    probe_res.get("all_losses_finite", False),
    probe_res.get("scenarios_passed", 0) == P48_SCENARIO_COUNT,
    probe_res.get("scenario_count", 0) == P48_SCENARIO_COUNT,
    probe_res.get("no_model", False),
    probe_res.get("no_vae", False),
    probe_res.get("no_encoder", False),
    probe_res.get("no_decoder", False),
    probe_res.get("no_dataset", False),
    probe_res.get("no_dataloader", False),
    probe_res.get("no_torch_optimizer", False),
    probe_res.get("no_scientific_conclusion", False),
    probe_res.get("status") == P48_STATUS_AVAILABLE,
]

verdict = "PASS" if all(all_checks) else "FAIL"

json_dict = direct_fit_robustness_probe_to_json_dict(probe_res)
output = dict(json_dict)
output["verdict"] = verdict
output["source_phase"] = "P48"

print(json.dumps(output, sort_keys=True, separators=(",", ":")))
