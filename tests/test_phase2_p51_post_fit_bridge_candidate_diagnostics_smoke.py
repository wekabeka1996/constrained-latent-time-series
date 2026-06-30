# tests/test_phase2_p51_post_fit_bridge_candidate_diagnostics_smoke.py

import json
import os
import subprocess
import sys

def test_p51_smoke_01_run_success():
    env = os.environ.copy()
    env["PYTHONPATH"] = "."
    
    # Run the smoke tool
    res = subprocess.run(
        [sys.executable, "tools/phase2/run_p51_post_fit_bridge_candidate_diagnostics_smoke.py"],
        capture_output=True,
        text=True,
        check=True,
        env=env
    )
    
    # Parse output JSON
    data = json.loads(res.stdout.strip())
    
    assert data["verdict"] == "PASS"
    assert data["source_phase"] == "P51"
    assert data["status"] == "post_fit_bridge_candidate_diagnostics_available_no_model_no_vae_no_gsb_no_science"
    assert data["bridge_target_count"] == 3
    assert data["targets_passed_loss_decrease"] == 3
    assert data["lambda_values"] == [0.25, 0.5, 0.75]
    assert data["all_targets_loss_decreased"] is True
    assert data["all_losses_finite"] is True
    assert data["all_ar_coefficients_finite"] is True
    assert data["all_garch_tensor_fields_finite"] is True
    assert data["all_signature_tensor_fields_finite"] is True
    
    # Verify boundaries and no claims
    assert data["no_model"] is True
    assert data["no_vae"] is True
    assert data["no_dataset"] is True
    assert data["no_torch_optimizer"] is True
    assert data["no_gsb_claim"] is True
    assert data["no_generation_claim"] is True
    assert data["no_scientific_conclusion"] is True
    assert data["realizability_claim"] == "post_fit_diagnostics_only_not_state_validity"
    
    # Verify no lists/arrays (e.g. the 100-long AR spectrum log) are leaked in diagnostics
    for r in data["target_diagnostics"]:
        for k, v in r.items():
            if isinstance(v, list):
                assert k == "ar_coefficients_tensor_shape"


def test_p51_smoke_02_rejects_args():
    res = subprocess.run(
        [sys.executable, "tools/phase2/run_p51_post_fit_bridge_candidate_diagnostics_smoke.py", "--invalid-arg"],
        capture_output=True,
        text=True
    )
    assert res.returncode != 0
    assert "Error:" in res.stderr
