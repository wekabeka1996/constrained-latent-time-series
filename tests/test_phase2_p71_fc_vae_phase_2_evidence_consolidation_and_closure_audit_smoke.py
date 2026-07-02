# tests/test_phase2_p71_fc_vae_phase_2_evidence_consolidation_and_closure_audit_smoke.py

import json
import os
import subprocess
import sys


def test_p71_smoke_01_run_success():
    env = os.environ.copy()
    env["PYTHONPATH"] = "."
    
    # Run the smoke tool
    res = subprocess.run(
        [sys.executable, "tools/phase2/run_p71_fc_vae_phase_2_evidence_consolidation_and_closure_audit_smoke.py"],
        capture_output=True,
        text=True,
        check=True,
        env=env
    )
    
    # Parse output JSON
    data = json.loads(res.stdout.strip())
    
    assert data["verdict"] == "PASS"
    assert data["source_phase"] == "P71"
    assert data["source_evidence_phase"] == "P70"
    assert data["status"] == "fc_vae_phase_2_evidence_consolidation_and_closure_audit_completed"
    assert data["contract_version"] == "phase2_p71_fc_vae_phase_2_evidence_consolidation_and_closure_audit_contract_v1"
    
    assert len(data["accepted_phase_2_ledger"]) == 22
    assert len(data["consolidated_supported_claims"]) > 0
    assert len(data["consolidated_unsupported_claims"]) > 0
    
    # boundary flags
    assert data["no_new_optimization"] is True
    assert data["no_direct_optimizer_created"] is True
    assert data["no_direct_model_created"] is True
    assert data["no_direct_torch_import"] is True
    assert data["no_dataset"] is True
    assert data["no_dataloader"] is True
    assert data["no_epoch_loop"] is True
    assert data["no_batch_loop"] is True
    assert data["no_scheduler"] is True
    assert data["no_checkpointing"] is True
    assert data["no_generalization_claim"] is True
    assert data["no_generation_claim"] is True
    assert data["no_gsb_claim"] is True
    assert data["no_scientific_conclusion_beyond_diagnostic_closure"] is True
    assert data["no_latent_learning_claim"] is True
    assert data["no_vae_success_claim"] is True
    assert data["no_convergence_claim"] is True
    assert data["no_semantic_geometry_proof_claim"] is True
    assert data["no_transfer_proof_claim"] is True
    assert data["no_seed_robustness_claim"] is True
    assert data["no_component_proof_claim"] is True
    assert data["no_label_specific_semantic_proof_claim"] is True
    assert data["no_numeric_identity_proof_claim"] is True
    assert data["no_perturbation_robustness_proof_claim"] is True
    assert data["no_coordinate_semantic_proof_claim"] is True
    assert data["no_latent_kl_explanatory_proof_claim"] is True
    assert data["no_kl_semantic_signal_proof_claim"] is True
    assert data["no_a_b_generate_c_claim"] is True
    assert data["no_beta_change"] is True


def test_p71_smoke_02_rejects_args():
    res = subprocess.run(
        [sys.executable, "tools/phase2/run_p71_fc_vae_phase_2_evidence_consolidation_and_closure_audit_smoke.py", "--invalid-arg"],
        capture_output=True,
        text=True
    )
    assert res.returncode != 0
    assert "Error:" in res.stderr
