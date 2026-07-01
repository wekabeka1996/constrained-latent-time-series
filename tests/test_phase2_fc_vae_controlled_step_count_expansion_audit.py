# tests/test_phase2_fc_vae_controlled_step_count_expansion_audit.py

import json
import math
import pathlib
import sys
import pytest
import re

# Crucial: No top-level torch import.

from src.phase2.fc_vae_controlled_step_count_expansion_audit import (
    FC_VAE_STEP_COUNT_EXPANSION_AUDIT_CONTRACT_VERSION,
    FC_VAE_STEP_COUNT_EXPANSION_AUDIT_KIND,
    FC_VAE_STEP_COUNT_EXPANSION_AUDIT_MODULE_NAME,
    FC_VAE_STEP_COUNT_EXPANSION_AUDIT_STATUS_AVAILABLE,
    P61_TARGET_COUNT,
    P61_TARGET_IDS,
    P61_STEP_COUNTS,
    P61_BASELINE_STEP_COUNT,
    P61_MAX_STEP_COUNT,
    P61_SEED,
    P61_BETA,
    P61_LEARNING_RATE,
    P61_OPTIMIZER_NAME,
    P61_EPS_MODE,
    P61_DOMINANCE_SHARE_WARNING_THRESHOLD,
    P61_DOMINANCE_SHARE_BLOCK_THRESHOLD,
    P61_LOSS_EXPLOSION_MULTIPLIER_BLOCK_THRESHOLD,
    P61_KL_EXPLOSION_MULTIPLIER_BLOCK_THRESHOLD,
    load_torch_for_p61_step_count_expansion_audit,
    run_p61_single_step_count,
    compare_p61_step_count_runs,
    run_fc_vae_controlled_step_count_expansion_audit_probe,
    fc_vae_controlled_step_count_expansion_audit_probe_to_json_dict,
)


def test_p61_01_constants_exact():
    assert FC_VAE_STEP_COUNT_EXPANSION_AUDIT_CONTRACT_VERSION == "phase2_p61_fc_vae_controlled_step_count_expansion_audit_contract_v1"
    assert FC_VAE_STEP_COUNT_EXPANSION_AUDIT_KIND == "fc_vae_controlled_step_count_expansion_audit_no_dataset_no_generalization"
    assert FC_VAE_STEP_COUNT_EXPANSION_AUDIT_MODULE_NAME == "src.phase2.fc_vae_controlled_step_count_expansion_audit"
    assert FC_VAE_STEP_COUNT_EXPANSION_AUDIT_STATUS_AVAILABLE == "fc_vae_controlled_step_count_expansion_audit_available_no_dataset_no_generalization"
    assert P61_TARGET_COUNT == 3
    assert P61_TARGET_IDS == ("bridge_lambda_0_25", "bridge_lambda_0_5", "bridge_lambda_0_75")
    assert P61_STEP_COUNTS == (5, 10, 20)
    assert P61_BASELINE_STEP_COUNT == 5
    assert P61_MAX_STEP_COUNT == 20
    assert P61_SEED == 61061
    assert P61_BETA == 0.001
    assert P61_LEARNING_RATE == 1e-4
    assert P61_OPTIMIZER_NAME == "SGD"
    assert P61_EPS_MODE == "zero"
    assert P61_DOMINANCE_SHARE_WARNING_THRESHOLD == 0.40
    assert P61_DOMINANCE_SHARE_BLOCK_THRESHOLD == 0.50
    assert P61_LOSS_EXPLOSION_MULTIPLIER_BLOCK_THRESHOLD == 1.25
    assert P61_KL_EXPLOSION_MULTIPLIER_BLOCK_THRESHOLD == 1.25


def test_p61_02_no_top_level_torch_import():
    filepath = "src/phase2/fc_vae_controlled_step_count_expansion_audit.py"
    with open(filepath, "r", encoding="utf-8") as f:
        code = f.read()
    lines = code.splitlines()
    for line in lines:
        if line.startswith("import torch") or line.startswith("from torch"):
            assert False, f"Forbidden top-level torch import found: {line}"


def test_p61_03_no_forbidden_libs():
    p = pathlib.Path("src/phase2/fc_vae_controlled_step_count_expansion_audit.py").read_text(encoding="utf-8")
    forbidden = ["numpy", "pandas", "scipy", "sklearn"]
    for lib in forbidden:
        assert f"import {lib}" not in p
        assert f"from {lib}" not in p


def test_p61_04_no_p47_p48_imports():
    p = pathlib.Path("src/phase2/fc_vae_controlled_step_count_expansion_audit.py").read_text(encoding="utf-8")
    assert "direct_raw_parameter_fit_smoke" not in p
    assert "direct_raw_parameter_fit_robustness_smoke" not in p


def test_p61_05_optimizer_use_limited():
    p = pathlib.Path("src/phase2/fc_vae_controlled_step_count_expansion_audit.py").read_text(encoding="utf-8")
    # Verify optimizer imports are local inside functions only
    for line in p.splitlines():
        if line.startswith("from torch.optim") or line.startswith("import torch.optim"):
            assert "    " in line, f"Optimizer import must be local: {line}"


def test_p61_06_no_adam_or_scheduler():
    p = pathlib.Path("src/phase2/fc_vae_controlled_step_count_expansion_audit.py").read_text(encoding="utf-8")
    assert "Adam" not in p
    assert "AdamW" not in p
    assert "RMSprop" not in p
    assert "lr_scheduler" not in p
    for line in p.splitlines():
        if "import" in line and "scheduler" in line.lower():
            assert False, f"Forbidden scheduler import: {line}"


def test_p61_07_no_dataset_dataloader_imports():
    p = pathlib.Path("src/phase2/fc_vae_controlled_step_count_expansion_audit.py").read_text(encoding="utf-8")
    assert "DataLoader" not in p
    assert "Dataset" not in p


def test_p61_08_no_epoch_batch_loop_strings():
    p = pathlib.Path("src/phase2/fc_vae_controlled_step_count_expansion_audit.py").read_text(encoding="utf-8")
    assert "for epoch" not in p.lower()
    assert "for batch" not in p.lower()


def test_p61_09_exactly_one_bounded_step_loop():
    p = pathlib.Path("src/phase2/fc_vae_controlled_step_count_expansion_audit.py").read_text(encoding="utf-8")
    matches = re.findall(r"for\s+\w+\s+in\s+range\(step_count\):", p)
    assert len(matches) == 1, "There must be exactly one step loop in P61 source matching the contract."


def test_p61_10_allowed_imports_present():
    p = pathlib.Path("src/phase2/fc_vae_controlled_step_count_expansion_audit.py").read_text(encoding="utf-8")
    assert "load_torch_for_p55_encoder_kl" in p
    assert "build_fc_vae_encoder_posterior_model" in p
    assert "compute_all_gradient_stats" in p
    assert "clear_model_grads" in p
    assert "collect_parameter_snapshot" in p
    assert "compare_parameter_snapshots" in p
    assert "compare_parameter_group_deltas" in p
    assert "build_p58_deterministic_target_set" in p
    assert "compute_p58_micro_training_objective" in p
    assert "compute_p60_target_balance_metrics" in p


def test_p61_11_single_run_rejects_invalid_step_count():
    with pytest.raises(ValueError):
        run_p61_single_step_count(15)


def test_p61_12_single_runs_execute_and_verify():
    run5 = run_p61_single_step_count(5)
    run10 = run_p61_single_step_count(10)
    run20 = run_p61_single_step_count(20)
    
    assert run5["passed"] is True
    assert run10["passed"] is True
    assert run20["passed"] is True
    
    assert run5["completed_step_count"] == 5
    assert run10["completed_step_count"] == 10
    assert run20["completed_step_count"] == 20
    
    assert run5["all_step_losses_finite"] is True
    assert run10["all_step_losses_finite"] is True
    assert run20["all_step_losses_finite"] is True
    
    assert run5["all_step_gradients_finite"] is True
    assert run5["all_step_gradients_present"] is True
    assert run5["all_parameter_deltas_finite"] is True
    assert run5["any_parameter_changed"] is True
    assert run5["all_expected_groups_changed"] is True
    assert run5["final_objective_finite"] is True
    
    assert run5["balance_metrics"]["all_targets_total_loss_decreased"] is True
    assert run5["balance_metrics"]["all_targets_reconstruction_loss_decreased"] is True
    assert run5["balance_metrics"]["dominance_blocker"] is False


def test_p61_13_cross_run_comparison_math():
    run5 = run_p61_single_step_count(5)
    run10 = run_p61_single_step_count(10)
    run20 = run_p61_single_step_count(20)
    
    run_summaries = {
        "steps_5": run5,
        "steps_10": run10,
        "steps_20": run20,
    }
    
    comp = compare_p61_step_count_runs(run_summaries)
    
    assert comp["cross_run_passed"] is True
    assert comp["objective_explosion_blocker_triggered"] is False
    assert comp["kl_explosion_blocker_triggered"] is False
    assert comp["dominance_blocker_across_runs"] is False
    assert comp["all_targets_improved_at_all_step_counts"] is True
    assert len(comp["per_target_improvement_matrix"]) == 3


def test_p61_14_aggregate_probe_pass():
    probe = run_fc_vae_controlled_step_count_expansion_audit_probe()
    assert probe["verdict"] == "PASS"
    assert probe["status"] == "fc_vae_controlled_step_count_expansion_audit_available_no_dataset_no_generalization"
    assert probe["target_count"] == 3
    assert probe["step_counts"] == [5, 10, 20]


def test_p61_15_serialization_excludes_tensors():
    probe = run_fc_vae_controlled_step_count_expansion_audit_probe()
    json_dict = fc_vae_controlled_step_count_expansion_audit_probe_to_json_dict(probe)
    
    for k, v in json_dict.items():
        if isinstance(v, dict):
            for sub_k, sub_v in v.items():
                if isinstance(sub_v, dict):
                    for k3, v3 in sub_v.items():
                        assert not hasattr(v3, "shape")
                elif isinstance(sub_v, list):
                    for item in sub_v:
                        assert not hasattr(item, "shape")
                else:
                    assert not hasattr(sub_v, "shape")
        assert not hasattr(v, "shape"), f"Tensor leaked in JSON: {k}"


def test_p61_16_boundary_flags():
    probe = run_fc_vae_controlled_step_count_expansion_audit_probe()
    assert probe["no_dataset"] is True
    assert probe["no_dataloader"] is True
    assert probe["no_epoch_loop"] is True
    assert probe["no_batch_loop"] is True
    assert probe["no_scheduler"] is True
    assert probe["no_checkpointing"] is True
    assert probe["no_weighted_training"] is True
    assert probe["uniform_objective_preserved"] is True
    
    assert probe["no_generalization_claim"] is True
    assert probe["no_generation_claim"] is True
    assert probe["no_gsb_claim"] is True
    assert probe["no_scientific_conclusion"] is True
    assert probe["no_latent_learning_claim"] is True
    assert probe["no_vae_success_claim"] is True
    assert probe["no_convergence_claim"] is True
    assert probe["no_semantic_geometry_proof_claim"] is True
    assert probe["realizability_claim"] == "controlled_step_count_expansion_audit_only_no_dataset_no_generalization"


def test_p61_17_no_forbidden_claims_in_source():
    p = pathlib.Path("src/phase2/fc_vae_controlled_step_count_expansion_audit.py").read_text(encoding="utf-8")
    p_lower = p.lower()
    
    assert "vae works" not in p_lower
    assert "latent space learned" not in p_lower
    assert "semantic geometry proven" not in p_lower
    assert "c generated" not in p_lower
    assert "gsb implemented" not in p_lower
    assert "posterior collapse solved" not in p_lower
    assert "scientific success" not in p_lower
    assert "vae solved" not in p_lower
    assert "model converged" not in p_lower
    assert "model trained" not in p_lower
    assert "model generalized" not in p_lower
    assert "generalization claim" not in p_lower
    assert "production ready" not in p_lower
    
    rpt = pathlib.Path("reports/PHASE_2_P61_FC_VAE_CONTROLLED_STEP_COUNT_EXPANSION_AUDIT_NO_DATASET_NO_GENERALIZATION_REPORT.md")
    if rpt.exists():
        text = rpt.read_text(encoding="utf-8").lower()
        assert "vae works" not in text
        assert "latent space learned" not in text
        assert "scientific success" not in text
        assert "model converged" not in text
        assert "model generalized" not in text
        assert "vae success" not in text
        assert "production ready" not in text


def test_p61_18_scope_gate():
    from tests.phase2_scope_gate_utils import enforce_phase_local_scope_gate_or_skip
    
    allowed = {
        "src/phase2/fc_vae_controlled_step_count_expansion_audit.py",
        "tools/phase2/run_p61_fc_vae_controlled_step_count_expansion_audit_smoke.py",
        "tests/test_phase2_fc_vae_controlled_step_count_expansion_audit.py",
        "tests/test_phase2_p61_fc_vae_controlled_step_count_expansion_audit_smoke.py",
        "reports/PHASE_2_P61_FC_VAE_CONTROLLED_STEP_COUNT_EXPANSION_AUDIT_NO_DATASET_NO_GENERALIZATION_REPORT.md",
    }
    
    enforce_phase_local_scope_gate_or_skip(
        expected_branch="phase2/p61-fc-vae-controlled-step-count-expansion-audit-no-dataset-no-generalization",
        base_commit="7dcd90d7cf3510979b911eddeeef1fbcd843aa04",
        allowed_files=allowed,
        phase_label="P61",
    )
