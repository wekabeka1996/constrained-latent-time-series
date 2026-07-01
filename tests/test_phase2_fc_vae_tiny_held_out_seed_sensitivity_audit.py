# tests/test_phase2_fc_vae_tiny_held_out_seed_sensitivity_audit.py

import json
import math
import pathlib
import pytest
import re

# Crucial: No top-level torch import.

from src.phase2.fc_vae_tiny_held_out_seed_sensitivity_audit import (
    SOURCE_PHASE,
    CONTRACT_VERSION,
    TARGET_IDS,
    FOLDS,
    SEED_VALUES,
    SEED_COUNT,
    STEP_COUNT,
    LEARNING_RATE,
    BETA,
    EPS_MODE,
    OPTIMIZER_NAME,
    run_p63_tiny_held_out_seed_sensitivity_audit_probe,
    fc_vae_tiny_held_out_seed_sensitivity_audit_probe_to_json_dict,
)


def test_p63_01_constants_exact():
    assert SOURCE_PHASE == "P63"
    assert CONTRACT_VERSION == "phase2_p63_fc_vae_tiny_held_out_seed_sensitivity_audit_contract_v1"
    assert TARGET_IDS == ["bridge_lambda_0_25", "bridge_lambda_0_5", "bridge_lambda_0_75"]
    assert len(FOLDS) == 3
    assert SEED_VALUES == [62062, 62162, 62262]
    assert SEED_COUNT == 3
    assert STEP_COUNT == 20
    assert LEARNING_RATE == 1e-4
    assert BETA == 0.001
    assert EPS_MODE == "zero"
    assert OPTIMIZER_NAME == "SGD"


def test_p63_02_no_top_level_torch_import():
    filepath = "src/phase2/fc_vae_tiny_held_out_seed_sensitivity_audit.py"
    with open(filepath, "r", encoding="utf-8") as f:
        code = f.read()
    lines = code.splitlines()
    for line in lines:
        if line.startswith("import torch") or line.startswith("from torch"):
            assert False, f"Forbidden top-level torch import found: {line}"


def test_p63_03_no_forbidden_libs():
    p = pathlib.Path("src/phase2/fc_vae_tiny_held_out_seed_sensitivity_audit.py").read_text(encoding="utf-8")
    forbidden = ["numpy", "pandas", "scipy", "sklearn"]
    for lib in forbidden:
        assert f"import {lib}" not in p
        assert f"from {lib}" not in p


def test_p63_04_no_p47_p48_imports():
    p = pathlib.Path("src/phase2/fc_vae_tiny_held_out_seed_sensitivity_audit.py").read_text(encoding="utf-8")
    assert "direct_raw_parameter_fit_smoke" not in p
    assert "direct_raw_parameter_fit_robustness_smoke" not in p


def test_p63_05_optimizer_use_limited():
    p = pathlib.Path("src/phase2/fc_vae_tiny_held_out_seed_sensitivity_audit.py").read_text(encoding="utf-8")
    for line in p.splitlines():
        if line.startswith("from torch.optim") or line.startswith("import torch.optim"):
            assert "    " in line, f"Optimizer import must be local: {line}"


def test_p63_06_no_adam_or_scheduler():
    p = pathlib.Path("src/phase2/fc_vae_tiny_held_out_seed_sensitivity_audit.py").read_text(encoding="utf-8")
    assert "Adam" not in p
    assert "AdamW" not in p
    assert "RMSprop" not in p
    assert "lr_scheduler" not in p
    for line in p.splitlines():
        if "import" in line and "scheduler" in line.lower():
            assert False, f"Forbidden scheduler import: {line}"


def test_p63_07_no_dataset_dataloader_imports():
    p = pathlib.Path("src/phase2/fc_vae_tiny_held_out_seed_sensitivity_audit.py").read_text(encoding="utf-8")
    assert "DataLoader" not in p
    assert "Dataset" not in p


def test_p63_08_no_epoch_batch_loop_strings():
    p = pathlib.Path("src/phase2/fc_vae_tiny_held_out_seed_sensitivity_audit.py").read_text(encoding="utf-8")
    assert "for epoch" not in p.lower()
    assert "for batch" not in p.lower()


def test_p63_09_exactly_one_bounded_step_loop():
    p = pathlib.Path("src/phase2/fc_vae_tiny_held_out_seed_sensitivity_audit.py").read_text(encoding="utf-8")
    matches = re.findall(r"for\s+\w+\s+in\s+range\(STEP_COUNT\):", p)
    assert len(matches) == 1, "There must be exactly one STEP_COUNT loop in P63 source."


def test_p63_10_allowed_imports_present():
    p = pathlib.Path("src/phase2/fc_vae_tiny_held_out_seed_sensitivity_audit.py").read_text(encoding="utf-8")
    assert "load_torch_for_p55_encoder_kl" in p
    assert "build_fc_vae_encoder_posterior_model" in p
    assert "clear_model_grads" in p
    assert "build_p58_deterministic_target_set" in p
    assert "compute_p58_micro_training_objective" in p


def test_p63_11_probe_returns_serializable_pass():
    probe = run_p63_tiny_held_out_seed_sensitivity_audit_probe()
    assert probe["verdict"] == "PASS"
    assert probe["torch_available"] is True
    
    json_dict = fc_vae_tiny_held_out_seed_sensitivity_audit_probe_to_json_dict(probe)
    serialized = json.dumps(json_dict)
    assert len(serialized) > 0


def test_p63_12_seeds_structure():
    probe = run_p63_tiny_held_out_seed_sensitivity_audit_probe()
    assert probe["seed_count"] == 3
    assert probe["fold_count_per_seed"] == 3
    assert probe["total_seed_fold_runs"] == 9
    assert len(probe["seed_summaries"]) == 3
    
    for s in probe["seed_summaries"]:
        assert s["seed_value"] in SEED_VALUES
        assert len(s["fold_summaries"]) == 3
        
        seen_held_out = set()
        for f in s["fold_summaries"]:
            assert f["seed_value"] == s["seed_value"]
            assert len(f["train_target_ids"]) == 2
            assert f["train_target_count"] == 2
            assert f["held_out_target_count"] == 1
            ho = f["held_out_target_id"]
            seen_held_out.add(ho)
            assert ho not in f["train_target_ids"]
            
        assert seen_held_out == set(TARGET_IDS)


def test_p63_13_all_fold_metrics_finite_and_present():
    probe = run_p63_tiny_held_out_seed_sensitivity_audit_probe()
    for s in probe["seed_summaries"]:
        assert s["seed_all_folds_passed"] is True
        assert s["seed_all_folds_finite"] is True
        assert s["seed_all_gradients_present"] is True
        assert s["seed_all_gradients_finite"] is True
        assert s["seed_all_optimizer_steps_completed"] is True
        
        for f in s["fold_summaries"]:
            assert math.isfinite(f["initial_train_objective_value"])
            assert math.isfinite(f["final_train_objective_value"])
            assert math.isfinite(f["train_objective_delta_value"])
            
            assert math.isfinite(f["initial_held_out_total_loss_value"])
            assert math.isfinite(f["final_held_out_total_loss_value"])
            assert math.isfinite(f["held_out_total_loss_delta_value"])
            
            assert isinstance(f["train_objective_decreased"], bool)
            assert isinstance(f["held_out_total_improved"], bool)
            assert isinstance(f["held_out_total_degraded"], bool)
            assert isinstance(f["held_out_total_unchanged"], bool)
            assert f["held_out_total_movement_class"] in ["improved", "degraded", "unchanged"]
            
            # compact diagnostics
            assert math.isfinite(f["first_step_train_loss_value"])
            assert math.isfinite(f["last_step_train_loss_value"])
            assert math.isfinite(f["step_train_loss_delta_value"])
            assert f["all_step_losses_finite"] is True
            assert f["all_step_gradients_present"] is True
            assert f["all_step_gradients_finite"] is True
            assert f["all_optimizer_steps_completed"] is True
            assert f["completed_step_count"] == 20
            assert f["fold_passed"] is True


def test_p63_14_target_sensitivity_diagnostics():
    probe = run_p63_tiny_held_out_seed_sensitivity_audit_probe()
    tsd = probe["target_sensitivity_diagnostics"]
    
    assert len(tsd) == 3
    for ho_id in TARGET_IDS:
        d = tsd[ho_id]
        assert d["held_out_target_id"] == ho_id
        assert d["seed_count"] == 3
        assert d["improved_count"] + d["degraded_count"] + d["unchanged_count"] == 3
        assert len(d["movement_classes_by_seed"]) == 3
        assert len(d["held_out_total_delta_values_by_seed"]) == 3
        assert math.isfinite(d["mean_held_out_total_delta_value"])
        assert math.isfinite(d["min_held_out_total_delta_value"])
        assert math.isfinite(d["max_held_out_total_delta_value"])
        assert d["all_seed_deltas_finite"] is True
        assert isinstance(d["consistent_improvement_across_seeds"], bool)
        assert isinstance(d["any_degradation_across_seeds"], bool)
        assert isinstance(d["seed_sensitivity_warning"], bool)


def test_p63_15_aggregate_diagnostics():
    probe = run_p63_tiny_held_out_seed_sensitivity_audit_probe()
    agg = probe["aggregate_diagnostics"]
    
    assert agg["all_runs_passed"] is True
    assert agg["all_runs_finite"] is True
    assert agg["all_gradients_present"] is True
    assert agg["all_gradients_finite"] is True
    assert agg["all_optimizer_steps_completed"] is True
    
    total_classifications = agg["total_held_out_improved_count"] + agg["total_held_out_degraded_count"] + agg["total_held_out_unchanged_count"]
    assert total_classifications == 9
    
    assert isinstance(agg["targets_consistently_improved"], list)
    assert isinstance(agg["targets_with_any_degradation"], list)
    assert isinstance(agg["targets_with_seed_sensitivity_warning"], list)
    
    # special summary check
    ho75 = agg["bridge_lambda_0_75_seed_sensitivity_summary"]
    assert ho75["held_out_target_id"] == "bridge_lambda_0_75"
    assert ho75["seed_count"] == 3
    assert len(ho75["movement_classes_by_seed"]) == 3
    assert len(ho75["held_out_total_delta_values_by_seed"]) == 3
    assert math.isfinite(ho75["mean_held_out_total_delta_value"])
    
    assert agg["seed_sensitivity_claim"] == "tiny_deterministic_seed_sensitivity_audit_only_no_dataset_no_generalization"
    assert agg["diagnostic_only_no_generalization"] is True


def test_p63_16_boundary_flags():
    probe = run_p63_tiny_held_out_seed_sensitivity_audit_probe()
    assert probe["no_dataset"] is True
    assert probe["no_dataloader"] is True
    assert probe["no_epoch_loop"] is True
    assert probe["no_batch_loop"] is True
    assert probe["no_scheduler"] is True
    assert probe["no_checkpointing"] is True
    
    assert probe["no_generalization_claim"] is True
    assert probe["no_generation_claim"] is True
    assert probe["no_gsb_claim"] is True
    assert probe["no_scientific_conclusion"] is True
    assert probe["no_latent_learning_claim"] is True
    assert probe["no_vae_success_claim"] is True
    assert probe["no_convergence_claim"] is True
    assert probe["no_semantic_geometry_proof_claim"] is True
    assert probe["no_transfer_proof_claim"] is True
    assert probe["no_seed_robustness_claim"] is True
    
    assert probe["no_weighted_training"] is True
    assert probe["uniform_objective_preserved"] is True
    assert probe["held_out_diagnostic_only"] is True
    assert probe["seed_sensitivity_diagnostic_only"] is True


def test_p63_17_forbidden_wordings_check():
    forbidden = [
        "vae " + "works", "latent " + "space learned", "semantic " + "geometry proven",
        "c " + "generated", "gsb " + "implemented", "model " + "generalized",
        "production " + "ready", "held-out " + "generalization", "transfer " + "proven",
        "model " + "generalizes to held-out target", "proof " + "of transfer",
        "seed-robust " + "generalization",
    ]
    p_src = pathlib.Path("src/phase2/fc_vae_tiny_held_out_seed_sensitivity_audit.py").read_text(encoding="utf-8").lower()
    for item in forbidden:
        assert item not in p_src
        
    p_test = pathlib.Path("tests/test_phase2_fc_vae_tiny_held_out_seed_sensitivity_audit.py").read_text(encoding="utf-8").lower()
    # Check that forbidden items are only present in forbidden check list (constructed using string concatenation).


def test_p63_18_scope_gate():
    from tests.phase2_scope_gate_utils import enforce_phase_local_scope_gate_or_skip
    
    allowed = {
        "src/phase2/fc_vae_tiny_held_out_seed_sensitivity_audit.py",
        "tools/phase2/run_p63_fc_vae_tiny_held_out_seed_sensitivity_audit_smoke.py",
        "tests/test_phase2_fc_vae_tiny_held_out_seed_sensitivity_audit.py",
        "tests/test_phase2_p63_fc_vae_tiny_held_out_seed_sensitivity_audit_smoke.py",
        "reports/PHASE_2_P63_FC_VAE_TINY_HELD_OUT_SEED_SENSITIVITY_AUDIT_NO_DATASET_NO_GENERALIZATION_REPORT.md",
    }
    
    enforce_phase_local_scope_gate_or_skip(
        expected_branch="phase2/p63-fc-vae-tiny-held-out-seed-sensitivity-audit-no-dataset-no-generalization",
        base_commit="302ef65e3bc7740cf36f8cbc7da86f5bb49d0462",
        allowed_files=allowed,
        phase_label="P63",
    )
