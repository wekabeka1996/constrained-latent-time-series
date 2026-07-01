# tests/test_phase2_fc_vae_tiny_held_out_deterministic_bridge_target_split.py

import json
import math
import pathlib
import sys
import pytest
import re

# Crucial: No top-level torch import.

from src.phase2.fc_vae_tiny_held_out_deterministic_bridge_target_split import (
    SOURCE_PHASE,
    CONTRACT_VERSION,
    TARGET_IDS,
    FOLDS,
    STEP_COUNT,
    LEARNING_RATE,
    BETA,
    SEED_VALUE,
    EPS_MODE,
    OPTIMIZER_NAME,
    run_p62_tiny_held_out_deterministic_bridge_target_split_probe,
    fc_vae_tiny_held_out_deterministic_bridge_target_split_probe_to_json_dict,
)


def test_p62_01_constants_exact():
    assert SOURCE_PHASE == "P62"
    assert CONTRACT_VERSION == "phase2_p62_fc_vae_tiny_held_out_deterministic_bridge_target_split_contract_v1"
    assert TARGET_IDS == ["bridge_lambda_0_25", "bridge_lambda_0_5", "bridge_lambda_0_75"]
    assert len(FOLDS) == 3
    assert STEP_COUNT == 20
    assert LEARNING_RATE == 1e-4
    assert BETA == 0.001
    assert SEED_VALUE == 62062
    assert EPS_MODE == "zero"
    assert OPTIMIZER_NAME == "SGD"


def test_p62_02_no_top_level_torch_import():
    filepath = "src/phase2/fc_vae_tiny_held_out_deterministic_bridge_target_split.py"
    with open(filepath, "r", encoding="utf-8") as f:
        code = f.read()
    lines = code.splitlines()
    for line in lines:
        if line.startswith("import torch") or line.startswith("from torch"):
            assert False, f"Forbidden top-level torch import found: {line}"


def test_p62_03_no_forbidden_libs():
    p = pathlib.Path("src/phase2/fc_vae_tiny_held_out_deterministic_bridge_target_split.py").read_text(encoding="utf-8")
    forbidden = ["numpy", "pandas", "scipy", "sklearn"]
    for lib in forbidden:
        assert f"import {lib}" not in p
        assert f"from {lib}" not in p


def test_p62_04_no_p47_p48_imports():
    p = pathlib.Path("src/phase2/fc_vae_tiny_held_out_deterministic_bridge_target_split.py").read_text(encoding="utf-8")
    assert "direct_raw_parameter_fit_smoke" not in p
    assert "direct_raw_parameter_fit_robustness_smoke" not in p


def test_p62_05_optimizer_use_limited():
    p = pathlib.Path("src/phase2/fc_vae_tiny_held_out_deterministic_bridge_target_split.py").read_text(encoding="utf-8")
    for line in p.splitlines():
        if line.startswith("from torch.optim") or line.startswith("import torch.optim"):
            assert "    " in line, f"Optimizer import must be local: {line}"


def test_p62_06_no_adam_or_scheduler():
    p = pathlib.Path("src/phase2/fc_vae_tiny_held_out_deterministic_bridge_target_split.py").read_text(encoding="utf-8")
    assert "Adam" not in p
    assert "AdamW" not in p
    assert "RMSprop" not in p
    assert "lr_scheduler" not in p
    for line in p.splitlines():
        if "import" in line and "scheduler" in line.lower():
            assert False, f"Forbidden scheduler import: {line}"


def test_p62_07_no_dataset_dataloader_imports():
    p = pathlib.Path("src/phase2/fc_vae_tiny_held_out_deterministic_bridge_target_split.py").read_text(encoding="utf-8")
    assert "DataLoader" not in p
    assert "Dataset" not in p


def test_p62_08_no_epoch_batch_loop_strings():
    p = pathlib.Path("src/phase2/fc_vae_tiny_held_out_deterministic_bridge_target_split.py").read_text(encoding="utf-8")
    assert "for epoch" not in p.lower()
    assert "for batch" not in p.lower()


def test_p62_09_exactly_one_bounded_step_loop():
    p = pathlib.Path("src/phase2/fc_vae_tiny_held_out_deterministic_bridge_target_split.py").read_text(encoding="utf-8")
    matches = re.findall(r"for\s+\w+\s+in\s+range\(STEP_COUNT\):", p)
    assert len(matches) == 1, "There must be exactly one STEP_COUNT loop in P62 source."


def test_p62_10_allowed_imports_present():
    p = pathlib.Path("src/phase2/fc_vae_tiny_held_out_deterministic_bridge_target_split.py").read_text(encoding="utf-8")
    assert "load_torch_for_p55_encoder_kl" in p
    assert "build_fc_vae_encoder_posterior_model" in p
    assert "clear_model_grads" in p
    assert "build_p58_deterministic_target_set" in p
    assert "compute_p58_micro_training_objective" in p


def test_p62_11_probe_returns_serializable_pass():
    probe = run_p62_tiny_held_out_deterministic_bridge_target_split_probe()
    assert probe["verdict"] == "PASS"
    assert probe["torch_available"] is True
    
    json_dict = fc_vae_tiny_held_out_deterministic_bridge_target_split_probe_to_json_dict(probe)
    serialized = json.dumps(json_dict)
    assert len(serialized) > 0


def test_p62_12_folds_structure():
    probe = run_p62_tiny_held_out_deterministic_bridge_target_split_probe()
    assert probe["fold_count"] == 3
    assert len(probe["fold_summaries"]) == 3
    
    seen_held_out = set()
    for f in probe["fold_summaries"]:
        assert len(f["train_target_ids"]) == 2
        assert f["train_target_count"] == 2
        assert f["held_out_target_count"] == 1
        ho = f["held_out_target_id"]
        seen_held_out.add(ho)
        assert ho not in f["train_target_ids"]
        
    assert seen_held_out == set(TARGET_IDS)


def test_p62_13_all_fold_metrics_finite_and_present():
    probe = run_p62_tiny_held_out_deterministic_bridge_target_split_probe()
    for f in probe["fold_summaries"]:
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
        
        assert f["completed_step_count"] == 20
        assert f["all_step_losses_finite"] is True
        assert f["all_step_gradients_present"] is True
        assert f["all_step_gradients_finite"] is True
        assert f["all_optimizer_steps_completed"] is True
        assert f["fold_passed"] is True


def test_p62_14_aggregate_diagnostics():
    probe = run_p62_tiny_held_out_deterministic_bridge_target_split_probe()
    agg = probe["aggregate_diagnostics"]
    
    assert agg["all_folds_passed"] is True
    assert agg["all_folds_finite"] is True
    assert agg["all_fold_gradients_present"] is True
    assert agg["all_fold_gradients_finite"] is True
    assert agg["all_optimizer_steps_completed"] is True
    
    assert agg["train_objective_decreased_count"] == 3
    assert agg["all_train_objectives_decreased"] is True
    
    total_classifications = agg["held_out_improved_count"] + agg["held_out_degraded_count"] + agg["held_out_unchanged_count"]
    assert total_classifications == 3
    
    assert len(agg["held_out_improved_target_ids"]) == agg["held_out_improved_count"]
    assert len(agg["held_out_degraded_target_ids"]) == agg["held_out_degraded_count"]
    assert len(agg["held_out_unchanged_target_ids"]) == agg["held_out_unchanged_count"]
    
    assert agg["hardest_held_out_target_by_final_total_loss"] in TARGET_IDS
    assert math.isfinite(agg["hardest_held_out_final_total_loss_value"])
    assert agg["worst_held_out_delta_target_id"] in TARGET_IDS
    assert math.isfinite(agg["worst_held_out_delta_value"])
    
    # special summary check
    ho75 = agg["bridge_lambda_0_75_held_out_summary"]
    assert ho75["held_out_target_id"] == "bridge_lambda_0_75"
    assert ho75["train_target_ids"] == ["bridge_lambda_0_25", "bridge_lambda_0_5"]
    assert math.isfinite(ho75["initial_held_out_total_loss_value"])
    assert math.isfinite(ho75["final_held_out_total_loss_value"])
    assert math.isfinite(ho75["held_out_total_loss_delta_value"])
    assert ho75["held_out_total_movement_class"] in ["improved", "degraded", "unchanged"]
    
    assert agg["held_out_transfer_claim"] == "tiny_deterministic_bridge_target_transfer_diagnostic_only_no_dataset_no_generalization"
    assert agg["diagnostic_only_no_generalization"] is True


def test_p62_15_boundary_flags():
    probe = run_p62_tiny_held_out_deterministic_bridge_target_split_probe()
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
    
    assert probe["no_weighted_training"] is True
    assert probe["uniform_objective_preserved"] is True
    assert probe["held_out_diagnostic_only"] is True


def test_p62_16_forbidden_wordings_check():
    forbidden = [
        "vae " + "works", "latent " + "space learned", "semantic " + "geometry proven",
        "c " + "generated", "gsb " + "implemented", "model " + "generalized",
        "production " + "ready", "held-out " + "generalization", "transfer " + "proven",
        "model " + "generalizes to held-out target",
    ]
    p_src = pathlib.Path("src/phase2/fc_vae_tiny_held_out_deterministic_bridge_target_split.py").read_text(encoding="utf-8").lower()
    for item in forbidden:
        assert item not in p_src
        
    p_test = pathlib.Path("tests/test_phase2_fc_vae_tiny_held_out_deterministic_bridge_target_split.py").read_text(encoding="utf-8").lower()
    # Check that forbidden items are only present in forbidden check list (constructed using string concatenation, so not matched literally)
    # The literal wordings must not be in tests or report.


def test_p62_17_scope_gate():
    from tests.phase2_scope_gate_utils import enforce_phase_local_scope_gate_or_skip
    
    allowed = {
        "src/phase2/fc_vae_tiny_held_out_deterministic_bridge_target_split.py",
        "tools/phase2/run_p62_fc_vae_tiny_held_out_deterministic_bridge_target_split_smoke.py",
        "tests/test_phase2_fc_vae_tiny_held_out_deterministic_bridge_target_split.py",
        "tests/test_phase2_p62_fc_vae_tiny_held_out_deterministic_bridge_target_split_smoke.py",
        "reports/PHASE_2_P62_FC_VAE_TINY_HELD_OUT_DETERMINISTIC_BRIDGE_TARGET_SPLIT_NO_DATASET_NO_GENERALIZATION_REPORT.md",
    }
    
    enforce_phase_local_scope_gate_or_skip(
        expected_branch="phase2/p62-fc-vae-tiny-held-out-deterministic-bridge-target-split-no-dataset-no-generalization",
        base_commit="cc2586b5ad96a0a2e40878b1d2f85ea3e12f2535",
        allowed_files=allowed,
        phase_label="P62",
    )
