# tests/test_phase2_fc_vae_held_out_component_attribution_audit.py

import json
import math
import pathlib
import pytest
import re

from src.phase2.fc_vae_held_out_component_attribution_audit import (
    SOURCE_PHASE,
    CONTRACT_VERSION,
    SOURCE_EVIDENCE_PHASE,
    COMPONENT_RESIDUAL_TOLERANCE,
    TARGET_IDS,
    SEED_VALUES,
    EXPECTED_SEED_FOLD_RUNS,
    BETA,
    run_p64_held_out_component_attribution_audit_probe,
    fc_vae_held_out_component_attribution_audit_probe_to_json_dict,
)


def test_p64_01_constants_exact():
    assert SOURCE_PHASE == "P64"
    assert CONTRACT_VERSION == "phase2_p64_fc_vae_held_out_component_attribution_audit_contract_v1"
    assert SOURCE_EVIDENCE_PHASE == "P63"
    assert COMPONENT_RESIDUAL_TOLERANCE == 1e-5
    assert TARGET_IDS == ["bridge_lambda_0_25", "bridge_lambda_0_5", "bridge_lambda_0_75"]
    assert SEED_VALUES == [62062, 62162, 62262]
    assert EXPECTED_SEED_FOLD_RUNS == 9
    assert BETA == 0.001


def test_p64_02_probe_returns_serializable_pass():
    probe = run_p64_held_out_component_attribution_audit_probe()
    assert probe["verdict"] == "PASS"
    
    json_dict = fc_vae_held_out_component_attribution_audit_probe_to_json_dict(probe)
    serialized = json.dumps(json_dict)
    assert len(serialized) > 0


def test_p64_03_p63_source_evidence_verdict_pass():
    probe = run_p64_held_out_component_attribution_audit_probe()
    assert probe["source_p63_verdict"] == "PASS"
    assert probe["source_p63_status"] == "fc_vae_tiny_held_out_seed_sensitivity_audit_available_no_dataset_no_generalization"


def test_p64_04_runs_count():
    probe = run_p64_held_out_component_attribution_audit_probe()
    assert probe["expected_seed_fold_runs"] == 9
    assert probe["observed_seed_fold_runs"] == 9
    assert len(probe["component_fold_summaries"]) == 9


def test_p64_05_fold_summary_fields_and_finiteness():
    probe = run_p64_held_out_component_attribution_audit_probe()
    for f in probe["component_fold_summaries"]:
        assert f["seed_value"] in SEED_VALUES
        assert f["held_out_target_id"] in TARGET_IDS
        assert len(f["train_target_ids"]) == 2
        
        assert math.isfinite(f["total_delta_value"])
        assert math.isfinite(f["reconstruction_delta_value"])
        assert math.isfinite(f["kl_delta_value"])
        assert math.isfinite(f["beta_weighted_kl_delta_value"])
        assert math.isfinite(f["reconstructed_total_delta_from_components"])
        assert math.isfinite(f["component_residual_value"])
        assert math.isfinite(f["component_residual_abs_value"])
        
        assert isinstance(f["component_residual_within_tolerance"], bool)
        assert f["component_residual_within_tolerance"] is True
        assert f["component_residual_abs_value"] <= COMPONENT_RESIDUAL_TOLERANCE
        
        assert isinstance(f["total_improved"], bool)
        assert isinstance(f["total_degraded"], bool)
        assert isinstance(f["reconstruction_improved"], bool)
        assert isinstance(f["reconstruction_degraded"], bool)
        assert isinstance(f["kl_decreased"], bool)
        assert isinstance(f["kl_increased"], bool)
        assert isinstance(f["beta_weighted_kl_helped_total"], bool)
        assert isinstance(f["beta_weighted_kl_hurt_total"], bool)
        
        assert isinstance(f["total_improvement_reconstruction_supported"], bool)
        assert isinstance(f["total_improvement_kl_only"], bool)
        assert isinstance(f["reconstruction_dominates_total_delta"], bool)
        assert isinstance(f["kl_dominates_total_delta"], bool)
        assert isinstance(f["kl_opposes_reconstruction"], bool)
        
        assert f["component_attribution_class"] in [
            "reconstruction_supported",
            "mixed_reconstruction_supported_kl_opposed",
            "kl_supported",
            "mixed_or_degraded",
        ]
        assert f["component_fold_passed"] is True


def test_p64_06_target_component_diagnostics():
    probe = run_p64_held_out_component_attribution_audit_probe()
    tcd = probe["target_component_diagnostics"]
    
    assert len(tcd) == 3
    for ho_id in TARGET_IDS:
        d = tcd[ho_id]
        assert d["held_out_target_id"] == ho_id
        assert d["seed_count"] == 3
        assert d["run_count"] == 3
        
        assert d["total_improved_count"] + d["mixed_or_degraded_count"] >= 3
        assert len(d["component_attribution_classes_by_seed"]) == 3
        assert len(d["total_delta_values_by_seed"]) == 3
        assert len(d["reconstruction_delta_values_by_seed"]) == 3
        assert len(d["beta_weighted_kl_delta_values_by_seed"]) == 3
        
        assert d["component_residual_max_abs"] <= COMPONENT_RESIDUAL_TOLERANCE
        assert d["all_component_residuals_within_tolerance"] is True
        assert isinstance(d["diagnostic_interpretation"], str)


def test_p64_07_aggregate_diagnostics():
    probe = run_p64_held_out_component_attribution_audit_probe()
    agg = probe["aggregate_diagnostics"]
    
    assert agg["source_p63_evidence_valid"] is True
    assert agg["all_component_folds_passed"] is True
    assert agg["all_component_values_finite"] is True
    assert agg["all_component_residuals_within_tolerance"] is True
    assert agg["max_component_residual_abs"] <= COMPONENT_RESIDUAL_TOLERANCE
    
    total_class_sum = (
        agg["reconstruction_supported_count"]
        + agg["kl_only_count"]
        + agg["mixed_or_degraded_count"]
    )
    assert total_class_sum == 9
    
    assert isinstance(agg["targets_reconstruction_supported_all_seeds"], list)
    assert isinstance(agg["targets_with_kl_only_any_seed"], list)
    assert isinstance(agg["targets_with_kl_opposition_any_seed"], list)
    
    ho75 = agg["bridge_lambda_0_75_component_summary"]
    assert ho75["held_out_target_id"] == "bridge_lambda_0_75"
    
    assert agg["component_attribution_claim"] == "held_out_component_attribution_audit_only_no_new_optimization_no_dataset_no_generalization"
    assert agg["diagnostic_only_no_generalization"] is True


def test_p64_08_boundary_flags():
    probe = run_p64_held_out_component_attribution_audit_probe()
    assert probe["no_new_optimization"] is True
    assert probe["no_direct_optimizer_created"] is True
    assert probe["no_direct_model_created"] is True
    assert probe["no_direct_torch_import"] is True
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
    assert probe["no_component_proof_claim"] is True
    
    assert probe["no_weighted_training"] is True
    assert probe["uniform_objective_preserved"] is True
    assert probe["held_out_diagnostic_only"] is True
    assert probe["component_attribution_diagnostic_only"] is True


def test_p64_09_no_tensors_in_output():
    probe = run_p64_held_out_component_attribution_audit_probe()
    serialized = json.dumps(probe)
    # Check that "tensor" string doesn't appear in value serialization
    assert "tensor" not in serialized.lower()


def test_p64_10_no_torch_or_optimizers_imported():
    p = pathlib.Path("src/phase2/fc_vae_held_out_component_attribution_audit.py").read_text(encoding="utf-8")
    assert "import torch" not in p
    assert "from torch" not in p
    for line in p.splitlines():
        trimmed = line.strip()
        if trimmed.startswith("import ") or trimmed.startswith("from "):
            if "optim" in trimmed or "torch" in trimmed:
                assert False, f"Forbidden import found: {line}"


def test_p64_11_forbidden_optimization_and_model_markers_absent():
    p = pathlib.Path("src/phase2/fc_vae_held_out_component_attribution_audit.py").read_text(encoding="utf-8")
    
    # We construct forbidden patterns using split string concatenation so this test file doesn't fail
    forbidden_patterns = [
        "optimizer" + ".step",
        "loss" + ".backward",
        "for step_idx " + "in range",
        "build_fc_vae_encoder_" + "posterior_model",
        "from torch" + ".optim",
        "SG" + "D(",
        "for epoch" + "",
        "for batch" + "",
    ]
    for pattern in forbidden_patterns:
        assert pattern not in p, f"Forbidden direct optimization/model marker found: {pattern}"


def test_p64_12_no_forbidden_libs():
    p = pathlib.Path("src/phase2/fc_vae_held_out_component_attribution_audit.py").read_text(encoding="utf-8")
    forbidden = ["numpy", "pandas", "scipy", "sklearn"]
    for lib in forbidden:
        assert f"import {lib}" not in p
        assert f"from {lib}" not in p


def test_p64_13_no_p47_p48_imports():
    p = pathlib.Path("src/phase2/fc_vae_held_out_component_attribution_audit.py").read_text(encoding="utf-8")
    assert "direct_raw_parameter_fit_smoke" not in p
    assert "direct_raw_parameter_fit_robustness_smoke" not in p


def test_p64_14_forbidden_wordings_check():
    forbidden = [
        "vae " + "works", "latent " + "space learned", "semantic " + "geometry proven",
        "c " + "generated", "gsb " + "implemented", "model " + "generalized",
        "production " + "ready", "held-out " + "generalization", "transfer " + "proven",
        "model " + "generalizes to held-out target", "proof " + "of transfer",
        "seed-robust " + "generalization", "component " + "proof",
    ]
    p_src = pathlib.Path("src/phase2/fc_vae_held_out_component_attribution_audit.py").read_text(encoding="utf-8").lower()
    for item in forbidden:
        assert item not in p_src


def test_p64_15_scope_gate():
    from tests.phase2_scope_gate_utils import enforce_phase_local_scope_gate_or_skip
    
    allowed = {
        "src/phase2/fc_vae_held_out_component_attribution_audit.py",
        "tools/phase2/run_p64_fc_vae_held_out_component_attribution_audit_smoke.py",
        "tests/test_phase2_fc_vae_held_out_component_attribution_audit.py",
        "tests/test_phase2_p64_fc_vae_held_out_component_attribution_audit_smoke.py",
        "reports/PHASE_2_P64_FC_VAE_HELD_OUT_COMPONENT_ATTRIBUTION_AUDIT_NO_NEW_OPTIMIZATION_NO_DATASET_NO_GENERALIZATION_REPORT.md",
    }
    
    enforce_phase_local_scope_gate_or_skip(
        expected_branch="phase2/p64-fc-vae-held-out-component-attribution-audit-no-new-optimization-no-dataset-no-generalization",
        base_commit="ac660bd3b6a4d2639021d90538fe756b33575a5e",
        allowed_files=allowed,
        phase_label="P64",
    )
