# tests/test_phase2_fc_vae_held_out_label_permutation_control_audit.py

import json
import math
import pathlib
import pytest
import re

from src.phase2.fc_vae_held_out_label_permutation_control_audit import (
    SOURCE_PHASE,
    CONTRACT_VERSION,
    SOURCE_EVIDENCE_PHASE,
    TARGET_IDS,
    SEED_VALUES,
    EXPECTED_SOURCE_COMPONENT_RUNS,
    PERMUTATION_IDS,
    NUMERIC_COMPARISON_TOLERANCE,
    run_p65_held_out_label_permutation_control_audit_probe,
    fc_vae_held_out_label_permutation_control_audit_probe_to_json_dict,
)


def test_p65_01_constants_exact():
    assert SOURCE_PHASE == "P65"
    assert CONTRACT_VERSION == "phase2_p65_fc_vae_held_out_label_permutation_control_audit_contract_v1"
    assert SOURCE_EVIDENCE_PHASE == "P64"
    assert TARGET_IDS == ["bridge_lambda_0_25", "bridge_lambda_0_5", "bridge_lambda_0_75"]
    assert SEED_VALUES == [62062, 62162, 62262]
    assert EXPECTED_SOURCE_COMPONENT_RUNS == 9
    assert PERMUTATION_IDS == ["cyclic_forward", "cyclic_backward"]
    assert NUMERIC_COMPARISON_TOLERANCE == 1e-12


def test_p65_02_probe_returns_serializable_pass():
    probe = run_p65_held_out_label_permutation_control_audit_probe()
    assert probe["verdict"] == "PASS"
    
    json_dict = fc_vae_held_out_label_permutation_control_audit_probe_to_json_dict(probe)
    serialized = json.dumps(json_dict)
    assert len(serialized) > 0


def test_p65_03_p64_source_evidence_verdict_pass():
    probe = run_p65_held_out_label_permutation_control_audit_probe()
    assert probe["source_p64_verdict"] == "PASS"
    assert probe["source_p64_status"] == "fc_vae_held_out_component_attribution_audit_available_no_dataset_no_generalization"


def test_p65_04_runs_count():
    probe = run_p65_held_out_label_permutation_control_audit_probe()
    assert probe["expected_source_component_runs"] == 9
    assert probe["observed_source_component_runs"] == 9


def test_p65_05_permutation_ids_and_maps():
    probe = run_p65_held_out_label_permutation_control_audit_probe()
    assert probe["permutation_ids"] == PERMUTATION_IDS
    
    for pid in PERMUTATION_IDS:
        p_sum = probe["permutation_summaries"][pid]
        assert p_sum["permutation_id"] == pid
        assert p_sum["mapping_is_bijective"] is True
        assert p_sum["mapping_has_no_fixed_points"] is True
        assert p_sum["control_record_count"] == 9
        assert p_sum["label_assignment_changed_count"] == 9
        assert p_sum["permutation_passed"] is True


def test_p65_06_target_summaries():
    probe = run_p65_held_out_label_permutation_control_audit_probe()
    
    # Original target summary contains exactly 3 targets
    assert len(probe["original_target_summary"]) == 3
    for tid in TARGET_IDS:
        orig = probe["original_target_summary"][tid]
        assert orig["held_out_target_id"] == tid
        assert orig["run_count"] == 3
        assert math.isfinite(orig["mean_total_delta_value"])
        assert math.isfinite(orig["mean_reconstruction_delta_value"])
        assert math.isfinite(orig["mean_beta_weighted_kl_delta_value"])
        
    for pid in PERMUTATION_IDS:
        p_sum = probe["permutation_summaries"][pid]
        assert len(p_sum["permuted_target_summaries"]) == 3
        assert len(p_sum["comparison_to_original"]) == 3
        
        for tid in TARGET_IDS:
            perm = p_sum["permuted_target_summaries"][tid]
            assert perm["held_out_target_id"] == tid
            assert perm["run_count"] == 3
            assert math.isfinite(perm["mean_total_delta_value"])
            assert math.isfinite(perm["mean_reconstruction_delta_value"])
            assert math.isfinite(perm["mean_beta_weighted_kl_delta_value"])
            
            comp = p_sum["comparison_to_original"][tid]
            assert comp["target_id"] == tid
            assert math.isfinite(comp["mean_total_delta_abs_diff"])
            assert math.isfinite(comp["mean_reconstruction_delta_abs_diff"])
            assert isinstance(comp["numeric_profile_changed"], bool)
            assert isinstance(comp["qualitative_profile_changed"], bool)


def test_p65_07_aggregate_diagnostics():
    probe = run_p65_held_out_label_permutation_control_audit_probe()
    agg = probe["aggregate_diagnostics"]
    
    assert agg["source_p64_evidence_valid"] is True
    assert agg["source_component_runs_count"] == 9
    assert agg["all_permutation_maps_bijective"] is True
    assert agg["all_permutation_maps_have_no_fixed_points"] is True
    assert agg["all_permutation_records_present"] is True
    assert agg["all_labels_changed_under_permutation"] is True
    assert agg["all_permutation_summaries_finite"] is True
    
    assert isinstance(agg["original_all_targets_reconstruction_supported"], bool)
    assert isinstance(agg["permutation_all_targets_reconstruction_supported"], bool)
    assert isinstance(agg["qualitative_all_improved_signal_label_invariant"], bool)
    assert isinstance(agg["qualitative_reconstruction_supported_signal_label_invariant"], bool)
    assert isinstance(agg["numeric_target_profiles_changed_under_permutation"], bool)
    
    assert isinstance(agg["targets_with_numeric_profile_change"], list)
    assert isinstance(agg["targets_with_qualitative_profile_change"], list)
    assert isinstance(agg["label_specificity_warning"], bool)
    assert isinstance(agg["negative_control_interpretation"], str)
    assert agg["label_permutation_control_claim"] == "label_permutation_control_audit_only_no_new_optimization_no_dataset_no_generalization"
    assert agg["diagnostic_only_no_generalization"] is True


def test_p65_08_boundary_flags():
    probe = run_p65_held_out_label_permutation_control_audit_probe()
    assert probe["no_new_optimization"] is True
    assert probe["no_direct_optimizer_created"] is True
    assert probe["no_direct_model_created"] is True
    assert probe["no_direct_torch_import"] is True
    assert probe["no_direct_p63_import"] is True
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
    assert probe["no_label_specific_semantic_proof_claim"] is True
    
    assert probe["no_weighted_training"] is True
    assert probe["uniform_objective_preserved"] is True
    assert probe["held_out_diagnostic_only"] is True
    assert probe["label_permutation_control_diagnostic_only"] is True


def test_p65_09_no_tensors_in_output():
    probe = run_p65_held_out_label_permutation_control_audit_probe()
    serialized = json.dumps(probe)
    assert "tensor" not in serialized.lower()


def test_p65_10_no_forbidden_imports():
    p = pathlib.Path("src/phase2/fc_vae_held_out_label_permutation_control_audit.py").read_text(encoding="utf-8")
    assert "import torch" not in p
    assert "from torch" not in p
    assert "import numpy" not in p
    assert "from numpy" not in p
    assert "import pandas" not in p
    assert "import scipy" not in p
    assert "import sklearn" not in p
    assert "import p63" not in p
    assert "import P63" not in p
    assert "fc_vae_tiny_held_out_seed_sensitivity_audit" not in p
    
    for line in p.splitlines():
        trimmed = line.strip()
        if trimmed.startswith("import ") or trimmed.startswith("from "):
            if "optim" in trimmed or "torch" in trimmed or "p63" in trimmed:
                assert False, f"Forbidden import found: {line}"


def test_p65_11_forbidden_optimization_and_model_markers_absent():
    p = pathlib.Path("src/phase2/fc_vae_held_out_label_permutation_control_audit.py").read_text(encoding="utf-8")
    
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


def test_p65_12_no_p47_p48_imports():
    p = pathlib.Path("src/phase2/fc_vae_held_out_label_permutation_control_audit.py").read_text(encoding="utf-8")
    assert "direct_raw_parameter_fit_smoke" not in p
    assert "direct_raw_parameter_fit_robustness_smoke" not in p


def test_p65_13_forbidden_wordings_check():
    forbidden = [
        "vae " + "works", "latent " + "space learned", "semantic " + "geometry proven",
        "c " + "generated", "gsb " + "implemented", "model " + "generalized",
        "production " + "ready", "held-out " + "generalization", "transfer " + "proven",
        "model " + "generalizes to held-out target", "proof " + "of transfer",
        "seed-robust " + "generalization", "component " + "proof",
        "label-specific " + "semantic proof",
    ]
    p_src = pathlib.Path("src/phase2/fc_vae_held_out_label_permutation_control_audit.py").read_text(encoding="utf-8").lower()
    for item in forbidden:
        assert item not in p_src


def test_p65_14_scope_gate():
    from tests.phase2_scope_gate_utils import enforce_phase_local_scope_gate_or_skip
    
    allowed = {
        "src/phase2/fc_vae_held_out_label_permutation_control_audit.py",
        "tools/phase2/run_p65_fc_vae_held_out_label_permutation_control_audit_smoke.py",
        "tests/test_phase2_fc_vae_held_out_label_permutation_control_audit.py",
        "tests/test_phase2_p65_fc_vae_held_out_label_permutation_control_audit_smoke.py",
        "reports/PHASE_2_P65_FC_VAE_HELD_OUT_LABEL_PERMUTATION_CONTROL_AUDIT_NO_NEW_OPTIMIZATION_NO_DATASET_NO_GENERALIZATION_REPORT.md",
    }
    
    enforce_phase_local_scope_gate_or_skip(
        expected_branch="phase2/p65-fc-vae-held-out-label-permutation-control-audit-no-new-optimization-no-dataset-no-generalization",
        base_commit="81fd0b9ac40f6664fe1f73e8dcca032a2ad8d0d5",
        allowed_files=allowed,
        phase_label="P65",
    )
