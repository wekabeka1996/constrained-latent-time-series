# tests/test_phase2_fc_vae_kl_signal_amplification_feasibility_audit.py

import json
import math
import pathlib
import pytest

from src.phase2.fc_vae_kl_signal_amplification_feasibility_audit import (
    SOURCE_PHASE,
    CONTRACT_VERSION,
    SOURCE_EVIDENCE_PHASE,
    TARGET_IDS,
    SEED_VALUES,
    EXPECTED_SOURCE_COMPONENT_RUNS,
    PROFILE_VECTOR_FIELDS,
    COORDINATE_IDS,
    VIEW_IDS,
    DISTANCE_EPSILON,
    KL_MATERIALITY_SHARE_THRESHOLD,
    AMPLIFICATION_TARGET_SHARES,
    AMPLIFICATION_REFERENCE_VIEWS,
    AMPLIFICATION_REASONABLE_UPPER_BOUND,
    run_p70_kl_signal_amplification_feasibility_audit_probe,
    fc_vae_kl_signal_amplification_feasibility_audit_probe_to_json_dict,
)


def test_p70_01_constants_exact():
    assert SOURCE_PHASE == "P70"
    assert CONTRACT_VERSION == "phase2_p70_fc_vae_kl_signal_amplification_feasibility_audit_contract_v1"
    assert SOURCE_EVIDENCE_PHASE == "P69"
    assert TARGET_IDS == ["bridge_lambda_0_25", "bridge_lambda_0_5", "bridge_lambda_0_75"]
    assert SEED_VALUES == [62062, 62162, 62262]
    assert EXPECTED_SOURCE_COMPONENT_RUNS == 9
    assert PROFILE_VECTOR_FIELDS == [
        "mean_total_delta_value",
        "mean_reconstruction_delta_value",
        "mean_beta_weighted_kl_delta_value",
    ]
    assert COORDINATE_IDS == ["total_delta", "reconstruction_delta", "beta_weighted_kl_delta"]
    assert VIEW_IDS == [
        "full_vector",
        "reconstruction_only",
        "total_only",
        "kl_only",
        "total_plus_reconstruction",
        "reconstruction_plus_kl",
        "total_plus_kl",
    ]
    assert DISTANCE_EPSILON == 1e-12
    assert KL_MATERIALITY_SHARE_THRESHOLD == 0.01
    assert AMPLIFICATION_TARGET_SHARES == [0.01, 0.10, 0.50]
    assert AMPLIFICATION_REFERENCE_VIEWS == ["reconstruction_only", "total_only", "total_plus_reconstruction", "full_vector"]
    assert AMPLIFICATION_REASONABLE_UPPER_BOUND == 10000.0


def test_p70_02_probe_returns_serializable_pass():
    probe = run_p70_kl_signal_amplification_feasibility_audit_probe()
    assert probe["verdict"] == "PASS"
    
    json_dict = fc_vae_kl_signal_amplification_feasibility_audit_probe_to_json_dict(probe)
    serialized = json.dumps(json_dict)
    assert len(serialized) > 0


def test_p70_03_p69_source_evidence_verdict_pass():
    probe = run_p70_kl_signal_amplification_feasibility_audit_probe()
    assert probe["source_p69_verdict"] == "PASS"
    assert probe["source_p69_status"] == "fc_vae_reconstruction_only_vs_kl_only_explanatory_boundary_audit_available_no_dataset_no_generalization"


def test_p70_04_runs_count():
    probe = run_p70_kl_signal_amplification_feasibility_audit_probe()
    assert probe["expected_source_component_runs"] == 9
    assert probe["observed_source_component_runs"] == 9


def test_p70_05_profile_vector_fields_and_views():
    probe = run_p70_kl_signal_amplification_feasibility_audit_probe()
    assert probe["profile_vector_fields"] == PROFILE_VECTOR_FIELDS
    assert probe["coordinate_ids"] == COORDINATE_IDS
    assert probe["view_ids"] == VIEW_IDS


def test_p70_06_ratio_diagnostics():
    probe = run_p70_kl_signal_amplification_feasibility_audit_probe()
    diags = probe["kl_to_reference_ratio_diagnostics"]
    
    assert len(diags) == 4
    for d in diags:
        assert d["reference_view_id"] in AMPLIFICATION_REFERENCE_VIEWS
        assert len(d["pair_records"]) == 6
        
        for r in d["pair_records"]:
            assert r["target_a"] in TARGET_IDS
            assert r["target_b"] in TARGET_IDS
            assert r["target_a"] != r["target_b"]
            assert math.isfinite(r["kl_l2_distance"])
            assert math.isfinite(r["reference_l2_distance"])
            assert math.isfinite(r["kl_to_reference_ratio"])
            assert math.isfinite(r["reference_to_kl_amplification_factor"])
            
            assert r["kl_to_reference_ratio"] > 0.0
            assert r["reference_to_kl_amplification_factor"] > 0.0
            
        assert math.isfinite(d["min_kl_to_reference_ratio"])
        assert math.isfinite(d["mean_kl_to_reference_ratio"])
        assert math.isfinite(d["max_kl_to_reference_ratio"])
        assert math.isfinite(d["min_reference_to_kl_amplification_factor"])
        assert math.isfinite(d["mean_reference_to_kl_amplification_factor"])
        assert math.isfinite(d["max_reference_to_kl_amplification_factor"])
        
        assert d["all_ratios_finite"] is True
        assert d["all_amplification_factors_finite"] is True


def test_p70_07_contribution_share_amplification_diagnostics():
    probe = run_p70_kl_signal_amplification_feasibility_audit_probe()
    diags = probe["contribution_share_amplification_diagnostics"]
    
    assert len(diags) == 3
    for d in diags:
        assert d["target_share"] in AMPLIFICATION_TARGET_SHARES
        assert len(d["pair_records"]) == 6
        
        for r in d["pair_records"]:
            assert r["target_a"] in TARGET_IDS
            assert r["target_b"] in TARGET_IDS
            assert r["target_a"] != r["target_b"]
            assert math.isfinite(r["total_squared_l2"])
            assert math.isfinite(r["reconstruction_squared_l2"])
            assert math.isfinite(r["kl_squared_l2"])
            assert math.isfinite(r["non_kl_squared_l2"])
            assert math.isfinite(r["required_amplification_factor"])
            
            assert r["required_amplification_factor"] > 0.0
            assert isinstance(r["within_reasonable_upper_bound"], bool)
            
        assert math.isfinite(d["min_required_amplification_factor"])
        assert math.isfinite(d["mean_required_amplification_factor"])
        assert math.isfinite(d["max_required_amplification_factor"])
        
        assert d["all_factors_finite"] is True
        assert isinstance(d["all_within_reasonable_upper_bound"], bool)
        assert isinstance(d["any_within_reasonable_upper_bound"], bool)


def test_p70_08_scaled_kl_scenarios():
    probe = run_p70_kl_signal_amplification_feasibility_audit_probe()
    scenarios = probe["scaled_kl_scenarios"]
    
    assert len(scenarios) >= 5
    for s in scenarios:
        assert s["amplification_factor"] > 0.0
        assert len(s["pair_records"]) == 6
        
        for r in s["pair_records"]:
            assert r["target_a"] in TARGET_IDS
            assert r["target_b"] in TARGET_IDS
            assert r["target_a"] != r["target_b"]
            assert math.isfinite(r["scaled_kl_l2_distance"])
            assert isinstance(r["scaled_kl_matches_or_exceeds_reconstruction"], bool)
            assert isinstance(r["scaled_kl_matches_or_exceeds_total"], bool)
            assert isinstance(r["scaled_kl_matches_or_exceeds_full"], bool)
            
        assert math.isfinite(s["min_scaled_kl_l2"])
        assert math.isfinite(s["mean_scaled_kl_l2"])
        assert math.isfinite(s["max_scaled_kl_l2"])
        assert isinstance(s["all_pairs_match_or_exceed_reconstruction"], bool)
        assert isinstance(s["all_pairs_match_or_exceed_total"], bool)
        assert isinstance(s["all_pairs_match_or_exceed_full"], bool)
        assert isinstance(s["any_pair_match_or_exceed_reconstruction"], bool)
        assert isinstance(s["any_pair_match_or_exceed_total"], bool)
        assert isinstance(s["any_pair_match_or_exceed_full"], bool)


def test_p70_09_aggregate_diagnostics():
    probe = run_p70_kl_signal_amplification_feasibility_audit_probe()
    agg = probe["aggregate_diagnostics"]
    
    assert agg["source_p69_evidence_valid"] is True
    assert agg["source_reconstruction_driven_numeric_separation"] is True
    assert agg["source_kl_only_sufficient"] is True
    assert agg["source_kl_only_explanatory"] is False
    assert agg["source_kl_negligible_by_share"] is True
    assert agg["source_latent_kl_explanatory_boundary"] == "not_supported_by_current_coordinate_contribution_evidence"
    assert agg["view_count"] == 7
    assert agg["non_self_pair_count"] == 6
    assert agg["all_view_distances_finite"] is True
    assert agg["kl_only_separates_above_epsilon"] is True
    assert agg["kl_signal_exists_above_epsilon"] is True
    assert agg["kl_signal_material_without_amplification"] is False
    
    assert math.isfinite(agg["mean_kl_to_reconstruction_ratio"])
    assert math.isfinite(agg["mean_reconstruction_to_kl_amplification_factor"])
    assert math.isfinite(agg["mean_kl_to_full_ratio"])
    assert math.isfinite(agg["mean_full_to_kl_amplification_factor"])
    
    assert math.isfinite(agg["amplification_factor_for_1_percent_share_mean"])
    assert math.isfinite(agg["amplification_factor_for_10_percent_share_mean"])
    assert math.isfinite(agg["amplification_factor_for_50_percent_share_mean"])
    
    assert isinstance(agg["one_percent_share_within_reasonable_bound"], bool)
    assert isinstance(agg["ten_percent_share_within_reasonable_bound"], bool)
    assert isinstance(agg["fifty_percent_share_within_reasonable_bound"], bool)
    
    assert agg["kl_amplification_feasibility_status"] == "kl_signal_exists_but_requires_extreme_amplification_even_for_1_percent_share"
    assert agg["kl_amplification_claim"] == "kl_signal_amplification_feasibility_audit_only_no_dataset_no_generalization"
    assert agg["diagnostic_only_no_generalization"] is True


def test_p70_10_boundary_flags():
    probe = run_p70_kl_signal_amplification_feasibility_audit_probe()
    assert probe["no_new_optimization"] is True
    assert probe["no_direct_optimizer_created"] is True
    assert probe["no_direct_model_created"] is True
    assert probe["no_direct_torch_import"] is True
    assert probe["no_direct_p68_import"] is True
    assert probe["no_direct_p67_import"] is True
    assert probe["no_direct_p66_import"] is True
    assert probe["no_direct_p65_import"] is True
    assert probe["no_direct_p64_import"] is True
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
    assert probe["no_numeric_identity_proof_claim"] is True
    assert probe["no_perturbation_robustness_proof_claim"] is True
    assert probe["no_coordinate_semantic_proof_claim"] is True
    assert probe["no_latent_kl_explanatory_proof_claim"] is True
    assert probe["no_kl_semantic_signal_proof_claim"] is True
    
    assert probe["no_beta_change"] is True
    assert probe["no_weighted_training"] is True
    assert probe["uniform_objective_preserved"] is True
    assert probe["held_out_diagnostic_only"] is True
    assert probe["kl_amplification_feasibility_diagnostic_only"] is True


def test_p70_11_no_tensors_in_output():
    probe = run_p70_kl_signal_amplification_feasibility_audit_probe()
    serialized = json.dumps(probe)
    assert "tensor" not in serialized.lower()


def test_p70_12_no_forbidden_imports():
    p = pathlib.Path("src/phase2/fc_vae_kl_signal_amplification_feasibility_audit.py").read_text(encoding="utf-8")
    assert "import torch" not in p
    assert "from torch" not in p
    assert "import numpy" not in p
    assert "from numpy" not in p
    assert "import pandas" not in p
    assert "import scipy" not in p
    assert "import sklearn" not in p
    assert "import p63" not in p
    assert "import P63" not in p
    assert "import p64" not in p
    assert "import P64" not in p
    assert "import p65" not in p
    assert "import P65" not in p
    assert "import p66" not in p
    assert "import P66" not in p
    assert "import p67" not in p
    assert "import P67" not in p
    assert "import p68" not in p
    assert "import P68" not in p
    
    for line in p.splitlines():
        trimmed = line.strip()
        if trimmed.startswith("import ") or trimmed.startswith("from "):
            for forbidden in ["optim", "torch", "p63", "p64", "p65", "p66", "p67", "p68"]:
                if forbidden in trimmed:
                    assert False, f"Forbidden import found: {line}"


def test_p70_13_forbidden_optimization_and_model_markers_absent():
    p = pathlib.Path("src/phase2/fc_vae_kl_signal_amplification_feasibility_audit.py").read_text(encoding="utf-8")
    
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


def test_p70_14_no_p47_p48_imports():
    p = pathlib.Path("src/phase2/fc_vae_kl_signal_amplification_feasibility_audit.py").read_text(encoding="utf-8")
    assert "direct_raw_parameter_fit_smoke" not in p
    assert "direct_raw_parameter_fit_robustness_smoke" not in p


def test_p70_15_forbidden_wordings_check():
    forbidden = [
        "vae " + "works",
        "latent " + "space " + "learned",
        "semantic " + "geometry " + "proven",
        "c " + "generated",
        "gsb " + "implemented",
        "model " + "generalized",
        "production " + "ready",
        "held-out " + "generalization",
        "transfer " + "proven",
        "model " + "generalizes " + "to held-out " + "target",
        "proof " + "of transfer",
        "seed-robust " + "generalization",
        "component " + "proof",
        "label-specific " + "semantic " + "proof",
        "numeric " + "identity " + "proof",
        "perturbation " + "robustness " + "proof",
        "coordinate " + "semantic " + "proof",
        "latent " + "kl " + "explanatory proof",
        "kl " + "semantic signal proof",
    ]
    p_src = pathlib.Path("src/phase2/fc_vae_kl_signal_amplification_feasibility_audit.py").read_text(encoding="utf-8").lower()
    for item in forbidden:
        assert item not in p_src


def test_p70_16_scope_gate():
    from tests.phase2_scope_gate_utils import enforce_phase_local_scope_gate_or_skip
    
    allowed = {
        "src/phase2/fc_vae_kl_signal_amplification_feasibility_audit.py",
        "tools/phase2/run_p70_fc_vae_kl_signal_amplification_feasibility_audit_smoke.py",
        "tests/test_phase2_fc_vae_kl_signal_amplification_feasibility_audit.py",
        "tests/test_phase2_p70_fc_vae_kl_signal_amplification_feasibility_audit_smoke.py",
        "reports/PHASE_2_P70_FC_VAE_KL_SIGNAL_AMPLIFICATION_FEASIBILITY_AUDIT_NO_NEW_OPTIMIZATION_NO_DATASET_NO_GENERALIZATION_REPORT.md",
    }
    
    enforce_phase_local_scope_gate_or_skip(
        expected_branch="phase2/p70-fc-vae-kl-signal-amplification-feasibility-audit-no-new-optimization-no-dataset-no-generalization",
        base_commit="f6e6d31b3292224c0a24f73227ce7ac6deaa7560",
        allowed_files=allowed,
        phase_label="P70",
    )
