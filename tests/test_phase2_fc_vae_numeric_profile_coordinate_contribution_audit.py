# tests/test_phase2_fc_vae_numeric_profile_coordinate_contribution_audit.py

import json
import math
import pathlib
import pytest
import re

from src.phase2.fc_vae_numeric_profile_coordinate_contribution_audit import (
    SOURCE_PHASE,
    CONTRACT_VERSION,
    SOURCE_EVIDENCE_PHASE,
    TARGET_IDS,
    SEED_VALUES,
    EXPECTED_SOURCE_COMPONENT_RUNS,
    PROFILE_VECTOR_FIELDS,
    DISTANCE_EPSILON,
    COORDINATE_IDS,
    CONTRIBUTION_DOMINANCE_THRESHOLD,
    KL_MATERIALITY_SHARE_THRESHOLD,
    run_p68_numeric_profile_coordinate_contribution_audit_probe,
    fc_vae_numeric_profile_coordinate_contribution_audit_probe_to_json_dict,
)


def test_p68_01_constants_exact():
    assert SOURCE_PHASE == "P68"
    assert CONTRACT_VERSION == "phase2_p68_fc_vae_numeric_profile_coordinate_contribution_audit_contract_v1"
    assert SOURCE_EVIDENCE_PHASE == "P67"
    assert TARGET_IDS == ["bridge_lambda_0_25", "bridge_lambda_0_5", "bridge_lambda_0_75"]
    assert SEED_VALUES == [62062, 62162, 62262]
    assert EXPECTED_SOURCE_COMPONENT_RUNS == 9
    assert PROFILE_VECTOR_FIELDS == [
        "mean_total_delta_value",
        "mean_reconstruction_delta_value",
        "mean_beta_weighted_kl_delta_value",
    ]
    assert DISTANCE_EPSILON == 1e-12
    assert COORDINATE_IDS == ["total_delta", "reconstruction_delta", "beta_weighted_kl_delta"]
    assert CONTRIBUTION_DOMINANCE_THRESHOLD == 0.9
    assert KL_MATERIALITY_SHARE_THRESHOLD == 0.01


def test_p68_02_probe_returns_serializable_pass():
    probe = run_p68_numeric_profile_coordinate_contribution_audit_probe()
    assert probe["verdict"] == "PASS"
    
    json_dict = fc_vae_numeric_profile_coordinate_contribution_audit_probe_to_json_dict(probe)
    serialized = json.dumps(json_dict)
    assert len(serialized) > 0


def test_p68_03_p67_source_evidence_verdict_pass():
    probe = run_p68_numeric_profile_coordinate_contribution_audit_probe()
    assert probe["source_p67_verdict"] == "PASS"
    assert probe["source_p67_status"] == "fc_vae_numeric_separation_perturbation_stability_audit_available_no_dataset_no_generalization"


def test_p68_04_runs_count():
    probe = run_p68_numeric_profile_coordinate_contribution_audit_probe()
    assert probe["expected_source_component_runs"] == 9
    assert probe["observed_source_component_runs"] == 9


def test_p68_05_profile_vector_fields():
    probe = run_p68_numeric_profile_coordinate_contribution_audit_probe()
    assert probe["profile_vector_fields"] == PROFILE_VECTOR_FIELDS
    assert probe["coordinate_ids"] == COORDINATE_IDS


def test_p68_06_original_numeric_profiles():
    probe = run_p68_numeric_profile_coordinate_contribution_audit_probe()
    profiles = probe["original_numeric_profiles"]
    
    assert len(profiles) == 3
    for p in profiles:
        assert p["target_id"] in TARGET_IDS
        assert p["profile_vector_fields"] == PROFILE_VECTOR_FIELDS
        assert len(p["profile_vector_values"]) == 3
        for val in p["profile_vector_values"]:
            assert math.isfinite(val)
        assert p["mean_total_delta_value"] == p["profile_vector_values"][0]
        assert p["mean_reconstruction_delta_value"] == p["profile_vector_values"][1]
        assert p["mean_beta_weighted_kl_delta_value"] == p["profile_vector_values"][2]


def test_p68_07_full_pairwise_distances():
    probe = run_p68_numeric_profile_coordinate_contribution_audit_probe()
    pairs = probe["full_pairwise_distances"]
    
    # Ordered pairs = 9 records
    assert len(pairs) == 9
    for p in pairs:
        assert p["target_a"] in TARGET_IDS
        assert p["target_b"] in TARGET_IDS
        assert math.isfinite(p["l1_distance"])
        assert math.isfinite(p["l2_distance"])
        assert math.isfinite(p["max_abs_distance"])
        assert math.isfinite(p["squared_l2_distance"])
        
        is_self = (p["target_a"] == p["target_b"])
        assert p["is_self_pair"] == is_self
        if is_self:
            assert p["l2_distance"] == 0.0
            assert p["separated_above_epsilon"] is False
        else:
            assert p["separated_above_epsilon"] == (p["l2_distance"] > DISTANCE_EPSILON)


def test_p68_08_coordinate_only_diagnostics():
    probe = run_p68_numeric_profile_coordinate_contribution_audit_probe()
    diags = probe["coordinate_only_diagnostics"]
    
    assert len(diags) == 3
    for d in diags:
        assert d["coordinate_id"] in COORDINATE_IDS
        assert len(d["pair_records"]) == 9
        
        for r in d["pair_records"]:
            assert r["target_a"] in TARGET_IDS
            assert r["target_b"] in TARGET_IDS
            assert math.isfinite(r["coordinate_abs_difference"])
            assert math.isfinite(r["coordinate_squared_difference"])
            
            is_self = (r["target_a"] == r["target_b"])
            assert r["is_self_pair"] == is_self
            if is_self:
                assert r["coordinate_abs_difference"] == 0.0
                assert r["coordinate_separates_pair"] is False
                
        assert isinstance(d["coordinate_separates_all_non_self_pairs"], bool)
        assert math.isfinite(d["min_non_self_coordinate_abs_diff"])
        assert math.isfinite(d["mean_non_self_coordinate_abs_diff"])
        assert math.isfinite(d["max_non_self_coordinate_abs_diff"])


def test_p68_09_leave_one_coordinate_out_diagnostics():
    probe = run_p68_numeric_profile_coordinate_contribution_audit_probe()
    diags = probe["leave_one_coordinate_out_diagnostics"]
    
    assert len(diags) == 3
    for d in diags:
        assert d["excluded_coordinate_id"] in COORDINATE_IDS
        assert len(d["remaining_coordinate_ids"]) == 2
        assert len(d["pair_records"]) == 9
        
        for r in d["pair_records"]:
            assert r["target_a"] in TARGET_IDS
            assert r["target_b"] in TARGET_IDS
            assert math.isfinite(r["ablation_l2_distance"])
            
            is_self = (r["target_a"] == r["target_b"])
            assert r["is_self_pair"] == is_self
            if is_self:
                assert r["ablation_l2_distance"] == 0.0
                assert r["ablation_preserves_pair_separation"] is False
                
        assert isinstance(d["ablation_preserves_all_non_self_separation"], bool)
        assert math.isfinite(d["min_non_self_ablation_l2_distance"])
        assert math.isfinite(d["mean_non_self_ablation_l2_distance"])
        assert math.isfinite(d["max_non_self_ablation_l2_distance"])


def test_p68_10_per_pair_contributions():
    probe = run_p68_numeric_profile_coordinate_contribution_audit_probe()
    contribs = probe["per_pair_coordinate_contributions"]
    
    # 6 non-self pairs
    assert len(contribs) == 6
    for c in contribs:
        assert c["target_a"] in TARGET_IDS
        assert c["target_b"] in TARGET_IDS
        assert c["target_a"] != c["target_b"]
        assert math.isfinite(c["full_squared_l2"])
        assert c["full_squared_l2"] > 0.0
        
        shares = c["coordinate_contribution_shares"]
        assert len(shares) == 3
        for k in COORDINATE_IDS:
            assert math.isfinite(shares[k])
            assert 0.0 <= shares[k] <= 1.0
            
        assert abs(sum(shares.values()) - 1.0) < 1e-6


def test_p68_11_aggregate_coordinate_contributions():
    probe = run_p68_numeric_profile_coordinate_contribution_audit_probe()
    aggs = probe["aggregate_coordinate_contributions"]
    
    assert len(aggs) == 3
    for a in aggs:
        assert a["coordinate_id"] in COORDINATE_IDS
        assert math.isfinite(a["mean_contribution_share"])
        assert math.isfinite(a["min_contribution_share"])
        assert math.isfinite(a["max_contribution_share"])
        assert isinstance(a["dominates_all_pairs"], bool)
        assert isinstance(a["dominates_any_pair"], bool)


def test_p68_12_aggregate_diagnostics():
    probe = run_p68_numeric_profile_coordinate_contribution_audit_probe()
    agg = probe["aggregate_diagnostics"]
    
    assert agg["source_p67_evidence_valid"] is True
    assert agg["source_numeric_separation_stable_under_tested_perturbations"] is True
    assert agg["source_qualitative_label_invariance_still_holds"] is True
    assert agg["original_profile_count"] == 3
    assert agg["profile_vector_field_count"] == 3
    assert agg["full_pairwise_distance_count"] == 9
    assert agg["non_self_pair_count"] == 6
    assert agg["coordinate_count"] == 3
    
    assert agg["all_original_profiles_consistent_across_p67_perturbation_summaries"] is True
    assert agg["all_full_non_self_pairs_separated"] is True
    
    assert isinstance(agg["coordinates_that_separate_all_non_self_pairs"], list)
    assert isinstance(agg["coordinates_that_do_not_separate_all_non_self_pairs"], list)
    assert isinstance(agg["leave_one_out_ablations_preserving_all_non_self_separation"], list)
    assert isinstance(agg["leave_one_out_ablations_breaking_any_non_self_separation"], list)
    
    assert agg["dominant_coordinate_id"] in COORDINATE_IDS
    assert math.isfinite(agg["dominant_coordinate_mean_share"])
    assert isinstance(agg["dominance_warning"], bool)
    
    assert math.isfinite(agg["kl_coordinate_mean_contribution_share"])
    assert isinstance(agg["kl_coordinate_separates_all_non_self_pairs"], bool)
    assert isinstance(agg["kl_coordinate_material"], bool)
    
    assert agg["coordinate_contribution_claim"] == "numeric_profile_coordinate_contribution_claim_no_dataset_no_generalization".replace("claim", "audit_only")
    assert agg["diagnostic_only_no_generalization"] is True


def test_p68_13_boundary_flags():
    probe = run_p68_numeric_profile_coordinate_contribution_audit_probe()
    assert probe["no_new_optimization"] is True
    assert probe["no_direct_optimizer_created"] is True
    assert probe["no_direct_model_created"] is True
    assert probe["no_direct_torch_import"] is True
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
    
    assert probe["no_weighted_training"] is True
    assert probe["uniform_objective_preserved"] is True
    assert probe["held_out_diagnostic_only"] is True
    assert probe["coordinate_contribution_diagnostic_only"] is True


def test_p68_14_no_tensors_in_output():
    probe = run_p68_numeric_profile_coordinate_contribution_audit_probe()
    serialized = json.dumps(probe)
    assert "tensor" not in serialized.lower()


def test_p68_15_no_forbidden_imports():
    p = pathlib.Path("src/phase2/fc_vae_numeric_profile_coordinate_contribution_audit.py").read_text(encoding="utf-8")
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
    
    for line in p.splitlines():
        trimmed = line.strip()
        if trimmed.startswith("import ") or trimmed.startswith("from "):
            for forbidden in ["optim", "torch", "p63", "p64", "p65", "p66"]:
                if forbidden in trimmed:
                    assert False, f"Forbidden import found: {line}"


def test_p68_16_forbidden_optimization_and_model_markers_absent():
    p = pathlib.Path("src/phase2/fc_vae_numeric_profile_coordinate_contribution_audit.py").read_text(encoding="utf-8")
    
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


def test_p68_17_no_p47_p48_imports():
    p = pathlib.Path("src/phase2/fc_vae_numeric_profile_coordinate_contribution_audit.py").read_text(encoding="utf-8")
    assert "direct_raw_parameter_fit_smoke" not in p
    assert "direct_raw_parameter_fit_robustness_smoke" not in p


def test_p68_18_forbidden_wordings_check():
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
    ]
    p_src = pathlib.Path("src/phase2/fc_vae_numeric_profile_coordinate_contribution_audit.py").read_text(encoding="utf-8").lower()
    for item in forbidden:
        assert item not in p_src


def test_p68_19_scope_gate():
    from tests.phase2_scope_gate_utils import enforce_phase_local_scope_gate_or_skip
    
    allowed = {
        "src/phase2/fc_vae_numeric_profile_coordinate_contribution_audit.py",
        "tools/phase2/run_p68_fc_vae_numeric_profile_coordinate_contribution_audit_smoke.py",
        "tests/test_phase2_fc_vae_numeric_profile_coordinate_contribution_audit.py",
        "tests/test_phase2_p68_fc_vae_numeric_profile_coordinate_contribution_audit_smoke.py",
        "reports/PHASE_2_P68_FC_VAE_NUMERIC_PROFILE_COORDINATE_CONTRIBUTION_AUDIT_NO_NEW_OPTIMIZATION_NO_DATASET_NO_GENERALIZATION_REPORT.md",
    }
    
    enforce_phase_local_scope_gate_or_skip(
        expected_branch="phase2/p68-fc-vae-numeric-profile-coordinate-contribution-audit-no-new-optimization-no-dataset-no-generalization",
        base_commit="f4d9449c90d906d582b916c7c9ecffaf9eac8c11",
        allowed_files=allowed,
        phase_label="P68",
    )
