# tests/test_phase2_fc_vae_target_identity_numeric_separation_audit.py

import json
import math
import pathlib
import pytest
import re

from src.phase2.fc_vae_target_identity_numeric_separation_audit import (
    SOURCE_PHASE,
    CONTRACT_VERSION,
    SOURCE_EVIDENCE_PHASE,
    TARGET_IDS,
    SEED_VALUES,
    EXPECTED_SOURCE_COMPONENT_RUNS,
    EXPECTED_PERMUTATION_IDS,
    DISTANCE_EPSILON,
    run_p66_target_identity_numeric_separation_audit_probe,
    fc_vae_target_identity_numeric_separation_audit_probe_to_json_dict,
)


def test_p66_01_constants_exact():
    assert SOURCE_PHASE == "P66"
    assert CONTRACT_VERSION == "phase2_p66_fc_vae_target_identity_numeric_separation_audit_contract_v1"
    assert SOURCE_EVIDENCE_PHASE == "P65"
    assert TARGET_IDS == ["bridge_lambda_0_25", "bridge_lambda_0_5", "bridge_lambda_0_75"]
    assert SEED_VALUES == [62062, 62162, 62262]
    assert EXPECTED_SOURCE_COMPONENT_RUNS == 9
    assert EXPECTED_PERMUTATION_IDS == ["cyclic_forward", "cyclic_backward"]
    assert DISTANCE_EPSILON == 1e-12


def test_p66_02_probe_returns_serializable_pass():
    probe = run_p66_target_identity_numeric_separation_audit_probe()
    assert probe["verdict"] == "PASS"
    
    json_dict = fc_vae_target_identity_numeric_separation_audit_probe_to_json_dict(probe)
    serialized = json.dumps(json_dict)
    assert len(serialized) > 0


def test_p66_03_p65_source_evidence_verdict_pass():
    probe = run_p66_target_identity_numeric_separation_audit_probe()
    assert probe["source_p65_verdict"] == "PASS"
    assert probe["source_p65_status"] == "fc_vae_held_out_label_permutation_control_audit_available_no_dataset_no_generalization"


def test_p66_04_runs_count():
    probe = run_p66_target_identity_numeric_separation_audit_probe()
    assert probe["expected_source_component_runs"] == 9
    assert probe["observed_source_component_runs"] == 9


def test_p66_05_profile_vector_fields():
    probe = run_p66_target_identity_numeric_separation_audit_probe()
    assert probe["profile_vector_fields"] == [
        "mean_total_delta_value",
        "mean_reconstruction_delta_value",
        "mean_beta_weighted_kl_delta_value",
    ]


def test_p66_06_original_numeric_profiles():
    probe = run_p66_target_identity_numeric_separation_audit_probe()
    profiles = probe["original_numeric_profiles"]
    
    assert len(profiles) == 3
    for p in profiles:
        assert p["target_id"] in TARGET_IDS
        assert p["profile_vector_fields"] == probe["profile_vector_fields"]
        assert len(p["profile_vector_values"]) == 3
        for val in p["profile_vector_values"]:
            assert math.isfinite(val)
        assert p["mean_total_delta_value"] == p["profile_vector_values"][0]
        assert p["mean_reconstruction_delta_value"] == p["profile_vector_values"][1]
        assert p["mean_beta_weighted_kl_delta_value"] == p["profile_vector_values"][2]


def test_p66_07_pairwise_distances():
    probe = run_p66_target_identity_numeric_separation_audit_probe()
    pairs = probe["original_pairwise_distances"]
    
    # Exactly 9 pairs (3 targets x 3 targets)
    assert len(pairs) == 9
    for p in pairs:
        assert p["target_a"] in TARGET_IDS
        assert p["target_b"] in TARGET_IDS
        assert math.isfinite(p["l1_distance"])
        assert math.isfinite(p["l2_distance"])
        assert math.isfinite(p["max_abs_distance"])
        assert p["l1_distance"] >= 0.0
        assert p["l2_distance"] >= 0.0
        assert p["max_abs_distance"] >= 0.0
        
        is_self = (p["target_a"] == p["target_b"])
        assert p["is_self_pair"] == is_self
        if is_self:
            assert p["l2_distance"] == 0.0
            assert p["separated_above_epsilon"] is False
        else:
            # check non-self distance properties
            assert p["separated_above_epsilon"] == (p["l2_distance"] > DISTANCE_EPSILON)


def test_p66_08_nearest_neighbor_diagnostics():
    probe = run_p66_target_identity_numeric_separation_audit_probe()
    nns = probe["nearest_neighbor_diagnostics"]
    
    assert len(nns) == 3
    for n in nns:
        assert n["target_id"] in TARGET_IDS
        assert n["self_distance_l2"] == 0.0
        assert n["nearest_non_self_target_id"] in TARGET_IDS
        assert n["nearest_non_self_target_id"] != n["target_id"]
        assert n["nearest_non_self_l2_distance"] >= 0.0
        assert n["nearest_non_self_l1_distance"] >= 0.0
        assert n["nearest_non_self_max_abs_distance"] >= 0.0
        assert n["self_separated_from_non_self"] == (n["nearest_non_self_l2_distance"] > DISTANCE_EPSILON)


def test_p66_09_permutation_mismatch_diagnostics():
    probe = run_p66_target_identity_numeric_separation_audit_probe()
    mismatches = probe["permutation_mismatch_diagnostics"]
    
    # 2 permutations x 3 targets = 6 records
    assert len(mismatches) == 6
    for m in mismatches:
        assert m["permutation_id"] in EXPECTED_PERMUTATION_IDS
        assert m["target_id"] in TARGET_IDS
        assert len(m["original_profile_vector_values"]) == 3
        assert len(m["permuted_profile_vector_values"]) == 3
        assert m["mismatched_profile_source_target_id"] in TARGET_IDS
        assert m["mismatched_profile_source_target_id"] != m["target_id"]
        
        assert math.isfinite(m["l1_distance"])
        assert math.isfinite(m["l2_distance"])
        assert math.isfinite(m["max_abs_distance"])
        assert m["mismatch_separated_above_epsilon"] == (m["l2_distance"] > DISTANCE_EPSILON)


def test_p66_10_aggregate_diagnostics():
    probe = run_p66_target_identity_numeric_separation_audit_probe()
    agg = probe["aggregate_diagnostics"]
    
    assert agg["source_p65_evidence_valid"] is True
    assert agg["source_label_specificity_warning"] is True
    assert agg["original_profile_count"] == 3
    assert agg["pairwise_distance_count"] == 9
    assert agg["non_self_pairwise_distance_count"] == 6
    
    assert math.isfinite(agg["min_non_self_l2_distance"])
    assert math.isfinite(agg["min_non_self_l1_distance"])
    assert math.isfinite(agg["min_non_self_max_abs_distance"])
    
    assert isinstance(agg["all_original_non_self_profiles_separated_above_epsilon"], bool)
    assert isinstance(agg["all_targets_self_nearest_when_self_allowed"], bool)
    assert isinstance(agg["all_targets_have_non_self_distance_above_epsilon"], bool)
    
    assert agg["permutation_mismatch_count"] == 6
    assert isinstance(agg["permutation_mismatch_separated_count"], int)
    assert isinstance(agg["all_permutation_mismatches_separated_above_epsilon"], bool)
    assert isinstance(agg["numeric_identity_separation_present"], bool)
    assert agg["qualitative_label_invariance_still_holds"] is True
    assert isinstance(agg["numeric_identity_vs_qualitative_invariance_interpretation"], str)
    assert agg["numeric_identity_separation_claim"] == "target_identity_numeric_separation_audit_only_no_new_optimization_no_dataset_no_generalization"
    assert agg["diagnostic_only_no_generalization"] is True


def test_p66_11_boundary_flags():
    probe = run_p66_target_identity_numeric_separation_audit_probe()
    assert probe["no_new_optimization"] is True
    assert probe["no_direct_optimizer_created"] is True
    assert probe["no_direct_model_created"] is True
    assert probe["no_direct_torch_import"] is True
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
    
    assert probe["no_weighted_training"] is True
    assert probe["uniform_objective_preserved"] is True
    assert probe["held_out_diagnostic_only"] is True
    assert probe["numeric_identity_separation_diagnostic_only"] is True


def test_p66_12_no_tensors_in_output():
    probe = run_p66_target_identity_numeric_separation_audit_probe()
    serialized = json.dumps(probe)
    assert "tensor" not in serialized.lower()


def test_p66_13_no_forbidden_imports():
    p = pathlib.Path("src/phase2/fc_vae_target_identity_numeric_separation_audit.py").read_text(encoding="utf-8")
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
    assert "fc_vae_tiny_held_out_seed_sensitivity_audit" not in p
    assert "fc_vae_held_out_component_attribution_audit" not in p
    
    for line in p.splitlines():
        trimmed = line.strip()
        if trimmed.startswith("import ") or trimmed.startswith("from "):
            if "optim" in trimmed or "torch" in trimmed or "p63" in trimmed or "p64" in trimmed:
                assert False, f"Forbidden import found: {line}"


def test_p66_14_forbidden_optimization_and_model_markers_absent():
    p = pathlib.Path("src/phase2/fc_vae_target_identity_numeric_separation_audit.py").read_text(encoding="utf-8")
    
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


def test_p66_15_no_p47_p48_imports():
    p = pathlib.Path("src/phase2/fc_vae_target_identity_numeric_separation_audit.py").read_text(encoding="utf-8")
    assert "direct_raw_parameter_fit_smoke" not in p
    assert "direct_raw_parameter_fit_robustness_smoke" not in p


def test_p66_16_forbidden_wordings_check():
    forbidden = [
        "vae " + "works", "latent " + "space learned", "semantic " + "geometry proven",
        "c " + "generated", "gsb " + "implemented", "model " + "generalized",
        "production " + "ready", "held-out " + "generalization", "transfer " + "proven",
        "model " + "generalizes to held-out target", "proof " + "of transfer",
        "seed-robust " + "generalization", "component " + "proof",
        "label-specific " + "semantic proof", "numeric " + "identity proof",
    ]
    p_src = pathlib.Path("src/phase2/fc_vae_target_identity_numeric_separation_audit.py").read_text(encoding="utf-8").lower()
    for item in forbidden:
        assert item not in p_src


def test_p66_17_scope_gate():
    from tests.phase2_scope_gate_utils import enforce_phase_local_scope_gate_or_skip
    
    allowed = {
        "src/phase2/fc_vae_target_identity_numeric_separation_audit.py",
        "tools/phase2/run_p66_fc_vae_target_identity_numeric_separation_audit_smoke.py",
        "tests/test_phase2_fc_vae_target_identity_numeric_separation_audit.py",
        "tests/test_phase2_p66_fc_vae_target_identity_numeric_separation_audit_smoke.py",
        "reports/PHASE_2_P66_FC_VAE_TARGET_IDENTITY_NUMERIC_SEPARATION_AUDIT_NO_NEW_OPTIMIZATION_NO_DATASET_NO_GENERALIZATION_REPORT.md",
    }
    
    enforce_phase_local_scope_gate_or_skip(
        expected_branch="phase2/p66-fc-vae-target-identity-numeric-separation-audit-no-new-optimization-no-dataset-no-generalization",
        base_commit="1610b8b500f41c8b44da3db98594f46fc4d1254b",
        allowed_files=allowed,
        phase_label="P66",
    )
