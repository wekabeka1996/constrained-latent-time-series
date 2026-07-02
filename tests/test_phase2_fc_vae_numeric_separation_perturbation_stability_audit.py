# tests/test_phase2_fc_vae_numeric_separation_perturbation_stability_audit.py

import json
import math
import pathlib
import pytest
import re

from src.phase2.fc_vae_numeric_separation_perturbation_stability_audit import (
    SOURCE_PHASE,
    CONTRACT_VERSION,
    SOURCE_EVIDENCE_PHASE,
    TARGET_IDS,
    SEED_VALUES,
    EXPECTED_SOURCE_COMPONENT_RUNS,
    PROFILE_VECTOR_FIELDS,
    DISTANCE_EPSILON,
    PERTURBATION_LEVELS,
    PERTURBATION_MODES,
    run_p67_numeric_separation_perturbation_stability_audit_probe,
    fc_vae_numeric_separation_perturbation_stability_audit_probe_to_json_dict,
)


def test_p67_01_constants_exact():
    assert SOURCE_PHASE == "P67"
    assert CONTRACT_VERSION == "phase2_p67_fc_vae_numeric_separation_perturbation_stability_audit_contract_v1"
    assert SOURCE_EVIDENCE_PHASE == "P66"
    assert TARGET_IDS == ["bridge_lambda_0_25", "bridge_lambda_0_5", "bridge_lambda_0_75"]
    assert SEED_VALUES == [62062, 62162, 62262]
    assert EXPECTED_SOURCE_COMPONENT_RUNS == 9
    assert PROFILE_VECTOR_FIELDS == [
        "mean_total_delta_value",
        "mean_reconstruction_delta_value",
        "mean_beta_weighted_kl_delta_value",
    ]
    assert DISTANCE_EPSILON == 1e-12
    assert PERTURBATION_LEVELS == [1e-9, 1e-6, 1e-4, 1e-3, 1e-2]
    assert PERTURBATION_MODES == [
        "all_positive",
        "all_negative",
        "alternating_by_coordinate",
        "toward_nearest_neighbor",
    ]


def test_p67_02_probe_returns_serializable_pass():
    probe = run_p67_numeric_separation_perturbation_stability_audit_probe()
    assert probe["verdict"] == "PASS"
    
    json_dict = fc_vae_numeric_separation_perturbation_stability_audit_probe_to_json_dict(probe)
    serialized = json.dumps(json_dict)
    assert len(serialized) > 0


def test_p67_03_p66_source_evidence_verdict_pass():
    probe = run_p67_numeric_separation_perturbation_stability_audit_probe()
    assert probe["source_p66_verdict"] == "PASS"
    assert probe["source_p66_status"] == "fc_vae_target_identity_numeric_separation_audit_available_no_dataset_no_generalization"


def test_p67_04_runs_count():
    probe = run_p67_numeric_separation_perturbation_stability_audit_probe()
    assert probe["expected_source_component_runs"] == 9
    assert probe["observed_source_component_runs"] == 9


def test_p67_05_grid_search_cases():
    probe = run_p67_numeric_separation_perturbation_stability_audit_probe()
    cases = probe["perturbation_summaries"]
    
    assert len(cases) == 20 # 5 levels x 4 modes
    
    for case in cases:
        assert case["perturbation_level"] in PERTURBATION_LEVELS
        assert case["perturbation_mode"] in PERTURBATION_MODES
        
        # each case has exactly 3 perturbed profiles
        assert len(case["perturbed_profiles"]) == 3
        for p in case["perturbed_profiles"]:
            assert p["target_id"] in TARGET_IDS
            assert len(p["original_profile_vector_values"]) == 3
            assert len(p["perturbed_profile_vector_values"]) == 3
            assert len(p["perturbation_delta_vector"]) == 3
            for v in p["perturbation_delta_vector"]:
                assert math.isfinite(v)
                
        # each case has exactly 9 pairwise distances
        assert len(case["pairwise_distances"]) == 9
        for dist in case["pairwise_distances"]:
            assert dist["target_a"] in TARGET_IDS
            assert dist["target_b"] in TARGET_IDS
            assert math.isfinite(dist["l1_distance"])
            assert math.isfinite(dist["l2_distance"])
            assert math.isfinite(dist["max_abs_distance"])
            assert dist["l1_distance"] >= 0.0
            assert dist["l2_distance"] >= 0.0
            assert dist["max_abs_distance"] >= 0.0
            
            is_self = (dist["target_a"] == dist["target_b"])
            assert dist["is_self_pair"] == is_self
            if is_self:
                assert dist["l2_distance"] == 0.0
                assert dist["separated_above_epsilon"] is False
                
        assert math.isfinite(case["min_non_self_l1_distance"])
        assert math.isfinite(case["min_non_self_l2_distance"])
        assert math.isfinite(case["min_non_self_max_abs_distance"])
        assert isinstance(case["all_non_self_profiles_separated_above_epsilon"], bool)
        assert isinstance(case["numeric_separation_preserved"], bool)
        assert isinstance(case["perturbation_summary_passed"], bool)
        
        # nearest-neighbor after perturbation
        assert len(case["nearest_neighbor_after_perturbation"]) == 3
        for nn in case["nearest_neighbor_after_perturbation"]:
            assert nn["target_id"] in TARGET_IDS
            assert nn["original_nearest_non_self_target_id"] in TARGET_IDS
            assert nn["perturbed_nearest_non_self_target_id"] in TARGET_IDS
            assert math.isfinite(nn["perturbed_nearest_non_self_l2_distance"])
            assert isinstance(nn["nearest_non_self_changed"], bool)
            
        assert isinstance(case["nearest_neighbor_changed_count"], int)


def test_p67_06_aggregate_diagnostics():
    probe = run_p67_numeric_separation_perturbation_stability_audit_probe()
    agg = probe["aggregate_diagnostics"]
    
    assert agg["source_p66_evidence_valid"] is True
    assert agg["source_numeric_identity_separation_present"] is True
    assert agg["source_qualitative_label_invariance_still_holds"] is True
    assert agg["perturbation_level_count"] == 5
    assert agg["perturbation_mode_count"] == 4
    assert agg["perturbation_case_count"] == 20
    
    assert isinstance(agg["perturbation_cases_passed_count"], int)
    assert agg["all_perturbation_cases_finite"] is True
    assert isinstance(agg["all_perturbation_cases_preserve_numeric_separation"], bool)
    assert math.isfinite(agg["min_observed_perturbed_non_self_l2_distance"])
    
    assert agg["weakest_perturbation_case"]["perturbation_level"] in PERTURBATION_LEVELS
    assert agg["weakest_perturbation_case"]["perturbation_mode"] in PERTURBATION_MODES
    assert math.isfinite(agg["weakest_perturbation_case"]["min_non_self_l2_distance"])
    
    assert isinstance(agg["nearest_neighbor_change_cases_count"], int)
    assert isinstance(agg["nearest_neighbor_change_cases"], list)
    assert isinstance(agg["numeric_separation_stable_under_tested_perturbations"], bool)
    
    assert agg["perturbation_stability_claim"] == "numeric_separation_perturbation_stability_audit_only_no_new_optimization_no_dataset_no_generalization"
    assert agg["diagnostic_only_no_generalization"] is True


def test_p67_07_boundary_flags():
    probe = run_p67_numeric_separation_perturbation_stability_audit_probe()
    assert probe["no_new_optimization"] is True
    assert probe["no_direct_optimizer_created"] is True
    assert probe["no_direct_model_created"] is True
    assert probe["no_direct_torch_import"] is True
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
    
    assert probe["no_weighted_training"] is True
    assert probe["uniform_objective_preserved"] is True
    assert probe["held_out_diagnostic_only"] is True
    assert probe["perturbation_stability_diagnostic_only"] is True


def test_p67_08_no_tensors_in_output():
    probe = run_p67_numeric_separation_perturbation_stability_audit_probe()
    serialized = json.dumps(probe)
    assert "tensor" not in serialized.lower()


def test_p67_09_no_forbidden_imports():
    p = pathlib.Path("src/phase2/fc_vae_numeric_separation_perturbation_stability_audit.py").read_text(encoding="utf-8")
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
    
    for line in p.splitlines():
        trimmed = line.strip()
        if trimmed.startswith("import ") or trimmed.startswith("from "):
            for forbidden in ["optim", "torch", "p63", "p64", "p65"]:
                if forbidden in trimmed:
                    assert False, f"Forbidden import found: {line}"


def test_p67_10_forbidden_optimization_and_model_markers_absent():
    p = pathlib.Path("src/phase2/fc_vae_numeric_separation_perturbation_stability_audit.py").read_text(encoding="utf-8")
    
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


def test_p67_11_no_p47_p48_imports():
    p = pathlib.Path("src/phase2/fc_vae_numeric_separation_perturbation_stability_audit.py").read_text(encoding="utf-8")
    assert "direct_raw_parameter_fit_smoke" not in p
    assert "direct_raw_parameter_fit_robustness_smoke" not in p


def test_p67_12_forbidden_wordings_check():
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
    ]
    p_src = pathlib.Path("src/phase2/fc_vae_numeric_separation_perturbation_stability_audit.py").read_text(encoding="utf-8").lower()
    for item in forbidden:
        assert item not in p_src


def test_p67_13_scope_gate():
    from tests.phase2_scope_gate_utils import enforce_phase_local_scope_gate_or_skip
    
    allowed = {
        "src/phase2/fc_vae_numeric_separation_perturbation_stability_audit.py",
        "tools/phase2/run_p67_fc_vae_numeric_separation_perturbation_stability_audit_smoke.py",
        "tests/test_phase2_fc_vae_numeric_separation_perturbation_stability_audit.py",
        "tests/test_phase2_p67_fc_vae_numeric_separation_perturbation_stability_audit_smoke.py",
        "reports/PHASE_2_P67_FC_VAE_NUMERIC_SEPARATION_PERTURBATION_STABILITY_AUDIT_NO_NEW_OPTIMIZATION_NO_DATASET_NO_GENERALIZATION_REPORT.md",
    }
    
    enforce_phase_local_scope_gate_or_skip(
        expected_branch="phase2/p67-fc-vae-numeric-separation-perturbation-stability-audit-no-new-optimization-no-dataset-no-generalization",
        base_commit="e88cb239e9e29e6b2742a658c69d7cb89c980feb",
        allowed_files=allowed,
        phase_label="P67",
    )
