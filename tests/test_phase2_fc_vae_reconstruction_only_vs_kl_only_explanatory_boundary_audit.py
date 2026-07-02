# tests/test_phase2_fc_vae_reconstruction_only_vs_kl_only_explanatory_boundary_audit.py

import json
import math
import pathlib
import pytest

from src.phase2.fc_vae_reconstruction_only_vs_kl_only_explanatory_boundary_audit import (
    SOURCE_PHASE,
    CONTRACT_VERSION,
    SOURCE_EVIDENCE_PHASE,
    TARGET_IDS,
    SEED_VALUES,
    EXPECTED_SOURCE_COMPONENT_RUNS,
    PROFILE_VECTOR_FIELDS,
    COORDINATE_IDS,
    DISTANCE_EPSILON,
    KL_MATERIALITY_SHARE_THRESHOLD,
    RECON_TOTAL_EXPLANATORY_SHARE_THRESHOLD,
    VIEW_IDS,
    run_p69_reconstruction_only_vs_kl_only_explanatory_boundary_audit_probe,
    fc_vae_reconstruction_only_vs_kl_only_explanatory_boundary_audit_probe_to_json_dict,
)


def test_p69_01_constants_exact():
    assert SOURCE_PHASE == "P69"
    assert CONTRACT_VERSION == "phase2_p69_fc_vae_reconstruction_only_vs_kl_only_explanatory_boundary_audit_contract_v1"
    assert SOURCE_EVIDENCE_PHASE == "P68"
    assert TARGET_IDS == ["bridge_lambda_0_25", "bridge_lambda_0_5", "bridge_lambda_0_75"]
    assert SEED_VALUES == [62062, 62162, 62262]
    assert EXPECTED_SOURCE_COMPONENT_RUNS == 9
    assert PROFILE_VECTOR_FIELDS == [
        "mean_total_delta_value",
        "mean_reconstruction_delta_value",
        "mean_beta_weighted_kl_delta_value",
    ]
    assert COORDINATE_IDS == ["total_delta", "reconstruction_delta", "beta_weighted_kl_delta"]
    assert DISTANCE_EPSILON == 1e-12
    assert KL_MATERIALITY_SHARE_THRESHOLD == 0.01
    assert RECON_TOTAL_EXPLANATORY_SHARE_THRESHOLD == 0.99
    assert VIEW_IDS == [
        "full_vector",
        "reconstruction_only",
        "total_only",
        "kl_only",
        "total_plus_reconstruction",
        "reconstruction_plus_kl",
        "total_plus_kl",
    ]


def test_p69_02_probe_returns_serializable_pass():
    probe = run_p69_reconstruction_only_vs_kl_only_explanatory_boundary_audit_probe()
    assert probe["verdict"] == "PASS"
    
    json_dict = fc_vae_reconstruction_only_vs_kl_only_explanatory_boundary_audit_probe_to_json_dict(probe)
    serialized = json.dumps(json_dict)
    assert len(serialized) > 0


def test_p69_03_p68_source_evidence_verdict_pass():
    probe = run_p69_reconstruction_only_vs_kl_only_explanatory_boundary_audit_probe()
    assert probe["source_p68_verdict"] == "PASS"
    assert probe["source_p68_status"] == "fc_vae_numeric_profile_coordinate_contribution_audit_available_no_dataset_no_generalization"


def test_p69_04_runs_count():
    probe = run_p69_reconstruction_only_vs_kl_only_explanatory_boundary_audit_probe()
    assert probe["expected_source_component_runs"] == 9
    assert probe["observed_source_component_runs"] == 9


def test_p69_05_profile_vector_fields():
    probe = run_p69_reconstruction_only_vs_kl_only_explanatory_boundary_audit_probe()
    assert probe["profile_vector_fields"] == PROFILE_VECTOR_FIELDS
    assert probe["coordinate_ids"] == COORDINATE_IDS


def test_p69_06_explanatory_view_count():
    probe = run_p69_reconstruction_only_vs_kl_only_explanatory_boundary_audit_probe()
    views = probe["explanatory_views"]
    
    assert len(views) == 7
    view_ids = [v["view_id"] for v in views]
    assert view_ids == VIEW_IDS


def test_p69_07_view_distances_and_ratios():
    probe = run_p69_reconstruction_only_vs_kl_only_explanatory_boundary_audit_probe()
    views = probe["explanatory_views"]
    
    for v in views:
        assert v["view_id"] in VIEW_IDS
        assert len(v["pairwise_distances"]) == 9
        
        for p in v["pairwise_distances"]:
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
                
        assert isinstance(v["view_separates_all_non_self_pairs"], bool)
        assert math.isfinite(v["min_non_self_l2_distance"])
        assert math.isfinite(v["mean_non_self_l2_distance"])
        assert math.isfinite(v["max_non_self_l2_distance"])
        
        assert math.isfinite(v["distance_ratio_to_full_min_l2"])
        assert math.isfinite(v["distance_ratio_to_full_mean_l2"])
        assert math.isfinite(v["distance_ratio_to_full_max_l2"])
        
        assert 0.0 <= v["distance_ratio_to_full_min_l2"] <= 1.05


def test_p69_08_aggregate_diagnostics():
    probe = run_p69_reconstruction_only_vs_kl_only_explanatory_boundary_audit_probe()
    agg = probe["aggregate_diagnostics"]
    
    assert agg["source_p68_evidence_valid"] is True
    assert agg["source_total_reconstruction_dominance_carried_forward"] is True
    assert agg["source_kl_coordinate_material"] is False
    assert math.isfinite(agg["source_kl_mean_contribution_share"])
    assert agg["source_qualitative_label_invariance_still_holds"] is True
    assert agg["view_count"] == 7
    assert agg["all_views_finite"] is True
    assert agg["full_vector_separates_all_non_self_pairs"] is True
    
    assert isinstance(agg["reconstruction_only_sufficient"], bool)
    assert isinstance(agg["total_only_sufficient"], bool)
    assert isinstance(agg["kl_only_sufficient"], bool)
    assert isinstance(agg["kl_only_explanatory"], bool)
    assert isinstance(agg["total_plus_reconstruction_sufficient"], bool)
    assert isinstance(agg["reconstruction_plus_kl_sufficient"], bool)
    assert isinstance(agg["total_plus_kl_sufficient"], bool)
    
    assert math.isfinite(agg["reconstruction_total_mean_contribution_share"])
    assert isinstance(agg["reconstruction_total_explains_full_separation"], bool)
    assert isinstance(agg["kl_negligible_by_share"], bool)
    assert isinstance(agg["kl_negligible_despite_coordinate_separation"], bool)
    assert isinstance(agg["reconstruction_driven_numeric_separation"], bool)
    
    assert agg["latent_kl_explanatory_boundary"] == "not_supported_by_current_coordinate_contribution_evidence"
    assert agg["primary_explanation"] == "total_plus_reconstruction"
    assert agg["kl_explanation_status"] == "coordinate_separates_but_negligible_share"
    
    assert agg["explanatory_boundary_claim"] == "reconstruction_only_vs_kl_only_explanatory_boundary_audit_only_no_dataset_no_generalization"
    assert agg["diagnostic_only_no_generalization"] is True


def test_p69_09_boundary_flags():
    probe = run_p69_reconstruction_only_vs_kl_only_explanatory_boundary_audit_probe()
    assert probe["no_new_optimization"] is True
    assert probe["no_direct_optimizer_created"] is True
    assert probe["no_direct_model_created"] is True
    assert probe["no_direct_torch_import"] is True
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
    
    assert probe["no_weighted_training"] is True
    assert probe["uniform_objective_preserved"] is True
    assert probe["held_out_diagnostic_only"] is True
    assert probe["explanatory_boundary_diagnostic_only"] is True


def test_p69_10_no_tensors_in_output():
    probe = run_p69_reconstruction_only_vs_kl_only_explanatory_boundary_audit_probe()
    serialized = json.dumps(probe)
    assert "tensor" not in serialized.lower()


def test_p69_11_no_forbidden_imports():
    p = pathlib.Path("src/phase2/fc_vae_reconstruction_only_vs_kl_only_explanatory_boundary_audit.py").read_text(encoding="utf-8")
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
    
    for line in p.splitlines():
        trimmed = line.strip()
        if trimmed.startswith("import ") or trimmed.startswith("from "):
            for forbidden in ["optim", "torch", "p63", "p64", "p65", "p66", "p67"]:
                if forbidden in trimmed:
                    assert False, f"Forbidden import found: {line}"


def test_p69_12_forbidden_optimization_and_model_markers_absent():
    p = pathlib.Path("src/phase2/fc_vae_reconstruction_only_vs_kl_only_explanatory_boundary_audit.py").read_text(encoding="utf-8")
    
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


def test_p69_13_no_p47_p48_imports():
    p = pathlib.Path("src/phase2/fc_vae_reconstruction_only_vs_kl_only_explanatory_boundary_audit.py").read_text(encoding="utf-8")
    assert "direct_raw_parameter_fit_smoke" not in p
    assert "direct_raw_parameter_fit_robustness_smoke" not in p


def test_p69_14_forbidden_wordings_check():
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
    ]
    p_src = pathlib.Path("src/phase2/fc_vae_reconstruction_only_vs_kl_only_explanatory_boundary_audit.py").read_text(encoding="utf-8").lower()
    for item in forbidden:
        assert item not in p_src


def test_p69_15_scope_gate():
    from tests.phase2_scope_gate_utils import enforce_phase_local_scope_gate_or_skip
    
    allowed = {
        "src/phase2/fc_vae_reconstruction_only_vs_kl_only_explanatory_boundary_audit.py",
        "tools/phase2/run_p69_fc_vae_reconstruction_only_vs_kl_only_explanatory_boundary_audit_smoke.py",
        "tests/test_phase2_fc_vae_reconstruction_only_vs_kl_only_explanatory_boundary_audit.py",
        "tests/test_phase2_p69_fc_vae_reconstruction_only_vs_kl_only_explanatory_boundary_audit_smoke.py",
        "reports/PHASE_2_P69_FC_VAE_RECONSTRUCTION_ONLY_VS_KL_ONLY_EXPLANATORY_BOUNDARY_AUDIT_NO_NEW_OPTIMIZATION_NO_DATASET_NO_GENERALIZATION_REPORT.md",
    }
    
    enforce_phase_local_scope_gate_or_skip(
        expected_branch="phase2/p69-fc-vae-reconstruction-only-vs-kl-only-explanatory-boundary-audit-no-new-optimization-no-dataset-no-generalization",
        base_commit="262cdee06251faa466108e4ed302b6d5a5dae0b7",
        allowed_files=allowed,
        phase_label="P69",
    )
