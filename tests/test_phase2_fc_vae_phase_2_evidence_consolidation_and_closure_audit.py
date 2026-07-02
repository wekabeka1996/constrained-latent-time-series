# tests/test_phase2_fc_vae_phase_2_evidence_consolidation_and_closure_audit.py

import json
import pathlib
import pytest

from src.phase2.fc_vae_phase_2_evidence_consolidation_and_closure_audit import (
    SOURCE_PHASE,
    CONTRACT_VERSION,
    SOURCE_EVIDENCE_PHASE,
    PHASE_2_LEDGER_START,
    PHASE_2_LEDGER_END,
    run_p71_phase_2_evidence_consolidation_and_closure_audit_probe,
    fc_vae_phase_2_evidence_consolidation_and_closure_audit_probe_to_json_dict,
)
from src.phase2.fc_vae_kl_signal_amplification_feasibility_audit import (
    TARGET_IDS,
    SEED_VALUES,
    EXPECTED_SOURCE_COMPONENT_RUNS,
    PROFILE_VECTOR_FIELDS,
    COORDINATE_IDS,
    VIEW_IDS,
)


def test_p71_01_constants_exact():
    assert SOURCE_PHASE == "P71"
    assert CONTRACT_VERSION == "phase2_p71_fc_vae_phase_2_evidence_consolidation_and_closure_audit_contract_v1"
    assert SOURCE_EVIDENCE_PHASE == "P70"
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
    assert PHASE_2_LEDGER_START == "P49"
    assert PHASE_2_LEDGER_END == "P70"


def test_p71_02_probe_returns_serializable_pass():
    probe = run_p71_phase_2_evidence_consolidation_and_closure_audit_probe()
    assert probe["verdict"] == "PASS"
    
    json_dict = fc_vae_phase_2_evidence_consolidation_and_closure_audit_probe_to_json_dict(probe)
    serialized = json.dumps(json_dict)
    assert len(serialized) > 0


def test_p71_03_p70_source_evidence_verdict_pass():
    probe = run_p71_phase_2_evidence_consolidation_and_closure_audit_probe()
    assert probe["source_p70_verdict"] == "PASS"
    assert probe["source_p70_status"] == "fc_vae_kl_signal_amplification_feasibility_audit_available_no_dataset_no_generalization"


def test_p71_04_runs_count():
    probe = run_p71_phase_2_evidence_consolidation_and_closure_audit_probe()
    assert probe["expected_source_component_runs"] == 9
    assert probe["observed_source_component_runs"] == 9


def test_p71_05_profile_vector_fields_and_views():
    probe = run_p71_phase_2_evidence_consolidation_and_closure_audit_probe()
    assert probe["profile_vector_fields"] == PROFILE_VECTOR_FIELDS
    assert probe["coordinate_ids"] == COORDINATE_IDS
    assert probe["view_ids"] == VIEW_IDS


def test_p71_06_ledger_contents():
    probe = run_p71_phase_2_evidence_consolidation_and_closure_audit_probe()
    ledger = probe["accepted_phase_2_ledger"]
    
    assert len(ledger) == 22
    phases = [entry["phase"] for entry in ledger]
    expected_phases = [f"P{i}" for i in range(49, 71)]
    assert phases == expected_phases
    
    for entry in ledger:
        assert entry["accepted_status"] == "accepted"
        assert isinstance(entry["evidence_type"], str)
        assert isinstance(entry["supported_claim"], str)
        assert isinstance(entry["unsupported_claims"], list)
        assert isinstance(entry["closure_relevance"], str)


def test_p71_07_claims_consolidation():
    probe = run_p71_phase_2_evidence_consolidation_and_closure_audit_probe()
    supported = probe["consolidated_supported_claims"]
    unsupported = probe["consolidated_unsupported_claims"]
    
    assert len(supported) > 0
    assert len(unsupported) > 0
    
    assert "signature_space_bridge_targets_constructed" in supported
    assert "phase_2_evidence_supports_reconstruction_driven_numeric_bridge_signal" in supported
    
    assert "dataset_generalization" in unsupported
    assert "transfer" in unsupported
    assert "semantic_geometry" in unsupported
    assert "vae_success" in unsupported
    assert "gsb_claim" not in unsupported  # actual forbidden word matches the target list
    assert "geometric_schrodinger_bridge" in unsupported


def test_p71_08_verdict_and_recommendation():
    probe = run_p71_phase_2_evidence_consolidation_and_closure_audit_probe()
    verdict = probe["phase_2_closure_verdict"]
    recommendation = probe["phase_3_recommendation"]
    
    assert verdict["verdict_id"] == "PHASE_2_COMPLETED_WITH_RECONSTRUCTION_DRIVEN_NUMERIC_BRIDGE_EVIDENCE_AND_NO_LATENT_SEMANTIC_GEOMETRY_PROOF"
    assert verdict["phase_2_should_close"] is True
    assert verdict["phase_3_required_for_latent_semantic_geometry"] is True
    assert verdict["phase_3_entry_boundary"] == "must_not_inherit_phase_2_reconstruction_driven_signal_as_semantic_latent_geometry_proof"
    
    assert recommendation["title"] == "PHASE_3_LATENT_GEOMETRY_AND_GENERATIVE_BRIDGE_VALIDATION_DESIGN"
    assert len(recommendation["proposed_axes"]) > 0


def test_p71_09_aggregate_diagnostics():
    probe = run_p71_phase_2_evidence_consolidation_and_closure_audit_probe()
    agg = probe["aggregate_diagnostics"]
    
    assert agg["source_p70_evidence_valid"] is True
    assert agg["phase_2_ledger_start"] == "P49"
    assert agg["phase_2_ledger_end"] == "P70"
    assert agg["accepted_phase_count"] == 22
    assert agg["supported_claim_count"] > 0
    assert agg["unsupported_claim_count"] > 0
    assert agg["phase_2_should_close"] is True
    assert agg["phase_2_final_verdict"] == "PHASE_2_COMPLETED_WITH_RECONSTRUCTION_DRIVEN_NUMERIC_BRIDGE_EVIDENCE_AND_NO_LATENT_SEMANTIC_GEOMETRY_PROOF"
    assert agg["reconstruction_driven_numeric_bridge_signal_supported"] is True
    assert agg["latent_semantic_geometry_supported"] is False
    assert agg["latent_kl_explanation_supported"] is False
    assert agg["gsb_supported"] is False
    assert agg["a_b_generate_c_supported"] is False
    assert agg["phase_3_required_for_latent_semantic_geometry"] is True
    assert agg["phase_3_entry_boundary"] == "must_not_inherit_phase_2_reconstruction_driven_signal_as_semantic_latent_geometry_proof"
    assert agg["diagnostic_only_no_generalization"] is True


def test_p71_10_boundary_flags():
    probe = run_p71_phase_2_evidence_consolidation_and_closure_audit_probe()
    assert probe["no_new_optimization"] is True
    assert probe["no_direct_optimizer_created"] is True
    assert probe["no_direct_model_created"] is True
    assert probe["no_direct_torch_import"] is True
    assert probe["no_direct_p69_import"] is True
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
    assert probe["no_scientific_conclusion_beyond_diagnostic_closure"] is True
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
    assert probe["no_a_b_generate_c_claim"] is True
    
    assert probe["no_beta_change"] is True
    assert probe["no_weighted_training"] is True
    assert probe["uniform_objective_preserved"] is True
    assert probe["held_out_diagnostic_only"] is True
    assert probe["phase_2_closure_diagnostic_only"] is True


def test_p71_11_no_tensors_in_output():
    probe = run_p71_phase_2_evidence_consolidation_and_closure_audit_probe()
    serialized = json.dumps(probe)
    assert "tensor" not in serialized.lower()


def test_p71_12_no_forbidden_imports():
    p = pathlib.Path("src/phase2/fc_vae_phase_2_evidence_consolidation_and_closure_audit.py").read_text(encoding="utf-8")
    assert "import torch" not in p
    assert "from torch" not in p
    assert "import numpy" not in p
    assert "from numpy" not in p
    assert "import pandas" not in p
    assert "import scipy" not in p
    assert "import sklearn" not in p
    assert "import p69" not in p
    assert "import P69" not in p
    assert "import p68" not in p
    assert "import P68" not in p
    assert "import p67" not in p
    assert "import P67" not in p
    assert "import p66" not in p
    assert "import P66" not in p
    assert "import p65" not in p
    assert "import P65" not in p
    assert "import p64" not in p
    assert "import P64" not in p
    assert "import p63" not in p
    assert "import P63" not in p
    
    for line in p.splitlines():
        trimmed = line.strip()
        if trimmed.startswith("import ") or trimmed.startswith("from "):
            for forbidden in ["optim", "torch", "p63", "p64", "p65", "p66", "p67", "p68", "p69"]:
                if forbidden in trimmed:
                    assert False, f"Forbidden import found: {line}"


def test_p71_13_forbidden_optimization_and_model_markers_absent():
    p = pathlib.Path("src/phase2/fc_vae_phase_2_evidence_consolidation_and_closure_audit.py").read_text(encoding="utf-8")
    
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


def test_p71_14_no_p47_p48_imports():
    p = pathlib.Path("src/phase2/fc_vae_phase_2_evidence_consolidation_and_closure_audit.py").read_text(encoding="utf-8")
    assert "direct_raw_parameter_fit_smoke" not in p
    assert "direct_raw_parameter_fit_robustness_smoke" not in p


def test_p71_15_forbidden_wordings_check():
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
    p_src = pathlib.Path("src/phase2/fc_vae_phase_2_evidence_consolidation_and_closure_audit.py").read_text(encoding="utf-8").lower()
    for item in forbidden:
        assert item not in p_src


def test_p71_16_scope_gate():
    from tests.phase2_scope_gate_utils import enforce_phase_local_scope_gate_or_skip
    
    allowed = {
        "src/phase2/fc_vae_phase_2_evidence_consolidation_and_closure_audit.py",
        "tools/phase2/run_p71_fc_vae_phase_2_evidence_consolidation_and_closure_audit_smoke.py",
        "tests/test_phase2_fc_vae_phase_2_evidence_consolidation_and_closure_audit.py",
        "tests/test_phase2_p71_fc_vae_phase_2_evidence_consolidation_and_closure_audit_smoke.py",
        "reports/PHASE_2_P71_FC_VAE_PHASE_2_EVIDENCE_CONSOLIDATION_AND_CLOSURE_AUDIT_NO_NEW_OPTIMIZATION_NO_DATASET_NO_GENERALIZATION_REPORT.md",
    }
    
    enforce_phase_local_scope_gate_or_skip(
        expected_branch="phase2/p71-fc-vae-phase-2-evidence-consolidation-and-closure-audit-no-new-optimization-no-dataset-no-generalization",
        base_commit="490b5075dc336fd7a00d4f53f5cdf915e5acefd8",
        allowed_files=allowed,
        phase_label="P71",
    )
