# tests/test_phase3_latent_operator_genesis_research_contract.py

import json
import pathlib
import pytest

from src.phase3.latent_operator_genesis_research_contract import (
    PHASE,
    PHASE_GROUP,
    PHASE_NAME,
    CONTRACT_VERSION,
    TRAINING_ALLOWED,
    DATASET_GENERATION_ALLOWED,
    MODEL_IMPLEMENTATION_ALLOWED,
    OPTIMIZATION_ALLOWED,
    TORCH_ALLOWED,
    BRIDGE_IMPLEMENTATION_ALLOWED,
    PRIMARY_EMPIRICAL_TARGET,
    FORBIDDEN_PRIMARY_CLAIM,
    PHASE3_PACKAGE_SEQUENCE,
    MANDATORY_BASELINES,
    MANDATORY_NEGATIVE_CONTROLS,
    OPERATOR_LEVEL_SUCCESS_CRITERIA,
    FORBIDDEN_CLAIMS,
    ALLOWED_CLAIMS,
    run_p68_latent_operator_genesis_research_contract_probe,
)


def test_p68_01_constants_exact():
    assert PHASE == "P68"
    assert PHASE_GROUP == "PHASE_3"
    assert PHASE_NAME == "Latent Operator Genesis"
    assert CONTRACT_VERSION == "phase3_p68_latent_operator_genesis_research_contract_v1"
    
    assert TRAINING_ALLOWED is False
    assert DATASET_GENERATION_ALLOWED is False
    assert MODEL_IMPLEMENTATION_ALLOWED is False
    assert OPTIMIZATION_ALLOWED is False
    assert TORCH_ALLOWED is False
    assert BRIDGE_IMPLEMENTATION_ALLOWED is False
    
    assert PRIMARY_EMPIRICAL_TARGET == "systematic_relational_operator_structure"
    assert FORBIDDEN_PRIMARY_CLAIM == "semantic_geometry_proven"


def test_p68_02_probe_returns_serializable_verdict():
    probe = run_p68_latent_operator_genesis_research_contract_probe()
    assert isinstance(probe, dict)
    assert probe["verdict"] == "P68_READY_FOR_REVIEW"
    
    serialized = json.dumps(probe)
    assert len(serialized) > 0


def test_p68_03_roadmap_sequence():
    probe = run_p68_latent_operator_genesis_research_contract_probe()
    seq = probe["phase3_package_sequence"]
    
    assert len(seq) == 11
    assert "P69_BASELINE_POINT_OFFSET_INTERPOLATION_HARNESS" in seq
    assert "P70A_PURE_NUMERIC_RELATION_TESTBED" in seq
    assert "P70B_SYNTHETIC_TIME_SERIES_RELATION_TESTBED" in seq
    assert "P71_RELATION_ENCODER_CONTRASTIVE_SIGNAL_SMOKE" in seq
    assert "P72_SPARSE_OPERATOR_BANK_MVP" in seq
    assert "P73_TRANSFER_AND_INVARIANT_PRESERVATION_AUDIT" in seq
    assert "P74_COMPOSITION_AND_ORDER_SENSITIVITY_AUDIT" in seq
    assert "P75_GLOBAL_NEGATIVE_CONTROLS_AND_COLLAPSE_AUDIT" in seq
    assert "P76_LEARNED_SEMANTIC_METRIC_PRE_BRIDGE_AUDIT" in seq
    assert "P77_OPTIONAL_RELATION_CONDITIONED_BRIDGE_PILOT" in seq
    assert "P78_PHASE_3_SYNTHESIS_VERDICT" in seq


def test_p68_04_mandatory_baselines():
    probe = run_p68_latent_operator_genesis_research_contract_probe()
    baselines = probe["mandatory_baselines"]
    
    assert "linear_latent_interpolation" in baselines
    assert "z_b_minus_z_a_offset_transfer" in baselines
    assert "mean_offset_per_relation_type" in baselines
    assert "no_relation_apply_or_decoder_baseline" in baselines
    assert "random_relation_vector_baseline" in baselines


def test_p68_05_mandatory_negative_controls():
    probe = run_p68_latent_operator_genesis_research_contract_probe()
    controls = probe["mandatory_negative_controls"]
    
    assert "label_permutation_per_module" in controls
    assert "random_pair_control" in controls
    assert "endpoint_pass_through_detection" in controls
    assert "z_b_minus_z_a_shortcut_detection" in controls
    assert "decoder_hallucination_check" in controls
    assert "operator_collapse_check" in controls


def test_p68_06_success_criteria():
    probe = run_p68_latent_operator_genesis_research_contract_probe()
    criteria = probe["operator_level_success_criteria"]
    
    assert "target_factor_change_above_baseline" in criteria
    assert "non_target_invariant_preservation_above_baseline" in criteria
    assert "heldout_base_state_transfer" in criteria
    assert "heldout_magnitude_transfer" in criteria
    assert "composition_consistency_when_ground_truth_supports_it" in criteria
    assert "seed_stability" in criteria


def test_p68_07_forbidden_and_allowed_claims():
    probe = run_p68_latent_operator_genesis_research_contract_probe()
    forbidden = probe["forbidden_claims"]
    allowed = probe["allowed_claims"]
    
    assert "semantic_geometry_is_proven" in forbidden
    assert "meaning_is_learned" in forbidden
    assert "schrodinger_bridge_creates_meaning" in forbidden
    
    assert "phase3_tests_systematic_relational_structure" in allowed
    assert "operator_level_signal_is_a_candidate_presemantic_structure" in allowed
    assert "bridge_methods_are_deferred_until_operator_and_metric_evidence_exist" in allowed


def test_p68_08_design_assertions():
    probe = run_p68_latent_operator_genesis_research_contract_probe()
    
    assert probe["p70a_required_before_p70b"] is True
    assert probe["contrastive_signal_required_for_p71"] is True
    assert probe["relation_labels_decoder_forbidden"] is True
    assert probe["permutation_control_required_per_module"] is True
    assert probe["bridge_deferred_until_metric_and_operator_evidence"] is True


def test_p68_09_no_torch_imports():
    for path_str in [
        "src/phase3/latent_operator_genesis_research_contract.py",
        "tools/phase3/run_p68_latent_operator_genesis_research_contract_smoke.py",
        "tests/test_phase3_latent_operator_genesis_research_contract.py",
        "tests/test_phase3_p68_latent_operator_genesis_research_contract_smoke.py",
    ]:
        p = pathlib.Path(path_str)
        if p.exists():
            content = p.read_text(encoding="utf-8")
            assert "import " + "torch" not in content
            assert "from " + "torch" not in content


def test_p68_10_no_unsupported_activities():
    # Make sure we didn't declare training or dataset gen or optimization as allowed
    probe = run_p68_latent_operator_genesis_research_contract_probe()
    assert probe["training_allowed"] is False
    assert probe["dataset_generation_allowed"] is False
    assert probe["model_implementation_allowed"] is False
    assert probe["optimization_allowed"] is False
    assert probe["bridge_implementation_allowed"] is False
    assert probe["diagnostic_only"] is True


def test_p68_11_forbidden_wordings_check():
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
    p_src = pathlib.Path("src/phase3/latent_operator_genesis_research_contract.py").read_text(encoding="utf-8").lower()
    for item in forbidden:
        assert item not in p_src


def test_p68_12_scope_gate():
    from tests.phase2_scope_gate_utils import enforce_phase_local_scope_gate_or_skip
    
    allowed = {
        "src/phase3/latent_operator_genesis_research_contract.py",
        "tools/phase3/run_p68_latent_operator_genesis_research_contract_smoke.py",
        "tests/test_phase3_latent_operator_genesis_research_contract.py",
        "tests/test_phase3_p68_latent_operator_genesis_research_contract_smoke.py",
        "reports/PHASE_3_P68_LATENT_OPERATOR_GENESIS_RESEARCH_CONTRACT_NO_TRAINING_NO_DATASET_NO_MODEL_REPORT.md",
    }
    
    enforce_phase_local_scope_gate_or_skip(
        expected_branch="phase3/p68-latent-operator-genesis-research-contract-no-training-no-dataset-no-model",
        base_commit="cf7c5c27543317424f61f627eee56cd060a1fa52",
        allowed_files=allowed,
        phase_label="P68",
    )
