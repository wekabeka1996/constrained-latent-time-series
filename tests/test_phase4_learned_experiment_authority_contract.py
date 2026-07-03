# tests/test_phase4_learned_experiment_authority_contract.py

import json
import os
import pathlib
import sys
import subprocess
from typing import Any

from src.phase4.learned_experiment_authority_contract import (
    run_p80_learned_experiment_authority_contract_probe,
)


def test_p80_01_probe_constants():
    res = run_p80_learned_experiment_authority_contract_probe()
    assert res["phase"] == "P80"
    assert res["phase_group"] == "PHASE_4"
    assert res["phase_name"] == "Learned Experiment Authority Contract"
    assert res["contract_version"] == "phase4_p80_learned_experiment_authority_contract_v1"
    assert res["source_hard_ablation_phase"] == "P79"
    assert res["verdict"] == "P80_READY_FOR_REVIEW"


def test_p80_02_source_contracts():
    res = run_p80_learned_experiment_authority_contract_probe()
    assert res["source_contracts_validated"] is True
    assert res["p79_hard_ablation_no_signal_preserved"] is True
    assert res["p79_no_learned_selector_evidence_preserved"] is True
    assert res["p79_bridge_not_ready_preserved"] is True
    
    assert res["sanity_summary"]["source_contracts_validated"] is True
    assert res["sanity_summary"]["p79_hard_ablation_no_signal_preserved"] is True
    assert res["sanity_summary"]["p79_no_learned_selector_evidence_preserved"] is True
    assert res["sanity_summary"]["p79_bridge_not_ready_preserved"] is True


def test_p80_03_permissions():
    res = run_p80_learned_experiment_authority_contract_probe()
    assert res["phase4_learned_experiments_allowed"] is True
    assert res["training_allowed"] is True
    assert res["numpy_allowed"] is True
    assert res["torch_allowed"] is True
    assert res["model_implementation_allowed"] is True
    assert res["learned_encoder_allowed"] is True
    assert res["learned_selector_allowed"] is True
    assert res["learned_metric_allowed"] is True
    assert res["optimization_allowed"] is True
    assert res["gpu_usage_allowed"] is True
    assert res["checkpoints_allowed_for_future_phases"] is True
    
    assert res["sanity_summary"]["phase4_learned_experiments_allowed"] is True
    assert res["sanity_summary"]["training_allowed"] is True
    assert res["sanity_summary"]["numpy_allowed"] is True
    assert res["sanity_summary"]["torch_allowed"] is True
    assert res["sanity_summary"]["model_implementation_allowed"] is True
    assert res["sanity_summary"]["learned_selector_allowed"] is True
    assert res["sanity_summary"]["learned_metric_allowed"] is True


def test_p80_04_bridge_blocked():
    res = run_p80_learned_experiment_authority_contract_probe()
    assert res["bridge_implementation_allowed"] is False
    assert res["bridge_ready"] is False
    assert res["semantic_metric_ready"] is False
    assert res["generation_claims_allowed"] is False
    assert res["semantic_geometry_claims_allowed"] is False
    
    assert res["sanity_summary"]["bridge_implementation_allowed"] is False
    assert res["sanity_summary"]["bridge_ready"] is False
    assert res["sanity_summary"]["semantic_metric_ready"] is False
    assert res["sanity_summary"]["p79_bridge_not_ready_preserved"] is True


def test_p80_05_governance_requirements():
    res = run_p80_learned_experiment_authority_contract_probe()
    assert res["leakage_gates_required"] is True
    assert res["baseline_comparison_required"] is True
    assert res["ablation_controls_required"] is True
    assert res["train_val_test_split_required"] is True
    assert res["reproducibility_required"] is True
    assert res["negative_controls_required"] is True
    
    assert res["sanity_summary"]["leakage_gates_required"] is True
    assert res["sanity_summary"]["baseline_comparison_required"] is True
    assert res["sanity_summary"]["ablation_controls_required"] is True
    assert res["sanity_summary"]["train_val_test_split_required"] is True
    assert res["sanity_summary"]["negative_controls_required"] is True


def test_p80_06_leakage_gates():
    res = run_p80_learned_experiment_authority_contract_probe()
    assert res["target_endpoint_used_for_selector"] is False
    assert res["target_delta_used_for_selector"] is False
    assert res["exact_relation_label_used_for_selector"] is False
    assert res["exact_operator_id_used_for_selector"] is False
    assert res["audit_metadata_used_for_selector"] is False


def test_p80_07_allowed_blocked_classes():
    res = run_p80_learned_experiment_authority_contract_probe()
    classes = res["allowed_phase4_experiment_classes"]
    
    # Must include selector, encoder, metric candidates
    assert "learned_relation_selector_candidate" in classes
    assert "learned_relation_encoder_candidate" in classes
    assert "learned_metric_candidate" in classes
    assert "contrastive_relation_metric_learning" in classes
    
    blocked = res["blocked_until_later_phases"]
    # Must include bridge and claims
    assert any("schro" in b and "bridge" in b for b in blocked)
    assert any("generation" in b for b in blocked)


def test_p80_08_evidence_criteria():
    res = run_p80_learned_experiment_authority_contract_probe()
    
    selector_criteria = res["learned_selector_evidence_criteria"]
    # Must include baseline beating and negative controls
    assert any("beats_majority_baseline" in c for c in selector_criteria)
    assert any("survives_negative_controls" in c for c in selector_criteria)
    
    metric_criteria = res["learned_metric_evidence_criteria"]
    # Must include metric not constructed from delta and transfer checks
    assert any("metric_not_constructed_from_target_delta" in c for c in metric_criteria)
    assert any("metric_transfers" in c for c in metric_criteria)
    
    bridge_criteria = res["bridge_readiness_criteria"]
    assert any("learned_selector_evidence_present" in c for c in bridge_criteria)


def test_p80_09_serializability_and_checks():
    res = run_p80_learned_experiment_authority_contract_probe()
    # Output must be JSON serializable
    dumped = json.dumps(res)
    assert isinstance(dumped, str)


def test_p80_10_cleanliness_checks():
    core_path = pathlib.Path("src/phase4/learned_experiment_authority_contract.py")
    src = core_path.read_text(encoding="utf-8")
    
    # Verify core source doesn't implement bridge or training modules
    assert "class Schrodinger" not in src
    assert "class Schrödinger" not in src
    assert "def train(" not in src
    assert "def fit(" not in src
    assert "Optimizer" not in src
    assert "DataLoader" not in src
    
    # Verify no model files or checkpoints exist in src/phase4/
    for f in core_path.parent.glob("*"):
        if f.suffix in [".pt", ".pth", ".ckpt", ".bin"]:
            raise AssertionError(f"Model file or checkpoint found in phase4 directory: {f}")


def test_p80_11_scope_gate():
    # Only 5 allowed files in the branch relative to base commit 8eec009127d87e6011a8962b7355faa52bcedd3b
    cmd = ["git", "diff", "--name-only", "8eec009127d87e6011a8962b7355faa52bcedd3b"]
    res = subprocess.run(cmd, capture_output=True, text=True, check=True)
    files = [f.strip() for f in res.stdout.split("\n") if f.strip()]
    
    allowed = {
        "src/phase4/learned_experiment_authority_contract.py",
        "tools/phase4/run_p80_learned_experiment_authority_contract_smoke.py",
        "tests/test_phase4_learned_experiment_authority_contract.py",
        "tests/test_phase4_p80_learned_experiment_authority_contract_smoke.py",
        "reports/PHASE_4_P80_LEARNED_EXPERIMENT_AUTHORITY_CONTRACT_WITH_LEAKAGE_GATES_NO_BRIDGE_REPORT.md",
    }
    for f in files:
        assert f in allowed, f"File {f} is not in the allowed list of changes for P80."
