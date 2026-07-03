# tests/test_phase4_tiny_learned_selector_pilot.py

import json
import os
import pathlib
import sys
import subprocess
from typing import Any

from src.phase4.tiny_learned_selector_pilot import (
    run_p82_tiny_learned_selector_pilot_probe,
)


def test_p82_01_probe_constants():
    res = run_p82_tiny_learned_selector_pilot_probe()
    assert res["phase"] == "P82"
    assert res["phase_group"] == "PHASE_4"
    assert res["phase_name"] == "Tiny Learned Selector Pilot"
    assert res["contract_version"] == "phase4_p82_tiny_learned_selector_pilot_v1"
    
    assert res["source_dataset_phase"] == "P81"
    assert res["source_authority_phase"] == "P80"
    
    assert res["verdict"] == "P82_READY_FOR_REVIEW"


def test_p82_02_source_contracts():
    res = run_p82_tiny_learned_selector_pilot_probe()
    assert res["source_contracts_validated"] is True
    assert res["p81_dataset_contract_preserved"] is True
    assert res["p81_selector_input_leakage_absent_preserved"] is True
    assert res["p81_bridge_not_ready_preserved"] is True
    
    assert res["sanity_summary"]["source_contracts_validated"] is True
    assert res["sanity_summary"]["p81_dataset_contract_preserved"] is True
    assert res["sanity_summary"]["p81_selector_input_leakage_absent_preserved"] is True
    assert res["sanity_summary"]["p81_bridge_not_ready_preserved"] is True


def test_p82_03_authority_inheritance():
    res = run_p82_tiny_learned_selector_pilot_probe()
    assert res["torch_allowed"] is True
    assert res["training_allowed"] is True
    assert res["model_implementation_allowed"] is True
    assert res["learned_selector_allowed"] is True


def test_p82_04_pilot_status():
    res = run_p82_tiny_learned_selector_pilot_probe()
    assert res["tiny_learned_selector_pilot_allowed"] is True
    assert res["tiny_learned_selector_trained"] is True
    assert res["clean_selector_input_used"] is True
    assert res["hint_passthrough_used_as_model_input"] is False
    
    assert res["sanity_summary"]["tiny_learned_selector_trained"] is True
    assert res["sanity_summary"]["clean_selector_input_used"] is True
    assert res["sanity_summary"]["hint_passthrough_used_as_model_input"] is False


def test_p82_05_input_leakage_flags():
    res = run_p82_tiny_learned_selector_pilot_probe()
    assert res["target_endpoint_used_for_selector_input"] is False
    assert res["target_delta_used_for_selector_input"] is False
    assert res["exact_relation_label_used_for_selector_input"] is False
    assert res["exact_operator_id_used_for_selector_input"] is False
    assert res["audit_metadata_used_for_selector_input"] is False
    assert res["relation_specific_hint_used_for_model_input"] is False
    
    assert res["sanity_summary"]["selector_input_leakage_detected"] is False


def test_p82_06_split_integrity():
    res = run_p82_tiny_learned_selector_pilot_probe()
    assert res["test_split_used_for_training"] is False
    assert res["test_labels_used_for_model_selection"] is False
    assert res["validation_used_for_model_selection"] is True
    
    assert res["sanity_summary"]["test_split_used_for_training"] is False
    assert res["sanity_summary"]["test_labels_used_for_model_selection"] is False
    assert res["sanity_summary"]["validation_used_for_model_selection"] is True


def test_p82_07_checkpoint_written():
    res = run_p82_tiny_learned_selector_pilot_probe()
    assert res["checkpoint_written"] is False
    assert res["sanity_summary"]["checkpoint_written"] is False


def test_p82_08_evidence_results():
    res = run_p82_tiny_learned_selector_pilot_probe()
    assert "learned_selector_evidence_present" in res
    assert "learned_selector_evidence_present" in res["sanity_summary"]
    
    # Check baseline results are populated
    assert "baseline_results" in res
    baselines = res["baseline_results"]
    assert "majority_selector_baseline" in baselines
    assert "source_only_baseline" in baselines
    assert "intensity_only_baseline" in baselines
    assert "source_plus_intensity_baseline" in baselines
    assert "hint_passthrough_baseline" in baselines
    
    # Check training results are populated
    assert "tiny_selector_training_results" in res
    training = res["tiny_selector_training_results"]
    assert training["training_completed"] is True
    assert training["epochs"] == 100
    assert training["model_class"] == "TinySelector"
    assert training["optimizer_class"] == "AdamW"
    assert training["loss_class"] == "CrossEntropyLoss"
    assert training["best_validation_accuracy"] >= 0.0
    assert training["test_accuracy_at_best"] >= 0.0


def test_p82_09_bridge_blocked():
    res = run_p82_tiny_learned_selector_pilot_probe()
    assert res["bridge_implementation_allowed"] is False
    assert res["bridge_ready"] is False
    assert res["learned_metric_evidence_present"] is False
    assert res["semantic_metric_ready"] is False
    assert res["generation_claims_allowed"] is False
    assert res["semantic_geometry_claims_allowed"] is False
    
    assert res["sanity_summary"]["learned_metric_evidence_present"] is False
    assert res["sanity_summary"]["semantic_metric_ready"] is False
    assert res["sanity_summary"]["bridge_ready"] is False


def test_p82_10_json_serializable():
    res = run_p82_tiny_learned_selector_pilot_probe()
    dumped = json.dumps(res)
    assert isinstance(dumped, str)


def test_p82_11_forbidden_imports_in_source():
    core_path = pathlib.Path("src/phase4/tiny_learned_selector_pilot.py")
    src = core_path.read_text(encoding="utf-8")
    
    forbidden_imports = [
        "import numpy",
        "import pandas",
        "import sklearn",
        "import random",
        "np.",
    ]
    for item in forbidden_imports:
        assert item not in src, f"Forbidden import found in source code: {item}"


def test_p82_12_no_file_writes():
    core_path = pathlib.Path("src/phase4/tiny_learned_selector_pilot.py")
    src = core_path.read_text(encoding="utf-8")
    
    # Ensure source does not contain open(..., 'w') or similar for checkpoints
    assert "torch.save" not in src
    assert "pickle.dump" not in src


def test_p82_13_forbidden_bridge_behavior():
    core_path = pathlib.Path("src/phase4/tiny_learned_selector_pilot.py")
    src = core_path.read_text(encoding="utf-8")
    
    forbidden_bridge = [
        "Schrodinger",
        "Schrödinger",
        "GeometricSchrodinger",
        "Brownian",
        "SDE",
        "score_model",
    ]
    for item in forbidden_bridge:
        assert item not in src, f"Forbidden bridge phrase found in source code: {item}"


def test_p82_14_scope_gate():
    # Only 5 allowed files in the branch relative to base commit 13bed8e389bfae89283b23676f74b5b9d4266c26
    cmd = ["git", "diff", "--name-only", "13bed8e389bfae89283b23676f74b5b9d4266c26"]
    res = subprocess.run(cmd, capture_output=True, text=True, check=True)
    files = [f.strip() for f in res.stdout.split("\n") if f.strip()]
    
    allowed = {
        "src/phase4/tiny_learned_selector_pilot.py",
        "tools/phase4/run_p82_tiny_learned_selector_pilot_smoke.py",
        "tests/test_phase4_tiny_learned_selector_pilot.py",
        "tests/test_phase4_p82_tiny_learned_selector_pilot_smoke.py",
        "reports/PHASE_4_P82_TINY_LEARNED_SELECTOR_PILOT_CLEAN_INPUT_ONLY_NO_BRIDGE_REPORT.md",
    }
    for f in files:
        assert f in allowed, f"File {f} is not in the allowed list of changes for P82."
