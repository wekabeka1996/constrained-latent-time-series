# tests/test_phase4_clean_input_identifiability_collision_audit.py

import json
import os
import pathlib
import sys
import subprocess
from typing import Any

from src.phase4.clean_input_identifiability_collision_audit import (
    run_p83_clean_input_identifiability_collision_audit_probe,
)


def test_p83_01_probe_constants():
    res = run_p83_clean_input_identifiability_collision_audit_probe()
    assert res["phase"] == "P83"
    assert res["phase_group"] == "PHASE_4"
    assert res["phase_name"] == "Clean Input Identifiability and Collision Audit"
    assert res["contract_version"] == "phase4_p83_clean_input_identifiability_collision_audit_v1"
    
    assert res["source_selector_pilot_phase"] == "P82"
    assert res["source_dataset_phase"] == "P81"
    
    assert res["verdict"] == "P83_READY_FOR_REVIEW"


def test_p83_02_source_contracts():
    res = run_p83_clean_input_identifiability_collision_audit_probe()
    assert res["source_contracts_validated"] is True
    assert res["p82_tiny_selector_trained_preserved"] is True
    assert res["p82_clean_input_only_preserved"] is True
    assert res["p82_no_learned_selector_evidence_preserved"] is True
    assert res["p82_bridge_not_ready_preserved"] is True
    
    assert res["sanity_summary"]["source_contracts_validated"] is True
    assert res["sanity_summary"]["p82_tiny_selector_trained_preserved"] is True
    assert res["sanity_summary"]["p82_clean_input_only_preserved"] is True
    assert res["sanity_summary"]["p82_no_learned_selector_evidence_preserved"] is True
    assert res["sanity_summary"]["p82_bridge_not_ready_preserved"] is True


def test_p83_03_no_training_performed():
    res = run_p83_clean_input_identifiability_collision_audit_probe()
    assert res["model_training_performed"] is False
    assert res["torch_training_performed"] is False
    assert res["new_model_implemented"] is False
    assert res["optimizer_created"] is False
    assert res["checkpoint_written"] is False
    
    assert res["sanity_summary"]["model_training_performed"] is False
    assert res["sanity_summary"]["torch_training_performed"] is False
    assert res["sanity_summary"]["new_model_implemented"] is False
    assert res["sanity_summary"]["optimizer_created"] is False
    assert res["sanity_summary"]["checkpoint_written"] is False


def test_p83_04_audit_execution_flags():
    res = run_p83_clean_input_identifiability_collision_audit_probe()
    assert res["clean_input_identifiability_audit_performed"] is True
    assert res["clean_input_label_collision_audit_performed"] is True
    assert res["deterministic_oracle_upper_bound_computed"] is True
    
    assert res["sanity_summary"]["clean_input_identifiability_audit_performed"] is True
    assert res["sanity_summary"]["clean_input_label_collision_audit_performed"] is True
    assert res["sanity_summary"]["deterministic_oracle_upper_bound_computed"] is True


def test_p83_05_collision_audits_exist():
    res = run_p83_clean_input_identifiability_collision_audit_probe()
    
    # Check audits exist for all three keys
    assert "full_clean_input_key" in res["collision_audits"]
    assert "p82_model_feature_key" in res["collision_audits"]
    assert "source_only_key" in res["collision_audits"]
    
    # Check split-aware structure has train/val/test/all
    assert "full_clean_input_key" in res["split_aware_collision_audits"]
    sa_full = res["split_aware_collision_audits"]["full_clean_input_key"]
    assert "train" in sa_full
    assert "validation" in sa_full
    assert "test" in sa_full
    assert "all" in sa_full


def test_p83_06_label_distribution():
    res = run_p83_clean_input_identifiability_collision_audit_probe()
    assert "label_distribution_audit" in res
    dist = res["label_distribution_audit"]
    assert dist["total_record_count"] == 114
    assert dist["label_distribution_all"] is not None


def test_p83_07_interpretation_verdict():
    res = run_p83_clean_input_identifiability_collision_audit_probe()
    assert "identifiability_interpretation" in res
    interpretation = res["identifiability_interpretation"]
    
    # Assert main expected diagnostic results are reported
    assert interpretation["clean_input_relation_identifiability_established"] is False
    assert interpretation["clean_input_label_collisions_present"] is True
    assert interpretation["model_capacity_not_primary_failure"] is True
    assert interpretation["additional_observation_context_required"] is True
    assert interpretation["recommended_next_phase"] == "P84_clean_query_observation_enrichment_contract"
    
    assert res["sanity_summary"]["clean_input_relation_identifiability_established"] is False
    assert res["sanity_summary"]["clean_input_label_collisions_present"] is True
    assert res["sanity_summary"]["model_capacity_not_primary_failure"] is True
    assert res["sanity_summary"]["additional_observation_context_required"] is True


def test_p83_08_bridge_blocked():
    res = run_p83_clean_input_identifiability_collision_audit_probe()
    assert res["bridge_implementation_allowed"] is False
    assert res["bridge_ready"] is False
    assert res["learned_selector_evidence_present"] is False
    assert res["learned_metric_evidence_present"] is False
    assert res["semantic_metric_ready"] is False
    assert res["generation_claims_allowed"] is False
    assert res["semantic_geometry_claims_allowed"] is False
    
    assert res["sanity_summary"]["learned_selector_evidence_present"] is False
    assert res["sanity_summary"]["learned_metric_evidence_present"] is False
    assert res["sanity_summary"]["bridge_ready"] is False


def test_p83_09_json_serializable():
    res = run_p83_clean_input_identifiability_collision_audit_probe()
    dumped = json.dumps(res)
    assert isinstance(dumped, str)


def test_p83_10_forbidden_imports_in_source():
    core_path = pathlib.Path("src/phase4/clean_input_identifiability_collision_audit.py")
    src = core_path.read_text(encoding="utf-8")
    
    forbidden_imports = [
        "import torch",
        "import numpy",
        "import pandas",
        "import sklearn",
        "import random",
        "torch.",
        "np.",
    ]
    for item in forbidden_imports:
        assert item not in src, f"Forbidden import found in source code: {item}"


def test_p83_11_forbidden_phrases_in_source():
    core_path = pathlib.Path("src/phase4/clean_input_identifiability_collision_audit.py")
    src = core_path.read_text(encoding="utf-8")
    
    forbidden_phrases = [
        "torch.nn.Module",
        "nn.Module",
        "def optimizer",
        "class Optimizer",
        ".backward(",
        ".step(",
    ]
    for item in forbidden_phrases:
        assert item not in src, f"Forbidden model/training phrase found in source code: {item}"


def test_p83_12_no_file_writes():
    core_path = pathlib.Path("src/phase4/clean_input_identifiability_collision_audit.py")
    src = core_path.read_text(encoding="utf-8")
    
    assert "open(" not in src
    assert "write(" not in src


def test_p83_13_forbidden_bridge_behavior():
    core_path = pathlib.Path("src/phase4/clean_input_identifiability_collision_audit.py")
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


def test_p83_14_scope_gate():
    # Only 5 allowed files in the branch relative to base commit 407576dc0f81ced81980ac6c683205ab926218fe
    cmd = ["git", "diff", "--name-only", "407576dc0f81ced81980ac6c683205ab926218fe"]
    res = subprocess.run(cmd, capture_output=True, text=True, check=True)
    files = [f.strip() for f in res.stdout.split("\n") if f.strip()]
    
    allowed = {
        "src/phase4/clean_input_identifiability_collision_audit.py",
        "tools/phase4/run_p83_clean_input_identifiability_collision_audit_smoke.py",
        "tests/test_phase4_clean_input_identifiability_collision_audit.py",
        "tests/test_phase4_p83_clean_input_identifiability_collision_audit_smoke.py",
        "reports/PHASE_4_P83_CLEAN_INPUT_IDENTIFIABILITY_AND_COLLISION_AUDIT_NO_TRAINING_NO_BRIDGE_REPORT.md",
    }
    for f in files:
        assert f in allowed, f"File {f} is not in the allowed list of changes for P83."
