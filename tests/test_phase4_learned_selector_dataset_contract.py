# tests/test_phase4_learned_selector_dataset_contract.py

import json
import os
import pathlib
import sys
import subprocess
from typing import Any

from src.phase4.learned_selector_dataset_contract import (
    run_p81_learned_selector_dataset_contract_probe,
    scan_keys,
)


def test_p81_01_probe_constants():
    res = run_p81_learned_selector_dataset_contract_probe()
    assert res["phase"] == "P81"
    assert res["phase_group"] == "PHASE_4"
    assert res["phase_name"] == "Learned Selector Dataset Contract and Split Builder"
    assert res["contract_version"] == "phase4_p81_learned_selector_dataset_contract_v1"
    
    assert res["source_authority_phase"] == "P80"
    assert res["source_vector_testbed_phase"] == "P70A"
    assert res["source_time_series_testbed_phase"] == "P70B"
    assert res["source_query_contract_phase"] == "P77"
    
    assert res["verdict"] == "P81_READY_FOR_REVIEW"


def test_p81_02_source_contracts():
    res = run_p81_learned_selector_dataset_contract_probe()
    assert res["source_contracts_validated"] is True
    assert res["p80_phase4_authority_preserved"] is True
    assert res["p80_bridge_not_ready_preserved"] is True
    assert res["p80_leakage_gates_preserved"] is True
    
    assert res["sanity_summary"]["source_contracts_validated"] is True
    assert res["sanity_summary"]["p80_phase4_authority_preserved"] is True
    assert res["sanity_summary"]["p80_bridge_not_ready_preserved"] is True
    assert res["sanity_summary"]["p80_leakage_gates_preserved"] is True


def test_p81_03_authority_inheritance():
    res = run_p81_learned_selector_dataset_contract_probe()
    assert res["phase4_learned_experiments_allowed"] is True
    assert res["training_allowed_by_phase4_authority"] is True
    assert res["numpy_allowed_by_phase4_authority"] is True
    assert res["torch_allowed_by_phase4_authority"] is True
    assert res["learned_selector_allowed_by_phase4_authority"] is True
    assert res["learned_metric_allowed_by_phase4_authority"] is True


def test_p81_04_no_training_performed():
    res = run_p81_learned_selector_dataset_contract_probe()
    assert res["model_training_performed"] is False
    assert res["torch_training_performed"] is False
    assert res["optimizer_created"] is False
    assert res["checkpoint_written"] is False
    
    assert res["sanity_summary"]["model_training_performed"] is False
    assert res["sanity_summary"]["torch_training_performed"] is False
    assert res["sanity_summary"]["optimizer_created"] is False
    assert res["sanity_summary"]["checkpoint_written"] is False


def test_p81_05_bridge_blocked():
    res = run_p81_learned_selector_dataset_contract_probe()
    assert res["bridge_implementation_allowed"] is False
    assert res["bridge_ready"] is False
    assert res["semantic_metric_ready"] is False
    assert res["generation_claims_allowed"] is False
    assert res["semantic_geometry_claims_allowed"] is False
    
    assert res["sanity_summary"]["bridge_implementation_allowed"] is False
    assert res["sanity_summary"]["bridge_ready"] is False
    assert res["learned_selector_evidence_present"] is False
    assert res["sanity_summary"]["learned_selector_evidence_present"] is False


def test_p81_06_dataset_contract_status():
    res = run_p81_learned_selector_dataset_contract_probe()
    assert res["learned_selector_dataset_contract_present"] is True
    assert res["selector_train_val_test_split_built"] is True
    assert res["target_label_available_for_supervised_training"] is True
    
    assert res["sanity_summary"]["learned_selector_dataset_contract_present"] is True
    assert res["sanity_summary"]["selector_train_val_test_split_built"] is True
    assert res["sanity_summary"]["target_label_available_for_supervised_training"] is True


def test_p81_07_split_counts_and_isolation():
    res = run_p81_learned_selector_dataset_contract_probe()
    audit = res["dataset_contract_audit"]
    
    assert audit["all_required_splits_present"] is True
    assert res["sanity_summary"]["all_required_splits_present"] is True
    
    assert audit["train_count"] > 0
    assert audit["validation_count"] > 0
    assert audit["test_count"] > 0
    assert audit["unknown_holdout_count"] == 0
    
    assert audit["train_val_test_split_isolated"] is True
    assert audit["test_split_used_for_training"] is False


def test_p81_08_leakage_audit():
    res = run_p81_learned_selector_dataset_contract_probe()
    audit = res["dataset_contract_audit"]
    
    assert audit["selector_input_leakage_detected"] is False
    assert res["sanity_summary"]["selector_input_leakage_detected"] is False
    
    assert audit["target_endpoint_used_for_selector_input"] is False
    assert audit["target_delta_used_for_selector_input"] is False
    assert audit["exact_relation_label_used_for_selector_input"] is False
    assert audit["exact_operator_id_used_for_selector_input"] is False
    assert audit["audit_metadata_used_for_selector_input"] is False
    assert audit["relation_specific_hint_used_for_main_selector_input"] is False


def test_p81_09_dataset_views():
    res = run_p81_learned_selector_dataset_contract_probe()
    views = res["dataset_views"]
    
    assert "clean_selector_input" in views
    assert "hint_passthrough_baseline_input" in views
    assert "label_evaluation_only" in views
    
    # Check sample records conform to views
    samples = res["sample_dataset_records"]
    assert len(samples) > 0
    
    forbidden_keys = {
        "_audit_metadata",
        "relation_type",
        "operator_id",
        "z_b",
        "params_b",
        "series_b",
        "z_end",
        "params_end",
        "series_end",
        "target_endpoint",
        "target_midpoint",
        "z_b_minus_z_a",
        "params_b_minus_params_a",
        "series_summary_delta",
        "relation_family_hint",
        "transformation_class_hint",
        "relation_axis_hint",
        "parameter_group_hint",
    }
    
    for r in samples:
        # 1. Main clean selector input view check
        sel_keys = scan_keys(r["selector_input"])
        for k in forbidden_keys:
            assert k not in sel_keys, f"Forbidden key '{k}' found in main selector input."
            
        # 2. Target relation label must only appear in label_evaluation
        assert "target_relation_label" in r["label_evaluation"]
        assert "target_relation_label" not in r["selector_input"]
        assert "target_relation_label" not in r["hint_passthrough_baseline_input"]
        
        # 3. Hint passthrough view may contain hints
        hints = r["hint_passthrough_baseline_input"]
        assert any(k in hints for k in ["relation_family_hint", "transformation_class_hint", "relation_axis_hint", "parameter_group_hint"])


def test_p81_10_baseline_contract():
    res = run_p81_learned_selector_dataset_contract_probe()
    bc = res["baseline_contract"]
    
    assert "majority_selector_baseline" in bc["required_baselines"]
    assert "source_only_baseline" in bc["required_baselines"]
    assert "intensity_only_baseline" in bc["required_baselines"]
    assert "hint_passthrough_baseline" in bc["required_baselines"]
    assert "oracle_operator_upper_bound" in bc["required_baselines"]
    
    assert "majority_selector_baseline" in bc["main_selector_must_beat"]
    assert "source_only_baseline" in bc["main_selector_must_beat"]
    assert "intensity_only_baseline" in bc["main_selector_must_beat"]
    
    assert bc["hint_passthrough_is_control_not_main_input"] is True
    assert bc["oracle_operator_is_upper_bound_not_training_input"] is True
    
    assert res["sanity_summary"]["hint_passthrough_baseline_present"] is True
    assert res["sanity_summary"]["baseline_contract_present"] is True


def test_p81_11_forbidden_imports_in_source():
    core_path = pathlib.Path("src/phase4/learned_selector_dataset_contract.py")
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
        assert item not in src, f"Forbidden import item found in source code: {item}"


def test_p81_12_forbidden_phrases_in_source():
    core_path = pathlib.Path("src/phase4/learned_selector_dataset_contract.py")
    src = core_path.read_text(encoding="utf-8")
    
    forbidden_phrases = [
        "DataLoader",
        "Dataset(",
        "Optimizer",
        "nn.Module",
        ".backward(",
        ".step(",
        "fit(",
        "train(",
    ]
    for item in forbidden_phrases:
        assert item not in src, f"Forbidden model/training phrase found in source code: {item}"


def test_p81_13_forbidden_bridge_phrases_in_source():
    core_path = pathlib.Path("src/phase4/learned_selector_dataset_contract.py")
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


def test_p81_14_scope_gate():
    # Only 5 allowed files in the branch relative to base commit 0d01a796a0cd075ebdcd0e1a3f94c9b1edc10e2c
    cmd = ["git", "diff", "--name-only", "0d01a796a0cd075ebdcd0e1a3f94c9b1edc10e2c"]
    res = subprocess.run(cmd, capture_output=True, text=True, check=True)
    files = [f.strip() for f in res.stdout.split("\n") if f.strip()]
    
    allowed = {
        "src/phase4/learned_selector_dataset_contract.py",
        "tools/phase4/run_p81_learned_selector_dataset_contract_smoke.py",
        "tests/test_phase4_learned_selector_dataset_contract.py",
        "tests/test_phase4_p81_learned_selector_dataset_contract_smoke.py",
        "reports/PHASE_4_P81_LEARNED_SELECTOR_DATASET_CONTRACT_AND_SPLIT_BUILDER_NO_MODEL_NO_BRIDGE_REPORT.md",
    }
    for f in files:
        assert f in allowed, f"File {f} is not in the allowed list of changes for P81."
