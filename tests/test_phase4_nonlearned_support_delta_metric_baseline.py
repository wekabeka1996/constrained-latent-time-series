# tests/test_phase4_nonlearned_support_delta_metric_baseline.py

import json
import os
import pathlib
import sys
import subprocess
from typing import Any

from src.phase4.nonlearned_support_delta_metric_baseline import (
    run_p88_nonlearned_support_delta_metric_baseline_probe,
    scan_keys,
    ALL_CLASSES,
)


def test_p88_01_probe_constants():
    res = run_p88_nonlearned_support_delta_metric_baseline_probe()
    assert res["phase"] == "P88"
    assert res["phase_group"] == "PHASE_4"
    assert res["phase_name"] == "Non-Learned Support-Delta Metric Baseline Formalization"
    assert res["contract_version"] == "phase4_p88_nonlearned_support_delta_metric_baseline_v1"
    
    assert res["source_few_shot_selector_phase"] == "P87"
    assert res["source_materializer_phase"] == "P86"
    assert res["source_externalized_context_phase"] == "P85"
    assert res["source_enrichment_phase"] == "P84"
    assert res["source_identifiability_phase"] == "P83"
    assert res["source_selector_pilot_phase"] == "P82"
    assert res["source_dataset_phase"] == "P81"
    
    assert res["verdict"] == "P88_READY_FOR_REVIEW"


def test_p88_02_no_torch_import():
    core_path = pathlib.Path("src/phase4/nonlearned_support_delta_metric_baseline.py")
    src = core_path.read_text(encoding="utf-8")
    
    # Assert absolutely NO torch import
    assert "import torch" not in src
    assert "torch." not in src


def test_p88_03_no_training_performed():
    res = run_p88_nonlearned_support_delta_metric_baseline_probe()
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


def test_p88_04_source_contracts():
    res = run_p88_nonlearned_support_delta_metric_baseline_probe()
    assert res["source_contracts_validated"] is True
    assert res["p87_no_learned_selector_evidence_preserved"] is True
    assert res["p87_nearest_support_delta_signal_preserved"] is True
    assert res["p87_recommended_p88_preserved"] is True
    assert res["p87_bridge_not_ready_preserved"] is True


def test_p88_05_strict_diagonal_policy_audit():
    res = run_p88_nonlearned_support_delta_metric_baseline_probe()
    assert res["support_split_policy_audit_defined"] is True
    assert res["same_split_support_strict_protocol_defined"] is True
    assert res["train_bank_support_diagnostic_protocol_defined"] is True
    
    # Assert same_split strict diagonal-only
    strict = res["primary_support_split_audit"]
    assert strict["same_split_diagonal_only"] is True
    assert strict["off_diagonal_support_count"] == 0
    assert strict["same_split_support_strict_pass"] is True
    
    assert strict["train_query_uses_validation_or_test_support"] is False
    assert strict["validation_query_uses_train_support"] is False
    assert strict["validation_query_uses_test_support"] is False
    assert strict["test_query_uses_train_support"] is False
    assert strict["test_query_uses_validation_support"] is False
    
    # Assert diagnostic train bank uses train support
    diag = res["diagnostic_train_bank_support_split_audit"]
    assert diag["train_bank_support_diagnostic_pass"] is True
    assert diag["test_query_uses_train_support"] is True


def test_p88_06_leakage_audit_and_boundaries():
    res = run_p88_nonlearned_support_delta_metric_baseline_probe()
    assert "metric_input_leakage_audit" in res
    
    audit = res["metric_input_leakage_audit"]
    assert audit["diagnostic_pass"] is True
    
    # Assert metric inputs exclude target endpoints, labels, and support IDs
    assert audit["support_record_id_used_for_metric_input"] is False
    assert audit["materialization_metadata_used_for_metric_input"] is False
    assert audit["evaluated_target_endpoint_used_for_metric_input"] is False
    assert audit["evaluated_target_delta_used_for_metric_input"] is False
    assert audit["exact_relation_label_used_for_metric_input"] is False
    assert audit["exact_operator_id_used_for_metric_input"] is False
    assert audit["audit_metadata_used_for_metric_input"] is False
    assert audit["relation_specific_hint_used_for_metric_input"] is False


def test_p88_07_distance_family_results():
    res = run_p88_nonlearned_support_delta_metric_baseline_probe()
    assert "distance_family_results" in res
    
    df = res["distance_family_results"]
    assert "l1" in df
    assert "l2" in df
    assert "cosine_safe" in df


def test_p88_08_ablation_results():
    res = run_p88_nonlearned_support_delta_metric_baseline_probe()
    assert "ablation_results" in res
    
    ablations = res["ablation_results"]
    modes = {a["mode"] for a in ablations}
    
    assert "delta_only" in modes
    assert "source_result_delta" in modes
    assert "delta_plus_invariants" in modes
    assert "full_support_metric" in modes
    
    # Negative controls
    assert "query_source_only_negative_control" in modes
    assert "shuffled_support_delta_negative_control" in modes
    assert "zero_delta_negative_control" in modes
    
    # Assert shuffled control verification flags
    assert res["shuffled_support_delta_control_implemented"] is True
    assert res["shuffled_control_target_labels_preserved"] is True
    assert res["shuffled_control_support_deltas_permuted"] is True
    assert res["target_labels_shifted"] is False
    assert res["true_labels_preserved"] is True
    
    assert res["sanity_summary"]["shuffled_support_delta_control_implemented"] is True
    assert res["sanity_summary"]["shuffled_control_target_labels_preserved"] is True
    assert res["sanity_summary"]["shuffled_control_support_deltas_permuted"] is True
    assert res["sanity_summary"]["target_labels_shifted"] is False
    assert res["sanity_summary"]["true_labels_preserved"] is True


def test_p88_09_best_metric_result():
    res = run_p88_nonlearned_support_delta_metric_baseline_probe()
    assert "best_metric_result" in res
    
    best = res["best_metric_result"]
    assert "mode" in best
    assert "distance_family" in best
    assert "accuracy" in best
    assert "predictions" in best
    
    for pred in best["predictions"]:
        assert "predicted_label" in pred
        assert pred["predicted_label"] in ALL_CLASSES


def test_p88_10_domain_and_relation_results():
    res = run_p88_nonlearned_support_delta_metric_baseline_probe()
    
    assert "domain_results" in res
    domains = res["domain_results"]
    assert "p70a_vector_world" in domains
    assert "p70b_time_series_parameter_world" in domains
    
    assert "relation_results" in res
    relations = res["relation_results"]
    for cls in ALL_CLASSES:
        # Check if evaluated classes are reported
        if cls in relations:
            assert "accuracy" in relations[cls]
            assert "correct_count" in relations[cls]
            assert "total_count" in relations[cls]


def test_p88_11_confusion_summary():
    res = run_p88_nonlearned_support_delta_metric_baseline_probe()
    assert "confusion_summary" in res
    
    conf = res["confusion_summary"]
    assert "translate_x_as_reflect_x" in conf
    assert "translate_x_as_nonlinear" in conf
    assert "reflect_x_as_translate_x" in conf
    assert "reflect_x_as_nonlinear" in conf
    assert "nonlinear_as_translate_x" in conf
    assert "nonlinear_as_reflect_x" in conf


def test_p88_12_evidence_rules():
    res = run_p88_nonlearned_support_delta_metric_baseline_probe()
    assert isinstance(res["nonlearned_metric_signal_present"], bool)
    
    # Non-learned metric signal must be True based on P87/P88 accuracies
    assert res["nonlearned_metric_signal_present"] is True
    
    # Learned selector, learned metric, semantic geometry, and bridge boundary must remain False
    assert res["learned_selector_evidence_present"] is False
    assert res["learned_metric_evidence_present"] is False
    assert res["semantic_metric_ready"] is False
    assert res["bridge_implementation_allowed"] is False
    assert res["bridge_ready"] is False
    assert res["generation_claims_allowed"] is False
    assert res["semantic_geometry_claims_allowed"] is False


def test_p88_13_forbidden_imports_in_source():
    core_path = pathlib.Path("src/phase4/nonlearned_support_delta_metric_baseline.py")
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


def test_p88_14_no_file_writes():
    core_path = pathlib.Path("src/phase4/nonlearned_support_delta_metric_baseline.py")
    src = core_path.read_text(encoding="utf-8")
    
    assert "open(" not in src
    assert "write(" not in src


def test_p88_15_scope_gate():
    # Only 5 allowed files in the branch relative to base commit 68488000f11b78c0ada322052d48478a7f434ab4
    cmd = ["git", "diff", "--name-only", "68488000f11b78c0ada322052d48478a7f434ab4"]
    res = subprocess.run(cmd, capture_output=True, text=True, check=True)
    files = [f.strip() for f in res.stdout.split("\n") if f.strip()]
    
    allowed = {
        "src/phase4/nonlearned_support_delta_metric_baseline.py",
        "tools/phase4/run_p88_nonlearned_support_delta_metric_baseline_smoke.py",
        "tests/test_phase4_nonlearned_support_delta_metric_baseline.py",
        "tests/test_phase4_p88_nonlearned_support_delta_metric_baseline_smoke.py",
        "reports/PHASE_4_P88_NONLEARNED_SUPPORT_DELTA_METRIC_BASELINE_FORMALIZATION_NO_TRAINING_NO_BRIDGE_REPORT.md",
    }
    for f in files:
        assert f in allowed, f"File {f} is not in the allowed list of changes for P88."
