# tests/test_phase4_synthetic_few_shot_selector_pilot.py

import json
import os
import pathlib
import sys
import subprocess
from typing import Any

from src.phase4.synthetic_few_shot_selector_pilot import (
    run_p87_synthetic_few_shot_selector_pilot_probe,
    scan_keys,
)


def test_p87_01_probe_constants():
    res = run_p87_synthetic_few_shot_selector_pilot_probe()
    assert res["phase"] == "P87"
    assert res["phase_group"] == "PHASE_4"
    assert res["phase_name"] == "Synthetic Few-Shot Selector Pilot from Materialized Support Episodes"
    assert res["contract_version"] == "phase4_p87_synthetic_few_shot_selector_pilot_v1"
    
    assert res["source_materializer_phase"] == "P86"
    assert res["source_externalized_context_phase"] == "P85"
    assert res["source_enrichment_phase"] == "P84"
    assert res["source_identifiability_phase"] == "P83"
    assert res["source_selector_pilot_phase"] == "P82"
    assert res["source_dataset_phase"] == "P81"
    
    assert res["verdict"] == "P87_READY_FOR_REVIEW"


def test_p87_02_source_contracts():
    res = run_p87_synthetic_few_shot_selector_pilot_probe()
    assert res["source_contracts_validated"] is True
    assert res["p86_materialized_support_preserved"] is True
    assert res["p86_sanitized_collision_key_preserved"] is True
    assert res["p86_p87_eligibility_preserved"] is True
    assert res["p86_bridge_not_ready_preserved"] is True
    
    assert res["sanity_summary"]["source_contracts_validated"] is True
    assert res["sanity_summary"]["p86_materialized_support_preserved"] is True
    assert res["sanity_summary"]["p86_sanitized_collision_key_preserved"] is True
    assert res["sanity_summary"]["p86_p87_eligibility_preserved"] is True
    assert res["sanity_summary"]["p86_bridge_not_ready_preserved"] is True


def test_p87_03_training_flags():
    res = run_p87_synthetic_few_shot_selector_pilot_probe()
    assert res["model_training_performed"] is True
    assert res["torch_training_performed"] is True
    assert res["new_model_implemented"] is True
    assert res["optimizer_created"] is True
    assert res["checkpoint_written"] is False
    
    assert res["sanity_summary"]["model_training_performed"] is True
    assert res["sanity_summary"]["torch_training_performed"] is True
    assert res["sanity_summary"]["new_model_implemented"] is True
    assert res["sanity_summary"]["optimizer_created"] is True
    assert res["sanity_summary"]["checkpoint_written"] is False


def test_p87_04_leakage_audit_and_boundaries():
    res = run_p87_synthetic_few_shot_selector_pilot_probe()
    assert "model_input_leakage_audit" in res
    
    audit = res["model_input_leakage_audit"]
    assert audit["diagnostic_pass"] is True
    
    assert audit["support_record_id_used_for_model_input"] is False
    assert audit["materialization_metadata_used_for_model_input"] is False
    assert audit["evaluated_target_endpoint_used_for_model_input"] is False
    assert audit["evaluated_target_delta_used_for_model_input"] is False
    assert audit["exact_relation_label_used_for_model_input"] is False
    assert audit["exact_operator_id_used_for_model_input"] is False
    assert audit["audit_metadata_used_for_model_input"] is False
    assert audit["relation_specific_hint_used_for_model_input"] is False
    
    assert res["sanity_summary"]["support_record_id_used_for_model_input"] is False
    assert res["sanity_summary"]["materialization_metadata_used_for_model_input"] is False
    assert res["sanity_summary"]["evaluated_target_endpoint_used_for_model_input"] is False
    assert res["sanity_summary"]["evaluated_target_delta_used_for_model_input"] is False
    assert res["sanity_summary"]["exact_relation_label_used_for_model_input"] is False
    assert res["sanity_summary"]["exact_operator_id_used_for_model_input"] is False
    assert res["sanity_summary"]["audit_metadata_used_for_model_input"] is False
    assert res["sanity_summary"]["relation_specific_hint_used_for_model_input"] is False


def test_p87_05_split_policy_setup():
    res = run_p87_synthetic_few_shot_selector_pilot_probe()
    assert res["support_split_policy_audit_defined"] is True
    assert res["same_split_support_strict_protocol_defined"] is True
    assert res["train_bank_support_diagnostic_protocol_defined"] is True
    
    assert res["sanity_summary"]["support_split_policy_audit_defined"] is True
    assert res["sanity_summary"]["same_split_support_strict_protocol_defined"] is True
    assert res["sanity_summary"]["train_bank_support_diagnostic_protocol_defined"] is True
    
    assert "primary_support_split_audit" in res
    assert "diagnostic_train_bank_support_split_audit" in res
    
    strict = res["primary_support_split_audit"]
    assert strict["same_split_support_strict_pass"] is True
    assert strict["train_query_uses_validation_or_test_support"] is False
    assert strict["validation_query_uses_test_support"] is False
    assert strict["test_query_uses_train_support"] is False
    
    diag = res["diagnostic_train_bank_support_split_audit"]
    assert diag["train_bank_support_diagnostic_pass"] is True
    assert diag["test_query_uses_train_support"] is True


def test_p87_06_baselines_and_results():
    res = run_p87_synthetic_few_shot_selector_pilot_probe()
    assert "baseline_results" in res
    assert "model_results" in res
    assert "domain_results" in res
    
    baselines = res["baseline_results"]
    assert "majority_accuracy" in baselines
    assert "query_source_only_accuracy" in baselines
    assert "nearest_support_delta_accuracy" in baselines
    assert "support_delta_only_accuracy" in baselines
    
    model = res["model_results"]
    assert "model_test_accuracy" in model
    
    domains = res["domain_results"]
    assert "p70a_vector_world" in domains
    assert "p70b_time_series_parameter_world" in domains
    
    # Assert separate domain reports
    assert "model_accuracy" in domains["p70a_vector_world"]
    assert "model_accuracy" in domains["p70b_time_series_parameter_world"]


def test_p87_07_confusion_summary():
    res = run_p87_synthetic_few_shot_selector_pilot_probe()
    assert "confusion_summary" in res
    conf = res["confusion_summary"]
    
    assert "translate_x_as_reflect_x" in conf
    assert "translate_x_as_nonlinear" in conf
    assert "reflect_x_as_translate_x" in conf
    assert "reflect_x_as_nonlinear" in conf
    assert "nonlinear_as_translate_x" in conf
    assert "nonlinear_as_reflect_x" in conf


def test_p87_08_evidence_rules():
    res = run_p87_synthetic_few_shot_selector_pilot_probe()
    # evidence may be True or False based on actual results, but must be boolean
    assert isinstance(res["learned_selector_evidence_present"], bool)
    assert res["sanity_summary"]["learned_selector_evidence_present"] == res["learned_selector_evidence_present"]
    
    # Learned metric, semantic metric, and bridge must remain strictly False
    assert res["learned_metric_evidence_present"] is False
    assert res["semantic_metric_ready"] is False
    assert res["bridge_implementation_allowed"] is False
    assert res["bridge_ready"] is False
    assert res["generation_claims_allowed"] is False
    assert res["semantic_geometry_claims_allowed"] is False
    
    assert res["sanity_summary"]["learned_metric_evidence_present"] is False
    assert res["sanity_summary"]["semantic_metric_ready"] is False
    assert res["sanity_summary"]["bridge_ready"] is False


def test_p87_09_bridge_boundary():
    res = run_p87_synthetic_few_shot_selector_pilot_probe()
    assert "bridge_boundary_after_few_shot_selector_pilot" in res
    bridge = res["bridge_boundary_after_few_shot_selector_pilot"]
    assert bridge["bridge_ready"] is False
    assert bridge["bridge_implementation_allowed"] is False
    assert bridge["learned_selector_evidence_present"] == res["learned_selector_evidence_present"]
    assert bridge["learned_metric_evidence_present"] is False
    assert bridge["semantic_metric_ready"] is False
    assert "learned_metric_evidence_not_present" in bridge["blocking_reasons"]


def test_p87_10_json_serializable():
    res = run_p87_synthetic_few_shot_selector_pilot_probe()
    dumped = json.dumps(res)
    assert isinstance(dumped, str)


def test_p87_11_forbidden_imports_in_source():
    core_path = pathlib.Path("src/phase4/synthetic_few_shot_selector_pilot.py")
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


def test_p87_12_no_file_writes():
    core_path = pathlib.Path("src/phase4/synthetic_few_shot_selector_pilot.py")
    src = core_path.read_text(encoding="utf-8")
    
    assert "open(" not in src
    assert "write(" not in src


def test_p87_13_forbidden_bridge_behavior():
    core_path = pathlib.Path("src/phase4/synthetic_few_shot_selector_pilot.py")
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


def test_p87_14_scope_gate():
    # Only 5 allowed files in the branch relative to base commit 6dc3535b492208ffd26015745645be6b7191e0f5
    cmd = ["git", "diff", "--name-only", "6dc3535b492208ffd26015745645be6b7191e0f5"]
    res = subprocess.run(cmd, capture_output=True, text=True, check=True)
    files = [f.strip() for f in res.stdout.split("\n") if f.strip()]
    
    allowed = {
        "src/phase4/synthetic_few_shot_selector_pilot.py",
        "tools/phase4/run_p87_synthetic_few_shot_selector_pilot_smoke.py",
        "tests/test_phase4_synthetic_few_shot_selector_pilot.py",
        "tests/test_phase4_p87_synthetic_few_shot_selector_pilot_smoke.py",
        "reports/PHASE_4_P87_SYNTHETIC_FEW_SHOT_SELECTOR_PILOT_FROM_MATERIALIZED_SUPPORT_EPISODES_NO_BRIDGE_REPORT.md",
    }
    for f in files:
        assert f in allowed, f"File {f} is not in the allowed list of changes for P87."
