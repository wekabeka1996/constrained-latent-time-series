# tests/test_phase4_support_pair_materializer_from_p70_relation_cases.py

import json
import os
import pathlib
import sys
import subprocess
from typing import Any

from src.phase4.support_pair_materializer_from_p70_relation_cases import (
    run_p86_support_pair_materializer_from_p70_relation_cases_probe,
    scan_keys,
)


def test_p86_01_probe_constants():
    res = run_p86_support_pair_materializer_from_p70_relation_cases_probe()
    assert res["phase"] == "P86"
    assert res["phase_group"] == "PHASE_4"
    assert res["phase_name"] == "Support Pair Materializer from P70 Relation Cases"
    assert res["contract_version"] == "phase4_p86_support_pair_materializer_from_p70_relation_cases_v1"
    
    assert res["source_externalized_context_phase"] == "P85"
    assert res["source_enrichment_phase"] == "P84"
    assert res["source_identifiability_phase"] == "P83"
    assert res["source_selector_pilot_phase"] == "P82"
    assert res["source_dataset_phase"] == "P81"
    assert res["source_synthetic_relation_phases"] == ["P70A", "P70B"]
    
    assert res["verdict"] == "P86_READY_FOR_REVIEW"


def test_p86_02_source_contracts():
    res = run_p86_support_pair_materializer_from_p70_relation_cases_probe()
    assert res["source_contracts_validated"] is True
    assert res["p85_externalized_context_contract_preserved"] is True
    assert res["p85_no_selector_evidence_preserved"] is True
    assert res["p85_support_pair_materializer_required_preserved"] is True
    assert res["p85_bridge_not_ready_preserved"] is True
    
    assert res["sanity_summary"]["source_contracts_validated"] is True
    assert res["sanity_summary"]["p85_externalized_context_contract_preserved"] is True
    assert res["sanity_summary"]["p85_no_selector_evidence_preserved"] is True
    assert res["sanity_summary"]["p85_support_pair_materializer_required_preserved"] is True
    assert res["sanity_summary"]["p85_bridge_not_ready_preserved"] is True


def test_p86_03_no_training_performed():
    res = run_p86_support_pair_materializer_from_p70_relation_cases_probe()
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


def test_p86_04_materialization_flags():
    res = run_p86_support_pair_materializer_from_p70_relation_cases_probe()
    assert res["support_pair_materializer_defined"] is True
    assert res["p70_relation_case_source_defined"] is True
    assert res["support_source_result_delta_schema_defined"] is True
    assert res["support_pair_materialization_attempted"] is True
    assert res["externalized_episode_records_with_materialized_support_defined"] is True
    
    assert res["sanity_summary"]["support_pair_materializer_defined"] is True
    assert res["sanity_summary"]["p70_relation_case_source_defined"] is True
    assert res["sanity_summary"]["support_source_result_delta_schema_defined"] is True
    assert res["sanity_summary"]["support_pair_materialization_attempted"] is True
    assert res["sanity_summary"]["externalized_episode_records_with_materialized_support_defined"] is True


def test_p86_05_selection_setup():
    res = run_p86_support_pair_materializer_from_p70_relation_cases_probe()
    assert res["support_selection_uses_relation_label_for_synthetic_dataset_construction"] is True
    assert res["support_selection_label_visible_to_selector"] is False
    assert res["support_context_constructed_by_synthetic_task_generator"] is True
    
    assert res["sanity_summary"]["support_selection_uses_relation_label_for_synthetic_dataset_construction"] is True
    assert res["sanity_summary"]["support_selection_label_visible_to_selector"] is False


def test_p86_06_target_leaks_absent():
    res = run_p86_support_pair_materializer_from_p70_relation_cases_probe()
    assert res["evaluated_target_endpoint_used_for_selector_input"] is False
    assert res["evaluated_target_delta_used_for_selector_input"] is False
    assert res["exact_relation_label_used_for_selector_input"] is False
    assert res["exact_operator_id_used_for_selector_input"] is False
    assert res["audit_metadata_used_for_selector_input"] is False
    assert res["relation_specific_hint_used_for_model_input"] is False
    
    assert res["sanity_summary"]["evaluated_target_endpoint_used_for_selector_input"] is False
    assert res["sanity_summary"]["evaluated_target_delta_used_for_selector_input"] is False
    assert res["sanity_summary"]["exact_relation_label_used_for_selector_input"] is False
    assert res["sanity_summary"]["exact_operator_id_used_for_selector_input"] is False
    assert res["sanity_summary"]["audit_metadata_used_for_selector_input"] is False
    assert res["sanity_summary"]["relation_specific_hint_used_for_model_input"] is False


def test_p86_07_feasibility_claims():
    res = run_p86_support_pair_materializer_from_p70_relation_cases_probe()
    assert res["valid_for_p87_synthetic_few_shot_selector_pilot"] is True
    assert res["valid_for_final_semantic_geometry_evidence"] is False
    assert res["valid_for_bridge_evidence"] is False
    assert res["external_real_query_source_still_required"] is True
    
    assert res["sanity_summary"]["support_pair_materialization_successful"] is True
    assert res["sanity_summary"]["valid_for_p87_synthetic_few_shot_selector_pilot"] is True
    assert res["sanity_summary"]["valid_for_final_semantic_geometry_evidence"] is False
    assert res["sanity_summary"]["valid_for_bridge_evidence"] is False
    assert res["sanity_summary"]["external_real_query_source_still_required"] is True


def test_p86_08_audits_present():
    res = run_p86_support_pair_materializer_from_p70_relation_cases_probe()
    assert "p70_relation_case_collection_audit" in res
    assert "support_pair_materialization_audit" in res
    assert "materialized_episode_input_leakage_audit" in res
    assert "materialized_episode_collision_reduction_audit" in res
    assert "support_pair_materialization_interpretation" in res
    
    # Assert leakage pass
    assert res["materialized_episode_input_leakage_audit"]["diagnostic_pass"] is True
    
    # Assert collision reduction succeeds with full materialized pair observations
    reduction = res["materialized_episode_collision_reduction_audit"]
    assert reduction["collision_fraction_reduced"] is True
    assert reduction["upper_bound_improved"] is True
    assert reduction["materialized_episode_collision_record_fraction"] < 0.25
    assert reduction["materialized_episode_upper_bound_accuracy"] > 0.85
    assert reduction["valid_for_p87_synthetic_few_shot_selector_pilot"] is True
    
    # Assert sanitization flags
    assert reduction["collision_key_sanitized"] is True
    assert reduction["support_record_id_used_in_collision_key"] is False
    assert reduction["materialization_metadata_used_in_collision_key"] is False
    assert reduction["label_or_hint_used_in_collision_key"] is False
    
    # Assert successful materialization audit
    mat = res["support_pair_materialization_audit"]
    assert mat["materialization_successful"] is True
    assert mat["support_pair_observation_available_fraction"] == 1.0


def test_p86_09_interpretation():
    res = run_p86_support_pair_materializer_from_p70_relation_cases_probe()
    interp = res["support_pair_materialization_interpretation"]
    assert interp["support_pair_materialization_successful"] is True
    assert interp["support_episode_context_feasibility_supported"] is True
    assert interp["valid_for_p87_synthetic_few_shot_selector_pilot"] is True
    assert interp["valid_for_final_semantic_geometry_evidence"] is False
    assert interp["recommended_next_phase"] == "P87_synthetic_few_shot_selector_pilot_no_bridge"


def test_p86_10_bridge_blocked():
    res = run_p86_support_pair_materializer_from_p70_relation_cases_probe()
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


def test_p86_11_sample_records():
    res = run_p86_support_pair_materializer_from_p70_relation_cases_probe()
    samples = res["sample_materialized_episode_records"]
    assert len(samples) > 0
    
    forbidden_keys = {
        "_audit_metadata",
        "relation_type",
        "true_relation_type",
        "operator_id",
        "relation_family_hint",
        "transformation_class_hint",
        "relation_axis_hint",
        "parameter_group_hint",
        "target_endpoint",
        "target_midpoint",
        "z_b",
        "params_b",
        "series_b",
        "z_b_minus_z_a",
        "params_b_minus_params_a",
        "series_summary_delta"
    }
    
    for ep in samples:
        selector_input = ep["materialized_episode_selector_input"]
        all_keys = scan_keys(selector_input)
        for k in forbidden_keys:
            assert k not in all_keys, f"Forbidden key '{k}' found in materialized episode selector input."


def test_p86_12_json_serializable():
    res = run_p86_support_pair_materializer_from_p70_relation_cases_probe()
    dumped = json.dumps(res)
    assert isinstance(dumped, str)


def test_p86_13_forbidden_imports_in_source():
    core_path = pathlib.Path("src/phase4/support_pair_materializer_from_p70_relation_cases.py")
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


def test_p86_14_forbidden_phrases_in_source():
    core_path = pathlib.Path("src/phase4/support_pair_materializer_from_p70_relation_cases.py")
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


def test_p86_15_no_file_writes():
    core_path = pathlib.Path("src/phase4/support_pair_materializer_from_p70_relation_cases.py")
    src = core_path.read_text(encoding="utf-8")
    
    assert "open(" not in src
    assert "write(" not in src


def test_p86_16_forbidden_bridge_behavior():
    core_path = pathlib.Path("src/phase4/support_pair_materializer_from_p70_relation_cases.py")
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


def test_p86_17_scope_gate():
    # Only 5 allowed files in the branch relative to base commit a6816729810a3f66fe59832ab6bfe29e97a3dd9c
    cmd = ["git", "diff", "--name-only", "a6816729810a3f66fe59832ab6bfe29e97a3dd9c"]
    res = subprocess.run(cmd, capture_output=True, text=True, check=True)
    files = [f.strip() for f in res.stdout.split("\n") if f.strip()]
    
    allowed = {
        "src/phase4/support_pair_materializer_from_p70_relation_cases.py",
        "tools/phase4/run_p86_support_pair_materializer_from_p70_relation_cases_smoke.py",
        "tests/test_phase4_support_pair_materializer_from_p70_relation_cases.py",
        "tests/test_phase4_p86_support_pair_materializer_from_p70_relation_cases_smoke.py",
        "reports/PHASE_4_P86_SUPPORT_PAIR_MATERIALIZER_FROM_P70_RELATION_CASES_NO_TRAINING_NO_BRIDGE_REPORT.md",
    }
    for f in files:
        assert f in allowed, f"File {f} is not in the allowed list of changes for P86."


def test_p86_18_collision_key_sanitization_rules():
    from src.phase4.support_pair_materializer_from_p70_relation_cases import (
        run_p86_support_pair_materializer_from_p70_relation_cases_probe,
        build_materialized_episode_selector_key_dict,
        scan_keys,
    )
    res = run_p86_support_pair_materializer_from_p70_relation_cases_probe()
    samples = res["sample_materialized_episode_records"]
    
    forbidden_keys = {
        "support_record_id",
        "materialization_source_phase",
        "materialization_method",
        "missing_result_summary",
        "missing_delta_summary",
        "support_pair_observation_available",
        "target_relation_label",
        "true_relation_type",
        "relation_type",
        "operator_id",
        "relation_family_hint",
        "transformation_class_hint",
        "relation_axis_hint",
        "parameter_group_hint",
        "_audit_metadata",
    }
    
    for ep in samples:
        key_dict = build_materialized_episode_selector_key_dict(ep)
        all_keys = scan_keys(key_dict)
        for k in forbidden_keys:
            assert k not in all_keys, f"Forbidden key '{k}' found in sanitized collision key dict: {key_dict}"
