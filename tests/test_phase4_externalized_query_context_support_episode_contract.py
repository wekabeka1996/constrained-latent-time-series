# tests/test_phase4_externalized_query_context_support_episode_contract.py

import json
import os
import pathlib
import sys
import subprocess
from typing import Any

from src.phase4.externalized_query_context_support_episode_contract import (
    run_p85_externalized_query_context_support_episode_contract_probe,
    scan_keys,
)


def test_p85_01_probe_constants():
    res = run_p85_externalized_query_context_support_episode_contract_probe()
    assert res["phase"] == "P85"
    assert res["phase_group"] == "PHASE_4"
    assert res["phase_name"] == "Externalized Query Context Source and Support Episode Contract"
    assert res["contract_version"] == "phase4_p85_externalized_query_context_support_episode_contract_v1"
    
    assert res["source_enrichment_phase"] == "P84"
    assert res["source_identifiability_phase"] == "P83"
    assert res["source_selector_pilot_phase"] == "P82"
    assert res["source_dataset_phase"] == "P81"
    
    assert res["verdict"] == "P85_READY_FOR_REVIEW"


def test_p85_02_source_contracts():
    res = run_p85_externalized_query_context_support_episode_contract_probe()
    assert res["source_contracts_validated"] is True
    assert res["p84_query_observation_contract_preserved"] is True
    assert res["p84_no_final_selector_evidence_preserved"] is True
    assert res["p84_real_external_context_required_preserved"] is True
    assert res["p84_bridge_not_ready_preserved"] is True
    
    assert res["sanity_summary"]["source_contracts_validated"] is True
    assert res["sanity_summary"]["p84_query_observation_contract_preserved"] is True
    assert res["sanity_summary"]["p84_no_final_selector_evidence_preserved"] is True
    assert res["sanity_summary"]["p84_real_external_context_required_preserved"] is True
    assert res["sanity_summary"]["p84_bridge_not_ready_preserved"] is True


def test_p85_03_no_training_performed():
    res = run_p85_externalized_query_context_support_episode_contract_probe()
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


def test_p85_04_episode_flags():
    res = run_p85_externalized_query_context_support_episode_contract_probe()
    assert res["externalized_query_context_source_defined"] is True
    assert res["support_episode_context_defined"] is True
    assert res["support_pair_result_delta_schema_defined"] is True
    assert res["episode_manifest_schema_defined"] is True
    assert res["support_invariant_context_schema_defined"] is True
    
    assert res["sanity_summary"]["externalized_query_context_source_defined"] is True
    assert res["sanity_summary"]["support_episode_context_defined"] is True
    assert res["sanity_summary"]["support_pair_result_delta_schema_defined"] is True
    assert res["sanity_summary"]["episode_manifest_schema_defined"] is True
    assert res["sanity_summary"]["support_invariant_context_schema_defined"] is True


def test_p85_05_selection_setup():
    res = run_p85_externalized_query_context_support_episode_contract_probe()
    assert res["support_selection_uses_relation_label_for_synthetic_dataset_construction"] is True
    assert res["support_selection_label_visible_to_selector"] is False
    assert res["support_context_constructed_by_synthetic_task_generator"] is True
    
    assert res["sanity_summary"]["support_selection_uses_relation_label_for_synthetic_dataset_construction"] is True
    assert res["sanity_summary"]["support_selection_label_visible_to_selector"] is False


def test_p85_06_target_leaks_absent():
    res = run_p85_externalized_query_context_support_episode_contract_probe()
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


def test_p85_07_feasibility_claims():
    res = run_p85_externalized_query_context_support_episode_contract_probe()
    assert res["valid_for_p86_synthetic_few_shot_selector_experiment"] is False
    assert res["valid_for_final_semantic_geometry_evidence"] is False
    assert res["valid_for_bridge_evidence"] is False
    assert res["external_real_query_source_still_required"] is True
    
    assert res["sanity_summary"]["valid_for_p86_synthetic_few_shot_selector_experiment"] is False
    assert res["sanity_summary"]["valid_for_final_semantic_geometry_evidence"] is False
    assert res["sanity_summary"]["valid_for_bridge_evidence"] is False
    assert res["sanity_summary"]["external_real_query_source_still_required"] is True


def test_p85_08_leakage_and_collision_reduction_audits():
    res = run_p85_externalized_query_context_support_episode_contract_probe()
    assert "episode_input_leakage_audit" in res
    assert "episode_collision_reduction_audit" in res
    
    leakage = res["episode_input_leakage_audit"]
    assert leakage["diagnostic_pass"] is True
    
    reduction = res["episode_collision_reduction_audit"]
    assert reduction["collision_fraction_reduced"] is False
    assert reduction["upper_bound_improved"] is False
    assert reduction["support_episode_limited_by_missing_result_or_delta_summaries"] is True


def test_p85_09_interpretation():
    res = run_p85_externalized_query_context_support_episode_contract_probe()
    assert "externalized_query_context_interpretation" in res
    interp = res["externalized_query_context_interpretation"]
    assert interp["externalized_query_context_source_defined"] is True
    assert interp["support_episode_context_feasibility_supported"] is False
    assert interp["valid_for_p86_synthetic_few_shot_selector_experiment"] is False
    assert interp["valid_for_final_semantic_geometry_evidence"] is False
    assert interp["recommended_next_phase"] == "P86_support_pair_materializer_from_p70_relation_cases_no_training_no_bridge"


def test_p85_10_bridge_blocked():
    res = run_p85_externalized_query_context_support_episode_contract_probe()
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


def test_p85_11_sample_episode_records():
    res = run_p85_externalized_query_context_support_episode_contract_probe()
    samples = res["sample_episode_records"]
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
        selector_input = ep["externalized_episode_selector_input"]
        all_keys = scan_keys(selector_input)
        for k in forbidden_keys:
            assert k not in all_keys, f"Forbidden key '{k}' found in episode selector input."


def test_p85_12_json_serializable():
    res = run_p85_externalized_query_context_support_episode_contract_probe()
    dumped = json.dumps(res)
    assert isinstance(dumped, str)


def test_p85_13_forbidden_imports_in_source():
    core_path = pathlib.Path("src/phase4/externalized_query_context_support_episode_contract.py")
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


def test_p85_14_forbidden_phrases_in_source():
    core_path = pathlib.Path("src/phase4/externalized_query_context_support_episode_contract.py")
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


def test_p85_15_no_file_writes():
    core_path = pathlib.Path("src/phase4/externalized_query_context_support_episode_contract.py")
    src = core_path.read_text(encoding="utf-8")
    
    assert "open(" not in src
    assert "write(" not in src


def test_p85_16_forbidden_bridge_behavior():
    core_path = pathlib.Path("src/phase4/externalized_query_context_support_episode_contract.py")
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


def test_p85_17_scope_gate():
    # Only 5 allowed files in the branch relative to base commit ab76fe11104a1cf0b1a30f9102ae3de843710551
    cmd = ["git", "diff", "--name-only", "ab76fe11104a1cf0b1a30f9102ae3de843710551"]
    res = subprocess.run(cmd, capture_output=True, text=True, check=True)
    files = [f.strip() for f in res.stdout.split("\n") if f.strip()]
    
    allowed = {
        "src/phase4/externalized_query_context_support_episode_contract.py",
        "tools/phase4/run_p85_externalized_query_context_support_episode_contract_smoke.py",
        "tests/test_phase4_externalized_query_context_support_episode_contract.py",
        "tests/test_phase4_p85_externalized_query_context_support_episode_contract_smoke.py",
        "reports/PHASE_4_P85_EXTERNALIZED_QUERY_CONTEXT_SOURCE_AND_SUPPORT_EPISODE_CONTRACT_NO_TRAINING_NO_BRIDGE_REPORT.md",
    }
    for f in files:
        assert f in allowed, f"File {f} is not in the allowed list of changes for P85."
