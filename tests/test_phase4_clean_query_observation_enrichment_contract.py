# tests/test_phase4_clean_query_observation_enrichment_contract.py

import json
import os
import pathlib
import sys
import subprocess
from typing import Any

from src.phase4.clean_query_observation_enrichment_contract import (
    run_p84_clean_query_observation_enrichment_contract_probe,
    scan_keys,
)


def test_p84_01_probe_constants():
    res = run_p84_clean_query_observation_enrichment_contract_probe()
    assert res["phase"] == "P84"
    assert res["phase_group"] == "PHASE_4"
    assert res["phase_name"] == "Clean Query-Observation Enrichment Contract"
    assert res["contract_version"] == "phase4_p84_clean_query_observation_enrichment_contract_v1"
    
    assert res["source_identifiability_phase"] == "P83"
    assert res["source_selector_pilot_phase"] == "P82"
    assert res["source_dataset_phase"] == "P81"
    
    assert res["verdict"] == "P84_READY_FOR_REVIEW"


def test_p84_02_source_contracts():
    res = run_p84_clean_query_observation_enrichment_contract_probe()
    assert res["source_contracts_validated"] is True
    assert res["p83_clean_input_underidentification_preserved"] is True
    assert res["p83_additional_context_required_preserved"] is True
    assert res["p83_bridge_not_ready_preserved"] is True
    
    assert res["sanity_summary"]["source_contracts_validated"] is True
    assert res["sanity_summary"]["p83_clean_input_underidentification_preserved"] is True
    assert res["sanity_summary"]["p83_additional_context_required_preserved"] is True
    assert res["sanity_summary"]["p83_bridge_not_ready_preserved"] is True


def test_p84_03_no_training_performed():
    res = run_p84_clean_query_observation_enrichment_contract_probe()
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


def test_p84_04_enrichment_flags():
    res = run_p84_clean_query_observation_enrichment_contract_probe()
    assert res["query_observation_enrichment_contract_present"] is True
    assert res["support_observation_context_defined"] is True
    assert res["invariant_observation_context_defined"] is True
    assert res["external_query_context_schema_defined"] is True
    
    assert res["sanity_summary"]["query_observation_enrichment_contract_present"] is True
    assert res["sanity_summary"]["support_observation_context_defined"] is True
    assert res["sanity_summary"]["invariant_observation_context_defined"] is True
    assert res["sanity_summary"]["external_query_context_schema_defined"] is True


def test_p84_05_target_leaks_absent():
    res = run_p84_clean_query_observation_enrichment_contract_probe()
    assert res["target_record_endpoint_used_for_selector_input"] is False
    assert res["target_record_delta_used_for_selector_input"] is False
    assert res["exact_relation_label_used_for_selector_input"] is False
    assert res["exact_operator_id_used_for_selector_input"] is False
    assert res["audit_metadata_used_for_selector_input"] is False
    assert res["relation_specific_hint_used_for_model_input"] is False
    
    assert res["sanity_summary"]["target_record_endpoint_used_for_selector_input"] is False
    assert res["sanity_summary"]["target_record_delta_used_for_selector_input"] is False
    assert res["sanity_summary"]["exact_relation_label_used_for_selector_input"] is False
    assert res["sanity_summary"]["exact_operator_id_used_for_selector_input"] is False
    assert res["sanity_summary"]["audit_metadata_used_for_selector_input"] is False
    assert res["sanity_summary"]["relation_specific_hint_used_for_model_input"] is False


def test_p84_06_synthetic_support_boundaries():
    res = run_p84_clean_query_observation_enrichment_contract_probe()
    assert res["synthetic_support_context_for_feasibility_only"] is True
    assert res["support_context_constructed_from_audit_label"] is True
    assert res["valid_for_final_selector_evidence"] is False
    assert res["valid_for_feasibility_collision_reduction_audit"] is True
    
    assert res["sanity_summary"]["synthetic_support_context_for_feasibility_only"] is True
    assert res["sanity_summary"]["support_context_constructed_from_audit_label"] is True
    assert res["sanity_summary"]["valid_for_final_selector_evidence"] is False
    assert res["sanity_summary"]["valid_for_feasibility_collision_reduction_audit"] is True


def test_p84_07_collision_reduction_audit():
    res = run_p84_clean_query_observation_enrichment_contract_probe()
    assert "collision_reduction_audit" in res
    audit = res["collision_reduction_audit"]
    assert audit["original_collision_record_fraction"] > 0.0
    assert audit["enriched_collision_record_fraction"] > 0.0
    assert audit["valid_for_final_selector_evidence"] is False
    assert audit["valid_for_feasibility_collision_reduction_audit"] is True


def test_p84_08_enrichment_interpretation():
    res = run_p84_clean_query_observation_enrichment_contract_probe()
    assert "enrichment_contract_interpretation" in res
    interp = res["enrichment_contract_interpretation"]
    assert interp["valid_for_final_selector_evidence"] is False
    assert interp["external_pre_target_query_context_required"] is True
    assert interp["recommended_next_phase"] in [
        "P85_externalized_query_context_dataset_or_support_set_contract",
        "P85_define_real_external_query_context_source",
    ]


def test_p84_09_bridge_blocked():
    res = run_p84_clean_query_observation_enrichment_contract_probe()
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


def test_p84_10_sample_enriched_records():
    res = run_p84_clean_query_observation_enrichment_contract_probe()
    samples = res["sample_enriched_records"]
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
    
    for r in samples:
        enriched_input = r["query_observation_enriched_selector_input"]
        all_keys = scan_keys(enriched_input)
        for k in forbidden_keys:
            assert k not in all_keys, f"Forbidden key '{k}' found in enriched selector input."


def test_p84_11_json_serializable():
    res = run_p84_clean_query_observation_enrichment_contract_probe()
    dumped = json.dumps(res)
    assert isinstance(dumped, str)


def test_p84_12_forbidden_imports_in_source():
    core_path = pathlib.Path("src/phase4/clean_query_observation_enrichment_contract.py")
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


def test_p84_13_forbidden_phrases_in_source():
    core_path = pathlib.Path("src/phase4/clean_query_observation_enrichment_contract.py")
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


def test_p84_14_no_file_writes():
    core_path = pathlib.Path("src/phase4/clean_query_observation_enrichment_contract.py")
    src = core_path.read_text(encoding="utf-8")
    
    assert "open(" not in src
    assert "write(" not in src


def test_p84_15_forbidden_bridge_behavior():
    core_path = pathlib.Path("src/phase4/clean_query_observation_enrichment_contract.py")
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


def test_p84_16_scope_gate():
    # Only 5 allowed files in the branch relative to base commit cace3df3d2cc3dda1319865cf0f897605c929da3
    cmd = ["git", "diff", "--name-only", "cace3df3d2cc3dda1319865cf0f897605c929da3"]
    res = subprocess.run(cmd, capture_output=True, text=True, check=True)
    files = [f.strip() for f in res.stdout.split("\n") if f.strip()]
    
    allowed = {
        "src/phase4/clean_query_observation_enrichment_contract.py",
        "tools/phase4/run_p84_clean_query_observation_enrichment_contract_smoke.py",
        "tests/test_phase4_clean_query_observation_enrichment_contract.py",
        "tests/test_phase4_p84_clean_query_observation_enrichment_contract_smoke.py",
        "reports/PHASE_4_P84_CLEAN_QUERY_OBSERVATION_ENRICHMENT_CONTRACT_NO_TRAINING_NO_BRIDGE_REPORT.md",
    }
    for f in files:
        assert f in allowed, f"File {f} is not in the allowed list of changes for P84."
