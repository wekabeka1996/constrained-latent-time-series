# tests/test_phase4_external_support_context_builder.py

import sys
import os
import json
import pytest

from src.phase4.external_support_context_builder import (
    run_p90_external_support_context_builder_probe,
    validate_source_contracts_for_p90,
    build_external_demonstration_records,
    build_external_query_records,
    build_external_support_context_for_query,
    audit_external_support_selection_leakage,
    encode_external_support_delta_context,
    audit_metric_shape_compatibility,
    PHASE,
    VERDICT
)

from src.phase4.support_pair_materializer_from_p70_relation_cases import (
    build_selector_dataset_records
)
from src.phase3.pure_numeric_relation_testbed import (
    run_p70a_pure_numeric_relation_testbed_probe
)
from src.phase3.synthetic_time_series_relation_testbed import (
    run_p70b_synthetic_time_series_relation_testbed_probe
)


def test_p90_01_probe_constants():
    res = run_p90_external_support_context_builder_probe()
    assert res["phase"] == PHASE
    assert res["verdict"] == VERDICT
    assert res["json_safe"] is True
    assert res["diagnostic_only"] is True


def test_p90_02_no_torch_import():
    # Assert torch is not in sys.modules, and check source file content
    assert "torch" not in sys.modules
    
    src_path = "src/phase4/external_support_context_builder.py"
    with open(src_path, "r", encoding="utf-8") as f:
        content = f.read()
    assert "import torch" not in content
    assert "from torch" not in content


def test_p90_03_no_training_performed():
    res = run_p90_external_support_context_builder_probe()
    assert res["model_training_performed"] is False
    assert res["torch_training_performed"] is False
    assert res["new_model_implemented"] is False
    assert res["optimizer_created"] is False
    assert res["checkpoint_written"] is False


def test_p90_04_source_contracts_and_reproduction():
    res = run_p90_external_support_context_builder_probe()
    assert res["source_contracts_validated"] is True
    
    # Assert P89 validations pass
    v = validate_source_contracts_for_p90()
    assert v["p89_validated"] is True
    assert v["p89_hard_generalization_preserved"] is True
    assert v["p89_aligned_controls_preserved"] is True


def test_p90_05_demonstration_and_query_schemas():
    p70a = run_p70a_pure_numeric_relation_testbed_probe()
    p70b = run_p70b_synthetic_time_series_relation_testbed_probe()
    
    demo_recs = build_external_demonstration_records(p70a, p70b)
    assert len(demo_recs) > 0
    
    # Schema check for demo record
    first_demo = demo_recs[0]
    assert "external_demo_id" in first_demo
    assert "domain" in first_demo
    assert "source_split" in first_demo
    assert "observable_context" in first_demo
    assert "selection_visible_metadata" in first_demo
    assert "audit_label_evaluation_only" in first_demo
    
    # Verify that labels are not used outside audit_label_evaluation_only
    obs_ctx_str = json.dumps(first_demo["observable_context"])
    assert first_demo["audit_label_evaluation_only"]["target_relation_label"] not in obs_ctx_str
    
    # Query schema check
    selector_records = build_selector_dataset_records(p70a, p70b)
    query_recs = build_external_query_records(selector_records)
    assert len(query_recs) > 0
    first_query = query_recs[0]
    assert "external_query_id" in first_query
    assert "dataset_record_id" in first_query
    assert "query_observation" in first_query
    assert "audit_label_evaluation_only" in first_query
    assert first_query["query_target_hidden"] is True
    assert first_query["query_delta_hidden"] is True


def test_p90_06_support_policies():
    p70a = run_p70a_pure_numeric_relation_testbed_probe()
    p70b = run_p70b_synthetic_time_series_relation_testbed_probe()
    demo_recs = build_external_demonstration_records(p70a, p70b)
    selector_records = build_selector_dataset_records(p70a, p70b)
    query_recs = build_external_query_records(selector_records)
    
    q = query_recs[0]
    
    # Oracle policy (uses label, diagnostic only)
    oracle_ctx = build_external_support_context_for_query(q, demo_recs, "label_selected_oracle_support")
    assert oracle_ctx["support_selection_audit"]["support_selection_uses_hidden_relation_label"] is True
    assert oracle_ctx["support_selection_audit"]["support_selection_non_label_selected"] is False
    assert oracle_ctx["support_selection_audit"]["valid_for_p91_external_context_metric_evaluation"] is False
    
    # Observable retrieval policy
    obs_ctx = build_external_support_context_for_query(q, demo_recs, "observable_domain_split_retrieval")
    assert obs_ctx["support_selection_audit"]["support_selection_uses_hidden_relation_label"] is False
    assert obs_ctx["support_selection_audit"]["support_selection_non_label_selected"] is True
    assert obs_ctx["support_selection_audit"]["valid_for_p91_external_context_metric_evaluation"] is True
    
    # Manifest policy
    man_ctx = build_external_support_context_for_query(q, demo_recs, "external_manifest_support")
    assert man_ctx["support_selection_audit"]["support_selection_uses_hidden_relation_label"] is False
    assert man_ctx["support_selection_audit"]["support_selection_non_label_selected"] is True
    assert man_ctx["support_selection_audit"]["valid_for_p91_external_context_metric_evaluation"] is True


def test_p90_07_leakage_and_audit():
    p70a = run_p70a_pure_numeric_relation_testbed_probe()
    p70b = run_p70b_synthetic_time_series_relation_testbed_probe()
    demo_recs = build_external_demonstration_records(p70a, p70b)
    selector_records = build_selector_dataset_records(p70a, p70b)
    query_recs = build_external_query_records(selector_records)
    
    # Build contexts for all queries under observable policy
    contexts = [build_external_support_context_for_query(q, demo_recs, "observable_domain_split_retrieval") for q in query_recs]
    leakage = audit_external_support_selection_leakage(contexts)
    
    assert leakage["hidden_relation_label_used_count"] == 0
    assert leakage["query_target_used_count"] == 0
    assert leakage["query_delta_used_count"] == 0
    assert leakage["operator_id_used_count"] == 0
    assert leakage["label_visible_to_selector_count"] == 0
    assert leakage["diagnostic_pass"] is True


def test_p90_08_metric_shape_compatibility():
    p70a = run_p70a_pure_numeric_relation_testbed_probe()
    p70b = run_p70b_synthetic_time_series_relation_testbed_probe()
    demo_recs = build_external_demonstration_records(p70a, p70b)
    selector_records = build_selector_dataset_records(p70a, p70b)
    query_recs = build_external_query_records(selector_records)
    
    contexts = [build_external_support_context_for_query(q, demo_recs, "observable_domain_split_retrieval") for q in query_recs]
    
    # Check individual vector encoding
    vec = encode_external_support_delta_context(contexts[0])
    assert len(vec) == 18
    assert all(isinstance(x, float) for x in vec)
    
    # Check collective shape audit
    shape_audit = audit_metric_shape_compatibility(contexts)
    assert shape_audit["expected_delta_only_dim"] == 18
    assert shape_audit["incompatible_context_count"] == 0
    assert shape_audit["all_contexts_metric_shape_compatible"] is True
    assert shape_audit["diagnostic_pass"] is True


def test_p90_09_readiness_and_bridge():
    res = run_p90_external_support_context_builder_probe()
    assert res["external_context_builder_ready"] is True
    assert res["non_label_selected_support_ready"] is True
    assert res["bridge_ready"] is False
    assert res["semantic_geometry_claims_allowed"] is False
    assert res["learned_selector_evidence_present"] is False
    assert res["learned_metric_evidence_present"] is False
    assert res["semantic_metric_ready"] is False


def test_p90_10_scope_gate():
    # Enforce exactly 5 planned files are created/modified
    planned_files = {
        "src/phase4/external_support_context_builder.py",
        "tools/phase4/run_p90_external_support_context_builder_smoke.py",
        "tests/test_phase4_external_support_context_builder.py",
        "tests/test_phase4_p90_external_support_context_builder_smoke.py",
        "reports/PHASE_4_P90_EXTERNAL_SUPPORT_CONTEXT_BUILDER_NO_TRAINING_NO_BRIDGE_REPORT.md",
    }
    
    # Verify that only these files exist under src/phase4/, tools/phase4/, and tests/ containing p90
    import pathlib
    paths = []
    for folder in ["src/phase4", "tools/phase4", "tests", "reports"]:
        if os.path.exists(folder):
            for p in pathlib.Path(folder).resolve().rglob("*"):
                if p.is_file():
                    rel_p = str(p.relative_to(pathlib.Path(os.getcwd()).resolve())).replace("\\", "/")
                    if "__pycache__" in rel_p or rel_p.endswith(".pyc"):
                        continue
                    if "p90" in rel_p.lower() or "external_support" in rel_p.lower():
                        paths.append(rel_p)
                        
    for p in paths:
        assert p in planned_files, f"Unexpected file created in codebase: {p}"
