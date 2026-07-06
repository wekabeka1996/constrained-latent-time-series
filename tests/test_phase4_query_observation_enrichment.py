# tests/test_phase4_query_observation_enrichment.py

import sys
import os
import json
import pathlib

from src.phase4.query_observation_enrichment import (
    run_p93_query_observation_enrichment_probe,
    validate_source_contracts_for_p93,
    PHASE,
    VERDICT,
    BASELINE_POLICIES,
    IMPROVED_POLICIES,
    ALL_RETRIEVAL_POLICIES,
)


def _probe():
    return run_p93_query_observation_enrichment_probe()


# ── 1. phase / verdict ────────────────────────────────────────────────────────

def test_p93_01_phase_verdict():
    r = _probe()
    assert r["phase"] == "P93"
    assert r["verdict"] == "P93_READY_FOR_REVIEW"
    assert r["json_safe"] is True
    assert r["diagnostic_only"] is True


# ── 2. scope gate ─────────────────────────────────────────────────────────────

def test_p93_02_scope_gate():
    planned = {
        "src/phase4/query_observation_enrichment.py",
        "tools/phase4/run_p93_query_observation_enrichment_smoke.py",
        "tests/test_phase4_query_observation_enrichment.py",
        "tests/test_phase4_p93_query_observation_enrichment_smoke.py",
        "reports/PHASE_4_P93_QUERY_OBSERVATION_ENRICHMENT_NO_TRAINING_NO_BRIDGE_REPORT.md",
    }
    found = []
    for folder in ["src/phase4", "tools/phase4", "tests", "reports"]:
        if os.path.exists(folder):
            for p in pathlib.Path(folder).resolve().rglob("*"):
                if p.is_file():
                    rel = str(p.relative_to(pathlib.Path(os.getcwd()).resolve())).replace("\\", "/")
                    if "__pycache__" in rel or rel.endswith(".pyc"):
                        continue
                    if "p93" in rel.lower() or rel == "src/phase4/query_observation_enrichment.py":
                        found.append(rel)
    for f in found:
        assert f in planned, f"Unexpected file: {f}"


# ── 3. no torch import ───────────────────────────────────────────────────────

def test_p93_03_no_torch_import():
    src = pathlib.Path("src/phase4/query_observation_enrichment.py").read_text(encoding="utf-8")
    assert "import torch" not in src
    assert "from torch" not in src


# ── 4. no numpy/pandas/sklearn/random ─────────────────────────────────────────

def test_p93_04_no_forbidden_imports():
    src = pathlib.Path("src/phase4/query_observation_enrichment.py").read_text(encoding="utf-8")
    assert "import numpy" not in src
    assert "import pandas" not in src
    assert "import sklearn" not in src
    assert "import random" not in src
    assert "from numpy" not in src
    assert "from pandas" not in src
    assert "from sklearn" not in src
    assert "from random" not in src


# ── 5. no model training ─────────────────────────────────────────────────────

def test_p93_05_no_training():
    r = _probe()
    assert r["model_training_performed"] is False
    assert r["torch_training_performed"] is False


# ── 6. no optimizer ───────────────────────────────────────────────────────────

def test_p93_06_no_optimizer():
    r = _probe()
    assert r["new_model_implemented"] is False
    assert r["optimizer_created"] is False


# ── 7. no checkpoint ──────────────────────────────────────────────────────────

def test_p93_07_no_checkpoint():
    r = _probe()
    assert r["checkpoint_written"] is False


# ── 8. P92 source validation ─────────────────────────────────────────────────

def test_p93_08_source_contracts():
    r = _probe()
    assert r["source_contracts_validated"] is True
    v = validate_source_contracts_for_p93()
    assert v["p92_validated"] is True


# ── 9. P92 near-threshold source similarity preserved ───────────────────────

def test_p93_09_near_threshold_similarity():
    v = validate_source_contracts_for_p93()
    assert v["p92_near_threshold_source_similarity_preserved"] is True


# ── 10. P92 retrieval signal absent preserved ───────────────────────────────

def test_p93_10_retrieval_signal_absent():
    v = validate_source_contracts_for_p93()
    assert v["p92_retrieval_signal_absent_preserved"] is True
    assert v["p92_improved_support_ready_preserved"] is True


# ── 11. enriched query records exist ─────────────────────────────────────────

def test_p93_11_enriched_query_records():
    r = _probe()
    assert r["enriched_query_record_count"] > 0


# ── 12. enriched support metadata records exist ──────────────────────────────

def test_p93_12_enriched_support_metadata():
    r = _probe()
    assert r["enriched_support_metadata_record_count"] > 0


# ── 13. enriched query audit no label/operator/target/delta/result ───────────

def test_p93_13_query_audit():
    r = _probe()
    # Check query leakage counts in audit
    la = r["enriched_policy_leakage_audits"]["enriched_source_signature_retrieval"]
    assert la["query_enrichment_uses_label_count"] == 0
    assert la["query_enrichment_uses_operator_id_count"] == 0
    assert la["query_enrichment_uses_target_count"] == 0
    assert la["query_enrichment_uses_delta_count"] == 0
    assert la["query_enrichment_uses_result_count"] == 0
    assert la["query_enrichment_uses_hidden_relation_family_count"] == 0


# ── 14. support metadata audit no label/operator/result/delta ────────────────

def test_p93_14_support_metadata_audit():
    r = _probe()
    la = r["enriched_policy_leakage_audits"]["enriched_source_signature_retrieval"]
    assert la["support_metadata_uses_label_count"] == 0
    assert la["support_metadata_uses_operator_id_count"] == 0
    assert la["support_metadata_uses_result_content_count"] == 0
    assert la["support_metadata_uses_delta_content_count"] == 0


# ── 15. enriched policies included ───────────────────────────────────────────

def test_p93_15_enriched_policies():
    r = _probe()
    for pol in IMPROVED_POLICIES:
        assert pol in r["enriched_policy_metric_results"]


# ── 16. all enriched policies are non-label-selected ─────────────────────────

def test_p93_16_non_label_selected():
    r = _probe()
    for pol in IMPROVED_POLICIES:
        la = r["enriched_policy_leakage_audits"][pol]
        assert la["non_label_selected_context_count"] == la["context_count"]


# ── 17. enriched distance does not use audit label ──────────────────────────

def test_p93_17_distance_no_label():
    src = pathlib.Path("src/phase4/query_observation_enrichment.py").read_text(encoding="utf-8")
    assert "audit_label_evaluation_only" not in src.split("def enriched_source_signature_distance")[1].split("def ")[0]


# ── 18. enriched distance does not use query target/delta ────────────────────

def test_p93_18_distance_no_query_target_delta():
    src = pathlib.Path("src/phase4/query_observation_enrichment.py").read_text(encoding="utf-8")
    func_body = src.split("def enriched_source_signature_distance")[1].split("def ")[0]
    assert "query_target" not in func_body
    assert "query_delta" not in func_body


# ── 19. enriched distance does not use support result/delta ──────────────────

def test_p93_19_distance_no_support_result_delta():
    src = pathlib.Path("src/phase4/query_observation_enrichment.py").read_text(encoding="utf-8")
    func_body = src.split("def enriched_source_signature_distance")[1].split("def ")[0]
    assert "support_result_summary" not in func_body
    assert "support_delta_summary" not in func_body


# ── 20. leakage audit passes for at least one enriched policy ────────────────

def test_p93_20_leakage_audit_passes():
    r = _probe()
    la = r["enriched_policy_leakage_audits"]["enriched_source_signature_retrieval"]
    assert la["diagnostic_pass"] is True


# ── 21. metric shape compatibility exists ─────────────────────────────────────

def test_p93_21_shape_compatibility():
    r = _probe()
    for pol in ALL_RETRIEVAL_POLICIES:
        sa = r["enriched_policy_shape_compatibility_audits"][pol]
        assert sa["diagnostic_pass"] is True


# ── 22. expected delta-only dim remains 18 ───────────────────────────────────

def test_p93_22_delta_only_dim():
    r = _probe()
    for pol in ALL_RETRIEVAL_POLICIES:
        sa = r["enriched_policy_shape_compatibility_audits"][pol]
        assert sa["expected_delta_only_dim"] == 18


# ── 23. shuffled controls preserve true labels ───────────────────────────────

def test_p93_23_shuffled_preserves_labels():
    r = _probe()
    for pol, nc in r["enriched_policy_negative_controls"].items():
        assert nc["shuffled_control_target_labels_preserved"] is True
        assert nc["true_labels_preserved"] is True


# ── 24. shuffled controls permute vectors ─────────────────────────────────────

def test_p93_24_shuffled_permutes_vectors():
    r = _probe()
    for pol, nc in r["enriched_policy_negative_controls"].items():
        assert nc["shuffled_control_support_deltas_permuted"] is True


# ── 25. target labels not rotated ─────────────────────────────────────────────

def test_p93_25_no_target_rotation():
    r = _probe()
    for pol, nc in r["enriched_policy_negative_controls"].items():
        assert nc["target_labels_shifted"] is False


# ── 26. zero-delta control exists ─────────────────────────────────────────────

def test_p93_26_zero_delta_exists():
    r = _probe()
    for pol, nc in r["enriched_policy_negative_controls"].items():
        assert "zero_delta_negative_control" in nc


# ── 27. enrichment proxy audit exists ─────────────────────────────────────────

def test_p93_27_proxy_audit_exists():
    r = _probe()
    assert r["enrichment_proxy_leakage_audit"]["audit_defined"] is True


# ── 28. query observation enrichment ready flag computed ─────────────────────

def test_p93_28_enrichment_ready_flag():
    r = _probe()
    assert isinstance(r["query_observation_enrichment_ready"], bool)


# ── 29. enriched retrieval signal flag computed from thresholds ───────────────

def test_p93_29_retrieval_signal_flag():
    r = _probe()
    assert isinstance(r["enriched_retrieval_signal_present"], bool)
    any_pass = False
    for pol in IMPROVED_POLICIES:
        if pol == "cross_domain_normalized_signature_retrieval":
            continue
        mr = r["enriched_policy_metric_results"][pol]
        nc = r["enriched_policy_negative_controls"][pol]
        la = r["enriched_policy_leakage_audits"][pol]
        sa = r["enriched_policy_shape_compatibility_audits"][pol]
        if (
            mr["effective_accuracy"] >= 0.40
            and mr["coverage"] >= 0.80
            and nc["beats_shuffled_by"] >= 0.10
            and nc["beats_zero_delta_by"] >= 0.10
            and la["diagnostic_pass"]
            and sa["diagnostic_pass"]
        ):
            any_pass = True
    assert r["enriched_retrieval_signal_present"] == any_pass


# ── 30. enriched external context metric support flag computed ───────────────

def test_p93_30_metric_supported_flag():
    r = _probe()
    assert isinstance(r["enriched_external_context_metric_supported"], bool)


# ── 31. bridge false ──────────────────────────────────────────────────────────

def test_p93_31_bridge_false():
    r = _probe()
    assert r["bridge_ready"] is False
    assert r["bridge_implementation_allowed"] is False


# ── 32. semantic geometry false ───────────────────────────────────────────────

def test_p93_32_semantic_geometry_false():
    r = _probe()
    assert r["semantic_geometry_claims_allowed"] is False
    assert r["learned_metric_evidence_present"] is False
    assert r["semantic_metric_ready"] is False


# ── 33. JSON serializable ────────────────────────────────────────────────────

def test_p93_33_json_serializable():
    r = _probe()
    s = json.dumps(r)
    assert isinstance(s, str)
