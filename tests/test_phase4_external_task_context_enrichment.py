# tests/test_phase4_external_task_context_enrichment.py

import sys
import os
import json
import pathlib

from src.phase4.external_task_context_enrichment import (
    run_p94_external_task_context_enrichment_probe,
    validate_source_contracts_for_p94,
    PHASE,
    VERDICT,
    ALL_P94_POLICIES,
)


def _probe():
    return run_p94_external_task_context_enrichment_probe()


# ── 1. phase / verdict ────────────────────────────────────────────────────────

def test_p94_01_phase_verdict():
    r = _probe()
    assert r["phase"] == "P94"
    assert r["verdict"] == "P94_READY_FOR_REVIEW"
    assert r["json_safe"] is True
    assert r["diagnostic_only"] is True


# ── 2. scope gate ─────────────────────────────────────────────────────────────

def test_p94_02_scope_gate():
    planned = {
        "src/phase4/external_task_context_enrichment.py",
        "tools/phase4/run_p94_external_task_context_enrichment_smoke.py",
        "tests/test_phase4_external_task_context_enrichment.py",
        "tests/test_phase4_p94_external_task_context_enrichment_smoke.py",
        "reports/PHASE_4_P94_EXTERNAL_TASK_CONTEXT_OR_SUPPORT_DEMONSTRATION_ENRICHMENT_NO_TRAINING_NO_BRIDGE_REPORT.md",
    }
    found = []
    for folder in ["src/phase4", "tools/phase4", "tests", "reports"]:
        if os.path.exists(folder):
            for p in pathlib.Path(folder).resolve().rglob("*"):
                if p.is_file():
                    rel = str(p.relative_to(pathlib.Path(os.getcwd()).resolve())).replace("\\", "/")
                    if "__pycache__" in rel or rel.endswith(".pyc"):
                        continue
                    if "p94" in rel.lower() or "external_task_context" in rel.lower():
                        found.append(rel)
    for f in found:
        assert f in planned, f"Unexpected file: {f}"


# ── 3. no torch import ───────────────────────────────────────────────────────

def test_p94_03_no_torch_import():
    src = pathlib.Path("src/phase4/external_task_context_enrichment.py").read_text(encoding="utf-8")
    assert "import torch" not in src
    assert "from torch" not in src


# ── 4. no numpy/pandas/sklearn/random ─────────────────────────────────────────

def test_p94_04_no_forbidden_imports():
    src = pathlib.Path("src/phase4/external_task_context_enrichment.py").read_text(encoding="utf-8")
    assert "import numpy" not in src
    assert "import pandas" not in src
    assert "import sklearn" not in src
    assert "import random" not in src


# ── 5. no model training ─────────────────────────────────────────────────────

def test_p94_05_no_training():
    r = _probe()
    assert r["model_training_performed"] is False
    assert r["torch_training_performed"] is False


# ── 6. no optimizer ───────────────────────────────────────────────────────────

def test_p94_06_no_optimizer():
    r = _probe()
    assert r["new_model_implemented"] is False
    assert r["optimizer_created"] is False


# ── 7. no checkpoint ──────────────────────────────────────────────────────────

def test_p94_07_no_checkpoint():
    r = _probe()
    assert r["checkpoint_written"] is False


# ── 8. P93 repaired source validation ─────────────────────────────────────────

def test_p94_08_source_contracts():
    r = _probe()
    assert r["source_contracts_validated"] is True
    v = validate_source_contracts_for_p94()
    assert v["p93_validated"] is True


# ── 9. P93 repaired baseline preserved ────────────────────────────────────────

def test_p94_09_repaired_baseline_preserved():
    v = validate_source_contracts_for_p94()
    assert v["p93_repaired_baseline_preserved"] is True
    assert v["p93_source_side_ceiling_preserved"] is True


# ── 10. external task context records exist ───────────────────────────────────

def test_p94_10_task_context_records_exist():
    r = _probe()
    assert r["external_task_context_record_count"] > 0


# ── 11. support demonstration context records exist ───────────────────────────

def test_p94_11_support_demo_context_records_exist():
    r = _probe()
    assert r["support_demonstration_context_record_count"] > 0


# ── 12–13. diagnostic oracle policies marked diagnostic only and excluded ─────

def test_p94_12_diagnostic_oracle_policies():
    r = _probe()
    for pol in ("diagnostic_oracle_task_context_retrieval", "diagnostic_support_delta_oracle_retrieval"):
        res = r["task_context_policy_metric_results"][pol]
        assert res["diagnostic_only"] is True
        assert res["valid_for_external_metric_evidence"] is False
    
    for pol in ("support_demo_context_without_query_task_card", "support_demo_context_with_non_label_task_manifest"):
        res = r["task_context_policy_metric_results"][pol]
        assert res["valid_for_external_metric_evidence"] is False
        assert res["diagnostic_only"] is True


def test_p94_13_oracle_excluded_from_evidence():
    r = _probe()
    # If oracle policies were used, task_context_signal_present might be True since oracle achieves 66.67%.
    # But because they are excluded, the flag is False.
    assert r["task_context_signal_present"] is False


# ── 14. non-label manifest policy uses no hidden labels ───────────────────────

def test_p94_14_non_label_uses_no_labels():
    r = _probe()
    la = r["task_context_policy_leakage_audits"]["non_label_external_task_manifest_retrieval"]
    assert la["hidden_relation_label_used_count"] == 0
    assert la["operator_id_used_count"] == 0


# ── 15. non-label manifest policy uses no query target/delta ──────────────────

def test_p94_15_non_label_uses_no_query_target_delta():
    r = _probe()
    la = r["task_context_policy_leakage_audits"]["non_label_external_task_manifest_retrieval"]
    assert la["query_target_used_count"] == 0
    assert la["query_delta_used_count"] == 0


# ── 16. support demo context schema exists ────────────────────────────────────

def test_p94_16_support_demo_schema():
    src = pathlib.Path("src/phase4/external_task_context_enrichment.py").read_text(encoding="utf-8")
    assert "support_source_descriptor" in src
    assert "support_result_descriptor" in src
    assert "support_delta_descriptor" in src
    assert "support_invariant_descriptor" in src


# ── 17. support demo context content use audited ──────────────────────────────

def test_p94_17_support_demo_context_audited():
    r = _probe()
    la = r["task_context_policy_leakage_audits"]["support_demo_context_without_query_task_card"]
    assert "support_result_content_used_for_selection_count" in la
    assert "support_delta_content_used_for_selection_count" in la
    assert "support_delta_shape_metadata_used_for_selection_count" in la


# ── 18. leakage audit exists ──────────────────────────────────────────────────

def test_p94_18_leakage_audit_exists():
    r = _probe()
    assert len(r["task_context_policy_leakage_audits"]) > 0


# ── 19. proxy audit exists ────────────────────────────────────────────────────

def test_p94_19_proxy_audit_exists():
    r = _probe()
    pa = r["task_context_proxy_leakage_audit"]
    assert pa["audit_defined"] is True


# ── 20. metric shape compatibility exists ─────────────────────────────────────

def test_p94_20_shape_compatibility():
    r = _probe()
    for pol in ALL_P94_POLICIES:
        sa = r["task_context_policy_shape_compatibility_audits"][pol]
        assert sa["diagnostic_pass"] is True


# ── 21. shuffled controls preserve true labels ───────────────────────────────

def test_p94_21_shuffled_preserves_labels():
    r = _probe()
    for pol, nc in r["task_context_policy_negative_controls"].items():
        assert nc["shuffled_control_target_labels_preserved"] is True
        assert nc["true_labels_preserved"] is True


# ── 22. shuffled controls permute vectors ─────────────────────────────────────

def test_p94_22_shuffled_permutes_vectors():
    r = _probe()
    for pol, nc in r["task_context_policy_negative_controls"].items():
        assert nc["shuffled_control_support_deltas_permuted"] is True


# ── 23. target labels not rotated ─────────────────────────────────────────────

def test_p94_23_no_target_rotation():
    r = _probe()
    for pol, nc in r["task_context_policy_negative_controls"].items():
        assert nc["target_labels_shifted"] is False


# ── 24. valid external task context flag computed ─────────────────────────────

def test_p94_24_valid_etc_computed():
    r = _probe()
    assert isinstance(r["valid_external_task_context_available"], bool)


# ── 25. support demonstration context ready flag computed ─────────────────────

def test_p94_25_sdc_ready_computed():
    r = _probe()
    assert isinstance(r["support_demonstration_context_ready"], bool)


# ── 26. task context signal flag computed from thresholds ─────────────────────

def test_p94_26_signal_flag_computed():
    r = _probe()
    assert isinstance(r["task_context_signal_present"], bool)
    # Excludes oracle and invalid evidence policies
    any_pass = False
    for pol in ALL_P94_POLICIES:
        mr = r["task_context_policy_metric_results"][pol]
        nc = r["task_context_policy_negative_controls"][pol]
        la = r["task_context_policy_leakage_audits"][pol]
        sa = r["task_context_policy_shape_compatibility_audits"][pol]
        if not mr["valid_for_external_metric_evidence"]:
            continue
        if (
            mr["effective_accuracy"] >= 0.40
            and mr["coverage"] >= 0.80
            and nc["beats_shuffled_by"] >= 0.10
            and nc["beats_zero_delta_by"] >= 0.10
            and la["diagnostic_pass"]
            and sa["diagnostic_pass"]
        ):
            any_pass = True
    assert r["task_context_signal_present"] == any_pass


# ── 27. task context external metric supported flag computed ──────────────────

def test_p94_27_metric_supported_flag():
    r = _probe()
    assert isinstance(r["task_context_external_metric_supported"], bool)


# ── 28. bridge false ──────────────────────────────────────────────────────────

def test_p94_28_bridge_false():
    r = _probe()
    assert r["bridge_ready"] is False
    assert r["bridge_implementation_allowed"] is False


# ── 29. semantic geometry false ───────────────────────────────────────────────

def test_p94_29_semantic_geometry_false():
    r = _probe()
    assert r["semantic_geometry_claims_allowed"] is False
    assert r["learned_metric_evidence_present"] is False
    assert r["semantic_metric_ready"] is False


# ── 30. JSON serializable ────────────────────────────────────────────────────

def test_p94_30_json_serializable():
    r = _probe()
    s = json.dumps(r)
    assert isinstance(s, str)
