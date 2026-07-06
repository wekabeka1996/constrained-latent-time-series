# tests/test_phase4_non_label_support_retrieval_improvement.py

import sys
import os
import json
import pathlib

from src.phase4.non_label_support_retrieval_improvement import (
    run_p92_non_label_support_retrieval_improvement_probe,
    validate_source_contracts_for_p92,
    PHASE,
    VERDICT,
    BASELINE_POLICIES,
    IMPROVED_POLICIES,
    ALL_RETRIEVAL_POLICIES,
)


def _probe():
    return run_p92_non_label_support_retrieval_improvement_probe()


# ── 1. phase / verdict ────────────────────────────────────────────────────────

def test_p92_01_phase_verdict():
    r = _probe()
    assert r["phase"] == "P92"
    assert r["verdict"] == "P92_READY_FOR_REVIEW"
    assert r["json_safe"] is True
    assert r["diagnostic_only"] is True


# ── 2. scope gate ─────────────────────────────────────────────────────────────

def test_p92_02_scope_gate():
    planned = {
        "src/phase4/non_label_support_retrieval_improvement.py",
        "tools/phase4/run_p92_non_label_support_retrieval_improvement_smoke.py",
        "tests/test_phase4_non_label_support_retrieval_improvement.py",
        "tests/test_phase4_p92_non_label_support_retrieval_improvement_smoke.py",
        "reports/PHASE_4_P92_NON_LABEL_SELECTED_SUPPORT_RETRIEVAL_IMPROVEMENT_NO_TRAINING_NO_BRIDGE_REPORT.md",
    }
    found = []
    for folder in ["src/phase4", "tools/phase4", "tests", "reports"]:
        if os.path.exists(folder):
            for p in pathlib.Path(folder).resolve().rglob("*"):
                if p.is_file():
                    rel = str(p.relative_to(pathlib.Path(os.getcwd()).resolve())).replace("\\", "/")
                    if "__pycache__" in rel or rel.endswith(".pyc"):
                        continue
                    if "p92" in rel.lower() or "non_label_support_retrieval" in rel.lower():
                        found.append(rel)
    for f in found:
        assert f in planned, f"Unexpected file: {f}"


# ── 3. no torch import ───────────────────────────────────────────────────────

def test_p92_03_no_torch_import():
    src = pathlib.Path("src/phase4/non_label_support_retrieval_improvement.py").read_text(encoding="utf-8")
    assert "import torch" not in src
    assert "from torch" not in src


# ── 4. no numpy/pandas/sklearn/random ─────────────────────────────────────────

def test_p92_04_no_forbidden_imports():
    src = pathlib.Path("src/phase4/non_label_support_retrieval_improvement.py").read_text(encoding="utf-8")
    assert "import numpy" not in src
    assert "import pandas" not in src
    assert "import sklearn" not in src
    assert "import random" not in src
    assert "from numpy" not in src
    assert "from pandas" not in src
    assert "from sklearn" not in src
    assert "from random" not in src


# ── 5. no model training ─────────────────────────────────────────────────────

def test_p92_05_no_training():
    r = _probe()
    assert r["model_training_performed"] is False
    assert r["torch_training_performed"] is False


# ── 6. no optimizer ───────────────────────────────────────────────────────────

def test_p92_06_no_optimizer():
    r = _probe()
    assert r["new_model_implemented"] is False
    assert r["optimizer_created"] is False


# ── 7. no checkpoint ──────────────────────────────────────────────────────────

def test_p92_07_no_checkpoint():
    r = _probe()
    assert r["checkpoint_written"] is False


# ── 8. P91 source validation ─────────────────────────────────────────────────

def test_p92_08_source_contracts():
    r = _probe()
    assert r["source_contracts_validated"] is True
    v = validate_source_contracts_for_p92()
    assert v["p91_validated"] is True


# ── 9. P91 non-label collapse preserved ───────────────────────────────────────

def test_p92_09_non_label_collapse_preserved():
    v = validate_source_contracts_for_p92()
    assert v["p91_non_label_collapse_preserved"] is True
    assert v["p91_oracle_signal_preserved"] is True


# ── 10. baseline policies included ───────────────────────────────────────────

def test_p92_10_baseline_policies():
    r = _probe()
    for pol in BASELINE_POLICIES:
        assert pol in r["retrieval_policy_metric_results"]


# ── 11. improved policies included ───────────────────────────────────────────

def test_p92_11_improved_policies():
    r = _probe()
    for pol in IMPROVED_POLICIES:
        assert pol in r["retrieval_policy_metric_results"]


# ── 12. all improved policies marked non-label-selected ──────────────────────

def test_p92_12_improved_non_label_selected():
    r = _probe()
    for pol in IMPROVED_POLICIES:
        la = r["retrieval_policy_leakage_audits"][pol]
        assert la["non_label_selected_context_count"] == la["context_count"]


# ── 13. hidden labels not used ───────────────────────────────────────────────

def test_p92_13_hidden_labels_not_used():
    r = _probe()
    for pol in IMPROVED_POLICIES:
        la = r["retrieval_policy_leakage_audits"][pol]
        assert la["hidden_relation_label_used_count"] == 0


# ── 14. operator IDs not used ────────────────────────────────────────────────

def test_p92_14_operator_ids_not_used():
    r = _probe()
    for pol in IMPROVED_POLICIES:
        la = r["retrieval_policy_leakage_audits"][pol]
        assert la["operator_id_used_count"] == 0


# ── 15. query target not used ────────────────────────────────────────────────

def test_p92_15_query_target_not_used():
    r = _probe()
    for pol in IMPROVED_POLICIES:
        la = r["retrieval_policy_leakage_audits"][pol]
        assert la["query_target_used_count"] == 0


# ── 16. query delta not used ─────────────────────────────────────────────────

def test_p92_16_query_delta_not_used():
    r = _probe()
    for pol in IMPROVED_POLICIES:
        la = r["retrieval_policy_leakage_audits"][pol]
        assert la["query_delta_used_count"] == 0


# ── 17. support result content not used ───────────────────────────────────────

def test_p92_17_support_result_not_used():
    r = _probe()
    for pol in IMPROVED_POLICIES:
        la = r["retrieval_policy_leakage_audits"][pol]
        assert la["support_result_content_used_for_selection_count"] == 0


# ── 18. support delta content not used ────────────────────────────────────────

def test_p92_18_support_delta_not_used():
    r = _probe()
    for pol in IMPROVED_POLICIES:
        la = r["retrieval_policy_leakage_audits"][pol]
        assert la["support_delta_content_used_for_selection_count"] == 0


# ── 19. support delta shape metadata audited ──────────────────────────────────

def test_p92_19_delta_shape_metadata_audited():
    r = _probe()
    for pol in IMPROVED_POLICIES:
        la = r["retrieval_policy_leakage_audits"][pol]
        assert "support_delta_shape_metadata_used_count" in la


# ── 20. metric shape compatibility exists ─────────────────────────────────────

def test_p92_20_shape_compatibility():
    r = _probe()
    for pol in ALL_RETRIEVAL_POLICIES:
        sa = r["retrieval_policy_shape_compatibility_audits"][pol]
        assert sa["diagnostic_pass"] is True


# ── 21. shuffled controls preserve true labels ───────────────────────────────

def test_p92_21_shuffled_preserves_labels():
    r = _probe()
    for pol, nc in r["retrieval_policy_negative_controls"].items():
        assert nc["shuffled_control_target_labels_preserved"] is True
        assert nc["true_labels_preserved"] is True


# ── 22. shuffled controls permute vectors ─────────────────────────────────────

def test_p92_22_shuffled_permutes_vectors():
    r = _probe()
    for pol, nc in r["retrieval_policy_negative_controls"].items():
        assert nc["shuffled_control_support_deltas_permuted"] is True


# ── 23. target labels not rotated ─────────────────────────────────────────────

def test_p92_23_no_target_rotation():
    r = _probe()
    for pol, nc in r["retrieval_policy_negative_controls"].items():
        assert nc["target_labels_shifted"] is False


# ── 24. zero-delta control exists ─────────────────────────────────────────────

def test_p92_24_zero_delta_exists():
    r = _probe()
    for pol, nc in r["retrieval_policy_negative_controls"].items():
        assert "zero_delta_negative_control" in nc


# ── 25. retrieval improvement flag computed from thresholds ───────────────────

def test_p92_25_improvement_flag_computed():
    r = _probe()
    assert isinstance(r["retrieval_improvement_signal_present"], bool)
    # Must be computed, not hardcoded—verify against actual results
    any_pass = False
    for pol in IMPROVED_POLICIES:
        mr = r["retrieval_policy_metric_results"][pol]
        nc = r["retrieval_policy_negative_controls"][pol]
        la = r["retrieval_policy_leakage_audits"][pol]
        sa = r["retrieval_policy_shape_compatibility_audits"][pol]
        if (
            mr["effective_accuracy"] >= 0.40
            and mr["coverage"] >= 0.80
            and nc["beats_shuffled_by"] >= 0.10
            and nc["beats_zero_delta_by"] >= 0.10
            and la["diagnostic_pass"]
            and sa["diagnostic_pass"]
        ):
            any_pass = True
    assert r["retrieval_improvement_signal_present"] == any_pass


# ── 26. improved support ready flag computed ──────────────────────────────────

def test_p92_26_improved_ready_computed():
    r = _probe()
    assert isinstance(r["improved_non_label_selected_support_ready"], bool)


# ── 27. bridge false ──────────────────────────────────────────────────────────

def test_p92_27_bridge_false():
    r = _probe()
    assert r["bridge_ready"] is False
    assert r["bridge_implementation_allowed"] is False


# ── 28. semantic geometry false ───────────────────────────────────────────────

def test_p92_28_semantic_geometry_false():
    r = _probe()
    assert r["semantic_geometry_claims_allowed"] is False
    assert r["learned_metric_evidence_present"] is False
    assert r["semantic_metric_ready"] is False


# ── 29. JSON serializable ────────────────────────────────────────────────────

def test_p92_29_json_serializable():
    r = _probe()
    s = json.dumps(r)
    assert isinstance(s, str)
