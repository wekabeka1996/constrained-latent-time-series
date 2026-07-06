# tests/test_phase4_external_context_metric_evaluation.py

import sys
import os
import json
import pathlib

from src.phase4.external_context_metric_evaluation import (
    run_p91_external_context_metric_evaluation_probe,
    validate_source_contracts_for_p91,
    PHASE,
    VERDICT,
)


def _probe():
    return run_p91_external_context_metric_evaluation_probe()


# ── 1. phase / verdict ────────────────────────────────────────────────────────

def test_p91_01_phase_verdict():
    r = _probe()
    assert r["phase"] == "P91"
    assert r["verdict"] == "P91_READY_FOR_REVIEW"
    assert r["json_safe"] is True
    assert r["diagnostic_only"] is True


# ── 2. scope gate ─────────────────────────────────────────────────────────────

def test_p91_02_scope_gate():
    planned = {
        "src/phase4/external_context_metric_evaluation.py",
        "tools/phase4/run_p91_external_context_metric_evaluation_smoke.py",
        "tests/test_phase4_external_context_metric_evaluation.py",
        "tests/test_phase4_p91_external_context_metric_evaluation_smoke.py",
        "reports/PHASE_4_P91_EXTERNAL_CONTEXT_METRIC_EVALUATION_NO_TRAINING_NO_BRIDGE_REPORT.md",
    }
    found = []
    for folder in ["src/phase4", "tools/phase4", "tests", "reports"]:
        if os.path.exists(folder):
            for p in pathlib.Path(folder).resolve().rglob("*"):
                if p.is_file():
                    rel = str(p.relative_to(pathlib.Path(os.getcwd()).resolve())).replace("\\", "/")
                    if "__pycache__" in rel or rel.endswith(".pyc"):
                        continue
                    if "p91" in rel.lower() or "external_context_metric" in rel.lower():
                        found.append(rel)
    for f in found:
        assert f in planned, f"Unexpected file: {f}"


# ── 3. no torch import ───────────────────────────────────────────────────────

def test_p91_03_no_torch_import():
    src = pathlib.Path("src/phase4/external_context_metric_evaluation.py").read_text(encoding="utf-8")
    assert "import torch" not in src
    assert "from torch" not in src


# ── 4–6. no training / optimizer / checkpoint ─────────────────────────────────

def test_p91_04_no_training():
    r = _probe()
    assert r["model_training_performed"] is False
    assert r["torch_training_performed"] is False


def test_p91_05_no_optimizer():
    r = _probe()
    assert r["new_model_implemented"] is False
    assert r["optimizer_created"] is False


def test_p91_06_no_checkpoint():
    r = _probe()
    assert r["checkpoint_written"] is False


# ── 7. P90 source validation ─────────────────────────────────────────────────

def test_p91_07_source_contracts():
    r = _probe()
    assert r["source_contracts_validated"] is True
    v = validate_source_contracts_for_p91()
    assert v["p90_validated"] is True
    assert v["p90_external_context_builder_ready_preserved"] is True
    assert v["p90_non_label_selected_support_ready_preserved"] is True


# ── 8. all three policies present ────────────────────────────────────────────

def test_p91_08_policy_results_present():
    r = _probe()
    pmr = r["policy_metric_results"]
    assert "label_selected_oracle_support" in pmr
    assert "observable_domain_split_retrieval" in pmr
    assert "external_manifest_support" in pmr


# ── 9. oracle is diagnostic only ─────────────────────────────────────────────

def test_p91_09_oracle_diagnostic():
    r = _probe()
    oracle = r["policy_metric_results"]["label_selected_oracle_support"]
    assert oracle["diagnostic_only"] is True
    assert oracle["valid_for_final_semantic_geometry_evidence"] is False


# ── 10–11. leakage clean for non-label-selected ──────────────────────────────

def test_p91_10_observable_leakage_clean():
    r = _probe()
    lv = r["leakage_verification"]["observable_domain_split_retrieval"]
    assert lv["hidden_relation_label_used_count"] == 0
    assert lv["query_target_used_count"] == 0
    assert lv["query_delta_used_count"] == 0
    assert lv["diagnostic_pass"] is True


def test_p91_11_manifest_leakage_clean():
    r = _probe()
    lv = r["leakage_verification"]["external_manifest_support"]
    assert lv["hidden_relation_label_used_count"] == 0
    assert lv["diagnostic_pass"] is True


# ── 12–16. negative controls ─────────────────────────────────────────────────

def test_p91_12_negative_controls_exist():
    r = _probe()
    for pol in ["label_selected_oracle_support", "observable_domain_split_retrieval", "external_manifest_support"]:
        assert pol in r["policy_negative_control_results"]


def test_p91_13_shuffled_preserves_labels():
    r = _probe()
    for pol, nc in r["policy_negative_control_results"].items():
        assert nc["shuffled_control_target_labels_preserved"] is True
        assert nc["true_labels_preserved"] is True


def test_p91_14_shuffled_permutes_vectors():
    r = _probe()
    for pol, nc in r["policy_negative_control_results"].items():
        assert nc["shuffled_control_support_deltas_permuted"] is True


def test_p91_15_no_target_rotation():
    r = _probe()
    for pol, nc in r["policy_negative_control_results"].items():
        assert nc["target_labels_shifted"] is False


def test_p91_16_zero_delta_exists():
    r = _probe()
    for pol, nc in r["policy_negative_control_results"].items():
        assert "zero_delta_negative_control" in nc


# ── 17–18. proxy leakage audit ───────────────────────────────────────────────

def test_p91_17_proxy_audit_exists():
    r = _probe()
    pa = r["proxy_leakage_audit"]
    assert pa["audit_defined"] is True
    assert "metadata_field_results" in pa
    assert "observable_source_shape" in pa["metadata_field_results"]


def test_p91_18_proxy_risk_reported():
    r = _probe()
    pa = r["proxy_leakage_audit"]
    assert "high_proxy_risk_fields" in pa
    assert isinstance(pa["high_proxy_risk_fields"], list)


# ── 19. unsupported classes handled ──────────────────────────────────────────

def test_p91_19_unsupported_no_default():
    r = _probe()
    for pol in ["observable_domain_split_retrieval", "external_manifest_support"]:
        preds = r["policy_metric_predictions"][pol]
        for p in preds:
            if p["unsupported"]:
                assert p["predicted_label"] is None
                assert p["is_correct"] is False


# ── 20–21. signal flags computed ─────────────────────────────────────────────

def test_p91_20_signal_flag_computed():
    r = _probe()
    assert isinstance(r["non_label_selected_metric_signal_present"], bool)


def test_p91_21_metric_supported_flag_computed():
    r = _probe()
    assert isinstance(r["external_context_metric_supported"], bool)


# ── 22–23. bridge / semantic geometry false ──────────────────────────────────

def test_p91_22_bridge_false():
    r = _probe()
    assert r["bridge_ready"] is False
    assert r["bridge_implementation_allowed"] is False


def test_p91_23_semantic_geometry_false():
    r = _probe()
    assert r["semantic_geometry_claims_allowed"] is False
    assert r["learned_metric_evidence_present"] is False
    assert r["semantic_metric_ready"] is False


# ── 24. JSON serializable ────────────────────────────────────────────────────

def test_p91_24_json_serializable():
    r = _probe()
    s = json.dumps(r)
    assert isinstance(s, str)
