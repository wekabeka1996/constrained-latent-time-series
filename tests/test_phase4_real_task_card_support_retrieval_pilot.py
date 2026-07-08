# tests/test_phase4_real_task_card_support_retrieval_pilot.py

import sys
import os
import json
import pathlib

from src.phase4.real_task_card_support_retrieval_pilot import (
    run_p96_real_task_card_support_retrieval_pilot_probe,
    validate_source_contracts_for_p96,
    tokenize_public_text,
    token_overlap_distance,
    preserve_constraint_distance,
    task_card_public_distance,
    PHASE,
    VERDICT,
    ALL_P96_POLICIES,
)


def _probe():
    return run_p96_real_task_card_support_retrieval_pilot_probe()


# ── 1. phase / verdict ────────────────────────────────────────────────────────

def test_p96_01_phase_verdict():
    r = _probe()
    assert r["phase"] == "P96"
    assert r["verdict"] == "P96_READY_FOR_REVIEW"
    assert r["json_safe"] is True
    assert r["diagnostic_only"] is True


# ── 2. scope gate ─────────────────────────────────────────────────────────────

def test_p96_02_scope_gate():
    planned = {
        "src/phase4/real_task_card_support_retrieval_pilot.py",
        "tools/phase4/run_p96_real_task_card_support_retrieval_pilot_smoke.py",
        "tests/test_phase4_real_task_card_support_retrieval_pilot.py",
        "tests/test_phase4_p96_real_task_card_support_retrieval_pilot_smoke.py",
        "reports/PHASE_4_P96_REAL_TASK_CARD_SUPPORT_RETRIEVAL_PILOT_NO_TRAINING_NO_BRIDGE_REPORT.md",
    }
    found = []
    for folder in ["src/phase4", "tools/phase4", "tests", "reports"]:
        if os.path.exists(folder):
            for p in pathlib.Path(folder).resolve().rglob("*"):
                if p.is_file():
                    rel = str(p.relative_to(pathlib.Path(os.getcwd()).resolve())).replace("\\", "/")
                    if "__pycache__" in rel or rel.endswith(".pyc"):
                        continue
                    if "p96" in rel.lower() or "real_task_card_support_retrieval" in rel.lower():
                        found.append(rel)
    for f in found:
        assert f in planned, f"Unexpected file: {f}"


# ── 3. no torch import ───────────────────────────────────────────────────────

def test_p96_03_no_torch_import():
    src = pathlib.Path("src/phase4/real_task_card_support_retrieval_pilot.py").read_text(encoding="utf-8")
    assert "import torch" not in src
    assert "from torch" not in src


# ── 4. no numpy/pandas/sklearn/random ─────────────────────────────────────────

def test_p96_04_no_forbidden_imports():
    src = pathlib.Path("src/phase4/real_task_card_support_retrieval_pilot.py").read_text(encoding="utf-8")
    assert "import numpy" not in src
    assert "import pandas" not in src
    assert "import sklearn" not in src
    assert "import random" not in src


# ── 5. no model training ─────────────────────────────────────────────────────

def test_p96_05_no_training():
    r = _probe()
    assert r["model_training_performed"] is False
    assert r["torch_training_performed"] is False


# ── 6. no optimizer ───────────────────────────────────────────────────────────

def test_p96_06_no_optimizer():
    r = _probe()
    assert r["new_model_implemented"] is False
    assert r["optimizer_created"] is False


# ── 7. no checkpoint ──────────────────────────────────────────────────────────

def test_p96_07_no_checkpoint():
    r = _probe()
    assert r["checkpoint_written"] is False


# ── 8. P95 source validation ──────────────────────────────────────────────────

def test_p96_08_source_contracts():
    r = _probe()
    assert r["source_contracts_validated"] is True
    v = validate_source_contracts_for_p96()
    assert v["p95_validated"] is True


# ── 9. P95 diagnostic-only evidence boundary preserved ────────────────────────

def test_p96_09_evidence_boundary_preserved():
    v = validate_source_contracts_for_p96()
    assert v["p95_valid_real_evidence_false"] is True
    assert v["p95_proxy_risk_true"] is True


# ── 10–12. task card catalog, query records, demo records loaded ──────────────

def test_p96_10_catalog_loaded():
    r = _probe()
    assert r["task_card_catalog_count"] == 10
    assert r["query_task_card_record_count"] > 0
    assert r["support_demo_card_record_count"] > 0


# ── 13. tokenization deterministic ────────────────────────────────────────────

def test_p96_13_tokenization():
    t1 = tokenize_public_text("Scale amplitude fluctuation")
    t2 = tokenize_public_text("scale AMPLITUDE, fluctuation.")
    assert t1 == t2
    assert "scale" in t1


# ── 14. token overlap distance behaves correctly ──────────────────────────────

def test_p96_14_token_overlap_distance():
    q = ["amplitude", "fluctuation"]
    s = ["amplitude", "fluctuation", "noise"]
    d = token_overlap_distance(q, s)
    assert 0.0 < d < 1.0


# ── 15. preserve constraint distance behaves correctly ────────────────────────

def test_p96_15_preserve_constraint_distance():
    q = {"length": True, "phase": False}
    s = {"length": True, "phase": True}
    d = preserve_constraint_distance(q, s)
    assert d == 0.5


# ── 16–17. public distance does not use task_card_id / compatible labels ───────

def test_p96_16_public_distance_safe():
    q = {
        "task_card_id": "oracle_id_1",
        "public_descriptor_tokens": ["amplitude"],
        "preserve_constraints": {"length": True},
        "change_intent": {"primary_change": "amplitude", "change_axis_description": "fluctuations"},
    }
    s = {
        "task_card_id": "oracle_id_2",
        "public_descriptor_tokens": ["amplitude"],
        "preserve_constraints": {"length": True},
        "change_intent": {"primary_change": "amplitude", "change_axis_description": "fluctuations"},
    }
    d = task_card_public_distance(q, s, "token_overlap")
    assert d == 0.0


# ── 18–19. diagnostic task_card_id policy marked diagnostic-only ──────────────

def test_p96_18_diagnostic_policies():
    r = _probe()
    for pol in ("diagnostic_task_card_id_exact_match", "diagnostic_compatible_label_mapping_oracle"):
        res = r["policy_metric_results"][pol]
        assert res["diagnostic_only"] is True
        assert res["valid_for_real_external_task_context_evidence"] is False


# ── 20–21. public-token policy not real evidence but valid for contract pilot ──

def test_p96_20_public_token_policies():
    r = _probe()
    for pol in (
        "task_card_public_token_overlap_retrieval",
        "task_card_change_preserve_signature_retrieval",
        "support_demo_public_caption_retrieval",
        "hybrid_task_card_plus_source_signature_retrieval",
    ):
        res = r["policy_metric_results"][pol]
        assert res["diagnostic_only"] is False
        assert res["valid_for_dataset_contract_pilot"] is True
        assert res["valid_for_real_external_task_context_evidence"] is False


# ── 24–25. metric results and negative controls produced ─────────────────────

def test_p96_24_results_produced():
    r = _probe()
    assert len(r["policy_metric_results"]) == len(ALL_P96_POLICIES)
    assert len(r["policy_negative_controls"]) == len(ALL_P96_POLICIES)


# ── 26–28. negative controls properties ───────────────────────────────────────

def test_p96_26_controls_properties():
    r = _probe()
    for pol, nc in r["policy_negative_controls"].items():
        assert nc["shuffled_control_target_labels_preserved"] is True
        assert nc["true_labels_preserved"] is True
        assert nc["target_labels_shifted"] is False


# ── 29–30. leakage audits exist and public policies have zero leakage ─────────

def test_p96_29_leakage_clean():
    r = _probe()
    for pol in (
        "task_card_public_token_overlap_retrieval",
        "task_card_change_preserve_signature_retrieval",
        "support_demo_public_caption_retrieval",
        "hybrid_task_card_plus_source_signature_retrieval",
    ):
        la = r["policy_leakage_audits"][pol]
        assert la["hidden_relation_label_used_count"] == 0
        assert la["operator_id_used_count"] == 0
        assert la["query_target_used_count"] == 0
        assert la["query_delta_used_count"] == 0
        assert la["query_result_used_count"] == 0
        assert la["diagnostic_pass"] is True


# ── 31. proxy audit present and proxy risk true ────────────────────────────────

def test_p96_31_proxy_audit():
    r = _probe()
    pa = r["policy_proxy_audit"]
    assert pa["audit_defined"] is True
    assert pa["proxy_risk_present"] is True


# ── 32–37. ready/signal/bridge flags ──────────────────────────────────────────

def test_p96_32_flags():
    r = _probe()
    assert r["task_card_retrieval_pipeline_ready"] is True
    assert r["task_card_external_metric_supported"] is False
    assert r["valid_external_task_context_available"] is False
    assert r["valid_for_real_external_task_context_evidence"] is False
    assert r["ready_for_real_data_collection"] is True
    assert r["bridge_ready"] is False
    assert r["semantic_geometry_claims_allowed"] is False


# ── 38. recommended P97 logic valid ───────────────────────────────────────────

def test_p96_38_recommended_p97():
    r = _probe()
    sig = r["task_card_retrieval_signal_present"]
    if sig:
        assert r["recommended_next_phase"] == "P97_real_task_card_raw_data_collection_contract_no_training_no_bridge"
    else:
        assert r["recommended_next_phase"] == "P97_task_card_retrieval_policy_repair_no_training_no_bridge"


# ── 39. JSON serializable ────────────────────────────────────────────────────

def test_p96_39_json_serializable():
    r = _probe()
    s = json.dumps(r)
    assert isinstance(s, str)
