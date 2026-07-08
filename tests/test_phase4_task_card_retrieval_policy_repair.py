# tests/test_phase4_task_card_retrieval_policy_repair.py

import sys
import os
import json
import pathlib

from src.phase4.task_card_retrieval_policy_repair import (
    run_p97_task_card_retrieval_policy_repair_probe,
    validate_source_contracts_for_p97,
    build_typed_operator_intent_signature,
    build_support_demo_visible_transformation_signature,
    typed_slot_distance,
    demo_compatibility_distance,
    PHASE,
    VERDICT,
    ALL_P97_POLICIES,
)


def _probe():
    return run_p97_task_card_retrieval_policy_repair_probe()


# ── 1. phase / verdict ────────────────────────────────────────────────────────

def test_p97_01_phase_verdict():
    r = _probe()
    assert r["phase"] == "P97"
    assert r["verdict"] == "P97_READY_FOR_REVIEW"
    assert r["json_safe"] is True
    assert r["diagnostic_only"] is True


# ── 2. scope gate ─────────────────────────────────────────────────────────────

def test_p97_02_scope_gate():
    planned = {
        "src/phase4/task_card_retrieval_policy_repair.py",
        "tools/phase4/run_p97_task_card_retrieval_policy_repair_smoke.py",
        "tests/test_phase4_task_card_retrieval_policy_repair.py",
        "tests/test_phase4_p97_task_card_retrieval_policy_repair_smoke.py",
        "reports/PHASE_4_P97_TASK_CARD_RETRIEVAL_POLICY_REPAIR_NO_TRAINING_NO_BRIDGE_REPORT.md",
    }
    found = []
    for folder in ["src/phase4", "tools/phase4", "tests", "reports"]:
        if os.path.exists(folder):
            for p in pathlib.Path(folder).resolve().rglob("*"):
                if p.is_file():
                    rel = str(p.relative_to(pathlib.Path(os.getcwd()).resolve())).replace("\\", "/")
                    if "__pycache__" in rel or rel.endswith(".pyc"):
                        continue
                    if "p97" in rel.lower() or "task_card_retrieval_policy_repair" in rel.lower():
                        found.append(rel)
    for f in found:
        assert f in planned, f"Unexpected file: {f}"


# ── 3. no forbidden imports ───────────────────────────────────────────────────

def test_p97_03_no_forbidden_imports():
    src = pathlib.Path("src/phase4/task_card_retrieval_policy_repair.py").read_text(encoding="utf-8")
    assert "import torch" not in src
    assert "from torch" not in src
    assert "import numpy" not in src
    assert "import pandas" not in src
    assert "import sklearn" not in src
    assert "import random" not in src


# ── 4–5. no model training/optimizer/checkpoint ───────────────────────────────

def test_p97_04_no_training_or_models():
    r = _probe()
    assert r["model_training_performed"] is False
    assert r["torch_training_performed"] is False
    assert r["new_model_implemented"] is False
    assert r["optimizer_created"] is False
    assert r["checkpoint_written"] is False


# ── 6–7. P96 source validation ────────────────────────────────────────────────

def test_p97_06_source_contracts():
    r = _probe()
    assert r["source_contracts_validated"] is True
    v = validate_source_contracts_for_p97()
    assert v["p96_validated"] is True
    assert v["p96_pipeline_ready"] is True
    assert v["p96_public_token_signal_absent"] is True


# ── 8–10. typed operator intent signature schema ──────────────────────────────

def test_p97_08_typed_operator_intent_schema():
    card = {
        "human_readable_instruction": "Shift the sequence coordinates along the horizontal axis.",
        "change_intent": {"primary_change": "translate", "change_axis_description": "horizontal axis"},
        "preserve_constraints": {"length": True},
        "forbidden_changes": [],
    }
    sig = build_typed_operator_intent_signature(card)
    assert sig["intent_family"] == "translation"
    assert sig["change_axis_type"] == "horizontal_coordinate"
    assert sig["change_direction_type"] == "shift"
    assert sig["signature_audit"]["uses_task_card_id"] is False
    assert sig["signature_audit"]["uses_compatible_label_mapping"] is False


# ── 11. support demo visible signature schema ──────────────────────────────────

def test_p97_11_support_demo_visible_schema():
    demo_card = {
        "support_task_card": {
            "human_readable_instruction": "Shift horizontal axis.",
            "change_intent": {"primary_change": "translate", "change_axis_description": "horizontal axis"},
            "preserve_constraints": {"length": True},
            "forbidden_changes": [],
        },
        "support_demo_public_view": {
            "visible_change_summary": {"delta_magnitude": 2.0, "delta_dimension": 0},
            "visible_preservation_summary": {"invariant_keys": ["length"]},
            "human_demo_caption": "Shift coordinate sequence horizontally.",
        }
    }
    sig = build_support_demo_visible_transformation_signature(demo_card)
    assert sig["demo_change_scale_type"] == "large"
    assert sig["demo_dimension_type"] == "dim_0"
    assert sig["signature_audit"]["uses_external_demo_id"] is False


# ── 12–13. typed slot distance properties ──────────────────────────────────────

def test_p97_12_typed_slot_distance():
    s1 = {
        "intent_family": "translation",
        "change_axis_type": "horizontal_coordinate",
        "change_direction_type": "shift",
        "geometry_action_type": "translate",
        "temporal_action_type": "none",
        "value_action_type": "none",
        "locality_type": "global",
        "periodicity_type": "non_periodic",
        "preserve_vector": {"length": True},
        "forbid_vector": {},
    }
    s2 = {
        "intent_family": "translation",
        "change_axis_type": "horizontal_coordinate",
        "change_direction_type": "shift",
        "geometry_action_type": "translate",
        "temporal_action_type": "none",
        "value_action_type": "none",
        "locality_type": "global",
        "periodicity_type": "non_periodic",
        "preserve_vector": {"length": True},
        "forbid_vector": {},
    }
    assert typed_slot_distance(s1, s2) == 0.0

    s3 = s2.copy()
    s3["intent_family"] = "scaling"
    assert typed_slot_distance(s1, s3) > 0.0


# ── 14. demo compatibility distance properties ───────────────────────────────

def test_p97_14_demo_compatibility_distance():
    q_sig = {
        "intent_family": "translation",
        "change_axis_type": "horizontal_coordinate",
        "change_direction_type": "shift",
        "geometry_action_type": "translate",
        "temporal_action_type": "none",
        "value_action_type": "none",
        "locality_type": "global",
        "periodicity_type": "non_periodic",
        "preserve_vector": {"length": True},
        "forbid_vector": {},
    }
    demo_sig = {
        "demo_task_intent_signature": q_sig,
        "demo_caption_intent_signature": q_sig,
    }
    assert demo_compatibility_distance(q_sig, demo_sig) == 0.0


# ── 15–19. repaired policies evaluation & types ────────────────────────────────

def test_p97_15_repaired_policies_run():
    r = _probe()
    assert len(r["policy_metric_results"]) == len(ALL_P97_POLICIES)

    for pol in ("diagnostic_task_card_id_exact_match", "diagnostic_compatible_label_mapping_oracle"):
        assert r["policy_metric_results"][pol]["diagnostic_only"] is True

    for pol in (
        "typed_operator_intent_signature_retrieval",
        "typed_intent_plus_preserve_constraint_retrieval",
        "demo_visible_transformation_signature_retrieval",
        "typed_intent_demo_compatibility_retrieval",
        "hybrid_typed_intent_demo_source_retrieval",
    ):
        mr = r["policy_metric_results"][pol]
        assert mr["diagnostic_only"] is False
        assert mr["valid_for_dataset_contract_pilot"] is True
        assert mr["valid_for_real_external_task_context_evidence"] is False


# ── 20. leakage audits pass for public policies ───────────────────────────────

def test_p97_20_leakage_clean():
    r = _probe()
    for pol in (
        "typed_operator_intent_signature_retrieval",
        "typed_intent_plus_preserve_constraint_retrieval",
        "demo_visible_transformation_signature_retrieval",
        "typed_intent_demo_compatibility_retrieval",
        "hybrid_typed_intent_demo_source_retrieval",
    ):
        la = r["policy_leakage_audits"][pol]
        assert la["hidden_relation_label_used_count"] == 0
        assert la["operator_id_used_count"] == 0
        assert la["query_target_used_count"] == 0
        assert la["query_delta_used_count"] == 0
        assert la["query_result_used_count"] == 0
        assert la["task_card_id_used_count"] == 0
        assert la["compatible_label_mapping_used_count"] == 0
        assert la["diagnostic_pass"] is True


# ── 21. proxy audit true ──────────────────────────────────────────────────────

def test_p97_21_proxy_audit():
    r = _probe()
    pa = r["policy_proxy_audit"]
    assert pa["audit_defined"] is True
    assert pa["proxy_risk_present"] is True


# ── 22–23. shuffled / typed-signature-shuffle controls exist ──────────────────

def test_p97_22_negative_controls():
    r = _probe()
    for pol in ALL_P97_POLICIES:
        nc = r["policy_negative_controls"][pol]
        assert nc["shuffled_control_target_labels_preserved"] is True
        assert nc["true_labels_preserved"] is True
        assert "typed_signature_shuffle_control" in nc


# ── 24–29. boundary & recommendation next phase logic ─────────────────────────

def test_p97_24_boundary_logic():
    r = _probe()
    assert r["repaired_retrieval_policy_ready"] is True
    assert r["repaired_task_card_external_metric_supported"] is False
    assert r["valid_external_task_context_available"] is False
    assert r["valid_for_real_external_task_context_evidence"] is False
    assert r["ready_for_real_data_collection"] is True
    assert r["bridge_ready"] is False
    assert r["semantic_geometry_claims_allowed"] is False

    sig = r["repaired_task_card_retrieval_signal_present"]
    if sig:
        assert r["recommended_next_phase"] == "P98_real_task_card_raw_data_collection_contract_no_training_no_bridge"
    else:
        assert r["recommended_next_phase"] == "P98_task_card_schema_enrichment_or_controlled_real_data_seed_no_training_no_bridge"


# ── 30. JSON serializable ────────────────────────────────────────────────────

def test_p97_30_json_serializable():
    r = _probe()
    s = json.dumps(r)
    assert isinstance(s, str)
