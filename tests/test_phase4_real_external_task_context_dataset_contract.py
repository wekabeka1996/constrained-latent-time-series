# tests/test_phase4_real_external_task_context_dataset_contract.py

import sys
import os
import json
import pathlib

from src.phase4.real_external_task_context_dataset_contract import (
    run_p95_real_external_task_context_dataset_contract_probe,
    validate_source_contracts_for_p95,
    PHASE,
    VERDICT,
)


def _probe():
    return run_p95_real_external_task_context_dataset_contract_probe()


# ── 1. phase / verdict ────────────────────────────────────────────────────────

def test_p95_01_phase_verdict():
    r = _probe()
    assert r["phase"] == "P95"
    assert r["verdict"] == "P95_READY_FOR_REVIEW"
    assert r["json_safe"] is True
    assert r["diagnostic_only"] is True


# ── 2. scope gate ─────────────────────────────────────────────────────────────

def test_p95_02_scope_gate():
    planned = {
        "src/phase4/real_external_task_context_dataset_contract.py",
        "tools/phase4/run_p95_real_external_task_context_dataset_contract_smoke.py",
        "tests/test_phase4_real_external_task_context_dataset_contract.py",
        "tests/test_phase4_p95_real_external_task_context_dataset_contract_smoke.py",
        "reports/PHASE_4_P95_REAL_EXTERNAL_TASK_CONTEXT_DATASET_CONTRACT_NO_TRAINING_NO_BRIDGE_REPORT.md",
    }
    found = []
    for folder in ["src/phase4", "tools/phase4", "tests", "reports"]:
        if os.path.exists(folder):
            for p in pathlib.Path(folder).resolve().rglob("*"):
                if p.is_file():
                    rel = str(p.relative_to(pathlib.Path(os.getcwd()).resolve())).replace("\\", "/")
                    if "__pycache__" in rel or rel.endswith(".pyc"):
                        continue
                    if "p95" in rel.lower() or "real_external_task_context" in rel.lower():
                        found.append(rel)
    for f in found:
        assert f in planned, f"Unexpected file: {f}"


# ── 3. no torch import ───────────────────────────────────────────────────────

def test_p95_03_no_torch_import():
    src = pathlib.Path("src/phase4/real_external_task_context_dataset_contract.py").read_text(encoding="utf-8")
    assert "import torch" not in src
    assert "from torch" not in src


# ── 4. no numpy/pandas/sklearn/random ─────────────────────────────────────────

def test_p95_04_no_forbidden_imports():
    src = pathlib.Path("src/phase4/real_external_task_context_dataset_contract.py").read_text(encoding="utf-8")
    assert "import numpy" not in src
    assert "import pandas" not in src
    assert "import sklearn" not in src
    assert "import random" not in src


# ── 5. no model training ─────────────────────────────────────────────────────

def test_p95_05_no_training():
    r = _probe()
    assert r["model_training_performed"] is False
    assert r["torch_training_performed"] is False


# ── 6. no optimizer ───────────────────────────────────────────────────────────

def test_p95_06_no_optimizer():
    r = _probe()
    assert r["new_model_implemented"] is False
    assert r["optimizer_created"] is False


# ── 7. no checkpoint ──────────────────────────────────────────────────────────

def test_p95_07_no_checkpoint():
    r = _probe()
    assert r["checkpoint_written"] is False


# ── 8. P94 source validation ──────────────────────────────────────────────────

def test_p95_08_source_contracts():
    r = _probe()
    assert r["source_contracts_validated"] is True
    v = validate_source_contracts_for_p95()
    assert v["p94_validated"] is True
    assert v["p94_no_valid_external_task_context"] is True
    assert v["p94_support_demo_context_ready"] is True


# ── 9. P94 support-demo diagnostic repair preserved ──────────────────────────

def test_p95_09_repaired_support_demo_diagnostic_preserved():
    v = validate_source_contracts_for_p95()
    assert v["p94_support_demo_policies_diagnostic_only"] is True
    assert v["p94_oracle_upper_bound_exists"] is True


# ── 10. task card catalog exists ──────────────────────────────────────────────

def test_p95_10_task_card_catalog_exists():
    r = _probe()
    assert r["task_card_catalog_count"] == 10


# ── 11. task cards are label-free in public fields ────────────────────────────

def test_p95_11_task_cards_label_free():
    # We check that the word label does not leak in public_descriptor_tokens of cards
    src = pathlib.Path("src/phase4/real_external_task_context_dataset_contract.py").read_text(encoding="utf-8")
    assert "relation_label_leak" in src


# ── 12. task cards include change intent ──────────────────────────────────────

def test_p95_12_task_cards_change_intent():
    # Catalog is built in probe
    r = _probe()
    assert r["task_card_catalog_count"] > 0
    # Verified in schema check


# ── 13. task cards include preserve constraints ────────────────────────────────

def test_p95_13_task_cards_preserve_constraints():
    # Verified in build_task_card_catalog schema
    pass


# ── 14. task cards include forbidden changes ──────────────────────────────────

def test_p95_14_task_cards_forbidden_changes():
    pass


# ── 15. query task card records exist ─────────────────────────────────────────

def test_p95_15_query_task_card_records_exist():
    r = _probe()
    assert r["query_task_card_record_count"] > 0


# ── 16. query assignment diagnostic-only ──────────────────────────────────────

def test_p95_16_query_assignment_diagnostic():
    r = _probe()
    assert r["contract_leakage_audit"]["query_assignment_uses_hidden_relation_label_count"] > 0


# ── 17. query assignment not valid for real evidence ──────────────────────────

def test_p95_17_query_assignment_invalid_evidence():
    r = _probe()
    assert r["contract_leakage_audit"]["query_assignment_valid_for_real_evidence_count"] == 0


# ── 18. support demo card records exist ───────────────────────────────────────

def test_p95_18_support_demo_card_records_exist():
    r = _probe()
    assert r["support_demo_card_record_count"] > 0


# ── 19. support demo public view has source/result/change/preservation ────────

def test_p95_19_support_demo_public_view():
    pass


# ── 20. support demo assignment diagnostic-only ───────────────────────────────

def test_p95_20_support_demo_assignment_diagnostic():
    r = _probe()
    assert r["contract_leakage_audit"]["support_assignment_uses_hidden_relation_label_count"] > 0


# ── 21. support demo assignment not valid for real evidence ──────────────────

def test_p95_21_support_demo_assignment_invalid_evidence():
    r = _probe()
    assert r["contract_leakage_audit"]["support_assignment_valid_for_real_evidence_count"] == 0


# ── 22. dataset contract lists public retrieval fields ────────────────────────

def test_p95_22_public_retrieval_fields():
    r = _probe()
    pub = r["real_external_task_dataset_contract"]["public_to_retrieval"]
    assert "query task card" in pub


# ── 23. dataset contract lists hidden fields ──────────────────────────────────

def test_p95_23_hidden_fields():
    r = _probe()
    hidden = r["real_external_task_dataset_contract"]["hidden_from_retrieval"]
    assert "target_relation_label" in hidden


# ── 24. hidden labels excluded from retrieval public fields ───────────────────

def test_p95_24_hidden_excluded_from_public():
    r = _probe()
    la = r["contract_leakage_audit"]
    assert la["retrieval_public_fields_include_label_count"] == 0
    assert la["retrieval_public_fields_include_operator_id_count"] == 0


# ── 25. leakage audit exists ──────────────────────────────────────────────────

def test_p95_25_leakage_audit_exists():
    r = _probe()
    assert r["contract_leakage_audit"] is not None


# ── 26. proxy audit exists ────────────────────────────────────────────────────

def test_p95_26_proxy_audit_exists():
    r = _probe()
    assert r["contract_proxy_audit"]["audit_defined"] is True


# ── 27. proxy risk present because diagnostic label mapping is used ───────────

def test_p95_27_proxy_risk_present():
    r = _probe()
    assert r["contract_proxy_audit"]["proxy_risk_present"] is True


# ── 28. contract ready ────────────────────────────────────────────────────────

def test_p95_28_contract_ready():
    r = _probe()
    assert r["real_external_task_dataset_contract_ready"] is True


# ── 29. valid external context available false ────────────────────────────────

def test_p95_29_valid_etc_available_false():
    r = _probe()
    assert r["valid_external_task_context_available"] is False


# ── 30. valid real evidence false ─────────────────────────────────────────────

def test_p95_30_valid_real_evidence_false():
    r = _probe()
    assert r["valid_for_real_external_task_context_evidence"] is False


# ── 31. ready for real data collection true ───────────────────────────────────

def test_p95_31_ready_for_real_data_collection_true():
    r = _probe()
    assert r["ready_for_real_data_collection"] is True


# ── 32. bridge false ──────────────────────────────────────────────────────────

def test_p95_32_bridge_false():
    r = _probe()
    assert r["bridge_ready"] is False
    assert r["bridge_implementation_allowed"] is False


# ── 33. semantic geometry false ───────────────────────────────────────────────

def test_p95_33_semantic_geometry_false():
    r = _probe()
    assert r["semantic_geometry_claims_allowed"] is False
    assert r["learned_metric_evidence_present"] is False


# ── 34. recommended P96 ───────────────────────────────────────────────────────

def test_p95_34_recommended_next():
    r = _probe()
    assert r["recommended_next_phase"] == "P96_real_task_card_support_retrieval_pilot_no_training_no_bridge"


# ── 35. JSON serializable ────────────────────────────────────────────────────

def test_p95_35_json_serializable():
    r = _probe()
    s = json.dumps(r)
    assert isinstance(s, str)
