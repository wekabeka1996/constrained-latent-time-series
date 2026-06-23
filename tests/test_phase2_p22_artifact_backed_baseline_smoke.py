import pytest
from tools.phase2.run_p22_artifact_backed_baseline_smoke import (
    default_p22_artifact_root,
    require_p14_smoke_artifacts_available,
    load_jsonl_records_until,
    model_spec_from_artifact_sample,
    select_artifact_reference_specs,
    build_p22_generation_request,
    run_single_p22_baseline,
    run_p22_artifact_backed_baseline_smoke,
)
from src.phase2.schema import FamilyId
from tools.phase2.static_scope_guard import check_static_scope, require_static_scope_pass


def test_p22_static_scope():
    """Validates scope compliance against static rules."""
    res = check_static_scope(
        "tools/phase2/run_p22_artifact_backed_baseline_smoke.py",
        forbidden_path_tokens=()
    )
    require_static_scope_pass(res)


def test_p22_require_artifacts_rejects_missing(tmp_path):
    with pytest.raises(ValueError, match="does not exist|failed audit|missing"):
        require_p14_smoke_artifacts_available(str(tmp_path / "missing"))


def test_p22_smoke_script_execution():
    """Runs the full P22 smoke test."""
    res = run_p22_artifact_backed_baseline_smoke()
    
    assert res["verdict"] == "PASS"
    assert res["contract"] == "phase2_p22_artifact_backed_baseline_smoke_v1"
    assert res["artifact_root"] == "phase2_artifacts/p14_smoke_dry_run"
    assert res["p14_audit_verified"] is True
    assert res["loaded_reference_count"] == 2
    assert res["baseline_count"] == 3
    
    # Check references
    refs = res["reference_summary"]["references"]
    assert len(refs) == 2
    roles = {r["role"] for r in refs}
    assert roles == {"arma_mean_source", "garch_volatility_source"}
    
    # Check baseline presence
    b_names = {b["baseline_name"] for b in res["baselines"]}
    assert b_names == {"copy_reference", "random_valid", "structural_composition_oracle"}
    
    for b in res["baselines"]:
        assert b["generation_summary"]["candidate_count"] == 6
        assert b["evaluation_summary"]["candidate_count"] == 6
        assert b["metric_bundle_bridge_verified"] is True
