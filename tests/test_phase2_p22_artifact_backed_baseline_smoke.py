import json
import pytest
import pathlib
from unittest.mock import patch

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
from src.phase2.schema import FamilyId, MeanFamily, VolatilityFamily, ModelSpec
from tools.phase2.static_scope_guard import check_static_scope, require_static_scope_pass


# 1. Static Scope Tests
def test_p22_static_scope():
    """Validates scope compliance against static rules without bypassing path tokens."""
    res = check_static_scope("tools/phase2/run_p22_artifact_backed_baseline_smoke.py")
    require_static_scope_pass(res)


def test_p22_static_scope_guard_behavior(tmp_path):
    """Verifies that static_scope_guard actually detects forbidden path tokens when they exist."""
    bad_code = "my_path = 'phase" + "2_artifacts/some_file'\n"
    p = tmp_path / "bad.py"
    p.write_text(bad_code, encoding="utf-8")
    res = check_static_scope(str(p))
    assert not res.passed
    assert "phase2_artifacts" in res.forbidden_path_token_hits


# 2. require_p14_smoke_artifacts_available Tests
def test_p22_require_artifacts_rejects_missing_directory(tmp_path):
    """Rejects if root directory doesn't exist."""
    with pytest.raises(ValueError, match="does not exist"):
        require_p14_smoke_artifacts_available(str(tmp_path / "non_existent_folder"))


def test_p22_require_artifacts_rejects_missing_samples(tmp_path):
    """Rejects if zero_shot_train/samples.jsonl is missing."""
    zero_shot = tmp_path / "zero_shot_train"
    zero_shot.mkdir(parents=True)
    (zero_shot / "manifest.json").write_text("{}", encoding="utf-8")
    with pytest.raises(ValueError, match="samples missing"):
        require_p14_smoke_artifacts_available(str(tmp_path))


def test_p22_require_artifacts_rejects_missing_manifest(tmp_path):
    """Rejects if zero_shot_train/manifest.json is missing."""
    zero_shot = tmp_path / "zero_shot_train"
    zero_shot.mkdir(parents=True)
    (zero_shot / "samples.jsonl").write_text("{}", encoding="utf-8")
    with pytest.raises(ValueError, match="manifest missing"):
        require_p14_smoke_artifacts_available(str(tmp_path))


def test_p22_require_artifacts_rejects_empty_arg():
    """Rejects if empty or invalid argument is passed."""
    with pytest.raises(ValueError, match="must be a non-empty str"):
        require_p14_smoke_artifacts_available("")


def test_p22_require_artifacts_audit_failure(tmp_path):
    """Rejects if audit fails (mocked to throw)."""
    zero_shot = tmp_path / "zero_shot_train"
    zero_shot.mkdir(parents=True)
    (zero_shot / "samples.jsonl").write_text("{}", encoding="utf-8")
    (zero_shot / "manifest.json").write_text("{}", encoding="utf-8")
    with patch("tools.phase2.audit_phase2_artifact_manifest.audit_phase2_artifacts", side_effect=ValueError("Mocked audit error")):
        with pytest.raises(ValueError, match="failed audit|Mocked audit error"):
            require_p14_smoke_artifacts_available(str(tmp_path))


def test_p22_require_artifacts_success_mocked(tmp_path):
    """Passes if audit succeeds (mocked happy path)."""
    zero_shot = tmp_path / "zero_shot_train"
    zero_shot.mkdir(parents=True)
    (zero_shot / "samples.jsonl").write_text("{}", encoding="utf-8")
    (zero_shot / "manifest.json").write_text("{}", encoding="utf-8")
    with patch("tools.phase2.audit_phase2_artifact_manifest.audit_phase2_artifacts", return_value={"verdict": "PASS"}) as mock_audit:
        require_p14_smoke_artifacts_available(str(tmp_path))
        mock_audit.assert_called_once_with(str(tmp_path))


# 3. load_jsonl_records_until Tests
def test_p22_load_jsonl_records_until_basic(tmp_path):
    """Loads records correctly with and without limits, skipping empty lines."""
    p = tmp_path / "test.jsonl"
    p.write_text('{"a": 1}\n\n{"b": 2}\n{"c": 3}\n', encoding="utf-8")
    
    # Load all
    recs = load_jsonl_records_until(str(p), max_records=None)
    assert len(recs) == 3
    assert recs[0] == (1, {"a": 1})
    assert recs[1] == (3, {"b": 2})  # line_num is 3 due to blank line at line 2
    assert recs[2] == (4, {"c": 3})
    
    # Load with limit
    recs_limit = load_jsonl_records_until(str(p), max_records=2)
    assert len(recs_limit) == 2
    assert recs_limit[0] == (1, {"a": 1})
    assert recs_limit[1] == (3, {"b": 2})


def test_p22_load_jsonl_records_until_malformed(tmp_path):
    """Raises ValueError indicating line number if JSON is malformed."""
    p = tmp_path / "test.jsonl"
    p.write_text('{"a": 1}\n{"b": 2\n', encoding="utf-8")  # line 2 is malformed
    with pytest.raises(ValueError, match="Malformed JSON at line 2"):
        load_jsonl_records_until(str(p), max_records=None)


def test_p22_load_jsonl_records_until_validation():
    """Validates path argument."""
    with pytest.raises(ValueError, match="path must be a non-empty str"):
        load_jsonl_records_until("", max_records=None)


# 4. model_spec_from_artifact_sample Tests
def test_p22_model_spec_from_artifact_sample_valid():
    """Correctly reconstructs ModelSpec from a valid artifact record dict."""
    record = {
        "model_spec": {
            "family_id": "ARMA",
            "mean_family": "ARMA",
            "volatility_family": "NONE",
            "p": 1,
            "q": 1,
            "r": 0,
            "s": 0,
            "ar_params": [0.5],
            "ma_params": [-0.3],
            "omega": None,
            "alpha_params": [],
            "beta_params": [],
            "constraint_flags": [1.0, 0.0, 0.0, 0.0],
            "provenance": [["phase", "p14"]]
        }
    }
    spec = model_spec_from_artifact_sample(record)
    assert isinstance(spec, ModelSpec)
    assert spec.family_id == FamilyId.ARMA
    assert spec.ar_params == (0.5,)
    assert spec.ma_params == (-0.3,)
    assert spec.omega is None


def test_p22_model_spec_from_artifact_sample_missing_model_spec():
    """Raises ValueError if 'model_spec' key is missing."""
    with pytest.raises(ValueError, match="record missing 'model_spec'"):
        model_spec_from_artifact_sample({"other_key": 123})


def test_p22_model_spec_from_artifact_sample_non_dict():
    """Raises ValueError if record is not a dict."""
    with pytest.raises(ValueError, match="record must be a dict"):
        model_spec_from_artifact_sample("not_a_dict")


def test_p22_model_spec_from_artifact_sample_missing_fields():
    """Raises ValueError if spec fields are missing."""
    record = {
        "model_spec": {
            "family_id": "ARMA",
            # missing mean_family
            "volatility_family": "NONE",
            "p": 1,
            "q": 1,
            "r": 0,
            "s": 0,
            "ar_params": [0.5],
            "ma_params": [-0.3],
            "omega": None,
            "alpha_params": [],
            "beta_params": [],
            "constraint_flags": [1.0, 0.0, 0.0, 0.0],
            "provenance": []
        }
    }
    with pytest.raises(ValueError, match="Missing or invalid spec fields"):
        model_spec_from_artifact_sample(record)


def test_p22_model_spec_from_artifact_sample_invalid_values():
    """Raises ValueError if values fail validation/math validity checks."""
    record = {
        "model_spec": {
            "family_id": "ARMA",
            "mean_family": "ARMA",
            "volatility_family": "NONE",
            "p": 1,
            "q": 1,
            "r": 0,
            "s": 0,
            "ar_params": [1.5],  # fails stationarity constraint (ar_param >= 1.0)
            "ma_params": [-0.3],
            "omega": None,
            "alpha_params": [],
            "beta_params": [],
            "constraint_flags": [1.0, 0.0, 0.0, 0.0],
            "provenance": [["phase", "p14"]]
        }
    }
    with pytest.raises(ValueError):
        model_spec_from_artifact_sample(record)


# 5. select_artifact_reference_specs Tests
def test_p22_select_artifact_reference_specs_valid(tmp_path):
    """Correctly selects ARMA and GARCH reference specs from JSONL records."""
    arma_rec = {
        "sample_id": "ARMA_1",
        "model_spec": {
            "family_id": "ARMA", "mean_family": "ARMA", "volatility_family": "NONE",
            "p": 1, "q": 1, "r": 0, "s": 0,
            "ar_params": [0.2], "ma_params": [0.3], "omega": None,
            "alpha_params": [], "beta_params": [],
            "constraint_flags": [1.0, 0.0, 0.0, 0.0], "provenance": [["phase", "p14"]]
        }
    }
    garch_rec = {
        "sample_id": "GARCH_1",
        "model_spec": {
            "family_id": "GARCH", "mean_family": "NONE", "volatility_family": "GARCH",
            "p": 0, "q": 0, "r": 1, "s": 1,
            "ar_params": [], "ma_params": [], "omega": 0.1,
            "alpha_params": [0.2], "beta_params": [0.3],
            "constraint_flags": [1.0, 0.0, 0.0, 0.0], "provenance": [["phase", "p14"]]
        }
    }
    p = tmp_path / "samples.jsonl"
    p.write_text(json.dumps(arma_rec) + "\n" + json.dumps(garch_rec) + "\n", encoding="utf-8")
    
    arma_info, garch_info = select_artifact_reference_specs(str(p))
    
    assert arma_info[1] == "ARMA_1"
    assert arma_info[2] == 1
    assert garch_info[1] == "GARCH_1"
    assert garch_info[2] == 2


def test_p22_select_artifact_reference_specs_missing_sample_id(tmp_path):
    """Raises ValueError if sample_id is missing on a record line."""
    rec = {
        # missing sample_id
        "model_spec": {
            "family_id": "ARMA", "mean_family": "ARMA", "volatility_family": "NONE",
            "p": 1, "q": 1, "r": 0, "s": 0,
            "ar_params": [0.2], "ma_params": [0.3], "omega": None,
            "alpha_params": [], "beta_params": [],
            "constraint_flags": [1.0, 0.0, 0.0, 0.0], "provenance": []
        }
    }
    p = tmp_path / "samples.jsonl"
    p.write_text(json.dumps(rec) + "\n", encoding="utf-8")
    with pytest.raises(ValueError, match="Missing sample_id at line 1"):
        select_artifact_reference_specs(str(p))


def test_p22_select_artifact_reference_specs_missing_arma(tmp_path):
    """Raises ValueError if no valid ARMA candidate is found in the samples."""
    garch_rec = {
        "sample_id": "GARCH_1",
        "model_spec": {
            "family_id": "GARCH", "mean_family": "NONE", "volatility_family": "GARCH",
            "p": 0, "q": 0, "r": 1, "s": 1,
            "ar_params": [], "ma_params": [], "omega": 0.1,
            "alpha_params": [0.2], "beta_params": [0.3],
            "constraint_flags": [1.0, 0.0, 0.0, 0.0], "provenance": []
        }
    }
    p = tmp_path / "samples.jsonl"
    p.write_text(json.dumps(garch_rec) + "\n", encoding="utf-8")
    with pytest.raises(ValueError, match="Required ARMA source not found"):
        select_artifact_reference_specs(str(p))


def test_p22_select_artifact_reference_specs_missing_garch(tmp_path):
    """Raises ValueError if no valid GARCH candidate is found in the samples."""
    arma_rec = {
        "sample_id": "ARMA_1",
        "model_spec": {
            "family_id": "ARMA", "mean_family": "ARMA", "volatility_family": "NONE",
            "p": 1, "q": 1, "r": 0, "s": 0,
            "ar_params": [0.2], "ma_params": [0.3], "omega": None,
            "alpha_params": [], "beta_params": [],
            "constraint_flags": [1.0, 0.0, 0.0, 0.0], "provenance": []
        }
    }
    p = tmp_path / "samples.jsonl"
    p.write_text(json.dumps(arma_rec) + "\n", encoding="utf-8")
    with pytest.raises(ValueError, match="Required GARCH source not found"):
        select_artifact_reference_specs(str(p))


def test_p22_select_artifact_reference_specs_malformed_json(tmp_path):
    """Raises ValueError if a line contains malformed JSON."""
    p = tmp_path / "samples.jsonl"
    p.write_text("{\n", encoding="utf-8")
    with pytest.raises(ValueError, match="Malformed JSON at line 1"):
        select_artifact_reference_specs(str(p))


# 6. build_p22_generation_request Tests
def test_p22_build_generation_request():
    """Validates baseline request construction for all baseline configurations."""
    arma_spec = ModelSpec(FamilyId.ARMA, MeanFamily.ARMA, VolatilityFamily.NONE, 1, 1, 0, 0, (0.2,), (0.3,), None, (), (), (1.0, 0.0, 0.0, 0.0), (('phase', 'p14'),))
    garch_spec = ModelSpec(FamilyId.GARCH, MeanFamily.NONE, VolatilityFamily.GARCH, 0, 0, 1, 1, (), (), 0.1, (0.2,), (0.3,), (1.0, 0.0, 0.0, 0.0), (('phase', 'p14'),))
    refs = (arma_spec, garch_spec)
    
    # 1. random_valid has templates & schedules
    req_rv = build_p22_generation_request("random_valid", refs)
    assert req_rv.baseline_name == "random_valid"
    assert req_rv.random_valid_generation_templates is not None
    assert req_rv.random_valid_family_schedule is not None
    assert req_rv.candidate_count == 6
    assert req_rv.seed == 22001
    
    # 2. copy_reference has no rv templates
    req_cr = build_p22_generation_request("copy_reference", refs)
    assert req_cr.baseline_name == "copy_reference"
    assert req_cr.random_valid_generation_templates is None
    assert req_cr.random_valid_family_schedule is None

    # 3. structural_composition_oracle
    req_sco = build_p22_generation_request("structural_composition_oracle", refs)
    assert req_sco.baseline_name == "structural_composition_oracle"


# 7. run_single_p22_baseline and summary no-raw-params Tests
def test_p22_run_single_baseline_no_raw_params():
    """Ensures baselines run and outputs do not contain raw ModelSpec parameters."""
    arma_spec = ModelSpec(FamilyId.ARMA, MeanFamily.ARMA, VolatilityFamily.NONE, 1, 1, 0, 0, (0.2,), (0.3,), None, (), (), (1.0, 0.0, 0.0, 0.0), (('phase', 'p14'),))
    garch_spec = ModelSpec(FamilyId.GARCH, MeanFamily.NONE, VolatilityFamily.GARCH, 0, 0, 1, 1, (), (), 0.1, (0.2,), (0.3,), (1.0, 0.0, 0.0, 0.0), (('phase', 'p14'),))
    refs = (arma_spec, garch_spec)
    
    for baseline in ["copy_reference", "random_valid", "structural_composition_oracle"]:
        res = run_single_p22_baseline(baseline, refs)
        assert res["baseline_name"] == baseline
        assert res["metric_bundle_bridge_verified"] is True
        
        # Check no-raw-params: the summaries should not contain ModelSpec fields like parameters
        gen_sum = res["generation_summary"]
        eval_sum = res["evaluation_summary"]
        
        forbidden_fields = {"ar_params", "ma_params", "alpha_params", "beta_params", "omega"}
        
        # Check generation summary
        assert len(forbidden_fields & gen_sum.keys()) == 0
        for rec in gen_sum.get("source_records_summary", []):
            assert len(forbidden_fields & rec.keys()) == 0
            
        # Check evaluation summary
        assert len(forbidden_fields & eval_sum.keys()) == 0


# 8. Real/Optional integration test
@pytest.mark.skipif(
    not (pathlib.Path("phase2" + "_artifacts/p14_smoke_dry_run/zero_shot_train/samples.jsonl").exists() and
         pathlib.Path("phase2" + "_artifacts/p14_smoke_dry_run/zero_shot_train/manifest.json").exists()),
    reason="Real P14 smoke artifacts are not available for integration test"
)
def test_p22_smoke_script_execution_real():
    """Runs the full P22 smoke test on real P14 artifacts when available."""
    res = run_p22_artifact_backed_baseline_smoke()
    
    assert res["verdict"] == "PASS"
    assert res["contract"] == "phase2_p22_artifact_backed_baseline_smoke_v1"
    assert res["artifact_root"] == "phase2" + "_artifacts/p14_smoke_dry_run"
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
