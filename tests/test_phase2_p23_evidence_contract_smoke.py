# tests/test_phase2_p23_evidence_contract_smoke.py

import json
import pathlib
import pytest
from unittest.mock import patch

from tools.phase2.run_p23_evidence_contract_smoke import (
    run_p23_evidence_contract_smoke,
    compact_json,
    main,
)
from src.phase2.evidence import EVIDENCE_CONTRACT_VERSION


# Predefined valid P22 smoke summary for monkeypatching
@pytest.fixture
def mock_p22_summary():
    return {
        "contract": "phase2_p22_artifact_backed_baseline_smoke_v1",
        "verdict": "PASS",
        "p14_audit_verified": True,
        "no_artifact_generation": True,
        "no_model_training": True,
        "no_final_comparison": True,
        "no_scientific_conclusion": True,
        "artifact_root": "phase2_artifacts/p14_smoke_dry_run",
        "loaded_reference_count": 2,
        "baseline_count": 3,
        "candidate_count_per_baseline": 6,
        "reference_summary": {
            "references": [
                {"role": "arma_mean_source", "sample_id": "ARMA_1", "line_number": 1, "family_id": "ARMA", "mean_family": "ARMA", "volatility_family": "NONE"},
                {"role": "garch_volatility_source", "sample_id": "GARCH_1", "line_number": 2, "family_id": "GARCH", "mean_family": "NONE", "volatility_family": "GARCH"}
            ]
        },
        "baselines": [
            {
                "baseline_name": "copy_reference",
                "metric_bundle_bridge_verified": True,
                "reason": "completed copy_reference",
                "generation_summary": {
                    "candidate_count": 6, "candidate_family_ids": ["ARMA"] * 6,
                    "source_records_summary": [{"source_reference_indices": [0], "seed": None}] * 6
                },
                "evaluation_summary": {
                    "validity_valid_count": 6, "validity_total_count": 6, "validity_valid_rate": 1.0,
                    "composition_pass_count": 0, "composition_pass_rate": 0.0,
                    "novelty_pass_count": 0, "novelty_pass_rate": 0.0, "distribution_mmd_rbf": 0.0,
                    "seed_stability_present": False, "candidate_record_count": 6
                }
            },
            {
                "baseline_name": "random_valid",
                "metric_bundle_bridge_verified": True,
                "reason": "completed random_valid",
                "generation_summary": {
                    "candidate_count": 6, "candidate_family_ids": ["ARMA"] * 6,
                    "source_records_summary": [{"source_reference_indices": [], "seed": 22001}] * 6
                },
                "evaluation_summary": {
                    "validity_valid_count": 6, "validity_total_count": 6, "validity_valid_rate": 1.0,
                    "composition_pass_count": 0, "composition_pass_rate": 0.0,
                    "novelty_pass_count": 0, "novelty_pass_rate": 0.0, "distribution_mmd_rbf": 0.1,
                    "seed_stability_present": False, "candidate_record_count": 6
                }
            },
            {
                "baseline_name": "structural_composition_oracle",
                "metric_bundle_bridge_verified": True,
                "reason": "completed structural_composition_oracle",
                "generation_summary": {
                    "candidate_count": 6, "candidate_family_ids": ["ARMA_GARCH"] * 6,
                    "source_records_summary": [{"source_reference_indices": [0, 1], "seed": None}] * 6
                },
                "evaluation_summary": {
                    "validity_valid_count": 6, "validity_total_count": 6, "validity_valid_rate": 1.0,
                    "composition_pass_count": 6, "composition_pass_rate": 1.0,
                    "novelty_pass_count": 0, "novelty_pass_rate": 0.0, "distribution_mmd_rbf": 0.2,
                    "seed_stability_present": False, "candidate_record_count": 6
                }
            }
        ]
    }


def test_p23_smoke_01_compact_json_sorts():
    data = {"b": 2, "a": 1}
    js = compact_json(data)
    assert js == '{"a":1,"b":2}'


def test_p23_smoke_02_main_rejects_args():
    with patch("sys.argv", ["run_p23_evidence_contract_smoke.py", "invalid_arg"]):
        assert main() != 0


def test_p23_smoke_03_run_evidence_smoke_monkeypatched(mock_p22_summary):
    with patch("tools.phase2.run_p23_evidence_contract_smoke.run_p22_artifact_backed_baseline_smoke", return_value=mock_p22_summary):
        res = run_p23_evidence_contract_smoke()
        assert res["verdict"] == "PASS"
        assert res["contract"] == EVIDENCE_CONTRACT_VERSION


def test_p23_smoke_04_monkeypatched_verdict_and_contract(mock_p22_summary):
    with patch("tools.phase2.run_p23_evidence_contract_smoke.run_p22_artifact_backed_baseline_smoke", return_value=mock_p22_summary):
        res = run_p23_evidence_contract_smoke()
        assert res["verdict"] == "PASS"
        assert res["contract"] == "phase2_p23_evidence_contract_v1"
        assert res["source_phase"] == "P22"


def test_p23_smoke_05_monkeypatched_no_raw_params(mock_p22_summary):
    with patch("tools.phase2.run_p23_evidence_contract_smoke.run_p22_artifact_backed_baseline_smoke", return_value=mock_p22_summary):
        res = run_p23_evidence_contract_smoke()
        # Recursively check no forbidden keys exist in final output
        from src.phase2.evidence import assert_no_raw_params
        assert_no_raw_params(res)


def test_p23_smoke_06_monkeypatched_no_forbidden_claims(mock_p22_summary):
    with patch("tools.phase2.run_p23_evidence_contract_smoke.run_p22_artifact_backed_baseline_smoke", return_value=mock_p22_summary):
        res = run_p23_evidence_contract_smoke()
        # Recursively check no forbidden claims are present
        from src.phase2.evidence import assert_no_forbidden_claims
        assert_no_forbidden_claims(res)


def test_p23_smoke_07_script_does_not_import_forbidden():
    p = pathlib.Path("tools/phase2/run_p23_evidence_contract_smoke.py").read_text(encoding="utf-8")
    forbidden = ["torch", "numpy", "pandas", "yaml", "argparse", "sklearn", "scipy"]
    for f in forbidden:
        assert f"import {f}" not in p
        assert f"from {f}" not in p


def test_p23_smoke_08_script_does_not_call_simulate_time_series():
    p = pathlib.Path("tools/phase2/run_p23_evidence_contract_smoke.py").read_text(encoding="utf-8")
    assert "simulate_time_series" not in p


def test_p23_smoke_09_script_does_not_call_build_dataset_in_memory():
    p = pathlib.Path("tools/phase2/run_p23_evidence_contract_smoke.py").read_text(encoding="utf-8")
    assert "build_dataset_in_memory" not in p


def test_p23_smoke_10_script_does_not_call_write_dataset_artifacts():
    p = pathlib.Path("tools/phase2/run_p23_evidence_contract_smoke.py").read_text(encoding="utf-8")
    assert "write_dataset_artifacts" not in p


def test_p23_smoke_11_script_does_not_call_run_phase2_artifact_generation():
    p = pathlib.Path("tools/phase2/run_p23_evidence_contract_smoke.py").read_text(encoding="utf-8")
    assert "run_phase2_artifact_generation" not in p


# 12. Optional integration test
@pytest.mark.skipif(
    not (pathlib.Path("phase2_artifacts/p14_smoke_dry_run/zero_shot_train/samples.jsonl").exists() and
         pathlib.Path("phase2_artifacts/p14_smoke_dry_run/zero_shot_train/manifest.json").exists()),
    reason="P14 smoke artifacts not present; P23 integration covered by explicit smoke command."
)
def test_p23_smoke_12_smoke_script_execution_real():
    res = run_p23_evidence_contract_smoke()
    assert res["verdict"] == "PASS"
    assert res["contract"] == "phase2_p23_evidence_contract_v1"
    assert res["source_phase"] == "P22"
    
    bundle = res["evidence_bundle"]
    assert bundle["metadata"]["contract_version"] == "phase2_p23_evidence_contract_v1"
    assert bundle["loaded_reference_count"] == 2
    assert bundle["baseline_count"] == 3
    assert bundle["candidate_count_per_baseline"] == 6
    assert bundle["verdict"] == "PASS"
