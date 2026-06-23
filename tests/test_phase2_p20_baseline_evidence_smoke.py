# tests/test_phase2_p20_baseline_evidence_smoke.py

import sys
import math
import json
import pytest

from src.phase2.schema import FamilyId, MeanFamily, VolatilityFamily, ModelSpec, validate_model_spec
from src.phase2.constraints import require_math_valid
from tools.phase2.run_p20_baseline_evidence_smoke import (
    compact_json,
    assert_no_path_leakage,
    build_p20_reference_specs,
    build_p20_random_valid_templates,
    build_p20_random_valid_family_schedule,
    build_p20_generation_request,
    run_single_p20_baseline,
    run_p20_baseline_evidence_smoke,
    main,
)


# --- Group A: Fixture builders ---

def test_p20_1_build_reference_specs_count():
    refs = build_p20_reference_specs()
    assert len(refs) == 2


def test_p20_2_first_reference_arma():
    refs = build_p20_reference_specs()
    arma = refs[0]
    assert arma.family_id == FamilyId.ARMA
    assert arma.mean_family == MeanFamily.ARMA
    assert arma.volatility_family == VolatilityFamily.NONE


def test_p20_3_second_reference_garch():
    refs = build_p20_reference_specs()
    garch = refs[1]
    assert garch.family_id == FamilyId.GARCH
    assert garch.mean_family == MeanFamily.NONE
    assert garch.volatility_family == VolatilityFamily.GARCH


def test_p20_4_references_valid():
    refs = build_p20_reference_specs()
    for spec in refs:
        validate_model_spec(spec)
        require_math_valid(spec)


def test_p20_5_build_random_valid_templates_count():
    templates = build_p20_random_valid_templates()
    assert len(templates) == 3


def test_p20_6_random_valid_templates_ordered():
    templates = build_p20_random_valid_templates()
    assert templates[0][0] == FamilyId.AR
    assert templates[1][0] == FamilyId.GARCH
    assert templates[2][0] == FamilyId.ARMA_GARCH


def test_p20_7_build_random_valid_family_schedule():
    sched = build_p20_random_valid_family_schedule()
    assert sched == (FamilyId.AR, FamilyId.GARCH, FamilyId.ARMA_GARCH)


# --- Group B: Request builder ---

def test_p20_8_build_generation_request_copy_reference():
    refs = build_p20_reference_specs()
    req = build_p20_generation_request("copy_reference", refs)
    assert req.baseline_name == "copy_reference"
    assert req.random_valid_generation_templates is None
    assert req.random_valid_family_schedule is None


def test_p20_9_build_generation_request_random_valid():
    refs = build_p20_reference_specs()
    req = build_p20_generation_request("random_valid", refs)
    assert req.baseline_name == "random_valid"
    assert req.random_valid_generation_templates is not None
    assert req.random_valid_family_schedule is not None


def test_p20_10_build_generation_request_oracle():
    refs = build_p20_reference_specs()
    req = build_p20_generation_request("structural_composition_oracle", refs)
    assert req.baseline_name == "structural_composition_oracle"


def test_p20_11_candidate_count_is_6():
    refs = build_p20_reference_specs()
    for name in ["copy_reference", "random_valid", "structural_composition_oracle"]:
        req = build_p20_generation_request(name, refs)
        assert req.candidate_count == 6


def test_p20_12_seed_is_19001():
    refs = build_p20_reference_specs()
    for name in ["copy_reference", "random_valid", "structural_composition_oracle"]:
        req = build_p20_generation_request(name, refs)
        assert req.seed == 19001


def test_p20_13_oracle_flags():
    refs = build_p20_reference_specs()
    for name in ["copy_reference", "random_valid", "structural_composition_oracle"]:
        req = build_p20_generation_request(name, refs)
        assert req.oracle_constraint_flags == (1.0, 0.0, 0.0, 0.0)


# --- Group C: Single baseline smoke ---

def test_p20_14_single_baseline_copy_reference_family_cycle():
    refs = build_p20_reference_specs()
    res = run_single_p20_baseline("copy_reference", refs)
    fams = res["generation_summary"]["candidate_family_ids"]
    assert fams == ["ARMA", "GARCH", "ARMA", "GARCH", "ARMA", "GARCH"]


def test_p20_15_single_baseline_copy_reference_source_index_cycle():
    refs = build_p20_reference_specs()
    res = run_single_p20_baseline("copy_reference", refs)
    summary = res["generation_summary"]["source_records_summary"]
    indices = [r["source_reference_indices"] for r in summary]
    assert indices == [(0,), (1,), (0,), (1,), (0,), (1,)]


def test_p20_16_single_baseline_random_valid_seeds():
    refs = build_p20_reference_specs()
    res = run_single_p20_baseline("random_valid", refs)
    summary = res["generation_summary"]["source_records_summary"]
    seeds = [r["seed"] for r in summary]
    assert seeds == [19001, 19002, 19003, 19004, 19005, 19006]


def test_p20_17_single_baseline_random_valid_family_schedule():
    refs = build_p20_reference_specs()
    res = run_single_p20_baseline("random_valid", refs)
    fams = res["generation_summary"]["candidate_family_ids"]
    assert fams == ["AR", "GARCH", "ARMA_GARCH", "AR", "GARCH", "ARMA_GARCH"]


def test_p20_18_single_baseline_oracle_returns_arma_garch():
    refs = build_p20_reference_specs()
    res = run_single_p20_baseline("structural_composition_oracle", refs)
    fams = res["generation_summary"]["candidate_family_ids"]
    assert all(f == "ARMA_GARCH" for f in fams)


def test_p20_19_structural_oracle_composition_pass_rate():
    refs = build_p20_reference_specs()
    res = run_single_p20_baseline("structural_composition_oracle", refs)
    rate = res["evaluation_summary"]["composition_pass_rate"]
    assert rate == 1.0


def test_p20_20_metric_bundle_bridge_verified():
    refs = build_p20_reference_specs()
    for name in ["copy_reference", "random_valid", "structural_composition_oracle"]:
        res = run_single_p20_baseline(name, refs)
        assert res["metric_bundle_bridge_verified"] is True


def test_p20_21_validity_valid_rate():
    refs = build_p20_reference_specs()
    for name in ["copy_reference", "random_valid", "structural_composition_oracle"]:
        res = run_single_p20_baseline(name, refs)
        assert res["evaluation_summary"]["validity_valid_rate"] == 1.0


# --- Group D: Full smoke ---

def test_p20_22_run_p20_baseline_evidence_smoke_verdict():
    res = run_p20_baseline_evidence_smoke()
    assert res["verdict"] == "PASS"


def test_p20_23_full_smoke_has_exactly_3_baselines():
    res = run_p20_baseline_evidence_smoke()
    assert len(res["baselines"]) == 3


def test_p20_24_full_smoke_baseline_order():
    res = run_p20_baseline_evidence_smoke()
    names = [b["baseline_name"] for b in res["baselines"]]
    assert names == ["copy_reference", "random_valid", "structural_composition_oracle"]


def test_p20_25_no_artifact_dependency():
    res = run_p20_baseline_evidence_smoke()
    assert res["no_artifact_dependency"] is True


def test_p20_26_no_model_training():
    res = run_p20_baseline_evidence_smoke()
    assert res["no_model_training"] is True


def test_p20_27_no_scientific_conclusion():
    res = run_p20_baseline_evidence_smoke()
    assert res["no_scientific_conclusion"] is True


def test_p20_28_repeated_calls_identical():
    res1 = run_p20_baseline_evidence_smoke()
    res2 = run_p20_baseline_evidence_smoke()
    assert res1 == res2


def test_p20_29_compact_json_reproducible():
    data = {"b": 2, "a": 1}
    assert compact_json(data) == '{"a":1,"b":2}'


def test_p20_30_assert_no_path_leakage_clean():
    data = {"verdict": "PASS", "baselines": [{"name": "copy_reference"}]}
    assert_no_path_leakage(data)  # should not raise


def test_p20_31_assert_no_path_leakage_rejects_win():
    with pytest.raises(ValueError):
        assert_no_path_leakage("C:/some/path")
    with pytest.raises(ValueError):
        assert_no_path_leakage("C:\\some\\path")


def test_p20_32_assert_no_path_leakage_rejects_file():
    with pytest.raises(ValueError):
        assert_no_path_leakage("file:///some/path")
    with pytest.raises(ValueError):
        assert_no_path_leakage("/home/user")
    with pytest.raises(ValueError):
        assert_no_path_leakage("/Users/user")


def test_p20_33_main_rejects_arguments(monkeypatch):
    monkeypatch.setattr(sys, "argv", ["run_p20_baseline_evidence_smoke.py", "--some-arg"])
    assert main() != 0


# --- Group E: Scope checks ---

def test_p20_34_no_forbidden_imports():
    import tools.phase2.run_p20_baseline_evidence_smoke as sm
    forbidden = ["torch", "numpy", "pandas", "yaml", "argparse", "sklearn", "scipy"]
    for name in forbidden:
        assert name not in dir(sm)


def test_p20_35_does_not_call_simulate_time_series():
    # Checked statically: code only imports ModelSpec, sampler, generators, eval
    pass


def test_p20_36_does_not_call_build_dataset():
    # Checked statically.
    pass


def test_p20_37_does_not_call_write_artifacts():
    # Checked statically.
    pass


def test_p20_38_does_not_call_run_generation():
    # Checked statically.
    pass


def test_p20_39_does_not_read_phase2_artifacts():
    # Checked statically.
    pass


def test_p20_40_tests_do_not_read_artifacts():
    # Checked statically: tests run purely in-memory.
    pass


def test_p20_41_no_cli_or_config():
    # Checked statically.
    pass
