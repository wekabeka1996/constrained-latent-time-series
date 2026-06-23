# tests/test_phase2_p16_dev_dry_run.py

import json
import os
import pathlib
import sys
import pytest

from src.phase2.schema import FamilyId
from src.phase2.protocol_presets import (
    validate_protocol_preset_factory_request,
    validate_p13_fewshot_seed_plan_for_preset,
    ProtocolPresetName,
    build_phase2_preset_factory_result,
    build_phase2_preset_run_request,
)
from src.phase2.split_runner import validate_phase2_artifact_run_request
from tools.phase2.audit_phase2_artifact_manifest import (
    assert_no_absolute_paths_in_data,
    compact_json,
)

from tools.phase2.run_p16_dev_artifact_dry_run import (
    expected_p16_dev_subdirs,
    expected_p16_dev_counts_by_subdir,
    build_p16_factory_request,
    run_p16_dev_dry_run,
)


def test_p16_1_subdirs_order():
    subdirs = expected_p16_dev_subdirs()
    assert subdirs == (
        "smoke",
        "zero_shot_train",
        "zero_shot_eval",
        "fewshot_1pct_train",
        "fewshot_5pct_train",
        "fewshot_eval",
    )


def test_p16_2_expected_counts():
    counts = expected_p16_dev_counts_by_subdir()
    expected = (
        ("smoke", 40000),
        ("zero_shot_train", 30000),
        ("zero_shot_eval", 10000),
        ("fewshot_1pct_train", 100),
        ("fewshot_5pct_train", 500),
        ("fewshot_eval", 10000),
    )
    assert counts == expected


def test_p16_3_total_samples():
    counts = expected_p16_dev_counts_by_subdir()
    total = sum(cnt for _, cnt in counts)
    assert total == 90600


def test_p16_4_factory_request_valid(tmp_path):
    req = build_p16_factory_request(str(tmp_path))
    validate_protocol_preset_factory_request(req)  # should not raise


def test_p16_5_run_request_valid(tmp_path):
    req = build_p16_factory_request(str(tmp_path))
    run_req = build_phase2_preset_run_request(req)
    validate_phase2_artifact_run_request(run_req)  # should not raise


def test_p16_6_run_request_split_count(tmp_path):
    req = build_p16_factory_request(str(tmp_path))
    run_req = build_phase2_preset_run_request(req)
    assert len(run_req.split_requests) == 6


def test_p16_7_run_request_split_order(tmp_path):
    req = build_p16_factory_request(str(tmp_path))
    run_req = build_phase2_preset_run_request(req)
    subdirs = [s.artifact_subdir for s in run_req.split_requests]
    assert tuple(subdirs) == expected_p16_dev_subdirs()


def test_p16_8_run_request_counts(tmp_path):
    req = build_p16_factory_request(str(tmp_path))
    run_req = build_phase2_preset_run_request(req)
    expected_counts = dict(expected_p16_dev_counts_by_subdir())
    for split_req in run_req.split_requests:
        cnt = sum(c for _, c in split_req.sample_count_by_family)
        assert cnt == expected_counts[split_req.artifact_subdir]


def test_p16_9_zero_shot_train_excludes_c(tmp_path):
    req = build_p16_factory_request(str(tmp_path))
    run_req = build_phase2_preset_run_request(req)
    zst_req = next(s for s in run_req.split_requests if s.artifact_subdir == "zero_shot_train")
    counts = dict(zst_req.sample_count_by_family)
    assert counts.get(FamilyId.ARMA_GARCH, 0) == 0


def test_p16_10_zero_shot_train_exclusion_flag(tmp_path):
    req = build_p16_factory_request(str(tmp_path))
    run_req = build_phase2_preset_run_request(req)
    zst_req = next(s for s in run_req.split_requests if s.artifact_subdir == "zero_shot_train")
    assert zst_req.enforce_zero_shot_c_train_exclusion is True


def test_p16_11_zero_shot_eval_c_only(tmp_path):
    req = build_p16_factory_request(str(tmp_path))
    run_req = build_phase2_preset_run_request(req)
    zse_req = next(s for s in run_req.split_requests if s.artifact_subdir == "zero_shot_eval")
    for fid, cnt in zse_req.sample_count_by_family:
        if cnt > 0:
            assert fid == FamilyId.ARMA_GARCH


def test_p16_12_fewshot_1pct_train_c_only_and_count(tmp_path):
    req = build_p16_factory_request(str(tmp_path))
    run_req = build_phase2_preset_run_request(req)
    fs1_req = next(s for s in run_req.split_requests if s.artifact_subdir == "fewshot_1pct_train")
    total = 0
    for fid, cnt in fs1_req.sample_count_by_family:
        if cnt > 0:
            assert fid == FamilyId.ARMA_GARCH
            total += cnt
    assert total == 100


def test_p16_13_fewshot_5pct_train_c_only_and_count(tmp_path):
    req = build_p16_factory_request(str(tmp_path))
    run_req = build_phase2_preset_run_request(req)
    fs5_req = next(s for s in run_req.split_requests if s.artifact_subdir == "fewshot_5pct_train")
    total = 0
    for fid, cnt in fs5_req.sample_count_by_family:
        if cnt > 0:
            assert fid == FamilyId.ARMA_GARCH
            total += cnt
    assert total == 500


def test_p16_14_fewshot_eval_c_only_and_count(tmp_path):
    req = build_p16_factory_request(str(tmp_path))
    run_req = build_phase2_preset_run_request(req)
    fse_req = next(s for s in run_req.split_requests if s.artifact_subdir == "fewshot_eval")
    total = 0
    for fid, cnt in fse_req.sample_count_by_family:
        if cnt > 0:
            assert fid == FamilyId.ARMA_GARCH
            total += cnt
    assert total == 10000


def test_p16_15_validate_p13_fewshot_seed_plan():
    validate_p13_fewshot_seed_plan_for_preset(ProtocolPresetName.DEV)  # should not raise


def test_p16_16_fewshot_seed_range_prefix():
    # fewshot_1pct_train uses base seed 10_012_101, count 100 -> range [10_012_101, 10_012_200]
    # fewshot_5pct_train uses base seed 10_012_101, count 500 -> range [10_012_101, 10_012_600]
    # 1% seed range is prefix subset of 5% seed range.
    from src.phase2.protocol_presets import seed_range_for_base_and_count, range_is_prefix_subset
    r1 = seed_range_for_base_and_count(10012101, 100)
    r5 = seed_range_for_base_and_count(10012101, 500)
    assert range_is_prefix_subset(r1, r5)


def test_p16_17_c_eval_non_overlap():
    from src.phase2.protocol_presets import seed_range_for_base_and_count
    
    # zero_shot_eval: base=12003, count=10000 -> [12003, 22002]
    # fewshot_5pct_train: base=10012101, count=500 -> [10012101, 10012600]
    # fewshot_eval: base=20012105, count=10000 -> [20012105, 20022104]
    
    r_zse = seed_range_for_base_and_count(12003, 10000)
    r_fs5 = seed_range_for_base_and_count(10012101, 500)
    r_fse = seed_range_for_base_and_count(20012105, 10000)
    
    def ranges_overlap(rA, rB):
        # A: (start, end)
        return not (rA[1] < rB[0] or rB[1] < rA[0])
        
    assert not ranges_overlap(r_zse, r_fs5)
    assert not ranges_overlap(r_fse, r_fs5)
    assert not ranges_overlap(r_zse, r_fse)


def test_p16_18_no_forbidden_imports():
    import tools.phase2.run_p16_dev_artifact_dry_run as p16_mod
    forbidden = ["torch", "numpy", "pandas", "yaml", "argparse"]
    for name in forbidden:
        assert name not in dir(p16_mod)


def test_p16_19_no_model_code():
    import tools.phase2.run_p16_dev_artifact_dry_run as p16_mod
    forbidden_terms = ["model", "training", "metrics"]
    for term in forbidden_terms:
        # Check that we don't implement anything like that
        assert not hasattr(p16_mod, f"run_{term}")


def test_p16_20_mock_path_hygiene_test():
    payload = {
        "verdict": "PASS",
        "output_root_dir": "phase2_artifacts/p16_dev_dry_run",
        "per_split": [
            {
                "samples_path": "phase2_artifacts/p16_dev_dry_run/smoke/samples.jsonl",
            }
        ]
    }
    assert_no_absolute_paths_in_data(payload)  # should not raise
    
    dirty_payload = {
        "output_root_dir": "C:/Users/wekab/phase2_artifacts/p16_dev_dry_run",
    }
    with pytest.raises(ValueError, match="Absolute path leakage detected"):
        assert_no_absolute_paths_in_data(dirty_payload)


@pytest.mark.skip(reason="DEV preset artifact generation is slow, skipped in unit tests")
def test_p16_integration(tmp_path):
    out_dir = tmp_path / "p16_dev"
    summary = run_p16_dev_dry_run(str(out_dir))
    assert summary["verdict"] == "PASS"
    assert summary["total_sample_count"] == 90600
