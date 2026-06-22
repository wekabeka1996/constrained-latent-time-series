# tests/test_phase2_p14_smoke_dry_run.py

import json
import os
import pathlib
import subprocess
import sys
import pytest

from src.phase2.schema import FamilyId
from src.phase2.sampler import validate_generation_request
from src.phase2.simulator import validate_simulation_request
from src.phase2.protocol_presets import (
    validate_protocol_preset_factory_request,
    validate_p13_fewshot_seed_plan_for_preset,
    ProtocolPresetName,
)
from src.phase2.split_runner import validate_phase2_artifact_run_request
from src.phase2.artifacts import sha256_file

from tools.phase2.run_p14_smoke_artifact_dry_run import (
    build_p14_generation_templates,
    build_p14_simulation_template,
    build_p14_factory_request,
    run_p14_smoke_dry_run,
)


def test_p14_1_generation_templates_valid():
    # 1. P14 dry-run template builders create valid GenerationRequest for all four families.
    templates = build_p14_generation_templates()
    assert len(templates) == 4
    families = {t[0] for t in templates}
    assert families == {FamilyId.AR, FamilyId.ARMA, FamilyId.GARCH, FamilyId.ARMA_GARCH}
    for fid, req in templates:
        assert req.family_id == fid
        validate_generation_request(req)  # should not raise


def test_p14_2_simulation_template_valid():
    # 2. P14 simulation template passes validate_simulation_request.
    sim_req = build_p14_simulation_template()
    validate_simulation_request(sim_req)  # should not raise


def test_p14_3_factory_request_valid(tmp_path):
    # 3. P14 factory request passes validate_protocol_preset_factory_request.
    req = build_p14_factory_request(str(tmp_path))
    validate_protocol_preset_factory_request(req)  # should not raise


def test_p14_4_factory_result_split_count(tmp_path):
    # 4. P14 factory result has split_count == 6.
    from src.phase2.protocol_presets import build_phase2_preset_factory_result
    req = build_p14_factory_request(str(tmp_path))
    res = build_phase2_preset_factory_result(req)
    assert res.split_count == 6


def test_p14_5_run_request_valid(tmp_path):
    # 5. P14 run request passes validate_phase2_artifact_run_request.
    from src.phase2.protocol_presets import build_phase2_preset_run_request
    req = build_p14_factory_request(str(tmp_path))
    run_req = build_phase2_preset_run_request(req)
    validate_phase2_artifact_run_request(run_req)  # should not raise


def test_p14_6_expected_split_counts(tmp_path):
    # 6. P14 expected split counts equal: smoke 4000, zero_shot_train 3000, zero_shot_eval 1000, fewshot_1pct_train 10, fewshot_5pct_train 50, fewshot_eval 1000, total 9060
    from src.phase2.protocol_presets import build_phase2_preset_run_request
    req = build_p14_factory_request(str(tmp_path))
    run_req = build_phase2_preset_run_request(req)
    
    expected = {
        "smoke": 4000,
        "zero_shot_train": 3000,
        "zero_shot_eval": 1000,
        "fewshot_1pct_train": 10,
        "fewshot_5pct_train": 50,
        "fewshot_eval": 1000,
    }
    
    total = 0
    for split_req in run_req.split_requests:
        cnt = sum(c for _, c in split_req.sample_count_by_family)
        assert cnt == expected[split_req.artifact_subdir]
        total += cnt
    assert total == 9060


def test_p14_7_creates_exactly_6_split_directories(tmp_path):
    # 7. Dry-run execution with tmp_path output creates exactly 6 split directories.
    out_dir = tmp_path / "smoke_dry_run"
    run_p14_smoke_dry_run(str(out_dir))
    
    subdirs = [p.name for p in out_dir.iterdir() if p.is_dir()]
    assert sorted(subdirs) == sorted([
        "smoke", "zero_shot_train", "zero_shot_eval",
        "fewshot_1pct_train", "fewshot_5pct_train", "fewshot_eval"
    ])
    assert len(subdirs) == 6


def test_p14_8_split_directory_contains_exactly_required_files(tmp_path):
    # 8. Each split directory contains exactly: samples.jsonl, manifest.json
    out_dir = tmp_path / "smoke_dry_run"
    run_p14_smoke_dry_run(str(out_dir))
    
    for split_dir in out_dir.iterdir():
        if split_dir.is_dir():
            files = sorted([f.name for f in split_dir.iterdir()])
            assert files == sorted(["manifest.json", "samples.jsonl"])


def test_p14_9_manifests_parse_as_json(tmp_path):
    # 9. Each manifest parses as JSON.
    out_dir = tmp_path / "smoke_dry_run"
    run_p14_smoke_dry_run(str(out_dir))
    
    for split_dir in out_dir.iterdir():
        if split_dir.is_dir():
            manifest_file = split_dir / "manifest.json"
            with open(manifest_file, "r") as f:
                data = json.load(f)
            assert isinstance(data, dict)


def test_p14_10_manifest_samples_sha256_matches(tmp_path):
    # 10. Each manifest samples_sha256 matches actual samples file.
    out_dir = tmp_path / "smoke_dry_run"
    run_p14_smoke_dry_run(str(out_dir))
    
    for split_dir in out_dir.iterdir():
        if split_dir.is_dir():
            manifest_file = split_dir / "manifest.json"
            samples_file = split_dir / "samples.jsonl"
            with open(manifest_file, "r") as f:
                data = json.load(f)
            
            actual_sha = sha256_file(str(samples_file))
            assert data["samples_sha256"] == actual_sha


def test_p14_11_zero_shot_train_c_count_zero(tmp_path):
    # 11. zero_shot_train manifest has zero_shot_c_train_count == 0.
    out_dir = tmp_path / "smoke_dry_run"
    run_p14_smoke_dry_run(str(out_dir))
    
    zst_manifest = out_dir / "zero_shot_train" / "manifest.json"
    with open(zst_manifest, "r") as f:
        data = json.load(f)
    assert data["zero_shot_c_train_count"] == 0


def test_p14_12_fewshot_seed_plan_validation_passes():
    # 12. fewshot 1%/5% seed plan validation passes.
    validate_p13_fewshot_seed_plan_for_preset(ProtocolPresetName.SMOKE)  # should not raise


def test_p14_13_no_files_created_outside_tmp_path(tmp_path):
    # 13. No files are created outside tmp_path.
    # When we run with tmp_path, only files under tmp_path are written.
    # We can check that the default directory was not touched or created during this test
    default_dir = pathlib.Path("phase2_artifacts/p14_smoke_dry_run")
    pre_exists = default_dir.exists()
    
    out_dir = tmp_path / "test_out"
    run_p14_smoke_dry_run(str(out_dir))
    
    assert default_dir.exists() == pre_exists


def test_p14_14_15_imports_and_models_scope():
    # 14. Script/module does not import torch/numpy/pandas/yaml/argparse.
    # 15. Script/module does not implement model/training/metrics.
    import tools.phase2.run_p14_smoke_artifact_dry_run as dry_run
    assert "torch" not in sys.modules or "torch" not in dir(dry_run)
    assert "numpy" not in sys.modules or "numpy" not in dir(dry_run)
    assert "pandas" not in dir(dry_run)
    assert "yaml" not in dir(dry_run)
    assert "argparse" not in dir(dry_run)
    assert "models" not in dir(dry_run)
    # Check execution functions are not in dir(dry_run)
    assert "build_dataset_in_memory" not in dir(dry_run)
    assert "write_dataset_artifacts" not in dir(dry_run)


def test_p14_16_script_stdout_json_parseable(tmp_path):
    # 16. Script stdout summary can be JSON parsed if script main is invoked via subprocess.
    out_dir = tmp_path / "smoke_dry_run_sub"
    script_path = pathlib.Path("tools/phase2/run_p14_smoke_artifact_dry_run.py").resolve()
    
    env = os.environ.copy()
    env["PYTHONPATH"] = str(pathlib.Path(".").resolve())

    res = subprocess.run(
        [sys.executable, str(script_path), str(out_dir)],
        capture_output=True,
        text=True,
        check=True,
        env=env
    )
    
    stdout_str = res.stdout.strip()
    summary = json.loads(stdout_str)
    assert summary["verdict"] == "PASS"
    assert summary["total_sample_count"] == 9060
    assert len(summary["per_split"]) == 6


def test_p14_17_script_exits_0(tmp_path):
    # 17. Script exits 0 when run on tmp_path or equivalent controlled output.
    out_dir = tmp_path / "smoke_dry_run_sub"
    script_path = pathlib.Path("tools/phase2/run_p14_smoke_artifact_dry_run.py").resolve()
    
    env = os.environ.copy()
    env["PYTHONPATH"] = str(pathlib.Path(".").resolve())

    res = subprocess.run(
        [sys.executable, str(script_path), str(out_dir)],
        capture_output=True,
        text=True,
        env=env
    )
    assert res.returncode == 0
