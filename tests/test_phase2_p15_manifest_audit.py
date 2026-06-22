# tests/test_phase2_p15_manifest_audit.py

import json
import os
import pathlib
import sys
import subprocess
import pytest

from tools.phase2.audit_phase2_artifact_manifest import (
    repository_root,
    normalize_repo_relative_path,
    sha256_file,
    audit_manifest_file,
    audit_phase2_artifact_root,
    audit_phase2_artifacts,
    compact_json,
    assert_no_absolute_paths_in_data,
)
from tools.phase2.run_p14_smoke_artifact_dry_run import run_p14_smoke_dry_run


def test_repository_root():
    root = repository_root()
    assert root.exists()
    assert root.is_dir()
    assert (root / "src").exists()


def test_normalize_repo_relative_path():
    root = repository_root()
    # Test path normalization under repo root
    test_path = root / "src" / "phase2" / "artifacts.py"
    rel_path = normalize_repo_relative_path(test_path, root)
    assert rel_path == "src/phase2/artifacts.py"
    
    # Test relative path preservation (should just resolve first)
    rel_test = "src/phase2/artifacts.py"
    assert normalize_repo_relative_path(root / rel_test, root) == "src/phase2/artifacts.py"


def test_normalize_repo_relative_path_outside_root(tmp_path):
    root = repository_root()
    # Test outside-root rejection
    outside = tmp_path / "some_file.txt"
    outside.touch()
    with pytest.raises(ValueError, match="is outside repository root"):
        normalize_repo_relative_path(outside, root)


def test_normalize_repo_relative_path_empty_rejection():
    root = repository_root()
    with pytest.raises(ValueError, match="Path cannot be"):
        normalize_repo_relative_path("", root)
    with pytest.raises(ValueError, match="Path cannot be"):
        normalize_repo_relative_path("   ", root)
    with pytest.raises(ValueError, match="Path cannot be"):
        normalize_repo_relative_path(None, root)


def test_absolute_path_leakage_rejection():
    # Test assert_no_absolute_paths_in_data with various absolute paths
    data_clean = {"path": "src/phase2/artifacts.py", "sub": ["another/relative/path"]}
    assert_no_absolute_paths_in_data(data_clean)  # should not raise
    
    bad_paths = [
        "C:/Users/name/repo",
        "C:\\Users\\name\\repo",
        "/home/user/repo",
        "/Users/name/repo",
        "file:///C:/Users/name/repo",
    ]
    for p in bad_paths:
        data_dirty = {"path": p}
        with pytest.raises(ValueError, match="Absolute path leakage detected"):
            assert_no_absolute_paths_in_data(data_dirty)


def test_audit_on_real_p14_dry_run(tmp_path):
    out_dir = tmp_path / "smoke_dry_run"
    
    # 1. Generate real artifacts
    summary = run_p14_smoke_dry_run(str(out_dir))
    
    # Verify the summary structure and data
    assert summary["verdict"] == "PASS"
    assert summary["total_split_count"] == 6
    assert summary["total_sample_count"] == 9060
    
    expected_splits = {
        "smoke": 4000,
        "zero_shot_train": 3000,
        "zero_shot_eval": 1000,
        "fewshot_1pct_train": 10,
        "fewshot_5pct_train": 50,
        "fewshot_eval": 1000,
    }
    for item in summary["per_split"]:
        subdir = item["artifact_subdir"]
        assert item["sample_count"] == expected_splits[subdir]
        
    # Verify paths are relative (no leakage)
    assert_no_absolute_paths_in_data(summary)


def test_audit_failures_corrupted_samples(tmp_path):
    out_dir = tmp_path / "smoke_dry_run"
    run_p14_smoke_dry_run(str(out_dir))
    
    # Corrupt samples file in one of the splits
    samples_file = out_dir / "smoke" / "samples.jsonl"
    with open(samples_file, "a") as f:
        f.write("\nCorrupted extra line")
        
    # Running audit should fail due to hash mismatch
    with pytest.raises(ValueError, match="Samples SHA256 mismatch"):
        audit_phase2_artifacts(str(out_dir))


def test_audit_failures_unexpected_file(tmp_path):
    out_dir = tmp_path / "smoke_dry_run"
    run_p14_smoke_dry_run(str(out_dir))
    
    # Add unexpected file in the root
    (out_dir / "unexpected.txt").touch()
    with pytest.raises(ValueError, match="Unexpected files at root level"):
        audit_phase2_artifacts(str(out_dir))
        
    # Remove unexpected file and add inside a subdir
    (out_dir / "unexpected.txt").unlink()
    (out_dir / "smoke" / "unexpected.txt").touch()
    with pytest.raises(ValueError, match="contains unexpected files"):
        audit_phase2_artifacts(str(out_dir))


def test_audit_failures_missing_manifest(tmp_path):
    out_dir = tmp_path / "smoke_dry_run"
    run_p14_smoke_dry_run(str(out_dir))
    
    # Remove manifest in a split
    (out_dir / "smoke" / "manifest.json").unlink()
    with pytest.raises(ValueError, match="contains unexpected files or is missing required ones"):
        audit_phase2_artifacts(str(out_dir))


def test_audit_failures_missing_samples(tmp_path):
    out_dir = tmp_path / "smoke_dry_run"
    run_p14_smoke_dry_run(str(out_dir))
    
    # Remove samples.jsonl in a split
    (out_dir / "smoke" / "samples.jsonl").unlink()
    with pytest.raises(ValueError, match="contains unexpected files or is missing required ones"):
        audit_phase2_artifacts(str(out_dir))


def test_stdout_json_has_no_absolute_paths(tmp_path):
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
    
    assert_no_absolute_paths_in_data(summary)


def test_no_forbidden_imports():
    import tools.phase2.audit_phase2_artifact_manifest as audit_mod
    import tools.phase2.run_p14_smoke_artifact_dry_run as dry_run
    
    forbidden = ["torch", "numpy", "pandas", "yaml", "argparse"]
    for name in forbidden:
        assert name not in dir(audit_mod)
        assert name not in dir(dry_run)

