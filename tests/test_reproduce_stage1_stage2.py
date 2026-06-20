import json
import os
from pathlib import Path

import numpy as np
import pytest
import torch

from src.reproduce import main


@pytest.fixture
def temp_environment(tmp_path):
    """Setup a valid temp config and a tiny valid checkpoint for testing."""
    config_dir = tmp_path / "configs"
    config_dir.mkdir()
    config_path = config_dir / "test_stage_1_2.yaml"
    
    models_dir = tmp_path / "models"
    models_dir.mkdir()
    ckpt_path = models_dir / "test_vae.pth"
    
    # Create a tiny dummy checkpoint that matches D_INPUT=40
    # and has the correct architecture keys.
    d_in = 40
    d_lat = 4
    d_h1 = 8
    d_h2 = 8
    
    dummy_state = {
        "encoder.0.weight": torch.randn(d_h1, d_in),
        "encoder.0.bias": torch.randn(d_h1),
        "encoder.2.weight": torch.randn(d_h2, d_h1),
        "encoder.2.bias": torch.randn(d_h2),
        "fc_mu.weight": torch.randn(d_lat, d_h2),
        "fc_mu.bias": torch.randn(d_lat),
        "fc_logvar.weight": torch.randn(d_lat, d_h2),
        "fc_logvar.bias": torch.randn(d_lat),
        "decoder.0.weight": torch.randn(d_h2, d_lat),
        "decoder.0.bias": torch.randn(d_h2),
        "decoder.2.weight": torch.randn(d_h1, d_h2),
        "decoder.2.bias": torch.randn(d_h1),
        "decoder.4.weight": torch.randn(d_in, d_h1),
        "decoder.4.bias": torch.randn(d_in),
    }
    torch.save(dummy_state, ckpt_path)
    
    config_content = f"""
project:
  name: econometric-vae-manifold
  mode: demo

model:
  architecture: "BetaVAE_MLP"
  input_dim: 40
  latent_dim: 4
  hidden_dims: [8, 8]
  checkpoint_path: "{str(ckpt_path).replace(os.sep, '/')}"

vector_schema:
  vector_dim: 40
  struct_dim: 10
  param_dim: 20
  stat_dim: 10
  max_lag_order: 5

generation:
  n_arma: 10
  n_garch: 10
  seed: 42
  mode: "clean_validated"
  allow_supervised_arma_garch: false
  synthetic_training_data_is_methodology: true

validation:
  activation_threshold: 0.5
  activation_epsilon: 0.1
  tolerance: 0.05

paths:
  market_data_csv: null
  output_dir: "results/generation_2/robustness/"

integrity:
  allow_random_fallbacks: false
  allow_fake_market_data: false
  allow_cached_artifacts: true
  require_artifact_provenance: true
  fail_fast_on_missing_required_artifacts: true
"""
    config_path.write_text(config_content, encoding="utf-8")
    
    return config_path, ckpt_path


def test_stage1_and_2_execution(temp_environment, monkeypatch, tmp_path):
    config_path, ckpt_path = temp_environment
    
    # We chdir into tmp_path so "results/" is created there
    monkeypatch.chdir(tmp_path)
    
    # Create legacy data folder just to verify it isn't touched
    legacy_dir = tmp_path / "data" / "generated"
    legacy_dir.mkdir(parents=True)
    legacy_file = legacy_dir / "old.npy"
    legacy_file.write_text("old data")

    args = ["--config", str(config_path), "--stages", "1,2"]
    exit_code = main(args)
    assert exit_code == 0
    
    repro_dirs = list(Path("results").glob("reproduction_*"))
    assert len(repro_dirs) == 1
    repro_dir = repro_dirs[0]
    
    manifest_path = repro_dir / "manifest.json"
    with open(manifest_path, "r", encoding="utf-8") as f:
        manifest = json.load(f)
        
    # test_stage1_manifest_records_outputs
    assert manifest["stages"]["stage_1_synthetic_methodology_dataset"]["status"] == "PASSED"
    assert "Z_train" in manifest["stages"]["stage_1_synthetic_methodology_dataset"]["outputs"]
    assert "train_labels" in manifest["stages"]["stage_1_synthetic_methodology_dataset"]["outputs"]
    
    # test_stage1_generates_expected_files
    data_dir = repro_dir / "data"
    assert (data_dir / "Z_train.npy").exists()
    assert (data_dir / "train_labels.npy").exists()
    
    # test_stage1_writes_only_inside_reproduction_dir
    # legacy_file must be untouched
    assert legacy_file.read_text() == "old data"
    assert len(list(legacy_dir.iterdir())) == 1
    
    # test_stage2_records_checkpoint_hash
    assert manifest["stages"]["stage_2_vae_checkpoint_verification"]["status"] == "PASSED"
    assert "checkpoint_sha256" in manifest["stages"]["stage_2_vae_checkpoint_verification"]["outputs"]
    
    # test_stage2_writes_checkpoint_verification_report
    report_path = repro_dir / "reports" / "checkpoint_verification.json"
    assert report_path.exists()
    with open(report_path, "r") as f:
        report = json.load(f)
    assert report["forward_pass_ok"] is True
    assert report["input_dim"] == 40
    assert report["output_shape"][1] == 40
    
    # test_future_stages_remain_skipped
    assert manifest["stages"]["stage_3_latent_extraction"]["status"] == "SKIPPED"
    assert manifest["stages"]["stage_4_structural_validation"]["status"] == "SKIPPED"
    assert manifest["stages"]["stage_5_corrected_geometry"]["status"] == "SKIPPED"
    assert manifest["stages"]["stage_6_empirical_stress_test"]["status"] == "SKIPPED"
    
    # test_runner_does_not_write_to_legacy_data_generated
    # Checked above (len(list(legacy_dir.iterdir())) == 1)
    
    # test_runner_does_not_create_fake_market_data
    # No CSV files should exist in repro_dir
    assert len(list(repro_dir.glob("**/*.csv"))) == 0
    
    # test_runner_does_not_create_geometry_outputs
    geometry_dir = repro_dir / "geometry"
    assert not geometry_dir.exists()


def test_stage2_requires_checkpoint_file(temp_environment, monkeypatch, tmp_path):
    config_path, ckpt_path = temp_environment
    monkeypatch.chdir(tmp_path)
    
    # Delete the checkpoint so it fails
    ckpt_path.unlink()
    
    args = ["--config", str(config_path), "--stages", "2"]
    exit_code = main(args)
    
    assert exit_code == 1
    
    repro_dirs = list(Path("results").glob("reproduction_*"))
    repro_dir = repro_dirs[0]
    manifest_path = repro_dir / "manifest.json"
    
    with open(manifest_path, "r", encoding="utf-8") as f:
        manifest = json.load(f)
        
    assert manifest["stages"]["stage_2_vae_checkpoint_verification"]["status"] == "FAILED"
    assert "Required file" in manifest["stages"]["stage_2_vae_checkpoint_verification"]["error_message"]
