"""
Tests for Stage 3 (Latent Representation Extraction) of the reproduction runner.
"""
import json
import os
from pathlib import Path

import numpy as np
import pytest
import torch

from src.reproduce import main


@pytest.fixture
def temp_environment(tmp_path):
    """Setup a valid temp config, a tiny valid checkpoint, and Stage 1 outputs for testing."""
    config_dir = tmp_path / "configs"
    config_dir.mkdir()
    config_path = config_dir / "test_stage3.yaml"
    
    models_dir = tmp_path / "models"
    models_dir.mkdir()
    ckpt_path = models_dir / "test_vae.pth"
    
    # Create a tiny dummy checkpoint that matches input_dim = 40, latent_dim = 8
    d_in = 40
    d_lat = 8
    d_h1 = 16
    d_h2 = 16
    
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
  latent_dim: 8
  hidden_dims: [16, 16]
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


def test_stage3_requires_stage1_outputs(temp_environment, monkeypatch, tmp_path):
    """Verify Stage 3 fails fast if Stage 1 outputs are missing."""
    config_path, ckpt_path = temp_environment
    monkeypatch.chdir(tmp_path)
    
    # Run only Stage 3 (without Stage 1/2 in same run)
    args = ["--config", str(config_path), "--stages", "3"]
    exit_code = main(args)
    assert exit_code == 1


def test_stage3_requires_checkpoint(temp_environment, monkeypatch, tmp_path):
    """Verify Stage 3 fails fast if checkpoint is missing."""
    config_path, ckpt_path = temp_environment
    monkeypatch.chdir(tmp_path)
    
    # Delete the checkpoint
    ckpt_path.unlink()
    
    args = ["--config", str(config_path), "--stages", "1,2,3"]
    exit_code = main(args)
    assert exit_code == 1
    
    repro_dirs = list(Path("results").glob("reproduction_*"))
    assert len(repro_dirs) == 1
    repro_dir = repro_dirs[0]
    manifest_path = repro_dir / "manifest.json"
    
    with open(manifest_path, "r", encoding="utf-8") as f:
        manifest = json.load(f)
        
    assert manifest["stages"]["stage_2_vae_checkpoint_verification"]["status"] == "FAILED"
    assert manifest["stages"]["stage_3_latent_extraction"]["status"] == "SKIPPED"


def test_stage3_writes_only_inside_reproduction_dir(temp_environment, monkeypatch, tmp_path):
    """Verify Stage 3 does not write outside the isolated reproduction folder (e.g. to root or data/generated)."""
    config_path, ckpt_path = temp_environment
    monkeypatch.chdir(tmp_path)
    
    # Create a decoy legacy directory
    legacy_dir = tmp_path / "data" / "generated"
    legacy_dir.mkdir(parents=True)
    legacy_file = legacy_dir / "old.npy"
    legacy_file.write_text("untouched")
    
    args = ["--config", str(config_path), "--stages", "1,2,3"]
    exit_code = main(args)
    assert exit_code == 0
    
    # Verify decoy remains untouched
    assert legacy_file.read_text() == "untouched"
    assert len(list(legacy_dir.iterdir())) == 1


def test_stage3_generates_latent_files(temp_environment, monkeypatch, tmp_path):
    """Verify Stage 3 outputs the correct numpy arrays inside latent/."""
    config_path, ckpt_path = temp_environment
    monkeypatch.chdir(tmp_path)
    
    args = ["--config", str(config_path), "--stages", "1,2,3"]
    exit_code = main(args)
    assert exit_code == 0
    
    repro_dir = list(Path("results").glob("reproduction_*"))[0]
    latent_dir = repro_dir / "latent"
    
    assert (latent_dir / "z_train.npy").exists()
    assert (latent_dir / "z_train_ar.npy").exists()
    assert (latent_dir / "z_train_garch.npy").exists()
    assert (latent_dir / "mu_train.npy").exists()
    assert (latent_dir / "logvar_train.npy").exists()
    
    # Refinement check: z_train equals mu_train
    z_train = np.load(latent_dir / "z_train.npy")
    mu_train = np.load(latent_dir / "mu_train.npy")
    logvar_train = np.load(latent_dir / "logvar_train.npy")
    
    np.testing.assert_array_equal(z_train, mu_train)
    
    # Shape checks: latent_dim = 8
    assert z_train.shape[1] == 8
    assert logvar_train.shape == z_train.shape


def test_stage3_splits_latents_by_labels(temp_environment, monkeypatch, tmp_path):
    """Verify Stage 3 splits latent embeddings correctly by AR/GARCH labels."""
    config_path, ckpt_path = temp_environment
    monkeypatch.chdir(tmp_path)
    
    args = ["--config", str(config_path), "--stages", "1,2,3"]
    exit_code = main(args)
    assert exit_code == 0
    
    repro_dir = list(Path("results").glob("reproduction_*"))[0]
    latent_dir = repro_dir / "latent"
    
    z_train = np.load(latent_dir / "z_train.npy")
    z_train_ar = np.load(latent_dir / "z_train_ar.npy")
    z_train_garch = np.load(latent_dir / "z_train_garch.npy")
    
    # Since config has n_arma=10 and n_garch=10, we expect 10 samples of each
    assert len(z_train_ar) == 10
    assert len(z_train_garch) == 10
    assert len(z_train) == 20
    
    # Assert they sum up to total length
    assert len(z_train_ar) + len(z_train_garch) == len(z_train)


def test_stage3_manifest_records_hashes(temp_environment, monkeypatch, tmp_path):
    """Verify Stage 3 updates the manifest with correct path/sha256 structure for all outputs."""
    config_path, ckpt_path = temp_environment
    monkeypatch.chdir(tmp_path)
    
    args = ["--config", str(config_path), "--stages", "1,2,3"]
    exit_code = main(args)
    assert exit_code == 0
    
    repro_dir = list(Path("results").glob("reproduction_*"))[0]
    manifest_path = repro_dir / "manifest.json"
    
    with open(manifest_path, "r", encoding="utf-8") as f:
        manifest = json.load(f)
        
    stage_3_outputs = manifest["stages"]["stage_3_latent_extraction"]["outputs"]
    
    for key in ["z_train", "z_train_ar", "z_train_garch", "mu_train", "logvar_train", "latent_report"]:
        assert key in stage_3_outputs
        assert "path" in stage_3_outputs[key]
        assert "sha256" in stage_3_outputs[key]
        
        # Verify physical path existence
        physical_path = repro_dir / stage_3_outputs[key]["path"]
        assert physical_path.exists()


def test_stage3_writes_latent_report(temp_environment, monkeypatch, tmp_path):
    """Verify Stage 3 writes reports/latent_extraction.json with correct format."""
    config_path, ckpt_path = temp_environment
    monkeypatch.chdir(tmp_path)
    
    args = ["--config", str(config_path), "--stages", "1,2,3"]
    exit_code = main(args)
    assert exit_code == 0
    
    repro_dir = list(Path("results").glob("reproduction_*"))[0]
    report_path = repro_dir / "reports" / "latent_extraction.json"
    
    assert report_path.exists()
    with open(report_path, "r", encoding="utf-8") as f:
        report = json.load(f)
        
    assert "input_path" in report
    assert "labels_path" in report
    assert "checkpoint_path" in report
    assert report["n_samples"] == 20 # n_arma=10 and n_garch=10 -> 20 samples
    assert report["latent_dim"] == 8
    assert report["label_counts"] == {"AR": 10, "GARCH": 10}
    assert "outputs" in report
    assert "z_train" in report["outputs"]
    assert "z_train_ar" in report["outputs"]
    assert "z_train_garch" in report["outputs"]
    assert "notes" in report


def test_stage3_does_not_create_generated_valid_thetas(temp_environment, monkeypatch, tmp_path):
    """Verify Stage 3 does not generate generated_valid_thetas.npy."""
    config_path, ckpt_path = temp_environment
    monkeypatch.chdir(tmp_path)
    
    args = ["--config", str(config_path), "--stages", "1,2,3"]
    exit_code = main(args)
    assert exit_code == 0
    
    repro_dir = list(Path("results").glob("reproduction_*"))[0]
    
    assert not (repro_dir / "data" / "generated_valid_thetas.npy").exists()
    assert len(list(repro_dir.glob("**/generated_valid_thetas*"))) == 0


def test_stage3_does_not_write_to_legacy_data_generated(temp_environment, monkeypatch, tmp_path):
    """Verify Stage 3 does not write output files to data/generated/."""
    config_path, ckpt_path = temp_environment
    monkeypatch.chdir(tmp_path)
    
    args = ["--config", str(config_path), "--stages", "1,2,3"]
    exit_code = main(args)
    assert exit_code == 0
    
    legacy_data_dir = Path("data/generated")
    assert not legacy_data_dir.exists()


def test_future_stages_remain_skipped_after_stage3(temp_environment, monkeypatch, tmp_path):
    """Verify stages 4, 5, 6, and 7 remain SKIPPED in manifest after Stage 3."""
    config_path, ckpt_path = temp_environment
    monkeypatch.chdir(tmp_path)
    
    args = ["--config", str(config_path), "--stages", "1,2,3"]
    exit_code = main(args)
    assert exit_code == 0
    
    repro_dir = list(Path("results").glob("reproduction_*"))[0]
    manifest_path = repro_dir / "manifest.json"
    
    with open(manifest_path, "r", encoding="utf-8") as f:
        manifest = json.load(f)
        
    assert manifest["stages"]["stage_3_latent_extraction"]["status"] == "PASSED"
    assert manifest["stages"]["stage_4_structural_validation"]["status"] == "SKIPPED"
    assert manifest["stages"]["stage_5_corrected_geometry"]["status"] == "SKIPPED"
    assert manifest["stages"]["stage_6_empirical_stress_test"]["status"] == "SKIPPED"
    assert manifest["stages"]["stage_7_claim_verdict"]["status"] == "SKIPPED"


def test_stage3_invalid_labels_fail_fast(temp_environment, monkeypatch, tmp_path):
    """Verify Stage 3 fails fast if training labels are invalid or group is empty."""
    config_path, ckpt_path = temp_environment
    monkeypatch.chdir(tmp_path)
    
    # Mock generate_training_dataset_from_config to return invalid labels
    import src.reproduce
    def mock_generate(cfg):
        X = np.random.randn(20, 40)
        y = np.array(["INVALID"] * 20, dtype=str)
        return X, y
        
    monkeypatch.setattr(src.reproduce, "generate_training_dataset_from_config", mock_generate)
    
    args = ["--config", str(config_path), "--stages", "1,2,3"]
    exit_code = main(args)
    
    # Should fail fast because of invalid labels
    assert exit_code == 1
    
    # Check that manifest records FAILED status for stage 3
    repro_dirs = list(Path("results").glob("reproduction_*"))
    assert len(repro_dirs) == 1
    repro_dir = repro_dirs[0]
    manifest_path = repro_dir / "manifest.json"
    
    with open(manifest_path, "r", encoding="utf-8") as f:
        manifest = json.load(f)
        
    assert manifest["stages"]["stage_3_latent_extraction"]["status"] == "FAILED"
    assert "Invalid label" in manifest["stages"]["stage_3_latent_extraction"]["error_message"]

