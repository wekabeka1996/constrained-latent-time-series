"""
Tests for the reproduction runner skeleton.
"""
import json
import os
from pathlib import Path

import pytest

from src.reproduce import main


@pytest.fixture
def temp_demo_config(tmp_path):
    """Create a temporary valid demo config for testing."""
    config_dir = tmp_path / "configs"
    config_dir.mkdir()
    config_path = config_dir / "test_demo.yaml"
    
    config_content = """
project:
  name: econometric-vae-manifold
  mode: demo

model:
  architecture: "BetaVAE_MLP"
  input_dim: 40
  latent_dim: 4
  hidden_dims: [128, 64, 32]
  checkpoint_path: "models/vae_beta50.pth"

vector_schema:
  vector_dim: 40
  struct_dim: 10
  param_dim: 20
  stat_dim: 10
  max_lag_order: 5

generation:
  n_arma: 100
  n_garch: 100
  seed: 42
  mode: "clean_validated"
  allow_supervised_arma_garch: false
  synthetic_training_data_is_methodology: false

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
    return config_path


def test_reproduce_dry_run(temp_demo_config, monkeypatch):
    """Test that the reproduction runner runs in dry-run mode and creates the expected directory structure."""
    
    # We need to change cwd to the tmp_path so it creates `results/` there.
    monkeypatch.chdir(temp_demo_config.parent.parent)

    # Run the reproducer
    args = ["--config", str(temp_demo_config), "--dry-run"]
    exit_code = main(args)
    
    assert exit_code == 0
    
    # Check that a results/reproduction_* directory was created
    results_dir = Path("results")
    assert results_dir.exists()
    
    subdirs = list(results_dir.glob("reproduction_*"))
    assert len(subdirs) == 1
    repro_dir = subdirs[0]
    
    # Check that config was copied
    copied_config = repro_dir / "config_snapshot.yaml"
    assert copied_config.exists()
    
    # Check manifest.json
    manifest_path = repro_dir / "manifest.json"
    assert manifest_path.exists()
    
    with open(manifest_path, "r", encoding="utf-8") as f:
        manifest = json.load(f)
        
    assert manifest["project"] == "econometric-vae-manifold"
    assert manifest["stages"]["stage_0_config_validation"]["status"] == "PASSED"
    # Ensure remaining stages are marked SKIPPED
    assert manifest["stages"]["stage_1_synthetic_methodology_dataset"]["status"] == "SKIPPED"
    assert manifest["stages"]["stage_5_corrected_geometry"]["status"] == "SKIPPED"


def test_reproduce_fails_fast_on_invalid_config(temp_demo_config, monkeypatch):
    """Test that the reproduction runner fails fast if the config allows random fallbacks."""
    
    # Modify config to violate integrity
    config_text = temp_demo_config.read_text(encoding="utf-8")
    config_text = config_text.replace("allow_random_fallbacks: false", "allow_random_fallbacks: true")
    temp_demo_config.write_text(config_text, encoding="utf-8")
    
    monkeypatch.chdir(temp_demo_config.parent.parent)

    args = ["--config", str(temp_demo_config), "--dry-run"]
    exit_code = main(args)
    
    # Should fail due to Stage 0 validation failure
    assert exit_code == 1
    
    repro_dir = list(Path("results").glob("reproduction_*"))[0]
    manifest_path = repro_dir / "manifest.json"
    
    with open(manifest_path, "r", encoding="utf-8") as f:
        manifest = json.load(f)
        
    assert manifest["stages"]["stage_0_config_validation"]["status"] == "FAILED"
    assert "allow_random_fallbacks" in manifest["stages"]["stage_0_config_validation"]["error_message"]
    assert manifest["stages"]["stage_1_synthetic_methodology_dataset"]["status"] == "SKIPPED"
