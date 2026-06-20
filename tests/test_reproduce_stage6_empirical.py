"""
Tests for Stage 6 (Empirical Stress Test with Real Data Provenance) of the reproduction runner.
"""
import csv
import json
import os
import hashlib
from pathlib import Path

import numpy as np
import pytest
import torch
import pandas as pd

from src.reproduce import main, run_stage_6, compute_sha256


@pytest.fixture
def temp_environment(tmp_path):
    """Setup a valid temp config, a tiny valid checkpoint, Stage 1 outputs, and mock claim matrix for testing."""
    config_dir = tmp_path / "configs"
    config_dir.mkdir()
    config_path = config_dir / "test_stage6.yaml"
    
    models_dir = tmp_path / "models"
    models_dir.mkdir()
    ckpt_path = models_dir / "test_vae.pth"
    
    # Create docs dir and mock CLAIM_VERDICT_MATRIX.md
    docs_dir = tmp_path / "docs"
    docs_dir.mkdir()
    claim_matrix_path = docs_dir / "CLAIM_VERDICT_MATRIX.md"
    claim_matrix_content = "# Claim Verdict Matrix\nMock claim matrix for testing Stage 6."
    claim_matrix_path.write_text(claim_matrix_content, encoding="utf-8")
    
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

geometry:
  max_geometry_points: 5
  finite_difference_eps: 1.0e-4
  metric_regularization_eps: 1.0e-8
"""
    config_path.write_text(config_content, encoding="utf-8")
    
    return config_path, ckpt_path


def test_stage6_requires_stage1_to_stage5(temp_environment, monkeypatch, tmp_path):
    """Verify Stage 6 fails fast if Stages 1-5 are not requested in the same run."""
    config_path, ckpt_path = temp_environment
    monkeypatch.chdir(tmp_path)
    
    # Run only stage 6
    args = ["--config", str(config_path), "--stages", "6"]
    exit_code = main(args)
    assert exit_code == 1
    
    # Run stages 1-4 and 6, missing stage 5
    args = ["--config", str(config_path), "--stages", "1,2,3,4,6"]
    exit_code = main(args)
    assert exit_code == 1


def test_stage6_null_market_data_marks_needs_real_data(temp_environment, monkeypatch, tmp_path):
    """Verify Case A (null market data) marks manifest stage_6 status as SKIPPED and reason NEEDS_REAL_DATA."""
    config_path, ckpt_path = temp_environment
    monkeypatch.chdir(tmp_path)
    
    # Run Stages 1-6 with default null market_data_csv
    args = ["--config", str(config_path), "--stages", "1,2,3,4,5,6"]
    exit_code = main(args)
    assert exit_code == 0
    
    repro_dirs = list(Path("results").glob("reproduction_*"))
    assert len(repro_dirs) == 1
    repro_dir = repro_dirs[0]
    
    manifest_path = repro_dir / "manifest.json"
    with open(manifest_path, "r", encoding="utf-8") as f:
        manifest = json.load(f)
        
    stage_6 = manifest["stages"]["stage_6_empirical_stress_test"]
    assert stage_6["status"] == "SKIPPED"
    assert "NEEDS_REAL_DATA" in stage_6["reason"]


def test_stage6_writes_empirical_summary(temp_environment, monkeypatch, tmp_path):
    """Verify Case A writes correct reports inside timestamped reproduction directory."""
    config_path, ckpt_path = temp_environment
    monkeypatch.chdir(tmp_path)
    
    args = ["--config", str(config_path), "--stages", "1,2,3,4,5,6"]
    exit_code = main(args)
    assert exit_code == 0
    
    repro_dir = list(Path("results").glob("reproduction_*"))[0]
    
    summary_path = repro_dir / "empirical_stress_test" / "empirical_summary.json"
    report_path = repro_dir / "reports" / "empirical_stress_test.json"
    
    assert summary_path.exists()
    assert report_path.exists()
    
    for p in [summary_path, report_path]:
        with open(p, "r", encoding="utf-8") as f:
            data = json.load(f)
        assert data["stage"] == "stage_6_empirical_stress_test"
        assert data["status"] == "NEEDS_REAL_DATA"
        assert data["verdict"] == "NEEDS_REAL_DATA"
        assert data["reason"] == "paths.market_data_csv is null or unavailable"
        assert data["data_policy"] == "local_csv_only_no_api_fetch"
        assert data["profitability_claims_made"] is False
        assert data["fake_data_used"] is False


def test_stage6_manifest_records_outputs(temp_environment, monkeypatch, tmp_path):
    """Verify Stage 6 outputs are logged in the manifest with path and sha256."""
    config_path, ckpt_path = temp_environment
    monkeypatch.chdir(tmp_path)
    
    args = ["--config", str(config_path), "--stages", "1,2,3,4,5,6"]
    exit_code = main(args)
    assert exit_code == 0
    
    repro_dir = list(Path("results").glob("reproduction_*"))[0]
    
    manifest_path = repro_dir / "manifest.json"
    with open(manifest_path, "r", encoding="utf-8") as f:
        manifest = json.load(f)
        
    outputs = manifest["stages"]["stage_6_empirical_stress_test"]["outputs"]
    assert "empirical_summary" in outputs
    assert "empirical_stress_test_report" in outputs
    
    summary_out = outputs["empirical_summary"]
    assert Path(summary_out["path"]).as_posix() == "empirical_stress_test/empirical_summary.json"
    assert len(summary_out["sha256"]) == 64


def test_stage6_does_not_create_fake_market_data(temp_environment, monkeypatch, tmp_path):
    """Verify Stage 6 does not create or write any dummy/fake CSV files to disk when market_data_csv is null."""
    config_path, ckpt_path = temp_environment
    monkeypatch.chdir(tmp_path)
    
    args = ["--config", str(config_path), "--stages", "1,2,3,4,5,6"]
    exit_code = main(args)
    assert exit_code == 0
    
    # Check that no CSV files other than validation_rows and path_lengths and invariants were generated
    generated_csvs = list(Path(".").glob("**/*.csv"))
    for csv_file in generated_csvs:
        # These are valid geometry/validation outputs, not market data
        assert csv_file.name in ("validation_rows.csv", "metric_invariants.csv", "path_lengths.csv")


def test_stage6_does_not_call_binance_api_by_default(temp_environment, monkeypatch, tmp_path):
    """Verify that Stage 6 does not call Binance/API or initiate network connection."""
    config_path, ckpt_path = temp_environment
    monkeypatch.chdir(tmp_path)
    
    # Block socket or mock requests to raise if called
    import urllib.request
    import urllib.parse
    
    def raise_network_error(*args, **kwargs):
        pytest.fail("Network API call was initiated during Stage 6 execution!")
        
    monkeypatch.setattr("urllib.request.urlopen", raise_network_error)
    # Also block pandas read_csv from reading URL if requested
    monkeypatch.setattr("pandas.read_csv", raise_network_error)
    
    # Since market_data_csv is null, it should not call requests or read_csv
    args = ["--config", str(config_path), "--stages", "1,2,3,4,5,6"]
    exit_code = main(args)
    assert exit_code == 0


def test_stage6_with_local_csv_profiles_market_data(temp_environment, monkeypatch, tmp_path):
    """Verify Case B: when local CSV is provided, Stage 6 profiles close-to-close returns and basic diagnostics."""
    config_path, ckpt_path = temp_environment
    monkeypatch.chdir(tmp_path)
    
    # Create dummy local market data CSV with 150 rows
    market_data_path = tmp_path / "market_data.csv"
    with open(market_data_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["timestamp", "open", "high", "low", "close", "volume"])
        # Write 120 rows
        close_val = 100.0
        for i in range(120):
            minute = i // 60
            second = i % 60
            timestamp = f"2026-06-20T10:{minute:02d}:{second:02d}"
            # Alternate close slightly
            close_val = close_val * 1.01 if i % 2 == 0 else close_val * 0.99
            writer.writerow([timestamp, 100.0, 105.0, 95.0, close_val, 1000.0])
            
    # Update config with market_data_csv path
    config_content = config_path.read_text(encoding="utf-8")
    config_content = config_content.replace("market_data_csv: null", f"market_data_csv: \"{str(market_data_path).replace(os.sep, '/')}\"")
    config_path.write_text(config_content, encoding="utf-8")
    
    args = ["--config", str(config_path), "--stages", "1,2,3,4,5,6"]
    exit_code = main(args)
    assert exit_code == 0
    
    repro_dir = list(Path("results").glob("reproduction_*"))[0]
    
    # Verify profile file
    profile_path = repro_dir / "empirical_stress_test" / "market_data_profile.json"
    assert profile_path.exists()
    
    with open(profile_path, "r", encoding="utf-8") as f:
        profile = json.load(f)
        
    assert profile["n_rows"] == 120
    assert "date_start" in profile
    assert "date_end" in profile
    assert profile["close_return_mean"] is not None
    assert profile["close_return_std"] is not None
    assert profile["naive_directional_baseline_accuracy"] is not None
    
    # Verify return diagnostics csv is written
    diagnostics_path = repro_dir / "empirical_stress_test" / "return_diagnostics.csv"
    assert diagnostics_path.exists()
    
    df = pd.read_csv(diagnostics_path)
    assert len(df) == 120
    assert "returns" in df.columns


def test_stage6_rejects_missing_close_column(temp_environment, monkeypatch, tmp_path):
    """Verify Stage 6 fails if market data CSV is missing the close column."""
    config_path, ckpt_path = temp_environment
    monkeypatch.chdir(tmp_path)
    
    market_data_path = tmp_path / "market_data_bad.csv"
    with open(market_data_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["timestamp", "open", "high", "low", "volume"])
        for i in range(120):
            writer.writerow([f"2026-06-20T10:00:{i:02d}", 100.0, 105.0, 95.0, 1000.0])
            
    config_content = config_path.read_text(encoding="utf-8")
    config_content = config_content.replace("market_data_csv: null", f"market_data_csv: \"{str(market_data_path).replace(os.sep, '/')}\"")
    config_path.write_text(config_content, encoding="utf-8")
    
    args = ["--config", str(config_path), "--stages", "1,2,3,4,5,6"]
    exit_code = main(args)
    assert exit_code == 1  # Should fail Stage 6


def test_stage6_rejects_too_few_rows(temp_environment, monkeypatch, tmp_path):
    """Verify Stage 6 fails if market data CSV has fewer than 100 rows."""
    config_path, ckpt_path = temp_environment
    monkeypatch.chdir(tmp_path)
    
    market_data_path = tmp_path / "market_data_short.csv"
    with open(market_data_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["timestamp", "close"])
        for i in range(50):
            writer.writerow([f"2026-06-20T10:00:{i:02d}", 100.0])
            
    config_content = config_path.read_text(encoding="utf-8")
    config_content = config_content.replace("market_data_csv: null", f"market_data_csv: \"{str(market_data_path).replace(os.sep, '/')}\"")
    config_path.write_text(config_content, encoding="utf-8")
    
    args = ["--config", str(config_path), "--stages", "1,2,3,4,5,6"]
    exit_code = main(args)
    assert exit_code == 1  # Should fail Stage 6 due to insufficient rows


def test_stage6_does_not_write_to_legacy_paths(temp_environment, monkeypatch, tmp_path):
    """Verify Stage 6 writes outputs ONLY inside timestamped reproduction folder, never to legacy paths."""
    config_path, ckpt_path = temp_environment
    monkeypatch.chdir(tmp_path)
    
    args = ["--config", str(config_path), "--stages", "1,2,3,4,5,6"]
    exit_code = main(args)
    assert exit_code == 0
    
    # Assert that no file was written to data/generated/ or root other than expected temporary files
    assert not Path("data/generated/").exists()
    assert not Path("results/generation_2/robustness/empirical_summary.json").exists()


def test_stage6_future_stage7_remains_skipped(temp_environment, monkeypatch, tmp_path):
    """Verify Stage 7 remains SKIPPED in manifest with reason 'Not implemented in Phase 1N'."""
    config_path, ckpt_path = temp_environment
    monkeypatch.chdir(tmp_path)
    
    args = ["--config", str(config_path), "--stages", "1,2,3,4,5,6"]
    exit_code = main(args)
    assert exit_code == 0
    
    repro_dir = list(Path("results").glob("reproduction_*"))[0]
    
    manifest_path = repro_dir / "manifest.json"
    with open(manifest_path, "r", encoding="utf-8") as f:
        manifest = json.load(f)
        
    # Stage 7 is now implemented, so if not requested, it should be "Not requested"
    stage_7 = manifest["stages"]["stage_7_claim_verdict"]
    assert stage_7["status"] == "SKIPPED"
    assert stage_7["reason"] == "Not requested"



def test_stage6_does_not_claim_profitability(temp_environment, monkeypatch, tmp_path):
    """Verify that Stage 6 outputs do not contain keys relating to profitability."""
    config_path, ckpt_path = temp_environment
    monkeypatch.chdir(tmp_path)
    
    args = ["--config", str(config_path), "--stages", "1,2,3,4,5,6"]
    exit_code = main(args)
    assert exit_code == 0
    
    repro_dir = list(Path("results").glob("reproduction_*"))[0]
    
    # Verify outputs
    summary_path = repro_dir / "empirical_stress_test" / "empirical_summary.json"
    report_path = repro_dir / "reports" / "empirical_stress_test.json"
    
    for p in [summary_path, report_path]:
        with open(p, "r", encoding="utf-8") as f:
            content = f.read().lower()
        # Verify no profitability indicators in content keys/values
        for word in ["profit", "pnl", "roi", "sharpe", "alpha", "strategy_profit"]:
            assert f'"{word}"' not in content


def test_stage6_local_csv_with_invalid_reconstructions_marks_out_of_scope(temp_environment, monkeypatch, tmp_path):
    """Verify that if local CSV exists, since VAE reconstructions are 0% valid, the verdict is OUT_OF_SCOPE."""
    config_path, ckpt_path = temp_environment
    monkeypatch.chdir(tmp_path)
    
    # Create dummy local market data CSV with 150 rows
    market_data_path = tmp_path / "market_data.csv"
    with open(market_data_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["timestamp", "close"])
        for i in range(120):
            minute = i // 60
            second = i % 60
            writer.writerow([f"2026-06-20T10:{minute:02d}:{second:02d}", 100.0])
            
    config_content = config_path.read_text(encoding="utf-8")
    config_content = config_content.replace("market_data_csv: null", f"market_data_csv: \"{str(market_data_path).replace(os.sep, '/')}\"")
    config_path.write_text(config_content, encoding="utf-8")
    
    args = ["--config", str(config_path), "--stages", "1,2,3,4,5,6"]
    exit_code = main(args)
    assert exit_code == 0
    
    repro_dir = list(Path("results").glob("reproduction_*"))[0]
    
    summary_path = repro_dir / "empirical_stress_test" / "empirical_summary.json"
    with open(summary_path, "r", encoding="utf-8") as f:
        summary = json.load(f)
        
    assert summary["status"] == "PROFILED_MARKET_DATA_ONLY"
    assert summary["verdict"] == "OUT_OF_SCOPE"
    assert summary["reason"] == "MODEL_SIGNALS_UNAVAILABLE_DUE_TO_RECONSTRUCTION_INVALIDITY"


def test_stage6_hashes_claim_matrix_project_input(temp_environment, monkeypatch, tmp_path):
    """Verify Stage 6 hashes project CLAIM_VERDICT_MATRIX.md and snapshots it in reports."""
    config_path, ckpt_path = temp_environment
    monkeypatch.chdir(tmp_path)
    
    # Create dummy market data CSV to run Case B
    market_data_path = tmp_path / "market_data.csv"
    with open(market_data_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["timestamp", "close"])
        for i in range(120):
            minute = i // 60
            second = i % 60
            writer.writerow([f"2026-06-20T10:{minute:02d}:{second:02d}", 100.0])
            
    config_content = config_path.read_text(encoding="utf-8")
    config_content = config_content.replace("market_data_csv: null", f"market_data_csv: \"{str(market_data_path).replace(os.sep, '/')}\"")
    config_path.write_text(config_content, encoding="utf-8")
    
    args = ["--config", str(config_path), "--stages", "1,2,3,4,5,6"]
    exit_code = main(args)
    assert exit_code == 0
    
    repro_dir = list(Path("results").glob("reproduction_*"))[0]
    
    # Check claim_verdict_matrix_snapshot.md exists in reports
    snapshot_path = repro_dir / "reports" / "claim_verdict_matrix_snapshot.md"
    assert snapshot_path.exists()
    
    # Verify hash is logged in manifest
    manifest_path = repro_dir / "manifest.json"
    with open(manifest_path, "r", encoding="utf-8") as f:
        manifest = json.load(f)
        
    stage_6_manifest = manifest["stages"]["stage_6_empirical_stress_test"]
    assert "claim_verdict_matrix" in stage_6_manifest["inputs"]
    
    logged_hash = stage_6_manifest["inputs"]["claim_verdict_matrix"]["sha256"]
    
    # Calculate expected hash of CLAIM_VERDICT_MATRIX.md
    expected_hash = compute_sha256(tmp_path / "docs" / "CLAIM_VERDICT_MATRIX.md")
    
    assert logged_hash == expected_hash
    assert stage_6_manifest["outputs"]["claim_verdict_matrix_snapshot"]["sha256"] == expected_hash
