"""
Tests for Stage 5 (Corrected Geometry Baseline) of the reproduction runner.
"""
import csv
import json
import os
from pathlib import Path

import numpy as np
import pytest
import torch

from src.reproduce import main, run_stage_5


@pytest.fixture
def temp_environment(tmp_path):
    """Setup a valid temp config, a tiny valid checkpoint, and Stage 1 outputs for testing."""
    config_dir = tmp_path / "configs"
    config_dir.mkdir()
    config_path = config_dir / "test_stage5.yaml"
    
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

geometry:
  max_geometry_points: 5
  finite_difference_eps: 1.0e-4
  metric_regularization_eps: 1.0e-8
"""
    config_path.write_text(config_content, encoding="utf-8")
    
    return config_path, ckpt_path


def test_stage5_requires_stage1_to_stage4(temp_environment, monkeypatch, tmp_path):
    """Verify Stage 5 fails fast if Stages 1-4 are not requested in the same run."""
    config_path, ckpt_path = temp_environment
    monkeypatch.chdir(tmp_path)
    
    args = ["--config", str(config_path), "--stages", "5"]
    exit_code = main(args)
    assert exit_code == 1
    
    args = ["--config", str(config_path), "--stages", "1,2,3,5"]
    exit_code = main(args)
    assert exit_code == 1


def test_stage5_requires_validation_rows(temp_environment, monkeypatch, tmp_path):
    """Verify Stage 5 fails if validation_rows.csv is missing."""
    config_path, ckpt_path = temp_environment
    monkeypatch.chdir(tmp_path)
    
    # Run Stages 1-4
    args = ["--config", str(config_path), "--stages", "1,2,3,4"]
    exit_code = main(args)
    assert exit_code == 0
    
    repro_dirs = list(Path("results").glob("reproduction_*"))
    assert len(repro_dirs) == 1
    repro_dir = repro_dirs[0]
    
    csv_path = repro_dir / "validation" / "validation_rows.csv"
    assert csv_path.exists()
    
    # Delete the csv file
    csv_path.unlink()
    
    manifest = {"stages": {}}
    res = run_stage_5(config_path, repro_dir, manifest)
    assert res is False
    assert manifest["stages"]["stage_5_corrected_geometry"]["status"] == "FAILED"


def test_stage5_selects_only_valid_input_rows(temp_environment, monkeypatch, tmp_path):
    """Verify Stage 5 selects only valid input rows from Stage 1."""
    config_path, ckpt_path = temp_environment
    monkeypatch.chdir(tmp_path)
    
    # Run Stages 1-5
    args = ["--config", str(config_path), "--stages", "1,2,3,4,5"]
    exit_code = main(args)
    assert exit_code == 0
    
    repro_dir = list(Path("results").glob("reproduction_*"))[0]
    
    # Read selection indices
    indices = np.load(repro_dir / "geometry" / "selected_indices.npy")
    
    # Verify they match rows in validation_rows.csv with source="input" and is_valid="TRUE"
    csv_path = repro_dir / "validation" / "validation_rows.csv"
    valid_input_indices = []
    with open(csv_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for r in reader:
            if r["source"] == "input" and r["is_valid"].upper() == "TRUE":
                valid_input_indices.append(int(r["row_index"]))
                
    # Our selection policy picks first N valid inputs
    expected_selected = np.array(valid_input_indices[:5])
    np.testing.assert_array_equal(indices, expected_selected)


def test_stage5_fails_if_no_valid_input_rows(temp_environment, monkeypatch, tmp_path):
    """Verify Stage 5 fails if there are no valid input vectors."""
    config_path, ckpt_path = temp_environment
    monkeypatch.chdir(tmp_path)
    
    args = ["--config", str(config_path), "--stages", "1,2,3,4"]
    exit_code = main(args)
    assert exit_code == 0
    
    repro_dir = list(Path("results").glob("reproduction_*"))[0]
    csv_path = repro_dir / "validation" / "validation_rows.csv"
    
    # Rewrite validation_rows.csv to contain no valid rows
    with open(csv_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        fieldnames = reader.fieldnames
        rows = list(reader)
        
    for r in rows:
        r["is_valid"] = "FALSE"
        
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
        
    # Run Stage 5 directly
    manifest = {"stages": {}}
    res = run_stage_5(config_path, repro_dir, manifest)
    assert res is False
    assert "No valid Stage 1" in manifest["stages"]["stage_5_corrected_geometry"]["error_message"]


def test_stage5_writes_only_inside_reproduction_dir(temp_environment, monkeypatch, tmp_path):
    """Verify Stage 5 does not write outside the isolated folder."""
    config_path, ckpt_path = temp_environment
    monkeypatch.chdir(tmp_path)
    
    legacy_dir = tmp_path / "data" / "generated"
    legacy_dir.mkdir(parents=True)
    legacy_file = legacy_dir / "old.npy"
    legacy_file.write_text("untouched")
    
    args = ["--config", str(config_path), "--stages", "1,2,3,4,5"]
    exit_code = main(args)
    assert exit_code == 0
    
    assert legacy_file.read_text() == "untouched"
    assert len(list(legacy_dir.iterdir())) == 1


def test_stage5_writes_geometry_summary(temp_environment, monkeypatch, tmp_path):
    """Verify Stage 5 generates geometry_summary.json with correct keys."""
    config_path, ckpt_path = temp_environment
    monkeypatch.chdir(tmp_path)
    
    args = ["--config", str(config_path), "--stages", "1,2,3,4,5"]
    exit_code = main(args)
    assert exit_code == 0
    
    repro_dir = list(Path("results").glob("reproduction_*"))[0]
    summary_path = repro_dir / "geometry" / "geometry_summary.json"
    assert summary_path.exists()
    
    with open(summary_path, "r", encoding="utf-8") as f:
        summary = json.load(f)
        
    assert "n_valid_input_rows" in summary
    assert "n_selected" in summary
    assert "invariants_summary" in summary
    assert "path_lengths_summary" in summary


def test_stage5_writes_metric_invariants_csv(temp_environment, monkeypatch, tmp_path):
    """Verify Stage 5 generates metric_invariants.csv with proper fields."""
    config_path, ckpt_path = temp_environment
    monkeypatch.chdir(tmp_path)
    
    args = ["--config", str(config_path), "--stages", "1,2,3,4,5"]
    exit_code = main(args)
    assert exit_code == 0
    
    repro_dir = list(Path("results").glob("reproduction_*"))[0]
    csv_path = repro_dir / "geometry" / "metric_invariants.csv"
    assert csv_path.exists()
    
    with open(csv_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        fieldnames = reader.fieldnames
        rows = list(reader)
        
    for col in ["selected_index", "original_row_index", "max_eigenvalue", "min_eigenvalue", "condition_number_before", "condition_number_after", "trace", "logdet", "is_symmetric", "is_psd", "finite_values_ok"]:
        assert col in fieldnames
    assert len(rows) == 5


def test_stage5_writes_selected_indices(temp_environment, monkeypatch, tmp_path):
    """Verify Stage 5 writes selected_indices.npy."""
    config_path, ckpt_path = temp_environment
    monkeypatch.chdir(tmp_path)
    
    args = ["--config", str(config_path), "--stages", "1,2,3,4,5"]
    exit_code = main(args)
    assert exit_code == 0
    
    repro_dir = list(Path("results").glob("reproduction_*"))[0]
    indices_path = repro_dir / "geometry" / "selected_indices.npy"
    assert indices_path.exists()
    indices = np.load(indices_path)
    assert indices.shape == (5,)


def test_stage5_manifest_records_hashes(temp_environment, monkeypatch, tmp_path):
    """Verify Stage 5 outputs record correct paths and hashes in manifest."""
    config_path, ckpt_path = temp_environment
    monkeypatch.chdir(tmp_path)
    
    args = ["--config", str(config_path), "--stages", "1,2,3,4,5"]
    exit_code = main(args)
    assert exit_code == 0
    
    repro_dir = list(Path("results").glob("reproduction_*"))[0]
    manifest_path = repro_dir / "manifest.json"
    
    with open(manifest_path, "r", encoding="utf-8") as f:
        manifest = json.load(f)
        
    stage_5 = manifest["stages"]["stage_5_corrected_geometry"]
    assert stage_5["status"] == "PASSED"
    
    outputs = stage_5["outputs"]
    for key in ["geometry_summary", "metric_invariants", "selected_indices", "pullback_metrics", "corrected_geometry_report"]:
        assert key in outputs
        assert "path" in outputs[key]
        assert "sha256" in outputs[key]
        assert (repro_dir / outputs[key]["path"]).exists()


def test_stage5_does_not_use_legacy_fisher_artifacts(temp_environment, monkeypatch, tmp_path):
    """Verify Stage 5 does not search or read legacy outputs."""
    config_path, ckpt_path = temp_environment
    monkeypatch.chdir(tmp_path)
    
    # Create decoy legacy metric
    legacy_dir = tmp_path / "results" / "generation_1" / "fisher_metrics"
    legacy_dir.mkdir(parents=True)
    legacy_metric = legacy_dir / "fisher_metric_0.npy"
    legacy_metric.write_text("should be ignored")
    
    args = ["--config", str(config_path), "--stages", "1,2,3,4,5"]
    exit_code = main(args)
    assert exit_code == 0
    
    assert legacy_metric.read_text() == "should be ignored"


def test_stage5_does_not_write_legacy_fisher_names(temp_environment, monkeypatch, tmp_path):
    """Verify Stage 5 outputs do not use legacy Fisher names."""
    config_path, ckpt_path = temp_environment
    monkeypatch.chdir(tmp_path)
    
    args = ["--config", str(config_path), "--stages", "1,2,3,4,5"]
    exit_code = main(args)
    assert exit_code == 0
    
    repro_dir = list(Path("results").glob("reproduction_*"))[0]
    assert len(list(repro_dir.glob("**/fisher_metric*"))) == 0
    assert len(list(repro_dir.glob("**/regularized_fisher*"))) == 0


def test_stage5_does_not_create_generated_valid_thetas(temp_environment, monkeypatch, tmp_path):
    """Verify Stage 5 does not create generated_valid_thetas.npy."""
    config_path, ckpt_path = temp_environment
    monkeypatch.chdir(tmp_path)
    
    args = ["--config", str(config_path), "--stages", "1,2,3,4,5"]
    exit_code = main(args)
    assert exit_code == 0
    
    repro_dir = list(Path("results").glob("reproduction_*"))[0]
    assert not (repro_dir / "geometry" / "generated_valid_thetas.npy").exists()
    assert len(list(repro_dir.glob("**/generated_valid_thetas*"))) == 0


def test_stage5_does_not_write_to_legacy_data_generated(temp_environment, monkeypatch, tmp_path):
    """Verify Stage 5 does not write to data/generated/."""
    config_path, ckpt_path = temp_environment
    monkeypatch.chdir(tmp_path)
    
    args = ["--config", str(config_path), "--stages", "1,2,3,4,5"]
    exit_code = main(args)
    assert exit_code == 0
    
    assert not Path("data/generated").exists()


def test_stage5_future_stages_remain_skipped(temp_environment, monkeypatch, tmp_path):
    """Verify Stage 6 and 7 remain SKIPPED in manifest with correct reason."""
    config_path, ckpt_path = temp_environment
    monkeypatch.chdir(tmp_path)
    
    args = ["--config", str(config_path), "--stages", "1,2,3,4,5"]
    exit_code = main(args)
    assert exit_code == 0
    
    repro_dir = list(Path("results").glob("reproduction_*"))[0]
    manifest_path = repro_dir / "manifest.json"
    
    with open(manifest_path, "r", encoding="utf-8") as f:
        manifest = json.load(f)
        
    # Stage 6 is now implemented, so if not requested, it should be "Not requested"
    assert manifest["stages"]["stage_6_empirical_stress_test"]["status"] == "SKIPPED"
    assert manifest["stages"]["stage_6_empirical_stress_test"]["reason"] == "Not requested"

    # Stage 7 is now implemented, so if not requested, it should be "Not requested"
    assert manifest["stages"]["stage_7_claim_verdict"]["status"] == "SKIPPED"
    assert manifest["stages"]["stage_7_claim_verdict"]["reason"] == "Not requested"




def test_stage5_report_labels_scope_as_valid_input_only(temp_environment, monkeypatch, tmp_path):
    """Verify reports/corrected_geometry.json contains scope and note mappings."""
    config_path, ckpt_path = temp_environment
    monkeypatch.chdir(tmp_path)
    
    args = ["--config", str(config_path), "--stages", "1,2,3,4,5"]
    exit_code = main(args)
    assert exit_code == 0
    
    repro_dir = list(Path("results").glob("reproduction_*"))[0]
    report_path = repro_dir / "reports" / "corrected_geometry.json"
    assert report_path.exists()
    
    with open(report_path, "r", encoding="utf-8") as f:
        report = json.load(f)
        
    assert report["stage"] == "stage_5_corrected_geometry"
    assert report["geometry_scope"] == "valid_stage1_input_latents_only"
    assert report["legacy_fisher_status"] == "INVALID_DO_NOT_CLAIM"
    assert "notes" in report
    
    notes = report["notes"]
    assert any("valid Stage 1 input vectors" in n for n in notes)
    assert any("LEGACY_LAYOUT_COLLISION" in n for n in notes)


def test_stage5_path_lengths_are_bounded(temp_environment, monkeypatch, tmp_path):
    """Verify Riemannian and Euclidean path lengths are computed and saved in path_lengths.csv."""
    config_path, ckpt_path = temp_environment
    monkeypatch.chdir(tmp_path)
    
    args = ["--config", str(config_path), "--stages", "1,2,3,4,5"]
    exit_code = main(args)
    assert exit_code == 0
    
    repro_dir = list(Path("results").glob("reproduction_*"))[0]
    path_lengths_path = repro_dir / "geometry" / "path_lengths.csv"
    assert path_lengths_path.exists()
    
    with open(path_lengths_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        fieldnames = reader.fieldnames
        rows = list(reader)
        
    assert "start_index" in fieldnames
    assert "end_index" in fieldnames
    assert "euclidean_length" in fieldnames
    assert "riemannian_length" in fieldnames
    
    # 5 selected points -> 4 segments
    assert len(rows) == 4
    
    # Check that lengths are positive floats
    for r in rows:
        assert float(r["euclidean_length"]) > 0.0
        assert float(r["riemannian_length"]) > 0.0


def test_stage5_metric_invariants_include_required_columns(temp_environment, monkeypatch, tmp_path):
    """Verify that metric_invariants.csv contains all the exact required invariant columns."""
    config_path, ckpt_path = temp_environment
    monkeypatch.chdir(tmp_path)
    
    args = ["--config", str(config_path), "--stages", "1,2,3,4,5"]
    exit_code = main(args)
    assert exit_code == 0
    
    repro_dir = list(Path("results").glob("reproduction_*"))[0]
    csv_path = repro_dir / "geometry" / "metric_invariants.csv"
    assert csv_path.exists()
    
    with open(csv_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        fieldnames = reader.fieldnames
        rows = list(reader)
        
    required_columns = [
        "jacobian_shape",
        "metric_shape",
        "metric_trace",
        "metric_logdet_or_null",
        "metric_condition_number",
        "min_eigenvalue",
        "max_eigenvalue",
        "is_symmetric",
        "is_positive_semidefinite_or_regularized",
        "finite_values_ok"
    ]
    
    for col in required_columns:
        assert col in fieldnames
        
    # Assert values in rows are populated correctly
    first_row = rows[0]
    assert first_row["jacobian_shape"] == "40x8"
    assert first_row["metric_shape"] == "8x8"
    assert float(first_row["metric_trace"]) > 0.0
    assert first_row["is_symmetric"] == "TRUE"
    assert first_row["is_positive_semidefinite_or_regularized"] in ("TRUE", "FALSE")
    assert first_row["finite_values_ok"] == "TRUE"
    
    # Verify metric_logdet_or_null is float or empty
    logdet_val = first_row["metric_logdet_or_null"]
    if logdet_val != "":
        assert float(logdet_val) is not None


def test_stage5_geometry_eps_accepts_yaml_string_scientific_notation(temp_environment, monkeypatch, tmp_path):
    """Verify that Stage 5 handles and coerces geometry eps scientific notation string format safely."""
    config_path, ckpt_path = temp_environment
    monkeypatch.chdir(tmp_path)
    
    # Overwrite config file to use string scientific notation for geometry epsilons
    config_text = config_path.read_text(encoding="utf-8")
    config_text = config_text.replace("finite_difference_eps: 1.0e-4", 'finite_difference_eps: "1e-4"')
    config_text = config_text.replace("metric_regularization_eps: 1.0e-8", 'metric_regularization_eps: "1e-8"')
    config_path.write_text(config_text, encoding="utf-8")
    
    args = ["--config", str(config_path), "--stages", "1,2,3,4,5"]
    exit_code = main(args)
    assert exit_code == 0
    
    repro_dir = list(Path("results").glob("reproduction_*"))[0]
    summary_path = repro_dir / "geometry" / "geometry_summary.json"
    assert summary_path.exists()
    
    with open(summary_path, "r", encoding="utf-8") as f:
        summary = json.load(f)
        
    # Ensure they were successfully coerced to floats in the summary
    assert isinstance(summary["finite_difference_eps"], float)
    assert isinstance(summary["metric_regularization_eps"], float)
    assert summary["finite_difference_eps"] == 1e-4
    assert summary["metric_regularization_eps"] == 1e-8
