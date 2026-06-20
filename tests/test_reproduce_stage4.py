"""
Tests for Stage 4 (Structural Validation) of the reproduction runner.
"""
import csv
import json
import os
from pathlib import Path

import numpy as np
import pytest
import torch

from src.reproduce import main, run_stage_4


@pytest.fixture
def temp_environment(tmp_path):
    """Setup a valid temp config, a tiny valid checkpoint, and Stage 1 outputs for testing."""
    config_dir = tmp_path / "configs"
    config_dir.mkdir()
    config_path = config_dir / "test_stage4.yaml"
    
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


def test_stage4_requires_stage1_stage2_stage3(temp_environment, monkeypatch, tmp_path):
    """Verify Stage 4 requires Stages 1, 2, and 3 in the same run."""
    config_path, ckpt_path = temp_environment
    monkeypatch.chdir(tmp_path)
    
    # Run only Stage 4 (without 1,2,3)
    args = ["--config", str(config_path), "--stages", "4"]
    exit_code = main(args)
    assert exit_code == 1
    
    # Run 1,4
    args = ["--config", str(config_path), "--stages", "1,4"]
    exit_code = main(args)
    assert exit_code == 1
    
    # Run 1,2,4
    args = ["--config", str(config_path), "--stages", "1,2,4"]
    exit_code = main(args)
    assert exit_code == 1


def test_stage4_requires_latent_outputs(temp_environment, monkeypatch, tmp_path):
    """Verify Stage 4 fails if required Stage 3 latent outputs are missing."""
    config_path, ckpt_path = temp_environment
    monkeypatch.chdir(tmp_path)
    
    # First, run 1, 2, 3 to initialize repro_dir and generate some outputs
    args = ["--config", str(config_path), "--stages", "1,2,3"]
    exit_code = main(args)
    assert exit_code == 0
    
    repro_dirs = list(Path("results").glob("reproduction_*"))
    assert len(repro_dirs) == 1
    repro_dir = repro_dirs[0]
    
    z_train_latent_path = repro_dir / "latent" / "z_train.npy"
    assert z_train_latent_path.exists()
    
    # Delete the latent file
    z_train_latent_path.unlink()
    
    # Call run_stage_4 directly and ensure it returns False
    manifest = {"stages": {}}
    res = run_stage_4(config_path, repro_dir, manifest)
    assert res is False
    assert manifest["stages"]["stage_4_structural_validation"]["status"] == "FAILED"


def test_stage4_writes_only_inside_reproduction_dir(temp_environment, monkeypatch, tmp_path):
    """Verify Stage 4 does not write outside the isolated reproduction folder."""
    config_path, ckpt_path = temp_environment
    monkeypatch.chdir(tmp_path)
    
    # Create decoy directories
    legacy_dir = tmp_path / "data" / "generated"
    legacy_dir.mkdir(parents=True)
    legacy_file = legacy_dir / "old.npy"
    legacy_file.write_text("untouched")
    
    args = ["--config", str(config_path), "--stages", "1,2,3,4"]
    exit_code = main(args)
    assert exit_code == 0
    
    # Verify decoy remains untouched
    assert legacy_file.read_text() == "untouched"
    assert len(list(legacy_dir.iterdir())) == 1


def test_stage4_writes_reconstructed_theta(temp_environment, monkeypatch, tmp_path):
    """Verify Stage 4 writes reconstructed_theta.npy with correct shape."""
    config_path, ckpt_path = temp_environment
    monkeypatch.chdir(tmp_path)
    
    args = ["--config", str(config_path), "--stages", "1,2,3,4"]
    exit_code = main(args)
    assert exit_code == 0
    
    repro_dirs = list(Path("results").glob("reproduction_*"))
    assert len(repro_dirs) == 1
    repro_dir = repro_dirs[0]
    
    recon_theta_path = repro_dir / "validation" / "reconstructed_theta.npy"
    assert recon_theta_path.exists()
    
    recon_theta = np.load(recon_theta_path)
    assert recon_theta.shape == (20, 40)  # n_arma=10, n_garch=10 -> 20 samples, dim 40


def test_stage4_does_not_create_generated_valid_thetas(temp_environment, monkeypatch, tmp_path):
    """Verify Stage 4 does not generate generated_valid_thetas.npy."""
    config_path, ckpt_path = temp_environment
    monkeypatch.chdir(tmp_path)
    
    args = ["--config", str(config_path), "--stages", "1,2,3,4"]
    exit_code = main(args)
    assert exit_code == 0
    
    repro_dirs = list(Path("results").glob("reproduction_*"))
    assert len(repro_dirs) == 1
    repro_dir = repro_dirs[0]
    
    assert not (repro_dir / "validation" / "generated_valid_thetas.npy").exists()
    assert not (repro_dir / "latent" / "generated_valid_thetas.npy").exists()
    assert len(list(repro_dir.glob("**/generated_valid_thetas*"))) == 0


def test_stage4_writes_validation_rows_csv(temp_environment, monkeypatch, tmp_path):
    """Verify Stage 4 writes validation_rows.csv with correct columns."""
    config_path, ckpt_path = temp_environment
    monkeypatch.chdir(tmp_path)
    
    args = ["--config", str(config_path), "--stages", "1,2,3,4"]
    exit_code = main(args)
    assert exit_code == 0
    
    repro_dirs = list(Path("results").glob("reproduction_*"))
    assert len(repro_dirs) == 1
    repro_dir = repro_dirs[0]
    
    csv_path = repro_dir / "validation" / "validation_rows.csv"
    assert csv_path.exists()
    
    with open(csv_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        fieldnames = reader.fieldnames
        rows = list(reader)
        
    for col in ["source", "row_index", "label", "is_valid", "structure_type", "invalid_reason"]:
        assert col in fieldnames
    
    # 20 samples * 2 sources (input, reconstruction) = 40 rows
    assert len(rows) == 40
    sources = [r["source"] for r in rows]
    assert sources.count("input") == 20
    assert sources.count("reconstruction") == 20


def test_stage4_writes_structural_validation_report(temp_environment, monkeypatch, tmp_path):
    """Verify Stage 4 writes reports/structural_validation.json with correct format."""
    config_path, ckpt_path = temp_environment
    monkeypatch.chdir(tmp_path)
    
    args = ["--config", str(config_path), "--stages", "1,2,3,4"]
    exit_code = main(args)
    assert exit_code == 0
    
    repro_dirs = list(Path("results").glob("reproduction_*"))
    assert len(repro_dirs) == 1
    repro_dir = repro_dirs[0]
    
    report_path = repro_dir / "reports" / "structural_validation.json"
    assert report_path.exists()
    
    with open(report_path, "r", encoding="utf-8") as f:
        report = json.load(f)
        
    assert report["stage"] == "stage_4_structural_validation"
    assert "inputs" in report
    assert "outputs" in report
    assert "input_vectors" in report
    assert "reconstructed_vectors" in report
    assert "notes" in report
    
    # Verify input_vectors and reconstructed_vectors have required subkeys
    for key in ["n_total", "n_valid", "valid_rate", "n_invalid", "invalid_rate", "label_counts", "structure_counts", "invalid_reason_counts"]:
        assert key in report["input_vectors"]
        assert key in report["reconstructed_vectors"]


def test_stage4_manifest_records_hashes(temp_environment, monkeypatch, tmp_path):
    """Verify Stage 4 outputs in manifest record correct path/sha256 structure."""
    config_path, ckpt_path = temp_environment
    monkeypatch.chdir(tmp_path)
    
    args = ["--config", str(config_path), "--stages", "1,2,3,4"]
    exit_code = main(args)
    assert exit_code == 0
    
    repro_dirs = list(Path("results").glob("reproduction_*"))
    assert len(repro_dirs) == 1
    repro_dir = repro_dirs[0]
    
    manifest_path = repro_dir / "manifest.json"
    
    with open(manifest_path, "r", encoding="utf-8") as f:
        manifest = json.load(f)
        
    stage_4 = manifest["stages"]["stage_4_structural_validation"]
    assert stage_4["status"] == "PASSED"
    
    outputs = stage_4["outputs"]
    required_outputs = [
        "input_validation_summary",
        "reconstruction_validation_summary",
        "reconstructed_theta",
        "validation_rows",
        "structural_validation_report"
    ]
    for key in required_outputs:
        assert key in outputs
        assert "path" in outputs[key]
        assert "sha256" in outputs[key]
        assert (repro_dir / outputs[key]["path"]).exists()


def test_stage4_future_stages_remain_skipped(temp_environment, monkeypatch, tmp_path):
    """Verify future stages remain SKIPPED in manifest with proper reasons."""
    config_path, ckpt_path = temp_environment
    monkeypatch.chdir(tmp_path)
    
    args = ["--config", str(config_path), "--stages", "1,2,3,4"]
    exit_code = main(args)
    assert exit_code == 0
    
    repro_dirs = list(Path("results").glob("reproduction_*"))
    assert len(repro_dirs) == 1
    repro_dir = repro_dirs[0]
    
    manifest_path = repro_dir / "manifest.json"
    
    with open(manifest_path, "r", encoding="utf-8") as f:
        manifest = json.load(f)
        
    assert manifest["stages"]["stage_5_corrected_geometry"]["status"] == "SKIPPED"
    assert manifest["stages"]["stage_5_corrected_geometry"]["reason"] == "Not requested"
    
    # Stage 6 is now implemented, so if not requested, it should be "Not requested"
    assert manifest["stages"]["stage_6_empirical_stress_test"]["status"] == "SKIPPED"
    assert manifest["stages"]["stage_6_empirical_stress_test"]["reason"] == "Not requested"

    # Stage 7 is now implemented, so if not requested, it should be "Not requested"
    assert manifest["stages"]["stage_7_claim_verdict"]["status"] == "SKIPPED"
    assert manifest["stages"]["stage_7_claim_verdict"]["reason"] == "Not requested"




def test_stage4_uses_canonical_validation_not_legacy_script():
    """Verify static check: src/reproduce.py does not call legacy scripts like Симуляція_2."""
    reproduce_py_path = Path("src/reproduce.py")
    content = reproduce_py_path.read_text(encoding="utf-8")
    
    # Assert we don't import or run Симуляція_2.py
    assert "Симуляція_2" not in content
    # Assert we import canonical modules
    assert "from src.validation import" in content
    assert "from src.vector_schema import" in content


def test_stage4_does_not_write_to_legacy_data_generated(temp_environment, monkeypatch, tmp_path):
    """Verify legacy data/generated/ folder is untouched."""
    config_path, ckpt_path = temp_environment
    monkeypatch.chdir(tmp_path)
    
    args = ["--config", str(config_path), "--stages", "1,2,3,4"]
    exit_code = main(args)
    assert exit_code == 0
    
    assert not Path("data/generated").exists()


def test_stage4_does_not_create_geometry_outputs(temp_environment, monkeypatch, tmp_path):
    """Verify Stage 4 does not write any geometry reports or results."""
    config_path, ckpt_path = temp_environment
    monkeypatch.chdir(tmp_path)
    
    args = ["--config", str(config_path), "--stages", "1,2,3,4"]
    exit_code = main(args)
    assert exit_code == 0
    
    repro_dirs = list(Path("results").glob("reproduction_*"))
    assert len(repro_dirs) == 1
    repro_dir = repro_dirs[0]
    
    assert len(list(repro_dir.glob("**/*geometry*"))) == 0
    assert not Path("geometry_report.json").exists()


def test_stage4_reconstruction_summary_contains_valid_rate(temp_environment, monkeypatch, tmp_path):
    """Verify reconstruction validation summary contains valid_rate key."""
    config_path, ckpt_path = temp_environment
    monkeypatch.chdir(tmp_path)
    
    args = ["--config", str(config_path), "--stages", "1,2,3,4"]
    exit_code = main(args)
    assert exit_code == 0
    
    repro_dirs = list(Path("results").glob("reproduction_*"))
    assert len(repro_dirs) == 1
    repro_dir = repro_dirs[0]
    
    recon_summary_path = repro_dir / "validation" / "reconstruction_validation_summary.json"
    assert recon_summary_path.exists()
    
    with open(recon_summary_path, "r", encoding="utf-8") as f:
        summary = json.load(f)
        
    assert "valid_rate" in summary
    assert isinstance(summary["valid_rate"], float)


def test_stage4_input_summary_contains_valid_rate(temp_environment, monkeypatch, tmp_path):
    """Verify input validation summary contains valid_rate key."""
    config_path, ckpt_path = temp_environment
    monkeypatch.chdir(tmp_path)
    
    args = ["--config", str(config_path), "--stages", "1,2,3,4"]
    exit_code = main(args)
    assert exit_code == 0
    
    repro_dirs = list(Path("results").glob("reproduction_*"))
    assert len(repro_dirs) == 1
    repro_dir = repro_dirs[0]
    
    input_summary_path = repro_dir / "validation" / "input_validation_summary.json"
    assert input_summary_path.exists()
    
    with open(input_summary_path, "r", encoding="utf-8") as f:
        summary = json.load(f)
        
    assert "valid_rate" in summary
    assert isinstance(summary["valid_rate"], float)


def test_stage4_reconstruction_invalidity_does_not_fail_stage(temp_environment, monkeypatch, tmp_path):
    """Verify that low/zero reconstruction validity rate does not fail the stage run."""
    config_path, ckpt_path = temp_environment
    monkeypatch.chdir(tmp_path)
    
    # We can mock the VAE decode method to return random/nonsense values that fail validation
    import src.vae
    original_decode = src.vae.VAE.decode
    
    # Decode to completely invalid parameters (e.g. all ones, which will violate stationarity/invertibility/GARCH)
    def mock_decode(self, z):
        return torch.ones(z.shape[0], 40)
        
    monkeypatch.setattr(src.vae.VAE, "decode", mock_decode)
    
    args = ["--config", str(config_path), "--stages", "1,2,3,4"]
    exit_code = main(args)
    
    # Stage should still complete with exit code 0
    assert exit_code == 0
    
    repro_dirs = list(Path("results").glob("reproduction_*"))
    assert len(repro_dirs) == 1
    repro_dir = repro_dirs[0]
    
    manifest_path = repro_dir / "manifest.json"
    
    with open(manifest_path, "r", encoding="utf-8") as f:
        manifest = json.load(f)
        
    # Validation status must still be PASSED
    assert manifest["stages"]["stage_4_structural_validation"]["status"] == "PASSED"
    
    # Check that reconstructed_vectors has valid_rate = 0.0 in reports/structural_validation.json
    report_path = repro_dir / "reports" / "structural_validation.json"
    with open(report_path, "r", encoding="utf-8") as f:
        report = json.load(f)
        
    assert report["reconstructed_vectors"]["valid_rate"] == 0.0


def test_stage4_reports_schema_status_and_final_structure_type(temp_environment, monkeypatch, tmp_path):
    """Verify that Stage 4 logs final_structure_type and schema_status columns in validation_rows.csv."""
    config_path, ckpt_path = temp_environment
    monkeypatch.chdir(tmp_path)
    
    args = ["--config", str(config_path), "--stages", "1,2,3,4"]
    exit_code = main(args)
    assert exit_code == 0
    
    repro_dirs = list(Path("results").glob("reproduction_*"))
    assert len(repro_dirs) == 1
    repro_dir = repro_dirs[0]
    
    csv_path = repro_dir / "validation" / "validation_rows.csv"
    assert csv_path.exists()
    
    with open(csv_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        fieldnames = reader.fieldnames
        rows = list(reader)
        
    # Check that the new columns are present
    assert "final_structure_type" in fieldnames
    assert "schema_status" in fieldnames
    
    # Check that schema_status has values like DISJOINT_COMPATIBLE or LEGACY_LAYOUT_COLLISION
    schema_statuses = [r["schema_status"] for r in rows]
    
    # Assert that all rows have valid values for schema_status and correctly mapped reasons
    allowed_statuses = {"DISJOINT_COMPATIBLE", "LEGACY_LAYOUT_COLLISION", "UNKNOWN_STRUCTURE"}
    for r in rows:
        assert r["schema_status"] in allowed_statuses
        if r["schema_status"] == "LEGACY_LAYOUT_COLLISION":
            assert r["final_structure_type"] == "INVALID"
            assert r["invalid_reason"] == "LEGACY_LAYOUT_COLLISION"
        elif r["schema_status"] == "UNKNOWN_STRUCTURE":
            assert r["final_structure_type"] == "INVALID"
            assert r["invalid_reason"] == "UNKNOWN_STRUCTURE"



def test_stage4_masks_inactive_orders_for_classification():
    """Verify that _classify_structure masks inactive orders based on the decoded model_type."""
    from src.reproduce import _classify_structure
    
    # Case 1: model_type is ARMA, meaning GARCH orders r/s must be ignored/masked
    # Raw orders say p=3, q=2, r=2, s=2. But because model_type is ARMA, r/s are masked out to 0.
    struct_type = _classify_structure("ARMA", p=3, q=2, r=2, s=2)
    assert struct_type == "ARMA"  # instead of ARMA_GARCH
    
    struct_type_ar = _classify_structure("ARMA", p=3, q=0, r=2, s=2)
    assert struct_type_ar == "AR"
    
    # Case 2: model_type is GARCH, meaning ARMA orders p/q must be ignored/masked
    struct_type_garch = _classify_structure("GARCH", p=3, q=2, r=2, s=2)
    assert struct_type_garch == "GARCH"  # instead of ARMA_GARCH
    
    # Case 3: model_type is ARMA-GARCH, meaning all are active
    struct_type_joint = _classify_structure("ARMA-GARCH", p=3, q=2, r=2, s=2)
    assert struct_type_joint == "ARMA_GARCH"
    
    struct_type_joint_ar = _classify_structure("ARMA-GARCH", p=3, q=0, r=2, s=2)
    assert struct_type_joint_ar == "AR_GARCH"

