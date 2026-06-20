"""
Tests for Stage 7 (Final Claim Verdict Report) of the reproduction runner.
"""
import csv
import json
import os
import hashlib
import shutil
from pathlib import Path

import numpy as np
import pytest
import torch

from src.reproduce import main, run_stage_7, compute_sha256


@pytest.fixture
def temp_environment(tmp_path):
    """Setup a valid temp config, a tiny VAE checkpoint, Stage 1-6 outputs, and project docs for Stage 7 testing."""
    config_dir = tmp_path / "configs"
    config_dir.mkdir()
    config_path = config_dir / "test_stage7.yaml"
    
    models_dir = tmp_path / "models"
    models_dir.mkdir()
    ckpt_path = models_dir / "test_vae.pth"
    
    # Create docs dir and mock required project-level files
    docs_dir = tmp_path / "docs"
    docs_dir.mkdir()
    
    claim_matrix_path = docs_dir / "CLAIM_VERDICT_MATRIX.md"
    claim_matrix_content = (
        "# Claim Verdict Matrix\n"
        "## Verdict summary\n"
        "| Status | Count |\n"
        "| :--- | ---: |\n"
        "| **CONFIRMED** | 0 |\n"
        "| **PARTIALLY_CONFIRMED** | 1 |\n"
        "| **CHANGED_DUE_TO_BUGFIX** | 1 |\n"
        "| **INVALIDATED** | 4 |\n"
        "| **NEEDS_REAL_DATA** | 1 |\n"
        "| **NOT_REPRODUCED** | 0 |\n"
        "| **OUT_OF_SCOPE** | 0 |\n"
        "Required anchors: CLAIM_01 CLAIM_02 CLAIM_03 CLAIM_04 CLAIM_05 CLAIM_06 CLAIM_07 "
        "LEGACY_LAYOUT_COLLISION INVALIDATED NEEDS_REAL_DATA"
    )
    claim_matrix_path.write_text(claim_matrix_content, encoding="utf-8")
    
    geometry_validity_path = docs_dir / "GEOMETRY_VALIDITY_STATUS.md"
    geometry_validity_path.write_text("Geometry validity status: INVALID_DO_NOT_CLAIM", encoding="utf-8")
    
    research_integrity_path = docs_dir / "RESEARCH_INTEGRITY.md"
    research_integrity_path.write_text("Research integrity: no fake data.", encoding="utf-8")
    
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


def test_stage7_requires_stage1_to_stage6(temp_environment, monkeypatch, tmp_path):
    """Verify Stage 7 fails fast if Stages 1-6 are not executed in the same run."""
    config_path, ckpt_path = temp_environment
    monkeypatch.chdir(tmp_path)
    
    # Run only Stage 7
    args = ["--config", str(config_path), "--stages", "7"]
    exit_code = main(args)
    assert exit_code == 1
    
    # Run Stages 1-5 and 7 (missing Stage 6)
    args = ["--config", str(config_path), "--stages", "1,2,3,4,5,7"]
    exit_code = main(args)
    assert exit_code == 1


def test_stage7_accepts_stage6_skipped_needs_real_data_if_report_exists(temp_environment, monkeypatch, tmp_path):
    """Verify Stage 7 accepts Stage 6 as skipped/NEEDS_REAL_DATA if the reports are correctly generated on disk."""
    config_path, ckpt_path = temp_environment
    monkeypatch.chdir(tmp_path)
    
    # Run Stages 1-7 (with default null market data in config, so Stage 6 is skipped)
    args = ["--config", str(config_path), "--stages", "1,2,3,4,5,6,7"]
    exit_code = main(args)
    assert exit_code == 0
    
    repro_dirs = list(Path("results").glob("reproduction_*"))
    assert len(repro_dirs) == 1
    repro_dir = repro_dirs[0]
    
    manifest_path = repro_dir / "manifest.json"
    with open(manifest_path, "r", encoding="utf-8") as f:
        manifest = json.load(f)
        
    assert manifest["stages"]["stage_6_empirical_stress_test"]["status"] == "SKIPPED"
    assert manifest["stages"]["stage_7_claim_verdict"]["status"] == "PASSED"


def test_stage7_requires_claim_verdict_matrix(temp_environment, monkeypatch, tmp_path):
    """Verify Stage 7 fails if project docs/CLAIM_VERDICT_MATRIX.md is missing."""
    config_path, ckpt_path = temp_environment
    monkeypatch.chdir(tmp_path)
    
    # First, run Stage 1-6 successfully
    args = ["--config", str(config_path), "--stages", "1,2,3,4,5,6"]
    exit_code = main(args)
    assert exit_code == 0
    
    repro_dir = list(Path("results").glob("reproduction_*"))[0]
    
    # Delete docs/CLAIM_VERDICT_MATRIX.md
    claim_matrix_path = tmp_path / "docs" / "CLAIM_VERDICT_MATRIX.md"
    claim_matrix_path.unlink()
    
    # Load manifest and run Stage 7 directly
    with open(repro_dir / "manifest.json", "r", encoding="utf-8") as f:
        manifest = json.load(f)
        
    res = run_stage_7(config_path, repro_dir, manifest)
    assert res is False


def test_stage7_requires_claim_matrix_hash_or_snapshot(temp_environment, monkeypatch, tmp_path):
    """Verify Stage 7 fails if claim matrix was not hashed or snapshotted in Stage 6 manifest entry."""
    config_path, ckpt_path = temp_environment
    monkeypatch.chdir(tmp_path)
    
    args = ["--config", str(config_path), "--stages", "1,2,3,4,5,6"]
    exit_code = main(args)
    assert exit_code == 0
    
    repro_dir = list(Path("results").glob("reproduction_*"))[0]
    manifest_path = repro_dir / "manifest.json"
    with open(manifest_path, "r", encoding="utf-8") as f:
        manifest = json.load(f)
        
    # Corrupt Stage 6 manifest entry by removing hashes
    stage_6 = manifest["stages"]["stage_6_empirical_stress_test"]
    if "inputs" in stage_6:
        stage_6.pop("inputs")
    if "outputs" in stage_6 and "claim_verdict_matrix_snapshot" in stage_6["outputs"]:
        stage_6["outputs"].pop("claim_verdict_matrix_snapshot")
        
    res = run_stage_7(config_path, repro_dir, manifest)
    assert res is False


def test_stage7_writes_final_claim_verdict_json(temp_environment, monkeypatch, tmp_path):
    """Verify that reports/final_claim_verdict.json is written with correct keys and content."""
    config_path, ckpt_path = temp_environment
    monkeypatch.chdir(tmp_path)
    
    args = ["--config", str(config_path), "--stages", "1,2,3,4,5,6,7"]
    exit_code = main(args)
    assert exit_code == 0
    
    repro_dir = list(Path("results").glob("reproduction_*"))[0]
    verdict_json_path = repro_dir / "reports" / "final_claim_verdict.json"
    
    assert verdict_json_path.exists()
    
    with open(verdict_json_path, "r", encoding="utf-8") as f:
        data = json.load(f)
        
    assert data["stage"] == "stage_7_claim_verdict"
    assert data["status"] == "PASSED"
    assert data["final_reproduction_status"] == "COMPLETED_WITH_LIMITATIONS"
    assert "structural_findings" in data
    assert "geometry_findings" in data
    assert "empirical_findings" in data
    assert "invalidated_legacy_claims" in data
    assert "surviving_claims" in data
    assert "safe_public_summary_draft" in data


def test_stage7_writes_final_claim_verdict_md(temp_environment, monkeypatch, tmp_path):
    """Verify reports/final_claim_verdict.md contains the required markdown structure."""
    config_path, ckpt_path = temp_environment
    monkeypatch.chdir(tmp_path)
    
    args = ["--config", str(config_path), "--stages", "1,2,3,4,5,6,7"]
    exit_code = main(args)
    assert exit_code == 0
    
    repro_dir = list(Path("results").glob("reproduction_*"))[0]
    verdict_md_path = repro_dir / "reports" / "final_claim_verdict.md"
    
    assert verdict_md_path.exists()
    md_content = verdict_md_path.read_text(encoding="utf-8")
    
    headers = [
        "# Final Claim Verdict",
        "## Scope",
        "## Evidence sources",
        "## Final reproduction status",
        "## Claim verdict summary",
        "## Structural findings",
        "## Geometry findings",
        "## Empirical findings",
        "## Invalidated legacy claims",
        "## Surviving claims",
        "## Claims requiring future work",
        "## README-safe draft summary",
        "## Forbidden public claims",
        "## Recommended next work",
        "## Final flags"
    ]
    
    for h in headers:
        assert h in md_content


def test_stage7_writes_readme_safe_summary_draft(temp_environment, monkeypatch, tmp_path):
    """Verify reports/readme_safe_summary_draft.md is generated."""
    config_path, ckpt_path = temp_environment
    monkeypatch.chdir(tmp_path)
    
    args = ["--config", str(config_path), "--stages", "1,2,3,4,5,6,7"]
    exit_code = main(args)
    assert exit_code == 0
    
    repro_dir = list(Path("results").glob("reproduction_*"))[0]
    readme_draft_path = repro_dir / "reports" / "readme_safe_summary_draft.md"
    assert readme_draft_path.exists()


def test_stage7_manifest_records_inputs_and_outputs(temp_environment, monkeypatch, tmp_path):
    """Verify Stage 7 manifest records all project-level inputs and Stage 7 reports with SHA-256."""
    config_path, ckpt_path = temp_environment
    monkeypatch.chdir(tmp_path)
    
    args = ["--config", str(config_path), "--stages", "1,2,3,4,5,6,7"]
    exit_code = main(args)
    assert exit_code == 0
    
    repro_dir = list(Path("results").glob("reproduction_*"))[0]
    manifest_path = repro_dir / "manifest.json"
    
    with open(manifest_path, "r", encoding="utf-8") as f:
        manifest = json.load(f)
        
    stage_7 = manifest["stages"]["stage_7_claim_verdict"]
    assert stage_7["status"] == "PASSED"
    
    # Check inputs
    inputs = stage_7["inputs"]
    assert "claim_verdict_matrix" in inputs
    assert "geometry_validity_status" in inputs
    assert "research_integrity" in inputs
    
    for k in ["claim_verdict_matrix", "geometry_validity_status", "research_integrity"]:
        assert len(inputs[k]["sha256"]) == 64
        
    # Check outputs
    outputs = stage_7["outputs"]
    assert "final_claim_verdict_json" in outputs
    assert "final_claim_verdict_md" in outputs
    assert "readme_safe_summary_draft" in outputs
    
    for k in ["final_claim_verdict_json", "final_claim_verdict_md", "readme_safe_summary_draft"]:
        assert len(outputs[k]["sha256"]) == 64


def test_stage7_invalidates_legacy_fisher_claims(temp_environment, monkeypatch, tmp_path):
    """Verify final claim report explicitly list legacy Fisher metrics as INVALIDATED/invalid."""
    config_path, ckpt_path = temp_environment
    monkeypatch.chdir(tmp_path)
    
    args = ["--config", str(config_path), "--stages", "1,2,3,4,5,6,7"]
    exit_code = main(args)
    assert exit_code == 0
    
    repro_dir = list(Path("results").glob("reproduction_*"))[0]
    verdict_md_path = repro_dir / "reports" / "final_claim_verdict.md"
    md_content = verdict_md_path.read_text(encoding="utf-8")
    
    assert "CLAIM_04: Pullback Fisher information metrics support latent manifold geometry claims" in md_content
    assert "CLAIM_05: SVD analysis shows the manifold is 2D, and geodesics are 5-10x shorter" in md_content
    assert "legacy Fisher metrics status: INVALID_DO_NOT_CLAIM" in md_content


def test_stage7_invalidates_reconstruction_validity_claims(temp_environment, monkeypatch, tmp_path):
    """Verify final claim report lists reconstruction validity claims as INVALIDATED due to collisions."""
    config_path, ckpt_path = temp_environment
    monkeypatch.chdir(tmp_path)
    
    args = ["--config", str(config_path), "--stages", "1,2,3,4,5,6,7"]
    exit_code = main(args)
    assert exit_code == 0
    
    repro_dir = list(Path("results").glob("reproduction_*"))[0]
    verdict_md_path = repro_dir / "reports" / "final_claim_verdict.md"
    md_content = verdict_md_path.read_text(encoding="utf-8")
    
    assert "CLAIM_02: VAE decoder outputs and reconstructed parameter vectors are structurally valid." in md_content
    assert "LEGACY_LAYOUT_COLLISION" in md_content
    assert "Reconstruction validity rate: 0.0%" in md_content


def test_stage7_marks_empirical_claim_needs_real_data_when_stage6_skipped(temp_environment, monkeypatch, tmp_path):
    """Verify Stage 7 marks empirical verdict as NEEDS_REAL_DATA if Stage 6 was skipped."""
    config_path, ckpt_path = temp_environment
    monkeypatch.chdir(tmp_path)
    
    args = ["--config", str(config_path), "--stages", "1,2,3,4,5,6,7"]
    exit_code = main(args)
    assert exit_code == 0
    
    repro_dir = list(Path("results").glob("reproduction_*"))[0]
    verdict_json_path = repro_dir / "reports" / "final_claim_verdict.json"
    
    with open(verdict_json_path, "r", encoding="utf-8") as f:
        data = json.load(f)
        
    assert data["empirical_findings"]["empirical_claim_status"] == "NEEDS_REAL_DATA"
    assert data["legacy_claim_status"]["empirical_no_edge_claim"] == "NEEDS_REAL_DATA"


def test_stage7_does_not_write_root_readme(temp_environment, monkeypatch, tmp_path):
    """Verify Stage 7 does not write or modify the root README.md."""
    config_path, ckpt_path = temp_environment
    
    # Create a dummy root README in tmp_path
    readme_path = tmp_path / "README.md"
    readme_content = "# Legacy Project README"
    readme_path.write_text(readme_content, encoding="utf-8")
    
    monkeypatch.chdir(tmp_path)
    
    args = ["--config", str(config_path), "--stages", "1,2,3,4,5,6,7"]
    exit_code = main(args)
    assert exit_code == 0
    
    # Verify README content remains identical
    assert readme_path.read_text(encoding="utf-8") == readme_content


def test_stage7_does_not_call_binance_api(temp_environment, monkeypatch, tmp_path):
    """Verify Stage 7 does not call Binance API or make web calls (executed offline)."""
    config_path, ckpt_path = temp_environment
    monkeypatch.chdir(tmp_path)
    
    def raise_err(*args, **kwargs):
        pytest.fail("Stage 7 attempted network communication!")
        
    monkeypatch.setattr("urllib.request.urlopen", raise_err)
    monkeypatch.setattr("pandas.read_csv", raise_err)
    
    args = ["--config", str(config_path), "--stages", "1,2,3,4,5,6,7"]
    exit_code = main(args)
    assert exit_code == 0


def test_stage7_does_not_create_fake_data(temp_environment, monkeypatch, tmp_path):
    """Verify Stage 7 does not generate fake data files."""
    config_path, ckpt_path = temp_environment
    monkeypatch.chdir(tmp_path)
    
    args = ["--config", str(config_path), "--stages", "1,2,3,4,5,6,7"]
    exit_code = main(args)
    assert exit_code == 0
    
    # The only files generated should be within reproduction output dir
    generated_files = list(Path(".").glob("**/*"))
    for gf in generated_files:
        if gf.is_file():
            # Check that it's located inside configs, models, docs, results, or is a temp test file
            assert any(p in gf.parts for p in ["configs", "models", "docs", "results"])


def test_stage7_does_not_create_generated_valid_thetas(temp_environment, monkeypatch, tmp_path):
    """Verify Stage 7 does not create generated_valid_thetas.npy."""
    config_path, ckpt_path = temp_environment
    monkeypatch.chdir(tmp_path)
    
    args = ["--config", str(config_path), "--stages", "1,2,3,4,5,6,7"]
    exit_code = main(args)
    assert exit_code == 0
    
    assert not Path("generated_valid_thetas.npy").exists()


def test_stage7_public_summary_avoids_forbidden_claims(temp_environment, monkeypatch, tmp_path):
    """Verify that the draft public summary avoids all forbidden claims."""
    config_path, ckpt_path = temp_environment
    monkeypatch.chdir(tmp_path)
    
    args = ["--config", str(config_path), "--stages", "1,2,3,4,5,6,7"]
    exit_code = main(args)
    assert exit_code == 0
    
    repro_dir = list(Path("results").glob("reproduction_*"))[0]
    readme_draft_path = repro_dir / "reports" / "readme_safe_summary_draft.md"
    content = readme_draft_path.read_text(encoding="utf-8").lower()
    
    forbidden_terms = [
        "generates valid", "valid AR/GARCH models", "valid hybrid structures",
        "proven 2D", "5-10x shorter", "proves no trading edge",
        "trading strategy", "alpha", "profit", "profitable"
    ]
    
    for term in forbidden_terms:
        assert term not in content


def test_stage7_forbidden_public_claims_are_listed(temp_environment, monkeypatch, tmp_path):
    """Verify forbidden public claims are listed inside the final verdict json and md reports."""
    config_path, ckpt_path = temp_environment
    monkeypatch.chdir(tmp_path)
    
    args = ["--config", str(config_path), "--stages", "1,2,3,4,5,6,7"]
    exit_code = main(args)
    assert exit_code == 0
    
    repro_dir = list(Path("results").glob("reproduction_*"))[0]
    
    verdict_json_path = repro_dir / "reports" / "final_claim_verdict.json"
    with open(verdict_json_path, "r", encoding="utf-8") as f:
        data = json.load(f)
        
    assert len(data["forbidden_public_claims"]) > 0
    
    verdict_md_path = repro_dir / "reports" / "final_claim_verdict.md"
    md_content = verdict_md_path.read_text(encoding="utf-8")
    assert "Forbidden public claims" in md_content


def test_stage7_does_not_require_fragile_markdown_table_formatting(temp_environment, monkeypatch, tmp_path):
    """Verify Stage 7 parses claim matrix robustly even if table layout changes slightly."""
    config_path, ckpt_path = temp_environment
    monkeypatch.chdir(tmp_path)
    
    # Modify docs/CLAIM_VERDICT_MATRIX.md to break markdown table formatting slightly
    claim_matrix_path = tmp_path / "docs" / "CLAIM_VERDICT_MATRIX.md"
    bad_table_content = (
        "# Claim Verdict Matrix\n"
        "## Verdict summary\n"
        "Status | Count\n"
        "CONFIRMED: 0\n"
        "PARTIALLY_CONFIRMED: 1\n"
        "Required anchors: CLAIM_01 CLAIM_02 CLAIM_03 CLAIM_04 CLAIM_05 CLAIM_06 CLAIM_07 "
        "LEGACY_LAYOUT_COLLISION INVALIDATED NEEDS_REAL_DATA"
    )
    claim_matrix_path.write_text(bad_table_content, encoding="utf-8")
    
    args = ["--config", str(config_path), "--stages", "1,2,3,4,5,6,7"]
    exit_code = main(args)
    # The run should still succeed robustly rather than failing due to format changes
    assert exit_code == 0
