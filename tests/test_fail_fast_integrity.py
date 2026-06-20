"""
tests/test_fail_fast_integrity.py
==================================
Tests for Phase 1E: Fail-Fast integrity verification and source scans.

Run from project root:
    python -m pytest tests/test_fail_fast_integrity.py -v
"""

from __future__ import annotations

import sys
from pathlib import Path
import pytest

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.io_utils import require_file
from src.config import load_config


def test_require_file_returns_existing_path(tmp_path: Path) -> None:
    # Arrange
    temp_file = tmp_path / "existing_artifact.npy"
    temp_file.write_bytes(b"dummy data")
    
    # Act
    resolved = require_file(temp_file, "Testing existence")
    
    # Assert
    assert resolved == temp_file
    assert resolved.exists()


def test_require_file_missing_raises_clear_error() -> None:
    # Arrange
    nonexistent = Path("nonexistent_artifact.npy")
    purpose_str = "For testing fail-fast behavior"
    
    # Act & Assert
    with pytest.raises(FileNotFoundError) as exc_info:
        require_file(nonexistent, purpose_str)
        
    err_msg = str(exc_info.value)
    assert "nonexistent_artifact.npy" in err_msg
    assert purpose_str in err_msg
    assert "forbidden" in err_msg.lower()
    assert "documented canonical pipeline" in err_msg.lower()


def test_generate_latent_vectors_has_no_dummy_data_function() -> None:
    # Arrange
    script_path = PROJECT_ROOT / "archive" / "legacy_code" / "generate_latent_vectors.py"
    content = script_path.read_text(encoding="utf-8")
    
    # Assert
    assert "def generate_dummy_data_if_needed" not in content
    assert "generate_dummy_data_if_needed" not in content


def test_generate_latent_vectors_has_no_random_uniform_fallback() -> None:
    # Arrange
    script_path = PROJECT_ROOT / "archive" / "legacy_code" / "generate_latent_vectors.py"
    content = script_path.read_text(encoding="utf-8")
    
    # Assert that there is no uniform random generation in the file
    assert "np.random.uniform" not in content
    assert "random.uniform" not in content


def test_latent_interpolation_has_no_random_normal_fallback() -> None:
    # Arrange
    script_path = PROJECT_ROOT / "archive" / "legacy_code" / "latent_space_interpolation_analysis.py"
    content = script_path.read_text(encoding="utf-8")
    
    # Assert that it no longer contains np.random.randn fallback
    # The file has np.random.choice for selecting pairs, but should not use randn
    assert "np.random.randn" not in content


def test_regularized_fisher_metric_has_no_bare_except() -> None:
    # Arrange
    script_path = PROJECT_ROOT / "archive" / "invalidated_geometry" / "regularized_fisher_metric.py"
    content = script_path.read_text(encoding="utf-8")
    
    # Check for bare 'except:' (with or without whitespace)
    # e.g., 'except:' or 'except :'
    import re
    bare_except_pattern = re.compile(r"except\s*:")
    matches = bare_except_pattern.findall(content)
    
    # Assert
    assert len(matches) == 0, f"Found bare except blocks in regularized_fisher_metric.py: {matches}"


def test_demo_config_disables_random_fallbacks() -> None:
    # Arrange
    demo_yaml = PROJECT_ROOT / "configs" / "demo.yaml"
    
    # Act
    cfg = load_config(demo_yaml)
    
    # Assert
    assert cfg.integrity.allow_random_fallbacks is False
    assert cfg.integrity.fail_fast_on_missing_required_artifacts is True
