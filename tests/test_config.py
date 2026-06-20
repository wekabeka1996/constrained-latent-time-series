"""
tests/test_config.py
====================
Tests for Phase 1C.5 — YAML config loading and Pydantic validation.

Run from the project root::

    python -m pytest tests/test_config.py -v
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from pydantic import ValidationError

from src.config import (
    AppConfig,
    GenerationConfigModel,
    ModelConfig,
    ValidationConfig,
    VectorSchemaConfig,
    load_config,
)
from src.vector_schema import MAX_LAG_ORDER, VECTOR_DIM

DEMO_YAML = PROJECT_ROOT / "configs" / "demo.yaml"


class TestDemoConfigLoads:
    def test_demo_config_loads(self) -> None:
        cfg = load_config(DEMO_YAML)
        assert isinstance(cfg, AppConfig)
        assert cfg.project.name == "econometric-vae-manifold"

    def test_config_matches_vector_schema_constants(self) -> None:
        cfg = load_config(DEMO_YAML)
        assert cfg.model.input_dim == VECTOR_DIM
        assert cfg.vector_schema.vector_dim == VECTOR_DIM
        assert cfg.vector_schema.max_lag_order == MAX_LAG_ORDER

    def test_generation_mode_is_explicit(self) -> None:
        cfg = load_config(DEMO_YAML)
        assert cfg.generation.mode in ("legacy_compatible", "clean_validated")
        # Demo config must explicitly use legacy mode
        assert cfg.generation.mode == "legacy_compatible"

    def test_config_threshold_ranges(self) -> None:
        cfg = load_config(DEMO_YAML)
        assert 0.0 < cfg.validation.activation_threshold < 1.0
        assert 0.0 <= cfg.validation.activation_epsilon < 0.5
        assert 0.0 < cfg.validation.tolerance < 1.0

    def test_config_checkpoint_path_is_string(self) -> None:
        cfg = load_config(DEMO_YAML)
        assert isinstance(cfg.model.checkpoint_path, str)
        assert len(cfg.model.checkpoint_path) > 0


class TestValidationRejects:
    def test_invalid_vector_dim_rejected(self) -> None:
        with pytest.raises(ValidationError, match="vector_dim"):
            VectorSchemaConfig(
                vector_dim=VECTOR_DIM + 1,  # wrong
                struct_dim=10,
                param_dim=20,
                stat_dim=10,
                max_lag_order=MAX_LAG_ORDER,
            )

    def test_invalid_threshold_rejected(self) -> None:
        with pytest.raises(ValidationError):
            ValidationConfig(
                activation_threshold=1.5,   # > 1: invalid
                activation_epsilon=0.05,
                tolerance=1e-8,
            )

    def test_invalid_generation_mode_rejected(self) -> None:
        with pytest.raises(ValidationError):
            GenerationConfigModel(
                n_arma=100,
                n_garch=100,
                seed=42,
                mode="something_unsupported",   # not in Literal
                allow_supervised_arma_garch=False,
            )

    def test_invalid_model_input_dim_rejected(self) -> None:
        with pytest.raises(ValidationError, match="input_dim"):
            ModelConfig(
                architecture="baseline_vae",
                input_dim=32,     # wrong: not VECTOR_DIM
                latent_dim=8,
                hidden_dims=[64, 32],
                checkpoint_path="models/vae_beta50.pth",
                checkpoint_role="test",
            )

    def test_file_not_found_raises(self) -> None:
        with pytest.raises(FileNotFoundError):
            load_config("configs/nonexistent_file.yaml")

    def test_integrity_random_fallbacks_disabled(self) -> None:
        cfg = load_config(DEMO_YAML)
        assert cfg.integrity.allow_random_fallbacks is False

    def test_integrity_fake_market_data_disabled(self) -> None:
        cfg = load_config(DEMO_YAML)
        assert cfg.integrity.allow_fake_market_data is False

    def test_integrity_requires_artifact_provenance(self) -> None:
        cfg = load_config(DEMO_YAML)
        assert cfg.integrity.require_artifact_provenance is True

    def test_fail_fast_on_missing_required_artifacts(self) -> None:
        cfg = load_config(DEMO_YAML)
        assert cfg.integrity.fail_fast_on_missing_required_artifacts is True

    def test_invalid_integrity_flags_rejected(self) -> None:
        # Pydantic should raise ValidationError because we defined
        # these must be False in the IntegrityConfig validator
        from src.config import IntegrityConfig
        with pytest.raises(ValidationError, match="must be false"):
            IntegrityConfig(allow_random_fallbacks=True)
            
        with pytest.raises(ValidationError, match="must be false"):
            IntegrityConfig(allow_fake_market_data=True)

    def test_market_data_can_be_null_without_fake_substitution(self) -> None:
        from src.config import PathsConfig
        paths = PathsConfig(market_data_csv=None, output_dir="results/demo")
        assert paths.market_data_csv is None
