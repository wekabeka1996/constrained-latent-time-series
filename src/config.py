"""
src/config.py
=============
Pydantic v2 configuration models for the econometric VAE project.

Loads YAML configuration files and validates them against the canonical
constants defined in ``src/vector_schema.py``.

Usage::

    from src.config import load_config
    cfg = load_config("configs/demo.yaml")
    print(cfg.model.checkpoint_path)
"""

from __future__ import annotations

from pathlib import Path
from typing import Annotated, Literal, Optional

import yaml
from pydantic import BaseModel, Field, model_validator

from .vector_schema import MAX_LAG_ORDER, VECTOR_DIM

# ---------------------------------------------------------------------------
# Sub-models
# ---------------------------------------------------------------------------

GenerationMode = Literal["legacy_compatible", "clean_validated"]


class ProjectConfig(BaseModel):
    """Top-level project identity."""
    name: str
    mode: str = "demo"


class ModelConfig(BaseModel):
    """VAE architecture and checkpoint metadata."""
    architecture: str
    input_dim: int
    latent_dim: int
    hidden_dims: list[int]
    checkpoint_path: str
    checkpoint_role: str = "unspecified"

    @model_validator(mode="after")
    def _input_dim_matches_schema(self) -> "ModelConfig":
        if self.input_dim != VECTOR_DIM:
            raise ValueError(
                f"model.input_dim must be {VECTOR_DIM} (VECTOR_DIM), got {self.input_dim}"
            )
        return self


class VectorSchemaConfig(BaseModel):
    """Explicit YAML copy of vector_schema.py constants — must match exactly."""
    vector_dim: int
    struct_dim: int
    param_dim: int
    stat_dim: int
    max_lag_order: int

    @model_validator(mode="after")
    def _dims_consistent(self) -> "VectorSchemaConfig":
        if self.vector_dim != VECTOR_DIM:
            raise ValueError(
                f"vector_schema.vector_dim must be {VECTOR_DIM}, got {self.vector_dim}"
            )
        if self.max_lag_order != MAX_LAG_ORDER:
            raise ValueError(
                f"vector_schema.max_lag_order must be {MAX_LAG_ORDER}, got {self.max_lag_order}"
            )
        if self.struct_dim + self.param_dim + self.stat_dim != self.vector_dim:
            raise ValueError(
                f"struct_dim + param_dim + stat_dim must equal vector_dim "
                f"({self.struct_dim} + {self.param_dim} + {self.stat_dim} != {self.vector_dim})"
            )
        return self


class GenerationConfigModel(BaseModel):
    """Training data generation parameters."""
    n_arma: int = Field(gt=0)
    n_garch: int = Field(gt=0)
    seed: int = 42
    mode: GenerationMode
    allow_supervised_arma_garch: bool = False
    # Explicitly asserts synthetic data is part of the research design,
    # not a substitute for real empirical data.
    synthetic_training_data_is_methodology: bool = False


class ValidationConfig(BaseModel):
    """Threshold values for the canonical validation module."""
    activation_threshold: Annotated[float, Field(gt=0.0, lt=1.0)]
    activation_epsilon: Annotated[float, Field(ge=0.0, lt=0.5)]
    tolerance: Annotated[float, Field(gt=0.0, lt=1.0)]


class PathsConfig(BaseModel):
    """Project filesystem paths (relative to project root by convention)."""
    # Real market data. Must be null if no real data is available.
    # Do NOT substitute with random/fake market series.
    market_data_csv: Optional[str] = None
    output_dir: str


class IntegrityConfig(BaseModel):
    """Research integrity enforcement flags.

    These are not optional preferences — they define the project's
    non-negotiable contract with its own research claims.
    """
    # If True, scripts may substitute missing artifacts with random arrays.
    # MUST be False in all public demo configs.
    allow_random_fallbacks: bool = False
    # If True, scripts may use synthetic/random data to simulate market prices.
    # MUST be False in all public demo configs.
    allow_fake_market_data: bool = False
    # Cached artifacts (e.g. Z_train.npy from a prior training run) are allowed
    # but must be labeled as such, not presented as freshly reproducible.
    allow_cached_artifacts: bool = True
    # Every artifact used as research evidence must have documented provenance.
    require_artifact_provenance: bool = True
    # If True, missing required inputs must crash the process.
    fail_fast_on_missing_required_artifacts: bool = True

    @model_validator(mode="after")
    def _integrity_contract_in_public_config(self) -> "IntegrityConfig":
        if self.allow_random_fallbacks:
            raise ValueError(
                "integrity.allow_random_fallbacks must be false. "
                "Random fallbacks substitute fabricated data for missing evidence."
            )
        if self.allow_fake_market_data:
            raise ValueError(
                "integrity.allow_fake_market_data must be false. "
                "Fake market data violates basic research credibility."
            )
        return self


class GeometryConfig(BaseModel):
    """Geometry calculation parameters."""
    max_geometry_points: int = Field(default=32, gt=0)
    finite_difference_eps: float = Field(default=1e-4, gt=0.0)
    metric_regularization_eps: float = Field(default=1e-8, gt=0.0)


# ---------------------------------------------------------------------------
# Root config
# ---------------------------------------------------------------------------

class AppConfig(BaseModel):
    """Top-level application configuration object."""
    project: ProjectConfig
    model: ModelConfig
    vector_schema: VectorSchemaConfig
    generation: GenerationConfigModel
    validation: ValidationConfig
    paths: PathsConfig
    integrity: IntegrityConfig = IntegrityConfig()
    geometry: GeometryConfig = GeometryConfig()


# ---------------------------------------------------------------------------
# Loader
# ---------------------------------------------------------------------------

def load_config(path: str | Path) -> AppConfig:
    """Load and validate a YAML configuration file.

    Parameters
    ----------
    path:
        Path to a YAML file (absolute or relative to cwd).

    Returns
    -------
    AppConfig:
        Fully validated configuration object.

    Raises
    ------
    FileNotFoundError:
        If the file does not exist.
    pydantic.ValidationError:
        If any field fails validation.
    yaml.YAMLError:
        If the file is not valid YAML.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")
    with path.open("r", encoding="utf-8") as f:
        raw = yaml.safe_load(f)
    return AppConfig.model_validate(raw)
