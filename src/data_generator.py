"""
src/data_generator.py
=====================
Canonical reproducible training data generator for the VAE.

This module replicates the original ``generate_structured_theta`` logic from
``Симуляція.py`` to preserve exact compatibility with existing checkpoints.

GENERATION MODES
----------------
legacy_compatible (current default):
    Reproduces the original disjoint training grammar:
      - Pure ARMA-like vectors (labeled "AR").
      - Pure GARCH vectors (labeled "GARCH").

    Quirks PRESERVED for checkpoint compatibility:
      1. ARMA vectors sample a q_order structural logit but do NOT generate
         any MA coefficients (theta). Only AR coefficients (phi) fill params.
      2. params[0] == phi_1 for AR vectors; params[0] == omega for GARCH.
         This collision is an inherent limitation of the original design.
      3. GARCH boundary bug: beta sampled from uniform(0, 0.4 - sum_alpha).
         When sum_alpha > 0.4, the upper bound is negative.  Legacy
         np.random.uniform silently swapped bounds, producing negative betas
         and models with persistence > 1.  This behaviour is reproduced here
         using explicit bound sorting, NOT corrected.

    DO NOT claim this generator produces mathematically valid GARCH models.
    DO NOT use this mode to benchmark econometric quality.
    USE this mode when retraining models that must match existing .pth weights.

clean_validated (NOT YET IMPLEMENTED):
    Future mode with strict GARCH constraints (persistence < 1, beta >= 0).
    Must NOT be used for retraining checkpoint-compatible models.

Combined ARMA-GARCH structures found in this project are latent-space
interpolation phenomena, NOT supervised training classes.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import numpy as np

from .vector_schema import (
    ARMA_ACTIVE_IDX,
    D_PARAM,
    D_STAT,
    D_STRUCT,
    GARCH_ACTIVE_IDX,
    MAX_LAG_ORDER,
    P_ORDER_IDX,
    Q_ORDER_IDX,
    R_ORDER_IDX,
    S_ORDER_IDX,
    STAT_SLICE,
    STRUCT_SLICE,
    VECTOR_DIM,
)


@dataclass(frozen=True)
class GenerationConfig:
    """Configuration for dataset generation."""
    n_arma: int = 100
    n_garch: int = 100
    seed: int = 42
    max_lag_order: int = MAX_LAG_ORDER


def generate_arma_vector(
    rng: np.random.Generator,
    config: GenerationConfig,
) -> np.ndarray:
    """Generate a single pure AR-like hypothesis vector.

    Matches the "ARMA" branch of the original `generate_structured_theta`.
    Note that while a `q_order` structural logit is sampled, no MA
    coefficients are actually inserted into the parameter slice.

    Parameters
    ----------
    rng:
        Numpy random generator instance.
    config:
        Configuration containing max_lag_order.

    Returns
    -------
    vector:
        A 1-D numpy array of length 40.
    """
    v = np.zeros(VECTOR_DIM, dtype=float)

    # 1. Structure
    struct = np.zeros(D_STRUCT, dtype=float)
    struct[ARMA_ACTIVE_IDX] = 1.0  # ARMA indicator active
    
    # Randomly sample logits in [-1, 1]
    p_logit = rng.uniform(-1.0, 1.0)
    q_logit = rng.uniform(-1.0, 1.0)
    struct[P_ORDER_IDX] = p_logit
    struct[Q_ORDER_IDX] = q_logit
    
    # Compute discrete order using the project's rounding formula
    p_order = round(((p_logit + 1.0) / 2.0) * config.max_lag_order)

    # 2. Parameters
    params = np.zeros(D_PARAM, dtype=float)
    if p_order > 0:
        phi = rng.uniform(-0.8, 0.8, size=p_order)
        params[:p_order] = phi
    # Note: original code generated NO theta (MA) coefficients.

    # 3. Stats
    stats = rng.uniform(-1.0, 1.0, size=D_STAT)

    # Assemble
    v[STRUCT_SLICE] = struct
    v[D_STRUCT : D_STRUCT + D_PARAM] = params
    v[STAT_SLICE] = stats

    return v


def generate_garch_vector(
    rng: np.random.Generator,
    config: GenerationConfig,
) -> np.ndarray:
    """Generate a single pure GARCH hypothesis vector.

    Matches the "GARCH" branch of the original `generate_structured_theta`.

    Parameters
    ----------
    rng:
        Numpy random generator instance.
    config:
        Configuration containing max_lag_order.

    Returns
    -------
    vector:
        A 1-D numpy array of length 40.
    """
    v = np.zeros(VECTOR_DIM, dtype=float)

    # 1. Structure
    struct = np.zeros(D_STRUCT, dtype=float)
    struct[GARCH_ACTIVE_IDX] = 1.0  # GARCH indicator active

    r_logit = rng.uniform(-1.0, 1.0)
    s_logit = rng.uniform(-1.0, 1.0)
    struct[R_ORDER_IDX] = r_logit
    struct[S_ORDER_IDX] = s_logit

    r_order = round(((r_logit + 1.0) / 2.0) * config.max_lag_order)
    s_order = round(((s_logit + 1.0) / 2.0) * config.max_lag_order)

    # 2. Parameters
    params = np.zeros(D_PARAM, dtype=float)
    omega = rng.uniform(0.1, 0.5)
    alpha = rng.uniform(0.0, 0.3, size=r_order)
    sum_alpha = np.sum(alpha)
    # Preserve original anomaly: if sum_alpha > 0.4, upper bound is negative.
    # Legacy np.random.uniform(0, negative) silently swapped bounds.
    # New rng.uniform requires low <= high, so we sort them explicitly.
    high_bound = 0.4 - sum_alpha
    low, high = min(0.0, high_bound), max(0.0, high_bound)
    beta = rng.uniform(low, high, size=s_order)

    params[0] = omega
    if r_order > 0:
        params[1 : 1 + r_order] = alpha
    if s_order > 0:
        params[1 + r_order : 1 + r_order + s_order] = beta

    # 3. Stats
    stats = rng.uniform(-1.0, 1.0, size=D_STAT)

    # Assemble
    v[STRUCT_SLICE] = struct
    v[D_STRUCT : D_STRUCT + D_PARAM] = params
    v[STAT_SLICE] = stats

    return v


def generate_training_dataset(
    config: GenerationConfig | None = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """Generate a full, disjoint synthetic training dataset.

    Replicates the exact shape and label structure of the original `Z_train.npy`
    and `train_labels.npy`.

    Parameters
    ----------
    config:
        Dataset configuration. Defaults to `GenerationConfig()`.

    Returns
    -------
    (X, y):
        X is a 2-D numpy array of shape (n_arma + n_garch, VECTOR_DIM).
        y is a 1-D numpy array of string labels ("AR" or "GARCH").
    """
    if config is None:
        config = GenerationConfig()

    rng = np.random.default_rng(config.seed)

    X_list = []
    y_list = []

    # Generate ARMA (AR-like)
    for _ in range(config.n_arma):
        X_list.append(generate_arma_vector(rng, config))
        y_list.append("AR")  # Preserving original label string

    # Generate GARCH
    for _ in range(config.n_garch):
        X_list.append(generate_garch_vector(rng, config))
        y_list.append("GARCH")

    X = np.stack(X_list, axis=0)
    y = np.array(y_list, dtype=str)

    return X, y


def generate_training_dataset_from_config(app_config: "AppConfig") -> Tuple[np.ndarray, np.ndarray]:  # noqa: F821
    """Bridge from ``AppConfig`` to ``generate_training_dataset``.

    Converts ``app_config.generation`` into a :class:`GenerationConfig` and
    delegates to :func:`generate_training_dataset`.

    Parameters
    ----------
    app_config:
        Loaded application configuration (see ``src/config.py``).

    Returns
    -------
    (X, y):
        Same as :func:`generate_training_dataset`.
    """
    # Import here to avoid circular dependency: config.py imports vector_schema,
    # not data_generator. data_generator does not import config at module load time.

    g = app_config.generation
    cfg = GenerationConfig(
        n_arma=g.n_arma,
        n_garch=g.n_garch,
        seed=g.seed,
        max_lag_order=app_config.vector_schema.max_lag_order,
    )
    return generate_training_dataset(cfg)
