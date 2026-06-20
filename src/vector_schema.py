"""
src/vector_schema.py
====================
Formal contract for the 40-dimensional model vector layout.

This module defines the canonical slices and indices used throughout the project
to read and write hypotheses in the VAE latent and parameter spaces.

Verdict from Phase 1B.5 Audit:
    MOSTLY_CONSISTENT_WITH_MINOR_AMBIGUITY

    The structure slice (0:10), parameter slice (10:30), and stats slice (30:40)
    are universally consistent across all scripts. The order decoding (MAX_LAG_ORDER=5)
    is also universal.

    **Critical Ambiguity (Coefficient Overlap):**
    In `Симуляція.py`, `generate_structured_theta` populates the parameter slice
    from index 0 for both ARMA and GARCH:
        - For ARMA, params[0:p] are AR coefficients (φ).
        - For GARCH, params[0] is ω, params[1:1+r] are α, etc.
    This means `v[10]` represents `φ₁` in pure ARMA, but `ω` in pure GARCH.
    The training data generator did NOT generate combined ARMA-GARCH examples.
    Therefore, the layout for a combined ARMA-GARCH parameter vector is formally
    UNDEFINED in the original code, leading to an inherent collision at `v[10]`.
    Until this is resolved in a future architecture update, the parameter mapping
    remains context-dependent.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import numpy as np

# ---------------------------------------------------------------------------
# Core Dimension Constants
# ---------------------------------------------------------------------------

VECTOR_DIM: int = 40
D_STRUCT: int = 10
D_PARAM: int = 20
D_STAT: int = 10
MAX_LAG_ORDER: int = 5

# ---------------------------------------------------------------------------
# Slices
# ---------------------------------------------------------------------------

STRUCT_SLICE = slice(0, 10)
PARAM_SLICE  = slice(10, 30)
STAT_SLICE   = slice(30, 40)

# ---------------------------------------------------------------------------
# Structural Indices (relative to the struct slice, or the full vector)
# ---------------------------------------------------------------------------

ARMA_ACTIVE_IDX: int = 0
GARCH_ACTIVE_IDX: int = 1
P_ORDER_IDX: int = 2
Q_ORDER_IDX: int = 3
R_ORDER_IDX: int = 4
S_ORDER_IDX: int = 5

RESERVED_STRUCT_SLICE = slice(6, 10)


@dataclass(frozen=True)
class VectorLayout:
    """Read-only contract of the vector dimensions and slices."""
    vector_dim: int = VECTOR_DIM
    struct_slice: slice = STRUCT_SLICE
    param_slice: slice = PARAM_SLICE
    stat_slice: slice = STAT_SLICE
    max_lag_order: int = MAX_LAG_ORDER


# ---------------------------------------------------------------------------
# Helper Functions
# ---------------------------------------------------------------------------

def assert_vector_shape(vector: np.ndarray) -> None:
    """Assert that the vector has the exact expected length of VECTOR_DIM.

    Parameters
    ----------
    vector:
        1-D or 2-D numpy array. If 2-D, the last dimension must be VECTOR_DIM.

    Raises
    ------
    ValueError:
        If the vector does not have the correct size.
    """
    if vector.ndim == 1:
        if len(vector) != VECTOR_DIM:
            raise ValueError(f"Expected 1D vector of length {VECTOR_DIM}, got {len(vector)}")
    elif vector.ndim == 2:
        if vector.shape[1] != VECTOR_DIM:
            raise ValueError(f"Expected 2D array with shape (..., {VECTOR_DIM}), got {vector.shape}")
    else:
        raise ValueError(f"Expected 1D or 2D array, got {vector.ndim}D array")


def split_vector(vector: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Split a full model vector into its three constituent parts.

    Parameters
    ----------
    vector:
        A 1-D numpy array of length VECTOR_DIM, or 2-D array of shape (N, VECTOR_DIM).

    Returns
    -------
    (struct_part, param_part, stat_part):
        Tuple of numpy arrays split according to the canonical slices.
    """
    assert_vector_shape(vector)
    if vector.ndim == 1:
        return (
            vector[STRUCT_SLICE],
            vector[PARAM_SLICE],
            vector[STAT_SLICE],
        )
    else:
        return (
            vector[:, STRUCT_SLICE],
            vector[:, PARAM_SLICE],
            vector[:, STAT_SLICE],
        )


def decode_order_logit(value: float, max_lag_order: int = MAX_LAG_ORDER) -> int:
    """Decode a continuous [-1, 1] logit into a discrete lag order.

    The continuous value is first shifted and scaled to [0, 1], then multiplied
    by max_lag_order, and rounded to the nearest integer.

    Parameters
    ----------
    value:
        Continuous logit, expected to be in [-1, 1].
    max_lag_order:
        The maximum discrete order allowed. Default is MAX_LAG_ORDER (5).

    Returns
    -------
    int:
        The decoded discrete lag order, typically in [0, max_lag_order].
    """
    # Note: Using standard round() which implements banker's rounding (round half to even).
    # This exactly matches the original Симуляція.py logic.
    return round(((value + 1.0) / 2.0) * max_lag_order)


def describe_layout() -> dict[str, object]:
    """Return a dictionary describing the canonical vector layout."""
    return {
        "VECTOR_DIM": VECTOR_DIM,
        "STRUCT_SLICE": (STRUCT_SLICE.start, STRUCT_SLICE.stop),
        "PARAM_SLICE": (PARAM_SLICE.start, PARAM_SLICE.stop),
        "STAT_SLICE": (STAT_SLICE.start, STAT_SLICE.stop),
        "MAX_LAG_ORDER": MAX_LAG_ORDER,
        "NOTE": "Coefficient overlap exists at params[0] (vector[10]). It represents phi_1 for ARMA and omega for GARCH."
    }
