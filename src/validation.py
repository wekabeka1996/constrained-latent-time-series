"""
src/validation.py
=================
Canonical validation module for the econometric-vae-manifold project.

This module is the single source of truth for:

  * Structure decoding  (decode_discrete_structure)
  * AR/MA polynomial root computation
  * AR stationarity and MA invertibility checks
  * GARCH parameter constraint checks
  * Combined ARMA-GARCH validity

Convention notes (verified from Симуляція.py, Симуляція_2.py,
latent_space_interpolation_analysis.py, analyze_roots_boundary_validity.py):

Vector layout (d_input = 40, tanh-normalised to [-1, 1]):
    v[0:10]  = structural sub-vector (D_STRUCT = 10)
        v[0] = ARMA activation logit
        v[1] = GARCH activation logit
        v[2] = p_order logit
        v[3] = q_order logit
        v[4] = r_order logit
        v[5] = s_order logit
        v[6:10] = reserved / unused
    v[10:30] = parameter sub-vector (D_PARAM = 20)
        params[0]         = ω (GARCH intercept)
        params[1:1+r]     = α coefficients (ARCH terms)
        params[1+r:1+r+s] = β coefficients (GARCH terms)
        params[0:p]       = φ  (AR coefficients, when no GARCH)
    v[30:40] = statistical-feature sub-vector (D_STAT = 10)

Order decoding:
    raw ∈ [-1, 1]  →  score = (raw + 1) / 2 ∈ [0, 1]
    order = round(score * MAX_LAG_ORDER)  where MAX_LAG_ORDER = 5

Type detection uses a threshold of 0.5 with an epsilon buffer of 0.05
(matching the convention in Симуляція.py and Симуляція_2.py).

Polynomial conventions (verified from analyze_roots_boundary_validity.py
and confirmed by numerical experiment 2026-06-18):

    The project uses the COMPANION (characteristic) polynomial form, not the
    lag-operator form.  numpy.roots is called with:

        AR: coeffs = [1, -φ₁, -φ₂, ..., -φₚ]
        MA: coeffs = [1,  θ₁,  θ₂, ...,  θ_q]

    np.roots([c₀, c₁, ..., cₙ]) finds z such that c₀·zⁿ + c₁·zⁿ⁻¹ + ... = 0.

    For AR(1) with φ=0.5: np.roots([1, -0.5]) → root=0.5 (the root IS φ).
    Stationarity in this companion form: |root| < 1  (i.e. |φ| < 1 for AR(1)).

    For MA(1) with θ=0.5: np.roots([1, 0.5]) → root=−0.5, |root|=0.5 < 1.
    Invertibility in this companion form: |root| < 1.

GARCH constraints (verified from Симуляція.py validate_theta_advanced):
    ω  >  0
    αᵢ >= 0  for all i
    βⱼ >= 0  for all j
    Σα + Σβ  < 1   (strict; >= 1 is invalid)
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence, Union

import numpy as np

from .vector_schema import (
    ARMA_ACTIVE_IDX as _IDX_ARMA_LOGIT,
    GARCH_ACTIVE_IDX as _IDX_GARCH_LOGIT,
    P_ORDER_IDX as _IDX_P_LOGIT,
    Q_ORDER_IDX as _IDX_Q_LOGIT,
    R_ORDER_IDX as _IDX_R_LOGIT,
    S_ORDER_IDX as _IDX_S_LOGIT,
    MAX_LAG_ORDER,
    decode_order_logit,
)

# Derived max orders (aliases for clarity in tests / docs)
DEFAULT_MAX_P: int = MAX_LAG_ORDER
DEFAULT_MAX_Q: int = MAX_LAG_ORDER
DEFAULT_MAX_R: int = MAX_LAG_ORDER
DEFAULT_MAX_S: int = MAX_LAG_ORDER

# ---------------------------------------------------------------------------
# Public dataclasses
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ModelStructure:
    """Decoded discrete structure of a generated hypothesis θ.

    Attributes
    ----------
    model_type:
        One of ``"ARMA"``, ``"GARCH"``, ``"ARMA-GARCH"``, or ``"UNKNOWN"``.
    p:
        AR lag order (0 if model has no ARMA component).
    q:
        MA lag order (0 if model has no ARMA component).
    r:
        ARCH lag order (0 if model has no GARCH component).
    s:
        GARCH lag order (0 if model has no GARCH component).
    """

    model_type: str
    p: int
    q: int
    r: int
    s: int


@dataclass(frozen=True)
class ValidationResult:
    """Result of :func:`validate_arma_garch`.

    Attributes
    ----------
    is_valid:
        ``True`` iff all applicable sub-checks pass.
    is_stationary_ar:
        ``True`` iff AR(p) polynomial roots all lie strictly outside the unit
        circle, or ``p == 0`` (trivially stationary).
    is_invertible_ma:
        ``True`` iff MA(q) polynomial roots all lie strictly outside the unit
        circle, or ``q == 0`` (trivially invertible).
    is_valid_garch:
        ``True`` iff all GARCH constraints hold, or no GARCH component.
    reason:
        Human-readable failure description; ``None`` when valid.
    """

    is_valid: bool
    is_stationary_ar: bool
    is_invertible_ma: bool
    is_valid_garch: bool
    reason: str | None = None


# ---------------------------------------------------------------------------
# AR polynomial root computation
# ---------------------------------------------------------------------------


def compute_ar_roots(
    phi: Union[Sequence[float], np.ndarray],
) -> np.ndarray:
    """Compute roots of the AR companion polynomial.

    The companion polynomial passed to ``np.roots`` is::

        [1, -φ₁, -φ₂, ..., -φₚ]

    which represents:  zᵖ - φ₁·zᵖ⁻¹ - φ₂·zᵖ⁻² - ... - φₚ = 0

    This matches the convention in ``analyze_roots_boundary_validity.py``::

        ar_coeffs = np.array([1, -phi[0], -phi[1]])
        ar_roots  = np.roots(ar_coeffs)

    **Stationarity condition in this form**: all |roots| < 1.
    For AR(1) with φ=0.5: root=0.5, |root|=0.5 < 1 → stationary.
    For AR(1) with φ=1.2: root=1.2, |root|=1.2 > 1 → non-stationary.

    Parameters
    ----------
    phi:
        AR coefficients [φ₁, φ₂, ..., φₚ].  Length-0 input returns an empty
        array (trivially stationary AR(0)).

    Returns
    -------
    roots:
        Complex-valued array of length ``p``.
    """
    phi = np.asarray(phi, dtype=float)
    if phi.size == 0:
        return np.array([], dtype=complex)
    coeffs = np.concatenate([[1.0], -phi])
    return np.roots(coeffs)


def compute_ma_roots(
    theta: Union[Sequence[float], np.ndarray],
) -> np.ndarray:
    """Compute roots of the MA companion polynomial.

    The companion polynomial passed to ``np.roots`` is::

        [1, θ₁, θ₂, ..., θ_q]

    which matches ``analyze_roots_boundary_validity.py``::

        ma_coeffs = np.array([1, theta[0], theta[1], theta[2]])
        ma_roots  = np.roots(ma_coeffs)

    **Invertibility condition in this form**: all |roots| < 1.
    For MA(1) with θ=0.5: root=−0.5, |root|=0.5 < 1 → invertible.
    For MA(1) with θ=1.2: root=−1.2, |root|=1.2 > 1 → non-invertible.

    Parameters
    ----------
    theta:
        MA coefficients [θ₁, θ₂, ..., θ_q].  Length-0 input returns an empty
        array (trivially invertible MA(0)).

    Returns
    -------
    roots:
        Complex-valued array of length ``q``.
    """
    theta = np.asarray(theta, dtype=float)
    if theta.size == 0:
        return np.array([], dtype=complex)
    coeffs = np.concatenate([[1.0], theta])
    return np.roots(coeffs)


# ---------------------------------------------------------------------------
# AR stationarity
# ---------------------------------------------------------------------------


def is_stationary_ar(
    phi: Union[Sequence[float], np.ndarray],
    tol: float = 1e-8,
) -> bool:
    """Return ``True`` iff the AR(p) process is covariance-stationary.

    Uses the **companion polynomial** convention (matching the project):
    stationarity requires all roots of ``np.roots([1, -φ₁, ..., -φₚ])``
    lie **strictly inside** the unit circle, i.e. ``|root| < 1``.

    This is equivalent to the standard stationarity condition:
    for AR(1), stationarity ⇔ |φ| < 1.

    An AR(0) process (empty ``phi``) is trivially stationary.

    Parameters
    ----------
    phi:
        AR coefficients [φ₁, ..., φₚ].
    tol:
        Numerical tolerance.  A root is stationary iff ``|root| < 1 - tol``.
        Default ``1e-8`` treats boundary roots (unit-root processes) as
        non-stationary.

    Returns
    -------
    bool
    """
    roots = compute_ar_roots(phi)
    if roots.size == 0:
        return True
    return bool(np.all(np.abs(roots) < 1.0 - tol))


# ---------------------------------------------------------------------------
# MA invertibility
# ---------------------------------------------------------------------------


def is_invertible_ma(
    theta: Union[Sequence[float], np.ndarray],
    tol: float = 1e-8,
) -> bool:
    """Return ``True`` iff the MA(q) process is invertible.

    Uses the **companion polynomial** convention (matching the project):
    invertibility requires all roots of ``np.roots([1, θ₁, ..., θ_q])``
    lie **strictly inside** the unit circle, i.e. ``|root| < 1``.

    This is equivalent to the standard invertibility condition:
    for MA(1), invertibility ⇔ |θ| < 1.

    An MA(0) process (empty ``theta``) is trivially invertible.

    Parameters
    ----------
    theta:
        MA coefficients [θ₁, ..., θ_q].
    tol:
        Numerical tolerance; same semantics as :func:`is_stationary_ar`.

    Returns
    -------
    bool
    """
    roots = compute_ma_roots(theta)
    if roots.size == 0:
        return True
    return bool(np.all(np.abs(roots) < 1.0 - tol))


# ---------------------------------------------------------------------------
# GARCH validity
# ---------------------------------------------------------------------------


def is_valid_garch(
    omega: float,
    alpha: Union[Sequence[float], np.ndarray],
    beta: Union[Sequence[float], np.ndarray],
    tol: float = 1e-8,
) -> bool:
    """Return ``True`` iff the GARCH(r,s) parameters satisfy all constraints.

    Constraints (verified from ``Симуляція.py::validate_theta_advanced``):

    1. ω > 0
    2. αᵢ >= 0  for all i
    3. βⱼ >= 0  for all j
    4. Σα + Σβ < 1  (strict inequality; ``>= 1`` is invalid)

    Parameters
    ----------
    omega:
        GARCH intercept (unconditional variance contribution).
    alpha:
        ARCH coefficients [α₁, ..., α_r].  May be empty for GARCH(0,s).
    beta:
        GARCH coefficients [β₁, ..., β_s].  May be empty for GARCH(r,0).
    tol:
        Numerical tolerance for the sign checks.  ``omega`` must be
        ``> tol``, coefficients must be ``>= -tol``.

    Returns
    -------
    bool
    """
    alpha = np.asarray(alpha, dtype=float)
    beta  = np.asarray(beta,  dtype=float)

    if omega <= tol:
        return False
    if alpha.size > 0 and np.any(alpha < -tol):
        return False
    if beta.size > 0 and np.any(beta < -tol):
        return False
    persistence = float(np.sum(alpha)) + float(np.sum(beta))
    if persistence >= 1.0:
        return False
    return True


# ---------------------------------------------------------------------------
# Structure decoding
# ---------------------------------------------------------------------------


def decode_discrete_structure(
    vector: Union[np.ndarray, Sequence[float]],
    threshold: float = 0.5,
    epsilon: float = 0.05,
) -> ModelStructure:
    """Decode the structural sub-vector of a VAE-decoded hypothesis.

    The structural sub-vector ``v`` (length ≥ 6, tanh-normalised to [-1, 1])
    encodes:

        v[0] → ARMA activation logit
        v[1] → GARCH activation logit
        v[2] → p_order logit
        v[3] → q_order logit
        v[4] → r_order logit
        v[5] → s_order logit

    The logits are first mapped from [-1, 1] to [0, 1]:

        score = (logit + 1) / 2

    Type detection logic (matches Симуляція_2.py, the most recent version):

        strong_arma  = arma_score > threshold + epsilon
        strong_garch = garch_score > threshold + epsilon
        weak_arma    = arma_score > threshold
        weak_garch   = garch_score > threshold

        ARMA-GARCH: strong_arma AND strong_garch
        ARMA:       strong_arma AND NOT weak_garch
        GARCH:      strong_garch AND NOT weak_arma
        ARMA-GARCH: weak_arma AND weak_garch  (both weakly active)
        ARMA:       weak_arma only
        GARCH:      weak_garch only
        UNKNOWN:    neither

    Order decoding:

        order = round((score + 1) / 2 * MAX_LAG_ORDER)

    but since the logit is already a scalar in [-1, 1], the formula is:

        order = round(((logit + 1) / 2) * MAX_LAG_ORDER)

    Parameters
    ----------
    vector:
        Structural sub-vector of length ≥ 6 with values in [-1, 1].
        If the full 40-dimensional hypothesis vector is passed, only the
        first :data:`D_STRUCT` elements are used.
    threshold:
        Decision boundary for type activation.  Default: 0.5.
    epsilon:
        Buffer zone around the threshold for "strong" activation.
        Default: 0.05.

    Returns
    -------
    ModelStructure
    """
    v = np.asarray(vector, dtype=float)
    # If full 40-dim vector is passed, take only struct portion
    if v.ndim != 1:
        raise ValueError(f"vector must be 1-D, got shape {v.shape}")

    # Default fallback when vector is too short
    if len(v) < 2:
        return ModelStructure(model_type="UNKNOWN", p=0, q=0, r=0, s=0)

    # --- Type logits → scores -----------------------------------------------
    arma_score  = float((v[_IDX_ARMA_LOGIT]  + 1.0) / 2.0)
    garch_score = float((v[_IDX_GARCH_LOGIT] + 1.0) / 2.0) if len(v) > 1 else 0.0

    strong_arma  = arma_score  > threshold + epsilon
    strong_garch = garch_score > threshold + epsilon
    weak_arma    = arma_score  > threshold
    weak_garch   = garch_score > threshold

    if strong_arma and strong_garch:
        model_type = "ARMA-GARCH"
    elif strong_arma and not weak_garch:
        model_type = "ARMA"
    elif strong_garch and not weak_arma:
        model_type = "GARCH"
    elif weak_arma and weak_garch:
        model_type = "ARMA-GARCH"
    elif weak_arma:
        model_type = "ARMA"
    elif weak_garch:
        model_type = "GARCH"
    else:
        model_type = "UNKNOWN"

    # --- Order logits → integer orders --------------------------------------
    def _decode_order(idx: int) -> int:
        if idx < len(v):
            return decode_order_logit(float(v[idx]), max_lag_order=MAX_LAG_ORDER)
        return 0

    p = _decode_order(_IDX_P_LOGIT)
    q = _decode_order(_IDX_Q_LOGIT)
    r = _decode_order(_IDX_R_LOGIT)
    s = _decode_order(_IDX_S_LOGIT)

    return ModelStructure(model_type=model_type, p=p, q=q, r=r, s=s)


# ---------------------------------------------------------------------------
# Combined ARMA-GARCH validation
# ---------------------------------------------------------------------------


def validate_arma_garch(
    phi:   Union[Sequence[float], np.ndarray, None] = None,
    theta: Union[Sequence[float], np.ndarray, None] = None,
    omega: Union[float, None] = None,
    alpha: Union[Sequence[float], np.ndarray, None] = None,
    beta:  Union[Sequence[float], np.ndarray, None] = None,
) -> ValidationResult:
    """Validate ARMA and/or GARCH parameters.

    Each argument is optional; omitting it (or passing ``None``) means the
    corresponding component is absent (orders = 0).

    Parameters
    ----------
    phi:
        AR coefficients.  ``None`` or empty → AR(0), trivially stationary.
    theta:
        MA coefficients.  ``None`` or empty → MA(0), trivially invertible.
    omega:
        GARCH intercept.  Required when ``alpha`` or ``beta`` is non-empty.
    alpha:
        ARCH coefficients.  ``None`` or empty → ARCH(0).
    beta:
        GARCH coefficients.  ``None`` or empty → GARCH(0).

    Returns
    -------
    ValidationResult
    """
    phi   = np.asarray(phi,   dtype=float) if phi   is not None else np.array([])
    theta = np.asarray(theta, dtype=float) if theta is not None else np.array([])
    alpha = np.asarray(alpha, dtype=float) if alpha is not None else np.array([])
    beta  = np.asarray(beta,  dtype=float) if beta  is not None else np.array([])

    # ---- AR check ----------------------------------------------------------
    stat = is_stationary_ar(phi)
    if not stat:
        return ValidationResult(
            is_valid=False,
            is_stationary_ar=False,
            is_invertible_ma=True,  # not tested yet
            is_valid_garch=True,    # not tested yet
            reason=(
                f"AR companion roots outside unit circle (non-stationary). "
                f"|roots|={np.abs(compute_ar_roots(phi)).tolist()}"
            ),
        )

    # ---- MA check ----------------------------------------------------------
    inv = is_invertible_ma(theta)
    if not inv:
        return ValidationResult(
            is_valid=False,
            is_stationary_ar=True,
            is_invertible_ma=False,
            is_valid_garch=True,    # not tested yet
            reason=(
                f"MA companion roots outside unit circle (non-invertible). "
                f"|roots|={np.abs(compute_ma_roots(theta)).tolist()}"
            ),
        )

    # ---- GARCH check -------------------------------------------------------
    has_garch = alpha.size > 0 or beta.size > 0
    if has_garch:
        omega_val = float(omega) if omega is not None else 0.0
        garch_ok = is_valid_garch(omega_val, alpha, beta)
        if not garch_ok:
            return ValidationResult(
                is_valid=False,
                is_stationary_ar=True,
                is_invertible_ma=True,
                is_valid_garch=False,
                reason=(
                    f"GARCH constraints violated: "
                    f"omega={omega_val}, alpha={alpha.tolist()}, "
                    f"beta={beta.tolist()}, "
                    f"persistence={float(np.sum(alpha)) + float(np.sum(beta)):.4f}"
                ),
            )
    else:
        garch_ok = True

    return ValidationResult(
        is_valid=True,
        is_stationary_ar=True,
        is_invertible_ma=True,
        is_valid_garch=garch_ok,
        reason=None,
    )
