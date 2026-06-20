"""
tests/test_validation.py
========================
Pytest suite for Phase 1B — validates src/validation.py.

Tests cover:
  * AR(1) stationarity (pass / fail)
  * MA(1) invertibility (pass / fail)
  * GARCH(1,1) constraints (pass / fail)
  * Combined validate_arma_garch (pass / fail)
  * decode_discrete_structure with hand-crafted structural vectors

Run from the project root:

    python -m pytest tests/test_validation.py -v
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.validation import (
    MAX_LAG_ORDER,
    ModelStructure,
    ValidationResult,
    compute_ar_roots,
    compute_ma_roots,
    decode_discrete_structure,
    is_invertible_ma,
    is_stationary_ar,
    is_valid_garch,
    validate_arma_garch,
)


# ===========================================================================
# TestARRoots
# ===========================================================================


class TestARRoots:
    """compute_ar_roots returns roots of the companion polynomial [1, -φ₁, ..., -φₚ].

    Convention: np.roots([1, -φ]) finds z s.t. z - φ = 0 → z = φ.
    Stationarity: |root| < 1.
    """

    def test_ar1_root_formula(self) -> None:
        """AR(1) with φ=0.5: np.roots([1, -0.5]) → root=0.5 (equals φ)."""
        roots = compute_ar_roots([0.5])
        assert len(roots) == 1
        assert abs(roots[0].real - 0.5) < 1e-10

    def test_ar1_unit_root(self) -> None:
        """AR(1) with φ=1.0 → root at z=1.0 (unit root, boundary)."""
        roots = compute_ar_roots([1.0])
        assert len(roots) == 1
        assert abs(abs(roots[0]) - 1.0) < 1e-10

    def test_empty_phi_returns_empty(self) -> None:
        roots = compute_ar_roots([])
        assert len(roots) == 0

    def test_ar2_num_roots(self) -> None:
        roots = compute_ar_roots([0.3, 0.2])
        assert len(roots) == 2


# ===========================================================================
# TestMAroots
# ===========================================================================


class TestMARoots:
    """compute_ma_roots returns roots of the companion polynomial [1, θ₁, ..., θ_q].

    Convention: np.roots([1, θ]) finds z s.t. z + θ = 0 → z = -θ.
    Invertibility: |root| < 1.
    """

    def test_ma1_root_formula(self) -> None:
        """MA(1) with θ=0.5: np.roots([1, 0.5]) → root=-0.5, |root|=0.5."""
        roots = compute_ma_roots([0.5])
        assert len(roots) == 1
        assert abs(roots[0].real - (-0.5)) < 1e-10
        assert abs(abs(roots[0]) - 0.5) < 1e-10

    def test_ma1_unit_root(self) -> None:
        """MA(1) with θ=1.0 → root=-1.0, |root|=1.0 (boundary)."""
        roots = compute_ma_roots([1.0])
        assert len(roots) == 1
        assert abs(abs(roots[0]) - 1.0) < 1e-10

    def test_empty_theta_returns_empty(self) -> None:
        roots = compute_ma_roots([])
        assert len(roots) == 0


# ===========================================================================
# TestIsStationaryAR
# ===========================================================================


class TestIsStationaryAR:
    """is_stationary_ar: companion roots must lie strictly INSIDE the unit circle."""

    # ---- required by spec: AR(1) examples ----------------------------------

    def test_ar1_stationary_phi_inside_unit_interval(self) -> None:
        """AR(1) φ=0.5 → companion root=0.5, |root|=0.5 < 1 → stationary."""
        assert is_stationary_ar([0.5]) is True

    def test_ar1_nonstationary_phi_above_one(self) -> None:
        """AR(1) φ=1.2 → companion root=1.2, |root|=1.2 > 1 → non-stationary."""
        assert is_stationary_ar([1.2]) is False

    # ---- additional boundary / negative tests -------------------------------

    def test_ar1_negative_stationary(self) -> None:
        """AR(1) φ=-0.7 → companion root=-0.7, |root|=0.7 < 1 → stationary."""
        assert is_stationary_ar([-0.7]) is True

    def test_ar1_unit_root_boundary(self) -> None:
        """AR(1) φ=1.0 → companion root=1.0, |root|=1.0 → non-stationary (boundary rejected)."""
        assert is_stationary_ar([1.0]) is False

    def test_ar2_stationary(self) -> None:
        """AR(2) φ=[0.3, 0.2]: companion roots both have |root| < 1 → stationary."""
        assert is_stationary_ar([0.3, 0.2]) is True

    def test_ar2_explosive(self) -> None:
        """AR(2) φ=[1.5, 0.5]: companion root has |root| > 1 → non-stationary."""
        assert is_stationary_ar([1.5, 0.5]) is False

    def test_empty_phi_is_trivially_stationary(self) -> None:
        """AR(0): no coefficients → trivially stationary."""
        assert is_stationary_ar([]) is True


# ===========================================================================
# TestIsInvertibleMA
# ===========================================================================


class TestIsInvertibleMA:
    """is_invertible_ma: MA companion roots must lie strictly INSIDE the unit circle."""

    # ---- required by spec: MA(1) examples ----------------------------------

    def test_ma1_invertible_theta_inside_unit_interval(self) -> None:
        """MA(1) θ=0.5 → companion root=-0.5, |root|=0.5 < 1 → invertible."""
        assert is_invertible_ma([0.5]) is True

    def test_ma1_noninvertible_theta_above_one(self) -> None:
        """MA(1) θ=1.2 → companion root=-1.2, |root|=1.2 > 1 → non-invertible."""
        assert is_invertible_ma([1.2]) is False

    # ---- additional tests ---------------------------------------------------

    def test_ma1_unit_root_boundary(self) -> None:
        """MA(1) θ=1.0 → companion root=-1.0, |root|=1.0 → non-invertible (boundary rejected)."""
        assert is_invertible_ma([1.0]) is False

    def test_ma1_negative_invertible(self) -> None:
        """MA(1) θ=-0.5 → companion root=0.5, |root|=0.5 < 1 → invertible."""
        assert is_invertible_ma([-0.5]) is True

    def test_empty_theta_is_trivially_invertible(self) -> None:
        """MA(0): no coefficients → trivially invertible."""
        assert is_invertible_ma([]) is True


# ===========================================================================
# TestIsValidGARCH
# ===========================================================================


class TestIsValidGARCH:
    """is_valid_garch: ω>0, all α≥0, all β≥0, Σα+Σβ<1."""

    # ---- required by spec ---------------------------------------------------

    def test_valid_garch_11(self) -> None:
        """Standard GARCH(1,1): ω=0.1, α=0.1, β=0.8 → persistence=0.9 < 1."""
        assert is_valid_garch(omega=0.1, alpha=[0.1], beta=[0.8]) is True

    def test_invalid_garch_negative_omega(self) -> None:
        """ω ≤ 0 is forbidden."""
        assert is_valid_garch(omega=-0.01, alpha=[0.1], beta=[0.8]) is False

    def test_invalid_garch_zero_omega(self) -> None:
        """ω = 0 is also forbidden (variance must have positive intercept)."""
        assert is_valid_garch(omega=0.0, alpha=[0.1], beta=[0.8]) is False

    def test_invalid_garch_negative_alpha(self) -> None:
        """Negative α violates non-negativity constraint."""
        assert is_valid_garch(omega=0.1, alpha=[-0.05], beta=[0.8]) is False

    def test_invalid_garch_persistence_ge_one(self) -> None:
        """Σα + Σβ ≥ 1 violates covariance-stationarity of GARCH."""
        assert is_valid_garch(omega=0.1, alpha=[0.3], beta=[0.7]) is False

    # ---- additional tests ---------------------------------------------------

    def test_invalid_garch_negative_beta(self) -> None:
        """Negative β violates non-negativity constraint."""
        assert is_valid_garch(omega=0.05, alpha=[0.1], beta=[-0.1]) is False

    def test_valid_garch_no_beta(self) -> None:
        """ARCH(1): β empty, ω>0, α=0.5 < 1 → valid."""
        assert is_valid_garch(omega=0.1, alpha=[0.5], beta=[]) is True

    def test_valid_garch_no_alpha(self) -> None:
        """GARCH(0,1): α empty, ω>0, β=0.9 < 1 → valid."""
        assert is_valid_garch(omega=0.1, alpha=[], beta=[0.9]) is True

    def test_invalid_garch_exact_unit_persistence(self) -> None:
        """Σα + Σβ = 1.0 exactly → invalid (strict inequality)."""
        assert is_valid_garch(omega=0.1, alpha=[0.5], beta=[0.5]) is False


# ===========================================================================
# TestValidateArmaGarch (combined)
# ===========================================================================


class TestValidateArmaGarch:
    """validate_arma_garch returns a ValidationResult dataclass."""

    # ---- required by spec ---------------------------------------------------

    def test_validate_arma_garch_valid_case(self) -> None:
        """Stationary AR(1) + invertible MA(1) + valid GARCH(1,1)."""
        result = validate_arma_garch(
            phi=[0.5],
            theta=[0.3],
            omega=0.1,
            alpha=[0.1],
            beta=[0.8],
        )
        assert isinstance(result, ValidationResult)
        assert result.is_valid is True
        assert result.is_stationary_ar is True
        assert result.is_invertible_ma is True
        assert result.is_valid_garch is True
        assert result.reason is None

    def test_validate_arma_garch_invalid_ar_case(self) -> None:
        """Non-stationary AR(1) makes the whole validation fail."""
        result = validate_arma_garch(
            phi=[1.2],
            theta=[0.3],
            omega=0.1,
            alpha=[0.1],
            beta=[0.8],
        )
        assert result.is_valid is False
        assert result.is_stationary_ar is False
        assert result.reason is not None

    def test_validate_arma_garch_invalid_garch_case(self) -> None:
        """Valid AR + valid MA but GARCH persistence ≥ 1 → invalid."""
        result = validate_arma_garch(
            phi=[0.5],
            theta=[0.3],
            omega=0.1,
            alpha=[0.5],
            beta=[0.5],
        )
        assert result.is_valid is False
        assert result.is_valid_garch is False
        assert result.reason is not None

    # ---- additional tests ---------------------------------------------------

    def test_validate_arma_only_no_garch(self) -> None:
        """Pure ARMA (no GARCH component) → valid when AR stationary."""
        result = validate_arma_garch(phi=[0.3], theta=[0.2])
        assert result.is_valid is True
        assert result.is_valid_garch is True  # trivially, no GARCH

    def test_validate_nonstationary_ar_no_garch(self) -> None:
        """Non-stationary AR with no GARCH → invalid."""
        result = validate_arma_garch(phi=[1.5])
        assert result.is_valid is False
        assert result.is_stationary_ar is False

    def test_validate_noninvertible_ma(self) -> None:
        """Stationary AR + non-invertible MA → invalid."""
        result = validate_arma_garch(phi=[0.3], theta=[1.5])
        assert result.is_valid is False
        assert result.is_invertible_ma is False

    def test_validate_null_case_all_none(self) -> None:
        """All None → trivially valid (AR(0) + MA(0) + no GARCH)."""
        result = validate_arma_garch()
        assert result.is_valid is True


# ===========================================================================
# TestDecodeDiscreteStructure
# ===========================================================================


class TestDecodeDiscreteStructure:
    """decode_discrete_structure decodes a structural sub-vector."""

    def _make_struct_vector(
        self,
        arma_raw: float,
        garch_raw: float,
        p_raw: float = -1.0,
        q_raw: float = -1.0,
        r_raw: float = -1.0,
        s_raw: float = -1.0,
        length: int = 10,
    ) -> np.ndarray:
        """Build a synthetic structural vector matching the known layout.

        Values are raw tanh outputs ∈ [-1, 1].
        """
        v = np.full(length, -1.0)   # all inactive by default
        v[0] = arma_raw   # ARMA logit
        v[1] = garch_raw  # GARCH logit
        v[2] = p_raw      # p_order logit
        v[3] = q_raw      # q_order logit
        v[4] = r_raw      # r_order logit
        v[5] = s_raw      # s_order logit
        return v

    # ---- Required spec tests ------------------------------------------------

    def test_strong_arma_signal_gives_arma_type(self) -> None:
        """ARMA logit >> 0.5 threshold with GARCH logit << 0.5 → type=ARMA."""
        # arma_raw=1.0 → score=1.0 (strong); garch_raw=-1.0 → score=0.0 (inactive)
        v = self._make_struct_vector(arma_raw=1.0, garch_raw=-1.0)
        result = decode_discrete_structure(v)
        assert isinstance(result, ModelStructure)
        assert result.model_type == "ARMA"

    def test_strong_garch_signal_gives_garch_type(self) -> None:
        """GARCH logit >> 0.5 threshold with ARMA logit << 0.5 → type=GARCH."""
        # garch_raw=1.0 → score=1.0 (strong); arma_raw=-1.0 → score=0.0 (inactive)
        v = self._make_struct_vector(arma_raw=-1.0, garch_raw=1.0)
        result = decode_discrete_structure(v)
        assert result.model_type == "GARCH"

    def test_both_strong_gives_arma_garch_type(self) -> None:
        """Both signals strong → ARMA-GARCH."""
        v = self._make_struct_vector(arma_raw=1.0, garch_raw=1.0)
        result = decode_discrete_structure(v)
        assert result.model_type == "ARMA-GARCH"

    def test_neither_signal_gives_unknown(self) -> None:
        """Both signals at 0 (score=0.5, weak) → UNKNOWN."""
        # score = (0 + 1) / 2 = 0.5; weak_arma = 0.5 > 0.5 → False
        v = self._make_struct_vector(arma_raw=0.0, garch_raw=0.0)
        result = decode_discrete_structure(v)
        assert result.model_type == "UNKNOWN"

    # ---- Order decoding tests -----------------------------------------------

    def test_order_decoding_max(self) -> None:
        """logit=1.0 → score=1.0 → order=MAX_LAG_ORDER."""
        v = self._make_struct_vector(
            arma_raw=1.0, garch_raw=-1.0,
            p_raw=1.0, q_raw=1.0,
        )
        result = decode_discrete_structure(v)
        assert result.p == MAX_LAG_ORDER
        assert result.q == MAX_LAG_ORDER

    def test_order_decoding_zero(self) -> None:
        """logit=-1.0 → score=0.0 → order=0."""
        v = self._make_struct_vector(
            arma_raw=1.0, garch_raw=-1.0,
            p_raw=-1.0, q_raw=-1.0,
        )
        result = decode_discrete_structure(v)
        assert result.p == 0
        assert result.q == 0

    def test_order_decoding_midpoint(self) -> None:
        """logit=0.0 → score=0.5 → order=round(0.5*5)=round(2.5)=2 or 3.

        Python's built-in round() uses banker's rounding (round half to even),
        so round(2.5) = 2.  We accept both 2 and 3 to avoid hardcoding a
        rounding-mode dependency.
        """
        v = self._make_struct_vector(
            arma_raw=1.0, garch_raw=-1.0,
            p_raw=0.0, q_raw=0.0,
        )
        result = decode_discrete_structure(v)
        assert result.p in (2, 3)
        assert result.q in (2, 3)

    def test_returns_model_structure_dataclass(self) -> None:
        v = self._make_struct_vector(arma_raw=1.0, garch_raw=-1.0)
        result = decode_discrete_structure(v)
        assert isinstance(result, ModelStructure)
        assert hasattr(result, "model_type")
        assert hasattr(result, "p")
        assert hasattr(result, "q")
        assert hasattr(result, "r")
        assert hasattr(result, "s")

    def test_too_short_vector_returns_unknown(self) -> None:
        """A vector shorter than 2 elements → UNKNOWN, all orders 0."""
        v = np.array([1.0])
        result = decode_discrete_structure(v)
        assert result.model_type == "UNKNOWN"
        assert result.p == 0

    def test_full_40d_vector_uses_first_slice(self) -> None:
        """Passing the full 40-dim vector should use v[0:6] for structure."""
        v = np.zeros(40)
        v[0] = 1.0    # strong ARMA
        v[1] = -1.0   # no GARCH
        result = decode_discrete_structure(v)
        assert result.model_type == "ARMA"
