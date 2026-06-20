"""
tests/test_vector_schema.py
===========================
Pytest suite for Phase 1B.5 — validates src/vector_schema.py and
checks existing artifact shapes.

Run from the project root:

    python -m pytest tests/test_vector_schema.py -v
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.vector_schema import (
    D_PARAM,
    D_STAT,
    D_STRUCT,
    MAX_LAG_ORDER,
    PARAM_SLICE,
    STAT_SLICE,
    STRUCT_SLICE,
    VECTOR_DIM,
    assert_vector_shape,
    decode_order_logit,
    describe_layout,
    split_vector,
)


class TestVectorConstants:
    """Verify that dimension constants match the established contract."""

    def test_vector_dim_is_40(self) -> None:
        assert VECTOR_DIM == 40
        assert D_STRUCT == 10
        assert D_PARAM == 20
        assert D_STAT == 10
        assert D_STRUCT + D_PARAM + D_STAT == VECTOR_DIM

    def test_slices_cover_exact_ranges(self) -> None:
        assert STRUCT_SLICE == slice(0, 10)
        assert PARAM_SLICE == slice(10, 30)
        assert STAT_SLICE == slice(30, 40)


class TestSplitVector:
    """Verify vector shape assertions and splitting logic."""

    def test_assert_vector_shape_valid_1d(self) -> None:
        v = np.zeros(VECTOR_DIM)
        # Should not raise
        assert_vector_shape(v)

    def test_assert_vector_shape_valid_2d(self) -> None:
        v = np.zeros((5, VECTOR_DIM))
        # Should not raise
        assert_vector_shape(v)

    def test_assert_vector_shape_invalid_length(self) -> None:
        v = np.zeros(VECTOR_DIM - 1)
        with pytest.raises(ValueError, match="Expected 1D vector"):
            assert_vector_shape(v)

    def test_assert_vector_shape_invalid_2d_dim(self) -> None:
        v = np.zeros((5, VECTOR_DIM + 1))
        with pytest.raises(ValueError, match="Expected 2D array"):
            assert_vector_shape(v)

    def test_split_vector_1d(self) -> None:
        v = np.arange(VECTOR_DIM)
        struct, param, stat = split_vector(v)
        assert struct.shape == (D_STRUCT,)
        assert param.shape == (D_PARAM,)
        assert stat.shape == (D_STAT,)
        assert struct[0] == 0
        assert param[0] == 10
        assert stat[0] == 30

    def test_split_vector_2d(self) -> None:
        v = np.zeros((3, VECTOR_DIM))
        v[:, 10] = 5.0
        struct, param, stat = split_vector(v)
        assert struct.shape == (3, D_STRUCT)
        assert param.shape == (3, D_PARAM)
        assert stat.shape == (3, D_STAT)
        assert np.all(param[:, 0] == 5.0)


class TestDecodeOrderLogit:
    """Verify logit decoding formula matches old Симуляція.py logic."""

    def test_decode_order_logit_minus_one(self) -> None:
        """logit=-1.0 → score=0.0 → order=0"""
        assert decode_order_logit(-1.0) == 0

    def test_decode_order_logit_plus_one(self) -> None:
        """logit=1.0 → score=1.0 → order=MAX_LAG_ORDER"""
        assert decode_order_logit(1.0) == MAX_LAG_ORDER

    def test_decode_order_logit_zero(self) -> None:
        """logit=0.0 → score=0.5 → order=round(2.5)=2 or 3 (banker's rounding)"""
        order = decode_order_logit(0.0)
        assert order in (2, 3)

    def test_decode_order_logit_custom_max(self) -> None:
        """Should scale properly with a custom max order."""
        assert decode_order_logit(1.0, max_lag_order=10) == 10


class TestDescribeLayout:
    """Verify the helper returns a valid dict with ambiguity warnings."""

    def test_describe_layout_contains_warning(self) -> None:
        info = describe_layout()
        assert info["VECTOR_DIM"] == 40
        assert "NOTE" in info
        assert "overlap" in str(info["NOTE"]).lower()


class TestArtifactShapes:
    """Verify shapes of existing artifacts, if present on disk."""

    def test_z_train_shape(self) -> None:
        p = PROJECT_ROOT / "Z_train.npy"
        if not p.exists():
            pytest.skip("Z_train.npy not found")
        z = np.load(p)
        assert z.ndim == 2, "Z_train should be a 2D array"
        assert z.shape[1] == VECTOR_DIM, f"Z_train should have {VECTOR_DIM} columns"

    def test_generated_valid_thetas_shape(self) -> None:
        p = PROJECT_ROOT / "generated_valid_thetas.npy"
        if not p.exists():
            pytest.skip("generated_valid_thetas.npy not found")
        z = np.load(p)
        assert z.ndim == 2, "generated_valid_thetas should be a 2D array"
        assert z.shape[1] == VECTOR_DIM, f"generated_valid_thetas should have {VECTOR_DIM} columns"
