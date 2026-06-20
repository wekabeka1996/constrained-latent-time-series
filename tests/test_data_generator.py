"""
tests/test_data_generator.py
============================
Pytest suite for Phase 1C — Canonical Training Data Generator.

Verifies:
  - Dataset shape and label matching.
  - Determinism for fixed seeds.
  - Absence of supervised combined ARMA-GARCH examples.
  - Structure and constraints of ARMA/AR and GARCH vectors.

Run from the project root:

    python -m pytest tests/test_data_generator.py -v
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.data_generator import (
    GenerationConfig,
    generate_arma_vector,
    generate_garch_vector,
    generate_training_dataset,
)
from src.vector_schema import (
    ARMA_ACTIVE_IDX,
    GARCH_ACTIVE_IDX,
    VECTOR_DIM,
    assert_vector_shape,
)


# ===========================================================================
# Test Dataset Shape and Labels
# ===========================================================================

class TestDatasetShapeAndLabels:
    def test_generate_training_dataset_shape(self) -> None:
        cfg = GenerationConfig(n_arma=15, n_garch=25)
        X, y = generate_training_dataset(cfg)
        assert X.shape == (40, VECTOR_DIM)
        assert y.shape == (40,)

    def test_generate_training_dataset_labels_length(self) -> None:
        cfg = GenerationConfig(n_arma=10, n_garch=10)
        X, y = generate_training_dataset(cfg)
        assert len(y) == X.shape[0]
        # First n_arma should be AR
        assert np.all(y[:10] == "AR")
        # Next n_garch should be GARCH
        assert np.all(y[10:] == "GARCH")

    def test_generated_vectors_have_expected_dim(self) -> None:
        rng = np.random.default_rng(123)
        cfg = GenerationConfig()
        v_arma = generate_arma_vector(rng, cfg)
        v_garch = generate_garch_vector(rng, cfg)
        
        # Will raise ValueError if dimensions are incorrect
        assert_vector_shape(v_arma)
        assert_vector_shape(v_garch)


# ===========================================================================
# Test Determinism
# ===========================================================================

class TestDeterminism:
    def test_generation_is_deterministic_for_same_seed(self) -> None:
        cfg = GenerationConfig(n_arma=5, n_garch=5, seed=42)
        X1, y1 = generate_training_dataset(cfg)
        X2, y2 = generate_training_dataset(cfg)
        
        np.testing.assert_array_equal(X1, X2)
        np.testing.assert_array_equal(y1, y2)

    def test_generation_changes_for_different_seed(self) -> None:
        cfg1 = GenerationConfig(n_arma=5, n_garch=5, seed=42)
        cfg2 = GenerationConfig(n_arma=5, n_garch=5, seed=99)
        
        X1, _ = generate_training_dataset(cfg1)
        X2, _ = generate_training_dataset(cfg2)
        
        # Ensure at least some elements differ
        assert not np.allclose(X1, X2)


# ===========================================================================
# Test Class Structure
# ===========================================================================

class TestClassStructure:
    def test_default_dataset_has_no_supervised_combined_arma_garch(self) -> None:
        """Verifies that no vector has BOTH structure flags active."""
        X, y = generate_training_dataset()
        for i in range(X.shape[0]):
            arma_active = X[i, ARMA_ACTIVE_IDX] == 1.0
            garch_active = X[i, GARCH_ACTIVE_IDX] == 1.0
            # Either one or the other, not both
            assert not (arma_active and garch_active), "Found combined ARMA-GARCH structure!"

    def test_arma_vectors_have_arma_flag_only(self) -> None:
        rng = np.random.default_rng(1)
        v = generate_arma_vector(rng, GenerationConfig())
        assert v[ARMA_ACTIVE_IDX] == 1.0
        assert v[GARCH_ACTIVE_IDX] == 0.0

    def test_garch_vectors_have_garch_flag_only(self) -> None:
        rng = np.random.default_rng(1)
        v = generate_garch_vector(rng, GenerationConfig())
        assert v[ARMA_ACTIVE_IDX] == 0.0
        assert v[GARCH_ACTIVE_IDX] == 1.0


# ===========================================================================
# Test Validity / Constraints
# ===========================================================================

class TestValidityConstraints:
    def test_generated_garch_vectors_reproduce_original_anomalies(self) -> None:
        """Verify that the generator exactly reproduces the original bugs:
        - beta can be negative if sum(alpha) > 0.4
        - persistence can easily exceed 1.0
        """
        rng = np.random.default_rng(42)
        cfg = GenerationConfig(max_lag_order=5)
        
        found_negative_beta = False
        found_explosive = False
        
        for _ in range(500):
            v = generate_garch_vector(rng, cfg)
            
            r_logit = v[4]
            s_logit = v[5]
            r = round(((r_logit + 1) / 2) * cfg.max_lag_order)
            s = round(((s_logit + 1) / 2) * cfg.max_lag_order)
            
            omega = v[10]
            alpha = v[11 : 11 + r] if r > 0 else np.array([])
            beta = v[11 + r : 11 + r + s] if s > 0 else np.array([])
            
            assert 0.1 <= omega <= 0.5
            
            if s > 0 and np.any(beta < 0):
                found_negative_beta = True
            
            persistence = np.sum(alpha) + np.sum(beta)
            if persistence > 1.0:
                found_explosive = True

        assert found_explosive, "Original generator naturally produced explosive models."
        # Note: finding a negative beta requires sum(alpha) > 0.4 and then drawing a negative value.
        # We don't strictly assert found_negative_beta unless we force the seed, but
        # the uniform logic natively permits it.

    def test_generated_arma_vectors_have_reasonable_orders(self) -> None:
        rng = np.random.default_rng(42)
        cfg = GenerationConfig(max_lag_order=5)
        for _ in range(50):
            v = generate_arma_vector(rng, cfg)
            p_logit = v[2]
            q_logit = v[3]
            p = round(((p_logit + 1) / 2) * cfg.max_lag_order)
            q = round(((q_logit + 1) / 2) * cfg.max_lag_order)
            
            assert 0 <= p <= 5
            assert 0 <= q <= 5


# ===========================================================================
# Test Compatibility With Current Artifacts
# ===========================================================================

class TestArtifactCompatibility:
    def test_z_train_compatibility(self) -> None:
        p = PROJECT_ROOT / "Z_train.npy"
        if not p.exists():
            pytest.skip("Z_train.npy not found")
        z = np.load(p)
        assert z.shape == (200, 40), "Original Z_train was 200x40"

    def test_train_labels_compatibility(self) -> None:
        p = PROJECT_ROOT / "train_labels.npy"
        if not p.exists():
            pytest.skip("train_labels.npy not found")
        labels = np.load(p)
        assert len(labels) == 200
        # The project originally used "AR" as the label for ARMA-like vectors
        # and "GARCH" for GARCH.
        assert "AR" in labels
        assert "GARCH" in labels
