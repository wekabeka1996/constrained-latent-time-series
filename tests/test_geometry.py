import numpy as np
import pytest
from src.geometry import (
    numerical_jacobian,
    fisher_metric_from_jacobian,
    symmetrize_metric,
    regularize_metric_svd,
    condition_number,
    path_length_euclidean,
    path_length_riemannian
)
from src.vae import VAE

def test_numerical_jacobian_linear_function():
    # f(z) = A @ z
    A = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
    def f(z):
        return A @ z
    
    z = np.array([0.5, -0.5])
    J = numerical_jacobian(f, z)
    
    np.testing.assert_allclose(J, A, atol=1e-10)

def test_numerical_jacobian_quadratic_function():
    # f(x, y) = [x^2, x*y]
    # J = [[2x, 0], [y, x]]
    def f(z):
        x, y = z
        return np.array([x**2, x*y])
        
    z = np.array([2.0, 3.0])
    expected_J = np.array([
        [4.0, 0.0],
        [3.0, 2.0]
    ])
    
    J = numerical_jacobian(f, z)
    np.testing.assert_allclose(J, expected_J, atol=1e-6)

def test_fisher_metric_from_jacobian_shape():
    J = np.random.randn(10, 3)
    G = fisher_metric_from_jacobian(J)
    assert G.shape == (3, 3)

def test_fisher_metric_is_symmetric():
    J = np.random.randn(10, 3)
    G = fisher_metric_from_jacobian(J)
    np.testing.assert_allclose(G, G.T, atol=1e-12)

def test_fisher_metric_is_positive_semidefinite():
    J = np.random.randn(10, 3)
    G = fisher_metric_from_jacobian(J)
    eigenvalues = np.linalg.eigvalsh(G)
    assert np.all(eigenvalues >= -1e-12)

def test_regularization_reduces_condition_number():
    # Create ill-conditioned matrix
    eigvals = np.array([100.0, 1.0, 1e-10])
    Q, _ = np.linalg.qr(np.random.randn(3, 3))
    G = Q @ np.diag(eigvals) @ Q.T
    
    cond_before = condition_number(G)
    res = regularize_metric_svd(G, min_singular_value=1e-4)
    cond_after = condition_number(res.metric)
    
    assert res.condition_number_before > 1e10 or res.condition_number_before == np.inf
    assert res.condition_number_after < 1e7
    assert cond_after < cond_before

def test_regularized_metric_is_positive_definite():
    G = np.zeros((3, 3))
    res = regularize_metric_svd(G, min_singular_value=1e-4)
    eigvals = np.linalg.eigvalsh(res.metric)
    assert np.all(eigvals >= 1e-4 - 1e-12)

def test_regularization_preserves_shape():
    G = np.eye(4)
    res = regularize_metric_svd(G)
    assert res.metric.shape == (4, 4)

def test_path_length_euclidean_straight_line():
    path = np.array([
        [0.0, 0.0],
        [3.0, 4.0]
    ])
    assert path_length_euclidean(path) == 5.0

def test_path_length_riemannian_identity_metric_matches_euclidean():
    path = np.array([
        [0.0, 0.0],
        [1.0, 1.0],
        [2.0, 3.0]
    ])
    def identity_metric(z):
        return np.eye(2)
        
    length_euc = path_length_euclidean(path)
    length_rie = path_length_riemannian(path, identity_metric)
    assert np.isclose(length_euc, length_rie)

def test_path_length_riemannian_rejects_bad_metric_shape():
    path = np.array([[0.0, 0.0], [1.0, 1.0]])
    def bad_shape_metric(z):
        return np.eye(3) # Should be 2x2
    
    with pytest.raises(ValueError, match="Metric shape"):
        path_length_riemannian(path, bad_shape_metric)

def test_path_length_riemannian_does_not_silently_fallback():
    path = np.array([[0.0, 0.0], [1.0, 1.0]])
    def failing_metric(z):
        raise RuntimeError("Metric failed")
    
    with pytest.raises(RuntimeError, match="Metric failed"):
        path_length_riemannian(path, failing_metric)

def test_vae_smoke_test():
    # Instantiate baseline VAE
    vae = VAE()
    
    # Define numpy wrapper for decoder
    import torch
    def decoder_wrapper(z_np):
        with torch.no_grad():
            z_tensor = torch.tensor(z_np, dtype=torch.float32).unsqueeze(0)
            theta_tensor = vae.decoder(z_tensor)
            return theta_tensor.squeeze(0).numpy().astype(np.float64)
            
    z_zero = np.zeros(8)
    J = numerical_jacobian(decoder_wrapper, z_zero)
    
    assert J.shape == (40, 8)
    
    G = fisher_metric_from_jacobian(J)
    assert G.shape == (8, 8)
    assert np.all(np.isfinite(G))
    
    # Check PSD
    eigvals = np.linalg.eigvalsh(G)
    assert np.all(eigvals >= -1e-6)
