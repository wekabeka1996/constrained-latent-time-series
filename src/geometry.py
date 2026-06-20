from dataclasses import dataclass
from typing import Callable
import numpy as np

@dataclass(frozen=True)
class MetricRegularizationResult:
    metric: np.ndarray
    singular_values: np.ndarray
    regularized_singular_values: np.ndarray
    condition_number_before: float
    condition_number_after: float

def numerical_jacobian(
    decoder: Callable[[np.ndarray], np.ndarray],
    z: np.ndarray,
    eps: float = 1e-4,
) -> np.ndarray:
    """
    Computes numerical Jacobian J = d(theta)/d(z) using central finite differences.
    """
    z = np.asarray(z, dtype=np.float64)
    d_z = z.shape[0]
    
    # Evaluate at center to find output dimension
    theta_center = decoder(z)
    d_theta = theta_center.shape[0]
    
    J = np.zeros((d_theta, d_z), dtype=np.float64)
    for j in range(d_z):
        e_j = np.zeros(d_z, dtype=np.float64)
        e_j[j] = 1.0
        
        theta_plus = decoder(z + eps * e_j)
        theta_minus = decoder(z - eps * e_j)
        
        J[:, j] = (theta_plus - theta_minus) / (2.0 * eps)
        
    return J

def fisher_metric_from_jacobian(jacobian: np.ndarray) -> np.ndarray:
    """
    Computes the pullback Fisher metric G = J^T J.
    """
    return jacobian.T @ jacobian

def symmetrize_metric(metric: np.ndarray) -> np.ndarray:
    """
    Ensures strict numerical symmetry of the metric.
    """
    return 0.5 * (metric + metric.T)

def regularize_metric_svd(
    metric: np.ndarray,
    min_singular_value: float = 1e-8,
) -> MetricRegularizationResult:
    """
    Regularizes symmetric metric using SVD/eigenvalue decomposition to ensure positive definiteness.
    """
    metric = symmetrize_metric(metric)
    
    # Since metric is symmetric positive semidefinite, eigenvalues == singular values.
    # We use eigh for numerical stability on symmetric matrices.
    eigenvalues, eigenvectors = np.linalg.eigh(metric)
    
    # Sort descending to match SVD convention
    idx = np.argsort(eigenvalues)[::-1]
    eigenvalues = eigenvalues[idx]
    eigenvectors = eigenvectors[:, idx]
    
    cond_before = eigenvalues[0] / (eigenvalues[-1] + 1e-15) if eigenvalues[-1] > 0 else np.inf
    
    # Regularize
    reg_eigenvalues = np.maximum(eigenvalues, min_singular_value)
    cond_after = reg_eigenvalues[0] / reg_eigenvalues[-1]
    
    # Reconstruct
    reg_metric = eigenvectors @ np.diag(reg_eigenvalues) @ eigenvectors.T
    reg_metric = symmetrize_metric(reg_metric)
    
    return MetricRegularizationResult(
        metric=reg_metric,
        singular_values=eigenvalues,
        regularized_singular_values=reg_eigenvalues,
        condition_number_before=float(cond_before),
        condition_number_after=float(cond_after)
    )

def condition_number(matrix: np.ndarray, eps: float = 1e-12) -> float:
    """
    Computes condition number.
    """
    s = np.linalg.svd(matrix, compute_uv=False)
    if s[-1] < eps:
        return float('inf')
    return float(s[0] / s[-1])

def path_length_euclidean(path: np.ndarray) -> float:
    """
    Computes Euclidean path length.
    """
    path = np.asarray(path)
    if len(path) < 2:
        return 0.0
    if path.ndim != 2:
        raise ValueError("Path must be a 2D array of shape (n_points, d_z)")
    diffs = path[1:] - path[:-1]
    return float(np.sum(np.linalg.norm(diffs, axis=1)))

def path_length_riemannian(
    path: np.ndarray,
    metric_fn: Callable[[np.ndarray], np.ndarray],
) -> float:
    """
    Computes Riemannian path length using midpoint evaluation.
    Does NOT silently fallback to Euclidean.
    """
    path = np.asarray(path)
    if len(path) < 2:
        return 0.0
    if path.ndim != 2:
        raise ValueError("Path must be a 2D array of shape (n_points, d_z)")
        
    length = 0.0
    for k in range(len(path) - 1):
        z_current = path[k]
        z_next = path[k+1]
        
        segment = z_next - z_current
        midpoint = 0.5 * (z_current + z_next)
        
        try:
            G = metric_fn(midpoint)
        except Exception as e:
            raise RuntimeError(f"Metric computation failed at midpoint {midpoint}: {e}") from e
            
        if G.shape != (len(segment), len(segment)):
            raise ValueError(f"Metric shape {G.shape} does not match path dimension {len(segment)}")
            
        quadratic_form = float(segment.T @ G @ segment)
        if quadratic_form < -1e-10:
            raise ValueError(f"Metric is not positive semi-definite (quadratic form = {quadratic_form})")
            
        length += np.sqrt(max(0.0, quadratic_form))
        
    return length
