# Geometry Validity Status

**Date:** 2026-06-18
**Phase:** 1D (Geometry Audit)
**Verdict:** `INVALID_DO_NOT_CLAIM`

## Summary

During Phase 1D of the research integrity audit, a P0 mathematical bug was discovered in the legacy Fisher metric computation script (`src/compute_fisher_metric.py`).

The script attempted to compute the numerical Jacobian by dividing the element-wise differences between two adjacent points on a 1D trajectory (`J[i, j] = dtheta[i] / dz[j]`). This mathematically fails to compute the partial derivatives required for a true Jacobian ($\frac{\partial \theta_i}{\partial z_j}$) because it does not perturb the latent space along orthogonal basis vectors using the VAE decoder.

As a result, the resulting pullback Fisher metrics $G = J^T J$ computed by the legacy code are mathematically meaningless.

## Impacted Artifacts

The following artifacts and derivatives generated prior to Phase 1D are invalidated:
- `results/generation_1/fisher_metrics/*.npy`
- `results/generation_1/regularized_fisher/*`
- Any plots, reports, or claims citing these metrics, singular value distributions (SVD), or geodesic distances.

## Required Action

1. **Do not use** the old artifacts as public evidence.
2. Any claims regarding the Fisher information metric, geodesic distances, SVD dimensionality, or manifold geometry must be **regenerated** using the new, mathematically correct implementation in `src/geometry.py`.
3. Until regeneration is complete, any reference to Fisher geometry in the README or reports must either be omitted or explicitly labeled as invalidated legacy results.

This is a strict research integrity correction to prevent the publication of mathematically unsound results.
