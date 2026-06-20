# Numerical Precision and Reproducibility Policy

To ensure mathematical reproducibility and honesty across all stages of this research prototype, we enforce the following numerical standards and seed discipline.

---

## 1. Floating-Point Representations

We do not use `Decimal` arithmetic, as it introduces unacceptable performance overhead for tensor operations and is incompatible with deep learning frameworks. Instead, we use standard floating-point types configured strategically:

* **PyTorch Inference:** Uses the native data type of the trained checkpoint (typically `float32`).
* **NumPy Computations:** Pullback metric evaluations, SVD operations, and numerical Jacobian differences in `src/geometry.py` use `float64` to maintain high numerical precision and stability.
* **Polynomial Root Finding:** Calculation of AR characteristic roots for stationarity validation uses complex double-precision numbers (`complex128`).

---

## 2. Tolerance Thresholds

NumPy array comparisons use relative (`rtol`) and absolute (`atol`) tolerances rather than strict equality, to account for minor compiler/hardware optimization variations:

* **Training Data Comparison:** `atol=1e-5`, `rtol=1e-5`
* **Latent Extraction Comparisons:** `atol=1e-4`, `rtol=1e-4`
* **Fisher Metric Elements:** Pullback matrices must be symmetric within `atol=1e-8`. SVD singular values are regularized using a strict threshold ($\sigma_i < 10^{-5}$).

---

## 3. Seed Discipline

Exact numeric replication of stochastic processes requires:
* **Global Seed Governance:** Global random seeds for NumPy (`np.random.seed`) and PyTorch (`torch.manual_seed`) are loaded directly from the YAML config.
* **Baseline Random Search:** The random baseline search in `src/baseline_random_search.py` must use a documented, fixed seed to prevent sliding benchmark results.
* **GPU Operations:** For runs on GPU, PyTorch deterministic mode (`torch.use_deterministic_algorithms(True)`) should be enabled if supported, though CPU execution is preferred for exact scientific consistency.

---

## 4. Run Identity and Versioning

Reproducibility is anchored by matching execution artifacts with their code context:
* **Code Hash:** Record the active code repository Git commit hash (or `null` if git is not initialized).
* **Config Snapshots:** Every run copies its source configuration file into the reproduction folder.
* **Inputs & Outputs Hash:** SHA-256 hashes of all input files loaded and output files saved are calculated and recorded in the run manifest.
