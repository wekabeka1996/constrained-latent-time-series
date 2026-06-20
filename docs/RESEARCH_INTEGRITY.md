# Research Integrity Policy

## Overview

The `econometric-vae-manifold` project is a quantitative research prototype, not a production trading system. Its value lies in the honest exploration of a hypothesis: *Can a Variational Autoencoder learn a continuous latent manifold of discrete econometric time-series models?*

To maintain the credibility of this research, we enforce a strict **No Fake Data Contract**. This policy ensures that all outputs, plots, and metrics presented in the repository are reproducible, traceable, and mathematically honest.

---

## 1. Allowed Data Categories

### A. Legitimate Synthetic Research Data
* **Allowed:** Generating pure AR or pure GARCH parameter vectors for training the VAE.
* **Why:** This is the core experimental design. The VAE is trained on disjoint synthetic structures to study whether it can generalize to combined ARMA-GARCH structures in its latent space.
* **Rule:** Synthetic training data must be generated using `src/data_generator.py` and must be clearly identified as methodological data, never as empirical data.

### B. Real Market Data
* **Allowed:** Backtesting or evaluating models against real financial time series (e.g., BTCUSDT returns).
* **Why:** To test the empirical relevance of the generated models.
* **Rule:** Real market data must come from real files with clear provenance (e.g., downloaded via Binance API or a stated CSV file). **Fake or randomly generated market data (e.g., `np.random.randn`) is strictly forbidden.**

### C. Generated Artifacts
* **Allowed:** Outputs of documented scripts (e.g., latent embeddings, interpolated trajectories, Fisher metrics).
* **Why:** These are the results of the research.
* **Rule:** Generated artifacts must be reproducible by running the associated scripts with the stated configuration.

### D. Cached Artifacts
* **Allowed:** Committing large or expensive-to-compute outputs (e.g., training data matrices, checkpoint weights).
* **Why:** For practical convenience and reproducibility of downstream analysis without retraining.
* **Rule:** Cached artifacts must be clearly labeled. Scripts using them must not claim to be freshly computing the results if they are merely loading a cache.

### E. Forbidden Data
* **Forbidden:** `np.random.uniform` or `np.random.randn` used as fallback when files are missing.
* **Forbidden:** Fake BTCUSDT/SP500 or any synthetic market price data pretending to be real.
* **Forbidden:** Placeholder CSVs used as evidence.

---

## 2. Fast-Fail Principle

If a required artifact, checkpoint, or market data file is missing:
* **DO NOT** substitute it with random arrays to make the script continue.
* **DO NOT** silence the error and output an empty or dummy plot.
* **DO** fail fast with a clear error message explaining what is missing and how to generate or obtain it.

---

## 3. Honest Limitations

* **Legacy Compatibility:** The original training grammar contained a boundary bug where GARCH persistence could exceed 1.0 due to inverted uniform bounds. This behavior is preserved *explicitly* in `legacy_compatible` generation mode to ensure compatibility with existing `vae_beta*.pth` checkpoints. It must not be presented as mathematically clean generation.
* **Geometry Validity:** During Phase 1D, the legacy Fisher information metric computation was found to be mathematically invalid (incorrect Jacobian formulation). Old geometry artifacts and derived claims are classified as `INVALID_DO_NOT_CLAIM`. See [GEOMETRY_VALIDITY_STATUS.md](GEOMETRY_VALIDITY_STATUS.md) for details. All future claims must use the canonical implementation in `src/geometry.py`.
* **Negative Results:** If backtests perform poorly or generated models are invalid, these negative results must be preserved honestly. They inform the boundaries of the VAE's capability. No result should be massaged to make the repo look better.

---

## 4. Configuration Enforcement

The YAML configuration (`src/config.py`) enforces this policy via the `integrity` block:

```yaml
integrity:
  allow_random_fallbacks: false
  allow_fake_market_data: false
  allow_cached_artifacts: true
  require_artifact_provenance: true
```

The config loader will reject any public demo config that attempts to enable random fallbacks or fake market data.
