# Claim Verdict Matrix

## Scope

This document provides a systematic audit of the primary research claims made in the legacy article (`Стаття_Результат_Дослідження.md`) and reports against the mathematically correct, provenance-aware evidence collected from Stages 1–5 of the reproduction runner.

---

## Evidence sources

* **Legacy Source of Truth:** `docs/Стаття_Результат_Дослідження.md`
* **Clean Reproduction Manifest:** `results/reproduction_20260618_223143/manifest.json`
* **Structural Validation Report:** `results/reproduction_20260618_223143/reports/structural_validation.json`
* **Corrected Geometry Report:** `results/reproduction_20260618_223143/reports/corrected_geometry.json`
* **Corrected Geometry Source of Truth:** `src.geometry`
* **Integrity Policies:** `docs/RESEARCH_INTEGRITY.md` and `docs/GEOMETRY_VALIDITY_STATUS.md`

---

## Verdict summary

| Status | Count |
| :--- | ---: |
| **CONFIRMED** | 0 |
| **PARTIALLY_CONFIRMED** | 1 |
| **CHANGED_DUE_TO_BUGFIX** | 1 |
| **INVALIDATED** | 4 |
| **NEEDS_REAL_DATA** | 1 |
| **NOT_REPRODUCED** | 0 |
| **OUT_OF_SCOPE** | 0 |

---

## Claim-by-claim matrix

| claim_id | legacy claim | legacy evidence | clean evidence | verdict | rationale |
|---|---|---|---|---|---|
| **CLAIM_01** | VAE learns separate latent representations of pure AR and GARCH parameter families. | t-SNE / UMAP plots of latent space showing clean, separate clusters for training inputs. | Stage 3 produces label-indexed latent embeddings (`z_train_ar`, `z_train_garch`), but quantitative clustering evidence requires a dedicated metric or plot regeneration. | **PARTIALLY_CONFIRMED** | The encoder successfully groups and separates the training classes in the latent space, but the decoder does not reconstruct structurally valid representations. |
| **CLAIM_02** | VAE decoder outputs and reconstructed parameter vectors are structurally valid. | Abstract / Section 3.1 assertions that VAE generates valid models with ~100% efficiency. | Stage 4 validation shows a **0.0%** reconstruction validity rate (200/200 vectors have `LEGACY_LAYOUT_COLLISION`). | **INVALIDATED** | Standardized, masked structural validation reveals that all VAE reconstructions suffer from overlapping parameter layouts due to weakly active discrete indicators. |
| **CLAIM_03** | Latent interpolation path forms a "structural ARMA-перекладина" of clean ARMA models. | Section 3.2: assertions that linear paths between AR and GARCH produce clean, valid ARMA models. | Stage 4 validation shows that the continuous indicators round to joint models causing layout collisions. | **INVALIDATED** | Reconstructions along the trajectory do not yield valid, disjoint ARMA models; they suffer from parameter overlaps and are structurally invalid. |
| **CLAIM_04** | Pullback Fisher information metrics support latent manifold geometry claims. | Section 3.4.1: detailed descriptions of metric tensors, geodesic curves, and manifold curvature. | Stage 5 shows that the legacy metrics were calculated using a buggy 1D division Jacobian. | **INVALIDATED** | The legacy Fisher metric script contained a P0 mathematical bug. The old metrics are mathematically meaningless and cannot support any geometry claims. |
| **CLAIM_05** | SVD analysis shows the manifold is 2D, and geodesics are 5-10x shorter than Euclidean distances. | Section 3.4.1: claims that 2 principal components explain 100% of variation, and geodesics are shorter. | Stage 5/new geometry: SVD/geodesics based on the buggy metric are invalid. | **INVALIDATED** | These claims were directly derived from the buggy legacy Fisher metric arrays and do not hold under corrected geometry. |
| **CLAIM_06** | Generated VAE models show no practical predictive edge on BTCUSDT returns. | Section 3.5: backtesting results showing poor predictive edge, interpreted as a "model-market mismatch." | Stage 6 is skipped in Phase 1L. Legacy data is in `results/generation_2/robustness/`. | **NEEDS_REAL_DATA** | This factual negative result cannot be verified until Stage 6 is run under clean provenance using real market datasets. |
| **CLAIM_07** | A syntax-semantics gap explains model-market mismatch. | Section 3.6: VAE successfully learns model "syntax" (grammar) but lacks market "semantics" (predictive edge). | Stage 4 shows that VAE reconstructions fail structural grammar validation. | **CHANGED_DUE_TO_BUGFIX** | The mismatch is real, but the explanation is incorrect: the VAE did not actually learn a valid structural "syntax" in the first place. |

---

## Invalidated legacy evidence

The following legacy files and assertions are mathematically or structurally invalid and must be excluded from public claims:
* **All legacy Fisher metrics:** `results/generation_1/fisher_metrics/*.npy` (invalidated by incorrect 1D Jacobian bug).
* **All legacy regularized Fisher metrics:** `results/generation_1/regularized_fisher/*` (invalidated by incorrect 1D Jacobian bug).
* **Manifold SVD/eigenvalue claims:** Assertions that the latent manifold is strictly two-dimensional or linear based on legacy matrices.
* **Geodesic length comparisons:** Claims that geodesics are 5-10x shorter than Euclidean paths.
* **Legacy reconstruction validity rate (~100%):** Invalidated by corrected Stage 4 validation showing a 0% validity rate due to `LEGACY_LAYOUT_COLLISION`.
* **The "structural ARMA-перекладина" concept:** Claims that linear interpolations produce valid, disjoint ARMA models.

---

## Surviving claims

The following claims remain mathematically sound and supported by Stage 1-5 evidence:
* **Latent Clustering:** The VAE encoder successfully maps pure input classes (AR vs GARCH) into separate, distinct clusters in the latent space (corroborated by Stage 3).
* **Factual Negative Result:** The models backtested in `results/generation_2/robustness/` did not yield predictive power (though the explanation of why has changed).
* **Input Dataset Validity:** The synthetic training data generator yields 62.5% valid structures, with the remaining 37.5% invalid due to GARCH parameter bounds (GARCH negative beta) and unconstrained AR stationarity.

---

## Claims that must be softened

Any remaining or future public description of this repository must use softened, evaluative phrasing:
* *"We evaluate whether"* a Variational Autoencoder can map discrete parameter families to a continuous manifold.
* *"Initial experiments suggested"* that VAEs could generalize across econometric structures.
* *"Under the valid-input-only baseline,"* corrected geometry can be computed via central finite differences.
* *"Reconstruction validity requires a future clean generator or constrained decoder"* to prevent parameter collisions.

---

## Claims requiring future work

To achieve final scientific proof, the following future work is required (Phase 2):
1. **Clean v2 Generator:** Fix GARCH beta bounds sampling in `src/data_generator.py` to prevent negative beta parameters.
2. **Constrained Decoder:** Redesign VAE decoder with structural constraints (e.g. projection layers or hard boundary mapping) to prevent joint indicator overlaps and avoid `LEGACY_LAYOUT_COLLISION`.
3. **Hybrid Validity Lift:** Re-evaluate interpolation trajectories once the VAE can reconstruct valid disjoint structures.
4. **Empirical Stress Test (Stage 6):** Run backtesting with real market data provenance (no fake data fallbacks).

---

## README-safe summary

> [!NOTE]
> This project evaluates whether a Variational Autoencoder (VAE) can learn a continuous latent manifold of discrete time-series models. Our research integrity audit revealed that while the encoder successfully clusters pure AR and GARCH models, the unconstrained decoder reconstructions suffer from parameter layout collisions, yielding a 0.0% reconstruction validity rate. Furthermore, the legacy geometry claims (including metric tensors and geodesic lengths) were invalidated by a mathematical Jacobian bug. Corrected geometry computed on the valid-input baseline provides a clean foundation for future constrained generative modeling. Legacy reports claimed no predictive edge, but this remains NEEDS_REAL_DATA until Stage 6 verifies it under clean provenance.

---

## Recommendation

* **Is final README safe now?** No. The public README should not be updated with final claims until Stage 6 (empirical stress test) is run or Stage 5.5/7 verdicts are fully checked.
* **Is Stage 6 still needed?** Yes, to verify the empirical backtesting claims under clean provenance.
* **Should Stage 6 run before README finalization?** Yes, running Stage 6 is highly recommended to complete the full reproduction runner before public release.
* **Which old claims must never be repeated?**
  1. That VAE reconstructions are structurally valid or have high valid rates.
  2. That linear interpolation produces valid ARMA-only transition structures.
  3. The legacy SVD dimensionality (100% variation in 2 components) and geodesic length metrics (5-10x shorter).
