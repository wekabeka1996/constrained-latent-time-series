# Findings in Plain Language

> **Summary:** A clean 7-stage reproduction runner was built from scratch.
> The runner confirmed one partial finding, corrected the explanation of
> another, and formally invalidated four major claims from the original
> research report. No public data was generated or claimed.

---

## What we tried to test

We asked a single scientific question:

> *Can a Variational Autoencoder (VAE) learn a continuous latent space that
> meaningfully organises discrete econometric time-series model families
> (pure AR and pure GARCH), and can points in that space be decoded back into
> structurally valid models?*

To explore this, the original research:
1. Trained a VAE on synthetic parameter vectors representing pure AR and pure
   GARCH models.
2. Examined whether the encoder separated the two families in latent space.
3. Decoded samples and interpolated trajectories and tested whether the
   outputs were structurally valid time-series models.
4. Tried to characterise the geometry of the learned latent space using
   Fisher information metrics.
5. Backtested generated models against a real market (BTCUSDT).

---

## What the legacy research claimed

The original article made six concrete claims:

| # | Claim |
|---|-------|
| CLAIM_01 | The VAE encoder learns separate latent regions for AR and GARCH inputs. |
| CLAIM_02 | The VAE decoder produces structurally valid parameter vectors (~100% validity). |
| CLAIM_03 | Linear interpolation in latent space produces valid ARMA transition models ("ARMA bridge"). |
| CLAIM_04 | Pullback Fisher information metrics characterise the geometry of the latent manifold. |
| CLAIM_05 | SVD shows the manifold is intrinsically 2D and geodesics are 5–10× shorter than Euclidean paths. |
| CLAIM_06 | Generated models show no practical predictive edge on BTCUSDT returns. |

An implicit seventh claim was that the "syntax–semantics gap" (model structure
vs. market relevance) explained the lack of predictive power.

---

## What the clean reproduction showed

A 7-stage reproduction runner (`src/reproduce.py`) was built and executed.
It uses only canonical, independently audited code modules.
The final status is **COMPLETED\_WITH\_LIMITATIONS**.

Key numbers from the latest run (`results/reproduction_20260620_085314/`):

| Metric | Value |
|---|---|
| Stage 1 input vectors generated | 200 |
| Stage 1 input vectors structurally valid | 125 / 200 = **62.5%** |
| Stage 4 VAE reconstructions structurally valid | 0 / 200 = **0.0%** |
| Stage 4 reconstruction failure reason | `LEGACY_LAYOUT_COLLISION` (all 200) |
| Stage 5 geometry scope | Valid Stage 1 input latents only |
| Stage 5 selected rows for geometry | 32 of 125 valid inputs |
| Stage 5 condition number (max) | 24.5 |
| Legacy Fisher metrics status | `INVALID_DO_NOT_CLAIM` |
| Stage 6 market data status | `NEEDS_REAL_DATA` |

---

## What survived

**CLAIM_01 (PARTIALLY CONFIRMED)**

The encoder does separate AR and GARCH inputs in latent space. The Stage 3
output produces distinct `z_train_ar` and `z_train_garch` embeddings, which
the legacy t-SNE/UMAP plots visually corroborate. This is the one encoder-side
finding that the clean run supports. However, the word "clusters" should be
used carefully until quantitative clustering metrics are computed under clean
provenance.

---

## What was invalidated

**CLAIM_02 — INVALIDATED**

The decoder does **not** produce structurally valid outputs. All 200
reconstructed vectors fail validation with `LEGACY_LAYOUT_COLLISION`. The root
cause is that the VAE decoder is unconstrained: when it decodes a latent point,
the two discrete indicator logits (AR-side and GARCH-side) can both be weakly
positive simultaneously. This causes the parameter layout classifier to treat
the output as a joint ARMA-GARCH model, which then has overlapping parameter
positions (slot `v[10]` is `phi_1` for ARMA but `omega` for GARCH). The
original claim of ~100% validity was based on a looser, pre-schema classifier.

**CLAIM_03 — INVALIDATED**

The "ARMA bridge" concept does not hold. If reconstructed single-type models
already fail structural validation (CLAIM_02), interpolated hybrids fail by the
same mechanism. There is no evidence of a valid structural transition path.

**CLAIM_04 — INVALIDATED**

The legacy Fisher metric code contained a P0 mathematical bug. It approximated
the Jacobian by dividing element-wise output differences by element-wise latent
differences along a 1D trajectory. This does not compute partial derivatives.
The correct Jacobian requires perturbing each latent dimension independently.
The resulting old `G = J^T J` matrices are numerically meaningless. See
[GEOMETRY\_VALIDITY\_STATUS.md](GEOMETRY_VALIDITY_STATUS.md).

**CLAIM_05 — INVALIDATED**

The dimensionality (2D) and geodesic length (5–10× shorter) claims were derived
directly from the buggy Fisher matrices. They are therefore also invalid. The
corrected Stage 5 geometry (computed on valid inputs only) provides a clean but
limited baseline — it does not reproduce these claims and cannot be used to
assert them.

**CLAIM_07 — CHANGED DUE TO BUGFIX**

The original explanation for the market mismatch was that the VAE learned model
"syntax" (structure) but lacked market "semantics" (predictive power). The
corrected finding is that the VAE did not actually learn valid structural syntax
in the first place. The mismatch remains real, but the explanation changes.

---

## Why the old Fisher/geodesic claims cannot be used

The legacy script `src/compute_fisher_metric.py` computed the Jacobian as:

```
J[i, j] = (theta_{k+1}[i] - theta_k[i]) / (z_{k+1}[j] - z_k[j])
```

This is a ratio of two element-wise differences along a fixed trajectory. It is
**not** a partial derivative of decoder output with respect to each latent
dimension. The correct numerical Jacobian requires:

```
J[i, j] = (decoder(z + eps * e_j)[i] - decoder(z)[i]) / eps
```

where `e_j` is the j-th standard basis vector. The corrected implementation
lives in `src/geometry.py` and was validated in Stage 5. The old `.npy` files
in `results/generation_1/fisher_metrics/` and
`results/generation_1/regularized_fisher/` are retained for audit history but
must never be cited as valid evidence.

---

## Why the decoder/reconstruction claim failed

The VAE was trained on disjoint inputs (pure AR or pure GARCH). Its discrete
indicator head has two outputs: one for the AR/ARMA component and one for the
GARCH component. In a structurally valid reconstruction, only one should be
above the 0.5 activation threshold at a time. However, the unconstrained
decoder learns to keep both indicators in the range 0.03–0.05 for "inactive"
components — these round above threshold and trigger a joint classification.
The joint ARMA-GARCH layout then collides at parameter slot `v[10]`, causing
`LEGACY_LAYOUT_COLLISION`. This is a structural model design issue, not a
data quality issue.

---

## What corrected geometry means here

Stage 5 computes pullback Fisher information metrics using the canonical,
mathematically correct implementation (`src/geometry.py`). It selects only the
32 latent encodings that correspond to structurally valid Stage 1 input vectors.
The resulting metrics are:
- All finite and numerically stable.
- All symmetric.
- Maximum condition number: 24.5 (well-conditioned).

This is a clean baseline, not a manifold characterisation. It establishes that
the corrected geometry infrastructure works and that geometry can be meaningfully
computed on the valid-input subset. Manifold claims require a working constrained
decoder before they can be extended to generated samples.

---

## What remains unresolved

**CLAIM_06 — NEEDS_REAL_DATA**

The empirical backtesting claim (no predictive edge on BTCUSDT) cannot be
verified without real market data supplied under clean provenance. No fake data,
dummy CSV, or random price series was used. Stage 6 is infrastructure-ready but
returns `NEEDS_REAL_DATA` until a real CSV is provided via `configs/demo.yaml`.

---

## What this project demonstrates as an engineering artifact

Even where the scientific claims were invalidated, the project demonstrates:

1. **Reproducible ML engineering:** A 7-stage config-driven reproduction runner
   with SHA-256 manifest tracking and incremental output isolation.
2. **Research integrity practice:** Systematic claim auditing, formal invalidation
   with root-cause evidence, and a written policy against fake data and silent
   fallbacks.
3. **Canonical schema design:** A unified `src/vector_schema.py` + `src/validation.py`
   + `src/data_generator.py` pipeline that separates structural semantics from
   model code.
4. **Corrected geometry infrastructure:** A numerically correct pullback metric
   implementation that is ready for future constrained generative experiments.
5. **Honest negative results:** The `LEGACY_LAYOUT_COLLISION` finding is itself
   a contribution — it identifies exactly why unconstrained VAE decoders fail
   for discrete-structured output spaces.

---

## What comes next

The reproduction runner output (`results/reproduction_*/`) contains a
`reports/final_claim_verdict.json` with a `recommended_next_work` field listing
the four highest-priority follow-up items:

1. **Clean v2 Generator:** Fix the GARCH beta sampling bound bug in
   `src/data_generator.py` so that all generated training vectors are valid.
2. **Constrained Decoder:** Add structural projection layers to the VAE decoder
   so that only one indicator can be active at a time — eliminating
   `LEGACY_LAYOUT_COLLISION`.
3. **Hybrid Validity Lift:** Re-run latent interpolation once the decoder is
   constrained, and check whether the "ARMA bridge" concept can be recovered.
4. **Stage 6 with Real Data:** Supply a real BTCUSDT CSV with clean provenance
   to run the empirical stress test under audit conditions.

See [README_BLUEPRINT.md](README_BLUEPRINT.md) for the planned public-facing
description and the full list of open research questions.
