# PHASE 2 METRIC CONTRACT
# All Metrics, Validity Definitions, Success/Failure Criteria

**Document version:** 1 (initial)
**Created:** 2026-06-20
**Branch:** phase2/p2-protocol-and-schema-design
**Status:** DRAFT — awaiting reviewer approval before any training begins

> [!IMPORTANT]
> **All TO_BE_DEFINED threshold values in this document must be resolved and this document
> marked APPROVED before any Phase 2 model is trained.**
> Post-hoc threshold changes are unconditionally forbidden (see `docs/PHASE_2_NO_RESULT_CHASING_RULES.md`).
> If a threshold is changed after training has begun, the run is invalid and must be discarded.

---

## 1. Input Validity Metrics

These metrics characterize the quality of the training data. They must be verified before any
training begins. If any input validity metric falls below its required level, the data generator
must be fixed and the data regenerated — not the threshold lowered.

### 1.1 A_input_valid_rate

```
A_input_valid_rate = (number of A-train samples passing all A-validity checks)
                     / (total number of A-train samples)
```

**Required validity checks for A-family:**
- AR stationarity: all roots of the AR polynomial strictly inside the unit circle
  (numerical tolerance: TO_BE_DEFINED — must match `validation.tolerance` in Phase 2 config; not `tol=1e-8` function default)
- MA invertibility: all roots of the MA polynomial strictly outside the unit circle

**Required value before training:** `A_input_valid_rate = 1.0` (100%)
Any value below 1.0 is a generator failure and must cause fail-fast.

### 1.2 B_input_valid_rate

```
B_input_valid_rate = (number of B-train samples passing all B-validity checks)
                     / (total number of B-train samples)
```

**Required validity checks for B-family:**
- ω > 0 (strictly positive)
- αᵢ ≥ 0 for all i
- βⱼ ≥ 0 for all j
- Σα + Σβ < 1

**Required value before training:** `B_input_valid_rate = 1.0` (100%)
The Phase 1 `legacy_compatible` generator achieved ~62.5% B_input_valid_rate due to the GARCH beta bug.
That is not acceptable for Phase 2. The Phase 2 generator must achieve 100%.

### 1.3 C_input_valid_rate (Holdout Set)

```
C_input_valid_rate = (number of C-holdout samples passing all C-validity checks)
                     / (total number of C-holdout samples)
```

**Required validity checks for C-family:**
- All A-family checks (stationarity, invertibility)
- All B-family checks (positivity, persistence < 1)

**Required value before evaluation:** `C_input_valid_rate = 1.0` (100%)
Any C-holdout sample that is not itself valid is not a valid evaluation target.

---

## 2. Generated C Metrics

These metrics characterize the quality of the model's output when asked to produce C-family samples.
All metrics are computed over the generated samples at evaluation time.

### 2.1 generated_C_valid_rate

```
generated_C_valid_rate = (number of generated samples that pass all C-validity checks)
                         / (total number of generated C-candidate samples)
```

**C-validity check:** A generated sample is valid-C if and only if ALL of:
- AR polynomial: all roots inside unit circle (stationarity)
- MA polynomial: all roots outside unit circle (invertibility)
- ω > 0
- αᵢ ≥ 0 for all i
- βⱼ ≥ 0 for all j
- Σα + Σβ < 1

**Threshold:** PROPOSED_PROTOCOL_THRESHOLD = TO_BE_DEFINED

Rationale for choosing the threshold (to be filled in before approval):
- The unconstrained VAE baseline is expected to achieve near 0% (per Phase 1 findings: 0.0% reconstruction validity)
- The supervised upper bound should achieve near 100% with a well-constrained architecture
- The proposed threshold for zero-shot C generalization claim should be significantly above random (random validity rate for ARMA-GARCH is TO_BE_DEFINED based on parameter space analysis)
- A threshold below 50% would be weak; above 80% would be strong
- The threshold must be set before training and may not be changed after

**Interim placeholder (not valid for training — must be replaced):**
PROPOSED_PROTOCOL_THRESHOLD = 50% — this number requires justification and must be replaced with
an approved value before Gate G3 is cleared.

### 2.2 generated_C_composition_score

**Definition:** A generated sample has valid composition if it satisfies the C-validity check
(§2.1) AND simultaneously has:
- Non-trivial mean component: at least one |φᵢ| > COMPOSITION_MEAN_THRESHOLD (TO_BE_DEFINED)
- Non-trivial volatility component: at least one αᵢ > COMPOSITION_VOL_THRESHOLD (TO_BE_DEFINED)

```
generated_C_composition_score = (number of generated samples with valid composition)
                                 / (total number of generated C-candidate samples)
```

This metric distinguishes "degenerate" C from genuine composition. A sample with all near-zero AR
coefficients and all near-zero ARCH coefficients would pass C-validity trivially (constant zero
process), but would not demonstrate genuine composition.

**Threshold:** PROPOSED_PROTOCOL_THRESHOLD = TO_BE_DEFINED

### 2.3 generated_C_novelty_score

**Definition:** A generated sample is novel if it is NOT reproduced by memorization of a specific
training sample. Novelty is measured as:

```
generated_C_novelty_score = (fraction of generated C-valid samples with minimum distance
                             to nearest A-train or B-train sample > NOVELTY_DISTANCE_THRESHOLD)
```

Distance metric: TO_BE_DEFINED (candidates: L2 distance in parameter space, Mahalanobis distance,
parameter-wise normalized distance). Must be committed before training.

**Threshold:** PROPOSED_PROTOCOL_THRESHOLD = TO_BE_DEFINED

**Note:** A model that memorizes and re-emits training samples will score high on validity but low
on novelty. Both validity and novelty are required for a positive C-generation claim.

### 2.4 generated_C_distance_to_A

**Definition:**
```
generated_C_distance_to_A = mean(min_distance(x_gen, A_train)) for x_gen in valid_C_generated
```

Where distance metric is the same as used for generated_C_novelty_score.

This metric measures how far the generated C samples are from the A-family training distribution.
Very small distance suggests the model is projecting ARMA-only, not ARMA-GARCH.

### 2.5 generated_C_distance_to_B

**Definition:**
```
generated_C_distance_to_B = mean(min_distance(x_gen, B_train)) for x_gen in valid_C_generated
```

Analogous to §2.4 but for the B-family. Very small distance suggests the model is projecting
GARCH-only, not ARMA-GARCH.

### 2.6 generated_C_distance_to_true_C_distribution

**Definition:**
```
generated_C_distance_to_true_C_distribution = distributional distance(
    distribution(generated valid C samples),
    distribution(C holdout samples)
)
```

Distributional distance metric: TO_BE_DEFINED (candidates: Maximum Mean Discrepancy, Wasserstein-1,
Fréchet distance in parameter space). Must be committed before training.

This metric measures whether the generated C distribution resembles the true C parameter distribution.
A model that generates C samples concentrated at the boundary between A and B (near-zero mean or
near-zero volatility components) will score poorly on this metric even if the samples are technically valid.

---

## 3. Series-Level Diagnostics

These diagnostics are secondary — they verify that the generated ARMA-GARCH parameters, when used
to simulate a time series, produce series with the expected statistical properties of ARMA-GARCH processes.

**Note:** These diagnostics require simulating time series from the generated parameters. The
simulation procedure must be specified before evaluation (TO_BE_DEFINED: number of timesteps,
burn-in period, simulation seed).

### 3.1 ACF/PACF Mean Dynamics Compatibility

Simulated series from generated C parameters should exhibit autocorrelation patterns consistent
with the mean order (p, q) of the generated ARMA component.

**Measurement:** Ljung-Box test on simulated series residuals (after removing GARCH effects).
Pass threshold: TO_BE_DEFINED.

### 3.2 Volatility Clustering Proxy

Simulated series from generated C parameters should exhibit volatility clustering (large absolute
returns followed by large absolute returns).

**Measurement:** Autocorrelation of squared returns at lag 1 through lag max(r, s).
Pass threshold: TO_BE_DEFINED.

### 3.3 Conditional Variance Stability

The conditional variance process derived from the GARCH parameters should be covariance-stationary
(persistence < 1 is already checked in C-validity, but the simulated process must also be numerically stable).

**Measurement:** Maximum absolute value of the simulated conditional variance over the simulation period.
If the conditional variance diverges, the simulation has failed.
Stability criterion: TO_BE_DEFINED.

### 3.4 Simulation Failure Rate

```
simulation_failure_rate = (number of generated C samples for which simulation diverges or errors)
                          / (total number of generated C-valid samples)
```

A high simulation failure rate despite high C-validity suggests the validity checks are insufficient
or the generated parameters are numerically edge cases.

**Required value:** simulation_failure_rate must be < TO_BE_DEFINED before a positive claim is made.

---

## 4. Reproducibility Metrics

### 4.1 Seed Stability

```
seed_stability_range = max(generated_C_valid_rate across seeds)
                       - min(generated_C_valid_rate across seeds)
```

**Requirement:** seed_stability_range < TO_BE_DEFINED
Minimum number of seeds: TO_BE_DEFINED (≥ 3 recommended; must be committed before training)

### 4.2 Run-to-Run Variance

For a fixed seed, two training runs with identical config and identical data should produce identical
results. If any non-determinism exists (GPU parallelism, OS scheduling), it must be documented.

**Measurement:** Standard deviation of `generated_C_valid_rate` across repeated identical runs.

### 4.3 Artifact Completeness

All required artifacts (see `docs/PHASE_2_ARTIFACT_CONTRACT.md`) must be present and non-empty
after a run. An incomplete artifact is a run failure.

```
artifact_completeness_score = (number of required artifacts present and non-empty)
                              / (total number of required artifacts)
```

**Required value:** `artifact_completeness_score = 1.0` — all artifacts must be present.

---

## 5. Required Validity Definitions

These definitions are mathematical and must be applied consistently across all metrics.
All tolerance values are TO_BE_DEFINED and must be explicit in the Phase 2 config.

### 5.1 AR Stationarity
An AR polynomial φ(z) = 1 − φ₁z − φ₂z² − ... − φₚzᵖ is stationary if and only if all roots of
φ(z) = 0 have |root| > 1, equivalently all roots of the reversed polynomial lie strictly outside
the unit circle, equivalently all roots of the companion polynomial lie strictly inside the unit circle.

Numerical implementation: `all(|roots| < 1 − tol)` where `tol = PHASE_2_CONFIG.validation.tolerance`.
This is equivalent to the Phase 1 implementation in `src/validation.py:is_stationary_ar` but
tol must be config-driven, not `tol=1e-8` function default.

### 5.2 MA Invertibility
An MA polynomial θ(z) = 1 + θ₁z + θ₂z² + ... + θ_qz^q is invertible if and only if all roots of
θ(z) = 0 have |root| > 1.

Numerical implementation: `all(|roots| < 1 − tol)` via the same convention.

### 5.3 GARCH Positivity
ω > tol AND αᵢ ≥ −tol for all i AND βⱼ ≥ −tol for all j
where tol = PHASE_2_CONFIG.validation.tolerance.

Note: the sign of tol is reversed from the stationarity check — here we allow values down to −tol
as "effectively non-negative" to handle floating-point noise.

### 5.4 GARCH Persistence
Σᵢ αᵢ + Σⱼ βⱼ < 1 − persistence_tol
where persistence_tol = TO_BE_DEFINED (separate from numerical tolerance; controls how close to the
boundary we allow; must be > 0 and < 1; must be explicit in Phase 2 config).

### 5.5 ARMA-GARCH Combined Validity
A generated sample is valid C if and only if ALL of:
- AR stationarity (§5.1)
- MA invertibility (§5.2)
- GARCH positivity (§5.3)
- GARCH persistence (§5.4)
- Non-trivial mean component: at least one |φᵢ| > COMPOSITION_MEAN_THRESHOLD (for composition score)
- Non-trivial volatility component: at least one αᵢ > COMPOSITION_VOL_THRESHOLD (for composition score)

---

## 6. Required Success Criteria

> [!CAUTION]
> The following numeric thresholds are PROPOSED_PROTOCOL_THRESHOLD values only.
> They are not approved. They must be replaced with APPROVED values before Gate G3 is cleared.
> Training is blocked until these are approved.

| Metric | PROPOSED_PROTOCOL_THRESHOLD | Status | Rationale |
|--------|----------------------------|--------|-----------|
| generated_C_valid_rate (zero-shot) | TO_BE_DEFINED | BLOCKED | Requires parameter space analysis to set a meaningful threshold above chance |
| generated_C_composition_score | TO_BE_DEFINED | BLOCKED | Requires defining COMPOSITION_MEAN_THRESHOLD and COMPOSITION_VOL_THRESHOLD |
| generated_C_novelty_score | TO_BE_DEFINED | BLOCKED | Requires defining distance metric and baseline novelty |
| seed_stability_range | TO_BE_DEFINED | BLOCKED | Requires pilot experiment data |
| Minimum number of seeds | TO_BE_DEFINED (≥ 3) | BLOCKED | Must be committed |
| simulation_failure_rate (max) | TO_BE_DEFINED | BLOCKED | Requires defining simulation procedure |

**Every TO_BE_DEFINED item in this table must be resolved before training begins.**

A success claim requires ALL of the following to be simultaneously true:
1. `generated_C_valid_rate` ≥ APPROVED threshold (zero-shot protocol)
2. `generated_C_composition_score` ≥ APPROVED threshold
3. `generated_C_novelty_score` ≥ APPROVED threshold
4. `seed_stability_range` ≤ APPROVED threshold
5. Unconstrained VAE baseline was evaluated under identical conditions and scored below the threshold
6. Split contract verified (split manifest hash matched, `C_count_in_zero_shot_train == 0`)
7. Full artifact archive present

---

## 7. Failure Criteria

The following outcomes constitute experimental failure and must be reported as such:

### Validity without Novelty Is Not Success
If `generated_C_valid_rate` is high but `generated_C_novelty_score` is low, the model is
reproducing training samples with valid structure but not composing new C instances.
This is memorization, not generalization. RESULT: FAILURE.

### Novelty without Validity Is Not Success
If `generated_C_novelty_score` is high but `generated_C_valid_rate` is low, the model is producing
novel vectors that happen to not be valid ARMA-GARCH instances.
This is creative noise, not structural composition. RESULT: FAILURE.

### Numeric Smoothness Is Not Success
A smooth interpolation path in latent space that passes through valid A and B regions does not
constitute C generation. A valid C sample must independently satisfy all C-validity constraints.
Interpolation path smoothness is not a metric in this contract. RESULT: Not applicable to C claim.

### Interpolation Path Beauty Is Not Success
Visual evidence of smooth latent manifolds, PCA trajectories, or UMAP visualizations does not
constitute a C-generation claim. Only the structural validity metrics in §2 constitute evidence.
RESULT: REJECTED as C-generation evidence.

### H0 Confirmed Is a Valid Negative Result
If no model (constrained or unconstrained) achieves `generated_C_valid_rate` above the threshold,
H0 is confirmed. This is a valid scientific outcome and must be preserved as such, not interpreted
as a failure of the experimental design.

---

## 8. Reviewer Approval Required

Mark as approved by appending:

```
## APPROVAL RECORD
Approved by: [reviewer name or GitHub handle]
Date: [YYYY-MM-DD]
Commit hash at approval: [full SHA]
Status: APPROVED
Resolved thresholds:
  - generated_C_valid_rate threshold: [value]%
  - generated_C_composition_score threshold: [value]%
  - generated_C_novelty_score threshold: [value]%
  - seed_stability_range threshold: [value]%
  - minimum seeds: [N]
  - simulation_failure_rate max: [value]%
  - validation.tolerance: [value]
  - persistence_tol: [value]
  - distance metric for novelty: [metric name]
  - distributional distance metric: [metric name]
  - COMPOSITION_MEAN_THRESHOLD: [value]
  - COMPOSITION_VOL_THRESHOLD: [value]
  - simulation timesteps: [N]
  - simulation burn-in: [N]
  - simulation seed: [value or seeding rule]
```
