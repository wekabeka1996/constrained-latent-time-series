# PHASE 2 RESEARCH PROTOCOL
# Problem Statement, Scope, Experiments, Baselines, Claim Language

**Document version:** 1 (initial)
**Created:** 2026-06-20
**Branch:** phase2/p2-protocol-and-schema-design
**Status:** DRAFT — awaiting reviewer approval before implementation begins

---

## 1. Problem Statement

Standard variational autoencoders (VAEs) learn a continuous latent space by encoding training samples
into a normal posterior and decoding sampled latent vectors back to the observation space.
For econometric time-series parameter vectors, this creates a well-known failure mode: the decoder
maps any point in the continuous latent space to a continuous numeric output, but this output is not
guaranteed to satisfy the discrete structural constraints of any econometric model family.

The fundamental question is whether a model trained on two structurally disjoint families A (ARMA mean
dynamics) and B (GARCH volatility dynamics) can produce structurally valid samples from the hybrid
family C (ARMA-GARCH) at inference time, without having been trained on any C example.

This requires two distinct capabilities:
1. **Compositional generalization** — the model must infer that C = A + B is valid structural combination.
2. **Structural validity** — the decoded output must simultaneously satisfy ARMA stationarity/invertibility
   AND GARCH non-negativity/persistence constraints.

Neither capability is provided by an unconstrained VAE architecture, which is the precise reason
this question requires investigation.

---

## 2. Scope

### In Scope
- Synthetic mathematical benchmark — parameter vectors from analytically defined families A, B, C
- VAE-based generative models with and without structural constraints
- Zero-shot C generation: model trained on A ∪ B only, evaluated on C
- Few-shot C generation: model trained on A ∪ B ∪ (small fraction of C)
- Supervised upper-bound: model trained on A ∪ B ∪ C
- Structural validity as the primary metric
- Comparison against defined baselines (see §7)
- Negative results are in scope and must be preserved

### Out of Scope
- Real market data as primary evaluation (market data is tertiary stress test only — see §4)
- Numeric forecasting accuracy
- Risk/profit metrics
- Production deployment
- Any claim about real financial data without explicit extension of the protocol
- ARIMA, SARIMA, or other extended families not defined in the family contract

### Explicitly Excluded from Phase 2 Training
- Any ARMA-GARCH (C-family) sample in the zero-shot training split
- Any hybrid model that simultaneously fits both mean and volatility dynamics
- Market data as training source (tertiary only)

---

## 3. Why Synthetic Mathematical Benchmark Is Primary

Real financial time series introduce confounds that prevent clean experimental control:
- Unknown true data-generating process
- Non-stationarity, regime changes, structural breaks
- Data cleaning and preprocessing choices that introduce bias
- No ground-truth label for whether a series is "AR" vs "GARCH" vs "ARMA-GARCH"

The synthetic mathematical benchmark provides:
- Exact ground-truth labels for every sample
- Known parameter ranges and validity constraints
- Controlled split: zero contamination of C in training is verifiable by construction
- Reproducibility: identical data can be regenerated from seed

**The benchmark is the primary experimental surface. A positive result on the benchmark is necessary
but not sufficient for a real-world claim.**

---

## 4. Why Real Market Data Is Only Tertiary Stress Test

Market data is a tertiary stress test because:
1. It cannot provide ground-truth family labels
2. The split contract (C absent from training) cannot be verified on unlabeled real data
3. Any structural validity claim on real market outputs would require a separate fitting procedure
   that introduces its own assumptions
4. Real data introduces confounds that obscure the structural composition signal

Market data will be used — if at all — only after:
- Positive synthetic benchmark results are obtained
- A separate market data protocol is defined (out of scope for Phase 2)

The `market_data_csv: null` setting in configs must remain null for Phase 2 zero-shot protocol.

---

## 5. Hypotheses

### H0 — Null Hypothesis
Unconstrained continuous latent generative models (standard VAE, β-VAE) trained only on A and B
cannot reliably generate valid unseen C without structural inductive bias.

**What would falsify H0:** A standard VAE with no structural constraints achieves
`generated_C_valid_rate ≥ PROPOSED_PROTOCOL_THRESHOLD` under the zero-shot protocol. If H0 is
falsified, it means the latent space geometry spontaneously encodes the A+B→C composition rule,
which would itself be a significant and unexpected finding.

### H1 — Alternative Hypothesis
Rule-aware / constrained / conditional / grammar-aware / compositional generative models can generate
valid C if the A+B→C composition rule is represented in architecture, grammar, loss, projection, or
evaluation protocol.

**What would support H1:** A constrained model achieves `generated_C_valid_rate` significantly above
the H0 baseline (unconstrained VAE), with the gap being statistically meaningful and stable across seeds.

### H2 — Anti-Interpolation Hypothesis
Pure latent interpolation may produce smooth numeric transitions, but structural validity of C requires
explicit constraints or learned compositional factors. Numeric smoothness is not structural validity.

**What would support H2:** Latent interpolation paths (midpoint decoding) produce lower structural
validity rates than constrained decoding under identical conditions.

---

## 6. Experiment Families

### Experiment E1 — Zero-Shot C (Primary)
- Training data: A ∪ B only
- No C sample in training under any label, in any split, under any mode
- Evaluation: C holdout set
- Primary metrics: `generated_C_valid_rate`, `generated_C_composition_score`, `generated_C_novelty_score`
- Success requires all three metrics above precommitted thresholds (APPROVED_PROTOCOL_THRESHOLD: see `docs/PHASE_2_METRIC_CONTRACT.md` §6)
- Models tested: all baselines and all constrained variants

### Experiment E2 — Few-Shot C at 1%
- Training data: A ∪ B ∪ (1% C by sample count)
- 1% C samples must be drawn from a pool disjoint from the zero-shot evaluation C holdout
- Explicit seed list, explicit sample ID manifest, explicit split manifest hash required
- Evaluation: same C holdout as E1
- Comparison: E2 vs E1 to measure few-shot gain from 1% C exposure

### Experiment E3 — Few-Shot C at 5%
- Same structure as E2 but with 5% C in training
- Separate seed list and sample ID manifest from E2
- Comparison: E3 vs E2 to measure gain from 5% vs 1% C exposure

### Experiment E4 — Supervised Upper Bound
- Training data: A ∪ B ∪ C (full C training set)
- Establishes the ceiling for `generated_C_valid_rate` under full supervision
- This result is a reference, not a claim about zero-shot capability
- If supervised ceiling < PROPOSED_PROTOCOL_THRESHOLD, the threshold must be revised and
  the revision must be documented before any training restarts

---

## 7. Required Baselines

The following baselines must be evaluated before any constrained model result is interpreted.
All baselines are subject to the same split contract, metric contract, and artifact contract.

### B1 — Unconstrained VAE (Phase 1 legacy)
- Architecture: standard VAE as in `src/vae.py` (d_input=40, d_latent=8, d_hidden=64/32)
- Training: standard ELBO loss
- Purpose: establishes the H0 baseline
- Expected result: low generated_C_valid_rate (per Phase 1 Phase 1P findings: 0.0% reconstruction validity)

### B2 — β-VAE
- Architecture: same as B1 but with β > 1 KL weight
- β value: APPROVED_PROTOCOL_VALUE: beta = 4.0
- Purpose: tests whether disentanglement pressure improves structural validity

### B3 — Conditional VAE (cVAE)
- Architecture: VAE conditioned on family label (A or B as one-hot or embedding)
- Purpose: tests whether explicit label conditioning helps C generalization
- Condition: label is A or B during training; what label is given at C evaluation time is: APPROVED_PROTOCOL_VALUE: evaluated at C using the interpolated label vector [0.5, 0.5] and also by evaluating both A [1, 0] and B [0, 1] label conditioning inputs separately.

### B4 — Gumbel / Categorical-Structure VAE
- Architecture: VAE with discrete or semi-discrete latent structure (e.g. Gumbel-softmax for family indicators)
- Purpose: tests whether explicit latent discretization for family type improves structural validity
- Specific variant: APPROVED_PROTOCOL_VALUE: Gumbel-Softmax estimator with a continuous relaxation of categorical variables (initial temperature = 1.0, annealed to 0.1).

### B5 — Grammar / Schema Decoder
- Architecture: decoder that decodes through a schema-constrained projection layer (parameter groups decoded separately)
- Purpose: tests whether architectural separation of mean and volatility components enables C generalization
- Specific architecture: BLOCKING_TO_BE_DEFINED: The specific network architecture for the grammar-projection layer depends on model capacity decisions and is deferred to the future implementation phase.

### B6 — Mixture-of-Experts / Composer
- Architecture: explicit composer that combines A-decoder output and B-decoder output via a learned composition gate
- Purpose: direct test of the A+B→C composition hypothesis
- Specific architecture: BLOCKING_TO_BE_DEFINED: The gating network architecture and experts count are deferred to the future implementation phase.

---

## 8. Negative Result Policy

Negative results are valid scientific outputs. The following negative results must be preserved, not deleted:

1. If `generated_C_valid_rate` < PROPOSED_PROTOCOL_THRESHOLD for all baselines and constrained models:
   this is a confirmed negative result and must be reported as such.
2. If H0 is confirmed (unconstrained models also fail): this is a partial confirmation of the
   structural inductive bias hypothesis.
3. If constrained models fail but unconstrained models succeed by chance: this is an anomalous
   positive result for H0 and must be investigated before any claim is made.
4. All run artifacts for failed experiments must be preserved (`docs/PHASE_2_ARTIFACT_CONTRACT.md`).
5. Failure cases must be logged in `failure_cases.csv` per run.

---

## 9. Claim Language Rules

### Allowed Claim Examples
- "Under the zero-shot protocol, the constrained model achieved generated_C_valid_rate = X% (95% CI: [lo, hi]) across N seeds, compared to baseline unconstrained VAE at Y%."
- "The ARMA-GARCH validity rate for constrained decoding was Z% vs W% for direct latent interpolation under identical conditions."
- "All N experimental seeds produced generated_C_valid_rate within [lo, hi]; we report the median with inter-seed range."
- "The model failed to generate valid C under zero-shot conditions (generated_C_valid_rate = 0%) confirming H0."

### Forbidden Claim Examples
- "The model discovered ARMA-GARCH structure in the latent space." — Not allowed without evidence.
- "The latent manifold encodes the A+B→C composition rule." — Not allowed without evidence.
- "Our model generates valid hybrid time series." — Too vague; must specify validity rate, confidence interval, seed stability, comparison baseline.
- "Phase 2 confirms zero-shot C generalization." — Not allowed unless all precommitted thresholds are met and documented.
- "The model learns to compose families." — Not allowed without composition score above precommitted threshold.

### What Must Be True Before Saying "Model Generated C"

All of the following must be documented in the artifact and verified against precommitted thresholds:
1. `generated_C_valid_rate` ≥ APPROVED_PROTOCOL_THRESHOLD (frozen before training)
2. `generated_C_composition_score` ≥ APPROVED_PROTOCOL_THRESHOLD (frozen before training)
3. `generated_C_novelty_score` ≥ APPROVED_PROTOCOL_THRESHOLD (frozen before training)
4. C was absent from zero-shot training (split manifest hash verified)
5. Result is stable across ≥ 5 seeds (APPROVED_PROTOCOL_VALUE: `minimum_model_repeat_seeds = 5`)
6. Unconstrained baseline was evaluated under identical conditions
7. Full artifact archive exists for the run
8. Model checkpoint hash is committed in the manifest
