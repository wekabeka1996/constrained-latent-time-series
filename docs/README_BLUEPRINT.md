# Constrained Latent Time-Series Geometry

> A research reproducibility study of Variational Autoencoders applied
> to discrete parametric time-series model families, with a full 7-stage
> reproduction runner, formal claim audit, and corrected Riemannian geometry.

---

## What this project is

This repository documents a research experiment that asked:

> *Can a Variational Autoencoder (VAE) learn a continuous, geometrically
> meaningful latent space over discrete econometric time-series model families
> (AR and GARCH parameters)?*

The project includes:

- A **synthetic methodology dataset generator** for pure AR and GARCH
  parameter vectors under explicit structural validity constraints.
- A **trained VAE model** that maps these parameter vectors to a continuous
  latent space.
- A **7-stage reproduction runner** (`src/reproduce.py`) that verifies all
  original research claims under strict provenance and integrity rules.
- A **corrected Riemannian geometry baseline** using canonical pullback Fisher
  information metrics computed via central finite differences.
- A **formal claim audit** documenting which original claims survived,
  which were invalidated, and which require future work.

---

## What this project is not

- **Not a trading system.** The project evaluates whether a VAE can represent
  econometric model structure. No live trading, strategy optimisation, or
  alpha generation is performed or implied.
- **Not a finished scientific product.** The reproduction runner revealed
  significant limitations in the original work. The repository is honest about
  these limitations.
- **Not a black-box pipeline.** Every stage writes an auditable SHA-256
  manifest with provenance tracking. No silent fallbacks or fake data.

---

## Main Result

**Final reproduction status: `COMPLETED_WITH_LIMITATIONS`**

The clean 7-stage reproduction confirmed partial encoder-side clustering
(CLAIM_01) and formally invalidated four major claims from the original work
(CLAIM_02 through CLAIM_05). The empirical stress test remains unresolved
pending real market data (CLAIM_06).

---

## Final Reproduction Verdict

| Claim | Status | Finding |
|---|---|---|
| CLAIM_01: Encoder separates AR/GARCH latent regions | **PARTIALLY_CONFIRMED** | Encoder clusters are visible; quantitative metrics pending |
| CLAIM_02: Decoder produces structurally valid outputs | **INVALIDATED** | 0/200 reconstructions valid; 100% `LEGACY_LAYOUT_COLLISION` |
| CLAIM_03: Interpolation produces valid ARMA transitions | **INVALIDATED** | Collision persists along trajectories |
| CLAIM_04: Fisher metrics characterise latent manifold | **INVALIDATED** | Legacy Jacobian was mathematically incorrect (P0 bug) |
| CLAIM_05: Manifold is 2D; geodesics 5–10× shorter | **INVALIDATED** | Derived from buggy Fisher matrices; does not hold |
| CLAIM_06: No predictive edge on BTCUSDT returns | **NEEDS_REAL_DATA** | Stage 6 infrastructure ready; real CSV required |
| CLAIM_07: Syntax–semantics gap explains market mismatch | **CHANGED_DUE_TO_BUGFIX** | VAE did not learn valid structural syntax in the first place |

See [`docs/CLAIM_VERDICT_MATRIX.md`](docs/CLAIM_VERDICT_MATRIX.md) and
[`docs/FINDINGS_SIMPLE.md`](docs/FINDINGS_SIMPLE.md) for full details.

---

## Key Findings

1. **Stage 1 input validity is 62.5%** (125/200 vectors). The remaining 37.5%
   fail due to unconstrained AR stationarity (32 rows) and a GARCH negative
   beta bug (37 rows) in the original data generator.
2. **Stage 4 reconstruction validity is 0.0%** (0/200 vectors). All VAE
   decoder outputs suffer from `LEGACY_LAYOUT_COLLISION`: both discrete
   indicators are weakly active simultaneously, producing an undefined
   joint parameter layout.
3. **Stage 5 corrected geometry** is computed on 32 selected valid-input
   latents. All metrics are finite, symmetric, and numerically stable
   (max condition number: 24.5). Legacy Fisher metrics are excluded.
4. **Legacy Fisher metrics are `INVALID_DO_NOT_CLAIM`**. A P0 mathematical
   bug in the original computation script produced meaningless matrices.
   See [`docs/GEOMETRY_VALIDITY_STATUS.md`](docs/GEOMETRY_VALIDITY_STATUS.md).

---

## What was invalidated

The following files in `archive/invalidated_geometry/` must not be used
as public evidence:

- `results/generation_1/fisher_metrics/*.npy` — buggy Jacobian
- `results/generation_1/regularized_fisher/*` — derived from buggy metrics
- Claims of 2D manifold dimensionality
- Claims of 5–10× geodesic shortening
- Claims of ~100% VAE reconstruction validity

---

## How to Reproduce

**Prerequisites:**

```bash
pip install -r requirements.txt
```

**Run all 7 stages:**

```bash
python -m src.reproduce --config configs/demo.yaml --stages 1,2,3,4,5,6,7
```

This will create a timestamped folder under `results/reproduction_<timestamp>/`
with a full manifest, validation reports, geometry baseline, and final claim
verdict. No root files are modified.

**Stage 6 (empirical stress test)** requires a real market data CSV. Provide it
in `configs/demo.yaml` under `paths.market_data_csv`. Without it, Stage 6
returns `NEEDS_REAL_DATA` (expected behaviour, not a failure).

**Run tests:**

```bash
python -m pytest tests -q
```

Expected: **210 tests pass**.

---

## Repository Structure

```
.
├── configs/demo.yaml          ← Reproduction config
├── docs/
│   ├── CLAIM_VERDICT_MATRIX.md   ← Claim-by-claim audit
│   ├── FINDINGS_SIMPLE.md        ← Plain-language summary
│   ├── GEOMETRY_VALIDITY_STATUS.md ← Fisher metric invalidity
│   ├── RESEARCH_INTEGRITY.md     ← No-fake-data contract
│   ├── REPRODUCTION_PLAN.md      ← Stage specifications
│   └── WEIGHTS_GOVERNANCE.md     ← Checkpoint policy
├── models/
│   ├── vae_beta50.pth            ← Public demo checkpoint (default in configs/demo.yaml)
│   └── vae_model_weights.pth     ← Retained legacy checkpoint (historical baseline)
├── requirements.txt
├── src/
│   ├── config.py             ← Config schema (Pydantic)
│   ├── data_generator.py     ← Synthetic data generator
│   ├── geometry.py           ← Corrected Fisher geometry
│   ├── io_utils.py           ← Fail-fast file utilities
│   ├── reproduce.py          ← 7-stage reproduction runner
│   ├── vae.py                ← VAE architecture
│   ├── validation.py         ← Structural validation
│   └── vector_schema.py      ← Parameter layout schema
└── tests/                    ← 210 tests across all modules
```

Legacy code and invalidated results are in `archive/` (see
[`docs/ARCHIVE_PLAN.md`](docs/ARCHIVE_PLAN.md)).

---

## Research Integrity

This project enforces a strict **No Fake Data Contract**
([`docs/RESEARCH_INTEGRITY.md`](docs/RESEARCH_INTEGRITY.md)):

- No random arrays used as fallback for missing files.
- No fake or synthetic market data.
- All generated artifacts are SHA-256 tracked in the reproduction manifest.
- Negative results (invalidated claims) are preserved and documented.
- The config loader rejects any config that enables random fallbacks.

---

## Current Limitations

1. **VAE decoder is unconstrained.** It does not enforce disjoint discrete
   indicator activation, causing `LEGACY_LAYOUT_COLLISION` in 100% of
   reconstructions.
2. **Data generator has a GARCH boundary bug.** Negative beta values are
   produced when `sum_alpha > 0.4`, violating non-negativity constraints.
3. **Stage 1 valid rate is only 62.5%.** A clean v2 generator would fix this.
4. **Corrected geometry is a baseline only.** It cannot support manifold
   claims until the decoder produces valid outputs.
5. **Empirical claim is unresolved.** Stage 6 requires real market data.

---

## Open Questions and Future Research Directions

The following questions define the natural Phase 2 research agenda.
They are framed as open questions — not proven capabilities.

**Decoder architecture:**
- Can a constrained decoder (e.g., with a hard-argmax or Gumbel-softmax
  indicator head) prevent `LEGACY_LAYOUT_COLLISION` entirely?
- What projection layer design would enforce disjoint parameter layout
  from a continuous latent variable?

**Data generation:**
- Can a clean v2 generator eliminate invalid GARCH and AR stationarity
  failures, bringing Stage 1 input validity to 100%?
- How does input validity rate affect encoder clustering quality?

**Interpolation validity:**
- Can latent interpolation produce structurally valid hybrid ARMA-GARCH
  structures once the decoder is constrained?
- Is the "ARMA bridge" concept recoverable under a constrained architecture?

**Geometry:**
- Can corrected geometry identify validity boundaries in latent space?
- Do valid-input latents form a lower-dimensional manifold distinct from
  invalid-input latents?

**Empirical evaluation:**
- What happens when clean real market data is provided under provenance rules?
- Is the no-predictive-edge result robust to different market periods and
  asset classes?

**Transfer:**
- Can the same VAE framework transfer to other constrained parametric model
  families (e.g., stable filters, DSGE models, control systems)?
- How does the approach generalise to higher-dimensional parameter spaces?

---

## Citation / Provenance Notes

This repository is a research reproducibility and research integrity project.

The original research (Generation 1) was conducted in 2025. The reproduction
runner and claim audit were conducted in 2026. The original article source is
retained in `archive/original_research/` for reference, but should not be cited
without the caveat that multiple claims were invalidated by the clean
reproduction.

The latest canonical reproduction anchor is:

```
results/reproduction_20260620_085314/reports/final_claim_verdict.json
```

All public claims should trace back to the outputs of that run.
