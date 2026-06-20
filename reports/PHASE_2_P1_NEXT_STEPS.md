# Phase 2 P1: Next-Step Plan

This document outlines the implementation plan and protocols for Phase 2 of the `constrained-latent-time-series` project. It lists the requirements for freezing parameters, files to be created/preserved, blockers, scope, and quality gates.

---

## 1. What Must Be Frozen Before Implementation

To ensure scientific and research integrity, the following components must be frozen and kept read-only before any code changes are implemented:

* **Mathematical Validation Rules**:
  - The stationarity checks for AR coefficients (roots of characteristic polynomial strictly outside the unit circle).
  - GARCH constraints: $\omega > 0, \alpha_i \ge 0, \beta_j \ge 0$, and $\sum \alpha_i + \sum \beta_j < 1$.
  - Threshold values: The validation threshold of 0.5 and epsilon buffer of 0.05 for structure classification.
* **Seeds**:
  - The random seed `42` for data generation and model training must remain the baseline default.
* **Baseline Reference Files**:
  - The legacy model checkpoints (`models/vae_beta50.pth` and `models/vae_model_weights.pth`) must remain untouched to allow comparison runs.
  - The original vector schema constants (`VECTOR_DIM = 40`, slices, index meanings) for the legacy models.

---

## 2. Which Files Should Be Created in Phase 2A

The following new files should be introduced to support the constrained architecture without corrupting the Phase 1 reproduction files:

* **`src/vector_schema_v2.py`**:
  - Defines the corrected, non-overlapping vector schema. It must isolate ARMA coefficients from GARCH coefficients, resolving the collision at `v[10]`.
* **`src/data_generator_v2.py`**:
  - Implements the corrected GARCH parameter generator (ensuring GARCH persistence $< 1$ and non-negative coefficients without swapped uniform bounds).
  - Can optionally generate hybrid ARMA-GARCH models under the new schema.
* **`src/vae_constrained.py`**:
  - Implements a constrained VAE decoder architecture. The decoder must include a Softmax or Gumbel-Softmax head on the model-family indicator slice to enforce mutual exclusivity, and projection layers to map parameter outputs to stable, stationary bounds.
* **`src/train_vae_v2.py`**:
  - Config-driven training pipeline that enforces strict input checks, validation splits, and logs the SHA-256 hash of the generated model weights.
* **`configs/phase2_constrained.yaml`**:
  - Dedicated configuration file for Phase 2 training and validation.
* **`tests/test_phase2_*.py`**:
  - Dedicated tests for the new generator, schema, and constrained VAE decoder.

---

## 3. Which Files Should NOT Be Touched

The following files must be kept read-only to preserve Phase 1 reproducibility:

* [src/vae.py](file:///c:/Users/wekab/Music/constrained-latent-time-series-main/constrained-latent-time-series-main/src/vae.py): Contains the baseline unconstrained VAE architecture.
* [src/data_generator.py](file:///c:/Users/wekab/Music/constrained-latent-time-series-main/constrained-latent-time-series-main/src/data_generator.py): Contains the legacy-compatible data generator.
* [src/reproduce.py](file:///c:/Users/wekab/Music/constrained-latent-time-series-main/constrained-latent-time-series-main/src/reproduce.py): Contains the 7-stage reproduction runner.
* `models/` directory: All existing `.pth` checkpoints.
* `tests/test_reproduce_*.py`: Pytest files checking Phase 1 reproducibility.

---

## 4. What Blockers Exist Before Generator v2

Before implementing the `data_generator_v2.py` script:
1. **Resolve parameter layout collision**: We must define a new, non-overlapping layout vector schema. A larger vector size (e.g. 50-dimensional) or dynamic offset indexing is required to house AR coefficients, MA coefficients, GARCH parameters, and stats without overlaps.
2. **Define strict sampling boundaries**: Formally specify the GARCH parameter sampling ranges to guarantee covariance-stationarity ($100\%$ input validity rate).

---

## 5. What Blockers Exist Before Any Training

Before training a new model:
1. **Enforce config discipline**: The Pydantic configuration loader (`src/config.py`) must be updated to raise an error if any parameter is missing from the configuration YAML file (no silent default fallbacks).
2. **Formulate train/validation split contracts**: Define a split strategy to protect against data leakage.

---

## 6. Recommended Prompt 2 Scope

The next task (Prompt 2) should cover:
1. Creation of the new layout schema (`src/vector_schema_v2.py`).
2. Implementation of the clean generator (`src/data_generator_v2.py`).
3. Implementation of the constrained decoder (`src/vae_constrained.py`).
4. Unit testing of the new generator and decoder constraints.

---

## 7. Recommended Acceptance Criteria for Prompt 2

* **Generator Validity**: The `data_generator_v2.py` must yield $100\%$ valid parameter vectors (AR stationarity, MA invertibility, and GARCH constraints checked via `validation.py`).
* **Architectural Guarantees**: The new constrained decoder model must guarantee disjoint family activations by design (no overlapping indicators, $0\%$ layout collisions).
* **Test Coverage**: $100\%$ of new unit tests in `tests/test_phase2_*.py` must pass.

---

## 8. Explicit List of "NO-GO Until Fixed" Issues

The following gates are strictly enforced:
* **NO-GO**: Do not train any new models until the layout collision at `v[10]` is resolved.
* **NO-GO**: Do not write any code that introduces silent configuration defaults. If a key is missing from the YAML file, the model must fail immediately.
* **NO-GO**: Do not delete or overwrite the historical VAE checkpoints.

---

## VERDICT

**VERDICT: READY_FOR_PHASE_2_PROTOCOL**

The roadmap and next steps are fully specified, providing a clear path forward for implementation.

- **Changed files**:
  - [reports/PHASE_2_P1_NEXT_STEPS.md](file:///c:/Users/wekab/Music/constrained-latent-time-series-main/constrained-latent-time-series-main/reports/PHASE_2_P1_NEXT_STEPS.md)
- **Commands run**:
  - None.
- **Tests run**:
  - None.
- **Evidence paths inspected**:
  - `configs/demo.yaml`
  - `src/config.py`
  - `src/reproduce.py`
  - `reports/PHASE_2_P1_REPO_INVENTORY.md`
  - `reports/PHASE_2_P1_FAILURE_MAP.md`
- **Known limitations**:
  - None.
- **Exact next recommended action**:
  - Wait for user and reviewer approval of the Phase 2 Protocol and reports.
