# PHASE 2 CONFIG DISCIPLINE
# No-Defaults Policy, Required Explicit Values, Fail-Fast Rules

**Document version:** 1 (initial)
**Created:** 2026-06-20
**Branch:** phase2/p2-protocol-and-schema-design
**Status:** DRAFT — awaiting reviewer approval

---

## 1. No Hidden Defaults Policy

Every research-affecting value used in Phase 2 experiments must be:
1. **Explicit** — stated in the Phase 2 config file
2. **Documented** — described in the protocol (this document or the linked metric/split contracts)
3. **Frozen before use** — committed to the config file before the experiment that uses it begins
4. **Auditable** — traceable through the artifact manifest to the exact value used in any run

A value is **research-affecting** if changing it would alter any of:
- Which samples are generated or included in training/evaluation
- Whether a sample is labeled as valid or invalid
- Which family a sample is classified as belonging to
- Whether a model checkpoint is accepted or rejected
- Whether a run passes or fails the split contract
- Whether a run's artifacts are complete or incomplete

If any research-affecting value is absent from the config file, the system must FAIL_FAST —
it must not substitute a default, silently proceed, or log a warning.

---

## 2. No Hardcoded Research Values Policy

No research-affecting value may be hardcoded in source code as:
- A module-level constant (e.g. `D_LATENT = 8` in `src/vae.py`)
- A class-level constant (e.g. `D_HIDDEN1: int = 64` in a model class)
- A Pydantic field default (e.g. `seed: int = 42` in a config class)
- A dataclass field default (e.g. `n_arma: int = 100` in a generator config)
- A function parameter default (e.g. `tol: float = 1e-8`, `threshold: float = 0.5`)
- An argparse default (e.g. `parser.add_argument('--epochs', default=100)`)

All of these patterns exist in Phase 1 code and are documented as **Phase 1 legacy baseline values**.
They must not be inherited by Phase 2 code.

If a constant is required for technical reasons (e.g. buffer size, string literals), it must be
annotated as non-research-affecting with an explicit comment explaining why.

---

## 3. Every Research-Affecting Value Must Be Explicit

### Required in Phase 2 Config (Complete List)

The following values must be present in the Phase 2 config file. If any is absent, FAIL_FAST.

```
# Data generation
generation.seed_a: <int>               # Seed for A-family generation — no default
generation.seed_b: <int>               # Seed for B-family generation — no default
generation.seed_c: <int>               # Seed for C-family generation — no default
generation.n_arma: <int>               # A-family sample count — no default
generation.n_garch: <int>              # B-family sample count — no default
generation.n_arma_garch: <int>         # C-family sample count — no default
generation.mode: <str>                 # Must be "clean_validated" for Phase 2; "legacy_compatible" is rejected
generation.allow_supervised_arma_garch: <bool>   # Must be explicitly set; no silent default

# Schema
schema.family_vocabulary: <list[str]>  # Explicit list of valid family_id values
schema.max_p: <int>                    # Maximum AR order — no default; must be > 0
schema.max_q: <int>                    # Maximum MA order — no default; must be ≥ 0
schema.max_r: <int>                    # Maximum ARCH order — no default; must be > 0
schema.max_s: <int>                    # Maximum GARCH order — no default; must be ≥ 0
schema.vector_dim: <int>               # Total dimension of the Phase 2 parameter vector — no default
schema.mean_component_offset: <int>    # Start index of mean parameter block — no default
schema.mean_component_length: <int>    # Length of mean parameter block — no default
schema.volatility_component_offset: <int>  # Start index of volatility parameter block — no default
schema.volatility_component_length: <int>  # Length of volatility parameter block — no default

# Model
model.architecture: <str>              # Named architecture; must match a registered model factory
model.latent_dim: <int>                # Latent space dimension — no default; "latent_dim=8" is a P1 legacy value
model.hidden_dims: <list[int]>         # Hidden layer widths — no default; "[64, 32]" is a P1 legacy value
model.input_dim: <int>                 # Must equal schema.vector_dim

# Training
training.seed: <int>                   # Training seed — no default; "seed=42" is a P1 legacy value
training.epochs: <int>                 # Training epochs — no default
training.batch_size: <int>             # Batch size — no default
training.learning_rate: <float>        # Learning rate — no default
training.beta: <float>                 # KL weight (β-VAE parameter) — no default; "beta=1.0" is a P1 legacy value

# Validation / metric contract
validation.tolerance: <float>          # Numerical tolerance for root checks — no default; "1e-8" is a P1 legacy value
validation.persistence_tol: <float>    # Persistence margin below 1 — no default
validation.composition_mean_threshold: <float>  # Min |φᵢ| for composition score — no default
validation.composition_vol_threshold: <float>   # Min αᵢ for composition score — no default
validation.activation_threshold: <float>        # If used: must be explicit; must not be "0.5" P1 legacy default
validation.activation_epsilon: <float>          # If used: must be explicit; must not be "0.05" P1 legacy default

# Metric thresholds (frozen before training)
metric_contract.generated_C_valid_rate_threshold: <float>      # Must be APPROVED before training
metric_contract.generated_C_composition_score_threshold: <float>
metric_contract.generated_C_novelty_score_threshold: <float>
metric_contract.seed_stability_range_threshold: <float>
metric_contract.min_seeds: <int>

# Split contract
split.zero_shot_c_count_required: 0   # Must be exactly 0; no deviation allowed
split.split_manifest_path: <str>       # Path to the pre-committed split manifest
split.split_manifest_hash: <str>       # SHA-256 hash of the split manifest — must match at runtime

# Artifact discipline
integrity.allow_cached_artifacts: false    # MUST be explicit; MUST be false for Phase 2
integrity.allow_random_fallbacks: false    # MUST be explicit; MUST be false
integrity.allow_fake_market_data: false    # MUST be explicit; MUST be false
integrity.require_artifact_provenance: true
integrity.fail_fast_on_missing_required_artifacts: true

# Paths
paths.output_dir: <str>               # Run output directory — no default
paths.checkpoint_path: <str>          # Model checkpoint path (if loading) — no default; null is allowed only if no checkpoint is used

# Geometry (if geometry experiments are included in Phase 2)
geometry.max_geometry_points: <int>   # Must be explicit; "32" is a P1 Pydantic default that must not be silently inherited
geometry.finite_difference_eps: <float>   # Must be explicit; "1e-4" is a P1 Pydantic default
geometry.metric_regularization_eps: <float>  # Must be explicit; "1e-8" is a P1 Pydantic default
```

---

## 4. Missing Key Must Fail-Fast

If any key from §3 is absent from the Phase 2 config file:

1. The config loader must raise an error before any computation begins
2. The error message must identify the missing key by name
3. The error must not be suppressed or caught silently
4. The run must not proceed in any mode (demo, test, or production)

**Forbidden fallback patterns:**
```python
# FORBIDDEN — silently uses a default
value = config.get("seed", 42)

# FORBIDDEN — Pydantic default
seed: int = 42

# FORBIDDEN — argparse default
parser.add_argument("--seed", type=int, default=42)

# FORBIDDEN — conditional default
if "seed" not in config:
    seed = 42  # use 42 as fallback

# REQUIRED — fail-fast
if "seed" not in config:
    raise KeyError("FAIL_FAST: 'generation.seed' is required in Phase 2 config but is absent")
```

---

## 5. Function-Level Defaults That Affect Research Results Are Forbidden

The following Phase 1 function signatures contain defaults that affect research results.
These defaults are **forbidden in Phase 2 callers** — all Phase 2 code must pass explicit values.

### From `src/validation.py`

```python
# Phase 1 signatures (FORBIDDEN to call without explicit args in Phase 2)
def is_stationary_ar(phi, tol=1e-8):         # tol must be explicit
def is_invertible_ma(theta, tol=1e-8):       # tol must be explicit
def is_valid_garch(omega, alpha, beta, tol=1e-8):  # tol must be explicit
def decode_discrete_structure(v, threshold=0.5, epsilon=0.05):  # both must be explicit
```

Phase 2 callers must always call these as:
```
is_stationary_ar(phi, tol=config.validation.tolerance)
decode_discrete_structure(v, threshold=config.validation.activation_threshold,
                             epsilon=config.validation.activation_epsilon)
```

If the Phase 2 schema abandons `decode_discrete_structure` in favor of explicit family IDs
(per `docs/PHASE_2_SCHEMA_SPEC.md` R1), this function may not be needed — but if used, it must
receive explicit values.

### From `src/data_generator.py`

```python
# Phase 1 dataclass defaults (FORBIDDEN for Phase 2)
n_arma: int = 100
n_garch: int = 100
seed: int = 42
```

Phase 2 generator config must require these as explicit fields with no default.

---

## 6. Pydantic/Dataclass Defaults That Affect Research Results Are Forbidden

The following Phase 1 Pydantic defaults must not be silently inherited by Phase 2:

| Class | Field | P1 Default | Action in Phase 2 |
|-------|-------|-----------|-------------------|
| `AppConfig` | `seed` | 42 | Must be required field with no default |
| `IntegrityConfig` | `allow_cached_artifacts` | True | Must be `false` and required |
| `GeometryConfig` | `max_geometry_points` | 32 | Must be required field in Phase 2 config |
| `GeometryConfig` | `finite_difference_eps` | 1e-4 | Must be required field |
| `GeometryConfig` | `metric_regularization_eps` | 1e-8 | Must be required field |
| `GenerationConfig` | `n_arma` | 100 | Must be required field |
| `GenerationConfig` | `n_garch` | 100 | Must be required field |
| `GenerationConfig` | `seed` | 42 | Must be required field |

The Phase 2 config schema (to be implemented in P3) must not define any Pydantic default for any
field in this list. If using Pydantic for Phase 2 config, use `...` (no default = required field)
or `Field(...)` (required) for all research-affecting values.

---

## 7. argparse Defaults That Affect Research Results Are Forbidden

`src/train_vae_model.py` and `src/reproduce.py` contain argparse defaults.
Phase 2 training must not use `src/train_vae_model.py` (it bypasses config discipline).
If a Phase 2 CLI is built, all research-affecting arguments must be required (no `default=`).

The only acceptable argparse default is the config file path itself (as a convenience):
```
parser.add_argument("--config", type=str, required=True)   # REQUIRED — no default path acceptable for Phase 2
```

---

## 8. Legacy Values Documented as P1 Baseline Values Only

The following values appear in Phase 1 code and must not be treated as Phase 2 defaults.
They are documented here as Phase 1 legacy baselines for reference only.

| Value | Location | P1 Role | Phase 2 Status |
|-------|----------|---------|----------------|
| `seed = 42` | `src/config.py:87`, `src/data_generator.py:68`, `src/train_vae_model.py:68` | Default seed | LEGACY — must not be inherited |
| `threshold = 0.5` | `src/validation.py:358` | Family detection threshold | LEGACY — must be explicit in config |
| `epsilon = 0.05` | `src/validation.py:359` | Threshold buffer | LEGACY — must be explicit in config |
| `latent_dim = 8` | `src/vae.py:42`, `src/train_vae_model.py:11` | Model capacity | LEGACY — must be explicit in config |
| `d_hidden1 = 64` | `src/vae.py:43` | Encoder/decoder width | LEGACY — must be explicit in config |
| `d_hidden2 = 32` | `src/vae.py:44` | Encoder bottleneck width | LEGACY — must be explicit in config |
| `n_arma = 100` | `src/data_generator.py:66`, `src/generate_enriched_training_data.py:47` | A-family sample count | LEGACY — must be explicit in config |
| `n_garch = 100` | `src/data_generator.py:67` | B-family sample count | LEGACY — must be explicit in config |
| `tol = 1e-8` | `src/validation.py:231,270,309` | Numerical tolerance | LEGACY — must be explicit in config |
| `beta = 1.0` | `src/train_vae_model.py:69` | KL weight | LEGACY — must be explicit in config |
| `epochs = 100` | `src/train_vae_model.py:67` | Training epochs | LEGACY — must be explicit in config |
| `max_geometry_points = 32` | `src/config.py:147` | Geometry sample count | LEGACY Pydantic default — must be explicit |
| `finite_difference_eps = 1e-4` | `src/config.py:148` | Jacobian step | LEGACY Pydantic default — must be explicit |
| `metric_regularization_eps = 1e-8` | `src/config.py:149` | Metric conditioning | LEGACY Pydantic default — must be explicit |
| `allow_cached_artifacts = True` | `src/config.py:124` | Provenance policy | LEGACY — must be `false` in Phase 2 |

---

## 9. Required Explicit Config Value: integrity.allow_cached_artifacts

The following value must be present in the Phase 2 config and must be set to `false`:

```yaml
integrity:
  allow_cached_artifacts: false
```

This is not a default. This is not negotiable. It must be explicit.

**Rationale:** In Phase 1, `allow_cached_artifacts: true` was the default (and the Pydantic default).
This means a Phase 1 run might have silently used cached artifacts from a previous run without
re-generating them. In Phase 2, every run must generate all artifacts fresh. A cached artifact
is an unverified artifact. An unverified artifact cannot be cited as evidence.

---

## 10. Missing Config Key Behavior: FAIL_FAST_REQUIRED

The following is the required behavior for any missing Phase 2 config key:

```
FAIL_FAST_REQUIRED

If any required config key is absent:
1. Raise an error immediately
2. Log the missing key name
3. Do not start data generation
4. Do not start training
5. Do not load a checkpoint
6. Do not run validation
7. Do not write any artifacts
8. Exit with a non-zero return code

A warning is not acceptable. A default substitution is not acceptable.
```

---

## 11. Environment Freeze Requirement

The Phase 2 config discipline applies to the software environment as well as config values.
Before any Phase 2 training run:

1. `pip freeze > requirements-phase2-freeze.txt` must be run and the file committed
2. The Python version must be documented in the freeze file header
3. The torch version must be verified as installed and functional
4. The freeze file hash must be included in the run manifest
5. Any difference between `requirements-phase2-freeze.txt` and the environment at run time
   must cause FAIL_FAST

This requirement supersedes the Phase 1 `requirements.txt` which only specifies minimum version bounds.
