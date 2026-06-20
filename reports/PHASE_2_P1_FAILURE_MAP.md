# PHASE_2_P1_FAILURE_MAP.md
# Phase 2 P1: Forensic Failure Map

**Report version:** 2 (post-review fix)
**Date:** 2026-06-20
**Branch:** phase2/p1-repo-inventory
**Commit:** 67f06e142bc0cab191ef22067f84e74e3c8755fe

> [!WARNING]
> All findings in this report are based on **static code audit** unless explicitly stated otherwise.
> Runtime behavior of VAE encode/decode/training cannot be verified because `torch` is not installed
> in the local environment. All such findings are marked **MISSING_EVIDENCE (runtime)**.

---

## Hardcoded/Default Config Audit Evidence

The following search was run to locate all hardcoded constants, silent defaults, and magic values:

```powershell
Command (PowerShell — ripgrep not available):
  Select-String -Path "src\*.py","tests\*.py","configs\*.yaml","pyproject.toml","README.md" `
    -Pattern "default=|seed|latent_dim|d_latent|hidden|threshold|epsilon|n_arma|n_garch|output_dir|
              checkpoint_path|allow_cached|np\.save|torch\.save|argparse|dataclass|BaseModel" `
    -CaseSensitive:$false

Note: ripgrep (rg) was not available in this environment. Select-String used instead.
```

Selected findings from `src/` only (full output stored in PHASE_2_P1_REVIEW_FIX_NOTES.md):

```
config.py:87:    seed: int = 42
config.py:124:   allow_cached_artifacts: bool = True
config.py:147:   max_geometry_points: int = Field(default=32, gt=0)
config.py:148:   finite_difference_eps: float = Field(default=1e-4, gt=0.0)
config.py:149:   metric_regularization_eps: float = Field(default=1e-8, gt=0.0)
config.py:165:   geometry: GeometryConfig = GeometryConfig()    ← entire block silently defaulted

data_generator.py:66:   n_arma: int = 100
data_generator.py:67:   n_garch: int = 100
data_generator.py:68:   seed: int = 42

generate_enriched_training_data.py:47:   n_arma = 100    ← bare literal hardcode
generate_enriched_training_data.py:77:   np.save('Z_train_enriched.npy', ...)   ← root-relative hardcoded path
generate_enriched_training_data.py:78:   np.save('train_labels_enriched.npy', ...)

reproduce.py:39:   default="configs/demo.yaml"   ← argparse default config path

train_vae_model.py:11:   D_LATENT = 8
train_vae_model.py:64:   parser.add_argument('--train_path', default='Z_train.npy')
train_vae_model.py:65:   parser.add_argument('--labels_path', default='train_labels.npy')
train_vae_model.py:66:   parser.add_argument('--save_path', default='vae_model_weights.pth')
train_vae_model.py:67:   parser.add_argument('--epochs', default=100)
train_vae_model.py:68:   parser.add_argument('--seed', default=42)
train_vae_model.py:69:   parser.add_argument('--beta', default=1.0)
train_vae_model.py:95:   torch.save(model.state_dict(), args.save_path)

vae.py:42:   D_LATENT: int = 8
vae.py:43:   D_HIDDEN1: int = 64
vae.py:44:   D_HIDDEN2: int = 32
vae.py:85:   d_hidden1: int = D_HIDDEN1
vae.py:86:   d_hidden2: int = D_HIDDEN2

validation.py:231:   tol: float = 1e-8      (is_stationary_ar)
validation.py:270:   tol: float = 1e-8      (is_invertible_ma)
validation.py:309:   tol: float = 1e-8      (is_valid_garch)
validation.py:358:   threshold: float = 0.5  (decode_discrete_structure)
validation.py:359:   epsilon: float = 0.05   (decode_discrete_structure)
```

---

## 1. Structural Validity Failure

### 1.1 VAE Decoder Produces Unconstrained Numeric Output

**Location:** `src/vae.py` — `VAE.decode()` → `self.decoder` → final `nn.Tanh()`

The VAE decoder maps latent vectors `z ∈ ℝ^8` to parameter vectors `x_hat ∈ [-1, 1]^40` via a fully-connected MLP with a Tanh output. There is no layer, projection, or penalty that enforces:

- AR polynomial roots strictly inside the unit circle (stationarity)
- GARCH non-negativity constraints (ω > 0, αᵢ ≥ 0, βⱼ ≥ 0)
- GARCH persistence strictly below 1 (Σα + Σβ < 1)
- Disjoint indicator activation (ARMA vs GARCH bits cannot both be strongly active)

**Observed consequence (Phase 1 reproduction — static evidence from docs/):**
- Stage 4 reconstruction validity: 0/200 = **0.0%**
- Failure mode: `LEGACY_LAYOUT_COLLISION` — both ARMA and GARCH indicators weakly active, creating an undefined joint parameter layout
- Source: `docs/evidence/publication_summary.json`, `docs/CLAIM_VERDICT_MATRIX.md`

**MISSING_EVIDENCE (runtime):** Direct runtime verification of the forward pass requires torch.

### 1.2 Implicit Family Type via Threshold Comparison

**Location:** `src/validation.py:356-460` — `decode_discrete_structure()`

Family membership (ARMA, GARCH, ARMA-GARCH, UNKNOWN) is determined by comparing continuous logit scores against a threshold (default 0.5) with an epsilon buffer (default 0.05). These parameters are:
- Function-level defaults (`threshold: float = 0.5, epsilon: float = 0.05`)
- NOT validated to match the config values in `configs/demo.yaml`
- NOT enforced to be consistent across modules

**Risk:** Different callers can silently override the threshold, creating inconsistent family classifications across modules.

---

## 2. Layout Collision Risk

### 2.1 Critical Ambiguity at Vector Index 10

**Location:** `src/vector_schema.py:16-26` — documented in module docstring

```
v[10] = φ₁  (first AR coefficient, for ARMA models)
v[10] = ω   (GARCH intercept, for GARCH models)
```

This is a **context-dependent overload**: the same memory slot carries a different mathematical meaning depending on which family indicator is active. The ambiguity is explicitly documented in `src/vector_schema.py`:

> "This means `v[10]` represents `φ₁` in pure ARMA, but `ω` in pure GARCH. [...] the layout for a combined ARMA-GARCH parameter vector is formally UNDEFINED."

**Consequence:** No valid ARMA-GARCH vector can be represented under the current schema. Any VAE interpolation between ARMA and GARCH endpoint latents that decodes to a vector with both indicators active will have a collision at `v[10]`.

### 2.2 Non-Canonical Layout in Enriched Generator

**Location:** `src/generate_enriched_training_data.py:57-69`

This script writes ARMA vectors using a non-standard layout:
```python
x[0] = 1   # ARMA active (binary, not logit)
x[1] = 1   # MA active (binary, not logit)
x[2] = 0   # GARCH inactive
x[3] = p   # raw integer order (not logit in [-1,1])
x[4] = q   # raw integer order (not logit in [-1,1])
x[10:10+p] = phi    # AR coefficients
x[20:20+q] = theta  # MA coefficients — places MA at different offset than GARCH
```

This diverges from the canonical schema (`ARMA_ACTIVE_IDX=0`, `GARCH_ACTIVE_IDX=1`, continuous logits at indices 2-5, params at 10:30 with family-dependent meaning). If these enriched vectors are fed into the canonical VAE, the misalignment creates silent numerical errors.

---

## 3. C Leakage Risk — Corrected Framing

> [!IMPORTANT]
> **Correction from v1 of this report:** The previous version incorrectly described ARMA-only examples as "C leakage." This framing was wrong under the Phase 2 benchmark intent.

### Correct Phase 2 family definitions:
- **A = AR / ARMA** (mean dynamics only)
- **B = GARCH** (volatility dynamics only)
- **C = ARMA-GARCH** (combined mean + volatility hybrid)

### Correct C-leakage definition:
C-leakage occurs when ARMA-GARCH or another explicit mean+volatility hybrid model appears in the zero-shot training set. This would invalidate the zero-shot generalization claim.

### Current status:
- The canonical generator (`src/data_generator.py`) generates **only A and B** — no C-leakage from this generator.
- The enriched generator (`src/generate_enriched_training_data.py`) generates **ARMA models** — these belong to the A-family or A-subfamily, **not C**.
- **True C-leakage risk:** There is **no formal split contract** and **no schema-level family guard** to verify that C-family models never appear in training. The risk is the absence of a guard, not confirmed C-leakage.

---

## 4. Hardcoded/Default Config Risk — Full Catalogue

The following table lists every research-affecting default found in the codebase. These are documented as **legacy baseline values**. Phase 2 must not silently inherit any of them — every value must be explicit in the Phase 2 protocol config and must fail-fast if absent.

| Location | Default | Research Impact | Phase 2 Action Required |
|---|---|---|---|
| `src/config.py:87` | `seed: int = 42` | Data generation | Must be explicit in Phase 2 config; no fallback |
| `src/config.py:124` | `allow_cached_artifacts: bool = True` | Provenance | Must be `false` by default in Phase 2 |
| `src/config.py:147` | `max_geometry_points = 32` | Geometry sampling | Must be explicit in Phase 2 config |
| `src/config.py:148` | `finite_difference_eps = 1e-4` | Jacobian accuracy | Must be explicit in Phase 2 config |
| `src/config.py:149` | `metric_regularization_eps = 1e-8` | Metric conditioning | Must be explicit in Phase 2 config |
| `src/config.py:165` | `geometry: GeometryConfig = GeometryConfig()` | Silently omits geometry block | Phase 2 config must require geometry section |
| `src/data_generator.py:66-68` | `n_arma=100, n_garch=100, seed=42` | Dataset size and seed | Must be config-driven, no dataclass default |
| `src/validation.py:231,270,309` | `tol: float = 1e-8` | Validity boundaries | Must match config; no function-level default |
| `src/validation.py:358-359` | `threshold=0.5, epsilon=0.05` | Family classification | Must be read from config; no function-level default |
| `src/vae.py:42-44` | `D_LATENT=8, D_HIDDEN1=64, D_HIDDEN2=32` | Model capacity | Must be explicit in Phase 2 config |
| `src/train_vae_model.py:64-69` | `epochs=100, seed=42, beta=1.0`, root-relative paths | Training behavior | This script should not be used in Phase 2 |
| `src/reproduce.py:39` | `default="configs/demo.yaml"` | Config path | Acceptable as CLI convenience; must be documented |
| `src/generate_enriched_training_data.py:47` | `n_arma = 100` (bare literal) | Dataset size | Script not in pipeline; annotate as non-canonical |

---

## 5. Metric Drift Risk

### 5.1 Validation Parameters as Function Defaults

`decode_discrete_structure()` in `src/validation.py` accepts `threshold` and `epsilon` as function parameters with defaults. Different callers can pass different values, and there is no mechanism to ensure they match `configs/demo.yaml validation.activation_threshold` and `validation.activation_epsilon`.

**Risk:** Post-hoc threshold tuning is architecturally possible. A researcher could call `decode_discrete_structure(vector, threshold=0.3)` and see a different family classification result without any audit trail.

**Mitigation needed:** Phase 2 must route all validation calls through the config-loaded values; function-level defaults must be removed or guarded.

### 5.2 Success Criteria Not Frozen Pre-Experiment

The existing pipeline has no mechanism to "freeze" success criteria before running experiments. Stage 7 generates a verdict after seeing Stage 4-6 results, which means thresholds could in principle be adjusted after results are known.

**Mitigation needed:** Phase 2 protocol must specify that all metric contracts and acceptance thresholds are committed to the repository before any data generation or model training begins.

---

## 6. Reproducibility Failure

### 6.1 P0 Jacobian Bug (Archived — Historical)

**Location:** `archive/invalidated_geometry/compute_fisher_metric.py`

The legacy Fisher metric script contained a mathematical error: instead of computing the full Jacobian matrix J ∈ ℝ^(d_theta × d_z) and forming the pullback metric G = J^T J, the computation used a 1D operation that produced scalar or rank-1 results rather than a valid d_z × d_z metric matrix.

**Consequence:** All legacy manifold geometry claims (2D dimensionality, geodesic shortening) are invalid.
**Status:** Archived; corrected implementation now in `src/geometry.py`.

### 6.2 Missing Lock File

`requirements.txt` specifies minimum version bounds but not exact installed versions. The actual environment at reproduction time may differ.

**MISSING_EVIDENCE:** `pip freeze` output not captured and not committed.

### 6.3 `code_git_commit` Hardcoded as `None`

**Location:** `src/reproduce.py:118` — `build_initial_manifest()`

```python
"code_git_commit": None,  # Not tracked in skeleton
```

The reproduction manifest does not auto-capture the git commit hash. This means any reproduction run cannot be definitively linked to a specific codebase state via the manifest alone.

### 6.4 GARCH Boundary Bug in `legacy_compatible` mode

**Location:** `src/data_generator.py:169-171`

```python
high_bound = 0.4 - sum_alpha
low, high = min(0.0, high_bound), max(0.0, high_bound)
beta = rng.uniform(low, high, size=s_order)
```

When `sum_alpha > 0.4`, `high_bound` is negative, causing `low=high_bound, high=0.0`. This reproduces the original bug (negative beta values, persistence > 1). **This is intentional in `legacy_compatible` mode** for checkpoint compatibility but means 37.5% of generated inputs are structurally invalid.

---

## 7. Result-Chasing Risk

### 7.1 Post-Hoc Threshold Tuning Architecturally Possible

As noted in §5.1, the `decode_discrete_structure()` function accepts threshold and epsilon as call-site parameters. Researchers can select thresholds after seeing results.

### 7.2 Standalone Training Script Bypasses Protocol

`src/train_vae_model.py` is an independent script that can train and save a VAE model without running Stages 0-3 of the reproduction runner. It writes directly to hardcoded root paths (`vae_model_weights.pth`) without config validation, provenance tracking, or split guards.

---

## 8. Training-Before-Protocol Risk

**Location:** `src/train_vae_model.py`

This script can be invoked with `python src/train_vae_model.py` and will:
1. Load `Z_train.npy` from the current directory (hardcoded default path)
2. Train a VAE model for 100 epochs (hardcoded default)
3. Save weights to `vae_model_weights.pth` (hardcoded default)

None of the following prerequisites are checked:
- Generator validity (Stage 1 results)
- Split validity (no split contract)
- Metric contract (thresholds not frozen)
- Config provenance (no config is loaded or checked)

**Risk:** A Phase 2 researcher could run this script before establishing any protocol, contaminating results without any audit trail.

---

## Known Limitations

1. **MISSING_EVIDENCE (runtime):** torch not installed — VAE encode/decode, checkpoint loading, Stage runner integration tests not verified at runtime. All VAE-related findings are from static code audit.
2. **MISSING_EVIDENCE:** No `pip freeze` output; exact environment reproducibility not guaranteed.
3. **MISSING_EVIDENCE:** No full reproduction run (`python -m src.reproduce ...`) was executed in this environment.
4. **Static audit of `archive/`:** Legacy scripts inspected by filename and content; not executed.
5. **ripgrep not available:** `Select-String` used instead; output format differs from `rg` but coverage is equivalent.

---

## P1 Acceptance Status

**P1_BLOCKED_BY_MISSING_TEST_ENV**

Forensic audit is complete and all failure modes documented. The block:
- torch not installed → runtime behavior of VAE is MISSING_EVIDENCE
- Does not prevent documentation or protocol design
- Must be resolved before any Phase 5+ runtime claim about model behavior
