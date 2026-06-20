# Weights Governance Policy

## Overview

This document defines the governance rules for all `.pth` checkpoint files in the
`econometric-vae-manifold` project. Weights remain binary `.pth` files. YAML
configuration (e.g. `configs/demo.yaml`) stores the **metadata** that governs
each checkpoint — including architecture parameters, role, and traceability info.

---

## 1. Public Demo Checkpoint

**Recommended default**: `models/vae_beta50.pth`

Rationale: Beta-KL weight of 50 produces the clearest latent-space cluster separation
between AR and GARCH regions, making it the most interpretable for demonstrations.
Size: ~45 KB.

---

## 2. Checkpoint Catalogue

| File | Location | Role | Architecture | Status |
|---|---|---|---|---|
| `vae_beta50.pth` | `models/` | **Public demo default** (referenced in configs) | baseline (40→64→32→z8→32→64→40) | ✅ Active |
| `vae_model_weights.pth` | `models/` | **Retained legacy baseline** (not used in demo) | baseline (40→64→32→z8→32→64→40) | ⚠️ Retained |
| `vae_beta01.pth` | `archive/legacy_models/` | Low-beta ablation | baseline | 🔬 Archived |
| `vae_beta05.pth` | `archive/legacy_models/` | Mid-beta ablation | baseline | 🔬 Archived |
| `vae_beta10.pth` | `archive/legacy_models/` | Mid-beta ablation | baseline | 🔬 Archived |
| `vae_beta20.pth` | `archive/legacy_models/` | Mid-beta ablation | baseline | 🔬 Archived |
| `vae_deeper.pth` | `archive/legacy_models/` | Deeper architecture experiment | deeper (≈79 KB) | 🔬 Archived |
| `vae_latent16.pth` | `archive/legacy_models/` | Latent dim=16 experiment | latent16 (≈48 KB) | 🔬 Archived |
| `vae_wider.pth` | `archive/legacy_models/` | Wider architecture experiment | wider (≈247 KB) | 🔬 Archived |
| `vae_weights_seed*.pth` | `archive/legacy_models/` | Robustness seeds | baseline | 🔬 Archived |
| `vae_weights.pth (ID_*)` | `archive/legacy_models/` | Architecture variants | variable | 🔬 Archived |

---

## 3. Required YAML Metadata Per Checkpoint

Each publicly-used checkpoint must have a corresponding YAML config entry. Minimum
required fields:

```yaml
model:
  architecture: baseline_vae        # Must match the VAE class definition
  input_dim: 40                      # Must equal VECTOR_DIM
  latent_dim: 8                      # Must match the trained model
  hidden_dims: [64, 32]             # Must match the trained model
  checkpoint_path: models/vae_beta50.pth
  checkpoint_role: public_demo_default   # One of: public_demo_default, ablation, archive, experimental
```

Optional traceability fields (recommended, deferred):

```yaml
model:
  training_seed: 42
  training_epochs: 100
  beta_kl: 50.0
  training_data: Z_train.npy         # Dataset used (not the data itself)
  sha256: <checksum>                 # See Section 5
  file_size_bytes: 46284
```

---

## 4. Where Weights Should Live

Current state (after Phase 1Q/1R cleanup):
The directory has been cleaned and organized as follows:

```
models/
  vae_beta50.pth        # Public demo checkpoint (default in configs/demo.yaml)
  vae_model_weights.pth # Retained legacy checkpoint (historical baseline)

archive/legacy_models/
  vae_beta01.pth        # Archived ablation
  vae_beta05.pth        # Archived ablation
  vae_beta10.pth        # Archived ablation
  vae_beta20.pth        # Archived ablation
  vae_deeper.pth        # Archived experimental deeper VAE
  vae_latent16.pth      # Archived experimental latent dim=16 VAE
  vae_wider.pth         # Archived experimental wider VAE
```

All legacy root duplicates of weights have been removed or archived.

---

## 5. Integrity Validation (Deferred)

A future hardening phase should add SHA-256 checksums and file sizes to each
YAML config entry. The loader should validate these on startup when present.

```python
# Planned validation (not yet implemented in src/config.py)
if cfg.model.sha256:
    actual = hashlib.sha256(Path(cfg.model.checkpoint_path).read_bytes()).hexdigest()
    assert actual == cfg.model.sha256, "Checkpoint integrity check failed"
```

This is deferred because:
- No current CI/CD to regenerate checksums when weights are updated.
- All existing checkpoints lack recorded metadata.

---

## 6. Architecture Compatibility Notes

> **Critical**: Loading a checkpoint into the wrong architecture raises a
> cryptic RuntimeError or silently loads partial weights.

| Checkpoint | Expected Architecture | Input → Latent | Notes |
|---|---|---|---|
| `vae_beta*.pth` | baseline VAE | 40 → 8 | hidden=[64,32] |
| `vae_deeper.pth` | deeper VAE | 40 → 8 | additional hidden layer |
| `vae_latent16.pth` | baseline VAE | 40 → 16 | different latent dim |
| `vae_wider.pth` | wider VAE | 40 → 8 | much larger hidden layers |

Use `model.architecture` and `model.latent_dim` in the YAML to prevent misloads.

---

## 7. Non-Goals

- This document does NOT govern training data files (`.npy`).
- This document does NOT govern analysis output files (`.csv`, `.png`).
- Weights are NOT stored inside YAML. YAML stores metadata only.
