# PHASE 2 P47 — DIRECT RAW PARAMETER FIT SMOKE
## NO MODEL / NO VAE / NO SCIENCE

---

## 1. Phase identifier
- **Phase**: P47
- **Branch**: `phase2/p47-direct-raw-parameter-fit-smoke-no-model-no-vae-no-science`
- **Base P46R Head**: `01353768a802dce56afe7fced0eedff650e6e62b`

---

## 2. Goal
Verify that the existing P44/P45/P46 constraint-signature-loss stack operates correctly under direct gradient optimization. Specifically, demonstrate that the combined moment/spectral matching loss decreases when directly optimizing raw candidate AR/GARCH parameter tensors against fixed target signatures.

---

## 3. Base branch/head
- Base Branch: `phase2/p46r-cumulative-scope-gate-policy-repair-no-model-no-math-change`
- Base Head Hash: `01353768a802dce56afe7fced0eedff650e6e62b`

---

## 4. Changed files
The following 7 files are changed/created:
1. `src/phase2/direct_raw_parameter_fit_smoke.py` [NEW]
2. `tools/phase2/run_p47_direct_raw_parameter_fit_smoke.py` [NEW]
3. `tests/test_phase2_direct_raw_parameter_fit_smoke.py` [NEW]
4. `tests/test_phase2_p47_direct_raw_parameter_fit_smoke.py` [NEW]
5. `src/phase2/__init__.py` [MODIFIED]
6. `tests/test_phase2_p46r_cumulative_scope_gate_policy.py` [MODIFIED]
7. `reports/PHASE_2_P47_DIRECT_RAW_PARAMETER_FIT_SMOKE_NO_MODEL_NO_VAE_NO_SCIENCE_REPORT.md` [NEW]

---

## 5. Why P47 is intentionally simple
P47 is designed as an optimization sanity check (smoke test). It avoids the complexities of neural architectures, datasets, batching, and training loops, verifying solely that our analytic signatures and matching losses are mathematically correct and fully differentiable.

---

## 6. Direct raw parameter fit design
1. **Target Parameters**: A set of fixed target raw AR/GARCH parameters (no gradients required).
2. **Candidate Parameters**: A set of raw AR/GARCH parameters initialized to zero, requiring gradients.
3. **Analytic Signatures**: Target and candidate signatures computed via:
   - PACF to stable AR coefficients (`pacf_to_stable_ar_coefficients`)
   - GARCH mass allocation from logits (`garch_mass_allocation_from_logits`)
   - Spectral signature (`ar_spectral_signature`)
   - Volatility/moment signature (`garch_moment_persistence_signature`)
   - Combined signature (`combined_ar_garch_signature`)
4. **Loss**: Computed via `combined_moment_spectral_matching_loss`.
5. **Optimization**: Exactly 20 steps of manual gradient descent with a learning rate of 0.05. No `torch.optim` or optimizer objects are instantiated.

---

## 7. Initial/final loss result
- **Initial Loss**: `0.1576555371284485`
- **Final Loss**: `0.02555064857006073`
- **Loss Decreased**: `True` (Delta = `0.13210488855838776`)

---

## 8. Boundary: no model/no VAE/no science
This implementation maintains strict boundaries:
- **No Neural Model**: No `torch.nn.Module` subclass or network structure.
- **No VAE**: No VAE encoders, decoders, latent sampling, or reparameterizations.
- **No Training Loop**: No datasets, dataloaders, epochs, or loss optimization over data samples.
- **No Scientific Claims**: This phase does not claim convergence, scientific success, or benchmark readiness. It is purely an integration sanity check of the differentiable math.

---

## 9. Smoke output
The compact, sorted JSON printed by the smoke script is:
```json
{"final_loss_finite":true,"final_loss_nonnegative":true,"final_loss_value":0.02555064857006073,"initial_loss_finite":true,"initial_loss_nonnegative":true,"initial_loss_value":0.1576555371284485,"kind":"direct_raw_parameter_fit_smoke_no_model_no_vae_no_science","loss_decrease_positive":true,"loss_decreased":true,"loss_delta_value":0.13210488855838776,"no_dataloader":true,"no_dataset":true,"no_decoder":true,"no_encoder":true,"no_model":true,"no_scientific_conclusion":true,"no_torch_optimizer":true,"no_vae":true,"reason":"p47_direct_raw_parameter_fit_probe_success","source_phase":"P47","status":"direct_raw_parameter_fit_smoke_available_no_model_no_vae_no_science","step_count":20,"torch_available":true,"verdict":"PASS"}
```

---

## 10. Focused tests
Ran focused P47 tests:
```bash
python -m pytest tests/test_phase2_direct_raw_parameter_fit_smoke.py tests/test_phase2_p47_direct_raw_parameter_fit_smoke.py -v
```
**Results**: 11 passed, 0 warnings.

---

## 11. Full curated tests

* Curated tests completed: **1751 passed, 8 skipped, 0 failed** (on the base `phase2/p47-direct-raw-parameter-fit-smoke-no-model-no-vae-no-science` branch).
* On the `phase2/p47r-report-only-full-curated-evidence-correction-no-code-change` branch, the result is **1750 passed, 9 skipped, 0 failed** due to `test_p47_09_scope_gate` correctly skipping as it is branch-aware.
* The skipped tests are expected explicit phase-local scope-gate skips on later cumulative branches.
* Final base remote head hash: `60809b24b92888302402944598a0d022e161a44f`

---

## 12. Scope gate
`test_p47_09_scope_gate` checks `git diff` against base head `01353768a802dce56afe7fced0eedff650e6e62b` and ensures only the 6 allowed files are modified or created.

---

## 13. Remaining blockers
None.

---

## 14. Final verdict
`P47_READY_FOR_REVIEW`
