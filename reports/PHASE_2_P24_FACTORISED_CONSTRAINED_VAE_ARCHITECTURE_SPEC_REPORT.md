# Phase 2 P24: Factorised Constrained VAE Architecture Spec Report

## 1. Task Summary
This task defines a strict neural model architecture specification for the Factorized Constrained VAE candidate (short alias: FC-VAE) with a typed ModelSpec decoder. This phase is documentation and contract only.

## 2. Base Commit Verification
- **Base branch**: `phase2/p23-evidence-contract-normalized-baseline-bundle`
- **Base HEAD commit**: `6354ea1c68e63064e0afdfba01f2303fe36a83e9`
- **P11 fix commit**: `12230a465c1d5d518a68263c036d80ddb8c1d0d6` (Verified present in commit history).
- **Accepted P23 & P22 files**: All verified present.
- **.gitignore**: Verified contains `phase2_artifacts/`.

## 3. Files Created
- `docs/PHASE_2_FACTORISED_CONSTRAINED_VAE_ARCHITECTURE.md`
- `docs/PHASE_2_MODEL_BOUNDARY_CONTRACT.md`
- `docs/PHASE_2_ARCHITECTURE_ABLATION_PLAN.md`
- `docs/PHASE_2_MODEL_TO_EVIDENCE_CONTRACT.md`
- `tests/test_phase2_p24_architecture_docs.py`
- `reports/PHASE_2_P24_FACTORISED_CONSTRAINED_VAE_ARCHITECTURE_SPEC_REPORT.md`

## 4. Files Modified
- None.

## 5. Files Not Changed
All accepted Phase 2 core files, baselines, and evidence contract modules remain completely unmodified.

## 6. Commands Run
- **Required test command**:
  `python -m pytest tests/test_phase2_schema.py tests/test_phase2_constraints.py tests/test_phase2_sampler.py tests/test_phase2_simulator.py tests/test_phase2_dataset.py tests/test_phase2_artifacts.py tests/test_phase2_split_runner.py tests/test_phase2_protocol_presets.py tests/test_phase2_p14_smoke_dry_run.py tests/test_phase2_p15_manifest_audit.py tests/test_phase2_p16_dev_dry_run.py tests/test_phase2_metrics.py tests/test_phase2_baseline_eval.py tests/test_phase2_baseline_generators.py tests/test_phase2_p20_baseline_evidence_smoke.py tests/test_phase2_static_scope_guard.py tests/test_phase2_p22_artifact_backed_baseline_smoke.py tests/test_phase2_evidence_contract.py tests/test_phase2_p23_evidence_contract_smoke.py tests/test_phase2_p24_architecture_docs.py -q`
- **Optional compatibility command**:
  `python -m pytest tests/test_config.py tests/test_data_generator.py tests/test_validation.py tests/test_vector_schema.py tests/test_fail_fast_integrity.py tests/test_phase2_schema.py tests/test_phase2_constraints.py tests/test_phase2_sampler.py tests/test_phase2_simulator.py tests/test_phase2_dataset.py tests/test_phase2_artifacts.py tests/test_phase2_split_runner.py tests/test_phase2_protocol_presets.py tests/test_phase2_p14_smoke_dry_run.py tests/test_phase2_p15_manifest_audit.py tests/test_phase2_p16_dev_dry_run.py tests/test_phase2_metrics.py tests/test_phase2_baseline_eval.py tests/test_phase2_baseline_generators.py tests/test_phase2_p20_baseline_evidence_smoke.py tests/test_phase2_static_scope_guard.py tests/test_phase2_p22_artifact_backed_baseline_smoke.py tests/test_phase2_evidence_contract.py tests/test_phase2_p23_evidence_contract_smoke.py tests/test_phase2_p24_architecture_docs.py -q`

## 7. Tests Run and Exact Results
- **Required Tests**: 712 passed, 1 skipped
- **Optional Compatibility Tests**: 803 passed, 5 skipped

## 8. Architecture Spec Summary
The `docs/PHASE_2_FACTORISED_CONSTRAINED_VAE_ARCHITECTURE.md` defines the Factorized Constrained VAE (FC-VAE) structure:
- **Latent space**: Split into partitioned `z_mean`, `z_volatility`, and `z_shared` components.
- **Encoder**: Independent mean, volatility, and residual branches.
- **Decoder**: Emits parameters through family, mean, volatility, and diagnostic heads.
- **Typed boundary**: Decoded parameters must parse into a valid `ModelSpec` and pass `validate_model_spec` and `require_math_valid`.
- **Constraint mode**: Recommends `reject-invalid mode` for v0 to ensure honest validity rates.

## 9. Model Boundary Contract Summary
The `docs/PHASE_2_MODEL_BOUNDARY_CONTRACT.md` establishes boundary rules:
- **Inputs**: Restricts training inputs to artifact-backed data and deterministic diagnostics.
- **Forbidden inputs**: Strictly bans Family C composite samples and labels from zero-shot training.
- **Validation gates**: Future model candidates must pass schema, math, zero-shot, and P23 safety checks.
- **Fail-fast triggers**: Enforces system termination on any contamination, schema violation, or safety leak.

## 10. Ablation Plan Summary
The `docs/PHASE_2_ARCHITECTURE_ABLATION_PLAN.md` defines baselines and neural candidates to isolate performance sources:
- **Baselines**: A0 (Copy), A1 (Random Valid), A2 (Oracle Composition).
- **Candidates**: M0 (Unconstrained), M1 (Single-latent Constrained), M2 (Factorized Unconstrained), M3 (FC-VAE primary candidate), M4 (Grammar-Prior constrained).

## 11. Model-to-Evidence Contract Summary
The `docs/PHASE_2_MODEL_TO_EVIDENCE_CONTRACT.md` governs model comparison:
- Maps model evaluation records to P23 metrics and generation schemas.
- Proposes future `EvidenceModelRecord` and `EvidenceComparisonBundle` extensions.
- Enforces safety filters (no raw parameter keys, no local paths, no forbidden success claims).

## 12. Scope Confirmation
It is explicitly confirmed that:
- No neural model code or PyTorch imports have been introduced.
- No VAE training loops, checkpointing, or optimizers exist.
- No training configs, CLI flags, or external configurations have been created.
- No generated dataset JSONL or manifest writing occurred.
- No new external dependencies have been added.

## 13. No P16 Artifact Dependency Confirmation
Confirmed that this specification does not depend on or access P16 dry-run artifacts.

## 14. Scientific Conclusion Disclaimer
This phase is documentation and contract only. No scientific conclusions are claimed regarding baseline or model candidate comparison.

## 15. Remaining Blockers
None.

## 16. Post-Commit/Push Evidence
- **Branch**: `phase2/p24-factorised-constrained-vae-architecture-spec`
- **Commit Hash**: `cd8bf5858cfd34c114387ea5be415f3debe8848d`
- **git ls-remote Hash**: `cd8bf5858cfd34c114387ea5be415f3debe8848d`

## 17. Final Verdict
```text id="p24_verdict"
P24_READY_FOR_REVIEW
```
