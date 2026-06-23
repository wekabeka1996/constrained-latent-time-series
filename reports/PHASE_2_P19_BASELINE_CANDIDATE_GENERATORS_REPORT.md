# Phase 2 P19: Baseline Candidate Generators Report

## 1. Task Summary
This task implements deterministic, standard-library-only baseline candidate generator primitives for Phase 2. It creates explicit in-memory `ModelSpec` candidate sets for three approved baseline names: `copy_reference`, `random_valid`, and `structural_composition_oracle`.

## 2. Base Commit Verification
- Base branch: `phase2/p18-baseline-metric-evaluation-harness`
- Base commit: `3eb39ad30b22279c3a120cd02e63707fc016e1f5` (incorporating P11 fix commit `12230a465c1d5d518a68263c036d80ddb8c1d0d6`)

## 3. Files Created
- `src/phase2/baseline_generators.py`
- `tests/test_phase2_baseline_generators.py`
- `reports/PHASE_2_P19_BASELINE_CANDIDATE_GENERATORS_REPORT.md`

## 4. Files Modified
- `src/phase2/__init__.py`

## 5. Files Not Changed
All other files in the repository remain unchanged.

## 6. Commands Run
- `python -m pytest tests/test_phase2_schema.py tests/test_phase2_constraints.py tests/test_phase2_sampler.py tests/test_phase2_simulator.py tests/test_phase2_dataset.py tests/test_phase2_artifacts.py tests/test_phase2_split_runner.py tests/test_phase2_protocol_presets.py tests/test_phase2_p14_smoke_dry_run.py tests/test_phase2_p15_manifest_audit.py tests/test_phase2_p16_dev_dry_run.py tests/test_phase2_metrics.py tests/test_phase2_baseline_eval.py tests/test_phase2_baseline_generators.py -q`
- `python -m pytest tests/test_config.py tests/test_data_generator.py tests/test_validation.py tests/test_vector_schema.py tests/test_fail_fast_integrity.py tests/test_phase2_schema.py tests/test_phase2_constraints.py tests/test_phase2_sampler.py tests/test_phase2_simulator.py tests/test_phase2_dataset.py tests/test_phase2_artifacts.py tests/test_phase2_split_runner.py tests/test_phase2_protocol_presets.py tests/test_phase2_p14_smoke_dry_run.py tests/test_phase2_p15_manifest_audit.py tests/test_phase2_p16_dev_dry_run.py tests/test_phase2_metrics.py tests/test_phase2_baseline_eval.py tests/test_phase2_baseline_generators.py -q`

## 7. Tests Run and Exact Results
- **Phase 2 Tests (Required)**: 559 passed, 1 skipped.
- **Compatibility Tests (Optional)**: 650 passed, 5 skipped.

## 8. Baseline Generator Constants
- `BASELINE_GENERATOR_CONTRACT_VERSION = "phase2_p19_baseline_generators_v1"`
- `SUPPORTED_BASELINE_GENERATOR_NAMES = ("copy_reference", "random_valid", "structural_composition_oracle")`
- `COPY_REFERENCE_BASELINE_NAME = "copy_reference"`
- `RANDOM_VALID_BASELINE_NAME = "random_valid"`
- `STRUCTURAL_COMPOSITION_ORACLE_BASELINE_NAME = "structural_composition_oracle"`

## 9. Public Dataclasses Implemented
- `BaselineCandidateSourceRecord`
- `BaselineGenerationRequest`
- `BaselineGenerationResult`

## 10. Validation Behavior
- `validate_baseline_generator_name` verifies that baseline names belong to supported lists.
- `validate_positive_int` checks for strictly positive integers, rejecting booleans.
- `validate_seed` checks for non-negative integers, rejecting booleans.
- `validate_reference_specs` ensures a tuple of valid and mathematically valid `ModelSpec` objects is provided.
- `validate_oracle_constraint_flags` checks that exactly 4 finite float/int flags are provided.
- `validate_random_valid_templates` ensures non-empty templates have unique `FamilyId` keys, and the schedule only requests defined templates.
- `validate_baseline_generation_request` validates requests according to baseline generator names and checks specific properties (like templates for random valid generators).

## 11. copy_reference Generator Behavior
- Cycles references deterministically: `reference_specs[i % len(reference_specs)]`.
- Clones candidates with clear provenance.
- Records exact source index, setting the seed field to `None`.

## 12. random_valid generator behavior
- Uses explicit `GenerationRequest` templates and a family schedule.
- Candidates are built by invoking the P7 `generate_model_spec` primitive with deterministic seeds `seed + i`.
- Properly extracts the `ModelSpec` structure from P7 `GenerationResult`.
- Records candidate seeds and sets source reference indices to empty.

## 13. structural_composition_oracle behavior
- Finds valid ARMA mean-only sources and valid GARCH volatility-only sources from reference specs.
- Composes ARMA_GARCH specifications by merging parameters from mean and volatility sources.
- Applies user-specified oracle constraint flags.
- Cycles sources deterministically and records the source indices used.

## 14. P18 Evaluation Request Bridge Behavior
- Converts a `BaselineGenerationResult` into a P18 `BaselineEvaluationRequest`.
- Validates the resulting request using `validate_baseline_evaluation_request` to ensure correct formatting and types.

## 15. Scope Confirmation
No neural model, no VAE, no training loop, no optimizer, no checkpointing, no metrics evaluation, no dataset JSONL writing, no config/YAML loader, no argparse CLI, and no new dependencies were introduced.

## 16. No Artifact Dependency Confirmation
Confirmed that the code does not read, write, or depend on any generated P14/P16 artifact files.

## 17. Remaining Blockers
None.

## 18. Post-Commit/Push Evidence
- Branch: `phase2/p19-baseline-candidate-generators`
- Commit: `6371c427df957dcdbcc441ab01cb5bcfb6b50a59`
- git ls-remote hash: `6371c427df957dcdbcc441ab01cb5bcfb6b50a59`

## 19. Final Verdict
**P19_READY_FOR_REVIEW**
