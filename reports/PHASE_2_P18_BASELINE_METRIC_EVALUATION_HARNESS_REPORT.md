# Phase 2 P18: Baseline Metric Evaluation Harness Report

## 1. Task Summary
This task implements a deterministic, standard-library-only baseline metric evaluation harness that applies the accepted Phase 2 P17 metric contract primitives to evaluate explicit in-memory baseline candidate `ModelSpec` sets and series values.

## 2. Base Commit Verification
- Base branch: `phase2/p17-metric-contract-primitives`
- Base commit: `e3ba726a4f2f4aff866da346e559f43faab83593` (incorporating P11 fix commit `12230a465c1d5d518a68263c036d80ddb8c1d0d6`)

## 3. Files Created
- `src/phase2/baseline_eval.py`
- `tests/test_phase2_baseline_eval.py`
- `reports/PHASE_2_P18_BASELINE_METRIC_EVALUATION_HARNESS_REPORT.md`

## 4. Files Modified
- `src/phase2/__init__.py`

## 5. Files Not Changed
All other files in the repository remain unchanged.

## 6. Commands Run
- `python -m pytest tests/test_phase2_schema.py tests/test_phase2_constraints.py tests/test_phase2_sampler.py tests/test_phase2_simulator.py tests/test_phase2_dataset.py tests/test_phase2_artifacts.py tests/test_phase2_split_runner.py tests/test_phase2_protocol_presets.py tests/test_phase2_p14_smoke_dry_run.py tests/test_phase2_p15_manifest_audit.py tests/test_phase2_p16_dev_dry_run.py tests/test_phase2_metrics.py tests/test_phase2_baseline_eval.py -q`
- `python -m pytest tests/test_config.py tests/test_data_generator.py tests/test_validation.py tests/test_vector_schema.py tests/test_fail_fast_integrity.py tests/test_phase2_schema.py tests/test_phase2_constraints.py tests/test_phase2_sampler.py tests/test_phase2_simulator.py tests/test_phase2_dataset.py tests/test_phase2_artifacts.py tests/test_phase2_split_runner.py tests/test_phase2_protocol_presets.py tests/test_phase2_p14_smoke_dry_run.py tests/test_phase2_p15_manifest_audit.py tests/test_phase2_p16_dev_dry_run.py tests/test_phase2_metrics.py tests/test_phase2_baseline_eval.py -q`

## 7. Tests Run and Exact Results
- **Phase 2 Tests (Required)**: 497 passed, 1 skipped.
- **Compatibility Tests (Optional)**: 588 passed, 5 skipped.

## 8. Baseline Contract Constants
- `BASELINE_EVAL_CONTRACT_VERSION = "phase2_p18_baseline_eval_v1"`
- `APPROVED_BASELINE_NAMES = ("copy_reference", "random_valid", "structural_composition_oracle")`

## 9. Public Dataclasses Implemented
- `BaselineEvaluationRequest`
- `CandidateMetricRecord`
- `BaselineEvaluationResult`

## 10. Request Validation Behavior
- Checks exact type instances and rejects invalid formats.
- Ensures `baseline_name` belongs to the approved baseline names.
- Enforces that `reference_specs` and `candidate_specs` contain valid and mathematically valid `ModelSpec` objects.
- Rejects empty `candidate_specs` or `reference_specs`.
- Validates dimensions and contents of `candidate_series_values` (length matching candidate specs, finite values, series length >= 2, no bools).
- Validates seed repeat values using P17 `compute_seed_stability` constraints (at least 5 valid probability values).
- Rejects requests containing file path fields to prevent external path leakage.

## 11. Candidate Record Behavior
- Evaluates individual candidate specs by preserving the index.
- Automatically calculates validity batch output, composition scores, and novelty scores against reference models.
- Optionally performs diagnostics on actual generated values.

## 12. Aggregate Evaluation Behavior
- Validates requests completely before beginning.
- Loops through all candidate specifications in their original order.
- Calculates overall validity batch result and distribution distances (using the standard-library MMD RBF algorithm).
- Aggregates composition/novelty pass rates, and returns a detailed `BaselineEvaluationResult`.

## 13. Metric Bundle Bridge Behavior
- Returns a P17 compatibility `MetricBundle` from a baseline result.
- Correctly skips empty series diagnostics to fetch the first non-None series diagnostic from the candidates.
- Integrates aggregate validity and seed stability.

## 14. Scope Confirmation
No neural model, no VAE, no training loop, no optimizer, no checkpointing, no new dependency, no argparse CLI, and no config/YAML loader was introduced.

## 15. No Artifact Dependency Confirmation
Confirmed that the code does not read, write, or depend on any P14/P16 generated artifact files.

## 16. Remaining Blockers
None.

## 17. Post-Commit/Push Evidence
- Branch: `phase2/p18-baseline-metric-evaluation-harness`
- Commit: `831aeac04455330e60edc4f0600ecd57b48f1fbd`
- git ls-remote hash: `831aeac04455330e60edc4f0600ecd57b48f1fbd`

## 18. Final Verdict
**P18_READY_FOR_REVIEW**
