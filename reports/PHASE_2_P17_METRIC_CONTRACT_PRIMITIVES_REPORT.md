# Phase 2 P17: Metric Contract Primitives Report

## 1. Task Summary
This task implements deterministic Phase 2 metric contract primitives for evaluating generated ModelSpec and time-series outputs. The metrics are implemented using only the Python standard library, with no dependencies on `numpy`, `scipy`, `pandas`, `scikit-learn`, or `torch`.

## 2. Base Commit Verification
- Base branch: `phase2/p16-dev-artifact-generation-dry-run`
- Base commit: `7b4d2e1fb4b146827453f0bab4f102ba09376b4e`

## 3. Files Created
- `src/phase2/metrics.py`
- `tests/test_phase2_metrics.py`
- `reports/PHASE_2_P17_METRIC_CONTRACT_PRIMITIVES_REPORT.md`

## 4. Files Modified
- `src/phase2/__init__.py`

## 5. Files Not Changed
All other files in the repository remain unchanged.

## 6. Commands Run
- `python -m pytest tests/test_phase2_schema.py tests/test_phase2_constraints.py tests/test_phase2_sampler.py tests/test_phase2_simulator.py tests/test_phase2_dataset.py tests/test_phase2_artifacts.py tests/test_phase2_split_runner.py tests/test_phase2_protocol_presets.py tests/test_phase2_p14_smoke_dry_run.py tests/test_phase2_p15_manifest_audit.py tests/test_phase2_p16_dev_dry_run.py tests/test_phase2_metrics.py -q`
- `python -m pytest tests/test_config.py tests/test_data_generator.py tests/test_validation.py tests/test_vector_schema.py tests/test_fail_fast_integrity.py tests/test_phase2_schema.py tests/test_phase2_constraints.py tests/test_phase2_sampler.py tests/test_phase2_simulator.py tests/test_phase2_dataset.py tests/test_phase2_artifacts.py tests/test_phase2_split_runner.py tests/test_phase2_protocol_presets.py tests/test_phase2_p14_smoke_dry_run.py tests/test_phase2_p15_manifest_audit.py tests/test_phase2_p16_dev_dry_run.py tests/test_phase2_metrics.py -q`

## 7. Tests Run and Exact Results
- **Phase 2 Tests (Required)**: 447 passed, 1 skipped.
- **Compatibility Tests (Optional)**: 538 passed, 5 skipped.

## 8. Implemented Metrics Details
The metrics layer is divided into key modules:
- **Validity Batch**: Evaluates the correctness of a batch of `ModelSpec` objects.
- **Composition**: Computes component scores for the mean and volatility families and aggregates them using a harmonic mean.
- **Novelty**: Computes distance metrics against reference models, using structural family distance and normalized parameter distance.
- **Distribution Distance**: Computes a biased Maximum Mean Discrepancy (MMD) with an RBF kernel.
- **Series Diagnostics**: Evaluates raw ACF, squared centered ACF, and volatility clustering scores on generated series values.
- **Seed Stability**: Validates variance of validity rates across multiple repeat seeds.

## 9. Post-Commit/Push Evidence
- Branch: `phase2/p17-metric-contract-primitives`
- Commit: `2cacaaf5ca002129b7a1420320094f190cf83225`
- git ls-remote hash: `2cacaaf5ca002129b7a1420320094f190cf83225`

## 10. Final Verdict
**P17_READY_FOR_REVIEW**
