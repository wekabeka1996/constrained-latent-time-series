# Phase 2 P22: Artifact-Backed Baseline Evidence Smoke Run Report

## 1. Task Summary
This task implements and executes a read-only artifact-backed baseline evidence smoke run using generated P14 smoke artifacts.
Following reviewer feedback (FIX2), we have:
- Restored direct, readable module imports and path strings in `run_p22_artifact_backed_baseline_smoke.py`.
- Customised the static scope rules for P22 to allow the exact P14 root path while strictly forbidding P16 paths, artifact generation calls, dry-runs, and third-party libraries (including `argparse`).
- Expanded the test suite in `tests/test_phase2_p22_artifact_backed_baseline_smoke.py` to 29 test cases, verifying main argument rejection, no P16 path dependencies, seed sequences, oracle composition rates, and decoupled mock-based assertions.

## 2. Base Commit Verification
- Base branch: `phase2/p22-artifact-backed-baseline-evidence-smoke`
- Baseline commit: `5e08895556273df9a62ff5fc55618646081d1b38` (HEAD of branch before fixes)

## 3. Files Created
- `reports/PHASE_2_P22_ARTIFACT_BACKED_BASELINE_EVIDENCE_SMOKE_REPORT.md` (This report)

## 4. Files Modified
- `tools/phase2/run_p22_artifact_backed_baseline_smoke.py`
- `tests/test_phase2_p22_artifact_backed_baseline_smoke.py`

## 5. Files Deleted
- `reports/PHASE_2_P22_ARTIFACT_BACKED_BASELINE_SMOKE_REPORT.md` (Old misnamed report)

## 6. Commands Run
- **Required Tests:**
  `python -m pytest tests/test_phase2_schema.py tests/test_phase2_constraints.py tests/test_phase2_sampler.py tests/test_phase2_simulator.py tests/test_phase2_dataset.py tests/test_phase2_artifacts.py tests/test_phase2_split_runner.py tests/test_phase2_protocol_presets.py tests/test_phase2_p14_smoke_dry_run.py tests/test_phase2_p15_manifest_audit.py tests/test_phase2_p16_dev_dry_run.py tests/test_phase2_metrics.py tests/test_phase2_baseline_eval.py tests/test_phase2_baseline_generators.py tests/test_phase2_p20_baseline_evidence_smoke.py tests/test_phase2_static_scope_guard.py tests/test_phase2_p22_artifact_backed_baseline_smoke.py -q`
- **Optional Compatibility Tests:**
  `python -m pytest tests/test_config.py tests/test_data_generator.py tests/test_validation.py tests/test_vector_schema.py tests/test_fail_fast_integrity.py tests/test_phase2_schema.py tests/test_phase2_constraints.py tests/test_phase2_sampler.py tests/test_phase2_simulator.py tests/test_phase2_dataset.py tests/test_phase2_artifacts.py tests/test_phase2_split_runner.py tests/test_phase2_protocol_presets.py tests/test_phase2_p14_smoke_dry_run.py tests/test_phase2_p15_manifest_audit.py tests/test_phase2_p16_dev_dry_run.py tests/test_phase2_metrics.py tests/test_phase2_baseline_eval.py tests/test_phase2_baseline_generators.py tests/test_phase2_p20_baseline_evidence_smoke.py tests/test_phase2_static_scope_guard.py tests/test_phase2_p22_artifact_backed_baseline_smoke.py -q`
- **Smoke Run:**
  `python -m tools.phase2.run_p22_artifact_backed_baseline_smoke`

## 7. Tests Run and Exact Results
- **Required Tests:** 649 passed, 1 skipped in 62.60s.
- **Optional Compatibility Tests:** 740 passed, 5 skipped in 61.97s.

## 8. Smoke Execution Command
```bash
python -m tools.phase2.run_p22_artifact_backed_baseline_smoke
```

## 9. Sanitized Stdout JSON Excerpt
```json
{"artifact_root":"phase2_artifacts/p14_smoke_dry_run","baseline_count":3,"baselines":[{"baseline_name":"copy_reference","evaluation_summary":{"baseline_name":"copy_reference","candidate_count":6,"candidate_record_count":6,"composition_pass_count":0,"composition_pass_rate":0.0,"distribution_mmd_rbf":0.0,"novelty_pass_count":0,"novelty_pass_rate":0.0,"reference_count":2,"seed_stability_present":false,"validity_total_count":6,"validity_valid_count":6,"validity_valid_rate":1.0},"generation_summary":{"baseline_name":"copy_reference","candidate_count":6,"candidate_family_ids":["ARMA","GARCH","ARMA","GARCH","ARMA","GARCH"],"source_records_summary":[{"candidate_index":0,"family_id":"ARMA","generator_name":"copy_reference","seed":null,"source_reference_indices":[0]},{"candidate_index":1,"family_id":"GARCH","generator_name":"copy_reference","seed":null,"source_reference_indices":[1]},{"candidate_index":2,"family_id":"ARMA","generator_name":"copy_reference","seed":null,"source_reference_indices":[0]},{"candidate_index":3,"family_id":"GARCH","generator_name":"copy_reference","seed":null,"source_reference_indices":[1]},{"candidate_index":4,"family_id":"ARMA","generator_name":"copy_reference","seed":null,"source_reference_indices":[0]},{"candidate_index":5,"family_id":"GARCH","generator_name":"copy_reference","seed":null,"source_reference_indices":[1]}]},"metric_bundle_bridge_verified":true,"reason":"p22_artifact_backed_baseline_smoke_single_baseline_completed: copy_reference"},{"baseline_name":"random_valid","evaluation_summary":{"baseline_name":"random_valid","candidate_count":6,"candidate_record_count":6,"composition_pass_count":2,"composition_pass_rate":0.3333333333333333,"distribution_mmd_rbf":0.467126027327986,"novelty_pass_count":0,"novelty_pass_rate":0.0,"reference_count":2,"seed_stability_present":false,"validity_total_count":6,"validity_valid_count":6,"validity_valid_rate":1.0},"generation_summary":{"baseline_name":"random_valid","candidate_count":6,"candidate_family_ids":["AR","GARCH","ARMA_GARCH","AR","GARCH","ARMA_GARCH"],"source_records_summary":[{"candidate_index":0,"family_id":"AR","generator_name":"random_valid","seed":22001,"source_reference_indices":[]},{"candidate_index":1,"family_id":"GARCH","generator_name":"random_valid","seed":22002,"source_reference_indices":[]},{"candidate_index":2,"family_id":"ARMA_GARCH","generator_name":"random_valid","seed":22003,"source_reference_indices":[]},{"candidate_index":3,"family_id":"AR","generator_name":"random_valid","seed":22004,"source_reference_indices":[]},{"candidate_index":4,"family_id":"GARCH","generator_name":"random_valid","seed":22005,"source_reference_indices":[]},{"candidate_index":5,"family_id":"ARMA_GARCH","generator_name":"random_valid","seed":22006,"source_reference_indices":[]}]},"metric_bundle_bridge_verified":true,"reason":"p22_artifact_backed_baseline_smoke_single_baseline_completed: random_valid"},{"baseline_name":"structural_composition_oracle","evaluation_summary":{"baseline_name":"structural_composition_oracle","candidate_count":6,"candidate_record_count":6,"composition_pass_count":6,"composition_pass_rate":1.0,"distribution_mmd_rbf":1.498526078428923,"novelty_pass_count":0,"novelty_pass_rate":0.0,"reference_count":2,"seed_stability_present":false,"validity_total_count":6,"validity_valid_count":6,"validity_valid_rate":1.0},"generation_summary":{"baseline_name":"structural_composition_oracle","candidate_count":6,"candidate_family_ids":["ARMA_GARCH","ARMA_GARCH","ARMA_GARCH","ARMA_GARCH","ARMA_GARCH","ARMA_GARCH"],"source_records_summary":[{"candidate_index":0,"family_id":"ARMA_GARCH","generator_name":"structural_composition_oracle","seed":null,"source_reference_indices":[0,1]},{"candidate_index":1,"family_id":"ARMA_GARCH","generator_name":"structural_composition_oracle","seed":null,"source_reference_indices":[0,1]},{"candidate_index":2,"family_id":"ARMA_GARCH","generator_name":"structural_composition_oracle","seed":null,"source_reference_indices":[0,1]},{"candidate_index":3,"family_id":"ARMA_GARCH","generator_name":"structural_composition_oracle","seed":null,"source_reference_indices":[0,1]},{"candidate_index":4,"family_id":"ARMA_GARCH","generator_name":"structural_composition_oracle","seed":null,"source_reference_indices":[0,1]},{"candidate_index":5,"family_id":"ARMA_GARCH","generator_name":"structural_composition_oracle","seed":null,"source_reference_indices":[0,1]}]},"metric_bundle_bridge_verified":true,"reason":"p22_artifact_backed_baseline_smoke_single_baseline_completed: structural_composition_oracle"}],"candidate_count_per_baseline":6,"contract":"phase2_p22_artifact_backed_baseline_smoke_v1","loaded_reference_count":2,"no_artifact_generation":true,"no_final_comparison":true,"no_model_training":true,"no_scientific_conclusion":true,"p14_audit_verified":true,"reason":"p22_artifact_backed_baseline_smoke_completed","reference_summary":{"artifact_root":"phase2_artifacts/p14_smoke_dry_run","loaded_reference_count":2,"references":[{"family_id":"ARMA","line_number":1001,"mean_family":"ARMA","role":"arma_mean_source","sample_id":"ARMA_zero_shot_train_000001_g12001_s1012001_29dcbe9e1c07","volatility_family":"NONE"},{"family_id":"GARCH","line_number":2001,"mean_family":"NONE","role":"garch_volatility_source","sample_id":"GARCH_zero_shot_train_000001_g12002_s1012002_6d653cb4f1b3","volatility_family":"GARCH"}],"zero_shot_train_samples_path":"phase2_artifacts\\p14_smoke_dry_run\\zero_shot_train\\samples.jsonl"},"verdict":"PASS"}
```

## 10. Reference Fixture Summary
The script successfully extracted 2 `ModelSpec` candidates from `zero_shot_train/samples.jsonl`:
- **ARMA mean source:** `sample_id: ARMA_zero_shot_train_000001_g12001_s1012001_29dcbe9e1c07` (extracted at line 1001, MeanFamily.ARMA, VolatilityFamily.NONE).
- **GARCH volatility source:** `sample_id: GARCH_zero_shot_train_000001_g12002_s1012002_6d653cb4f1b3` (extracted at line 2001, MeanFamily.NONE, VolatilityFamily.GARCH).

## 11. Baseline Generation Summary
- **copy_reference:** Successfully cloned references to generate 3 ARMA and 3 GARCH baseline candidate copies.
- **random_valid:** Generated 6 valid candidates (`AR`, `GARCH`, `ARMA_GARCH`) using seed sequence `22001` through `22006` in order.
- **structural_composition_oracle:** Merged parameter fields of the reference specs to create 6 valid composite `ARMA_GARCH` candidates.

## 12. Baseline Evaluation Summary
- All candidate models possess a validity pass rate of `1.0`.
- The structural composition oracle obtained a `composition_pass_rate` of `1.0` (as expected).
- The random_valid baseline uses exact seed sequences `22001..22006` as verified programmatically.
- Evaluation metrics (validity, composition, novelty, distribution MMD-RBF) were evaluated deterministically without any raw parameter leakage.

## 13. MetricBundle Bridge Verification
Programmatic checks verify that `build_baseline_metric_bundle` successfully parses evaluation results and produces a valid `MetricBundle` instance.

## 14. Determinism Verification
Verification confirms that consecutive executions of the baseline evaluation harness with identical reference sets produce exact identical outputs.

## 15. Scope Confirmation
No neural model training, dataset artifact writing, or VAE optimization was introduced. Command line arguments are strictly rejected (verified by unit test).
No forbidden imports (`torch`, `numpy`, `pandas`, `yaml`, `argparse`, `sklearn`, `scipy`) are present in the runner script.

## 16. No P14/P16 Artifact Dependency Confirmation
The unit tests do not depend on the local existence of P14/P16 artifacts. The integration test skips gracefully if artifacts are absent.
The static checks verify that no general `P16` path references exist in the script. The exact root path `"phase2_artifacts/p14_smoke_dry_run"` is allowed.

## 17. Scientific Conclusion Disclaimer
This run constitutes a bounded baseline stack sanity execution (smoke evidence) only. No final baseline performance comparison or model evaluation conclusions are made.

## 18. Remaining Blockers
None.

## 19. Post-Commit/Push Evidence
- **Branch:** `phase2/p22-artifact-backed-baseline-evidence-smoke`
- **Commit Hash:** `447debd0b38763ce738f6f4d9302aca54b22e7ad`
- **git ls-remote Hash:** `447debd0b38763ce738f6f4d9302aca54b22e7ad`

## 20. Final Verdict
```text id="p22_verdict"
P22_READY_FOR_REVIEW
```
