# Phase 2 P20: Bounded Baseline Evidence Smoke Run Report

## 1. Task Summary
This task implements and executes a small deterministic in-memory baseline evidence smoke run using the accepted Phase 2 baseline generation and evaluation stack.

## 2. Base Commit Verification
- Base branch: `phase2/p19-baseline-candidate-generators`
- Base commit: `922f655e97b26bed78945fb419e939da5428487a` (incorporating P11 fix commit `12230a465c1d5d518a68263c036d80ddb8c1d0d6`)

## 3. Files Created
- `tools/phase2/run_p20_baseline_evidence_smoke.py`
- `tests/test_phase2_p20_baseline_evidence_smoke.py`
- `reports/PHASE_2_P20_BOUNDED_BASELINE_EVIDENCE_SMOKE_RUN_REPORT.md`

## 4. Files Modified
None.

## 5. Files Not Changed
All other files in the repository remain unchanged.

## 6. Commands Run
- `python -m pytest tests/test_phase2_schema.py tests/test_phase2_constraints.py tests/test_phase2_sampler.py tests/test_phase2_simulator.py tests/test_phase2_dataset.py tests/test_phase2_artifacts.py tests/test_phase2_split_runner.py tests/test_phase2_protocol_presets.py tests/test_phase2_p14_smoke_dry_run.py tests/test_phase2_p15_manifest_audit.py tests/test_phase2_p16_dev_dry_run.py tests/test_phase2_metrics.py tests/test_phase2_baseline_eval.py tests/test_phase2_baseline_generators.py tests/test_phase2_p20_baseline_evidence_smoke.py -q`
- `python -m pytest tests/test_config.py tests/test_data_generator.py tests/test_validation.py tests/test_vector_schema.py tests/test_fail_fast_integrity.py tests/test_phase2_schema.py tests/test_phase2_constraints.py tests/test_phase2_sampler.py tests/test_phase2_simulator.py tests/test_phase2_dataset.py tests/test_phase2_artifacts.py tests/test_phase2_split_runner.py tests/test_phase2_protocol_presets.py tests/test_phase2_p14_smoke_dry_run.py tests/test_phase2_p15_manifest_audit.py tests/test_phase2_p16_dev_dry_run.py tests/test_phase2_metrics.py tests/test_phase2_baseline_eval.py tests/test_phase2_baseline_generators.py tests/test_phase2_p20_baseline_evidence_smoke.py -q`
- `$env:PYTHONPATH="."; python tools/phase2/run_p20_baseline_evidence_smoke.py`

## 7. Tests Run and Exact Results
- **Phase 2 Tests (Required)**: 600 passed, 1 skipped.
- **Compatibility Tests (Optional)**: 691 passed, 5 skipped.

## 8. Smoke Execution Command
```bash
$env:PYTHONPATH="."; python tools/phase2/run_p20_baseline_evidence_smoke.py
```

## 9. Sanitized Stdout JSON Excerpt
```json
{"baseline_count":3,"baselines":[{"baseline_name":"copy_reference","evaluation_summary":{"baseline_name":"copy_reference","candidate_count":6,"candidate_record_count":6,"composition_pass_count":0,"composition_pass_rate":0.0,"distribution_mmd_rbf":0.0,"novelty_pass_count":0,"novelty_pass_rate":0.0,"reference_count":2,"seed_stability_present":false,"validity_total_count":6,"validity_valid_count":6,"validity_valid_rate":1.0},"generation_summary":{"baseline_name":"copy_reference","candidate_count":6,"candidate_family_ids":["ARMA","GARCH","ARMA","GARCH","ARMA","GARCH"],"source_records_summary":[{"candidate_index":0,"family_id":"ARMA","generator_name":"copy_reference","seed":null,"source_reference_indices":[0]},{"candidate_index":1,"family_id":"GARCH","generator_name":"copy_reference","seed":null,"source_reference_indices":[1]},{"candidate_index":2,"family_id":"ARMA","generator_name":"copy_reference","seed":null,"source_reference_indices":[0]},{"candidate_index":3,"family_id":"GARCH","generator_name":"copy_reference","seed":null,"source_reference_indices":[1]},{"candidate_index":4,"family_id":"ARMA","generator_name":"copy_reference","seed":null,"source_reference_indices":[0]},{"candidate_index":5,"family_id":"GARCH","generator_name":"copy_reference","seed":null,"source_reference_indices":[1]}]},"metric_bundle_bridge_verified":true,"reason":"p20_bounded_baseline_evidence_smoke_single_baseline_completed: copy_reference"},{"baseline_name":"random_valid","evaluation_summary":{"baseline_name":"random_valid","candidate_count":6,"candidate_record_count":6,"composition_pass_count":2,"composition_pass_rate":0.3333333333333333,"distribution_mmd_rbf":0.5048600943969466,"novelty_pass_count":0,"novelty_pass_rate":0.0,"reference_count":2,"seed_stability_present":false,"validity_total_count":6,"validity_valid_count":6,"validity_valid_rate":1.0},"generation_summary":{"baseline_name":"random_valid","candidate_count":6,"candidate_family_ids":["AR","GARCH","ARMA_GARCH","AR","GARCH","ARMA_GARCH"],"source_records_summary":[{"candidate_index":0,"family_id":"AR","generator_name":"random_valid","seed":19001,"source_reference_indices":[]},{"candidate_index":1,"family_id":"GARCH","generator_name":"random_valid","seed":19002,"source_reference_indices":[]},{"candidate_index":2,"family_id":"ARMA_GARCH","generator_name":"random_valid","seed":19003,"source_reference_indices":[]},{"candidate_index":3,"family_id":"AR","generator_name":"random_valid","seed":19004,"source_reference_indices":[]},{"candidate_index":4,"family_id":"GARCH","generator_name":"random_valid","seed":19005,"source_reference_indices":[]},{"candidate_index":5,"family_id":"ARMA_GARCH","generator_name":"random_valid","seed":19006,"source_reference_indices":[]}]},"metric_bundle_bridge_verified":true,"reason":"p20_bounded_baseline_evidence_smoke_single_baseline_completed: random_valid"},{"baseline_name":"structural_composition_oracle","evaluation_summary":{"baseline_name":"structural_composition_oracle","candidate_count":6,"candidate_record_count":6,"composition_pass_count":6,"composition_pass_rate":1.0,"distribution_mmd_rbf":1.498811304940202,"novelty_pass_count":0,"novelty_pass_rate":0.0,"reference_count":2,"seed_stability_present":false,"validity_total_count":6,"validity_valid_count":6,"validity_valid_rate":1.0},"generation_summary":{"baseline_name":"structural_composition_oracle","candidate_count":6,"candidate_family_ids":["ARMA_GARCH","ARMA_GARCH","ARMA_GARCH","ARMA_GARCH","ARMA_GARCH","ARMA_GARCH"],"source_records_summary":[{"candidate_index":0,"family_id":"ARMA_GARCH","generator_name":"structural_composition_oracle","seed":null,"source_reference_indices":[0,1]},{"candidate_index":1,"family_id":"ARMA_GARCH","generator_name":"structural_composition_oracle","seed":null,"source_reference_indices":[0,1]},{"candidate_index":2,"family_id":"ARMA_GARCH","generator_name":"structural_composition_oracle","seed":null,"source_reference_indices":[0,1]},{"candidate_index":3,"family_id":"ARMA_GARCH","generator_name":"structural_composition_oracle","seed":null,"source_reference_indices":[0,1]},{"candidate_index":4,"family_id":"ARMA_GARCH","generator_name":"structural_composition_oracle","seed":null,"source_reference_indices":[0,1]},{"candidate_index":5,"family_id":"ARMA_GARCH","generator_name":"structural_composition_oracle","seed":null,"source_reference_indices":[0,1]}]},"metric_bundle_bridge_verified":true,"reason":"p20_bounded_baseline_evidence_smoke_single_baseline_completed: structural_composition_oracle"}],"candidate_count_per_baseline":6,"contract":"phase2_p20_bounded_baseline_evidence_smoke_v1","no_artifact_dependency":true,"no_model_training":true,"no_scientific_conclusion":true,"reason":"p20_bounded_baseline_evidence_smoke_completed","reference_count":2,"verdict":"PASS"}
```

## 10. Reference Fixture Summary
Built 2 baseline reference fixtures in memory:
- **ARMA mean-only**: `FamilyId.ARMA`, `MeanFamily.ARMA`, `VolatilityFamily.NONE`, `p=1`, `q=1`, `ar_params=(0.35,)`, `ma_params=(-0.25,)`.
- **GARCH volatility-only**: `FamilyId.GARCH`, `MeanFamily.NONE`, `VolatilityFamily.GARCH`, `r=1`, `s=1`, `omega=0.40`, `alpha_params=(0.08,)`, `beta_params=(0.75,)`.

## 11. Baseline Generation Summary
- **copy_reference**: Successfully cycled through reference specs to produce 6 candidates with appropriate provenance tags.
- **random_valid**: Correctly dispatched custom generation requests with templates and explicit family schedule to produce 6 deterministic candidates with seeds `19001` through `19006`.
- **structural_composition_oracle**: Successfully merged parameters from mean and volatility sources to output 6 `ARMA_GARCH` candidates.

## 12. Baseline Evaluation Summary
- Aggregate validity for all candidates is `1.0`.
- The structural oracle correctly obtained a `composition_pass_rate` of `1.0`.
- Distribution distance `mmd_rbf` was computed deterministically across references and candidates.

## 13. MetricBundle Bridge Verification
Programmatic checks verify that `build_baseline_metric_bundle` is active and generates a valid `MetricBundle` from baseline evaluation results.

## 14. Determinism Verification
Repeated calls to generators, evaluation, and bridge return identical summaries, proving absolute determinism of the baseline evaluation stack.

## 15. Scope Confirmation
No neural model, no VAE, no training loop, no optimizer, no checkpointing, no final baseline comparison report, no generated dataset JSONL, no argparse CLI, and no new dependencies were introduced.

## 16. No P14/P16 Artifact Dependency Confirmation
Confirmed that the code does not read, write, or depend on any P14/P16 generated artifact files.

## 17. Scientific Conclusion Disclaimer
This execution represents a bounded deterministic baseline stack check (smoke evidence) only. No scientific conclusions are claimed regarding baseline performance or model results.

## 18. Remaining Blockers
None.

## 19. Post-Commit/Push Evidence
- Branch: `phase2/p20-bounded-baseline-evidence-smoke-run`
- Commit: `3041b511d587367b706322234b0228d8dea827e7`
- git ls-remote hash: `3041b511d587367b706322234b0228d8dea827e7`

## 20. Final Verdict
**P20_READY_FOR_REVIEW**
