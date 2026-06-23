# Phase 2 P21 Scope and Process Hardening Report

## 1. Task Summary
Harden the accepted P20 baseline smoke layer by:
- Replacing comment-only P20 scope tests with real static source checks.
- Removing unused imports (`math`, `Dict`, `Optional`) from the P20 smoke script.
- Adding a small reusable static scope guard utility (`tools/phase2/static_scope_guard.py`).
- Ensuring P20 runtime behavior is preserved exactly.
- Confirming that no neural models, training, or artifact-backed evaluations are introduced.

## 2. Base Commit Verification
- HEAD contains accepted P11 fix commit: `12230a465c1d5d518a68263c036d80ddb8c1d0d6` (Verified via `git branch --contains`).
- HEAD contains accepted P20 files:
  - `tools/phase2/run_p20_baseline_evidence_smoke.py`
  - `tests/test_phase2_p20_baseline_evidence_smoke.py`
  - `reports/PHASE_2_P20_BOUNDED_BASELINE_EVIDENCE_SMOKE_RUN_REPORT.md`
- `.gitignore` includes `phase2_artifacts/`.

## 3. Files Created
- `tools/phase2/static_scope_guard.py`
- `tests/test_phase2_static_scope_guard.py`
- `reports/PHASE_2_P21_SCOPE_AND_PROCESS_HARDENING_REPORT.md`

## 4. Files Modified
- `tools/phase2/run_p20_baseline_evidence_smoke.py`
- `tests/test_phase2_p20_baseline_evidence_smoke.py`

## 5. Files Not Changed
- `src/phase2/schema.py`
- `src/phase2/constraints.py`
- `src/phase2/sampler.py`
- `src/phase2/simulator.py`
- `src/phase2/dataset.py`
- `src/phase2/artifacts.py`
- `src/phase2/split_runner.py`
- `src/phase2/protocol_presets.py`
- `src/phase2/metrics.py`
- `src/phase2/baseline_eval.py`
- `src/phase2/baseline_generators.py`
- `src/phase2/__init__.py`
- `tools/phase2/run_p14_smoke_artifact_dry_run.py`
- `tools/phase2/audit_phase2_artifact_manifest.py`
- `tools/phase2/run_p16_dev_artifact_dry_run.py`
- `reports/PHASE_2_P20_BOUNDED_BASELINE_EVIDENCE_SMOKE_RUN_REPORT.md`
- `configs/`
- `models/`
- `archive/`
- `docs/`
- `requirements.txt`
- `pyproject.toml`
- `README.md`
- `ROADMAP.md`

## 6. Commands Run
- Required pytest command:
  `python -m pytest tests/test_phase2_schema.py tests/test_phase2_constraints.py tests/test_phase2_sampler.py tests/test_phase2_simulator.py tests/test_phase2_dataset.py tests/test_phase2_artifacts.py tests/test_phase2_split_runner.py tests/test_phase2_protocol_presets.py tests/test_phase2_p14_smoke_dry_run.py tests/test_phase2_p15_manifest_audit.py tests/test_phase2_p16_dev_dry_run.py tests/test_phase2_metrics.py tests/test_phase2_baseline_eval.py tests/test_phase2_baseline_generators.py tests/test_phase2_p20_baseline_evidence_smoke.py tests/test_phase2_static_scope_guard.py -q`
- Optional compatibility command:
  `python -m pytest tests/test_config.py tests/test_data_generator.py tests/test_validation.py tests/test_vector_schema.py tests/test_fail_fast_integrity.py tests/test_phase2_schema.py tests/test_phase2_constraints.py tests/test_phase2_sampler.py tests/test_phase2_simulator.py tests/test_phase2_dataset.py tests/test_phase2_artifacts.py tests/test_phase2_split_runner.py tests/test_phase2_protocol_presets.py tests/test_phase2_p14_smoke_dry_run.py tests/test_phase2_p15_manifest_audit.py tests/test_phase2_p16_dev_dry_run.py tests/test_phase2_metrics.py tests/test_phase2_baseline_eval.py tests/test_phase2_baseline_generators.py tests/test_phase2_p20_baseline_evidence_smoke.py tests/test_phase2_static_scope_guard.py -q`
- Smoke run command:
  `$env:PYTHONPATH="."; python tools/phase2/run_p20_baseline_evidence_smoke.py`

## 7. Tests Run and Exact Results
- **Required Tests**: 620 passed, 1 skipped in 61.55s.
- **Optional Compatibility Tests**: 711 passed, 5 skipped in 61.22s.

## 8. Static Scope Guard API Summary
The `tools/phase2/static_scope_guard.py` module defines:
- **Constants**:
  - `FORBIDDEN_THIRD_PARTY_IMPORT_ROOTS`: `("torch", "numpy", "pandas", "yaml", "sklearn", "scipy")`
  - `FORBIDDEN_PHASE2_RUNTIME_CALLS`: `("simulate_time_series", "build_dataset_in_memory", "write_dataset_artifacts", "run_phase2_artifact_generation")`
  - `FORBIDDEN_ARTIFACT_PATH_TOKENS`: `("phase2_artifacts", "P14", "P16")`
- **Dataclass**:
  - `StaticScopeCheckResult` (frozen): containing `path`, `forbidden_import_hits`, `forbidden_call_hits`, `forbidden_path_token_hits`, `passed`, `reason`.
- **Functions**:
  - `read_text_file(path: str) -> str`
  - `parse_python_source(source: str) -> ast.AST`
  - `collect_import_roots(source: str) -> tuple[str, ...]`
  - `collect_call_names(source: str) -> tuple[str, ...]`
  - `collect_string_token_hits(source: str, tokens: tuple[str, ...]) -> tuple[str, ...]`
  - `check_static_scope(...) -> StaticScopeCheckResult`
  - `require_static_scope_pass(result: StaticScopeCheckResult) -> None`

## 9. P20 Script Cleanup Summary
- Unused imports `math`, `Dict`, `Optional` were removed from `tools/phase2/run_p20_baseline_evidence_smoke.py`.
- No functional behavior or output formatting was changed.
- No filesystem read/write or argparse/config loader dependency was added.

## 10. P20 Scope Test Hardening Summary
Comment-only P20 scope tests in `tests/test_phase2_p20_baseline_evidence_smoke.py` were replaced with real static checks using `tools.phase2.static_scope_guard`:
- `test_p20_35_does_not_call_simulate_time_series`
- `test_p20_36_does_not_call_build_dataset`
- `test_p20_37_does_not_call_write_artifacts`
- `test_p20_38_does_not_call_run_generation`
- `test_p20_39_does_not_read_phase2_artifacts`
- `test_p20_40_tests_do_not_read_artifacts` (AST constant string checking; uses concatenation of literals to prevent self-reference detection failure)
- `test_p20_41_no_cli_or_config`

## 11. P20 Smoke Re-Execution Summary
- Command: `$env:PYTHONPATH="."; python tools/phase2/run_p20_baseline_evidence_smoke.py`
- Exit Code: `0`
- Format: Parseable compact JSON.
- Leak check: Verified zero local path leaks.

## 12. Behavior Preservation Evidence
Stdout JSON matches P20 exactly:
```json
{"baseline_count":3,"baselines":[{"baseline_name":"copy_reference","evaluation_summary":{"baseline_name":"copy_reference","candidate_count":6,"candidate_record_count":6,"composition_pass_count":0,"composition_pass_rate":0.0,"distribution_mmd_rbf":0.0,"novelty_pass_count":0,"novelty_pass_rate":0.0,"reference_count":2,"seed_stability_present":false,"validity_total_count":6,"validity_valid_count":6,"validity_valid_rate":1.0},"generation_summary":{"baseline_name":"copy_reference","candidate_count":6,"candidate_family_ids":["ARMA","GARCH","ARMA","GARCH","ARMA","GARCH"],"source_records_summary":[{"candidate_index":0,"family_id":"ARMA","generator_name":"copy_reference","seed":null,"source_reference_indices":[0]},{"candidate_index":1,"family_id":"GARCH","generator_name":"copy_reference","seed":null,"source_reference_indices":[1]},{"candidate_index":2,"family_id":"ARMA","generator_name":"copy_reference","seed":null,"source_reference_indices":[0]},{"candidate_index":3,"family_id":"GARCH","generator_name":"copy_reference","seed":null,"source_reference_indices":[1]},{"candidate_index":4,"family_id":"ARMA","generator_name":"copy_reference","seed":null,"source_reference_indices":[0]},{"candidate_index":5,"family_id":"GARCH","generator_name":"copy_reference","seed":null,"source_reference_indices":[1]}]},"metric_bundle_bridge_verified":true,"reason":"p20_bounded_baseline_evidence_smoke_single_baseline_completed: copy_reference"},{"baseline_name":"random_valid","evaluation_summary":{"baseline_name":"random_valid","candidate_count":6,"candidate_record_count":6,"composition_pass_count":2,"composition_pass_rate":0.3333333333333333,"distribution_mmd_rbf":0.5048600943969466,"novelty_pass_count":0,"novelty_pass_rate":0.0,"reference_count":2,"seed_stability_present":false,"validity_total_count":6,"validity_valid_count":6,"validity_valid_rate":1.0},"generation_summary":{"baseline_name":"random_valid","candidate_count":6,"candidate_family_ids":["AR","GARCH","ARMA_GARCH","AR","GARCH","ARMA_GARCH"],"source_records_summary":[{"candidate_index":0,"family_id":"AR","generator_name":"random_valid","seed":19001,"source_reference_indices":[]},{"candidate_index":1,"family_id":"GARCH","generator_name":"random_valid","seed":19002,"source_reference_indices":[]},{"candidate_index":2,"family_id":"ARMA_GARCH","generator_name":"random_valid","seed":19003,"source_reference_indices":[]},{"candidate_index":3,"family_id":"AR","generator_name":"random_valid","seed":19004,"source_reference_indices":[]},{"candidate_index":4,"family_id":"GARCH","generator_name":"random_valid","seed":19005,"source_reference_indices":[]},{"candidate_index":5,"family_id":"ARMA_GARCH","generator_name":"random_valid","seed":19006,"source_reference_indices":[]}]},"metric_bundle_bridge_verified":true,"reason":"p20_bounded_baseline_evidence_smoke_single_baseline_completed: random_valid"},{"baseline_name":"structural_composition_oracle","evaluation_summary":{"baseline_name":"structural_composition_oracle","candidate_count":6,"candidate_record_count":6,"composition_pass_count":6,"composition_pass_rate":1.0,"distribution_mmd_rbf":1.498811304940202,"novelty_pass_count":0,"novelty_pass_rate":0.0,"reference_count":2,"seed_stability_present":false,"validity_total_count":6,"validity_valid_count":6,"validity_valid_rate":1.0},"generation_summary":{"baseline_name":"structural_composition_oracle","candidate_count":6,"candidate_family_ids":["ARMA_GARCH","ARMA_GARCH","ARMA_GARCH","ARMA_GARCH","ARMA_GARCH","ARMA_GARCH"],"source_records_summary":[{"candidate_index":0,"family_id":"ARMA_GARCH","generator_name":"structural_composition_oracle","seed":null,"source_reference_indices":[0,1]},{"candidate_index":1,"family_id":"ARMA_GARCH","generator_name":"structural_composition_oracle","seed":null,"source_reference_indices":[0,1]},{"candidate_index":2,"family_id":"ARMA_GARCH","generator_name":"structural_composition_oracle","seed":null,"source_reference_indices":[0,1]},{"candidate_index":3,"family_id":"ARMA_GARCH","generator_name":"structural_composition_oracle","seed":null,"source_reference_indices":[0,1]},{"candidate_index":4,"family_id":"ARMA_GARCH","generator_name":"structural_composition_oracle","seed":null,"source_reference_indices":[0,1]},{"candidate_index":5,"family_id":"ARMA_GARCH","generator_name":"structural_composition_oracle","seed":null,"source_reference_indices":[0,1]}]},"metric_bundle_bridge_verified":true,"reason":"p20_bounded_baseline_evidence_smoke_single_baseline_completed: structural_composition_oracle"}],"candidate_count_per_baseline":6,"contract":"phase2_p20_bounded_baseline_evidence_smoke_v1","no_artifact_dependency":true,"no_model_training":true,"no_scientific_conclusion":true,"reason":"p20_bounded_baseline_evidence_smoke_completed","reference_count":2,"verdict":"PASS"}
```

## 13. Scope Confirmation
No model, training, VAE, optimizer, checkpointing, dataset writing (JSONL/manifest), CLI frameworks, argparse, YAML configurations, or third-party dependencies (numpy, pandas, torch, scipy, sklearn) were introduced or imported in the code.

## 14. No P14/P16 Artifact Dependency Confirmation
Confirmed that the code does not read, require, or have dependencies on generated P14/P16 artifacts.

## 15. Remaining Blockers
None.

## 16. Post-Commit/Push Evidence
- **Branch**: `phase2/p21-scope-process-hardening`
- **Commit Hash**: `a396ff31df53a89478dd0ea67d266a0ff960ec19`
- **git ls-remote Hash**: `a396ff31df53a89478dd0ea67d266a0ff960ec19`

## 17. Final Verdict
`P21_READY_FOR_REVIEW`
