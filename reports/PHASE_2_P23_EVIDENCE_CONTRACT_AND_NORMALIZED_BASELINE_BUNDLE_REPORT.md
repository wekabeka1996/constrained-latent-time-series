# Phase 2 P23: Evidence Contract and Normalized Baseline Bundle Report

## 1. Task Summary
This task converts the accepted P22 artifact-backed baseline smoke summary into a deterministic, schema-like evidence bundle that conforms to a strict evidence contract. We have:
- Created the core evidence contract module `src/phase2/evidence.py`.
- Updated `src/phase2/__init__.py` to export all constants, classes, and validators of the evidence contract.
- Created `tools/phase2/run_p23_evidence_contract_smoke.py` to pipe the P22 smoke test summary through the normalization layer.
- Added comprehensive unit tests and smoke tests ensuring that all validations, conversions, exports, and scope restrictions are strictly verified.
- Avoided all VAE/training loops, new dependencies, and local path or raw parameter leakages.

## 2. Base Commit Verification
- **Base branch:** `phase2/p22-artifact-backed-baseline-evidence-smoke`
- **Baseline HEAD commit:** `95b09ba19438b584fc630eea1bfeced57baee5de` (Verified that HEAD contains P11 fix commit `12230a465c1d5d518a68263c036d80ddb8c1d0d6` and P21 scope guard).

## 3. Files Created
- `src/phase2/evidence.py`
- `tests/test_phase2_evidence_contract.py`
- `tools/phase2/run_p23_evidence_contract_smoke.py`
- `tests/test_phase2_p23_evidence_contract_smoke.py`
- `reports/PHASE_2_P23_EVIDENCE_CONTRACT_AND_NORMALIZED_BASELINE_BUNDLE_REPORT.md`

## 4. Files Modified
- `src/phase2/__init__.py`

## 5. Files Not Changed
All other files, including the accepted P22 runner `tools/phase2/run_p22_artifact_backed_baseline_smoke.py` and its tests `tests/test_phase2_p22_artifact_backed_baseline_smoke.py` remain completely unchanged.

## 6. Commands Run
- **Required test command:**
  `python -m pytest tests/test_phase2_schema.py tests/test_phase2_constraints.py tests/test_phase2_sampler.py tests/test_phase2_simulator.py tests/test_phase2_dataset.py tests/test_phase2_artifacts.py tests/test_phase2_split_runner.py tests/test_phase2_protocol_presets.py tests/test_phase2_p14_smoke_dry_run.py tests/test_phase2_p15_manifest_audit.py tests/test_phase2_p16_dev_dry_run.py tests/test_phase2_metrics.py tests/test_phase2_baseline_eval.py tests/test_phase2_baseline_generators.py tests/test_phase2_p20_baseline_evidence_smoke.py tests/test_phase2_static_scope_guard.py tests/test_phase2_p22_artifact_backed_baseline_smoke.py tests/test_phase2_evidence_contract.py tests/test_phase2_p23_evidence_contract_smoke.py -q`
- **Optional compatibility command:**
  `python -m pytest tests/test_config.py tests/test_data_generator.py tests/test_validation.py tests/test_vector_schema.py tests/test_fail_fast_integrity.py tests/test_phase2_schema.py tests/test_phase2_constraints.py tests/test_phase2_sampler.py tests/test_phase2_simulator.py tests/test_phase2_dataset.py tests/test_phase2_artifacts.py tests/test_phase2_split_runner.py tests/test_phase2_protocol_presets.py tests/test_phase2_p14_smoke_dry_run.py tests/test_phase2_p15_manifest_audit.py tests/test_phase2_p16_dev_dry_run.py tests/test_phase2_metrics.py tests/test_phase2_baseline_eval.py tests/test_phase2_baseline_generators.py tests/test_phase2_p20_baseline_evidence_smoke.py tests/test_phase2_static_scope_guard.py tests/test_phase2_p22_artifact_backed_baseline_smoke.py tests/test_phase2_evidence_contract.py tests/test_phase2_p23_evidence_contract_smoke.py -q`
- **Smoke execution command:**
  `$env:PYTHONPATH="."; python tools/phase2/run_p23_evidence_contract_smoke.py`

## 7. Tests Run and Exact Results
- **Required Tests:** 702 passed, 1 skipped.
- **Optional Compatibility Tests:** 793 passed, 5 skipped.

## 8. Evidence Contract API Summary
The `src/phase2/evidence.py` module defines:
- **Constants:**
  - `EVIDENCE_CONTRACT_VERSION`: `"phase2_p23_evidence_contract_v1"`
  - `EVIDENCE_RUN_KIND_BASELINE_ARTIFACT_SMOKE`: `"baseline_artifact_smoke"`
- **Serialization Functions:**
  - `evidence_bundle_to_json_dict`: Converts the frozen bundle into a standard, sorted, schema-like JSON-friendly dict using an unchecked private serializer helper `_evidence_bundle_to_json_dict_unchecked`.
  - `compact_evidence_json`: Exports the JSON dict as compact string using `json.dumps(..., sort_keys=True, separators=(",", ":"))`.

## 9. Evidence Dataclasses Summary
Defined 6 frozen dataclasses:
1. `EvidenceRunMetadata`: Tracks version, run kind, source phase (`"P22"`), artifact root, audit validation status, and scope confirmation boolean flags.
2. `EvidenceReferenceRecord`: Stores role, sample id, line number, and spec structure strings (family id, mean family, volatility family).
3. `EvidenceMetricRecord`: Maps validity, composition, and novelty counts and rates, distribution distance, and stability presence.
4. `EvidenceGenerationRecord`: Records candidate count, family ids tuple, parent references indices, and seeds used.
5. `EvidenceBaselineRecord`: Associates baseline name with generation/metric records and bridge verification flags.
6. `EvidenceBundle`: Encapsulates metadata, references, baselines, and verdict.

## 10. Evidence Validation Summary
Implements value-level type/bounds checks and structure-level safety checks:
- `validate_non_empty_str`, `validate_bool`, `validate_non_negative_int`, `validate_positive_int`, `validate_probability`, `validate_optional_finite_float`.
- `assert_no_local_path_leakage`: Rejects absolute paths (`C:/`, `/home/`, etc.).
- `assert_no_raw_params`: Rejects dictionary keys or values containing raw param names (`ar_params`, etc.) in key-like form.
- `assert_no_forbidden_claims`: Rejects words like `"best"`, `"winner"`, `"solved"`, `"model works"` (while allowing `"no_final_comparison"` and `"no_scientific_conclusion"`).
- `validate_evidence_bundle` calls the above recursively.

## 11. P22-to-P23 Normalization Summary
The conversion function `normalize_p22_smoke_summary_to_evidence_bundle` maps the P22 smoke run output to the strict `EvidenceBundle` structure:
- Extracts references (ARMA mean, GARCH volatility sources) mapping their roles.
- Maps `copy_reference`, `random_valid`, and `structural_composition_oracle` baseline summaries in order.
- Validates the entire bundle prior to returning it.

## 12. P23 Smoke Command and Result
- **Smoke execution command:**
  `$env:PYTHONPATH="."; python tools/phase2/run_p23_evidence_contract_smoke.py`
- **Exit code:** 0
- **Status:** PASS (contract version matches, no path leaks, no parameter leaks, no forbidden claims).

## 13. Sanitized Stdout JSON Excerpt
```json
{"contract":"phase2_p23_evidence_contract_v1","evidence_bundle":{"baseline_count":3,"baselines":[{"baseline_name":"copy_reference","generation":{"candidate_count":6,"candidate_family_ids":["ARMA","GARCH","ARMA","GARCH","ARMA","GARCH"],"seed_sequence":[null,null,null,null,null,null],"source_reference_indices":[[0],[1],[0],[1],[0],[1]]},"metric_bundle_bridge_verified":true,"metrics":{"candidate_record_count":6,"composition_pass_count":0,"composition_pass_rate":0.0,"distribution_mmd_rbf":0.0,"novelty_pass_count":0,"novelty_pass_rate":0.0,"seed_stability_present":false,"validity_total_count":6,"validity_valid_count":6,"validity_valid_rate":1.0},"reason":"p22_artifact_backed_baseline_smoke_single_baseline_completed: copy_reference"},{"baseline_name":"random_valid","generation":{"candidate_count":6,"candidate_family_ids":["AR","GARCH","ARMA_GARCH","AR","GARCH","ARMA_GARCH"],"seed_sequence":[22001,22002,22003,22004,22005,22006],"source_reference_indices":[[],[],[],[],[],[]]},"metric_bundle_bridge_verified":true,"metrics":{"candidate_record_count":6,"composition_pass_count":2,"composition_pass_rate":0.3333333333333333,"distribution_mmd_rbf":0.467126027327986,"novelty_pass_count":0,"novelty_pass_rate":0.0,"seed_stability_present":false,"validity_total_count":6,"validity_valid_count":6,"validity_valid_rate":1.0},"reason":"p22_artifact_backed_baseline_smoke_single_baseline_completed: random_valid"},{"baseline_name":"structural_composition_oracle","generation":{"candidate_count":6,"candidate_family_ids":["ARMA_GARCH","ARMA_GARCH","ARMA_GARCH","ARMA_GARCH","ARMA_GARCH","ARMA_GARCH"],"seed_sequence":[null,null,null,null,null,null],"source_reference_indices":[[0,1],[0,1],[0,1],[0,1],[0,1],[0,1]]},"metric_bundle_bridge_verified":true,"metrics":{"candidate_record_count":6,"composition_pass_count":6,"composition_pass_rate":1.0,"distribution_mmd_rbf":1.498526078428923,"novelty_pass_count":0,"novelty_pass_rate":0.0,"seed_stability_present":false,"validity_total_count":6,"validity_valid_count":6,"validity_valid_rate":1.0},"reason":"p22_artifact_backed_baseline_smoke_single_baseline_completed: structural_composition_oracle"}],"candidate_count_per_baseline":6,"loaded_reference_count":2,"metadata":{"artifact_root":"phase2_artifacts/p14_smoke_dry_run","contract_version":"phase2_p23_evidence_contract_v1","no_artifact_generation":true,"no_final_comparison":true,"no_model_training":true,"no_scientific_conclusion":true,"p14_audit_verified":true,"reason":"normalized_from_p22_artifact_backed_baseline_smoke","run_kind":"baseline_artifact_smoke","source_phase":"P22"},"references":[{"family_id":"ARMA","line_number":1001,"mean_family":"ARMA","role":"arma_mean_source","sample_id":"ARMA_zero_shot_train_000001_g12001_s1012001_29dcbe9e1c07","volatility_family":"NONE"},{"family_id":"GARCH","line_number":2001,"mean_family":"NONE","role":"garch_volatility_source","sample_id":"GARCH_zero_shot_train_000001_g12002_s1012002_6d653cb4f1b3","volatility_family":"GARCH"}],"verdict":"PASS"},"no_artifact_generation":true,"no_final_comparison":true,"no_model_training":true,"no_scientific_conclusion":true,"reason":"p23_evidence_contract_smoke_completed","source_phase":"P22","verdict":"PASS"}
```

## 14. Raw Parameter Leakage Check
The raw parameter leakage checks recursively verify that none of the dictionary keys or values contain forbidden key strings (`ar_params`, etc.) in key-like format.
Result: **PASS (zero leaks detected)**.

## 15. Forbidden Claim Check
The forbidden claim checker validates that no scientific success, model performance, or solver claims are made.
Result: **PASS (all claims verified clean; scope flags allowed correctly)**.

## 16. Scope Confirmation
It is explicitly confirmed that:
- No neural model or VAE is introduced.
- No training loops or optimizers are present.
- No model checkpointing or model evaluations were run.
- No generated dataset JSONL or manifest file writing occurred during P23.
- No argparse or config file loader was added.
- No new external dependencies are imported.

## 17. No P16 Artifact Dependency Confirmation
Verified that the P23 code does not read, write, or have dependencies on P16 dev dry-run artifacts. Static scope checks pass with strict forbidden tokens checks.

## 18. Scientific Conclusion Disclaimer
This execution represents a bounded evidence contract normalization check (smoke run) only. No scientific conclusions are claimed regarding baseline performance or model comparative metrics.

## 19. Remaining Blockers
None.

## 20. Post-Commit/Push Evidence
- **Branch:** `phase2/p23-evidence-contract-normalized-baseline-bundle`
- **Commit Hash:** `5e3d6ea2c9c5fdc15bd5fef42739c8b172b626b1`
- **git ls-remote Hash:** `5e3d6ea2c9c5fdc15bd5fef42739c8b172b626b1`

## 21. Final Verdict
```text id="p23_verdict"
P23_READY_FOR_REVIEW
```
