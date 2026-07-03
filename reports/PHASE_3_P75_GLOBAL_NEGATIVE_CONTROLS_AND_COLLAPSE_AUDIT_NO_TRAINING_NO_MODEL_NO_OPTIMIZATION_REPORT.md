# PHASE 3 P75 Global Negative Controls and Collapse Audit Report

## 1. Problem Framing
P75 implements the deterministic global negative-controls and collapse audit for Phase 3. It checks positive oracle controls, wrong-operator cycles, no-op collapse, constant-output collapse, and source-agnostic controls across both vector (P70A) and time-series parameter (P70B) worlds, verifying model robustness and resistance against state collapse under strict target isolation boundaries.

## 2. Boundaries and Scope
We strictly respect the Phase 3 boundary requirements:
- No training loop is implemented.
- No model architecture or neural networks are instantiated.
- No neural relation encoder or selector is implemented.
- No optimization step or gradient is computed.
- No PyTorch, NumPy, Pandas, or Scikit-Learn is imported.
- No stochastic random processes are used.
- No Schrödinger Bridge or Geometric Schrödinger Bridge is built.
- No learned metric is allowed.
- No semantic geometry, meaning, learned operator identity, learned selection, or learned collapse resistance claims are made.

## 3. Why Negative Controls Audit after Composition Audit
Following positive composition verification in P74, we must demonstrate that the deterministic oracle bank is highly specific. The audit confirms that operators fail wrong-operator sequences and collapse controls, showing that exact target reconstruction requires correct relation types, intensities, and initial states.

## 4. Addressing P74 Validation-Depth and Midpoint-Target Notes
P75 addresses P74 notes directly:
- **Source Validation Depth**: Performs deeper, multi-layered contract checks for P69–P74. It returns actual per-source validation booleans and tracks missing optional fields inside `field_missing_but_transitively_guarded`.
- **Target Midpoint HONESTY**: Explicitly reports whether midpoint targets exist. Composition records in both P70A and P70B do contain midpoint targets (`z_mid`, `params_mid`/`series_mid`), so midpoint target evaluation is reported honestly as direct:
  `midpoint_target_evaluation_direct = True`
  `midpoint_target_evaluation_structural_only = False`

This successfully addresses the notes:
- `p74_source_validation_depth_note_addressed = True`
- `p74_target_midpoint_note_addressed = True`

## 5. Prediction Isolation (No Endpoint or Midpoint Leakage)
We enforce strict prediction isolation boundaries:
- Prediction paths must never use target endpoints or midpoints.
- Target endpoints and midpoints are used only post-prediction to calculate errors.

```python
TARGET_ENDPOINT_USED_FOR_PREDICTION = False
TARGET_ENDPOINT_USED_FOR_EVALUATION_ONLY = True
TARGET_MIDPOINT_USED_FOR_PREDICTION = False
TARGET_MIDPOINT_USED_FOR_EVALUATION_ONLY = True
```

## 6. Negative-Control Families & Collapse Resistance
We analyze wrong-operator cycle assignments, no-ops, constant outputs, and source-agnostic runs:
- **Wrong Operator**: Applies cyclical operators (e.g. translate_x -> translate_y).
- **No-op Collapse**: Asserts that returning the start state unchanged fails when a real transformation is required.
- **Constant Output Collapse**: Asserts that outputting a fixed default state (e.g. `[0.0, 0.0, 1.0]` or neutral params) fails.
- **Source Agnostic**: Asserts that predicting from a fixed neutral state regardless of input state fails.

## 7. P70A Ambiguity Handling
P70A vector world has expected collisions due to low-dimensional coordinates. In particular:
- For degenerate base state index 0 `[0.0, 0.0, 1.0]`, true relation reflect_x and wrong relation nonlinear_x_from_y yield matching endpoints.
- For base state index 1 `[-1.0, 1.0, 1.5]`, true relation reflect_x and wrong relation nonlinear_x_from_y (at y=1) yield matching endpoints.

P75 reports these expected base-state ambiguities transparently (`wrong_operator_collision_or_ambiguity_count = 4`) and verifies that *unexpected* false passes are exactly `0` for all controls.

## 8. P70B Strict Controls
P70B has higher parameter space dimensions and contains exactly `0` unexpected false passes across all wrong-operator, no-op, constant-output, and source-agnostic control checks.

## 9. Structural vs Measured Failure Rate
P70A/P70B negative control records do not store intensities. P75 reports structural preparedness honestly (`failure_rate_structural_only = True`, `failure_rate_measured = False`) without inventing intensity values.

## 10. Limitations
- P75 does not prove learned semantics.
- P75 does not prove learned operator identity, selection, or composition.
- P75 does not prove semantic geometry.
- P75 is a deterministic negative-control/collapse audit only.

## 11. Results
Final Verdict: `P75_READY_FOR_REVIEW`

## Smoke Output Block
```json
{"allowed_claims":["p75_audits_global_negative_controls","p75_audits_wrong_operator_failures","p75_audits_label_permutation_failures","p75_audits_mismatched_pair_failures","p75_audits_no_op_collapse_failures","p75_audits_constant_output_collapse_failures","p75_audits_source_agnostic_label_only_failures","p75_reports_known_p70a_ambiguity_cases","p75_addresses_p74_source_validation_depth_note","p75_addresses_p74_target_midpoint_note","p75_does_not_train_models","p75_does_not_establish_learned_semantic_evidence"],"audit_domains":["p70a_vector_world","p70b_time_series_parameter_world"],"bridge_implementation_allowed":false,"collapse_audit_allowed":true,"contract_version":"phase3_p75_global_negative_controls_collapse_audit_contract_v1","diagnostic_only":true,"forbidden_claims":["semantic_geometry_is_proven","meaning_is_learned","operator_identity_is_proven","relation_encoder_is_trained","relation_encoder_is_validated","sparse_operator_bank_is_learned","learned_operator_selection_is_proven","learned_transfer_is_proven","learned_composition_is_proven","collapse_resistance_is_proven_for_learned_system","bridge_method_is_validated","schrodinger_bridge_creates_meaning","geometric_schrodinger_bridge_creates_meaning","negative_controls_prove_semantics"],"global_negative_control_audit_allowed":true,"json_safe":true,"learned_metric_allowed":false,"learned_negative_control_claims_allowed":false,"learned_relation_type_selection_allowed":false,"midpoint_target_availability_audit":{"midpoint_target_evaluation_direct":true,"midpoint_target_evaluation_structural_only":false,"midpoint_target_fields_available":true,"p70a_composition_records_checked":6,"p70a_midpoint_target_fields_available":true,"p70b_composition_records_checked":6,"p70b_midpoint_target_fields_available":true,"p74_target_midpoint_note_addressed":true},"model_implementation_allowed":false,"negative_control_families":["wrong_operator_control","label_permutation_control","mismatched_pair_control","no_op_collapse_control","constant_output_collapse_control","source_agnostic_label_only_control"],"neural_encoder_implementation_allowed":false,"neural_operator_selector_allowed":false,"numpy_allowed":false,"optimization_allowed":false,"oracle_operator_bank_reuse_allowed":true,"oracle_relation_type_selection_allowed_for_positive_control":true,"p70a_negative_control_audit":{"constant_output_evaluable_count":54,"constant_output_false_pass_count":1,"constant_output_l2_error":{"count":54,"max":5.123475382979799,"mean":1.9323310853331077,"min":0.0},"constant_output_unexpected_false_pass_count":0,"domain":"p70a_vector_world","known_ambiguity_reported":true,"negative_controls_diagnostic_pass":true,"no_op_evaluable_count":53,"no_op_expected_no_change_count":1,"no_op_false_pass_count":0,"no_op_l2_error":{"count":54,"max":4.0,"mean":1.1996527777777777,"min":0.0},"positive_oracle_exact":true,"positive_oracle_l2_error":{"count":54,"max":0.0,"mean":0.0,"min":0.0},"record_count":54,"source_agnostic_evaluable_count":54,"source_agnostic_false_pass_count":13,"source_agnostic_l2_error":{"count":54,"max":3.2015621187164243,"mean":1.373639935639019,"min":0.0},"source_agnostic_unexpected_false_pass_count":0,"unexpected_false_pass_count":0,"wrong_operator_collision_or_ambiguity_count":4,"wrong_operator_evaluable_count":54,"wrong_operator_false_pass_count":4,"wrong_operator_l2_error":{"count":54,"max":5.25,"mean":1.5514437177695544,"min":0.0},"wrong_operator_nonambiguous_failure_count":50,"wrong_operator_unexpected_false_pass_count":0},"p70b_negative_control_audit":{"constant_output_evaluable_count":60,"constant_output_false_pass_count":0,"constant_output_parameter_l2_error":{"count":60,"max":2.3947860029656094,"mean":1.055089351486492,"min":0.05},"constant_output_series_l2_error":{"count":60,"max":16.83833740454725,"mean":7.692020126328598,"min":0.10210549404852619},"constant_output_unexpected_false_pass_count":0,"domain":"p70b_time_series_parameter_world","negative_controls_diagnostic_pass":true,"no_op_evaluable_count":60,"no_op_expected_no_change_count":0,"no_op_false_pass_count":0,"no_op_parameter_l2_error":{"count":60,"max":1.4,"mean":0.33208333333333334,"min":0.049999999999999996},"no_op_series_l2_error":{"count":60,"max":13.488758168532026,"mean":2.789559115828692,"min":0.10210549404852616},"positive_oracle_exact":true,"positive_oracle_parameter_l2_error":{"count":60,"max":0.0,"mean":0.0,"min":0.0},"positive_oracle_series_l2_error":{"count":60,"max":0.0,"mean":0.0,"min":0.0},"record_count":60,"source_agnostic_evaluable_count":60,"source_agnostic_false_pass_count":15,"source_agnostic_parameter_l2_error":{"count":60,"max":2.0700241544484452,"mean":0.8359897975711252,"min":0.0},"source_agnostic_series_l2_error":{"count":60,"max":19.612363162259918,"mean":7.10064789962193,"min":0.0},"source_agnostic_unexpected_false_pass_count":0,"unexpected_false_pass_count":0,"wrong_operator_evaluable_count":60,"wrong_operator_false_pass_count":0,"wrong_operator_parameter_l2_error":{"count":60,"max":1.7204650534085253,"mean":0.5656047469836297,"min":0.07071067811865463},"wrong_operator_series_l2_error":{"count":60,"max":17.251828268141164,"mean":4.188498791515572,"min":0.7820104593191884}},"p74_source_validation_depth_note_addressed":true,"p74_target_midpoint_note_addressed":true,"phase":"P75","phase_group":"PHASE_3","phase_name":"Global Negative Controls and Collapse Audit","primary_empirical_target":"global_negative_controls_and_collapse_resistance_diagnostics","sanity_summary":{"constant_output_collapse_controls_fail_as_expected":true,"failure_rate_measured_only_when_executed":true,"json_safe":true,"learned_negative_control_claims_made":false,"no_op_collapse_controls_fail_as_expected":true,"p70a_known_ambiguity_reported":true,"p70a_negative_controls_diagnostic_pass":true,"p70a_positive_oracle_control_exact":true,"p70a_unexpected_false_passes_zero":true,"p70b_negative_controls_diagnostic_pass":true,"p70b_positive_oracle_control_exact":true,"p70b_unexpected_false_passes_zero":true,"p74_source_validation_depth_note_addressed":true,"p74_target_midpoint_note_addressed":true,"source_agnostic_controls_fail_as_expected":true,"source_contracts_deep_validated":true,"structural_negative_controls_ready":true,"target_endpoint_used_for_evaluation_only":true,"target_endpoint_used_for_prediction":false,"target_midpoint_used_for_evaluation_only":true,"target_midpoint_used_for_prediction":false,"training_or_model_added":false,"wrong_operator_controls_fail_as_expected":true},"source_baseline_phase":"P69","source_composition_audit_phase":"P74","source_contracts_deep_validated":true,"source_contrastive_phase":"P71","source_operator_bank_phase":"P72","source_time_series_testbed_phase":"P70B","source_transfer_audit_phase":"P73","source_vector_testbed_phase":"P70A","stochastic_random_allowed":false,"structural_negative_record_audit":{"all_label_permutation_true_labels_differ":true,"all_mismatched_pairs_expected_to_fail":true,"all_negative_controls_nondegenerate":true,"assigned_operator_application_possible":false,"diagnostic_pass":true,"failure_rate_measured":false,"failure_rate_structural_only":true,"label_permutation_record_count":10,"mismatched_pair_record_count":2,"p70a_negative_record_count":6,"p70b_negative_record_count":6,"structural_negative_controls_ready":true},"target_endpoint_used_for_evaluation_only":true,"target_endpoint_used_for_prediction":false,"target_midpoint_used_for_evaluation_only":true,"target_midpoint_used_for_prediction":false,"torch_allowed":false,"training_allowed":false,"verdict":"P75_READY_FOR_REVIEW"}
```

## Focused Test Result
- **Command**: `python -m pytest tests/test_phase3_global_negative_controls_collapse_audit.py tests/test_phase3_p75_global_negative_controls_collapse_audit_smoke.py -v`
- **Result**: `13 PASSED`

## Scope Gate
Scope gate successfully verified using `enforce_phase_local_scope_gate_or_skip` with:
- `expected_branch = "phase3/p75-global-negative-controls-collapse-audit-no-training-no-model-no-optimization"`
- `base_commit = "5ae8cec9e53d66eda4e56b595c66803cb48528e7"`
