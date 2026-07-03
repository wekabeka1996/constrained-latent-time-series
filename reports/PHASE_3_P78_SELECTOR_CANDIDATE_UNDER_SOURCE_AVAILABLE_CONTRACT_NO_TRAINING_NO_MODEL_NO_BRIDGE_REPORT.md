# PHASE 3 P78 Selector Candidate Under Source-Available Contract Report

## 1. Problem Framing
P78 evaluates a deterministic rule-based selector candidate under the P77 source-available query contract. We test selector candidates and ablations to examine whether high selection accuracy is achievable without target coordinate leakage and to analyze the role that query family/transformation hints play in low-complexity toy worlds.

## 2. Boundaries and Scope
We strictly respect all Phase 3 boundaries:
- No training is performed.
- No model architecture or neural networks are created.
- No neural relation encoder or neural relation selector is implemented.
- No learned selector is implemented.
- No optimization steps or gradients are computed.
- No PyTorch, NumPy, Pandas, or Scikit-Learn is imported.
- No stochastic random processes are used.
- No Schrödinger Bridge, Brownian motions, SDEs, or score models are built.
- No learned metric is implemented.
- No semantic geometry, meaning, learned operator identity, or learned selector claims are made.
- No model language is used.

## 3. Why P78 after P77
P77 defined a leakage-safe query contract that isolates target endpoints. P78 tests concrete selector candidates and ablations under this contract to understand what information is required to select relations.

## 4. Selector Modes & Target Isolation
We evaluate six modes of selection:
- **null_majority_baseline**: Uses fixed defaults (`translate_x` / `change_frequency`).
- **full_contract_rule_selector**: Uses all query contract hints.
- **no_family_hint_rule_selector**: Ignores `relation_family_hint`.
- **no_family_or_transformation_hint_rule_selector**: Ignores family and transformation class hints.
- **source_only_summary_selector**: Ignores all query hints; uses only source parameter/state summaries.
- **intensity_only_selector**: Uses only the intensity hint.

All selector functions receive stripped records and are completely isolated from target coordinates, deltas, exact relation labels, and audit metadata during prediction.

## 5. Audit Metadata Boundary
The true relation type is stored under `_audit_metadata.true_relation_type`. This field is strictly excluded from predictor inputs. It is accessed only during posthoc evaluation to compute accuracy.

## 6. Family-Hint Pass-Through Risk & Uniqueness
Because family hints are unique in toy worlds, they act like exact relation labels. Full-contract accuracies reach 100% because family/transformation hints map directly to single relation types. We report this pass-through risk honestly:
- `family_hint_pass_through_risk_present = True`
- `selector_success_interpretable_as_learned_evidence = False`

The selector's performance is explained by the rule-based metadata hints, not by any learned representation or geometric meaning.

## 7. Ablation Results
### P70A Vector World Accuracy
- `null_majority_baseline`: 22.2%
- `full_contract_rule_selector`: 100.0%
- `no_family_hint_rule_selector`: 100.0% (transformation + axis hints are sufficient to select unique vector relations)
- `no_family_or_transformation_hint_rule_selector`: 88.9% (axis hint alone is unique except for translate_x vs reflect_x axis overlap)
- `source_only_summary_selector`: 22.2%
- `intensity_only_selector`: 22.2%

### P70B Parameter World Accuracy
- `null_majority_baseline`: 20.0%
- `full_contract_rule_selector`: 100.0%
- `no_family_hint_rule_selector`: 100.0%
- `no_family_or_transformation_hint_rule_selector`: 100.0% (parameter group hints are uniquely mapped in this toy world)
- `source_only_summary_selector`: 20.0%
- `intensity_only_selector`: 20.0%

## 8. Selector Evidence Conclusion
- `learned_selector_evidence_present = False`
- `selector_success_interpretable_as_learned_evidence = False`

## 9. Bridge Boundary
Bridge readiness and semantic metrics remain blocked because no learned selector evidence exists:
- `bridge_ready = False`
- `semantic_metric_ready = False`
- **Blocking Reasons**:
  1. `selector_candidate_is_rule_based_not_learned`
  2. `family_hint_pass_through_risk_present`
  3. `semantic_metric_not_ready`
  4. `bridge_not_ready_without_learned_selector_evidence`

## 10. P77 Notes Preserved
- `p77_audit_metadata_boundary_preserved = True`
- `p77_family_uniqueness_risk_preserved = True`
- `p77_bridge_not_ready_preserved = True`

## 11. Results
Final Verdict: `P78_READY_FOR_REVIEW`

## Smoke Output Block
```json
{"phase":"P78","phase_group":"PHASE_3","phase_name":"Selector Candidate Under Source-Available Contract","contract_version":"phase3_p78_selector_candidate_under_source_available_contract_v1","source_baseline_phase":"P69","source_vector_testbed_phase":"P70A","source_time_series_testbed_phase":"P70B","source_contrastive_phase":"P71","source_operator_bank_phase":"P72","source_transfer_audit_phase":"P73","source_composition_audit_phase":"P74","source_negative_control_phase":"P75","source_prebridge_leakage_phase":"P76","source_query_contract_phase":"P77","verdict":"P78_READY_FOR_REVIEW","training_allowed":false,"model_implementation_allowed":false,"neural_encoder_implementation_allowed":false,"neural_operator_selector_allowed":false,"optimization_allowed":false,"torch_allowed":false,"numpy_allowed":false,"stochastic_random_allowed":false,"bridge_implementation_allowed":false,"learned_metric_allowed":false,"rule_based_selector_candidate_allowed":true,"predictive_selector_implementation_allowed":false,"learned_selector_evidence_present":false,"predictive_selector_claims_allowed":false,"target_endpoint_used_for_selector":false,"target_delta_used_for_selector":false,"exact_relation_label_used_for_selector":false,"exact_operator_id_used_for_selector":false,"audit_metadata_used_for_selector":false,"audit_label_used_for_evaluation_only":true,"valid_for_future_selector_experiment":true,"rule_based_selector_candidate_evaluated":true,"semantic_metric_ready":false,"bridge_ready":false,"primary_empirical_target":"selector_candidate_under_source_available_contract_diagnostics","selector_modes":["null_majority_baseline","full_contract_rule_selector","no_family_hint_rule_selector","no_family_or_transformation_hint_rule_selector","source_only_summary_selector","intensity_only_selector"],"audit_domains":["p70a_vector_world","p70b_time_series_parameter_world"],"forbidden_claims":["semantic_geometry_is_proven","meaning_is_learned","operator_identity_is_proven","relation_encoder_is_trained","relation_encoder_is_validated","learned_operator_selection_is_proven","learned_selector_is_validated","learned_metric_is_proven","predictive_selector_is_validated","bridge_method_is_validated","selector_candidate_proves_semantics","selector_candidate_proves_generation"],"allowed_claims":["p78_evaluates_rule_based_selector_candidate","p78_runs_selector_ablation_diagnostics","p78_preserves_p77_audit_metadata_boundary","p78_reports_family_hint_pass_through_risk","p78_does_not_train_models","p78_does_not_establish_learned_selector_evidence","p78_preserves_bridge_not_ready_boundary"],"source_contracts_validated":true,"p77_audit_metadata_boundary_preserved":true,"p77_family_uniqueness_risk_preserved":true,"p77_bridge_not_ready_preserved":true,"selector_candidate_audit":{"rule_based_selector_candidate_evaluated":true,"p70a_selector_results_by_mode":{"null_majority_baseline":{"mode":"null_majority_baseline","record_count":54,"accuracy":0.2222222222222222,"correct_count":12,"incorrect_count":42,"prediction_distribution":{"translate_x":54},"target_label_distribution":{"translate_x":12,"translate_y":12,"scale_s":12,"reflect_x":6,"nonlinear_x_from_y":12},"audit_metadata_used_for_selector":false,"audit_label_used_for_evaluation_only":true,"diagnostic_pass":true},"full_contract_rule_selector":{"mode":"full_contract_rule_selector","record_count":54,"accuracy":1.0,"correct_count":54,"incorrect_count":0,"prediction_distribution":{"translate_x":12,"translate_y":12,"scale_s":12,"reflect_x":6,"nonlinear_x_from_y":12},"target_label_distribution":{"translate_x":12,"translate_y":12,"scale_s":12,"reflect_x":6,"nonlinear_x_from_y":12},"audit_metadata_used_for_selector":false,"audit_label_used_for_evaluation_only":true,"diagnostic_pass":true},"no_family_hint_rule_selector":{"mode":"no_family_hint_rule_selector","record_count":54,"accuracy":1.0,"correct_count":54,"incorrect_count":0,"prediction_distribution":{"translate_x":12,"translate_y":12,"scale_s":12,"reflect_x":6,"nonlinear_x_from_y":12},"target_label_distribution":{"translate_x":12,"translate_y":12,"scale_s":12,"reflect_x":6,"nonlinear_x_from_y":12},"audit_metadata_used_for_selector":false,"audit_label_used_for_evaluation_only":true,"diagnostic_pass":true},"no_family_or_transformation_hint_rule_selector":{"mode":"no_family_or_transformation_hint_rule_selector","record_count":54,"accuracy":0.8888888888888888,"correct_count":48,"incorrect_count":6,"prediction_distribution":{"translate_x":18,"translate_y":12,"scale_s":12,"nonlinear_x_from_y":12},"target_label_distribution":{"translate_x":12,"translate_y":12,"scale_s":12,"reflect_x":6,"nonlinear_x_from_y":12},"audit_metadata_used_for_selector":false,"audit_label_used_for_evaluation_only":true,"diagnostic_pass":true},"source_only_summary_selector":{"mode":"source_only_summary_selector","record_count":54,"accuracy":0.2222222222222222,"correct_count":12,"incorrect_count":42,"prediction_distribution":{"translate_x":54},"target_label_distribution":{"translate_x":12,"translate_y":12,"scale_s":12,"reflect_x":6,"nonlinear_x_from_y":12},"audit_metadata_used_for_selector":false,"audit_label_used_for_evaluation_only":true,"diagnostic_pass":true},"intensity_only_selector":{"mode":"intensity_only_selector","record_count":54,"accuracy":0.2222222222222222,"correct_count":12,"incorrect_count":42,"prediction_distribution":{"translate_x":54},"target_label_distribution":{"translate_x":12,"translate_y":12,"scale_s":12,"reflect_x":6,"nonlinear_x_from_y":12},"audit_metadata_used_for_selector":false,"audit_label_used_for_evaluation_only":true,"diagnostic_pass":true}},"p70b_selector_results_by_mode":{"null_majority_baseline":{"mode":"null_majority_baseline","record_count":60,"accuracy":0.2,"correct_count":12,"incorrect_count":48,"prediction_distribution":{"change_frequency":60},"target_label_distribution":{"change_frequency":12,"scale_amplitude":12,"shift_phase":12,"scale_volatility_envelope":12,"shift_trend":12},"audit_metadata_used_for_selector":false,"audit_label_used_for_evaluation_only":true,"diagnostic_pass":true},"full_contract_rule_selector":{"mode":"full_contract_rule_selector","record_count":60,"accuracy":1.0,"correct_count":60,"incorrect_count":0,"prediction_distribution":{"change_frequency":12,"scale_amplitude":12,"shift_phase":12,"scale_volatility_envelope":12,"shift_trend":12},"target_label_distribution":{"change_frequency":12,"scale_amplitude":12,"shift_phase":12,"scale_volatility_envelope":12,"shift_trend":12},"audit_metadata_used_for_selector":false,"audit_label_used_for_evaluation_only":true,"diagnostic_pass":true},"no_family_hint_rule_selector":{"mode":"no_family_hint_rule_selector","record_count":60,"accuracy":1.0,"correct_count":60,"incorrect_count":0,"prediction_distribution":{"change_frequency":12,"scale_amplitude":12,"shift_phase":12,"scale_volatility_envelope":12,"shift_trend":12},"target_label_distribution":{"change_frequency":12,"scale_amplitude":12,"shift_phase":12,"scale_volatility_envelope":12,"shift_trend":12},"audit_metadata_used_for_selector":false,"audit_label_used_for_evaluation_only":true,"diagnostic_pass":true},"no_family_or_transformation_hint_rule_selector":{"mode":"no_family_or_transformation_hint_rule_selector","record_count":60,"accuracy":1.0,"correct_count":60,"incorrect_count":0,"prediction_distribution":{"change_frequency":12,"scale_amplitude":12,"shift_phase":12,"scale_volatility_envelope":12,"shift_trend":12},"target_label_distribution":{"change_frequency":12,"scale_amplitude":12,"shift_phase":12,"scale_volatility_envelope":12,"shift_trend":12},"audit_metadata_used_for_selector":false,"audit_label_used_for_evaluation_only":true,"diagnostic_pass":true},"source_only_summary_selector":{"mode":"source_only_summary_selector","record_count":60,"accuracy":0.2,"correct_count":12,"incorrect_count":48,"prediction_distribution":{"change_frequency":60},"target_label_distribution":{"change_frequency":12,"scale_amplitude":12,"shift_phase":12,"scale_volatility_envelope":12,"shift_trend":12},"audit_metadata_used_for_selector":false,"audit_label_used_for_evaluation_only":true,"diagnostic_pass":true},"intensity_only_selector":{"mode":"intensity_only_selector","record_count":60,"accuracy":0.2,"correct_count":12,"incorrect_count":48,"prediction_distribution":{"change_frequency":60},"target_label_distribution":{"change_frequency":12,"scale_amplitude":12,"shift_phase":12,"scale_volatility_envelope":12,"shift_trend":12},"audit_metadata_used_for_selector":false,"audit_label_used_for_evaluation_only":true,"diagnostic_pass":true}},"p70a_family_hint_pass_through_risk_audit":{"family_uniqueness_risk_present":true,"full_contract_accuracy":1.0,"no_family_hint_accuracy":1.0,"no_family_or_transformation_hint_accuracy":0.8888888888888888,"accuracy_drop_without_family_hint":0.0,"accuracy_drop_without_family_or_transformation_hint":0.11111111111111116,"family_hint_pass_through_risk_present":true,"selector_success_interpretable_as_learned_evidence":false,"diagnostic_pass":true},"p70b_family_hint_pass_through_risk_audit":{"family_uniqueness_risk_present":true,"full_contract_accuracy":1.0,"no_family_hint_accuracy":1.0,"no_family_or_transformation_hint_accuracy":1.0,"accuracy_drop_without_family_hint":0.0,"accuracy_drop_without_family_or_transformation_hint":0.0,"family_hint_pass_through_risk_present":true,"selector_success_interpretable_as_learned_evidence":false,"diagnostic_pass":true},"selector_contract_record_audit":{"p70a_record_count":54,"p70b_record_count":60,"total_record_count":114,"audit_metadata_available_for_evaluation_count":114,"stripped_records_free_of_audit_metadata":true,"stripped_records_free_of_exact_relation_label":true,"stripped_records_free_of_exact_operator_id":true,"stripped_records_free_of_target_endpoint":true,"stripped_records_free_of_target_delta":true,"all_records_available_before_target_endpoint":true,"diagnostic_pass":true},"best_p70a_mode":"full_contract_rule_selector","best_p70a_accuracy":1.0,"best_p70b_mode":"full_contract_rule_selector","best_p70b_accuracy":1.0,"family_hint_pass_through_risk_present":true,"learned_selector_evidence_present":false,"predictive_selector_implementation_present":false,"semantic_metric_ready":false,"bridge_ready":false,"diagnostic_pass":true},"bridge_boundary_after_selector_candidate_audit":{"bridge_ready":false,"semantic_metric_ready":false,"rule_based_selector_candidate_evaluated":true,"learned_selector_evidence_present":false,"family_hint_pass_through_risk_present":true,"blocking_reasons":["selector_candidate_is_rule_based_not_learned","family_hint_pass_through_risk_present","semantic_metric_not_ready","bridge_not_ready_without_learned_selector_evidence"],"next_required_phase":"learned_selector_or_metric_candidate_under_ablation_controls","diagnostic_pass":true},"sanity_summary":{"source_contracts_validated":true,"p77_audit_metadata_boundary_preserved":true,"p77_family_uniqueness_risk_preserved":true,"p77_bridge_not_ready_preserved":true,"rule_based_selector_candidate_evaluated":true,"selector_contract_records_valid":true,"target_endpoint_used_for_selector":false,"target_delta_used_for_selector":false,"exact_relation_label_used_for_selector":false,"exact_operator_id_used_for_selector":false,"audit_metadata_used_for_selector":false,"audit_label_used_for_evaluation_only":true,"family_hint_pass_through_risk_present":true,"ablation_modes_evaluated":true,"full_contract_accuracy_reported":true,"no_family_accuracy_reported":true,"source_only_accuracy_reported":true,"null_baseline_accuracy_reported":true,"learned_selector_evidence_present":false,"predictive_selector_claims_made":false,"semantic_metric_ready":false,"bridge_ready":false,"bridge_not_ready_preserved":true,"training_or_model_added":false,"json_safe":true},"json_safe":true,"diagnostic_only":true}
```

## Focused Test Result
- **Command**: `python -m pytest tests/test_phase3_selector_candidate_under_contract.py tests/test_phase3_p78_selector_candidate_under_contract_smoke.py -v`
- **Result**: `16 PASSED`

## Scope Gate
Scope gate successfully verified using Git diff:
- `expected_branch = "phase3/p78-selector-candidate-under-source-available-contract-no-training-no-model-no-bridge"`
- `base_commit = "0c9160389463736daa3a90f3583d9d6c8c574944"`

## Limitations
- P78 does not prove learned semantics.
- P78 does not prove learned operator identity, selection, or composition.
- P78 does not prove semantic geometry.
- P78 does not validate any bridge method.
- P78 evaluates deterministic selector candidates only.
