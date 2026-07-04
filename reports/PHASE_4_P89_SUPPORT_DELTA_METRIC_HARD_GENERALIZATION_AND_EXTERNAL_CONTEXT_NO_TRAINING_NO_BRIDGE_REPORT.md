# PHASE 4 P89 Support-Delta Metric Hard Generalization Report

## 1. Problem Framing
P89 evaluates whether the non-learned support-delta metric baseline generalized across harder splits, relation families, support construction variants, and externalized context constraints. The objective is to stress-test the metric baseline and define the contract for external context.

## 2. P88 Inheritance
P88 source contracts are fully validated:
- `p88_nonlearned_metric_signal_preserved = True`
- `p88_real_shuffled_control_preserved = True`
- `p88_no_training_preserved = True`
- `p88_bridge_not_ready_preserved = True`
- `p88_recommended_p89_preserved = True`

## 3. No-Training/No-Torch Boundary
P89 operates strictly under the no-training boundary:
- `model_training_performed = False`
- `torch_training_performed = False`
- `new_model_implemented = False`
- `optimizer_created = False`
- `checkpoint_written = False`
- `torch` is not imported anywhere in the core module.

## 4. Metric Reconstruction
We rebuilt all selector dataset records directly from P70/P81/P86 materializer logic. We recursively sanitize all metric inputs against target label/endpoint leakage.

## 5. P88 Reproduction
- **`same_split_support_strict`**: **66.67%** (0.6667) effective accuracy on test split, matching P88 results exactly.

## 6. Hard Generalization Policies
We stress-tested the baseline on 4 hard generalization policies:
- **`train_to_heldout_base_state`**: Train prototypes on train split; evaluate on validation split queries using train supports.
  - Effective Accuracy: **70.00%** (0.7000)
  - Coverage: **100.00%**
- **`train_to_heldout_magnitude`**: Train prototypes on train split; evaluate on test split queries using train supports.
  - Effective Accuracy: **66.67%** (0.6667)
  - Coverage: **100.00%**
- **`leave_relation_family_out`**: Holds out `translate_x` and `change_frequency` from the prototype bank.
  - Effective Accuracy: **55.56%**
  - Coverage: **77.78%**
  - Accuracy on Supported: **71.43%**
  - Unsupported classes return `unsupported = True` instead of silently defaulting.
- **`leave_domain_out`**: Train on P70A, evaluate P70B (and vice-versa).
  - Effective Accuracy: **0.00%**
  - Coverage: **0.00%** (disjoint features and classes, reported inapplicable honestly).

## 7. Negative Controls and Policy Alignment
Negative controls are computed separately for each generalization policy without fallback constants:

### 1. Controls for `train_to_heldout_base_state` (Accuracy: 70.00%):
- **`query_source_only_negative_control`**: **40.00%** (margin: **30.00%** >= 15%)
- **`shuffled_support_delta_negative_control`**: **40.00%** (margin: **30.00%** >= 10%)
- **`zero_delta_negative_control`**: **10.00%** (margin: **60.00%** >= 10%)

### 2. Controls for `train_to_heldout_magnitude` (Accuracy: 66.67%):
- **`query_source_only_negative_control`**: **44.44%** (margin: **22.23%** >= 15%)
- **`shuffled_support_delta_negative_control`**: **38.89%** (margin: **27.78%** >= 10%)
- **`zero_delta_negative_control`**: **11.11%** (margin: **55.56%** >= 10%)

### Hard Policy Control Aligned Flags:
- `hard_policy_controls_aligned = True`
- `hard_policy_fallback_controls_used = False`

### Best Generalization Policies:
- **Best Raw Hard Policy**: `"train_to_heldout_base_state"` (effective accuracy: **70.00%**)
- **Best Evidence-Passing Hard Policy**: `"train_to_heldout_base_state"` (since it passes all control thresholds and has the highest accuracy among passing policies).

## 8. Domain Results (on `train_to_heldout_magnitude` policy)
- **P70A (Vector World)**: **75.00%** (6/8 correct)
- **P70B (Time Series Parameter World)**: **60.00%** (6/10 correct)

## 9. Relation-Level Results (on `train_to_heldout_magnitude` policy)
- `translate_y`: **100.00%** (2/2 correct)
- `scale_s`: **100.00%** (2/2 correct)
- `nonlinear_x_from_y`: **100.00%** (2/2 correct)
- `change_frequency`: **100.00%** (2/2 correct)
- `scale_amplitude`: **100.00%** (2/2 correct)
- `scale_volatility_envelope`: **100.00%** (2/2 correct)
- `translate_x`: **0.00%** (0/2 correct)
- `shift_phase`: **0.00%** (0/2 correct)
- `shift_trend`: **0.00%** (0/2 correct)

## 10. `nonlinear_x_from_y` Stress Result
Stress test details:
- `nonlinear_x_from_y` total: **2**
- `nonlinear_x_from_y` correct: **2**
- `nonlinear_x_from_y` predicted as `translate_x`: **0**

Under the prototype metric, `nonlinear_x_from_y` is successfully classified and not confused with `translate_x`.

## 11. External Context Contract
We defined the external context readiness contract:
- `external_context_contract_defined = True`
- `requires_real_external_support_demonstrations = True`
- `requires_query_observation_source = True`
- `requires_non_label_selected_support = True`
- `synthetic_label_selected_support_still_present = True`
- `valid_for_final_semantic_geometry_evidence = False`
- `valid_for_bridge_input_contract = False`

The support episodes remain label-selected synthetically. Moving to true generalization requires real observed A-to-B support pairs.

## 12. Evidence Verdict
- `hard_generalization_supported = True`
*Reasoning*: Best strict reproduction is **66.67%** (>= 60%), best hard policy (`train_to_heldout_base_state`) is **70.00%** (>= 50%), and it beats all corresponding controls (qo by **30.00%** >= 15%, shuffled by **30.00%** >= 10%, zero delta by **60.00%** >= 10%).

## 13. Bridge Boundary
Schrödinger Bridges remain strictly blocked:
- `bridge_ready = False`
- `bridge_implementation_allowed = False`
- `learned_selector_evidence_present = False`
- `learned_metric_evidence_present = False`

## 14. Limitations
- P89 does not train a model.
- P89 does not prove learned metric evidence.
- P89 does not prove semantic geometry.
- P89 does not validate any bridge.
- P89 still uses synthetic label-selected support construction.

## 15. Final Verdict
`P89_READY_FOR_REVIEW`

## Output Flags Verified:
- `shuffled_support_delta_control_implemented = True`
- `shuffled_control_target_labels_preserved = True`
- `shuffled_control_support_deltas_permuted = True`
- `target_labels_shifted = False`
- `true_labels_preserved = True`
- `hard_policy_controls_aligned = True`
- `hard_policy_fallback_controls_used = False`

## Smoke Output Block
```json
{"phase":"P89","phase_group":"PHASE_4","phase_name":"Support-Delta Metric Hard Generalization and External Context","contract_version":"phase4_p89_support_delta_metric_hard_generalization_v1","source_nonlearned_metric_phase":"P88","source_few_shot_selector_phase":"P87","source_materializer_phase":"P86","source_externalized_context_phase":"P85","source_enrichment_phase":"P84","source_identifiability_phase":"P83","source_selector_pilot_phase":"P82","source_dataset_phase":"P81","verdict":"P89_READY_FOR_REVIEW","model_training_performed":false,"torch_training_performed":false,"new_model_implemented":false,"optimizer_created":false,"checkpoint_written":false,"source_contracts_validated":true,"p88_reproduction_result":{"effective_accuracy":0.6666666666666666,"sample_count":18,"mode":"delta_only","distance_family":"l2"},"hard_generalization_results":{"train_to_heldout_base_state":{"effective_accuracy":0.7,"coverage":1.0,"accuracy_on_supported":0.7},"train_to_heldout_magnitude":{"effective_accuracy":0.6666666666666666,"coverage":1.0,"accuracy_on_supported":0.6666666666666666},"leave_relation_family_out":{"effective_accuracy":0.5555555555555556,"coverage":0.7777777777777778,"accuracy_on_supported":0.7142857142857143,"sample_count":18,"supported_sample_count":14,"unsupported_sample_count":4,"held_out_classes":["change_frequency","translate_x"],"predictions":[{"episode_id":"episode_same_split_support_strict_p81_record_p70a_46","domain":"p70a_vector_world","split":"test","true_label":"translate_x","predicted_label":"nonlinear_x_from_y","unsupported":true,"is_correct":false,"distance_family":"l2","mode":"delta_only"},{"episode_id":"episode_same_split_support_strict_p81_record_p70a_47","domain":"p70a_vector_world","split":"test","true_label":"translate_x","predicted_label":"nonlinear_x_from_y","unsupported":true,"is_correct":false,"distance_family":"l2","mode":"delta_only"},{"episode_id":"episode_same_split_support_strict_p81_record_p70a_48","domain":"p70a_vector_world","split":"test","true_label":"translate_y","predicted_label":"translate_y","unsupported":false,"is_correct":true,"distance_family":"l2","mode":"delta_only"},{"episode_id":"episode_same_split_support_strict_p81_record_p70a_49","domain":"p70a_vector_world","split":"test","true_label":"translate_y","predicted_label":"translate_y","unsupported":false,"is_correct":true,"distance_family":"l2","mode":"delta_only"},{"episode_id":"episode_same_split_support_strict_p81_record_p70a_50","domain":"p70a_vector_world","split":"test","true_label":"scale_s","predicted_label":"scale_s","unsupported":false,"is_correct":true,"distance_family":"l2","mode":"delta_only"},{"episode_id":"episode_same_split_support_strict_p81_record_p70a_51","domain":"p70a_vector_world","split":"test","true_label":"scale_s","predicted_label":"scale_s","unsupported":false,"is_correct":true,"distance_family":"l2","mode":"delta_only"},{"episode_id":"episode_same_split_support_strict_p81_record_p70a_52","domain":"p70a_vector_world","split":"test","true_label":"nonlinear_x_from_y","predicted_label":"nonlinear_x_from_y","unsupported":false,"is_correct":true,"distance_family":"l2","mode":"delta_only"},{"episode_id":"episode_same_split_support_strict_p81_record_p70a_53","domain":"p70a_vector_world","split":"test","true_label":"nonlinear_x_from_y","predicted_label":"nonlinear_x_from_y","unsupported":false,"is_correct":true,"distance_family":"l2","mode":"delta_only"},{"episode_id":"episode_same_split_support_strict_p81_record_p70b_50","domain":"p70b_time_series_parameter_world","split":"test","true_label":"change_frequency","predicted_label":"shift_phase","unsupported":true,"is_correct":false,"distance_family":"l2","mode":"delta_only"},{"episode_id":"episode_same_split_support_strict_p81_record_p70b_51","domain":"p70b_time_series_parameter_world","split":"test","true_label":"change_frequency","predicted_label":"shift_phase","unsupported":true,"is_correct":false,"distance_family":"l2","mode":"delta_only"},{"episode_id":"episode_same_split_support_strict_p81_record_p70b_52","domain":"p70b_time_series_parameter_world","split":"test","true_label":"scale_amplitude","predicted_label":"scale_amplitude","unsupported":false,"is_correct":true,"distance_family":"l2","mode":"delta_only"},{"episode_id":"episode_same_split_support_strict_p81_record_p70b_53","domain":"p70b_time_series_parameter_world","split":"test","true_label":"scale_amplitude","predicted_label":"scale_amplitude","unsupported":false,"is_correct":true,"distance_family":"l2","mode":"delta_only"},{"episode_id":"episode_same_split_support_strict_p81_record_p70b_54","domain":"p70b_time_series_parameter_world","split":"test","true_label":"shift_phase","predicted_label":"scale_amplitude","unsupported":false,"is_correct":false,"distance_family":"l2","mode":"delta_only"},{"episode_id":"episode_same_split_support_strict_p81_record_p70b_55","domain":"p70b_time_series_parameter_world","split":"test","true_label":"shift_phase","predicted_label":"scale_amplitude","unsupported":false,"is_correct":false,"distance_family":"l2","mode":"delta_only"},{"episode_id":"episode_same_split_support_strict_p81_record_p70b_56","domain":"p70b_time_series_parameter_world","split":"test","true_label":"scale_volatility_envelope","predicted_label":"shift_trend","unsupported":false,"is_correct":false,"distance_family":"l2","mode":"delta_only"},{"episode_id":"episode_same_split_support_strict_p81_record_p70b_57","domain":"p70b_time_series_parameter_world","split":"test","true_label":"scale_volatility_envelope","predicted_label":"shift_trend","unsupported":false,"is_correct":false,"distance_family":"l2","mode":"delta_only"},{"episode_id":"episode_same_split_support_strict_p81_record_p70b_58","domain":"p70b_time_series_parameter_world","split":"test","true_label":"shift_trend","predicted_label":"shift_trend","unsupported":false,"is_correct":true,"distance_family":"l2","mode":"delta_only"},{"episode_id":"episode_same_split_support_strict_p81_record_p70b_59","domain":"p70b_time_series_parameter_world","split":"test","true_label":"shift_trend","predicted_label":"shift_trend","unsupported":false,"is_correct":true,"distance_family":"l2","mode":"delta_only"}]},"leave_domain_out":{"train_p70a_test_p70b":{"effective_accuracy":0.0,"coverage":0.0,"accuracy_on_supported":0.0},"train_p70b_test_p70a":{"effective_accuracy":0.0,"coverage":0.0,"accuracy_on_supported":0.0},"note":"Domain out results yield 0 coverage as relation classes and features are completely disjoint."}},"negative_control_results":{"query_source_only_negative_control":0.4,"shuffled_support_delta_negative_control":0.4,"zero_delta_negative_control":0.1,"shuffled_support_delta_control_implemented":true,"shuffled_control_target_labels_preserved":true,"shuffled_control_support_deltas_permuted":true,"target_labels_shifted":false,"true_labels_preserved":true},"domain_results":{"p70a_vector_world":{"sample_count":8,"effective_accuracy":0.75},"p70b_time_series_parameter_world":{"sample_count":10,"effective_accuracy":0.6}},"relation_results":{"translate_x":{"accuracy":0.0,"correct_count":0,"total_count":2},"translate_y":{"accuracy":1.0,"correct_count":2,"total_count":2},"scale_s":{"accuracy":1.0,"correct_count":2,"total_count":2},"nonlinear_x_from_y":{"accuracy":1.0,"correct_count":2,"total_count":2},"change_frequency":{"accuracy":1.0,"correct_count":2,"total_count":2},"scale_amplitude":{"accuracy":1.0,"correct_count":2,"total_count":2},"shift_phase":{"accuracy":0.0,"correct_count":0,"total_count":2},"scale_volatility_envelope":{"accuracy":1.0,"correct_count":2,"total_count":2},"shift_trend":{"accuracy":0.0,"correct_count":0,"total_count":2}},"nonlinear_x_from_y_stress_result":{"nonlinear_x_from_y_total":2,"nonlinear_x_from_y_correct":2,"nonlinear_x_from_y_predicted_as_translate_x":0,"note":"Evaluated on train_to_heldout_magnitude policy."},"metric_input_leakage_audit":{"metric_input_count":114,"support_record_id_used_for_metric_input":false,"materialization_metadata_used_for_metric_input":false,"evaluated_target_endpoint_used_for_metric_input":false,"evaluated_target_delta_used_for_metric_input":false,"exact_relation_label_used_for_metric_input":false,"exact_operator_id_used_for_metric_input":false,"audit_metadata_used_for_metric_input":false,"relation_specific_hint_used_for_metric_input":false,"diagnostic_pass":true},"support_policy_flow_audits":{"same_split_support_strict":{"support_policy":"same_split_support_strict","episode_count":114,"available_episode_count":114,"unavailable_episode_count":0,"unsupported_label_count":0,"query_split_to_support_split_counts":{"train":{"train":152,"validation":0,"test":0},"validation":{"train":0,"validation":20,"test":0},"test":{"train":0,"validation":0,"test":18}},"off_diagonal_support_count":0,"same_split_diagonal_only":true,"diagnostic_pass":true},"train_to_heldout_base_state":{"support_policy":"train_to_heldout_base_state","episode_count":114,"available_episode_count":114,"unavailable_episode_count":0,"unsupported_label_count":0,"query_split_to_support_split_counts":{"train":{"train":152,"validation":0,"test":0},"validation":{"train":40,"validation":0,"test":0},"test":{"train":36,"validation":0,"test":0}},"off_diagonal_support_count":76,"same_split_diagonal_only":false,"diagnostic_pass":true},"train_to_heldout_magnitude":{"support_policy":"train_to_heldout_magnitude","episode_count":114,"available_episode_count":114,"unavailable_episode_count":0,"unsupported_label_count":0,"query_split_to_support_split_counts":{"train":{"train":152,"validation":0,"test":0},"validation":{"train":40,"validation":0,"test":0},"test":{"train":36,"validation":0,"test":0}},"off_diagonal_support_count":76,"same_split_diagonal_only":false,"diagnostic_pass":true},"leave_relation_family_out":{"support_policy":"leave_relation_family_out","episode_count":114,"available_episode_count":114,"unavailable_episode_count":0,"unsupported_label_count":0,"query_split_to_support_split_counts":{"train":{"train":152,"validation":0,"test":0},"validation":{"train":0,"validation":20,"test":0},"test":{"train":0,"validation":0,"test":18}},"off_diagonal_support_count":0,"same_split_diagonal_only":true,"diagnostic_pass":true},"leave_domain_out":{"support_policy":"leave_domain_out","episode_count":114,"available_episode_count":114,"unavailable_episode_count":0,"unsupported_label_count":0,"query_split_to_support_split_counts":{"train":{"train":152,"validation":0,"test":0},"validation":{"train":0,"validation":20,"test":0},"test":{"train":0,"validation":0,"test":18}},"off_diagonal_support_count":0,"same_split_diagonal_only":true,"diagnostic_pass":true}},"external_context_contract":{"external_context_contract_defined":true,"requires_real_external_support_demonstrations":true,"requires_query_observation_source":true,"requires_non_label_selected_support":true,"synthetic_label_selected_support_still_present":true,"valid_for_final_semantic_geometry_evidence":false,"valid_for_bridge_input_contract":false,"recommended_external_context_source":["human_defined_relation_demonstrations","procedurally_generated_query_tasks_without_label_visible_to_selector","real_dataset_with observed A_to_B support pairs"]},"hard_policy_control_results":{"train_to_heldout_base_state":{"query_source_only_negative_control":0.4,"shuffled_support_delta_negative_control":0.4,"zero_delta_negative_control":0.1},"train_to_heldout_magnitude":{"query_source_only_negative_control":0.4444444444444444,"shuffled_support_delta_negative_control":0.3888888888888889,"zero_delta_negative_control":0.1111111111111111}},"best_raw_hard_policy":"train_to_heldout_base_state","best_evidence_passing_hard_policy":"train_to_heldout_base_state","hard_policy_controls_aligned":true,"hard_policy_fallback_controls_used":false,"nonlearned_metric_signal_present":true,"hard_generalization_supported":true,"learned_selector_evidence_present":false,"learned_metric_evidence_present":false,"semantic_metric_ready":false,"bridge_implementation_allowed":false,"bridge_ready":false,"generation_claims_allowed":false,"semantic_geometry_claims_allowed":false,"recommended_next_phase":"P90_external_support_context_builder_no_training_no_bridge","bridge_boundary_after_hard_generalization":{"bridge_ready":false,"bridge_implementation_allowed":false,"nonlearned_metric_signal_present":true,"hard_generalization_supported":true,"learned_selector_evidence_present":false,"learned_metric_evidence_present":false,"semantic_metric_ready":false,"generation_claims_allowed":false,"semantic_geometry_claims_allowed":false,"blocking_reasons":["learned_metric_evidence_not_present","semantic_metric_not_ready","external_context_not_real_or_non_label_selected","bridge_input_contract_not_defined","bridge_validation_not_run"],"diagnostic_pass":true},"json_safe":true,"diagnostic_only":true}
```
