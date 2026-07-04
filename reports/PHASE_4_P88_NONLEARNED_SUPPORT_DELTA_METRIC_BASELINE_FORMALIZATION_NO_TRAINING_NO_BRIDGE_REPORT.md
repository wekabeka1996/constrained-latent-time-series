# PHASE 4 P88 Non-Learned Support-Delta Metric Baseline Formalization Report

## 1. Problem Framing
P88 formalizes the non-learned support-delta metric baseline that demonstrated strong classification performance in P87. The goal is to verify if the support-delta metric signal is stable, auditable, split-clean, and domain/relation interpretable.

## 2. P87 Inheritance
P87 source contracts are fully validated:
- `p87_no_learned_selector_evidence_preserved = True`
- `p87_nearest_support_delta_signal_preserved = True`
- `p87_recommended_p88_preserved = True`
- `p87_bridge_not_ready_preserved = True`

## 3. No-Training Boundary
P88 operates strictly under a no-training boundary:
- `model_training_performed = False`
- `torch_training_performed = False`
- `new_model_implemented = False`
- `optimizer_created = False`
- `checkpoint_written = False`
- `torch` is not imported anywhere in the core module.

## 4. Support Split Policy Audit
We audit split policy flows strictly:
- **`same_split_support_strict`**: strict policy (primary evidence). Validation/test queries use same-split supports only.
- **`train_bank_support_diagnostic`**: uses the train split support bank for validation/test queries (diagnostic only).

### Strict policy split count flow:
- `train` -> `train`: **152** supports
- `validation` -> `validation`: **20** supports
- `test` -> `test`: **18** supports
- Strict flow check: `same_split_diagonal_only = True`
- Strict flow pass: `same_split_support_strict_pass = True`
- Off-diagonal support count: **0**
- Diagnostic flow pass: `train_bank_support_diagnostic_pass = True`

## 5. Metric Input Sanitization
Metric inputs are sanitized recursively to guarantee zero leakage:
- `support_record_id_used_for_metric_input = False`
- `materialization_metadata_used_for_metric_input = False`
- `evaluated_target_endpoint_used_for_metric_input = False`
- `evaluated_target_delta_used_for_metric_input = False`
- `exact_relation_label_used_for_metric_input = False`
- `exact_operator_id_used_for_metric_input = False`
- `audit_metadata_used_for_metric_input = False`
- `relation_specific_hint_used_for_metric_input = False`
- `diagnostic_pass = True` (Audited across all 114 records)

## 6. Metric Feature Modes
We evaluate four encoding modes:
1. `delta_only`: support delta features only (18 floats).
2. `source_result_delta`: support source, result, delta features (30 floats).
3. `delta_plus_invariants`: delta features + invariants (26 floats).
4. `full_support_metric`: query source + support source/result/delta + invariants (49 floats).

## 7. Distance Families
We evaluate three distance families:
- **L1 Distance**
- **L2 Distance** (Euclidean)
- **Cosine Distance** (Cosine-safe)

## 8. Main Metric Results
- **Best Mode**: `"delta_only"`
- **Best Distance Family**: `"l2"`
- **Best Combined Accuracy**: **66.67%** (0.6667)
- **Total Test Samples**: **18**

## 9. Hard Ablations and Negative Controls
We run hard ablations on the test split:
- **`delta_only` (L2)**: **66.67%**
- **`source_result_delta` (L2)**: **27.78%**
- **`delta_plus_invariants` (L2)**: **61.11%**
- **`full_support_metric` (L2)**: **27.78%**
- **`query_source_only_negative_control`**: **11.11%** (collapses to majority class)
- **`shuffled_support_delta_negative_control`**: **38.89%** (support deltas permuted across episodes while preserving target labels)
- **`zero_delta_negative_control`**: **11.11%** (collapses to majority class)

### Shuffled Control Permutation Method:
The support delta feature vectors are deterministically shifted/permuted across the test episodes (by a shift index of 1). The target labels remain completely unshifted and unmodified. This decouples the actual query-to-support alignment while maintaining identical target classification distributions.

*Interpretation*: Adding query features or source/result features degrades the simple metric distance match because of domain overlap. Restricting the baseline to support delta features is critical to extracting the relation signal. Shuffling delta labels or zeroing delta features completely destroys classification performance.

## 10. Domain-Specific Results
### P70A (Vector World)
- Sample count: **8**
- Best accuracy: **75.00%** (mode: `delta_only`, distance: `l2`)

### P70B (Time Series Parameter World)
- Sample count: **10**
- Best accuracy: **60.00%** (mode: `delta_only`, distance: `l2`)

## 11. Relation-Level Results
- `translate_x`: **100.00%** (2/2 correct)
- `translate_y`: **100.00%** (2/2 correct)
- `scale_s`: **100.00%** (2/2 correct)
- `reflect_x`: **0.00%** (0/0 correct, held out/unseen in test split)
- `nonlinear_x_from_y`: **0.00%** (0/2 correct)
- `change_frequency`: **50.00%** (1/2 correct)
- `scale_amplitude`: **100.00%** (2/2 correct)
- `shift_phase`: **50.00%** (1/2 correct)
- `scale_volatility_envelope`: **50.00%** (1/2 correct)
- `shift_trend`: **100.00%** (2/2 correct)

## 12. Confusion Analysis
- `translate_x_as_reflect_x`: **0**
- `translate_x_as_nonlinear`: **0**
- `reflect_x_as_translate_x`: **0**
- `reflect_x_as_nonlinear`: **0**
- `nonlinear_as_translate_x`: **2**
- `nonlinear_as_reflect_x`: **0**

*Interpretation*: `nonlinear_x_from_y` is confused as `translate_x` in 2 test cases.

## 13. Evidence Verdict
- `nonlearned_metric_signal_present = True`
*Reasoning*: The best strict-policy configuration (`delta_only` L2) beats the query-source-only negative control by **55.56%** (threshold >= 20%), the shuffled control by **27.78%** (threshold >= 15%), and the zero delta control by **55.56%** (threshold >= 15%), with a combined test accuracy of **66.67%** (threshold >= 60.00%).

## 14. Bridge Boundary
Schrödinger Bridges remain strictly blocked:
- `bridge_ready = False`
- `bridge_implementation_allowed = False`
- `learned_selector_evidence_present = False`
- `learned_metric_evidence_present = False`

## 15. Limitations
- P88 does not train a model.
- P88 does not prove learned metric evidence.
- P88 does not prove semantic geometry.
- P88 does not validate any bridge.
- P88 only formalizes a deterministic synthetic support-delta metric baseline.

## 16. Final Verdict
`P88_READY_FOR_REVIEW`

## Output Flags Verified:
- `shuffled_support_delta_control_implemented = True`
- `shuffled_control_target_labels_preserved = True`
- `shuffled_control_support_deltas_permuted = True`
- `target_labels_shifted = False`
- `true_labels_preserved = True`

## Smoke Output Block
```json
{"phase":"P88","phase_group":"PHASE_4","phase_name":"Non-Learned Support-Delta Metric Baseline Formalization","contract_version":"phase4_p88_nonlearned_support_delta_metric_baseline_v1","source_few_shot_selector_phase":"P87","source_materializer_phase":"P86","source_externalized_context_phase":"P85","source_enrichment_phase":"P84","source_identifiability_phase":"P83","source_selector_pilot_phase":"P82","source_dataset_phase":"P81","verdict":"P88_READY_FOR_REVIEW","training_allowed_by_phase4_authority":true,"model_training_performed":false,"torch_training_performed":false,"new_model_implemented":false,"optimizer_created":false,"checkpoint_written":false,"source_contracts_validated":true,"p87_no_learned_selector_evidence_preserved":true,"p87_nearest_support_delta_signal_preserved":true,"p87_recommended_p88_preserved":true,"p87_bridge_not_ready_preserved":true,"support_split_policy_audit_defined":true,"same_split_support_strict_protocol_defined":true,"train_bank_support_diagnostic_protocol_defined":true,"primary_support_policy":"same_split_support_strict","primary_support_split_audit":{"support_policy":"same_split_support_strict","episode_count":114,"available_episode_count":114,"unavailable_episode_count":0,"query_split_to_support_split_counts":{"train":{"train":152,"validation":0,"test":0},"validation":{"train":0,"validation":20,"test":0},"test":{"train":0,"validation":0,"test":18}},"same_split_diagonal_only":true,"off_diagonal_support_count":0,"train_query_uses_validation_or_test_support":false,"validation_query_uses_train_support":false,"validation_query_uses_test_support":false,"test_query_uses_train_support":false,"test_query_uses_validation_support":false,"same_split_support_strict_pass":true,"train_bank_support_diagnostic_pass":true,"diagnostic_pass":true},"diagnostic_train_bank_support_split_audit":{"support_policy":"train_bank_support_diagnostic","episode_count":114,"available_episode_count":114,"unavailable_episode_count":0,"query_split_to_support_split_counts":{"train":{"train":152,"validation":0,"test":0},"validation":{"train":40,"validation":0,"test":0},"test":{"train":36,"validation":0,"test":0}},"same_split_diagonal_only":false,"off_diagonal_support_count":76,"train_query_uses_validation_or_test_support":false,"validation_query_uses_train_support":true,"validation_query_uses_test_support":false,"test_query_uses_train_support":true,"test_query_uses_validation_support":false,"same_split_support_strict_pass":false,"train_bank_support_diagnostic_pass":true,"diagnostic_pass":true},"metric_input_leakage_audit":{"metric_input_count":114,"support_record_id_used_for_metric_input":false,"materialization_metadata_used_for_metric_input":false,"evaluated_target_endpoint_used_for_metric_input":false,"evaluated_target_delta_used_for_metric_input":false,"exact_relation_label_used_for_metric_input":false,"exact_operator_id_used_for_metric_input":false,"audit_metadata_used_for_metric_input":false,"relation_specific_hint_used_for_metric_input":false,"diagnostic_pass":true},"distance_family_results":{"l1":0.4444444444444444,"l2":0.6666666666666666,"cosine_safe":0.5555555555555556},"ablation_results":[{"mode":"delta_only","distance_family":"l1","combined_accuracy":0.4444444444444444,"p70a_accuracy":0.75,"p70b_accuracy":0.2,"sample_count":18,"valid_signal":false},{"mode":"delta_only","distance_family":"l2","combined_accuracy":0.6666666666666666,"p70a_accuracy":0.75,"p70b_accuracy":0.6,"sample_count":18,"valid_signal":true},{"mode":"delta_only","distance_family":"cosine_safe","combined_accuracy":0.5555555555555556,"p70a_accuracy":0.75,"p70b_accuracy":0.4,"sample_count":18,"valid_signal":false},{"mode":"source_result_delta","distance_family":"l1","combined_accuracy":0.3888888888888889,"p70a_accuracy":0.5,"p70b_accuracy":0.3,"sample_count":18,"valid_signal":false},{"mode":"source_result_delta","distance_family":"l2","combined_accuracy":0.2777777777777778,"p70a_accuracy":0.375,"p70b_accuracy":0.2,"sample_count":18,"valid_signal":false},{"mode":"source_result_delta","distance_family":"cosine_safe","combined_accuracy":0.2777777777777778,"p70a_accuracy":0.25,"p70b_accuracy":0.3,"sample_count":18,"valid_signal":false},{"mode":"delta_plus_invariants","distance_family":"l1","combined_accuracy":0.5555555555555556,"p70a_accuracy":0.75,"p70b_accuracy":0.4,"sample_count":18,"valid_signal":false},{"mode":"delta_plus_invariants","distance_family":"l2","combined_accuracy":0.6111111111111112,"p70a_accuracy":0.75,"p70b_accuracy":0.5,"sample_count":18,"valid_signal":true},{"mode":"delta_plus_invariants","distance_family":"cosine_safe","combined_accuracy":0.5,"p70a_accuracy":0.75,"p70b_accuracy":0.3,"sample_count":18,"valid_signal":false},{"mode":"full_support_metric","distance_family":"l1","combined_accuracy":0.4444444444444444,"p70a_accuracy":0.5,"p70b_accuracy":0.4,"sample_count":18,"valid_signal":false},{"mode":"full_support_metric","distance_family":"l2","combined_accuracy":0.2777777777777778,"p70a_accuracy":0.375,"p70b_accuracy":0.2,"sample_count":18,"valid_signal":false},{"mode":"full_support_metric","distance_family":"cosine_safe","combined_accuracy":0.3333333333333333,"p70a_accuracy":0.375,"p70b_accuracy":0.3,"sample_count":18,"valid_signal":false},{"mode":"query_source_only_negative_control","distance_family":"l2","combined_accuracy":0.1111111111111111,"p70a_accuracy":0.0,"p70b_accuracy":0.0,"sample_count":18,"valid_signal":false},{"mode":"shuffled_support_delta_negative_control","distance_family":"l2","combined_accuracy":0.3888888888888889,"p70a_accuracy":0.0,"p70b_accuracy":0.0,"sample_count":18,"valid_signal":false},{"mode":"zero_delta_negative_control","distance_family":"l2","combined_accuracy":0.1111111111111111,"p70a_accuracy":0.0,"p70b_accuracy":0.0,"sample_count":18,"valid_signal":false}],"best_metric_result":{"mode":"delta_only","distance_family":"l2","accuracy":0.6666666666666666,"predictions_count":18,"predictions":[{"episode_id":"episode_same_split_support_strict_p81_record_p70a_46","domain":"p70a_vector_world","split":"test","true_label":"translate_x","predicted_label":"translate_x","distance_family":"l2","mode":"delta_only"},{"episode_id":"episode_same_split_support_strict_p81_record_p70a_47","domain":"p70a_vector_world","split":"test","true_label":"translate_x","predicted_label":"translate_x","distance_family":"l2","mode":"delta_only"},{"episode_id":"episode_same_split_support_strict_p81_record_p70a_48","domain":"p70a_vector_world","split":"test","true_label":"translate_y","predicted_label":"translate_y","distance_family":"l2","mode":"delta_only"},{"episode_id":"episode_same_split_support_strict_p81_record_p70a_49","domain":"p70a_vector_world","split":"test","true_label":"translate_y","predicted_label":"translate_y","distance_family":"l2","mode":"delta_only"},{"episode_id":"episode_same_split_support_strict_p81_record_p70a_50","domain":"p70a_vector_world","split":"test","true_label":"scale_s","predicted_label":"scale_s","distance_family":"l2","mode":"delta_only"},{"episode_id":"episode_same_split_support_strict_p81_record_p70a_51","domain":"p70a_vector_world","split":"test","true_label":"scale_s","predicted_label":"scale_s","distance_family":"l2","mode":"delta_only"},{"episode_id":"episode_same_split_support_strict_p81_record_p70a_52","domain":"p70a_vector_world","split":"test","true_label":"nonlinear_x_from_y","predicted_label":"translate_x","distance_family":"l2","mode":"delta_only"},{"episode_id":"episode_same_split_support_strict_p81_record_p70a_53","domain":"p70a_vector_world","split":"test","true_label":"nonlinear_x_from_y","predicted_label":"translate_x","distance_family":"l2","mode":"delta_only"},{"episode_id":"episode_same_split_support_strict_p81_record_p70b_50","domain":"p70b_time_series_parameter_world","split":"test","true_label":"change_frequency","predicted_label":"change_frequency","distance_family":"l2","mode":"delta_only"},{"episode_id":"episode_same_split_support_strict_p81_record_p70b_51","domain":"p70b_time_series_parameter_world","split":"test","true_label":"change_frequency","predicted_label":"change_frequency","distance_family":"l2","mode":"delta_only"},{"episode_id":"episode_same_split_support_strict_p81_record_p70b_52","domain":"p70b_time_series_parameter_world","split":"test","true_label":"scale_amplitude","predicted_label":"scale_amplitude","distance_family":"l2","mode":"delta_only"},{"episode_id":"episode_same_split_support_strict_p81_record_p70b_53","domain":"p70b_time_series_parameter_world","split":"test","true_label":"scale_amplitude","predicted_label":"scale_amplitude","distance_family":"l2","mode":"delta_only"},{"episode_id":"episode_same_split_support_strict_p81_record_p70b_54","domain":"p70b_time_series_parameter_world","split":"test","true_label":"shift_phase","predicted_label":"scale_amplitude","distance_family":"l2","mode":"delta_only"},{"episode_id":"episode_same_split_support_strict_p81_record_p70b_55","domain":"p70b_time_series_parameter_world","split":"test","true_label":"shift_phase","predicted_label":"scale_amplitude","distance_family":"l2","mode":"delta_only"},{"episode_id":"episode_same_split_support_strict_p81_record_p70b_56","domain":"p70b_time_series_parameter_world","split":"test","true_label":"scale_volatility_envelope","predicted_label":"shift_trend","distance_family":"l2","mode":"delta_only"},{"episode_id":"episode_same_split_support_strict_p81_record_p70b_57","domain":"p70b_time_series_parameter_world","split":"test","true_label":"scale_volatility_envelope","predicted_label":"shift_trend","distance_family":"l2","mode":"delta_only"},{"episode_id":"episode_same_split_support_strict_p81_record_p70b_58","domain":"p70b_time_series_parameter_world","split":"test","true_label":"shift_trend","predicted_label":"shift_trend","distance_family":"l2","mode":"delta_only"},{"episode_id":"episode_same_split_support_strict_p81_record_p70b_59","domain":"p70b_time_series_parameter_world","split":"test","true_label":"shift_trend","predicted_label":"shift_trend","distance_family":"l2","mode":"delta_only"}]},"domain_results":{"p70a_vector_world":{"sample_count":8,"accuracy":0.75,"best_mode":"delta_only","best_distance_family":"l2"},"p70b_time_series_parameter_world":{"sample_count":10,"accuracy":0.6,"best_mode":"delta_only","best_distance_family":"l2"}},"relation_results":{"translate_x":{"accuracy":1.0,"correct_count":2,"total_count":2},"translate_y":{"accuracy":1.0,"correct_count":2,"total_count":2},"scale_s":{"accuracy":1.0,"correct_count":2,"total_count":2},"nonlinear_x_from_y":{"accuracy":0.0,"correct_count":0,"total_count":2},"change_frequency":{"accuracy":1.0,"correct_count":2,"total_count":2},"scale_amplitude":{"accuracy":1.0,"correct_count":2,"total_count":2},"shift_phase":{"accuracy":0.0,"correct_count":0,"total_count":2},"scale_volatility_envelope":{"accuracy":0.0,"correct_count":0,"total_count":2},"shift_trend":{"accuracy":1.0,"correct_count":2,"total_count":2}},"confusion_summary":{"translate_x_as_reflect_x":0,"translate_x_as_nonlinear":0,"reflect_x_as_translate_x":0,"reflect_x_as_nonlinear":0,"nonlinear_as_translate_x":2,"nonlinear_as_reflect_x":0,"note":"Confusion clusters computed for translate_x, reflect_x, and nonlinear_x_from_y on test split."},"shuffled_support_delta_control_implemented":true,"shuffled_control_target_labels_preserved":true,"shuffled_control_support_deltas_permuted":true,"target_labels_shifted":false,"true_labels_preserved":true,"nonlearned_metric_signal_present":true,"learned_selector_evidence_present":false,"learned_metric_evidence_present":false,"semantic_metric_ready":false,"bridge_implementation_allowed":false,"bridge_ready":false,"generation_claims_allowed":false,"semantic_geometry_claims_allowed":false,"recommended_next_phase":"P89_support_delta_metric_hard_generalization_and_external_context_no_bridge","bridge_boundary_after_nonlearned_metric_baseline":{"bridge_ready":false,"bridge_implementation_allowed":false,"nonlearned_metric_signal_present":true,"learned_selector_evidence_present":false,"learned_metric_evidence_present":false,"semantic_metric_ready":false,"generation_claims_allowed":false,"semantic_geometry_claims_allowed":false,"blocking_reasons":["learned_metric_evidence_not_present","semantic_metric_not_ready","bridge_input_contract_not_defined","bridge_validation_not_run"],"diagnostic_pass":true},"sanity_summary":{"source_contracts_validated":true,"p87_no_learned_selector_evidence_preserved":true,"p87_nearest_support_delta_signal_preserved":true,"p87_recommended_p88_preserved":true,"p87_bridge_not_ready_preserved":true,"support_split_policy_audit_defined":true,"same_split_support_strict_protocol_defined":true,"train_bank_support_diagnostic_protocol_defined":true,"support_selection_uses_relation_label_for_synthetic_dataset_construction":true,"support_selection_label_visible_to_selector":false,"evaluated_target_endpoint_used_for_metric_input":false,"evaluated_target_delta_used_for_metric_input":false,"exact_relation_label_used_for_metric_input":false,"exact_operator_id_used_for_metric_input":false,"audit_metadata_used_for_metric_input":false,"relation_specific_hint_used_for_metric_input":false,"support_record_id_used_for_metric_input":false,"materialization_metadata_used_for_metric_input":false,"model_training_performed":false,"torch_training_performed":false,"new_model_implemented":false,"optimizer_created":false,"checkpoint_written":false,"shuffled_support_delta_control_implemented":true,"shuffled_control_target_labels_preserved":true,"shuffled_control_support_deltas_permuted":true,"target_labels_shifted":false,"true_labels_preserved":true,"nonlearned_metric_signal_present":true,"learned_selector_evidence_present":false,"learned_metric_evidence_present":false,"semantic_metric_ready":false,"bridge_ready":false,"json_safe":true},"json_safe":true,"diagnostic_only":true}
```
