# PHASE 4 P87 Synthetic Few-Shot Selector Pilot Report

## 1. Problem Framing
P87 implements the first learned few-shot selector pilot from P86 materialized support episodes. We address whether a selector can identify the transformation relation of a query from externalized support demonstrations without seeing labels, hints, support IDs, materialization metadata, or query targets.

## 2. P86 Inheritance
P86 source contracts are fully validated:
- `p86_materialized_support_preserved = True`
- `p86_sanitized_collision_key_preserved = True`
- `p86_p87_eligibility_preserved = True`
- `p86_bridge_not_ready_preserved = True`

## 3. Support Split Policy and Audit
We implement and audit two split policies to avoid silent leakage:
- **`same_split_support_strict`**: strict policy (primary evidence). Validation/test queries use same-split supports only.
- **`train_bank_support_diagnostic`**: uses the train split support bank for validation/test queries (diagnostic only).

### Strict policy split count flow:
- `train` -> `train`: **152** supports
- `validation` -> `validation`: **20** supports
- `test` -> `test`: **18** supports
- Strict flow check: `same_split_support_strict_pass = True`
- Diagnostic flow check: `train_bank_support_diagnostic_pass = True`

## 4. Sanitized Model Input Contract
Model inputs are sanitized recursively to guarantee zero leakage:
- `support_record_id_used_for_model_input = False`
- `materialization_metadata_used_for_model_input = False`
- `evaluated_target_endpoint_used_for_model_input = False`
- `evaluated_target_delta_used_for_model_input = False`
- `exact_relation_label_used_for_model_input = False`
- `exact_operator_id_used_for_model_input = False`
- `audit_metadata_used_for_model_input = False`
- `relation_specific_hint_used_for_model_input = False`
- `diagnostic_pass = True`

## 5. Feature Encoding
- **Query features**: 11 floats (domain one-hot, intensity, P70A state properties, P70B parameter properties).
- **Support pair features**: 30 floats (15 per pair, padding missing pairs with zeros).
- **Invariant features**: 8 floats (counts, means, flags).
- **Total Feature Dimension**: **49**

## 6. Baselines
We compare the neural model against four baselines:
1. **Majority Class Baseline**: Predicts the most frequent class in train split (**11.11%** accuracy).
2. **Query-Source-Only tiny selector**: Equivalent to P82-style input but run on split records (**22.22%** accuracy).
3. **Nearest Support Delta Baseline**: Prototype distance metric match using support deltas only (**66.67%** accuracy).
4. **Support-Delta-Only linear classifier**: Trained on support delta features only (**55.56%** accuracy).

## 7. Model Architecture/Training
- **Model**: `TinyFewShotSelector` (input_dim=49, hidden=32, num_labels=10)
- **Optimizer**: AdamW (lr=0.01, epochs=150)
- **Loss**: CrossEntropyLoss
- **Environment**: In-memory only (no model/binary checkpoints written).

## 8. Combined Results
- **Test Sample Count**: **18**
- **Model Test Accuracy**: **22.22%**
- **Majority Class Accuracy**: **11.11%**
- **Query-Source-Only Accuracy**: **22.22%**
- **Nearest Support Delta Accuracy**: **66.67%** (Wins!)
- **Support-Delta-Only Accuracy**: **55.56%**

## 9. Domain-Specific Results
### P70A (Vector World)
- Sample count: **8**
- Model accuracy: **25.00%**
- Majority accuracy: **25.00%**
- Query-source-only accuracy: **0.00%**
- Nearest support delta accuracy: **75.00%**
- Support-delta-only accuracy: **75.00%**

### P70B (Time Series Parameter World)
- Sample count: **10**
- Model accuracy: **20.00%**
- Majority accuracy: **0.00%**
- Query-source-only accuracy: **40.00%**
- Nearest support delta accuracy: **60.00%**
- Support-delta-only accuracy: **40.00%**

## 10. Confusion Analysis
- `translate_x_as_reflect_x`: **0**
- `translate_x_as_nonlinear`: **0**
- `reflect_x_as_translate_x`: **0**
- `reflect_x_as_nonlinear`: **0**
- `nonlinear_as_translate_x`: **2**
- `nonlinear_as_reflect_x`: **0**

*Interpretation*: Confusion exists between `nonlinear_x_from_y` and `translate_x`, with the model predicting `translate_x` for 2 actual `nonlinear_x_from_y` episodes.

## 11. Evidence Verdict
- `learned_selector_evidence_present = False`
*Reason*: While the model slightly beats the majority class, it fails to beat the query-source-only baseline by at least 10% and fails to reach the required 60.00% test accuracy threshold.
However, the **non-learned** nearest support delta baseline shows a very strong accuracy of **66.67%**, proving that support deltas contain powerful relation identifier signal.

## 12. Bridge Boundary
Schrödinger Bridges remain strictly blocked:
- `bridge_ready = False`
- `bridge_implementation_allowed = False`
- `learned_selector_evidence_present = False`
- `learned_metric_evidence_present = False`

## 13. Limitations
- P87 does not prove semantic geometry.
- P87 does not prove learned metric.
- P87 does not validate any bridge.
- P87 only tests synthetic few-shot relation selection on P70-derived support episodes.

## 14. Final Verdict
`P87_READY_FOR_REVIEW`

## Smoke Output Block
```json
{"phase":"P87","phase_group":"PHASE_4","phase_name":"Synthetic Few-Shot Selector Pilot from Materialized Support Episodes","contract_version":"phase4_p87_synthetic_few_shot_selector_pilot_v1","source_materializer_phase":"P86","source_externalized_context_phase":"P85","source_enrichment_phase":"P84","source_identifiability_phase":"P83","source_selector_pilot_phase":"P82","source_dataset_phase":"P81","verdict":"P87_READY_FOR_REVIEW","training_allowed_by_phase4_authority":true,"model_training_performed":true,"torch_training_performed":true,"new_model_implemented":true,"optimizer_created":true,"checkpoint_written":false,"source_contracts_validated":true,"p86_materialized_support_preserved":true,"p86_sanitized_collision_key_preserved":true,"p86_p87_eligibility_preserved":true,"p86_bridge_not_ready_preserved":true,"support_split_policy_audit_defined":true,"same_split_support_strict_protocol_defined":true,"train_bank_support_diagnostic_protocol_defined":true,"primary_support_policy":"same_split_support_strict","primary_support_split_audit":{"support_policy":"same_split_support_strict","episode_count":114,"available_episode_count":114,"unavailable_episode_count":0,"query_split_to_support_split_counts":{"train":{"train":152,"validation":0,"test":0},"validation":{"train":0,"validation":20,"test":0},"test":{"train":0,"validation":0,"test":18}},"train_query_uses_validation_or_test_support":false,"validation_query_uses_test_support":false,"test_query_uses_train_support":false,"same_split_support_strict_pass":true,"train_bank_support_diagnostic_pass":false,"diagnostic_pass":true},"diagnostic_train_bank_support_split_audit":{"support_policy":"train_bank_support_diagnostic","episode_count":114,"available_episode_count":114,"unavailable_episode_count":0,"query_split_to_support_split_counts":{"train":{"train":152,"validation":0,"test":0},"validation":{"train":40,"validation":0,"test":0},"test":{"train":36,"validation":0,"test":0}},"train_query_uses_validation_or_test_support":false,"validation_query_uses_test_support":false,"test_query_uses_train_support":true,"same_split_support_strict_pass":false,"train_bank_support_diagnostic_pass":true,"diagnostic_pass":true},"model_input_leakage_audit":{"support_record_id_used_for_model_input":false,"materialization_metadata_used_for_model_input":false,"evaluated_target_endpoint_used_for_model_input":false,"evaluated_target_delta_used_for_model_input":false,"exact_relation_label_used_for_model_input":false,"exact_operator_id_used_for_model_input":false,"audit_metadata_used_for_model_input":false,"relation_specific_hint_used_for_model_input":false,"diagnostic_pass":true},"feature_dim":49,"num_labels":10,"baseline_results":{"majority_accuracy":0.1111111111111111,"query_source_only_accuracy":0.2222222222222222,"nearest_support_delta_accuracy":0.6666666666666666,"support_delta_only_accuracy":0.5555555555555556},"model_results":{"test_sample_count":18,"model_test_accuracy":0.2222222222222222},"domain_results":{"p70a_vector_world":{"sample_count":8,"majority_accuracy":0.25,"query_source_only_accuracy":0.0,"nearest_support_delta_accuracy":0.75,"support_delta_only_accuracy":0.75,"model_accuracy":0.25},"p70b_time_series_parameter_world":{"sample_count":10,"majority_accuracy":0.0,"query_source_only_accuracy":0.4,"nearest_support_delta_accuracy":0.6,"support_delta_only_accuracy":0.4,"model_accuracy":0.2}},"confusion_summary":{"translate_x_as_reflect_x":0,"translate_x_as_nonlinear":0,"reflect_x_as_translate_x":0,"reflect_x_as_nonlinear":0,"nonlinear_as_translate_x":2,"nonlinear_as_reflect_x":0,"note":"Confusion clusters computed for translate_x, reflect_x, and nonlinear_x_from_y on test split."},"learned_selector_evidence_present":false,"learned_metric_evidence_present":false,"semantic_metric_ready":false,"bridge_implementation_allowed":false,"bridge_ready":false,"generation_claims_allowed":false,"semantic_geometry_claims_allowed":false,"recommended_next_phase":"P88_nonlearned_support_delta_metric_baseline_formalization_no_bridge","bridge_boundary_after_few_shot_selector_pilot":{"bridge_ready":false,"bridge_implementation_allowed":false,"learned_selector_evidence_present":false,"learned_metric_evidence_present":false,"semantic_metric_ready":false,"generation_claims_allowed":false,"semantic_geometry_claims_allowed":false,"blocking_reasons":["learned_metric_evidence_not_present","semantic_metric_not_ready","bridge_input_contract_not_defined","bridge_validation_not_run"],"diagnostic_pass":true},"sanity_summary":{"source_contracts_validated":true,"p86_materialized_support_preserved":true,"p86_sanitized_collision_key_preserved":true,"p86_p87_eligibility_preserved":true,"p86_bridge_not_ready_preserved":true,"support_split_policy_audit_defined":true,"same_split_support_strict_protocol_defined":true,"train_bank_support_diagnostic_protocol_defined":true,"support_selection_uses_relation_label_for_synthetic_dataset_construction":true,"support_selection_label_visible_to_selector":false,"evaluated_target_endpoint_used_for_model_input":false,"evaluated_target_delta_used_for_model_input":false,"exact_relation_label_used_for_model_input":false,"exact_operator_id_used_for_model_input":false,"audit_metadata_used_for_model_input":false,"relation_specific_hint_used_for_model_input":false,"support_record_id_used_for_model_input":false,"materialization_metadata_used_for_model_input":false,"model_training_performed":true,"torch_training_performed":true,"new_model_implemented":true,"optimizer_created":true,"checkpoint_written":false,"learned_selector_evidence_present":false,"learned_metric_evidence_present":false,"semantic_metric_ready":false,"bridge_ready":false,"json_safe":true},"json_safe":true,"diagnostic_only":true}
```
