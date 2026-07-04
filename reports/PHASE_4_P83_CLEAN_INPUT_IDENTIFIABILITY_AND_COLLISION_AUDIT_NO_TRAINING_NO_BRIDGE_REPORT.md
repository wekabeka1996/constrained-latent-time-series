# PHASE 4 P83 Clean Input Identifiability and Collision Audit Report

## 1. Problem Framing
P83 investigates why the P82 tiny selector pilot failed to establish learned selector evidence. The model trained successfully and avoided target leakage, but failed to beat the source-only baseline on the test split. P83 audits whether identical or near-identical clean inputs map to multiple different relation labels (label collisions), resulting in an underdetermined task.

## 2. P82 Inheritance
P82 results are successfully validated and inherited:
- `p82_tiny_selector_trained_preserved = True`
- `p82_clean_input_only_preserved = True`
- `p82_no_learned_selector_evidence_preserved = True`
- `p82_bridge_not_ready_preserved = True`

## 3. Identifiability Question & Keys Analyzed
P83 answers: **Can the relation label be determined from the clean selector input at all?**
We analyze three keys constructed from the dataset records in canonical form:
1. **full_clean_input_key**: canonical JSON of domain, intensity hint, context world, and summaries (no split or record ids).
2. **p82_model_feature_key**: canonical representation of P82's 13 float features in pure Python.
3. **source_only_key**: clean key omitting intensity.

## 4. Label Distribution Analysis
The dataset contains 114 total records:
- **p70a (Vector World)**: 54 records (12 each for translate_x, translate_y, scale_s, nonlinear_x_from_y; 6 for reflect_x)
- **p70b (Parameter World)**: 60 records (12 each for change_frequency, scale_amplitude, shift_phase, scale_volatility_envelope, shift_trend)
- Splits are highly balanced (8 train, 2 validation, 2 test per class for p70a/p70b classes).

## 5. Collision Analysis & Deterministic Upper Bound
We grouped the records by each key and analyzed collisions and deterministic upper bounds:
- **full_clean_input_key**:
  - Unique keys: 46
  - Collided keys: 24
  - Records in collided keys: 92 (80.70%)
  - Max labels per key: 5
  - Deterministic Upper-Bound Accuracy: **40.35%** (all splits), **55.56%** (test split)
- **p82_model_feature_key**:
  - Unique keys: 22
  - Collided keys: 15
  - Records in collided keys: 96 (84.21%)
  - Max labels per key: 5
  - Deterministic Upper-Bound Accuracy: **40.35%** (all splits), **55.56%** (test split)
- **source_only_key**:
  - Unique keys: 12
  - Collided keys: 12
  - Records in collided keys: 114 (100.00%)
  - Max labels per key: 5
  - Deterministic Upper-Bound Accuracy: **21.05%** (all splits), **22.22%** (test split)

## 6. Split-Aware Collision Analysis (p82_model_feature_key)
- **Train split**: 84.21% collided records, upper-bound accuracy **36.84%**
- **Validation split**: 70.00% collided records, upper-bound accuracy **50.00%**
- **Test split**: 66.67% collided records, upper-bound accuracy **55.56%**

## 7. Interpretation
- `clean_input_relation_identifiability_established = False`
- `clean_input_label_collisions_present = True`
- `model_capacity_not_primary_failure = True`
- `additional_observation_context_required = True`
- `recommended_next_phase = "P84_clean_query_observation_enrichment_contract"`

The high rate of label collisions (84.21% on features) proves that the same clean features correspond to multiple different labels. The failure is an **information/identifiability underdetermination**, not a model capacity failure. Adding capacity or model parameters cannot resolve this collision. We must enrich selector inputs with query-observation context (e.g., source state \(z_a\) and intermediate/target endpoints under governance restrictions).

## 8. Bridge Boundary
Schrödinger Bridges remain strictly blocked:
- `bridge_ready = False`
- `bridge_implementation_allowed = False`
- `learned_selector_evidence_present = False`
- `learned_metric_evidence_present = False`

Blocking reasons:
1. `learned_selector_evidence_not_present`
2. `learned_metric_evidence_not_present`
3. `clean_input_identifiability_not_established`
4. `bridge_input_contract_not_defined`
5. `bridge_validation_not_run`

## 9. Results
Final Verdict: `P83_READY_FOR_REVIEW`

## Smoke Output Block
```json
{"phase":"P83","phase_group":"PHASE_4","phase_name":"Clean Input Identifiability and Collision Audit","contract_version":"phase4_p83_clean_input_identifiability_collision_audit_v1","source_selector_pilot_phase":"P82","source_dataset_phase":"P81","verdict":"P83_READY_FOR_REVIEW","training_allowed_by_phase4_authority":true,"model_training_performed":false,"torch_training_performed":false,"new_model_implemented":false,"optimizer_created":false,"checkpoint_written":false,"clean_selector_input_used":true,"hint_passthrough_used_as_model_input":false,"target_endpoint_used_for_selector_input":false,"target_delta_used_for_selector_input":false,"exact_relation_label_used_for_selector_input":false,"exact_operator_id_used_for_selector_input":false,"audit_metadata_used_for_selector_input":false,"relation_specific_hint_used_for_model_input":false,"clean_input_identifiability_audit_performed":true,"clean_input_label_collision_audit_performed":true,"deterministic_oracle_upper_bound_computed":true,"learned_selector_evidence_present":false,"learned_metric_evidence_present":false,"semantic_metric_ready":false,"bridge_implementation_allowed":false,"bridge_ready":false,"generation_claims_allowed":false,"semantic_geometry_claims_allowed":false,"source_contracts_validated":true,"p82_tiny_selector_trained_preserved":true,"p82_clean_input_only_preserved":true,"p82_no_learned_selector_evidence_preserved":true,"p82_bridge_not_ready_preserved":true,"label_distribution_audit":{"total_record_count":114,"label_distribution_all":{"translate_x":12,"translate_y":12,"scale_s":12,"reflect_x":6,"nonlinear_x_from_y":12,"change_frequency":12,"scale_amplitude":12,"shift_phase":12,"scale_volatility_envelope":12,"shift_trend":12},"label_distribution_by_split":{"train":{"translate_x":8,"translate_y":8,"scale_s":8,"reflect_x":4,"nonlinear_x_from_y":8,"change_frequency":8,"scale_amplitude":8,"shift_phase":8,"scale_volatility_envelope":8,"shift_trend":8},"validation":{"translate_x":2,"translate_y":2,"scale_s":2,"reflect_x":2,"nonlinear_x_from_y":2,"change_frequency":2,"scale_amplitude":2,"shift_phase":2,"scale_volatility_envelope":2,"shift_trend":2},"test":{"translate_x":2,"translate_y":2,"scale_s":2,"nonlinear_x_from_y":2,"change_frequency":2,"scale_amplitude":2,"shift_phase":2,"scale_volatility_envelope":2,"shift_trend":2}},"label_distribution_by_domain":{"p70a_vector_world":{"translate_x":12,"translate_y":12,"scale_s":12,"reflect_x":6,"nonlinear_x_from_y":12},"p70b_time_series_parameter_world":{"change_frequency":12,"scale_amplitude":12,"shift_phase":12,"scale_volatility_envelope":12,"shift_trend":12}},"diagnostic_pass":true},"collision_audits":{"full_clean_input_key":{"key_name":"full_clean_input_key","record_count":114,"unique_key_count":46,"collided_key_count":24,"non_collided_key_count":22,"records_in_collided_keys":92,"collision_record_fraction":0.8070175438596491,"max_labels_per_key":5,"mean_labels_per_key":2.4782608695652173,"deterministic_identifiability_upper_bound_accuracy":0.40350877192982454,"diagnostic_pass":true},"p82_model_feature_key":{"key_name":"p82_model_feature_key","record_count":114,"unique_key_count":22,"collided_key_count":15,"non_collided_key_count":7,"records_in_collided_keys":96,"collision_record_fraction":0.8421052631578947,"max_labels_per_key":5,"mean_labels_per_key":3.1363636363636362,"deterministic_identifiability_upper_bound_accuracy":0.40350877192982454,"diagnostic_pass":true},"source_only_key":{"key_name":"source_only_key","record_count":114,"unique_key_count":12,"collided_key_count":12,"non_collided_key_count":0,"records_in_collided_keys":114,"collision_record_fraction":1.0,"max_labels_per_key":5,"mean_labels_per_key":5.0,"deterministic_identifiability_upper_bound_accuracy":0.21052631578947367,"diagnostic_pass":true}},"split_aware_collision_audits":{"full_clean_input_key":{"key_name":"full_clean_input_key","train":{"key_name":"full_clean_input_key","record_count":76,"unique_key_count":28,"collided_key_count":16,"non_collided_key_count":12,"records_in_collided_keys":64,"collision_record_fraction":0.8421052631578947,"max_labels_per_key":5,"mean_labels_per_key":2.7142857142857144,"deterministic_identifiability_upper_bound_accuracy":0.3684210526315789,"diagnostic_pass":true},"validation":{"key_name":"full_clean_input_key","record_count":20,"unique_key_count":10,"collided_key_count":4,"non_collided_key_count":6,"records_in_collided_keys":14,"collision_record_fraction":0.7,"max_labels_per_key":4,"mean_labels_per_key":2.0,"deterministic_identifiability_upper_bound_accuracy":0.5,"diagnostic_pass":true},"test":{"key_name":"full_clean_input_key","record_count":18,"unique_key_count":10,"collided_key_count":4,"non_collided_key_count":6,"records_in_collided_keys":12,"collision_record_fraction":0.6666666666666666,"max_labels_per_key":4,"mean_labels_per_key":1.8,"deterministic_identifiability_upper_bound_accuracy":0.5555555555555556,"diagnostic_pass":true},"all":{"key_name":"full_clean_input_key","record_count":114,"unique_key_count":46,"collided_key_count":24,"non_collided_key_count":22,"records_in_collided_keys":92,"collision_record_fraction":0.8070175438596491,"max_labels_per_key":5,"mean_labels_per_key":2.4782608695652173,"deterministic_identifiability_upper_bound_accuracy":0.40350877192982454,"diagnostic_pass":true},"diagnostic_pass":true},"p82_model_feature_key":{"key_name":"p82_model_feature_key","train":{"key_name":"p82_model_feature_key","record_count":76,"unique_key_count":13,"collided_key_count":10,"non_collided_key_count":3,"records_in_collided_keys":64,"collision_record_fraction":0.8421052631578947,"max_labels_per_key":5,"mean_labels_per_key":3.5384615384615383,"deterministic_identifiability_upper_bound_accuracy":0.3684210526315789,"diagnostic_pass":true},"validation":{"key_name":"p82_model_feature_key","record_count":20,"unique_key_count":7,"collided_key_count":3,"non_collided_key_count":4,"records_in_collided_keys":14,"collision_record_fraction":0.7,"max_labels_per_key":4,"mean_labels_per_key":2.142857142857143,"deterministic_identifiability_upper_bound_accuracy":0.5,"diagnostic_pass":true},"test":{"key_name":"p82_model_feature_key","record_count":18,"unique_key_count":6,"collided_key_count":3,"non_collided_key_count":3,"records_in_collided_keys":12,"collision_record_fraction":0.6666666666666666,"max_labels_per_key":4,"mean_labels_per_key":2.1666666666666665,"deterministic_identifiability_upper_bound_accuracy":0.5555555555555556,"diagnostic_pass":true},"all":{"key_name":"p82_model_feature_key","record_count":114,"unique_key_count":22,"collided_key_count":15,"non_collided_key_count":7,"records_in_collided_keys":96,"collision_record_fraction":0.8421052631578947,"max_labels_per_key":5,"mean_labels_per_key":3.1363636363636362,"deterministic_identifiability_upper_bound_accuracy":0.40350877192982454,"diagnostic_pass":true},"diagnostic_pass":true},"source_only_key":{"key_name":"source_only_key","train":{"key_name":"source_only_key","record_count":76,"unique_key_count":8,"collided_key_count":8,"non_collided_key_count":0,"records_in_collided_keys":76,"collision_record_fraction":1.0,"max_labels_per_key":5,"mean_labels_per_key":5.0,"deterministic_identifiability_upper_bound_accuracy":0.21052631578947367,"diagnostic_pass":true},"validation":{"key_name":"source_only_key","record_count":20,"unique_key_count":4,"collided_key_count":4,"non_collided_key_count":0,"records_in_collided_keys":20,"collision_record_fraction":1.0,"max_labels_per_key":5,"mean_labels_per_key":5.0,"deterministic_identifiability_upper_bound_accuracy":0.2,"diagnostic_pass":true},"test":{"key_name":"source_only_key","record_count":18,"unique_key_count":4,"collided_key_count":4,"non_collided_key_count":0,"records_in_collided_keys":18,"collision_record_fraction":1.0,"max_labels_per_key":5,"mean_labels_per_key":4.5,"deterministic_identifiability_upper_bound_accuracy":0.2222222222222222,"diagnostic_pass":true},"all":{"key_name":"source_only_key","record_count":114,"unique_key_count":12,"collided_key_count":12,"non_collided_key_count":0,"records_in_collided_keys":114,"collision_record_fraction":1.0,"max_labels_per_key":5,"mean_labels_per_key":5.0,"deterministic_identifiability_upper_bound_accuracy":0.21052631578947367,"diagnostic_pass":true},"diagnostic_pass":true}},"identifiability_interpretation":{"clean_input_relation_identifiability_established":false,"clean_input_label_collisions_present":true,"model_capacity_not_primary_failure":true,"additional_observation_context_required":true,"recommended_next_phase":"P84_clean_query_observation_enrichment_contract","diagnostic_pass":true},"bridge_boundary_after_identifiability_audit":{"bridge_ready":false,"bridge_implementation_allowed":false,"learned_selector_evidence_present":false,"learned_metric_evidence_present":false,"semantic_metric_ready":false,"generation_claims_allowed":false,"semantic_geometry_claims_allowed":false,"blocking_reasons":["learned_selector_evidence_not_present","learned_metric_evidence_not_present","clean_input_identifiability_not_established","bridge_input_contract_not_defined","bridge_validation_not_run"],"diagnostic_pass":true},"sanity_summary":{"source_contracts_validated":true,"p82_tiny_selector_trained_preserved":true,"p82_clean_input_only_preserved":true,"p82_no_learned_selector_evidence_preserved":true,"p82_bridge_not_ready_preserved":true,"clean_input_identifiability_audit_performed":true,"clean_input_label_collision_audit_performed":true,"deterministic_oracle_upper_bound_computed":true,"clean_input_relation_identifiability_established":false,"clean_input_label_collisions_present":true,"model_capacity_not_primary_failure":true,"additional_observation_context_required":true,"model_training_performed":false,"torch_training_performed":false,"new_model_implemented":false,"optimizer_created":false,"checkpoint_written":false,"learned_selector_evidence_present":false,"learned_metric_evidence_present":false,"semantic_metric_ready":false,"bridge_ready":false,"json_safe":true},"json_safe":true,"diagnostic_only":true}
```

## Focused Test Result
- **Command**: `python -m pytest tests/test_phase4_clean_input_identifiability_collision_audit.py tests/test_phase4_p83_clean_input_identifiability_collision_audit_smoke.py -v`
- **Result**: `16 PASSED`

## Scope Gate
Scope gate successfully verified using Git diff:
- `expected_branch = "phase4/p83-clean-input-identifiability-and-collision-audit-no-training-no-bridge"`
- `base_commit = "407576dc0f81ced81980ac6c683205ab926218fe"`

## Limitations
- P83 does not train a model.
- P83 does not prove learned selector evidence or learned metric evidence.
- P83 does not prove semantic geometry.
- P83 does not validate any bridge.
- P83 only audits identifiability/collisions of current clean inputs.
