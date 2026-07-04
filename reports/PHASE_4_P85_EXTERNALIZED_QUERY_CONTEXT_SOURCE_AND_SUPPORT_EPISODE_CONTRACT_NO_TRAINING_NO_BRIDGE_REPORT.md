# PHASE 4 P85 Externalized Query Context Source and Support Episode Contract Report

## 1. Problem Framing
P85 follows the P84 finding that source-only support observations do not reduce label collisions. P85 formalizes the contract for externalized query episodes, where support observations become part of the task query itself rather than hidden label metadata pass-through.

## 2. P84 Inheritance
P84 validation is successfully inherited:
- `p84_query_observation_contract_preserved = True`
- `p84_no_final_selector_evidence_preserved = True`
- `p84_real_external_context_required_preserved = True`
- `p84_bridge_not_ready_preserved = True`

## 3. Externalized Query Context Source
To ensure rigorous validation without label leaks, P85 separates three design concepts:
1. **Selector Input**: The query source input and support pair observations. Excludes target labels/hints.
2. **Episode Manifest**: Synthetically constructed by the task generator using hidden labels, but not visible to the selector.
3. **Audit View**: Holds labels for evaluation.

## 4. Support Episode Schema
The externalized support episode contract defines:
- **`query_source_input`**: Domain, intensity, context world, parameter and state summaries (no target z_b/endpoints).
- **`externalized_support_episode_context`**:
  - `support_pairs`: domain, context, source summary, result summary, and delta summary.
  - `support_invariant_context`: invariant summary characteristics.
  - `episode_context_metadata`.

## 5. Leakage Boundaries
The selector input contains zero target record endpoints, deltas, labels, operator IDs, or relation-specific hints of the evaluated record:
- `evaluated_target_endpoint_used_for_selector_input = False`
- `evaluated_target_delta_used_for_selector_input = False`
- `exact_relation_label_used_for_selector_input = False`
- `exact_operator_id_used_for_selector_input = False`
- `audit_metadata_used_for_selector_input = False`
- `relation_specific_hint_used_for_model_input = False`

## 6. Support Result/Delta Availability
Since P81 records only contain source parameter/state summaries, support result and delta summaries are unavailable. Therefore:
```text
support_pair_observation_available = False
support_episode_limited_by_missing_result_or_delta_summaries = True
```

## 7. Episode Collision Audit
As result/delta summaries are absent, the collision audit excludes support pairs from the collision keys:
- **Original Collision Record Fraction**: **80.70%** (all splits)
- **Episode Collision Record Fraction**: **80.70%** (all splits)
- **Collision Fraction Reduced**: **False**
- **Original Upper Bound Accuracy**: **40.35%** (all splits)
- **Episode Upper Bound Accuracy**: **40.35%** (all splits)
- **Upper Bound Improved**: **False**

No collision reduction is claimed.

## 8. Interpretation
- `externalized_query_context_source_defined = True`
- `support_episode_context_feasibility_supported = False`
- `valid_for_p86_synthetic_few_shot_selector_experiment = False`
- `valid_for_final_semantic_geometry_evidence = False`
- `external_real_query_source_still_required = True`
- `recommended_next_phase = "P86_support_pair_materializer_from_p70_relation_cases_no_training_no_bridge"`

Because the current synthetic records lack result/delta summaries, we cannot run synthetic P86 few-shot selectors yet. In P86, we must build a support pair materializer that draws full source-result-delta pairs directly from P70 relation case probes.

## 9. Bridge Boundary
Schrödinger Bridges remain strictly blocked:
- `bridge_ready = False`
- `bridge_implementation_allowed = False`
- `learned_selector_evidence_present = False`
- `learned_metric_evidence_present = False`

Blocking reasons:
1. `learned_selector_evidence_not_present`
2. `learned_metric_evidence_not_present`
3. `external_real_query_source_still_required`
4. `bridge_input_contract_not_defined`
5. `bridge_validation_not_run`

## 10. Results
Final Verdict: `P85_READY_FOR_REVIEW`

## Smoke Output Block
```json
{"phase":"P85","phase_group":"PHASE_4","phase_name":"Externalized Query Context Source and Support Episode Contract","contract_version":"phase4_p85_externalized_query_context_support_episode_contract_v1","source_enrichment_phase":"P84","source_identifiability_phase":"P83","source_selector_pilot_phase":"P82","source_dataset_phase":"P81","verdict":"P85_READY_FOR_REVIEW","training_allowed_by_phase4_authority":true,"model_training_performed":false,"torch_training_performed":false,"new_model_implemented":false,"optimizer_created":false,"checkpoint_written":false,"externalized_query_context_source_defined":true,"support_episode_context_defined":true,"support_pair_result_delta_schema_defined":true,"episode_manifest_schema_defined":true,"support_invariant_context_schema_defined":true,"support_selection_uses_relation_label_for_synthetic_dataset_construction":true,"support_selection_label_visible_to_selector":false,"support_context_constructed_by_synthetic_task_generator":true,"valid_for_p86_synthetic_few_shot_selector_experiment":false,"valid_for_final_semantic_geometry_evidence":false,"valid_for_bridge_evidence":false,"external_real_query_source_still_required":true,"evaluated_target_endpoint_used_for_selector_input":false,"evaluated_target_delta_used_for_selector_input":false,"exact_relation_label_used_for_selector_input":false,"exact_operator_id_used_for_selector_input":false,"audit_metadata_used_for_selector_input":false,"relation_specific_hint_used_for_model_input":false,"learned_selector_evidence_present":false,"learned_metric_evidence_present":false,"semantic_metric_ready":false,"bridge_implementation_allowed":false,"bridge_ready":false,"generation_claims_allowed":false,"semantic_geometry_claims_allowed":false,"source_contracts_validated":true,"p84_query_observation_contract_preserved":true,"p84_no_final_selector_evidence_preserved":true,"p84_real_external_context_required_preserved":true,"p84_bridge_not_ready_preserved":true,"dataset_record_count":114,"episode_record_count":114,"episode_manifest_count":114,"episode_input_leakage_audit":{"evaluated_target_endpoint_used_for_selector_input":false,"evaluated_target_delta_used_for_selector_input":false,"exact_relation_label_used_for_selector_input":false,"exact_operator_id_used_for_selector_input":false,"audit_metadata_used_for_selector_input":false,"relation_specific_hint_used_for_model_input":false,"support_selection_label_visible_to_selector":false,"diagnostic_pass":true},"episode_collision_reduction_audit":{"original_collision_record_fraction":0.8070175438596491,"episode_collision_record_fraction":0.8070175438596491,"collision_fraction_reduced":false,"original_upper_bound_accuracy":0.40350877192982454,"episode_upper_bound_accuracy":0.40350877192982454,"upper_bound_improved":false,"support_pair_observation_available_count":0,"support_episode_limited_by_missing_result_or_delta_summaries":true,"valid_for_p86_synthetic_few_shot_selector_experiment":false,"valid_for_final_semantic_geometry_evidence":false,"valid_for_bridge_evidence":false,"diagnostic_pass":true},"externalized_query_context_interpretation":{"externalized_query_context_source_defined":true,"support_episode_context_feasibility_supported":false,"valid_for_p86_synthetic_few_shot_selector_experiment":false,"valid_for_final_semantic_geometry_evidence":false,"external_real_query_source_still_required":true,"recommended_next_phase":"P86_support_pair_materializer_from_p70_relation_cases_no_training_no_bridge","diagnostic_pass":true},"bridge_boundary_after_externalized_query_context":{"bridge_ready":false,"bridge_implementation_allowed":false,"learned_selector_evidence_present":false,"learned_metric_evidence_present":false,"semantic_metric_ready":false,"generation_claims_allowed":false,"semantic_geometry_claims_allowed":false,"blocking_reasons":["learned_selector_evidence_not_present","learned_metric_evidence_not_present","external_real_query_source_still_required","bridge_input_contract_not_defined","bridge_validation_not_run"],"diagnostic_pass":true},"sample_episode_records":[{"episode_id":"episode_p81_record_p70a_0","query_record_id":"p81_record_p70a_0","split":"train","domain":"p70a_vector_world","externalized_episode_selector_input":{"query_source_input":{"domain":"p70a_vector_world","query_intensity_hint":0.5,"context_world":"p70a","source_state_summary":{"z_a_dim":3,"z_a_abs_sum":1.0,"z_a_sign_pattern":[0,0,1]}},"externalized_support_episode_context":{"support_pair_count":2,"support_pairs":[{"support_record_id":"p81_record_p70a_1","support_domain":"p70a_vector_world","support_context_world":"p70a","support_result_summary":{},"support_delta_summary":{},"support_pair_observation_available":false,"missing_result_summary":true,"missing_delta_summary":true,"support_source_summary":{"z_a_dim":3,"z_a_abs_sum":1.0,"z_a_sign_pattern":[0,0,1]}},{"support_record_id":"p81_record_p70a_2","support_domain":"p70a_vector_world","support_context_world":"p70a","support_result_summary":{},"support_delta_summary":{},"support_pair_observation_available":false,"missing_result_summary":true,"missing_delta_summary":true,"support_source_summary":{"z_a_dim":3,"z_a_abs_sum":3.5,"z_a_sign_pattern":[1,1,1]}}],"support_invariant_context":{"support_pair_count":2,"support_pair_observation_available_count":0,"result_summary_available_count":0,"delta_summary_available_count":0,"support_delta_abs_sum_mean":0.0,"support_delta_nonzero_count_mean":0.0,"support_invariant_context_available":false,"support_episode_limited_by_missing_result_or_delta_summaries":true},"episode_context_metadata":{"context_source_kind":"externalized_synthetic_support_episode","support_selection_uses_relation_label_for_dataset_construction":true,"support_selection_label_visible_to_selector":false,"valid_for_p86_synthetic_few_shot_selector_experiment":true,"valid_for_final_semantic_geometry_evidence":false,"valid_for_bridge_evidence":false,"external_real_query_source_still_required":true}}},"externalized_episode_manifest":{"episode_id":"episode_p81_record_p70a_0","query_record_id":"p81_record_p70a_0","support_record_ids":["p81_record_p70a_1","p81_record_p70a_2"],"split":"train","domain":"p70a_vector_world","construction_source":"synthetic_task_generator","support_selection_uses_relation_label_for_dataset_construction":true,"support_selection_label_visible_to_selector":false,"valid_for_p86_synthetic_few_shot_selector_experiment":true,"valid_for_final_semantic_geometry_evidence":false,"valid_for_bridge_evidence":false},"audit_label_evaluation_only":{"target_relation_label":"translate_x","domain":"p70a_vector_world","descriptor_id":"p70a_descriptor_train_style_repeated_instances_0","split":"train"},"episode_input_leakage_audit":{"evaluated_target_endpoint_used_for_selector_input":false,"evaluated_target_delta_used_for_selector_input":false,"exact_relation_label_used_for_selector_input":false,"exact_operator_id_used_for_selector_input":false,"audit_metadata_used_for_selector_input":false,"relation_specific_hint_used_for_model_input":false,"support_selection_label_visible_to_selector":false,"diagnostic_pass":true}},{"episode_id":"episode_p81_record_p70a_1","query_record_id":"p81_record_p70a_1","split":"train","domain":"p70a_vector_world","externalized_episode_selector_input":{"query_source_input":{"domain":"p70a_vector_world","query_intensity_hint":1.0,"context_world":"p70a","source_state_summary":{"z_a_dim":3,"z_a_abs_sum":1.0,"z_a_sign_pattern":[0,0,1]}},"externalized_support_episode_context":{"support_pair_count":2,"support_pairs":[{"support_record_id":"p81_record_p70a_0","support_domain":"p70a_vector_world","support_context_world":"p70a","support_result_summary":{},"support_delta_summary":{},"support_pair_observation_available":false,"missing_result_summary":true,"missing_delta_summary":true,"support_source_summary":{"z_a_dim":3,"z_a_abs_sum":1.0,"z_a_sign_pattern":[0,0,1]}},{"support_record_id":"p81_record_p70a_2","support_domain":"p70a_vector_world","support_context_world":"p70a","support_result_summary":{},"support_delta_summary":{},"support_pair_observation_available":false,"missing_result_summary":true,"missing_delta_summary":true,"support_source_summary":{"z_a_dim":3,"z_a_abs_sum":3.5,"z_a_sign_pattern":[1,1,1]}}],"support_invariant_context":{"support_pair_count":2,"support_pair_observation_available_count":0,"result_summary_available_count":0,"delta_summary_available_count":0,"support_delta_abs_sum_mean":0.0,"support_delta_nonzero_count_mean":0.0,"support_invariant_context_available":false,"support_episode_limited_by_missing_result_or_delta_summaries":true},"episode_context_metadata":{"context_source_kind":"externalized_synthetic_support_episode","support_selection_uses_relation_label_for_dataset_construction":true,"support_selection_label_visible_to_selector":false,"valid_for_p86_synthetic_few_shot_selector_experiment":true,"valid_for_final_semantic_geometry_evidence":false,"valid_for_bridge_evidence":false,"external_real_query_source_still_required":true}}},"externalized_episode_manifest":{"episode_id":"episode_p81_record_p70a_1","query_record_id":"p81_record_p70a_1","support_record_ids":["p81_record_p70a_0","p81_record_p70a_2"],"split":"train","domain":"p70a_vector_world","construction_source":"synthetic_task_generator","support_selection_uses_relation_label_for_dataset_construction":true,"support_selection_label_visible_to_selector":false,"valid_for_p86_synthetic_few_shot_selector_experiment":true,"valid_for_final_semantic_geometry_evidence":false,"valid_for_bridge_evidence":false},"audit_label_evaluation_only":{"target_relation_label":"translate_x","domain":"p70a_vector_world","descriptor_id":"p70a_descriptor_train_style_repeated_instances_1","split":"train"},"episode_input_leakage_audit":{"evaluated_target_endpoint_used_for_selector_input":false,"evaluated_target_delta_used_for_selector_input":false,"exact_relation_label_used_for_selector_input":false,"exact_operator_id_used_for_selector_input":false,"audit_metadata_used_for_selector_input":false,"relation_specific_hint_used_for_model_input":false,"support_selection_label_visible_to_selector":false,"diagnostic_pass":true}},{"episode_id":"episode_p81_record_p70a_2","query_record_id":"p81_record_p70a_2","split":"train","domain":"p70a_vector_world","externalized_episode_selector_input":{"query_source_input":{"domain":"p70a_vector_world","query_intensity_hint":0.5,"context_world":"p70a","source_state_summary":{"z_a_dim":3,"z_a_abs_sum":3.5,"z_a_sign_pattern":[1,1,1]}},"externalized_support_episode_context":{"support_pair_count":2,"support_pairs":[{"support_record_id":"p81_record_p70a_0","support_domain":"p70a_vector_world","support_context_world":"p70a","support_result_summary":{},"support_delta_summary":{},"support_pair_observation_available":false,"missing_result_summary":true,"missing_delta_summary":true,"support_source_summary":{"z_a_dim":3,"z_a_abs_sum":1.0,"z_a_sign_pattern":[0,0,1]}},{"support_record_id":"p81_record_p70a_1","support_domain":"p70a_vector_world","support_context_world":"p70a","support_result_summary":{},"support_delta_summary":{},"support_pair_observation_available":false,"missing_result_summary":true,"missing_delta_summary":true,"support_source_summary":{"z_a_dim":3,"z_a_abs_sum":1.0,"z_a_sign_pattern":[0,0,1]}}],"support_invariant_context":{"support_pair_count":2,"support_pair_observation_available_count":0,"result_summary_available_count":0,"delta_summary_available_count":0,"support_delta_abs_sum_mean":0.0,"support_delta_nonzero_count_mean":0.0,"support_invariant_context_available":false,"support_episode_limited_by_missing_result_or_delta_summaries":true},"episode_context_metadata":{"context_source_kind":"externalized_synthetic_support_episode","support_selection_uses_relation_label_for_dataset_construction":true,"support_selection_label_visible_to_selector":false,"valid_for_p86_synthetic_few_shot_selector_experiment":true,"valid_for_final_semantic_geometry_evidence":false,"valid_for_bridge_evidence":false,"external_real_query_source_still_required":true}}},"externalized_episode_manifest":{"episode_id":"episode_p81_record_p70a_2","query_record_id":"p81_record_p70a_2","support_record_ids":["p81_record_p70a_0","p81_record_p70a_1"],"split":"train","domain":"p70a_vector_world","construction_source":"synthetic_task_generator","support_selection_uses_relation_label_for_dataset_construction":true,"support_selection_label_visible_to_selector":false,"valid_for_p86_synthetic_few_shot_selector_experiment":true,"valid_for_final_semantic_geometry_evidence":false,"valid_for_bridge_evidence":false},"audit_label_evaluation_only":{"target_relation_label":"translate_x","domain":"p70a_vector_world","descriptor_id":"p70a_descriptor_train_style_repeated_instances_2","split":"train"},"episode_input_leakage_audit":{"evaluated_target_endpoint_used_for_selector_input":false,"evaluated_target_delta_used_for_selector_input":false,"exact_relation_label_used_for_selector_input":false,"exact_operator_id_used_for_selector_input":false,"audit_metadata_used_for_selector_input":false,"relation_specific_hint_used_for_model_input":false,"support_selection_label_visible_to_selector":false,"diagnostic_pass":true}}],"sample_record_count":3,"sanity_summary":{"source_contracts_validated":true,"p84_query_observation_contract_preserved":true,"p84_no_final_selector_evidence_preserved":true,"p84_real_external_context_required_preserved":true,"p84_bridge_not_ready_preserved":true,"externalized_query_context_source_defined":true,"support_episode_context_defined":true,"support_pair_result_delta_schema_defined":true,"episode_manifest_schema_defined":true,"support_invariant_context_schema_defined":true,"support_selection_uses_relation_label_for_synthetic_dataset_construction":true,"support_selection_label_visible_to_selector":false,"evaluated_target_endpoint_used_for_selector_input":false,"evaluated_target_delta_used_for_selector_input":false,"exact_relation_label_used_for_selector_input":false,"exact_operator_id_used_for_selector_input":false,"audit_metadata_used_for_selector_input":false,"relation_specific_hint_used_for_model_input":false,"valid_for_p86_synthetic_few_shot_selector_experiment":false,"valid_for_final_semantic_geometry_evidence":false,"valid_for_bridge_evidence":false,"external_real_query_source_still_required":true,"model_training_performed":false,"torch_training_performed":false,"new_model_implemented":false,"optimizer_created":false,"checkpoint_written":false,"learned_selector_evidence_present":false,"learned_metric_evidence_present":false,"semantic_metric_ready":false,"bridge_ready":false,"json_safe":true},"json_safe":true,"diagnostic_only":true}
```

## Focused Test Result
- **Command**: `python -m pytest tests/test_phase4_externalized_query_context_support_episode_contract.py tests/test_phase4_p85_externalized_query_context_support_episode_contract_smoke.py -v`
- **Result**: `19 PASSED`

## Scope Gate
Scope gate successfully verified using Git diff:
- `expected_branch = "phase4/p85-externalized-query-context-source-support-episode-contract-no-training-no-bridge"`
- `base_commit = "ab76fe11104a1cf0b1a30f9102ae3de843710551"`

## Limitations
- P85 does not train a model.
- P85 does not prove learned selector evidence or learned metric evidence.
- P85 does not prove semantic geometry.
- P85 does not validate any bridge.
- Synthetic support construction may use labels for dataset construction, but labels are not visible to selector input.
- Final semantic evidence requires an external real query context source.
