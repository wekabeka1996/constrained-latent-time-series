# PHASE 4 P81 Learned Selector Dataset Contract Report

## 1. Problem Framing
P81 creates the clean dataset contract and deterministic split builder required before training Phase 4 learned selector models. We define selector input schemas, map splits, enforce target and hint isolation gates, and verify that dataset views are ready for P82 training without target endpoints or delta leakages.

## 2. P80 Inheritance
P81 inherits P80 Phase 4 authority rules:
- `phase4_learned_experiments_allowed = True`
- PyTorch and NumPy are permitted under leakage gates.
- P81 itself is a dataset governance step and does not perform model training or bridge operations.

## 3. Dataset Views
The contract defines three main dataset views:
1. **clean_selector_input**: Main input view for learned selectors. Contains only domain, source summaries, query intensity, context, and split origin. Contains zero endpoints, deltas, or hints.
2. **hint_passthrough_baseline_input**: Control baseline view containing relation hints (family, class, axis, parameter group). Not for training main selectors.
3. **label_evaluation_only**: Evaluation view containing target labels used as supervised training targets.

## 4. Selector Input Leakage Gates
The leakage gates verify that `clean_selector_input` contains no target endpoints, target deltas, exact labels, operator IDs, or query hints:
- `target_endpoint_used_for_selector_input = False`
- `target_delta_used_for_selector_input = False`
- `exact_relation_label_used_for_selector_input = False`
- `exact_operator_id_used_for_selector_input = False`
- `relation_specific_hint_used_for_main_selector_input = False`
- `selector_input_leakage_detected = False`

## 5. Split Policy & Dataset Counts
Original splits are deterministically mapped to model splits:
- `train_style_repeated_instances` -> `train` (76 records)
- `heldout_base_state` -> `validation` (20 records)
- `heldout_magnitude` -> `test` (18 records)
- Total record count: 114 records
  - P70A Vector World: 54 records
  - P70B Parameter World: 60 records
  - Unknown holdout count: 0

## 6. Baseline Contract
The baseline contract defines required baselines that future selectors must beat:
- **Must Beat**: `majority_selector_baseline`, `source_only_baseline`, `intensity_only_baseline`.
- **Controls/Upper Bounds**: `hint_passthrough_baseline`, `oracle_operator_upper_bound`.

## 7. What P81 Does Not Do
- `model_training_performed = False`
- `torch_training_performed = False`
- `optimizer_created = False`
- `checkpoint_written = False`
- `learned_selector_evidence_present = False`
- `bridge_implementation_allowed = False`
- `bridge_ready = False`

No model training has occurred; P81 only prepares the dataset contract for P82.

## 8. Results
Final Verdict: `P81_READY_FOR_REVIEW`

## Smoke Output Block
```json
{"phase":"P81","phase_group":"PHASE_4","phase_name":"Learned Selector Dataset Contract and Split Builder","contract_version":"phase4_p81_learned_selector_dataset_contract_v1","source_authority_phase":"P80","source_vector_testbed_phase":"P70A","source_time_series_testbed_phase":"P70B","source_query_contract_phase":"P77","verdict":"P81_READY_FOR_REVIEW","phase4_learned_experiments_allowed":true,"training_allowed_by_phase4_authority":true,"numpy_allowed_by_phase4_authority":true,"torch_allowed_by_phase4_authority":true,"learned_selector_allowed_by_phase4_authority":true,"learned_metric_allowed_by_phase4_authority":true,"model_training_performed":false,"torch_training_performed":false,"optimizer_created":false,"checkpoint_written":false,"bridge_implementation_allowed":false,"bridge_ready":false,"semantic_metric_ready":false,"generation_claims_allowed":false,"semantic_geometry_claims_allowed":false,"learned_selector_dataset_contract_present":true,"selector_train_val_test_split_built":true,"target_label_available_for_supervised_training":true,"learned_selector_evidence_present":false,"target_endpoint_used_for_selector_input":false,"target_delta_used_for_selector_input":false,"exact_relation_label_used_for_selector_input":false,"exact_operator_id_used_for_selector_input":false,"audit_metadata_used_for_selector_input":false,"relation_specific_hint_used_for_main_selector_input":false,"dataset_views":["clean_selector_input","hint_passthrough_baseline_input","label_evaluation_only"],"split_mapping":{"train_style_repeated_instances":"train","heldout_base_state":"validation","heldout_magnitude":"test"},"required_splits":["train","validation","test"],"source_contracts_validated":true,"p80_phase4_authority_preserved":true,"p80_bridge_not_ready_preserved":true,"p80_leakage_gates_preserved":true,"dataset_contract_audit":{"dataset_record_count":114,"p70a_record_count":54,"p70b_record_count":60,"train_count":76,"validation_count":20,"test_count":18,"unknown_holdout_count":0,"all_required_splits_present":true,"selector_input_leakage_detected":false,"label_available_for_supervised_training":true,"hint_passthrough_baseline_present":true,"target_endpoint_used_for_selector_input":false,"target_delta_used_for_selector_input":false,"exact_relation_label_used_for_selector_input":false,"exact_operator_id_used_for_selector_input":false,"audit_metadata_used_for_selector_input":false,"relation_specific_hint_used_for_main_selector_input":false,"train_val_test_split_isolated":true,"test_split_used_for_training":false,"diagnostic_pass":true},"baseline_contract":{"required_baselines":["majority_selector_baseline","source_only_baseline","intensity_only_baseline","source_plus_intensity_baseline","hint_passthrough_baseline","oracle_operator_upper_bound"],"main_selector_must_beat":["majority_selector_baseline","source_only_baseline","intensity_only_baseline"],"hint_passthrough_is_control_not_main_input":true,"oracle_operator_is_upper_bound_not_training_input":true},"sample_dataset_records":[{"dataset_record_id":"p81_record_p70a_0","domain":"p70a_vector_world","source_descriptor_id":"p70a_descriptor_train_style_repeated_instances_0","source_split_origin":"train_style_repeated_instances","split":"train","selector_input":{"domain":"p70a_vector_world","query_intensity_hint":0.5,"context_world":"p70a","source_split_origin":"train_style_repeated_instances","source_state_summary":{"z_a_dim":3,"z_a_abs_sum":1.0,"z_a_sign_pattern":[0,0,1]}},"hint_passthrough_baseline_input":{"query_intensity_hint":0.5,"domain":"p70a_vector_world","relation_family_hint":"axis_shift_family","transformation_class_hint":"shift_like","relation_axis_hint":"x"},"label_evaluation":{"target_relation_label":"translate_x","domain":"p70a_vector_world","descriptor_id":"p70a_descriptor_train_style_repeated_instances_0","split":"train"},"input_leakage_audit":{"target_endpoint_present":false,"target_delta_present":false,"exact_label_present_in_selector_input":false,"exact_operator_id_present_in_selector_input":false,"audit_metadata_present_in_selector_input":false,"relation_specific_hint_present_in_main_selector_input":false,"diagnostic_pass":true}},{"dataset_record_id":"p81_record_p70a_1","domain":"p70a_vector_world","source_descriptor_id":"p70a_descriptor_train_style_repeated_instances_1","source_split_origin":"train_style_repeated_instances","split":"train","selector_input":{"domain":"p70a_vector_world","query_intensity_hint":1.0,"context_world":"p70a","source_split_origin":"train_style_repeated_instances","source_state_summary":{"z_a_dim":3,"z_a_abs_sum":1.0,"z_a_sign_pattern":[0,0,1]}},"hint_passthrough_baseline_input":{"query_intensity_hint":1.0,"domain":"p70a_vector_world","relation_family_hint":"axis_shift_family","transformation_class_hint":"shift_like","relation_axis_hint":"x"},"label_evaluation":{"target_relation_label":"translate_x","domain":"p70a_vector_world","descriptor_id":"p70a_descriptor_train_style_repeated_instances_1","split":"train"},"input_leakage_audit":{"target_endpoint_present":false,"target_delta_present":false,"exact_label_present_in_selector_input":false,"exact_operator_id_present_in_selector_input":false,"audit_metadata_present_in_selector_input":false,"relation_specific_hint_present_in_main_selector_input":false,"diagnostic_pass":true}},{"dataset_record_id":"p81_record_p70a_2","domain":"p70a_vector_world","source_descriptor_id":"p70a_descriptor_train_style_repeated_instances_2","source_split_origin":"train_style_repeated_instances","split":"train","selector_input":{"domain":"p70a_vector_world","query_intensity_hint":0.5,"context_world":"p70a","source_split_origin":"train_style_repeated_instances","source_state_summary":{"z_a_dim":3,"z_a_abs_sum":3.5,"z_a_sign_pattern":[1,1,1]}},"hint_passthrough_baseline_input":{"query_intensity_hint":0.5,"domain":"p70a_vector_world","relation_family_hint":"axis_shift_family","transformation_class_hint":"shift_like","relation_axis_hint":"x"},"label_evaluation":{"target_relation_label":"translate_x","domain":"p70a_vector_world","descriptor_id":"p70a_descriptor_train_style_repeated_instances_2","split":"train"},"input_leakage_audit":{"target_endpoint_present":false,"target_delta_present":false,"exact_label_present_in_selector_input":false,"exact_operator_id_present_in_selector_input":false,"audit_metadata_present_in_selector_input":false,"relation_specific_hint_present_in_main_selector_input":false,"diagnostic_pass":true}}],"sample_record_count":3,"sanity_summary":{"source_contracts_validated":true,"p80_phase4_authority_preserved":true,"p80_bridge_not_ready_preserved":true,"p80_leakage_gates_preserved":true,"learned_selector_dataset_contract_present":true,"selector_train_val_test_split_built":true,"all_required_splits_present":true,"selector_input_leakage_detected":false,"target_label_available_for_supervised_training":true,"hint_passthrough_baseline_present":true,"baseline_contract_present":true,"model_training_performed":false,"torch_training_performed":false,"optimizer_created":false,"checkpoint_written":false,"learned_selector_evidence_present":false,"bridge_implementation_allowed":false,"bridge_ready":false,"json_safe":true},"json_safe":true,"diagnostic_only":true}
```

## Focused Test Result
- **Command**: `python -m pytest tests/test_phase4_learned_selector_dataset_contract.py tests/test_phase4_p81_learned_selector_dataset_contract_smoke.py -v`
- **Result**: `16 PASSED`

## Scope Gate
Scope gate successfully verified using Git diff:
- `expected_branch = "phase4/p81-learned-selector-dataset-contract-and-split-builder-no-model-no-bridge"`
- `base_commit = "0d01a796a0cd075ebdcd0e1a3f94c9b1edc10e2c"`

## Limitations
- P81 only builds the dataset/split contract.
- P81 does not prove learned selector evidence or learned metric evidence.
- P81 does not prove semantic geometry or latent coordinate meaning.
- P81 does not validate any bridge method.
- P81 prepares the P82 learned selector pilot.
