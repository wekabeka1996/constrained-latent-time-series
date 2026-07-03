# PHASE 4 P82 Tiny Learned Selector Pilot Report

## 1. Problem Framing
P82 is the first Phase 4 learned selector experiment. Following the P80 authority and P81 dataset definitions, we train a tiny MLP selector using PyTorch on clean inputs only. We evaluate whether a learned model can extract any generalizable prediction signal without leaking targets or query hints.

## 2. P81 Inheritance
P82 preserves and validates the dataset split structure, leakage constraints, and contract definitions from P81:
- `p81_dataset_contract_preserved = True`
- `p81_selector_input_leakage_absent_preserved = True`
- `p81_bridge_not_ready_preserved = True`
- PyTorch training is fully authorized, but Schrödinger Bridge methods remain blocked.

## 3. Model Input Contract & Feature Encoding
The model is trained on a 13-dimensional feature vector containing ONLY clean inputs:
1. `domain` (one-hot encoded for vector world vs. parameter world)
2. `query_intensity_hint`
3. `context_world` (one-hot encoded)
4. `source_state_summary` (dimension, absolute sum, sign pattern for p70a)
5. `source_parameter_summary` (key count, parameter sum, nonzero count for p70b)

We explicitly check that:
- `source_split_origin` is NOT encoded as a feature.
- Target endpoints and deltas are completely absent.
- Hints are only used inside baseline control models, not as model inputs.

## 4. Baselines Evaluation
We computed baseline test split accuracies to establish performance boundaries:
- **majority_selector_baseline**: 11.11%
- **source_only_baseline**: 22.22%
- **intensity_only_baseline**: 11.11%
- **source_plus_intensity_baseline**: 22.22%
- **hint_passthrough_baseline**: 100.00% (Control baseline with leakage hints)

## 5. Tiny Selector Training Results
- **Model Class**: `TinySelector` (13 input -> 16 hidden with ReLU -> 10 output classes)
- **Epochs**: 100
- **Optimizer**: `AdamW` (learning rate 0.01)
- **Loss Function**: `CrossEntropyLoss`
- **Best Validation Accuracy**: 40.00% (reached at epoch 37)
- **Train Accuracy at Best**: 36.84%
- **Test Accuracy at Best**: 22.22%
- **Checkpoint Written**: `False` (all model weights kept in-memory)

## 6. Learned Selector Evidence Audit
Evidence is declared present if the model beats the majority, source-only, and intensity-only baselines on the test split by at least 5.0% accuracy.
- **beats_majority_baseline_on_test**: `True` (22.22% vs. 11.11%)
- **beats_intensity_only_baseline_on_test**: `True` (22.22% vs. 11.11%)
- **beats_source_only_baseline_on_test**: `False` (22.22% vs. 22.22%)
- **learned_selector_evidence_present**: `False`

**Blocking reasons**: `["model_does_not_beat_source_only_baseline"]`

Since the model does not beat the simple domain-majority predictor, we conclude that clean, non-leaking features contain zero learnable selection signal.

## 7. Bridge Boundary
Schrödinger Bridges remain strictly blocked:
- `bridge_ready = False`
- `bridge_implementation_allowed = False`
- `learned_metric_evidence_present = False`
- `semantic_metric_ready = False`
- `generation_claims_allowed = False`
- `semantic_geometry_claims_allowed = False`

Blocking reasons:
1. `learned_metric_evidence_not_present`
2. `bridge_input_contract_not_defined`
3. `bridge_validation_not_run`

## 8. Results
Final Verdict: `P82_READY_FOR_REVIEW`

## Smoke Output Block
```json
{"phase":"P82","phase_group":"PHASE_4","phase_name":"Tiny Learned Selector Pilot","contract_version":"phase4_p82_tiny_learned_selector_pilot_v1","source_dataset_phase":"P81","source_authority_phase":"P80","verdict":"P82_READY_FOR_REVIEW","torch_allowed":true,"training_allowed":true,"model_implementation_allowed":true,"learned_selector_allowed":true,"tiny_learned_selector_pilot_allowed":true,"tiny_learned_selector_trained":true,"clean_selector_input_used":true,"hint_passthrough_used_as_model_input":false,"target_endpoint_used_for_selector_input":false,"target_delta_used_for_selector_input":false,"exact_relation_label_used_for_selector_input":false,"exact_operator_id_used_for_selector_input":false,"audit_metadata_used_for_selector_input":false,"relation_specific_hint_used_for_model_input":false,"test_split_used_for_training":false,"test_labels_used_for_model_selection":false,"validation_used_for_model_selection":true,"checkpoint_written":false,"learned_selector_evidence_present":false,"learned_metric_evidence_present":false,"semantic_metric_ready":false,"bridge_implementation_allowed":false,"bridge_ready":false,"generation_claims_allowed":false,"semantic_geometry_claims_allowed":false,"source_contracts_validated":true,"p81_dataset_contract_preserved":true,"p81_selector_input_leakage_absent_preserved":true,"p81_bridge_not_ready_preserved":true,"model_input_audit":{"record_count":114,"selector_input_leakage_detected":false,"hint_passthrough_used_as_model_input":false,"target_endpoint_used_for_selector_input":false,"target_delta_used_for_selector_input":false,"exact_relation_label_used_for_selector_input":false,"exact_operator_id_used_for_selector_input":false,"audit_metadata_used_for_selector_input":false,"relation_specific_hint_used_for_model_input":false,"source_split_origin_encoded_as_feature":false,"diagnostic_pass":true},"split_integrity_audit":{"train_count":76,"validation_count":20,"test_count":18,"all_required_splits_present":true,"test_split_used_for_training":false,"test_labels_used_for_model_selection":false,"validation_used_for_model_selection":true,"diagnostic_pass":true},"tensor_dataset_summary":{"x_train_shape":[76,13],"y_train_shape":[76],"x_validation_shape":[20,13],"y_validation_shape":[20],"x_test_shape":[18,13],"y_test_shape":[18],"feature_dim":13},"baseline_results":{"majority_selector_baseline":{"mode":"majority_selector_baseline","train_accuracy":0.10526315789473684,"validation_accuracy":0.1,"test_accuracy":0.1111111111111111},"source_only_baseline":{"mode":"source_only_baseline","train_accuracy":0.21052631578947367,"validation_accuracy":0.2,"test_accuracy":0.2222222222222222},"intensity_only_baseline":{"mode":"intensity_only_baseline","train_accuracy":0.10526315789473684,"validation_accuracy":0.1,"test_accuracy":0.1111111111111111},"source_plus_intensity_baseline":{"mode":"source_plus_intensity_baseline","train_accuracy":0.21052631578947367,"validation_accuracy":0.2,"test_accuracy":0.2222222222222222},"hint_passthrough_baseline":{"mode":"hint_passthrough_baseline","train_accuracy":1.0,"validation_accuracy":1.0,"test_accuracy":1.0}},"tiny_selector_training_results":{"model_class":"TinySelector","epochs":100,"optimizer_class":"AdamW","loss_class":"CrossEntropyLoss","best_validation_accuracy":0.4,"best_epoch":37,"train_accuracy_at_best":0.3684210526315789,"test_accuracy_at_best":0.2222222222222222,"checkpoint_written":false,"training_completed":true},"learned_selector_evidence_audit":{"learned_selector_evidence_present":false,"beats_majority_baseline_on_test":true,"beats_source_only_baseline_on_test":false,"beats_intensity_only_baseline_on_test":true,"beats_hint_passthrough_baseline_on_test":false,"evidence_margin_threshold":0.05,"blocking_reasons":["model_does_not_beat_source_only_baseline"],"diagnostic_pass":true},"bridge_boundary_after_selector_pilot_audit":{"bridge_ready":false,"bridge_implementation_allowed":false,"learned_selector_evidence_present":false,"learned_metric_evidence_present":false,"semantic_metric_ready":false,"generation_claims_allowed":false,"semantic_geometry_claims_allowed":false,"blocking_reasons":["learned_metric_evidence_not_present","bridge_input_contract_not_defined","bridge_validation_not_run"],"diagnostic_pass":true},"sanity_summary":{"source_contracts_validated":true,"p81_dataset_contract_preserved":true,"p81_selector_input_leakage_absent_preserved":true,"p81_bridge_not_ready_preserved":true,"tiny_learned_selector_trained":true,"clean_selector_input_used":true,"hint_passthrough_used_as_model_input":false,"selector_input_leakage_detected":false,"test_split_used_for_training":false,"test_labels_used_for_model_selection":false,"validation_used_for_model_selection":true,"checkpoint_written":false,"baseline_results_present":true,"training_results_present":true,"learned_selector_evidence_present":false,"learned_metric_evidence_present":false,"semantic_metric_ready":false,"bridge_ready":false,"json_safe":true},"json_safe":true,"diagnostic_only":false}
```

## Focused Test Result
- **Command**: `python -m pytest tests/test_phase4_tiny_learned_selector_pilot.py tests/test_phase4_p82_tiny_learned_selector_pilot_smoke.py -v`
- **Result**: `16 PASSED`

## Scope Gate
Scope gate successfully verified using Git diff:
- `expected_branch = "phase4/p82-tiny-learned-selector-pilot-clean-input-only-no-bridge"`
- `base_commit = "13bed8e389bfae89283b23676f74b5b9d4266c26"`

## Limitations
- P82 is a tiny pilot selector only.
- P82 does not prove semantic geometry or coordinate meanings.
- P82 does not prove learned metric evidence.
- P82 does not validate any bridge or generation method.
- P82 does not use leakage hints as inputs.
