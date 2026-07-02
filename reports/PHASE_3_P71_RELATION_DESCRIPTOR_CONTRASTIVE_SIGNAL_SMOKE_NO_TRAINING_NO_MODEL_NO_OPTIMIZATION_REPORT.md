# PHASE 3 P71 Relation Descriptor + Contrastive Signal Smoke Report

## 1. Problem Framing
P71 is the first relation-signal smoke test after the baselines and testbeds are established. It evaluates whether deterministic descriptors constructed from endpoint coordinates (without knowing relation type labels) contain sufficient separability to identify transformations.

## 2. Boundaries and Scope
We strictly respect the Phase 3 boundary requirements:
- No training loop is implemented.
- No model architecture or neural networks are instantiated.
- No optimization step or gradient is computed.
- No PyTorch, NumPy, Pandas, or Scikit-Learn is imported.
- No stochastic random processes are used.
- No Schrödinger Bridge or Geometric Schrödinger Bridge is built.
- No learned metric is allowed.
- No semantic geometry, meaning, operator identity, transfer, composition, or generation success claims are made.

## 3. Why Deterministic Descriptors First
Before we train a neural relation encoder, we must establish whether the underlying transition math exposes a separable contrastive signal. Building hand-engineered descriptors from endpoints first allows us to prove that the testbeds (P70A and P70B) are mathematically structured to support relation separation. It also verifies that our negative controls (label permutations and mismatched pairs) are correctly set up.

## 4. Label Discipline
Relation type labels are **never** used during descriptor construction. The descriptor functions take only endpoints (e.g. `z_a` and `z_b`) as inputs. Labels are used exclusively by the evaluator to partition positive (same relation) and negative (different relation) pairs for contrastive scoring.

## 5. Descriptor Views
We define three deterministic descriptor views:
1. **p70a_vector_delta_descriptor**: A 9-dimensional vector containing coordinate delta, absolute coordinate delta, and L2-normalized direction.
2. **p70b_parameter_delta_descriptor**: A 15-dimensional vector containing parameter delta, absolute parameter delta, and L2-normalized direction.
3. **p70b_series_summary_delta_descriptor**: An 18-dimensional vector containing delta, absolute delta, and L2-normalized direction over six time-series summary stats (mean, min, max, energy, first value, last value).

## 6. Contrastive Metrics
We evaluate:
- **mean_positive_l2 / mean_negative_l2**: Average distance between descriptors of the same / different relation.
- **separation_margin**: Difference between mean negative and mean positive L2 distances.
- **nearest_same_relation_top1_accuracy**: Top-1 nearest-neighbor accuracy.
- **different_relation_collision_count**: Number of instances where different relations map to descriptors within `1e-9` distance.

## 7. Negative Controls
- **Label Permutation Records**: Asserts that cycling relation labels (while leaving endpoints and descriptors unchanged) will cause evaluator failures. The damage is structural only since no model is trained, showing that the evaluator detects mismatch.
- **Mismatched Pair Records**: Ensures mismatched pairings are nondegenerate and will fail to map to valid relation profiles.

## 8. Previous Contract Validation
P71 successfully validates P69, P70A, and P70B contracts:
- P69 baselines exist and torch/numpy/model/bridge flags are false.
- P70A pure numeric testbed exists and sanity summary is correct.
- P70B synthetic time-series testbed exists, is nondegenerate, and has 0 endpoint collisions.

## 9. Limitations
- P71 does not prove a learned relation representation.
- P71 does not prove operator identity, transfer, composition, or semantic geometry.
- The P70A vector descriptor exhibits diagnostic ambiguity (13 collisions between reflect_x and translate_x, which both only modify index 0). This is expected and reported as a warning.

## 10. Results
Final Verdict: `P71_READY_FOR_REVIEW`

## Smoke Output Block
```json
{"allowed_claims":["p71_evaluates_relation_signal_availability","p71_uses_deterministic_relation_descriptors","p71_uses_labels_only_for_evaluation","p71_checks_label_permutation_damage","p71_checks_mismatched_pair_controls","p71_reports_descriptor_collision_warnings","p71_does_not_train_models","p71_does_not_establish_operator_level_evidence"],"bridge_implementation_allowed":false,"contract_version":"phase3_p71_relation_descriptor_contrastive_signal_smoke_contract_v1","contrastive_evaluation_allowed":true,"control_views":["true_relation_labels","assigned_permuted_relation_labels","mismatched_pair_controls"],"descriptor_construction_allowed":true,"descriptor_evaluations":{"p70a_vector_delta_descriptor":{"diagnostic_pass":true,"different_relation_collision_count":13,"mean_negative_l2":2.401737538738568,"mean_positive_l2":0.8033734166777934,"mean_positive_l2_by_type":{"nonlinear_x_from_y":0.7828682220279634,"reflect_x":4.2797292178324335,"scale_s":0.8775700230797331,"translate_x":0.4040610178208843,"translate_y":0.4040610178208843},"nearest_same_relation_top1_accuracy":0.8333333333333334,"negative_pair_count":512,"positive_pair_count":118,"record_count":36,"relation_type_count":5,"same_relation_pair_count_by_type":{"nonlinear_x_from_y":28,"reflect_x":6,"scale_s":28,"translate_x":28,"translate_y":28},"separation_margin":1.598364122060775},"p70b_parameter_delta_descriptor":{"diagnostic_pass":true,"different_relation_collision_count":0,"mean_negative_l2":1.571415842178986,"mean_positive_l2":0.18839344955898732,"mean_positive_l2_by_type":{"change_frequency":0.20203050891044216,"scale_amplitude":0.3863833482912206,"scale_volatility_envelope":0.11111677990074319,"shift_phase":0.20203050891044216,"shift_trend":0.040406101782088436},"nearest_same_relation_top1_accuracy":1.0,"negative_pair_count":640,"positive_pair_count":140,"record_count":40,"relation_type_count":5,"same_relation_pair_count_by_type":{"change_frequency":28,"scale_amplitude":28,"scale_volatility_envelope":28,"shift_phase":28,"shift_trend":28},"separation_margin":1.3830223926199987},"p70b_series_summary_delta_descriptor":{"diagnostic_pass":true,"different_relation_collision_count":0,"mean_negative_l2":28.81516203335223,"mean_positive_l2":16.352464077076775,"mean_positive_l2_by_type":{"change_frequency":5.002677604819881,"scale_amplitude":60.2892320451774,"scale_volatility_envelope":13.315488969162724,"shift_phase":2.3681781233743204,"shift_trend":0.7867436428495556},"nearest_same_relation_top1_accuracy":0.625,"negative_pair_count":640,"positive_pair_count":140,"record_count":40,"relation_type_count":5,"same_relation_pair_count_by_type":{"change_frequency":28,"scale_amplitude":28,"scale_volatility_envelope":28,"shift_phase":28,"shift_trend":28},"separation_margin":12.462697956275456}},"descriptor_views":["p70a_vector_delta_descriptor","p70b_parameter_delta_descriptor","p70b_series_summary_delta_descriptor"],"diagnostic_only":true,"forbidden_claims":["semantic_geometry_is_proven","meaning_is_learned","operator_identity_is_proven","relation_encoder_is_trained","relation_encoder_is_validated","sparse_operator_bank_is_validated","transfer_is_proven","composition_is_proven","bridge_method_is_validated","schrodinger_bridge_creates_meaning","geometric_schrodinger_bridge_creates_meaning","contrastive_smoke_proves_semantics"],"json_safe":true,"label_permutation_evaluation":{"all_records_nondegenerate":true,"all_true_labels_differ_from_assigned":true,"assigned_label_collision_count":0,"damage_expected":true,"label_permutation_record_count":5},"learned_metric_allowed":false,"mismatched_pair_evaluation":{"all_expected_to_fail_relation_identity":true,"all_records_nondegenerate":true,"mismatched_pair_record_count":1},"model_implementation_allowed":false,"neural_encoder_implementation_allowed":false,"numpy_allowed":false,"optimization_allowed":false,"p70a_descriptor_ambiguity_warning":true,"phase":"P71","phase_group":"PHASE_3","phase_name":"Relation Descriptor Contrastive Signal Smoke","primary_empirical_target":"relation_signal_availability_under_contrastive_diagnostics","relation_labels_allowed_for_descriptor_construction":false,"relation_labels_allowed_for_evaluation_only":true,"sanity_summary":{"all_descriptor_views_present":true,"json_safe":true,"label_permutation_records_ready":true,"labels_used_for_descriptor_construction":false,"labels_used_for_evaluation_only":true,"mismatched_pair_records_ready":true,"p70a_descriptor_ambiguity_warning":true,"p70a_vector_descriptor_diagnostic_pass":true,"p70b_parameter_descriptor_strong_pass":true,"p70b_series_summary_descriptor_diagnostic_pass":true,"source_contracts_validated":true},"source_baseline_phase":"P69","source_contracts_validated":true,"source_time_series_testbed_phase":"P70B","source_vector_testbed_phase":"P70A","stochastic_random_allowed":false,"torch_allowed":false,"training_allowed":false,"verdict":"P71_READY_FOR_REVIEW"}
```

## Focused Test Result
- **Command**: `python -m pytest tests/test_phase3_relation_contrastive_signal_smoke.py tests/test_phase3_p71_relation_contrastive_signal_smoke.py -v`
- **Result**: `16 PASSED`

## Scope Gate
Scope gate successfully verified using `enforce_phase_local_scope_gate_or_skip` with:
- `expected_branch = "phase3/p71-relation-descriptor-contrastive-signal-smoke-no-training-no-model-no-optimization"`
- `base_commit = "8e96be8de31075a3d8815d6c7e635b6b9361a9ef"`
