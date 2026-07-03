# PHASE 4 P80 Learned Experiment Authority Contract Report

## 1. Problem Framing
P80 establishes the authority and governance contract for Phase 4 learned experiments. Since Phase 3 deterministic audits proved that hard-ablated queries contain zero selection signal above the null baseline, further progress requires learned models. P80 defines the rules that allow future training, torch, and numpy experiments under strict leakage gates, while keeping the generative bridge blocked.

## 2. P79 Inheritance
P80 inherits and validates all outcomes of the P79 hard-ablation selector evidence gate:
- `hard_ablated_selector_signal_present = False`
- `hard_ablated_selector_beats_null_baseline = False`
- `learned_selector_evidence_present = False`
- `bridge_ready = False`

The source contract validations ensure that we do not base Phase 4 on leaking components.

## 3. New Phase 4 Authority
Phase 4 opens up governed permissions for learned experiments, allowing:
- PyTorch and NumPy imports;
- Supervised/contrastive model training;
- Custom encoder, selector, and metric learning modules;
- Training/Validation/Testing split isolation;
- GPU acceleration;
- Future training checkpoints and logs;
- Random seeds and reproducibility controls.

## 4. What Remains Blocked
The following are strictly blocked in P80 and cannot be claimed or implemented yet:
- Schrödinger Bridge methods (diffusion bridges, score-based bridges);
- Semantic geometry proofs or VAE-latent coordinate meaning claims;
- Generative reconstruction and A+B→C composition claims.

## 5. Required Leakage Gates
All future learned experiment modules must enforce the following leakage gates:
- No target endpoint in selector inputs;
- No target delta in selector inputs;
- No exact relation label pass-through in selector inputs;
- No exact operator ID pass-through in selector inputs;
- No audit metadata access during prediction;
- Strict isolation of the test split from training and model selection.

## 6. Required Baselines
Every learned experiment must compare its performance against:
- P69 null baselines;
- Majority selector baseline;
- Random selector baseline with fixed seed;
- Source-only baseline;
- Intensity-only baseline;
- Hint pass-through baseline;
- Oracle operator upper bound.

## 7. Learned Selector Evidence Criteria
A selector candidate can only claim learned evidence if it:
- Is trained only on the training split;
- Chooses model checkpoints without looking at test split labels;
- Beats the majority, source-only, and intensity-only baselines on the heldout test split;
- Survives relation-specific hint ablation;
- Survives negative control checks.

## 8. Learned Metric Evidence Criteria
A metric candidate can only claim learned evidence if:
- Same-relation pairs are closer than different-relation pairs on heldout test data;
- The metric is not constructed from the target delta;
- The metric survives label permutation and mismatched pair controls;
- The metric transfers to heldout base states and magnitudes.

## 9. Bridge Readiness Criteria
The bridge can only be declared ready when:
- Learned selector evidence is present;
- Learned metric evidence is present;
- Target leakage is completely absent;
- Negative controls are passed;
- The bridge input contract is defined.

Current satisfying status: `bridge_readiness_currently_satisfied = False`
Blocking reasons:
1. `learned_selector_evidence_not_yet_present`
2. `learned_metric_evidence_not_yet_present`
3. `bridge_input_contract_not_yet_defined`

## 10. Results
Final Verdict: `P80_READY_FOR_REVIEW`

## Smoke Output Block
```json
{"phase":"P80","phase_group":"PHASE_4","phase_name":"Learned Experiment Authority Contract","contract_version":"phase4_p80_learned_experiment_authority_contract_v1","source_hard_ablation_phase":"P79","verdict":"P80_READY_FOR_REVIEW","phase4_learned_experiments_allowed":true,"training_allowed":true,"numpy_allowed":true,"torch_allowed":true,"model_implementation_allowed":true,"learned_encoder_allowed":true,"learned_selector_allowed":true,"learned_metric_allowed":true,"optimization_allowed":true,"gpu_usage_allowed":true,"checkpoints_allowed_for_future_phases":true,"bridge_implementation_allowed":false,"bridge_ready":false,"semantic_metric_ready":false,"generation_claims_allowed":false,"semantic_geometry_claims_allowed":false,"target_endpoint_used_for_selector":false,"target_delta_used_for_selector":false,"exact_relation_label_used_for_selector":false,"exact_operator_id_used_for_selector":false,"audit_metadata_used_for_selector":false,"leakage_gates_required":true,"baseline_comparison_required":true,"ablation_controls_required":true,"train_val_test_split_required":true,"reproducibility_required":true,"negative_controls_required":true,"allowed_phase4_experiment_classes":["learned_relation_selector_candidate","learned_relation_encoder_candidate","learned_metric_candidate","contrastive_relation_metric_learning","supervised_selector_training","operator_embedding_probe"],"blocked_until_later_phases":["schrodinger_bridge","geometric_schrodinger_bridge","diffusion_bridge","score_based_bridge","a_plus_b_to_c_generation_claim","semantic_geometry_claim"],"required_leakage_gates":["no_target_endpoint_in_selector_input","no_target_delta_in_selector_input","no_exact_relation_label_pass_through","no_exact_operator_id_pass_through","no_audit_metadata_in_selector_input","train_val_test_split_isolated","test_split_never_used_for_training","baseline_comparison_required","ablation_controls_required","negative_controls_required"],"required_baselines":["p69_null_baselines","majority_selector_baseline","random_selector_baseline_with_fixed_seed","source_only_baseline","intensity_only_baseline","hint_pass_through_baseline","oracle_operator_upper_bound"],"learned_selector_evidence_criteria":["trained_only_on_train_split","selected_model_chosen_without_test_labels","beats_majority_baseline_on_heldout_test","beats_source_only_baseline_on_heldout_test","beats_intensity_only_baseline_on_heldout_test","survives_relation_specific_hint_ablation","survives_negative_controls","reports_confidence_intervals_or_bootstrap_where_possible"],"learned_metric_evidence_criteria":["same_relation_pairs_closer_than_different_relation_pairs_on_test","metric_not_constructed_from_target_delta","metric_survives_label_permutation_control","metric_survives_mismatched_pair_control","metric_transfers_to_heldout_base_states","metric_transfers_to_heldout_magnitudes"],"bridge_readiness_criteria":["learned_selector_evidence_present","learned_metric_evidence_present","target_leakage_absent","negative_controls_passed","oracle_upper_bound_available","bridge_input_contract_defined","generation_claims_still_disallowed_until_bridge_validation"],"source_contracts_validated":true,"p79_hard_ablation_no_signal_preserved":true,"p79_no_learned_selector_evidence_preserved":true,"p79_bridge_not_ready_preserved":true,"phase4_authority_audit":{"phase4_learned_experiments_allowed":true,"training_allowed":true,"numpy_allowed":true,"torch_allowed":true,"model_implementation_allowed":true,"learned_selector_allowed":true,"learned_metric_allowed":true,"bridge_implementation_allowed":false,"bridge_ready":false,"leakage_gates_required":true,"baseline_comparison_required":true,"ablation_controls_required":true,"train_val_test_split_required":true,"negative_controls_required":true,"allowed_phase4_experiment_classes":["learned_relation_selector_candidate","learned_relation_encoder_candidate","learned_metric_candidate","contrastive_relation_metric_learning","supervised_selector_training","operator_embedding_probe"],"blocked_until_later_phases":["schrodinger_bridge","geometric_schrodinger_bridge","diffusion_bridge","score_based_bridge","a_plus_b_to_c_generation_claim","semantic_geometry_claim"],"required_leakage_gates":["no_target_endpoint_in_selector_input","no_target_delta_in_selector_input","no_exact_relation_label_pass_through","no_exact_operator_id_pass_through","no_audit_metadata_in_selector_input","train_val_test_split_isolated","test_split_never_used_for_training","baseline_comparison_required","ablation_controls_required","negative_controls_required"],"required_baselines":["p69_null_baselines","majority_selector_baseline","random_selector_baseline_with_fixed_seed","source_only_baseline","intensity_only_baseline","hint_pass_through_baseline","oracle_operator_upper_bound"],"learned_selector_evidence_criteria":["trained_only_on_train_split","selected_model_chosen_without_test_labels","beats_majority_baseline_on_heldout_test","beats_source_only_baseline_on_heldout_test","beats_intensity_only_baseline_on_heldout_test","survives_relation_specific_hint_ablation","survives_negative_controls","reports_confidence_intervals_or_bootstrap_where_possible"],"learned_metric_evidence_criteria":["same_relation_pairs_closer_than_different_relation_pairs_on_test","metric_not_constructed_from_target_delta","metric_survives_label_permutation_control","metric_survives_mismatched_pair_control","metric_transfers_to_heldout_base_states","metric_transfers_to_heldout_magnitudes"],"bridge_readiness_criteria":["learned_selector_evidence_present","learned_metric_evidence_present","target_leakage_absent","negative_controls_passed","oracle_upper_bound_available","bridge_input_contract_defined","generation_claims_still_disallowed_until_bridge_validation"],"bridge_readiness_currently_satisfied":false,"bridge_blocking_reasons":["learned_selector_evidence_not_yet_present","learned_metric_evidence_not_yet_present","bridge_input_contract_not_yet_defined"],"diagnostic_pass":true},"sanity_summary":{"source_contracts_validated":true,"p79_hard_ablation_no_signal_preserved":true,"p79_no_learned_selector_evidence_preserved":true,"p79_bridge_not_ready_preserved":true,"phase4_learned_experiments_allowed":true,"training_allowed":true,"numpy_allowed":true,"torch_allowed":true,"model_implementation_allowed":true,"learned_selector_allowed":true,"learned_metric_allowed":true,"bridge_implementation_allowed":false,"bridge_ready":false,"semantic_metric_ready":false,"leakage_gates_required":true,"baseline_comparison_required":true,"ablation_controls_required":true,"train_val_test_split_required":true,"negative_controls_required":true,"bridge_readiness_currently_satisfied":false,"json_safe":true},"json_safe":true,"diagnostic_only":true}
```

## Focused Test Result
- **Command**: `python -m pytest tests/test_phase4_learned_experiment_authority_contract.py tests/test_phase4_p80_learned_experiment_authority_contract_smoke.py -v`
- **Result**: `13 PASSED`

## Scope Gate
Scope gate successfully verified using Git diff:
- `expected_branch = "phase4/p80-learned-experiment-authority-contract-with-leakage-gates-no-bridge"`
- `base_commit = "8eec009127d87e6011a8962b7355faa52bcedd3b"`

## Limitations
- P80 does not train a model.
- P80 does not establish learned selector evidence.
- P80 does not establish learned metric evidence.
- P80 does not prove semantic geometry.
- P80 does not validate any bridge method.
- P80 only opens governed authority for future learned experiments.
