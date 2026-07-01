# PHASE_2_P64 Report

## 1. Phase Identifier
`PHASE_2_P64_FC_VAE_HELD_OUT_COMPONENT_ATTRIBUTION_AUDIT_NO_NEW_OPTIMIZATION_NO_DATASET_NO_GENERALIZATION`

## 2. Goal
Implement a component-level held-out attribution audit for the accepted P63 tiny deterministic held-out seed-sensitivity diagnostic. P64 consumes the accepted P63 probe output and audits whether held-out total loss movement is primarily explained by reconstruction movement or by the beta-weighted KL component, without performing any new optimization or model execution.

## 3. Base Branch/Head
- **Base Branch:** `phase2/p63-fc-vae-tiny-held-out-seed-sensitivity-audit-no-dataset-no-generalization`
- **Base Head Commit:** `ac660bd3b6a4d2639021d90538fe756b33575a5e`

## 4. Changed Files
| File | Status | Description |
|------|--------|-------------|
| `src/phase2/fc_vae_held_out_component_attribution_audit.py` | NEW | Component decomposition and attribution logic over P63 summaries |
| `tools/phase2/run_p64_fc_vae_held_out_component_attribution_audit_smoke.py` | NEW | Smoke script outputting compact JSON |
| `tests/test_phase2_fc_vae_held_out_component_attribution_audit.py` | NEW | Unit tests checking parameters, boundaries, and direct-fit absence |
| `tests/test_phase2_p64_fc_vae_held_out_component_attribution_audit_smoke.py` | NEW | Smoke integration test |
| `reports/PHASE_2_P64_FC_VAE_HELD_OUT_COMPONENT_ATTRIBUTION_AUDIT_NO_NEW_OPTIMIZATION_NO_DATASET_NO_GENERALIZATION_REPORT.md` | NEW | This report |

## 5. Why P64 is Component Attribution Diagnostic Only
P64 is a component-level held-out attribution audit over accepted P63 evidence only. It does not perform new optimization and does not prove dataset generalization, transfer, semantic geometry, latent learning, generation, GSB, or VAE success.

P64 audits whether P63 held-out total-loss movement is reconstruction-supported or KL-supported; component attribution is diagnostic and does not establish generalization or transfer.

## 6. Source P63 Evidence Contract
- **Source Contract Version:** `phase2_p63_fc_vae_tiny_held_out_seed_sensitivity_audit_contract_v1`
- **P63 Status:** `fc_vae_tiny_held_out_seed_sensitivity_audit_available_no_dataset_no_generalization`
- **P63 Verdict:** `PASS`
- **P63 Seeds Audit Folds:** 9 runs (3 seeds × 3 folds)

## 7. Component Decomposition Formula
The total loss delta on the held-out target is decomposed into reconstruction and beta-weighted KL components:
$$\Delta \text{Total} \approx \Delta \text{Reconstruction} + \beta \times \Delta \text{KL}$$

Where:
- $\beta = 0.001$
- $\text{Component Residual} = \Delta \text{Total} - (\Delta \text{Reconstruction} + \beta \times \Delta \text{KL})$

## 8. Component Residual Tolerance
- **Tolerance limit:** $1.0 \times 10^{-5}$
- **Max observed residual magnitude:** $2.785 \times 10^{-7}$
- **Pass Status:** Passed (all residuals are well within the tolerance threshold).

## 9. Per-Target Component Attribution Diagnostics
- **Target `bridge_lambda_0_25`:**
  - Seed 62062: `reconstruction_supported`
  - Seed 62162: `reconstruction_supported`
  - Seed 62262: `reconstruction_supported`
  - Total Improved Count: 3/3
  - Reconstruction Supported Count: 3/3
  - KL Opposition (Hurt total): 1 run (Seed 62062, KL increased by $+3.83 \times 10^{-5}$ but total still improved)
- **Target `bridge_lambda_0_5`:**
  - Seed 62062: `reconstruction_supported`
  - Seed 62162: `reconstruction_supported`
  - Seed 62262: `reconstruction_supported`
  - Total Improved Count: 3/3
  - Reconstruction Supported Count: 3/3
  - KL Opposition (Hurt total): 1 run (Seed 62062, KL increased by $+1.92 \times 10^{-4}$ but total still improved)
- **Target `bridge_lambda_0_75` (Hardest Target):**
  - Seed 62062: `reconstruction_supported`
  - Seed 62162: `reconstruction_supported`
  - Seed 62262: `reconstruction_supported`
  - Total Improved Count: 3/3
  - Reconstruction Supported Count: 3/3
  - KL Opposition (Hurt total): 1 run (Seed 62062, KL increased by $+2.87 \times 10^{-4}$ but total still improved)

## 10. Special Note on `bridge_lambda_0_75`
For the hardest target `bridge_lambda_0_75`, all 3 seeds show `"reconstruction_supported"` movement. Under seed `62162` (high loss initialization), the total loss decreased by $-4.0170$ which was dominated by reconstruction loss decrease of $-4.0170$ (with KL component contributing $-1.24 \times 10^{-5}$ helper movement).

## 11. Diagnostic Warnings / KL Opposition
Under seed `62062` (base seed), we observe a minor KL opposition diagnostic across all 3 folds: the reconstruction loss decreases enough to dominate and improve the total loss, while the KL loss increases slightly (contributing positive deltas to total loss). This behaves as a diagnostic signal for latent regularisation balance and does not fail the audit.

## 12. Aggregate Diagnostics
- `source_p63_evidence_valid`: `true`
- `all_component_folds_passed`: `true`
- `all_component_values_finite`: `true`
- `all_component_residuals_within_tolerance`: `true`
- `max_component_residual_abs`: `2.785325050602694e-07`
- `total_improved_count`: `9`
- `reconstruction_improved_count`: `9`
- `kl_increased_count`: `3`
- `kl_decreased_count`: `6`
- `reconstruction_supported_count`: `9`
- `kl_only_count`: `0`
- `mixed_or_degraded_count`: `0`
- `targets_reconstruction_supported_all_seeds`: `["bridge_lambda_0_25", "bridge_lambda_0_5", "bridge_lambda_0_75"]`
- `targets_with_kl_opposition_any_seed`: `["bridge_lambda_0_25", "bridge_lambda_0_5", "bridge_lambda_0_75"]`

## 13. Boundary Flags
- `no_new_optimization = true`
- `no_direct_optimizer_created = true`
- `no_direct_model_created = true`
- `no_direct_torch_import = true`
- `no_dataset = true`
- `no_dataloader = true`
- `no_epoch_loop = true`
- `no_batch_loop = true`
- `no_scheduler = true`
- `no_checkpointing = true`
- `no_generalization_claim = true`
- `no_generation_claim = true`
- `no_gsb_claim = true`
- `no_scientific_conclusion = true`
- `no_latent_learning_claim = true`
- `no_vae_success_claim = true`
- `no_convergence_claim = true`
- `no_semantic_geometry_proof_claim = true`
- `no_transfer_proof_claim = true`
- `no_seed_robustness_claim = true`
- `no_component_proof_claim = true`
- `no_weighted_training = true`
- `uniform_objective_preserved = true`
- `held_out_diagnostic_only = true`
- `component_attribution_diagnostic_only = true`

## 14. Smoke Output
```json
{"aggregate_diagnostics":{"all_component_folds_passed":true,"all_component_residuals_within_tolerance":true,"all_component_values_finite":true,"beta_weighted_kl_helped_count":6,"beta_weighted_kl_hurt_count":3,"bridge_lambda_0_75_component_summary":{"all_component_residuals_within_tolerance":true,"beta_weighted_kl_delta_values_by_seed":[2.8720498085021974e-07,-1.2449294328689575e-05,-7.396936416625977e-08],"beta_weighted_kl_helped_count":2,"beta_weighted_kl_hurt_count":1,"component_attribution_classes_by_seed":["reconstruction_supported","reconstruction_supported","reconstruction_supported"],"component_residual_max_abs":5.152821547227404e-08,"diagnostic_interpretation":"Target bridge_lambda_0_75 showed total improvement of 3 with 3 reconstruction-supported runs.","held_out_target_id":"bridge_lambda_0_75","kl_decreased_count":2,"kl_increased_count":1,"kl_only_count":0,"mixed_or_degraded_count":0,"reconstruction_delta_values_by_seed":[-0.006418466567993164,-4.017014265060425,-0.007493257522583008],"reconstruction_improved_count":3,"reconstruction_supported_count":3,"run_count":3,"seed_count":3,"total_delta_values_by_seed":[-0.006418168544769287,-4.017026662826538,-0.007493317127227783],"total_improved_count":3},"component_attribution_claim":"held_out_component_attribution_audit_only_no_new_optimization_no_dataset_no_generalization","diagnostic_only_no_generalization":true,"kl_decreased_count":6,"kl_increased_count":3,"kl_only_count":0,"max_component_residual_abs":2.785325050602694e-07,"mixed_or_degraded_count":0,"reconstruction_improved_count":9,"reconstruction_supported_count":9,"source_p63_evidence_valid":true,"targets_reconstruction_supported_all_seeds":["bridge_lambda_0_25","bridge_lambda_0_5","bridge_lambda_0_75"],"targets_with_kl_only_any_seed":[],"targets_with_kl_opposition_any_seed":["bridge_lambda_0_25","bridge_lambda_0_5","bridge_lambda_0_75"],"total_degraded_count":0,"total_improved_count":9},"beta":0.001,"component_fold_summaries":[{"beta_weighted_kl_delta_value":3.8385391235351564e-08,"beta_weighted_kl_helped_total":false,"beta_weighted_kl_hurt_total":true,"component_attribution_class":"reconstruction_supported","component_fold_passed":true,"component_residual_abs_value":2.1219253540025185e-08,"component_residual_value":2.1219253540025185e-08,"fold_id":0,"held_out_target_id":"bridge_lambda_0_25","kl_decreased":false,"kl_delta_value":3.838539123535156e-05,"kl_dominates_total_delta":false,"kl_increased":true,"kl_opposes_reconstruction":true,"reconstructed_total_delta_from_components":-0.007277807950973511,"reconstruction_degraded":false,"reconstruction_delta_value":-0.007277846336364746,"reconstruction_dominates_total_delta":true,"reconstruction_improved":true,"seed_value":62062,"total_degraded":false,"total_delta_value":-0.007277786731719971,"total_improved":true,"total_improvement_kl_only":false,"total_improvement_reconstruction_supported":true,"train_target_ids":["bridge_lambda_0_5","bridge_lambda_0_75"]},{"beta_weighted_kl_delta_value":1.927018165588379e-07,"beta_weighted_kl_helped_total":false,"beta_weighted_kl_hurt_total":true,"component_attribution_class":"reconstruction_supported","component_fold_passed":true,"component_residual_abs_value":1.3887882232242743e-08,"component_residual_value":-1.3887882232242743e-08,"fold_id":1,"held_out_target_id":"bridge_lambda_0_5","kl_decreased":false,"kl_delta_value":0.0001927018165588379,"kl_dominates_total_delta":false,"kl_increased":true,"kl_opposes_reconstruction":true,"reconstructed_total_delta_from_components":-0.0070807794928550724,"reconstruction_degraded":false,"reconstruction_delta_value":-0.007080972194671631,"reconstruction_dominates_total_delta":true,"reconstruction_improved":true,"seed_value":62062,"total_degraded":false,"total_delta_value":-0.007080793380737305,"total_improved":true,"total_improvement_kl_only":false,"total_improvement_reconstruction_supported":true,"train_target_ids":["bridge_lambda_0_25","bridge_lambda_0_75"]},{"beta_weighted_kl_delta_value":2.8720498085021974e-07,"beta_weighted_kl_helped_total":false,"beta_weighted_kl_hurt_total":true,"component_attribution_class":"reconstruction_supported","component_fold_passed":true,"component_residual_abs_value":1.0818243027094221e-08,"component_residual_value":1.0818243027094221e-08,"fold_id":2,"held_out_target_id":"bridge_lambda_0_75","kl_decreased":false,"kl_delta_value":0.0002872049808502197,"kl_dominates_total_delta":false,"kl_increased":true,"kl_opposes_reconstruction":true,"reconstructed_total_delta_from_components":-0.006418179363012314,"reconstruction_degraded":false,"reconstruction_delta_value":-0.006418466567993164,"reconstruction_dominates_total_delta":true,"reconstruction_improved":true,"seed_value":62062,"total_degraded":false,"total_delta_value":-0.006418168544769287,"total_improved":true,"total_improvement_kl_only":false,"total_improvement_reconstruction_supported":true,"train_target_ids":["bridge_lambda_0_25","bridge_lambda_0_5"]},{"beta_weighted_kl_delta_value":-1.1943578720092773e-05,"beta_weighted_kl_helped_total":true,"beta_weighted_kl_hurt_total":false,"component_attribution_class":"reconstruction_supported","component_fold_passed":true,"component_residual_abs_value":1.418590547253018e-07,"component_residual_value":1.418590547253018e-07,"fold_id":0,"held_out_target_id":"bridge_lambda_0_25","kl_decreased":true,"kl_delta_value":-0.011943578720092773,"kl_dominates_total_delta":false,"kl_increased":false,"kl_opposes_reconstruction":false,"reconstructed_total_delta_from_components":-3.5989177453517915,"reconstruction_degraded":false,"reconstruction_delta_value":-3.5989058017730713,"reconstruction_dominates_total_delta":true,"reconstruction_improved":true,"seed_value":62162,"total_degraded":false,"total_delta_value":-3.598917603492737,"total_improved":true,"total_improvement_kl_only":false,"total_improvement_reconstruction_supported":true,"train_target_ids":["bridge_lambda_0_5","bridge_lambda_0_75"]},{"beta_weighted_kl_delta_value":-1.17226243019104e-05,"beta_weighted_kl_helped_total":true,"beta_weighted_kl_hurt_total":false,"component_attribution_class":"reconstruction_supported","component_fold_passed":true,"component_residual_abs_value":2.785325050602694e-07,"component_residual_value":2.785325050602694e-07,"fold_id":1,"held_out_target_id":"bridge_lambda_0_5","kl_decreased":true,"kl_delta_value":-0.0117226243019104,"kl_dominates_total_delta":false,"kl_increased":false,"kl_opposes_reconstruction":false,"reconstructed_total_delta_from_components":-3.818659584105015,"reconstruction_degraded":false,"reconstruction_delta_value":-3.818647861480713,"reconstruction_dominates_total_delta":true,"reconstruction_improved":true,"seed_value":62162,"total_degraded":false,"total_delta_value":-3.8186593055725098,"total_improved":true,"total_improvement_kl_only":false,"total_improvement_reconstruction_supported":true,"train_target_ids":["bridge_lambda_0_25","bridge_lambda_0_75"]},{"beta_weighted_kl_delta_value":-1.2449294328689575e-05,"beta_weighted_kl_helped_total":true,"beta_weighted_kl_hurt_total":false,"component_attribution_class":"reconstruction_supported","component_fold_passed":true,"component_residual_abs_value":5.152821547227404e-08,"component_residual_value":5.152821547227404e-08,"fold_id":2,"held_out_target_id":"bridge_lambda_0_75","kl_decreased":true,"kl_delta_value":-0.012449294328689575,"kl_dominates_total_delta":false,"kl_increased":false,"kl_opposes_reconstruction":false,"reconstructed_total_delta_from_components":-4.017026714354754,"reconstruction_degraded":false,"reconstruction_delta_value":-4.017014265060425,"reconstruction_dominates_total_delta":true,"reconstruction_improved":true,"seed_value":62162,"total_degraded":false,"total_delta_value":-4.017026662826538,"total_improved":true,"total_improvement_kl_only":false,"total_improvement_reconstruction_supported":true,"train_target_ids":["bridge_lambda_0_25","bridge_lambda_0_5"]},{"beta_weighted_kl_delta_value":-4.09930944442749e-08,"beta_weighted_kl_helped_total":true,"beta_weighted_kl_hurt_total":false,"component_attribution_class":"reconstruction_supported","component_fold_passed":true,"component_residual_abs_value":4.099309444498267e-08,"component_residual_value":4.099309444498267e-08,"fold_id":0,"held_out_target_id":"bridge_lambda_0_25","kl_decreased":true,"kl_delta_value":-4.09930944442749e-05,"kl_dominates_total_delta":false,"kl_increased":false,"kl_opposes_reconstruction":false,"reconstructed_total_delta_from_components":-0.008434694275259972,"reconstruction_degraded":false,"reconstruction_delta_value":-0.008434653282165527,"reconstruction_dominates_total_delta":true,"reconstruction_improved":true,"seed_value":62262,"total_degraded":false,"total_delta_value":-0.008434653282165527,"total_improved":true,"total_improvement_kl_only":false,"total_improvement_reconstruction_supported":true,"train_target_ids":["bridge_lambda_0_5","bridge_lambda_0_75"]},{"beta_weighted_kl_delta_value":-5.3077936172485356e-08,"beta_weighted_kl_helped_total":true,"beta_weighted_kl_hurt_total":false,"component_attribution_class":"reconstruction_supported","component_fold_passed":true,"component_residual_abs_value":6.526708602391795e-09,"component_residual_value":-6.526708602391795e-09,"fold_id":1,"held_out_target_id":"bridge_lambda_0_5","kl_decreased":true,"kl_delta_value":-5.307793617248535e-05,"kl_dominates_total_delta":false,"kl_increased":false,"kl_opposes_reconstruction":false,"reconstructed_total_delta_from_components":-0.008327596813440323,"reconstruction_degraded":false,"reconstruction_delta_value":-0.00832754373550415,"reconstruction_dominates_total_delta":true,"reconstruction_improved":true,"seed_value":62262,"total_degraded":false,"total_delta_value":-0.008327603340148926,"total_improved":true,"total_improvement_kl_only":false,"total_improvement_reconstruction_supported":true,"train_target_ids":["bridge_lambda_0_25","bridge_lambda_0_75"]},{"beta_weighted_kl_delta_value":-7.396936416625977e-08,"beta_weighted_kl_helped_total":true,"beta_weighted_kl_hurt_total":false,"component_attribution_class":"reconstruction_supported","component_fold_passed":true,"component_residual_abs_value":1.4364719390543013e-08,"component_residual_value":1.4364719390543013e-08,"fold_id":2,"held_out_target_id":"bridge_lambda_0_75","kl_decreased":true,"kl_delta_value":-7.396936416625977e-05,"kl_dominates_total_delta":false,"kl_increased":false,"kl_opposes_reconstruction":false,"reconstructed_total_delta_from_components":-0.007493331491947174,"reconstruction_degraded":false,"reconstruction_delta_value":-0.007493257522583008,"reconstruction_dominates_total_delta":true,"reconstruction_improved":true,"seed_value":62262,"total_degraded":false,"total_delta_value":-0.007493317127227783,"total_improved":true,"total_improvement_kl_only":false,"total_improvement_reconstruction_supported":true,"train_target_ids":["bridge_lambda_0_25","bridge_lambda_0_5"]}],"component_residual_tolerance":1e-05,"contract_version":"phase2_p64_fc_vae_held_out_component_attribution_audit_contract_v1","expected_seed_fold_runs":9,"held_out_diagnostic_only":true,"kind":"fc_vae_held_out_component_attribution_audit_no_dataset_no_generalization","no_batch_loop":true,"no_checkpointing":true,"no_component_proof_claim":true,"no_convergence_claim":true,"no_dataloader":true,"no_dataset":true,"no_direct_model_created":true,"no_direct_optimizer_created":true,"no_direct_torch_import":true,"no_epoch_loop":true,"no_generalization_claim":true,"no_generation_claim":true,"no_gsb_claim":true,"no_latent_learning_claim":true,"no_new_optimization":true,"no_scheduler":true,"no_scientific_conclusion":true,"no_seed_robustness_claim":true,"no_semantic_geometry_proof_claim":true,"no_transfer_proof_claim":true,"no_vae_success_claim":true,"no_weighted_training":true,"observed_seed_fold_runs":9,"reason":"fc_vae_held_out_component_attribution_audit_probe_success","seed_values":[62062,62162,62262],"source_evidence_phase":"P63","source_p63_learning_rate":0.0001,"source_p63_status":"fc_vae_tiny_held_out_seed_sensitivity_audit_available_no_dataset_no_generalization","source_p63_step_count":20,"source_p63_verdict":"PASS","source_phase":"P64","status":"fc_vae_held_out_component_attribution_audit_available_no_dataset_no_generalization","target_component_diagnostics":{"bridge_lambda_0_25":{"all_component_residuals_within_tolerance":true,"beta_weighted_kl_delta_values_by_seed":[3.8385391235351564e-08,-1.1943578720092773e-05,-4.09930944442749e-08],"beta_weighted_kl_helped_count":2,"beta_weighted_kl_hurt_count":1,"component_attribution_classes_by_seed":["reconstruction_supported","reconstruction_supported","reconstruction_supported"],"component_residual_max_abs":1.418590547253018e-07,"diagnostic_interpretation":"Target bridge_lambda_0_25 showed total improvement of 3 with 3 reconstruction-supported runs.","held_out_target_id":"bridge_lambda_0_25","kl_decreased_count":2,"kl_increased_count":1,"kl_only_count":0,"mixed_or_degraded_count":0,"reconstruction_delta_values_by_seed":[-0.007277846336364746,-3.5989058017730713,-0.008434653282165527],"reconstruction_improved_count":3,"reconstruction_supported_count":3,"run_count":3,"seed_count":3,"total_delta_values_by_seed":[-0.007277786731719971,-3.598917603492737,-0.008434653282165527],"total_improved_count":3},"bridge_lambda_0_5":{"all_component_residuals_within_tolerance":true,"beta_weighted_kl_delta_values_by_seed":[1.927018165588379e-07,-1.17226243019104e-05,-5.3077936172485356e-08],"beta_weighted_kl_helped_count":2,"beta_weighted_kl_hurt_count":1,"component_attribution_classes_by_seed":["reconstruction_supported","reconstruction_supported","reconstruction_supported"],"component_residual_max_abs":2.785325050602694e-07,"diagnostic_interpretation":"Target bridge_lambda_0_5 showed total improvement of 3 with 3 reconstruction-supported runs.","held_out_target_id":"bridge_lambda_0_5","kl_decreased_count":2,"kl_increased_count":1,"kl_only_count":0,"mixed_or_degraded_count":0,"reconstruction_delta_values_by_seed":[-0.007080972194671631,-3.818647861480713,-0.00832754373550415],"reconstruction_improved_count":3,"reconstruction_supported_count":3,"run_count":3,"seed_count":3,"total_delta_values_by_seed":[-0.007080793380737305,-3.8186593055725098,-0.008327603340148926],"total_improved_count":3},"bridge_lambda_0_75":{"all_component_residuals_within_tolerance":true,"beta_weighted_kl_delta_values_by_seed":[2.8720498085021974e-07,-1.2449294328689575e-05,-7.396936416625977e-08],"beta_weighted_kl_helped_count":2,"beta_weighted_kl_hurt_count":1,"component_attribution_classes_by_seed":["reconstruction_supported","reconstruction_supported","reconstruction_supported"],"component_residual_max_abs":5.152821547227404e-08,"diagnostic_interpretation":"Target bridge_lambda_0_75 showed total improvement of 3 with 3 reconstruction-supported runs.","held_out_target_id":"bridge_lambda_0_75","kl_decreased_count":2,"kl_increased_count":1,"kl_only_count":0,"mixed_or_degraded_count":0,"reconstruction_delta_values_by_seed":[-0.006418466567993164,-4.017014265060425,-0.007493257522583008],"reconstruction_improved_count":3,"reconstruction_supported_count":3,"run_count":3,"seed_count":3,"total_delta_values_by_seed":[-0.006418168544769287,-4.017026662826538,-0.007493317127227783],"total_improved_count":3}},"target_ids":["bridge_lambda_0_25","bridge_lambda_0_5","bridge_lambda_0_75"],"uniform_objective_preserved":true,"verdict":"PASS"}
```

## 15. Focused Test Result
`python -m pytest tests/test_phase2_fc_vae_held_out_component_attribution_audit.py tests/test_phase2_p64_fc_vae_held_out_component_attribution_audit_smoke.py -v`
- **Result:** `17 passed` ✅

## 16. Full Curated Test Result
- **Result:** `3108 passed, 42 skipped` ✅

## 17. Scope Gate
The P64 scope gate validates that only the allowed 5 files were changed relative to the accepted P63 head commit `ac660bd3b6a4d2639021d90538fe756b33575a5e`.

## 18. Remaining Blockers
None.

## 19. Final Verdict
`P64_READY_FOR_REVIEW`

“P64 audits whether P63 held-out total-loss movement is reconstruction-supported or KL-supported; component attribution is diagnostic and does not establish generalization or transfer.”
