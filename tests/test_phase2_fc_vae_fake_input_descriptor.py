# tests/test_phase2_fc_vae_fake_input_descriptor.py

import dataclasses
import json
import pathlib
import subprocess
from tests.phase2_scope_gate_utils import enforce_phase_local_scope_gate_or_skip
import pytest
from dataclasses import is_dataclass

from src.phase2.fc_vae_fake_input_descriptor import (
    FC_VAE_FAKE_INPUT_DESCRIPTOR_CONTRACT_VERSION,
    FC_VAE_FAKE_INPUT_DESCRIPTOR_KIND,
    FC_VAE_FAKE_INPUT_DESCRIPTOR_MODULE_NAME,
    FC_VAE_FAKE_INPUT_DESCRIPTOR_STATUS_TORCH_UNAVAILABLE,
    FC_VAE_FAKE_INPUT_DESCRIPTOR_STATUS_BATCH_UNAVAILABLE,
    FC_VAE_FAKE_INPUT_DESCRIPTOR_STATUS_DESCRIPTOR_ONLY,
    SUPPORTED_FC_VAE_FAKE_INPUT_DESCRIPTOR_STATUSES,
    FC_VAE_FAKE_INPUT_DESCRIPTOR_SOURCE_KIND,
    FC_VAE_FAKE_INPUT_DESCRIPTOR_DISTRIBUTION_KIND,
    FC_VAE_FAKE_INPUT_DESCRIPTOR_DTYPE_INTENT,
    DEFAULT_P33_DESCRIPTOR_SEED,
    DEFAULT_P33_MIN_VALUE,
    DEFAULT_P33_MAX_VALUE,
    FCVAEFakeInputDescriptorRequest,
    FCVAEFakeInputDescriptorShape,
    FCVAEFakeInputDescriptorPolicy,
    FCVAEFakeInputDescriptorMetadata,
    FCVAEFakeInputDescriptorResult,
    validate_non_empty_str,
    validate_bool,
    validate_non_negative_int,
    validate_positive_int,
    validate_numeric_value,
    validate_value_range,
    validate_shape_tuple,
    validate_fake_input_descriptor_status,
    assert_no_local_path_leakage,
    assert_no_forbidden_claims,
    validate_fake_input_descriptor_request,
    validate_fake_input_descriptor_shape,
    validate_fake_input_descriptor_policy,
    validate_fake_input_descriptor_metadata,
    validate_fake_input_descriptor_result,
    build_fake_input_descriptor_request_from_p32_default,
    build_fake_input_descriptor_shape,
    build_fake_input_descriptor_policy,
    build_fake_input_descriptor_metadata,
    build_fake_input_descriptor_result,
    run_fake_input_descriptor_probe,
    fake_input_descriptor_request_to_json_dict,
    fake_input_descriptor_shape_to_json_dict,
    fake_input_descriptor_policy_to_json_dict,
    fake_input_descriptor_metadata_to_json_dict,
    fake_input_descriptor_result_to_json_dict,
    compact_fake_input_descriptor_json,
)


# 1. constants exact
def test_p33_01_constants():
    assert FC_VAE_FAKE_INPUT_DESCRIPTOR_CONTRACT_VERSION == "phase2_p33_fake_input_descriptor_contract_v1"
    assert FC_VAE_FAKE_INPUT_DESCRIPTOR_KIND == "deterministic_fake_input_descriptor_no_values"
    assert FC_VAE_FAKE_INPUT_DESCRIPTOR_MODULE_NAME == "src.phase2.fc_vae_fake_input_descriptor"
    assert FC_VAE_FAKE_INPUT_DESCRIPTOR_STATUS_TORCH_UNAVAILABLE == "blocked_torch_unavailable"
    assert FC_VAE_FAKE_INPUT_DESCRIPTOR_STATUS_BATCH_UNAVAILABLE == "blocked_input_batch_unavailable"
    assert FC_VAE_FAKE_INPUT_DESCRIPTOR_STATUS_DESCRIPTOR_ONLY == "descriptor_only_no_values_in_p33"
    assert SUPPORTED_FC_VAE_FAKE_INPUT_DESCRIPTOR_STATUSES == (
        "blocked_torch_unavailable",
        "blocked_input_batch_unavailable",
        "descriptor_only_no_values_in_p33",
    )
    assert FC_VAE_FAKE_INPUT_DESCRIPTOR_SOURCE_KIND == "synthetic_descriptor_only"
    assert FC_VAE_FAKE_INPUT_DESCRIPTOR_DISTRIBUTION_KIND == "bounded_uniform_descriptor"
    assert FC_VAE_FAKE_INPUT_DESCRIPTOR_DTYPE_INTENT == "float32_future_tensor_intent"
    assert DEFAULT_P33_DESCRIPTOR_SEED == 1337
    assert DEFAULT_P33_MIN_VALUE == -1.0
    assert DEFAULT_P33_MAX_VALUE == 1.0


# 2. dataclasses frozen
def test_p33_02_dataclasses_frozen():
    for cls in (
        FCVAEFakeInputDescriptorRequest,
        FCVAEFakeInputDescriptorShape,
        FCVAEFakeInputDescriptorPolicy,
        FCVAEFakeInputDescriptorMetadata,
        FCVAEFakeInputDescriptorResult,
    ):
        assert is_dataclass(cls)
        req = build_fake_input_descriptor_request_from_p32_default()
        with pytest.raises(Exception):
            req.reason = "modified"


# 3. primitive validators accept/reject invalid values
def test_p33_03_primitive_validators():
    validate_non_empty_str("valid", "field")
    with pytest.raises(TypeError):
        validate_non_empty_str(123, "field")
    with pytest.raises(ValueError):
        validate_non_empty_str("", "field")
    with pytest.raises(ValueError):
        validate_non_empty_str(" invalid  ", "field")

    validate_bool(True, "flag")
    validate_bool(False, "flag")
    with pytest.raises(TypeError):
        validate_bool(1, "flag")

    validate_non_negative_int(0, "val")
    validate_non_negative_int(5, "val")
    with pytest.raises(ValueError):
        validate_non_negative_int(-1, "val")
    with pytest.raises(TypeError):
        validate_non_negative_int(True, "val")

    validate_positive_int(1, "val")
    with pytest.raises(ValueError):
        validate_positive_int(0, "val")
    with pytest.raises(TypeError):
        validate_positive_int(False, "val")


# 4. numeric validator accepts int/float and rejects bool/string
def test_p33_04_numeric_validator():
    validate_numeric_value(1, "val")
    validate_numeric_value(-1.5, "val")
    with pytest.raises(TypeError):
        validate_numeric_value(True, "val")
    with pytest.raises(TypeError):
        validate_numeric_value("1.5", "val")


# 5. value range validator rejects min >= max
def test_p33_05_value_range_validator():
    validate_value_range(-1.0, 1.0)
    with pytest.raises(ValueError):
        validate_value_range(1.0, -1.0)
    with pytest.raises(ValueError):
        validate_value_range(0.0, 0.0)


# 6. shape tuple validator accepts exact tuple
def test_p33_06_shape_tuple_accepts():
    validate_shape_tuple((2, 32))


# 7. shape tuple validator rejects list/non-int/wrong length
def test_p33_07_shape_tuple_rejects():
    with pytest.raises(TypeError):
        validate_shape_tuple([2, 32])
    with pytest.raises(ValueError):
        validate_shape_tuple((2, 32, 1))
    with pytest.raises(TypeError):
        validate_shape_tuple((2.5, 32))
    with pytest.raises(TypeError):
        validate_shape_tuple((2, True))
    with pytest.raises(ValueError):
        validate_shape_tuple((2, -5))


# 8. status validator accepts all supported statuses
def test_p33_08_status_validator():
    for status in SUPPORTED_FC_VAE_FAKE_INPUT_DESCRIPTOR_STATUSES:
        validate_fake_input_descriptor_status(status)
    with pytest.raises(ValueError):
        validate_fake_input_descriptor_status("unknown")


# 9. local path and forbidden claim guards work
def test_p33_09_guards():
    assert_no_local_path_leakage("safe string")
    for pattern in ("file:///", "C:/", "C:\\", "/home/", "/Users/"):
        with pytest.raises(ValueError):
            assert_no_local_path_leakage(f"leak {pattern}")

    assert_no_forbidden_claims("safe conclusion")
    # Allowed
    assert_no_forbidden_claims("no_scientific_conclusion")
    assert_no_forbidden_claims("no_final_comparison")
    assert_no_forbidden_claims("descriptor_only_no_values_in_p33")
    # Forbidden
    for claim in ("model works", "scientific success", "solved", "best", "winner", "production ready", "state of the art"):
        with pytest.raises(ValueError):
            assert_no_forbidden_claims(f"our {claim} claim")


# 10. request builder returns seed=1337, min=-1.0, max=1.0
def test_p33_10_request_builder_defaults():
    req = build_fake_input_descriptor_request_from_p32_default()
    assert req.descriptor_seed == 1337
    assert req.min_value == -1.0
    assert req.max_value == 1.0


# 11. request builder returns batch_size=2 and input_flat_dim=32
def test_p33_11_request_builder_shape():
    req = build_fake_input_descriptor_request_from_p32_default()
    assert req.batch_size == 2
    assert req.input_flat_dim == 32


# 12. request rejects wrong contract
def test_p33_12_request_wrong_contract():
    req = build_fake_input_descriptor_request_from_p32_default()
    bad = dataclasses.replace(req, contract_version="wrong")
    with pytest.raises(ValueError):
        validate_fake_input_descriptor_request(bad)


# 13. request rejects wrong descriptor kind
def test_p33_13_request_wrong_descriptor_kind():
    req = build_fake_input_descriptor_request_from_p32_default()
    bad = dataclasses.replace(req, descriptor_kind="wrong")
    with pytest.raises(ValueError):
        validate_fake_input_descriptor_request(bad)


# 14. request rejects wrong source kind
def test_p33_14_request_wrong_source_kind():
    req = build_fake_input_descriptor_request_from_p32_default()
    bad = dataclasses.replace(req, source_kind="wrong")
    with pytest.raises(ValueError):
        validate_fake_input_descriptor_request(bad)


# 15. request rejects wrong distribution kind
def test_p33_15_request_wrong_dist_kind():
    req = build_fake_input_descriptor_request_from_p32_default()
    bad = dataclasses.replace(req, distribution_kind="wrong")
    with pytest.raises(ValueError):
        validate_fake_input_descriptor_request(bad)


# 16. request rejects wrong dtype intent
def test_p33_16_request_wrong_dtype_intent():
    req = build_fake_input_descriptor_request_from_p32_default()
    bad = dataclasses.replace(req, dtype_intent="wrong")
    with pytest.raises(ValueError):
        validate_fake_input_descriptor_request(bad)


# 17. request rejects negative seed
def test_p33_17_request_negative_seed():
    req = build_fake_input_descriptor_request_from_p32_default()
    bad = dataclasses.replace(req, descriptor_seed=-5)
    with pytest.raises(ValueError):
        validate_fake_input_descriptor_request(bad)


# 18. request rejects min >= max
def test_p33_18_request_range_invalid():
    req = build_fake_input_descriptor_request_from_p32_default()
    bad = dataclasses.replace(req, min_value=2.0, max_value=1.0)
    with pytest.raises(ValueError):
        validate_fake_input_descriptor_request(bad)


# 19. request rejects allow_rng_execution True
def test_p33_19_request_allow_rng():
    req = build_fake_input_descriptor_request_from_p32_default()
    bad = dataclasses.replace(req, allow_rng_execution_in_p33=True)
    with pytest.raises(ValueError):
        validate_fake_input_descriptor_request(bad)


# 20. request rejects allow_value_materialization True
def test_p33_20_request_allow_val():
    req = build_fake_input_descriptor_request_from_p32_default()
    bad = dataclasses.replace(req, allow_value_materialization_in_p33=True)
    with pytest.raises(ValueError):
        validate_fake_input_descriptor_request(bad)


# 21. request rejects allow_array_materialization True
def test_p33_21_request_allow_arr():
    req = build_fake_input_descriptor_request_from_p32_default()
    bad = dataclasses.replace(req, allow_array_materialization_in_p33=True)
    with pytest.raises(ValueError):
        validate_fake_input_descriptor_request(bad)


# 22. request rejects allow_tensor_materialization True
def test_p33_22_request_allow_tensor():
    req = build_fake_input_descriptor_request_from_p32_default()
    bad = dataclasses.replace(req, allow_tensor_materialization_in_p33=True)
    with pytest.raises(ValueError):
        validate_fake_input_descriptor_request(bad)


# 23. request rejects allow_forward_execution True
def test_p33_23_request_allow_forward():
    req = build_fake_input_descriptor_request_from_p32_default()
    bad = dataclasses.replace(req, allow_forward_execution_in_p33=True)
    with pytest.raises(ValueError):
        validate_fake_input_descriptor_request(bad)


# 24. shape builder validates rank=2 and tuple=(2,32)
def test_p33_24_shape_builder():
    req = build_fake_input_descriptor_request_from_p32_default()
    shape = build_fake_input_descriptor_shape(req)
    validate_fake_input_descriptor_shape(shape)
    assert shape.rank == 2
    assert shape.shape_tuple == (2, 32)


# 25. shape rejects rank != 2
def test_p33_25_shape_rank_invalid():
    req = build_fake_input_descriptor_request_from_p32_default()
    shape = build_fake_input_descriptor_shape(req)
    bad = dataclasses.replace(shape, rank=3)
    with pytest.raises(ValueError):
        validate_fake_input_descriptor_shape(bad)


# 26. shape rejects wrong tuple
def test_p33_26_shape_tuple_invalid():
    req = build_fake_input_descriptor_request_from_p32_default()
    shape = build_fake_input_descriptor_shape(req)
    bad = dataclasses.replace(shape, shape_tuple=(3, 32))
    with pytest.raises(ValueError):
        validate_fake_input_descriptor_shape(bad)


# 27. shape rejects source tuple mismatch
def test_p33_27_shape_source_tuple_mismatch():
    req = build_fake_input_descriptor_request_from_p32_default()
    shape = build_fake_input_descriptor_shape(req)
    bad = dataclasses.replace(shape, source_batch_shape_tuple=(3, 32))
    with pytest.raises(ValueError):
        validate_fake_input_descriptor_shape(bad)


# 28. shape rejects shape_matches_p32 False
def test_p33_28_shape_matches_p32_false():
    req = build_fake_input_descriptor_request_from_p32_default()
    shape = build_fake_input_descriptor_shape(req)
    bad = dataclasses.replace(shape, shape_matches_p32=False)
    with pytest.raises(ValueError):
        validate_fake_input_descriptor_shape(bad)


# 29. policy builder validates descriptor fields
def test_p33_29_policy_builder():
    req = build_fake_input_descriptor_request_from_p32_default()
    pol = build_fake_input_descriptor_policy(req)
    validate_fake_input_descriptor_policy(pol)
    assert pol.descriptor_seed == 1337
    assert pol.min_value == -1.0
    assert pol.max_value == 1.0


# 30. policy rejects rng_executed True
def test_p33_30_policy_rng_executed():
    req = build_fake_input_descriptor_request_from_p32_default()
    pol = build_fake_input_descriptor_policy(req)
    bad = dataclasses.replace(pol, rng_executed=True)
    with pytest.raises(ValueError):
        validate_fake_input_descriptor_policy(bad)


# 31. policy rejects values_materialized True
def test_p33_31_policy_values_materialized():
    req = build_fake_input_descriptor_request_from_p32_default()
    pol = build_fake_input_descriptor_policy(req)
    bad = dataclasses.replace(pol, values_materialized=True)
    with pytest.raises(ValueError):
        validate_fake_input_descriptor_policy(bad)


# 32. policy rejects array_materialized True
def test_p33_32_policy_array_materialized():
    req = build_fake_input_descriptor_request_from_p32_default()
    pol = build_fake_input_descriptor_policy(req)
    bad = dataclasses.replace(pol, array_materialized=True)
    with pytest.raises(ValueError):
        validate_fake_input_descriptor_policy(bad)


# 33. policy rejects tensor_materialized True
def test_p33_33_policy_tensor_materialized():
    req = build_fake_input_descriptor_request_from_p32_default()
    pol = build_fake_input_descriptor_policy(req)
    bad = dataclasses.replace(pol, tensor_materialized=True)
    with pytest.raises(ValueError):
        validate_fake_input_descriptor_policy(bad)


# 34. metadata builder validates
def test_p33_34_metadata_builder():
    req = build_fake_input_descriptor_request_from_p32_default()
    meta = build_fake_input_descriptor_metadata(req)
    validate_fake_input_descriptor_metadata(meta)
    assert meta.descriptor_created is True


# 35. metadata rejects descriptor_created False
def test_p33_35_metadata_desc_created_false():
    req = build_fake_input_descriptor_request_from_p32_default()
    meta = build_fake_input_descriptor_metadata(req)
    bad = dataclasses.replace(meta, descriptor_created=False)
    with pytest.raises(ValueError):
        validate_fake_input_descriptor_metadata(bad)


# 36. metadata rejects batch_shape_declared False
def test_p33_36_metadata_batch_shape_declared_false():
    req = build_fake_input_descriptor_request_from_p32_default()
    meta = build_fake_input_descriptor_metadata(req)
    bad = dataclasses.replace(meta, batch_shape_declared=False)
    with pytest.raises(ValueError):
        validate_fake_input_descriptor_metadata(bad)


# 37. metadata rejects rng_execution_attempted True
def test_p33_37_metadata_rng_attempted():
    req = build_fake_input_descriptor_request_from_p32_default()
    meta = build_fake_input_descriptor_metadata(req)
    bad = dataclasses.replace(meta, rng_execution_attempted=True)
    with pytest.raises(ValueError):
        validate_fake_input_descriptor_metadata(bad)


# 38. metadata rejects value_materialization_attempted True
def test_p33_38_metadata_value_attempted():
    req = build_fake_input_descriptor_request_from_p32_default()
    meta = build_fake_input_descriptor_metadata(req)
    bad = dataclasses.replace(meta, value_materialization_attempted=True)
    with pytest.raises(ValueError):
        validate_fake_input_descriptor_metadata(bad)


# 39. metadata rejects array_materialization_attempted True
def test_p33_39_metadata_array_attempted():
    req = build_fake_input_descriptor_request_from_p32_default()
    meta = build_fake_input_descriptor_metadata(req)
    bad = dataclasses.replace(meta, array_materialization_attempted=True)
    with pytest.raises(ValueError):
        validate_fake_input_descriptor_metadata(bad)


# 40. metadata rejects tensor_materialization_attempted True
def test_p33_40_metadata_tensor_attempted():
    req = build_fake_input_descriptor_request_from_p32_default()
    meta = build_fake_input_descriptor_metadata(req)
    bad = dataclasses.replace(meta, tensor_materialization_attempted=True)
    with pytest.raises(ValueError):
        validate_fake_input_descriptor_metadata(bad)


# 41. metadata rejects forward_execution_attempted True
def test_p33_41_metadata_forward_attempted():
    req = build_fake_input_descriptor_request_from_p32_default()
    meta = build_fake_input_descriptor_metadata(req)
    bad = dataclasses.replace(meta, forward_execution_attempted=True)
    with pytest.raises(ValueError):
        validate_fake_input_descriptor_metadata(bad)


# 42. metadata rejects output_generation_attempted True
def test_p33_42_metadata_output_attempted():
    req = build_fake_input_descriptor_request_from_p32_default()
    meta = build_fake_input_descriptor_metadata(req)
    bad = dataclasses.replace(meta, output_generation_attempted=True)
    with pytest.raises(ValueError):
        validate_fake_input_descriptor_metadata(bad)


# 43. result builder validates
def test_p33_43_result_builder():
    req = build_fake_input_descriptor_request_from_p32_default()
    res = build_fake_input_descriptor_result(req)
    validate_fake_input_descriptor_result(res)


# 44. result rejects descriptor_available_in_p33 False
def test_p33_44_result_desc_available_false():
    req = build_fake_input_descriptor_request_from_p32_default()
    res = build_fake_input_descriptor_result(req)
    bad = dataclasses.replace(res, descriptor_available_in_p33=False)
    with pytest.raises(ValueError):
        validate_fake_input_descriptor_result(bad)


# 45. result rejects rng_execution_available True
def test_p33_45_result_rng_available():
    req = build_fake_input_descriptor_request_from_p32_default()
    res = build_fake_input_descriptor_result(req)
    bad = dataclasses.replace(res, rng_execution_available_in_p33=True)
    with pytest.raises(ValueError):
        validate_fake_input_descriptor_result(bad)


# 46. result rejects value_materialization_available True
def test_p33_46_result_value_available():
    req = build_fake_input_descriptor_request_from_p32_default()
    res = build_fake_input_descriptor_result(req)
    bad = dataclasses.replace(res, value_materialization_available_in_p33=True)
    with pytest.raises(ValueError):
        validate_fake_input_descriptor_result(bad)


# 47. result rejects array_materialization_available True
def test_p33_47_result_array_available():
    req = build_fake_input_descriptor_request_from_p32_default()
    res = build_fake_input_descriptor_result(req)
    bad = dataclasses.replace(res, array_materialization_available_in_p33=True)
    with pytest.raises(ValueError):
        validate_fake_input_descriptor_result(bad)


# 48. result rejects tensor_materialization_available True
def test_p33_48_result_tensor_available():
    req = build_fake_input_descriptor_request_from_p32_default()
    res = build_fake_input_descriptor_result(req)
    bad = dataclasses.replace(res, tensor_materialization_available_in_p33=True)
    with pytest.raises(ValueError):
        validate_fake_input_descriptor_result(bad)


# 49. result rejects forward_execution_available True
def test_p33_49_result_forward_available():
    req = build_fake_input_descriptor_request_from_p32_default()
    res = build_fake_input_descriptor_result(req)
    bad = dataclasses.replace(res, forward_execution_available_in_p33=True)
    with pytest.raises(ValueError):
        validate_fake_input_descriptor_result(bad)


# 50. result rejects output_generation_available True
def test_p33_50_result_output_available():
    req = build_fake_input_descriptor_request_from_p32_default()
    res = build_fake_input_descriptor_result(req)
    bad = dataclasses.replace(res, output_generation_available_in_p33=True)
    with pytest.raises(ValueError):
        validate_fake_input_descriptor_result(bad)


# 51. result rejects no_rng_execution False
def test_p33_51_result_no_rng():
    req = build_fake_input_descriptor_request_from_p32_default()
    res = build_fake_input_descriptor_result(req)
    bad = dataclasses.replace(res, no_rng_execution=False)
    with pytest.raises(ValueError):
        validate_fake_input_descriptor_result(bad)


# 52. result rejects no_values_materialized False
def test_p33_52_result_no_values():
    req = build_fake_input_descriptor_request_from_p32_default()
    res = build_fake_input_descriptor_result(req)
    bad = dataclasses.replace(res, no_values_materialized=False)
    with pytest.raises(ValueError):
        validate_fake_input_descriptor_result(bad)


# 53. result rejects no_array_created False
def test_p33_53_result_no_array():
    req = build_fake_input_descriptor_request_from_p32_default()
    res = build_fake_input_descriptor_result(req)
    bad = dataclasses.replace(res, no_array_created=False)
    with pytest.raises(ValueError):
        validate_fake_input_descriptor_result(bad)


# 54. result rejects no_tensor_created False
def test_p33_54_result_no_tensor():
    req = build_fake_input_descriptor_request_from_p32_default()
    res = build_fake_input_descriptor_result(req)
    bad = dataclasses.replace(res, no_tensor_created=False)
    with pytest.raises(ValueError):
        validate_fake_input_descriptor_result(bad)


# 55. result rejects no_forward_execution False
def test_p33_55_result_no_forward():
    req = build_fake_input_descriptor_request_from_p32_default()
    res = build_fake_input_descriptor_result(req)
    bad = dataclasses.replace(res, no_forward_execution=False)
    with pytest.raises(ValueError):
        validate_fake_input_descriptor_result(bad)


# 56. result rejects no_output_generation False
def test_p33_56_result_no_output():
    req = build_fake_input_descriptor_request_from_p32_default()
    res = build_fake_input_descriptor_result(req)
    bad = dataclasses.replace(res, no_output_generation=False)
    with pytest.raises(ValueError):
        validate_fake_input_descriptor_result(bad)


# 57. result rejects no_training_loop False
def test_p33_57_result_no_training():
    req = build_fake_input_descriptor_request_from_p32_default()
    res = build_fake_input_descriptor_result(req)
    bad = dataclasses.replace(res, no_training_loop=False)
    with pytest.raises(ValueError):
        validate_fake_input_descriptor_result(bad)


# 58. result rejects no_optimizer False
def test_p33_58_result_no_optimizer():
    req = build_fake_input_descriptor_request_from_p32_default()
    res = build_fake_input_descriptor_result(req)
    bad = dataclasses.replace(res, no_optimizer=False)
    with pytest.raises(ValueError):
        validate_fake_input_descriptor_result(bad)


# 59. result rejects no_checkpointing False
def test_p33_59_result_no_checkpoint():
    req = build_fake_input_descriptor_request_from_p32_default()
    res = build_fake_input_descriptor_result(req)
    bad = dataclasses.replace(res, no_checkpointing=False)
    with pytest.raises(ValueError):
        validate_fake_input_descriptor_result(bad)


# 60. result rejects no_artifact_generation False
def test_p33_60_result_no_artifact():
    req = build_fake_input_descriptor_request_from_p32_default()
    res = build_fake_input_descriptor_result(req)
    bad = dataclasses.replace(res, no_artifact_generation=False)
    with pytest.raises(ValueError):
        validate_fake_input_descriptor_result(bad)


# 61. result rejects no_final_comparison False
def test_p33_61_result_no_final():
    req = build_fake_input_descriptor_request_from_p32_default()
    res = build_fake_input_descriptor_result(req)
    bad = dataclasses.replace(res, no_final_comparison=False)
    with pytest.raises(ValueError):
        validate_fake_input_descriptor_result(bad)


# 62. result rejects no_scientific_conclusion False
def test_p33_62_result_no_scientific():
    req = build_fake_input_descriptor_request_from_p32_default()
    res = build_fake_input_descriptor_result(req)
    bad = dataclasses.replace(res, no_scientific_conclusion=False)
    with pytest.raises(ValueError):
        validate_fake_input_descriptor_result(bad)


# 63. probe returns valid result
def test_p33_63_probe():
    res = run_fake_input_descriptor_probe()
    assert isinstance(res, FCVAEFakeInputDescriptorResult)
    validate_fake_input_descriptor_result(res)


# 64. serializers return JSON-safe dict
def test_p33_64_serializers():
    req = build_fake_input_descriptor_request_from_p32_default()
    res = build_fake_input_descriptor_result(req)
    d1 = fake_input_descriptor_request_to_json_dict(req)
    d2 = fake_input_descriptor_shape_to_json_dict(res.shape)
    d3 = fake_input_descriptor_policy_to_json_dict(res.policy)
    d4 = fake_input_descriptor_metadata_to_json_dict(res.metadata)
    d5 = fake_input_descriptor_result_to_json_dict(res)
    for d in (d1, d2, d3, d4, d5):
        assert isinstance(d, dict)


# 65. compact JSON sorted/parseable
def test_p33_65_compact_json():
    res = run_fake_input_descriptor_probe()
    js = compact_fake_input_descriptor_json(res)
    assert isinstance(js, str)
    d = json.loads(js)
    assert d["contract_version"] == FC_VAE_FAKE_INPUT_DESCRIPTOR_CONTRACT_VERSION


# 66. serialized result has no local paths
def test_p33_66_serialized_no_local_paths():
    res = run_fake_input_descriptor_probe()
    js = compact_fake_input_descriptor_json(res)
    for pattern in ("file:///", "C:/", "C:\\", "/home/", "/Users/"):
        assert pattern not in js
        assert pattern.lower() not in js.lower()


# 67. serialized result has no forbidden success claims
def test_p33_67_serialized_no_forbidden_claims():
    res = run_fake_input_descriptor_probe()
    js = compact_fake_input_descriptor_json(res)
    normalized = js.lower()
    cleaned = (normalized
               .replace("no_final_comparison", "")
               .replace("no_scientific_conclusion", "")
               .replace("descriptor_only_no_values_in_p33", ""))
    for claim in ("model works", "scientific success", "solved", "best", "winner", "production ready", "state of the art"):
        assert claim not in cleaned


# 68. serialized result contains no tensor/array/value payload
def test_p33_68_serialized_no_payloads():
    res = run_fake_input_descriptor_probe()
    js = compact_fake_input_descriptor_json(res)
    d = json.loads(js)
    assert "tensor" not in d
    assert "array" not in d
    assert "values" not in d


# 69. shape tuples serialized as JSON lists
def test_p33_69_shape_tuples_lists():
    res = run_fake_input_descriptor_probe()
    js = compact_fake_input_descriptor_json(res)
    d = json.loads(js)
    assert isinstance(d["shape"]["shape_tuple"], list)
    assert d["shape"]["shape_tuple"] == [2, 32]
    assert isinstance(d["shape"]["source_batch_shape_tuple"], list)
    assert d["shape"]["source_batch_shape_tuple"] == [2, 32]


# 70. module has no torch import
def test_p33_70_no_torch_import():
    content = pathlib.Path("src/phase2/fc_vae_fake_input_descriptor.py").read_text(encoding="utf-8")
    assert "import torch" not in content
    assert "from torch" not in content


# 71. module has no numpy import
def test_p33_71_no_numpy_import():
    content = pathlib.Path("src/phase2/fc_vae_fake_input_descriptor.py").read_text(encoding="utf-8")
    assert "import numpy" not in content
    assert "from numpy" not in content


# 72. module has no random/secrets import
def test_p33_72_no_random_secrets_import():
    content = pathlib.Path("src/phase2/fc_vae_fake_input_descriptor.py").read_text(encoding="utf-8")
    assert "import random" not in content
    assert "from random" not in content
    assert "import secrets" not in content
    assert "from secrets" not in content


# 73. module has no `def forward`
def test_p33_73_no_def_forward():
    content = pathlib.Path("src/phase2/fc_vae_fake_input_descriptor.py").read_text(encoding="utf-8")
    assert "def forward(" not in content
    assert "def forward " not in content


# 74. module has no `forward(` definition
def test_p33_74_no_forward_call():
    content = pathlib.Path("src/phase2/fc_vae_fake_input_descriptor.py").read_text(encoding="utf-8")
    assert "forward(" not in content


# 75. module has no tensor/array/value constructors
def test_p33_75_no_constructors():
    content = pathlib.Path("src/phase2/fc_vae_fake_input_descriptor.py").read_text(encoding="utf-8")
    for pattern in ("torch.tensor", "np.array", "np.zeros", "torch.zeros", "random.uniform"):
        assert pattern not in content


# 76. module has no train/fit/loss/optimizer/checkpoint functions
def test_p33_76_no_training_functions():
    content = pathlib.Path("src/phase2/fc_vae_fake_input_descriptor.py").read_text(encoding="utf-8")
    for pattern in ("def train", "def fit", "def loss", "def optimizer", "def checkpoint"):
        assert pattern not in content


# 77. smoke script has no torch/numpy/random/secrets import
def test_p33_77_smoke_script_imports():
    content = pathlib.Path("tools/phase2/run_p33_fake_input_descriptor_smoke.py").read_text(encoding="utf-8")
    for pattern in ("torch", "numpy", "random", "secrets"):
        assert f"import {pattern}" not in content
        assert f"from {pattern}" not in content


# 78. `src/phase2/__init__.py` has no monkeypatch symbols
def test_p33_78_init_no_monkeypatch():
    content = pathlib.Path("src/phase2/__init__.py").read_text(encoding="utf-8")
    assert "subprocess.run =" not in content
    assert "_patched_run" not in content
    assert "_original_run" not in content


# 79. P33-owned scope gate compares against accepted P32 branch
def test_p33_79_scope_gate():
    """Phase-local scope gate for P33. Skips on non-P33 branches."""
    allowed = {
        "src/phase2/fc_vae_fake_input_descriptor.py",
        "src/phase2/__init__.py",
        "tests/test_phase2_fc_vae_fake_input_descriptor.py",
        "tools/phase2/run_p33_fake_input_descriptor_smoke.py",
        "tests/test_phase2_p33_fake_input_descriptor_smoke.py",
        "reports/PHASE_2_P33_DETERMINISTIC_FAKE_INPUT_DESCRIPTOR_NO_VALUES_REPORT.md",
    }
    enforce_phase_local_scope_gate_or_skip(
        expected_branch="phase2/p33-deterministic-fake-input-descriptor-no-values",
        base_commit="phase2/p32-forward-input-batch-contract-no-tensor",
        allowed_files=allowed,
        phase_label="P33",
    )
