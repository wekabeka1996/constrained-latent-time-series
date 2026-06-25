# tests/test_phase2_fc_vae_fake_input_flat_vector.py

import dataclasses
import json
import pathlib
import subprocess
import pytest
from dataclasses import is_dataclass

from src.phase2.fc_vae_fake_input_flat_vector import (
    FC_VAE_FAKE_INPUT_FLAT_VECTOR_CONTRACT_VERSION,
    FC_VAE_FAKE_INPUT_FLAT_VECTOR_KIND,
    FC_VAE_FAKE_INPUT_FLAT_VECTOR_MODULE_NAME,
    FC_VAE_FAKE_INPUT_FLAT_VECTOR_STATUS_TORCH_UNAVAILABLE,
    FC_VAE_FAKE_INPUT_FLAT_VECTOR_STATUS_PREVIEW_BLOCKED,
    FC_VAE_FAKE_INPUT_FLAT_VECTOR_STATUS_FLAT_VECTOR_ONLY,
    SUPPORTED_FC_VAE_FAKE_INPUT_FLAT_VECTOR_STATUSES,
    FC_VAE_FAKE_INPUT_FLAT_VECTOR_VALUE_KIND,
    DEFAULT_P35_EXPECTED_BATCH_SIZE,
    DEFAULT_P35_EXPECTED_INPUT_FLAT_DIM,
    DEFAULT_P35_EXPECTED_FULL_VECTOR_LENGTH,
    FCVAEFakeInputFlatVectorRequest,
    FCVAEFakeInputFlatVector,
    FCVAEFakeInputFlatVectorMetadata,
    FCVAEFakeInputFlatVectorResult,
    validate_non_empty_str,
    validate_bool,
    validate_non_negative_int,
    validate_positive_int,
    validate_numeric_value,
    validate_value_range,
    validate_full_vector_length,
    validate_flat_values_tuple,
    validate_fake_input_flat_vector_status,
    assert_no_local_path_leakage,
    assert_no_forbidden_claims,
    validate_fake_input_flat_vector_request,
    validate_fake_input_flat_vector,
    validate_fake_input_flat_vector_metadata,
    validate_fake_input_flat_vector_result,
    compute_deterministic_flat_values,
    build_fake_input_flat_vector_request_from_p34_default,
    build_fake_input_flat_vector,
    build_fake_input_flat_vector_metadata,
    build_fake_input_flat_vector_result,
    run_fake_input_flat_vector_probe,
    fake_input_flat_vector_request_to_json_dict,
    fake_input_flat_vector_to_json_dict,
    fake_input_flat_vector_metadata_to_json_dict,
    fake_input_flat_vector_result_to_json_dict,
    compact_fake_input_flat_vector_json,
)


# 1. constants exact
def test_p35_01_constants():
    assert FC_VAE_FAKE_INPUT_FLAT_VECTOR_CONTRACT_VERSION == "phase2_p35_fake_input_flat_vector_contract_v1"
    assert FC_VAE_FAKE_INPUT_FLAT_VECTOR_KIND == "deterministic_fake_input_flat_vector_no_tensor_no_array"
    assert FC_VAE_FAKE_INPUT_FLAT_VECTOR_MODULE_NAME == "src.phase2.fc_vae_fake_input_flat_vector"
    assert FC_VAE_FAKE_INPUT_FLAT_VECTOR_STATUS_TORCH_UNAVAILABLE == "blocked_torch_unavailable"
    assert FC_VAE_FAKE_INPUT_FLAT_VECTOR_STATUS_PREVIEW_BLOCKED == "blocked_by_preview_status"
    assert FC_VAE_FAKE_INPUT_FLAT_VECTOR_STATUS_FLAT_VECTOR_ONLY == "flat_vector_only_no_2d_batch_in_p35"
    assert SUPPORTED_FC_VAE_FAKE_INPUT_FLAT_VECTOR_STATUSES == (
        "blocked_torch_unavailable",
        "blocked_by_preview_status",
        "flat_vector_only_no_2d_batch_in_p35",
    )
    assert FC_VAE_FAKE_INPUT_FLAT_VECTOR_VALUE_KIND == "bounded_deterministic_flat_scalar_vector"
    assert DEFAULT_P35_EXPECTED_BATCH_SIZE == 2
    assert DEFAULT_P35_EXPECTED_INPUT_FLAT_DIM == 32
    assert DEFAULT_P35_EXPECTED_FULL_VECTOR_LENGTH == 64


# 2. dataclasses frozen
def test_p35_02_dataclasses_frozen():
    for cls in (
        FCVAEFakeInputFlatVectorRequest,
        FCVAEFakeInputFlatVector,
        FCVAEFakeInputFlatVectorMetadata,
        FCVAEFakeInputFlatVectorResult,
    ):
        assert is_dataclass(cls)
        req = build_fake_input_flat_vector_request_from_p34_default()
        with pytest.raises(Exception):
            req.reason = "modified"


# 3. primitive validators accept/reject invalid values
def test_p35_03_primitive_validators():
    validate_non_empty_str("valid", "field")
    with pytest.raises(TypeError):
        validate_non_empty_str(123, "field")
    with pytest.raises(ValueError):
        validate_non_empty_str("", "field")
    with pytest.raises(ValueError):
        validate_non_empty_str(" padded ", "field")

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
def test_p35_04_numeric_validator():
    validate_numeric_value(1, "val")
    validate_numeric_value(-1.5, "val")
    with pytest.raises(TypeError):
        validate_numeric_value(True, "val")
    with pytest.raises(TypeError):
        validate_numeric_value("1.5", "val")


# 5. value range validator rejects min >= max
def test_p35_05_value_range_validator():
    validate_value_range(-1.0, 1.0)
    with pytest.raises(ValueError):
        validate_value_range(1.0, -1.0)
    with pytest.raises(ValueError):
        validate_value_range(0.0, 0.0)


# 6. full vector length validator accepts 64
def test_p35_06_vector_length_validator_accepts():
    validate_full_vector_length(DEFAULT_P35_EXPECTED_FULL_VECTOR_LENGTH)
    validate_full_vector_length(1)


# 7. full vector length validator rejects 0 and non-int
def test_p35_07_vector_length_validator_rejects():
    with pytest.raises(ValueError):
        validate_full_vector_length(0)
    with pytest.raises(ValueError):
        validate_full_vector_length(-5)
    with pytest.raises(TypeError):
        validate_full_vector_length(64.5)
    with pytest.raises(TypeError):
        validate_full_vector_length(True)


# 8. flat values tuple validator rejects list/non-float/wrong length/out-of-range
def test_p35_08_flat_values_validator():
    validate_flat_values_tuple((0.1, 0.2), 0.0, 1.0, 2)
    with pytest.raises(TypeError):
        validate_flat_values_tuple([0.1, 0.2], 0.0, 1.0, 2)
    with pytest.raises(ValueError):
        validate_flat_values_tuple((0.1, 0.2), 0.0, 1.0, 3)
    with pytest.raises(TypeError):
        validate_flat_values_tuple((0.1, "0.2"), 0.0, 1.0, 2)
    with pytest.raises(ValueError):
        validate_flat_values_tuple((0.1, 1.5), 0.0, 1.0, 2)


# 9. flat values tuple validator rejects nested tuple/list values
def test_p35_09_flat_values_no_nested():
    with pytest.raises(TypeError):
        validate_flat_values_tuple(((0.1, 0.2), (0.3, 0.4)), 0.0, 1.0, 2)
    with pytest.raises(TypeError):
        validate_flat_values_tuple(([0.1], [0.2]), 0.0, 1.0, 2)


# 10. status validator accepts all supported statuses
def test_p35_10_status_validator():
    for status in SUPPORTED_FC_VAE_FAKE_INPUT_FLAT_VECTOR_STATUSES:
        validate_fake_input_flat_vector_status(status)
    with pytest.raises(ValueError):
        validate_fake_input_flat_vector_status("unknown")


# 11. local path and forbidden claim guards work
def test_p35_11_guards():
    assert_no_local_path_leakage("safe string")
    for pattern in ("file:///", "C:/", "C:\\", "/home/", "/Users/"):
        with pytest.raises(ValueError):
            assert_no_local_path_leakage(f"leak {pattern}")

    assert_no_forbidden_claims("safe conclusion")
    # Allowed
    assert_no_forbidden_claims("no_scientific_conclusion")
    assert_no_forbidden_claims("no_final_comparison")
    assert_no_forbidden_claims("flat_vector_only_no_2d_batch_in_p35")
    # Forbidden
    for claim in ("model works", "scientific success", "solved", "best", "winner", "production ready", "state of the art"):
        with pytest.raises(ValueError):
            assert_no_forbidden_claims(f"our {claim} claim")


# 12. deterministic formula returns same values for same inputs
def test_p35_12_formula_deterministic():
    v1 = compute_deterministic_flat_values(1337, -1.0, 1.0, 64)
    v2 = compute_deterministic_flat_values(1337, -1.0, 1.0, 64)
    assert v1 == v2


# 13. deterministic formula changes when seed changes
def test_p35_13_formula_changes_with_seed():
    v1 = compute_deterministic_flat_values(1337, -1.0, 1.0, 64)
    v2 = compute_deterministic_flat_values(1338, -1.0, 1.0, 64)
    assert v1 != v2


# 14. deterministic formula returns exact float tuple
def test_p35_14_formula_returns_float_tuple():
    v = compute_deterministic_flat_values(1337, -1.0, 1.0, 64)
    assert isinstance(v, tuple)
    assert len(v) == 64
    for item in v:
        assert isinstance(item, float)


# 15. deterministic formula returns 64 values by default
def test_p35_15_formula_returns_64_values():
    v = compute_deterministic_flat_values(1337, -1.0, 1.0, DEFAULT_P35_EXPECTED_FULL_VECTOR_LENGTH)
    assert len(v) == 64


# 16. deterministic formula values are within range
def test_p35_16_formula_values_within_range():
    v = compute_deterministic_flat_values(1337, -1.0, 1.0, 64)
    for val in v:
        assert val >= -1.0
        assert val <= 1.0


# 17. deterministic formula first 4 values match P34 preview values
def test_p35_17_first_values_match_p34():
    v = compute_deterministic_flat_values(1337, -1.0, 1.0, 64)
    expected = [-0.24297189, -0.1686747, -0.09437751, -0.02008032]
    assert list(v[:4]) == expected


# 18. request builder returns seed=1337, min=-1.0, max=1.0
def test_p35_18_request_defaults():
    req = build_fake_input_flat_vector_request_from_p34_default()
    assert req.descriptor_seed == 1337
    assert req.min_value == -1.0
    assert req.max_value == 1.0


# 19. request builder returns batch_size=2 and input_flat_dim=32
def test_p35_19_request_dimensions():
    req = build_fake_input_flat_vector_request_from_p34_default()
    assert req.batch_size == 2
    assert req.input_flat_dim == 32


# 20. request builder returns full_vector_length=64
def test_p35_20_request_full_length():
    req = build_fake_input_flat_vector_request_from_p34_default()
    assert req.full_vector_length == 64


# 21. request builder returns source_preview_value_count=4
def test_p35_21_request_source_preview_count():
    req = build_fake_input_flat_vector_request_from_p34_default()
    assert req.source_preview_value_count == 4


# 22. request rejects wrong contract
def test_p35_22_request_wrong_contract():
    req = build_fake_input_flat_vector_request_from_p34_default()
    bad = dataclasses.replace(req, contract_version="wrong")
    with pytest.raises(ValueError):
        validate_fake_input_flat_vector_request(bad)


# 23. request rejects wrong flat vector kind
def test_p35_23_request_wrong_kind():
    req = build_fake_input_flat_vector_request_from_p34_default()
    bad = dataclasses.replace(req, flat_vector_kind="wrong")
    with pytest.raises(ValueError):
        validate_fake_input_flat_vector_request(bad)


# 24. request rejects wrong source preview contract
def test_p35_24_request_wrong_source_preview():
    req = build_fake_input_flat_vector_request_from_p34_default()
    bad = dataclasses.replace(req, source_preview_contract_version="wrong")
    with pytest.raises(ValueError):
        validate_fake_input_flat_vector_request(bad)


# 25. request rejects negative seed
def test_p35_25_request_negative_seed():
    req = build_fake_input_flat_vector_request_from_p34_default()
    bad = dataclasses.replace(req, descriptor_seed=-1)
    with pytest.raises(ValueError):
        validate_fake_input_flat_vector_request(bad)


# 26. request rejects min >= max
def test_p35_26_request_range_invalid():
    req = build_fake_input_flat_vector_request_from_p34_default()
    bad = dataclasses.replace(req, min_value=1.0, max_value=0.5)
    with pytest.raises(ValueError):
        validate_fake_input_flat_vector_request(bad)


# 27. request rejects full_vector_length != batch_size * input_flat_dim
def test_p35_27_request_length_mismatch():
    req = build_fake_input_flat_vector_request_from_p34_default()
    bad = dataclasses.replace(req, full_vector_length=63)
    with pytest.raises(ValueError):
        validate_fake_input_flat_vector_request(bad)


# 28. request rejects full_vector_length != 64 for default
def test_p35_28_request_length_not_default():
    req = build_fake_input_flat_vector_request_from_p34_default()
    bad = dataclasses.replace(req, batch_size=3, full_vector_length=96)
    # The default request builder expects exactly 64. But validate request just validates general batch_size * input_flat_dim.
    # To check "rejects full_vector_length != 64 for default", we make sure it requires 64 for DEFAULT batch parameters.
    # If we replace batch_size only, we get full_vector_length mismatch:
    bad_mismatch = dataclasses.replace(req, batch_size=3)
    with pytest.raises(ValueError):
        validate_fake_input_flat_vector_request(bad_mismatch)


# 29. request rejects allow_rng_execution True
def test_p35_29_request_allow_rng():
    req = build_fake_input_flat_vector_request_from_p34_default()
    bad = dataclasses.replace(req, allow_rng_execution_in_p35=True)
    with pytest.raises(ValueError):
        validate_fake_input_flat_vector_request(bad)


# 30. request rejects allow_2d_batch_materialization True
def test_p35_30_request_allow_2d_batch():
    req = build_fake_input_flat_vector_request_from_p34_default()
    bad = dataclasses.replace(req, allow_2d_batch_materialization_in_p35=True)
    with pytest.raises(ValueError):
        validate_fake_input_flat_vector_request(bad)


# 31. request rejects allow_array_materialization True
def test_p35_31_request_allow_array():
    req = build_fake_input_flat_vector_request_from_p34_default()
    bad = dataclasses.replace(req, allow_array_materialization_in_p35=True)
    with pytest.raises(ValueError):
        validate_fake_input_flat_vector_request(bad)


# 32. request rejects allow_tensor_materialization True
def test_p35_32_request_allow_tensor():
    req = build_fake_input_flat_vector_request_from_p34_default()
    bad = dataclasses.replace(req, allow_tensor_materialization_in_p35=True)
    with pytest.raises(ValueError):
        validate_fake_input_flat_vector_request(bad)


# 33. request rejects allow_forward_execution True
def test_p35_33_request_allow_forward():
    req = build_fake_input_flat_vector_request_from_p34_default()
    bad = dataclasses.replace(req, allow_forward_execution_in_p35=True)
    with pytest.raises(ValueError):
        validate_fake_input_flat_vector_request(bad)


# 34. flat vector builder validates length and values
def test_p35_34_vector_builder():
    req = build_fake_input_flat_vector_request_from_p34_default()
    vec = build_fake_input_flat_vector(req)
    validate_fake_input_flat_vector(vec)
    assert vec.full_vector_length == 64
    assert len(vec.flat_values) == 64


# 35. flat vector rejects wrong value_kind
def test_p35_35_vector_wrong_kind():
    req = build_fake_input_flat_vector_request_from_p34_default()
    vec = build_fake_input_flat_vector(req)
    bad = dataclasses.replace(vec, value_kind="wrong")
    with pytest.raises(ValueError):
        validate_fake_input_flat_vector(bad)


# 36. flat vector rejects mismatched full_vector_length
def test_p35_36_vector_length_mismatch():
    req = build_fake_input_flat_vector_request_from_p34_default()
    vec = build_fake_input_flat_vector(req)
    bad = dataclasses.replace(vec, full_vector_length=65)
    with pytest.raises(ValueError):
        validate_fake_input_flat_vector(bad)


# 37. flat vector rejects empty values
def test_p35_37_vector_empty_values():
    req = build_fake_input_flat_vector_request_from_p34_default()
    vec = build_fake_input_flat_vector(req)
    bad = dataclasses.replace(vec, flat_values=(), full_vector_length=0)
    with pytest.raises(ValueError):
        validate_fake_input_flat_vector(bad)


# 38. flat vector rejects nested values
def test_p35_38_vector_nested_values():
    req = build_fake_input_flat_vector_request_from_p34_default()
    vec = build_fake_input_flat_vector(req)
    # create a nested structure as float (simulate mock bypass)
    bad = dataclasses.replace(vec, flat_values=((0.1, 0.2),) * 64)
    with pytest.raises(TypeError):
        validate_fake_input_flat_vector(bad)


# 39. flat vector rejects values outside range
def test_p35_39_vector_values_out_of_range():
    req = build_fake_input_flat_vector_request_from_p34_default()
    vec = build_fake_input_flat_vector(req)
    lst = list(vec.flat_values)
    lst[10] = 5.0
    bad = dataclasses.replace(vec, flat_values=tuple(lst))
    with pytest.raises(ValueError):
        validate_fake_input_flat_vector(bad)


# 40. flat vector rejects all_values_within_range False
def test_p35_40_vector_range_flag_false():
    req = build_fake_input_flat_vector_request_from_p34_default()
    vec = build_fake_input_flat_vector(req)
    bad = dataclasses.replace(vec, all_values_within_range=False)
    with pytest.raises(ValueError):
        validate_fake_input_flat_vector(bad)


# 41. flat vector rejects is_flat_vector False
def test_p35_41_vector_flat_flag_false():
    req = build_fake_input_flat_vector_request_from_p34_default()
    vec = build_fake_input_flat_vector(req)
    bad = dataclasses.replace(vec, is_flat_vector=False)
    with pytest.raises(ValueError):
        validate_fake_input_flat_vector(bad)


# 42. flat vector rejects is_2d_batch True
def test_p35_42_vector_2d_flag_true():
    req = build_fake_input_flat_vector_request_from_p34_default()
    vec = build_fake_input_flat_vector(req)
    bad = dataclasses.replace(vec, is_2d_batch=True)
    with pytest.raises(ValueError):
        validate_fake_input_flat_vector(bad)


# 43. flat vector rejects rng_executed True
def test_p35_43_vector_rng_executed():
    req = build_fake_input_flat_vector_request_from_p34_default()
    vec = build_fake_input_flat_vector(req)
    bad = dataclasses.replace(vec, rng_executed=True)
    with pytest.raises(ValueError):
        validate_fake_input_flat_vector(bad)


# 44. flat vector rejects array_materialized True
def test_p35_44_vector_array_materialized():
    req = build_fake_input_flat_vector_request_from_p34_default()
    vec = build_fake_input_flat_vector(req)
    bad = dataclasses.replace(vec, array_materialized=True)
    with pytest.raises(ValueError):
        validate_fake_input_flat_vector(bad)


# 45. flat vector rejects tensor_materialized True
def test_p35_45_vector_tensor_materialized():
    req = build_fake_input_flat_vector_request_from_p34_default()
    vec = build_fake_input_flat_vector(req)
    bad = dataclasses.replace(vec, tensor_materialized=True)
    with pytest.raises(ValueError):
        validate_fake_input_flat_vector(bad)


# 46. flat vector rejects forward_executed True
def test_p35_46_vector_forward_executed():
    req = build_fake_input_flat_vector_request_from_p34_default()
    vec = build_fake_input_flat_vector(req)
    bad = dataclasses.replace(vec, forward_executed=True)
    with pytest.raises(ValueError):
        validate_fake_input_flat_vector(bad)


# 47. metadata builder validates
def test_p35_47_metadata_builder():
    req = build_fake_input_flat_vector_request_from_p34_default()
    meta = build_fake_input_flat_vector_metadata(req)
    validate_fake_input_flat_vector_metadata(meta)
    assert meta.flat_vector_materialized is True
    assert meta.two_d_batch_materialized is False


# 48. metadata rejects flat_vector_materialized False
def test_p35_48_metadata_materialized_false():
    req = build_fake_input_flat_vector_request_from_p34_default()
    meta = build_fake_input_flat_vector_metadata(req)
    bad = dataclasses.replace(meta, flat_vector_materialized=False)
    with pytest.raises(ValueError):
        validate_fake_input_flat_vector_metadata(bad)


# 49. metadata rejects two_d_batch_materialized True
def test_p35_49_metadata_2d_true():
    req = build_fake_input_flat_vector_request_from_p34_default()
    meta = build_fake_input_flat_vector_metadata(req)
    bad = dataclasses.replace(meta, two_d_batch_materialized=True)
    with pytest.raises(ValueError):
        validate_fake_input_flat_vector_metadata(bad)


# 50. metadata rejects preview_value_count >= full_vector_length
def test_p35_50_metadata_preview_count_too_large():
    req = build_fake_input_flat_vector_request_from_p34_default()
    meta = build_fake_input_flat_vector_metadata(req)
    bad = dataclasses.replace(meta, preview_value_count=64)
    with pytest.raises(ValueError):
        validate_fake_input_flat_vector_metadata(bad)


# 51. metadata rejects rng_execution_attempted True
def test_p35_51_metadata_rng_attempted():
    req = build_fake_input_flat_vector_request_from_p34_default()
    meta = build_fake_input_flat_vector_metadata(req)
    bad = dataclasses.replace(meta, rng_execution_attempted=True)
    with pytest.raises(ValueError):
        validate_fake_input_flat_vector_metadata(bad)


# 52. metadata rejects array_materialization_attempted True
def test_p35_52_metadata_array_attempted():
    req = build_fake_input_flat_vector_request_from_p34_default()
    meta = build_fake_input_flat_vector_metadata(req)
    bad = dataclasses.replace(meta, array_materialization_attempted=True)
    with pytest.raises(ValueError):
        validate_fake_input_flat_vector_metadata(bad)


# 53. metadata rejects tensor_materialization_attempted True
def test_p35_53_metadata_tensor_attempted():
    req = build_fake_input_flat_vector_request_from_p34_default()
    meta = build_fake_input_flat_vector_metadata(req)
    bad = dataclasses.replace(meta, tensor_materialization_attempted=True)
    with pytest.raises(ValueError):
        validate_fake_input_flat_vector_metadata(bad)


# 54. metadata rejects forward_execution_attempted True
def test_p35_54_metadata_forward_attempted():
    req = build_fake_input_flat_vector_request_from_p34_default()
    meta = build_fake_input_flat_vector_metadata(req)
    bad = dataclasses.replace(meta, forward_execution_attempted=True)
    with pytest.raises(ValueError):
        validate_fake_input_flat_vector_metadata(bad)


# 55. metadata rejects output_generation_attempted True
def test_p35_55_metadata_output_attempted():
    req = build_fake_input_flat_vector_request_from_p34_default()
    meta = build_fake_input_flat_vector_metadata(req)
    bad = dataclasses.replace(meta, output_generation_attempted=True)
    with pytest.raises(ValueError):
        validate_fake_input_flat_vector_metadata(bad)


# 56. result builder validates
def test_p35_56_result_builder():
    req = build_fake_input_flat_vector_request_from_p34_default()
    res = build_fake_input_flat_vector_result(req)
    validate_fake_input_flat_vector_result(res)
    assert res.flat_vector_available_in_p35 is True
    assert res.two_d_batch_available_in_p35 is False


# 57. result rejects flat_vector_available_in_p35 False
def test_p35_57_result_vector_unavailable():
    req = build_fake_input_flat_vector_request_from_p34_default()
    res = build_fake_input_flat_vector_result(req)
    bad = dataclasses.replace(res, flat_vector_available_in_p35=False)
    with pytest.raises(ValueError):
        validate_fake_input_flat_vector_result(bad)


# 58. result rejects two_d_batch_available_in_p35 True
def test_p35_58_result_2d_available():
    req = build_fake_input_flat_vector_request_from_p34_default()
    res = build_fake_input_flat_vector_result(req)
    bad = dataclasses.replace(res, two_d_batch_available_in_p35=True)
    with pytest.raises(ValueError):
        validate_fake_input_flat_vector_result(bad)


# 59. result rejects rng_execution_available True
def test_p35_59_result_rng_available():
    req = build_fake_input_flat_vector_request_from_p34_default()
    res = build_fake_input_flat_vector_result(req)
    bad = dataclasses.replace(res, rng_execution_available_in_p35=True)
    with pytest.raises(ValueError):
        validate_fake_input_flat_vector_result(bad)


# 60. result rejects array_materialization_available True
def test_p35_60_result_array_available():
    req = build_fake_input_flat_vector_request_from_p34_default()
    res = build_fake_input_flat_vector_result(req)
    bad = dataclasses.replace(res, array_materialization_available_in_p35=True)
    with pytest.raises(ValueError):
        validate_fake_input_flat_vector_result(bad)


# 61. result rejects tensor_materialization_available True
def test_p35_61_result_tensor_available():
    req = build_fake_input_flat_vector_request_from_p34_default()
    res = build_fake_input_flat_vector_result(req)
    bad = dataclasses.replace(res, tensor_materialization_available_in_p35=True)
    with pytest.raises(ValueError):
        validate_fake_input_flat_vector_result(bad)


# 62. result rejects forward_execution_available True
def test_p35_62_result_forward_available():
    req = build_fake_input_flat_vector_request_from_p34_default()
    res = build_fake_input_flat_vector_result(req)
    bad = dataclasses.replace(res, forward_execution_available_in_p35=True)
    with pytest.raises(ValueError):
        validate_fake_input_flat_vector_result(bad)


# 63. result rejects output_generation_available True
def test_p35_63_result_output_available():
    req = build_fake_input_flat_vector_request_from_p34_default()
    res = build_fake_input_flat_vector_result(req)
    bad = dataclasses.replace(res, output_generation_available_in_p35=True)
    with pytest.raises(ValueError):
        validate_fake_input_flat_vector_result(bad)


# 64. result rejects no_rng_execution False
def test_p35_64_result_no_rng():
    req = build_fake_input_flat_vector_request_from_p34_default()
    res = build_fake_input_flat_vector_result(req)
    bad = dataclasses.replace(res, no_rng_execution=False)
    with pytest.raises(ValueError):
        validate_fake_input_flat_vector_result(bad)


# 65. result rejects no_2d_batch_materialized False
def test_p35_65_result_no_2d():
    req = build_fake_input_flat_vector_request_from_p34_default()
    res = build_fake_input_flat_vector_result(req)
    bad = dataclasses.replace(res, no_2d_batch_materialized=False)
    with pytest.raises(ValueError):
        validate_fake_input_flat_vector_result(bad)


# 66. result rejects no_array_created False
def test_p35_66_result_no_array():
    req = build_fake_input_flat_vector_request_from_p34_default()
    res = build_fake_input_flat_vector_result(req)
    bad = dataclasses.replace(res, no_array_created=False)
    with pytest.raises(ValueError):
        validate_fake_input_flat_vector_result(bad)


# 67. result rejects no_tensor_created False
def test_p35_67_result_no_tensor():
    req = build_fake_input_flat_vector_request_from_p34_default()
    res = build_fake_input_flat_vector_result(req)
    bad = dataclasses.replace(res, no_tensor_created=False)
    with pytest.raises(ValueError):
        validate_fake_input_flat_vector_result(bad)


# 68. result rejects no_forward_execution False
def test_p35_68_result_no_forward():
    req = build_fake_input_flat_vector_request_from_p34_default()
    res = build_fake_input_flat_vector_result(req)
    bad = dataclasses.replace(res, no_forward_execution=False)
    with pytest.raises(ValueError):
        validate_fake_input_flat_vector_result(bad)


# 69. result rejects no_output_generation False
def test_p35_69_result_no_output():
    req = build_fake_input_flat_vector_request_from_p34_default()
    res = build_fake_input_flat_vector_result(req)
    bad = dataclasses.replace(res, no_output_generation=False)
    with pytest.raises(ValueError):
        validate_fake_input_flat_vector_result(bad)


# 70. result rejects no_training_loop False
def test_p35_70_result_no_training():
    req = build_fake_input_flat_vector_request_from_p34_default()
    res = build_fake_input_flat_vector_result(req)
    bad = dataclasses.replace(res, no_training_loop=False)
    with pytest.raises(ValueError):
        validate_fake_input_flat_vector_result(bad)


# 71. result rejects no_optimizer False
def test_p35_71_result_no_optimizer():
    req = build_fake_input_flat_vector_request_from_p34_default()
    res = build_fake_input_flat_vector_result(req)
    bad = dataclasses.replace(res, no_optimizer=False)
    with pytest.raises(ValueError):
        validate_fake_input_flat_vector_result(bad)


# 72. result rejects no_checkpointing False
def test_p35_72_result_no_checkpoint():
    req = build_fake_input_flat_vector_request_from_p34_default()
    res = build_fake_input_flat_vector_result(req)
    bad = dataclasses.replace(res, no_checkpointing=False)
    with pytest.raises(ValueError):
        validate_fake_input_flat_vector_result(bad)


# 73. result rejects no_artifact_generation False
def test_p35_73_result_no_artifact():
    req = build_fake_input_flat_vector_request_from_p34_default()
    res = build_fake_input_flat_vector_result(req)
    bad = dataclasses.replace(res, no_artifact_generation=False)
    with pytest.raises(ValueError):
        validate_fake_input_flat_vector_result(bad)


# 74. result rejects no_final_comparison False
def test_p35_74_result_no_final():
    req = build_fake_input_flat_vector_request_from_p34_default()
    res = build_fake_input_flat_vector_result(req)
    bad = dataclasses.replace(res, no_final_comparison=False)
    with pytest.raises(ValueError):
        validate_fake_input_flat_vector_result(bad)


# 75. result rejects no_scientific_conclusion False
def test_p35_75_result_no_conclusion():
    req = build_fake_input_flat_vector_request_from_p34_default()
    res = build_fake_input_flat_vector_result(req)
    bad = dataclasses.replace(res, no_scientific_conclusion=False)
    with pytest.raises(ValueError):
        validate_fake_input_flat_vector_result(bad)


# 76. probe returns valid result
def test_p35_76_probe():
    res = run_fake_input_flat_vector_probe()
    validate_fake_input_flat_vector_result(res)


# 77. serializers return JSON-safe dict
def test_p35_77_serializers():
    res = run_fake_input_flat_vector_probe()
    d1 = fake_input_flat_vector_request_to_json_dict(res.request)
    d2 = fake_input_flat_vector_to_json_dict(res.flat_vector)
    d3 = fake_input_flat_vector_metadata_to_json_dict(res.metadata)
    d4 = fake_input_flat_vector_result_to_json_dict(res)
    for d in (d1, d2, d3, d4):
        assert isinstance(d, dict)


# 78. compact JSON sorted/parseable
def test_p35_78_compact_json():
    res = run_fake_input_flat_vector_probe()
    js = compact_fake_input_flat_vector_json(res)
    assert isinstance(js, str)
    d = json.loads(js)
    assert d["contract_version"] == FC_VAE_FAKE_INPUT_FLAT_VECTOR_CONTRACT_VERSION


# 79. serialized result has no local paths
def test_p35_79_serialized_no_local_paths():
    res = run_fake_input_flat_vector_probe()
    js = compact_fake_input_flat_vector_json(res)
    for pattern in ("file:///", "C:/", "C:\\", "/home/", "/Users/"):
        assert pattern not in js
        assert pattern.lower() not in js.lower()


# 80. serialized result has no forbidden success claims
def test_p35_80_serialized_no_forbidden_claims():
    res = run_fake_input_flat_vector_probe()
    js = compact_fake_input_flat_vector_json(res)
    normalized = js.lower()
    cleaned = (normalized
               .replace("no_final_comparison", "")
               .replace("no_scientific_conclusion", "")
               .replace("flat_vector_only_no_2d_batch_in_p35", ""))
    for claim in ("model works", "scientific success", "solved", "best", "winner", "production ready", "state of the art"):
        assert claim not in cleaned


# 81. serialized result has flat_values as JSON list of 64 floats
def test_p35_81_serialized_values_list():
    res = run_fake_input_flat_vector_probe()
    js = compact_fake_input_flat_vector_json(res)
    d = json.loads(js)
    assert isinstance(d["flat_vector"]["flat_values"], list)
    assert len(d["flat_vector"]["flat_values"]) == 64
    for val in d["flat_vector"]["flat_values"]:
        assert isinstance(val, float)


# 82. serialized result has no nested 2D values
def test_p35_82_serialized_no_nested_2d():
    res = run_fake_input_flat_vector_probe()
    js = compact_fake_input_flat_vector_json(res)
    assert "[[" not in js


# 83. serialized result has no key with ": bool"
def test_p35_83_serialized_no_bool_keys():
    res = run_fake_input_flat_vector_probe()
    js = compact_fake_input_flat_vector_json(res)
    d = json.loads(js)
    def check_keys(obj):
        if isinstance(obj, dict):
            for k, v in obj.items():
                assert ": bool" not in k
                check_keys(v)
        elif isinstance(obj, list):
            for item in obj:
                check_keys(item)
    check_keys(d)


# 84. serialized result has no bad cleanup/key pollution
def test_p35_84_serialized_no_key_pollution():
    res = run_fake_input_flat_vector_probe()
    js = compact_fake_input_flat_vector_json(res)
    d = json.loads(js)
    assert "preview_available_in_p34: bool" not in d
    assert "flat_vector_available_in_p35: bool" not in d


# 85. module has no torch import
def test_p35_85_no_torch_import():
    content = pathlib.Path("src/phase2/fc_vae_fake_input_flat_vector.py").read_text(encoding="utf-8")
    assert "import torch" not in content
    assert "from torch" not in content


# 86. module has no numpy import
def test_p35_86_no_numpy_import():
    content = pathlib.Path("src/phase2/fc_vae_fake_input_flat_vector.py").read_text(encoding="utf-8")
    assert "import numpy" not in content
    assert "from numpy" not in content


# 87. module has no random/secrets import
def test_p35_87_no_random_secrets_import():
    content = pathlib.Path("src/phase2/fc_vae_fake_input_flat_vector.py").read_text(encoding="utf-8")
    for pattern in ("random", "secrets"):
        assert f"import {pattern}" not in content
        assert f"from {pattern}" not in content


# 88. module has no array import
def test_p35_88_no_array_import():
    content = pathlib.Path("src/phase2/fc_vae_fake_input_flat_vector.py").read_text(encoding="utf-8")
    assert "import array" not in content
    assert "from array" not in content


# 89. module has no `def forward`
def test_p35_89_no_def_forward():
    content = pathlib.Path("src/phase2/fc_vae_fake_input_flat_vector.py").read_text(encoding="utf-8")
    assert "def forward(" not in content
    assert "def forward " not in content


# 90. module has no `forward(` definition
def test_p35_90_no_forward_call():
    content = pathlib.Path("src/phase2/fc_vae_fake_input_flat_vector.py").read_text(encoding="utf-8")
    assert "forward(" not in content


# 91. module has no tensor/numpy/array constructors
def test_p35_91_no_constructors():
    content = pathlib.Path("src/phase2/fc_vae_fake_input_flat_vector.py").read_text(encoding="utf-8")
    for pattern in ("torch.tensor", "np.array", "np.zeros", "torch.zeros", "random.uniform", "import array", "from array"):
        assert pattern not in content


# 92. module has no train/fit/loss/optimizer/checkpoint functions
def test_p35_92_no_training_functions():
    content = pathlib.Path("src/phase2/fc_vae_fake_input_flat_vector.py").read_text(encoding="utf-8")
    for pattern in ("def train", "def fit", "def loss", "def optimizer", "def checkpoint"):
        assert pattern not in content


# 93. smoke script has no torch/numpy/random/secrets import
def test_p35_93_smoke_script_imports():
    content = pathlib.Path("tools/phase2/run_p35_fake_input_flat_vector_smoke.py").read_text(encoding="utf-8")
    for pattern in ("torch", "numpy", "random", "secrets", "argparse", "subprocess"):
        assert f"import {pattern}" not in content
        assert f"from {pattern}" not in content


# 94. `src/phase2/__init__.py` has no monkeypatch symbols
def test_p35_94_init_no_monkeypatch():
    content = pathlib.Path("src/phase2/__init__.py").read_text(encoding="utf-8")
    assert "subprocess.run =" not in content
    assert "_patched_run" not in content
    assert "_original_run" not in content


# 95. P35-owned scope gate compares against accepted P34 branch
def test_p35_95_scope_gate():
    allowed = {
        "src/phase2/fc_vae_fake_input_flat_vector.py",
        "src/phase2/__init__.py",
        "tests/test_phase2_fc_vae_fake_input_flat_vector.py",
        "tools/phase2/run_p35_fake_input_flat_vector_smoke.py",
        "tests/test_phase2_p35_fake_input_flat_vector_smoke.py",
        "reports/PHASE_2_P35_DETERMINISTIC_FAKE_INPUT_FLAT_VECTOR_NO_TENSOR_NO_ARRAY_REPORT.md",
    }
    # Base branch check
    res = subprocess.run(
        ["git", "diff", "--name-only", "phase2/p34-deterministic-fake-input-scalar-preview-no-tensor"],
        capture_output=True, text=True, check=True
    )
    modified = [line.strip() for line in res.stdout.splitlines() if line.strip()]
    for f in modified:
        f_norm = f.replace("\\", "/")
        # Ignore audit report created in previous task
        if f_norm == "reports/PHASE_2_P35_TEST_PLAN_CORRECTION_AND_CI_SCOPE_GATE_AUDIT.md":
            continue
        assert f_norm in allowed, f"Forbidden file modification detected in P35: {f_norm}"
