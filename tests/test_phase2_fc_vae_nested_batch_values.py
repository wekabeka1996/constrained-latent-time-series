# tests/test_phase2_fc_vae_nested_batch_values.py

import dataclasses
import json
import pathlib
import subprocess
import pytest
from dataclasses import is_dataclass
from typing import Tuple

from src.phase2.fc_vae_nested_batch_values import (
    FC_VAE_NESTED_BATCH_VALUES_CONTRACT_VERSION,
    FC_VAE_NESTED_BATCH_VALUES_KIND,
    FC_VAE_NESTED_BATCH_VALUES_MODULE_NAME,
    FC_VAE_NESTED_BATCH_VALUES_STATUS_TORCH_UNAVAILABLE,
    FC_VAE_NESTED_BATCH_VALUES_STATUS_BATCH_VIEW_BLOCKED,
    FC_VAE_NESTED_BATCH_VALUES_STATUS_NESTED_VALUES_ONLY,
    SUPPORTED_FC_VAE_NESTED_BATCH_VALUES_STATUSES,
    FC_VAE_NESTED_BATCH_VALUES_ORDER_KIND,
    DEFAULT_P37_BATCH_SIZE,
    DEFAULT_P37_INPUT_FLAT_DIM,
    DEFAULT_P37_FULL_VECTOR_LENGTH,
    DEFAULT_P37_SHAPE_TUPLE,
    FCVAENestedBatchValuesRequest,
    FCVAENestedBatchValues,
    FCVAENestedBatchValuesMetadata,
    FCVAENestedBatchValuesResult,
    validate_non_empty_str,
    validate_bool,
    validate_positive_int,
    validate_shape_tuple,
    validate_full_vector_length,
    validate_flat_values_tuple,
    validate_nested_batch_values_tuple,
    validate_nested_batch_values_status,
    assert_no_local_path_leakage,
    assert_no_forbidden_claims,
    validate_nested_batch_values_request,
    validate_nested_batch_values,
    validate_nested_batch_values_metadata,
    validate_nested_batch_values_result,
    build_nested_tuple_from_flat_values,
    flatten_nested_batch_values,
    build_nested_batch_values_request_from_p36_default,
    build_nested_batch_values,
    build_nested_batch_values_metadata,
    build_nested_batch_values_result,
    run_nested_batch_values_probe,
    nested_batch_values_request_to_json_dict,
    nested_batch_values_to_json_dict,
    nested_batch_values_metadata_to_json_dict,
    nested_batch_values_result_to_json_dict,
    compact_nested_batch_values_json,
)


# 1. Constants exact
def test_p37_01_constants_exact():
    assert FC_VAE_NESTED_BATCH_VALUES_CONTRACT_VERSION == "phase2_p37_nested_batch_values_contract_v1"
    assert FC_VAE_NESTED_BATCH_VALUES_KIND == "controlled_nested_python_tuple_batch_values_no_tensor_no_array"
    assert FC_VAE_NESTED_BATCH_VALUES_MODULE_NAME == "src.phase2.fc_vae_nested_batch_values"
    assert FC_VAE_NESTED_BATCH_VALUES_STATUS_TORCH_UNAVAILABLE == "blocked_torch_unavailable"
    assert FC_VAE_NESTED_BATCH_VALUES_STATUS_BATCH_VIEW_BLOCKED == "blocked_by_batch_view_status"
    assert FC_VAE_NESTED_BATCH_VALUES_STATUS_NESTED_VALUES_ONLY == "nested_values_only_no_tensor_no_forward_in_p37"
    assert SUPPORTED_FC_VAE_NESTED_BATCH_VALUES_STATUSES == (
        "blocked_torch_unavailable",
        "blocked_by_batch_view_status",
        "nested_values_only_no_tensor_no_forward_in_p37",
    )
    assert FC_VAE_NESTED_BATCH_VALUES_ORDER_KIND == "row_major_flat_to_nested_tuple"
    assert DEFAULT_P37_BATCH_SIZE == 2
    assert DEFAULT_P37_INPUT_FLAT_DIM == 32
    assert DEFAULT_P37_FULL_VECTOR_LENGTH == 64
    assert DEFAULT_P37_SHAPE_TUPLE == (2, 32)


# 2. Dataclasses frozen
def test_p37_02_dataclasses_frozen():
    for cls in (
        FCVAENestedBatchValuesRequest,
        FCVAENestedBatchValues,
        FCVAENestedBatchValuesMetadata,
        FCVAENestedBatchValuesResult,
    ):
        assert is_dataclass(cls)
        req = build_nested_batch_values_request_from_p36_default()
        with pytest.raises(Exception):
            req.reason = "modified"


# 3. Validator non-empty str
def test_p37_03_validator_non_empty_str():
    validate_non_empty_str("valid", "name")
    with pytest.raises(TypeError):
        validate_non_empty_str(123, "name")
    with pytest.raises(ValueError):
        validate_non_empty_str("", "name")
    with pytest.raises(ValueError):
        validate_non_empty_str(" padded ", "name")


# 4. Validator bool
def test_p37_04_validator_bool():
    validate_bool(True, "flag")
    validate_bool(False, "flag")
    with pytest.raises(TypeError):
        validate_bool(1, "flag")


# 5. Validator positive int
def test_p37_05_validator_positive_int():
    validate_positive_int(1, "val")
    with pytest.raises(ValueError):
        validate_positive_int(0, "val")
    with pytest.raises(TypeError):
        validate_positive_int(True, "val")


# 6. Validator shape tuple valid
def test_p37_06_validator_shape_tuple_valid():
    validate_shape_tuple((2, 32))


# 7. Validator shape tuple type
def test_p37_07_validator_shape_tuple_type():
    with pytest.raises(TypeError):
        validate_shape_tuple([2, 32])


# 8. Validator shape tuple length
def test_p37_08_validator_shape_tuple_length():
    with pytest.raises(ValueError):
        validate_shape_tuple((2, 32, 1))


# 9. Validator shape tuple elements type
def test_p37_09_validator_shape_tuple_elements_type():
    with pytest.raises(TypeError):
        validate_shape_tuple((2.5, 32))
    with pytest.raises(TypeError):
        validate_shape_tuple((2, True))


# 10. Validator shape tuple elements val
def test_p37_10_validator_shape_tuple_elements_val():
    with pytest.raises(ValueError):
        validate_shape_tuple((0, 32))


# 11. Validator shape tuple values
def test_p37_11_validator_shape_tuple_values():
    with pytest.raises(ValueError):
        validate_shape_tuple((1, 32))
    with pytest.raises(ValueError):
        validate_shape_tuple((2, 16))


# 12. Validator full vector length valid
def test_p37_12_validator_full_vector_length_valid():
    validate_full_vector_length(64)


# 13. Validator full vector length type
def test_p37_13_validator_full_vector_length_type():
    with pytest.raises(TypeError):
        validate_full_vector_length(64.5)
    with pytest.raises(TypeError):
        validate_full_vector_length(True)


# 14. Validator full vector length val
def test_p37_14_validator_full_vector_length_val():
    with pytest.raises(ValueError):
        validate_full_vector_length(65)


# 15. Validator flat values tuple valid
def test_p37_15_validator_flat_values_tuple_valid():
    t = tuple(float(x) for x in range(64))
    validate_flat_values_tuple(t)


# 16. Validator flat values tuple type
def test_p37_16_validator_flat_values_tuple_type():
    with pytest.raises(TypeError):
        validate_flat_values_tuple([float(x) for x in range(64)])


# 17. Validator flat values tuple length
def test_p37_17_validator_flat_values_tuple_length():
    t = tuple(float(x) for x in range(65))
    with pytest.raises(ValueError):
        validate_flat_values_tuple(t)


# 18. Validator flat values tuple element type
def test_p37_18_validator_flat_values_tuple_element_type():
    t = tuple(x for x in range(64))
    with pytest.raises(TypeError):
        validate_flat_values_tuple(t)


# 19. Validator nested batch values tuple valid
def test_p37_19_validator_nested_batch_values_tuple_valid():
    row = tuple(1.0 for _ in range(32))
    validate_nested_batch_values_tuple((row, row))


# 20. Validator nested batch values tuple type
def test_p37_20_validator_nested_batch_values_tuple_type():
    row = tuple(1.0 for _ in range(32))
    with pytest.raises(TypeError):
        validate_nested_batch_values_tuple([row, row])


# 21. Validator nested batch values tuple length
def test_p37_21_validator_nested_batch_values_tuple_length():
    row = tuple(1.0 for _ in range(32))
    with pytest.raises(ValueError):
        validate_nested_batch_values_tuple((row, row, row))


# 22. Validator nested batch values tuple row type
def test_p37_22_validator_nested_batch_values_tuple_row_type():
    row_t = tuple(1.0 for _ in range(32))
    row_l = list(1.0 for _ in range(32))
    with pytest.raises(TypeError):
        validate_nested_batch_values_tuple((row_t, row_l))


# 23. Validator nested batch values tuple row length
def test_p37_23_validator_nested_batch_values_tuple_row_length():
    row1 = tuple(1.0 for _ in range(32))
    row2 = tuple(1.0 for _ in range(31))
    with pytest.raises(ValueError):
        validate_nested_batch_values_tuple((row1, row2))


# 24. Validator nested batch values tuple element type
def test_p37_24_validator_nested_batch_values_tuple_element_type():
    row1 = tuple(1.0 for _ in range(32))
    row2 = tuple(x for x in range(32))
    with pytest.raises(TypeError):
        validate_nested_batch_values_tuple((row1, row2))


# 25. Validator nested batch values status valid
def test_p37_25_validator_nested_batch_values_status_valid():
    for s in SUPPORTED_FC_VAE_NESTED_BATCH_VALUES_STATUSES:
        validate_nested_batch_values_status(s)


# 26. Validator nested batch values status type
def test_p37_26_validator_nested_batch_values_status_type():
    with pytest.raises(TypeError):
        validate_nested_batch_values_status(123)


# 27. Validator nested batch values status value
def test_p37_27_validator_nested_batch_values_status_value():
    with pytest.raises(ValueError):
        validate_nested_batch_values_status("unknown")


# 28. Local path leakage guard
def test_p37_28_local_path_leakage_guard():
    assert_no_local_path_leakage("safe string")
    for pat in ("file:///", "C:/", "C:\\", "/home/", "/Users/"):
        with pytest.raises(ValueError):
            assert_no_local_path_leakage(f"leak {pat}")


# 29. Forbidden claims guard
def test_p37_29_forbidden_claims_guard():
    assert_no_forbidden_claims("safe conclusion")
    assert_no_forbidden_claims("no_scientific_conclusion")
    assert_no_forbidden_claims("no_final_comparison")
    assert_no_forbidden_claims("nested_values_only_no_tensor_no_forward_in_p37")
    for claim in ("model works", "scientific success", "solved", "best", "winner", "production ready", "state of the art"):
        with pytest.raises(ValueError):
            assert_no_forbidden_claims(f"our {claim} claim")


# 30. Validate request valid
def test_p37_30_validate_request_valid():
    req = build_nested_batch_values_request_from_p36_default()
    validate_nested_batch_values_request(req)


# 31. Validate request type
def test_p37_31_validate_request_type():
    with pytest.raises(TypeError):
        validate_nested_batch_values_request("not request")


# 32. Validate request contract version
def test_p37_32_validate_request_contract_version():
    req = build_nested_batch_values_request_from_p36_default()
    bad = dataclasses.replace(req, contract_version="wrong")
    with pytest.raises(ValueError):
        validate_nested_batch_values_request(bad)


# 33. Validate request nested values kind
def test_p37_33_validate_request_nested_values_kind():
    req = build_nested_batch_values_request_from_p36_default()
    bad = dataclasses.replace(req, nested_values_kind="wrong")
    with pytest.raises(ValueError):
        validate_nested_batch_values_request(bad)


# 34. Validate request architecture id
def test_p37_34_validate_request_architecture_id():
    req = build_nested_batch_values_request_from_p36_default()
    bad = dataclasses.replace(req, architecture_id="wrong")
    with pytest.raises(ValueError):
        validate_nested_batch_values_request(bad)


# 35. Validate request source flat vector contract version
def test_p37_35_validate_request_source_flat_vector():
    req = build_nested_batch_values_request_from_p36_default()
    bad = dataclasses.replace(req, source_flat_vector_contract_version="wrong")
    with pytest.raises(ValueError):
        validate_nested_batch_values_request(bad)


# 36. Validate request source batch view contract version
def test_p37_36_validate_request_source_batch_view():
    req = build_nested_batch_values_request_from_p36_default()
    bad = dataclasses.replace(req, source_batch_view_contract_version="wrong")
    with pytest.raises(ValueError):
        validate_nested_batch_values_request(bad)


# 37. Validate request batch size
def test_p37_37_validate_request_batch_size():
    req = build_nested_batch_values_request_from_p36_default()
    bad = dataclasses.replace(req, batch_size=3)
    with pytest.raises(ValueError):
        validate_nested_batch_values_request(bad)


# 38. Validate request input flat dim
def test_p37_38_validate_request_input_flat_dim():
    req = build_nested_batch_values_request_from_p36_default()
    bad = dataclasses.replace(req, input_flat_dim=31)
    with pytest.raises(ValueError):
        validate_nested_batch_values_request(bad)


# 39. Validate request full vector length
def test_p37_39_validate_request_full_vector_length():
    req = build_nested_batch_values_request_from_p36_default()
    bad = dataclasses.replace(req, full_vector_length=65)
    with pytest.raises(ValueError):
        validate_nested_batch_values_request(bad)


# 40. Validate request shape tuple
def test_p37_40_validate_request_shape_tuple():
    req = build_nested_batch_values_request_from_p36_default()
    bad = dataclasses.replace(req, shape_tuple=(2, 31))
    with pytest.raises(ValueError):
        validate_nested_batch_values_request(bad)


# 41. Validate request order kind
def test_p37_41_validate_request_order_kind():
    req = build_nested_batch_values_request_from_p36_default()
    bad = dataclasses.replace(req, order_kind="wrong")
    with pytest.raises(ValueError):
        validate_nested_batch_values_request(bad)


# 42. Validate request allow nested
def test_p37_42_validate_request_allow_nested():
    req = build_nested_batch_values_request_from_p36_default()
    bad = dataclasses.replace(req, allow_nested_values_in_p37=False)
    with pytest.raises(ValueError):
        validate_nested_batch_values_request(bad)


# 43. Validate request allow array
def test_p37_43_validate_request_allow_array():
    req = build_nested_batch_values_request_from_p36_default()
    bad = dataclasses.replace(req, allow_array_materialization_in_p37=True)
    with pytest.raises(ValueError):
        validate_nested_batch_values_request(bad)


# 44. Validate request allow tensor
def test_p37_44_validate_request_allow_tensor():
    req = build_nested_batch_values_request_from_p36_default()
    bad = dataclasses.replace(req, allow_tensor_materialization_in_p37=True)
    with pytest.raises(ValueError):
        validate_nested_batch_values_request(bad)


# 45. Validate request allow forward
def test_p37_45_validate_request_allow_forward():
    req = build_nested_batch_values_request_from_p36_default()
    bad = dataclasses.replace(req, allow_forward_execution_in_p37=True)
    with pytest.raises(ValueError):
        validate_nested_batch_values_request(bad)


# 46. Validate request reason
def test_p37_46_validate_request_reason():
    req = build_nested_batch_values_request_from_p36_default()
    bad = dataclasses.replace(req, reason="")
    with pytest.raises(ValueError):
        validate_nested_batch_values_request(bad)


# 47. Validate nested values valid
def test_p37_47_validate_nested_values_valid():
    req = build_nested_batch_values_request_from_p36_default()
    vals = build_nested_batch_values(req)
    validate_nested_batch_values(vals)


# 48. Validate nested values type
def test_p37_48_validate_nested_values_type():
    with pytest.raises(TypeError):
        validate_nested_batch_values("not nested")


# 49. Validate nested values contract
def test_p37_49_validate_nested_values_contract():
    req = build_nested_batch_values_request_from_p36_default()
    vals = build_nested_batch_values(req)
    bad = dataclasses.replace(vals, contract_version="wrong")
    with pytest.raises(ValueError):
        validate_nested_batch_values(bad)


# 50. Validate nested values kind
def test_p37_50_validate_nested_values_kind():
    req = build_nested_batch_values_request_from_p36_default()
    vals = build_nested_batch_values(req)
    bad = dataclasses.replace(vals, nested_values_kind="wrong")
    with pytest.raises(ValueError):
        validate_nested_batch_values(bad)


# 51. Validate nested values order
def test_p37_51_validate_nested_values_order():
    req = build_nested_batch_values_request_from_p36_default()
    vals = build_nested_batch_values(req)
    bad = dataclasses.replace(vals, order_kind="wrong")
    with pytest.raises(ValueError):
        validate_nested_batch_values(bad)


# 52. Validate nested values shape
def test_p37_52_validate_nested_values_shape():
    req = build_nested_batch_values_request_from_p36_default()
    vals = build_nested_batch_values(req)
    bad = dataclasses.replace(vals, shape_tuple=(2, 31))
    with pytest.raises(ValueError):
        validate_nested_batch_values(bad)


# 52b. Validate nested values batch size
def test_p37_52b_validate_nested_values_batch_size():
    req = build_nested_batch_values_request_from_p36_default()
    vals = build_nested_batch_values(req)
    bad = dataclasses.replace(vals, batch_size=3)
    with pytest.raises(ValueError):
        validate_nested_batch_values(bad)


# 53. Validate nested values input flat dim
def test_p37_53_validate_nested_values_input_flat_dim():
    req = build_nested_batch_values_request_from_p36_default()
    vals = build_nested_batch_values(req)
    bad = dataclasses.replace(vals, input_flat_dim=31)
    with pytest.raises(ValueError):
        validate_nested_batch_values(bad)


# 54. Validate nested values full vector length
def test_p37_54_validate_nested_values_full_vector_length():
    req = build_nested_batch_values_request_from_p36_default()
    vals = build_nested_batch_values(req)
    bad = dataclasses.replace(vals, full_vector_length=65)
    with pytest.raises(ValueError):
        validate_nested_batch_values(bad)


# 55. Validate nested values row count
def test_p37_55_validate_nested_values_row_count():
    req = build_nested_batch_values_request_from_p36_default()
    vals = build_nested_batch_values(req)
    bad = dataclasses.replace(vals, row_count=3)
    with pytest.raises(ValueError):
        validate_nested_batch_values(bad)


# 56. Validate nested values row lengths
def test_p37_56_validate_nested_values_row_lengths():
    req = build_nested_batch_values_request_from_p36_default()
    vals = build_nested_batch_values(req)
    bad = dataclasses.replace(vals, row_lengths=(32, 31))
    with pytest.raises(ValueError):
        validate_nested_batch_values(bad)


# 57. Validate nested values total scalar count
def test_p37_57_validate_nested_values_total_scalar_count():
    req = build_nested_batch_values_request_from_p36_default()
    vals = build_nested_batch_values(req)
    bad = dataclasses.replace(vals, total_scalar_count=65)
    with pytest.raises(ValueError):
        validate_nested_batch_values(bad)


# 58. Validate nested values flattened matches
def test_p37_58_validate_nested_values_flattened_matches():
    req = build_nested_batch_values_request_from_p36_default()
    vals = build_nested_batch_values(req)
    bad = dataclasses.replace(vals, flattened_matches_source_flat_vector=False)
    with pytest.raises(ValueError):
        validate_nested_batch_values(bad)


# 59. Validate nested values all exact float
def test_p37_59_validate_nested_values_all_exact_float():
    req = build_nested_batch_values_request_from_p36_default()
    vals = build_nested_batch_values(req)
    bad = dataclasses.replace(vals, all_values_are_exact_float=False)
    with pytest.raises(ValueError):
        validate_nested_batch_values(bad)


# 60. Validate nested values all within range
def test_p37_60_validate_nested_values_all_within_range():
    req = build_nested_batch_values_request_from_p36_default()
    vals = build_nested_batch_values(req)
    bad = dataclasses.replace(vals, all_values_within_source_range=False)
    with pytest.raises(ValueError):
        validate_nested_batch_values(bad)


# 61. Validate nested values array materialized
def test_p37_61_validate_nested_values_array_materialized():
    req = build_nested_batch_values_request_from_p36_default()
    vals = build_nested_batch_values(req)
    bad = dataclasses.replace(vals, array_materialized=True)
    with pytest.raises(ValueError):
        validate_nested_batch_values(bad)


# 62. Validate nested values tensor materialized
def test_p37_62_validate_nested_values_tensor_materialized():
    req = build_nested_batch_values_request_from_p36_default()
    vals = build_nested_batch_values(req)
    bad = dataclasses.replace(vals, tensor_materialized=True)
    with pytest.raises(ValueError):
        validate_nested_batch_values(bad)


# 63. Validate nested values forward executed
def test_p37_63_validate_nested_values_forward_executed():
    req = build_nested_batch_values_request_from_p36_default()
    vals = build_nested_batch_values(req)
    bad = dataclasses.replace(vals, forward_executed=True)
    with pytest.raises(ValueError):
        validate_nested_batch_values(bad)


# 64. Validate nested values output generated
def test_p37_64_validate_nested_values_output_generated():
    req = build_nested_batch_values_request_from_p36_default()
    vals = build_nested_batch_values(req)
    bad = dataclasses.replace(vals, output_generated=True)
    with pytest.raises(ValueError):
        validate_nested_batch_values(bad)


# 65. Validate nested values reason
def test_p37_65_validate_nested_values_reason():
    req = build_nested_batch_values_request_from_p36_default()
    vals = build_nested_batch_values(req)
    bad = dataclasses.replace(vals, reason="")
    with pytest.raises(ValueError):
        validate_nested_batch_values(bad)


# 66. Validate metadata valid
def test_p37_66_validate_metadata_valid():
    req = build_nested_batch_values_request_from_p36_default()
    meta = build_nested_batch_values_metadata(req)
    validate_nested_batch_values_metadata(meta)


# 67. Validate metadata type
def test_p37_67_validate_metadata_type():
    with pytest.raises(TypeError):
        validate_nested_batch_values_metadata("not metadata")


# 68. Validate metadata contract
def test_p37_68_validate_metadata_contract():
    req = build_nested_batch_values_request_from_p36_default()
    meta = build_nested_batch_values_metadata(req)
    bad = dataclasses.replace(meta, contract_version="wrong")
    with pytest.raises(ValueError):
        validate_nested_batch_values_metadata(bad)


# 69. Validate metadata source flat vector
def test_p37_69_validate_metadata_source_flat():
    req = build_nested_batch_values_request_from_p36_default()
    meta = build_nested_batch_values_metadata(req)
    bad = dataclasses.replace(meta, source_flat_vector_contract_version="wrong")
    with pytest.raises(ValueError):
        validate_nested_batch_values_metadata(bad)


# 70. Validate metadata source batch view
def test_p37_70_validate_metadata_source_batch():
    req = build_nested_batch_values_request_from_p36_default()
    meta = build_nested_batch_values_metadata(req)
    bad = dataclasses.replace(meta, source_batch_view_contract_version="wrong")
    with pytest.raises(ValueError):
        validate_nested_batch_values_metadata(bad)


# 71. Validate metadata flat vector status
def test_p37_71_validate_metadata_flat_vector_status():
    req = build_nested_batch_values_request_from_p36_default()
    meta = build_nested_batch_values_metadata(req)
    bad = dataclasses.replace(meta, flat_vector_status="")
    with pytest.raises(ValueError):
        validate_nested_batch_values_metadata(bad)


# 72. Validate metadata batch view status
def test_p37_72_validate_metadata_batch_view_status():
    req = build_nested_batch_values_request_from_p36_default()
    meta = build_nested_batch_values_metadata(req)
    bad = dataclasses.replace(meta, batch_view_status="")
    with pytest.raises(ValueError):
        validate_nested_batch_values_metadata(bad)


# 73. Validate metadata torch available
def test_p37_73_validate_metadata_torch_available():
    req = build_nested_batch_values_request_from_p36_default()
    meta = build_nested_batch_values_metadata(req)
    # torch_available can be True or False, just type check
    bad = dataclasses.replace(meta, torch_available=1)
    with pytest.raises(TypeError):
        validate_nested_batch_values_metadata(bad)


# 74. Validate metadata flat vector available
def test_p37_74_validate_metadata_flat_vector_available():
    req = build_nested_batch_values_request_from_p36_default()
    meta = build_nested_batch_values_metadata(req)
    bad = dataclasses.replace(meta, flat_vector_available_in_p35=False)
    with pytest.raises(ValueError):
        validate_nested_batch_values_metadata(bad)


# 75. Validate metadata batch view available
def test_p37_75_validate_metadata_batch_view_available():
    req = build_nested_batch_values_request_from_p36_default()
    meta = build_nested_batch_values_metadata(req)
    bad = dataclasses.replace(meta, batch_view_available_in_p36=False)
    with pytest.raises(ValueError):
        validate_nested_batch_values_metadata(bad)


# 76. Validate metadata flat vector length
def test_p37_76_validate_metadata_flat_vector_length():
    req = build_nested_batch_values_request_from_p36_default()
    meta = build_nested_batch_values_metadata(req)
    bad = dataclasses.replace(meta, flat_vector_length=65)
    with pytest.raises(ValueError):
        validate_nested_batch_values_metadata(bad)


# 77. Validate metadata shape
def test_p37_77_validate_metadata_shape():
    req = build_nested_batch_values_request_from_p36_default()
    meta = build_nested_batch_values_metadata(req)
    bad = dataclasses.replace(meta, shape_tuple=(2, 31))
    with pytest.raises(ValueError):
        validate_nested_batch_values_metadata(bad)


# 78. Validate metadata batch size
def test_p37_78_validate_metadata_shape_batch_size():
    req = build_nested_batch_values_request_from_p36_default()
    meta = build_nested_batch_values_metadata(req)
    bad = dataclasses.replace(meta, batch_size=3)
    with pytest.raises(ValueError):
        validate_nested_batch_values_metadata(bad)


# 79. Validate metadata input flat dim
def test_p37_79_validate_metadata_input_flat_dim():
    req = build_nested_batch_values_request_from_p36_default()
    meta = build_nested_batch_values_metadata(req)
    bad = dataclasses.replace(meta, input_flat_dim=31)
    with pytest.raises(ValueError):
        validate_nested_batch_values_metadata(bad)


# 80. Validate metadata nested materialized
def test_p37_80_validate_metadata_nested_materialized():
    req = build_nested_batch_values_request_from_p36_default()
    meta = build_nested_batch_values_metadata(req)
    bad = dataclasses.replace(meta, nested_values_materialized=False)
    with pytest.raises(ValueError):
        validate_nested_batch_values_metadata(bad)


# 81. Validate metadata attempts
def test_p37_81_validate_metadata_attempts():
    req = build_nested_batch_values_request_from_p36_default()
    meta = build_nested_batch_values_metadata(req)
    for flag in ("array_materialization_attempted", "tensor_materialization_attempted", "forward_execution_attempted", "output_generation_attempted"):
        bad = dataclasses.replace(meta, **{flag: True})
        with pytest.raises(ValueError):
            validate_nested_batch_values_metadata(bad)


# 82. Validate metadata reason
def test_p37_82_validate_metadata_reason():
    req = build_nested_batch_values_request_from_p36_default()
    meta = build_nested_batch_values_metadata(req)
    bad = dataclasses.replace(meta, reason="")
    with pytest.raises(ValueError):
        validate_nested_batch_values_metadata(bad)


# 83. Validate result valid
def test_p37_83_validate_result_valid():
    req = build_nested_batch_values_request_from_p36_default()
    res = build_nested_batch_values_result(req)
    validate_nested_batch_values_result(res)


# 84. Validate result type
def test_p37_84_validate_result_type():
    with pytest.raises(TypeError):
        validate_nested_batch_values_result("not result")


# 85. Validate result contract
def test_p37_85_validate_result_contract():
    req = build_nested_batch_values_request_from_p36_default()
    res = build_nested_batch_values_result(req)
    bad = dataclasses.replace(res, contract_version="wrong")
    with pytest.raises(ValueError):
        validate_nested_batch_values_result(bad)


# 86. Validate result availabilities
def test_p37_86_validate_result_availabilities():
    req = build_nested_batch_values_request_from_p36_default()
    res = build_nested_batch_values_result(req)
    for flag in ("array_materialization_available_in_p37", "tensor_materialization_available_in_p37", "forward_execution_available_in_p37", "output_generation_available_in_p37"):
        bad = dataclasses.replace(res, **{flag: True})
        with pytest.raises(ValueError):
            validate_nested_batch_values_result(bad)


# 87. Validate result no flags
def test_p37_87_validate_result_no_flags():
    req = build_nested_batch_values_request_from_p36_default()
    res = build_nested_batch_values_result(req)
    for flag in ("no_array_created", "no_tensor_created", "no_forward_execution", "no_output_generation", "no_training_loop", "no_optimizer", "no_checkpointing", "no_artifact_generation", "no_final_comparison", "no_scientific_conclusion"):
        bad = dataclasses.replace(res, **{flag: False})
        with pytest.raises(ValueError):
            validate_nested_batch_values_result(bad)


# 88. Validate result status coherence
def test_p37_88_validate_result_status_coherence():
    req = build_nested_batch_values_request_from_p36_default()
    res = build_nested_batch_values_result(req)
    bad = dataclasses.replace(res, status="blocked_torch_unavailable")
    # if torch is actually unavailable, this is fine. If it is available, it will fail.
    # We can test status coherence by modifying metadata and result status in matching/mismatched pairs.
    meta = dataclasses.replace(res.metadata, torch_available=False)
    res_mismatched = dataclasses.replace(res, metadata=meta, status=FC_VAE_NESTED_BATCH_VALUES_STATUS_NESTED_VALUES_ONLY)
    with pytest.raises(ValueError):
        validate_nested_batch_values_result(res_mismatched)

    res_matched = dataclasses.replace(res, metadata=meta, status=FC_VAE_NESTED_BATCH_VALUES_STATUS_TORCH_UNAVAILABLE)
    validate_nested_batch_values_result(res_matched)


# 89. Validate result reason
def test_p37_89_validate_result_reason():
    req = build_nested_batch_values_request_from_p36_default()
    res = build_nested_batch_values_result(req)
    bad = dataclasses.replace(res, reason="")
    with pytest.raises(ValueError):
        validate_nested_batch_values_result(bad)


# 90. Helper build_nested_tuple
def test_p37_90_helpers_build_nested_tuple():
    flat = tuple(float(x) for x in range(64))
    res = build_nested_tuple_from_flat_values(flat, 2, 32)
    assert type(res) is tuple
    assert len(res) == 2
    assert type(res[0]) is tuple
    assert type(res[1]) is tuple
    assert len(res[0]) == 32
    assert len(res[1]) == 32


# 91. Helper flatten_nested
def test_p37_91_helpers_flatten_nested():
    row1 = tuple(float(x) for x in range(32))
    row2 = tuple(float(x) for x in range(32, 64))
    nested = (row1, row2)
    flat = flatten_nested_batch_values(nested)
    assert type(flat) is tuple
    assert len(flat) == 64
    assert flat == row1 + row2


# 92. Helper reconstruction
def test_p37_92_helpers_reconstruction():
    flat = tuple(float(x) for x in range(64))
    nested = build_nested_tuple_from_flat_values(flat, 2, 32)
    flat_recon = flatten_nested_batch_values(nested)
    assert flat_recon == flat


# 93. Helper segmentation
def test_p37_93_helpers_segmentation():
    flat = tuple(float(x) for x in range(64))
    nested = build_nested_tuple_from_flat_values(flat, 2, 32)
    assert nested[0] == flat[0:32]
    assert nested[1] == flat[32:64]


# 94. Helper first 4 match P35/P34 default preview values
def test_p37_94_helpers_first_four_match():
    # first 4 elements of default fake input flat vector in P35 match preview values
    req = build_nested_batch_values_request_from_p36_default()
    vals = build_nested_batch_values(req)
    expected = [-0.24297189, -0.1686747, -0.09437751, -0.02008032]
    assert list(vals.nested_batch_values[0][:4]) == expected


# 95. Request builder validates
def test_p37_95_request_builder():
    req = build_nested_batch_values_request_from_p36_default()
    validate_nested_batch_values_request(req)
    assert req.allow_nested_values_in_p37 is True
    assert req.allow_array_materialization_in_p37 is False
    assert req.allow_tensor_materialization_in_p37 is False
    assert req.allow_forward_execution_in_p37 is False


# 96. Nested batch values builder validates
def test_p37_96_nested_batch_values_builder():
    req = build_nested_batch_values_request_from_p36_default()
    vals = build_nested_batch_values(req)
    validate_nested_batch_values(vals)
    assert vals.flattened_matches_source_flat_vector is True
    assert vals.row_count == 2
    assert vals.row_lengths == (32, 32)
    assert vals.total_scalar_count == 64


# 97. Metadata builder validates
def test_p37_97_metadata_builder():
    req = build_nested_batch_values_request_from_p36_default()
    meta = build_nested_batch_values_metadata(req)
    validate_nested_batch_values_metadata(meta)
    assert meta.nested_values_materialized is True


# 98. Result builder validates
def test_p37_98_result_builder():
    req = build_nested_batch_values_request_from_p36_default()
    res = build_nested_batch_values_result(req)
    validate_nested_batch_values_result(res)
    assert res.nested_values_available_in_p37 is True
    assert res.status in SUPPORTED_FC_VAE_NESTED_BATCH_VALUES_STATUSES


# 99. Run probe
def test_p37_99_run_probe():
    res = run_nested_batch_values_probe()
    validate_nested_batch_values_result(res)


# 100. Serializers JSON safe
def test_p37_100_serializers_json_safe():
    res = run_nested_batch_values_probe()
    d = nested_batch_values_result_to_json_dict(res)
    assert isinstance(d, dict)
    js = compact_nested_batch_values_json(res)
    assert isinstance(js, str)
    assert "nested_batch_values" in js


# 101. Serialized result has nested_batch_values as list-of-lists
def test_p37_101_serializers_nested_list_of_lists():
    res = run_nested_batch_values_probe()
    js = compact_nested_batch_values_json(res)
    d = json.loads(js)
    assert isinstance(d["nested_values"]["nested_batch_values"], list)
    assert isinstance(d["nested_values"]["nested_batch_values"][0], list)
    assert isinstance(d["nested_values"]["nested_batch_values"][1], list)
    assert len(d["nested_values"]["nested_batch_values"]) == 2
    assert len(d["nested_values"]["nested_batch_values"][0]) == 32


# 102. Serializers no key pollution
def test_p37_102_serializers_no_key_pollution():
    res = run_nested_batch_values_probe()
    js = compact_nested_batch_values_json(res)
    assert ": bool" not in js


# 103. Module imports clean
def test_p37_103_module_imports_clean():
    content = pathlib.Path("src/phase2/fc_vae_nested_batch_values.py").read_text(encoding="utf-8")
    for pattern in (
        "import torch", "from torch",
        "import numpy", "from numpy",
        "import random", "from random",
        "import secrets", "from secrets",
        "import array", "from array",
        "import argparse", "from argparse",
        "import subprocess", "from subprocess",
    ):
        assert pattern not in content


# 104. Module has no forward and no model constructors
def test_p37_104_module_no_forward_and_model_constructors():
    content = pathlib.Path("src/phase2/fc_vae_nested_batch_values.py").read_text(encoding="utf-8")
    assert "def forward(" not in content
    assert "def forward " not in content
    assert "forward(" not in content
    for pattern in ("torch.tensor", "np.array", "np.zeros", "torch.zeros"):
        assert pattern not in content


# 105. Module has no train/loss/optimizer
def test_p37_105_module_no_train_loss_optimizer():
    content = pathlib.Path("src/phase2/fc_vae_nested_batch_values.py").read_text(encoding="utf-8")
    for pattern in ("def train", "def fit", "def loss", "def optimizer", "def checkpoint"):
        assert pattern not in content


# 106. Init file monkeypatch check
def test_p37_106_init_monkeypatch():
    content = pathlib.Path("src/phase2/__init__.py").read_text(encoding="utf-8")
    assert "subprocess.run" not in content
    assert "subprocess.Popen" not in content


# 107. Scope gate
def test_p37_107_scope_gate():
    allowed = {
        "src/phase2/fc_vae_nested_batch_values.py",
        "src/phase2/__init__.py",
        "tests/test_phase2_fc_vae_nested_batch_values.py",
        "tools/phase2/run_p37_nested_batch_values_smoke.py",
        "tests/test_phase2_p37_nested_batch_values_smoke.py",
        "reports/PHASE_2_P37_CONTROLLED_NESTED_BATCH_VALUES_NO_TENSOR_NO_ARRAY_REPORT.md",
    }
    # Base branch check
    res = subprocess.run(
        ["git", "diff", "--name-only", "phase2/p36-flat-vector-2d-batch-view-contract-no-nested-values"],
        capture_output=True, text=True, check=True
    )
    modified = [line.strip() for line in res.stdout.splitlines() if line.strip()]
    for f in modified:
        f_norm = f.replace("\\", "/")
        assert f_norm in allowed, f"Forbidden file modification detected in P37: {f_norm}"
