# tests/test_phase2_fc_vae_flat_vector_batch_view.py

import dataclasses
import json
import pathlib
import subprocess
import pytest
from dataclasses import is_dataclass

from src.phase2.fc_vae_flat_vector_batch_view import (
    FC_VAE_FLAT_VECTOR_BATCH_VIEW_CONTRACT_VERSION,
    FC_VAE_FLAT_VECTOR_BATCH_VIEW_KIND,
    FC_VAE_FLAT_VECTOR_BATCH_VIEW_MODULE_NAME,
    FC_VAE_FLAT_VECTOR_BATCH_VIEW_STATUS_TORCH_UNAVAILABLE,
    FC_VAE_FLAT_VECTOR_BATCH_VIEW_STATUS_FLAT_VECTOR_BLOCKED,
    FC_VAE_FLAT_VECTOR_BATCH_VIEW_STATUS_VIEW_ONLY,
    SUPPORTED_FC_VAE_FLAT_VECTOR_BATCH_VIEW_STATUSES,
    FC_VAE_FLAT_VECTOR_BATCH_VIEW_ORDER_KIND,
    DEFAULT_P36_VIEW_RANK,
    DEFAULT_P36_BATCH_SIZE,
    DEFAULT_P36_INPUT_FLAT_DIM,
    DEFAULT_P36_FULL_VECTOR_LENGTH,
    DEFAULT_P36_SHAPE_TUPLE,
    FCVAEFlatVectorBatchViewRequest,
    FCVAEFlatVectorBatchViewShape,
    FCVAEFlatVectorBatchViewMetadata,
    FCVAEFlatVectorBatchViewResult,
    validate_non_empty_str,
    validate_bool,
    validate_non_negative_int,
    validate_positive_int,
    validate_shape_tuple,
    validate_full_vector_length,
    validate_view_rank,
    validate_row_major_formula,
    validate_flat_vector_batch_view_status,
    assert_no_local_path_leakage,
    assert_no_forbidden_claims,
    validate_flat_vector_batch_view_request,
    validate_flat_vector_batch_view_shape,
    validate_flat_vector_batch_view_metadata,
    validate_flat_vector_batch_view_result,
    compute_row_major_flat_index,
    build_flat_vector_batch_view_request_from_p35_default,
    build_flat_vector_batch_view_shape,
    build_flat_vector_batch_view_metadata,
    build_flat_vector_batch_view_result,
    run_flat_vector_batch_view_probe,
    flat_vector_batch_view_request_to_json_dict,
    flat_vector_batch_view_shape_to_json_dict,
    flat_vector_batch_view_metadata_to_json_dict,
    flat_vector_batch_view_result_to_json_dict,
    compact_flat_vector_batch_view_json,
)


# 1. constants exact
def test_p36_01_constants_exact():
    assert FC_VAE_FLAT_VECTOR_BATCH_VIEW_CONTRACT_VERSION == "phase2_p36_flat_vector_2d_batch_view_contract_v1"
    assert FC_VAE_FLAT_VECTOR_BATCH_VIEW_KIND == "flat_vector_2d_batch_view_metadata_no_nested_values"
    assert FC_VAE_FLAT_VECTOR_BATCH_VIEW_MODULE_NAME == "src.phase2.fc_vae_flat_vector_batch_view"
    assert FC_VAE_FLAT_VECTOR_BATCH_VIEW_STATUS_TORCH_UNAVAILABLE == "blocked_torch_unavailable"
    assert FC_VAE_FLAT_VECTOR_BATCH_VIEW_STATUS_FLAT_VECTOR_BLOCKED == "blocked_by_flat_vector_status"
    assert FC_VAE_FLAT_VECTOR_BATCH_VIEW_STATUS_VIEW_ONLY == "view_metadata_only_no_2d_values_in_p36"
    assert SUPPORTED_FC_VAE_FLAT_VECTOR_BATCH_VIEW_STATUSES == (
        "blocked_torch_unavailable",
        "blocked_by_flat_vector_status",
        "view_metadata_only_no_2d_values_in_p36",
    )
    assert FC_VAE_FLAT_VECTOR_BATCH_VIEW_ORDER_KIND == "row_major_flat_index_view"
    assert DEFAULT_P36_VIEW_RANK == 2
    assert DEFAULT_P36_BATCH_SIZE == 2
    assert DEFAULT_P36_INPUT_FLAT_DIM == 32
    assert DEFAULT_P36_FULL_VECTOR_LENGTH == 64
    assert DEFAULT_P36_SHAPE_TUPLE == (2, 32)


# 2. dataclasses frozen
def test_p36_02_dataclasses_frozen():
    for cls in (
        FCVAEFlatVectorBatchViewRequest,
        FCVAEFlatVectorBatchViewShape,
        FCVAEFlatVectorBatchViewMetadata,
        FCVAEFlatVectorBatchViewResult,
    ):
        assert is_dataclass(cls)
        req = build_flat_vector_batch_view_request_from_p35_default()
        with pytest.raises(Exception):
            req.reason = "modified"


# 3. primitive validator non_empty_str
def test_p36_03_primitive_validator_non_empty_str():
    validate_non_empty_str("valid", "field")
    with pytest.raises(TypeError):
        validate_non_empty_str(123, "field")
    with pytest.raises(ValueError):
        validate_non_empty_str("", "field")
    with pytest.raises(ValueError):
        validate_non_empty_str(" padded ", "field")


# 4. primitive validator bool
def test_p36_04_primitive_validator_bool():
    validate_bool(True, "flag")
    validate_bool(False, "flag")
    with pytest.raises(TypeError):
        validate_bool(1, "flag")


# 5. primitive validator non_negative_int
def test_p36_05_primitive_validator_non_negative_int():
    validate_non_negative_int(0, "val")
    validate_non_negative_int(5, "val")
    with pytest.raises(ValueError):
        validate_non_negative_int(-1, "val")
    with pytest.raises(TypeError):
        validate_non_negative_int(True, "val")


# 6. primitive validator positive_int
def test_p36_06_primitive_validator_positive_int():
    validate_positive_int(1, "val")
    with pytest.raises(ValueError):
        validate_positive_int(0, "val")
    with pytest.raises(TypeError):
        validate_positive_int(False, "val")


# 7. shape tuple validator accepts valid
def test_p36_07_shape_tuple_validator_accepts_valid():
    validate_shape_tuple((2, 32))
    validate_shape_tuple((1, 10))


# 8. shape tuple validator rejects list
def test_p36_08_shape_tuple_validator_rejects_list():
    with pytest.raises(TypeError):
        validate_shape_tuple([2, 32])


# 9. shape tuple validator rejects invalid elements
def test_p36_09_shape_tuple_validator_rejects_invalid_elements():
    with pytest.raises(ValueError):
        validate_shape_tuple((0, 32))
    with pytest.raises(TypeError):
        validate_shape_tuple((2.5, 32))
    with pytest.raises(TypeError):
        validate_shape_tuple((2, True))


# 10. shape tuple validator rejects wrong length
def test_p36_10_shape_tuple_validator_rejects_wrong_length():
    with pytest.raises(ValueError):
        validate_shape_tuple((2, 32, 1))
    with pytest.raises(ValueError):
        validate_shape_tuple((2,))


# 11. full vector length validator accepts valid
def test_p36_11_full_vector_length_validator_accepts_valid():
    validate_full_vector_length(64)


# 12. full vector length validator rejects invalid
def test_p36_12_full_vector_length_validator_rejects_invalid():
    with pytest.raises(ValueError):
        validate_full_vector_length(0)
    with pytest.raises(ValueError):
        validate_full_vector_length(-5)
    with pytest.raises(TypeError):
        validate_full_vector_length(64.5)
    with pytest.raises(TypeError):
        validate_full_vector_length(True)


# 13. view rank validator accepts 2
def test_p36_13_view_rank_validator_accepts_2():
    validate_view_rank(2)


# 14. view rank validator rejects invalid
def test_p36_14_view_rank_validator_rejects_invalid():
    with pytest.raises(ValueError):
        validate_view_rank(3)
    with pytest.raises(TypeError):
        validate_view_rank(2.0)
    with pytest.raises(TypeError):
        validate_view_rank(True)


# 15. row major formula validator accepts valid
def test_p36_15_row_major_formula_validator_accepts_valid():
    validate_row_major_formula("flat_index = batch_index * input_flat_dim + feature_index")


# 16. row major formula validator rejects invalid
def test_p36_16_row_major_formula_validator_rejects_invalid():
    with pytest.raises(ValueError):
        validate_row_major_formula("flat_index = row * dim + col")
    with pytest.raises(TypeError):
        validate_row_major_formula(123)


# 17. status validator accepts valid
def test_p36_17_status_validator_accepts_valid():
    for status in SUPPORTED_FC_VAE_FLAT_VECTOR_BATCH_VIEW_STATUSES:
        validate_flat_vector_batch_view_status(status)


# 18. status validator rejects invalid
def test_p36_18_status_validator_rejects_invalid():
    with pytest.raises(ValueError):
        validate_flat_vector_batch_view_status("unknown")
    with pytest.raises(TypeError):
        validate_flat_vector_batch_view_status(123)


# 19. local path and forbidden claim guards work
def test_p36_19_local_path_leakage_guard():
    assert_no_local_path_leakage("safe string")
    for pattern in ("file:///", "C:/", "C:\\", "/home/", "/Users/"):
        with pytest.raises(ValueError):
            assert_no_local_path_leakage(f"leak {pattern}")


# 20. forbidden success claims guard
def test_p36_20_forbidden_claims_guard():
    assert_no_forbidden_claims("safe conclusion")
    # Allowed
    assert_no_forbidden_claims("no_scientific_conclusion")
    assert_no_forbidden_claims("no_final_comparison")
    assert_no_forbidden_claims("view_metadata_only_no_2d_values_in_p36")
    # Forbidden
    for claim in ("model works", "scientific success", "solved", "best", "winner", "production ready", "state of the art"):
        with pytest.raises(ValueError):
            assert_no_forbidden_claims(f"our {claim} claim")


# 21. row-major index helper returns 0 for (0,0,32)
def test_p36_21_row_major_index_helper_0_0():
    assert compute_row_major_flat_index(0, 0, 32) == 0


# 22. row-major index helper returns 31 for (0,31,32)
def test_p36_22_row_major_index_helper_0_31():
    assert compute_row_major_flat_index(0, 31, 32) == 31


# 23. row-major index helper returns 32 for (1,0,32)
def test_p36_23_row_major_index_helper_1_0():
    assert compute_row_major_flat_index(1, 0, 32) == 32


# 24. row-major index helper returns 63 for (1,31,32)
def test_p36_24_row_major_index_helper_1_31():
    assert compute_row_major_flat_index(1, 31, 32) == 63


# 25. row-major index helper rejects bool
def test_p36_25_row_major_index_helper_rejects_bool():
    with pytest.raises(TypeError):
        compute_row_major_flat_index(True, 0, 32)
    with pytest.raises(TypeError):
        compute_row_major_flat_index(0, False, 32)


# 26. row-major index helper rejects float
def test_p36_26_row_major_index_helper_rejects_float():
    with pytest.raises(TypeError):
        compute_row_major_flat_index(0.5, 0, 32)


# 27. row-major index helper rejects negative
def test_p36_27_row_major_index_helper_rejects_negative():
    with pytest.raises(ValueError):
        compute_row_major_flat_index(-1, 0, 32)
    with pytest.raises(ValueError):
        compute_row_major_flat_index(0, -1, 32)


# 28. row-major index helper rejects invalid dim
def test_p36_28_row_major_index_helper_rejects_invalid_dim():
    with pytest.raises(ValueError):
        compute_row_major_flat_index(0, 0, 0)
    with pytest.raises(ValueError):
        compute_row_major_flat_index(0, 0, -32)


# 29. request builder validates
def test_p36_29_request_builder_validates():
    req = build_flat_vector_batch_view_request_from_p35_default()
    validate_flat_vector_batch_view_request(req)


# 30. request builder copies P35 contract
def test_p36_30_request_builder_copies_p35_contract():
    req = build_flat_vector_batch_view_request_from_p35_default()
    assert req.source_flat_vector_contract_version == "phase2_p35_fake_input_flat_vector_contract_v1"


# 31. request builder shape (2,32)
def test_p36_31_request_builder_shape():
    req = build_flat_vector_batch_view_request_from_p35_default()
    assert req.shape_tuple == (2, 32)


# 32. request builder full vector length 64
def test_p36_32_request_builder_full_length():
    req = build_flat_vector_batch_view_request_from_p35_default()
    assert req.full_vector_length == 64


# 33. request builder all allow flags false
def test_p36_33_request_builder_all_allow_flags_false():
    req = build_flat_vector_batch_view_request_from_p35_default()
    assert req.allow_nested_values_in_p36 is False
    assert req.allow_2d_batch_materialization_in_p36 is False
    assert req.allow_array_materialization_in_p36 is False
    assert req.allow_tensor_materialization_in_p36 is False
    assert req.allow_forward_execution_in_p36 is False


# 34. request rejects wrong contract
def test_p36_34_request_validator_rejects_wrong_contract():
    req = build_flat_vector_batch_view_request_from_p35_default()
    bad = dataclasses.replace(req, contract_version="wrong")
    with pytest.raises(ValueError):
        validate_flat_vector_batch_view_request(bad)


# 35. request rejects wrong kind
def test_p36_35_request_validator_rejects_wrong_kind():
    req = build_flat_vector_batch_view_request_from_p35_default()
    bad = dataclasses.replace(req, view_kind="wrong")
    with pytest.raises(ValueError):
        validate_flat_vector_batch_view_request(bad)


# 36. request rejects wrong architecture_id
def test_p36_36_request_validator_rejects_wrong_architecture_id():
    req = build_flat_vector_batch_view_request_from_p35_default()
    bad = dataclasses.replace(req, architecture_id="wrong")
    with pytest.raises(ValueError):
        validate_flat_vector_batch_view_request(bad)


# 37. request rejects wrong source contract
def test_p36_37_request_validator_rejects_wrong_source_flat_vector():
    req = build_flat_vector_batch_view_request_from_p35_default()
    bad = dataclasses.replace(req, source_flat_vector_contract_version="wrong")
    with pytest.raises(ValueError):
        validate_flat_vector_batch_view_request(bad)


# 38. request rejects wrong batch size
def test_p36_38_request_validator_rejects_wrong_batch_size():
    req = build_flat_vector_batch_view_request_from_p35_default()
    bad = dataclasses.replace(req, batch_size=3, full_vector_length=96, shape_tuple=(3, 32))
    with pytest.raises(ValueError):
        validate_flat_vector_batch_view_request(bad)


# 39. request rejects wrong input dim
def test_p36_39_request_validator_rejects_wrong_input_dim():
    req = build_flat_vector_batch_view_request_from_p35_default()
    bad = dataclasses.replace(req, input_flat_dim=16, full_vector_length=32, shape_tuple=(2, 16))
    with pytest.raises(ValueError):
        validate_flat_vector_batch_view_request(bad)


# 40. request rejects wrong full vector length
def test_p36_40_request_validator_rejects_wrong_full_length():
    req = build_flat_vector_batch_view_request_from_p35_default()
    bad = dataclasses.replace(req, full_vector_length=65)
    with pytest.raises(ValueError):
        validate_flat_vector_batch_view_request(bad)


# 41. request rejects wrong view rank
def test_p36_41_request_validator_rejects_wrong_view_rank():
    req = build_flat_vector_batch_view_request_from_p35_default()
    bad = dataclasses.replace(req, view_rank=3)
    with pytest.raises(ValueError):
        validate_flat_vector_batch_view_request(bad)


# 42. request rejects wrong shape tuple
def test_p36_42_request_validator_rejects_wrong_shape_tuple():
    req = build_flat_vector_batch_view_request_from_p35_default()
    bad = dataclasses.replace(req, shape_tuple=(2, 16))
    with pytest.raises(ValueError):
        validate_flat_vector_batch_view_request(bad)


# 43. request rejects wrong view order kind
def test_p36_43_request_validator_rejects_wrong_view_order_kind():
    req = build_flat_vector_batch_view_request_from_p35_default()
    bad = dataclasses.replace(req, view_order_kind="wrong")
    with pytest.raises(ValueError):
        validate_flat_vector_batch_view_request(bad)


# 44. request rejects nested values allow true
def test_p36_44_request_validator_rejects_allow_nested_values():
    req = build_flat_vector_batch_view_request_from_p35_default()
    bad = dataclasses.replace(req, allow_nested_values_in_p36=True)
    with pytest.raises(ValueError):
        validate_flat_vector_batch_view_request(bad)


# 45. request rejects 2D materialization allow true
def test_p36_45_request_validator_rejects_allow_2d_batch():
    req = build_flat_vector_batch_view_request_from_p35_default()
    bad = dataclasses.replace(req, allow_2d_batch_materialization_in_p36=True)
    with pytest.raises(ValueError):
        validate_flat_vector_batch_view_request(bad)


# 46. request rejects array allow true
def test_p36_46_request_validator_rejects_allow_array():
    req = build_flat_vector_batch_view_request_from_p35_default()
    bad = dataclasses.replace(req, allow_array_materialization_in_p36=True)
    with pytest.raises(ValueError):
        validate_flat_vector_batch_view_request(bad)


# 47. request rejects tensor allow true
def test_p36_47_request_validator_rejects_allow_tensor():
    req = build_flat_vector_batch_view_request_from_p35_default()
    bad = dataclasses.replace(req, allow_tensor_materialization_in_p36=True)
    with pytest.raises(ValueError):
        validate_flat_vector_batch_view_request(bad)


# 48. request rejects forward allow true
def test_p36_48_request_validator_rejects_allow_forward():
    req = build_flat_vector_batch_view_request_from_p35_default()
    bad = dataclasses.replace(req, allow_forward_execution_in_p36=True)
    with pytest.raises(ValueError):
        validate_flat_vector_batch_view_request(bad)


# 49. shape builder validates
def test_p36_49_shape_builder_validates():
    req = build_flat_vector_batch_view_request_from_p35_default()
    shape = build_flat_vector_batch_view_shape(req)
    validate_flat_vector_batch_view_shape(shape)


# 50. shape has no nested values
def test_p36_50_shape_has_no_nested_values():
    req = build_flat_vector_batch_view_request_from_p35_default()
    shape = build_flat_vector_batch_view_shape(req)
    assert shape.nested_values_materialized is False


# 51. shape has no 2D batch
def test_p36_51_shape_has_no_2d_batch():
    req = build_flat_vector_batch_view_request_from_p35_default()
    shape = build_flat_vector_batch_view_shape(req)
    assert shape.two_d_batch_materialized is False


# 52. shape has no array/tensor
def test_p36_52_shape_has_no_array():
    req = build_flat_vector_batch_view_request_from_p35_default()
    shape = build_flat_vector_batch_view_shape(req)
    assert shape.array_materialized is False
    assert shape.tensor_materialized is False


# 53. shape has values_copied_from_flat_vector False
def test_p36_53_shape_has_no_tensor():
    req = build_flat_vector_batch_view_request_from_p35_default()
    shape = build_flat_vector_batch_view_shape(req)
    assert shape.values_copied_from_flat_vector is False


# 54. shape rejects nested_values_materialized True
def test_p36_54_shape_has_values_copied_false():
    req = build_flat_vector_batch_view_request_from_p35_default()
    shape = build_flat_vector_batch_view_shape(req)
    bad = dataclasses.replace(shape, nested_values_materialized=True)
    with pytest.raises(ValueError):
        validate_flat_vector_batch_view_shape(bad)


# 55. shape rejects two_d_batch_materialized True
def test_p36_55_shape_validator_rejects_nested_values_materialized():
    req = build_flat_vector_batch_view_request_from_p35_default()
    shape = build_flat_vector_batch_view_shape(req)
    bad = dataclasses.replace(shape, two_d_batch_materialized=True)
    with pytest.raises(ValueError):
        validate_flat_vector_batch_view_shape(bad)


# 56. shape rejects array_materialized True
def test_p36_56_shape_validator_rejects_2d_batch_materialized():
    req = build_flat_vector_batch_view_request_from_p35_default()
    shape = build_flat_vector_batch_view_shape(req)
    bad = dataclasses.replace(shape, array_materialized=True)
    with pytest.raises(ValueError):
        validate_flat_vector_batch_view_shape(bad)


# 57. shape rejects tensor_materialized True
def test_p36_57_shape_validator_rejects_array_materialized():
    req = build_flat_vector_batch_view_request_from_p35_default()
    shape = build_flat_vector_batch_view_shape(req)
    bad = dataclasses.replace(shape, tensor_materialized=True)
    with pytest.raises(ValueError):
        validate_flat_vector_batch_view_shape(bad)


# 58. shape rejects values_copied True
def test_p36_58_shape_validator_rejects_tensor_materialized():
    req = build_flat_vector_batch_view_request_from_p35_default()
    shape = build_flat_vector_batch_view_shape(req)
    bad = dataclasses.replace(shape, values_copied_from_flat_vector=True)
    with pytest.raises(ValueError):
        validate_flat_vector_batch_view_shape(bad)


# 59. metadata builder validates
def test_p36_59_shape_validator_rejects_values_copied():
    req = build_flat_vector_batch_view_request_from_p35_default()
    meta = build_flat_vector_batch_view_metadata(req)
    validate_flat_vector_batch_view_metadata(meta)


# 60. metadata flat_vector_reused_without_copy True
def test_p36_60_metadata_builder_validates():
    req = build_flat_vector_batch_view_request_from_p35_default()
    meta = build_flat_vector_batch_view_metadata(req)
    assert meta.flat_vector_reused_without_copy is True


# 61. metadata rejects nested_values_attempted True
def test_p36_61_metadata_flat_vector_reused_without_copy():
    req = build_flat_vector_batch_view_request_from_p35_default()
    meta = build_flat_vector_batch_view_metadata(req)
    bad = dataclasses.replace(meta, nested_values_attempted=True)
    with pytest.raises(ValueError):
        validate_flat_vector_batch_view_metadata(bad)


# 62. metadata rejects 2D materialization attempted True
def test_p36_62_metadata_validator_rejects_nested_values_attempted():
    req = build_flat_vector_batch_view_request_from_p35_default()
    meta = build_flat_vector_batch_view_metadata(req)
    bad = dataclasses.replace(meta, two_d_batch_materialization_attempted=True)
    with pytest.raises(ValueError):
        validate_flat_vector_batch_view_metadata(bad)


# 63. metadata rejects array attempted True
def test_p36_63_metadata_validator_rejects_2d_batch_attempted():
    req = build_flat_vector_batch_view_request_from_p35_default()
    meta = build_flat_vector_batch_view_metadata(req)
    bad = dataclasses.replace(meta, array_materialization_attempted=True)
    with pytest.raises(ValueError):
        validate_flat_vector_batch_view_metadata(bad)


# 64. metadata rejects tensor attempted True
def test_p36_64_metadata_validator_rejects_array_attempted():
    req = build_flat_vector_batch_view_request_from_p35_default()
    meta = build_flat_vector_batch_view_metadata(req)
    bad = dataclasses.replace(meta, tensor_materialization_attempted=True)
    with pytest.raises(ValueError):
        validate_flat_vector_batch_view_metadata(bad)


# 65. metadata rejects forward attempted True
def test_p36_65_metadata_validator_rejects_tensor_attempted():
    req = build_flat_vector_batch_view_request_from_p35_default()
    meta = build_flat_vector_batch_view_metadata(req)
    bad = dataclasses.replace(meta, forward_execution_attempted=True)
    with pytest.raises(ValueError):
        validate_flat_vector_batch_view_metadata(bad)


# 66. metadata rejects output attempted True
def test_p36_66_metadata_validator_rejects_forward_attempted():
    req = build_flat_vector_batch_view_request_from_p35_default()
    meta = build_flat_vector_batch_view_metadata(req)
    bad = dataclasses.replace(meta, output_generation_attempted=True)
    with pytest.raises(ValueError):
        validate_flat_vector_batch_view_metadata(bad)


# 67. result builder validates
def test_p36_67_metadata_validator_rejects_output_attempted():
    req = build_flat_vector_batch_view_request_from_p35_default()
    res = build_flat_vector_batch_view_result(req)
    validate_flat_vector_batch_view_result(res)


# 68. result batch_view_available True
def test_p36_68_result_builder_validates():
    req = build_flat_vector_batch_view_request_from_p35_default()
    res = build_flat_vector_batch_view_result(req)
    assert res.batch_view_available_in_p36 is True


# 69. result nested_values_available False
def test_p36_69_result_batch_view_available():
    req = build_flat_vector_batch_view_request_from_p35_default()
    res = build_flat_vector_batch_view_result(req)
    assert res.nested_values_available_in_p36 is False


# 70. result 2D batch available False
def test_p36_70_result_nested_values_available():
    req = build_flat_vector_batch_view_request_from_p35_default()
    res = build_flat_vector_batch_view_result(req)
    assert res.two_d_batch_available_in_p36 is False


# 71. result array availability False
def test_p36_71_result_2d_batch_available():
    req = build_flat_vector_batch_view_request_from_p35_default()
    res = build_flat_vector_batch_view_result(req)
    assert res.array_materialization_available_in_p36 is False


# 72. result tensor availability False
def test_p36_72_result_array_available():
    req = build_flat_vector_batch_view_request_from_p35_default()
    res = build_flat_vector_batch_view_result(req)
    assert res.tensor_materialization_available_in_p36 is False


# 73. result forward availability False
def test_p36_73_result_tensor_available():
    req = build_flat_vector_batch_view_request_from_p35_default()
    res = build_flat_vector_batch_view_result(req)
    assert res.forward_execution_available_in_p36 is False


# 74. result output availability False
def test_p36_74_result_forward_available():
    req = build_flat_vector_batch_view_request_from_p35_default()
    res = build_flat_vector_batch_view_result(req)
    assert res.output_generation_available_in_p36 is False


# 75. result all no_* flags True
def test_p36_75_result_output_available():
    req = build_flat_vector_batch_view_request_from_p35_default()
    res = build_flat_vector_batch_view_result(req)
    assert res.no_nested_values is True
    assert res.no_2d_batch_materialized is True
    assert res.no_array_created is True
    assert res.no_tensor_created is True
    assert res.no_forward_execution is True
    assert res.no_output_generation is True
    assert res.no_training_loop is True
    assert res.no_optimizer is True
    assert res.no_checkpointing is True
    assert res.no_artifact_generation is True
    assert res.no_final_comparison is True
    assert res.no_scientific_conclusion is True


# 76. result rejects no_nested_values False
def test_p36_76_result_all_no_flags():
    req = build_flat_vector_batch_view_request_from_p35_default()
    res = build_flat_vector_batch_view_result(req)
    bad = dataclasses.replace(res, no_nested_values=False)
    with pytest.raises(ValueError):
        validate_flat_vector_batch_view_result(bad)


# 77. result rejects no_2d_batch_materialized False
def test_p36_77_result_validator_rejects_no_nested_values_false():
    req = build_flat_vector_batch_view_request_from_p35_default()
    res = build_flat_vector_batch_view_result(req)
    bad = dataclasses.replace(res, no_2d_batch_materialized=False)
    with pytest.raises(ValueError):
        validate_flat_vector_batch_view_result(bad)


# 78. result rejects no_array_created False
def test_p36_78_result_validator_rejects_no_2d_batch_false():
    req = build_flat_vector_batch_view_request_from_p35_default()
    res = build_flat_vector_batch_view_result(req)
    bad = dataclasses.replace(res, no_array_created=False)
    with pytest.raises(ValueError):
        validate_flat_vector_batch_view_result(bad)


# 79. result rejects no_tensor_created False
def test_p36_79_result_validator_rejects_no_array_false():
    req = build_flat_vector_batch_view_request_from_p35_default()
    res = build_flat_vector_batch_view_result(req)
    bad = dataclasses.replace(res, no_tensor_created=False)
    with pytest.raises(ValueError):
        validate_flat_vector_batch_view_result(bad)


# 80. result rejects no_forward_execution False
def test_p36_80_result_validator_rejects_no_tensor_false():
    req = build_flat_vector_batch_view_request_from_p35_default()
    res = build_flat_vector_batch_view_result(req)
    bad = dataclasses.replace(res, no_forward_execution=False)
    with pytest.raises(ValueError):
        validate_flat_vector_batch_view_result(bad)


# 81. result rejects no_output_generation False
def test_p36_81_result_validator_rejects_no_forward_false():
    req = build_flat_vector_batch_view_request_from_p35_default()
    res = build_flat_vector_batch_view_result(req)
    bad = dataclasses.replace(res, no_output_generation=False)
    with pytest.raises(ValueError):
        validate_flat_vector_batch_view_result(bad)


# 82. result rejects no_training_loop False
def test_p36_82_result_validator_rejects_no_output_false():
    req = build_flat_vector_batch_view_request_from_p35_default()
    res = build_flat_vector_batch_view_result(req)
    bad = dataclasses.replace(res, no_training_loop=False)
    with pytest.raises(ValueError):
        validate_flat_vector_batch_view_result(bad)


# 83. result rejects no_optimizer False
def test_p36_83_result_validator_rejects_no_training_false():
    req = build_flat_vector_batch_view_request_from_p35_default()
    res = build_flat_vector_batch_view_result(req)
    bad = dataclasses.replace(res, no_optimizer=False)
    with pytest.raises(ValueError):
        validate_flat_vector_batch_view_result(bad)


# 84. result rejects no_checkpointing False
def test_p36_84_result_validator_rejects_no_optimizer_false():
    req = build_flat_vector_batch_view_request_from_p34_default = build_flat_vector_batch_view_request_from_p35_default()
    res = build_flat_vector_batch_view_result(req)
    bad = dataclasses.replace(res, no_checkpointing=False)
    with pytest.raises(ValueError):
        validate_flat_vector_batch_view_result(bad)


# 85. result rejects no_artifact_generation False
def test_p36_85_result_validator_rejects_no_checkpoint_false():
    req = build_flat_vector_batch_view_request_from_p35_default()
    res = build_flat_vector_batch_view_result(req)
    bad = dataclasses.replace(res, no_artifact_generation=False)
    with pytest.raises(ValueError):
        validate_flat_vector_batch_view_result(bad)


# 86. result rejects no_final_comparison False
def test_p36_86_result_validator_rejects_no_artifact_false():
    req = build_flat_vector_batch_view_request_from_p35_default()
    res = build_flat_vector_batch_view_result(req)
    bad = dataclasses.replace(res, no_final_comparison=False)
    with pytest.raises(ValueError):
        validate_flat_vector_batch_view_result(bad)


# 87. result rejects no_scientific_conclusion False
def test_p36_87_result_validator_rejects_no_final_comparison_false():
    req = build_flat_vector_batch_view_request_from_p35_default()
    res = build_flat_vector_batch_view_result(req)
    bad = dataclasses.replace(res, no_scientific_conclusion=False)
    with pytest.raises(ValueError):
        validate_flat_vector_batch_view_result(bad)


# 88. probe returns valid result
def test_p36_88_result_validator_rejects_no_conclusion_false():
    res = run_flat_vector_batch_view_probe()
    validate_flat_vector_batch_view_result(res)


# 89. serializers JSON-safe
def test_p36_89_probe():
    res = run_flat_vector_batch_view_probe()
    d1 = flat_vector_batch_view_request_to_json_dict(res.request)
    d2 = flat_vector_batch_view_shape_to_json_dict(res.shape)
    d3 = flat_vector_batch_view_metadata_to_json_dict(res.metadata)
    d4 = flat_vector_batch_view_result_to_json_dict(res)
    for d in (d1, d2, d3, d4):
        assert isinstance(d, dict)


# 90. compact JSON parseable/sorted
def test_p36_90_serializers():
    res = run_flat_vector_batch_view_probe()
    js = compact_flat_vector_batch_view_json(res)
    assert isinstance(js, str)
    d = json.loads(js)
    assert d["contract_version"] == FC_VAE_FLAT_VECTOR_BATCH_VIEW_CONTRACT_VERSION


# 91. serialized result has shape_tuple as [2, 32]
def test_p36_91_compact_json():
    res = run_flat_vector_batch_view_probe()
    js = compact_flat_vector_batch_view_json(res)
    d = json.loads(js)
    assert d["shape"]["shape_tuple"] == [2, 32]
    assert d["request"]["shape_tuple"] == [2, 32]
    assert d["metadata"]["shape_tuple"] == [2, 32]


# 92. serialized result has no flat_values key
def test_p36_92_serialized_shape_tuple_is_list():
    res = run_flat_vector_batch_view_probe()
    js = compact_flat_vector_batch_view_json(res)
    assert "flat_values" not in js


# 93. serialized result has no nested lists except shape_tuple
def test_p36_93_serialized_has_no_flat_values_or_nested_values():
    res = run_flat_vector_batch_view_probe()
    js = compact_flat_vector_batch_view_json(res)
    # the only nested lists should be the shape_tuple ([2, 32])
    # check that we do not have nested value structures
    assert "[[" not in js


# 94. serialized result has no key with ": bool"
def test_p36_94_serialized_no_key_pollution():
    res = run_flat_vector_batch_view_probe()
    js = compact_flat_vector_batch_view_json(res)
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


# 95. module has no torch/numpy/random/secrets/array import
def test_p36_95_module_import_boundary():
    content = pathlib.Path("src/phase2/fc_vae_flat_vector_batch_view.py").read_text(encoding="utf-8")
    for pattern in ("import torch", "from torch", "import numpy", "from numpy", "import random", "from random", "import secrets", "from secrets", "import array", "from array"):
        assert pattern not in content
    # constructors
    for pattern in ("torch.tensor", "np.array", "np.zeros", "torch.zeros", "random.uniform", "import array", "from array"):
        assert pattern not in content
    # def forward
    assert "def forward(" not in content
    assert "def forward " not in content
    assert "forward(" not in content
    # train/fit/loss/optimizer/checkpoint functions
    for pattern in ("def train", "def fit", "def loss", "def optimizer", "def checkpoint"):
        assert pattern not in content


# 96. P36-owned scope gate compares against accepted P35 branch
def test_p36_96_scope_gate():
    allowed = {
        "src/phase2/fc_vae_flat_vector_batch_view.py",
        "src/phase2/__init__.py",
        "tests/test_phase2_fc_vae_flat_vector_batch_view.py",
        "tools/phase2/run_p36_flat_vector_batch_view_smoke.py",
        "tests/test_phase2_p36_flat_vector_batch_view_smoke.py",
        "reports/PHASE_2_P36_FLAT_VECTOR_2D_BATCH_VIEW_CONTRACT_NO_NESTED_VALUES_NO_TENSOR_NO_ARRAY_REPORT.md",
    }
    # Base branch check
    res = subprocess.run(
        ["git", "diff", "--name-only", "phase2/p35-deterministic-fake-input-flat-vector-no-tensor"],
        capture_output=True, text=True, check=True
    )
    modified = [line.strip() for line in res.stdout.splitlines() if line.strip()]
    for f in modified:
        f_norm = f.replace("\\", "/")
        assert f_norm in allowed, f"Forbidden file modification detected in P36: {f_norm}"
