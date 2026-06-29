# tests/test_phase2_fc_vae_fake_input_preview.py

import dataclasses
import json
import pathlib
import subprocess
from tests.phase2_scope_gate_utils import enforce_phase_local_scope_gate_or_skip
import pytest
from dataclasses import is_dataclass

from src.phase2.fc_vae_fake_input_preview import (
    FC_VAE_FAKE_INPUT_PREVIEW_CONTRACT_VERSION,
    FC_VAE_FAKE_INPUT_PREVIEW_KIND,
    FC_VAE_FAKE_INPUT_PREVIEW_MODULE_NAME,
    FC_VAE_FAKE_INPUT_PREVIEW_STATUS_TORCH_UNAVAILABLE,
    FC_VAE_FAKE_INPUT_PREVIEW_STATUS_DESCRIPTOR_BLOCKED,
    FC_VAE_FAKE_INPUT_PREVIEW_STATUS_PREVIEW_ONLY,
    SUPPORTED_FC_VAE_FAKE_INPUT_PREVIEW_STATUSES,
    FC_VAE_FAKE_INPUT_PREVIEW_VALUE_KIND,
    DEFAULT_P34_PREVIEW_VALUE_COUNT,
    MIN_P34_PREVIEW_VALUE_COUNT,
    MAX_P34_PREVIEW_VALUE_COUNT,
    FCVAEFakeInputPreviewRequest,
    FCVAEFakeInputScalarPreview,
    FCVAEFakeInputPreviewMetadata,
    FCVAEFakeInputPreviewResult,
    validate_non_empty_str,
    validate_bool,
    validate_non_negative_int,
    validate_positive_int,
    validate_numeric_value,
    validate_value_range,
    validate_preview_value_count,
    validate_preview_values_tuple,
    validate_fake_input_preview_status,
    assert_no_local_path_leakage,
    assert_no_forbidden_claims,
    validate_fake_input_preview_request,
    validate_fake_input_scalar_preview,
    validate_fake_input_preview_metadata,
    validate_fake_input_preview_result,
    compute_deterministic_preview_values,
    build_fake_input_preview_request_from_p33_default,
    build_fake_input_scalar_preview,
    build_fake_input_preview_metadata,
    build_fake_input_preview_result,
    run_fake_input_preview_probe,
    fake_input_preview_request_to_json_dict,
    fake_input_scalar_preview_to_json_dict,
    fake_input_preview_metadata_to_json_dict,
    fake_input_preview_result_to_json_dict,
    compact_fake_input_preview_json,
)


# 1. constants exact
def test_p34_01_constants():
    assert FC_VAE_FAKE_INPUT_PREVIEW_CONTRACT_VERSION == "phase2_p34_fake_input_scalar_preview_contract_v1"
    assert FC_VAE_FAKE_INPUT_PREVIEW_KIND == "deterministic_fake_input_scalar_preview_no_tensor_no_array"
    assert FC_VAE_FAKE_INPUT_PREVIEW_MODULE_NAME == "src.phase2.fc_vae_fake_input_preview"
    assert FC_VAE_FAKE_INPUT_PREVIEW_STATUS_TORCH_UNAVAILABLE == "blocked_torch_unavailable"
    assert FC_VAE_FAKE_INPUT_PREVIEW_STATUS_DESCRIPTOR_BLOCKED == "blocked_by_descriptor_status"
    assert FC_VAE_FAKE_INPUT_PREVIEW_STATUS_PREVIEW_ONLY == "scalar_preview_only_no_full_batch_in_p34"
    assert SUPPORTED_FC_VAE_FAKE_INPUT_PREVIEW_STATUSES == (
        "blocked_torch_unavailable",
        "blocked_by_descriptor_status",
        "scalar_preview_only_no_full_batch_in_p34",
    )
    assert FC_VAE_FAKE_INPUT_PREVIEW_VALUE_KIND == "bounded_deterministic_scalar_preview"
    assert DEFAULT_P34_PREVIEW_VALUE_COUNT == 4
    assert MIN_P34_PREVIEW_VALUE_COUNT == 1
    assert MAX_P34_PREVIEW_VALUE_COUNT == 8


# 2. dataclasses frozen
def test_p34_02_dataclasses_frozen():
    for cls in (
        FCVAEFakeInputPreviewRequest,
        FCVAEFakeInputScalarPreview,
        FCVAEFakeInputPreviewMetadata,
        FCVAEFakeInputPreviewResult,
    ):
        assert is_dataclass(cls)
        req = build_fake_input_preview_request_from_p33_default()
        with pytest.raises(Exception):
            req.reason = "modified"


# 3. primitive validators accept/reject invalid values
def test_p34_03_primitive_validators():
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
def test_p34_04_numeric_validator():
    validate_numeric_value(1, "val")
    validate_numeric_value(-1.5, "val")
    with pytest.raises(TypeError):
        validate_numeric_value(True, "val")
    with pytest.raises(TypeError):
        validate_numeric_value("1.5", "val")


# 5. value range validator rejects min >= max
def test_p34_05_value_range_validator():
    validate_value_range(-1.0, 1.0)
    with pytest.raises(ValueError):
        validate_value_range(1.0, -1.0)
    with pytest.raises(ValueError):
        validate_value_range(0.0, 0.0)


# 6. preview count validator accepts min/default/max
def test_p34_06_preview_count_validator_accepts():
    validate_preview_value_count(MIN_P34_PREVIEW_VALUE_COUNT)
    validate_preview_value_count(DEFAULT_P34_PREVIEW_VALUE_COUNT)
    validate_preview_value_count(MAX_P34_PREVIEW_VALUE_COUNT)


# 7. preview count validator rejects 0 and MAX+1
def test_p34_07_preview_count_validator_rejects():
    with pytest.raises(ValueError):
        validate_preview_value_count(0)
    with pytest.raises(ValueError):
        validate_preview_value_count(MAX_P34_PREVIEW_VALUE_COUNT + 1)
    with pytest.raises(TypeError):
        validate_preview_value_count(2.5)
    with pytest.raises(TypeError):
        validate_preview_value_count(True)


# 8. preview tuple validator rejects list/non-float/wrong length/out-of-range
def test_p34_08_preview_tuple_validator():
    validate_preview_values_tuple((0.1, 0.2), 0.0, 1.0, 2)
    with pytest.raises(TypeError):
        validate_preview_values_tuple([0.1, 0.2], 0.0, 1.0, 2)
    with pytest.raises(ValueError):
        validate_preview_values_tuple((0.1, 0.2), 0.0, 1.0, 3)
    with pytest.raises(TypeError):
        validate_preview_values_tuple((0.1, "0.2"), 0.0, 1.0, 2)
    with pytest.raises(ValueError):
        validate_preview_values_tuple((0.1, 1.5), 0.0, 1.0, 2)


# 9. status validator accepts all supported statuses
def test_p34_09_status_validator():
    for status in SUPPORTED_FC_VAE_FAKE_INPUT_PREVIEW_STATUSES:
        validate_fake_input_preview_status(status)
    with pytest.raises(ValueError):
        validate_fake_input_preview_status("unknown")


# 10. local path and forbidden claim guards work
def test_p34_10_guards():
    assert_no_local_path_leakage("safe string")
    for pattern in ("file:///", "C:/", "C:\\", "/home/", "/Users/"):
        with pytest.raises(ValueError):
            assert_no_local_path_leakage(f"leak {pattern}")

    assert_no_forbidden_claims("safe conclusion")
    # Allowed
    assert_no_forbidden_claims("no_scientific_conclusion")
    assert_no_forbidden_claims("no_final_comparison")
    assert_no_forbidden_claims("scalar_preview_only_no_full_batch_in_p34")
    # Forbidden
    for claim in ("model works", "scientific success", "solved", "best", "winner", "production ready", "state of the art"):
        with pytest.raises(ValueError):
            assert_no_forbidden_claims(f"our {claim} claim")


# 11. deterministic formula returns same values for same inputs
def test_p34_11_formula_deterministic():
    v1 = compute_deterministic_preview_values(1337, -1.0, 1.0, 4)
    v2 = compute_deterministic_preview_values(1337, -1.0, 1.0, 4)
    assert v1 == v2


# 12. deterministic formula changes when seed changes
def test_p34_12_formula_changes_with_seed():
    v1 = compute_deterministic_preview_values(1337, -1.0, 1.0, 4)
    v2 = compute_deterministic_preview_values(1338, -1.0, 1.0, 4)
    assert v1 != v2


# 13. deterministic formula returns exact float tuple
def test_p34_13_formula_returns_float_tuple():
    v = compute_deterministic_preview_values(1337, -1.0, 1.0, 4)
    assert isinstance(v, tuple)
    assert len(v) == 4
    for item in v:
        assert isinstance(item, float)


# 14. deterministic formula values are within range
def test_p34_14_formula_values_within_range():
    v = compute_deterministic_preview_values(1337, -1.0, 1.0, 8)
    for val in v:
        assert val >= -1.0
        assert val <= 1.0


# 15. request builder returns seed=1337, min=-1.0, max=1.0
def test_p34_15_request_defaults():
    req = build_fake_input_preview_request_from_p33_default()
    assert req.descriptor_seed == 1337
    assert req.min_value == -1.0
    assert req.max_value == 1.0


# 16. request builder returns batch_size=2 and input_flat_dim=32
def test_p34_16_request_dimensions():
    req = build_fake_input_preview_request_from_p33_default()
    assert req.batch_size == 2
    assert req.input_flat_dim == 32


# 17. request builder returns full_batch_scalar_count=64
def test_p34_17_request_full_count():
    req = build_fake_input_preview_request_from_p33_default()
    assert req.full_batch_scalar_count == 64


# 18. request builder returns preview_value_count=4
def test_p34_18_request_preview_count():
    req = build_fake_input_preview_request_from_p33_default()
    assert req.preview_value_count == 4


# 19. request rejects wrong contract
def test_p34_19_request_wrong_contract():
    req = build_fake_input_preview_request_from_p33_default()
    bad = dataclasses.replace(req, contract_version="wrong")
    with pytest.raises(ValueError):
        validate_fake_input_preview_request(bad)


# 20. request rejects wrong preview kind
def test_p34_20_request_wrong_preview_kind():
    req = build_fake_input_preview_request_from_p33_default()
    bad = dataclasses.replace(req, preview_kind="wrong")
    with pytest.raises(ValueError):
        validate_fake_input_preview_request(bad)


# 21. request rejects wrong source descriptor contract
def test_p34_21_request_wrong_source_descriptor():
    req = build_fake_input_preview_request_from_p33_default()
    bad = dataclasses.replace(req, source_descriptor_contract_version="wrong")
    with pytest.raises(ValueError):
        validate_fake_input_preview_request(bad)


# 22. request rejects negative seed
def test_p34_22_request_negative_seed():
    req = build_fake_input_preview_request_from_p33_default()
    bad = dataclasses.replace(req, descriptor_seed=-1)
    with pytest.raises(ValueError):
        validate_fake_input_preview_request(bad)


# 23. request rejects min >= max
def test_p34_23_request_range_invalid():
    req = build_fake_input_preview_request_from_p33_default()
    bad = dataclasses.replace(req, min_value=1.0, max_value=0.5)
    with pytest.raises(ValueError):
        validate_fake_input_preview_request(bad)


# 24. request rejects preview_value_count >= full_batch_scalar_count
def test_p34_24_request_preview_count_too_large():
    req = build_fake_input_preview_request_from_p33_default()
    bad = dataclasses.replace(req, preview_value_count=64)
    with pytest.raises(ValueError):
        validate_fake_input_preview_request(bad)


# 25. request rejects allow_rng_execution True
def test_p34_25_request_allow_rng():
    req = build_fake_input_preview_request_from_p33_default()
    bad = dataclasses.replace(req, allow_rng_execution_in_p34=True)
    with pytest.raises(ValueError):
        validate_fake_input_preview_request(bad)


# 26. request rejects allow_full_batch_materialization True
def test_p34_26_request_allow_full_batch():
    req = build_fake_input_preview_request_from_p33_default()
    bad = dataclasses.replace(req, allow_full_batch_materialization_in_p34=True)
    with pytest.raises(ValueError):
        validate_fake_input_preview_request(bad)


# 27. request rejects allow_array_materialization True
def test_p34_27_request_allow_array():
    req = build_fake_input_preview_request_from_p33_default()
    bad = dataclasses.replace(req, allow_array_materialization_in_p34=True)
    with pytest.raises(ValueError):
        validate_fake_input_preview_request(bad)


# 28. request rejects allow_tensor_materialization True
def test_p34_28_request_allow_tensor():
    req = build_fake_input_preview_request_from_p33_default()
    bad = dataclasses.replace(req, allow_tensor_materialization_in_p34=True)
    with pytest.raises(ValueError):
        validate_fake_input_preview_request(bad)


# 29. request rejects allow_forward_execution True
def test_p34_29_request_allow_forward():
    req = build_fake_input_preview_request_from_p33_default()
    bad = dataclasses.replace(req, allow_forward_execution_in_p34=True)
    with pytest.raises(ValueError):
        validate_fake_input_preview_request(bad)


# 30. preview builder validates preview count and values
def test_p34_30_preview_builder():
    req = build_fake_input_preview_request_from_p33_default()
    preview = build_fake_input_scalar_preview(req)
    validate_fake_input_scalar_preview(preview)
    assert preview.preview_value_count == 4
    assert len(preview.preview_values) == 4


# 31. preview rejects wrong value_kind
def test_p34_31_preview_wrong_kind():
    req = build_fake_input_preview_request_from_p33_default()
    preview = build_fake_input_scalar_preview(req)
    bad = dataclasses.replace(preview, value_kind="wrong")
    with pytest.raises(ValueError):
        validate_fake_input_scalar_preview(bad)


# 32. preview rejects mismatched preview count
def test_p34_32_preview_count_mismatch():
    req = build_fake_input_preview_request_from_p33_default()
    preview = build_fake_input_scalar_preview(req)
    bad = dataclasses.replace(preview, preview_value_count=5)
    with pytest.raises(ValueError):
        validate_fake_input_scalar_preview(bad)


# 33. preview rejects empty values
def test_p34_33_preview_empty_values():
    req = build_fake_input_preview_request_from_p33_default()
    preview = build_fake_input_scalar_preview(req)
    bad = dataclasses.replace(preview, preview_values=(), preview_value_count=0)
    with pytest.raises(ValueError):
        validate_fake_input_scalar_preview(bad)


# 34. preview rejects values outside range
def test_p34_34_preview_values_out_of_range():
    req = build_fake_input_preview_request_from_p33_default()
    preview = build_fake_input_scalar_preview(req)
    bad = dataclasses.replace(preview, preview_values=(1.5, 0.1, 0.2, 0.3))
    with pytest.raises(ValueError):
        validate_fake_input_scalar_preview(bad)


# 35. preview rejects all_values_within_range False
def test_p34_35_preview_all_within_range_false():
    req = build_fake_input_preview_request_from_p33_default()
    preview = build_fake_input_scalar_preview(req)
    bad = dataclasses.replace(preview, all_values_within_range=False)
    with pytest.raises(ValueError):
        validate_fake_input_scalar_preview(bad)


# 36. preview rejects rng_executed True
def test_p34_36_preview_rng_executed():
    req = build_fake_input_preview_request_from_p33_default()
    preview = build_fake_input_scalar_preview(req)
    bad = dataclasses.replace(preview, rng_executed=True)
    with pytest.raises(ValueError):
        validate_fake_input_scalar_preview(bad)


# 37. preview rejects full_batch_materialized True
def test_p34_37_preview_full_batch_materialized():
    req = build_fake_input_preview_request_from_p33_default()
    preview = build_fake_input_scalar_preview(req)
    bad = dataclasses.replace(preview, full_batch_materialized=True)
    with pytest.raises(ValueError):
        validate_fake_input_scalar_preview(bad)


# 38. preview rejects array_materialized True
def test_p34_38_preview_array_materialized():
    req = build_fake_input_preview_request_from_p33_default()
    preview = build_fake_input_scalar_preview(req)
    bad = dataclasses.replace(preview, array_materialized=True)
    with pytest.raises(ValueError):
        validate_fake_input_scalar_preview(bad)


# 39. preview rejects tensor_materialized True
def test_p34_39_preview_tensor_materialized():
    req = build_fake_input_preview_request_from_p33_default()
    preview = build_fake_input_scalar_preview(req)
    bad = dataclasses.replace(preview, tensor_materialized=True)
    with pytest.raises(ValueError):
        validate_fake_input_scalar_preview(bad)


# 40. metadata builder validates
def test_p34_40_metadata_builder():
    req = build_fake_input_preview_request_from_p33_default()
    meta = build_fake_input_preview_metadata(req)
    validate_fake_input_preview_metadata(meta)
    assert meta.preview_is_partial is True


# 41. metadata rejects preview_is_partial False
def test_p34_41_metadata_partial_false():
    req = build_fake_input_preview_request_from_p33_default()
    meta = build_fake_input_preview_metadata(req)
    bad = dataclasses.replace(meta, preview_is_partial=False)
    with pytest.raises(ValueError):
        validate_fake_input_preview_metadata(bad)


# 42. metadata rejects preview_value_count >= full_batch_scalar_count
def test_p34_42_metadata_count_mismatch():
    req = build_fake_input_preview_request_from_p33_default()
    meta = build_fake_input_preview_metadata(req)
    bad = dataclasses.replace(meta, preview_value_count=64)
    with pytest.raises(ValueError):
        validate_fake_input_preview_metadata(bad)


# 43. metadata rejects rng_execution_attempted True
def test_p34_43_metadata_rng_attempted():
    req = build_fake_input_preview_request_from_p33_default()
    meta = build_fake_input_preview_metadata(req)
    bad = dataclasses.replace(meta, rng_execution_attempted=True)
    with pytest.raises(ValueError):
        validate_fake_input_preview_metadata(bad)


# 44. metadata rejects full_batch_materialization_attempted True
def test_p34_44_metadata_full_attempted():
    req = build_fake_input_preview_request_from_p33_default()
    meta = build_fake_input_preview_metadata(req)
    bad = dataclasses.replace(meta, full_batch_materialization_attempted=True)
    with pytest.raises(ValueError):
        validate_fake_input_preview_metadata(bad)


# 45. metadata rejects array_materialization_attempted True
def test_p34_45_metadata_array_attempted():
    req = build_fake_input_preview_request_from_p33_default()
    meta = build_fake_input_preview_metadata(req)
    bad = dataclasses.replace(meta, array_materialization_attempted=True)
    with pytest.raises(ValueError):
        validate_fake_input_preview_metadata(bad)


# 46. metadata rejects tensor_materialization_attempted True
def test_p34_46_metadata_tensor_attempted():
    req = build_fake_input_preview_request_from_p33_default()
    meta = build_fake_input_preview_metadata(req)
    bad = dataclasses.replace(meta, tensor_materialization_attempted=True)
    with pytest.raises(ValueError):
        validate_fake_input_preview_metadata(bad)


# 47. metadata rejects forward_execution_attempted True
def test_p34_47_metadata_forward_attempted():
    req = build_fake_input_preview_request_from_p33_default()
    meta = build_fake_input_preview_metadata(req)
    bad = dataclasses.replace(meta, forward_execution_attempted=True)
    with pytest.raises(ValueError):
        validate_fake_input_preview_metadata(bad)


# 48. metadata rejects output_generation_attempted True
def test_p34_48_metadata_output_attempted():
    req = build_fake_input_preview_request_from_p33_default()
    meta = build_fake_input_preview_metadata(req)
    bad = dataclasses.replace(meta, output_generation_attempted=True)
    with pytest.raises(ValueError):
        validate_fake_input_preview_metadata(bad)


# 49. result builder validates
def test_p34_49_result_builder():
    req = build_fake_input_preview_request_from_p33_default()
    res = build_fake_input_preview_result(req)
    validate_fake_input_preview_result(res)


# 50. result rejects preview_available_in_p34 False
def test_p34_50_result_preview_unavailable():
    req = build_fake_input_preview_request_from_p33_default()
    res = build_fake_input_preview_result(req)
    bad = dataclasses.replace(res, preview_available_in_p34=False)
    with pytest.raises(ValueError):
        validate_fake_input_preview_result(bad)


# 51. result rejects full_batch_available_in_p34 True
def test_p34_51_result_full_batch_available():
    req = build_fake_input_preview_request_from_p33_default()
    res = build_fake_input_preview_result(req)
    bad = dataclasses.replace(res, full_batch_available_in_p34=True)
    with pytest.raises(ValueError):
        validate_fake_input_preview_result(bad)


# 52. result rejects rng_execution_available True
def test_p34_52_result_rng_available():
    req = build_fake_input_preview_request_from_p33_default()
    res = build_fake_input_preview_result(req)
    bad = dataclasses.replace(res, rng_execution_available_in_p34=True)
    with pytest.raises(ValueError):
        validate_fake_input_preview_result(bad)


# 53. result rejects array_materialization_available True
def test_p34_53_result_array_available():
    req = build_fake_input_preview_request_from_p33_default()
    res = build_fake_input_preview_result(req)
    bad = dataclasses.replace(res, array_materialization_available_in_p34=True)
    with pytest.raises(ValueError):
        validate_fake_input_preview_result(bad)


# 54. result rejects tensor_materialization_available True
def test_p34_54_result_tensor_available():
    req = build_fake_input_preview_request_from_p33_default()
    res = build_fake_input_preview_result(req)
    bad = dataclasses.replace(res, tensor_materialization_available_in_p34=True)
    with pytest.raises(ValueError):
        validate_fake_input_preview_result(bad)


# 55. result rejects forward_execution_available True
def test_p34_55_result_forward_available():
    req = build_fake_input_preview_request_from_p33_default()
    res = build_fake_input_preview_result(req)
    bad = dataclasses.replace(res, forward_execution_available_in_p34=True)
    with pytest.raises(ValueError):
        validate_fake_input_preview_result(bad)


# 56. result rejects output_generation_available True
def test_p34_56_result_output_available():
    req = build_fake_input_preview_request_from_p33_default()
    res = build_fake_input_preview_result(req)
    bad = dataclasses.replace(res, output_generation_available_in_p34=True)
    with pytest.raises(ValueError):
        validate_fake_input_preview_result(bad)


# 57. result rejects no_rng_execution False
def test_p34_57_result_no_rng():
    req = build_fake_input_preview_request_from_p33_default()
    res = build_fake_input_preview_result(req)
    bad = dataclasses.replace(res, no_rng_execution=False)
    with pytest.raises(ValueError):
        validate_fake_input_preview_result(bad)


# 58. result rejects no_full_batch_materialized False
def test_p34_58_result_no_full_batch():
    req = build_fake_input_preview_request_from_p33_default()
    res = build_fake_input_preview_result(req)
    bad = dataclasses.replace(res, no_full_batch_materialized=False)
    with pytest.raises(ValueError):
        validate_fake_input_preview_result(bad)


# 59. result rejects no_array_created False
def test_p34_59_result_no_array():
    req = build_fake_input_preview_request_from_p33_default()
    res = build_fake_input_preview_result(req)
    bad = dataclasses.replace(res, no_array_created=False)
    with pytest.raises(ValueError):
        validate_fake_input_preview_result(bad)


# 60. result rejects no_tensor_created False
def test_p34_60_result_no_tensor():
    req = build_fake_input_preview_request_from_p33_default()
    res = build_fake_input_preview_result(req)
    bad = dataclasses.replace(res, no_tensor_created=False)
    with pytest.raises(ValueError):
        validate_fake_input_preview_result(bad)


# 61. result rejects no_forward_execution False
def test_p34_61_result_no_forward():
    req = build_fake_input_preview_request_from_p33_default()
    res = build_fake_input_preview_result(req)
    bad = dataclasses.replace(res, no_forward_execution=False)
    with pytest.raises(ValueError):
        validate_fake_input_preview_result(bad)


# 62. result rejects no_output_generation False
def test_p34_62_result_no_output():
    req = build_fake_input_preview_request_from_p33_default()
    res = build_fake_input_preview_result(req)
    bad = dataclasses.replace(res, no_output_generation=False)
    with pytest.raises(ValueError):
        validate_fake_input_preview_result(bad)


# 63. result rejects no_training_loop False
def test_p34_63_result_no_training():
    req = build_fake_input_preview_request_from_p33_default()
    res = build_fake_input_preview_result(req)
    bad = dataclasses.replace(res, no_training_loop=False)
    with pytest.raises(ValueError):
        validate_fake_input_preview_result(bad)


# 64. result rejects no_optimizer False
def test_p34_64_result_no_optimizer():
    req = build_fake_input_preview_request_from_p33_default()
    res = build_fake_input_preview_result(req)
    bad = dataclasses.replace(res, no_optimizer=False)
    with pytest.raises(ValueError):
        validate_fake_input_preview_result(bad)


# 65. result rejects no_checkpointing False
def test_p34_65_result_no_checkpoint():
    req = build_fake_input_preview_request_from_p33_default()
    res = build_fake_input_preview_result(req)
    bad = dataclasses.replace(res, no_checkpointing=False)
    with pytest.raises(ValueError):
        validate_fake_input_preview_result(bad)


# 66. result rejects no_artifact_generation False
def test_p34_66_result_no_artifact():
    req = build_fake_input_preview_request_from_p33_default()
    res = build_fake_input_preview_result(req)
    bad = dataclasses.replace(res, no_artifact_generation=False)
    with pytest.raises(ValueError):
        validate_fake_input_preview_result(bad)


# 67. result rejects no_final_comparison False
def test_p34_67_result_no_final():
    req = build_fake_input_preview_request_from_p33_default()
    res = build_fake_input_preview_result(req)
    bad = dataclasses.replace(res, no_final_comparison=False)
    with pytest.raises(ValueError):
        validate_fake_input_preview_result(bad)


# 68. result rejects no_scientific_conclusion False
def test_p34_68_result_no_conclusion():
    req = build_fake_input_preview_request_from_p33_default()
    res = build_fake_input_preview_result(req)
    bad = dataclasses.replace(res, no_scientific_conclusion=False)
    with pytest.raises(ValueError):
        validate_fake_input_preview_result(bad)


# 69. probe returns valid result
def test_p34_69_probe():
    res = run_fake_input_preview_probe()
    assert isinstance(res, FCVAEFakeInputPreviewResult)
    validate_fake_input_preview_result(res)


# 70. serializers return JSON-safe dict
def test_p34_70_serializers():
    req = build_fake_input_preview_request_from_p33_default()
    res = build_fake_input_preview_result(req)
    d1 = fake_input_preview_request_to_json_dict(req)
    d2 = fake_input_scalar_preview_to_json_dict(res.preview)
    d3 = fake_input_preview_metadata_to_json_dict(res.metadata)
    d4 = fake_input_preview_result_to_json_dict(res)
    for d in (d1, d2, d3, d4):
        assert isinstance(d, dict)


# 71. compact JSON sorted/parseable
def test_p34_71_compact_json():
    res = run_fake_input_preview_probe()
    js = compact_fake_input_preview_json(res)
    assert isinstance(js, str)
    d = json.loads(js)
    assert d["contract_version"] == FC_VAE_FAKE_INPUT_PREVIEW_CONTRACT_VERSION


# 72. serialized result has no local paths
def test_p34_72_serialized_no_local_paths():
    res = run_fake_input_preview_probe()
    js = compact_fake_input_preview_json(res)
    for pattern in ("file:///", "C:/", "C:\\", "/home/", "/Users/"):
        assert pattern not in js
        assert pattern.lower() not in js.lower()


# 73. serialized result has no forbidden success claims
def test_p34_73_serialized_no_forbidden_claims():
    res = run_fake_input_preview_probe()
    js = compact_fake_input_preview_json(res)
    normalized = js.lower()
    cleaned = (normalized
               .replace("no_final_comparison", "")
               .replace("no_scientific_conclusion", "")
               .replace("scalar_preview_only_no_full_batch_in_p34", ""))
    for claim in ("model works", "scientific success", "solved", "best", "winner", "production ready", "state of the art"):
        assert claim not in cleaned


# 74. serialized result has preview_values as JSON list
def test_p34_74_serialized_values_list():
    res = run_fake_input_preview_probe()
    js = compact_fake_input_preview_json(res)
    d = json.loads(js)
    assert isinstance(d["preview"]["preview_values"], list)
    assert len(d["preview"]["preview_values"]) == 4


# 75. serialized result has no full batch payload
def test_p34_75_serialized_no_full_batch():
    res = run_fake_input_preview_probe()
    js = compact_fake_input_preview_json(res)
    d = json.loads(js)
    assert "full_batch" not in d
    assert "batch_data" not in d


# 76. serialized result has no nested 2D values
def test_p34_76_serialized_no_nested_2d():
    res = run_fake_input_preview_probe()
    js = compact_fake_input_preview_json(res)
    assert "[[" not in js


# 77. module has no torch import
def test_p34_77_no_torch_import():
    content = pathlib.Path("src/phase2/fc_vae_fake_input_preview.py").read_text(encoding="utf-8")
    assert "import torch" not in content
    assert "from torch" not in content


# 78. module has no numpy import
def test_p34_78_no_numpy_import():
    content = pathlib.Path("src/phase2/fc_vae_fake_input_preview.py").read_text(encoding="utf-8")
    assert "import numpy" not in content
    assert "from numpy" not in content


# 79. module has no random/secrets import
def test_p34_79_no_random_secrets_import():
    content = pathlib.Path("src/phase2/fc_vae_fake_input_preview.py").read_text(encoding="utf-8")
    for pattern in ("random", "secrets"):
        assert f"import {pattern}" not in content
        assert f"from {pattern}" not in content


# 80. module has no `def forward`
def test_p34_80_no_def_forward():
    content = pathlib.Path("src/phase2/fc_vae_fake_input_preview.py").read_text(encoding="utf-8")
    assert "def forward(" not in content
    assert "def forward " not in content


# 81. module has no `forward(` definition
def test_p34_81_no_forward_call():
    content = pathlib.Path("src/phase2/fc_vae_fake_input_preview.py").read_text(encoding="utf-8")
    assert "forward(" not in content


# 82. module has no tensor/array/full-batch constructors
def test_p34_82_no_constructors():
    content = pathlib.Path("src/phase2/fc_vae_fake_input_preview.py").read_text(encoding="utf-8")
    for pattern in ("torch.tensor", "np.array", "np.zeros", "torch.zeros", "random.uniform", "import array", "from array"):
        assert pattern not in content


# 83. module has no train/fit/loss/optimizer/checkpoint functions
def test_p34_83_no_training_functions():
    content = pathlib.Path("src/phase2/fc_vae_fake_input_preview.py").read_text(encoding="utf-8")
    for pattern in ("def train", "def fit", "def loss", "def optimizer", "def checkpoint"):
        assert pattern not in content


# 84. smoke script has no torch/numpy/random/secrets import
def test_p34_84_smoke_script_imports():
    content = pathlib.Path("tools/phase2/run_p34_fake_input_preview_smoke.py").read_text(encoding="utf-8")
    for pattern in ("torch", "numpy", "random", "secrets"):
        assert f"import {pattern}" not in content
        assert f"from {pattern}" not in content


# 85. `src/phase2/__init__.py` has no monkeypatch symbols
def test_p34_85_init_no_monkeypatch():
    content = pathlib.Path("src/phase2/__init__.py").read_text(encoding="utf-8")
    assert "subprocess.run =" not in content
    assert "_patched_run" not in content
    assert "_original_run" not in content


# 86. P34-owned scope gate compares against accepted P33 branch
def test_p34_86_scope_gate():
    """Phase-local scope gate for P34. Skips on non-P34 branches."""
    allowed = {
        "src/phase2/fc_vae_fake_input_preview.py",
        "src/phase2/__init__.py",
        "tests/test_phase2_fc_vae_fake_input_preview.py",
        "tools/phase2/run_p34_fake_input_preview_smoke.py",
        "tests/test_phase2_p34_fake_input_preview_smoke.py",
        "reports/PHASE_2_P34_DETERMINISTIC_FAKE_INPUT_SCALAR_PREVIEW_NO_TENSOR_NO_ARRAY_REPORT.md",
    }
    enforce_phase_local_scope_gate_or_skip(
        expected_branch="phase2/p34-deterministic-fake-input-scalar-preview-no-tensor",
        base_commit="phase2/p33-deterministic-fake-input-descriptor-no-values",
        allowed_files=allowed,
        phase_label="P34",
    )
def test_p34_87_serialized_keys_no_bool_suffix():
    res = run_fake_input_preview_probe()
    js = compact_fake_input_preview_json(res)
    d = json.loads(js)
    def check_keys(obj):
        if isinstance(obj, dict):
            for k, v in obj.items():
                assert ": bool" not in k, f"Key '{k}' contains forbidden ': bool' suffix"
                check_keys(v)
        elif isinstance(obj, list):
            for item in obj:
                check_keys(item)
    check_keys(d)


# 88. serialized result must contain exactly preview_available_in_p34
def test_p34_88_serialized_keys_exact_preview_available():
    res = run_fake_input_preview_probe()
    js = compact_fake_input_preview_json(res)
    d = json.loads(js)
    assert "preview_available_in_p34" in d
    assert d["preview_available_in_p34"] is True
    assert "preview_available_in_p34: bool" not in d


# 89. source file must not contain "preview_available_in_p34: bool"
def test_p34_89_source_no_bool_annotation():
    content = pathlib.Path("src/phase2/fc_vae_fake_input_preview.py").read_text(encoding="utf-8")
    assert '"preview_available_in_p34: bool"' not in content
    assert "'preview_available_in_p34: bool'" not in content


# 90. source file must not contain self-review text such as "Wait,", "Good catch", or "Let's"
def test_p34_90_source_no_self_review_text():
    content = pathlib.Path("src/phase2/fc_vae_fake_input_preview.py").read_text(encoding="utf-8")
    for word in ("Wait,", "Good catch", "Let's"):
        assert word.lower() not in content.lower(), f"Source code contains self-review/commentary text: {word}"
