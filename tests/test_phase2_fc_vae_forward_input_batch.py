# tests/test_phase2_fc_vae_forward_input_batch.py

import dataclasses
import json
import pathlib
import subprocess
from tests.phase2_scope_gate_utils import enforce_phase_local_scope_gate_or_skip
import re
import pytest
from dataclasses import is_dataclass

import src.phase2
from src.phase2.fc_vae_forward_input_batch import (
    FC_VAE_FORWARD_INPUT_BATCH_CONTRACT_VERSION,
    FC_VAE_FORWARD_INPUT_BATCH_KIND,
    FC_VAE_FORWARD_INPUT_BATCH_MODULE_NAME,
    FC_VAE_FORWARD_INPUT_BATCH_STATUS_MATERIALIZATION_BLOCKED,
    FC_VAE_FORWARD_INPUT_BATCH_STATUS_FORWARD_BOUNDARY_BLOCKED,
    FC_VAE_FORWARD_INPUT_BATCH_STATUS_TORCH_UNAVAILABLE,
    SUPPORTED_FC_VAE_FORWARD_INPUT_BATCH_STATUSES,
    FC_VAE_FORWARD_INPUT_BATCH_SHAPE_KIND,
    DEFAULT_P32_BATCH_SIZE,
    MIN_P32_BATCH_SIZE,
    MAX_P32_BATCH_SIZE,
    FCVAEForwardInputBatchRequest,
    FCVAEForwardInputBatchShape,
    FCVAEForwardInputBatchMetadata,
    FCVAEForwardInputBatchResult,
    validate_non_empty_str,
    validate_bool,
    validate_positive_int,
    validate_batch_size,
    validate_shape_tuple,
    validate_input_batch_status,
    assert_no_local_path_leakage,
    assert_no_forbidden_claims,
    validate_forward_input_batch_request,
    validate_forward_input_batch_shape,
    validate_forward_input_batch_metadata,
    validate_forward_input_batch_result,
    build_forward_input_batch_request_from_p31_default,
    build_forward_input_batch_shape,
    build_forward_input_batch_metadata,
    build_forward_input_batch_result,
    run_forward_input_batch_probe,
    forward_input_batch_request_to_json_dict,
    forward_input_batch_shape_to_json_dict,
    forward_input_batch_metadata_to_json_dict,
    forward_input_batch_result_to_json_dict,
    compact_forward_input_batch_json,
)


# 1. constants exact
def test_p32_01_constants():
    assert FC_VAE_FORWARD_INPUT_BATCH_CONTRACT_VERSION == "phase2_p32_forward_input_batch_contract_v1"
    assert FC_VAE_FORWARD_INPUT_BATCH_KIND == "forward_input_batch_metadata_no_tensor"
    assert FC_VAE_FORWARD_INPUT_BATCH_MODULE_NAME == "src.phase2.fc_vae_forward_input_batch"
    assert FC_VAE_FORWARD_INPUT_BATCH_STATUS_MATERIALIZATION_BLOCKED == "input_batch_materialization_blocked_in_p32"
    assert FC_VAE_FORWARD_INPUT_BATCH_STATUS_FORWARD_BOUNDARY_BLOCKED == "blocked_by_forward_boundary"
    assert FC_VAE_FORWARD_INPUT_BATCH_STATUS_TORCH_UNAVAILABLE == "blocked_torch_unavailable"
    assert len(SUPPORTED_FC_VAE_FORWARD_INPUT_BATCH_STATUSES) == 3
    assert "input_batch_materialization_blocked_in_p32" in SUPPORTED_FC_VAE_FORWARD_INPUT_BATCH_STATUSES
    assert "blocked_by_forward_boundary" in SUPPORTED_FC_VAE_FORWARD_INPUT_BATCH_STATUSES
    assert "blocked_torch_unavailable" in SUPPORTED_FC_VAE_FORWARD_INPUT_BATCH_STATUSES
    assert FC_VAE_FORWARD_INPUT_BATCH_SHAPE_KIND == "declared_input_batch_shape_no_tensor"
    assert DEFAULT_P32_BATCH_SIZE == 2
    assert MIN_P32_BATCH_SIZE == 1
    assert MAX_P32_BATCH_SIZE == 16


# 2. dataclasses frozen
def test_p32_02_dataclasses_frozen():
    for cls in (FCVAEForwardInputBatchRequest, FCVAEForwardInputBatchShape, FCVAEForwardInputBatchMetadata, FCVAEForwardInputBatchResult):
        assert is_dataclass(cls)
        req = build_forward_input_batch_request_from_p31_default()
        with pytest.raises(Exception):
            req.reason = "modified"


# 3. primitive validators accept/reject invalid values
def test_p32_03_primitive_validators():
    validate_non_empty_str("valid_string", "field")
    with pytest.raises(TypeError):
        validate_non_empty_str(123, "field")
    with pytest.raises(ValueError):
        validate_non_empty_str("", "field")
    with pytest.raises(ValueError):
        validate_non_empty_str("  padded  ", "field")

    validate_bool(True, "bool_field")
    validate_bool(False, "bool_field")
    with pytest.raises(TypeError):
        validate_bool(1, "bool_field")

    validate_positive_int(1, "pos_int")
    with pytest.raises(ValueError):
        validate_positive_int(0, "pos_int")
    with pytest.raises(TypeError):
        validate_positive_int(True, "pos_int")


# 4. batch size validator accepts min/default/max
def test_p32_04_batch_size_accepts():
    validate_batch_size(MIN_P32_BATCH_SIZE)
    validate_batch_size(DEFAULT_P32_BATCH_SIZE)
    validate_batch_size(MAX_P32_BATCH_SIZE)


# 5. batch size validator rejects 0 and MAX+1
def test_p32_05_batch_size_rejects():
    with pytest.raises(ValueError):
        validate_batch_size(0)
    with pytest.raises(ValueError):
        validate_batch_size(MAX_P32_BATCH_SIZE + 1)
    with pytest.raises(TypeError):
        validate_batch_size(1.5)
    with pytest.raises(TypeError):
        validate_batch_size(True)


# 6. shape tuple validator accepts exact tuple
def test_p32_06_shape_tuple_accepts():
    validate_shape_tuple((2, 32))


# 7. shape tuple validator rejects list/non-int/wrong length
def test_p32_07_shape_tuple_rejects():
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
def test_p32_08_status_validator():
    for status in SUPPORTED_FC_VAE_FORWARD_INPUT_BATCH_STATUSES:
        validate_input_batch_status(status)
    with pytest.raises(ValueError):
        validate_input_batch_status("unknown_status")
    with pytest.raises(TypeError):
        validate_input_batch_status(123)


# 9. local path and forbidden claim guards work
def test_p32_09_guards():
    assert_no_local_path_leakage("safe string")
    for leak in ["file:///", "C:/", "C:\\", "/home/", "/Users/"]:
        with pytest.raises(ValueError):
            assert_no_local_path_leakage(f"leak {leak}")

    assert_no_forbidden_claims("safe conclusion")
    # Allowed
    assert_no_forbidden_claims("no_scientific_conclusion")
    assert_no_forbidden_claims("no_final_comparison")
    assert_no_forbidden_claims("input_batch_materialization_blocked_in_p32")
    # Forbidden
    for claim in ["model works", "scientific success", "solved", "best", "winner", "production ready", "state of the art"]:
        with pytest.raises(ValueError):
            assert_no_forbidden_claims(f"our {claim} claim")


# 10. request builder returns batch_size=2 and input_flat_dim=32
def test_p32_10_request_builder():
    req = build_forward_input_batch_request_from_p31_default()
    assert req.batch_size == 2
    assert req.input_flat_dim == 32


# 11. request rejects wrong contract
def test_p32_11_request_wrong_contract():
    req = build_forward_input_batch_request_from_p31_default()
    bad = dataclasses.replace(req, contract_version="wrong")
    with pytest.raises(ValueError):
        validate_forward_input_batch_request(bad)


# 12. request rejects wrong batch kind
def test_p32_12_request_wrong_batch_kind():
    req = build_forward_input_batch_request_from_p31_default()
    bad = dataclasses.replace(req, batch_kind="wrong")
    with pytest.raises(ValueError):
        validate_forward_input_batch_request(bad)


# 13. request rejects wrong architecture
def test_p32_13_request_wrong_arch():
    req = build_forward_input_batch_request_from_p31_default()
    bad = dataclasses.replace(req, architecture_id="wrong")
    with pytest.raises(ValueError):
        validate_forward_input_batch_request(bad)


# 14. request rejects invalid source contract
def test_p32_14_request_wrong_source_contract():
    req = build_forward_input_batch_request_from_p31_default()
    bad = dataclasses.replace(req, source_forward_boundary_contract_version="wrong")
    with pytest.raises(ValueError):
        validate_forward_input_batch_request(bad)


# 15. request rejects allow_tensor_materialization True
def test_p32_15_request_allow_tensor():
    req = build_forward_input_batch_request_from_p31_default()
    bad = dataclasses.replace(req, allow_tensor_materialization_in_p32=True)
    with pytest.raises(ValueError):
        validate_forward_input_batch_request(bad)


# 16. request rejects allow_array_materialization True
def test_p32_16_request_allow_array():
    req = build_forward_input_batch_request_from_p31_default()
    bad = dataclasses.replace(req, allow_array_materialization_in_p32=True)
    with pytest.raises(ValueError):
        validate_forward_input_batch_request(bad)


# 17. request rejects allow_forward_execution True
def test_p32_17_request_allow_execution():
    req = build_forward_input_batch_request_from_p31_default()
    bad = dataclasses.replace(req, allow_forward_execution_in_p32=True)
    with pytest.raises(ValueError):
        validate_forward_input_batch_request(bad)


# 18. request rejects allow_output_generation True
def test_p32_18_request_allow_output():
    req = build_forward_input_batch_request_from_p31_default()
    bad = dataclasses.replace(req, allow_output_generation_in_p32=True)
    with pytest.raises(ValueError):
        validate_forward_input_batch_request(bad)


# 19. shape builder validates rank=2 and tuple=(2,32)
def test_p32_19_shape_builder():
    req = build_forward_input_batch_request_from_p31_default()
    shape = build_forward_input_batch_shape(req)
    validate_forward_input_batch_shape(shape)
    assert shape.rank == 2
    assert shape.shape_tuple == (2, 32)


# 20. shape rejects rank != 2
def test_p32_20_shape_wrong_rank():
    req = build_forward_input_batch_request_from_p31_default()
    shape = build_forward_input_batch_shape(req)
    bad = dataclasses.replace(shape, rank=3)
    with pytest.raises(ValueError):
        validate_forward_input_batch_shape(bad)


# 21. shape rejects wrong tuple
def test_p32_21_shape_wrong_tuple():
    req = build_forward_input_batch_request_from_p31_default()
    shape = build_forward_input_batch_shape(req)
    bad = dataclasses.replace(shape, shape_tuple=(3, 32))
    with pytest.raises(ValueError):
        validate_forward_input_batch_shape(bad)


# 22. shape rejects tensor_materialized True
def test_p32_22_shape_tensor_true():
    req = build_forward_input_batch_request_from_p31_default()
    shape = build_forward_input_batch_shape(req)
    bad = dataclasses.replace(shape, tensor_materialized=True)
    with pytest.raises(ValueError):
        validate_forward_input_batch_shape(bad)


# 23. shape rejects array_materialized True
def test_p32_23_shape_array_true():
    req = build_forward_input_batch_request_from_p31_default()
    shape = build_forward_input_batch_shape(req)
    bad = dataclasses.replace(shape, array_materialized=True)
    with pytest.raises(ValueError):
        validate_forward_input_batch_shape(bad)


# 24. shape rejects values_materialized True
def test_p32_24_shape_values_true():
    req = build_forward_input_batch_request_from_p31_default()
    shape = build_forward_input_batch_shape(req)
    bad = dataclasses.replace(shape, values_materialized=True)
    with pytest.raises(ValueError):
        validate_forward_input_batch_shape(bad)


# 25. metadata builder validates
def test_p32_25_metadata_builder():
    req = build_forward_input_batch_request_from_p31_default()
    meta = build_forward_input_batch_metadata(req)
    validate_forward_input_batch_metadata(meta)
    assert meta.batch_shape_declared is True


# 26. metadata rejects batch_shape_declared False
def test_p32_26_metadata_shape_declared_false():
    req = build_forward_input_batch_request_from_p31_default()
    meta = build_forward_input_batch_metadata(req)
    bad = dataclasses.replace(meta, batch_shape_declared=False)
    with pytest.raises(ValueError):
        validate_forward_input_batch_metadata(bad)


# 27. metadata rejects tensor_materialization_attempted True
def test_p32_27_metadata_tensor_attempted():
    req = build_forward_input_batch_request_from_p31_default()
    meta = build_forward_input_batch_metadata(req)
    bad = dataclasses.replace(meta, tensor_materialization_attempted=True)
    with pytest.raises(ValueError):
        validate_forward_input_batch_metadata(bad)


# 28. metadata rejects array_materialization_attempted True
def test_p32_28_metadata_array_attempted():
    req = build_forward_input_batch_request_from_p31_default()
    meta = build_forward_input_batch_metadata(req)
    bad = dataclasses.replace(meta, array_materialization_attempted=True)
    with pytest.raises(ValueError):
        validate_forward_input_batch_metadata(bad)


# 29. metadata rejects values_materialization_attempted True
def test_p32_29_metadata_values_attempted():
    req = build_forward_input_batch_request_from_p31_default()
    meta = build_forward_input_batch_metadata(req)
    bad = dataclasses.replace(meta, values_materialization_attempted=True)
    with pytest.raises(ValueError):
        validate_forward_input_batch_metadata(bad)


# 30. metadata rejects forward_execution_attempted True
def test_p32_30_metadata_forward_attempted():
    req = build_forward_input_batch_request_from_p31_default()
    meta = build_forward_input_batch_metadata(req)
    bad = dataclasses.replace(meta, forward_execution_attempted=True)
    with pytest.raises(ValueError):
        validate_forward_input_batch_metadata(bad)


# 31. metadata rejects output_generation_attempted True
def test_p32_31_metadata_output_attempted():
    req = build_forward_input_batch_request_from_p31_default()
    meta = build_forward_input_batch_metadata(req)
    bad = dataclasses.replace(meta, output_generation_attempted=True)
    with pytest.raises(ValueError):
        validate_forward_input_batch_metadata(bad)


# 32. result builder validates
def test_p32_32_result_builder():
    req = build_forward_input_batch_request_from_p31_default()
    res = build_forward_input_batch_result(req)
    validate_forward_input_batch_result(res)


# 33. result rejects input_batch_available_in_p32 True
def test_p32_33_result_batch_available():
    req = build_forward_input_batch_request_from_p31_default()
    res = build_forward_input_batch_result(req)
    bad = dataclasses.replace(res, input_batch_available_in_p32=True)
    with pytest.raises(ValueError):
        validate_forward_input_batch_result(bad)


# 34. result rejects tensor_materialization_available True
def test_p32_34_result_tensor_available():
    req = build_forward_input_batch_request_from_p31_default()
    res = build_forward_input_batch_result(req)
    bad = dataclasses.replace(res, tensor_materialization_available_in_p32=True)
    with pytest.raises(ValueError):
        validate_forward_input_batch_result(bad)


# 35. result rejects array_materialization_available True
def test_p32_35_result_array_available():
    req = build_forward_input_batch_request_from_p31_default()
    res = build_forward_input_batch_result(req)
    bad = dataclasses.replace(res, array_materialization_available_in_p32=True)
    with pytest.raises(ValueError):
        validate_forward_input_batch_result(bad)


# 36. result rejects values_materialization_available True
def test_p32_36_result_values_available():
    req = build_forward_input_batch_request_from_p31_default()
    res = build_forward_input_batch_result(req)
    bad = dataclasses.replace(res, values_materialization_available_in_p32=True)
    with pytest.raises(ValueError):
        validate_forward_input_batch_result(bad)


# 37. result rejects forward_execution_available True
def test_p32_37_result_execution_available():
    req = build_forward_input_batch_request_from_p31_default()
    res = build_forward_input_batch_result(req)
    bad = dataclasses.replace(res, forward_execution_available_in_p32=True)
    with pytest.raises(ValueError):
        validate_forward_input_batch_result(bad)


# 38. result rejects output_generation_available True
def test_p32_38_result_output_available():
    req = build_forward_input_batch_request_from_p31_default()
    res = build_forward_input_batch_result(req)
    bad = dataclasses.replace(res, output_generation_available_in_p32=True)
    with pytest.raises(ValueError):
        validate_forward_input_batch_result(bad)


# 39. result rejects no_tensor_created False
def test_p32_39_result_no_tensor():
    req = build_forward_input_batch_request_from_p31_default()
    res = build_forward_input_batch_result(req)
    bad = dataclasses.replace(res, no_tensor_created=False)
    with pytest.raises(ValueError):
        validate_forward_input_batch_result(bad)


# 40. result rejects no_array_created False
def test_p32_40_result_no_array():
    req = build_forward_input_batch_request_from_p31_default()
    res = build_forward_input_batch_result(req)
    bad = dataclasses.replace(res, no_array_created=False)
    with pytest.raises(ValueError):
        validate_forward_input_batch_result(bad)


# 41. result rejects no_values_materialized False
def test_p32_41_result_no_values():
    req = build_forward_input_batch_request_from_p31_default()
    res = build_forward_input_batch_result(req)
    bad = dataclasses.replace(res, no_values_materialized=False)
    with pytest.raises(ValueError):
        validate_forward_input_batch_result(bad)


# 42. result rejects no_forward_execution False
def test_p32_42_result_no_forward():
    req = build_forward_input_batch_request_from_p31_default()
    res = build_forward_input_batch_result(req)
    bad = dataclasses.replace(res, no_forward_execution=False)
    with pytest.raises(ValueError):
        validate_forward_input_batch_result(bad)


# 43. result rejects no_output_generation False
def test_p32_43_result_no_output():
    req = build_forward_input_batch_request_from_p31_default()
    res = build_forward_input_batch_result(req)
    bad = dataclasses.replace(res, no_output_generation=False)
    with pytest.raises(ValueError):
        validate_forward_input_batch_result(bad)


# 44. result rejects no_training_loop False
def test_p32_44_result_no_training():
    req = build_forward_input_batch_request_from_p31_default()
    res = build_forward_input_batch_result(req)
    bad = dataclasses.replace(res, no_training_loop=False)
    with pytest.raises(ValueError):
        validate_forward_input_batch_result(bad)


# 45. result rejects no_optimizer False
def test_p32_45_result_no_optimizer():
    req = build_forward_input_batch_request_from_p31_default()
    res = build_forward_input_batch_result(req)
    bad = dataclasses.replace(res, no_optimizer=False)
    with pytest.raises(ValueError):
        validate_forward_input_batch_result(bad)


# 46. result rejects no_checkpointing False
def test_p32_46_result_no_checkpoint():
    req = build_forward_input_batch_request_from_p31_default()
    res = build_forward_input_batch_result(req)
    bad = dataclasses.replace(res, no_checkpointing=False)
    with pytest.raises(ValueError):
        validate_forward_input_batch_result(bad)


# 47. result rejects no_artifact_generation False
def test_p32_47_result_no_artifact():
    req = build_forward_input_batch_request_from_p31_default()
    res = build_forward_input_batch_result(req)
    bad = dataclasses.replace(res, no_artifact_generation=False)
    with pytest.raises(ValueError):
        validate_forward_input_batch_result(bad)


# 48. result rejects no_final_comparison False
def test_p32_48_result_no_final_comp():
    req = build_forward_input_batch_request_from_p31_default()
    res = build_forward_input_batch_result(req)
    bad = dataclasses.replace(res, no_final_comparison=False)
    with pytest.raises(ValueError):
        validate_forward_input_batch_result(bad)


# 49. result rejects no_scientific_conclusion False
def test_p32_49_result_no_sci_conclusion():
    req = build_forward_input_batch_request_from_p31_default()
    res = build_forward_input_batch_result(req)
    bad = dataclasses.replace(res, no_scientific_conclusion=False)
    with pytest.raises(ValueError):
        validate_forward_input_batch_result(bad)


# 50. probe returns valid result
def test_p32_50_probe():
    res = run_forward_input_batch_probe()
    assert isinstance(res, FCVAEForwardInputBatchResult)
    validate_forward_input_batch_result(res)


# 51. serializers return JSON-safe dict
def test_p32_51_serializers():
    req = build_forward_input_batch_request_from_p31_default()
    res = build_forward_input_batch_result(req)
    d1 = forward_input_batch_request_to_json_dict(req)
    d2 = forward_input_batch_shape_to_json_dict(res.shape)
    d3 = forward_input_batch_metadata_to_json_dict(res.metadata)
    d4 = forward_input_batch_result_to_json_dict(res)
    assert isinstance(d1, dict)
    assert isinstance(d2, dict)
    assert isinstance(d3, dict)
    assert isinstance(d4, dict)


# 52. compact JSON sorted/parseable
def test_p32_52_compact_json():
    res = run_forward_input_batch_probe()
    js = compact_forward_input_batch_json(res)
    assert isinstance(js, str)
    d = json.loads(js)
    assert d["contract_version"] == FC_VAE_FORWARD_INPUT_BATCH_CONTRACT_VERSION


# 53. serialized result has no local paths
def test_p32_53_serialized_no_local_paths():
    res = run_forward_input_batch_probe()
    js = compact_forward_input_batch_json(res)
    for path in ["file:///", "C:/", "C:\\", "/home/", "/Users/"]:
        assert path not in js
        assert path.lower() not in js.lower()


# 54. serialized result has no forbidden success claims
def test_p32_54_serialized_no_claims():
    res = run_forward_input_batch_probe()
    js = compact_forward_input_batch_json(res)
    normalized = js.lower()
    cleaned = (normalized
               .replace("no_final_comparison", "")
               .replace("no_scientific_conclusion", "")
               .replace("input_batch_materialization_blocked_in_p32", ""))
    for claim in ["model works", "scientific success", "solved", "best", "winner", "production ready", "state of the art"]:
        assert claim not in cleaned


# 55. serialized result contains no tensor/array/value payload
def test_p32_55_serialized_no_payloads():
    res = run_forward_input_batch_probe()
    js = compact_forward_input_batch_json(res)
    d = json.loads(js)
    assert "tensor" not in d
    assert "array" not in d
    assert "values" not in d


# 56. shape_tuple serialized as JSON list
def test_p32_56_shape_tuple_json_list():
    res = run_forward_input_batch_probe()
    js = compact_forward_input_batch_json(res)
    d = json.loads(js)
    assert isinstance(d["shape"]["shape_tuple"], list)
    assert d["shape"]["shape_tuple"] == [2, 32]


# 57. module has no torch import
def test_p32_57_no_torch_import():
    p = pathlib.Path("src/phase2/fc_vae_forward_input_batch.py").read_text(encoding="utf-8")
    assert "import torch" not in p
    assert "from torch" not in p


# 58. module has no numpy import
def test_p32_58_no_numpy_import():
    p = pathlib.Path("src/phase2/fc_vae_forward_input_batch.py").read_text(encoding="utf-8")
    assert "import numpy" not in p
    assert "from numpy" not in p


# 59. module has no `def forward`
def test_p32_59_no_def_forward():
    p = pathlib.Path("src/phase2/fc_vae_forward_input_batch.py").read_text(encoding="utf-8")
    assert "def forward(" not in p
    assert "def forward " not in p


# 60. module has no `forward(` definition
def test_p32_60_no_forward_definition():
    p = pathlib.Path("src/phase2/fc_vae_forward_input_batch.py").read_text(encoding="utf-8")
    assert "forward(" not in p


# 61. module has no tensor/array/materialized value constructors
def test_p32_61_no_constructors():
    p = pathlib.Path("src/phase2/fc_vae_forward_input_batch.py").read_text(encoding="utf-8")
    forbidden = ["torch.tensor", "np.array", "np.zeros", "torch.zeros"]
    for f in forbidden:
        assert f not in p


# 62. module has no train/fit/loss/optimizer/checkpoint functions
def test_p32_62_no_training_code():
    p = pathlib.Path("src/phase2/fc_vae_forward_input_batch.py").read_text(encoding="utf-8")
    forbidden = ["def train", "def fit", "def loss", "def optimizer", "def checkpoint"]
    for f in forbidden:
        assert f not in p


# 63. smoke script has no torch/numpy import
def test_p32_63_smoke_script_imports():
    p = pathlib.Path("tools/phase2/run_p32_forward_input_batch_smoke.py").read_text(encoding="utf-8")
    assert "import torch" not in p
    assert "from torch" not in p
    assert "import numpy" not in p
    assert "from numpy" not in p


# 64. `src/phase2/__init__.py` has no monkeypatch symbols
def test_p32_64_init_monkeypatch():
    p = pathlib.Path("src/phase2/__init__.py").read_text(encoding="utf-8")
    assert "subprocess.run =" not in p
    assert "_patched_run" not in p
    assert "_original_run" not in p


# 65. P32-owned scope gate compares against accepted P31 branch
def test_p32_65_scope_gate():
    """Phase-local scope gate for P32. Skips on non-P32 branches."""
    allowed = {
        "src/phase2/fc_vae_forward_input_batch.py",
        "src/phase2/__init__.py",
        "tests/test_phase2_fc_vae_forward_input_batch.py",
        "tools/phase2/run_p32_forward_input_batch_smoke.py",
        "tests/test_phase2_p32_forward_input_batch_smoke.py",
        "reports/PHASE_2_P32_FORWARD_INPUT_BATCH_CONTRACT_NO_TENSOR_MATERIALIZATION_REPORT.md",
    }
    enforce_phase_local_scope_gate_or_skip(
        expected_branch="phase2/p32-forward-input-batch-contract-no-tensor",
        base_commit="phase2/p31-noop-forward-boundary-contract",
        allowed_files=allowed,
        phase_label="P32",
    )
