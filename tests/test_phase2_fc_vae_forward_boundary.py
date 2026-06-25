# tests/test_phase2_fc_vae_forward_boundary.py

import dataclasses
import json
import pathlib
import subprocess
import re
import pytest
from dataclasses import is_dataclass

import src.phase2
from src.phase2.fc_vae_forward_boundary import (
    FC_VAE_FORWARD_BOUNDARY_CONTRACT_VERSION,
    FC_VAE_FORWARD_BOUNDARY_KIND,
    FC_VAE_FORWARD_BOUNDARY_MODULE_NAME,
    FC_VAE_FORWARD_BOUNDARY_STATUS_TORCH_UNAVAILABLE,
    FC_VAE_FORWARD_BOUNDARY_STATUS_CONSTRUCTOR_NOT_BOUND,
    FC_VAE_FORWARD_BOUNDARY_STATUS_NOOP_BLOCKED,
    SUPPORTED_FC_VAE_FORWARD_BOUNDARY_STATUSES,
    FC_VAE_FORWARD_BOUNDARY_OUTPUT_KIND,
    FCVAEForwardBoundaryRequest,
    FCVAEForwardBoundaryOutputShape,
    FCVAEForwardBoundaryMetadata,
    FCVAEForwardBoundaryResult,
    validate_non_empty_str,
    validate_bool,
    validate_positive_int,
    validate_non_negative_int,
    validate_forward_boundary_status,
    assert_no_local_path_leakage,
    assert_no_forbidden_claims,
    validate_forward_boundary_request,
    validate_forward_boundary_output_shape,
    validate_forward_boundary_metadata,
    validate_forward_boundary_result,
    build_forward_boundary_request_from_p27_smoke_contracts,
    build_forward_boundary_output_shape,
    build_forward_boundary_metadata,
    build_forward_boundary_result,
    run_forward_boundary_probe,
    forward_boundary_request_to_json_dict,
    forward_boundary_output_shape_to_json_dict,
    forward_boundary_metadata_to_json_dict,
    forward_boundary_result_to_json_dict,
    compact_forward_boundary_json,
)


# 1. Constants exact
def test_p31_01_constants():
    assert FC_VAE_FORWARD_BOUNDARY_CONTRACT_VERSION == "phase2_p31_noop_forward_boundary_contract_v1"
    assert FC_VAE_FORWARD_BOUNDARY_KIND == "noop_forward_boundary_contract"
    assert FC_VAE_FORWARD_BOUNDARY_MODULE_NAME == "src.phase2.fc_vae_forward_boundary"
    assert FC_VAE_FORWARD_BOUNDARY_STATUS_TORCH_UNAVAILABLE == "blocked_torch_unavailable"
    assert FC_VAE_FORWARD_BOUNDARY_STATUS_CONSTRUCTOR_NOT_BOUND == "blocked_constructor_not_bound"
    assert FC_VAE_FORWARD_BOUNDARY_STATUS_NOOP_BLOCKED == "noop_forward_blocked_in_p31"
    assert len(SUPPORTED_FC_VAE_FORWARD_BOUNDARY_STATUSES) == 3
    assert "blocked_torch_unavailable" in SUPPORTED_FC_VAE_FORWARD_BOUNDARY_STATUSES
    assert "blocked_constructor_not_bound" in SUPPORTED_FC_VAE_FORWARD_BOUNDARY_STATUSES
    assert "noop_forward_blocked_in_p31" in SUPPORTED_FC_VAE_FORWARD_BOUNDARY_STATUSES
    assert FC_VAE_FORWARD_BOUNDARY_OUTPUT_KIND == "declared_noop_modelspec_logits_shape"


# 2. Dataclasses frozen
def test_p31_02_dataclasses_frozen():
    for cls in (FCVAEForwardBoundaryRequest, FCVAEForwardBoundaryOutputShape, FCVAEForwardBoundaryMetadata, FCVAEForwardBoundaryResult):
        assert is_dataclass(cls)
        req = build_forward_boundary_request_from_p27_smoke_contracts()
        with pytest.raises(Exception):
            req.reason = "modified"


# 3. Primitive validators accept/reject invalid values
def test_p31_03_primitive_validators():
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

    validate_non_negative_int(0, "non_neg_int")
    validate_non_negative_int(5, "non_neg_int")
    with pytest.raises(ValueError):
        validate_non_negative_int(-1, "non_neg_int")


# 4. Status validator accepts all supported statuses
def test_p31_04_status_validator():
    for status in SUPPORTED_FC_VAE_FORWARD_BOUNDARY_STATUSES:
        validate_forward_boundary_status(status)
    with pytest.raises(ValueError):
        validate_forward_boundary_status("unknown_status")
    with pytest.raises(TypeError):
        validate_forward_boundary_status(123)


# 5. Local path and forbidden claim guards work
def test_p31_05_guards():
    assert_no_local_path_leakage("safe string")
    for leak in ["file:///", "C:/", "C:\\", "/home/", "/Users/"]:
        with pytest.raises(ValueError):
            assert_no_local_path_leakage(f"leak {leak}")

    assert_no_forbidden_claims("safe conclusion")
    # Allowed negative/no-op flags
    assert_no_forbidden_claims("no_scientific_conclusion")
    assert_no_forbidden_claims("no_final_comparison")
    assert_no_forbidden_claims("noop_forward_blocked_in_p31")
    assert_no_forbidden_claims("forward_execution_available_in_p31=false")
    # Forbidden
    for claim in ["model works", "scientific success", "solved", "best", "winner", "production ready", "state of the art"]:
        with pytest.raises(ValueError):
            assert_no_forbidden_claims(f"our {claim} claim")


# 6. Request builder returns P27 smoke dims
def test_p31_06_request_builder_smoke_dims():
    req = build_forward_boundary_request_from_p27_smoke_contracts()
    assert req.input_flat_dim == 32
    assert req.z_mean_dim == 8
    assert req.z_volatility_dim == 8
    assert req.z_shared_dim == 4
    assert req.declared_family_head_dim == 3
    assert req.declared_mean_head_dim == 3
    assert req.declared_volatility_head_dim == 3
    assert req.declared_diagnostic_head_dim == 4


# 7. Request rejects wrong contract
def test_p31_07_request_wrong_contract():
    req = build_forward_boundary_request_from_p27_smoke_contracts()
    bad = dataclasses.replace(req, contract_version="wrong")
    with pytest.raises(ValueError):
        validate_forward_boundary_request(bad)


# 8. Request rejects wrong boundary kind
def test_p31_08_request_wrong_boundary_kind():
    req = build_forward_boundary_request_from_p27_smoke_contracts()
    bad = dataclasses.replace(req, boundary_kind="wrong")
    with pytest.raises(ValueError):
        validate_forward_boundary_request(bad)


# 9. Request rejects wrong architecture
def test_p31_09_request_wrong_architecture():
    req = build_forward_boundary_request_from_p27_smoke_contracts()
    bad = dataclasses.replace(req, architecture_id="wrong")
    with pytest.raises(ValueError):
        validate_forward_boundary_request(bad)


# 10. Request rejects allow_execution True
def test_p31_10_request_allow_execution_true():
    req = build_forward_boundary_request_from_p27_smoke_contracts()
    bad = dataclasses.replace(req, allow_execution_in_p31=True)
    with pytest.raises(ValueError):
        validate_forward_boundary_request(bad)


# 11. Request rejects allow_tensor_allocation True
def test_p31_11_request_allow_tensor_true():
    req = build_forward_boundary_request_from_p27_smoke_contracts()
    bad = dataclasses.replace(req, allow_tensor_allocation_in_p31=True)
    with pytest.raises(ValueError):
        validate_forward_boundary_request(bad)


# 12. Request rejects allow_output_generation True
def test_p31_12_request_allow_output_true():
    req = build_forward_boundary_request_from_p27_smoke_contracts()
    bad = dataclasses.replace(req, allow_output_generation_in_p31=True)
    with pytest.raises(ValueError):
        validate_forward_boundary_request(bad)


# 13. Request rejects allow_training True
def test_p31_13_request_allow_training_true():
    req = build_forward_boundary_request_from_p27_smoke_contracts()
    bad = dataclasses.replace(req, allow_training_in_p31=True)
    with pytest.raises(ValueError):
        validate_forward_boundary_request(bad)


# 14. Output shape builder validates declared dims and total
def test_p31_14_output_shape_builder():
    req = build_forward_boundary_request_from_p27_smoke_contracts()
    shape = build_forward_boundary_output_shape(req)
    validate_forward_boundary_output_shape(shape)
    assert shape.total_declared_output_dim == 13


# 15. Output shape rejects generated_output True
def test_p31_15_output_shape_rejects_generated_output():
    req = build_forward_boundary_request_from_p27_smoke_contracts()
    shape = build_forward_boundary_output_shape(req)
    bad = dataclasses.replace(shape, generated_output=True)
    with pytest.raises(ValueError):
        validate_forward_boundary_output_shape(bad)


# 16. Output shape rejects wrong total_declared_output_dim
def test_p31_16_output_shape_rejects_wrong_total():
    req = build_forward_boundary_request_from_p27_smoke_contracts()
    shape = build_forward_boundary_output_shape(req)
    bad = dataclasses.replace(shape, total_declared_output_dim=10)
    with pytest.raises(ValueError):
        validate_forward_boundary_output_shape(bad)


# 17. Metadata builder validates whether torch available or not
def test_p31_17_metadata_builder():
    req = build_forward_boundary_request_from_p27_smoke_contracts()
    meta = build_forward_boundary_metadata(req)
    validate_forward_boundary_metadata(meta)


# 18. Metadata rejects execution_attempted True
def test_p31_18_metadata_execution_attempted():
    req = build_forward_boundary_request_from_p27_smoke_contracts()
    meta = build_forward_boundary_metadata(req)
    bad = dataclasses.replace(meta, execution_attempted=True)
    with pytest.raises(ValueError):
        validate_forward_boundary_metadata(bad)


# 19. Metadata rejects tensor_allocation_attempted True
def test_p31_19_metadata_tensor_attempted():
    req = build_forward_boundary_request_from_p27_smoke_contracts()
    meta = build_forward_boundary_metadata(req)
    bad = dataclasses.replace(meta, tensor_allocation_attempted=True)
    with pytest.raises(ValueError):
        validate_forward_boundary_metadata(bad)


# 20. Metadata rejects output_generation_attempted True
def test_p31_20_metadata_output_attempted():
    req = build_forward_boundary_request_from_p27_smoke_contracts()
    meta = build_forward_boundary_metadata(req)
    bad = dataclasses.replace(meta, output_generation_attempted=True)
    with pytest.raises(ValueError):
        validate_forward_boundary_metadata(bad)


# 21. Metadata rejects module_object_returned True
def test_p31_21_metadata_module_object_returned():
    req = build_forward_boundary_request_from_p27_smoke_contracts()
    meta = build_forward_boundary_metadata(req)
    bad = dataclasses.replace(meta, module_object_returned=True)
    with pytest.raises(ValueError):
        validate_forward_boundary_metadata(bad)


# 22. Metadata rejects defines_forward True
def test_p31_22_metadata_defines_forward():
    req = build_forward_boundary_request_from_p27_smoke_contracts()
    meta = build_forward_boundary_metadata(req)
    bad = dataclasses.replace(meta, defines_forward=True)
    with pytest.raises(ValueError):
        validate_forward_boundary_metadata(bad)


# 23. Metadata rejects defines_layers True
def test_p31_23_metadata_defines_layers():
    req = build_forward_boundary_request_from_p27_smoke_contracts()
    meta = build_forward_boundary_metadata(req)
    bad = dataclasses.replace(meta, defines_layers=True)
    with pytest.raises(ValueError):
        validate_forward_boundary_metadata(bad)


# 24. Metadata rejects nonzero parameter_count
def test_p31_24_metadata_nonzero_param_count():
    req = build_forward_boundary_request_from_p27_smoke_contracts()
    meta = build_forward_boundary_metadata(req)
    bad = dataclasses.replace(meta, parameter_count=1)
    with pytest.raises(ValueError):
        validate_forward_boundary_metadata(bad)


# 25. Metadata rejects nonzero buffer_count
def test_p31_25_metadata_nonzero_buffer_count():
    req = build_forward_boundary_request_from_p27_smoke_contracts()
    meta = build_forward_boundary_metadata(req)
    bad = dataclasses.replace(meta, buffer_count=1)
    with pytest.raises(ValueError):
        validate_forward_boundary_metadata(bad)


# 26. Result builder validates whether torch available or not
def test_p31_26_result_builder():
    req = build_forward_boundary_request_from_p27_smoke_contracts()
    res = build_forward_boundary_result(req)
    validate_forward_boundary_result(res)


# 27. Result rejects torch_required_for_p31 True
def test_p31_27_result_torch_required_p31():
    req = build_forward_boundary_request_from_p27_smoke_contracts()
    res = build_forward_boundary_result(req)
    bad = dataclasses.replace(res, torch_required_for_p31=True)
    with pytest.raises(ValueError):
        validate_forward_boundary_result(bad)


# 28. Result rejects forward_execution_available True
def test_p31_28_result_execution_available():
    req = build_forward_boundary_request_from_p27_smoke_contracts()
    res = build_forward_boundary_result(req)
    bad = dataclasses.replace(res, forward_execution_available_in_p31=True)
    with pytest.raises(ValueError):
        validate_forward_boundary_result(bad)


# 29. Result rejects tensor_allocation_available True
def test_p31_29_result_tensor_available():
    req = build_forward_boundary_request_from_p27_smoke_contracts()
    res = build_forward_boundary_result(req)
    bad = dataclasses.replace(res, tensor_allocation_available_in_p31=True)
    with pytest.raises(ValueError):
        validate_forward_boundary_result(bad)


# 30. Result rejects output_generation_available True
def test_p31_30_result_output_available():
    req = build_forward_boundary_request_from_p27_smoke_contracts()
    res = build_forward_boundary_result(req)
    bad = dataclasses.replace(res, output_generation_available_in_p31=True)
    with pytest.raises(ValueError):
        validate_forward_boundary_result(bad)


# 31. Result rejects training_available True
def test_p31_31_result_training_available():
    req = build_forward_boundary_request_from_p27_smoke_contracts()
    res = build_forward_boundary_result(req)
    bad = dataclasses.replace(res, training_available_in_p31=True)
    with pytest.raises(ValueError):
        validate_forward_boundary_result(bad)


# 32. Result rejects loss_available True
def test_p31_32_result_loss_available():
    req = build_forward_boundary_request_from_p27_smoke_contracts()
    res = build_forward_boundary_result(req)
    bad = dataclasses.replace(res, loss_available_in_p31=True)
    with pytest.raises(ValueError):
        validate_forward_boundary_result(bad)


# 33. Result rejects optimizer_available True
def test_p31_33_result_optimizer_available():
    req = build_forward_boundary_request_from_p27_smoke_contracts()
    res = build_forward_boundary_result(req)
    bad = dataclasses.replace(res, optimizer_available_in_p31=True)
    with pytest.raises(ValueError):
        validate_forward_boundary_result(bad)


# 34. Result rejects checkpointing_available True
def test_p31_34_result_checkpointing_available():
    req = build_forward_boundary_request_from_p27_smoke_contracts()
    res = build_forward_boundary_result(req)
    bad = dataclasses.replace(res, checkpointing_available_in_p31=True)
    with pytest.raises(ValueError):
        validate_forward_boundary_result(bad)


# 35. Result rejects artifact_generation_available True
def test_p31_35_result_artifact_available():
    req = build_forward_boundary_request_from_p27_smoke_contracts()
    res = build_forward_boundary_result(req)
    bad = dataclasses.replace(res, artifact_generation_available_in_p31=True)
    with pytest.raises(ValueError):
        validate_forward_boundary_result(bad)


# 36. Result rejects no_final_comparison False
def test_p31_36_result_no_final_comparison():
    req = build_forward_boundary_request_from_p27_smoke_contracts()
    res = build_forward_boundary_result(req)
    bad = dataclasses.replace(res, no_final_comparison=False)
    with pytest.raises(ValueError):
        validate_forward_boundary_result(bad)


# 37. Result rejects no_scientific_conclusion False
def test_p31_37_result_no_scientific_conclusion():
    req = build_forward_boundary_request_from_p27_smoke_contracts()
    res = build_forward_boundary_result(req)
    bad = dataclasses.replace(res, no_scientific_conclusion=False)
    with pytest.raises(ValueError):
        validate_forward_boundary_result(bad)


# 38. Probe returns valid result
def test_p31_38_probe():
    res = run_forward_boundary_probe()
    assert isinstance(res, FCVAEForwardBoundaryResult)
    validate_forward_boundary_result(res)


# 39. Serializers return JSON-safe dict
def test_p31_39_serializers():
    req = build_forward_boundary_request_from_p27_smoke_contracts()
    res = build_forward_boundary_result(req)
    d1 = forward_boundary_request_to_json_dict(req)
    d2 = forward_boundary_output_shape_to_json_dict(res.declared_output_shape)
    d3 = forward_boundary_metadata_to_json_dict(res.metadata)
    d4 = forward_boundary_result_to_json_dict(res)
    assert isinstance(d1, dict)
    assert isinstance(d2, dict)
    assert isinstance(d3, dict)
    assert isinstance(d4, dict)


# 40. Compact JSON sorted/parseable
def test_p31_40_compact_json():
    res = run_forward_boundary_probe()
    js = compact_forward_boundary_json(res)
    assert isinstance(js, str)
    d = json.loads(js)
    assert d["contract_version"] == FC_VAE_FORWARD_BOUNDARY_CONTRACT_VERSION


# 41. Serialized result has no local paths
def test_p31_41_serialized_no_local_paths():
    res = run_forward_boundary_probe()
    js = compact_forward_boundary_json(res)
    for path in ["file:///", "C:/", "C:\\", "/home/", "/Users/"]:
        assert path not in js
        assert path.lower() not in js.lower()


# 42. Serialized result has no forbidden success claims
def test_p31_42_serialized_no_claims():
    res = run_forward_boundary_probe()
    js = compact_forward_boundary_json(res)
    normalized = js.lower()
    cleaned = (normalized
               .replace("no_final_comparison", "")
               .replace("no_scientific_conclusion", "")
               .replace("noop_forward_blocked_in_p31", "")
               .replace("forward_execution_available_in_p31=false", ""))
    for claim in ["model works", "scientific success", "solved", "best", "winner", "production ready", "state of the art"]:
        assert claim not in cleaned


# 43. Serialized result has no module object
def test_p31_43_serialized_no_module():
    res = run_forward_boundary_probe()
    js = compact_forward_boundary_json(res)
    d = json.loads(js)
    assert "module" not in d


# 44. Serialized result has generated_output False
def test_p31_44_serialized_generated_output_false():
    res = run_forward_boundary_probe()
    js = compact_forward_boundary_json(res)
    d = json.loads(js)
    assert d["declared_output_shape"]["generated_output"] is False


# 45. Module has no top-level torch import
def test_p31_45_no_top_level_torch_import():
    p = pathlib.Path("src/phase2/fc_vae_forward_boundary.py").read_text(encoding="utf-8")
    for line in p.splitlines():
        if line.strip().startswith("#"):
            continue
        assert not line.startswith("import torch"), f"Top-level import torch found: {line}"
        assert not line.startswith("from torch"), f"Top-level from torch found: {line}"


# 46. Module has no local torch import
def test_p31_46_no_local_torch_import():
    p = pathlib.Path("src/phase2/fc_vae_forward_boundary.py").read_text(encoding="utf-8")
    assert "import torch" not in p
    assert "from torch" not in p


# 47. Module has no `def forward`
def test_p31_47_no_def_forward():
    p = pathlib.Path("src/phase2/fc_vae_forward_boundary.py").read_text(encoding="utf-8")
    assert "def forward(" not in p
    assert "def forward " not in p


# 48. Module has no `forward(` definition
def test_p31_48_no_forward_definition():
    p = pathlib.Path("src/phase2/fc_vae_forward_boundary.py").read_text(encoding="utf-8")
    assert "forward(" not in p


# 49. Module has no torch.nn.Linear/Conv/Sequential/Parameter
def test_p31_49_no_forbidden_layers():
    p = pathlib.Path("src/phase2/fc_vae_forward_boundary.py").read_text(encoding="utf-8")
    forbidden = ["nn.Linear", "nn.Conv", "nn.Sequential", "nn.Parameter", "nn.Module"]
    for layer in forbidden:
        assert layer not in p


# 50. Module has no train/fit/loss/optimizer/checkpoint functions
def test_p31_50_no_training_code():
    p = pathlib.Path("src/phase2/fc_vae_forward_boundary.py").read_text(encoding="utf-8")
    forbidden_fns = [
        "def train(", "def fit(", "def train_step(", "def training_loop(",
        "def save_checkpoint(", "def load_checkpoint(", "def loss(", "def optimizer("
    ]
    for fn in forbidden_fns:
        assert fn not in p


# 51. Smoke script has no torch import
def test_p31_51_smoke_script_no_torch():
    p = pathlib.Path("tools/phase2/run_p31_forward_boundary_smoke.py").read_text(encoding="utf-8")
    assert "import torch" not in p
    assert "from torch" not in p


# 52. `src/phase2/__init__.py` has no monkeypatch symbols
def test_p31_52_init_no_subprocess_monkeypatch():
    p = pathlib.Path("src/phase2/__init__.py").read_text(encoding="utf-8")
    assert "subprocess.run =" not in p
    assert "_patched_run" not in p
    assert "_original_run" not in p


# 53. P31-owned scope gate compares against accepted P30 branch and allows only the specified 6 files
def test_p31_53_scope_gate():
    allowed = {
        "src/phase2/fc_vae_forward_boundary.py",
        "src/phase2/__init__.py",
        "tests/test_phase2_fc_vae_forward_boundary.py",
        "tools/phase2/run_p31_forward_boundary_smoke.py",
        "tests/test_phase2_p31_forward_boundary_smoke.py",
        "reports/PHASE_2_P31_NOOP_FORWARD_BOUNDARY_CONTRACT_REPORT.md",
    }
    res = subprocess.run(
        ["git", "diff", "--name-only", "phase2/p30-torch-module-constructor-spec-binding"],
        capture_output=True, text=True, check=True
    )
    modified = [line.strip() for line in res.stdout.splitlines() if line.strip()]
    for f in modified:
        f_norm = f.replace("\\", "/")
        assert f_norm in allowed, f"Forbidden file modification detected in P31: {f_norm}"
