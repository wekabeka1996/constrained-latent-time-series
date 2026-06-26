# tests/test_phase2_p40_forward_eligibility_gate_smoke.py

import json
import pathlib
import sys
import pytest
from typing import Any

from src.phase2.torch_boundary import TorchDependencyStatus
from src.phase2.fc_vae_tensor_materialization import FCVAETensorMaterializationResult
from src.phase2.fc_vae_forward_eligibility_gate import (
    run_forward_eligibility_gate_probe,
    build_forward_eligibility_request_from_defaults,
)
from tools.phase2.run_p40_forward_eligibility_gate_smoke import (
    run_p40_forward_eligibility_gate_smoke,
    compact_json,
    main,
)

# 01. compact_json sorts keys
def test_p40_smoke_01_compact_json_sorts():
    data = {"b": 2, "a": 1}
    js = compact_json(data)
    assert js == '{"a":1,"b":2}'

# 02. main rejects args
def test_p40_smoke_02_main_rejects_args():
    orig = sys.argv
    try:
        sys.argv = ["script.py", "extra"]
        code = main()
        assert code != 0
    finally:
        sys.argv = orig

# 03. smoke returns PASS
def test_p40_smoke_03_verdict_pass():
    res = run_p40_forward_eligibility_gate_smoke()
    assert res["verdict"] == "PASS"

# 04. contract exact
def test_p40_smoke_04_contract_exact():
    res = run_p40_forward_eligibility_gate_smoke()
    assert res["contract"] == "phase2_p40_forward_eligibility_gate_contract_v1"

# 05. source_phase P40
def test_p40_smoke_05_source_phase():
    res = run_p40_forward_eligibility_gate_smoke()
    assert res["source_phase"] == "P40"

# 06. status blocked_by_model_implementation_unavailable
def test_p40_smoke_06_status_expected():
    res = run_p40_forward_eligibility_gate_smoke()
    assert res["status"] == "blocked_by_model_implementation_unavailable"

# 07. torch_available True
def test_p40_smoke_07_torch_available():
    res = run_p40_forward_eligibility_gate_smoke()
    assert res["torch_available"] is True

# 08. tensor_materialized_in_p39 True
def test_p40_smoke_08_tensor_materialized():
    res = run_p40_forward_eligibility_gate_smoke()
    assert res["tensor_materialized_in_p39"] is True

# 09. tensor dtype
def test_p40_smoke_09_tensor_dtype():
    res = run_p40_forward_eligibility_gate_smoke()
    assert res["tensor_dtype_name"] == "torch.float32"

# 10. tensor device
def test_p40_smoke_10_tensor_device():
    res = run_p40_forward_eligibility_gate_smoke()
    assert res["tensor_device_type"] == "cpu"

# 11. tensor shape
def test_p40_smoke_11_tensor_shape():
    res = run_p40_forward_eligibility_gate_smoke()
    assert res["tensor_shape_tuple"] == [2, 32]
    assert res["tensor_numel"] == 64

# 12. requires_grad False
def test_p40_smoke_12_requires_grad():
    res = run_p40_forward_eligibility_gate_smoke()
    assert res["tensor_requires_grad"] is False

# 13. tensor_is_floating_point True
def test_p40_smoke_13_is_floating():
    res = run_p40_forward_eligibility_gate_smoke()
    assert res["tensor_is_floating_point"] is True

# 14. values_match True
def test_p40_smoke_14_values_match():
    res = run_p40_forward_eligibility_gate_smoke()
    assert res["tensor_values_match_p37_nested_values"] is True

# 15. model_skeleton_status ready_for_future_implementation
def test_p40_smoke_15_model_skeleton_status():
    res = run_p40_forward_eligibility_gate_smoke()
    assert res["model_skeleton_status"] == "ready_for_future_implementation"

# 16. model_no_implementation True
def test_p40_smoke_16_model_no_implementation():
    res = run_p40_forward_eligibility_gate_smoke()
    assert res["model_no_implementation"] is True

# 17. model_implementation_available False
def test_p40_smoke_17_model_implementation_available():
    res = run_p40_forward_eligibility_gate_smoke()
    assert res["model_implementation_available"] is False

# 18. model_forward_contract_declared True
def test_p40_smoke_18_forward_contract_declared():
    res = run_p40_forward_eligibility_gate_smoke()
    assert res["model_forward_contract_declared"] is True

# 19. model_forward_implemented_in_p27 False
def test_p40_smoke_19_forward_implemented_false():
    res = run_p40_forward_eligibility_gate_smoke()
    assert res["model_forward_implemented_in_p27"] is False

# 20. input_flat_dim_matches True
def test_p40_smoke_20_input_flat_dim_matches():
    res = run_p40_forward_eligibility_gate_smoke()
    assert res["input_flat_dim_matches"] is True

# 21. shape_matches True
def test_p40_smoke_21_shape_matches():
    res = run_p40_forward_eligibility_gate_smoke()
    assert res["shape_matches"] is True

# 22. dtype_matches True
def test_p40_smoke_22_dtype_matches():
    res = run_p40_forward_eligibility_gate_smoke()
    assert res["dtype_matches"] is True

# 23. device_matches True
def test_p40_smoke_23_device_matches():
    res = run_p40_forward_eligibility_gate_smoke()
    assert res["device_matches"] is True

# 24. latent_layout_matches True
def test_p40_smoke_24_latent_layout_matches():
    res = run_p40_forward_eligibility_gate_smoke()
    assert res["latent_layout_matches"] is True

# 25. decoder_contract_available True
def test_p40_smoke_25_decoder_contract_available():
    res = run_p40_forward_eligibility_gate_smoke()
    assert res["decoder_contract_available"] is True

# 26. forward_eligible_in_p40 False
def test_p40_smoke_26_forward_eligible():
    res = run_p40_forward_eligibility_gate_smoke()
    assert res["forward_eligible_in_p40"] is False

# 27. no_forward_execution True
def test_p40_smoke_27_no_forward():
    res = run_p40_forward_eligibility_gate_smoke()
    assert res["no_forward_execution"] is True
    assert res["forward_executed_in_p40"] is False

# 28. no_output_generation True
def test_p40_smoke_28_no_output():
    res = run_p40_forward_eligibility_gate_smoke()
    assert res["no_output_generation"] is True
    assert res["output_generated_in_p40"] is False

# 29. no_training_loop True
def test_p40_smoke_29_no_training():
    res = run_p40_forward_eligibility_gate_smoke()
    assert res["no_training_loop"] is True
    assert res["training_executed_in_p40"] is False

# 30. no_optimizer True
def test_p40_smoke_30_no_optimizer():
    res = run_p40_forward_eligibility_gate_smoke()
    assert res["no_optimizer"] is True

# 31. mock torch unavailable scenario - verdict
def test_p40_smoke_31_mock_torch_unavail_verdict(monkeypatch):
    monkeypatch.setattr(
        "src.phase2.fc_vae_forward_eligibility_gate.build_torch_dependency_status",
        lambda policy: TorchDependencyStatus(
            contract_version="phase2_p26_torch_boundary_contract_v1",
            backend_name="torch",
            available=False,
            policy="optional",
            import_safe=False,
            top_level_import_required=False,
            reason="mocked_unavailable",
        )
    )
    res = run_p40_forward_eligibility_gate_smoke()
    assert res["verdict"] == "PASS"
    assert res["status"] == "blocked_torch_unavailable"

# 32. mock torch unavailable - status
def test_p40_smoke_32_mock_torch_unavail_status(monkeypatch):
    monkeypatch.setattr(
        "src.phase2.fc_vae_forward_eligibility_gate.build_torch_dependency_status",
        lambda policy: TorchDependencyStatus(
            contract_version="phase2_p26_torch_boundary_contract_v1",
            backend_name="torch",
            available=False,
            policy="optional",
            import_safe=False,
            top_level_import_required=False,
            reason="mocked_unavailable",
        )
    )
    res = run_p40_forward_eligibility_gate_smoke()
    assert res["torch_available"] is False

# 33. mock torch unavailable - no_forward
def test_p40_smoke_33_mock_torch_unavail_no_forward(monkeypatch):
    monkeypatch.setattr(
        "src.phase2.fc_vae_forward_eligibility_gate.build_torch_dependency_status",
        lambda policy: TorchDependencyStatus(
            contract_version="phase2_p26_torch_boundary_contract_v1", backend_name="torch", available=False, policy="optional", import_safe=False, top_level_import_required=False, reason="mocked"
        )
    )
    res = run_p40_forward_eligibility_gate_smoke()
    assert res["no_forward_execution"] is True

# 34. mock torch unavailable - no_output
def test_p40_smoke_34_mock_torch_unavail_no_output(monkeypatch):
    monkeypatch.setattr(
        "src.phase2.fc_vae_forward_eligibility_gate.build_torch_dependency_status",
        lambda policy: TorchDependencyStatus(
            contract_version="phase2_p26_torch_boundary_contract_v1", backend_name="torch", available=False, policy="optional", import_safe=False, top_level_import_required=False, reason="mocked"
        )
    )
    res = run_p40_forward_eligibility_gate_smoke()
    assert res["no_output_generation"] is True

# 35. mock torch unavailable - no_training
def test_p40_smoke_35_mock_torch_unavail_no_training(monkeypatch):
    monkeypatch.setattr(
        "src.phase2.fc_vae_forward_eligibility_gate.build_torch_dependency_status",
        lambda policy: TorchDependencyStatus(
            contract_version="phase2_p26_torch_boundary_contract_v1", backend_name="torch", available=False, policy="optional", import_safe=False, top_level_import_required=False, reason="mocked"
        )
    )
    res = run_p40_forward_eligibility_gate_smoke()
    assert res["no_training_loop"] is True

# 36. mock torch unavailable - no_optimizer
def test_p40_smoke_36_mock_torch_unavail_no_optimizer(monkeypatch):
    monkeypatch.setattr(
        "src.phase2.fc_vae_forward_eligibility_gate.build_torch_dependency_status",
        lambda policy: TorchDependencyStatus(
            contract_version="phase2_p26_torch_boundary_contract_v1", backend_name="torch", available=False, policy="optional", import_safe=False, top_level_import_required=False, reason="mocked"
        )
    )
    res = run_p40_forward_eligibility_gate_smoke()
    assert res["no_optimizer"] is True

# 37. mock torch unavailable - no_checkpoint
def test_p40_smoke_37_mock_torch_unavail_no_checkpoint(monkeypatch):
    monkeypatch.setattr(
        "src.phase2.fc_vae_forward_eligibility_gate.build_torch_dependency_status",
        lambda policy: TorchDependencyStatus(
            contract_version="phase2_p26_torch_boundary_contract_v1", backend_name="torch", available=False, policy="optional", import_safe=False, top_level_import_required=False, reason="mocked"
        )
    )
    res = run_p40_forward_eligibility_gate_smoke()
    assert res["no_checkpointing"] is True

# 38. mock tensor unavailable - verdict
def test_p40_smoke_38_mock_tensor_unavail_verdict(monkeypatch):
    monkeypatch.setattr(
        "src.phase2.fc_vae_forward_eligibility_gate.run_tensor_materialization_probe",
        lambda: FCVAETensorMaterializationResult(
            contract_version="phase2_p39_torch_tensor_materialization_contract_v1",
            request=build_forward_eligibility_request_from_defaults(), # close dummy
            metadata=None,
            status="blocked_torch_unavailable",
            materialized_tensor=None,
            tensor_materialization_available_in_p39=False,
            tensor_materialized_in_p39=False,
            forward_execution_available_in_p39=False,
            output_generation_available_in_p39=False,
            training_available_in_p39=False,
            no_forward_execution=True,
            no_output_generation=True,
            no_training_loop=True,
            no_optimizer=True,
            no_checkpointing=True,
            no_artifact_generation=True,
            no_final_comparison=True,
            no_scientific_conclusion=True,
            reason="mocked_unavailable",
        )
    )
    res = run_p40_forward_eligibility_gate_smoke()
    assert res["verdict"] == "PASS"
    assert res["status"] == "blocked_by_tensor_materialization_unavailable"

# 39. mock tensor unavailable - flag
def test_p40_smoke_39_mock_tensor_unavail_flag(monkeypatch):
    monkeypatch.setattr(
        "src.phase2.fc_vae_forward_eligibility_gate.run_tensor_materialization_probe",
        lambda: FCVAETensorMaterializationResult(
            contract_version="phase2_p39_torch_tensor_materialization_contract_v1",
            request=build_forward_eligibility_request_from_defaults(),
            metadata=None,
            status="blocked_torch_unavailable",
            materialized_tensor=None,
            tensor_materialization_available_in_p39=False,
            tensor_materialized_in_p39=False,
            forward_execution_available_in_p39=False,
            output_generation_available_in_p39=False,
            training_available_in_p39=False,
            no_forward_execution=True,
            no_output_generation=True,
            no_training_loop=True,
            no_optimizer=True,
            no_checkpointing=True,
            no_artifact_generation=True,
            no_final_comparison=True,
            no_scientific_conclusion=True,
            reason="mocked_unavailable",
        )
    )
    res = run_p40_forward_eligibility_gate_smoke()
    assert res["tensor_materialized_in_p39"] is False

# 40. mock tensor unavailable - no_forward
def test_p40_smoke_40_mock_tensor_unavail_no_forward(monkeypatch):
    monkeypatch.setattr(
        "src.phase2.fc_vae_forward_eligibility_gate.run_tensor_materialization_probe",
        lambda: FCVAETensorMaterializationResult(
            contract_version="phase2_p39_torch_tensor_materialization_contract_v1",
            request=build_forward_eligibility_request_from_defaults(),
            metadata=None,
            status="blocked_torch_unavailable",
            materialized_tensor=None,
            tensor_materialization_available_in_p39=False,
            tensor_materialized_in_p39=False,
            forward_execution_available_in_p39=False,
            output_generation_available_in_p39=False,
            training_available_in_p39=False,
            no_forward_execution=True,
            no_output_generation=True,
            no_training_loop=True,
            no_optimizer=True,
            no_checkpointing=True,
            no_artifact_generation=True,
            no_final_comparison=True,
            no_scientific_conclusion=True,
            reason="mocked_unavailable",
        )
    )
    res = run_p40_forward_eligibility_gate_smoke()
    assert res["no_forward_execution"] is True

# 41. mock tensor unavailable - no_output
def test_p40_smoke_41_mock_tensor_unavail_no_output(monkeypatch):
    monkeypatch.setattr(
        "src.phase2.fc_vae_forward_eligibility_gate.run_tensor_materialization_probe",
        lambda: FCVAETensorMaterializationResult(
            contract_version="phase2_p39_torch_tensor_materialization_contract_v1",
            request=build_forward_eligibility_request_from_defaults(),
            metadata=None,
            status="blocked_torch_unavailable",
            materialized_tensor=None,
            tensor_materialization_available_in_p39=False,
            tensor_materialized_in_p39=False,
            forward_execution_available_in_p39=False,
            output_generation_available_in_p39=False,
            training_available_in_p39=False,
            no_forward_execution=True,
            no_output_generation=True,
            no_training_loop=True,
            no_optimizer=True,
            no_checkpointing=True,
            no_artifact_generation=True,
            no_final_comparison=True,
            no_scientific_conclusion=True,
            reason="mocked_unavailable",
        )
    )
    res = run_p40_forward_eligibility_gate_smoke()
    assert res["no_output_generation"] is True

# 42. mock tensor unavailable - no_training
def test_p40_smoke_42_mock_tensor_unavail_no_training(monkeypatch):
    monkeypatch.setattr(
        "src.phase2.fc_vae_forward_eligibility_gate.run_tensor_materialization_probe",
        lambda: FCVAETensorMaterializationResult(
            contract_version="phase2_p39_torch_tensor_materialization_contract_v1",
            request=build_forward_eligibility_request_from_defaults(),
            metadata=None,
            status="blocked_torch_unavailable",
            materialized_tensor=None,
            tensor_materialization_available_in_p39=False,
            tensor_materialized_in_p39=False,
            forward_execution_available_in_p39=False,
            output_generation_available_in_p39=False,
            training_available_in_p39=False,
            no_forward_execution=True,
            no_output_generation=True,
            no_training_loop=True,
            no_optimizer=True,
            no_checkpointing=True,
            no_artifact_generation=True,
            no_final_comparison=True,
            no_scientific_conclusion=True,
            reason="mocked_unavailable",
        )
    )
    res = run_p40_forward_eligibility_gate_smoke()
    assert res["no_training_loop"] is True

# 43. mock tensor unavailable - no_optimizer
def test_p40_smoke_43_mock_tensor_unavail_no_optimizer(monkeypatch):
    monkeypatch.setattr(
        "src.phase2.fc_vae_forward_eligibility_gate.run_tensor_materialization_probe",
        lambda: FCVAETensorMaterializationResult(
            contract_version="phase2_p39_torch_tensor_materialization_contract_v1",
            request=build_forward_eligibility_request_from_defaults(),
            metadata=None,
            status="blocked_torch_unavailable",
            materialized_tensor=None,
            tensor_materialization_available_in_p39=False,
            tensor_materialized_in_p39=False,
            forward_execution_available_in_p39=False,
            output_generation_available_in_p39=False,
            training_available_in_p39=False,
            no_forward_execution=True,
            no_output_generation=True,
            no_training_loop=True,
            no_optimizer=True,
            no_checkpointing=True,
            no_artifact_generation=True,
            no_final_comparison=True,
            no_scientific_conclusion=True,
            reason="mocked_unavailable",
        )
    )
    res = run_p40_forward_eligibility_gate_smoke()
    assert res["no_optimizer"] is True

# 44. mock tensor unavailable - no_checkpoint
def test_p40_smoke_44_mock_tensor_unavail_no_checkpoint(monkeypatch):
    monkeypatch.setattr(
        "src.phase2.fc_vae_forward_eligibility_gate.run_tensor_materialization_probe",
        lambda: FCVAETensorMaterializationResult(
            contract_version="phase2_p39_torch_tensor_materialization_contract_v1",
            request=build_forward_eligibility_request_from_defaults(),
            metadata=None,
            status="blocked_torch_unavailable",
            materialized_tensor=None,
            tensor_materialization_available_in_p39=False,
            tensor_materialized_in_p39=False,
            forward_execution_available_in_p39=False,
            output_generation_available_in_p39=False,
            training_available_in_p39=False,
            no_forward_execution=True,
            no_output_generation=True,
            no_training_loop=True,
            no_optimizer=True,
            no_checkpointing=True,
            no_artifact_generation=True,
            no_final_comparison=True,
            no_scientific_conclusion=True,
            reason="mocked_unavailable",
        )
    )
    res = run_p40_forward_eligibility_gate_smoke()
    assert res["no_checkpointing"] is True


# 45. JSON parseable
def test_p40_smoke_45_json_parseable():
    res = run_p40_forward_eligibility_gate_smoke()
    js = compact_json(res)
    parsed = json.loads(js)
    assert parsed["verdict"] == "PASS"

# 46. smoke output has no raw tensor object
def test_p40_smoke_46_no_raw_tensor_object():
    res = run_p40_forward_eligibility_gate_smoke()
    js = compact_json(res)
    assert "MockTensor" not in js
    assert "Tensor" not in js

# 47. smoke output has no nested_batch_values
def test_p40_smoke_47_no_nested_batch_values():
    res = run_p40_forward_eligibility_gate_smoke()
    js = compact_json(res)
    assert "nested_batch_values" not in js

# 48. smoke output has no flat_values
def test_p40_smoke_48_no_flat_values():
    res = run_p40_forward_eligibility_gate_smoke()
    js = compact_json(res)
    assert "flat_values" not in js

# 49. no forbidden success claims
def test_p40_smoke_49_no_success_claims():
    res = run_p40_forward_eligibility_gate_smoke()
    js = compact_json(res)
    assert "model works" not in js.lower()

# 50. smoke script imports clean
def test_p40_smoke_50_imports_clean():
    content = pathlib.Path("tools/phase2/run_p40_forward_eligibility_gate_smoke.py").read_text(encoding="utf-8")
    for pattern in ("import torch", "from torch", "import numpy", "from numpy"):
        assert pattern not in content
