# tests/test_phase2_p39_tensor_materialization_smoke.py

import json
import pathlib
import pytest
import sys
from typing import Any, Tuple


from tools.phase2.run_p39_tensor_materialization_smoke import (
    run_p39_tensor_materialization_smoke,
    compact_json,
    main,
)

# Mock classes for testing when torch is available
class MockDevice:
    def __init__(self, device_type: str):
        self.type = device_type

class MockTensor:
    def __init__(self, data: Any, dtype: str, device: str):
        self.data = data
        self.dtype = dtype
        self.device = MockDevice(device)
        self.requires_grad = False
    def numel(self) -> int:
        return 64
    def is_floating_point(self) -> bool:
        return True
    def detach(self) -> Any:
        return self
    def cpu(self) -> Any:
        return self
    def tolist(self) -> Any:
        return [list(row) for row in self.data]
    @property
    def shape(self) -> Tuple[int, int]:
        return (2, 32)
    def requires_grad_(self, val: bool) -> None:
        self.requires_grad = val

class MockCuda:
    @staticmethod
    def is_available() -> bool:
        return False

class MockTorch:
    __version__ = "2.0.0+mock"
    float32 = "torch.float32"
    cuda = MockCuda
    @staticmethod
    def tensor(data: Any, dtype: str, device: str) -> MockTensor:
        return MockTensor(data, dtype, device)


# 1. compact_json sorts keys
def test_p39_smoke_01_compact_json_sorts():
    data = {"b": 2, "a": 1}
    js = compact_json(data)
    assert js == '{"a":1,"b":2}'

# 2. main rejects args
def test_p39_smoke_02_main_rejects_args():
    orig = sys.argv
    try:
        sys.argv = ["script.py", "extra"]
        code = main()
        assert code != 0
    finally:
        sys.argv = orig

# 3. smoke returns BLOCKED if torch unavailable
def test_p39_smoke_03_verdict_blocked_if_torch_unavailable():
    res = run_p39_tensor_materialization_smoke()
    assert res["verdict"] == "BLOCKED"

# 4. status blocked if torch unavailable
def test_p39_smoke_04_status_blocked_if_torch_unavailable():
    res = run_p39_tensor_materialization_smoke()
    assert res["status"] == "blocked_torch_unavailable"

# 5. torch_available false if torch unavailable
def test_p39_smoke_05_torch_available_false_if_torch_unavailable():
    res = run_p39_tensor_materialization_smoke()
    assert res["torch_available"] is False

# 6. tensor_materialized_in_p39 false if torch unavailable
def test_p39_smoke_06_tensor_materialized_in_p39_false_if_torch_unavailable():
    res = run_p39_tensor_materialization_smoke()
    assert res["tensor_materialized_in_p39"] is False

# 7. contract exact
def test_p39_smoke_07_contract_exact():
    res = run_p39_tensor_materialization_smoke()
    assert res["contract"] == "phase2_p39_torch_tensor_materialization_contract_v1"

# 8. source_phase P39
def test_p39_smoke_08_source_phase():
    res = run_p39_tensor_materialization_smoke()
    assert res["source_phase"] == "P39"

# 9. target_framework
def test_p39_smoke_09_target_framework():
    res = run_p39_tensor_materialization_smoke()
    assert res["target_framework"] == "torch"

# 10. target_dtype
def test_p39_smoke_10_target_dtype():
    res = run_p39_tensor_materialization_smoke()
    assert res["target_dtype"] == "float32"

# 11. target_device
def test_p39_smoke_11_target_device():
    res = run_p39_tensor_materialization_smoke()
    assert res["target_device"] == "cpu"

# 12. target_layout_kind
def test_p39_smoke_12_target_layout_kind():
    res = run_p39_tensor_materialization_smoke()
    assert res["target_layout_kind"] == "row_major_2d_batch_tensor"

# 13. shape_tuple
def test_p39_smoke_13_shape_tuple():
    res = run_p39_tensor_materialization_smoke()
    assert res["shape_tuple"] == [2, 32]

# 14. batch_size
def test_p39_smoke_14_batch_size():
    res = run_p39_tensor_materialization_smoke()
    assert res["batch_size"] == 2

# 15. input_flat_dim
def test_p39_smoke_15_input_flat_dim():
    res = run_p39_tensor_materialization_smoke()
    assert res["input_flat_dim"] == 32

# 16. full_vector_length
def test_p39_smoke_16_full_vector_length():
    res = run_p39_tensor_materialization_smoke()
    assert res["full_vector_length"] == 64

# 17. no_forward_execution
def test_p39_smoke_17_no_forward_execution():
    res = run_p39_tensor_materialization_smoke()
    assert res["no_forward_execution"] is True

# 18. no_output_generation
def test_p39_smoke_18_no_output_generation():
    res = run_p39_tensor_materialization_smoke()
    assert res["no_output_generation"] is True

# 19. no_training_loop
def test_p39_smoke_19_no_training_loop():
    res = run_p39_tensor_materialization_smoke()
    assert res["no_training_loop"] is True

# 20. no_optimizer
def test_p39_smoke_20_no_optimizer():
    res = run_p39_tensor_materialization_smoke()
    assert res["no_optimizer"] is True

# 21. no_checkpointing
def test_p39_smoke_21_no_checkpointing():
    res = run_p39_tensor_materialization_smoke()
    assert res["no_checkpointing"] is True

# 22. no_artifact_generation
def test_p39_smoke_22_no_artifact_generation():
    res = run_p39_tensor_materialization_smoke()
    assert res["no_artifact_generation"] is True

# 23. no_final_comparison
def test_p39_smoke_23_no_final_comparison():
    res = run_p39_tensor_materialization_smoke()
    assert res["no_final_comparison"] is True

# 24. no_scientific_conclusion
def test_p39_smoke_24_no_scientific_conclusion():
    res = run_p39_tensor_materialization_smoke()
    assert res["no_scientific_conclusion"] is True

# 25. JSON parseable
def test_p39_smoke_25_json_parseable():
    res = run_p39_tensor_materialization_smoke()
    js = compact_json(res)
    parsed = json.loads(js)
    assert parsed["verdict"] == "BLOCKED"

# 26. smoke output has no raw tensor object
def test_p39_smoke_26_no_raw_tensor_object():
    res = run_p39_tensor_materialization_smoke()
    js = compact_json(res)
    assert "MockTensor" not in js
    assert "Tensor" not in js

# 27. smoke output has no full nested_batch_values
def test_p39_smoke_27_no_full_nested_batch_values():
    res = run_p39_tensor_materialization_smoke()
    js = compact_json(res)
    assert '"nested_batch_values":' not in js

# 28. smoke output has no flat_values
def test_p39_smoke_28_no_flat_values():
    res = run_p39_tensor_materialization_smoke()
    js = compact_json(res)
    assert "flat_values" not in js

# 29. smoke output has no key with ": bool"
def test_p39_smoke_29_no_key_with_colon_bool():
    res = run_p39_tensor_materialization_smoke()
    js = compact_json(res)
    assert ": bool" not in js

# 30. smoke output has no success/scientific claims
def test_p39_smoke_30_no_success_claims():
    res = run_p39_tensor_materialization_smoke()
    js = compact_json(res)
    forbidden = ["model works", "scientific success", "solved", "best", "winner", "production ready", "state of the art"]
    for word in forbidden:
        assert word not in js.lower()

# 31. smoke script imports clean
def test_p39_smoke_31_imports_clean():
    content = pathlib.Path("tools/phase2/run_p39_tensor_materialization_smoke.py").read_text(encoding="utf-8")
    for pattern in ("import torch", "from torch", "import numpy", "from numpy"):
        assert pattern not in content


# 32-44: mocked tests when torch is available
def test_p39_smoke_32_verdict_pass_if_torch_available_mocked(monkeypatch):
    monkeypatch.setattr("src.phase2.fc_vae_tensor_materialization.load_torch_for_p39_materialization", lambda: MockTorch)
    from src.phase2.torch_boundary import TorchDependencyStatus
    monkeypatch.setattr("src.phase2.fc_vae_tensor_materialization.build_torch_dependency_status", lambda policy: TorchDependencyStatus(
        contract_version="phase2_p26_torch_boundary_contract_v1", backend_name="torch", available=True, policy="optional", import_safe=True, top_level_import_required=False, reason="mocked"
    ))
    res = run_p39_tensor_materialization_smoke()
    assert res["verdict"] == "PASS"

def test_p39_smoke_33_status_materialized_if_torch_available_mocked(monkeypatch):
    monkeypatch.setattr("src.phase2.fc_vae_tensor_materialization.load_torch_for_p39_materialization", lambda: MockTorch)
    from src.phase2.torch_boundary import TorchDependencyStatus
    monkeypatch.setattr("src.phase2.fc_vae_tensor_materialization.build_torch_dependency_status", lambda policy: TorchDependencyStatus(
        contract_version="phase2_p26_torch_boundary_contract_v1", backend_name="torch", available=True, policy="optional", import_safe=True, top_level_import_required=False, reason="mocked"
    ))
    res = run_p39_tensor_materialization_smoke()
    assert res["status"] == "torch_tensor_materialized_no_forward_no_training_in_p39"

def test_p39_smoke_34_torch_available_true_if_torch_available_mocked(monkeypatch):
    monkeypatch.setattr("src.phase2.fc_vae_tensor_materialization.load_torch_for_p39_materialization", lambda: MockTorch)
    from src.phase2.torch_boundary import TorchDependencyStatus
    monkeypatch.setattr("src.phase2.fc_vae_tensor_materialization.build_torch_dependency_status", lambda policy: TorchDependencyStatus(
        contract_version="phase2_p26_torch_boundary_contract_v1", backend_name="torch", available=True, policy="optional", import_safe=True, top_level_import_required=False, reason="mocked"
    ))
    res = run_p39_tensor_materialization_smoke()
    assert res["torch_available"] is True

def test_p39_smoke_35_tensor_materialized_in_p39_true_if_torch_available_mocked(monkeypatch):
    monkeypatch.setattr("src.phase2.fc_vae_tensor_materialization.load_torch_for_p39_materialization", lambda: MockTorch)
    from src.phase2.torch_boundary import TorchDependencyStatus
    monkeypatch.setattr("src.phase2.fc_vae_tensor_materialization.build_torch_dependency_status", lambda policy: TorchDependencyStatus(
        contract_version="phase2_p26_torch_boundary_contract_v1", backend_name="torch", available=True, policy="optional", import_safe=True, top_level_import_required=False, reason="mocked"
    ))
    res = run_p39_tensor_materialization_smoke()
    assert res["tensor_materialized_in_p39"] is True

def test_p39_smoke_36_tensor_shape_tuple_if_torch_available_mocked(monkeypatch):
    monkeypatch.setattr("src.phase2.fc_vae_tensor_materialization.load_torch_for_p39_materialization", lambda: MockTorch)
    from src.phase2.torch_boundary import TorchDependencyStatus
    monkeypatch.setattr("src.phase2.fc_vae_tensor_materialization.build_torch_dependency_status", lambda policy: TorchDependencyStatus(
        contract_version="phase2_p26_torch_boundary_contract_v1", backend_name="torch", available=True, policy="optional", import_safe=True, top_level_import_required=False, reason="mocked"
    ))
    res = run_p39_tensor_materialization_smoke()
    assert res["tensor_shape_tuple"] == [2, 32]

def test_p39_smoke_37_tensor_numel_if_torch_available_mocked(monkeypatch):
    monkeypatch.setattr("src.phase2.fc_vae_tensor_materialization.load_torch_for_p39_materialization", lambda: MockTorch)
    from src.phase2.torch_boundary import TorchDependencyStatus
    monkeypatch.setattr("src.phase2.fc_vae_tensor_materialization.build_torch_dependency_status", lambda policy: TorchDependencyStatus(
        contract_version="phase2_p26_torch_boundary_contract_v1", backend_name="torch", available=True, policy="optional", import_safe=True, top_level_import_required=False, reason="mocked"
    ))
    res = run_p39_tensor_materialization_smoke()
    assert res["tensor_numel"] == 64

def test_p39_smoke_38_tensor_device_type_if_torch_available_mocked(monkeypatch):
    monkeypatch.setattr("src.phase2.fc_vae_tensor_materialization.load_torch_for_p39_materialization", lambda: MockTorch)
    from src.phase2.torch_boundary import TorchDependencyStatus
    monkeypatch.setattr("src.phase2.fc_vae_tensor_materialization.build_torch_dependency_status", lambda policy: TorchDependencyStatus(
        contract_version="phase2_p26_torch_boundary_contract_v1", backend_name="torch", available=True, policy="optional", import_safe=True, top_level_import_required=False, reason="mocked"
    ))
    res = run_p39_tensor_materialization_smoke()
    assert res["tensor_device_type"] == "cpu"

def test_p39_smoke_39_tensor_dtype_name_if_torch_available_mocked(monkeypatch):
    monkeypatch.setattr("src.phase2.fc_vae_tensor_materialization.load_torch_for_p39_materialization", lambda: MockTorch)
    from src.phase2.torch_boundary import TorchDependencyStatus
    monkeypatch.setattr("src.phase2.fc_vae_tensor_materialization.build_torch_dependency_status", lambda policy: TorchDependencyStatus(
        contract_version="phase2_p26_torch_boundary_contract_v1", backend_name="torch", available=True, policy="optional", import_safe=True, top_level_import_required=False, reason="mocked"
    ))
    res = run_p39_tensor_materialization_smoke()
    assert res["tensor_dtype_name"] == "torch.float32"

def test_p39_smoke_40_tensor_requires_grad_if_torch_available_mocked(monkeypatch):
    monkeypatch.setattr("src.phase2.fc_vae_tensor_materialization.load_torch_for_p39_materialization", lambda: MockTorch)
    from src.phase2.torch_boundary import TorchDependencyStatus
    monkeypatch.setattr("src.phase2.fc_vae_tensor_materialization.build_torch_dependency_status", lambda policy: TorchDependencyStatus(
        contract_version="phase2_p26_torch_boundary_contract_v1", backend_name="torch", available=True, policy="optional", import_safe=True, top_level_import_required=False, reason="mocked"
    ))
    res = run_p39_tensor_materialization_smoke()
    assert res["tensor_requires_grad"] is False

def test_p39_smoke_41_tensor_is_floating_point_if_torch_available_mocked(monkeypatch):
    monkeypatch.setattr("src.phase2.fc_vae_tensor_materialization.load_torch_for_p39_materialization", lambda: MockTorch)
    from src.phase2.torch_boundary import TorchDependencyStatus
    monkeypatch.setattr("src.phase2.fc_vae_tensor_materialization.build_torch_dependency_status", lambda policy: TorchDependencyStatus(
        contract_version="phase2_p26_torch_boundary_contract_v1", backend_name="torch", available=True, policy="optional", import_safe=True, top_level_import_required=False, reason="mocked"
    ))
    res = run_p39_tensor_materialization_smoke()
    assert res["tensor_is_floating_point"] is True

def test_p39_smoke_42_tensor_values_match_if_torch_available_mocked(monkeypatch):
    monkeypatch.setattr("src.phase2.fc_vae_tensor_materialization.load_torch_for_p39_materialization", lambda: MockTorch)
    from src.phase2.torch_boundary import TorchDependencyStatus
    monkeypatch.setattr("src.phase2.fc_vae_tensor_materialization.build_torch_dependency_status", lambda policy: TorchDependencyStatus(
        contract_version="phase2_p26_torch_boundary_contract_v1", backend_name="torch", available=True, policy="optional", import_safe=True, top_level_import_required=False, reason="mocked"
    ))
    res = run_p39_tensor_materialization_smoke()
    assert res["tensor_values_match_p37_nested_values"] is True

def test_p39_smoke_43_first_row_first_4_values_if_torch_available_mocked(monkeypatch):
    monkeypatch.setattr("src.phase2.fc_vae_tensor_materialization.load_torch_for_p39_materialization", lambda: MockTorch)
    from src.phase2.torch_boundary import TorchDependencyStatus
    monkeypatch.setattr("src.phase2.fc_vae_tensor_materialization.build_torch_dependency_status", lambda policy: TorchDependencyStatus(
        contract_version="phase2_p26_torch_boundary_contract_v1", backend_name="torch", available=True, policy="optional", import_safe=True, top_level_import_required=False, reason="mocked"
    ))
    res = run_p39_tensor_materialization_smoke()
    expected = [-0.24297189, -0.1686747, -0.09437751, -0.02008032]
    for v, exp in zip(res["first_row_first_4_values"], expected):
        assert abs(v - exp) < 1e-6

def test_p39_smoke_44_second_row_first_4_values_if_torch_available_mocked(monkeypatch):
    monkeypatch.setattr("src.phase2.fc_vae_tensor_materialization.load_torch_for_p39_materialization", lambda: MockTorch)
    from src.phase2.torch_boundary import TorchDependencyStatus
    monkeypatch.setattr("src.phase2.fc_vae_tensor_materialization.build_torch_dependency_status", lambda policy: TorchDependencyStatus(
        contract_version="phase2_p26_torch_boundary_contract_v1", backend_name="torch", available=True, policy="optional", import_safe=True, top_level_import_required=False, reason="mocked"
    ))
    res = run_p39_tensor_materialization_smoke()
    expected = [0.13253012, 0.20682731, 0.2811245, 0.35542169]
    for v, exp in zip(res["second_row_first_4_values"], expected):
        assert abs(v - exp) < 1e-6


# 45-50: availability fields & main
def test_p39_smoke_45_forward_execution_available_false():
    res = run_p39_tensor_materialization_smoke()
    assert res["forward_execution_available_in_p39"] is False

def test_p39_smoke_46_output_generation_available_false():
    res = run_p39_tensor_materialization_smoke()
    assert res["output_generation_available_in_p39"] is False

def test_p39_smoke_47_training_available_false():
    res = run_p39_tensor_materialization_smoke()
    assert res["training_available_in_p39"] is False

def test_p39_smoke_48_main_no_args_ok():
    orig = sys.argv
    try:
        sys.argv = ["script.py"]
        code = main()
        assert code == 0
    finally:
        sys.argv = orig

def test_p39_smoke_49_torch_available_field_present():
    res = run_p39_tensor_materialization_smoke()
    assert "torch_available" in res

def test_p39_smoke_50_verdict_present():
    res = run_p39_tensor_materialization_smoke()
    assert "verdict" in res
