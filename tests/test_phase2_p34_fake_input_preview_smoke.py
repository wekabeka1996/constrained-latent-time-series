# tests/test_phase2_p34_fake_input_preview_smoke.py

import json
import pathlib
import pytest

from tools.phase2.run_p34_fake_input_preview_smoke import (
    run_p34_fake_input_preview_smoke,
    compact_json,
    main,
)


# 1. compact_json sorts keys
def test_p33_smoke_01_compact_json_sorts():
    data = {"b": 2, "a": 1}
    js = compact_json(data)
    assert js == '{"a":1,"b":2}'


# 2. main rejects args
def test_p33_smoke_02_main_rejects_args():
    import sys
    orig = sys.argv
    try:
        sys.argv = ["script.py", "extra"]
        code = main()
        assert code != 0
    finally:
        sys.argv = orig


# 3. smoke returns PASS
def test_p33_smoke_03_verdict_pass():
    res = run_p34_fake_input_preview_smoke()
    assert res["verdict"] == "PASS"


# 4. contract matches
def test_p33_smoke_04_contract_matches():
    res = run_p34_fake_input_preview_smoke()
    assert res["contract"] == "phase2_p34_fake_input_scalar_preview_contract_v1"


# 5. source_phase P34
def test_p33_smoke_05_source_phase():
    res = run_p34_fake_input_preview_smoke()
    assert res["source_phase"] == "P34"


# 6. descriptor_seed == 1337
def test_p33_smoke_06_descriptor_seed():
    res = run_p34_fake_input_preview_smoke()
    assert res["descriptor_seed"] == 1337


# 7. min_value == -1.0
def test_p33_smoke_07_min_value():
    res = run_p34_fake_input_preview_smoke()
    assert res["min_value"] == -1.0


# 8. max_value == 1.0
def test_p33_smoke_08_max_value():
    res = run_p34_fake_input_preview_smoke()
    assert res["max_value"] == 1.0


# 9. batch_size == 2
def test_p33_smoke_09_batch_size():
    res = run_p34_fake_input_preview_smoke()
    assert res["batch_size"] == 2


# 10. input_flat_dim == 32
def test_p33_smoke_10_input_flat_dim():
    res = run_p34_fake_input_preview_smoke()
    assert res["input_flat_dim"] == 32


# 11. full_batch_scalar_count == 64
def test_p33_smoke_11_full_batch_scalar_count():
    res = run_p34_fake_input_preview_smoke()
    assert res["full_batch_scalar_count"] == 64


# 12. preview_value_count == 4
def test_p33_smoke_12_preview_value_count():
    res = run_p34_fake_input_preview_smoke()
    assert res["preview_value_count"] == 4


# 13. preview_values is a JSON list of 4 floats
def test_p33_smoke_13_preview_values_format():
    res = run_p34_fake_input_preview_smoke()
    assert isinstance(res["preview_values"], list)
    assert len(res["preview_values"]) == 4
    for val in res["preview_values"]:
        assert isinstance(val, float)


# 14. every preview value is within range
def test_p33_smoke_14_preview_values_range():
    res = run_p34_fake_input_preview_smoke()
    for val in res["preview_values"]:
        assert val >= -1.0
        assert val <= 1.0


# 15. preview_is_partial True
def test_p33_smoke_15_preview_is_partial():
    res = run_p34_fake_input_preview_smoke()
    assert res["preview_is_partial"] is True


# 16. all_values_within_range True
def test_p33_smoke_16_all_within_range():
    res = run_p34_fake_input_preview_smoke()
    assert res["all_values_within_range"] is True


# 17. preview_available_in_p34 True
def test_p33_smoke_17_preview_available():
    res = run_p34_fake_input_preview_smoke()
    assert res["preview_available_in_p34"] is True


# 18. full_batch_available_in_p34 False
def test_p33_smoke_18_full_batch_available():
    res = run_p34_fake_input_preview_smoke()
    assert res["full_batch_available_in_p34"] is False


# 19. rng_execution_attempted False
def test_p33_smoke_19_rng_execution_attempted():
    res = run_p34_fake_input_preview_smoke()
    assert res["rng_execution_attempted"] is False


# 20. full_batch_materialization_attempted False
def test_p33_smoke_20_full_batch_materialization_attempted():
    res = run_p34_fake_input_preview_smoke()
    assert res["full_batch_materialization_attempted"] is False


# 21. array_materialization_attempted False
def test_p33_smoke_21_array_materialization_attempted():
    res = run_p34_fake_input_preview_smoke()
    assert res["array_materialization_attempted"] is False


# 22. tensor_materialization_attempted False
def test_p33_smoke_22_tensor_materialization_attempted():
    res = run_p34_fake_input_preview_smoke()
    assert res["tensor_materialization_attempted"] is False


# 23. forward_execution_attempted False
def test_p33_smoke_23_forward_execution_attempted():
    res = run_p34_fake_input_preview_smoke()
    assert res["forward_execution_attempted"] is False


# 24. output_generation_attempted False
def test_p33_smoke_24_output_generation_attempted():
    res = run_p34_fake_input_preview_smoke()
    assert res["output_generation_attempted"] is False


# 25. rng_executed False
def test_p33_smoke_25_rng_executed():
    res = run_p34_fake_input_preview_smoke()
    assert res["rng_executed"] is False


# 26. full_batch_materialized False
def test_p33_smoke_26_full_batch_materialized():
    res = run_p34_fake_input_preview_smoke()
    assert res["full_batch_materialized"] is False


# 27. array_materialized False
def test_p33_smoke_27_array_materialized():
    res = run_p34_fake_input_preview_smoke()
    assert res["array_materialized"] is False


# 28. tensor_materialized False
def test_p33_smoke_28_tensor_materialized():
    res = run_p34_fake_input_preview_smoke()
    assert res["tensor_materialized"] is False


# 29. all no_* flags True
def test_p33_smoke_29_no_flags():
    res = run_p34_fake_input_preview_smoke()
    assert res["no_rng_execution"] is True
    assert res["no_full_batch_materialized"] is True
    assert res["no_array_created"] is True
    assert res["no_tensor_created"] is True
    assert res["no_forward_execution"] is True
    assert res["no_output_generation"] is True
    assert res["no_training_loop"] is True
    assert res["no_optimizer"] is True
    assert res["no_checkpointing"] is True
    assert res["no_artifact_generation"] is True
    assert res["no_final_comparison"] is True
    assert res["no_scientific_conclusion"] is True


# 30. JSON parseable
def test_p33_smoke_30_json_parseable():
    res = run_p34_fake_input_preview_smoke()
    js = compact_json(res)
    d = json.loads(js)
    assert d["verdict"] == "PASS"


# 31. smoke script no torch import
def test_p33_smoke_31_no_torch_import():
    content = pathlib.Path("tools/phase2/run_p34_fake_input_preview_smoke.py").read_text(encoding="utf-8")
    assert "import torch" not in content
    assert "from torch" not in content


# 32. smoke script no numpy import
def test_p33_smoke_32_no_numpy_import():
    content = pathlib.Path("tools/phase2/run_p34_fake_input_preview_smoke.py").read_text(encoding="utf-8")
    assert "import numpy" not in content
    assert "from numpy" not in content


# 33. smoke script no random/secrets import
def test_p33_smoke_33_no_random_secrets():
    content = pathlib.Path("tools/phase2/run_p34_fake_input_preview_smoke.py").read_text(encoding="utf-8")
    assert "import random" not in content
    assert "from random" not in content
    assert "import secrets" not in content
    assert "from secrets" not in content


# 34. smoke script no argparse
def test_p33_smoke_34_no_argparse():
    content = pathlib.Path("tools/phase2/run_p34_fake_input_preview_smoke.py").read_text(encoding="utf-8")
    assert "argparse" not in content


# 35. smoke script no subprocess
def test_p33_smoke_35_no_subprocess():
    content = pathlib.Path("tools/phase2/run_p34_fake_input_preview_smoke.py").read_text(encoding="utf-8")
    assert "subprocess" not in content


# 36. smoke output no success/scientific claims
def test_p33_smoke_36_no_claims():
    res = run_p34_fake_input_preview_smoke()
    js = compact_json(res)
    normalized = js.lower()
    cleaned = (normalized
               .replace("no_final_comparison", "")
               .replace("no_scientific_conclusion", "")
               .replace("scalar_preview_only_no_full_batch_in_p34", ""))
    for claim in ("model works", "scientific success", "solved", "best", "winner", "production ready", "state of the art"):
        assert claim not in cleaned


# 37. smoke output has no full batch payload
def test_p33_smoke_37_no_full_batch_payload():
    res = run_p34_fake_input_preview_smoke()
    js = compact_json(res)
    d = json.loads(js)
    assert "full_batch" not in d
    assert "batch_data" not in d
    if "result" in d:
        assert "full_batch" not in d["result"]
        assert "batch_data" not in d["result"]


# 38. smoke output has no nested 2D values
def test_p33_smoke_38_no_nested_2d():
    res = run_p34_fake_input_preview_smoke()
    js = compact_json(res)
    assert "[[" not in js
