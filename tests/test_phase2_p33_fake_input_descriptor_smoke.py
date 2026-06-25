# tests/test_phase2_p33_fake_input_descriptor_smoke.py

import json
import pathlib
import pytest

from tools.phase2.run_p33_fake_input_descriptor_smoke import (
    run_p33_fake_input_descriptor_smoke,
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
    res = run_p33_fake_input_descriptor_smoke()
    assert res["verdict"] == "PASS"


# 4. contract matches
def test_p33_smoke_04_contract_matches():
    res = run_p33_fake_input_descriptor_smoke()
    assert res["contract"] == "phase2_p33_fake_input_descriptor_contract_v1"


# 5. source_phase P33
def test_p33_smoke_05_source_phase():
    res = run_p33_fake_input_descriptor_smoke()
    assert res["source_phase"] == "P33"


# 6. descriptor_seed == 1337
def test_p33_smoke_06_descriptor_seed():
    res = run_p33_fake_input_descriptor_smoke()
    assert res["descriptor_seed"] == 1337


# 7. min_value == -1.0
def test_p33_smoke_07_min_value():
    res = run_p33_fake_input_descriptor_smoke()
    assert res["min_value"] == -1.0


# 8. max_value == 1.0
def test_p33_smoke_08_max_value():
    res = run_p33_fake_input_descriptor_smoke()
    assert res["max_value"] == 1.0


# 9. batch_size == 2
def test_p33_smoke_09_batch_size():
    res = run_p33_fake_input_descriptor_smoke()
    assert res["batch_size"] == 2


# 10. input_flat_dim == 32
def test_p33_smoke_10_input_flat_dim():
    res = run_p33_fake_input_descriptor_smoke()
    assert res["input_flat_dim"] == 32


# 11. rank == 2
def test_p33_smoke_11_rank():
    res = run_p33_fake_input_descriptor_smoke()
    assert res["rank"] == 2


# 12. shape_tuple == [2, 32]
def test_p33_smoke_12_shape_tuple():
    res = run_p33_fake_input_descriptor_smoke()
    assert res["shape_tuple"] == [2, 32]


# 13. shape_matches_p32 True
def test_p33_smoke_13_shape_matches_p32():
    res = run_p33_fake_input_descriptor_smoke()
    assert res["shape_matches_p32"] is True


# 14. descriptor_available_in_p33 True
def test_p33_smoke_14_descriptor_available():
    res = run_p33_fake_input_descriptor_smoke()
    assert res["descriptor_available_in_p33"] is True


# 15. rng_execution_attempted False
def test_p33_smoke_15_rng_attempted():
    res = run_p33_fake_input_descriptor_smoke()
    assert res["rng_execution_attempted"] is False


# 16. value_materialization_attempted False
def test_p33_smoke_16_value_attempted():
    res = run_p33_fake_input_descriptor_smoke()
    assert res["value_materialization_attempted"] is False


# 17. array_materialization_attempted False
def test_p33_smoke_17_array_attempted():
    res = run_p33_fake_input_descriptor_smoke()
    assert res["array_materialization_attempted"] is False


# 18. tensor_materialization_attempted False
def test_p33_smoke_18_tensor_attempted():
    res = run_p33_fake_input_descriptor_smoke()
    assert res["tensor_materialization_attempted"] is False


# 19. forward_execution_attempted False
def test_p33_smoke_19_forward_attempted():
    res = run_p33_fake_input_descriptor_smoke()
    assert res["forward_execution_attempted"] is False


# 20. output_generation_attempted False
def test_p33_smoke_20_output_attempted():
    res = run_p33_fake_input_descriptor_smoke()
    assert res["output_generation_attempted"] is False


# 21. rng_executed False
def test_p33_smoke_21_rng_executed():
    res = run_p33_fake_input_descriptor_smoke()
    assert res["rng_executed"] is False


# 22. values_materialized False
def test_p33_smoke_22_values_materialized():
    res = run_p33_fake_input_descriptor_smoke()
    assert res["values_materialized"] is False


# 23. array_materialized False
def test_p33_smoke_23_array_materialized():
    res = run_p33_fake_input_descriptor_smoke()
    assert res["array_materialized"] is False


# 24. tensor_materialized False
def test_p33_smoke_24_tensor_materialized():
    res = run_p33_fake_input_descriptor_smoke()
    assert res["tensor_materialized"] is False


# 25. all no_* flags True
def test_p33_smoke_25_no_flags():
    res = run_p33_fake_input_descriptor_smoke()
    assert res["no_rng_execution"] is True
    assert res["no_values_materialized"] is True
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


# 26. JSON parseable
def test_p33_smoke_26_json_parseable():
    res = run_p33_fake_input_descriptor_smoke()
    js = compact_json(res)
    d = json.loads(js)
    assert d["verdict"] == "PASS"


# 27. smoke script no torch import
def test_p33_smoke_27_no_torch_import():
    content = pathlib.Path("tools/phase2/run_p33_fake_input_descriptor_smoke.py").read_text(encoding="utf-8")
    assert "import torch" not in content
    assert "from torch" not in content


# 28. smoke script no numpy import
def test_p33_smoke_28_no_numpy_import():
    content = pathlib.Path("tools/phase2/run_p33_fake_input_descriptor_smoke.py").read_text(encoding="utf-8")
    assert "import numpy" not in content
    assert "from numpy" not in content


# 29. smoke script no random/secrets import
def test_p33_smoke_29_no_random_secrets():
    content = pathlib.Path("tools/phase2/run_p33_fake_input_descriptor_smoke.py").read_text(encoding="utf-8")
    assert "import random" not in content
    assert "from random" not in content
    assert "import secrets" not in content
    assert "from secrets" not in content


# 30. smoke script no argparse
def test_p33_smoke_30_no_argparse():
    content = pathlib.Path("tools/phase2/run_p33_fake_input_descriptor_smoke.py").read_text(encoding="utf-8")
    assert "argparse" not in content


# 31. smoke script no subprocess
def test_p33_smoke_31_no_subprocess():
    content = pathlib.Path("tools/phase2/run_p33_fake_input_descriptor_smoke.py").read_text(encoding="utf-8")
    assert "subprocess" not in content


# 32. smoke output no success/scientific claims
def test_p33_smoke_32_no_claims():
    res = run_p33_fake_input_descriptor_smoke()
    js = compact_json(res)
    normalized = js.lower()
    cleaned = (normalized
               .replace("no_final_comparison", "")
               .replace("no_scientific_conclusion", "")
               .replace("descriptor_only_no_values_in_p33", ""))
    for claim in ("model works", "scientific success", "solved", "best", "winner", "production ready", "state of the art"):
        assert claim not in cleaned
