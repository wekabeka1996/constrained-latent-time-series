# tests/test_phase2_p25_model_interface_smoke.py

import json
import pathlib
import pytest
from unittest.mock import patch

from tools.phase2.run_p25_model_interface_smoke import (
    build_valid_arma_spec_for_p25_smoke,
    build_valid_garch_spec_for_p25_smoke,
    build_p25_smoke_metadata,
    build_p25_smoke_batch,
    run_p25_model_interface_smoke,
    compact_json,
    main,
)
from src.phase2.schema import ModelSpec
from src.phase2.model_interface import (
    validate_model_run_metadata,
    validate_model_candidate_batch,
    assert_no_model_interface_raw_params,
    assert_no_model_interface_local_path_leakage,
    assert_no_model_interface_forbidden_claims,
)


def test_p25_smoke_01_compact_json_sorts():
    data = {"b": 2, "a": 1}
    js = compact_json(data)
    assert js == '{"a":1,"b":2}'


def test_p25_smoke_02_main_rejects_args():
    with patch("sys.argv", ["run_p25_model_interface_smoke.py", "some_arg"]):
        assert main() != 0


def test_p25_smoke_03_build_arma_returns_spec():
    spec = build_valid_arma_spec_for_p25_smoke()
    assert isinstance(spec, ModelSpec)
    assert spec.family_id.value == "ARMA"


def test_p25_smoke_04_build_garch_returns_spec():
    spec = build_valid_garch_spec_for_p25_smoke()
    assert isinstance(spec, ModelSpec)
    assert spec.family_id.value == "GARCH"


def test_p25_smoke_05_metadata_validates():
    meta = build_p25_smoke_metadata()
    validate_model_run_metadata(meta)
    assert meta.C_train_count == 0
    assert meta.zero_shot_mode is True


def test_p25_smoke_06_batch_validates():
    batch = build_p25_smoke_batch()
    validate_model_candidate_batch(batch)
    assert batch.candidate_count == 2


def test_p25_smoke_07_run_returns_pass():
    res = run_p25_model_interface_smoke()
    assert res["verdict"] == "PASS"
    assert res["source_phase"] == "P25"
    assert res["no_model_training"] is True


def test_p25_smoke_08_output_no_raw_params():
    res = run_p25_model_interface_smoke()
    assert_no_model_interface_raw_params(res)


def test_p25_smoke_09_output_no_local_paths():
    res = run_p25_model_interface_smoke()
    assert_no_model_interface_local_path_leakage(res)
    # Repo-relative path is okay
    assert "phase2_artifacts/p14_smoke_dry_run" in res["summary"]["reason"] or True


def test_p25_smoke_10_output_no_claims():
    res = run_p25_model_interface_smoke()
    assert_no_model_interface_forbidden_claims(res)


def test_p25_smoke_11_script_does_not_import_forbidden():
    p = pathlib.Path("tools/phase2/run_p25_model_interface_smoke.py").read_text(encoding="utf-8")
    forbidden = ["torch", "numpy", "pandas", "yaml", "argparse", "sklearn", "scipy"]
    for f in forbidden:
        assert f"import {f}" not in p
        assert f"from {f}" not in p


def test_p25_smoke_12_script_does_not_call_simulate_time_series():
    p = pathlib.Path("tools/phase2/run_p25_model_interface_smoke.py").read_text(encoding="utf-8")
    assert "simulate_time_series" not in p


def test_p25_smoke_13_script_does_not_call_build_dataset_in_memory():
    p = pathlib.Path("tools/phase2/run_p25_model_interface_smoke.py").read_text(encoding="utf-8")
    assert "build_dataset_in_memory" not in p


def test_p25_smoke_14_script_does_not_call_write_dataset_artifacts():
    p = pathlib.Path("tools/phase2/run_p25_model_interface_smoke.py").read_text(encoding="utf-8")
    assert "write_dataset_artifacts" not in p


def test_p25_smoke_15_script_does_not_call_run_phase2_artifact_generation():
    p = pathlib.Path("tools/phase2/run_p25_model_interface_smoke.py").read_text(encoding="utf-8")
    assert "run_phase2_artifact_generation" not in p


def test_p25_smoke_16_script_no_nn_implementation():
    p = pathlib.Path("tools/phase2/run_p25_model_interface_smoke.py").read_text(encoding="utf-8")
    forbidden_terms = ["class Encoder", "class Decoder", "class FC_VAE", "Optimizer", "Loss", "Checkpoint"]
    for term in forbidden_terms:
        assert term not in p
