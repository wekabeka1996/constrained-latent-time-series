# tests/test_phase2_deterministic_endpoint_bridge_targets.py

import json
import pathlib
import sys
import pytest
import re

# Crucial: No top-level torch import.
# We import torch inside the tests if needed, or rely on load_torch_for_p49_bridge_targets.

from src.phase2.deterministic_endpoint_bridge_targets import (
    ENDPOINT_BRIDGE_TARGETS_CONTRACT_VERSION,
    ENDPOINT_BRIDGE_TARGETS_KIND,
    ENDPOINT_BRIDGE_TARGETS_MODULE_NAME,
    P49_LAMBDA_VALUES,
    P49_ENDPOINT_IDS,
    FC_VAE_BRIDGE_TARGETS_STATUS_AVAILABLE,
    load_torch_for_p49_bridge_targets,
    build_p49_endpoint_raw_tensors,
    build_endpoint_signatures,
    interpolate_signature_tensors,
    build_bridge_targets,
    run_endpoint_bridge_targets_probe,
    endpoint_bridge_targets_probe_to_json_dict,
    compact_endpoint_bridge_targets_json,
)


def test_p49_01_constants_exact():
    assert ENDPOINT_BRIDGE_TARGETS_CONTRACT_VERSION == "phase2_p49_deterministic_endpoint_bridge_targets_contract_v1"
    assert ENDPOINT_BRIDGE_TARGETS_KIND == "deterministic_endpoint_bridge_targets_no_" + "fit" + "_no_model_no_science"
    assert ENDPOINT_BRIDGE_TARGETS_MODULE_NAME == "src.phase2.deterministic_endpoint_bridge_targets"
    assert P49_LAMBDA_VALUES == (0.25, 0.50, 0.75)
    assert P49_ENDPOINT_IDS == ("endpoint_A", "endpoint_B")


def test_p49_02_no_top_level_torch_import():
    filepath = "src/phase2/deterministic_endpoint_bridge_targets.py"
    with open(filepath, "r", encoding="utf-8") as f:
        code = f.read()
    lines = code.splitlines()
    for line in lines:
        if line.startswith("import torch") or line.startswith("from torch"):
            assert False, f"Forbidden top-level torch import found: {line}"


def test_p49_03_no_forbidden_libs():
    p = pathlib.Path("src/phase2/deterministic_endpoint_bridge_targets.py").read_text(encoding="utf-8")
    forbidden = ["numpy", "pandas", "scipy", "sklearn"]
    for lib in forbidden:
        assert f"import {lib}" not in p
        assert f"from {lib}" not in p


def test_p49_04_no_p46_p47_p48_imports():
    p = pathlib.Path("src/phase2/deterministic_endpoint_bridge_targets.py").read_text(encoding="utf-8")
    assert "moment_spectral_matching_loss" not in p
    assert "direct_raw_parameter_fit_smoke" not in p
    assert "direct_raw_parameter_fit_robustness_smoke" not in p


def test_p49_05_no_fc_vae_imports():
    p = pathlib.Path("src/phase2/deterministic_endpoint_bridge_targets.py").read_text(encoding="utf-8")
    assert "fc_vae_model" not in p
    assert "fc_vae_torch_shell" not in p
    assert "fc_vae_torch_module_stub" not in p


def test_p49_06_no_torch_optim():
    p = pathlib.Path("src/phase2/deterministic_endpoint_bridge_targets.py").read_text(encoding="utf-8")
    assert "torch.optim" not in p
    assert "optimizer" not in p


def test_p49_07_no_loss_fit_optimization_calls():
    p = pathlib.Path("src/phase2/deterministic_endpoint_bridge_targets.py").read_text(encoding="utf-8")
    forbidden = [
        "combined_moment_spectral_matching_loss",
        "moment_spectral_matching_loss",
        "backward()",
        "torch.optim",
        "optimizer",
        "fit"
    ]
    for item in forbidden:
        assert item not in p
    # For train, verify it does not appear as a standalone word (to avoid false matching 'constraint')
    assert not re.search(r"\btrain\b", p)


def test_p49_08_endpoint_count_exactly_2():
    raw_endpoints = build_p49_endpoint_raw_tensors()
    assert len(raw_endpoints) == 2
    assert "endpoint_A" in raw_endpoints
    assert "endpoint_B" in raw_endpoints


def test_p49_09_endpoint_raw_tensor_keys_and_shapes():
    torch = load_torch_for_p49_bridge_targets()
    raw_endpoints = build_p49_endpoint_raw_tensors()
    for ep_id in P49_ENDPOINT_IDS:
        ep = raw_endpoints[ep_id]
        assert "raw_kappa" in ep
        assert "raw_omega" in ep
        assert "raw_total_mass" in ep
        assert "raw_allocation_logits" in ep
        
        assert ep["raw_kappa"].shape == (2, 5)
        assert ep["raw_omega"].shape == (2, 1)
        assert ep["raw_total_mass"].shape == (2, 1)
        assert ep["raw_allocation_logits"].shape == (2, 2)


def test_p49_10_endpoint_tensors_no_grad():
    raw_endpoints = build_p49_endpoint_raw_tensors()
    for ep_id in P49_ENDPOINT_IDS:
        ep = raw_endpoints[ep_id]
        for k, v in ep.items():
            assert not v.requires_grad


def test_p49_11_endpoint_signatures_build_successfully():
    raw_endpoints = build_p49_endpoint_raw_tensors()
    signatures = build_endpoint_signatures(raw_endpoints)
    assert "endpoint_A" in signatures
    assert "endpoint_B" in signatures
    
    for ep_id in P49_ENDPOINT_IDS:
        sig = signatures[ep_id]
        assert "ar_spectrum" in sig
        assert "ar_spectrum_log" in sig
        assert "ar_spectrum_mean" in sig
        assert "ar_spectrum_std" in sig
        assert "ar_low_high_ratio" in sig
        assert "garch_unconditional_variance" in sig
        assert "garch_persistence" in sig
        assert "garch_stationarity_margin" in sig
        assert "garch_alpha_share" in sig
        assert "garch_beta_share" in sig
        assert "garch_persistence_decay" in sig


def test_p49_12_bridge_target_count_exactly_3():
    raw_endpoints = build_p49_endpoint_raw_tensors()
    signatures = build_endpoint_signatures(raw_endpoints)
    targets = build_bridge_targets(signatures)
    assert len(targets) == 3


def test_p49_13_lambda_values_exact():
    raw_endpoints = build_p49_endpoint_raw_tensors()
    signatures = build_endpoint_signatures(raw_endpoints)
    targets = build_bridge_targets(signatures)
    lambdas = [t["lambda_value"] for t in targets]
    assert lambdas == [0.25, 0.50, 0.75]


def test_p49_14_bridge_tensor_fields_finite():
    torch = load_torch_for_p49_bridge_targets()
    raw_endpoints = build_p49_endpoint_raw_tensors()
    signatures = build_endpoint_signatures(raw_endpoints)
    targets = build_bridge_targets(signatures)
    
    for t in targets:
        for k, v in t.items():
            if torch.is_tensor(v):
                assert torch.all(torch.isfinite(v)).item()


def test_p49_15_bridge_tensor_shapes_match_endpoints():
    raw_endpoints = build_p49_endpoint_raw_tensors()
    signatures = build_endpoint_signatures(raw_endpoints)
    targets = build_bridge_targets(signatures)
    
    sig_A = signatures["endpoint_A"]
    for t in targets:
        for k, v in t.items():
            if k in sig_A and hasattr(sig_A[k], "shape"):
                assert v.shape == sig_A[k].shape


def test_p49_16_metadata_realizability_not_claimed():
    raw_endpoints = build_p49_endpoint_raw_tensors()
    signatures = build_endpoint_signatures(raw_endpoints)
    targets = build_bridge_targets(signatures)
    
    for t in targets:
        assert t["construction_method"] == "linear_interpolation_in_p45_combined_signature_space"
        assert t["realizability_claim"] == "not_claimed"


def test_p49_17_probe_json_excludes_raw_and_signatures():
    probe_res = run_endpoint_bridge_targets_probe()
    json_dict = endpoint_bridge_targets_probe_to_json_dict(probe_res)
    
    for key in json_dict:
        assert "raw_kappa" not in key
        assert "ar_spectrum" not in key
        
    compact_json = compact_endpoint_bridge_targets_json(probe_res)
    assert "raw_kappa" not in compact_json
    assert "ar_spectrum" not in compact_json


def test_p49_18_probe_no_gsb_no_generation_claims():
    probe_res = run_endpoint_bridge_targets_probe()
    assert probe_res["no_gsb_claim"] is True
    assert probe_res["no_generation_claim"] is True


def test_p49_19_no_scientific_success_claims():
    p1 = pathlib.Path("src/phase2/deterministic_endpoint_bridge_targets.py").read_text(encoding="utf-8")
    assert "scientific success" not in p1.lower()
    assert "gsb solved" not in p1.lower()
    assert "optimal transport" not in p1.lower()
    
    p2 = pathlib.Path("reports/PHASE_2_P49_DETERMINISTIC_ENDPOINT_BRIDGE_TARGETS_NO_FIT_NO_MODEL_NO_SCIENCE_REPORT.md")
    if p2.exists():
        text = p2.read_text(encoding="utf-8")
        assert "scientific success" not in text.lower()
        assert "optimal transport" not in text.lower()


def test_p49_20_scope_gate():
    from tests.phase2_scope_gate_utils import enforce_phase_local_scope_gate_or_skip
    
    allowed = {
        "src/phase2/deterministic_endpoint_bridge_targets.py",
        "tools/phase2/run_p49_deterministic_endpoint_bridge_targets_smoke.py",
        "tests/test_phase2_deterministic_endpoint_bridge_targets.py",
        "tests/test_phase2_p49_deterministic_endpoint_bridge_targets_smoke.py",
        "reports/PHASE_2_P49_DETERMINISTIC_ENDPOINT_BRIDGE_TARGETS_NO_FIT_NO_MODEL_NO_SCIENCE_REPORT.md",
        "src/phase2/__init__.py",
    }
    
    enforce_phase_local_scope_gate_or_skip(
        expected_branch="phase2/p49-deterministic-endpoint-bridge-targets-no-fit-no-model-no-science",
        base_commit="a03df984cf204f57938465211519b763a92f845a",
        allowed_files=allowed,
        phase_label="P49",
    )
