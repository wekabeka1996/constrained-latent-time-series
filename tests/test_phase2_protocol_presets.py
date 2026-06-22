# tests/test_phase2_protocol_presets.py

import pytest
import pathlib
import sys
from dataclasses import replace

from src.phase2.schema import FamilyId, ModelSpec, MeanFamily, VolatilityFamily
from src.phase2.sampler import GenerationRequest
from src.phase2.simulator import SimulationRequest
from src.phase2.dataset import SplitName
from src.phase2.split_runner import (
    SplitArtifactRequest,
    Phase2ArtifactRunRequest,
    validate_split_artifact_request,
    validate_phase2_artifact_run_request,
)
from src.phase2.protocol_presets import (
    ProtocolPresetName,
    ProtocolPresetFactoryRequest,
    ProtocolPresetFactoryResult,
    APPROVED_PHASE2_PRESET_SAMPLE_COUNTS,
    APPROVED_PHASE2_BASE_SEEDS_BY_FAMILY,
    APPROVED_PHASE2_ARTIFACT_SUBDIR_BY_SPLIT,
    APPROVED_PHASE2_SAMPLES_FILENAME,
    APPROVED_PHASE2_MANIFEST_FILENAME,
    APPROVED_PHASE2_FEWSHOT_COUNTS_BY_PRESET,
    APPROVED_PHASE2_C_SEED_ROOT_BY_C_SPLIT,
    APPROVED_PHASE2_C_SEED_RANGE_STRIDE,
    APPROVED_PHASE2_C_SEED_BASE_BY_C_SPLIT,
    validate_protocol_preset_factory_request,
    get_preset_sample_count_by_family,
    build_split_request_for_preset,
    build_phase2_preset_run_request,
    build_phase2_preset_factory_result,
    get_fewshot_counts_for_preset,
    seed_range_for_base_and_count,
    ranges_overlap,
    range_is_prefix_subset,
    validate_p13_fewshot_seed_plan_for_preset,
)


# ==============================================================================
# Helper builders for test requests
# ==============================================================================

def get_test_gen_template(family_id: FamilyId) -> GenerationRequest:
    if family_id == FamilyId.AR:
        return GenerationRequest(
            family_id=FamilyId.AR,
            seed=0,
            p=1, q=0, r=0, s=0,
            ar_range=(-0.4, 0.4),
            ma_range=(-0.4, 0.4),
            omega_range=(0.1, 1.0),
            alpha_range=(0.0, 0.4),
            beta_range=(0.0, 0.4),
            constraint_flags=(1.0, 0.0, 0.0, 0.0),
            provenance=(("phase", "p13"),),
            max_attempts=100,
        )
    elif family_id == FamilyId.ARMA:
        return GenerationRequest(
            family_id=FamilyId.ARMA,
            seed=0,
            p=1, q=1, r=0, s=0,
            ar_range=(-0.4, 0.4),
            ma_range=(-0.4, 0.4),
            omega_range=(0.1, 1.0),
            alpha_range=(0.0, 0.4),
            beta_range=(0.0, 0.4),
            constraint_flags=(1.0, 0.0, 0.0, 0.0),
            provenance=(("phase", "p13"),),
            max_attempts=100,
        )
    elif family_id == FamilyId.GARCH:
        return GenerationRequest(
            family_id=FamilyId.GARCH,
            seed=0,
            p=0, q=0, r=1, s=1,
            ar_range=(-0.4, 0.4),
            ma_range=(-0.4, 0.4),
            omega_range=(0.1, 0.5),
            alpha_range=(0.1, 0.3),
            beta_range=(0.1, 0.5),
            constraint_flags=(1.0, 0.0, 0.0, 0.0),
            provenance=(("phase", "p13"),),
            max_attempts=100,
        )
    elif family_id == FamilyId.ARMA_GARCH:
        return GenerationRequest(
            family_id=FamilyId.ARMA_GARCH,
            seed=0,
            p=1, q=1, r=1, s=1,
            ar_range=(-0.3, 0.3),
            ma_range=(-0.3, 0.3),
            omega_range=(0.1, 0.5),
            alpha_range=(0.1, 0.3),
            beta_range=(0.1, 0.5),
            constraint_flags=(1.0, 0.0, 0.0, 0.0),
            provenance=(("phase", "p13"),),
            max_attempts=100,
        )
    else:
        raise ValueError(f"Unknown family: {family_id}")


def get_test_sim_template() -> SimulationRequest:
    spec = ModelSpec(
        family_id=FamilyId.AR,
        mean_family=MeanFamily.AR,
        volatility_family=VolatilityFamily.NONE,
        p=1, q=0, r=0, s=0,
        ar_params=(0.1,), ma_params=(), omega=None, alpha_params=(), beta_params=(),
        constraint_flags=(1.0, 0.0, 0.0, 0.0), provenance=(),
    )
    return SimulationRequest(
        spec=spec,
        seed=0,
        length=20,
        burn_in=5,
        innovation_distribution="gaussian",
        innovation_std=1.0,
        initial_value=0.0,
        initial_innovation=0.0,
        initial_variance=1.0,
        max_abs_value=1000.0,
    )


def make_valid_factory_request(
    protocol_name="test_protocol",
    preset_name=ProtocolPresetName.SMOKE,
    output_root_dir="results/test",
    create_parent_dirs=True,
    overwrite_existing=True,
    include_values=True,
    include_innovations=True,
    include_variances=True,
    json_sort_keys=True,
    json_indent=2,
    sample_id_hash_len=8,
) -> ProtocolPresetFactoryRequest:
    generation_template_by_family = (
        (FamilyId.AR, get_test_gen_template(FamilyId.AR)),
        (FamilyId.ARMA, get_test_gen_template(FamilyId.ARMA)),
        (FamilyId.GARCH, get_test_gen_template(FamilyId.GARCH)),
        (FamilyId.ARMA_GARCH, get_test_gen_template(FamilyId.ARMA_GARCH)),
    )
    return ProtocolPresetFactoryRequest(
        protocol_name=protocol_name,
        preset_name=preset_name,
        output_root_dir=output_root_dir,
        generation_template_by_family=generation_template_by_family,
        simulation_template=get_test_sim_template(),
        create_parent_dirs=create_parent_dirs,
        overwrite_existing=overwrite_existing,
        include_values=include_values,
        include_innovations=include_innovations,
        include_variances=include_variances,
        json_sort_keys=json_sort_keys,
        json_indent=json_indent,
        sample_id_hash_len=sample_id_hash_len,
    )


# ==============================================================================
# Group A: Fewshot constants (Tests 1-3)
# ==============================================================================

def test_a_fewshot_constants_1():
    # 1. APPROVED_PHASE2_FEWSHOT_COUNTS_BY_PRESET has SMOKE=(10,50), DEV=(100,500), MAIN=(1000,5000).
    assert type(APPROVED_PHASE2_FEWSHOT_COUNTS_BY_PRESET) is tuple
    counts_dict = {item[0]: (item[1], item[2]) for item in APPROVED_PHASE2_FEWSHOT_COUNTS_BY_PRESET}
    assert counts_dict[ProtocolPresetName.SMOKE] == (10, 50)
    assert counts_dict[ProtocolPresetName.DEV] == (100, 500)
    assert counts_dict[ProtocolPresetName.MAIN] == (1000, 5000)


def test_a_fewshot_constants_2():
    # 2. APPROVED_PHASE2_C_SEED_BASE_BY_C_SPLIT has zero_shot_eval=12003, fewshot_train=10_012_101, fewshot_eval=20_012_105.
    assert type(APPROVED_PHASE2_C_SEED_BASE_BY_C_SPLIT) is tuple
    seeds = dict(APPROVED_PHASE2_C_SEED_BASE_BY_C_SPLIT)
    assert seeds["zero_shot_eval"] == 12003
    assert seeds["fewshot_train"] == 10012101
    assert seeds["fewshot_eval"] == 20012105


def test_a_fewshot_constants_3():
    # 3. Artifact subdirs include fewshot_1pct_train and fewshot_5pct_train.
    subdirs = [subdir for _, subdir in APPROVED_PHASE2_ARTIFACT_SUBDIR_BY_SPLIT]
    assert "fewshot_1pct_train" in subdirs
    assert "fewshot_5pct_train" in subdirs
    assert "fewshot_eval" in subdirs


# ==============================================================================
# Group B: Seed range helpers (Tests 4-12)
# ==============================================================================

def test_b_seed_range_helpers_4():
    # 4. seed_range_for_base_and_count(12101, 10) returns (12101, 12110).
    assert seed_range_for_base_and_count(12101, 10) == (12101, 12110)


def test_b_seed_range_helpers_5():
    # 5. seed_range_for_base_and_count rejects count <= 0.
    with pytest.raises(ValueError, match="count must be a positive integer"):
        seed_range_for_base_and_count(12101, 0)
    with pytest.raises(ValueError, match="count must be a positive integer"):
        seed_range_for_base_and_count(12101, -5)


def test_b_seed_range_helpers_6_7():
    # 6. ranges_overlap detects overlap.
    assert ranges_overlap((10, 20), (15, 25)) is True
    assert ranges_overlap((10, 20), (20, 30)) is True
    assert ranges_overlap((10, 20), (5, 12)) is True
    # 7. ranges_overlap detects non-overlap.
    assert ranges_overlap((10, 20), (21, 30)) is False
    assert ranges_overlap((10, 20), (1, 9)) is False


def test_b_seed_range_helpers_8_9():
    # 8. range_is_prefix_subset accepts 1% inside 5%.
    assert range_is_prefix_subset((12101, 12110), (12101, 12150)) is True
    assert range_is_prefix_subset((100, 200), (100, 200)) is True
    # 9. range_is_prefix_subset rejects non-prefix overlap.
    assert range_is_prefix_subset((101, 200), (100, 200)) is False
    assert range_is_prefix_subset((100, 250), (100, 200)) is False


def test_b_seed_range_helpers_10_11_12():
    # 10. validate_p13_fewshot_seed_plan_for_preset passes for SMOKE.
    # 11. validate_p13_fewshot_seed_plan_for_preset passes for DEV.
    # 12. validate_p13_fewshot_seed_plan_for_preset passes for MAIN.
    validate_p13_fewshot_seed_plan_for_preset(ProtocolPresetName.SMOKE)
    validate_p13_fewshot_seed_plan_for_preset(ProtocolPresetName.DEV)
    validate_p13_fewshot_seed_plan_for_preset(ProtocolPresetName.MAIN)


# ==============================================================================
# Group C: Split request generation (Tests 13-27)
# ==============================================================================

def test_c_split_request_generation_13_14():
    # 13. build_split_request_for_preset requires valid split_name/artifact_subdir pair.
    # 14. invalid split_name/artifact_subdir pair is rejected.
    req = make_valid_factory_request()
    
    # Valid
    split_req = build_split_request_for_preset(
        ProtocolPresetName.SMOKE,
        SplitName.SMOKE,
        "smoke",
        req.generation_template_by_family,
        req.simulation_template,
        8
    )
    assert isinstance(split_req, SplitArtifactRequest)

    # Invalid subdir
    with pytest.raises(ValueError, match="Invalid split_name and artifact_subdir pair"):
        build_split_request_for_preset(
            ProtocolPresetName.SMOKE,
            SplitName.SMOKE,
            "zero_shot_train",
            req.generation_template_by_family,
            req.simulation_template,
            8
        )


def test_c_split_request_generation_15():
    # 15. smoke split includes all A/B/C families.
    req = make_valid_factory_request(preset_name=ProtocolPresetName.SMOKE)
    split_req = build_split_request_for_preset(
        req.preset_name,
        SplitName.SMOKE,
        "smoke",
        req.generation_template_by_family,
        req.simulation_template,
        req.sample_id_hash_len
    )
    counts = dict(split_req.sample_count_by_family)
    assert set(counts.keys()) == {FamilyId.AR, FamilyId.ARMA, FamilyId.GARCH, FamilyId.ARMA_GARCH}
    assert counts[FamilyId.AR] == 1000
    assert counts[FamilyId.ARMA] == 1000
    assert counts[FamilyId.GARCH] == 1000
    assert counts[FamilyId.ARMA_GARCH] == 1000


def test_c_split_request_generation_16_17():
    # 16. zero_shot_train excludes ARMA_GARCH.
    # 17. zero_shot_train has enforce_zero_shot_c_train_exclusion=True.
    req = make_valid_factory_request(preset_name=ProtocolPresetName.SMOKE)
    split_req = build_split_request_for_preset(
        req.preset_name,
        SplitName.ZERO_SHOT_TRAIN,
        "zero_shot_train",
        req.generation_template_by_family,
        req.simulation_template,
        req.sample_id_hash_len
    )
    counts = dict(split_req.sample_count_by_family)
    assert FamilyId.ARMA_GARCH not in counts
    assert split_req.enforce_zero_shot_c_train_exclusion is True


def test_c_split_request_generation_18():
    # 18. zero_shot_eval is C-only with base seed 12003.
    req = make_valid_factory_request(preset_name=ProtocolPresetName.SMOKE)
    split_req = build_split_request_for_preset(
        req.preset_name,
        SplitName.ZERO_SHOT_EVAL,
        "zero_shot_eval",
        req.generation_template_by_family,
        req.simulation_template,
        req.sample_id_hash_len
    )
    counts = dict(split_req.sample_count_by_family)
    assert list(counts.keys()) == [FamilyId.ARMA_GARCH]
    assert counts[FamilyId.ARMA_GARCH] == 1000
    
    seeds = dict(split_req.base_seed_by_family)
    assert seeds[FamilyId.ARMA_GARCH] == 12003


def test_c_split_request_generation_19_20():
    # 19. fewshot_1pct_train is C-only with count 10/100/1000 depending preset.
    # 20. fewshot_1pct_train uses ARMA_GARCH base seed 10_012_101.
    req = make_valid_factory_request(preset_name=ProtocolPresetName.SMOKE)
    split_req = build_split_request_for_preset(
        req.preset_name,
        SplitName.FEWSHOT_TRAIN,
        "fewshot_1pct_train",
        req.generation_template_by_family,
        req.simulation_template,
        req.sample_id_hash_len
    )
    counts = dict(split_req.sample_count_by_family)
    assert list(counts.keys()) == [FamilyId.ARMA_GARCH]
    assert counts[FamilyId.ARMA_GARCH] == 10
    
    seeds = dict(split_req.base_seed_by_family)
    assert seeds[FamilyId.ARMA_GARCH] == 10012101


def test_c_split_request_generation_21_22():
    # 21. fewshot_5pct_train is C-only with count 50/500/5000 depending preset.
    # 22. fewshot_5pct_train uses ARMA_GARCH base seed 10_012_101.
    req = make_valid_factory_request(preset_name=ProtocolPresetName.SMOKE)
    split_req = build_split_request_for_preset(
        req.preset_name,
        SplitName.FEWSHOT_TRAIN,
        "fewshot_5pct_train",
        req.generation_template_by_family,
        req.simulation_template,
        req.sample_id_hash_len
    )
    counts = dict(split_req.sample_count_by_family)
    assert list(counts.keys()) == [FamilyId.ARMA_GARCH]
    assert counts[FamilyId.ARMA_GARCH] == 50
    
    seeds = dict(split_req.base_seed_by_family)
    assert seeds[FamilyId.ARMA_GARCH] == 10012101


def test_c_split_request_generation_23():
    # 23. fewshot_eval is C-only with base seed 20_012_105.
    req = make_valid_factory_request(preset_name=ProtocolPresetName.SMOKE)
    split_req = build_split_request_for_preset(
        req.preset_name,
        SplitName.FEWSHOT_EVAL,
        "fewshot_eval",
        req.generation_template_by_family,
        req.simulation_template,
        req.sample_id_hash_len
    )
    counts = dict(split_req.sample_count_by_family)
    assert list(counts.keys()) == [FamilyId.ARMA_GARCH]
    assert counts[FamilyId.ARMA_GARCH] == 1000
    
    seeds = dict(split_req.base_seed_by_family)
    assert seeds[FamilyId.ARMA_GARCH] == 20012105


def test_c_split_request_generation_24_25_26_27():
    # 24. fewshot_5pct_train count is greater than fewshot_1pct_train count.
    # 25. fewshot_1pct seed range is prefix subset of fewshot_5pct seed range.
    # 26. zero_shot_eval seed range does not overlap fewshot_5pct_train.
    # 27. fewshot_eval seed range does not overlap fewshot_5pct_train.
    req = make_valid_factory_request(preset_name=ProtocolPresetName.SMOKE)
    
    s_1pct = build_split_request_for_preset(req.preset_name, SplitName.FEWSHOT_TRAIN, "fewshot_1pct_train", req.generation_template_by_family, req.simulation_template, 8)
    s_5pct = build_split_request_for_preset(req.preset_name, SplitName.FEWSHOT_TRAIN, "fewshot_5pct_train", req.generation_template_by_family, req.simulation_template, 8)
    s_z_eval = build_split_request_for_preset(req.preset_name, SplitName.ZERO_SHOT_EVAL, "zero_shot_eval", req.generation_template_by_family, req.simulation_template, 8)
    s_f_eval = build_split_request_for_preset(req.preset_name, SplitName.FEWSHOT_EVAL, "fewshot_eval", req.generation_template_by_family, req.simulation_template, 8)

    c_1 = dict(s_1pct.sample_count_by_family)[FamilyId.ARMA_GARCH]
    c_5 = dict(s_5pct.sample_count_by_family)[FamilyId.ARMA_GARCH]
    c_z = dict(s_z_eval.sample_count_by_family)[FamilyId.ARMA_GARCH]
    c_f = dict(s_f_eval.sample_count_by_family)[FamilyId.ARMA_GARCH]

    assert c_5 > c_1

    r_1 = seed_range_for_base_and_count(dict(s_1pct.base_seed_by_family)[FamilyId.ARMA_GARCH], c_1)
    r_5 = seed_range_for_base_and_count(dict(s_5pct.base_seed_by_family)[FamilyId.ARMA_GARCH], c_5)
    r_z = seed_range_for_base_and_count(dict(s_z_eval.base_seed_by_family)[FamilyId.ARMA_GARCH], c_z)
    r_f = seed_range_for_base_and_count(dict(s_f_eval.base_seed_by_family)[FamilyId.ARMA_GARCH], c_f)

    assert range_is_prefix_subset(r_1, r_5) is True
    assert ranges_overlap(r_5, r_z) is False
    assert ranges_overlap(r_5, r_f) is False
    assert ranges_overlap(r_z, r_f) is False


# ==============================================================================
# Group D: Full run request (Tests 28-36)
# ==============================================================================

def test_d_run_request_28_29_30():
    # 28. build_phase2_preset_run_request returns 6 split requests.
    # 29. split order is smoke, zero_shot_train, zero_shot_eval, fewshot_1pct_train, fewshot_5pct_train, fewshot_eval.
    # 30. duplicate SplitName.FEWSHOT_TRAIN appears twice but different artifact_subdir and passes validate_phase2_artifact_run_request.
    req = make_valid_factory_request()
    run_req = build_phase2_preset_run_request(req)
    assert isinstance(run_req, Phase2ArtifactRunRequest)
    assert len(run_req.split_requests) == 6

    expected = [
        (SplitName.SMOKE, "smoke"),
        (SplitName.ZERO_SHOT_TRAIN, "zero_shot_train"),
        (SplitName.ZERO_SHOT_EVAL, "zero_shot_eval"),
        (SplitName.FEWSHOT_TRAIN, "fewshot_1pct_train"),
        (SplitName.FEWSHOT_TRAIN, "fewshot_5pct_train"),
        (SplitName.FEWSHOT_EVAL, "fewshot_eval"),
    ]
    
    actual = [(s.split_name, s.artifact_subdir) for s in run_req.split_requests]
    assert actual == expected

    # Passes runner's validation
    validate_phase2_artifact_run_request(run_req)  # should not raise


def test_d_run_request_31_32():
    # 31. count_by_split in factory result preserves the 6-split order.
    # 32. factory result split_count is 6.
    req = make_valid_factory_request()
    res = build_phase2_preset_factory_result(req)
    assert isinstance(res, ProtocolPresetFactoryResult)
    assert res.split_count == 6

    expected_splits = [
        SplitName.SMOKE,
        SplitName.ZERO_SHOT_TRAIN,
        SplitName.ZERO_SHOT_EVAL,
        SplitName.FEWSHOT_TRAIN,
        SplitName.FEWSHOT_TRAIN,
        SplitName.FEWSHOT_EVAL,
    ]
    assert [split for split, _ in res.count_by_split] == expected_splits


def test_d_run_request_33_34_35():
    # 33. factory result total_requested_samples is correct for SMOKE.
    # smoke: SMOKE (4000) + ZERO_SHOT_TRAIN (3000) + ZERO_SHOT_EVAL (1000) + FEWSHOT_1PCT (10) + FEWSHOT_5PCT (50) + FEWSHOT_EVAL (1000) = 9060
    req_smoke = make_valid_factory_request(preset_name=ProtocolPresetName.SMOKE)
    res_smoke = build_phase2_preset_factory_result(req_smoke)
    assert res_smoke.total_requested_samples == 9060

    # 34. factory result total_requested_samples is correct for DEV.
    # dev: SMOKE (40000) + ZERO_SHOT_TRAIN (30000) + ZERO_SHOT_EVAL (10000) + FEWSHOT_1PCT (100) + FEWSHOT_5PCT (500) + FEWSHOT_EVAL (10000) = 90600
    req_dev = make_valid_factory_request(preset_name=ProtocolPresetName.DEV)
    res_dev = build_phase2_preset_factory_result(req_dev)
    assert res_dev.total_requested_samples == 90600

    # 35. factory result total_requested_samples is correct for MAIN.
    # main: SMOKE (400000) + ZERO_SHOT_TRAIN (300000) + ZERO_SHOT_EVAL (100000) + FEWSHOT_1PCT (1000) + FEWSHOT_5PCT (5000) + FEWSHOT_EVAL (100000) = 906000
    req_main = make_valid_factory_request(preset_name=ProtocolPresetName.MAIN)
    res_main = build_phase2_preset_factory_result(req_main)
    assert res_main.total_requested_samples == 906000


def test_d_run_request_36():
    # 36. no files or directories are created by factory.
    # Verify that nonexistent output directory was not created on disk
    req = make_valid_factory_request(output_root_dir="never_ever_created_by_preset_factory_p13")
    build_phase2_preset_run_request(req)
    assert not pathlib.Path("never_ever_created_by_preset_factory_p13").exists()


# ==============================================================================
# Group E: No-execution / scope (Tests 37-43)
# ==============================================================================

def test_e_no_execution_scope_37_39():
    # 37. protocol_presets.py does not call run_phase2_artifact_generation.
    # 38. protocol_presets.py does not call write_dataset_artifacts.
    # 39. protocol_presets.py does not call build_dataset_in_memory.
    import src.phase2.protocol_presets as presets
    assert "run_phase2_artifact_generation" not in dir(presets)
    assert "write_dataset_artifacts" not in dir(presets)
    assert "build_dataset_in_memory" not in dir(presets)


def test_e_no_execution_scope_40_43():
    # 40. protocol_presets.py does not import torch.
    # 41. protocol_presets.py does not import numpy.
    # 42. protocol_presets.py does not import pandas/yaml/argparse.
    # 43. protocol_presets.py does not import model/training modules.
    import src.phase2.protocol_presets as presets
    assert "torch" not in sys.modules or "torch" not in dir(presets)
    assert "numpy" not in sys.modules or "numpy" not in dir(presets)
    assert "pandas" not in dir(presets)
    assert "yaml" not in dir(presets)
    assert "argparse" not in dir(presets)
    assert "models" not in dir(presets)


# ==============================================================================
# Preserved P12 Request Validation Tests
# ==============================================================================

def test_preserved_p12_validation_reject_raw_non_request():
    with pytest.raises(ValueError, match="must be a ProtocolPresetFactoryRequest instance"):
        validate_protocol_preset_factory_request("not_a_request")


def test_preserved_p12_validation_reject_empty_protocol_name():
    req = make_valid_factory_request(protocol_name="")
    with pytest.raises(ValueError, match="protocol_name must be a non-empty string"):
        validate_protocol_preset_factory_request(req)


def test_preserved_p12_validation_reject_raw_string_preset_name():
    req = replace(make_valid_factory_request(), preset_name="smoke")
    with pytest.raises(ValueError, match="preset_name must be a ProtocolPresetName enum member"):
        validate_protocol_preset_factory_request(req)


def test_preserved_p12_validation_reject_empty_output_root_dir():
    req = make_valid_factory_request(output_root_dir="   ")
    with pytest.raises(ValueError, match="output_root_dir must be a non-empty string"):
        validate_protocol_preset_factory_request(req)


def test_preserved_p12_validation_reject_list_templates():
    req = replace(make_valid_factory_request(), generation_template_by_family=[])
    with pytest.raises(ValueError, match="generation_template_by_family must be exactly a tuple"):
        validate_protocol_preset_factory_request(req)


def test_preserved_p12_validation_reject_duplicate_family_template():
    dup_templates = (
        (FamilyId.AR, get_test_gen_template(FamilyId.AR)),
        (FamilyId.AR, get_test_gen_template(FamilyId.AR)),
        (FamilyId.ARMA, get_test_gen_template(FamilyId.ARMA)),
        (FamilyId.GARCH, get_test_gen_template(FamilyId.GARCH)),
        (FamilyId.ARMA_GARCH, get_test_gen_template(FamilyId.ARMA_GARCH)),
    )
    req = replace(make_valid_factory_request(), generation_template_by_family=dup_templates)
    with pytest.raises(ValueError, match="Duplicate family template entry"):
        validate_protocol_preset_factory_request(req)


def test_preserved_p12_validation_reject_raw_string_family_key():
    bad_keys = (
        ("AR", get_test_gen_template(FamilyId.AR)),
        (FamilyId.ARMA, get_test_gen_template(FamilyId.ARMA)),
        (FamilyId.GARCH, get_test_gen_template(FamilyId.GARCH)),
        (FamilyId.ARMA_GARCH, get_test_gen_template(FamilyId.ARMA_GARCH)),
    )
    req = replace(make_valid_factory_request(), generation_template_by_family=bad_keys)
    with pytest.raises(ValueError, match="family key in generation_template_by_family must be a FamilyId instance"):
        validate_protocol_preset_factory_request(req)


def test_preserved_p12_validation_reject_non_generation_request():
    bad_templates = (
        (FamilyId.AR, "not_a_template"),
        (FamilyId.ARMA, get_test_gen_template(FamilyId.ARMA)),
        (FamilyId.GARCH, get_test_gen_template(FamilyId.GARCH)),
        (FamilyId.ARMA_GARCH, get_test_gen_template(FamilyId.ARMA_GARCH)),
    )
    req = replace(make_valid_factory_request(), generation_template_by_family=bad_templates)
    with pytest.raises(ValueError, match="template in generation_template_by_family must be a GenerationRequest instance"):
        validate_protocol_preset_factory_request(req)


def test_preserved_p12_validation_reject_template_family_mismatch():
    bad_mismatch = (
        (FamilyId.AR, get_test_gen_template(FamilyId.ARMA)),
        (FamilyId.ARMA, get_test_gen_template(FamilyId.ARMA)),
        (FamilyId.GARCH, get_test_gen_template(FamilyId.GARCH)),
        (FamilyId.ARMA_GARCH, get_test_gen_template(FamilyId.ARMA_GARCH)),
    )
    req = replace(make_valid_factory_request(), generation_template_by_family=bad_mismatch)
    with pytest.raises(ValueError, match="does not match key"):
        validate_protocol_preset_factory_request(req)


def test_preserved_p12_validation_reject_invalid_generation_template():
    bad_gen_req = replace(get_test_gen_template(FamilyId.AR), max_attempts=0)
    templates = (
        (FamilyId.AR, bad_gen_req),
        (FamilyId.ARMA, get_test_gen_template(FamilyId.ARMA)),
        (FamilyId.GARCH, get_test_gen_template(FamilyId.GARCH)),
        (FamilyId.ARMA_GARCH, get_test_gen_template(FamilyId.ARMA_GARCH)),
    )
    req = replace(make_valid_factory_request(), generation_template_by_family=templates)
    with pytest.raises(ValueError, match="max_attempts must be a positive int"):
        validate_protocol_preset_factory_request(req)


def test_preserved_p12_validation_reject_non_simulation_request():
    req = replace(make_valid_factory_request(), simulation_template="not_a_simulation_template")
    with pytest.raises(ValueError, match="simulation_template must be a SimulationRequest instance"):
        validate_protocol_preset_factory_request(req)


def test_preserved_p12_validation_reject_invalid_simulation_template():
    bad_sim_req = replace(get_test_sim_template(), length=0)
    req = replace(make_valid_factory_request(), simulation_template=bad_sim_req)
    with pytest.raises(ValueError, match="length must be a positive int"):
        validate_protocol_preset_factory_request(req)


def test_preserved_p12_validation_reject_non_bool_create_parent_dirs():
    req = replace(make_valid_factory_request(), create_parent_dirs="True")
    with pytest.raises(ValueError, match="create_parent_dirs must be a bool"):
        validate_protocol_preset_factory_request(req)


def test_preserved_p12_validation_reject_non_bool_overwrite_existing():
    req = replace(make_valid_factory_request(), overwrite_existing=1)
    with pytest.raises(ValueError, match="overwrite_existing must be a bool"):
        validate_protocol_preset_factory_request(req)


def test_preserved_p12_validation_reject_non_bool_include_values():
    req = replace(make_valid_factory_request(), include_values=None)
    with pytest.raises(ValueError, match="include_values must be a bool"):
        validate_protocol_preset_factory_request(req)


def test_preserved_p12_validation_reject_non_bool_include_innovations():
    req = replace(make_valid_factory_request(), include_innovations="False")
    with pytest.raises(ValueError, match="include_innovations must be a bool"):
        validate_protocol_preset_factory_request(req)


def test_preserved_p12_validation_reject_non_bool_include_variances():
    req = replace(make_valid_factory_request(), include_variances=0.0)
    with pytest.raises(ValueError, match="include_variances must be a bool"):
        validate_protocol_preset_factory_request(req)


def test_preserved_p12_validation_reject_all_include_flags_false():
    req = make_valid_factory_request(include_values=False, include_innovations=False, include_variances=False)
    with pytest.raises(ValueError, match="At least one include flag must be True"):
        validate_protocol_preset_factory_request(req)


def test_preserved_p12_validation_reject_non_bool_json_sort_keys():
    req = replace(make_valid_factory_request(), json_sort_keys="True")
    with pytest.raises(ValueError, match="json_sort_keys must be a bool"):
        validate_protocol_preset_factory_request(req)


def test_preserved_p12_validation_reject_bool_json_indent():
    req = replace(make_valid_factory_request(), json_indent=True)
    with pytest.raises(ValueError, match="json_indent must be a non-negative int"):
        validate_protocol_preset_factory_request(req)


def test_preserved_p12_validation_reject_negative_json_indent():
    req = replace(make_valid_factory_request(), json_indent=-1)
    with pytest.raises(ValueError, match="json_indent must be a non-negative int"):
        validate_protocol_preset_factory_request(req)


def test_preserved_p12_validation_reject_sample_id_hash_len_too_short():
    req = replace(make_valid_factory_request(), sample_id_hash_len=7)
    with pytest.raises(ValueError, match="sample_id_hash_len must be an int >= 8"):
        validate_protocol_preset_factory_request(req)


def test_preserved_p12_validation_reject_output_root_dir_exists_as_file(tmp_path):
    file_path = tmp_path / "already_a_file.txt"
    file_path.write_text("content")
    req = make_valid_factory_request(output_root_dir=str(file_path))
    with pytest.raises(ValueError, match="exists and is a file"):
        validate_protocol_preset_factory_request(req)


def test_preserved_p12_validation_reject_missing_required_templates():
    # Reject missing AR template
    missing_ar = (
        (FamilyId.ARMA, get_test_gen_template(FamilyId.ARMA)),
        (FamilyId.GARCH, get_test_gen_template(FamilyId.GARCH)),
        (FamilyId.ARMA_GARCH, get_test_gen_template(FamilyId.ARMA_GARCH)),
    )
    req = replace(make_valid_factory_request(), generation_template_by_family=missing_ar)
    with pytest.raises(ValueError, match="Missing generation templates for required families:.*AR"):
        validate_protocol_preset_factory_request(req)
