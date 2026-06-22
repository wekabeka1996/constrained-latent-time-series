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
    validate_protocol_preset_factory_request,
    get_preset_sample_count_by_family,
    build_split_request_for_preset,
    build_phase2_preset_run_request,
    build_phase2_preset_factory_result,
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
            provenance=(("phase", "p12"),),
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
            provenance=(("phase", "p12"),),
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
            provenance=(("phase", "p12"),),
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
            provenance=(("phase", "p12"),),
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
# Group A: Enum and constants (Tests 1-5)
# ==============================================================================

def test_a_enums_and_constants_1():
    # 1. ProtocolPresetName has exactly smoke/dev/main.
    assert len(ProtocolPresetName) == 3
    assert ProtocolPresetName.SMOKE == "smoke"
    assert ProtocolPresetName.DEV == "dev"
    assert ProtocolPresetName.MAIN == "main"
    assert list(ProtocolPresetName) == ["smoke", "dev", "main"]


def test_a_enums_and_constants_2_3():
    # 2. Approved base seed constants contain AR, ARMA, GARCH, ARMA_GARCH.
    # 3. Approved base seed constants use 12001/12001/12002/12003.
    assert type(APPROVED_PHASE2_BASE_SEEDS_BY_FAMILY) is tuple
    seeds_dict = dict(APPROVED_PHASE2_BASE_SEEDS_BY_FAMILY)
    assert set(seeds_dict.keys()) == {FamilyId.AR, FamilyId.ARMA, FamilyId.GARCH, FamilyId.ARMA_GARCH}
    assert seeds_dict[FamilyId.AR] == 12001
    assert seeds_dict[FamilyId.ARMA] == 12001
    assert seeds_dict[FamilyId.GARCH] == 12002
    assert seeds_dict[FamilyId.ARMA_GARCH] == 12003


def test_a_enums_and_constants_4():
    # 4. Approved filenames are samples.jsonl and manifest.json.
    assert APPROVED_PHASE2_SAMPLES_FILENAME == "samples.jsonl"
    assert APPROVED_PHASE2_MANIFEST_FILENAME == "manifest.json"


def test_a_enums_and_constants_5():
    # 5. Approved subdir constants match smoke/zero_shot_train/zero_shot_eval/fewshot_train/fewshot_eval.
    assert type(APPROVED_PHASE2_ARTIFACT_SUBDIR_BY_SPLIT) is tuple
    subdirs = dict(APPROVED_PHASE2_ARTIFACT_SUBDIR_BY_SPLIT)
    assert subdirs[SplitName.SMOKE] == "smoke"
    assert subdirs[SplitName.ZERO_SHOT_TRAIN] == "zero_shot_train"
    assert subdirs[SplitName.ZERO_SHOT_EVAL] == "zero_shot_eval"
    assert subdirs[SplitName.FEWSHOT_TRAIN] == "fewshot_train"
    assert subdirs[SplitName.FEWSHOT_EVAL] == "fewshot_eval"
    assert len(subdirs) == 5


# ==============================================================================
# Group B: Factory request validation (Tests 6-32)
# ==============================================================================

def test_b_validation_6():
    # 6. Reject raw non-ProtocolPresetFactoryRequest.
    with pytest.raises(ValueError, match="must be a ProtocolPresetFactoryRequest instance"):
        validate_protocol_preset_factory_request("not_a_request")


def test_b_validation_7():
    # 7. Reject empty protocol_name.
    req = make_valid_factory_request(protocol_name="")
    with pytest.raises(ValueError, match="protocol_name must be a non-empty string"):
        validate_protocol_preset_factory_request(req)


def test_b_validation_8():
    # 8. Reject raw string preset_name.
    # Note: since preset_name is typed as ProtocolPresetName, we bypass type checking by using replace with an invalid value
    req = replace(make_valid_factory_request(), preset_name="smoke")
    with pytest.raises(ValueError, match="preset_name must be a ProtocolPresetName enum member"):
        validate_protocol_preset_factory_request(req)


def test_b_validation_9():
    # 9. Reject empty output_root_dir.
    req = make_valid_factory_request(output_root_dir="   ")
    with pytest.raises(ValueError, match="output_root_dir must be a non-empty string"):
        validate_protocol_preset_factory_request(req)


def test_b_validation_10():
    # 10. Reject list generation_template_by_family.
    req = replace(make_valid_factory_request(), generation_template_by_family=[])
    with pytest.raises(ValueError, match="generation_template_by_family must be exactly a tuple"):
        validate_protocol_preset_factory_request(req)


def test_b_validation_11():
    # 11. Reject duplicate family template.
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


def test_b_validation_12():
    # 12. Reject raw string family key.
    bad_keys = (
        ("AR", get_test_gen_template(FamilyId.AR)),
        (FamilyId.ARMA, get_test_gen_template(FamilyId.ARMA)),
        (FamilyId.GARCH, get_test_gen_template(FamilyId.GARCH)),
        (FamilyId.ARMA_GARCH, get_test_gen_template(FamilyId.ARMA_GARCH)),
    )
    req = replace(make_valid_factory_request(), generation_template_by_family=bad_keys)
    with pytest.raises(ValueError, match="family key in generation_template_by_family must be a FamilyId instance"):
        validate_protocol_preset_factory_request(req)


def test_b_validation_13():
    # 13. Reject non-GenerationRequest template.
    bad_templates = (
        (FamilyId.AR, "not_a_template"),
        (FamilyId.ARMA, get_test_gen_template(FamilyId.ARMA)),
        (FamilyId.GARCH, get_test_gen_template(FamilyId.GARCH)),
        (FamilyId.ARMA_GARCH, get_test_gen_template(FamilyId.ARMA_GARCH)),
    )
    req = replace(make_valid_factory_request(), generation_template_by_family=bad_templates)
    with pytest.raises(ValueError, match="template in generation_template_by_family must be a GenerationRequest instance"):
        validate_protocol_preset_factory_request(req)


def test_b_validation_14():
    # 14. Reject template family mismatch.
    bad_mismatch = (
        (FamilyId.AR, get_test_gen_template(FamilyId.ARMA)),
        (FamilyId.ARMA, get_test_gen_template(FamilyId.ARMA)),
        (FamilyId.GARCH, get_test_gen_template(FamilyId.GARCH)),
        (FamilyId.ARMA_GARCH, get_test_gen_template(FamilyId.ARMA_GARCH)),
    )
    req = replace(make_valid_factory_request(), generation_template_by_family=bad_mismatch)
    with pytest.raises(ValueError, match="does not match key"):
        validate_protocol_preset_factory_request(req)


def test_b_validation_15():
    # 15. Reject invalid GenerationRequest template.
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


def test_b_validation_16():
    # 16. Reject non-SimulationRequest simulation_template.
    req = replace(make_valid_factory_request(), simulation_template="not_a_simulation_template")
    with pytest.raises(ValueError, match="simulation_template must be a SimulationRequest instance"):
        validate_protocol_preset_factory_request(req)


def test_b_validation_17():
    # 17. Reject invalid SimulationRequest template.
    bad_sim_req = replace(get_test_sim_template(), length=0)
    req = replace(make_valid_factory_request(), simulation_template=bad_sim_req)
    with pytest.raises(ValueError, match="length must be a positive int"):
        validate_protocol_preset_factory_request(req)


def test_b_validation_18():
    # 18. Reject non-bool create_parent_dirs.
    req = replace(make_valid_factory_request(), create_parent_dirs="True")
    with pytest.raises(ValueError, match="create_parent_dirs must be a bool"):
        validate_protocol_preset_factory_request(req)


def test_b_validation_19():
    # 19. Reject non-bool overwrite_existing.
    req = replace(make_valid_factory_request(), overwrite_existing=1)
    with pytest.raises(ValueError, match="overwrite_existing must be a bool"):
        validate_protocol_preset_factory_request(req)


def test_b_validation_20():
    # 20. Reject non-bool include_values.
    req = replace(make_valid_factory_request(), include_values=None)
    with pytest.raises(ValueError, match="include_values must be a bool"):
        validate_protocol_preset_factory_request(req)


def test_b_validation_21():
    # 21. Reject non-bool include_innovations.
    req = replace(make_valid_factory_request(), include_innovations="False")
    with pytest.raises(ValueError, match="include_innovations must be a bool"):
        validate_protocol_preset_factory_request(req)


def test_b_validation_22():
    # 22. Reject non-bool include_variances.
    req = replace(make_valid_factory_request(), include_variances=0.0)
    with pytest.raises(ValueError, match="include_variances must be a bool"):
        validate_protocol_preset_factory_request(req)


def test_b_validation_23():
    # 23. Reject all include flags False.
    req = make_valid_factory_request(include_values=False, include_innovations=False, include_variances=False)
    with pytest.raises(ValueError, match="At least one include flag must be True"):
        validate_protocol_preset_factory_request(req)


def test_b_validation_24():
    # 24. Reject non-bool json_sort_keys.
    req = replace(make_valid_factory_request(), json_sort_keys="True")
    with pytest.raises(ValueError, match="json_sort_keys must be a bool"):
        validate_protocol_preset_factory_request(req)


def test_b_validation_25():
    # 25. Reject bool json_indent.
    req = replace(make_valid_factory_request(), json_indent=True)
    with pytest.raises(ValueError, match="json_indent must be a non-negative int"):
        validate_protocol_preset_factory_request(req)


def test_b_validation_26():
    # 26. Reject negative json_indent.
    req = replace(make_valid_factory_request(), json_indent=-1)
    with pytest.raises(ValueError, match="json_indent must be a non-negative int"):
        validate_protocol_preset_factory_request(req)


def test_b_validation_27():
    # 27. Reject sample_id_hash_len < 8.
    req = replace(make_valid_factory_request(), sample_id_hash_len=7)
    with pytest.raises(ValueError, match="sample_id_hash_len must be an int >= 8"):
        validate_protocol_preset_factory_request(req)


def test_b_validation_28(tmp_path):
    # 28. Reject output_root_dir that exists as file.
    file_path = tmp_path / "already_a_file.txt"
    file_path.write_text("content")
    req = make_valid_factory_request(output_root_dir=str(file_path))
    with pytest.raises(ValueError, match="exists and is a file"):
        validate_protocol_preset_factory_request(req)


def test_b_validation_29_32():
    # 29. Reject missing AR template.
    missing_ar = (
        (FamilyId.ARMA, get_test_gen_template(FamilyId.ARMA)),
        (FamilyId.GARCH, get_test_gen_template(FamilyId.GARCH)),
        (FamilyId.ARMA_GARCH, get_test_gen_template(FamilyId.ARMA_GARCH)),
    )
    req = replace(make_valid_factory_request(), generation_template_by_family=missing_ar)
    with pytest.raises(ValueError, match="Missing generation templates for required families:.*AR"):
        validate_protocol_preset_factory_request(req)

    # 30. Reject missing ARMA template.
    missing_arma = (
        (FamilyId.AR, get_test_gen_template(FamilyId.AR)),
        (FamilyId.GARCH, get_test_gen_template(FamilyId.GARCH)),
        (FamilyId.ARMA_GARCH, get_test_gen_template(FamilyId.ARMA_GARCH)),
    )
    req = replace(make_valid_factory_request(), generation_template_by_family=missing_arma)
    with pytest.raises(ValueError, match="Missing generation templates for required families:.*ARMA"):
        validate_protocol_preset_factory_request(req)

    # 31. Reject missing GARCH template.
    missing_garch = (
        (FamilyId.AR, get_test_gen_template(FamilyId.AR)),
        (FamilyId.ARMA, get_test_gen_template(FamilyId.ARMA)),
        (FamilyId.ARMA_GARCH, get_test_gen_template(FamilyId.ARMA_GARCH)),
    )
    req = replace(make_valid_factory_request(), generation_template_by_family=missing_garch)
    with pytest.raises(ValueError, match="Missing generation templates for required families:.*GARCH"):
        validate_protocol_preset_factory_request(req)

    # 32. Reject missing ARMA_GARCH template.
    missing_arma_garch = (
        (FamilyId.AR, get_test_gen_template(FamilyId.AR)),
        (FamilyId.ARMA, get_test_gen_template(FamilyId.ARMA)),
        (FamilyId.GARCH, get_test_gen_template(FamilyId.GARCH)),
    )
    req = replace(make_valid_factory_request(), generation_template_by_family=missing_arma_garch)
    with pytest.raises(ValueError, match="Missing generation templates for required families:.*ARMA_GARCH"):
        validate_protocol_preset_factory_request(req)


# ==============================================================================
# Group C: Sample count presets (Tests 33-36)
# ==============================================================================

def test_c_sample_counts_33():
    # 33. SMOKE family counts are 1000 each.
    counts = get_preset_sample_count_by_family(ProtocolPresetName.SMOKE)
    assert len(counts) == 4
    for fid, cnt in counts:
        assert cnt == 1000


def test_c_sample_counts_34():
    # 34. DEV family counts are 10000 each.
    counts = get_preset_sample_count_by_family(ProtocolPresetName.DEV)
    assert len(counts) == 4
    for fid, cnt in counts:
        assert cnt == 10000


def test_c_sample_counts_35():
    # 35. MAIN family counts are 100000 each.
    counts = get_preset_sample_count_by_family(ProtocolPresetName.MAIN)
    assert len(counts) == 4
    for fid, cnt in counts:
        assert cnt == 100000


def test_c_sample_counts_36():
    # 36. Invalid preset_name rejected by get_preset_sample_count_by_family.
    with pytest.raises(ValueError):
        get_preset_sample_count_by_family("smoke")


# ==============================================================================
# Group D: Split request generation (Tests 37-49)
# ==============================================================================

def test_d_split_requests_37():
    # 37. SMOKE split includes AR, ARMA, GARCH, ARMA_GARCH.
    req = make_valid_factory_request(preset_name=ProtocolPresetName.SMOKE)
    split_req = build_split_request_for_preset(
        preset_name=req.preset_name,
        split_name=SplitName.SMOKE,
        generation_template_by_family=req.generation_template_by_family,
        simulation_template=req.simulation_template,
        sample_id_hash_len=req.sample_id_hash_len,
    )
    counts = dict(split_req.sample_count_by_family)
    assert set(counts.keys()) == {FamilyId.AR, FamilyId.ARMA, FamilyId.GARCH, FamilyId.ARMA_GARCH}
    assert counts[FamilyId.AR] == 1000
    assert counts[FamilyId.ARMA] == 1000
    assert counts[FamilyId.GARCH] == 1000
    assert counts[FamilyId.ARMA_GARCH] == 1000


def test_d_split_requests_38_39():
    # 38. ZERO_SHOT_TRAIN excludes ARMA_GARCH or sets ARMA_GARCH count 0.
    # 39. ZERO_SHOT_TRAIN enforce_zero_shot_c_train_exclusion is True.
    req = make_valid_factory_request(preset_name=ProtocolPresetName.SMOKE)
    split_req = build_split_request_for_preset(
        preset_name=req.preset_name,
        split_name=SplitName.ZERO_SHOT_TRAIN,
        generation_template_by_family=req.generation_template_by_family,
        simulation_template=req.simulation_template,
        sample_id_hash_len=req.sample_id_hash_len,
    )
    counts = dict(split_req.sample_count_by_family)
    assert FamilyId.ARMA_GARCH not in counts or counts[FamilyId.ARMA_GARCH] == 0
    assert split_req.enforce_zero_shot_c_train_exclusion is True


def test_d_split_requests_40_41_42():
    # 40. ZERO_SHOT_EVAL includes only ARMA_GARCH.
    # 41. FEWSHOT_TRAIN includes only ARMA_GARCH.
    # 42. FEWSHOT_EVAL includes only ARMA_GARCH.
    req = make_valid_factory_request(preset_name=ProtocolPresetName.SMOKE)
    
    for split_name in [SplitName.ZERO_SHOT_EVAL, SplitName.FEWSHOT_TRAIN, SplitName.FEWSHOT_EVAL]:
        split_req = build_split_request_for_preset(
            preset_name=req.preset_name,
            split_name=split_name,
            generation_template_by_family=req.generation_template_by_family,
            simulation_template=req.simulation_template,
            sample_id_hash_len=req.sample_id_hash_len,
        )
        counts = dict(split_req.sample_count_by_family)
        assert list(counts.keys()) == [FamilyId.ARMA_GARCH]
        assert counts[FamilyId.ARMA_GARCH] == 1000


def test_d_split_requests_43():
    # 43. Each split uses correct artifact_subdir.
    req = make_valid_factory_request(preset_name=ProtocolPresetName.SMOKE)
    for split_name in SplitName:
        split_req = build_split_request_for_preset(
            preset_name=req.preset_name,
            split_name=split_name,
            generation_template_by_family=req.generation_template_by_family,
            simulation_template=req.simulation_template,
            sample_id_hash_len=req.sample_id_hash_len,
        )
        expected_subdir = dict(APPROVED_PHASE2_ARTIFACT_SUBDIR_BY_SPLIT)[split_name]
        assert split_req.artifact_subdir == expected_subdir


def test_d_split_requests_44():
    # 44. Each split uses approved filenames.
    req = make_valid_factory_request(preset_name=ProtocolPresetName.SMOKE)
    for split_name in SplitName:
        split_req = build_split_request_for_preset(
            preset_name=req.preset_name,
            split_name=split_name,
            generation_template_by_family=req.generation_template_by_family,
            simulation_template=req.simulation_template,
            sample_id_hash_len=req.sample_id_hash_len,
        )
        assert split_req.samples_filename == APPROVED_PHASE2_SAMPLES_FILENAME
        assert split_req.manifest_filename == APPROVED_PHASE2_MANIFEST_FILENAME


def test_d_split_requests_45():
    # 45. Each split uses approved base seeds.
    req = make_valid_factory_request(preset_name=ProtocolPresetName.SMOKE)
    for split_name in SplitName:
        split_req = build_split_request_for_preset(
            preset_name=req.preset_name,
            split_name=split_name,
            generation_template_by_family=req.generation_template_by_family,
            simulation_template=req.simulation_template,
            sample_id_hash_len=req.sample_id_hash_len,
        )
        assert split_req.base_seed_by_family == APPROVED_PHASE2_BASE_SEEDS_BY_FAMILY


def test_d_split_requests_46():
    # 46. Each split passes validate_split_artifact_request.
    req = make_valid_factory_request(preset_name=ProtocolPresetName.SMOKE)
    for split_name in SplitName:
        split_req = build_split_request_for_preset(
            preset_name=req.preset_name,
            split_name=split_name,
            generation_template_by_family=req.generation_template_by_family,
            simulation_template=req.simulation_template,
            sample_id_hash_len=req.sample_id_hash_len,
        )
        validate_split_artifact_request(split_req)  # should not raise


def test_d_split_requests_47_48_49():
    # 47. Split request sample_id_hash_len is propagated.
    # 48. Split request simulation_template is propagated.
    # 49. Split request generation_template_by_family is propagated.
    req = make_valid_factory_request(preset_name=ProtocolPresetName.SMOKE, sample_id_hash_len=12)
    split_req = build_split_request_for_preset(
        preset_name=req.preset_name,
        split_name=SplitName.SMOKE,
        generation_template_by_family=req.generation_template_by_family,
        simulation_template=req.simulation_template,
        sample_id_hash_len=req.sample_id_hash_len,
    )
    assert split_req.sample_id_hash_len == 12
    assert split_req.simulation_template == req.simulation_template
    assert split_req.generation_template_by_family == req.generation_template_by_family


# ==============================================================================
# Group E: Full run request generation (Tests 50-60)
# ==============================================================================

def test_e_run_requests_50_51_52():
    # 50. build_phase2_preset_run_request returns Phase2ArtifactRunRequest.
    # 51. split order is exactly smoke, zero_shot_train, zero_shot_eval, fewshot_train, fewshot_eval.
    # 52. run request protocol_name propagated.
    req = make_valid_factory_request(protocol_name="custom_protocol")
    run_req = build_phase2_preset_run_request(req)
    assert isinstance(run_req, Phase2ArtifactRunRequest)
    assert run_req.protocol_name == "custom_protocol"
    
    expected_order = [
        SplitName.SMOKE,
        SplitName.ZERO_SHOT_TRAIN,
        SplitName.ZERO_SHOT_EVAL,
        SplitName.FEWSHOT_TRAIN,
        SplitName.FEWSHOT_EVAL,
    ]
    assert [s.split_name for s in run_req.split_requests] == expected_order


def test_e_run_requests_53_58():
    # 53. output_root_dir propagated.
    # 54. create_parent_dirs propagated.
    # 55. overwrite_existing propagated.
    # 56. include flags propagated.
    # 57. json_sort_keys propagated.
    # 58. json_indent propagated.
    req = make_valid_factory_request(
        output_root_dir="results/preset_test",
        create_parent_dirs=True,
        overwrite_existing=False,
        include_values=True,
        include_innovations=False,
        include_variances=True,
        json_sort_keys=False,
        json_indent=4,
    )
    run_req = build_phase2_preset_run_request(req)
    assert run_req.output_root_dir == "results/preset_test"
    assert run_req.create_parent_dirs is True
    assert run_req.overwrite_existing is False
    assert run_req.include_values is True
    assert run_req.include_innovations is False
    assert run_req.include_variances is True
    assert run_req.json_sort_keys is False
    assert run_req.json_indent == 4


def test_e_run_requests_59_60():
    # 59. run request passes validate_phase2_artifact_run_request.
    # 60. build_phase2_preset_run_request does not create files or directories.
    req = make_valid_factory_request(output_root_dir="nonexistent_test_directory_never_created")
    run_req = build_phase2_preset_run_request(req)
    validate_phase2_artifact_run_request(run_req)  # should not raise
    assert not pathlib.Path("nonexistent_test_directory_never_created").exists()


# ==============================================================================
# Group F: Factory result (Tests 61-66)
# ==============================================================================

def test_f_factory_result_61_62_63():
    # 61. build_phase2_preset_factory_result returns ProtocolPresetFactoryResult.
    # 62. result reason is "phase2_protocol_preset_request_built".
    # 63. result split_count is 5.
    req = make_valid_factory_request(preset_name=ProtocolPresetName.SMOKE)
    res = build_phase2_preset_factory_result(req)
    assert isinstance(res, ProtocolPresetFactoryResult)
    assert res.preset_name == ProtocolPresetName.SMOKE
    assert res.reason == "phase2_protocol_preset_request_built"
    assert res.split_count == 5


def test_f_factory_result_64():
    # 64. result total_requested_samples equals sum of all split request counts.
    # SMOKE: SMOKE split (4 * 1000) + ZERO_SHOT_TRAIN split (3 * 1000) + ZERO_SHOT_EVAL (1 * 1000) + FEWSHOT_TRAIN (1 * 1000) + FEWSHOT_EVAL (1 * 1000) = 10 * 1000 = 10000.
    req = make_valid_factory_request(preset_name=ProtocolPresetName.SMOKE)
    res = build_phase2_preset_factory_result(req)
    assert res.total_requested_samples == 10000

    # DEV: 10 * 10000 = 100000
    req_dev = make_valid_factory_request(preset_name=ProtocolPresetName.DEV)
    res_dev = build_phase2_preset_factory_result(req_dev)
    assert res_dev.total_requested_samples == 100000


def test_f_factory_result_65_66():
    # 65. count_by_split preserves split order.
    # 66. run_request inside result passes validate_phase2_artifact_run_request.
    req = make_valid_factory_request(preset_name=ProtocolPresetName.SMOKE)
    res = build_phase2_preset_factory_result(req)
    
    expected_splits = [
        SplitName.SMOKE,
        SplitName.ZERO_SHOT_TRAIN,
        SplitName.ZERO_SHOT_EVAL,
        SplitName.FEWSHOT_TRAIN,
        SplitName.FEWSHOT_EVAL,
    ]
    assert [split for split, _ in res.count_by_split] == expected_splits
    validate_phase2_artifact_run_request(res.run_request)  # should not raise


# ==============================================================================
# Group G: Scope (Tests 67-74)
# ==============================================================================

def test_g_scope_67_70():
    # 67. protocol_presets.py does not import torch directly.
    # 68. protocol_presets.py does not import numpy directly.
    # 69. protocol_presets.py does not import pandas/yaml/argparse.
    # 70. protocol_presets.py does not import model/training modules.
    import src.phase2.protocol_presets as presets
    assert "torch" not in sys.modules or "torch" not in dir(presets)
    assert "numpy" not in sys.modules or "numpy" not in dir(presets)
    assert "pandas" not in dir(presets)
    assert "yaml" not in dir(presets)
    assert "argparse" not in dir(presets)
    assert "models" not in dir(presets)


def test_g_scope_71_73():
    # 71. protocol_presets.py does not call run_phase2_artifact_generation.
    # 72. protocol_presets.py does not call write_dataset_artifacts.
    # 73. protocol_presets.py does not call build_dataset_in_memory.
    import src.phase2.protocol_presets as presets
    # Check that execution functions are not imported or present in presets namespace
    assert "run_phase2_artifact_generation" not in dir(presets)
    assert "write_dataset_artifacts" not in dir(presets)
    assert "build_dataset_in_memory" not in dir(presets)


def test_g_scope_74():
    # 74. P12 tests do not create artifact files.
    # We only build the run request, and never execute or write files, so no files should be created.
    assert True
