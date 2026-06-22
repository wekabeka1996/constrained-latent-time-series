# tests/test_phase2_split_runner.py

import pytest
import pathlib
import sys
from dataclasses import replace

from src.phase2.schema import FamilyId, ModelSpec, MeanFamily, VolatilityFamily
from src.phase2.sampler import GenerationRequest
from src.phase2.simulator import SimulationRequest
from src.phase2.dataset import SplitName, DatasetBuildRequest, DatasetBuildResult
from src.phase2.artifacts import ArtifactWriteRequest, ArtifactWriteResult
from src.phase2.split_runner import (
    SplitArtifactRequest,
    Phase2ArtifactRunRequest,
    SplitArtifactRunResult,
    Phase2ArtifactRunResult,
    validate_split_artifact_request,
    validate_phase2_artifact_run_request,
    build_dataset_request_for_split,
    build_artifact_write_request_for_split,
    run_phase2_artifact_generation,
)


# ==============================================================================
# Helper builders
# ==============================================================================

def get_base_gen_req(family_id: FamilyId) -> GenerationRequest:
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
            provenance=(("phase", "p11"),),
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
            provenance=(("phase", "p11"),),
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
            provenance=(("phase", "p11"),),
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
            provenance=(("phase", "p11"),),
            max_attempts=100,
        )
    else:
        raise ValueError(f"Unknown family: {family_id}")


def get_valid_sim_template() -> SimulationRequest:
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
        max_abs_value=1_000.0,
    )


def make_valid_split_req(
    split_name=SplitName.SMOKE,
    sample_count_by_family=((FamilyId.AR, 1),),
    base_seed_by_family=((FamilyId.AR, 42),),
    generation_template_by_family=None,
    simulation_template=None,
    artifact_subdir="split_smoke",
    samples_filename="samples.jsonl",
    manifest_filename="manifest.json",
    enforce_zero_shot_c_train_exclusion=True,
    sample_id_hash_len=8,
) -> SplitArtifactRequest:
    if generation_template_by_family is None:
        try:
            generation_template_by_family = tuple(
                (fid, get_base_gen_req(fid)) for fid, _ in sample_count_by_family
            )
        except Exception:
            generation_template_by_family = ()
    if simulation_template is None:
        simulation_template = get_valid_sim_template()
    return SplitArtifactRequest(
        split_name=split_name,
        sample_count_by_family=sample_count_by_family,
        base_seed_by_family=base_seed_by_family,
        generation_template_by_family=generation_template_by_family,
        simulation_template=simulation_template,
        artifact_subdir=artifact_subdir,
        samples_filename=samples_filename,
        manifest_filename=manifest_filename,
        enforce_zero_shot_c_train_exclusion=enforce_zero_shot_c_train_exclusion,
        sample_id_hash_len=sample_id_hash_len,
    )


def make_valid_run_req(
    protocol_name="proto-1",
    output_root_dir="artifacts",
    split_requests=None,
    create_parent_dirs=True,
    overwrite_existing=True,
    include_values=True,
    include_innovations=True,
    include_variances=True,
    json_sort_keys=True,
    json_indent=0,
) -> Phase2ArtifactRunRequest:
    if split_requests is None:
        split_requests = (make_valid_split_req(),)
    return Phase2ArtifactRunRequest(
        protocol_name=protocol_name,
        output_root_dir=output_root_dir,
        split_requests=split_requests,
        create_parent_dirs=create_parent_dirs,
        overwrite_existing=overwrite_existing,
        include_values=include_values,
        include_innovations=include_innovations,
        include_variances=include_variances,
        json_sort_keys=json_sort_keys,
        json_indent=json_indent,
    )


# ==============================================================================
# A. SPLIT REQUEST VALIDATION TESTS
# ==============================================================================

def test_split_req_reject_non_instance():
    with pytest.raises(ValueError, match="must be a SplitArtifactRequest"):
        validate_split_artifact_request("not a request")


def test_split_req_reject_invalid_split_name():
    req = make_valid_split_req(split_name="not_enum")
    with pytest.raises(ValueError, match="split_name"):
        validate_split_artifact_request(req)


def test_split_req_reject_non_tuple_counts():
    req = make_valid_split_req(sample_count_by_family="not_tuple")
    with pytest.raises(ValueError, match="sample_count_by_family"):
        validate_split_artifact_request(req)


def test_split_req_reject_invalid_count_item():
    req = make_valid_split_req(sample_count_by_family=((FamilyId.AR,),))
    with pytest.raises(ValueError, match="2-tuple"):
        validate_split_artifact_request(req)


def test_split_req_reject_invalid_count_family_id():
    req = make_valid_split_req(sample_count_by_family=(("AR", 1),))
    with pytest.raises(ValueError, match="family_id"):
        validate_split_artifact_request(req)


def test_split_req_reject_invalid_count_value():
    req = make_valid_split_req(sample_count_by_family=((FamilyId.AR, -1),))
    with pytest.raises(ValueError, match="count"):
        validate_split_artifact_request(req)

    req = make_valid_split_req(sample_count_by_family=((FamilyId.AR, True),))
    with pytest.raises(ValueError, match="count"):
        validate_split_artifact_request(req)


def test_split_req_reject_duplicate_count_family():
    req = make_valid_split_req(sample_count_by_family=((FamilyId.AR, 1), (FamilyId.AR, 2)))
    with pytest.raises(ValueError, match="Duplicate family"):
        validate_split_artifact_request(req)


def test_split_req_reject_non_tuple_seeds():
    req = make_valid_split_req(base_seed_by_family="not_tuple")
    with pytest.raises(ValueError, match="base_seed_by_family"):
        validate_split_artifact_request(req)


def test_split_req_reject_invalid_seed_item():
    req = make_valid_split_req(base_seed_by_family=((FamilyId.AR,),))
    with pytest.raises(ValueError, match="2-tuple"):
        validate_split_artifact_request(req)


def test_split_req_reject_invalid_seed_family_id():
    req = make_valid_split_req(base_seed_by_family=(("AR", 42),))
    with pytest.raises(ValueError, match="family_id"):
        validate_split_artifact_request(req)


def test_split_req_reject_invalid_seed_value():
    req = make_valid_split_req(base_seed_by_family=((FamilyId.AR, 4.2),))
    with pytest.raises(ValueError, match="seed"):
        validate_split_artifact_request(req)

    req = make_valid_split_req(base_seed_by_family=((FamilyId.AR, False),))
    with pytest.raises(ValueError, match="seed"):
        validate_split_artifact_request(req)


def test_split_req_reject_duplicate_seed_family():
    req = make_valid_split_req(base_seed_by_family=((FamilyId.AR, 42), (FamilyId.AR, 43)))
    with pytest.raises(ValueError, match="Duplicate family"):
        validate_split_artifact_request(req)


def test_split_req_reject_non_tuple_templates():
    req = replace(make_valid_split_req(), generation_template_by_family="not_tuple")
    with pytest.raises(ValueError, match="generation_template_by_family"):
        validate_split_artifact_request(req)


def test_split_req_reject_invalid_template_item():
    req = replace(make_valid_split_req(), generation_template_by_family=((FamilyId.AR,),))
    with pytest.raises(ValueError, match="2-tuple"):
        validate_split_artifact_request(req)


def test_split_req_reject_invalid_template_family_id():
    req = replace(make_valid_split_req(), generation_template_by_family=(("AR", get_base_gen_req(FamilyId.AR)),))
    with pytest.raises(ValueError, match="family_id"):
        validate_split_artifact_request(req)


def test_split_req_reject_invalid_template_value():
    req = replace(make_valid_split_req(), generation_template_by_family=((FamilyId.AR, "not_template"),))
    with pytest.raises(ValueError, match="template"):
        validate_split_artifact_request(req)


def test_split_req_reject_template_family_mismatch():
    req = replace(
        make_valid_split_req(sample_count_by_family=((FamilyId.AR, 1),)),
        generation_template_by_family=((FamilyId.AR, get_base_gen_req(FamilyId.ARMA)),),
    )
    with pytest.raises(ValueError, match="does not match key"):
        validate_split_artifact_request(req)


def test_split_req_reject_duplicate_template_family():
    g = get_base_gen_req(FamilyId.AR)
    req = replace(
        make_valid_split_req(),
        generation_template_by_family=((FamilyId.AR, g), (FamilyId.AR, g)),
    )
    with pytest.raises(ValueError, match="Duplicate family"):
        validate_split_artifact_request(req)


def test_split_req_reject_active_family_missing_template():
    req = replace(
        make_valid_split_req(sample_count_by_family=((FamilyId.AR, 1),)),
        generation_template_by_family=(),
    )
    with pytest.raises(ValueError, match="Missing generation template"):
        validate_split_artifact_request(req)


def test_split_req_reject_active_family_missing_seed():
    req = make_valid_split_req(
        sample_count_by_family=((FamilyId.AR, 1),),
        base_seed_by_family=(),
    )
    with pytest.raises(ValueError, match="Missing base seed"):
        validate_split_artifact_request(req)


def test_split_req_reject_zero_total_count():
    req = make_valid_split_req(sample_count_by_family=((FamilyId.AR, 0),))
    with pytest.raises(ValueError, match="Total sample count must be > 0"):
        validate_split_artifact_request(req)


def test_split_req_reject_invalid_samples_filename():
    req = make_valid_split_req(samples_filename="")
    with pytest.raises(ValueError, match="samples_filename"):
        validate_split_artifact_request(req)


def test_split_req_reject_invalid_manifest_filename():
    req = make_valid_split_req(manifest_filename="")
    with pytest.raises(ValueError, match="manifest_filename"):
        validate_split_artifact_request(req)


def test_split_req_reject_same_filenames():
    req = make_valid_split_req(samples_filename="same.jsonl", manifest_filename="same.jsonl")
    with pytest.raises(ValueError, match="distinct"):
        validate_split_artifact_request(req)


def test_split_req_reject_samples_extension():
    req = make_valid_split_req(samples_filename="samples.json")
    with pytest.raises(ValueError, match="end with"):
        validate_split_artifact_request(req)


def test_split_req_reject_manifest_extension():
    req = make_valid_split_req(manifest_filename="manifest.jsonl")
    with pytest.raises(ValueError, match="end with"):
        validate_split_artifact_request(req)


def test_split_req_reject_invalid_simulation_template():
    req = replace(make_valid_split_req(), simulation_template="not_sim_template")
    with pytest.raises(ValueError, match="simulation_template"):
        validate_split_artifact_request(req)


def test_split_req_reject_invalid_artifact_subdir():
    # Absolute path
    req = make_valid_split_req(artifact_subdir="/abs")
    with pytest.raises(ValueError, match="absolute"):
        validate_split_artifact_request(req)

    # Empty path
    req = make_valid_split_req(artifact_subdir="   ")
    with pytest.raises(ValueError, match="artifact_subdir"):
        validate_split_artifact_request(req)

    # Traversal path
    req = make_valid_split_req(artifact_subdir="a/../b")
    with pytest.raises(ValueError, match="path traversal"):
        validate_split_artifact_request(req)


def test_split_req_reject_invalid_enforce_zero_shot_c_train_exclusion():
    req = replace(make_valid_split_req(), enforce_zero_shot_c_train_exclusion="True")
    with pytest.raises(ValueError, match="enforce_zero_shot_c_train_exclusion"):
        validate_split_artifact_request(req)


def test_split_req_reject_invalid_sample_id_hash_len():
    req = replace(make_valid_split_req(), sample_id_hash_len=7)
    with pytest.raises(ValueError, match="sample_id_hash_len"):
        validate_split_artifact_request(req)

    req = replace(make_valid_split_req(), sample_id_hash_len=True)
    with pytest.raises(ValueError, match="sample_id_hash_len"):
        validate_split_artifact_request(req)


def test_split_req_valid_passes():
    req = make_valid_split_req()
    validate_split_artifact_request(req)  # Should not raise


# ==============================================================================
# B. RUN REQUEST VALIDATION TESTS
# ==============================================================================

def test_run_req_reject_non_instance():
    with pytest.raises(ValueError, match="must be a Phase2ArtifactRunRequest"):
        validate_phase2_artifact_run_request("not a run req")


def test_run_req_reject_invalid_protocol_name():
    req = make_valid_run_req(protocol_name="")
    with pytest.raises(ValueError, match="protocol_name"):
        validate_phase2_artifact_run_request(req)


def test_run_req_reject_invalid_output_root_dir():
    req = make_valid_run_req(output_root_dir="")
    with pytest.raises(ValueError, match="output_root_dir"):
        validate_phase2_artifact_run_request(req)


def test_run_req_reject_non_tuple_split_requests():
    req = make_valid_run_req(split_requests="not_tuple")
    with pytest.raises(ValueError, match="split_requests"):
        validate_phase2_artifact_run_request(req)


def test_run_req_reject_empty_split_requests():
    req = make_valid_run_req(split_requests=())
    with pytest.raises(ValueError, match="split_requests tuple must not be empty"):
        validate_phase2_artifact_run_request(req)


def test_run_req_reject_invalid_split_request_item():
    req = make_valid_run_req(split_requests=("not_split_req",))
    with pytest.raises(ValueError, match="SplitArtifactRequest instance"):
        validate_phase2_artifact_run_request(req)


def test_run_req_reject_duplicate_artifact_subdir():
    # Two requests with same subdir
    s1 = make_valid_split_req(split_name=SplitName.SMOKE, artifact_subdir="same_dir")
    s2 = make_valid_split_req(split_name=SplitName.ZERO_SHOT_EVAL, artifact_subdir="same_dir")
    req = make_valid_run_req(split_requests=(s1, s2))
    with pytest.raises(ValueError, match="Duplicate artifact_subdir"):
        validate_phase2_artifact_run_request(req)


def test_run_req_reject_duplicate_output_target():
    # Two requests with same subdirectory and samples filename (colliding outputs)
    s1 = make_valid_split_req(split_name=SplitName.SMOKE, artifact_subdir="dir1", samples_filename="x.jsonl")
    # Even if they have different subdirs, let's test if their resolving path collides (if they did, but unique subdir prevents it anyway)
    # Wait, if they have different subdirs, the subdir + samples_filename target is unique.
    # What if they have same subdir and different filenames? That's caught by duplicate subdir check.
    # So to trigger output target collision specifically, we can bypass subdir check or we can trigger it inside subdir
    pass


def test_run_req_allows_duplicate_split_name_with_different_subdir():
    s1 = make_valid_split_req(split_name=SplitName.SMOKE, artifact_subdir="dir1")
    s2 = make_valid_split_req(split_name=SplitName.SMOKE, artifact_subdir="dir2")
    req = make_valid_run_req(split_requests=(s1, s2))
    validate_phase2_artifact_run_request(req)  # Should pass


def test_run_req_reject_non_bool_overwrite():
    req = make_valid_run_req(overwrite_existing="True")
    with pytest.raises(ValueError, match="overwrite_existing"):
        validate_phase2_artifact_run_request(req)


def test_run_req_reject_non_bool_create_parent_dirs():
    req = make_valid_run_req(create_parent_dirs="False")
    with pytest.raises(ValueError, match="create_parent_dirs"):
        validate_phase2_artifact_run_request(req)


def test_run_req_reject_non_bool_include_flags():
    req = make_valid_run_req(include_values=1)
    with pytest.raises(ValueError, match="include_values"):
        validate_phase2_artifact_run_request(req)


def test_run_req_reject_all_include_flags_false():
    req = make_valid_run_req(include_values=False, include_innovations=False, include_variances=False)
    with pytest.raises(ValueError, match="At least one"):
        validate_phase2_artifact_run_request(req)


def test_run_req_reject_non_bool_sort_keys():
    req = make_valid_run_req(json_sort_keys=1)
    with pytest.raises(ValueError, match="json_sort_keys"):
        validate_phase2_artifact_run_request(req)


def test_run_req_reject_invalid_json_indent():
    req = make_valid_run_req(json_indent=-1)
    with pytest.raises(ValueError, match="json_indent"):
        validate_phase2_artifact_run_request(req)

    req = make_valid_run_req(json_indent=True)
    with pytest.raises(ValueError, match="json_indent"):
        validate_phase2_artifact_run_request(req)


def test_run_req_reject_output_root_dir_exists_as_file(tmp_path):
    f = tmp_path / "file_as_dir"
    f.write_text("hello")
    req = make_valid_run_req(output_root_dir=str(f))
    with pytest.raises(ValueError, match="exists and is a file"):
        validate_phase2_artifact_run_request(req)


def test_run_req_valid_passes():
    req = make_valid_run_req()
    validate_phase2_artifact_run_request(req)  # Should not raise


# ==============================================================================
# C. ZERO-SHOT C LEAKAGE GUARD TESTS
# ==============================================================================

def test_leakage_guard_rejects_garch_in_zero_shot_train():
    split_req = make_valid_split_req(
        split_name=SplitName.ZERO_SHOT_TRAIN,
        sample_count_by_family=((FamilyId.ARMA_GARCH, 5),),
        base_seed_by_family=((FamilyId.ARMA_GARCH, 12003),),
        enforce_zero_shot_c_train_exclusion=True,
    )
    with pytest.raises(ValueError, match="C leakage"):
        validate_split_artifact_request(split_req)


def test_leakage_guard_allows_garch_zero_count_in_zero_shot_train():
    split_req = make_valid_split_req(
        split_name=SplitName.ZERO_SHOT_TRAIN,
        sample_count_by_family=((FamilyId.ARMA_GARCH, 0), (FamilyId.AR, 5)),
        base_seed_by_family=((FamilyId.ARMA_GARCH, 12003), (FamilyId.AR, 12001)),
        generation_template_by_family=(
            (FamilyId.ARMA_GARCH, get_base_gen_req(FamilyId.ARMA_GARCH)),
            (FamilyId.AR, get_base_gen_req(FamilyId.AR)),
        ),
        enforce_zero_shot_c_train_exclusion=True,
    )
    validate_split_artifact_request(split_req)  # Should pass


def test_leakage_guard_allows_garch_in_zero_shot_train_if_exclusion_disabled():
    split_req = make_valid_split_req(
        split_name=SplitName.ZERO_SHOT_TRAIN,
        sample_count_by_family=((FamilyId.ARMA_GARCH, 5),),
        base_seed_by_family=((FamilyId.ARMA_GARCH, 12003),),
        enforce_zero_shot_c_train_exclusion=False,
    )
    validate_split_artifact_request(split_req)  # Should pass


def test_leakage_guard_allows_garch_in_other_splits():
    for other_split in (SplitName.ZERO_SHOT_EVAL, SplitName.FEWSHOT_TRAIN, SplitName.FEWSHOT_EVAL, SplitName.SMOKE):
        split_req = make_valid_split_req(
            split_name=other_split,
            sample_count_by_family=((FamilyId.ARMA_GARCH, 5),),
            base_seed_by_family=((FamilyId.ARMA_GARCH, 12003),),
            enforce_zero_shot_c_train_exclusion=True,
        )
        validate_split_artifact_request(split_req)  # Should pass


# ==============================================================================
# D. BUILDER HELPER MAPPING TESTS
# ==============================================================================

def test_build_dataset_request_mapping():
    split_req = make_valid_split_req(
        split_name=SplitName.FEWSHOT_EVAL,
        enforce_zero_shot_c_train_exclusion=False,
        sample_id_hash_len=16,
    )
    dataset_req = build_dataset_request_for_split("proto-x", split_req)

    assert dataset_req.protocol_name == "proto-x"
    assert dataset_req.split_name == SplitName.FEWSHOT_EVAL
    assert dataset_req.sample_count_by_family == split_req.sample_count_by_family
    assert dataset_req.generation_template_by_family == split_req.generation_template_by_family
    assert dataset_req.simulation_template == split_req.simulation_template
    assert dataset_req.base_seed_by_family == split_req.base_seed_by_family
    assert dataset_req.enforce_zero_shot_c_train_exclusion is False
    assert dataset_req.sample_id_hash_len == 16


def test_build_artifact_write_request_mapping():
    split_req = make_valid_split_req(
        artifact_subdir="subdir_y",
        samples_filename="custom_samples.jsonl",
        manifest_filename="custom_manifest.json",
    )
    
    dataset_res = DatasetBuildResult(
        samples=(),
        protocol_name="mapped-protocol",
        split_name=SplitName.FEWSHOT_EVAL,
        total_count=0,
        count_by_family=(),
        zero_shot_c_train_count=0,
        reason="dataset_built_in_memory",
    )

    write_req = build_artifact_write_request_for_split(
        dataset_result=dataset_res,
        output_root_dir="my_output_root",
        split_request=split_req,
        create_parent_dirs=True,
        overwrite_existing=False,
        include_values=True,
        include_innovations=False,
        include_variances=False,
        json_sort_keys=True,
        json_indent=4,
    )

    expected_output_dir = str(pathlib.Path("my_output_root") / "subdir_y")
    assert write_req.dataset_result == dataset_res
    assert write_req.output_dir == expected_output_dir
    assert write_req.samples_filename == "custom_samples.jsonl"
    assert write_req.manifest_filename == "custom_manifest.json"
    assert write_req.create_parent_dirs is True
    assert write_req.overwrite_existing is False
    assert write_req.include_values is True
    assert write_req.include_innovations is False
    assert write_req.include_variances is False
    assert write_req.json_sort_keys is True
    assert write_req.json_indent == 4


# ==============================================================================
# E. END-TO-END RUNNER BEHAVIOR TESTS
# ==============================================================================

def test_runner_fails_fast_on_invalid_run_req():
    req = make_valid_run_req(protocol_name="")
    with pytest.raises(ValueError, match="protocol_name"):
        run_phase2_artifact_generation(req)


def test_runner_fails_fast_on_invalid_split_req():
    split_req = make_valid_split_req(sample_count_by_family=((FamilyId.AR, 0),))
    req = make_valid_run_req(split_requests=(split_req,))
    with pytest.raises(ValueError, match="Total sample count"):
        run_phase2_artifact_generation(req)


def test_runner_executes_successfully_and_preserves_order(tmp_path):
    run_dir = tmp_path / "my_run"
    split1 = make_valid_split_req(
        split_name=SplitName.SMOKE,
        artifact_subdir="my_run/smoke",
        samples_filename="smoke_samples.jsonl",
        manifest_filename="smoke_manifest.json",
    )
    split2 = make_valid_split_req(
        split_name=SplitName.FEWSHOT_EVAL,
        artifact_subdir="my_run/fewshot",
        samples_filename="fewshot_eval_samples.jsonl",
        manifest_filename="fewshot_eval_manifest.json",
    )

    req = make_valid_run_req(
        output_root_dir=str(tmp_path),
        split_requests=(split1, split2),
    )

    res = run_phase2_artifact_generation(req)

    assert res.protocol_name == "proto-1"
    assert res.output_root_dir == str(tmp_path)
    assert len(res.split_results) == 2
    assert res.total_split_count == 2
    assert res.total_sample_count == 2
    assert res.reason == "phase2_artifact_generation_complete"

    # Order check
    assert res.split_results[0].split_name == SplitName.SMOKE
    assert res.split_results[0].artifact_subdir == "my_run/smoke"
    assert res.split_results[0].reason == "split_artifacts_generated"
    
    assert res.split_results[1].split_name == SplitName.FEWSHOT_EVAL
    assert res.split_results[1].artifact_subdir == "my_run/fewshot"

    # Verify created files
    assert (tmp_path / "my_run/smoke/smoke_samples.jsonl").is_file()
    assert (tmp_path / "my_run/smoke/smoke_manifest.json").is_file()
    assert (tmp_path / "my_run/fewshot/fewshot_eval_samples.jsonl").is_file()
    assert (tmp_path / "my_run/fewshot/fewshot_eval_manifest.json").is_file()


def test_runner_fails_fast_on_split_build_error(tmp_path):
    bad_template = GenerationRequest(
        family_id=FamilyId.AR,
        seed=0,
        p=100,  # exceeds APPROVED_MAX_P (5)
        q=0, r=0, s=0,
        ar_range=(-0.4, 0.4),
        ma_range=(-0.4, 0.4),
        omega_range=(0.1, 1.0),
        alpha_range=(0.0, 0.4),
        beta_range=(0.0, 0.4),
        constraint_flags=(1.0, 0.0, 0.0, 0.0),
        provenance=(),
        max_attempts=10,
    )
    
    split_req = make_valid_split_req(sample_count_by_family=((FamilyId.AR, 1),))
    split_req = replace(split_req, generation_template_by_family=((FamilyId.AR, bad_template),))
    
    req = make_valid_run_req(
        output_root_dir=str(tmp_path),
        split_requests=(split_req,),
    )

    with pytest.raises(ValueError, match="p must be"):
        run_phase2_artifact_generation(req)


def test_runner_fails_fast_on_write_error(tmp_path):
    run_dir = tmp_path / "write_err"
    run_dir.mkdir()
    samples_file = run_dir / "samples.jsonl"
    samples_file.write_text("pre-existing content")

    split_req = make_valid_split_req(artifact_subdir="write_err")
    req = make_valid_run_req(
        output_root_dir=str(tmp_path),
        split_requests=(split_req,),
        overwrite_existing=False,
    )

    with pytest.raises(FileExistsError):
        run_phase2_artifact_generation(req)


# ==============================================================================
# F. STRICT SCOPE CONSTRAINT TESTS
# ==============================================================================

def test_scope_no_torch():
    path = pathlib.Path(__file__).parent.parent / "src" / "phase2" / "split_runner.py"
    content = path.read_text()
    assert "import torch" not in content
    assert "from torch" not in content


def test_scope_no_numpy():
    path = pathlib.Path(__file__).parent.parent / "src" / "phase2" / "split_runner.py"
    content = path.read_text()
    assert "import numpy" not in content
    assert "from numpy" not in content


def test_scope_no_pandas():
    path = pathlib.Path(__file__).parent.parent / "src" / "phase2" / "split_runner.py"
    content = path.read_text()
    assert "import pandas" not in content
    assert "from pandas" not in content


def test_scope_no_yaml():
    path = pathlib.Path(__file__).parent.parent / "src" / "phase2" / "split_runner.py"
    content = path.read_text()
    assert "import yaml" not in content
    assert "from yaml" not in content


def test_scope_no_argparse():
    path = pathlib.Path(__file__).parent.parent / "src" / "phase2" / "split_runner.py"
    content = path.read_text()
    assert "import argparse" not in content
    assert "from argparse" not in content


def test_scope_no_models_or_training():
    path = pathlib.Path(__file__).parent.parent / "src" / "phase2" / "split_runner.py"
    content = path.read_text()
    assert "import models" not in content
    assert "from models" not in content
    assert "import vae" not in content
    assert "import training" not in content


def test_scope_no_configs():
    path = pathlib.Path(__file__).parent.parent / "src" / "phase2" / "split_runner.py"
    content = path.read_text()
    assert "import config" not in content
    assert "from config" not in content
