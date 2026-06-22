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
    samples_filename="samples.jsonl",
    manifest_filename="manifest.json",
) -> SplitArtifactRequest:
    if generation_template_by_family is None:
        try:
            generation_template_by_family = tuple(
                (fid, get_base_gen_req(fid)) for fid, _ in sample_count_by_family
            )
        except Exception:
            generation_template_by_family = ()
    return SplitArtifactRequest(
        split_name=split_name,
        sample_count_by_family=sample_count_by_family,
        base_seed_by_family=base_seed_by_family,
        generation_template_by_family=generation_template_by_family,
        samples_filename=samples_filename,
        manifest_filename=manifest_filename,
    )


def make_valid_run_req(
    protocol_name="proto-1",
    artifact_dir="artifacts",
    artifact_subdir="run_1",
    splits=None,
    simulation_template=None,
    overwrite_existing=True,
    create_parent_dirs=True,
    include_values=True,
    include_innovations=True,
    include_variances=True,
    json_sort_keys=True,
    json_indent=0,
    enforce_zero_shot_c_train_exclusion=True,
    sample_id_hash_len=8,
) -> Phase2ArtifactRunRequest:
    if splits is None:
        splits = (make_valid_split_req(),)
    if simulation_template is None:
        simulation_template = get_valid_sim_template()
    return Phase2ArtifactRunRequest(
        protocol_name=protocol_name,
        artifact_dir=artifact_dir,
        artifact_subdir=artifact_subdir,
        splits=splits,
        simulation_template=simulation_template,
        overwrite_existing=overwrite_existing,
        create_parent_dirs=create_parent_dirs,
        include_values=include_values,
        include_innovations=include_innovations,
        include_variances=include_variances,
        json_sort_keys=json_sort_keys,
        json_indent=json_indent,
        enforce_zero_shot_c_train_exclusion=enforce_zero_shot_c_train_exclusion,
        sample_id_hash_len=sample_id_hash_len,
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


def test_run_req_reject_invalid_artifact_dir():
    req = make_valid_run_req(artifact_dir="")
    with pytest.raises(ValueError, match="artifact_dir"):
        validate_phase2_artifact_run_request(req)


def test_run_req_reject_non_str_artifact_subdir():
    req = make_valid_run_req(artifact_subdir=123)
    with pytest.raises(ValueError, match="artifact_subdir"):
        validate_phase2_artifact_run_request(req)


def test_run_req_reject_empty_artifact_subdir():
    req = make_valid_run_req(artifact_subdir="   ")
    with pytest.raises(ValueError, match="artifact_subdir"):
        validate_phase2_artifact_run_request(req)


def test_run_req_reject_absolute_artifact_subdir():
    req = make_valid_run_req(artifact_subdir="/abs")
    with pytest.raises(ValueError, match="absolute"):
        validate_phase2_artifact_run_request(req)

    req = make_valid_run_req(artifact_subdir="C:\\abs")
    with pytest.raises(ValueError, match="absolute"):
        validate_phase2_artifact_run_request(req)


def test_run_req_reject_traversal_artifact_subdir():
    req = make_valid_run_req(artifact_subdir="run/../dir")
    with pytest.raises(ValueError, match="path traversal"):
        validate_phase2_artifact_run_request(req)

    req = make_valid_run_req(artifact_subdir="..")
    with pytest.raises(ValueError, match="path traversal"):
        validate_phase2_artifact_run_request(req)


def test_run_req_reject_non_tuple_splits():
    req = make_valid_run_req(splits="not_tuple")
    with pytest.raises(ValueError, match="splits"):
        validate_phase2_artifact_run_request(req)


def test_run_req_reject_empty_splits():
    req = make_valid_run_req(splits=())
    with pytest.raises(ValueError, match="splits tuple must not be empty"):
        validate_phase2_artifact_run_request(req)


def test_run_req_reject_invalid_split_item():
    req = make_valid_run_req(splits=("not_split_req",))
    with pytest.raises(ValueError, match="SplitArtifactRequest instance"):
        validate_phase2_artifact_run_request(req)


def test_run_req_reject_duplicate_splits():
    req = make_valid_run_req(splits=(make_valid_split_req(SplitName.SMOKE), make_valid_split_req(SplitName.SMOKE)))
    with pytest.raises(ValueError, match="Duplicate split_name"):
        validate_phase2_artifact_run_request(req)


def test_run_req_reject_invalid_simulation_template():
    req = make_valid_run_req(simulation_template="not_sim_template")
    with pytest.raises(ValueError, match="simulation_template"):
        validate_phase2_artifact_run_request(req)


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
    with pytest.raises(ValueError, match="At least one of"):
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


def test_run_req_reject_non_bool_leakage_exclusion():
    req = make_valid_run_req(enforce_zero_shot_c_train_exclusion="True")
    with pytest.raises(ValueError, match="enforce_zero_shot_c_train_exclusion"):
        validate_phase2_artifact_run_request(req)


def test_run_req_reject_invalid_hash_len():
    req = make_valid_run_req(sample_id_hash_len=7)
    with pytest.raises(ValueError, match="sample_id_hash_len"):
        validate_phase2_artifact_run_request(req)

    req = make_valid_run_req(sample_id_hash_len=True)
    with pytest.raises(ValueError, match="sample_id_hash_len"):
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
    )
    req = make_valid_run_req(splits=(split_req,), enforce_zero_shot_c_train_exclusion=True)
    with pytest.raises(ValueError, match="C leakage detected"):
        validate_phase2_artifact_run_request(req)


def test_leakage_guard_allows_garch_zero_count_in_zero_shot_train():
    split_req = make_valid_split_req(
        split_name=SplitName.ZERO_SHOT_TRAIN,
        sample_count_by_family=((FamilyId.ARMA_GARCH, 0), (FamilyId.AR, 5)),
        base_seed_by_family=((FamilyId.ARMA_GARCH, 12003), (FamilyId.AR, 12001)),
        generation_template_by_family=(
            (FamilyId.ARMA_GARCH, get_base_gen_req(FamilyId.ARMA_GARCH)),
            (FamilyId.AR, get_base_gen_req(FamilyId.AR)),
        ),
    )
    req = make_valid_run_req(splits=(split_req,), enforce_zero_shot_c_train_exclusion=True)
    validate_phase2_artifact_run_request(req)  # Should pass


def test_leakage_guard_allows_garch_in_zero_shot_train_if_exclusion_disabled():
    split_req = make_valid_split_req(
        split_name=SplitName.ZERO_SHOT_TRAIN,
        sample_count_by_family=((FamilyId.ARMA_GARCH, 5),),
        base_seed_by_family=((FamilyId.ARMA_GARCH, 12003),),
    )
    req = make_valid_run_req(splits=(split_req,), enforce_zero_shot_c_train_exclusion=False)
    validate_phase2_artifact_run_request(req)  # Should pass


def test_leakage_guard_allows_garch_in_other_splits():
    for other_split in (SplitName.ZERO_SHOT_EVAL, SplitName.FEWSHOT_TRAIN, SplitName.FEWSHOT_EVAL, SplitName.SMOKE):
        split_req = make_valid_split_req(
            split_name=other_split,
            sample_count_by_family=((FamilyId.ARMA_GARCH, 5),),
            base_seed_by_family=((FamilyId.ARMA_GARCH, 12003),),
        )
        req = make_valid_run_req(splits=(split_req,), enforce_zero_shot_c_train_exclusion=True)
        validate_phase2_artifact_run_request(req)  # Should pass


# ==============================================================================
# D. BUILDER HELPER MAPPING TESTS
# ==============================================================================

def test_build_dataset_request_mapping():
    run_req = make_valid_run_req(protocol_name="mapped-protocol", sample_id_hash_len=12)
    split_req = make_valid_split_req(split_name=SplitName.FEWSHOT_EVAL)
    dataset_req = build_dataset_request_for_split(run_req, split_req)

    assert dataset_req.protocol_name == "mapped-protocol"
    assert dataset_req.split_name == SplitName.FEWSHOT_EVAL
    assert dataset_req.sample_count_by_family == split_req.sample_count_by_family
    assert dataset_req.generation_template_by_family == split_req.generation_template_by_family
    assert dataset_req.simulation_template == run_req.simulation_template
    assert dataset_req.base_seed_by_family == split_req.base_seed_by_family
    assert dataset_req.enforce_zero_shot_c_train_exclusion == run_req.enforce_zero_shot_c_train_exclusion
    assert dataset_req.sample_id_hash_len == 12


def test_build_artifact_write_request_mapping():
    run_req = make_valid_run_req(
        artifact_dir="my_artifacts",
        artifact_subdir="run_dir",
        include_values=False,
        overwrite_existing=False,
    )
    split_req = make_valid_split_req(
        samples_filename="custom_samples.jsonl",
        manifest_filename="custom_manifest.json",
    )
    
    # Dummy DatasetBuildResult
    dataset_res = DatasetBuildResult(
        samples=(),
        protocol_name="mapped-protocol",
        split_name=SplitName.FEWSHOT_EVAL,
        total_count=0,
        count_by_family=(),
        zero_shot_c_train_count=0,
        reason="dataset_built_in_memory",
    )

    write_req = build_artifact_write_request_for_split(run_req, split_req, dataset_res)

    expected_output_dir = str(pathlib.Path("my_artifacts") / "run_dir")
    assert write_req.dataset_result == dataset_res
    assert write_req.output_dir == expected_output_dir
    assert write_req.samples_filename == "custom_samples.jsonl"
    assert write_req.manifest_filename == "custom_manifest.json"
    assert write_req.create_parent_dirs == run_req.create_parent_dirs
    assert write_req.overwrite_existing is False
    assert write_req.include_values is False
    assert write_req.include_innovations == run_req.include_innovations
    assert write_req.include_variances == run_req.include_variances
    assert write_req.json_sort_keys == run_req.json_sort_keys
    assert write_req.json_indent == run_req.json_indent


# ==============================================================================
# E. END-TO-END RUNNER BEHAVIOR TESTS
# ==============================================================================

def test_runner_fails_fast_on_invalid_run_req():
    # protocol_name is empty
    req = make_valid_run_req(protocol_name="")
    with pytest.raises(ValueError, match="protocol_name"):
        run_phase2_artifact_generation(req)


def test_runner_fails_fast_on_invalid_split_req():
    # zero count
    split_req = make_valid_split_req(sample_count_by_family=((FamilyId.AR, 0),))
    req = make_valid_run_req(splits=(split_req,))
    with pytest.raises(ValueError, match="Total sample count"):
        run_phase2_artifact_generation(req)


def test_runner_executes_successfully_and_preserves_order(tmp_path):
    run_dir = tmp_path / "my_run"
    split1 = make_valid_split_req(
        split_name=SplitName.SMOKE,
        samples_filename="smoke_samples.jsonl",
        manifest_filename="smoke_manifest.json",
    )
    split2 = make_valid_split_req(
        split_name=SplitName.FEWSHOT_EVAL,
        samples_filename="fewshot_eval_samples.jsonl",
        manifest_filename="fewshot_eval_manifest.json",
    )

    req = make_valid_run_req(
        artifact_dir=str(tmp_path),
        artifact_subdir="my_run",
        splits=(split1, split2),
    )

    res = run_phase2_artifact_generation(req)

    assert res.status == "success"
    assert res.protocol_name == "proto-1"
    assert len(res.split_results) == 2

    # Check order preservation
    assert res.split_results[0].split_name == SplitName.SMOKE
    assert res.split_results[1].split_name == SplitName.FEWSHOT_EVAL

    # Verify files created
    assert (run_dir / "smoke_samples.jsonl").is_file()
    assert (run_dir / "smoke_manifest.json").is_file()
    assert (run_dir / "fewshot_eval_samples.jsonl").is_file()
    assert (run_dir / "fewshot_eval_manifest.json").is_file()

    # Verify sha256 output
    assert len(res.split_results[0].samples_sha256) == 64
    assert len(res.split_results[0].manifest_sha256) == 64


def test_runner_fails_fast_on_split_build_error(tmp_path):
    # Construct a request that will fail validation or stationarity
    # E.g. make generation templates that fail stationarity or use invalid bounds.
    # To cause a build failure, we can provide an invalid parameter length or order in the template,
    # which will cause generation/validation to raise ValueError inside build_dataset_in_memory.
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
    
    # We bypass validate_split_artifact_request by setting bad_template.
    # But validate_generation_request (called in dataset request validation) will fail.
    split_req = make_valid_split_req(sample_count_by_family=((FamilyId.AR, 1),))
    split_req = replace(split_req, generation_template_by_family=((FamilyId.AR, bad_template),))
    
    req = make_valid_run_req(
        artifact_dir=str(tmp_path),
        artifact_subdir="fail_run",
        splits=(split_req,),
    )

    with pytest.raises(ValueError, match="p must be"):
        run_phase2_artifact_generation(req)


def test_runner_fails_fast_on_write_error(tmp_path):
    run_dir = tmp_path / "write_err"
    run_dir.mkdir()
    
    # Pre-create the file to force FileExistsError when overwrite_existing=False
    samples_file = run_dir / "samples.jsonl"
    samples_file.write_text("already here")

    split_req = make_valid_split_req()
    req = make_valid_run_req(
        artifact_dir=str(tmp_path),
        artifact_subdir="write_err",
        splits=(split_req,),
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
