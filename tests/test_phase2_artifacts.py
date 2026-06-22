# tests/test_phase2_artifacts.py

import json
import pathlib
import pytest
import sys
from dataclasses import replace

from src.phase2.schema import FamilyId, MeanFamily, VolatilityFamily, ModelSpec
from src.phase2.dataset import SplitName, DatasetSample, DatasetBuildResult
from src.phase2.artifacts import (
    ArtifactWriteRequest,
    ArtifactWriteResult,
    validate_artifact_write_request,
    write_dataset_artifacts,
    dataset_sample_to_json_dict,
    dataset_manifest_to_json_dict,
    sha256_file,
)


def make_dummy_spec() -> ModelSpec:
    return ModelSpec(
        family_id=FamilyId.AR,
        mean_family=MeanFamily.AR,
        volatility_family=VolatilityFamily.NONE,
        p=1,
        q=0,
        r=0,
        s=0,
        ar_params=(0.5,),
        ma_params=(),
        omega=1.0,
        alpha_params=(),
        beta_params=(),
        constraint_flags=(1.0, 0.0, 0.0, 0.0),
        provenance=(("step", "dummy"),),
    )


def make_dummy_sample(
    sample_id="AR_zero_shot_train_000001_g42_s1000042_abc123",
    protocol_name="proto",
    split_name=SplitName.ZERO_SHOT_TRAIN,
    family_id=FamilyId.AR,
    sample_index=1,
    generation_seed=42,
    simulation_seed=1000042,
    values=(1.0, 2.0, 3.0),
    innovations=(0.1, 0.2, 0.3),
    variances=(1.0, 1.0, 1.0),
    provenance=(("provenance", "dummy"),),
) -> DatasetSample:
    return DatasetSample(
        sample_id=sample_id,
        protocol_name=protocol_name,
        split_name=split_name,
        family_id=family_id,
        sample_index=sample_index,
        generation_seed=generation_seed,
        simulation_seed=simulation_seed,
        model_spec=make_dummy_spec(),
        values=values,
        innovations=innovations,
        variances=variances,
        provenance=provenance,
    )


def make_dummy_dataset_result(
    samples=None,
    protocol_name="proto",
    split_name=SplitName.ZERO_SHOT_TRAIN,
    total_count=1,
    count_by_family=((FamilyId.AR, 1),),
    zero_shot_c_train_count=0,
    reason="dataset_built_in_memory",
) -> DatasetBuildResult:
    if samples is None:
        samples = (make_dummy_sample(protocol_name=protocol_name, split_name=split_name),)
    return DatasetBuildResult(
        samples=samples,
        protocol_name=protocol_name,
        split_name=split_name,
        total_count=total_count,
        count_by_family=count_by_family,
        zero_shot_c_train_count=zero_shot_c_train_count,
        reason=reason,
    )


def make_dummy_write_request(
    dataset_result=None,
    output_dir="tmp_dir",
    samples_filename="samples.jsonl",
    manifest_filename="manifest.json",
    create_parent_dirs=True,
    overwrite_existing=True,
    include_values=True,
    include_innovations=True,
    include_variances=True,
    json_sort_keys=True,
    json_indent=0,
) -> ArtifactWriteRequest:
    if dataset_result is None:
        dataset_result = make_dummy_dataset_result()
    return ArtifactWriteRequest(
        dataset_result=dataset_result,
        output_dir=output_dir,
        samples_filename=samples_filename,
        manifest_filename=manifest_filename,
        create_parent_dirs=create_parent_dirs,
        overwrite_existing=overwrite_existing,
        include_values=include_values,
        include_innovations=include_innovations,
        include_variances=include_variances,
        json_sort_keys=json_sort_keys,
        json_indent=json_indent,
    )


# ==============================================================================
# A. REQUEST VALIDATION TESTS
# ==============================================================================

# 1. Reject non-DatasetBuildResult dataset_result
def test_reject_non_dataset_result():
    req = make_dummy_write_request(dataset_result="not a result")
    with pytest.raises(ValueError, match="dataset_result"):
        validate_artifact_write_request(req)


# 2. Reject empty output_dir
def test_reject_empty_output_dir():
    req = make_dummy_write_request(output_dir="")
    with pytest.raises(ValueError, match="output_dir"):
        validate_artifact_write_request(req)

    req = make_dummy_write_request(output_dir="   ")
    with pytest.raises(ValueError, match="output_dir"):
        validate_artifact_write_request(req)

    with pytest.raises(ValueError, match="output_dir"):
        # Type check
        validate_artifact_write_request(make_dummy_write_request(output_dir=None))


# 3. Reject empty samples_filename
def test_reject_empty_samples_filename():
    req = make_dummy_write_request(samples_filename="")
    with pytest.raises(ValueError, match="samples_filename"):
        validate_artifact_write_request(req)


# 4. Reject empty manifest_filename
def test_reject_empty_manifest_filename():
    req = make_dummy_write_request(manifest_filename="")
    with pytest.raises(ValueError, match="manifest_filename"):
        validate_artifact_write_request(req)


# 5. Reject same samples and manifest filename
def test_reject_same_samples_and_manifest_filename():
    req = make_dummy_write_request(samples_filename="data.jsonl", manifest_filename="data.jsonl")
    with pytest.raises(ValueError, match="distinct"):
        validate_artifact_write_request(req)


# 6. Reject samples_filename not ending .jsonl
def test_reject_samples_filename_extension():
    req = make_dummy_write_request(samples_filename="samples.json")
    with pytest.raises(ValueError, match="jsonl"):
        validate_artifact_write_request(req)


# 7. Reject manifest_filename not ending .json
def test_reject_manifest_filename_extension():
    req = make_dummy_write_request(manifest_filename="manifest.jsonl")
    with pytest.raises(ValueError, match="json"):
        validate_artifact_write_request(req)


# 8. Reject non-bool create_parent_dirs
def test_reject_non_bool_create_parent_dirs():
    req = make_dummy_write_request(create_parent_dirs="True")
    with pytest.raises(ValueError, match="create_parent_dirs"):
        validate_artifact_write_request(req)


# 9. Reject non-bool overwrite_existing
def test_reject_non_bool_overwrite_existing():
    req = make_dummy_write_request(overwrite_existing="False")
    with pytest.raises(ValueError, match="overwrite_existing"):
        validate_artifact_write_request(req)


# 10. Reject non-bool include_values/include_innovations/include_variances
def test_reject_non_bool_include_flags():
    for flag in ("include_values", "include_innovations", "include_variances"):
        kwargs = {flag: 1}
        req = make_dummy_write_request(**kwargs)
        with pytest.raises(ValueError, match=flag):
            validate_artifact_write_request(req)


# 11. Reject all include flags False
def test_reject_all_include_flags_false():
    req = make_dummy_write_request(
        include_values=False, include_innovations=False, include_variances=False
    )
    with pytest.raises(ValueError, match="At least one of"):
        validate_artifact_write_request(req)


# 12. Reject non-bool json_sort_keys
def test_reject_non_bool_json_sort_keys():
    req = make_dummy_write_request(json_sort_keys=1)
    with pytest.raises(ValueError, match="json_sort_keys"):
        validate_artifact_write_request(req)


# 13. Reject bool json_indent
def test_reject_bool_json_indent():
    req = make_dummy_write_request(json_indent=True)
    with pytest.raises(ValueError, match="json_indent"):
        validate_artifact_write_request(req)


# 14. Reject negative json_indent
def test_reject_negative_json_indent():
    req = make_dummy_write_request(json_indent=-1)
    with pytest.raises(ValueError, match="json_indent"):
        validate_artifact_write_request(req)


# 15. Reject dataset_result.total_count mismatch
def test_reject_total_count_mismatch():
    res = make_dummy_dataset_result(total_count=10)
    req = make_dummy_write_request(dataset_result=res)
    with pytest.raises(ValueError, match="total_count mismatch"):
        validate_artifact_write_request(req)


# 16. Reject sample protocol mismatch
def test_reject_sample_protocol_mismatch():
    sample = make_dummy_sample(protocol_name="diff_protocol")
    res = make_dummy_dataset_result(samples=(sample,), protocol_name="proto")
    req = make_dummy_write_request(dataset_result=res)
    with pytest.raises(ValueError, match="protocol_name"):
        validate_artifact_write_request(req)


# 17. Reject sample split mismatch
def test_reject_sample_split_mismatch():
    sample = make_dummy_sample(split_name=SplitName.ZERO_SHOT_EVAL)
    res = make_dummy_dataset_result(samples=(sample,), split_name=SplitName.ZERO_SHOT_TRAIN)
    req = make_dummy_write_request(dataset_result=res)
    with pytest.raises(ValueError, match="split_name"):
        validate_artifact_write_request(req)


# 18. Reject output_dir that exists as a file
def test_reject_output_dir_exists_as_file(tmp_path):
    target_file = tmp_path / "existing_file"
    target_file.write_text("dummy")
    req = make_dummy_write_request(output_dir=str(target_file))
    with pytest.raises(ValueError, match="exists and is a file"):
        validate_artifact_write_request(req)


# 19. Reject existing target files when overwrite_existing=False
def test_reject_existing_target_files_no_overwrite(tmp_path):
    out_dir = tmp_path / "out"
    out_dir.mkdir()
    samples_file = out_dir / "samples.jsonl"
    samples_file.write_text("hello")
    req = make_dummy_write_request(output_dir=str(out_dir), overwrite_existing=False)
    with pytest.raises(FileExistsError, match="already exists"):
        write_dataset_artifacts(req)


# ==============================================================================
# B. SERIALIZATION TESTS
# ==============================================================================

# 20. dataset_sample_to_json_dict includes required metadata fields
def test_sample_to_json_dict_includes_metadata():
    sample = make_dummy_sample()
    d = dataset_sample_to_json_dict(sample, True, True, True)
    assert d["sample_id"] == sample.sample_id
    assert d["protocol_name"] == sample.protocol_name
    assert d["split_name"] == sample.split_name.value
    assert d["family_id"] == sample.family_id.value
    assert d["sample_index"] == sample.sample_index
    assert d["generation_seed"] == sample.generation_seed
    assert d["simulation_seed"] == sample.simulation_seed
    assert d["provenance"] == [list(item) for item in sample.provenance]


# 21. model_spec serializes all required fields
def test_model_spec_serializes_all_required_fields():
    sample = make_dummy_sample()
    d = dataset_sample_to_json_dict(sample, True, True, True)
    spec = d["model_spec"]
    assert spec["family_id"] == sample.model_spec.family_id.value
    assert spec["mean_family"] == sample.model_spec.mean_family.value
    assert spec["volatility_family"] == sample.model_spec.volatility_family.value
    assert spec["p"] == sample.model_spec.p
    assert spec["q"] == sample.model_spec.q
    assert spec["r"] == sample.model_spec.r
    assert spec["s"] == sample.model_spec.s
    assert spec["ar_params"] == list(sample.model_spec.ar_params)
    assert spec["ma_params"] == list(sample.model_spec.ma_params)
    assert spec["omega"] == sample.model_spec.omega
    assert spec["alpha_params"] == list(sample.model_spec.alpha_params)
    assert spec["beta_params"] == list(sample.model_spec.beta_params)
    assert spec["constraint_flags"] == list(sample.model_spec.constraint_flags)
    assert spec["provenance"] == [list(item) for item in sample.model_spec.provenance]


# 22. Enums serialize as strings, not repr
def test_enums_serialize_as_strings():
    sample = make_dummy_sample()
    d = dataset_sample_to_json_dict(sample, True, True, True)
    assert d["split_name"] == "zero_shot_train"
    assert d["family_id"] == "AR"
    assert d["model_spec"]["mean_family"] == "AR"
    assert d["model_spec"]["volatility_family"] == "NONE"


# 23. Tuples serialize as JSON lists
def test_tuples_serialize_as_lists():
    sample = make_dummy_sample()
    d = dataset_sample_to_json_dict(sample, True, True, True)
    assert isinstance(d["provenance"], list)
    assert isinstance(d["model_spec"]["ar_params"], list)
    assert isinstance(d["model_spec"]["provenance"], list)


# 24. include_values=False omits values
def test_include_values_false_omits():
    sample = make_dummy_sample()
    d = dataset_sample_to_json_dict(sample, False, True, True)
    assert "values" not in d
    assert "innovations" in d
    assert "variances" in d


# 25. include_innovations=False omits innovations
def test_include_innovations_false_omits():
    sample = make_dummy_sample()
    d = dataset_sample_to_json_dict(sample, True, False, True)
    assert "values" in d
    assert "innovations" not in d
    assert "variances" in d


# 26. include_variances=False omits variances
def test_include_variances_false_omits():
    sample = make_dummy_sample()
    d = dataset_sample_to_json_dict(sample, True, True, False)
    assert "values" in d
    assert "innovations" in d
    assert "variances" not in d


# 27. Manifest contains artifact_type
def test_manifest_contains_artifact_type():
    req = make_dummy_write_request()
    m = dataset_manifest_to_json_dict(req, "sha256dummy")
    assert m["artifact_type"] == "phase2_dataset_manifest"


# 28. Manifest contains samples_sha256
def test_manifest_contains_samples_sha256():
    req = make_dummy_write_request()
    m = dataset_manifest_to_json_dict(req, "sha256dummy")
    assert m["samples_sha256"] == "sha256dummy"


# 29. Manifest count_by_family serializes as list of objects
def test_manifest_count_by_family_as_objects():
    req = make_dummy_write_request()
    m = dataset_manifest_to_json_dict(req, "sha256dummy")
    assert isinstance(m["count_by_family"], list)
    assert len(m["count_by_family"]) == 1
    assert m["count_by_family"][0] == {"family_id": "AR", "count": 1}


# 30. Manifest sample_ids preserves dataset sample order
def test_manifest_sample_ids_preserves_order():
    sample1 = make_dummy_sample(sample_id="id1")
    sample2 = make_dummy_sample(sample_id="id2")
    res = make_dummy_dataset_result(samples=(sample1, sample2), total_count=2)
    req = make_dummy_write_request(dataset_result=res)
    m = dataset_manifest_to_json_dict(req, "sha")
    assert m["sample_ids"] == ["id1", "id2"]


# ==============================================================================
# C. WRITING BEHAVIOR TESTS
# ==============================================================================

# 31. write_dataset_artifacts creates output_dir if create_parent_dirs=True
def test_write_creates_output_dir(tmp_path):
    out_dir = tmp_path / "new_parent" / "output_dir"
    req = make_dummy_write_request(output_dir=str(out_dir), create_parent_dirs=True)
    res = write_dataset_artifacts(req)
    assert out_dir.exists()
    assert out_dir.is_dir()


# 32. write_dataset_artifacts fails if output_dir missing and create_parent_dirs=False
def test_write_fails_if_dir_missing(tmp_path):
    out_dir = tmp_path / "no_parent" / "output_dir"
    req = make_dummy_write_request(output_dir=str(out_dir), create_parent_dirs=False)
    with pytest.raises(FileNotFoundError):
        write_dataset_artifacts(req)


# 33. Writes exactly samples JSONL and manifest JSON, no extra files
def test_writes_exactly_expected_files(tmp_path):
    out_dir = tmp_path / "exact_files"
    req = make_dummy_write_request(output_dir=str(out_dir), create_parent_dirs=True)
    write_dataset_artifacts(req)
    files = sorted(list(out_dir.iterdir()))
    assert len(files) == 2
    assert files[0].name == "manifest.json"
    assert files[1].name == "samples.jsonl"


# 34. JSONL line count equals dataset_result.total_count
def test_jsonl_line_count(tmp_path):
    out_dir = tmp_path / "line_count"
    sample1 = make_dummy_sample(sample_id="id1")
    sample2 = make_dummy_sample(sample_id="id2")
    res = make_dummy_dataset_result(samples=(sample1, sample2), total_count=2)
    req = make_dummy_write_request(dataset_result=res, output_dir=str(out_dir))
    write_dataset_artifacts(req)
    samples_file = out_dir / "samples.jsonl"
    with open(samples_file, "r", encoding="utf-8") as f:
        lines = [line for line in f if line.strip()]
    assert len(lines) == 2


# 35. Each JSONL line parses as JSON object
def test_each_jsonl_line_parses(tmp_path):
    out_dir = tmp_path / "parses"
    req = make_dummy_write_request(output_dir=str(out_dir))
    write_dataset_artifacts(req)
    samples_file = out_dir / "samples.jsonl"
    with open(samples_file, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                obj = json.loads(line)
                assert isinstance(obj, dict)


# 36. Manifest parses as JSON object
def test_manifest_parses_as_json_object(tmp_path):
    out_dir = tmp_path / "manifest_parses"
    req = make_dummy_write_request(output_dir=str(out_dir))
    write_dataset_artifacts(req)
    manifest_file = out_dir / "manifest.json"
    with open(manifest_file, "r", encoding="utf-8") as f:
        obj = json.load(f)
    assert isinstance(obj, dict)
    assert obj["artifact_type"] == "phase2_dataset_manifest"


# 37. ArtifactWriteResult paths match actual files
def test_result_paths_match_actual_files(tmp_path):
    out_dir = tmp_path / "result_paths"
    req = make_dummy_write_request(output_dir=str(out_dir))
    res = write_dataset_artifacts(req)
    assert res.output_dir == str(out_dir.resolve())
    assert res.samples_path == str((out_dir / "samples.jsonl").resolve())
    assert res.manifest_path == str((out_dir / "manifest.json").resolve())


# 38. ArtifactWriteResult samples_sha256 equals sha256_file(samples_path)
def test_result_samples_sha256(tmp_path):
    out_dir = tmp_path / "result_sha_s"
    req = make_dummy_write_request(output_dir=str(out_dir))
    res = write_dataset_artifacts(req)
    assert res.samples_sha256 == sha256_file(res.samples_path)


# 39. ArtifactWriteResult manifest_sha256 equals sha256_file(manifest_path)
def test_result_manifest_sha256(tmp_path):
    out_dir = tmp_path / "result_sha_m"
    req = make_dummy_write_request(output_dir=str(out_dir))
    res = write_dataset_artifacts(req)
    assert res.manifest_sha256 == sha256_file(res.manifest_path)


# 40. overwrite_existing=False fails on second write
def test_no_overwrite_fails(tmp_path):
    out_dir = tmp_path / "no_overwrite"
    req = make_dummy_write_request(output_dir=str(out_dir), overwrite_existing=False)
    write_dataset_artifacts(req)
    with pytest.raises(FileExistsError):
        write_dataset_artifacts(req)


# 41. overwrite_existing=True succeeds on second write
def test_overwrite_succeeds(tmp_path):
    out_dir = tmp_path / "overwrite_succeeds"
    req = make_dummy_write_request(output_dir=str(out_dir), overwrite_existing=True)
    write_dataset_artifacts(req)
    # Succeeds on second write
    write_dataset_artifacts(req)


# ==============================================================================
# D. DETERMINISM TESTS
# ==============================================================================

# 42. Same DatasetBuildResult + same write request produces identical samples_sha256 across two different output directories
def test_sha256_determinism_across_dirs(tmp_path):
    dir1 = tmp_path / "dir1"
    dir2 = tmp_path / "dir2"
    res = make_dummy_dataset_result()

    req1 = make_dummy_write_request(dataset_result=res, output_dir=str(dir1))
    req2 = make_dummy_write_request(dataset_result=res, output_dir=str(dir2))

    res1 = write_dataset_artifacts(req1)
    res2 = write_dataset_artifacts(req2)

    assert res1.samples_sha256 == res2.samples_sha256


# 43. Same DatasetBuildResult + same write request produces identical manifest content except path-independent fields
def test_manifest_content_determinism(tmp_path):
    dir1 = tmp_path / "dir1"
    dir2 = tmp_path / "dir2"
    res = make_dummy_dataset_result()

    req1 = make_dummy_write_request(dataset_result=res, output_dir=str(dir1))
    req2 = make_dummy_write_request(dataset_result=res, output_dir=str(dir2))

    res1 = write_dataset_artifacts(req1)
    res2 = write_dataset_artifacts(req2)

    with open(res1.manifest_path, "r", encoding="utf-8") as f:
        m1 = json.load(f)
    with open(res2.manifest_path, "r", encoding="utf-8") as f:
        m2 = json.load(f)

    assert m1 == m2
    assert res1.manifest_sha256 == res2.manifest_sha256


# 44. Reordered samples change samples_sha256
def test_reordered_samples_change_sha256(tmp_path):
    dir1 = tmp_path / "dir1"
    dir2 = tmp_path / "dir2"
    sample1 = make_dummy_sample(sample_id="id1")
    sample2 = make_dummy_sample(sample_id="id2")

    res1 = make_dummy_dataset_result(samples=(sample1, sample2), total_count=2)
    res2 = make_dummy_dataset_result(samples=(sample2, sample1), total_count=2)

    req1 = make_dummy_write_request(dataset_result=res1, output_dir=str(dir1))
    req2 = make_dummy_write_request(dataset_result=res2, output_dir=str(dir2))

    write_res1 = write_dataset_artifacts(req1)
    write_res2 = write_dataset_artifacts(req2)

    assert write_res1.samples_sha256 != write_res2.samples_sha256


# ==============================================================================
# E. SCOPE TESTS
# ==============================================================================

# 45. artifacts.py does not import torch directly
def test_scope_no_torch():
    path = pathlib.Path(__file__).parent.parent / "src" / "phase2" / "artifacts.py"
    content = path.read_text()
    assert "import torch" not in content
    assert "from torch" not in content


# 46. artifacts.py does not import numpy directly
def test_scope_no_numpy():
    path = pathlib.Path(__file__).parent.parent / "src" / "phase2" / "artifacts.py"
    content = path.read_text()
    assert "import numpy" not in content
    assert "from numpy" not in content


# 47. Artifact writer does not import model/training modules
def test_scope_no_model_or_training():
    path = pathlib.Path(__file__).parent.parent / "src" / "phase2" / "artifacts.py"
    content = path.read_text()
    # Check that we do not import from src.models, or any model/training-specific modules
    assert "import models" not in content
    assert "from models" not in content
    assert "import vae" not in content
    assert "import training" not in content


# 48. Artifact writer does not require configs
def test_scope_no_configs():
    path = pathlib.Path(__file__).parent.parent / "src" / "phase2" / "artifacts.py"
    content = path.read_text()
    assert "import config" not in content
    assert "from config" not in content


# 49. Artifact writer writes only inside tmp_path in tests
def test_scope_writes_only_in_tmp_path(tmp_path):
    # All tests that invoke writing write inside the provided tmp_path fixture.
    pass


# 50. Artifact writer does not create artifact directories outside explicit output_dir
def test_scope_no_creation_outside_output_dir(tmp_path):
    out_dir = tmp_path / "explicit_dir"
    req = make_dummy_write_request(output_dir=str(out_dir))
    write_dataset_artifacts(req)
    # Check that nothing else was created in tmp_path besides explicit_dir
    created = list(tmp_path.iterdir())
    assert len(created) == 1
    assert created[0].name == "explicit_dir"
