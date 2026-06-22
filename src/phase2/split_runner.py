# src/phase2/split_runner.py

import pathlib
from dataclasses import dataclass
from src.phase2.schema import FamilyId
from src.phase2.sampler import GenerationRequest
from src.phase2.simulator import SimulationRequest, validate_simulation_request
from src.phase2.dataset import (
    SplitName,
    DatasetBuildRequest,
    DatasetBuildResult,
    build_dataset_in_memory,
)
from src.phase2.artifacts import (
    ArtifactWriteRequest,
    ArtifactWriteResult,
    write_dataset_artifacts,
)


@dataclass(frozen=True)
class SplitArtifactRequest:
    split_name: SplitName
    sample_count_by_family: tuple[tuple[FamilyId, int], ...]
    base_seed_by_family: tuple[tuple[FamilyId, int], ...]
    generation_template_by_family: tuple[tuple[FamilyId, GenerationRequest], ...]
    simulation_template: SimulationRequest
    artifact_subdir: str
    samples_filename: str
    manifest_filename: str
    enforce_zero_shot_c_train_exclusion: bool
    sample_id_hash_len: int


@dataclass(frozen=True)
class Phase2ArtifactRunRequest:
    protocol_name: str
    output_root_dir: str
    split_requests: tuple[SplitArtifactRequest, ...]
    create_parent_dirs: bool
    overwrite_existing: bool
    include_values: bool
    include_innovations: bool
    include_variances: bool
    json_sort_keys: bool
    json_indent: int


@dataclass(frozen=True)
class SplitArtifactRunResult:
    split_name: SplitName
    artifact_subdir: str
    dataset_total_count: int
    count_by_family: tuple[tuple[FamilyId, int], ...]
    zero_shot_c_train_count: int
    samples_path: str
    manifest_path: str
    samples_sha256: str
    manifest_sha256: str
    sample_ids: tuple[str, ...]
    reason: str


@dataclass(frozen=True)
class Phase2ArtifactRunResult:
    protocol_name: str
    output_root_dir: str
    split_results: tuple[SplitArtifactRunResult, ...]
    total_sample_count: int
    total_split_count: int
    reason: str


def validate_split_artifact_request(request: SplitArtifactRequest) -> None:
    if not isinstance(request, SplitArtifactRequest):
        raise ValueError("request must be a SplitArtifactRequest instance")
    if not isinstance(request.split_name, SplitName):
        raise ValueError("split_name must be a SplitName enum instance")

    # Validate collections are exactly tuples
    if type(request.sample_count_by_family) is not tuple:
        raise ValueError("sample_count_by_family must be exactly a tuple")
    if type(request.base_seed_by_family) is not tuple:
        raise ValueError("base_seed_by_family must be exactly a tuple")
    if type(request.generation_template_by_family) is not tuple:
        raise ValueError("generation_template_by_family must be exactly a tuple")

    # Validate simulation_template
    if not isinstance(request.simulation_template, SimulationRequest):
        raise ValueError("simulation_template must be a SimulationRequest instance")

    # Validate artifact_subdir
    if not isinstance(request.artifact_subdir, str):
        raise ValueError("artifact_subdir must be a string")
    sub_stripped = request.artifact_subdir.strip()
    if len(sub_stripped) == 0:
        raise ValueError("artifact_subdir must be non-empty and not just whitespace")
    
    sub_path = pathlib.Path(request.artifact_subdir)
    if sub_path.is_absolute():
        raise ValueError("artifact_subdir must not be an absolute path")
    if request.artifact_subdir.startswith("/") or request.artifact_subdir.startswith("\\") or ":" in request.artifact_subdir:
        raise ValueError("artifact_subdir must not be an absolute path")
    if any(p == ".." for p in sub_path.parts) or ".." in request.artifact_subdir.replace("\\", "/").split("/"):
        raise ValueError("artifact_subdir must not contain path traversal using '..'")

    # Validate filenames
    if not isinstance(request.samples_filename, str) or len(request.samples_filename.strip()) == 0:
        raise ValueError("samples_filename must be a non-empty string")
    if not isinstance(request.manifest_filename, str) or len(request.manifest_filename.strip()) == 0:
        raise ValueError("manifest_filename must be a non-empty string")
    if request.samples_filename == request.manifest_filename:
        raise ValueError("samples_filename and manifest_filename must be distinct")
    if not request.samples_filename.endswith(".jsonl"):
        raise ValueError("samples_filename must end with '.jsonl'")
    if not request.manifest_filename.endswith(".json"):
        raise ValueError("manifest_filename must end with '.json'")

    # Validate exclusion flag and hash len
    if not isinstance(request.enforce_zero_shot_c_train_exclusion, bool):
        raise ValueError("enforce_zero_shot_c_train_exclusion must be a bool")
    if type(request.sample_id_hash_len) is not int or isinstance(request.sample_id_hash_len, bool) or request.sample_id_hash_len < 8:
        raise ValueError("sample_id_hash_len must be an int >= 8")

    # Validate duplicates in counts
    seen_counts = set()
    for item in request.sample_count_by_family:
        if type(item) is not tuple or len(item) != 2:
            raise ValueError("Each element in sample_count_by_family must be a 2-tuple")
        fid, count = item
        if not isinstance(fid, FamilyId):
            raise ValueError("family_id in sample_count_by_family must be a FamilyId instance")
        if type(count) is not int or isinstance(count, bool) or count < 0:
            raise ValueError("count in sample_count_by_family must be a non-negative int")
        if fid in seen_counts:
            raise ValueError(f"Duplicate family {fid} in sample_count_by_family")
        seen_counts.add(fid)

    # Validate duplicates in seeds
    seen_seeds = set()
    for item in request.base_seed_by_family:
        if type(item) is not tuple or len(item) != 2:
            raise ValueError("Each element in base_seed_by_family must be a 2-tuple")
        fid, seed = item
        if not isinstance(fid, FamilyId):
            raise ValueError("family_id in base_seed_by_family must be a FamilyId instance")
        if type(seed) is not int or isinstance(seed, bool):
            raise ValueError("seed in base_seed_by_family must be an int")
        if fid in seen_seeds:
            raise ValueError(f"Duplicate family {fid} in base_seed_by_family")
        seen_seeds.add(fid)

    # Validate duplicates in templates
    seen_gens = set()
    for item in request.generation_template_by_family:
        if type(item) is not tuple or len(item) != 2:
            raise ValueError("Each element in generation_template_by_family must be a 2-tuple")
        fid, template = item
        if not isinstance(fid, FamilyId):
            raise ValueError("family_id in generation_template_by_family must be a FamilyId instance")
        if not isinstance(template, GenerationRequest):
            raise ValueError("template in generation_template_by_family must be a GenerationRequest")
        if template.family_id != fid:
            raise ValueError(f"GenerationRequest family_id {template.family_id} does not match key {fid}")
        if fid in seen_gens:
            raise ValueError(f"Duplicate family {fid} in generation_template_by_family")
        from src.phase2.sampler import validate_generation_request
        validate_generation_request(template)
        seen_gens.add(fid)

    # Validate simulation_template itself
    validate_simulation_request(request.simulation_template)

    # Active families checks
    for fid, count in request.sample_count_by_family:
        if count > 0:
            if fid not in seen_gens:
                raise ValueError(f"Missing generation template for active family {fid}")
            if fid not in seen_seeds:
                raise ValueError(f"Missing base seed for active family {fid}")

    # Total sample count check
    total_count = sum(c for _, c in request.sample_count_by_family)
    if total_count <= 0:
        raise ValueError("Total sample count must be > 0")

    # Zero-shot C leakage guard
    if request.split_name == SplitName.ZERO_SHOT_TRAIN and request.enforce_zero_shot_c_train_exclusion:
        arma_garch_count = 0
        for fid, count in request.sample_count_by_family:
            if fid == FamilyId.ARMA_GARCH:
                arma_garch_count = count
        if arma_garch_count > 0:
            raise ValueError(
                f"C leakage detected in split request: ARMA_GARCH count is {arma_garch_count} (> 0), "
                f"which is strictly prohibited in split zero_shot_train."
            )


def validate_phase2_artifact_run_request(request: Phase2ArtifactRunRequest) -> None:
    if not isinstance(request, Phase2ArtifactRunRequest):
        raise ValueError("request must be a Phase2ArtifactRunRequest instance")
    if not isinstance(request.protocol_name, str) or len(request.protocol_name.strip()) == 0:
        raise ValueError("protocol_name must be a non-empty string")
    if not isinstance(request.output_root_dir, str) or len(request.output_root_dir.strip()) == 0:
        raise ValueError("output_root_dir must be a non-empty string")

    # Validate split_requests
    if type(request.split_requests) is not tuple:
        raise ValueError("split_requests must be exactly a tuple")
    if len(request.split_requests) == 0:
        raise ValueError("split_requests tuple must not be empty")

    # Validate write options
    if not isinstance(request.create_parent_dirs, bool):
        raise ValueError("create_parent_dirs must be a bool")
    if not isinstance(request.overwrite_existing, bool):
        raise ValueError("overwrite_existing must be a bool")

    if not isinstance(request.include_values, bool):
        raise ValueError("include_values must be a bool")
    if not isinstance(request.include_innovations, bool):
        raise ValueError("include_innovations must be a bool")
    if not isinstance(request.include_variances, bool):
        raise ValueError("include_variances must be a bool")
    if not (request.include_values or request.include_innovations or request.include_variances):
        raise ValueError("At least one include flag must be True")

    if not isinstance(request.json_sort_keys, bool):
        raise ValueError("json_sort_keys must be a bool")
    if type(request.json_indent) is not int or isinstance(request.json_indent, bool) or request.json_indent < 0:
        raise ValueError("json_indent must be a non-negative int")

    # Validate output_root_dir exists and is a file
    p_root = pathlib.Path(request.output_root_dir)
    if p_root.exists() and p_root.is_file():
        raise ValueError(f"output_root_dir '{request.output_root_dir}' exists and is a file")

    seen_subdirs = set()
    seen_split_subdir_pairs = set()
    seen_output_targets = set()

    for split_req in request.split_requests:
        # Every split request must pass validate_split_artifact_request
        validate_split_artifact_request(split_req)

        # Reject duplicate artifact_subdir
        if split_req.artifact_subdir in seen_subdirs:
            raise ValueError(f"Duplicate artifact_subdir '{split_req.artifact_subdir}' detected.")
        seen_subdirs.add(split_req.artifact_subdir)

        # Reject duplicate split_name + artifact_subdir pair
        pair = (split_req.split_name, split_req.artifact_subdir)
        if pair in seen_split_subdir_pairs:
            raise ValueError(f"Duplicate split_name and artifact_subdir pair '{pair}' detected.")
        seen_split_subdir_pairs.add(pair)

        # Reject duplicate output targets under output_root_dir
        samples_target = (split_req.artifact_subdir, split_req.samples_filename)
        manifest_target = (split_req.artifact_subdir, split_req.manifest_filename)

        if samples_target in seen_output_targets:
            raise ValueError(f"Duplicate output target samples path '{samples_target}' detected.")
        seen_output_targets.add(samples_target)

        if manifest_target in seen_output_targets:
            raise ValueError(f"Duplicate output target manifest path '{manifest_target}' detected.")
        seen_output_targets.add(manifest_target)


def build_dataset_request_for_split(
    protocol_name: str,
    split_request: SplitArtifactRequest,
) -> DatasetBuildRequest:
    return DatasetBuildRequest(
        protocol_name=protocol_name,
        split_name=split_request.split_name,
        sample_count_by_family=split_request.sample_count_by_family,
        generation_template_by_family=split_request.generation_template_by_family,
        simulation_template=split_request.simulation_template,
        base_seed_by_family=split_request.base_seed_by_family,
        enforce_zero_shot_c_train_exclusion=split_request.enforce_zero_shot_c_train_exclusion,
        sample_id_hash_len=split_request.sample_id_hash_len,
    )


def build_artifact_write_request_for_split(
    dataset_result: DatasetBuildResult,
    output_root_dir: str,
    split_request: SplitArtifactRequest,
    create_parent_dirs: bool,
    overwrite_existing: bool,
    include_values: bool,
    include_innovations: bool,
    include_variances: bool,
    json_sort_keys: bool,
    json_indent: int,
) -> ArtifactWriteRequest:
    output_dir = str(pathlib.Path(output_root_dir) / split_request.artifact_subdir)
    return ArtifactWriteRequest(
        dataset_result=dataset_result,
        output_dir=output_dir,
        samples_filename=split_request.samples_filename,
        manifest_filename=split_request.manifest_filename,
        create_parent_dirs=create_parent_dirs,
        overwrite_existing=overwrite_existing,
        include_values=include_values,
        include_innovations=include_innovations,
        include_variances=include_variances,
        json_sort_keys=json_sort_keys,
        json_indent=json_indent,
    )


def run_phase2_artifact_generation(request: Phase2ArtifactRunRequest) -> Phase2ArtifactRunResult:
    # Validate the run request before executing any split
    validate_phase2_artifact_run_request(request)

    split_results = []
    for split_req in request.split_requests:
        validate_split_artifact_request(split_req)

        build_req = build_dataset_request_for_split(request.protocol_name, split_req)
        build_res = build_dataset_in_memory(build_req)

        write_req = build_artifact_write_request_for_split(
            dataset_result=build_res,
            output_root_dir=request.output_root_dir,
            split_request=split_req,
            create_parent_dirs=request.create_parent_dirs,
            overwrite_existing=request.overwrite_existing,
            include_values=request.include_values,
            include_innovations=request.include_innovations,
            include_variances=request.include_variances,
            json_sort_keys=request.json_sort_keys,
            json_indent=request.json_indent,
        )
        write_res = write_dataset_artifacts(write_req)

        sample_ids = tuple(s.sample_id for s in build_res.samples)
        split_result = SplitArtifactRunResult(
            split_name=split_req.split_name,
            artifact_subdir=split_req.artifact_subdir,
            dataset_total_count=build_res.total_count,
            count_by_family=build_res.count_by_family,
            zero_shot_c_train_count=build_res.zero_shot_c_train_count,
            samples_path=write_res.samples_path,
            manifest_path=write_res.manifest_path,
            samples_sha256=write_res.samples_sha256,
            manifest_sha256=write_res.manifest_sha256,
            sample_ids=sample_ids,
            reason="split_artifacts_generated",
        )
        split_results.append(split_result)

    total_sample_count = sum(r.dataset_total_count for r in split_results)
    return Phase2ArtifactRunResult(
        protocol_name=request.protocol_name,
        output_root_dir=request.output_root_dir,
        split_results=tuple(split_results),
        total_sample_count=total_sample_count,
        total_split_count=len(split_results),
        reason="phase2_artifact_generation_complete",
    )
