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
    samples_filename: str
    manifest_filename: str


@dataclass(frozen=True)
class Phase2ArtifactRunRequest:
    protocol_name: str
    artifact_dir: str
    artifact_subdir: str
    splits: tuple[SplitArtifactRequest, ...]
    simulation_template: SimulationRequest
    overwrite_existing: bool
    create_parent_dirs: bool
    include_values: bool
    include_innovations: bool
    include_variances: bool
    json_sort_keys: bool
    json_indent: int
    enforce_zero_shot_c_train_exclusion: bool
    sample_id_hash_len: int


@dataclass(frozen=True)
class SplitArtifactRunResult:
    split_name: SplitName
    samples_path: str
    manifest_path: str
    samples_sha256: str
    manifest_sha256: str
    sample_count: int
    count_by_family: tuple[tuple[FamilyId, int], ...]
    write_result: ArtifactWriteResult


@dataclass(frozen=True)
class Phase2ArtifactRunResult:
    protocol_name: str
    artifact_dir: str
    artifact_subdir: str
    split_results: tuple[SplitArtifactRunResult, ...]
    status: str


def validate_split_artifact_request(request: SplitArtifactRequest) -> None:
    if not isinstance(request, SplitArtifactRequest):
        raise ValueError("request must be a SplitArtifactRequest instance")
    if not isinstance(request.split_name, SplitName):
        raise ValueError("split_name must be a SplitName enum instance")

    # Validate sample_count_by_family
    if type(request.sample_count_by_family) is not tuple:
        raise ValueError("sample_count_by_family must be exactly a tuple")
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

    # Validate base_seed_by_family
    if type(request.base_seed_by_family) is not tuple:
        raise ValueError("base_seed_by_family must be exactly a tuple")
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

    # Validate generation_template_by_family
    if type(request.generation_template_by_family) is not tuple:
        raise ValueError("generation_template_by_family must be exactly a tuple")
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
        seen_gens.add(fid)

    # Active families checks
    for fid, count in request.sample_count_by_family:
        if count > 0:
            if fid not in seen_gens:
                raise ValueError(f"Missing generation template for active family {fid}")
            if fid not in seen_seeds:
                raise ValueError(f"Missing base seed for active family {fid}")

    # Total count check
    total_count = sum(c for _, c in request.sample_count_by_family)
    if total_count <= 0:
        raise ValueError("Total sample count must be > 0")

    # Filename validations
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


def validate_phase2_artifact_run_request(request: Phase2ArtifactRunRequest) -> None:
    if not isinstance(request, Phase2ArtifactRunRequest):
        raise ValueError("request must be a Phase2ArtifactRunRequest instance")
    if not isinstance(request.protocol_name, str) or len(request.protocol_name.strip()) == 0:
        raise ValueError("protocol_name must be a non-empty string")
    if not isinstance(request.artifact_dir, str) or len(request.artifact_dir.strip()) == 0:
        raise ValueError("artifact_dir must be a non-empty string")

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

    # Validate splits tuple
    if type(request.splits) is not tuple:
        raise ValueError("splits must be exactly a tuple")
    if len(request.splits) == 0:
        raise ValueError("splits tuple must not be empty")

    seen_split_names = set()
    for split_req in request.splits:
        if not isinstance(split_req, SplitArtifactRequest):
            raise ValueError("Each element in splits must be a SplitArtifactRequest instance")
        validate_split_artifact_request(split_req)
        if split_req.split_name in seen_split_names:
            raise ValueError(f"Duplicate split_name '{split_req.split_name.value}' in splits")
        seen_split_names.add(split_req.split_name)

    # Validate simulation_template
    if not isinstance(request.simulation_template, SimulationRequest):
        raise ValueError("simulation_template must be a SimulationRequest instance")
    validate_simulation_request(request.simulation_template)

    # Validate write preferences
    if not isinstance(request.overwrite_existing, bool):
        raise ValueError("overwrite_existing must be a bool")
    if not isinstance(request.create_parent_dirs, bool):
        raise ValueError("create_parent_dirs must be a bool")
    if not isinstance(request.include_values, bool):
        raise ValueError("include_values must be a bool")
    if not isinstance(request.include_innovations, bool):
        raise ValueError("include_innovations must be a bool")
    if not isinstance(request.include_variances, bool):
        raise ValueError("include_variances must be a bool")
    if not (request.include_values or request.include_innovations or request.include_variances):
        raise ValueError("At least one of include_values, include_innovations, or include_variances must be True")

    if not isinstance(request.json_sort_keys, bool):
        raise ValueError("json_sort_keys must be a bool")
    if type(request.json_indent) is not int or isinstance(request.json_indent, bool) or request.json_indent < 0:
        raise ValueError("json_indent must be a non-negative int")

    # Zero-shot C Train exclusion & Hash length
    if not isinstance(request.enforce_zero_shot_c_train_exclusion, bool):
        raise ValueError("enforce_zero_shot_c_train_exclusion must be a bool")
    if type(request.sample_id_hash_len) is not int or isinstance(request.sample_id_hash_len, bool) or request.sample_id_hash_len < 8:
        raise ValueError("sample_id_hash_len must be an int >= 8")

    # Leakage check: split request zero_shot_train must not request ARMA_GARCH > 0 if exclusion is True
    if request.enforce_zero_shot_c_train_exclusion:
        for split_req in request.splits:
            if split_req.split_name == SplitName.ZERO_SHOT_TRAIN:
                for fid, count in split_req.sample_count_by_family:
                    if fid == FamilyId.ARMA_GARCH and count > 0:
                        raise ValueError(
                            f"C leakage detected in split request: ARMA_GARCH count is {count} (> 0), "
                            f"which is strictly prohibited in split zero_shot_train."
                        )


def build_dataset_request_for_split(
    run_req: Phase2ArtifactRunRequest, split_req: SplitArtifactRequest
) -> DatasetBuildRequest:
    return DatasetBuildRequest(
        protocol_name=run_req.protocol_name,
        split_name=split_req.split_name,
        sample_count_by_family=split_req.sample_count_by_family,
        generation_template_by_family=split_req.generation_template_by_family,
        simulation_template=run_req.simulation_template,
        base_seed_by_family=split_req.base_seed_by_family,
        enforce_zero_shot_c_train_exclusion=run_req.enforce_zero_shot_c_train_exclusion,
        sample_id_hash_len=run_req.sample_id_hash_len,
    )


def build_artifact_write_request_for_split(
    run_req: Phase2ArtifactRunRequest,
    split_req: SplitArtifactRequest,
    build_res: DatasetBuildResult,
) -> ArtifactWriteRequest:
    output_dir = str(pathlib.Path(run_req.artifact_dir) / run_req.artifact_subdir)
    return ArtifactWriteRequest(
        dataset_result=build_res,
        output_dir=output_dir,
        samples_filename=split_req.samples_filename,
        manifest_filename=split_req.manifest_filename,
        create_parent_dirs=run_req.create_parent_dirs,
        overwrite_existing=run_req.overwrite_existing,
        include_values=run_req.include_values,
        include_innovations=run_req.include_innovations,
        include_variances=run_req.include_variances,
        json_sort_keys=run_req.json_sort_keys,
        json_indent=run_req.json_indent,
    )


def run_phase2_artifact_generation(request: Phase2ArtifactRunRequest) -> Phase2ArtifactRunResult:
    # Validate the run request before executing any split
    validate_phase2_artifact_run_request(request)

    split_results = []
    for split_req in request.splits:
        # Validate split request before constructing DatasetBuildRequest
        validate_split_artifact_request(split_req)

        build_req = build_dataset_request_for_split(request, split_req)
        build_res = build_dataset_in_memory(build_req)

        write_req = build_artifact_write_request_for_split(request, split_req, build_res)
        write_res = write_dataset_artifacts(write_req)

        split_result = SplitArtifactRunResult(
            split_name=split_req.split_name,
            samples_path=write_res.samples_path,
            manifest_path=write_res.manifest_path,
            samples_sha256=write_res.samples_sha256,
            manifest_sha256=write_res.manifest_sha256,
            sample_count=write_res.sample_count,
            count_by_family=write_res.count_by_family,
            write_result=write_res,
        )
        split_results.append(split_result)

    return Phase2ArtifactRunResult(
        protocol_name=request.protocol_name,
        artifact_dir=request.artifact_dir,
        artifact_subdir=request.artifact_subdir,
        split_results=tuple(split_results),
        status="success",
    )
