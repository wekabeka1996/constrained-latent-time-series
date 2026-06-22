# src/phase2/protocol_presets.py

from dataclasses import dataclass
from enum import Enum
import pathlib
from typing import Any

from src.phase2.schema import FamilyId
from src.phase2.sampler import GenerationRequest, validate_generation_request
from src.phase2.simulator import SimulationRequest, validate_simulation_request
from src.phase2.dataset import SplitName
from src.phase2.split_runner import (
    SplitArtifactRequest,
    Phase2ArtifactRunRequest,
    validate_phase2_artifact_run_request,
)


class ProtocolPresetName(str, Enum):
    SMOKE = "smoke"
    DEV = "dev"
    MAIN = "main"


@dataclass(frozen=True)
class ProtocolPresetFactoryRequest:
    protocol_name: str
    preset_name: ProtocolPresetName
    output_root_dir: str
    generation_template_by_family: tuple[tuple[FamilyId, GenerationRequest], ...]
    simulation_template: SimulationRequest
    create_parent_dirs: bool
    overwrite_existing: bool
    include_values: bool
    include_innovations: bool
    include_variances: bool
    json_sort_keys: bool
    json_indent: int
    sample_id_hash_len: int


@dataclass(frozen=True)
class ProtocolPresetFactoryResult:
    run_request: Phase2ArtifactRunRequest
    preset_name: ProtocolPresetName
    protocol_name: str
    total_requested_samples: int
    split_count: int
    count_by_split: tuple[tuple[SplitName, int], ...]
    reason: str


# Public constants as tuples of tuples
APPROVED_PHASE2_PRESET_SAMPLE_COUNTS = (
    ("smoke", 1000),
    ("dev", 10000),
    ("main", 100000),
)

APPROVED_PHASE2_BASE_SEEDS_BY_FAMILY = (
    (FamilyId.AR, 12001),
    (FamilyId.ARMA, 12001),
    (FamilyId.GARCH, 12002),
    (FamilyId.ARMA_GARCH, 12003),
)

APPROVED_PHASE2_ARTIFACT_SUBDIR_BY_SPLIT = (
    (SplitName.SMOKE, "smoke"),
    (SplitName.ZERO_SHOT_TRAIN, "zero_shot_train"),
    (SplitName.ZERO_SHOT_EVAL, "zero_shot_eval"),
    (SplitName.FEWSHOT_TRAIN, "fewshot_train"),
    (SplitName.FEWSHOT_EVAL, "fewshot_eval"),
)

APPROVED_PHASE2_SAMPLES_FILENAME = "samples.jsonl"
APPROVED_PHASE2_MANIFEST_FILENAME = "manifest.json"


def validate_protocol_preset_factory_request(request: ProtocolPresetFactoryRequest) -> None:
    if not isinstance(request, ProtocolPresetFactoryRequest):
        raise ValueError("request must be a ProtocolPresetFactoryRequest instance")

    if not isinstance(request.protocol_name, str) or len(request.protocol_name.strip()) == 0:
        raise ValueError("protocol_name must be a non-empty string")

    if not isinstance(request.preset_name, ProtocolPresetName):
        raise ValueError("preset_name must be a ProtocolPresetName enum member")

    if not isinstance(request.output_root_dir, str) or len(request.output_root_dir.strip()) == 0:
        raise ValueError("output_root_dir must be a non-empty string")

    # Validate generation templates
    if type(request.generation_template_by_family) is not tuple:
        raise ValueError("generation_template_by_family must be exactly a tuple")

    seen_families = set()
    for item in request.generation_template_by_family:
        if type(item) is not tuple or len(item) != 2:
            raise ValueError("Each element in generation_template_by_family must be a 2-tuple")
        fid, template = item
        if not isinstance(fid, FamilyId):
            raise ValueError("family key in generation_template_by_family must be a FamilyId instance")
        if fid in seen_families:
            raise ValueError(f"Duplicate family template entry for {fid}")
        if not isinstance(template, GenerationRequest):
            raise ValueError("template in generation_template_by_family must be a GenerationRequest instance")
        if template.family_id != fid:
            raise ValueError(f"GenerationRequest family_id {template.family_id} does not match key {fid}")
        validate_generation_request(template)
        seen_families.add(fid)

    required_families = {FamilyId.AR, FamilyId.ARMA, FamilyId.GARCH, FamilyId.ARMA_GARCH}
    missing_families = required_families - seen_families
    if missing_families:
        raise ValueError(f"Missing generation templates for required families: {missing_families}")

    # Validate simulation template
    if not isinstance(request.simulation_template, SimulationRequest):
        raise ValueError("simulation_template must be a SimulationRequest instance")
    validate_simulation_request(request.simulation_template)

    # Validate bool flags
    if type(request.create_parent_dirs) is not bool:
        raise ValueError("create_parent_dirs must be a bool")
    if type(request.overwrite_existing) is not bool:
        raise ValueError("overwrite_existing must be a bool")
    if type(request.include_values) is not bool:
        raise ValueError("include_values must be a bool")
    if type(request.include_innovations) is not bool:
        raise ValueError("include_innovations must be a bool")
    if type(request.include_variances) is not bool:
        raise ValueError("include_variances must be a bool")

    if not (request.include_values or request.include_innovations or request.include_variances):
        raise ValueError("At least one include flag must be True")

    if type(request.json_sort_keys) is not bool:
        raise ValueError("json_sort_keys must be a bool")

    # Validate integers
    if type(request.json_indent) is not int or isinstance(request.json_indent, bool) or request.json_indent < 0:
        raise ValueError("json_indent must be a non-negative int")
    if type(request.sample_id_hash_len) is not int or isinstance(request.sample_id_hash_len, bool) or request.sample_id_hash_len < 8:
        raise ValueError("sample_id_hash_len must be an int >= 8")

    # Validate output root dir exists as file check
    p = pathlib.Path(request.output_root_dir)
    if p.exists() and p.is_file():
        raise ValueError(f"output_root_dir '{request.output_root_dir}' exists and is a file")


def get_preset_sample_count_by_family(preset_name: ProtocolPresetName) -> tuple[tuple[FamilyId, int], ...]:
    if not isinstance(preset_name, ProtocolPresetName):
        raise ValueError("preset_name must be a ProtocolPresetName enum member")
    
    val = 0
    if preset_name == ProtocolPresetName.SMOKE:
        val = 1000
    elif preset_name == ProtocolPresetName.DEV:
        val = 10000
    elif preset_name == ProtocolPresetName.MAIN:
        val = 100000
    else:
        raise ValueError(f"Unknown preset_name: {preset_name}")

    return (
        (FamilyId.AR, val),
        (FamilyId.ARMA, val),
        (FamilyId.GARCH, val),
        (FamilyId.ARMA_GARCH, val),
    )


def build_split_request_for_preset(
    preset_name: ProtocolPresetName,
    split_name: SplitName,
    generation_template_by_family: tuple[tuple[FamilyId, GenerationRequest], ...],
    simulation_template: SimulationRequest,
    sample_id_hash_len: int,
) -> SplitArtifactRequest:
    if not isinstance(preset_name, ProtocolPresetName):
        raise ValueError("preset_name must be a ProtocolPresetName enum member")
    if not isinstance(split_name, SplitName):
        raise ValueError("split_name must be a SplitName enum member")

    # Get sample count per family based on preset name
    family_preset_counts = get_preset_sample_count_by_family(preset_name)
    count_map = dict(family_preset_counts)

    if split_name == SplitName.SMOKE:
        sample_count_by_family = (
            (FamilyId.AR, count_map[FamilyId.AR]),
            (FamilyId.ARMA, count_map[FamilyId.ARMA]),
            (FamilyId.GARCH, count_map[FamilyId.GARCH]),
            (FamilyId.ARMA_GARCH, count_map[FamilyId.ARMA_GARCH]),
        )
        enforce_zero_shot_c_train_exclusion = False
    elif split_name == SplitName.ZERO_SHOT_TRAIN:
        sample_count_by_family = (
            (FamilyId.AR, count_map[FamilyId.AR]),
            (FamilyId.ARMA, count_map[FamilyId.ARMA]),
            (FamilyId.GARCH, count_map[FamilyId.GARCH]),
        )
        enforce_zero_shot_c_train_exclusion = True
    elif split_name == SplitName.ZERO_SHOT_EVAL:
        sample_count_by_family = (
            (FamilyId.ARMA_GARCH, count_map[FamilyId.ARMA_GARCH]),
        )
        enforce_zero_shot_c_train_exclusion = False
    elif split_name == SplitName.FEWSHOT_TRAIN:
        sample_count_by_family = (
            (FamilyId.ARMA_GARCH, count_map[FamilyId.ARMA_GARCH]),
        )
        enforce_zero_shot_c_train_exclusion = False
    elif split_name == SplitName.FEWSHOT_EVAL:
        sample_count_by_family = (
            (FamilyId.ARMA_GARCH, count_map[FamilyId.ARMA_GARCH]),
        )
        enforce_zero_shot_c_train_exclusion = False
    else:
        raise ValueError(f"Unknown split_name: {split_name}")

    subdir = None
    for s_name, sub in APPROVED_PHASE2_ARTIFACT_SUBDIR_BY_SPLIT:
        if s_name == split_name:
            subdir = sub
            break
    if subdir is None:
        raise ValueError(f"No subdir found for split {split_name}")

    return SplitArtifactRequest(
        split_name=split_name,
        sample_count_by_family=sample_count_by_family,
        base_seed_by_family=APPROVED_PHASE2_BASE_SEEDS_BY_FAMILY,
        generation_template_by_family=generation_template_by_family,
        simulation_template=simulation_template,
        artifact_subdir=subdir,
        samples_filename=APPROVED_PHASE2_SAMPLES_FILENAME,
        manifest_filename=APPROVED_PHASE2_MANIFEST_FILENAME,
        enforce_zero_shot_c_train_exclusion=enforce_zero_shot_c_train_exclusion,
        sample_id_hash_len=sample_id_hash_len,
    )


def build_phase2_preset_run_request(request: ProtocolPresetFactoryRequest) -> Phase2ArtifactRunRequest:
    validate_protocol_preset_factory_request(request)

    splits = (
        SplitName.SMOKE,
        SplitName.ZERO_SHOT_TRAIN,
        SplitName.ZERO_SHOT_EVAL,
        SplitName.FEWSHOT_TRAIN,
        SplitName.FEWSHOT_EVAL,
    )

    split_requests = []
    for s_name in splits:
        split_req = build_split_request_for_preset(
            preset_name=request.preset_name,
            split_name=s_name,
            generation_template_by_family=request.generation_template_by_family,
            simulation_template=request.simulation_template,
            sample_id_hash_len=request.sample_id_hash_len,
        )
        split_requests.append(split_req)

    run_request = Phase2ArtifactRunRequest(
        protocol_name=request.protocol_name,
        output_root_dir=request.output_root_dir,
        split_requests=tuple(split_requests),
        create_parent_dirs=request.create_parent_dirs,
        overwrite_existing=request.overwrite_existing,
        include_values=request.include_values,
        include_innovations=request.include_innovations,
        include_variances=request.include_variances,
        json_sort_keys=request.json_sort_keys,
        json_indent=request.json_indent,
    )

    # Cross-verify using split runner's validation
    validate_phase2_artifact_run_request(run_request)
    return run_request


def build_phase2_preset_factory_result(request: ProtocolPresetFactoryRequest) -> ProtocolPresetFactoryResult:
    # No defaults.
    run_req = build_phase2_preset_run_request(request)

    count_by_split = []
    total_requested_samples = 0
    for split_req in run_req.split_requests:
        split_total = sum(c for _, c in split_req.sample_count_by_family)
        count_by_split.append((split_req.split_name, split_total))
        total_requested_samples += split_total

    return ProtocolPresetFactoryResult(
        run_request=run_req,
        preset_name=request.preset_name,
        protocol_name=request.protocol_name,
        total_requested_samples=total_requested_samples,
        split_count=len(run_req.split_requests),
        count_by_split=tuple(count_by_split),
        reason="phase2_protocol_preset_request_built",
    )
