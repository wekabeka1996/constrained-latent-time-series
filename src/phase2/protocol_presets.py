# src/phase2/protocol_presets.py

from dataclasses import dataclass
from enum import Enum
import pathlib

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

APPROVED_PHASE2_FEWSHOT_COUNTS_BY_PRESET = (
    (ProtocolPresetName.SMOKE, 10, 50),
    (ProtocolPresetName.DEV, 100, 500),
    (ProtocolPresetName.MAIN, 1000, 5000),
)

APPROVED_PHASE2_ARTIFACT_SUBDIR_BY_SPLIT = (
    (SplitName.SMOKE, "smoke"),
    (SplitName.ZERO_SHOT_TRAIN, "zero_shot_train"),
    (SplitName.ZERO_SHOT_EVAL, "zero_shot_eval"),
    (SplitName.FEWSHOT_TRAIN, "fewshot_1pct_train"),
    (SplitName.FEWSHOT_TRAIN, "fewshot_5pct_train"),
    (SplitName.FEWSHOT_EVAL, "fewshot_eval"),
)

APPROVED_PHASE2_C_SEED_ROOT_BY_C_SPLIT = (
    ("zero_shot_eval", 12003),
    ("fewshot_train", 12101),
    ("fewshot_eval", 12105),
)

APPROVED_PHASE2_C_SEED_RANGE_STRIDE = 10_000_000

APPROVED_PHASE2_C_SEED_BASE_BY_C_SPLIT = (
    ("zero_shot_eval", 12003),
    ("fewshot_train", 10_012_101),
    ("fewshot_eval", 20_012_105),
)

APPROVED_PHASE2_SAMPLES_FILENAME = "samples.jsonl"
APPROVED_PHASE2_MANIFEST_FILENAME = "manifest.json"


def get_fewshot_counts_for_preset(
    preset_name: ProtocolPresetName,
) -> tuple[int, int]:
    if not isinstance(preset_name, ProtocolPresetName):
        raise ValueError("preset_name must be a ProtocolPresetName enum member")
    for name, c1, c5 in APPROVED_PHASE2_FEWSHOT_COUNTS_BY_PRESET:
        if name == preset_name:
            return (c1, c5)
    raise ValueError(f"Unknown preset_name: {preset_name}")


def seed_range_for_base_and_count(
    base_seed: int,
    count: int,
) -> tuple[int, int]:
    if type(count) is not int or isinstance(count, bool) or count <= 0:
        raise ValueError("count must be a positive integer")
    if type(base_seed) is not int or isinstance(base_seed, bool):
        raise ValueError("base_seed must be an integer")
    return (base_seed, base_seed + count - 1)


def ranges_overlap(
    left: tuple[int, int],
    right: tuple[int, int],
) -> bool:
    if type(left) is not tuple or len(left) != 2 or type(right) is not tuple or len(right) != 2:
        raise ValueError("ranges must be 2-tuples")
    x1, x2 = left
    y1, y2 = right
    if type(x1) is not int or type(x2) is not int or type(y1) is not int or type(y2) is not int:
        raise ValueError("range bounds must be integers")
    return max(x1, y1) <= min(x2, y2)


def range_is_prefix_subset(
    smaller: tuple[int, int],
    larger: tuple[int, int],
) -> bool:
    if type(smaller) is not tuple or len(smaller) != 2 or type(larger) is not tuple or len(larger) != 2:
        raise ValueError("ranges must be 2-tuples")
    s1, s2 = smaller
    l1, l2 = larger
    if type(s1) is not int or type(s2) is not int or type(l1) is not int or type(l2) is not int:
        raise ValueError("range bounds must be integers")
    return s1 == l1 and s2 <= l2


def validate_p13_fewshot_seed_plan_for_preset(
    preset_name: ProtocolPresetName,
) -> None:
    if not isinstance(preset_name, ProtocolPresetName):
        raise ValueError("preset_name must be a ProtocolPresetName enum member")

    # Get preset counts
    preset_counts = get_preset_sample_count_by_family(preset_name)
    count_map = dict(preset_counts)
    n_eval = count_map[FamilyId.ARMA_GARCH]

    n_1, n_5 = get_fewshot_counts_for_preset(preset_name)

    # Base seeds from approved constant
    seeds_dict = dict(APPROVED_PHASE2_C_SEED_BASE_BY_C_SPLIT)
    base_eval = seeds_dict["zero_shot_eval"]
    base_train = seeds_dict["fewshot_train"]
    base_fewshot_eval = seeds_dict["fewshot_eval"]

    # Generation Ranges
    r_1 = seed_range_for_base_and_count(base_train, n_1)
    r_5 = seed_range_for_base_and_count(base_train, n_5)
    r_z_eval = seed_range_for_base_and_count(base_eval, n_eval)
    r_f_eval = seed_range_for_base_and_count(base_fewshot_eval, n_eval)

    # Overlap and Prefix validations for generation
    if not range_is_prefix_subset(r_1, r_5):
        raise ValueError(f"Fewshot 1% range {r_1} is not a prefix subset of 5% range {r_5}")
    if ranges_overlap(r_5, r_z_eval):
        raise ValueError(f"Fewshot 5% train range {r_5} overlaps with zero_shot_eval range {r_z_eval}")
    if ranges_overlap(r_5, r_f_eval):
        raise ValueError(f"Fewshot 5% train range {r_5} overlaps with fewshot_eval range {r_f_eval}")
    if ranges_overlap(r_z_eval, r_f_eval):
        raise ValueError(f"Zero_shot_eval range {r_z_eval} overlaps with fewshot_eval range {r_f_eval}")

    # Simulation Ranges (offset by 1_000_000)
    s_1 = seed_range_for_base_and_count(base_train + 1_000_000, n_1)
    s_5 = seed_range_for_base_and_count(base_train + 1_000_000, n_5)
    s_z_eval = seed_range_for_base_and_count(base_eval + 1_000_000, n_eval)
    s_f_eval = seed_range_for_base_and_count(base_fewshot_eval + 1_000_000, n_eval)

    # Overlap and Prefix validations for simulation
    if not range_is_prefix_subset(s_1, s_5):
        raise ValueError(f"Simulation Fewshot 1% range {s_1} is not a prefix subset of 5% range {s_5}")
    if ranges_overlap(s_5, s_z_eval):
        raise ValueError(f"Simulation Fewshot 5% train range {s_5} overlaps with zero_shot_eval range {s_z_eval}")
    if ranges_overlap(s_5, s_f_eval):
        raise ValueError(f"Simulation Fewshot 5% train range {s_5} overlaps with fewshot_eval range {s_f_eval}")
    if ranges_overlap(s_z_eval, s_f_eval):
        raise ValueError(f"Simulation Zero_shot_eval range {s_z_eval} overlaps with fewshot_eval range {s_f_eval}")


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

    # Validate seed plan non-overlap and prefix requirements
    validate_p13_fewshot_seed_plan_for_preset(request.preset_name)


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
    artifact_subdir: str,
    generation_template_by_family: tuple[tuple[FamilyId, GenerationRequest], ...],
    simulation_template: SimulationRequest,
    sample_id_hash_len: int,
) -> SplitArtifactRequest:
    if not isinstance(preset_name, ProtocolPresetName):
        raise ValueError("preset_name must be a ProtocolPresetName enum member")
    if not isinstance(split_name, SplitName):
        raise ValueError("split_name must be a SplitName enum member")
    if not isinstance(artifact_subdir, str):
        raise ValueError("artifact_subdir must be a string")

    # Get sample count per family based on preset name
    family_preset_counts = get_preset_sample_count_by_family(preset_name)
    count_map = dict(family_preset_counts)
    fewshot_1pct, fewshot_5pct = get_fewshot_counts_for_preset(preset_name)

    if split_name == SplitName.SMOKE and artifact_subdir == "smoke":
        sample_count_by_family = (
            (FamilyId.AR, count_map[FamilyId.AR]),
            (FamilyId.ARMA, count_map[FamilyId.ARMA]),
            (FamilyId.GARCH, count_map[FamilyId.GARCH]),
            (FamilyId.ARMA_GARCH, count_map[FamilyId.ARMA_GARCH]),
        )
        base_seeds = APPROVED_PHASE2_BASE_SEEDS_BY_FAMILY
        enforce_zero_shot_c_train_exclusion = False

    elif split_name == SplitName.ZERO_SHOT_TRAIN and artifact_subdir == "zero_shot_train":
        sample_count_by_family = (
            (FamilyId.AR, count_map[FamilyId.AR]),
            (FamilyId.ARMA, count_map[FamilyId.ARMA]),
            (FamilyId.GARCH, count_map[FamilyId.GARCH]),
        )
        base_seeds = APPROVED_PHASE2_BASE_SEEDS_BY_FAMILY
        enforce_zero_shot_c_train_exclusion = True

    elif split_name == SplitName.ZERO_SHOT_EVAL and artifact_subdir == "zero_shot_eval":
        sample_count_by_family = (
            (FamilyId.ARMA_GARCH, count_map[FamilyId.ARMA_GARCH]),
        )
        base_seeds = (
            (FamilyId.AR, 12001),
            (FamilyId.ARMA, 12001),
            (FamilyId.GARCH, 12002),
            (FamilyId.ARMA_GARCH, 12003),
        )
        enforce_zero_shot_c_train_exclusion = False

    elif split_name == SplitName.FEWSHOT_TRAIN and artifact_subdir == "fewshot_1pct_train":
        sample_count_by_family = (
            (FamilyId.ARMA_GARCH, fewshot_1pct),
        )
        base_seeds = (
            (FamilyId.AR, 12001),
            (FamilyId.ARMA, 12001),
            (FamilyId.GARCH, 12002),
            (FamilyId.ARMA_GARCH, 10_012_101),
        )
        enforce_zero_shot_c_train_exclusion = False

    elif split_name == SplitName.FEWSHOT_TRAIN and artifact_subdir == "fewshot_5pct_train":
        sample_count_by_family = (
            (FamilyId.ARMA_GARCH, fewshot_5pct),
        )
        base_seeds = (
            (FamilyId.AR, 12001),
            (FamilyId.ARMA, 12001),
            (FamilyId.GARCH, 12002),
            (FamilyId.ARMA_GARCH, 10_012_101),
        )
        enforce_zero_shot_c_train_exclusion = False

    elif split_name == SplitName.FEWSHOT_EVAL and artifact_subdir == "fewshot_eval":
        sample_count_by_family = (
            (FamilyId.ARMA_GARCH, count_map[FamilyId.ARMA_GARCH]),
        )
        base_seeds = (
            (FamilyId.AR, 12001),
            (FamilyId.ARMA, 12001),
            (FamilyId.GARCH, 12002),
            (FamilyId.ARMA_GARCH, 20_012_105),
        )
        enforce_zero_shot_c_train_exclusion = False

    else:
        raise ValueError(f"Invalid split_name and artifact_subdir pair: ({split_name}, {artifact_subdir})")

    return SplitArtifactRequest(
        split_name=split_name,
        sample_count_by_family=sample_count_by_family,
        base_seed_by_family=base_seeds,
        generation_template_by_family=generation_template_by_family,
        simulation_template=simulation_template,
        artifact_subdir=artifact_subdir,
        samples_filename=APPROVED_PHASE2_SAMPLES_FILENAME,
        manifest_filename=APPROVED_PHASE2_MANIFEST_FILENAME,
        enforce_zero_shot_c_train_exclusion=enforce_zero_shot_c_train_exclusion,
        sample_id_hash_len=sample_id_hash_len,
    )


def build_phase2_preset_run_request(request: ProtocolPresetFactoryRequest) -> Phase2ArtifactRunRequest:
    validate_protocol_preset_factory_request(request)

    splits = (
        (SplitName.SMOKE, "smoke"),
        (SplitName.ZERO_SHOT_TRAIN, "zero_shot_train"),
        (SplitName.ZERO_SHOT_EVAL, "zero_shot_eval"),
        (SplitName.FEWSHOT_TRAIN, "fewshot_1pct_train"),
        (SplitName.FEWSHOT_TRAIN, "fewshot_5pct_train"),
        (SplitName.FEWSHOT_EVAL, "fewshot_eval"),
    )

    split_requests = []
    for s_name, subdir in splits:
        split_req = build_split_request_for_preset(
            preset_name=request.preset_name,
            split_name=s_name,
            artifact_subdir=subdir,
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
