# src/phase2/dataset.py

import json
import hashlib
from dataclasses import dataclass, replace
from enum import Enum

from src.phase2.schema import FamilyId, ModelSpec, validate_model_spec
from src.phase2.sampler import GenerationRequest, validate_generation_request, generate_model_spec
from src.phase2.simulator import SimulationRequest, validate_simulation_request, simulate_time_series


class SplitName(str, Enum):
    ZERO_SHOT_TRAIN = "zero_shot_train"
    ZERO_SHOT_EVAL = "zero_shot_eval"
    FEWSHOT_TRAIN = "fewshot_train"
    FEWSHOT_EVAL = "fewshot_eval"
    SMOKE = "smoke"


@dataclass(frozen=True)
class DatasetBuildRequest:
    protocol_name: str
    split_name: SplitName
    sample_count_by_family: tuple[tuple[FamilyId, int], ...]
    generation_template_by_family: tuple[tuple[FamilyId, GenerationRequest], ...]
    simulation_template: SimulationRequest
    base_seed_by_family: tuple[tuple[FamilyId, int], ...]
    enforce_zero_shot_c_train_exclusion: bool
    sample_id_hash_len: int


@dataclass(frozen=True)
class DatasetSample:
    sample_id: str
    protocol_name: str
    split_name: SplitName
    family_id: FamilyId
    sample_index: int
    generation_seed: int
    simulation_seed: int
    model_spec: ModelSpec
    values: tuple[float, ...]
    innovations: tuple[float, ...]
    variances: tuple[float, ...]
    provenance: tuple[tuple[str, str], ...]


@dataclass(frozen=True)
class DatasetBuildResult:
    samples: tuple[DatasetSample, ...]
    protocol_name: str
    split_name: SplitName
    total_count: int
    count_by_family: tuple[tuple[FamilyId, int], ...]
    zero_shot_c_train_count: int
    reason: str


def make_sample_id(
    protocol_name: str,
    split_name: SplitName,
    family_id: FamilyId,
    sample_index: int,
    generation_seed: int,
    simulation_seed: int,
    model_spec: ModelSpec,
    hash_len: int,
) -> str:
    if not isinstance(protocol_name, str):
        raise ValueError("protocol_name must be a string")
    if not isinstance(split_name, SplitName):
        raise ValueError("split_name must be a SplitName enum instance")
    if not isinstance(family_id, FamilyId):
        raise ValueError("family_id must be a FamilyId enum instance")
    if type(sample_index) is not int or sample_index <= 0:
        raise ValueError("sample_index must be a positive int")
    if type(generation_seed) is not int or isinstance(generation_seed, bool):
        raise ValueError("generation_seed must be an int")
    if type(simulation_seed) is not int or isinstance(simulation_seed, bool):
        raise ValueError("simulation_seed must be an int")
    if not isinstance(model_spec, ModelSpec):
        raise ValueError("model_spec must be a ModelSpec instance")
    if type(hash_len) is not int or isinstance(hash_len, bool) or hash_len < 8 or hash_len > 64:
        raise ValueError("hash_len must be an int between 8 and 64")

    payload = {
        "protocol_name": protocol_name,
        "split_name": split_name.value,
        "family_id": family_id.value,
        "sample_index": sample_index,
        "generation_seed": generation_seed,
        "simulation_seed": simulation_seed,
        "p": model_spec.p,
        "q": model_spec.q,
        "r": model_spec.r,
        "s": model_spec.s,
        "ar_params": list(model_spec.ar_params),
        "ma_params": list(model_spec.ma_params),
        "omega": model_spec.omega,
        "alpha_params": list(model_spec.alpha_params),
        "beta_params": list(model_spec.beta_params),
        "constraint_flags": list(model_spec.constraint_flags),
        "provenance": [list(item) for item in model_spec.provenance],
    }

    payload_json = json.dumps(payload, separators=(",", ":"), sort_keys=True)
    h = hashlib.sha256(payload_json.encode("utf-8")).hexdigest()
    truncated_hash = h[:hash_len]

    index_str = f"{sample_index:06d}"
    return f"{family_id.value}_{split_name.value}_{index_str}_g{generation_seed}_s{simulation_seed}_{truncated_hash}"


def validate_dataset_build_request(request: DatasetBuildRequest) -> None:
    # 1. protocol_name: str, non-empty
    if not isinstance(request.protocol_name, str):
        raise ValueError("protocol_name must be a string")
    if len(request.protocol_name.strip()) == 0:
        raise ValueError("protocol_name must be non-empty")

    # 2. split_name: SplitName
    if not isinstance(request.split_name, SplitName):
        raise ValueError("split_name must be a SplitName enum instance")

    # 3. enforce_zero_shot_c_train_exclusion: bool
    if not isinstance(request.enforce_zero_shot_c_train_exclusion, bool):
        raise ValueError("enforce_zero_shot_c_train_exclusion must be a bool")

    # 4. sample_id_hash_len: int, not bool, >= 8
    if type(request.sample_id_hash_len) is not int or isinstance(request.sample_id_hash_len, bool):
        raise ValueError("sample_id_hash_len must be an int, not bool or other type")
    if request.sample_id_hash_len < 8:
        raise ValueError(f"sample_id_hash_len must be >= 8, got {request.sample_id_hash_len}")

    # 5. sample_count_by_family: exactly a tuple
    if type(request.sample_count_by_family) is not tuple:
        raise ValueError("sample_count_by_family must be exactly a tuple")

    seen_count_families: set[FamilyId] = set()
    for item in request.sample_count_by_family:
        if type(item) is not tuple or len(item) != 2:
            raise ValueError("Each element in sample_count_by_family must be a tuple of length 2")
        f_id, count = item
        if not isinstance(f_id, FamilyId):
            raise ValueError("family_id in sample_count_by_family must be a FamilyId enum instance")
        if type(count) is not int or isinstance(count, bool) or count < 0:
            raise ValueError("count in sample_count_by_family must be a non-negative int, not bool")
        if f_id in seen_count_families:
            raise ValueError(f"Duplicate family_id {f_id} in sample_count_by_family")
        seen_count_families.add(f_id)

    # 6. base_seed_by_family: exactly a tuple
    if type(request.base_seed_by_family) is not tuple:
        raise ValueError("base_seed_by_family must be exactly a tuple")

    seen_seed_families: set[FamilyId] = set()
    for item in request.base_seed_by_family:
        if type(item) is not tuple or len(item) != 2:
            raise ValueError("Each element in base_seed_by_family must be a tuple of length 2")
        f_id, seed = item
        if not isinstance(f_id, FamilyId):
            raise ValueError("family_id in base_seed_by_family must be a FamilyId enum instance")
        if type(seed) is not int or isinstance(seed, bool):
            raise ValueError("seed in base_seed_by_family must be an int, not bool")
        if f_id in seen_seed_families:
            raise ValueError(f"Duplicate family_id {f_id} in base_seed_by_family")
        seen_seed_families.add(f_id)

    # 7. generation_template_by_family: exactly a tuple
    if type(request.generation_template_by_family) is not tuple:
        raise ValueError("generation_template_by_family must be exactly a tuple")

    seen_gen_families: set[FamilyId] = set()
    for item in request.generation_template_by_family:
        if type(item) is not tuple or len(item) != 2:
            raise ValueError("Each element in generation_template_by_family must be a tuple of length 2")
        f_id, gen_req = item
        if not isinstance(f_id, FamilyId):
            raise ValueError("family_id in generation_template_by_family must be a FamilyId enum instance")
        if not isinstance(gen_req, GenerationRequest):
            raise ValueError("template in generation_template_by_family must be a GenerationRequest instance")
        # Family mismatch check
        if gen_req.family_id != f_id:
            raise ValueError(
                f"generation_template_by_family key {f_id} does not match "
                f"GenerationRequest.family_id {gen_req.family_id}"
            )
        # Validate the template itself
        validate_generation_request(gen_req)
        if f_id in seen_gen_families:
            raise ValueError(f"Duplicate family_id {f_id} in generation_template_by_family")
        seen_gen_families.add(f_id)

    # 8. simulation_template: SimulationRequest + passes validate_simulation_request
    if not isinstance(request.simulation_template, SimulationRequest):
        raise ValueError("simulation_template must be a SimulationRequest instance")
    # Note: We only validate the template's non-spec fields; the spec will be replaced per sample.
    # Validate the simulation template as-is (it must be at least structurally valid).
    validate_simulation_request(request.simulation_template)

    # 9. Active families must have seeds and templates
    for f_id, count in request.sample_count_by_family:
        if count > 0:
            if f_id not in seen_gen_families:
                raise ValueError(f"Missing generation template for active family {f_id}")
            if f_id not in seen_seed_families:
                raise ValueError(f"Missing base seed for active family {f_id}")

    # 10. Total count must be > 0
    total = sum(count for _, count in request.sample_count_by_family)
    if total <= 0:
        raise ValueError("Total sample count must be > 0; all counts are 0 or empty")

    # 11. Zero-shot C leakage guard
    if request.split_name == SplitName.ZERO_SHOT_TRAIN and request.enforce_zero_shot_c_train_exclusion:
        arma_garch_count = 0
        for f_id, count in request.sample_count_by_family:
            if f_id == FamilyId.ARMA_GARCH:
                arma_garch_count = count
        if arma_garch_count > 0:
            raise ValueError(
                f"C leakage detected: ARMA_GARCH count is {arma_garch_count} (> 0), "
                f"which is strictly prohibited in split zero_shot_train."
            )


def build_dataset_in_memory(request: DatasetBuildRequest) -> DatasetBuildResult:
    validate_dataset_build_request(request)

    gen_templates = dict(request.generation_template_by_family)
    base_seeds = dict(request.base_seed_by_family)

    samples: list[DatasetSample] = []

    for family_id, count in request.sample_count_by_family:
        if count <= 0:
            continue

        gen_template = gen_templates[family_id]
        base_seed = base_seeds[family_id]

        for i in range(count):
            sample_index = i + 1
            generation_seed = base_seed + i
            simulation_seed = base_seed + 1_000_000 + i

            # Replace seed in generation request, do not mutate original
            sample_gen_req = replace(gen_template, seed=generation_seed)
            gen_res = generate_model_spec(sample_gen_req)
            generated_spec = gen_res.spec

            # Validate generated spec
            validate_model_spec(generated_spec)

            # Build augmented provenance: original spec provenance + builder entries
            augmented_provenance = (
                generated_spec.provenance
                + (
                    ("protocol_name", request.protocol_name),
                    ("split_name", request.split_name.value),
                    ("generation_seed", str(generation_seed)),
                    ("simulation_seed", str(simulation_seed)),
                    ("sample_index", str(sample_index)),
                )
            )

            # Replace spec and seed in simulation request, do not mutate original
            sample_sim_req = replace(
                request.simulation_template,
                spec=generated_spec,
                seed=simulation_seed,
            )
            sim_res = simulate_time_series(sample_sim_req)

            sample_id = make_sample_id(
                protocol_name=request.protocol_name,
                split_name=request.split_name,
                family_id=family_id,
                sample_index=sample_index,
                generation_seed=generation_seed,
                simulation_seed=simulation_seed,
                model_spec=generated_spec,
                hash_len=request.sample_id_hash_len,
            )

            sample = DatasetSample(
                sample_id=sample_id,
                protocol_name=request.protocol_name,
                split_name=request.split_name,
                family_id=family_id,
                sample_index=sample_index,
                generation_seed=generation_seed,
                simulation_seed=simulation_seed,
                model_spec=generated_spec,
                values=sim_res.values,
                innovations=sim_res.innovations,
                variances=sim_res.variances,
                provenance=augmented_provenance,
            )
            samples.append(sample)

    # Build count_by_family in the same order as request.sample_count_by_family
    built_counts: dict[FamilyId, int] = {}
    for sample in samples:
        built_counts[sample.family_id] = built_counts.get(sample.family_id, 0) + 1

    count_by_family = tuple(
        (f_id, built_counts.get(f_id, 0))
        for f_id, _ in request.sample_count_by_family
    )

    # zero_shot_c_train_count: actual ARMA_GARCH count in built results
    zero_shot_c_train_count = built_counts.get(FamilyId.ARMA_GARCH, 0)

    return DatasetBuildResult(
        samples=tuple(samples),
        protocol_name=request.protocol_name,
        split_name=request.split_name,
        total_count=len(samples),
        count_by_family=count_by_family,
        zero_shot_c_train_count=zero_shot_c_train_count,
        reason="dataset_built_in_memory",
    )
