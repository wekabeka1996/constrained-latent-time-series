# src/phase2/dataset.py

import json
import hashlib
from dataclasses import dataclass, replace
from enum import Enum
from typing import Optional

from src.phase2.schema import FamilyId, ModelSpec
from src.phase2.sampler import GenerationRequest, generate_model_spec
from src.phase2.simulator import SimulationRequest, simulate_time_series


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
    spec: ModelSpec
    values: tuple[float, ...]
    innovations: tuple[float, ...]
    variances: tuple[float, ...]


@dataclass(frozen=True)
class DatasetBuildResult:
    samples: tuple[DatasetSample, ...]
    split_name: SplitName


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
    if type(hash_len) is not int or hash_len <= 0 or hash_len > 64:
        raise ValueError("hash_len must be a positive int between 1 and 64")

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
    if not isinstance(request.protocol_name, str):
        raise ValueError("protocol_name must be a string")
    if not isinstance(request.split_name, SplitName):
        raise ValueError("split_name must be a SplitName enum instance")
    if not isinstance(request.enforce_zero_shot_c_train_exclusion, bool):
        raise ValueError("enforce_zero_shot_c_train_exclusion must be a bool")
    if type(request.sample_id_hash_len) is not int or isinstance(request.sample_id_hash_len, bool) or request.sample_id_hash_len <= 0:
        raise ValueError("sample_id_hash_len must be a positive int")

    if type(request.sample_count_by_family) is not tuple:
        raise ValueError("sample_count_by_family must be exactly a tuple")
    
    seen_counts = set()
    for item in request.sample_count_by_family:
        if type(item) is not tuple or len(item) != 2:
            raise ValueError("Each element in sample_count_by_family must be a tuple of length 2")
        f_id, count = item
        if not isinstance(f_id, FamilyId):
            raise ValueError("family_id in sample_count_by_family must be a FamilyId enum instance")
        if type(count) is not int or isinstance(count, bool) or count < 0:
            raise ValueError("count in sample_count_by_family must be a non-negative int")
        if f_id in seen_counts:
            raise ValueError(f"Duplicate family_id {f_id} in sample_count_by_family")
        seen_counts.add(f_id)

    if type(request.base_seed_by_family) is not tuple:
        raise ValueError("base_seed_by_family must be exactly a tuple")
    
    seen_seeds = set()
    for item in request.base_seed_by_family:
        if type(item) is not tuple or len(item) != 2:
            raise ValueError("Each element in base_seed_by_family must be a tuple of length 2")
        f_id, seed = item
        if not isinstance(f_id, FamilyId):
            raise ValueError("family_id in base_seed_by_family must be a FamilyId enum instance")
        if type(seed) is not int or isinstance(seed, bool):
            raise ValueError("seed in base_seed_by_family must be an int")
        if f_id in seen_seeds:
            raise ValueError(f"Duplicate family_id {f_id} in base_seed_by_family")
        seen_seeds.add(f_id)

    if type(request.generation_template_by_family) is not tuple:
        raise ValueError("generation_template_by_family must be exactly a tuple")
    
    seen_gen = set()
    for item in request.generation_template_by_family:
        if type(item) is not tuple or len(item) != 2:
            raise ValueError("Each element in generation_template_by_family must be a tuple of length 2")
        f_id, gen_req = item
        if not isinstance(f_id, FamilyId):
            raise ValueError("family_id in generation_template_by_family must be a FamilyId enum instance")
        if not isinstance(gen_req, GenerationRequest):
            raise ValueError("gen_req in generation_template_by_family must be a GenerationRequest instance")
        if f_id in seen_gen:
            raise ValueError(f"Duplicate family_id {f_id} in generation_template_by_family")
        seen_gen.add(f_id)

    if not isinstance(request.simulation_template, SimulationRequest):
        raise ValueError("simulation_template must be a SimulationRequest instance")

    for f_id, count in request.sample_count_by_family:
        if count > 0:
            if f_id not in seen_gen:
                raise ValueError(f"Missing generation template for active family {f_id}")
            if f_id not in seen_seeds:
                raise ValueError(f"Missing base seed for active family {f_id}")

    if request.split_name == SplitName.ZERO_SHOT_TRAIN and request.enforce_zero_shot_c_train_exclusion:
        arma_garch_count = 0
        for f_id, count in request.sample_count_by_family:
            if f_id == FamilyId.ARMA_GARCH:
                arma_garch_count = count
        if arma_garch_count > 0:
            raise ValueError(
                f"C leakage detected: ARMA_GARCH count is {arma_garch_count} (> 0), "
                f"which is strictly prohibited in split {request.split_name.value}."
            )


def build_dataset_in_memory(request: DatasetBuildRequest) -> DatasetBuildResult:
    validate_dataset_build_request(request)

    gen_templates = dict(request.generation_template_by_family)
    base_seeds = dict(request.base_seed_by_family)

    samples = []

    for family_id, count in request.sample_count_by_family:
        if count <= 0:
            continue

        gen_template = gen_templates[family_id]
        base_seed = base_seeds[family_id]

        for i in range(count):
            sample_index = i + 1
            generation_seed = base_seed + i
            simulation_seed = base_seed + 1_000_000 + i

            sample_gen_req = replace(gen_template, seed=generation_seed)
            gen_res = generate_model_spec(sample_gen_req)
            spec = gen_res.spec

            sample_sim_req = replace(
                request.simulation_template,
                spec=spec,
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
                model_spec=spec,
                hash_len=request.sample_id_hash_len,
            )

            sample = DatasetSample(
                sample_id=sample_id,
                spec=spec,
                values=sim_res.values,
                innovations=sim_res.innovations,
                variances=sim_res.variances,
            )
            samples.append(sample)

    return DatasetBuildResult(
        samples=tuple(samples),
        split_name=request.split_name,
    )
