# tests/test_phase2_dataset.py

import pytest
import math
import json
import hashlib
from dataclasses import replace

from src.phase2.schema import FamilyId, ModelSpec, MeanFamily, VolatilityFamily
from src.phase2.sampler import GenerationRequest
from src.phase2.simulator import SimulationRequest
from src.phase2.dataset import (
    SplitName,
    DatasetBuildRequest,
    DatasetSample,
    DatasetBuildResult,
    make_sample_id,
    validate_dataset_build_request,
    build_dataset_in_memory,
)


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
            provenance=(("key", "val"),),
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
            provenance=(("key", "val"),),
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
            provenance=(("key", "val"),),
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
            provenance=(("key", "val"),),
            max_attempts=100,
        )
    else:
        raise ValueError(f"Unknown family: {family_id}")


def get_base_sim_req() -> SimulationRequest:
    # Dummy ModelSpec that is mathematically valid
    dummy_spec = ModelSpec(
        family_id=FamilyId.AR,
        mean_family=MeanFamily.AR,
        volatility_family=VolatilityFamily.NONE,
        p=1, q=0, r=0, s=0,
        ar_params=(0.1,), ma_params=(), omega=None, alpha_params=(), beta_params=(),
        constraint_flags=(1.0, 0.0, 0.0, 0.0), provenance=(),
    )
    return SimulationRequest(
        spec=dummy_spec,
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


def make_dataset_build_request(
    protocol_name="p9-test",
    split_name=SplitName.SMOKE,
    sample_count_by_family=((FamilyId.AR, 2), (FamilyId.ARMA, 1)),
    generation_template_by_family=None,
    simulation_template=None,
    base_seed_by_family=((FamilyId.AR, 100), (FamilyId.ARMA, 200)),
    enforce_zero_shot_c_train_exclusion=True,
    sample_id_hash_len=12,
    **kwargs,
) -> DatasetBuildRequest:
    if generation_template_by_family is None:
        generation_template_by_family = (
            (FamilyId.AR, get_base_gen_req(FamilyId.AR)),
            (FamilyId.ARMA, get_base_gen_req(FamilyId.ARMA)),
            (FamilyId.GARCH, get_base_gen_req(FamilyId.GARCH)),
            (FamilyId.ARMA_GARCH, get_base_gen_req(FamilyId.ARMA_GARCH)),
        )
    if simulation_template is None:
        simulation_template = get_base_sim_req()

    params = {
        "protocol_name": protocol_name,
        "split_name": split_name,
        "sample_count_by_family": sample_count_by_family,
        "generation_template_by_family": generation_template_by_family,
        "simulation_template": simulation_template,
        "base_seed_by_family": base_seed_by_family,
        "enforce_zero_shot_c_train_exclusion": enforce_zero_shot_c_train_exclusion,
        "sample_id_hash_len": sample_id_hash_len,
    }
    params.update(kwargs)
    return DatasetBuildRequest(**params)


# --- A. SplitName Verification ---

def test_split_name_values():
    expected_values = {
        "zero_shot_train",
        "zero_shot_eval",
        "fewshot_train",
        "fewshot_eval",
        "smoke",
    }
    actual_values = {s.value for s in SplitName}
    assert actual_values == expected_values
    assert len(SplitName) == 5


# --- B. Validation Checks ---

def test_validate_request_types():
    # protocol_name must be a string
    with pytest.raises(ValueError) as exc:
        validate_dataset_build_request(make_dataset_build_request(protocol_name=123))
    assert "protocol_name must be a string" in str(exc.value)

    # split_name must be a SplitName
    with pytest.raises(ValueError) as exc:
        validate_dataset_build_request(make_dataset_build_request(split_name="smoke"))
    assert "split_name must be a SplitName enum instance" in str(exc.value)

    # enforce_zero_shot_c_train_exclusion must be a bool
    with pytest.raises(ValueError) as exc:
        validate_dataset_build_request(make_dataset_build_request(enforce_zero_shot_c_train_exclusion="True"))
    assert "enforce_zero_shot_c_train_exclusion must be a bool" in str(exc.value)

    # sample_id_hash_len must be a positive int
    with pytest.raises(ValueError) as exc:
        validate_dataset_build_request(make_dataset_build_request(sample_id_hash_len="12"))
    assert "sample_id_hash_len must be a positive int" in str(exc.value)

    with pytest.raises(ValueError) as exc:
        validate_dataset_build_request(make_dataset_build_request(sample_id_hash_len=-1))
    assert "sample_id_hash_len must be a positive int" in str(exc.value)


def test_validate_tuple_structured_fields():
    # Check sample_count_by_family is tuple, not list
    with pytest.raises(ValueError) as exc:
        validate_dataset_build_request(make_dataset_build_request(sample_count_by_family=[(FamilyId.AR, 1)]))
    assert "sample_count_by_family must be exactly a tuple" in str(exc.value)

    # Check duplicate family in sample_count_by_family
    with pytest.raises(ValueError) as exc:
        validate_dataset_build_request(make_dataset_build_request(sample_count_by_family=((FamilyId.AR, 1), (FamilyId.AR, 2))))
    assert "Duplicate family_id" in str(exc.value)

    # Check count is non-negative int, not bool or other type
    with pytest.raises(ValueError) as exc:
        validate_dataset_build_request(make_dataset_build_request(sample_count_by_family=((FamilyId.AR, True),)))
    assert "count in sample_count_by_family must be a non-negative int" in str(exc.value)

    # Check base_seed_by_family is tuple, not list
    with pytest.raises(ValueError) as exc:
        validate_dataset_build_request(make_dataset_build_request(base_seed_by_family=[(FamilyId.AR, 100)]))
    assert "base_seed_by_family must be exactly a tuple" in str(exc.value)

    # Check generation_template_by_family is tuple, not list
    with pytest.raises(ValueError) as exc:
        validate_dataset_build_request(make_dataset_build_request(generation_template_by_family=[(FamilyId.AR, get_base_gen_req(FamilyId.AR))]))
    assert "generation_template_by_family must be exactly a tuple" in str(exc.value)


def test_missing_templates_or_seeds():
    # Family AR has count > 0, but no generation template is provided for AR
    with pytest.raises(ValueError) as exc:
        validate_dataset_build_request(
            make_dataset_build_request(
                sample_count_by_family=((FamilyId.AR, 1),),
                generation_template_by_family=((FamilyId.ARMA, get_base_gen_req(FamilyId.ARMA)),),
                base_seed_by_family=((FamilyId.AR, 100),),
            )
        )
    assert "Missing generation template for active family" in str(exc.value)

    # Family AR has count > 0, but no base seed is provided for AR
    with pytest.raises(ValueError) as exc:
        validate_dataset_build_request(
            make_dataset_build_request(
                sample_count_by_family=((FamilyId.AR, 1),),
                generation_template_by_family=((FamilyId.AR, get_base_gen_req(FamilyId.AR)),),
                base_seed_by_family=((FamilyId.ARMA, 100),),
            )
        )
    assert "Missing base seed for active family" in str(exc.value)


# --- C. Zero-Shot C Leakage Guard ---

def test_zero_shot_c_leakage_guard():
    # ZERO_SHOT_TRAIN split + enforce_zero_shot_c_train_exclusion == True
    # If ARMA_GARCH count > 0, must raise ValueError containing "C leakage" and "zero_shot_train"
    with pytest.raises(ValueError) as exc:
        validate_dataset_build_request(
            make_dataset_build_request(
                split_name=SplitName.ZERO_SHOT_TRAIN,
                sample_count_by_family=((FamilyId.ARMA_GARCH, 1),),
                base_seed_by_family=((FamilyId.ARMA_GARCH, 100),),
                enforce_zero_shot_c_train_exclusion=True,
            )
        )
    err_str = str(exc.value)
    assert "C leakage" in err_str
    assert "zero_shot_train" in err_str

    # ARMA_GARCH count == 0 is allowed
    validate_dataset_build_request(
        make_dataset_build_request(
            split_name=SplitName.ZERO_SHOT_TRAIN,
            sample_count_by_family=((FamilyId.ARMA_GARCH, 0),),
            base_seed_by_family=((FamilyId.ARMA_GARCH, 100),),
            enforce_zero_shot_c_train_exclusion=True,
        )
    )

    # ARMA_GARCH omitted is allowed
    validate_dataset_build_request(
        make_dataset_build_request(
            split_name=SplitName.ZERO_SHOT_TRAIN,
            sample_count_by_family=((FamilyId.AR, 2),),
            base_seed_by_family=((FamilyId.AR, 100),),
            enforce_zero_shot_c_train_exclusion=True,
        )
    )

    # Other splits (e.g. ZERO_SHOT_EVAL) are allowed to have ARMA_GARCH > 0
    validate_dataset_build_request(
        make_dataset_build_request(
            split_name=SplitName.ZERO_SHOT_EVAL,
            sample_count_by_family=((FamilyId.ARMA_GARCH, 1),),
            base_seed_by_family=((FamilyId.ARMA_GARCH, 100),),
            enforce_zero_shot_c_train_exclusion=True,
        )
    )


# --- D. Hashing and ID Correctness ---

def test_make_sample_id_correctness():
    spec = ModelSpec(
        family_id=FamilyId.AR,
        mean_family=MeanFamily.AR,
        volatility_family=VolatilityFamily.NONE,
        p=1, q=0, r=0, s=0,
        ar_params=(0.2,), ma_params=(), omega=None, alpha_params=(), beta_params=(),
        constraint_flags=(1.0, 0.0, 0.0, 0.0), provenance=(("key", "value"),),
    )
    
    sample_id = make_sample_id(
        protocol_name="my-protocol",
        split_name=SplitName.ZERO_SHOT_EVAL,
        family_id=FamilyId.AR,
        sample_index=5,
        generation_seed=12003,
        simulation_seed=1012003,
        model_spec=spec,
        hash_len=12,
    )
    
    # Expected format: AR_zero_shot_eval_000005_g12003_s1012003_{hash}
    prefix = "AR_zero_shot_eval_000005_g12003_s1012003_"
    assert sample_id.startswith(prefix)
    hash_part = sample_id[len(prefix):]
    assert len(hash_part) == 12

    # Manually compute the hash payload and verify matching hash
    expected_payload = {
        "protocol_name": "my-protocol",
        "split_name": "zero_shot_eval",
        "family_id": "AR",
        "sample_index": 5,
        "generation_seed": 12003,
        "simulation_seed": 1012003,
        "p": 1,
        "q": 0,
        "r": 0,
        "s": 0,
        "ar_params": [0.2],
        "ma_params": [],
        "omega": None,
        "alpha_params": [],
        "beta_params": [],
        "constraint_flags": [1.0, 0.0, 0.0, 0.0],
        "provenance": [["key", "value"]],
    }
    payload_json = json.dumps(expected_payload, separators=(",", ":"), sort_keys=True)
    expected_hash = hashlib.sha256(payload_json.encode("utf-8")).hexdigest()[:12]
    assert hash_part == expected_hash


def test_make_sample_id_invalid_parameters():
    spec = ModelSpec(
        family_id=FamilyId.AR,
        mean_family=MeanFamily.AR,
        volatility_family=VolatilityFamily.NONE,
        p=1, q=0, r=0, s=0,
        ar_params=(0.2,), ma_params=(), omega=None, alpha_params=(), beta_params=(),
        constraint_flags=(1.0, 0.0, 0.0, 0.0), provenance=(),
    )
    
    with pytest.raises(ValueError):
        make_sample_id("protocol", "not_a_splitname", FamilyId.AR, 1, 100, 200, spec, 12)
        
    with pytest.raises(ValueError):
        make_sample_id("protocol", SplitName.SMOKE, FamilyId.AR, -1, 100, 200, spec, 12)

    with pytest.raises(ValueError):
        make_sample_id("protocol", SplitName.SMOKE, FamilyId.AR, 1, True, 200, spec, 12)


# --- E. Dataset Build End-to-End ---

def test_build_dataset_in_memory_success():
    req = make_dataset_build_request(
        sample_count_by_family=((FamilyId.AR, 2), (FamilyId.ARMA, 1)),
        base_seed_by_family=((FamilyId.AR, 1000), (FamilyId.ARMA, 2000)),
    )
    
    res = build_dataset_in_memory(req)
    
    assert isinstance(res, DatasetBuildResult)
    assert res.split_name == SplitName.SMOKE
    assert isinstance(res.samples, tuple)
    assert len(res.samples) == 3

    # Check sample details for Family AR
    # Sample index 1: generation_seed = 1000, simulation_seed = 1001000
    # Sample index 2: generation_seed = 1001, simulation_seed = 1001001
    sample0 = res.samples[0]
    assert sample0.spec.family_id == FamilyId.AR
    assert "AR_smoke_000001_g1000_s1001000_" in sample0.sample_id
    assert isinstance(sample0.values, tuple)
    assert len(sample0.values) == req.simulation_template.length
    assert len(sample0.innovations) == req.simulation_template.length
    assert len(sample0.variances) == req.simulation_template.length

    sample1 = res.samples[1]
    assert sample1.spec.family_id == FamilyId.AR
    assert "AR_smoke_000002_g1001_s1001001_" in sample1.sample_id

    # Check sample details for Family ARMA
    # Sample index 1: generation_seed = 2000, simulation_seed = 1002000
    sample2 = res.samples[2]
    assert sample2.spec.family_id == FamilyId.ARMA
    assert "ARMA_smoke_000001_g2000_s1002000_" in sample2.sample_id


def test_build_all_four_families():
    req = make_dataset_build_request(
        split_name=SplitName.FEWSHOT_EVAL,
        sample_count_by_family=(
            (FamilyId.AR, 1),
            (FamilyId.ARMA, 1),
            (FamilyId.GARCH, 1),
            (FamilyId.ARMA_GARCH, 1),
        ),
        base_seed_by_family=(
            (FamilyId.AR, 1000),
            (FamilyId.ARMA, 2000),
            (FamilyId.GARCH, 3000),
            (FamilyId.ARMA_GARCH, 4000),
        ),
    )
    
    res = build_dataset_in_memory(req)
    assert len(res.samples) == 4
    for i, family_id in enumerate([FamilyId.AR, FamilyId.ARMA, FamilyId.GARCH, FamilyId.ARMA_GARCH]):
        assert res.samples[i].spec.family_id == family_id


# --- F. Code Constraints & Scope ---

def test_no_torch_imported():
    import subprocess
    import sys
    script = "import sys; import src.phase2.dataset; sys.exit(1 if 'torch' in sys.modules else 0)"
    result = subprocess.run([sys.executable, "-c", script], capture_output=True)
    assert result.returncode == 0, "torch was imported by phase2.dataset"


def test_no_direct_numpy_in_dataset():
    import os
    dataset_path = os.path.join("src", "phase2", "dataset.py")
    with open(dataset_path, "r", encoding="utf-8") as f:
        content = f.read()
    
    assert "import numpy" not in content, "Direct 'import numpy' found in dataset.py"
    assert "from numpy" not in content, "Direct 'from numpy' found in dataset.py"


def test_no_file_io_in_dataset_builder():
    # Verify that build_dataset_in_memory does not write to any file
    # (Checking content of dataset.py for file operations)
    import os
    dataset_path = os.path.join("src", "phase2", "dataset.py")
    with open(dataset_path, "r", encoding="utf-8") as f:
        content = f.read()
    
    forbidden_calls = ["open(", "write(", ".to_csv", ".to_json", "save("]
    for call in forbidden_calls:
        assert call not in content, f"Forbidden call '{call}' found in dataset.py"
