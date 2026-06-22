# tests/test_phase2_dataset.py

import pytest
import math
import json
import hashlib
from dataclasses import replace

from src.phase2.schema import FamilyId, ModelSpec, MeanFamily, VolatilityFamily, validate_model_spec
from src.phase2.constraints import require_math_valid
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


# ==============================================================================
# Fixtures / Helper builders
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
            provenance=(("phase", "p9"),),
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
            provenance=(("phase", "p9"),),
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
            provenance=(("phase", "p9"),),
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
            provenance=(("phase", "p9"),),
            max_attempts=100,
        )
    else:
        raise ValueError(f"Unknown family: {family_id}")


def get_valid_sim_template() -> SimulationRequest:
    # Build a valid SimulationRequest with a valid AR spec as placeholder
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


def make_build_request(
    protocol_name="p9-test",
    split_name=SplitName.SMOKE,
    sample_count_by_family=None,
    generation_template_by_family=None,
    simulation_template=None,
    base_seed_by_family=None,
    enforce_zero_shot_c_train_exclusion=True,
    sample_id_hash_len=12,
    **kwargs,
) -> DatasetBuildRequest:
    if sample_count_by_family is None:
        sample_count_by_family = ((FamilyId.AR, 2), (FamilyId.ARMA, 1))
    if generation_template_by_family is None:
        generation_template_by_family = (
            (FamilyId.AR, get_base_gen_req(FamilyId.AR)),
            (FamilyId.ARMA, get_base_gen_req(FamilyId.ARMA)),
            (FamilyId.GARCH, get_base_gen_req(FamilyId.GARCH)),
            (FamilyId.ARMA_GARCH, get_base_gen_req(FamilyId.ARMA_GARCH)),
        )
    if simulation_template is None:
        simulation_template = get_valid_sim_template()
    if base_seed_by_family is None:
        base_seed_by_family = (
            (FamilyId.AR, 1000),
            (FamilyId.ARMA, 2000),
            (FamilyId.GARCH, 3000),
            (FamilyId.ARMA_GARCH, 4000),
        )
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


# ==============================================================================
# A. SplitName Verification
# ==============================================================================

def test_split_name_values():
    expected_values = {
        "zero_shot_train",
        "zero_shot_eval",
        "fewshot_train",
        "fewshot_eval",
        "smoke",
    }
    assert {s.value for s in SplitName} == expected_values
    assert len(SplitName) == 5


# ==============================================================================
# B. DatasetSample field contract (tests 1-8)
# ==============================================================================

def test_dataset_sample_has_protocol_name():
    # 1. DatasetSample has protocol_name
    req = make_build_request(
        sample_count_by_family=((FamilyId.AR, 1),),
        base_seed_by_family=((FamilyId.AR, 100), (FamilyId.ARMA, 200), (FamilyId.GARCH, 300), (FamilyId.ARMA_GARCH, 400)),
    )
    res = build_dataset_in_memory(req)
    assert res.samples[0].protocol_name == "p9-test"


def test_dataset_sample_has_split_name():
    # 2. DatasetSample has split_name
    req = make_build_request(
        split_name=SplitName.ZERO_SHOT_EVAL,
        sample_count_by_family=((FamilyId.AR, 1),),
        base_seed_by_family=((FamilyId.AR, 100), (FamilyId.ARMA, 200), (FamilyId.GARCH, 300), (FamilyId.ARMA_GARCH, 400)),
    )
    res = build_dataset_in_memory(req)
    assert res.samples[0].split_name == SplitName.ZERO_SHOT_EVAL


def test_dataset_sample_has_family_id():
    # 3. DatasetSample has family_id
    req = make_build_request(
        sample_count_by_family=((FamilyId.ARMA, 1),),
        base_seed_by_family=((FamilyId.AR, 100), (FamilyId.ARMA, 200), (FamilyId.GARCH, 300), (FamilyId.ARMA_GARCH, 400)),
    )
    res = build_dataset_in_memory(req)
    assert res.samples[0].family_id == FamilyId.ARMA


def test_dataset_sample_has_sample_index():
    # 4. DatasetSample has sample_index (1-based)
    req = make_build_request(
        sample_count_by_family=((FamilyId.AR, 2),),
        base_seed_by_family=((FamilyId.AR, 100), (FamilyId.ARMA, 200), (FamilyId.GARCH, 300), (FamilyId.ARMA_GARCH, 400)),
    )
    res = build_dataset_in_memory(req)
    assert res.samples[0].sample_index == 1
    assert res.samples[1].sample_index == 2


def test_dataset_sample_has_generation_seed():
    # 5. DatasetSample has generation_seed (base + i, 0-based)
    req = make_build_request(
        sample_count_by_family=((FamilyId.AR, 2),),
        base_seed_by_family=((FamilyId.AR, 1000), (FamilyId.ARMA, 200), (FamilyId.GARCH, 300), (FamilyId.ARMA_GARCH, 400)),
    )
    res = build_dataset_in_memory(req)
    assert res.samples[0].generation_seed == 1000
    assert res.samples[1].generation_seed == 1001


def test_dataset_sample_has_simulation_seed():
    # 6. DatasetSample has simulation_seed (base + 1_000_000 + i)
    req = make_build_request(
        sample_count_by_family=((FamilyId.AR, 2),),
        base_seed_by_family=((FamilyId.AR, 1000), (FamilyId.ARMA, 200), (FamilyId.GARCH, 300), (FamilyId.ARMA_GARCH, 400)),
    )
    res = build_dataset_in_memory(req)
    assert res.samples[0].simulation_seed == 1_001_000
    assert res.samples[1].simulation_seed == 1_001_001


def test_dataset_sample_uses_model_spec_not_spec():
    # 7. DatasetSample uses model_spec field, not spec
    req = make_build_request(
        sample_count_by_family=((FamilyId.AR, 1),),
        base_seed_by_family=((FamilyId.AR, 100), (FamilyId.ARMA, 200), (FamilyId.GARCH, 300), (FamilyId.ARMA_GARCH, 400)),
    )
    res = build_dataset_in_memory(req)
    sample = res.samples[0]
    assert hasattr(sample, "model_spec")
    assert not hasattr(sample, "spec")
    assert isinstance(sample.model_spec, ModelSpec)


def test_dataset_sample_has_provenance():
    # 8. DatasetSample has provenance tuple
    req = make_build_request(
        sample_count_by_family=((FamilyId.AR, 1),),
        base_seed_by_family=((FamilyId.AR, 100), (FamilyId.ARMA, 200), (FamilyId.GARCH, 300), (FamilyId.ARMA_GARCH, 400)),
    )
    res = build_dataset_in_memory(req)
    sample = res.samples[0]
    assert isinstance(sample.provenance, tuple)
    assert len(sample.provenance) >= 5


# ==============================================================================
# C. DatasetBuildResult field contract (tests 9-13)
# ==============================================================================

def test_dataset_build_result_has_protocol_name():
    # 9. DatasetBuildResult has protocol_name
    req = make_build_request(protocol_name="my-protocol", sample_count_by_family=((FamilyId.AR, 1),), base_seed_by_family=((FamilyId.AR, 100), (FamilyId.ARMA, 200), (FamilyId.GARCH, 300), (FamilyId.ARMA_GARCH, 400)))
    res = build_dataset_in_memory(req)
    assert res.protocol_name == "my-protocol"


def test_dataset_build_result_has_total_count():
    # 10. DatasetBuildResult has total_count
    req = make_build_request(
        sample_count_by_family=((FamilyId.AR, 2), (FamilyId.ARMA, 1)),
        base_seed_by_family=((FamilyId.AR, 100), (FamilyId.ARMA, 200), (FamilyId.GARCH, 300), (FamilyId.ARMA_GARCH, 400)),
    )
    res = build_dataset_in_memory(req)
    assert res.total_count == 3


def test_dataset_build_result_has_count_by_family():
    # 11. DatasetBuildResult has count_by_family
    req = make_build_request(
        sample_count_by_family=((FamilyId.AR, 2), (FamilyId.ARMA, 1)),
        base_seed_by_family=((FamilyId.AR, 100), (FamilyId.ARMA, 200), (FamilyId.GARCH, 300), (FamilyId.ARMA_GARCH, 400)),
    )
    res = build_dataset_in_memory(req)
    assert isinstance(res.count_by_family, tuple)
    counts = dict(res.count_by_family)
    assert counts[FamilyId.AR] == 2
    assert counts[FamilyId.ARMA] == 1


def test_dataset_build_result_has_zero_shot_c_train_count():
    # 12. DatasetBuildResult has zero_shot_c_train_count
    req = make_build_request(
        split_name=SplitName.FEWSHOT_TRAIN,
        sample_count_by_family=((FamilyId.AR, 1), (FamilyId.ARMA_GARCH, 1)),
        base_seed_by_family=((FamilyId.AR, 100), (FamilyId.ARMA, 200), (FamilyId.GARCH, 300), (FamilyId.ARMA_GARCH, 400)),
    )
    res = build_dataset_in_memory(req)
    assert hasattr(res, "zero_shot_c_train_count")
    assert res.zero_shot_c_train_count == 1


def test_dataset_build_result_has_reason():
    # 13. DatasetBuildResult has reason
    req = make_build_request(
        sample_count_by_family=((FamilyId.AR, 1),),
        base_seed_by_family=((FamilyId.AR, 100), (FamilyId.ARMA, 200), (FamilyId.GARCH, 300), (FamilyId.ARMA_GARCH, 400)),
    )
    res = build_dataset_in_memory(req)
    assert res.reason == "dataset_built_in_memory"


# ==============================================================================
# D. Request Validation Edge Cases (tests 14-19)
# ==============================================================================

def test_protocol_name_empty_string_rejected():
    # 14. protocol_name empty string rejected
    with pytest.raises(ValueError) as exc:
        validate_dataset_build_request(make_build_request(protocol_name=""))
    assert "non-empty" in str(exc.value) or "protocol_name" in str(exc.value)

    with pytest.raises(ValueError):
        validate_dataset_build_request(make_build_request(protocol_name="   "))


def test_sample_id_hash_len_below_8_rejected():
    # 15. sample_id_hash_len < 8 rejected
    with pytest.raises(ValueError) as exc:
        validate_dataset_build_request(make_build_request(sample_id_hash_len=7))
    assert "sample_id_hash_len" in str(exc.value) or ">= 8" in str(exc.value)

    with pytest.raises(ValueError):
        validate_dataset_build_request(make_build_request(sample_id_hash_len=0))

    with pytest.raises(ValueError):
        validate_dataset_build_request(make_build_request(sample_id_hash_len=-1))

    # 8 exactly should be accepted
    validate_dataset_build_request(make_build_request(sample_id_hash_len=8))


def test_zero_total_count_rejected():
    # 16. zero total count rejected
    with pytest.raises(ValueError) as exc:
        validate_dataset_build_request(
            make_build_request(
                sample_count_by_family=((FamilyId.AR, 0), (FamilyId.ARMA, 0)),
            )
        )
    assert "count" in str(exc.value).lower() or "0" in str(exc.value)


def test_generation_template_family_mismatch_rejected():
    # 17. generation template family_id mismatch rejected
    # Create an ARMA template but assign it to the AR key
    arma_gen_req = get_base_gen_req(FamilyId.ARMA)
    with pytest.raises(ValueError) as exc:
        validate_dataset_build_request(
            make_build_request(
                generation_template_by_family=((FamilyId.AR, arma_gen_req),),
                sample_count_by_family=((FamilyId.AR, 1),),
            )
        )
    err = str(exc.value).lower()
    assert "mismatch" in err or "family_id" in err or "does not match" in err


def test_invalid_generation_request_template_rejected():
    # 18. invalid GenerationRequest template rejected through validate_generation_request
    # Build a GenerationRequest with bad parameters (p=0 for AR family)
    bad_gen_req = GenerationRequest(
        family_id=FamilyId.AR,
        seed=0,
        p=0,   # AR requires p > 0
        q=0, r=0, s=0,
        ar_range=(-0.4, 0.4),
        ma_range=(-0.4, 0.4),
        omega_range=(0.1, 1.0),
        alpha_range=(0.0, 0.4),
        beta_range=(0.0, 0.4),
        constraint_flags=(1.0, 0.0, 0.0, 0.0),
        provenance=(),
        max_attempts=100,
    )
    with pytest.raises(ValueError):
        validate_dataset_build_request(
            make_build_request(
                generation_template_by_family=((FamilyId.AR, bad_gen_req),),
                sample_count_by_family=((FamilyId.AR, 1),),
            )
        )


def test_invalid_simulation_request_template_rejected():
    # 19. invalid SimulationRequest template rejected through validate_simulation_request
    spec = ModelSpec(
        family_id=FamilyId.AR,
        mean_family=MeanFamily.AR,
        volatility_family=VolatilityFamily.NONE,
        p=1, q=0, r=0, s=0,
        ar_params=(0.1,), ma_params=(), omega=None, alpha_params=(), beta_params=(),
        constraint_flags=(1.0, 0.0, 0.0, 0.0), provenance=(),
    )
    bad_sim_req = SimulationRequest(
        spec=spec,
        seed=0,
        length=-10,  # invalid
        burn_in=0,
        innovation_distribution="gaussian",
        innovation_std=1.0,
        initial_value=0.0,
        initial_innovation=0.0,
        initial_variance=1.0,
        max_abs_value=1_000.0,
    )
    with pytest.raises(ValueError):
        validate_dataset_build_request(make_build_request(simulation_template=bad_sim_req))


# ==============================================================================
# E. Zero-Shot C Leakage Guard (tests 20-24)
# ==============================================================================

def test_zero_shot_train_c_count_gt_0_fails():
    # 20. zero_shot_train C count > 0 fails
    with pytest.raises(ValueError) as exc:
        validate_dataset_build_request(
            make_build_request(
                split_name=SplitName.ZERO_SHOT_TRAIN,
                sample_count_by_family=((FamilyId.ARMA_GARCH, 1),),
                enforce_zero_shot_c_train_exclusion=True,
            )
        )
    err = str(exc.value)
    assert "C leakage" in err
    assert "zero_shot_train" in err


def test_zero_shot_train_c_omitted_passes():
    # 21. zero_shot_train C omitted passes
    validate_dataset_build_request(
        make_build_request(
            split_name=SplitName.ZERO_SHOT_TRAIN,
            sample_count_by_family=((FamilyId.AR, 1),),
            enforce_zero_shot_c_train_exclusion=True,
        )
    )


def test_zero_shot_train_c_count_0_passes():
    # 22. zero_shot_train C count == 0 passes
    validate_dataset_build_request(
        make_build_request(
            split_name=SplitName.ZERO_SHOT_TRAIN,
            sample_count_by_family=((FamilyId.AR, 1), (FamilyId.ARMA_GARCH, 0)),
            enforce_zero_shot_c_train_exclusion=True,
        )
    )


def test_zero_shot_eval_c_count_gt_0_passes():
    # 23. zero_shot_eval C count > 0 passes
    validate_dataset_build_request(
        make_build_request(
            split_name=SplitName.ZERO_SHOT_EVAL,
            sample_count_by_family=((FamilyId.ARMA_GARCH, 1),),
            enforce_zero_shot_c_train_exclusion=True,
        )
    )


def test_fewshot_train_c_count_gt_0_passes():
    # 24. fewshot_train C count > 0 passes
    validate_dataset_build_request(
        make_build_request(
            split_name=SplitName.FEWSHOT_TRAIN,
            sample_count_by_family=((FamilyId.ARMA_GARCH, 1),),
            enforce_zero_shot_c_train_exclusion=True,
        )
    )


# ==============================================================================
# F. Generated specs validity (tests 25-26)
# ==============================================================================

def test_all_generated_model_specs_pass_validate_model_spec():
    # 25. all generated model_specs pass validate_model_spec
    req = make_build_request(
        sample_count_by_family=((FamilyId.AR, 2), (FamilyId.ARMA, 1)),
        base_seed_by_family=((FamilyId.AR, 1000), (FamilyId.ARMA, 2000), (FamilyId.GARCH, 3000), (FamilyId.ARMA_GARCH, 4000)),
    )
    res = build_dataset_in_memory(req)
    for sample in res.samples:
        validate_model_spec(sample.model_spec)


def test_all_generated_model_specs_pass_require_math_valid():
    # 26. all generated model_specs pass require_math_valid
    req = make_build_request(
        sample_count_by_family=((FamilyId.AR, 2), (FamilyId.GARCH, 1)),
        base_seed_by_family=((FamilyId.AR, 1000), (FamilyId.ARMA, 2000), (FamilyId.GARCH, 3000), (FamilyId.ARMA_GARCH, 4000)),
    )
    res = build_dataset_in_memory(req)
    for sample in res.samples:
        require_math_valid(sample.model_spec)


# ==============================================================================
# G. Uniqueness, Counts, Correctness (tests 27-30)
# ==============================================================================

def test_sample_ids_unique():
    # 27. sample IDs unique
    req = make_build_request(
        sample_count_by_family=((FamilyId.AR, 3), (FamilyId.ARMA, 2)),
        base_seed_by_family=((FamilyId.AR, 1000), (FamilyId.ARMA, 2000), (FamilyId.GARCH, 3000), (FamilyId.ARMA_GARCH, 4000)),
    )
    res = build_dataset_in_memory(req)
    ids = [s.sample_id for s in res.samples]
    assert len(ids) == len(set(ids)), "Sample IDs must all be unique"


def test_total_count_correct():
    # 28. total_count correct
    req = make_build_request(
        sample_count_by_family=((FamilyId.AR, 2), (FamilyId.ARMA, 3), (FamilyId.GARCH, 1)),
        base_seed_by_family=((FamilyId.AR, 1000), (FamilyId.ARMA, 2000), (FamilyId.GARCH, 3000), (FamilyId.ARMA_GARCH, 4000)),
    )
    res = build_dataset_in_memory(req)
    assert res.total_count == 6


def test_count_by_family_correct():
    # 29. count_by_family correct
    req = make_build_request(
        sample_count_by_family=((FamilyId.AR, 2), (FamilyId.ARMA, 3), (FamilyId.GARCH, 0)),
        base_seed_by_family=((FamilyId.AR, 1000), (FamilyId.ARMA, 2000), (FamilyId.GARCH, 3000), (FamilyId.ARMA_GARCH, 4000)),
    )
    res = build_dataset_in_memory(req)
    counts = dict(res.count_by_family)
    assert counts[FamilyId.AR] == 2
    assert counts[FamilyId.ARMA] == 3
    assert counts[FamilyId.GARCH] == 0


def test_zero_shot_c_train_count_correct():
    # 30. zero_shot_c_train_count correct
    req = make_build_request(
        split_name=SplitName.FEWSHOT_EVAL,
        sample_count_by_family=((FamilyId.AR, 1), (FamilyId.ARMA_GARCH, 2)),
        base_seed_by_family=((FamilyId.AR, 1000), (FamilyId.ARMA, 2000), (FamilyId.GARCH, 3000), (FamilyId.ARMA_GARCH, 4000)),
    )
    res = build_dataset_in_memory(req)
    assert res.zero_shot_c_train_count == 2


# ==============================================================================
# H. Provenance contract (tests 31-36)
# ==============================================================================

def test_sample_provenance_contains_protocol_name():
    # 31. sample provenance contains protocol_name
    req = make_build_request(
        protocol_name="p9-fix-test",
        sample_count_by_family=((FamilyId.AR, 1),),
        base_seed_by_family=((FamilyId.AR, 100), (FamilyId.ARMA, 200), (FamilyId.GARCH, 300), (FamilyId.ARMA_GARCH, 400)),
    )
    res = build_dataset_in_memory(req)
    prov_dict = dict(res.samples[0].provenance)
    assert "protocol_name" in prov_dict
    assert prov_dict["protocol_name"] == "p9-fix-test"


def test_sample_provenance_contains_split_name():
    # 32. sample provenance contains split_name
    req = make_build_request(
        split_name=SplitName.ZERO_SHOT_EVAL,
        sample_count_by_family=((FamilyId.AR, 1),),
        base_seed_by_family=((FamilyId.AR, 100), (FamilyId.ARMA, 200), (FamilyId.GARCH, 300), (FamilyId.ARMA_GARCH, 400)),
    )
    res = build_dataset_in_memory(req)
    prov_dict = dict(res.samples[0].provenance)
    assert "split_name" in prov_dict
    assert prov_dict["split_name"] == "zero_shot_eval"


def test_sample_provenance_contains_generation_seed():
    # 33. sample provenance contains generation_seed
    req = make_build_request(
        sample_count_by_family=((FamilyId.AR, 2),),
        base_seed_by_family=((FamilyId.AR, 1000), (FamilyId.ARMA, 200), (FamilyId.GARCH, 300), (FamilyId.ARMA_GARCH, 400)),
    )
    res = build_dataset_in_memory(req)
    prov0 = dict(res.samples[0].provenance)
    prov1 = dict(res.samples[1].provenance)
    assert prov0["generation_seed"] == "1000"
    assert prov1["generation_seed"] == "1001"


def test_sample_provenance_contains_simulation_seed():
    # 34. sample provenance contains simulation_seed
    req = make_build_request(
        sample_count_by_family=((FamilyId.AR, 1),),
        base_seed_by_family=((FamilyId.AR, 1000), (FamilyId.ARMA, 200), (FamilyId.GARCH, 300), (FamilyId.ARMA_GARCH, 400)),
    )
    res = build_dataset_in_memory(req)
    prov_dict = dict(res.samples[0].provenance)
    assert prov_dict["simulation_seed"] == "1001000"


def test_sample_provenance_contains_sample_index():
    # 35. sample provenance contains sample_index
    req = make_build_request(
        sample_count_by_family=((FamilyId.AR, 2),),
        base_seed_by_family=((FamilyId.AR, 100), (FamilyId.ARMA, 200), (FamilyId.GARCH, 300), (FamilyId.ARMA_GARCH, 400)),
    )
    res = build_dataset_in_memory(req)
    prov0 = dict(res.samples[0].provenance)
    prov1 = dict(res.samples[1].provenance)
    assert prov0["sample_index"] == "1"
    assert prov1["sample_index"] == "2"


def test_original_model_spec_provenance_not_mutated():
    # 36. original ModelSpec provenance not mutated
    # The generated spec's provenance should remain as the original generator set it
    req = make_build_request(
        sample_count_by_family=((FamilyId.AR, 1),),
        base_seed_by_family=((FamilyId.AR, 100), (FamilyId.ARMA, 200), (FamilyId.GARCH, 300), (FamilyId.ARMA_GARCH, 400)),
    )
    res = build_dataset_in_memory(req)
    sample = res.samples[0]
    # The model_spec.provenance is what was in the sampler output; it comes from the GenerationRequest
    # which has provenance=(("phase", "p9"),), so the model_spec.provenance should only have that entry
    model_spec_prov_keys = [k for k, _ in sample.model_spec.provenance]
    # model_spec provenance must NOT contain builder-injected keys
    assert "protocol_name" not in model_spec_prov_keys
    assert "split_name" not in model_spec_prov_keys
    assert "generation_seed" not in model_spec_prov_keys
    assert "simulation_seed" not in model_spec_prov_keys
    assert "sample_index" not in model_spec_prov_keys
    # The sample provenance includes the model_spec prov AND builder entries
    all_prov_keys = [k for k, _ in sample.provenance]
    assert "protocol_name" in all_prov_keys
    assert "split_name" in all_prov_keys


# ==============================================================================
# I. Hashing and ID Correctness
# ==============================================================================

def test_make_sample_id_format_and_hash():
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

    prefix = "AR_zero_shot_eval_000005_g12003_s1012003_"
    assert sample_id.startswith(prefix)
    hash_part = sample_id[len(prefix):]
    assert len(hash_part) == 12

    # Verify SHA256 hash manually
    expected_payload = {
        "protocol_name": "my-protocol",
        "split_name": "zero_shot_eval",
        "family_id": "AR",
        "sample_index": 5,
        "generation_seed": 12003,
        "simulation_seed": 1012003,
        "p": 1, "q": 0, "r": 0, "s": 0,
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


def test_make_sample_id_hash_len_below_8_rejected():
    spec = ModelSpec(
        family_id=FamilyId.AR,
        mean_family=MeanFamily.AR,
        volatility_family=VolatilityFamily.NONE,
        p=1, q=0, r=0, s=0,
        ar_params=(0.2,), ma_params=(), omega=None, alpha_params=(), beta_params=(),
        constraint_flags=(1.0, 0.0, 0.0, 0.0), provenance=(),
    )
    with pytest.raises(ValueError):
        make_sample_id("p", SplitName.SMOKE, FamilyId.AR, 1, 100, 200, spec, 7)


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

    with pytest.raises(ValueError):
        make_sample_id("protocol", SplitName.SMOKE, FamilyId.AR, 1, 100, 200, spec, True)


# ==============================================================================
# J. Full End-to-End Builds
# ==============================================================================

def test_build_all_four_families():
    req = make_build_request(
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
    families = [s.family_id for s in res.samples]
    assert FamilyId.AR in families
    assert FamilyId.ARMA in families
    assert FamilyId.GARCH in families
    assert FamilyId.ARMA_GARCH in families


def test_build_seed_policy_correctness():
    # Seed policy: generation_seed = base + i, simulation_seed = base + 1_000_000 + i (0-based i)
    req = make_build_request(
        sample_count_by_family=((FamilyId.AR, 3),),
        base_seed_by_family=((FamilyId.AR, 500), (FamilyId.ARMA, 200), (FamilyId.GARCH, 300), (FamilyId.ARMA_GARCH, 400)),
    )
    res = build_dataset_in_memory(req)
    for i, sample in enumerate(res.samples):
        expected_gen_seed = 500 + i
        expected_sim_seed = 500 + 1_000_000 + i
        assert sample.generation_seed == expected_gen_seed, f"Sample {i}: generation_seed mismatch"
        assert sample.simulation_seed == expected_sim_seed, f"Sample {i}: simulation_seed mismatch"
        assert sample.sample_index == i + 1, f"Sample {i}: sample_index mismatch"


def test_build_values_length():
    req = make_build_request(
        sample_count_by_family=((FamilyId.AR, 1),),
        base_seed_by_family=((FamilyId.AR, 100), (FamilyId.ARMA, 200), (FamilyId.GARCH, 300), (FamilyId.ARMA_GARCH, 400)),
    )
    res = build_dataset_in_memory(req)
    sample = res.samples[0]
    assert len(sample.values) == req.simulation_template.length
    assert len(sample.innovations) == req.simulation_template.length
    assert len(sample.variances) == req.simulation_template.length
    assert isinstance(sample.values, tuple)
    assert isinstance(sample.innovations, tuple)
    assert isinstance(sample.variances, tuple)


# ==============================================================================
# K. Tuple-Structured Field Validation
# ==============================================================================

def test_validate_tuple_fields_not_lists():
    with pytest.raises(ValueError) as exc:
        validate_dataset_build_request(make_build_request(sample_count_by_family=[(FamilyId.AR, 1)]))
    assert "sample_count_by_family must be exactly a tuple" in str(exc.value)

    with pytest.raises(ValueError) as exc:
        validate_dataset_build_request(make_build_request(base_seed_by_family=[(FamilyId.AR, 100)]))
    assert "base_seed_by_family must be exactly a tuple" in str(exc.value)

    with pytest.raises(ValueError) as exc:
        validate_dataset_build_request(make_build_request(
            generation_template_by_family=[(FamilyId.AR, get_base_gen_req(FamilyId.AR))]
        ))
    assert "generation_template_by_family must be exactly a tuple" in str(exc.value)


def test_validate_count_bool_rejected():
    with pytest.raises(ValueError) as exc:
        validate_dataset_build_request(make_build_request(
            sample_count_by_family=((FamilyId.AR, True),)
        ))
    assert "count in sample_count_by_family must be a non-negative int" in str(exc.value)


def test_validate_seed_bool_rejected():
    with pytest.raises(ValueError) as exc:
        validate_dataset_build_request(make_build_request(
            base_seed_by_family=((FamilyId.AR, True), (FamilyId.ARMA, 200), (FamilyId.GARCH, 300), (FamilyId.ARMA_GARCH, 400))
        ))
    assert "seed in base_seed_by_family must be an int, not bool" in str(exc.value)


def test_validate_duplicate_family_rejected():
    with pytest.raises(ValueError) as exc:
        validate_dataset_build_request(make_build_request(
            sample_count_by_family=((FamilyId.AR, 1), (FamilyId.AR, 2))
        ))
    assert "Duplicate family_id" in str(exc.value)


# ==============================================================================
# L. Scope: no torch, no numpy, no files (tests 37-40)
# ==============================================================================

def test_no_torch_imported_by_dataset():
    # 38. dataset.py does not import torch directly
    import subprocess
    import sys
    script = "import sys; import src.phase2.dataset; sys.exit(1 if 'torch' in sys.modules else 0)"
    result = subprocess.run([sys.executable, "-c", script], capture_output=True)
    assert result.returncode == 0, "torch was imported by phase2.dataset"


def test_no_direct_numpy_in_dataset():
    # 37. dataset.py does not import numpy directly
    import os
    dataset_path = os.path.join("src", "phase2", "dataset.py")
    with open(dataset_path, "r", encoding="utf-8") as f:
        content = f.read()

    assert "import numpy" not in content, "Direct 'import numpy' found in dataset.py"
    assert "from numpy" not in content, "Direct 'from numpy' found in dataset.py"


def test_no_file_io_in_dataset():
    # 39. builder does not create files
    import os
    dataset_path = os.path.join("src", "phase2", "dataset.py")
    with open(dataset_path, "r", encoding="utf-8") as f:
        content = f.read()

    forbidden_calls = ["open(", ".write(", ".to_csv", ".to_json", "save(", "Path("]
    for call in forbidden_calls:
        assert call not in content, f"Forbidden file operation '{call}' found in dataset.py"


def test_no_config_required():
    # 40. builder does not require configs
    # Just verifying dataset.py does not import from configs module
    import os
    dataset_path = os.path.join("src", "phase2", "dataset.py")
    with open(dataset_path, "r", encoding="utf-8") as f:
        content = f.read()

    assert "configs" not in content, "'configs' module import found in dataset.py"
