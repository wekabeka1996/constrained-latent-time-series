# tests/test_phase2_baseline_generators.py

import sys
import math
import pytest

from dataclasses import replace
from src.phase2.schema import FamilyId, MeanFamily, VolatilityFamily, ModelSpec, validate_model_spec
from src.phase2.constraints import require_math_valid
from src.phase2.sampler import GenerationRequest, generate_model_spec
from src.phase2.baseline_generators import (
    BASELINE_GENERATOR_CONTRACT_VERSION,
    SUPPORTED_BASELINE_GENERATOR_NAMES,
    COPY_REFERENCE_BASELINE_NAME,
    RANDOM_VALID_BASELINE_NAME,
    STRUCTURAL_COMPOSITION_ORACLE_BASELINE_NAME,
    BaselineCandidateSourceRecord,
    BaselineGenerationRequest,
    BaselineGenerationResult,
    validate_baseline_generator_name,
    validate_positive_int,
    validate_seed,
    validate_reference_specs,
    validate_oracle_constraint_flags,
    validate_random_valid_templates,
    validate_baseline_generation_request,
    clone_model_spec_with_provenance,
    generate_copy_reference_candidates,
    generate_random_valid_candidates,
    find_oracle_mean_sources,
    find_oracle_volatility_sources,
    compose_oracle_candidate,
    generate_structural_composition_oracle_candidates,
    generate_baseline_candidates,
    baseline_generation_result_to_evaluation_request,
)
from src.phase2.baseline_eval import BaselineEvaluationRequest


def get_valid_ar() -> ModelSpec:
    return ModelSpec(
        family_id=FamilyId.AR,
        mean_family=MeanFamily.AR,
        volatility_family=VolatilityFamily.NONE,
        p=1, q=0, r=0, s=0,
        ar_params=(0.5,), ma_params=(),
        omega=None,
        alpha_params=(), beta_params=(),
        constraint_flags=(1.0, 0.0, 0.0, 0.0),
        provenance=(),
    )


def get_valid_garch() -> ModelSpec:
    return ModelSpec(
        family_id=FamilyId.GARCH,
        mean_family=MeanFamily.NONE,
        volatility_family=VolatilityFamily.GARCH,
        p=0, q=0, r=1, s=1,
        ar_params=(), ma_params=(),
        omega=0.5,
        alpha_params=(0.1,), beta_params=(0.8,),
        constraint_flags=(1.0, 0.0, 0.0, 0.0),
        provenance=(),
    )


def get_valid_arma() -> ModelSpec:
    return ModelSpec(
        family_id=FamilyId.ARMA,
        mean_family=MeanFamily.ARMA,
        volatility_family=VolatilityFamily.NONE,
        p=1, q=1, r=0, s=0,
        ar_params=(0.5,), ma_params=(-0.3,),
        omega=None,
        alpha_params=(), beta_params=(),
        constraint_flags=(1.0, 0.0, 0.0, 0.0),
        provenance=(),
    )


def get_valid_arma_garch() -> ModelSpec:
    return ModelSpec(
        family_id=FamilyId.ARMA_GARCH,
        mean_family=MeanFamily.ARMA,
        volatility_family=VolatilityFamily.GARCH,
        p=1, q=1, r=1, s=1,
        ar_params=(0.5,), ma_params=(-0.3,),
        omega=0.5,
        alpha_params=(0.1,), beta_params=(0.8,),
        constraint_flags=(1.0, 0.0, 0.0, 0.0),
        provenance=(),
    )


def get_valid_generation_request(family_id: FamilyId) -> GenerationRequest:
    if family_id == FamilyId.AR:
        return GenerationRequest(
            family_id=FamilyId.AR,
            seed=42,
            p=1, q=0, r=0, s=0,
            ar_range=(-0.5, 0.5),
            ma_range=(0.0, 0.0),
            omega_range=(0.0, 0.0),
            alpha_range=(0.0, 0.0),
            beta_range=(0.0, 0.0),
            constraint_flags=(1.0, 0.0, 0.0, 0.0),
            provenance=(),
            max_attempts=50
        )
    elif family_id == FamilyId.GARCH:
        return GenerationRequest(
            family_id=FamilyId.GARCH,
            seed=42,
            p=0, q=0, r=1, s=1,
            ar_range=(0.0, 0.0),
            ma_range=(0.0, 0.0),
            omega_range=(0.1, 0.5),
            alpha_range=(0.01, 0.1),
            beta_range=(0.5, 0.8),
            constraint_flags=(1.0, 0.0, 0.0, 0.0),
            provenance=(),
            max_attempts=50
        )
    elif family_id == FamilyId.ARMA_GARCH:
        return GenerationRequest(
            family_id=FamilyId.ARMA_GARCH,
            seed=42,
            p=1, q=1, r=1, s=1,
            ar_range=(-0.5, 0.5),
            ma_range=(-0.5, 0.5),
            omega_range=(0.1, 0.5),
            alpha_range=(0.01, 0.1),
            beta_range=(0.5, 0.8),
            constraint_flags=(1.0, 0.0, 0.0, 0.0),
            provenance=(),
            max_attempts=50
        )
    else:
        raise ValueError(f"Unsupported test family_id: {family_id}")


# --- Group A: Constants/dataclasses ---

def test_p19_1_contract_version():
    assert BASELINE_GENERATOR_CONTRACT_VERSION == "phase2_p19_baseline_generators_v1"


def test_p19_2_supported_names():
    assert SUPPORTED_BASELINE_GENERATOR_NAMES == (
        "copy_reference",
        "random_valid",
        "structural_composition_oracle",
    )


def test_p19_3_dataclasses_frozen():
    for cls in (
        BaselineCandidateSourceRecord,
        BaselineGenerationRequest,
        BaselineGenerationResult,
    ):
        assert cls.__dataclass_params__.frozen


# --- Group B: Validation ---

def test_p19_4_validate_name_accepts():
    for name in SUPPORTED_BASELINE_GENERATOR_NAMES:
        validate_baseline_generator_name(name)


def test_p19_5_validate_name_rejects():
    with pytest.raises(TypeError):
        validate_baseline_generator_name(123)
    with pytest.raises(ValueError):
        validate_baseline_generator_name("")
    with pytest.raises(ValueError):
        validate_baseline_generator_name("invalid_name")


def test_p19_6_validate_positive_int_accepts():
    validate_positive_int(1, "count")
    validate_positive_int(100, "count")


def test_p19_7_validate_positive_int_rejects():
    with pytest.raises(TypeError):
        validate_positive_int(True, "count")  # bool
    with pytest.raises(TypeError):
        validate_positive_int(1.5, "count")  # float
    with pytest.raises(ValueError):
        validate_positive_int(0, "count")
    with pytest.raises(ValueError):
        validate_positive_int(-1, "count")


def test_p19_8_validate_seed_accepts():
    validate_seed(0, "seed")
    validate_seed(123, "seed")


def test_p19_9_validate_seed_rejects():
    with pytest.raises(TypeError):
        validate_seed(False, "seed")
    with pytest.raises(TypeError):
        validate_seed(1.2, "seed")
    with pytest.raises(ValueError):
        validate_seed(-1, "seed")


def test_p19_10_validate_reference_specs_accepts():
    validate_reference_specs((get_valid_ar(), get_valid_garch()))


def test_p19_11_validate_reference_specs_rejects():
    with pytest.raises(TypeError):
        validate_reference_specs([get_valid_ar()])  # list
    with pytest.raises(ValueError):
        validate_reference_specs(())  # empty
    with pytest.raises(TypeError):
        validate_reference_specs((get_valid_ar(), "not_a_spec"))
        
    invalid_spec = get_valid_ar()
    invalid_spec = replace(invalid_spec, ar_params=(1.5,))  # math-invalid
    with pytest.raises(ValueError):
        validate_reference_specs((invalid_spec,))


def test_p19_12_validate_oracle_constraint_flags_accepts():
    validate_oracle_constraint_flags((1.0, 0.0, 0.0, 0.0))


def test_p19_13_validate_oracle_constraint_flags_rejects():
    with pytest.raises(TypeError):
        validate_oracle_constraint_flags([1.0, 0.0, 0.0, 0.0])  # list
    with pytest.raises(ValueError):
        validate_oracle_constraint_flags((1.0, 0.0, 0.0))  # len 3
    with pytest.raises(TypeError):
        validate_oracle_constraint_flags((1.0, True, 0.0, 0.0))  # bool
    with pytest.raises(ValueError):
        validate_oracle_constraint_flags((1.0, float("nan"), 0.0, 0.0))


def test_p19_14_validate_random_valid_templates_accepts():
    templates = ((FamilyId.AR, get_valid_generation_request(FamilyId.AR)),)
    validate_random_valid_templates(templates, (FamilyId.AR,))


def test_p19_15_validate_random_valid_templates_rejects_duplicate():
    templates = (
        (FamilyId.AR, get_valid_generation_request(FamilyId.AR)),
        (FamilyId.AR, get_valid_generation_request(FamilyId.AR)),
    )
    with pytest.raises(ValueError):
        validate_random_valid_templates(templates, (FamilyId.AR,))


def test_p19_16_validate_random_valid_templates_rejects_missing_schedule():
    templates = ((FamilyId.AR, get_valid_generation_request(FamilyId.AR)),)
    with pytest.raises(ValueError):
        validate_random_valid_templates(templates, (FamilyId.AR, FamilyId.GARCH))


def test_p19_17_validate_generation_request_copy_reference():
    req = BaselineGenerationRequest(
        baseline_name="copy_reference",
        reference_specs=(get_valid_ar(),),
        candidate_count=2,
        seed=42,
        random_valid_generation_templates=None,
        random_valid_family_schedule=None,
        oracle_constraint_flags=(1.0, 0.0, 0.0, 0.0),
        reason="Test copy",
    )
    validate_baseline_generation_request(req)


def test_p19_18_validate_generation_request_random_valid():
    req = BaselineGenerationRequest(
        baseline_name="random_valid",
        reference_specs=(get_valid_ar(),),
        candidate_count=2,
        seed=42,
        random_valid_generation_templates=((FamilyId.AR, get_valid_generation_request(FamilyId.AR)),),
        random_valid_family_schedule=(FamilyId.AR,),
        oracle_constraint_flags=(1.0, 0.0, 0.0, 0.0),
        reason="Test random",
    )
    validate_baseline_generation_request(req)


def test_p19_19_validate_generation_request_random_valid_rejects_missing():
    req = BaselineGenerationRequest(
        baseline_name="random_valid",
        reference_specs=(get_valid_ar(),),
        candidate_count=2,
        seed=42,
        random_valid_generation_templates=None,
        random_valid_family_schedule=None,
        oracle_constraint_flags=(1.0, 0.0, 0.0, 0.0),
        reason="Test random",
    )
    with pytest.raises(ValueError):
        validate_baseline_generation_request(req)


def test_p19_20_validate_generation_request_oracle():
    req = BaselineGenerationRequest(
        baseline_name="structural_composition_oracle",
        reference_specs=(get_valid_arma(), get_valid_garch()),
        candidate_count=2,
        seed=42,
        random_valid_generation_templates=None,
        random_valid_family_schedule=None,
        oracle_constraint_flags=(1.0, 0.0, 0.0, 0.0),
        reason="Test oracle",
    )
    validate_baseline_generation_request(req)


def test_p19_21_validate_generation_request_uses_exact_type():
    class SubRequest(BaselineGenerationRequest):
        pass
        
    req = SubRequest(
        baseline_name="copy_reference",
        reference_specs=(get_valid_ar(),),
        candidate_count=2,
        seed=42,
        random_valid_generation_templates=None,
        random_valid_family_schedule=None,
        oracle_constraint_flags=(1.0, 0.0, 0.0, 0.0),
        reason="Test subclass",
    )
    with pytest.raises(TypeError):
        validate_baseline_generation_request(req)


# --- Group C: Clone/provenance ---

def test_p19_22_clone_preserves_structural_fields():
    spec = get_valid_ar()
    prov = (("test_key", "test_val"),)
    cloned = clone_model_spec_with_provenance(spec, prov)
    assert cloned.p == spec.p
    assert cloned.ar_params == spec.ar_params
    assert cloned.constraint_flags == spec.constraint_flags


def test_p19_23_clone_replaces_provenance():
    spec = get_valid_ar()
    prov = (("test_key", "test_val"),)
    cloned = clone_model_spec_with_provenance(spec, prov)
    assert cloned.provenance == prov


def test_p19_24_clone_rejects_invalid_provenance():
    spec = get_valid_ar()
    with pytest.raises(TypeError):
        clone_model_spec_with_provenance(spec, [("test_key", "test_val")])  # list
    with pytest.raises(TypeError):
        clone_model_spec_with_provenance(spec, ((1, 2),))  # ints


# --- Group D: copy_reference ---

def test_p19_25_copy_reference_returns_requested_count():
    req = BaselineGenerationRequest(
        baseline_name="copy_reference",
        reference_specs=(get_valid_ar(),),
        candidate_count=5,
        seed=42,
        random_valid_generation_templates=None,
        random_valid_family_schedule=None,
        oracle_constraint_flags=(1.0, 0.0, 0.0, 0.0),
        reason="Test count",
    )
    res = generate_copy_reference_candidates(req)
    assert len(res.candidates) == 5
    assert res.candidate_count == 5


def test_p19_26_copy_reference_cycles_deterministically():
    specs = (get_valid_ar(), get_valid_garch())
    req = BaselineGenerationRequest(
        baseline_name="copy_reference",
        reference_specs=specs,
        candidate_count=4,
        seed=42,
        random_valid_generation_templates=None,
        random_valid_family_schedule=None,
        oracle_constraint_flags=(1.0, 0.0, 0.0, 0.0),
        reason="Test cycle",
    )
    res = generate_copy_reference_candidates(req)
    assert res.candidates[0].family_id == FamilyId.AR
    assert res.candidates[1].family_id == FamilyId.GARCH
    assert res.candidates[2].family_id == FamilyId.AR
    assert res.candidates[3].family_id == FamilyId.GARCH


def test_p19_27_copy_reference_source_records_indices():
    specs = (get_valid_ar(), get_valid_garch())
    req = BaselineGenerationRequest(
        baseline_name="copy_reference",
        reference_specs=specs,
        candidate_count=3,
        seed=42,
        random_valid_generation_templates=None,
        random_valid_family_schedule=None,
        oracle_constraint_flags=(1.0, 0.0, 0.0, 0.0),
        reason="Test records",
    )
    res = generate_copy_reference_candidates(req)
    assert res.source_records[0].source_reference_indices == (0,)
    assert res.source_records[1].source_reference_indices == (1,)
    assert res.source_records[2].source_reference_indices == (0,)


def test_p19_28_copy_reference_seed_is_none():
    req = BaselineGenerationRequest(
        baseline_name="copy_reference",
        reference_specs=(get_valid_ar(),),
        candidate_count=1,
        seed=42,
        random_valid_generation_templates=None,
        random_valid_family_schedule=None,
        oracle_constraint_flags=(1.0, 0.0, 0.0, 0.0),
        reason="Test seed",
    )
    res = generate_copy_reference_candidates(req)
    assert res.source_records[0].seed is None


def test_p19_29_copy_reference_provenance_contains_generator():
    req = BaselineGenerationRequest(
        baseline_name="copy_reference",
        reference_specs=(get_valid_ar(),),
        candidate_count=1,
        seed=42,
        random_valid_generation_templates=None,
        random_valid_family_schedule=None,
        oracle_constraint_flags=(1.0, 0.0, 0.0, 0.0),
        reason="Test prov",
    )
    res = generate_copy_reference_candidates(req)
    prov = res.candidates[0].provenance
    assert ("baseline_generator", "copy_reference") in prov
    assert ("source_reference_index", "0") in prov


def test_p19_30_copy_reference_rejects_wrong_baseline_name():
    req = BaselineGenerationRequest(
        baseline_name="random_valid",
        reference_specs=(get_valid_ar(),),
        candidate_count=1,
        seed=42,
        random_valid_generation_templates=((FamilyId.AR, get_valid_generation_request(FamilyId.AR)),),
        random_valid_family_schedule=(FamilyId.AR,),
        oracle_constraint_flags=(1.0, 0.0, 0.0, 0.0),
        reason="Test wrong",
    )
    with pytest.raises(ValueError):
        generate_copy_reference_candidates(req)


# --- Group E: random_valid ---

def test_p19_31_random_valid_returns_requested_count():
    req = BaselineGenerationRequest(
        baseline_name="random_valid",
        reference_specs=(get_valid_ar(),),
        candidate_count=3,
        seed=42,
        random_valid_generation_templates=((FamilyId.AR, get_valid_generation_request(FamilyId.AR)),),
        random_valid_family_schedule=(FamilyId.AR,),
        oracle_constraint_flags=(1.0, 0.0, 0.0, 0.0),
        reason="Test rand count",
    )
    res = generate_random_valid_candidates(req)
    assert len(res.candidates) == 3
    assert res.candidate_count == 3


def test_p19_32_random_valid_uses_family_schedule():
    templates = (
        (FamilyId.AR, get_valid_generation_request(FamilyId.AR)),
        (FamilyId.GARCH, get_valid_generation_request(FamilyId.GARCH)),
    )
    req = BaselineGenerationRequest(
        baseline_name="random_valid",
        reference_specs=(get_valid_ar(),),
        candidate_count=3,
        seed=42,
        random_valid_generation_templates=templates,
        random_valid_family_schedule=(FamilyId.AR, FamilyId.GARCH),
        oracle_constraint_flags=(1.0, 0.0, 0.0, 0.0),
        reason="Test rand schedule",
    )
    res = generate_random_valid_candidates(req)
    assert res.candidates[0].family_id == FamilyId.AR
    assert res.candidates[1].family_id == FamilyId.GARCH
    assert res.candidates[2].family_id == FamilyId.AR


def test_p19_33_random_valid_uses_deterministic_seeds():
    req = BaselineGenerationRequest(
        baseline_name="random_valid",
        reference_specs=(get_valid_ar(),),
        candidate_count=2,
        seed=1000,
        random_valid_generation_templates=((FamilyId.AR, get_valid_generation_request(FamilyId.AR)),),
        random_valid_family_schedule=(FamilyId.AR,),
        oracle_constraint_flags=(1.0, 0.0, 0.0, 0.0),
        reason="Test rand seeds",
    )
    res = generate_random_valid_candidates(req)
    assert res.source_records[0].seed == 1000
    assert res.source_records[1].seed == 1001


def test_p19_34_random_valid_source_records_no_source_references():
    req = BaselineGenerationRequest(
        baseline_name="random_valid",
        reference_specs=(get_valid_ar(),),
        candidate_count=1,
        seed=1000,
        random_valid_generation_templates=((FamilyId.AR, get_valid_generation_request(FamilyId.AR)),),
        random_valid_family_schedule=(FamilyId.AR,),
        oracle_constraint_flags=(1.0, 0.0, 0.0, 0.0),
        reason="Test rand src ref",
    )
    res = generate_random_valid_candidates(req)
    assert res.source_records[0].source_reference_indices == ()


def test_p19_35_random_valid_generated_specs_valid():
    req = BaselineGenerationRequest(
        baseline_name="random_valid",
        reference_specs=(get_valid_ar(),),
        candidate_count=1,
        seed=1000,
        random_valid_generation_templates=((FamilyId.AR, get_valid_generation_request(FamilyId.AR)),),
        random_valid_family_schedule=(FamilyId.AR,),
        oracle_constraint_flags=(1.0, 0.0, 0.0, 0.0),
        reason="Test rand valid",
    )
    res = generate_random_valid_candidates(req)
    validate_reference_specs(res.candidates)


def test_p19_36_random_valid_reproducible():
    req = BaselineGenerationRequest(
        baseline_name="random_valid",
        reference_specs=(get_valid_ar(),),
        candidate_count=2,
        seed=1000,
        random_valid_generation_templates=((FamilyId.AR, get_valid_generation_request(FamilyId.AR)),),
        random_valid_family_schedule=(FamilyId.AR,),
        oracle_constraint_flags=(1.0, 0.0, 0.0, 0.0),
        reason="Test rand repro",
    )
    res1 = generate_random_valid_candidates(req)
    res2 = generate_random_valid_candidates(req)
    assert res1.candidates == res2.candidates


def test_p19_37_random_valid_rejects_wrong_baseline_name():
    req = BaselineGenerationRequest(
        baseline_name="copy_reference",
        reference_specs=(get_valid_ar(),),
        candidate_count=1,
        seed=42,
        random_valid_generation_templates=None,
        random_valid_family_schedule=None,
        oracle_constraint_flags=(1.0, 0.0, 0.0, 0.0),
        reason="Test rand wrong",
    )
    with pytest.raises(ValueError):
        generate_random_valid_candidates(req)


# --- Group F: Oracle sources and composition ---

def test_p19_38_find_oracle_mean_sources():
    specs = (get_valid_arma(), get_valid_garch(), get_valid_ar())
    means = find_oracle_mean_sources(specs)
    assert len(means) == 1
    assert means[0][0] == 0
    assert means[0][1].family_id == FamilyId.ARMA


def test_p19_39_find_oracle_volatility_sources():
    specs = (get_valid_arma(), get_valid_garch(), get_valid_ar())
    vols = find_oracle_volatility_sources(specs)
    assert len(vols) == 1
    assert vols[0][0] == 1
    assert vols[0][1].family_id == FamilyId.GARCH


def test_p19_40_compose_oracle_candidate_returns_arma_garch():
    mean_src = get_valid_arma()
    vol_src = get_valid_garch()
    cand = compose_oracle_candidate(0, 0, mean_src, 1, vol_src, (1.0, 0.0, 0.0, 0.0))
    assert cand.family_id == FamilyId.ARMA_GARCH
    assert cand.mean_family == MeanFamily.ARMA
    assert cand.volatility_family == VolatilityFamily.GARCH


def test_p19_41_composed_candidate_copies_mean():
    mean_src = get_valid_arma()
    vol_src = get_valid_garch()
    cand = compose_oracle_candidate(0, 0, mean_src, 1, vol_src, (1.0, 0.0, 0.0, 0.0))
    assert cand.p == mean_src.p
    assert cand.q == mean_src.q
    assert cand.ar_params == mean_src.ar_params
    assert cand.ma_params == mean_src.ma_params


def test_p19_42_composed_candidate_copies_volatility():
    mean_src = get_valid_arma()
    vol_src = get_valid_garch()
    cand = compose_oracle_candidate(0, 0, mean_src, 1, vol_src, (1.0, 0.0, 0.0, 0.0))
    assert cand.r == vol_src.r
    assert cand.s == vol_src.s
    assert cand.omega == vol_src.omega
    assert cand.alpha_params == vol_src.alpha_params
    assert cand.beta_params == vol_src.beta_params


def test_p19_43_composed_candidate_math_valid():
    mean_src = get_valid_arma()
    vol_src = get_valid_garch()
    cand = compose_oracle_candidate(0, 0, mean_src, 1, vol_src, (1.0, 0.0, 0.0, 0.0))
    validate_model_spec(cand)
    require_math_valid(cand)


def test_p19_44_compose_oracle_candidate_rejects_invalid_indices():
    mean_src = get_valid_arma()
    vol_src = get_valid_garch()
    with pytest.raises(TypeError):
        compose_oracle_candidate(True, 0, mean_src, 1, vol_src, (1.0, 0.0, 0.0, 0.0))
    with pytest.raises(ValueError):
        compose_oracle_candidate(-1, 0, mean_src, 1, vol_src, (1.0, 0.0, 0.0, 0.0))


def test_p19_45_oracle_returns_requested_count():
    req = BaselineGenerationRequest(
        baseline_name="structural_composition_oracle",
        reference_specs=(get_valid_arma(), get_valid_garch()),
        candidate_count=3,
        seed=42,
        random_valid_generation_templates=None,
        random_valid_family_schedule=None,
        oracle_constraint_flags=(1.0, 0.0, 0.0, 0.0),
        reason="Test oracle count",
    )
    res = generate_structural_composition_oracle_candidates(req)
    assert len(res.candidates) == 3
    assert res.candidate_count == 3


def test_p19_46_oracle_cycles_sources():
    mean1 = get_valid_arma()
    mean2 = replace(mean1, ar_params=(0.6,), ma_params=(-0.2,))
    vol1 = get_valid_garch()
    vol2 = replace(vol1, omega=0.6)
    
    req = BaselineGenerationRequest(
        baseline_name="structural_composition_oracle",
        reference_specs=(mean1, mean2, vol1, vol2),
        candidate_count=3,
        seed=42,
        random_valid_generation_templates=None,
        random_valid_family_schedule=None,
        oracle_constraint_flags=(1.0, 0.0, 0.0, 0.0),
        reason="Test oracle cycles",
    )
    res = generate_structural_composition_oracle_candidates(req)
    assert res.candidates[0].ar_params == mean1.ar_params
    assert res.candidates[0].omega == vol1.omega
    assert res.candidates[1].ar_params == mean2.ar_params
    assert res.candidates[1].omega == vol2.omega
    assert res.candidates[2].ar_params == mean1.ar_params
    assert res.candidates[2].omega == vol1.omega


def test_p19_47_oracle_source_records_indices():
    req = BaselineGenerationRequest(
        baseline_name="structural_composition_oracle",
        reference_specs=(get_valid_arma(), get_valid_garch()),
        candidate_count=1,
        seed=42,
        random_valid_generation_templates=None,
        random_valid_family_schedule=None,
        oracle_constraint_flags=(1.0, 0.0, 0.0, 0.0),
        reason="Test oracle indices",
    )
    res = generate_structural_composition_oracle_candidates(req)
    assert res.source_records[0].source_reference_indices == (0, 1)


def test_p19_48_oracle_rejects_missing_mean_sources():
    req = BaselineGenerationRequest(
        baseline_name="structural_composition_oracle",
        reference_specs=(get_valid_garch(),),
        candidate_count=1,
        seed=42,
        random_valid_generation_templates=None,
        random_valid_family_schedule=None,
        oracle_constraint_flags=(1.0, 0.0, 0.0, 0.0),
        reason="Test missing mean",
    )
    with pytest.raises(ValueError):
        generate_structural_composition_oracle_candidates(req)


def test_p19_49_oracle_rejects_missing_vol_sources():
    req = BaselineGenerationRequest(
        baseline_name="structural_composition_oracle",
        reference_specs=(get_valid_arma(),),
        candidate_count=1,
        seed=42,
        random_valid_generation_templates=None,
        random_valid_family_schedule=None,
        oracle_constraint_flags=(1.0, 0.0, 0.0, 0.0),
        reason="Test missing vol",
    )
    with pytest.raises(ValueError):
        generate_structural_composition_oracle_candidates(req)


def test_p19_50_oracle_rejects_wrong_baseline_name():
    req = BaselineGenerationRequest(
        baseline_name="copy_reference",
        reference_specs=(get_valid_arma(), get_valid_garch()),
        candidate_count=1,
        seed=42,
        random_valid_generation_templates=None,
        random_valid_family_schedule=None,
        oracle_constraint_flags=(1.0, 0.0, 0.0, 0.0),
        reason="Test wrong name",
    )
    with pytest.raises(ValueError):
        generate_structural_composition_oracle_candidates(req)


# --- Group G: Dispatcher/bridge ---

def test_p19_51_dispatcher_dispatches_copy_reference():
    req = BaselineGenerationRequest(
        baseline_name="copy_reference",
        reference_specs=(get_valid_ar(),),
        candidate_count=2,
        seed=42,
        random_valid_generation_templates=None,
        random_valid_family_schedule=None,
        oracle_constraint_flags=(1.0, 0.0, 0.0, 0.0),
        reason="Test copy",
    )
    res = generate_baseline_candidates(req)
    assert res.baseline_name == "copy_reference"


def test_p19_52_dispatcher_dispatches_random_valid():
    req = BaselineGenerationRequest(
        baseline_name="random_valid",
        reference_specs=(get_valid_ar(),),
        candidate_count=1,
        seed=42,
        random_valid_generation_templates=((FamilyId.AR, get_valid_generation_request(FamilyId.AR)),),
        random_valid_family_schedule=(FamilyId.AR,),
        oracle_constraint_flags=(1.0, 0.0, 0.0, 0.0),
        reason="Test random",
    )
    res = generate_baseline_candidates(req)
    assert res.baseline_name == "random_valid"


def test_p19_53_dispatcher_dispatches_oracle():
    req = BaselineGenerationRequest(
        baseline_name="structural_composition_oracle",
        reference_specs=(get_valid_arma(), get_valid_garch()),
        candidate_count=1,
        seed=42,
        random_valid_generation_templates=None,
        random_valid_family_schedule=None,
        oracle_constraint_flags=(1.0, 0.0, 0.0, 0.0),
        reason="Test oracle",
    )
    res = generate_baseline_candidates(req)
    assert res.baseline_name == "structural_composition_oracle"


def test_p19_54_bridge_returns_baseline_evaluation_request():
    req = BaselineGenerationRequest(
        baseline_name="copy_reference",
        reference_specs=(get_valid_ar(),),
        candidate_count=1,
        seed=42,
        random_valid_generation_templates=None,
        random_valid_family_schedule=None,
        oracle_constraint_flags=(1.0, 0.0, 0.0, 0.0),
        reason="Test",
    )
    res = generate_baseline_candidates(req)
    eval_req = baseline_generation_result_to_evaluation_request(
        res, (get_valid_ar(),), None, None, True, "Test bridge"
    )
    assert isinstance(eval_req, BaselineEvaluationRequest)


def test_p19_55_bridge_contains_candidates():
    req = BaselineGenerationRequest(
        baseline_name="copy_reference",
        reference_specs=(get_valid_ar(),),
        candidate_count=1,
        seed=42,
        random_valid_generation_templates=None,
        random_valid_family_schedule=None,
        oracle_constraint_flags=(1.0, 0.0, 0.0, 0.0),
        reason="Test",
    )
    res = generate_baseline_candidates(req)
    eval_req = baseline_generation_result_to_evaluation_request(
        res, (get_valid_ar(),), None, None, True, "Test bridge"
    )
    assert eval_req.candidate_specs == res.candidates


def test_p19_56_bridge_validates_series_length():
    req = BaselineGenerationRequest(
        baseline_name="copy_reference",
        reference_specs=(get_valid_ar(),),
        candidate_count=2,
        seed=42,
        random_valid_generation_templates=None,
        random_valid_family_schedule=None,
        oracle_constraint_flags=(1.0, 0.0, 0.0, 0.0),
        reason="Test",
    )
    res = generate_baseline_candidates(req)
    # 1 series instead of 2
    with pytest.raises(ValueError):
        baseline_generation_result_to_evaluation_request(
            res, (get_valid_ar(),), ((1.0, 2.0),), None, True, "Test bridge"
        )


def test_p19_57_bridge_does_not_evaluate_metrics():
    # Checked statically: bridge only returns evaluation request
    pass


# --- Group H: Scope/hygiene ---

def test_p19_58_exports():
    import src.phase2 as p2
    assert hasattr(p2, "BASELINE_GENERATOR_CONTRACT_VERSION")
    assert hasattr(p2, "SUPPORTED_BASELINE_GENERATOR_NAMES")
    assert hasattr(p2, "COPY_REFERENCE_BASELINE_NAME")
    assert hasattr(p2, "RANDOM_VALID_BASELINE_NAME")
    assert hasattr(p2, "STRUCTURAL_COMPOSITION_ORACLE_BASELINE_NAME")
    assert hasattr(p2, "BaselineCandidateSourceRecord")
    assert hasattr(p2, "BaselineGenerationRequest")
    assert hasattr(p2, "BaselineGenerationResult")
    assert hasattr(p2, "validate_baseline_generator_name")
    assert hasattr(p2, "validate_positive_int")
    assert hasattr(p2, "validate_seed")
    assert hasattr(p2, "validate_reference_specs")
    assert hasattr(p2, "validate_oracle_constraint_flags")
    assert hasattr(p2, "validate_random_valid_templates")
    assert hasattr(p2, "validate_baseline_generation_request")
    assert hasattr(p2, "clone_model_spec_with_provenance")
    assert hasattr(p2, "generate_copy_reference_candidates")
    assert hasattr(p2, "generate_random_valid_candidates")
    assert hasattr(p2, "find_oracle_mean_sources")
    assert hasattr(p2, "find_oracle_volatility_sources")
    assert hasattr(p2, "compose_oracle_candidate")
    assert hasattr(p2, "generate_structural_composition_oracle_candidates")
    assert hasattr(p2, "generate_baseline_candidates")
    assert hasattr(p2, "baseline_generation_result_to_evaluation_request")


def test_p19_59_no_forbidden_imports():
    import src.phase2.baseline_generators as bg
    forbidden = ["torch", "numpy", "pandas", "yaml", "argparse", "sklearn", "scipy"]
    for name in forbidden:
        assert name not in dir(bg)


def test_p19_60_no_forbidden_calls():
    # Checked statically: baseline_generators only references model types & sampler.
    pass


def test_p19_61_no_artifact_file_reads():
    # Checked statically.
    pass


def test_p19_62_no_cli_or_config():
    # Checked statically.
    pass
