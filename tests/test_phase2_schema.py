# tests/test_phase2_schema.py

import pytest
from src.phase2.schema import (
    APPROVED_MAX_P,
    APPROVED_MAX_Q,
    APPROVED_MAX_R,
    APPROVED_MAX_S,
    SCHEMA_V2_MIN_FLAT_DIM,
    FamilyId,
    MeanFamily,
    VolatilityFamily,
    ModelSpec,
    validate_model_spec,
    schema_v2_index_map,
    to_flat_boundary_vector,
    from_flat_boundary_vector,
)
from dataclasses import replace

def get_valid_ar_spec():
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
        omega=None,
        alpha_params=(),
        beta_params=(),
        constraint_flags=(1.0, 0.0, 0.0, 0.0),
        provenance=(("source", "test"),)
    )

def get_valid_arma_spec(p=1, q=1):
    return ModelSpec(
        family_id=FamilyId.ARMA,
        mean_family=MeanFamily.ARMA,
        volatility_family=VolatilityFamily.NONE,
        p=p,
        q=q,
        r=0,
        s=0,
        ar_params=(0.5,) * p,
        ma_params=(0.3,) * q,
        omega=None,
        alpha_params=(),
        beta_params=(),
        constraint_flags=(1.0, 0.0, 0.0, 0.0),
        provenance=()
    )

def get_valid_garch_spec():
    return ModelSpec(
        family_id=FamilyId.GARCH,
        mean_family=MeanFamily.NONE,
        volatility_family=VolatilityFamily.GARCH,
        p=0,
        q=0,
        r=1,
        s=1,
        ar_params=(),
        ma_params=(),
        omega=0.1,
        alpha_params=(0.2,),
        beta_params=(0.7,),
        constraint_flags=(1.0, 0.0, 0.0, 0.0),
        provenance=()
    )

def get_valid_c_spec():
    return ModelSpec(
        family_id=FamilyId.ARMA_GARCH,
        mean_family=MeanFamily.ARMA,
        volatility_family=VolatilityFamily.GARCH,
        p=2,
        q=1,
        r=1,
        s=1,
        ar_params=(0.5, -0.2),
        ma_params=(0.3,),
        omega=0.15,
        alpha_params=(0.1,),
        beta_params=(0.8,),
        constraint_flags=(1.0, 1.0, 0.0, 0.0),
        provenance=(("git_commit", "b18f29f"),)
    )

# 1. Valid A-family AR spec passes
def test_valid_ar_spec_passes():
    spec = get_valid_ar_spec()
    validate_model_spec(spec)

# 2. Valid A-family ARMA spec passes
def test_valid_arma_spec_passes():
    spec = get_valid_arma_spec()
    validate_model_spec(spec)

# 3. Valid A-family pure MA as ARMA(0,q) passes if labeled ARMA
def test_valid_pure_ma_passes():
    spec = get_valid_arma_spec(p=0, q=2)
    validate_model_spec(spec)

# 4. Valid B-family GARCH spec passes
def test_valid_garch_spec_passes():
    spec = get_valid_garch_spec()
    validate_model_spec(spec)

# 5. Valid C-family ARMA_GARCH spec passes
def test_valid_c_spec_passes():
    spec = get_valid_c_spec()
    validate_model_spec(spec)

# 6. AR spec with volatility params fails
def test_ar_with_volatility_params_fails():
    spec = ModelSpec(
        family_id=FamilyId.AR,
        mean_family=MeanFamily.AR,
        volatility_family=VolatilityFamily.NONE,
        p=1,
        q=0,
        r=1,  # invalid
        s=0,
        ar_params=(0.5,),
        ma_params=(),
        omega=None,
        alpha_params=(0.1,),  # invalid
        beta_params=(),
        constraint_flags=(1.0, 0.0, 0.0, 0.0),
        provenance=()
    )
    with pytest.raises(ValueError):
        validate_model_spec(spec)

# 7. GARCH spec with AR params fails
def test_garch_with_ar_params_fails():
    spec = ModelSpec(
        family_id=FamilyId.GARCH,
        mean_family=MeanFamily.NONE,
        volatility_family=VolatilityFamily.GARCH,
        p=1,  # invalid
        q=0,
        r=1,
        s=1,
        ar_params=(0.5,),  # invalid
        ma_params=(),
        omega=0.1,
        alpha_params=(0.2,),
        beta_params=(0.7,),
        constraint_flags=(1.0, 0.0, 0.0, 0.0),
        provenance=()
    )
    with pytest.raises(ValueError):
        validate_model_spec(spec)

# 8. ARMA_GARCH spec without omega fails
def test_c_spec_without_omega_fails():
    spec = ModelSpec(
        family_id=FamilyId.ARMA_GARCH,
        mean_family=MeanFamily.ARMA,
        volatility_family=VolatilityFamily.GARCH,
        p=1,
        q=1,
        r=1,
        s=1,
        ar_params=(0.5,),
        ma_params=(0.3,),
        omega=None,  # invalid
        alpha_params=(0.1,),
        beta_params=(0.8,),
        constraint_flags=(1.0, 0.0, 0.0, 0.0),
        provenance=()
    )
    with pytest.raises(ValueError):
        validate_model_spec(spec)

# 9. ARMA_GARCH spec with wrong alpha/beta lengths fails
def test_c_spec_with_wrong_param_lengths_fails():
    spec = ModelSpec(
        family_id=FamilyId.ARMA_GARCH,
        mean_family=MeanFamily.ARMA,
        volatility_family=VolatilityFamily.GARCH,
        p=1,
        q=1,
        r=1,
        s=1,
        ar_params=(0.5,),
        ma_params=(0.3,),
        omega=0.1,
        alpha_params=(0.1, 0.2),  # invalid (length 2 instead of r=1)
        beta_params=(0.8,),
        constraint_flags=(1.0, 0.0, 0.0, 0.0),
        provenance=()
    )
    with pytest.raises(ValueError):
        validate_model_spec(spec)

# 10. Order bounds over max_p/max_q/max_r/max_s fail
def test_order_bounds_exceeded_fails():
    spec = ModelSpec(
        family_id=FamilyId.AR,
        mean_family=MeanFamily.AR,
        volatility_family=VolatilityFamily.NONE,
        p=APPROVED_MAX_P + 1,  # invalid
        q=0,
        r=0,
        s=0,
        ar_params=(0.5,) * (APPROVED_MAX_P + 1),
        ma_params=(),
        omega=None,
        alpha_params=(),
        beta_params=(),
        constraint_flags=(1.0, 0.0, 0.0, 0.0),
        provenance=()
    )
    with pytest.raises(ValueError):
        validate_model_spec(spec)

# 11. Negative orders fail
def test_negative_orders_fail():
    spec = ModelSpec(
        family_id=FamilyId.AR,
        mean_family=MeanFamily.AR,
        volatility_family=VolatilityFamily.NONE,
        p=-1,  # invalid
        q=0,
        r=0,
        s=0,
        ar_params=(),
        ma_params=(),
        omega=None,
        alpha_params=(),
        beta_params=(),
        constraint_flags=(1.0, 0.0, 0.0, 0.0),
        provenance=()
    )
    with pytest.raises(ValueError):
        validate_model_spec(spec)

# 12. Schema index map has no overlap
def test_schema_index_map_no_overlap():
    idx_map = schema_v2_index_map()
    keys_to_check = ["ar_params", "ma_params", "omega", "alpha_params", "beta_params"]
    
    omega_set = idx_map["omega"]
    sets = {k: idx_map[k] for k in keys_to_check if k != "omega"}
    sets["omega"] = omega_set
    
    for k1, s1 in sets.items():
        for k2, s2 in sets.items():
            if k1 != k2:
                intersection = s1.intersection(s2)
                assert not intersection, f"Index overlap detected between {k1} and {k2} at {intersection}"

# 13. schema_v2_min_flat_dim equals 32
def test_schema_min_flat_dim_equals_32():
    assert SCHEMA_V2_MIN_FLAT_DIM == 32

# 14. to_flat_boundary_vector returns length 32
def test_to_flat_boundary_vector_length():
    spec = get_valid_c_spec()
    vec = to_flat_boundary_vector(spec)
    assert len(vec) == 32

# 15. to_flat_boundary_vector refuses invalid spec
def test_to_flat_boundary_vector_refuses_invalid():
    spec = ModelSpec(
        family_id=FamilyId.AR,
        mean_family=MeanFamily.AR,
        volatility_family=VolatilityFamily.NONE,
        p=-1,
        q=0,
        r=0,
        s=0,
        ar_params=(),
        ma_params=(),
        omega=None,
        alpha_params=(),
        beta_params=(),
        constraint_flags=(1.0, 0.0, 0.0, 0.0),
        provenance=()
    )
    with pytest.raises(ValueError):
        to_flat_boundary_vector(spec)

# 16. from_flat_boundary_vector round-trips a valid C-family spec created by to_flat_boundary_vector
def test_round_trip_c_spec():
    spec_orig = get_valid_c_spec()
    vec = to_flat_boundary_vector(spec_orig)
    spec_recon = from_flat_boundary_vector(vec)
    
    assert spec_recon.family_id == spec_orig.family_id
    assert spec_recon.mean_family == spec_orig.mean_family
    assert spec_recon.volatility_family == spec_orig.volatility_family
    assert spec_recon.p == spec_orig.p
    assert spec_recon.q == spec_orig.q
    assert spec_recon.r == spec_orig.r
    assert spec_recon.s == spec_orig.s
    assert spec_recon.ar_params == spec_orig.ar_params
    assert spec_recon.ma_params == spec_orig.ma_params
    assert spec_recon.omega == spec_orig.omega
    assert spec_recon.alpha_params == spec_orig.alpha_params
    assert spec_recon.beta_params == spec_orig.beta_params
    assert spec_recon.constraint_flags == spec_orig.constraint_flags
    # provenance is expected to be empty tuple on reconstruction
    assert spec_recon.provenance == ()

# 17. from_flat_boundary_vector rejects wrong length
def test_from_flat_boundary_vector_wrong_length():
    with pytest.raises(ValueError):
        from_flat_boundary_vector([0.0] * 31)
    with pytest.raises(ValueError):
        from_flat_boundary_vector([0.0] * 33)

# 18. from_flat_boundary_vector rejects unknown family code
def test_from_flat_boundary_vector_unknown_family_code():
    vec = [0.0] * 32
    vec[0] = 99.0  # unknown family code
    with pytest.raises(ValueError):
        from_flat_boundary_vector(vec)

# 19. No threshold-based decoding is used
def test_no_threshold_based_decoding():
    vec = [0.0] * 32
    # FamilyId code AR is 1.0. Test that a nearby fractional value fails
    vec[0] = 1.1  # invalid code, must reject exactly
    vec[1] = 1.0  # mean_family code AR (1.0)
    vec[2] = 0.0  # volatility NONE
    vec[3] = 1.0  # p=1
    with pytest.raises(ValueError) as excinfo:
        from_flat_boundary_vector(vec)
    assert "No threshold decoding allowed" in str(excinfo.value)

# 20. Legacy v[10] collision is not present in Phase 2 index map
def test_legacy_v10_collision_avoided():
    idx_map = schema_v2_index_map()
    # Check that ar_params (12:17) and volatility params (omega=7, alpha_params=22:24, beta_params=24:26) are disjoint
    ar_indices = idx_map["ar_params"]
    vol_indices = idx_map["omega"].union(idx_map["alpha_params"]).union(idx_map["beta_params"])
    assert ar_indices.isdisjoint(vol_indices)

# 21. Immutability checks (Correction 6)
def test_modelspec_immutability():
    spec = get_valid_c_spec()
    assert isinstance(spec.ar_params, tuple)
    assert isinstance(spec.ma_params, tuple)
    assert isinstance(spec.alpha_params, tuple)
    assert isinstance(spec.beta_params, tuple)
    assert isinstance(spec.constraint_flags, tuple)
    assert isinstance(spec.provenance, tuple)
    
    # Verify that trying to modify fields raises AttributeError (frozen dataclass)
    with pytest.raises(AttributeError):
        spec.p = 3  # type: ignore
        
    with pytest.raises(AttributeError):
        spec.ar_params = (0.1, 0.2)  # type: ignore

# 22. Enum type strictness
def test_enum_types_strictness():
    spec = ModelSpec(
        family_id="AR",  # type: ignore (invalid, must be Enum)
        mean_family=MeanFamily.AR,
        volatility_family=VolatilityFamily.NONE,
        p=1, q=0, r=0, s=0,
        ar_params=(0.5,), ma_params=(), omega=None, alpha_params=(), beta_params=(),
        constraint_flags=(1.0, 0.0, 0.0, 0.0), provenance=()
    )
    with pytest.raises(ValueError) as excinfo:
        validate_model_spec(spec)
    assert "must be a FamilyId enum instance" in str(excinfo.value)

# --- Tuple Container Strictness Tests ---
def test_ar_params_as_list_rejected():
    spec = replace(get_valid_ar_spec(), ar_params=[0.5]) # type: ignore
    with pytest.raises(ValueError, match="ar_params must be exactly a tuple"): validate_model_spec(spec)

def test_ma_params_as_list_rejected():
    spec = replace(get_valid_arma_spec(), ma_params=[0.3]) # type: ignore
    with pytest.raises(ValueError, match="ma_params must be exactly a tuple"): validate_model_spec(spec)

def test_alpha_params_as_list_rejected():
    spec = replace(get_valid_garch_spec(), alpha_params=[0.2]) # type: ignore
    with pytest.raises(ValueError, match="alpha_params must be exactly a tuple"): validate_model_spec(spec)

def test_beta_params_as_list_rejected():
    spec = replace(get_valid_garch_spec(), beta_params=[0.7]) # type: ignore
    with pytest.raises(ValueError, match="beta_params must be exactly a tuple"): validate_model_spec(spec)

def test_constraint_flags_as_list_rejected():
    spec = replace(get_valid_ar_spec(), constraint_flags=[1.0, 0.0, 0.0, 0.0]) # type: ignore
    with pytest.raises(ValueError, match="constraint_flags must be exactly a tuple"): validate_model_spec(spec)

def test_provenance_as_list_rejected():
    spec = replace(get_valid_ar_spec(), provenance=[("a", "b")]) # type: ignore
    with pytest.raises(ValueError, match="provenance must be exactly a tuple"): validate_model_spec(spec)

def test_provenance_entry_as_list_rejected():
    spec = replace(get_valid_ar_spec(), provenance=(["a", "b"],)) # type: ignore
    with pytest.raises(ValueError, match="provenance entry must be exactly a tuple"): validate_model_spec(spec)

def test_provenance_entry_length_rejected():
    spec = replace(get_valid_ar_spec(), provenance=(("a",),)) # type: ignore
    with pytest.raises(ValueError, match="provenance entry length must be exactly 2"): validate_model_spec(spec)

def test_provenance_key_not_str_rejected():
    spec = replace(get_valid_ar_spec(), provenance=((1, "b"),)) # type: ignore
    with pytest.raises(ValueError, match="provenance key must be a string"): validate_model_spec(spec)

def test_provenance_value_not_str_rejected():
    spec = replace(get_valid_ar_spec(), provenance=(("a", 2),)) # type: ignore
    with pytest.raises(ValueError, match="provenance value must be a string"): validate_model_spec(spec)

# --- Tuple Element Type Strictness Tests ---
def test_ar_params_containing_bool_rejected():
    spec = replace(get_valid_ar_spec(), ar_params=(True,)) # type: ignore
    with pytest.raises(ValueError, match="ar_params items must not be bool"): validate_model_spec(spec)

def test_ma_params_containing_string_rejected():
    spec = replace(get_valid_arma_spec(), ma_params=("0.3",)) # type: ignore
    with pytest.raises(ValueError, match="ma_params items must be int or float"): validate_model_spec(spec)

def test_alpha_params_containing_nan_rejected():
    spec = replace(get_valid_garch_spec(), alpha_params=(float("nan"),))
    with pytest.raises(ValueError, match="alpha_params items must be finite"): validate_model_spec(spec)

def test_beta_params_containing_inf_rejected():
    spec = replace(get_valid_garch_spec(), beta_params=(float("inf"),))
    with pytest.raises(ValueError, match="beta_params items must be finite"): validate_model_spec(spec)

def test_constraint_flags_containing_bool_rejected():
    spec = replace(get_valid_ar_spec(), constraint_flags=(True, 0.0, 0.0, 0.0)) # type: ignore
    with pytest.raises(ValueError, match="constraint_flags items must not be bool"): validate_model_spec(spec)

# --- Boundary Vector Decoding Tests ---
def test_from_flat_boundary_vector_bool_p_rejected():
    vec = to_flat_boundary_vector(get_valid_ar_spec())
    vec[3] = True # type: ignore
    with pytest.raises(ValueError, match="got bool"): from_flat_boundary_vector(vec)

def test_from_flat_boundary_vector_bool_q_rejected():
    vec = to_flat_boundary_vector(get_valid_ar_spec())
    vec[4] = True # type: ignore
    with pytest.raises(ValueError, match="got bool"): from_flat_boundary_vector(vec)

def test_from_flat_boundary_vector_bool_r_rejected():
    vec = to_flat_boundary_vector(get_valid_ar_spec())
    vec[5] = True # type: ignore
    with pytest.raises(ValueError, match="got bool"): from_flat_boundary_vector(vec)

def test_from_flat_boundary_vector_bool_s_rejected():
    vec = to_flat_boundary_vector(get_valid_ar_spec())
    vec[6] = True # type: ignore
    with pytest.raises(ValueError, match="got bool"): from_flat_boundary_vector(vec)

def test_from_flat_boundary_vector_string_order_rejected():
    vec = to_flat_boundary_vector(get_valid_ar_spec())
    vec[3] = "1" # type: ignore
    with pytest.raises(ValueError, match="must be a float or int"): from_flat_boundary_vector(vec)

def test_from_flat_boundary_vector_nan_order_rejected():
    vec = to_flat_boundary_vector(get_valid_ar_spec())
    vec[3] = float("nan")
    with pytest.raises(ValueError, match="must be finite"): from_flat_boundary_vector(vec)

def test_from_flat_boundary_vector_inf_order_rejected():
    vec = to_flat_boundary_vector(get_valid_ar_spec())
    vec[3] = float("inf")
    with pytest.raises(ValueError, match="must be finite"): from_flat_boundary_vector(vec)

def test_from_flat_boundary_vector_fractional_order_rejected():
    vec = to_flat_boundary_vector(get_valid_ar_spec())
    vec[3] = 1.5
    with pytest.raises(ValueError, match="non-negative integer"): from_flat_boundary_vector(vec)

