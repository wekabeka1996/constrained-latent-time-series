# tests/test_phase2_constraints.py

import pytest
from dataclasses import replace
from src.phase2.schema import (
    ModelSpec, FamilyId, MeanFamily, VolatilityFamily
)
from src.phase2.constraints import (
    validate_ar_stationarity,
    validate_ma_invertibility,
    validate_garch_constraints,
    validate_math_constraints,
    require_math_valid,
    require_constraint_pass,
    ConstraintResult,
    PERSISTENCE_TOL
)

def get_base_ar() -> ModelSpec:
    return ModelSpec(
        family_id=FamilyId.AR,
        mean_family=MeanFamily.AR,
        volatility_family=VolatilityFamily.NONE,
        p=1, q=0, r=0, s=0,
        ar_params=(0.5,), ma_params=(), omega=None, alpha_params=(), beta_params=(),
        constraint_flags=(1.0, 0.0, 0.0, 0.0), provenance=()
    )

def get_base_arma() -> ModelSpec:
    return ModelSpec(
        family_id=FamilyId.ARMA,
        mean_family=MeanFamily.ARMA,
        volatility_family=VolatilityFamily.NONE,
        p=1, q=1, r=0, s=0,
        ar_params=(0.4,), ma_params=(0.4,), omega=None, alpha_params=(), beta_params=(),
        constraint_flags=(1.0, 0.0, 0.0, 0.0), provenance=()
    )

def get_base_garch() -> ModelSpec:
    return ModelSpec(
        family_id=FamilyId.GARCH,
        mean_family=MeanFamily.NONE,
        volatility_family=VolatilityFamily.GARCH,
        p=0, q=0, r=1, s=1,
        ar_params=(), ma_params=(), omega=0.1, alpha_params=(0.1,), beta_params=(0.8,),
        constraint_flags=(1.0, 0.0, 0.0, 0.0), provenance=()
    )

def get_base_c() -> ModelSpec:
    return ModelSpec(
        family_id=FamilyId.ARMA_GARCH,
        mean_family=MeanFamily.ARMA,
        volatility_family=VolatilityFamily.GARCH,
        p=1, q=1, r=1, s=1,
        ar_params=(0.4,), ma_params=(0.4,), omega=0.1, alpha_params=(0.1,), beta_params=(0.8,),
        constraint_flags=(1.0, 0.0, 0.0, 0.0), provenance=()
    )

# --- A. AR stationarity ---

def test_ar_stationarity_phi_0_5_passes():
    spec = get_base_ar()
    res = validate_ar_stationarity(spec)
    assert res.is_valid

def test_ar_stationarity_phi_1_0_fails():
    spec = replace(get_base_ar(), ar_params=(1.0,))
    res = validate_ar_stationarity(spec)
    assert not res.is_valid

def test_ar_stationarity_phi_1_1_fails():
    spec = replace(get_base_ar(), ar_params=(1.1,))
    res = validate_ar_stationarity(spec)
    assert not res.is_valid

def test_arma_ar_part_phi_0_4_passes():
    spec = get_base_arma()
    res = validate_ar_stationarity(spec)
    assert res.is_valid

def test_arma_garch_ar_part_phi_0_4_passes():
    spec = get_base_c()
    res = validate_ar_stationarity(spec)
    assert res.is_valid

def test_garch_only_returns_not_applicable_no_mean_component():
    spec = get_base_garch()
    res = validate_ar_stationarity(spec)
    assert res.is_valid
    assert res.reason == "not_applicable_no_mean_component"

def test_pure_ma_returns_not_applicable_no_ar_terms():
    spec = replace(get_base_arma(), p=0, ar_params=())
    res = validate_ar_stationarity(spec)
    assert res.is_valid
    assert res.reason == "not_applicable_no_ar_terms"

def test_ar_trailing_exact_zero_trimmed():
    spec = replace(get_base_ar(), p=2, ar_params=(0.5, 0.0))
    res = validate_ar_stationarity(spec)
    assert res.is_valid
    assert dict(res.details)["effective_order"] == "1"

# --- B. MA invertibility ---

def test_ma_invertibility_theta_0_5_passes():
    spec = replace(get_base_arma(), p=0, q=1, ar_params=(), ma_params=(0.5,))
    res = validate_ma_invertibility(spec)
    assert res.is_valid

def test_ma_invertibility_theta_1_0_fails():
    spec = replace(get_base_arma(), p=0, q=1, ar_params=(), ma_params=(1.0,))
    res = validate_ma_invertibility(spec)
    assert not res.is_valid

def test_ma_invertibility_theta_1_1_fails():
    spec = replace(get_base_arma(), p=0, q=1, ar_params=(), ma_params=(1.1,))
    res = validate_ma_invertibility(spec)
    assert not res.is_valid

def test_ar_q_0_returns_not_applicable_no_ma_terms():
    spec = get_base_ar()
    res = validate_ma_invertibility(spec)
    assert res.is_valid
    assert res.reason == "not_applicable_no_ma_terms"

def test_garch_only_returns_not_applicable_no_ma_terms():
    spec = get_base_garch()
    res = validate_ma_invertibility(spec)
    assert res.is_valid
    assert res.reason == "not_applicable_no_ma_terms"

def test_arma_garch_ma_part_theta_0_4_passes():
    spec = get_base_c()
    res = validate_ma_invertibility(spec)
    assert res.is_valid

def test_ma_trailing_exact_zero_trimmed():
    spec = replace(get_base_arma(), p=0, q=2, ar_params=(), ma_params=(0.5, 0.0))
    res = validate_ma_invertibility(spec)
    assert res.is_valid
    assert dict(res.details)["effective_order"] == "1"

# --- C. GARCH constraints ---

def test_garch_1_1_passes():
    spec = get_base_garch()
    res = validate_garch_constraints(spec)
    assert res.is_valid

def test_garch_omega_0_0_fails():
    spec = replace(get_base_garch(), omega=0.0)
    res = validate_garch_constraints(spec)
    assert not res.is_valid
    assert res.reason == "invalid_omega"

def test_garch_negative_alpha_fails():
    spec = replace(get_base_garch(), alpha_params=(-0.1,))
    res = validate_garch_constraints(spec)
    assert not res.is_valid
    assert res.reason == "negative_alpha"

def test_garch_negative_beta_fails():
    spec = replace(get_base_garch(), beta_params=(-0.8,))
    res = validate_garch_constraints(spec)
    assert not res.is_valid
    assert res.reason == "negative_beta"

def test_garch_persistence_exactly_1_fails():
    spec = replace(get_base_garch(), alpha_params=(0.2,), beta_params=(0.8,))
    res = validate_garch_constraints(spec)
    assert not res.is_valid
    assert res.reason == "invalid_persistence"

def test_garch_persistence_above_1_fails():
    spec = replace(get_base_garch(), alpha_params=(0.3,), beta_params=(0.8,))
    res = validate_garch_constraints(spec)
    assert not res.is_valid
    assert res.reason == "invalid_persistence"

def test_garch_persistence_at_or_above_1_minus_tol_fails():
    val = 1.0 - PERSISTENCE_TOL
    spec = replace(get_base_garch(), alpha_params=(val,), beta_params=(0.0,))
    res = validate_garch_constraints(spec)
    assert not res.is_valid
    assert res.reason == "invalid_persistence"

def test_ar_returns_not_applicable_no_volatility_component():
    spec = get_base_ar()
    res = validate_garch_constraints(spec)
    assert res.is_valid
    assert res.reason == "not_applicable_no_volatility_component"

def test_arma_garch_valid_volatility_passes():
    spec = get_base_c()
    res = validate_garch_constraints(spec)
    assert res.is_valid

# --- D. Composite validation ---

def test_require_math_valid_ar_passes():
    require_math_valid(get_base_ar())

def test_require_math_valid_garch_passes():
    require_math_valid(get_base_garch())

def test_require_math_valid_arma_garch_passes():
    require_math_valid(get_base_c())

def test_require_math_valid_nonstationary_ar_fails():
    spec = replace(get_base_ar(), ar_params=(1.0,))
    with pytest.raises(ValueError) as exc:
        require_math_valid(spec)
    assert "ar_stationarity" in str(exc.value)

def test_require_math_valid_noninvertible_ma_fails():
    spec = replace(get_base_arma(), p=0, q=1, ar_params=(), ma_params=(1.0,))
    with pytest.raises(ValueError) as exc:
        require_math_valid(spec)
    assert "ma_invertibility" in str(exc.value)

def test_require_math_valid_invalid_garch_persistence_fails():
    spec = replace(get_base_garch(), alpha_params=(0.5,), beta_params=(0.6,))
    with pytest.raises(ValueError) as exc:
        require_math_valid(spec)
    assert "garch_constraints" in str(exc.value)

def test_validate_math_constraints_returns_tuple():
    res = validate_math_constraints(get_base_ar())
    assert isinstance(res, tuple)
    for r in res:
        assert isinstance(r, ConstraintResult)

def test_constraint_result_details_is_tuple():
    res = validate_ar_stationarity(get_base_ar())
    assert isinstance(res.details, tuple)

# --- E. Dependency / scope ---

def test_no_torch_imported():
    import subprocess
    import sys
    script = "import sys; import src.phase2.constraints; sys.exit(1 if 'torch' in sys.modules else 0)"
    result = subprocess.run([sys.executable, "-c", script], capture_output=True)
    assert result.returncode == 0, "torch was imported by phase2.constraints"

def test_no_forbidden_modules_imported():
    import subprocess
    import sys
    script = "import sys; import src.phase2.constraints; forbidden = ['src.generator', 'src.dataset', 'src.models', 'src.configs']; sys.exit(1 if any(f in sys.modules for f in forbidden) else 0)"
    result = subprocess.run([sys.executable, "-c", script], capture_output=True)
    assert result.returncode == 0, "A forbidden module was imported by phase2.constraints"
