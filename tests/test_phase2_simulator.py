# tests/test_phase2_simulator.py

import pytest
import math
from src.phase2.schema import (
    FamilyId,
    ModelSpec,
    MeanFamily,
    VolatilityFamily,
)
from src.phase2.simulator import (
    SimulationRequest,
    SimulationResult,
    validate_simulation_request,
    simulate_time_series,
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


def make_sim_request(
    spec=None,
    seed=123,
    length=100,
    burn_in=10,
    innovation_distribution="gaussian",
    innovation_std=1.0,
    initial_value=0.0,
    initial_innovation=0.0,
    initial_variance=1.0,
    max_abs_value=1_000_000.0,
    **kwargs
):
    if spec is None:
        spec = get_base_ar()
    params = {
        "spec": spec,
        "seed": seed,
        "length": length,
        "burn_in": burn_in,
        "innovation_distribution": innovation_distribution,
        "innovation_std": innovation_std,
        "initial_value": initial_value,
        "initial_innovation": initial_innovation,
        "initial_variance": initial_variance,
        "max_abs_value": max_abs_value,
    }
    params.update(kwargs)
    return SimulationRequest(**params)


# --- A. Request Validation ---

def test_reject_non_modelspec():
    # 1. Reject raw non-ModelSpec spec.
    with pytest.raises(ValueError) as exc:
        validate_simulation_request(make_sim_request(spec="not_a_modelspec"))
    assert "spec must be a ModelSpec instance" in str(exc.value)


def test_reject_invalid_schema_spec():
    # 2. Reject invalid schema spec.
    from src.phase2.schema import APPROVED_MAX_P
    from dataclasses import replace
    bad_spec = replace(get_base_ar(), p=APPROVED_MAX_P + 1, ar_params=(0.5,) * (APPROVED_MAX_P + 1))
    with pytest.raises(ValueError):
        validate_simulation_request(make_sim_request(spec=bad_spec))


def test_reject_mathematically_invalid_spec():
    # 3. Reject mathematically invalid spec.
    from dataclasses import replace
    bad_spec = replace(get_base_ar(), ar_params=(1.0,))
    with pytest.raises(ValueError):
        validate_simulation_request(make_sim_request(spec=bad_spec))


def test_reject_bool_seed():
    # 4. Reject bool seed.
    with pytest.raises(ValueError):
        validate_simulation_request(make_sim_request(seed=True))


def test_reject_non_positive_length():
    # 5. Reject non-positive length.
    with pytest.raises(ValueError):
        validate_simulation_request(make_sim_request(length=0))
    with pytest.raises(ValueError):
        validate_simulation_request(make_sim_request(length=-10))


def test_reject_negative_burn_in():
    # 6. Reject negative burn_in.
    with pytest.raises(ValueError):
        validate_simulation_request(make_sim_request(burn_in=-5))


def test_reject_bool_length_burn_in():
    # 7. Reject bool length/burn_in.
    with pytest.raises(ValueError):
        validate_simulation_request(make_sim_request(length=True))
    with pytest.raises(ValueError):
        validate_simulation_request(make_sim_request(burn_in=True))


def test_reject_unsupported_distribution():
    # 8. Reject unsupported innovation_distribution.
    with pytest.raises(ValueError):
        validate_simulation_request(make_sim_request(innovation_distribution="student-t"))


def test_reject_string_innovation_std():
    # 9. Reject string innovation_std.
    with pytest.raises(ValueError):
        validate_simulation_request(make_sim_request(innovation_std="1.0"))


def test_reject_bool_innovation_std():
    # 10. Reject bool innovation_std.
    with pytest.raises(ValueError):
        validate_simulation_request(make_sim_request(innovation_std=True))


def test_reject_nan_innovation_std():
    # 11. Reject NaN innovation_std.
    with pytest.raises(ValueError):
        validate_simulation_request(make_sim_request(innovation_std=float("nan")))


def test_reject_inf_initial_value():
    # 12. Reject inf initial_value.
    with pytest.raises(ValueError):
        validate_simulation_request(make_sim_request(initial_value=float("inf")))


def test_reject_non_positive_initial_variance():
    # 13. Reject non-positive initial_variance.
    with pytest.raises(ValueError):
        validate_simulation_request(make_sim_request(initial_variance=0.0))
    with pytest.raises(ValueError):
        validate_simulation_request(make_sim_request(initial_variance=-5.0))


def test_reject_non_positive_max_abs_value():
    # 14. Reject non-positive max_abs_value.
    with pytest.raises(ValueError):
        validate_simulation_request(make_sim_request(max_abs_value=0.0))
    with pytest.raises(ValueError):
        validate_simulation_request(make_sim_request(max_abs_value=-10.0))


# --- B. Valid Simulation ---

def test_simulate_valid_ar():
    # 15. Simulate valid AR spec.
    # 20. Result values length equals request.length.
    # 21. Result innovations length equals request.length.
    # 22. Result variances length equals request.length.
    # 23. All returned values are finite.
    # 24. All returned innovations are finite.
    # 25. All returned variances are finite and > 0.
    # 26. Result outputs are tuples, not lists.
    # 27. Result metadata matches request seed/family/length/burn_in.
    req = make_sim_request(spec=get_base_ar(), length=50, burn_in=20, seed=42)
    res = simulate_time_series(req)
    assert res.family_id == FamilyId.AR
    assert res.request_seed == 42
    assert res.length == 50
    assert res.burn_in == 20
    assert len(res.values) == 50
    assert len(res.innovations) == 50
    assert len(res.variances) == 50
    assert isinstance(res.values, tuple)
    assert isinstance(res.innovations, tuple)
    assert isinstance(res.variances, tuple)
    assert all(math.isfinite(x) for x in res.values)
    assert all(math.isfinite(x) for x in res.innovations)
    assert all(math.isfinite(x) and x > 0 for x in res.variances)


def test_simulate_valid_arma():
    # 16. Simulate valid ARMA spec.
    req = make_sim_request(spec=get_base_arma(), length=40)
    res = simulate_time_series(req)
    assert res.family_id == FamilyId.ARMA
    assert len(res.values) == 40


def test_simulate_valid_pure_ma():
    # 17. Simulate valid pure MA as ARMA(0,q).
    from dataclasses import replace
    ma_spec = replace(get_base_arma(), p=0, ar_params=())
    req = make_sim_request(spec=ma_spec, length=35)
    res = simulate_time_series(req)
    assert res.family_id == FamilyId.ARMA
    assert len(res.values) == 35


def test_simulate_valid_garch():
    # 18. Simulate valid GARCH spec.
    req = make_sim_request(spec=get_base_garch(), length=60)
    res = simulate_time_series(req)
    assert res.family_id == FamilyId.GARCH
    assert len(res.values) == 60


def test_simulate_valid_arma_garch():
    # 19. Simulate valid ARMA_GARCH spec.
    req = make_sim_request(spec=get_base_c(), length=75)
    res = simulate_time_series(req)
    assert res.family_id == FamilyId.ARMA_GARCH
    assert len(res.values) == 75


# --- C. Determinism ---

def test_determinism():
    # 28. Same request produces identical values, innovations, variances.
    req1 = make_sim_request(spec=get_base_c(), seed=888, length=50)
    req2 = make_sim_request(spec=get_base_c(), seed=888, length=50)
    
    res1 = simulate_time_series(req1)
    res2 = simulate_time_series(req2)
    
    assert res1.values == res2.values
    assert res1.innovations == res2.innovations
    assert res1.variances == res2.variances


def test_different_seeds():
    # 29. Different seed produces valid result; do not assert it must differ.
    req1 = make_sim_request(spec=get_base_ar(), seed=10, length=30)
    req2 = make_sim_request(spec=get_base_ar(), seed=20, length=30)
    
    res1 = simulate_time_series(req1)
    res2 = simulate_time_series(req2)
    
    assert len(res1.values) == 30
    assert len(res2.values) == 30


# --- D. Burn-in ---

def test_burn_in_behavior():
    # 30. Same request with burn_in=0 returns length values.
    # 31. Same request with burn_in>0 returns length values.
    # 32. Burn-in changes the generated returned path for a deterministic seed in at least one simple AR or GARCH case.
    req_no_burn = make_sim_request(spec=get_base_ar(), seed=555, length=50, burn_in=0)
    req_with_burn = make_sim_request(spec=get_base_ar(), seed=555, length=50, burn_in=15)
    
    res_no_burn = simulate_time_series(req_no_burn)
    res_with_burn = simulate_time_series(req_with_burn)
    
    assert len(res_no_burn.values) == 50
    assert len(res_with_burn.values) == 50
    assert res_no_burn.values != res_with_burn.values


# --- E. Runtime Failure Behavior ---

def test_runtime_failure_max_abs_value():
    # 33. Very small max_abs_value raises ValueError.
    # 34. Failure message for max_abs_value includes index/value.
    req = make_sim_request(spec=get_base_ar(), length=50, max_abs_value=0.001)
    with pytest.raises(ValueError) as exc:
        simulate_time_series(req)
    assert "step" in str(exc.value) or "index" in str(exc.value)
    assert "exceeds max_abs_value" in str(exc.value)


def test_variance_validation_failure_prevents_bad_path():
    # 35. Bad variance runtime path raises ValueError if possible via constructed invalid initial variance is already caught at validation; document this in test with validation failure.
    with pytest.raises(ValueError) as exc:
        validate_simulation_request(make_sim_request(initial_variance=-5.0))
    assert "initial_variance" in str(exc.value)


# --- F. Scope ---

def test_no_torch_imported():
    # 36. Tests do not import torch.
    import subprocess
    import sys
    script = "import sys; import src.phase2.simulator; sys.exit(1 if 'torch' in sys.modules else 0)"
    result = subprocess.run([sys.executable, "-c", script], capture_output=True)
    assert result.returncode == 0, "torch was imported by phase2.simulator"


def test_no_forbidden_modules_imported():
    # 37. Tests do not import dataset/model/training modules.
    import subprocess
    import sys
    script = "import sys; import src.phase2.simulator; forbidden = ['src.generator', 'src.dataset', 'src.models', 'src.configs']; sys.exit(1 if any(f in sys.modules for f in forbidden) else 0)"
    result = subprocess.run([sys.executable, "-c", script], capture_output=True)
    assert result.returncode == 0, "A forbidden module was imported by phase2.simulator"


def test_no_direct_numpy_in_simulator():
    # 38. Simulator does not import numpy directly (standard library only).
    import os
    simulator_path = os.path.join("src", "phase2", "simulator.py")
    with open(simulator_path, "r", encoding="utf-8") as f:
        content = f.read()
    
    assert "import numpy" not in content, "Direct 'import numpy' found in simulator.py"
    assert "from numpy" not in content, "Direct 'from numpy' found in simulator.py"
