# tests/test_phase2_sampler.py

import pytest
import math
from src.phase2.schema import (
    APPROVED_MAX_P,
    APPROVED_MAX_Q,
    APPROVED_MAX_R,
    APPROVED_MAX_S,
    FamilyId,
    ModelSpec,
    validate_model_spec,
)
from src.phase2.constraints import require_math_valid
from src.phase2.sampler import (
    GenerationRequest,
    GenerationResult,
    validate_generation_request,
    generate_model_spec,
)


def make_request(
    family_id=FamilyId.AR,
    seed=123,
    p=1,
    q=0,
    r=0,
    s=0,
    ar_range=(-0.5, 0.5),
    ma_range=(-0.5, 0.5),
    omega_range=(0.1, 1.0),
    alpha_range=(0.0, 0.4),
    beta_range=(0.0, 0.4),
    constraint_flags=(1.0, 0.0, 0.0, 0.0),
    provenance=(),
    max_attempts=100,
    **kwargs,
):
    params = {
        "family_id": family_id,
        "seed": seed,
        "p": p,
        "q": q,
        "r": r,
        "s": s,
        "ar_range": ar_range,
        "ma_range": ma_range,
        "omega_range": omega_range,
        "alpha_range": alpha_range,
        "beta_range": beta_range,
        "constraint_flags": constraint_flags,
        "provenance": provenance,
        "max_attempts": max_attempts,
    }
    params.update(kwargs)
    return GenerationRequest(**params)


# --- A. Request Validation ---

def test_reject_string_family_id():
    # 1. Reject raw string family_id.
    with pytest.raises(ValueError) as exc:
        validate_generation_request(make_request(family_id="AR"))
    assert "family_id must be a FamilyId instance" in str(exc.value)


def test_reject_bool_seed():
    # 2. Reject bool seed.
    with pytest.raises(ValueError) as exc:
        validate_generation_request(make_request(seed=True))
    assert "seed must be an int" in str(exc.value)


def test_reject_bool_orders():
    # 3. Reject bool order p/q/r/s.
    for field in ["p", "q", "r", "s"]:
        kwargs = {field: True}
        with pytest.raises(ValueError) as exc:
            validate_generation_request(make_request(**kwargs))
        assert f"{field} must be an int" in str(exc.value)


def test_reject_invalid_max_attempts():
    # 4. Reject max_attempts <= 0.
    with pytest.raises(ValueError) as exc:
        validate_generation_request(make_request(max_attempts=0))
    assert "max_attempts must be a positive int" in str(exc.value)

    with pytest.raises(ValueError) as exc:
        validate_generation_request(make_request(max_attempts=-5))
    assert "max_attempts must be a positive int" in str(exc.value)

    with pytest.raises(ValueError) as exc:
        validate_generation_request(make_request(max_attempts=True))
    assert "max_attempts must be a positive int" in str(exc.value)


def test_reject_list_range():
    # 5. Reject list range instead of tuple.
    with pytest.raises(ValueError) as exc:
        validate_generation_request(make_request(ar_range=[-0.5, 0.5]))
    assert "ar_range must be exactly a tuple" in str(exc.value)


def test_reject_bool_range_endpoint():
    # 6. Reject range with bool endpoint.
    with pytest.raises(ValueError) as exc:
        validate_generation_request(make_request(ar_range=(False, 0.5)))
    assert "ar_range endpoints must be float or int" in str(exc.value)


def test_reject_string_range_endpoint():
    # 7. Reject range with string endpoint.
    with pytest.raises(ValueError) as exc:
        validate_generation_request(make_request(ar_range=(-0.5, "0.5")))
    assert "ar_range endpoints must be float or int" in str(exc.value)


def test_reject_nan_range_endpoint():
    # 8. Reject range with NaN endpoint.
    with pytest.raises(ValueError) as exc:
        validate_generation_request(make_request(ar_range=(float("nan"), 0.5)))
    assert "ar_range endpoints must be finite" in str(exc.value)


def test_reject_inf_range_endpoint():
    # 9. Reject range with inf endpoint.
    with pytest.raises(ValueError) as exc:
        validate_generation_request(make_request(ar_range=(-0.5, float("inf"))))
    assert "ar_range endpoints must be finite" in str(exc.value)


def test_reject_low_greater_than_high():
    # 10. Reject low > high.
    with pytest.raises(ValueError) as exc:
        validate_generation_request(make_request(ar_range=(0.5, -0.5)))
    assert "low endpoint must be <= high endpoint" in str(exc.value)


def test_reject_list_constraint_flags():
    # 11. Reject constraint_flags as list.
    with pytest.raises(ValueError) as exc:
        validate_generation_request(make_request(constraint_flags=[1.0, 0.0, 0.0, 0.0]))
    assert "constraint_flags must be exactly a tuple" in str(exc.value)


def test_reject_list_provenance():
    # 12. Reject provenance as list.
    with pytest.raises(ValueError) as exc:
        validate_generation_request(make_request(provenance=[("a", "b")]))
    assert "provenance must be exactly a tuple" in str(exc.value)


def test_reject_list_provenance_entry():
    # 13. Reject provenance entry as list.
    with pytest.raises(ValueError) as exc:
        validate_generation_request(make_request(provenance=(["a", "b"],)))
    assert "provenance entry must be exactly a tuple" in str(exc.value)


def test_reject_ar_with_q_greater_than_zero():
    # 14. Reject AR request with q > 0.
    with pytest.raises(ValueError) as exc:
        validate_generation_request(make_request(family_id=FamilyId.AR, p=1, q=1))
    assert "AR family requires p > 0, q == 0, r == 0, s == 0" in str(exc.value)


def test_reject_arma_with_no_terms():
    # 15. Reject ARMA request with p + q == 0.
    with pytest.raises(ValueError) as exc:
        validate_generation_request(make_request(family_id=FamilyId.ARMA, p=0, q=0))
    assert "ARMA family requires p >= 0, q >= 0, p+q > 0, r == 0, s == 0" in str(exc.value)


def test_reject_garch_with_mean_terms():
    # 16. Reject GARCH request with p > 0 or q > 0.
    with pytest.raises(ValueError) as exc:
        validate_generation_request(make_request(family_id=FamilyId.GARCH, p=1, q=0, r=1, s=1))
    assert "GARCH family requires p == 0, q == 0, r > 0, s > 0" in str(exc.value)


def test_reject_arma_garch_with_no_volatility_terms():
    # 17. Reject ARMA_GARCH request with r == 0 or s == 0.
    with pytest.raises(ValueError) as exc:
        validate_generation_request(make_request(family_id=FamilyId.ARMA_GARCH, p=1, q=1, r=0, s=1))
    assert "ARMA_GARCH family requires p >= 0, q >= 0, p+q > 0, r > 0, s > 0" in str(exc.value)


def test_reject_order_over_approved_max():
    # 18. Reject order over approved max.
    with pytest.raises(ValueError) as exc:
        validate_generation_request(make_request(family_id=FamilyId.AR, p=APPROVED_MAX_P + 1))
    assert f"p must be between 0 and {APPROVED_MAX_P}" in str(exc.value)


def test_reject_invalid_garch_sign_ranges():
    # GARCH/ARMA_GARCH request validation rejects ranges that cannot produce valid sign constraints
    with pytest.raises(ValueError) as exc:
        validate_generation_request(make_request(
            family_id=FamilyId.GARCH, p=0, q=0, r=1, s=1, omega_range=(0.0, 0.0)
        ))
    assert "omega_range high must be > 0" in str(exc.value)

    with pytest.raises(ValueError) as exc:
        validate_generation_request(make_request(
            family_id=FamilyId.GARCH, p=0, q=0, r=1, s=1, alpha_range=(-0.5, -0.1)
        ))
    assert "alpha_range high must be >= 0" in str(exc.value)

    with pytest.raises(ValueError) as exc:
        validate_generation_request(make_request(
            family_id=FamilyId.GARCH, p=0, q=0, r=1, s=1, beta_range=(-0.5, -0.1)
        ))
    assert "beta_range high must be >= 0" in str(exc.value)


# --- B. Valid Generation & C. Determinism ---

def test_generate_valid_ar():
    # 19. Generate valid AR spec.
    # 24. Each generated spec passes validate_model_spec.
    # 25. Each generated spec passes require_math_valid.
    req = make_request(family_id=FamilyId.AR, p=2, q=0, r=0, s=0, ar_range=(-0.4, 0.4))
    res = generate_model_spec(req)
    assert res.family_id == FamilyId.AR
    assert res.attempts >= 1
    assert res.spec.p == 2
    assert res.spec.q == 0
    assert len(res.spec.ar_params) == 2
    assert all(-0.4 <= phi <= 0.4 for phi in res.spec.ar_params)
    validate_model_spec(res.spec)
    require_math_valid(res.spec)


def test_generate_valid_arma():
    # 20. Generate valid ARMA spec.
    req = make_request(family_id=FamilyId.ARMA, p=2, q=2, r=0, s=0, ar_range=(-0.3, 0.3), ma_range=(-0.3, 0.3))
    res = generate_model_spec(req)
    assert res.family_id == FamilyId.ARMA
    assert res.spec.p == 2
    assert res.spec.q == 2
    assert len(res.spec.ar_params) == 2
    assert len(res.spec.ma_params) == 2
    validate_model_spec(res.spec)
    require_math_valid(res.spec)


def test_generate_valid_pure_ma():
    # 21. Generate valid pure MA spec as ARMA(0,q).
    req = make_request(family_id=FamilyId.ARMA, p=0, q=2, r=0, s=0, ma_range=(-0.4, 0.4))
    res = generate_model_spec(req)
    assert res.family_id == FamilyId.ARMA
    assert res.spec.p == 0
    assert res.spec.q == 2
    assert len(res.spec.ma_params) == 2
    validate_model_spec(res.spec)
    require_math_valid(res.spec)


def test_generate_valid_garch():
    # 22. Generate valid GARCH spec.
    req = make_request(
        family_id=FamilyId.GARCH, p=0, q=0, r=1, s=1,
        omega_range=(0.1, 0.5), alpha_range=(0.1, 0.3), beta_range=(0.1, 0.5)
    )
    res = generate_model_spec(req)
    assert res.family_id == FamilyId.GARCH
    assert res.spec.omega is not None and 0.1 <= res.spec.omega <= 0.5
    assert len(res.spec.alpha_params) == 1
    assert len(res.spec.beta_params) == 1
    validate_model_spec(res.spec)
    require_math_valid(res.spec)


def test_generate_valid_arma_garch():
    # 23. Generate valid ARMA_GARCH spec.
    req = make_request(
        family_id=FamilyId.ARMA_GARCH, p=1, q=1, r=1, s=1,
        ar_range=(-0.2, 0.2), ma_range=(-0.2, 0.2),
        omega_range=(0.1, 0.5), alpha_range=(0.1, 0.3), beta_range=(0.1, 0.5)
    )
    res = generate_model_spec(req)
    assert res.family_id == FamilyId.ARMA_GARCH
    validate_model_spec(res.spec)
    require_math_valid(res.spec)


def test_determinism():
    # 26. Same request produces identical spec and attempts.
    # 27. Same seed and same request does not mutate provenance or constraint flags.
    req1 = make_request(family_id=FamilyId.AR, seed=42, p=2, q=0, ar_range=(-0.8, 0.8))
    req2 = make_request(family_id=FamilyId.AR, seed=42, p=2, q=0, ar_range=(-0.8, 0.8))
    
    res1 = generate_model_spec(req1)
    res2 = generate_model_spec(req2)
    
    assert res1.spec == res2.spec
    assert res1.attempts == res2.attempts
    assert res1.spec.provenance == req1.provenance
    assert res1.spec.constraint_flags == req1.constraint_flags


def test_different_seeds():
    # 28. Different explicit seed may produce a valid spec; do not assert it must differ.
    req1 = make_request(family_id=FamilyId.AR, seed=100, p=2, q=0, ar_range=(-0.8, 0.8))
    req2 = make_request(family_id=FamilyId.AR, seed=200, p=2, q=0, ar_range=(-0.8, 0.8))
    
    res1 = generate_model_spec(req1)
    res2 = generate_model_spec(req2)
    
    assert isinstance(res1.spec, ModelSpec)
    assert isinstance(res2.spec, ModelSpec)


# --- D. Failure Behavior ---

def test_impossible_ar_range():
    # 29. Impossible AR range, e.g. ar_range=(1.2, 1.2), fails after max_attempts.
    # 32. Failure message includes family_id, seed, and max_attempts.
    req = make_request(family_id=FamilyId.AR, seed=123, p=1, q=0, ar_range=(1.2, 1.2), max_attempts=5)
    with pytest.raises(ValueError) as exc:
        generate_model_spec(req)
    msg = str(exc.value)
    assert "AR" in msg
    assert "123" in msg
    assert "5" in msg
    assert "no_valid_spec_found" in msg


def test_impossible_ma_range():
    # 30. Impossible MA range, e.g. ma_range=(1.2, 1.2), fails after max_attempts.
    req = make_request(family_id=FamilyId.ARMA, seed=123, p=0, q=1, ma_range=(1.2, 1.2), max_attempts=5)
    with pytest.raises(ValueError) as exc:
        generate_model_spec(req)
    msg = str(exc.value)
    assert "ARMA" in msg
    assert "123" in msg
    assert "5" in msg
    assert "no_valid_spec_found" in msg


def test_impossible_garch_persistence():
    # 31. Impossible GARCH persistence range, e.g. alpha_range=(0.9, 0.9), beta_range=(0.9, 0.9), fails after max_attempts.
    req = make_request(
        family_id=FamilyId.GARCH, seed=123, p=0, q=0, r=1, s=1,
        omega_range=(0.1, 0.2), alpha_range=(0.9, 0.9), beta_range=(0.9, 0.9),
        max_attempts=5
    )
    with pytest.raises(ValueError) as exc:
        generate_model_spec(req)
    msg = str(exc.value)
    assert "GARCH" in msg
    assert "123" in msg
    assert "5" in msg
    assert "no_valid_spec_found" in msg


# --- E. Scope & Dependencies ---

def test_no_torch_imported():
    # 33. Tests do not import torch.
    import subprocess
    import sys
    script = "import sys; import src.phase2.sampler; sys.exit(1 if 'torch' in sys.modules else 0)"
    result = subprocess.run([sys.executable, "-c", script], capture_output=True)
    assert result.returncode == 0, "torch was imported by phase2.sampler"


def test_no_forbidden_modules_imported():
    # 34. Tests do not import generator/dataset/model/training modules.
    import subprocess
    import sys
    script = "import sys; import src.phase2.sampler; forbidden = ['src.generator', 'src.dataset', 'src.models', 'src.configs']; sys.exit(1 if any(f in sys.modules for f in forbidden) else 0)"
    result = subprocess.run([sys.executable, "-c", script], capture_output=True)
    assert result.returncode == 0, "A forbidden module was imported by phase2.sampler"


def test_no_direct_numpy_in_sampler():
    # Sampler does not import numpy directly (standard library only).
    import os
    sampler_path = os.path.join("src", "phase2", "sampler.py")
    with open(sampler_path, "r", encoding="utf-8") as f:
        content = f.read()
    
    assert "import numpy" not in content, "Direct 'import numpy' found in sampler.py"
    assert "from numpy" not in content, "Direct 'from numpy' found in sampler.py"
