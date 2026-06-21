# src/phase2/sampler.py

import math
import random
from dataclasses import dataclass
from typing import Optional

from src.phase2.schema import (
    APPROVED_MAX_P,
    APPROVED_MAX_Q,
    APPROVED_MAX_R,
    APPROVED_MAX_S,
    FamilyId,
    MeanFamily,
    VolatilityFamily,
    ModelSpec,
    validate_model_spec,
)
from src.phase2.constraints import require_math_valid


@dataclass(frozen=True)
class GenerationRequest:
    family_id: FamilyId
    seed: int
    p: int
    q: int
    r: int
    s: int
    ar_range: tuple[float, float]
    ma_range: tuple[float, float]
    omega_range: tuple[float, float]
    alpha_range: tuple[float, float]
    beta_range: tuple[float, float]
    constraint_flags: tuple[float, float, float, float]
    provenance: tuple[tuple[str, str], ...]
    max_attempts: int


@dataclass(frozen=True)
class GenerationResult:
    spec: ModelSpec
    attempts: int
    request_seed: int
    family_id: FamilyId
    reason: str


def _validate_range(val: tuple[float, float], name: str) -> None:
    if type(val) is not tuple:
        raise ValueError(f"{name} must be exactly a tuple, got {type(val)}")
    if len(val) != 2:
        raise ValueError(f"{name} must have length exactly 2, got {len(val)}")
    for endpoint in val:
        if type(endpoint) not in (int, float):
            raise ValueError(f"{name} endpoints must be float or int, got {type(endpoint)}")
        if not math.isfinite(endpoint):
            raise ValueError(f"{name} endpoints must be finite, got {endpoint}")
    if val[0] > val[1]:
        raise ValueError(f"{name} low endpoint must be <= high endpoint, got {val}")


def validate_generation_request(request: GenerationRequest) -> None:
    # 1. family_id is FamilyId instance
    if not isinstance(request.family_id, FamilyId):
        raise ValueError("family_id must be a FamilyId instance")

    # 2. seed is int but not bool
    if type(request.seed) is not int:
        raise ValueError("seed must be an int, not bool or other type")

    # 3. p/q/r/s are int but not bool
    for order_name, order_val in [
        ("p", request.p),
        ("q", request.q),
        ("r", request.r),
        ("s", request.s),
    ]:
        if type(order_val) is not int:
            raise ValueError(f"{order_name} must be an int, not bool or other type")

    # 4. max_attempts is int but not bool and > 0
    if type(request.max_attempts) is not int or request.max_attempts <= 0:
        raise ValueError("max_attempts must be a positive int, not bool or other type")

    # 5. Structurally validate all ranges always
    _validate_range(request.ar_range, "ar_range")
    _validate_range(request.ma_range, "ma_range")
    _validate_range(request.omega_range, "omega_range")
    _validate_range(request.alpha_range, "alpha_range")
    _validate_range(request.beta_range, "beta_range")

    # 6. constraint_flags validation
    if type(request.constraint_flags) is not tuple:
        raise ValueError("constraint_flags must be exactly a tuple")
    if len(request.constraint_flags) != 4:
        raise ValueError("constraint_flags must have length exactly 4")
    for val in request.constraint_flags:
        if type(val) not in (int, float):
            raise ValueError(f"constraint_flags elements must be float or int, got {type(val)}")
        if not math.isfinite(val):
            raise ValueError(f"constraint_flags elements must be finite, got {val}")

    # 7. provenance validation
    if type(request.provenance) is not tuple:
        raise ValueError("provenance must be exactly a tuple")
    for entry in request.provenance:
        if type(entry) is not tuple:
            raise ValueError("provenance entry must be exactly a tuple")
        if len(entry) != 2:
            raise ValueError("provenance entry must have length exactly 2")
        if type(entry[0]) is not str or type(entry[1]) is not str:
            raise ValueError("provenance entry elements must be strings")

    # 8. Order bounds checking
    if not (0 <= request.p <= APPROVED_MAX_P):
        raise ValueError(f"p must be between 0 and {APPROVED_MAX_P}, got {request.p}")
    if not (0 <= request.q <= APPROVED_MAX_Q):
        raise ValueError(f"q must be between 0 and {APPROVED_MAX_Q}, got {request.q}")
    if not (0 <= request.r <= APPROVED_MAX_R):
        raise ValueError(f"r must be between 0 and {APPROVED_MAX_R}, got {request.r}")
    if not (0 <= request.s <= APPROVED_MAX_S):
        raise ValueError(f"s must be between 0 and {APPROVED_MAX_S}, got {request.s}")

    # 9. Order/family consistency validation
    if request.family_id == FamilyId.AR:
        if request.p <= 0 or request.q != 0 or request.r != 0 or request.s != 0:
            raise ValueError(
                f"AR family requires p > 0, q == 0, r == 0, s == 0. "
                f"Got p={request.p}, q={request.q}, r={request.r}, s={request.s}"
            )
    elif request.family_id == FamilyId.ARMA:
        if request.p < 0 or request.q < 0 or (request.p + request.q == 0) or request.r != 0 or request.s != 0:
            raise ValueError(
                f"ARMA family requires p >= 0, q >= 0, p+q > 0, r == 0, s == 0. "
                f"Got p={request.p}, q={request.q}, r={request.r}, s={request.s}"
            )
    elif request.family_id == FamilyId.GARCH:
        if request.p != 0 or request.q != 0 or request.r <= 0 or request.s <= 0:
            raise ValueError(
                f"GARCH family requires p == 0, q == 0, r > 0, s > 0. "
                f"Got p={request.p}, q={request.q}, r={request.r}, s={request.s}"
            )
    elif request.family_id == FamilyId.ARMA_GARCH:
        if request.p < 0 or request.q < 0 or (request.p + request.q == 0) or request.r <= 0 or request.s <= 0:
            raise ValueError(
                f"ARMA_GARCH family requires p >= 0, q >= 0, p+q > 0, r > 0, s > 0. "
                f"Got p={request.p}, q={request.q}, r={request.r}, s={request.s}"
            )

    # 10. GARCH/ARMA_GARCH sign constraints check on ranges
    if request.family_id in (FamilyId.GARCH, FamilyId.ARMA_GARCH):
        if request.omega_range[1] <= 0:
            raise ValueError(f"omega_range high must be > 0 for GARCH/ARMA_GARCH, got {request.omega_range[1]}")
        if request.alpha_range[1] < 0:
            raise ValueError(f"alpha_range high must be >= 0 for GARCH/ARMA_GARCH, got {request.alpha_range[1]}")
        if request.beta_range[1] < 0:
            raise ValueError(f"beta_range high must be >= 0 for GARCH/ARMA_GARCH, got {request.beta_range[1]}")


def _sample_float(rng: random.Random, value_range: tuple[float, float]) -> float:
    """Samples a single float uniformly from the given range [low, high]."""
    return rng.uniform(value_range[0], value_range[1])


def _sample_tuple(rng: random.Random, count: int, value_range: tuple[float, float]) -> tuple[float, ...]:
    """Samples a tuple of floats of length `count` uniformly from the given range [low, high]."""
    return tuple(_sample_float(rng, value_range) for _ in range(count))


def generate_model_spec(request: GenerationRequest) -> GenerationResult:
    # Validate request
    validate_generation_request(request)

    # Deterministic RNG initialization
    rng = random.Random(request.seed)

    # Rejection sampling loop
    for attempt in range(1, request.max_attempts + 1):
        # Sample parameter tuples depending on orders
        ar_params = _sample_tuple(rng, request.p, request.ar_range) if request.p > 0 else ()
        ma_params = _sample_tuple(rng, request.q, request.ma_range) if request.q > 0 else ()
        
        # Sample GARCH parameters
        if request.family_id in (FamilyId.GARCH, FamilyId.ARMA_GARCH):
            omega = _sample_float(rng, request.omega_range)
            alpha_params = _sample_tuple(rng, request.r, request.alpha_range)
            beta_params = _sample_tuple(rng, request.s, request.beta_range)
        else:
            omega = None
            alpha_params = ()
            beta_params = ()

        # Map to mean_family and volatility_family
        if request.family_id == FamilyId.AR:
            mean_family = MeanFamily.AR
            volatility_family = VolatilityFamily.NONE
        elif request.family_id == FamilyId.ARMA:
            mean_family = MeanFamily.ARMA
            volatility_family = VolatilityFamily.NONE
        elif request.family_id == FamilyId.GARCH:
            mean_family = MeanFamily.NONE
            volatility_family = VolatilityFamily.GARCH
        elif request.family_id == FamilyId.ARMA_GARCH:
            mean_family = MeanFamily.ARMA
            volatility_family = VolatilityFamily.GARCH
        else:
            raise ValueError(f"Unsupported family_id: {request.family_id}")

        # Construct candidate spec
        spec = ModelSpec(
            family_id=request.family_id,
            mean_family=mean_family,
            volatility_family=volatility_family,
            p=request.p,
            q=request.q,
            r=request.r,
            s=request.s,
            ar_params=ar_params,
            ma_params=ma_params,
            omega=omega,
            alpha_params=alpha_params,
            beta_params=beta_params,
            constraint_flags=request.constraint_flags,
            provenance=request.provenance,
        )

        # Validate spec through standard schema guards and mathematical validity guards
        try:
            validate_model_spec(spec)
            require_math_valid(spec)
            # If both pass, return the result
            return GenerationResult(
                spec=spec,
                attempts=attempt,
                request_seed=request.seed,
                family_id=request.family_id,
                reason="valid_spec_generated",
            )
        except ValueError:
            # Continue to next attempt
            continue

    # Exceeded max_attempts without finding a valid spec
    raise ValueError(
        f"Failed to generate valid ModelSpec for family_id={request.family_id.value} "
        f"with seed={request.seed} after max_attempts={request.max_attempts}. "
        f"Reason: no_valid_spec_found"
    )
