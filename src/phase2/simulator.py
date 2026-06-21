# src/phase2/simulator.py

import math
import random
from dataclasses import dataclass
from typing import Optional

from src.phase2.schema import (
    FamilyId,
    ModelSpec,
    validate_model_spec,
)
from src.phase2.constraints import require_math_valid


@dataclass(frozen=True)
class SimulationRequest:
    spec: ModelSpec
    seed: int
    length: int
    burn_in: int
    innovation_distribution: str
    innovation_std: float
    initial_value: float
    initial_innovation: float
    initial_variance: float
    max_abs_value: float


@dataclass(frozen=True)
class SimulationResult:
    values: tuple[float, ...]
    innovations: tuple[float, ...]
    variances: tuple[float, ...]
    request_seed: int
    family_id: FamilyId
    length: int
    burn_in: int
    reason: str


def validate_simulation_request(request: SimulationRequest) -> None:
    # 1. spec is ModelSpec instance
    if not isinstance(request.spec, ModelSpec):
        raise ValueError("spec must be a ModelSpec instance")

    # 2. validate_model_spec and require_math_valid
    validate_model_spec(request.spec)
    require_math_valid(request.spec)

    # 3. seed is int but not bool
    if type(request.seed) is not int:
        raise ValueError("seed must be an int, not bool")

    # 4. length is int, not bool, > 0
    if type(request.length) is not int or request.length <= 0:
        raise ValueError("length must be a positive int, not bool")

    # 5. burn_in is int, not bool, >= 0
    if type(request.burn_in) is not int or request.burn_in < 0:
        raise ValueError("burn_in must be a non-negative int, not bool")

    # 6. innovation_distribution is exactly "gaussian"
    if request.innovation_distribution != "gaussian":
        raise ValueError("innovation_distribution must be exactly 'gaussian'")

    # 7. innovation_std validation
    if type(request.innovation_std) not in (int, float) or not math.isfinite(request.innovation_std) or request.innovation_std <= 0:
        raise ValueError("innovation_std must be a positive finite float/int, not bool")

    # 8. initial_value validation
    if type(request.initial_value) not in (int, float) or not math.isfinite(request.initial_value):
        raise ValueError("initial_value must be a finite float/int, not bool")

    # 9. initial_innovation validation
    if type(request.initial_innovation) not in (int, float) or not math.isfinite(request.initial_innovation):
        raise ValueError("initial_innovation must be a finite float/int, not bool")

    # 10. initial_variance validation
    if type(request.initial_variance) not in (int, float) or not math.isfinite(request.initial_variance) or request.initial_variance <= 0:
        raise ValueError("initial_variance must be a positive finite float/int, not bool")

    # 11. max_abs_value validation
    if type(request.max_abs_value) not in (int, float) or not math.isfinite(request.max_abs_value) or request.max_abs_value <= 0:
        raise ValueError("max_abs_value must be a positive finite float/int, not bool")


def simulate_time_series(request: SimulationRequest) -> SimulationResult:
    # Validate the request
    validate_simulation_request(request)

    # Initialize random number generator
    rng = random.Random(request.seed)

    # Setup history lookback size
    spec = request.spec
    lookback = max(1, spec.p, spec.q, spec.r, spec.s)

    # Pre-populate history arrays using the required explicit initializations
    y_hist = [request.initial_value] * lookback
    epsilon_hist = [request.initial_innovation] * lookback
    h_hist = [request.initial_variance] * lookback

    total_steps = request.length + request.burn_in

    # Volatility and Mean simulation loop
    for t in range(total_steps):
        idx = lookback + t

        # 1. Compute variance h_t
        if spec.family_id in (FamilyId.GARCH, FamilyId.ARMA_GARCH):
            # GARCH variance recursion
            omega = spec.omega
            alpha_sum = sum(spec.alpha_params[i] * (epsilon_hist[idx - 1 - i] ** 2) for i in range(spec.r))
            beta_sum = sum(spec.beta_params[j] * h_hist[idx - 1 - j] for j in range(spec.s))
            h_t = omega + alpha_sum + beta_sum
        else:
            # AR / ARMA constant variance
            h_t = request.innovation_std ** 2

        # Validate h_t
        if not math.isfinite(h_t) or h_t <= 0:
            raise ValueError(f"Invalid variance calculated at step {t}: {h_t}")

        # 2. Draw standard normal innovation and scale
        z_t = rng.gauss(0.0, 1.0)
        if spec.family_id in (FamilyId.GARCH, FamilyId.ARMA_GARCH):
            epsilon_t = math.sqrt(h_t) * z_t
        else:
            epsilon_t = request.innovation_std * z_t

        # Validate epsilon_t
        if not math.isfinite(epsilon_t):
            raise ValueError(f"Non-finite innovation calculated at step {t}: {epsilon_t}")

        # 3. Compute mean equation value y_t
        if spec.family_id == FamilyId.GARCH:
            y_t = epsilon_t
        else:
            ar_sum = sum(spec.ar_params[i] * y_hist[idx - 1 - i] for i in range(spec.p))
            ma_sum = sum(spec.ma_params[j] * epsilon_hist[idx - 1 - j] for j in range(spec.q))
            y_t = ar_sum + epsilon_t + ma_sum

        # Validate y_t finiteness and absolute bound
        if not math.isfinite(y_t):
            raise ValueError(f"Non-finite value calculated at step {t}: {y_t}")
        if abs(y_t) > request.max_abs_value:
            raise ValueError(f"Value {y_t} at step {t} exceeds max_abs_value threshold: {request.max_abs_value}")

        # Append to history buffers
        y_hist.append(y_t)
        epsilon_hist.append(epsilon_t)
        h_hist.append(h_t)

    # Slice out burn-in and return the final length paths as tuples
    start_slice = lookback + request.burn_in
    values = tuple(y_hist[start_slice:])
    innovations = tuple(epsilon_hist[start_slice:])
    variances = tuple(h_hist[start_slice:])

    return SimulationResult(
        values=values,
        innovations=innovations,
        variances=variances,
        request_seed=request.seed,
        family_id=spec.family_id,
        length=request.length,
        burn_in=request.burn_in,
        reason="simulation_completed",
    )
