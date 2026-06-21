# src/phase2/constraints.py

import math
from dataclasses import dataclass
import numpy as np

from src.phase2.schema import (
    ModelSpec,
    FamilyId,
    validate_model_spec
)

VALIDATION_TOLERANCE = 1e-8
PERSISTENCE_TOL = 1e-8
ROOT_BOUNDARY_MARGIN = 1e-6

@dataclass(frozen=True)
class ConstraintResult:
    is_valid: bool
    check_name: str
    reason: str
    details: tuple[tuple[str, str], ...]

def require_constraint_pass(result: ConstraintResult) -> None:
    if not result.is_valid:
        raise ValueError(f"Constraint {result.check_name} failed: {result.reason}")

def _trim_trailing_exact_zeros(params: tuple[float, ...]) -> tuple[float, ...]:
    trimmed = list(params)
    while trimmed and trimmed[-1] == 0.0:
        trimmed.pop()
    return tuple(trimmed)

def _roots_outside_unit_circle(poly_coeffs: list[float]) -> bool:
    # poly_coeffs should be in descending powers: [a_n, a_{n-1}, ..., a_0]
    roots = np.roots(poly_coeffs)
    for r in roots:
        if not np.isfinite(r):
            return False
        if abs(r) <= 1.0 + ROOT_BOUNDARY_MARGIN:
            return False
    return True

def validate_ar_stationarity(spec: ModelSpec) -> ConstraintResult:
    validate_model_spec(spec)
    check_name = "ar_stationarity"
    
    if spec.family_id == FamilyId.GARCH:
        return ConstraintResult(True, check_name, "not_applicable_no_mean_component", ())
        
    if spec.p == 0:
        return ConstraintResult(True, check_name, "not_applicable_no_ar_terms", ())
        
    trimmed_ar = _trim_trailing_exact_zeros(spec.ar_params)
    effective_order = len(trimmed_ar)
    
    if effective_order == 0:
        return ConstraintResult(True, check_name, "no_effective_ar_terms", ())
        
    # AR Polynomial: 1 - phi_1 z - phi_2 z^2 - ... - phi_p z^p = 0
    # numpy.roots expects [a_p, a_{p-1}, ..., a_0]
    # So: [-phi_p, ..., -phi_1, 1.0]
    poly_coeffs = [-phi for phi in reversed(trimmed_ar)] + [1.0]
    
    if _roots_outside_unit_circle(poly_coeffs):
        return ConstraintResult(True, check_name, "valid", (("effective_order", str(effective_order)),))
    else:
        return ConstraintResult(False, check_name, "nonstationary_roots", (("effective_order", str(effective_order)),))

def validate_ma_invertibility(spec: ModelSpec) -> ConstraintResult:
    validate_model_spec(spec)
    check_name = "ma_invertibility"
    
    if spec.family_id in (FamilyId.AR, FamilyId.GARCH):
        return ConstraintResult(True, check_name, "not_applicable_no_ma_terms", ())
        
    if spec.q == 0:
        return ConstraintResult(True, check_name, "not_applicable_no_ma_terms", ())
        
    trimmed_ma = _trim_trailing_exact_zeros(spec.ma_params)
    effective_order = len(trimmed_ma)
    
    if effective_order == 0:
        return ConstraintResult(True, check_name, "no_effective_ma_terms", ())
        
    # MA Polynomial: 1 + theta_1 z + theta_2 z^2 + ... + theta_q z^q = 0
    # numpy.roots expects [a_q, a_{q-1}, ..., a_0]
    # So: [theta_q, ..., theta_1, 1.0]
    poly_coeffs = [theta for theta in reversed(trimmed_ma)] + [1.0]
    
    if _roots_outside_unit_circle(poly_coeffs):
        return ConstraintResult(True, check_name, "valid", (("effective_order", str(effective_order)),))
    else:
        return ConstraintResult(False, check_name, "noninvertible_roots", (("effective_order", str(effective_order)),))

def validate_garch_constraints(spec: ModelSpec) -> ConstraintResult:
    validate_model_spec(spec)
    check_name = "garch_constraints"
    
    if spec.family_id in (FamilyId.AR, FamilyId.ARMA):
        return ConstraintResult(True, check_name, "not_applicable_no_volatility_component", ())
        
    # GARCH / ARMA_GARCH checks
    if spec.omega is None or spec.omega <= 0:
        return ConstraintResult(False, check_name, "invalid_omega", ())
        
    for alpha in spec.alpha_params:
        if alpha < 0:
            return ConstraintResult(False, check_name, "negative_alpha", ())
            
    for beta in spec.beta_params:
        if beta < 0:
            return ConstraintResult(False, check_name, "negative_beta", ())
            
    persistence = sum(spec.alpha_params) + sum(spec.beta_params)
    details = (("persistence", str(persistence)),)
    
    if persistence >= 1.0 - PERSISTENCE_TOL:
        return ConstraintResult(False, check_name, "invalid_persistence", details)
        
    return ConstraintResult(True, check_name, "valid", details)

def validate_math_constraints(spec: ModelSpec) -> tuple[ConstraintResult, ...]:
    return (
        validate_ar_stationarity(spec),
        validate_ma_invertibility(spec),
        validate_garch_constraints(spec)
    )

def require_math_valid(spec: ModelSpec) -> None:
    results = validate_math_constraints(spec)
    failures = [res for res in results if not res.is_valid]
    if failures:
        msg = "; ".join(f"{f.check_name}: {f.reason}" for f in failures)
        raise ValueError(f"Mathematical validation failed: {msg}")
