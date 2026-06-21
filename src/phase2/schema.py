# src/phase2/schema.py

from dataclasses import dataclass
from enum import Enum
from typing import Optional

# Approved protocol constants from P3
APPROVED_MAX_P = 5
APPROVED_MAX_Q = 5
APPROVED_MAX_R = 2
APPROVED_MAX_S = 2
SCHEMA_V2_MIN_FLAT_DIM = 32

class FamilyId(str, Enum):
    AR = "AR"
    ARMA = "ARMA"
    GARCH = "GARCH"
    ARMA_GARCH = "ARMA_GARCH"

class MeanFamily(str, Enum):
    NONE = "NONE"
    AR = "AR"
    ARMA = "ARMA"

class VolatilityFamily(str, Enum):
    NONE = "NONE"
    GARCH = "GARCH"

@dataclass(frozen=True)
class ModelSpec:
    family_id: FamilyId
    mean_family: MeanFamily
    volatility_family: VolatilityFamily
    p: int
    q: int
    r: int
    s: int
    ar_params: tuple[float, ...]
    ma_params: tuple[float, ...]
    omega: Optional[float]
    alpha_params: tuple[float, ...]
    beta_params: tuple[float, ...]
    constraint_flags: tuple[float, float, float, float]
    provenance: tuple[tuple[str, str], ...]

def validate_order_bounds(spec: ModelSpec) -> None:
    if not (0 <= spec.p <= APPROVED_MAX_P):
        raise ValueError(f"p must be between 0 and {APPROVED_MAX_P}, got {spec.p}")
    if not (0 <= spec.q <= APPROVED_MAX_Q):
        raise ValueError(f"q must be between 0 and {APPROVED_MAX_Q}, got {spec.q}")
    if not (0 <= spec.r <= APPROVED_MAX_R):
        raise ValueError(f"r must be between 0 and {APPROVED_MAX_R}, got {spec.r}")
    if not (0 <= spec.s <= APPROVED_MAX_S):
        raise ValueError(f"s must be between 0 and {APPROVED_MAX_S}, got {spec.s}")

def is_a_family(spec: ModelSpec) -> bool:
    return spec.family_id in (FamilyId.AR, FamilyId.ARMA)

def is_b_family(spec: ModelSpec) -> bool:
    return spec.family_id == FamilyId.GARCH

def is_c_family(spec: ModelSpec) -> bool:
    return spec.family_id == FamilyId.ARMA_GARCH

def validate_enum_types(spec: ModelSpec) -> None:
    if not isinstance(spec.family_id, FamilyId):
        raise ValueError(f"family_id must be a FamilyId enum instance, got {type(spec.family_id)}")
    if not isinstance(spec.mean_family, MeanFamily):
        raise ValueError(f"mean_family must be a MeanFamily enum instance, got {type(spec.mean_family)}")
    if not isinstance(spec.volatility_family, VolatilityFamily):
        raise ValueError(f"volatility_family must be a VolatilityFamily enum instance, got {type(spec.volatility_family)}")

def validate_immutable_tuple_fields(spec: ModelSpec) -> None:
    for field_name in ["ar_params", "ma_params", "alpha_params", "beta_params", "constraint_flags", "provenance"]:
        val = getattr(spec, field_name)
        if type(val) is not tuple:
            raise ValueError(f"{field_name} must be exactly a tuple, got {type(val)}")

def validate_family_consistency(spec: ModelSpec) -> None:
    if is_a_family(spec):
        if spec.family_id == FamilyId.AR:
            if spec.p <= 0 or spec.q != 0:
                raise ValueError(f"AR family requires p > 0 and q == 0, got p={spec.p}, q={spec.q}")
            if spec.mean_family != MeanFamily.AR:
                raise ValueError(f"AR family requires mean_family to be AR, got {spec.mean_family}")
        elif spec.family_id == FamilyId.ARMA:
            if spec.p < 0 or spec.q < 0 or (spec.p + spec.q == 0):
                raise ValueError(f"ARMA family requires p>=0, q>=0 and p+q > 0, got p={spec.p}, q={spec.q}")
            if spec.mean_family != MeanFamily.ARMA:
                raise ValueError(f"ARMA family requires mean_family to be ARMA, got {spec.mean_family}")
        
        if spec.volatility_family != VolatilityFamily.NONE:
            raise ValueError(f"A-family requires volatility_family to be NONE, got {spec.volatility_family}")
        if spec.r != 0 or spec.s != 0:
            raise ValueError(f"A-family requires r==0 and s==0, got r={spec.r}, s={spec.s}")
        if spec.omega is not None:
            raise ValueError(f"A-family requires omega to be None, got {spec.omega}")
        if len(spec.alpha_params) != 0 or len(spec.beta_params) != 0:
            raise ValueError("A-family requires empty alpha/beta parameters")
        
    elif is_b_family(spec):
        if spec.mean_family != MeanFamily.NONE:
            raise ValueError(f"B-family requires mean_family to be NONE, got {spec.mean_family}")
        if spec.volatility_family != VolatilityFamily.GARCH:
            raise ValueError(f"B-family requires volatility_family to be GARCH, got {spec.volatility_family}")
        if spec.p != 0 or spec.q != 0:
            raise ValueError(f"B-family requires p==0 and q==0, got p={spec.p}, q={spec.q}")
        if spec.r <= 0 or spec.s <= 0:
            raise ValueError(f"B-family requires r > 0 and s > 0, got r={spec.r}, s={spec.s}")
        if spec.omega is None:
            raise ValueError("B-family requires non-None omega")
        if len(spec.ar_params) != 0 or len(spec.ma_params) != 0:
            raise ValueError("B-family requires empty ar/ma parameters")
            
    elif is_c_family(spec):
        if spec.mean_family != MeanFamily.ARMA:
            raise ValueError(f"C-family requires mean_family to be ARMA, got {spec.mean_family}")
        if spec.volatility_family != VolatilityFamily.GARCH:
            raise ValueError(f"C-family requires volatility_family to be GARCH, got {spec.volatility_family}")
        if spec.p < 0 or spec.q < 0 or (spec.p + spec.q == 0):
            raise ValueError(f"C-family requires p>=0, q>=0 and p+q > 0, got p={spec.p}, q={spec.q}")
        if spec.r <= 0 or spec.s <= 0:
            raise ValueError(f"C-family requires r > 0 and s > 0, got r={spec.r}, s={spec.s}")
        if spec.omega is None:
            raise ValueError("C-family requires non-None omega")
            
    else:
        raise ValueError(f"Unknown or unsupported family_id: {spec.family_id}")

def validate_parameter_lengths(spec: ModelSpec) -> None:
    if len(spec.ar_params) != spec.p:
        raise ValueError(f"ar_params length must equal p ({spec.p}), got {len(spec.ar_params)}")
    if len(spec.ma_params) != spec.q:
        raise ValueError(f"ma_params length must equal q ({spec.q}), got {len(spec.ma_params)}")
    if len(spec.alpha_params) != spec.r:
        raise ValueError(f"alpha_params length must equal r ({spec.r}), got {len(spec.alpha_params)}")
    if len(spec.beta_params) != spec.s:
        raise ValueError(f"beta_params length must equal s ({spec.s}), got {len(spec.beta_params)}")
    if len(spec.constraint_flags) != 4:
        raise ValueError(f"constraint_flags length must be exactly 4, got {len(spec.constraint_flags)}")

def schema_v2_index_map() -> dict:
    return {
        "family_id": {0},
        "mean_family": {1},
        "volatility_family": {2},
        "p": {3},
        "q": {4},
        "r": {5},
        "s": {6},
        "omega": {7},
        "constraint_flags": set(range(8, 12)),
        "ar_params": set(range(12, 17)),
        "ma_params": set(range(17, 22)),
        "alpha_params": set(range(22, 24)),
        "beta_params": set(range(24, 26)),
        "reserved": set(range(26, 32))
    }

def validate_no_legacy_collision(spec: ModelSpec) -> None:
    idx_map = schema_v2_index_map()
    keys_to_check = ["ar_params", "ma_params", "omega", "alpha_params", "beta_params"]
    omega_set = idx_map["omega"]
    sets = {k: idx_map[k] for k in keys_to_check if k != "omega"}
    sets["omega"] = omega_set
    
    for k1, s1 in sets.items():
        for k2, s2 in sets.items():
            if k1 != k2:
                intersection = s1.intersection(s2)
                if intersection:
                    raise ValueError(f"Index collision detected between {k1} and {k2} at indices {intersection}")

def validate_model_spec(spec: ModelSpec) -> None:
    validate_enum_types(spec)
    validate_immutable_tuple_fields(spec)
    validate_order_bounds(spec)
    validate_family_consistency(spec)
    validate_parameter_lengths(spec)
    validate_no_legacy_collision(spec)

# Explicit enum numeric codes for boundary flat vectors
FAMILY_ID_MAP = {
    FamilyId.AR: 1.0,
    FamilyId.ARMA: 2.0,
    FamilyId.GARCH: 3.0,
    FamilyId.ARMA_GARCH: 4.0
}
FAMILY_ID_REV = {v: k for k, v in FAMILY_ID_MAP.items()}

MEAN_FAMILY_MAP = {
    MeanFamily.NONE: 0.0,
    MeanFamily.AR: 1.0,
    MeanFamily.ARMA: 2.0
}
MEAN_FAMILY_REV = {v: k for k, v in MEAN_FAMILY_MAP.items()}

VOLATILITY_FAMILY_MAP = {
    VolatilityFamily.NONE: 0.0,
    VolatilityFamily.GARCH: 1.0
}
VOLATILITY_FAMILY_REV = {v: k for k, v in VOLATILITY_FAMILY_MAP.items()}

def to_flat_boundary_vector(spec: ModelSpec) -> list[float]:
    validate_model_spec(spec)
    
    vector = [0.0] * SCHEMA_V2_MIN_FLAT_DIM
    
    vector[0] = FAMILY_ID_MAP[spec.family_id]
    vector[1] = MEAN_FAMILY_MAP[spec.mean_family]
    vector[2] = VOLATILITY_FAMILY_MAP[spec.volatility_family]
    vector[3] = float(spec.p)
    vector[4] = float(spec.q)
    vector[5] = float(spec.r)
    vector[6] = float(spec.s)
    vector[7] = spec.omega if spec.omega is not None else 0.0
    
    for i, val in enumerate(spec.constraint_flags):
        vector[8 + i] = val
        
    for i in range(5):
        if i < len(spec.ar_params):
            vector[12 + i] = spec.ar_params[i]
            
    for i in range(5):
        if i < len(spec.ma_params):
            vector[17 + i] = spec.ma_params[i]
            
    for i in range(2):
        if i < len(spec.alpha_params):
            vector[22 + i] = spec.alpha_params[i]
            
    for i in range(2):
        if i < len(spec.beta_params):
            vector[24 + i] = spec.beta_params[i]
            
    return vector

def _decode_non_negative_integer_code(val: float, name: str) -> int:
    try:
        f_val = float(val)
    except (TypeError, ValueError):
        raise ValueError(f"{name} must be convertible to float, got {type(val)}")
    if not f_val.is_integer() or f_val < 0:
        raise ValueError(f"{name} order must be a non-negative integer, got {f_val}")
    return int(f_val)

def from_flat_boundary_vector(vector: list[float]) -> ModelSpec:
    if len(vector) != SCHEMA_V2_MIN_FLAT_DIM:
        raise ValueError(f"Vector length must be exactly {SCHEMA_V2_MIN_FLAT_DIM}, got {len(vector)}")
        
    fid_code = vector[0]
    mf_code = vector[1]
    vf_code = vector[2]
    
    if fid_code not in FAMILY_ID_REV:
        raise ValueError(f"Invalid family_id code: {fid_code}. No threshold decoding allowed.")
    if mf_code not in MEAN_FAMILY_REV:
        raise ValueError(f"Invalid mean_family code: {mf_code}. No threshold decoding allowed.")
    if vf_code not in VOLATILITY_FAMILY_REV:
        raise ValueError(f"Invalid volatility_family code: {vf_code}. No threshold decoding allowed.")
        
    family_id = FAMILY_ID_REV[fid_code]
    mean_family = MEAN_FAMILY_REV[mf_code]
    volatility_family = VOLATILITY_FAMILY_REV[vf_code]
    
    p = _decode_non_negative_integer_code(vector[3], "p")
    q = _decode_non_negative_integer_code(vector[4], "q")
    r = _decode_non_negative_integer_code(vector[5], "r")
    s = _decode_non_negative_integer_code(vector[6], "s")
    
    omega_val = vector[7]
    if family_id in (FamilyId.AR, FamilyId.ARMA):
        if omega_val != 0.0:
            raise ValueError(f"A-family requires omega to be 0.0 in flat vector, got {omega_val}")
        omega = None
    else:
        omega = omega_val
        
    constraint_flags = (vector[8], vector[9], vector[10], vector[11])
    
    ar_params = tuple(vector[12:12 + p])
    for idx in range(12 + p, 17):
        if vector[idx] != 0.0:
            raise ValueError(f"Padding index {idx} in ar_params must be 0.0, got {vector[idx]}")
            
    ma_params = tuple(vector[17:17 + q])
    for idx in range(17 + q, 22):
        if vector[idx] != 0.0:
            raise ValueError(f"Padding index {idx} in ma_params must be 0.0, got {vector[idx]}")
            
    alpha_params = tuple(vector[22:22 + r])
    for idx in range(22 + r, 24):
        if vector[idx] != 0.0:
            raise ValueError(f"Padding index {idx} in alpha_params must be 0.0, got {vector[idx]}")
            
    beta_params = tuple(vector[24:24 + s])
    for idx in range(24 + s, 26):
        if vector[idx] != 0.0:
            raise ValueError(f"Padding index {idx} in beta_params must be 0.0, got {vector[idx]}")
            
    for idx in range(26, 32):
        if vector[idx] != 0.0:
            raise ValueError(f"Reserved index {idx} must be 0.0, got {vector[idx]}")
            
    # Provenance is empty tuple by design when restoring from a flat vector representation
    provenance = ()
    
    spec = ModelSpec(
        family_id=family_id,
        mean_family=mean_family,
        volatility_family=volatility_family,
        p=p,
        q=q,
        r=r,
        s=s,
        ar_params=ar_params,
        ma_params=ma_params,
        omega=omega,
        alpha_params=alpha_params,
        beta_params=beta_params,
        constraint_flags=constraint_flags,
        provenance=provenance
    )
    
    validate_model_spec(spec)
    return spec
