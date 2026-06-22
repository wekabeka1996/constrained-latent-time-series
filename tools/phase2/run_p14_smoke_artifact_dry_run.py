# tools/phase2/run_p14_smoke_artifact_dry_run.py

import json
import os
import pathlib
import sys

from src.phase2.schema import FamilyId, ModelSpec, MeanFamily, VolatilityFamily
from src.phase2.sampler import GenerationRequest
from src.phase2.simulator import SimulationRequest
from src.phase2.dataset import SplitName
from src.phase2.split_runner import run_phase2_artifact_generation
from src.phase2.artifacts import sha256_file
from src.phase2.protocol_presets import (
    ProtocolPresetName,
    ProtocolPresetFactoryRequest,
    build_phase2_preset_factory_result,
    validate_p13_fewshot_seed_plan_for_preset,
    seed_range_for_base_and_count,
    range_is_prefix_subset,
)


def build_p14_generation_templates() -> tuple[tuple[FamilyId, GenerationRequest], ...]:
    ar = GenerationRequest(
        family_id=FamilyId.AR,
        seed=0,
        p=1, q=0, r=0, s=0,
        ar_range=(-0.4, 0.4),
        ma_range=(-0.2, 0.2),
        omega_range=(0.1, 1.0),
        alpha_range=(0.05, 0.2),
        beta_range=(0.05, 0.4),
        constraint_flags=(1.0, 0.0, 0.0, 0.0),
        provenance=(("phase", "p14"), ("template", "AR")),
        max_attempts=100,
    )
    arma = GenerationRequest(
        family_id=FamilyId.ARMA,
        seed=0,
        p=1, q=1, r=0, s=0,
        ar_range=(-0.35, 0.35),
        ma_range=(-0.35, 0.35),
        omega_range=(0.1, 1.0),
        alpha_range=(0.05, 0.2),
        beta_range=(0.05, 0.4),
        constraint_flags=(1.0, 0.0, 0.0, 0.0),
        provenance=(("phase", "p14"), ("template", "ARMA")),
        max_attempts=100,
    )
    garch = GenerationRequest(
        family_id=FamilyId.GARCH,
        seed=0,
        p=0, q=0, r=1, s=1,
        ar_range=(-0.2, 0.2),
        ma_range=(-0.2, 0.2),
        omega_range=(0.1, 0.5),
        alpha_range=(0.05, 0.2),
        beta_range=(0.1, 0.6),
        constraint_flags=(1.0, 0.0, 0.0, 0.0),
        provenance=(("phase", "p14"), ("template", "GARCH")),
        max_attempts=100,
    )
    arma_garch = GenerationRequest(
        family_id=FamilyId.ARMA_GARCH,
        seed=0,
        p=1, q=1, r=1, s=1,
        ar_range=(-0.3, 0.3),
        ma_range=(-0.3, 0.3),
        omega_range=(0.1, 0.5),
        alpha_range=(0.05, 0.2),
        beta_range=(0.1, 0.6),
        constraint_flags=(1.0, 0.0, 0.0, 0.0),
        provenance=(("phase", "p14"), ("template", "ARMA_GARCH")),
        max_attempts=100,
    )
    return (
        (FamilyId.AR, ar),
        (FamilyId.ARMA, arma),
        (FamilyId.GARCH, garch),
        (FamilyId.ARMA_GARCH, arma_garch),
    )


def build_p14_simulation_template() -> SimulationRequest:
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
        length=64,
        burn_in=16,
        innovation_distribution="gaussian",
        innovation_std=1.0,
        initial_value=0.0,
        initial_innovation=0.0,
        initial_variance=1.0,
        max_abs_value=1000.0,
    )


def build_p14_factory_request(output_root_dir: str) -> ProtocolPresetFactoryRequest:
    return ProtocolPresetFactoryRequest(
        protocol_name="phase2_p14_smoke_dry_run",
        preset_name=ProtocolPresetName.SMOKE,
        output_root_dir=output_root_dir,
        generation_template_by_family=build_p14_generation_templates(),
        simulation_template=build_p14_simulation_template(),
        create_parent_dirs=True,
        overwrite_existing=True,
        include_values=True,
        include_innovations=True,
        include_variances=True,
        json_sort_keys=True,
        json_indent=2,
        sample_id_hash_len=12,
    )


def run_p14_smoke_dry_run(output_root_dir: str) -> dict:
    request = build_p14_factory_request(output_root_dir)
    factory_result = build_phase2_preset_factory_result(request)
    run_request = factory_result.run_request

    # Run execution
    run_phase2_artifact_generation(run_request)

    # Use audit utility
    from tools.phase2.audit_phase2_artifact_manifest import audit_phase2_artifacts
    return audit_phase2_artifacts(output_root_dir)


def main() -> int:
    if len(sys.argv) > 1:
        default_output = sys.argv[1]
    else:
        default_output = "phase2_artifacts/p14_smoke_dry_run"
    try:
        summary = run_p14_smoke_dry_run(default_output)
        
        from tools.phase2.audit_phase2_artifact_manifest import (
            compact_json,
            assert_no_absolute_paths_in_data
        )
        
        assert_no_absolute_paths_in_data(summary)
        # Print compact JSON to stdout only on success
        print(compact_json(summary))
        return 0
    except Exception as e:
        print(f"FAIL: Dry run validation failed. Error: {e}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())

