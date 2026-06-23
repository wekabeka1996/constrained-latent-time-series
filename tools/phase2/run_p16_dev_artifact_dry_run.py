# tools/phase2/run_p16_dev_artifact_dry_run.py

import json
import os
import pathlib
import sys

from src.phase2.schema import FamilyId
from src.phase2.split_runner import run_phase2_artifact_generation
from src.phase2.protocol_presets import (
    ProtocolPresetName,
    ProtocolPresetFactoryRequest,
    build_phase2_preset_factory_result,
    validate_p13_fewshot_seed_plan_for_preset,
)
from tools.phase2.run_p14_smoke_artifact_dry_run import (
    build_p14_generation_templates,
    build_p14_simulation_template,
)
from tools.phase2.audit_phase2_artifact_manifest import (
    audit_phase2_artifact_root,
    repository_root,
    compact_json,
    assert_no_absolute_paths_in_data,
)


def expected_p16_dev_subdirs() -> tuple[str, ...]:
    return (
        "smoke",
        "zero_shot_train",
        "zero_shot_eval",
        "fewshot_1pct_train",
        "fewshot_5pct_train",
        "fewshot_eval",
    )


def expected_p16_dev_counts_by_subdir() -> tuple[tuple[str, int], ...]:
    return (
        ("smoke", 40000),
        ("zero_shot_train", 30000),
        ("zero_shot_eval", 10000),
        ("fewshot_1pct_train", 100),
        ("fewshot_5pct_train", 500),
        ("fewshot_eval", 10000),
    )


def build_p16_factory_request(output_root_dir: str) -> ProtocolPresetFactoryRequest:
    return ProtocolPresetFactoryRequest(
        protocol_name="phase2_p16_dev_dry_run",
        preset_name=ProtocolPresetName.DEV,
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


def run_p16_dev_dry_run(output_root_dir: str) -> dict:
    request = build_p16_factory_request(output_root_dir)
    factory_result = build_phase2_preset_factory_result(request)
    run_request = factory_result.run_request

    # Run execution
    run_phase2_artifact_generation(run_request)

    # Use audit utility
    subdirs = list(expected_p16_dev_subdirs())
    counts = dict(expected_p16_dev_counts_by_subdir())
    
    # Determine repo root fallback if outside repository
    root = repository_root()
    output_path = pathlib.Path(output_root_dir).resolve()
    try:
        output_path.relative_to(root)
    except ValueError:
        # Fallback if outside repository (e.g., in tmp_path tests)
        root = output_path.parent

    summary = audit_phase2_artifact_root(
        output_root_dir=output_root_dir,
        expected_subdirs=subdirs,
        expected_counts_by_subdir=counts,
        repo_root=root,
    )

    # Perform P13 DEV seed plan verification
    validate_p13_fewshot_seed_plan_for_preset(ProtocolPresetName.DEV)
    summary["fewshot_seed_plan_verified"] = True
    summary["c_eval_holdout_non_overlap_verified"] = True

    # Assert no absolute paths
    assert_no_absolute_paths_in_data(summary)

    return summary


def main() -> int:
    if len(sys.argv) > 1:
        default_output = sys.argv[1]
    else:
        default_output = "phase2_artifacts/p16_dev_dry_run"
    try:
        summary = run_p16_dev_dry_run(default_output)
        print(compact_json(summary))
        return 0
    except Exception as e:
        print(f"FAIL: DEV dry run validation failed. Error: {e}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
