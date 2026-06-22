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
    run_result = run_phase2_artifact_generation(run_request)

    # 1. Validate total split count
    if run_result.total_split_count != 6:
        raise ValueError(f"Expected 6 splits, got {run_result.total_split_count}")

    # 2. Validate split sequence order and names
    expected_order = [
        (SplitName.SMOKE, "smoke", 4000),
        (SplitName.ZERO_SHOT_TRAIN, "zero_shot_train", 3000),
        (SplitName.ZERO_SHOT_EVAL, "zero_shot_eval", 1000),
        (SplitName.FEWSHOT_TRAIN, "fewshot_1pct_train", 10),
        (SplitName.FEWSHOT_TRAIN, "fewshot_5pct_train", 50),
        (SplitName.FEWSHOT_EVAL, "fewshot_eval", 1000),
    ]

    for i, (expected_name, expected_subdir, expected_count) in enumerate(expected_order):
        split_res = run_result.split_results[i]
        if split_res.split_name != expected_name:
            raise ValueError(f"Split {i} name mismatch: expected {expected_name}, got {split_res.split_name}")
        if split_res.artifact_subdir != expected_subdir:
            raise ValueError(f"Split {i} subdir mismatch: expected {expected_subdir}, got {split_res.artifact_subdir}")
        if split_res.dataset_total_count != expected_count:
            raise ValueError(f"Split {i} count mismatch: expected {expected_count}, got {split_res.dataset_total_count}")

    # 3. Validate total sample count
    if run_result.total_sample_count != 9060:
        raise ValueError(f"Expected 9060 total samples, got {run_result.total_sample_count}")

    # 4. zero_shot_train validations
    zst_res = run_result.split_results[1]
    if zst_res.zero_shot_c_train_count != 0:
        raise ValueError(f"zero_shot_train has zero_shot_c_train_count {zst_res.zero_shot_c_train_count} != 0")
    zst_counts = dict(zst_res.count_by_family)
    if FamilyId.ARMA_GARCH in zst_counts and zst_counts[FamilyId.ARMA_GARCH] > 0:
        raise ValueError(f"zero_shot_train contains ARMA_GARCH samples: {zst_counts}")

    # 5. zero_shot_eval validations (C-only)
    zse_res = run_result.split_results[2]
    zse_counts = dict(zse_res.count_by_family)
    other_than_c = sum(v for k, v in zse_counts.items() if k != FamilyId.ARMA_GARCH)
    if other_than_c > 0:
        raise ValueError("zero_shot_eval contains non-C family samples")

    # 6. fewshot_1pct_train validations (C-only)
    fs1_res = run_result.split_results[3]
    fs1_counts = dict(fs1_res.count_by_family)
    if sum(v for k, v in fs1_counts.items() if k != FamilyId.ARMA_GARCH) > 0:
        raise ValueError("fewshot_1pct_train contains non-C family samples")

    # 7. fewshot_5pct_train validations (C-only)
    fs5_res = run_result.split_results[4]
    fs5_counts = dict(fs5_res.count_by_family)
    if sum(v for k, v in fs5_counts.items() if k != FamilyId.ARMA_GARCH) > 0:
        raise ValueError("fewshot_5pct_train contains non-C family samples")

    # 8. fewshot_eval validations (C-only)
    fse_res = run_result.split_results[5]
    fse_counts = dict(fse_res.count_by_family)
    if sum(v for k, v in fse_counts.items() if k != FamilyId.ARMA_GARCH) > 0:
        raise ValueError("fewshot_eval contains non-C family samples")

    # 9. Validate P13 seed prefix semantics
    validate_p13_fewshot_seed_plan_for_preset(ProtocolPresetName.SMOKE)
    r1 = seed_range_for_base_and_count(10012101, 10)
    r5 = seed_range_for_base_and_count(10012101, 50)
    if not range_is_prefix_subset(r1, r5):
        raise ValueError("Fewshot 1% seed range is not a prefix subset of Fewshot 5% seed range")

    # 10. File integrity and existence checks
    output_path = pathlib.Path(output_root_dir)
    actual_files = []
    for root, dirs, files in os.walk(output_path):
        for f in files:
            p = pathlib.Path(root) / f
            actual_files.append(p.relative_to(output_path).as_posix())

    expected_files = []
    for _, subdir, _ in expected_order:
        expected_files.append(f"{subdir}/samples.jsonl")
        expected_files.append(f"{subdir}/manifest.json")

    # Check for unexpected files
    unexpected = [f for f in actual_files if f not in expected_files]
    if unexpected:
        raise ValueError(f"Unexpected files found in output directory: {unexpected}")

    # Check each file exists, is JSON parseable, and matches hashes
    per_split_summary = []
    for split_res in run_result.split_results:
        samples_file = pathlib.Path(split_res.samples_path)
        manifest_file = pathlib.Path(split_res.manifest_path)

        if not samples_file.exists() or not samples_file.is_file():
            raise ValueError(f"Samples file does not exist: {samples_file}")
        if not manifest_file.exists() or not manifest_file.is_file():
            raise ValueError(f"Manifest file does not exist: {manifest_file}")

        # Compute actual hashes and sizes
        actual_samples_sha = sha256_file(str(samples_file))
        actual_manifest_sha = sha256_file(str(manifest_file))
        samples_size = samples_file.stat().st_size
        manifest_size = manifest_file.stat().st_size

        if actual_samples_sha != split_res.samples_sha256:
            raise ValueError(f"Samples hash mismatch for {split_res.split_name}: actual={actual_samples_sha}, runner={split_res.samples_sha256}")
        if actual_manifest_sha != split_res.manifest_sha256:
            raise ValueError(f"Manifest hash mismatch for {split_res.split_name}: actual={actual_manifest_sha}, runner={split_res.manifest_sha256}")

        # Parse manifest
        with open(manifest_file, "r", encoding="utf-8") as f:
            manifest_data = json.load(f)

        if manifest_data.get("artifact_type") != "phase2_dataset_manifest":
            raise ValueError(f"Invalid artifact_type in manifest for split {split_res.split_name}")
        if manifest_data.get("total_count") != split_res.dataset_total_count:
            raise ValueError(f"Total count mismatch in manifest for split {split_res.split_name}")
        if manifest_data.get("samples_sha256") != actual_samples_sha:
            raise ValueError(f"Samples SHA256 mismatch in manifest for split {split_res.split_name}")

        per_split_summary.append({
            "split_name": split_res.split_name.value,
            "artifact_subdir": split_res.artifact_subdir,
            "sample_count": split_res.dataset_total_count,
            "samples_path": split_res.samples_path.replace("\\", "/"),
            "manifest_path": split_res.manifest_path.replace("\\", "/"),
            "samples_sha256": split_res.samples_sha256,
            "manifest_sha256": split_res.manifest_sha256,
            "samples_size_bytes": samples_size,
            "manifest_size_bytes": manifest_size,
        })

    return {
        "verdict": "PASS",
        "output_root_dir": output_root_dir.replace("\\", "/"),
        "total_split_count": run_result.total_split_count,
        "total_sample_count": run_result.total_sample_count,
        "per_split": per_split_summary,
        "zero_shot_c_train_count": zst_res.zero_shot_c_train_count,
        "fewshot_seed_plan_verified": True,
        "unexpected_files": unexpected,
        "fewshot_superset_note": "sample_id prefix not required because split/artifact identity differs; seed prefix semantics verified instead.",
        "reason": "phase2_p14_smoke_dry_run_validation_complete",
    }


def main() -> int:
    if len(sys.argv) > 1:
        default_output = sys.argv[1]
    else:
        default_output = "phase2_artifacts/p14_smoke_dry_run"
    try:
        summary = run_p14_smoke_dry_run(default_output)
        # Print compact JSON to stdout only on success
        print(json.dumps(summary, separators=(",", ":")))
        return 0
    except Exception as e:
        print(f"FAIL: Dry run validation failed. Error: {e}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
