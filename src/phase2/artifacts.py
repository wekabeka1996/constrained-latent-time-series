# src/phase2/artifacts.py

import json
import hashlib
import pathlib
from dataclasses import dataclass
from typing import Any, Tuple

from src.phase2.schema import FamilyId
from src.phase2.dataset import SplitName, DatasetSample, DatasetBuildResult


@dataclass(frozen=True)
class ArtifactWriteRequest:
    dataset_result: DatasetBuildResult
    output_dir: str
    samples_filename: str
    manifest_filename: str
    create_parent_dirs: bool
    overwrite_existing: bool
    include_values: bool
    include_innovations: bool
    include_variances: bool
    json_sort_keys: bool
    json_indent: int


@dataclass(frozen=True)
class ArtifactWriteResult:
    output_dir: str
    samples_path: str
    manifest_path: str
    samples_sha256: str
    manifest_sha256: str
    sample_count: int
    protocol_name: str
    split_name: SplitName
    count_by_family: Tuple[Tuple[FamilyId, int], ...]
    reason: str


def validate_artifact_write_request(request: ArtifactWriteRequest) -> None:
    # 1. dataset_result is DatasetBuildResult
    if not isinstance(request.dataset_result, DatasetBuildResult):
        raise ValueError("dataset_result must be a DatasetBuildResult instance")

    # 2. output_dir is str and non-empty
    if not isinstance(request.output_dir, str):
        raise ValueError("output_dir must be a string")
    if len(request.output_dir.strip()) == 0:
        raise ValueError("output_dir must be non-empty")

    # 3. samples_filename is str and non-empty
    if not isinstance(request.samples_filename, str):
        raise ValueError("samples_filename must be a string")
    if len(request.samples_filename.strip()) == 0:
        raise ValueError("samples_filename must be non-empty")

    # 4. manifest_filename is str and non-empty
    if not isinstance(request.manifest_filename, str):
        raise ValueError("manifest_filename must be a string")
    if len(request.manifest_filename.strip()) == 0:
        raise ValueError("manifest_filename must be non-empty")

    # 5. samples_filename != manifest_filename
    if request.samples_filename == request.manifest_filename:
        raise ValueError("samples_filename and manifest_filename must be distinct")

    # 6. samples_filename endswith ".jsonl"
    if not request.samples_filename.endswith(".jsonl"):
        raise ValueError("samples_filename must end with '.jsonl'")

    # 7. manifest_filename endswith ".json"
    if not request.manifest_filename.endswith(".json"):
        raise ValueError("manifest_filename must end with '.json'")

    # 8. create_parent_dirs is bool
    if not isinstance(request.create_parent_dirs, bool):
        raise ValueError("create_parent_dirs must be a bool")

    # 9. overwrite_existing is bool
    if not isinstance(request.overwrite_existing, bool):
        raise ValueError("overwrite_existing must be a bool")

    # 10. include_values is bool
    if not isinstance(request.include_values, bool):
        raise ValueError("include_values must be a bool")

    # 11. include_innovations is bool
    if not isinstance(request.include_innovations, bool):
        raise ValueError("include_innovations must be a bool")

    # 12. include_variances is bool
    if not isinstance(request.include_variances, bool):
        raise ValueError("include_variances must be a bool")

    # 13. json_sort_keys is bool
    if not isinstance(request.json_sort_keys, bool):
        raise ValueError("json_sort_keys must be a bool")

    # 14. json_indent is int, not bool, and >= 0
    if type(request.json_indent) is not int or isinstance(request.json_indent, bool):
        raise ValueError("json_indent must be an int, not bool or other type")
    if request.json_indent < 0:
        raise ValueError("json_indent must be >= 0")

    # 15. dataset_result.samples is tuple
    if type(request.dataset_result.samples) is not tuple:
        raise ValueError("dataset_result.samples must be a tuple")

    # 16. dataset_result.total_count == len(dataset_result.samples)
    if request.dataset_result.total_count != len(request.dataset_result.samples):
        raise ValueError("dataset_result.total_count mismatch with samples tuple length")

    # 17. dataset_result.reason == "dataset_built_in_memory"
    if request.dataset_result.reason != "dataset_built_in_memory":
        raise ValueError("dataset_result.reason must be 'dataset_built_in_memory'")

    # 18. every sample is DatasetSample
    for sample in request.dataset_result.samples:
        if not isinstance(sample, DatasetSample):
            raise ValueError("Every sample in dataset_result must be a DatasetSample instance")
        
        # 19. every sample belongs to dataset_result.protocol_name and split_name
        if sample.protocol_name != request.dataset_result.protocol_name:
            raise ValueError(f"Sample protocol_name '{sample.protocol_name}' does not match dataset protocol_name '{request.dataset_result.protocol_name}'")
        if sample.split_name != request.dataset_result.split_name:
            raise ValueError(f"Sample split_name '{sample.split_name}' does not match dataset split_name '{request.dataset_result.split_name}'")

        # 20. every sample values/innovations/variances are tuples
        if type(sample.values) is not tuple:
            raise ValueError("sample.values must be a tuple")
        if type(sample.innovations) is not tuple:
            raise ValueError("sample.innovations must be a tuple")
        if type(sample.variances) is not tuple:
            raise ValueError("sample.variances must be a tuple")

    # 21. At least one of include_values/include_innovations/include_variances must be True
    if not (request.include_values or request.include_innovations or request.include_variances):
        raise ValueError("At least one of include_values, include_innovations, or include_variances must be True")

    # 22. Path checks
    output_path = pathlib.Path(request.output_dir).resolve()
    if output_path.exists() and output_path.is_file():
        raise ValueError(f"output_dir {request.output_dir} exists and is a file")

    # Path safety check to prevent writing outside output_dir
    samples_path = (output_path / request.samples_filename).resolve()
    manifest_path = (output_path / request.manifest_filename).resolve()
    if output_path not in samples_path.parents:
        raise ValueError(f"samples_filename '{request.samples_filename}' must stay within output_dir")
    if output_path not in manifest_path.parents:
        raise ValueError(f"manifest_filename '{request.manifest_filename}' must stay within output_dir")


def dataset_sample_to_json_dict(
    sample: DatasetSample,
    include_values: bool,
    include_innovations: bool,
    include_variances: bool,
) -> dict:
    spec = sample.model_spec
    model_spec_dict = {
        "family_id": spec.family_id.value,
        "mean_family": spec.mean_family.value,
        "volatility_family": spec.volatility_family.value,
        "p": spec.p,
        "q": spec.q,
        "r": spec.r,
        "s": spec.s,
        "ar_params": list(spec.ar_params),
        "ma_params": list(spec.ma_params),
        "omega": spec.omega,
        "alpha_params": list(spec.alpha_params),
        "beta_params": list(spec.beta_params),
        "constraint_flags": list(spec.constraint_flags),
        "provenance": [list(item) for item in spec.provenance],
    }

    sample_dict = {
        "sample_id": sample.sample_id,
        "protocol_name": sample.protocol_name,
        "split_name": sample.split_name.value,
        "family_id": sample.family_id.value,
        "sample_index": sample.sample_index,
        "generation_seed": sample.generation_seed,
        "simulation_seed": sample.simulation_seed,
        "model_spec": model_spec_dict,
        "provenance": [list(item) for item in sample.provenance],
    }

    if include_values:
        sample_dict["values"] = list(sample.values)
    if include_innovations:
        sample_dict["innovations"] = list(sample.innovations)
    if include_variances:
        sample_dict["variances"] = list(sample.variances)

    return sample_dict


def dataset_manifest_to_json_dict(
    request: ArtifactWriteRequest,
    samples_sha256: str,
) -> dict:
    count_by_family_list = []
    for f_id, count in request.dataset_result.count_by_family:
        count_by_family_list.append({
            "family_id": f_id.value,
            "count": count
        })

    manifest = {
        "artifact_type": "phase2_dataset_manifest",
        "protocol_name": request.dataset_result.protocol_name,
        "split_name": request.dataset_result.split_name.value,
        "total_count": request.dataset_result.total_count,
        "count_by_family": count_by_family_list,
        "zero_shot_c_train_count": request.dataset_result.zero_shot_c_train_count,
        "samples_filename": request.samples_filename,
        "samples_sha256": samples_sha256,
        "sample_ids": [s.sample_id for s in request.dataset_result.samples],
        "include_values": request.include_values,
        "include_innovations": request.include_innovations,
        "include_variances": request.include_variances,
        "json_sort_keys": request.json_sort_keys,
        "json_indent": request.json_indent,
        "reason": "artifact_manifest_created",
    }
    return manifest


def sha256_file(path: str) -> str:
    if not isinstance(path, str):
        raise ValueError("path must be a string")
    p = pathlib.Path(path)
    if not p.is_file():
        raise FileNotFoundError(f"File {path} does not exist or is not a file")
    
    h = hashlib.sha256()
    with open(p, "rb") as f:
        while chunk := f.read(8192):
            h.update(chunk)
    return h.hexdigest().lower()


def write_dataset_artifacts(request: ArtifactWriteRequest) -> ArtifactWriteResult:
    # 1. Validate the request and check initial path correctness
    validate_artifact_write_request(request)

    output_path = pathlib.Path(request.output_dir).resolve()
    samples_path = (output_path / request.samples_filename).resolve()
    manifest_path = (output_path / request.manifest_filename).resolve()

    # 2. Check if output_dir missing and handle according to create_parent_dirs
    if not output_path.exists():
        if request.create_parent_dirs:
            output_path.mkdir(parents=True, exist_ok=True)
        else:
            raise FileNotFoundError(
                f"Output directory {request.output_dir} does not exist and create_parent_dirs is False"
            )

    # 3. Check if target files exist and overwrite_existing is False
    if not request.overwrite_existing:
        if samples_path.exists():
            raise FileExistsError(
                f"Samples file {samples_path} already exists and overwrite_existing is False"
            )
        if manifest_path.exists():
            raise FileExistsError(
                f"Manifest file {manifest_path} already exists and overwrite_existing is False"
            )

    # 4. Serialize samples JSONL
    # Each sample line is a single line, even if json_indent > 0
    lines = []
    for sample in request.dataset_result.samples:
        s_dict = dataset_sample_to_json_dict(
            sample,
            include_values=request.include_values,
            include_innovations=request.include_innovations,
            include_variances=request.include_variances,
        )
        line = json.dumps(s_dict, sort_keys=request.json_sort_keys, separators=(",", ":"))
        lines.append(line)
    
    # We append a trailing newline for JSONL consistency
    samples_text = "\n".join(lines) + "\n"
    samples_bytes = samples_text.encode("utf-8")
    samples_sha256 = hashlib.sha256(samples_bytes).hexdigest()

    # Write samples first
    samples_path.write_bytes(samples_bytes)

    # 5. Serialize manifest JSON
    # Manifest respects json_indent
    manifest_dict = dataset_manifest_to_json_dict(request, samples_sha256)
    if request.json_indent > 0:
        manifest_text = json.dumps(
            manifest_dict,
            sort_keys=request.json_sort_keys,
            indent=request.json_indent,
        )
    else:
        manifest_text = json.dumps(
            manifest_dict,
            sort_keys=request.json_sort_keys,
            separators=(",", ":"),
        )
    
    manifest_bytes = manifest_text.encode("utf-8")
    manifest_sha256 = hashlib.sha256(manifest_bytes).hexdigest()

    # Write manifest second
    manifest_path.write_bytes(manifest_bytes)

    return ArtifactWriteResult(
        output_dir=str(output_path),
        samples_path=str(samples_path),
        manifest_path=str(manifest_path),
        samples_sha256=samples_sha256,
        manifest_sha256=manifest_sha256,
        sample_count=request.dataset_result.total_count,
        protocol_name=request.dataset_result.protocol_name,
        split_name=request.dataset_result.split_name,
        count_by_family=request.dataset_result.count_by_family,
        reason="artifact_manifest_created",
    )
