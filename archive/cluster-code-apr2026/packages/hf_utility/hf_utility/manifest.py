"""
RACA-PROJECT-MANIFEST management for tracking all HuggingFace artifacts.

The manifest is stored as a dataset at {org}/RACA-PROJECT-MANIFEST and contains
metadata about all uploaded datasets in the organization.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from typing import Any, Optional

from datasets import Dataset, Features, Value, load_dataset, concatenate_datasets
from huggingface_hub import HfApi

from hf_utility.config import get_org, get_manifest_repo
from hf_utility.retry import retry_on_timeout, _is_conflict_error, CONFLICT_BASE_WAIT, API_TIMEOUT_WAIT

import random
import time

# Artifact-visibility fields — all optional (None default).
ARTIFACT_FIELDS: tuple[str, ...] = (
    "experiment_id",
    "run_id",
    "artifact_type",
    "visualizer_type",
    "artifact_group",
    "parent_artifact",
    "size_bytes",
)

_STANDARD_KEYS: frozenset[str] = frozenset({
    "script_name",
    "model",
    "hyperparameters",
    "input_datasets",
    "description",
    *ARTIFACT_FIELDS,
})


def _get_current_timestamp() -> str:
    return datetime.now(timezone.utc).isoformat()


def get_manifest() -> Optional[Dataset]:
    """Load the RACA-PROJECT-MANIFEST dataset from HuggingFace.

    Returns Dataset if exists, None if not found or empty.
    """
    try:
        manifest = retry_on_timeout(load_dataset, get_manifest_repo(), split="train")

        # Backfill any missing columns from older manifests
        for col in ARTIFACT_FIELDS:
            if col not in manifest.column_names:
                default = -1 if col == "size_bytes" else None
                manifest = manifest.add_column(col, [default] * len(manifest))

        return manifest
    except Exception as e:
        error_str = str(e).lower()
        if any(p in error_str for p in ["doesn't exist", "404", "no data", "corresponds to no data"]):
            return None
        raise


def _create_empty_manifest() -> Dataset:
    features = Features({
        "dataset_name": Value("string"),
        "script_name": Value("string"),
        "model": Value("string"),
        "hyperparameters": Value("string"),
        "input_datasets": Value("string"),
        "description": Value("string"),
        "tags": Value("string"),
        "custom_metadata": Value("string"),
        "created": Value("string"),
        "updated": Value("string"),
        "experiment_id": Value("string"),
        "run_id": Value("string"),
        "artifact_type": Value("string"),
        "visualizer_type": Value("string"),
        "artifact_group": Value("string"),
        "parent_artifact": Value("string"),
        "size_bytes": Value("int64"),
    })
    return Dataset.from_dict({col: [] for col in features}, features=features)


def _manifest_to_dict(manifest: Dataset) -> dict:
    result = {}
    for i in range(len(manifest)):
        row = manifest[i]
        result[row["dataset_name"]] = {"index": i, "data": row}
    return result


def _build_manifest_update(dataset_name: str, metadata: dict, tags: list[str]) -> Dataset:
    manifest = get_manifest()
    current_time = _get_current_timestamp()

    entry: dict[str, Any] = {
        "dataset_name": dataset_name,
        "script_name": metadata.get("script_name", ""),
        "model": metadata.get("model", ""),
        "hyperparameters": json.dumps(metadata.get("hyperparameters", {})),
        "input_datasets": json.dumps(metadata.get("input_datasets", [])),
        "description": metadata.get("description", ""),
        "tags": json.dumps(tags),
        "custom_metadata": json.dumps({
            k: v for k, v in metadata.items() if k not in _STANDARD_KEYS
        }),
        "updated": current_time,
        "experiment_id": metadata.get("experiment_id"),
        "run_id": metadata.get("run_id"),
        "artifact_type": metadata.get("artifact_type"),
        "visualizer_type": metadata.get("visualizer_type"),
        "artifact_group": metadata.get("artifact_group"),
        "parent_artifact": metadata.get("parent_artifact"),
        "size_bytes": metadata.get("size_bytes", -1),
    }

    if manifest is None:
        entry["created"] = current_time
        return Dataset.from_dict({k: [v] for k, v in entry.items()})

    manifest_dict = _manifest_to_dict(manifest)

    if dataset_name in manifest_dict:
        existing = manifest_dict[dataset_name]["data"]
        entry["created"] = existing["created"]

        # Preserve existing artifact fields when caller doesn't pass them
        for field in ARTIFACT_FIELDS:
            if entry.get(field) is None and existing.get(field) is not None:
                entry[field] = existing[field]

        manifest_data = manifest.to_dict()
        idx = manifest_dict[dataset_name]["index"]
        for key, value in entry.items():
            manifest_data[key][idx] = value
        return Dataset.from_dict(manifest_data)
    else:
        entry["created"] = current_time
        new_entry = Dataset.from_dict({k: [v] for k, v in entry.items()})
        return concatenate_datasets([manifest, new_entry])


def update_manifest_entry(
    dataset_name: str,
    metadata: dict,
    tags: list[str],
    max_retries: int = 5,
) -> None:
    """Update or create an entry in the RACA-PROJECT-MANIFEST.

    The entire read-modify-push cycle is retried on 409/412 conflicts.
    """
    manifest_repo = get_manifest_repo()

    for attempt in range(max_retries):
        try:
            new_manifest = _build_manifest_update(dataset_name, metadata, tags)
            new_manifest.push_to_hub(manifest_repo, private=False)
            _update_manifest_readme(new_manifest)
            print(f"Updated manifest entry for: {dataset_name}")
            return
        except Exception as e:
            error_str = str(e).lower()
            is_conflict = _is_conflict_error(error_str)
            is_retryable = is_conflict or "timeout" in error_str or "429" in error_str

            if is_retryable and attempt < max_retries - 1:
                if is_conflict:
                    wait = CONFLICT_BASE_WAIT * (attempt + 1) + random.uniform(0, 5)
                    print(f"Manifest conflict, retrying in {wait:.1f}s "
                          f"(attempt {attempt + 1}/{max_retries})...")
                else:
                    wait = API_TIMEOUT_WAIT
                    print(f"API timeout/rate limit, waiting {wait}s before retry...")
                time.sleep(wait)
            else:
                raise


def remove_manifest_entry(dataset_name: str) -> bool:
    """Remove an entry from the RACA-PROJECT-MANIFEST.

    Returns True if found and removed, False if not found.
    """
    manifest = get_manifest()
    if manifest is None:
        return False

    manifest_dict = _manifest_to_dict(manifest)
    if dataset_name not in manifest_dict:
        return False

    manifest_data = manifest.to_dict()
    idx = manifest_dict[dataset_name]["index"]

    new_data = {}
    for key, values in manifest_data.items():
        new_data[key] = [v for i, v in enumerate(values) if i != idx]

    if all(len(v) == 0 for v in new_data.values()):
        new_manifest = _create_empty_manifest()
    else:
        new_manifest = Dataset.from_dict(new_data)

    manifest_repo = get_manifest_repo()
    retry_on_timeout(new_manifest.push_to_hub, manifest_repo, private=False)
    _update_manifest_readme(new_manifest)

    print(f"Removed manifest entry for: {dataset_name}")
    return True


def _update_manifest_readme(manifest: Dataset) -> None:
    org = get_org()
    manifest_repo = get_manifest_repo()
    num_entries = len(manifest)

    readme_content = f"""---
license: mit
---

# RACA-PROJECT-MANIFEST

Central registry of all datasets in the `{org}` organization.

- **Total Datasets Tracked**: {num_entries}
- **Last Updated**: {_get_current_timestamp()}

## Usage

```python
from datasets import load_dataset

manifest = load_dataset("{manifest_repo}", split="train")
print(f"Tracking {{len(manifest)}} datasets")
```

"""

    try:
        api = HfApi()
        retry_on_timeout(
            api.upload_file,
            path_or_fileobj=readme_content.encode(),
            path_in_repo="README.md",
            repo_id=manifest_repo,
            repo_type="dataset",
        )
    except Exception as e:
        print(f"Warning: Failed to update manifest README: {e}")


def dataset_exists_in_manifest(dataset_name: str) -> bool:
    manifest = get_manifest()
    if manifest is None:
        return False
    return dataset_name in _manifest_to_dict(manifest)
