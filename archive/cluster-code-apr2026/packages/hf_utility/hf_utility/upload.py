"""Dataset upload with automatic README and manifest management."""

from __future__ import annotations

import json
import os
import warnings
from typing import Optional

from datasets import Dataset
from huggingface_hub import HfApi

from hf_utility.config import get_org
from hf_utility.manifest import update_manifest_entry
from hf_utility.retry import retry_on_timeout


def _warn_empty_columns(dataset: Dataset) -> None:
    """Raise if any string column is 100% empty — likely a parsing bug."""
    empty_cols = []
    for col in dataset.column_names:
        if getattr(dataset.features[col], "dtype", None) != "string":
            continue
        values = dataset[col]
        if len(values) == 0:
            continue
        if sum(1 for v in values if v and str(v).strip()) == 0:
            empty_cols.append(col)

    if empty_cols:
        msg = (
            f"All-empty string columns detected: {empty_cols}. "
            f"This usually means a parsing bug. "
            f"Set HF_ALLOW_EMPTY_COLUMNS=1 to upload anyway."
        )
        if os.environ.get("HF_ALLOW_EMPTY_COLUMNS") == "1":
            warnings.warn(msg)
        else:
            raise ValueError(msg)


def _generate_readme(
    dataset: Dataset,
    dataset_name: str,
    metadata: dict,
    tags: list[str],
    org: str,
    column_descriptions: Optional[dict[str, str]] = None,
    experiment_doc_link: Optional[str] = None,
) -> str:
    if column_descriptions is None:
        column_descriptions = {}

    columns_table = "| Column | Type | Description |\n|--------|------|-------------|\n"
    for col_name in dataset.column_names:
        col_type = str(dataset.features[col_name])
        description = column_descriptions.get(col_name, "*No description provided*")
        columns_table += f"| {col_name} | {col_type} | {description} |\n"

    metadata_json = json.dumps(metadata, indent=2)

    readme = f"""---
license: mit
tags:
{chr(10).join(f'- {tag}' for tag in tags)}
---

# {dataset_name}

{metadata.get('description', 'Dataset uploaded via hf_utility.')}

## Dataset Info

- **Rows**: {len(dataset)}
- **Columns**: {len(dataset.column_names)}

## Columns

{columns_table}

## Generation Parameters

```json
{metadata_json}
```

"""

    if experiment_doc_link:
        readme += f"""## Experiment Documentation

For complete experiment details, see [{experiment_doc_link}]({experiment_doc_link})

"""

    readme += f"""## Usage

```python
from datasets import load_dataset

dataset = load_dataset("{org}/{dataset_name}", split="train")
print(f"Loaded {{len(dataset)}} rows")
```

---

"""

    return readme


def push_dataset_to_hub(
    dataset: Dataset,
    dataset_name: str,
    metadata: dict,
    tags: list[str],
    org: Optional[str] = None,
    column_descriptions: Optional[dict[str, str]] = None,
    experiment_doc_link: Optional[str] = None,
    readme: Optional[str] = None,
    config_name: Optional[str] = None,
    skip_readme: bool = False,
    skip_manifest: bool = False,
) -> str:
    """Upload a dataset to HuggingFace Hub with automatic README and manifest.

    Args:
        dataset: The Dataset object to upload.
        dataset_name: Name for the dataset (without org prefix).
        metadata: Dict with at least: script_name, model, description.
        tags: List of tags.
        org: HF org/user. If None, resolved via get_org().
        column_descriptions: Optional {column: description} dict.
        experiment_doc_link: Optional URL to experiment docs.
        readme: Custom README content. None = auto-generate.
        config_name: Optional config/subset name for multi-config repos.
        skip_readme: Skip README generation.
        skip_manifest: Skip manifest update.

    Returns:
        Full repository path (e.g., "my-org/my-dataset").
    """
    resolved_org = org or get_org()
    repo_id = f"{resolved_org}/{dataset_name}"

    required_fields = ["script_name", "model", "description"]
    for field in required_fields:
        if field not in metadata:
            raise ValueError(f"Missing required metadata field: {field}")

    metadata.setdefault("hyperparameters", {})
    metadata.setdefault("input_datasets", [])

    _warn_empty_columns(dataset)

    config_label = f" (config={config_name})" if config_name else ""
    print(f"Uploading dataset to {repo_id}{config_label}...")

    push_kwargs = {"private": False}
    if config_name:
        push_kwargs["config_name"] = config_name
    retry_on_timeout(dataset.push_to_hub, repo_id, **push_kwargs)

    if not skip_readme:
        if readme is None:
            readme_content = _generate_readme(
                dataset=dataset,
                dataset_name=dataset_name,
                metadata=metadata,
                tags=tags,
                org=resolved_org,
                column_descriptions=column_descriptions,
                experiment_doc_link=experiment_doc_link,
            )
        else:
            readme_content = readme

        api = HfApi()
        retry_on_timeout(
            api.upload_file,
            path_or_fileobj=readme_content.encode(),
            path_in_repo="README.md",
            repo_id=repo_id,
            repo_type="dataset",
        )
        print(f"Uploaded README to {repo_id}")

    if not skip_manifest:
        update_manifest_entry(dataset_name=dataset_name, metadata=metadata, tags=tags)

    print(f"Successfully uploaded: {repo_id}")
    return repo_id


def upload_file_to_repo(
    dataset_name: str,
    file_path: str,
    path_in_repo: str,
    org: Optional[str] = None,
    max_size_bytes: int = 2 * 1024 * 1024,
) -> str:
    """Upload a file to an existing dataset repo.

    Returns the URL of the uploaded file.
    """
    resolved_org = org or get_org()
    repo_id = f"{resolved_org}/{dataset_name}"

    file_size = os.path.getsize(file_path)
    if file_size > max_size_bytes:
        raise ValueError(
            f"File {file_path} is {file_size:,} bytes, exceeds max {max_size_bytes:,} bytes."
        )

    api = HfApi()
    retry_on_timeout(
        api.upload_file,
        path_or_fileobj=file_path,
        path_in_repo=path_in_repo,
        repo_id=repo_id,
        repo_type="dataset",
    )

    url = f"https://huggingface.co/datasets/{repo_id}/resolve/main/{path_in_repo}"
    print(f"Uploaded {file_path} to {repo_id}/{path_in_repo}")
    return url
