"""Dataset and model deletion with manifest management."""

from __future__ import annotations

import re
from typing import Optional

from huggingface_hub import HfApi, list_datasets

from hf_utility.config import get_org
from hf_utility.manifest import remove_manifest_entry
from hf_utility.retry import retry_on_timeout


def list_org_datasets(
    org: Optional[str] = None,
    pattern: Optional[str] = None,
) -> list[str]:
    """List all datasets in the org, optionally filtered by regex."""
    resolved_org = org or get_org()
    datasets = retry_on_timeout(list_datasets, author=resolved_org)

    names = []
    for ds in datasets:
        name = ds.id.replace(f"{resolved_org}/", "")
        if pattern is None or re.search(pattern, name):
            names.append(name)
    return sorted(names)


def list_org_models(
    org: Optional[str] = None,
    pattern: Optional[str] = None,
) -> list[str]:
    """List all models in the org, optionally filtered by regex."""
    from huggingface_hub import list_models

    resolved_org = org or get_org()
    models = retry_on_timeout(list_models, author=resolved_org)

    names = []
    for model in models:
        name = model.id.replace(f"{resolved_org}/", "")
        if pattern is None or re.search(pattern, name):
            names.append(name)
    return sorted(names)


def delete_dataset(
    dataset_name: str,
    org: Optional[str] = None,
    confirm: bool = True,
    update_manifest: bool = True,
) -> bool:
    """Delete a single dataset from HuggingFace Hub.

    Returns True if deleted, False if cancelled or not found.
    """
    resolved_org = org or get_org()
    repo_id = f"{resolved_org}/{dataset_name}"

    if confirm:
        response = input(f"Delete dataset '{repo_id}'? [y/N]: ").strip().lower()
        if response != "y":
            print(f"Skipped: {repo_id}")
            return False

    api = HfApi()
    try:
        retry_on_timeout(api.delete_repo, repo_id=repo_id, repo_type="dataset")
        print(f"Deleted: {repo_id}")

        if update_manifest:
            removed = remove_manifest_entry(dataset_name)
            if removed:
                print(f"Removed from manifest: {dataset_name}")
            else:
                print(f"Note: {dataset_name} was not in manifest")

        return True
    except Exception as e:
        if "404" in str(e) or "not found" in str(e).lower():
            print(f"Not found: {repo_id}")
            return False
        raise


def delete_datasets(
    pattern: str,
    org: Optional[str] = None,
    confirm: bool = True,
    force: bool = False,
    update_manifest: bool = True,
    repo_type: str = "dataset",
) -> list[str]:
    """Delete datasets or models matching a regex pattern.

    Returns list of deleted names.
    """
    if repo_type not in ("dataset", "model"):
        raise ValueError(f"repo_type must be 'dataset' or 'model', got: {repo_type}")

    resolved_org = org or get_org()

    if repo_type == "dataset":
        matching = list_org_datasets(org=resolved_org, pattern=pattern)
    else:
        matching = list_org_models(org=resolved_org, pattern=pattern)

    if not matching:
        print(f"No {repo_type}s matching '{pattern}' in {resolved_org}")
        return []

    print(f"Found {len(matching)} {repo_type}(s) matching '{pattern}':")
    for name in matching:
        print(f"  - {name}")

    if not force:
        print()

    deleted = []
    api = HfApi()

    for name in matching:
        repo_id = f"{resolved_org}/{name}"

        should_delete = force
        if not force and confirm:
            response = input(f"Delete '{repo_id}'? [y/N]: ").strip().lower()
            should_delete = response == "y"
        elif not force:
            should_delete = True

        if not should_delete:
            print(f"Skipped: {repo_id}")
            continue

        try:
            retry_on_timeout(api.delete_repo, repo_id=repo_id, repo_type=repo_type)
            print(f"Deleted: {repo_id}")
            deleted.append(name)

            if update_manifest and repo_type == "dataset":
                removed = remove_manifest_entry(name)
                if removed:
                    print(f"  Removed from manifest")
        except Exception as e:
            if "404" in str(e) or "not found" in str(e).lower():
                print(f"Not found (already deleted?): {repo_id}")
            else:
                print(f"Error deleting {repo_id}: {e}")

    print(f"\nDeleted {len(deleted)} {repo_type}(s)")
    return deleted


def delete_models(
    pattern: str,
    org: Optional[str] = None,
    confirm: bool = True,
    force: bool = False,
) -> list[str]:
    """Delete models matching a regex pattern."""
    return delete_datasets(
        pattern=pattern,
        org=org,
        confirm=confirm,
        force=force,
        update_manifest=False,
        repo_type="model",
    )
