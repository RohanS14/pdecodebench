"""
HuggingFace Utility — dataset upload, deletion, and manifest tracking.

Org is resolved at runtime from (in order):
  1. Explicit `org` parameter on each function call
  2. HF_ORG environment variable
  3. .raca/config.yaml hf_org field
  4. HuggingFace username (from token)

Usage:
    from hf_utility import push_dataset_to_hub, delete_datasets

    push_dataset_to_hub(
        dataset=my_dataset,
        dataset_name="my-experiment-v1",
        metadata={
            "script_name": "generate.py",
            "model": "meta-llama/Llama-3.1-8B",
            "description": "Experiment results",
        },
        tags=["experiment-1", "baseline"],
    )

    delete_datasets(pattern=r"test-.*", confirm=True)
"""

from hf_utility.config import get_org, get_manifest_repo
from hf_utility.upload import push_dataset_to_hub, upload_file_to_repo
from hf_utility.delete import delete_datasets, delete_dataset
from hf_utility.manifest import (
    get_manifest,
    update_manifest_entry,
    remove_manifest_entry,
)

__all__ = [
    "get_org",
    "get_manifest_repo",
    "push_dataset_to_hub",
    "upload_file_to_repo",
    "delete_datasets",
    "delete_dataset",
    "get_manifest",
    "update_manifest_entry",
    "remove_manifest_entry",
]
