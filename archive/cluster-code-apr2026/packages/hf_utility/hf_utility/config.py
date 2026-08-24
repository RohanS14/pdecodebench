"""
Org resolution for hf_utility.

Resolution order:
  1. HF_ORG environment variable
  2. .raca/config.yaml hf_org field (walks up from cwd)
  3. HuggingFace username from logged-in token
  4. Raises RuntimeError
"""

from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path


def _find_raca_config() -> dict | None:
    """Walk up from cwd looking for .raca/config.yaml."""
    try:
        import yaml
    except ImportError:
        return None

    current = Path.cwd()
    while current != current.parent:
        config_path = current / ".raca" / "config.yaml"
        if config_path.is_file():
            with config_path.open() as f:
                return yaml.safe_load(f) or {}
        current = current.parent
    return None


def _hf_username() -> str | None:
    """Get the logged-in HuggingFace username, or None."""
    try:
        from huggingface_hub import HfApi
        api = HfApi()
        info = api.whoami()
        return info.get("name")
    except Exception:
        return None


@lru_cache(maxsize=1)
def get_org() -> str:
    """Resolve the HuggingFace org to upload to.

    Returns the org name string (e.g. "my-research-org" or "username").
    Raises RuntimeError if no org can be determined.
    """
    # 1. Environment variable
    env_org = os.environ.get("HF_ORG")
    if env_org:
        return env_org

    # 2. .raca/config.yaml
    config = _find_raca_config()
    if config and config.get("hf_org"):
        return config["hf_org"]

    # 3. HF username
    username = _hf_username()
    if username:
        return username

    raise RuntimeError(
        "Cannot determine HuggingFace org. Set one of:\n"
        "  - HF_ORG environment variable\n"
        "  - hf_org field in .raca/config.yaml\n"
        "  - Log in with `huggingface-cli login`"
    )


def get_manifest_repo() -> str:
    """Return the full repo ID for the RACA-PROJECT-MANIFEST dataset."""
    return f"{get_org()}/RACA-PROJECT-MANIFEST"
