"""
KeyHandler — centralized API key management.

SETUP:
  1. Copy this file to key_handler.py (same directory)
  2. Fill in your actual API keys below
  3. key_handler.py is gitignored and will never be committed
  4. Call KeyHandler.set_env_key() at the top of any script that needs API access

USAGE:
  from key_handler import KeyHandler
  KeyHandler.set_env_key()
"""

import os


class KeyHandler:
    # --- Core LLM providers ---
    openai_key: str = "your-openai-api-key"
    anthropic_key: str = "your-anthropic-api-key"

    # --- HuggingFace ---
    hf_key: str = "your-hf-token"

    # --- Other inference providers ---
    together_key: str = "your-together-api-key"
    openrouter_key: str = "your-openrouter-api-key"

    # --- Compute ---
    runpod_key: str = "your-runpod-api-key"

    # --- Experiment tracking ---
    wandb_key: str = "your-wandb-api-key"

    @classmethod
    def set_env_key(cls) -> None:
        """Inject all configured keys into os.environ.

        Keys that still hold a placeholder value (starting with "your-" or empty)
        are skipped — only real keys are injected.
        """
        _mappings: list[tuple[str, list[str]]] = [
            ("openai_key",     ["OPENAI_API_KEY"]),
            ("anthropic_key",  ["ANTHROPIC_API_KEY"]),
            ("hf_key",         ["HF_TOKEN", "HF_API_KEY"]),
            ("together_key",   ["TOGETHER_API_KEY"]),
            ("openrouter_key", ["OPENROUTER_API_KEY"]),
            ("runpod_key",     ["RUNPOD_API_KEY"]),
            ("wandb_key",      ["WANDB_API_KEY"]),
        ]

        for attr, env_vars in _mappings:
            value: str = getattr(cls, attr, "")
            if not value or value.startswith("your-"):
                continue
            for env_var in env_vars:
                os.environ[env_var] = value
