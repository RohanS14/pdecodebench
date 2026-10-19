"""Shared dataset loading for the eval pipeline.

The jul28 release ships as CSV (`data/merged_mod_jul28.csv` and friends); every
version up to v5 shipped as xlsx and now lives under `data/archive/`. This picks
the right reader so either can be passed to --dataset.
"""
from pathlib import Path

import pandas as pd

# Canonical dataset: both sources (human + synthetic), 256 rows, 8 mod_types.
DEFAULT_MOD_DATASET = "data/merged_mod_jul28.csv"
DEFAULT_BASE_DATASET = "data/merged_base_jul28.csv"


def load_dataset(path) -> pd.DataFrame:
    return pd.read_csv(path) if str(path).endswith(".csv") else pd.read_excel(path)
