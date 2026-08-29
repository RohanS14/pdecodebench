"""The registry and the runtime reasoning sets must not drift apart.

`data/model_registry.csv` supplies the release_date that the generational trend
figure plots against, and the reasoning_mode that says how each model was asked to
think. Neither is derivable from the results rows, so if the registry disagrees with
`run_cross_modal_consistency.py` the figure mislabels its own x-axis and nothing
crashes to say so.

That is not hypothetical. F9: `Qwen3-32B` was recorded in one place as a thinking
model and run as a non-thinking one, because the config key that said so was read by
nothing. These tests are the check that would have caught it.
"""
import csv
import os

import pytest

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
REGISTRY = os.path.join(ROOT, "data", "model_registry.csv")

import sys                                                        # noqa: E402
sys.path.insert(0, ROOT)
from cross_modal_consistency.eval.run_cross_modal_consistency import (         # noqa: E402
    ALWAYS_THINKING, TOGGLEABLE, supports)

EXPECTED_COLUMNS = [
    "model_id", "short", "family", "generation", "release_date",
    "params_total_b", "params_active_b", "context_len", "reasoning_mode",
    "think_tags", "open_training_data", "verified_on",
]


def _rows():
    with open(REGISTRY, newline="") as fh:
        return list(csv.DictReader(fh))


def test_registry_has_the_expected_columns():
    with open(REGISTRY, newline="") as fh:
        assert next(csv.reader(fh)) == EXPECTED_COLUMNS


def test_every_runnable_model_has_a_registry_row():
    """A model that runs but has no row plots at an unknown date -- i.e. silently
    drops off the trend figure rather than erroring."""
    known = {r["model_id"] for r in _rows()}
    missing = (TOGGLEABLE | ALWAYS_THINKING) - known
    assert not missing, f"no registry row for: {sorted(missing)}"


@pytest.mark.parametrize("row", _rows(), ids=lambda r: r["model_id"])
def test_reasoning_mode_agrees_with_the_runtime_sets(row):
    """The registry's claim about how a model thinks must match how it is actually
    invoked. This is the F9 guard."""
    mid, mode = row["model_id"], row["reasoning_mode"]
    if mode == "toggle":
        assert mid in TOGGLEABLE, f"{mid} says toggle but is not in TOGGLEABLE"
        assert mid not in ALWAYS_THINKING
    elif mode == "always":
        assert mid in ALWAYS_THINKING, f"{mid} says always but is not in ALWAYS_THINKING"
        assert mid not in TOGGLEABLE
    else:
        pytest.fail(f"{mid}: unexpected reasoning_mode {mode!r}")


@pytest.mark.parametrize("row", _rows(), ids=lambda r: r["model_id"])
def test_every_registered_model_can_run_the_on_arm(row):
    """The roster is reasoning-only, so a model that cannot honour --thinking on has
    no place in it."""
    assert supports(row["model_id"], "on")


@pytest.mark.parametrize("row", _rows(), ids=lambda r: r["model_id"])
def test_release_date_is_a_sortable_iso_date(row):
    """The trend figure sorts on this string; a stray format would reorder the
    x-axis without failing."""
    d = row["release_date"]
    assert len(d) == 10 and d[4] == d[7] == "-", d
    y, m, day = d.split("-")
    assert 2024 <= int(y) <= 2030 and 1 <= int(m) <= 12 and 1 <= int(day) <= 31


@pytest.mark.parametrize("row", _rows(), ids=lambda r: r["model_id"])
def test_total_params_are_present_and_in_the_roster_band(row):
    """Size is held near-constant on purpose -- it is what keeps recency from being
    confounded with scale. A model outside the band breaks that control silently."""
    total = float(row["params_total_b"])
    assert 27.0 <= total <= 36.0, f"{row['model_id']} at {total}B leaves the band"


@pytest.mark.parametrize("row", _rows(), ids=lambda r: r["model_id"])
def test_context_length_leaves_room_for_the_prompt(row):
    """Prompts run to ~9k median and 32.5k max. A model whose context cannot hold a
    worst-case prompt plus a usable trace would truncate systematically rather than
    in the tail."""
    assert int(row["context_len"]) >= 40960, row["model_id"]
