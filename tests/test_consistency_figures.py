"""Figures must build from degenerate inputs, not raise and not silently mislead.

Every case here is a real partial-run shape: a reasoning arm that has not been run,
a single condition, a model that produced nothing. The figures are built during a
run, not after it, so "the data is incomplete" is the normal case rather than the
exceptional one.
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import pytest

from viz.consistency import figures as F
from viz.consistency.synth import Effects, generate
from viz.consistency.constants import SCHEMA_COLUMNS


@pytest.fixture(scope="module")
def df():
    return generate(Effects(n_solvers=6, models=("m1", "m2")))


@pytest.mark.parametrize("name", list(F.FIGURES))
def test_every_figure_builds(df, name):
    fig = F.FIGURES[name](df)
    assert fig is not None
    plt.close(fig)


@pytest.mark.parametrize("name", list(F.FIGURES))
def test_every_figure_survives_an_empty_frame(name):
    fig = F.FIGURES[name](pd.DataFrame(columns=list(SCHEMA_COLUMNS)))
    assert fig is not None
    plt.close(fig)


@pytest.mark.parametrize("name", list(F.FIGURES))
def test_every_figure_survives_a_missing_reasoning_arm(df, name):
    """The case the spec calls out: reasoning=on absent."""
    only_off = df[df["reasoning"].eq("off")]
    assert not only_off.empty
    fig = F.FIGURES[name](only_off)
    assert fig is not None
    plt.close(fig)


@pytest.mark.parametrize("name", list(F.FIGURES))
def test_every_figure_survives_a_single_naming_level(df, name):
    fig = F.FIGURES[name](df[df["naming"].eq("real")])
    assert fig is not None
    plt.close(fig)


@pytest.mark.parametrize("name", list(F.FIGURES))
def test_every_figure_survives_clean_items_only(df, name):
    """A0 alone: no corrupted rows anywhere, so localization is undefined throughout."""
    fig = F.FIGURES[name](df[df["condition"].eq("A0")])
    assert fig is not None
    plt.close(fig)


def test_font_sizes_never_drop_below_the_paper_minimum(df):
    from viz.consistency import style
    style.apply()
    fig = F.fig1_blame_matrix(df)
    sizes = [t.get_fontsize() for ax in fig.axes for t in ax.texts]
    sizes += [t.get_fontsize() for ax in fig.axes
              for t in ax.get_xticklabels() + ax.get_yticklabels()]
    assert sizes and min(sizes) >= style.MIN_FONT_PT - 1e-9
    plt.close(fig)


def test_figure_width_fits_the_text_column(df):
    from viz.consistency import style
    for fn in F.FIGURES.values():
        fig = fn(df)
        assert fig.get_size_inches()[0] <= style.TEXT_WIDTH_IN + 1e-9
        plt.close(fig)


def test_saves_both_vector_and_raster(df, tmp_path):
    from viz.consistency import style
    fig = F.fig2_trust_scatter(df)
    pdf, png = style.save(fig, "t", outdir=str(tmp_path))
    assert pdf.endswith(".pdf") and png.endswith(".png")
    assert (tmp_path / "t.pdf").stat().st_size > 0
    assert (tmp_path / "t.png").stat().st_size > 0
    plt.close(fig)
