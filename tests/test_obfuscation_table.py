"""The obfuscation contrast as LaTeX: same numbers as the figure, no broken .tex.

This table is written by a script and dropped into a paper. Nobody watches it
compile, so the things that can only be caught by compiling -- a stray em dash, an
unescaped percent, a row with the wrong number of cells -- are checked here.
"""
import re

import pytest

from viz.consistency import figures as F
from viz.consistency import tables as T
from viz.consistency.synth import Effects, generate


@pytest.fixture(scope="module")
def tex():
    return T.obfuscation_latex(generate(Effects(n_solvers=8)), n_boot=200)


def _body(tex):
    return [ln for ln in tex.splitlines()
            if ln.endswith(r"\\") and not ln.startswith(r"\multicolumn")
            and "&" in ln and not ln.startswith("&")]


def _ncol(tex):
    spec = re.search(r"\\begin\{tabular\}\{(.*?)\}\n", tex).group(1)
    return len(re.sub(r"@\{[^}]*\}", "", spec))


def test_every_row_has_the_declared_number_of_columns(tex):
    ncol = _ncol(tex)
    assert ncol == 3, "the table is three columns: label, before->after, delta [CI]"
    for ln in _body(tex):
        assert len(ln.rsplit(r"\\", 1)[0].split("&")) == ncol, ln


def test_no_characters_pdflatex_cannot_set(tex):
    """Row labels come from the figure helpers, which use an em dash and a curly
    apostrophe. Under pdflatex those are a build error, not a typography choice."""
    bad = [c for c in tex if ord(c) > 127]
    assert not bad, f"non-ASCII in .tex: {sorted(set(bad))}"


def test_bare_percent_signs_are_escaped(tex):
    for m in re.finditer(r"(?<!\\)%", tex):
        pytest.fail(f"unescaped % at {m.start()}: {tex[m.start()-40:m.start()+10]!r}")


def test_deltas_are_math_mode_not_hyphens(tex):
    """A text hyphen is shorter than the plus it lines up against in the column."""
    for ln in _body(tex):
        cell = ln.split("&")[2].rsplit(r"\\", 1)[0]
        # Nothing outside a $...$ group but the thin space between delta and CI:
        # every sign in the cell is therefore a math minus, not a text hyphen.
        outside = re.sub(r"\$[^$]*\$", "", cell).replace(r"\,", "").strip()
        assert not outside, (ln, outside)


def test_a_bound_that_rounds_to_zero_borrows_a_decimal(tex):
    """A starred row whose interval prints as touching 0.0 reads as a contradiction.

    The corrected bound on the trajectory row is -0.008pp: clear of zero, which is
    why it is starred, and invisible at one decimal.
    """
    for ln in _body(tex):
        if "^{*}" not in ln:
            continue
        ci = ln.split("&")[2]
        assert "0.0]" not in ci and "[0.0," not in ci, ln


def test_the_table_reports_the_figures_numbers(tex):
    """Both read obfuscation_stats; neither recomputes."""
    import inspect
    assert "obfuscation_stats(" in inspect.getsource(T.obfuscation_latex)
    assert "obfuscation_stats(" in inspect.getsource(F.fig5_obfuscation_contrast)


def test_significance_is_the_corrected_interval_not_the_raw_one(df_real=None):
    """A row is starred iff its Bonferroni-corrected interval clears zero -- the
    same rule the figure fills its marker on."""
    df = generate(Effects(n_solvers=8))
    st = F.obfuscation_stats(df, n_boot=200)
    tex = T.obfuscation_latex(df, n_boot=200)
    for r in st["rows"]:
        starred = any("^{*}" in ln for ln in _body(tex)
                      if ln.split("&")[0].strip().startswith(
                          {"C": "code", "T": "traj", "D": "desc",
                           "M": "math"}.get(r["category"], "zzz")))
        clears = not (r["clo"] <= 0 <= r["chi"])
        assert starred == clears, (r["category"], starred, clears)


def test_the_table_is_narrower_than_the_text_column(tex):
    """The side-by-side version measured 6.7in against a 5.5in column and only fit
    at 7pt. Three columns in one stack is what makes a single group narrow enough."""
    widest = max(len(re.sub(r"[\\${}^*]", "", ln.rsplit(r"\\", 1)[0]))
                 for ln in _body(tex))
    assert widest < 62, f"{widest} chars is wider than a 5.5in column at 8pt"


def test_the_stack_stays_short(tex):
    """Row count is the height. A block heading is a row like any other, which is
    why the two over-all-items rows lead unheaded rather than under a fourth one."""
    rows = [ln for ln in tex.splitlines() if ln.endswith(r"\\")]
    assert len(rows) <= 17, f"{len(rows)} typeset rows"
    assert sum("multicolumn" in ln for ln in rows) == 2
