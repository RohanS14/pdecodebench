"""LaTeX output. booktabs, no vertical rules, numbers pre-formatted here.

Formatting lives in Python rather than in siunitx macros so the .tex is portable
into whatever the workshop template turns out to be, and so the same strings can be
reused by the dashboard without a LaTeX pass.
"""
from .constants import CONDITIONS
from . import metrics as M

NA = "--"


def _fmt(rate, lo, hi, digits=3):
    if rate != rate:            # NaN
        return NA
    return f"{rate:.{digits}f} [{lo:.{digits}f}, {hi:.{digits}f}]"


def main_results_latex(df, caption=None, label="tab:consistency-main"):
    """Rows = condition; columns = detection, localization, [judge], n. CIs inline.

    The judge column is EMITTED ONLY IF there is judge data. No LLM-judge pass has
    been run over the real justifications, so `judge_correct` is absent from the real
    frames by design (adapter.py leaves it out rather than inventing it) and every
    judge cell would render as "--" under a caption promising "the rate at which an
    LLM judge finds the justification names the real defect". A column of dashes
    described that way reads as "the judge found nothing", which is a result; the
    truth is that the measurement does not exist. So the column and the sentence
    describing it are dropped together, and the caption says so.
    """
    t = M.main_table(df)
    # Read off the INPUT, not off t. metrics.judge_rate() coerces an all-null
    # judge_correct column to 0.0 rather than NaN, so a frame carrying the column but
    # no judgements renders as a confident row of zeroes -- "the judge confirmed
    # nothing" -- which is worse than the dashes it replaced.
    has_judge = ("judge_correct" in getattr(df, "columns", [])
                 and df["judge_correct"].notna().any())
    caption = caption or (
        "Cross-representation consistency by condition. Detection accuracy is scored "
        "against the truth that A0 items agree; localization is conditional on the "
        "item having been correctly flagged"
        + ("; judge-confirmed is the rate at which an LLM judge finds the "
           "justification names the real defect" if has_judge else
           ". No LLM-judge pass has been run over the justifications, so no "
           "judge-confirmed column is reported")
        + ". Brackets give 95\\% Wilson intervals.")
    ncol = 4 if has_judge else 3
    header = r"condition & detection acc. & localization acc. & "
    if has_judge:
        header += r"judge-confirmed & "
    lines = [
        r"\begin{table}[t]", r"\centering", r"\small",
        r"\begin{tabular}{l" + "r" * ncol + "}", r"\toprule",
        header + r"$n$ \\", r"\midrule",
    ]
    order = [c for c in CONDITIONS if c in set(t.get("condition", []))]
    for c in order:
        r = t[t["condition"].eq(c)].iloc[0]
        cells = [c,
                 _fmt(r.detection_rate, r.detection_lo, r.detection_hi),
                 _fmt(r.localization_rate, r.localization_lo, r.localization_hi)]
        if has_judge:
            cells.append(_fmt(r.judge_rate, r.judge_lo, r.judge_hi))
        cells.append(str(int(r.n)))
        lines.append(" & ".join(cells) + r" \\")
    if not order:
        lines.append(rf"\multicolumn{{{ncol + 1}}}{{c}}{{no rows}} \\")
    lines += [
        r"\bottomrule", r"\end{tabular}",
        rf"\caption{{{caption}}}", rf"\label{{{label}}}", r"\end{table}",
    ]
    return "\n".join(lines)


def write_main_results(df, path="figures/table_main.tex", **kw):
    import os
    os.makedirs(os.path.dirname(os.path.abspath(path)) or ".", exist_ok=True)
    tex = main_results_latex(df, **kw)
    with open(path, "w") as f:
        f.write(tex + "\n")
    return path


# ── obfuscation contrast ─────────────────────────────────────────────────────
# The figure form of this (fig5_obfuscation_contrast) is thirteen rows in three
# blocks stacked down a portrait axis: at text-column width it runs most of a page
# for a set of numbers a reader has to read off the labels anyway, because every row
# already prints its before, its after and its delta beside the dot. The interval is
# the only thing the dot plot adds that a table cannot, and an interval column says
# it in less space. The figure stays -- it is the right form on screen, where there
# is no page to run out of -- but the paper takes this.
# Row labels come from the same helpers the figures use, so they arrive with the
# typography a figure wants and LaTeX does not: an em dash, a curly apostrophe. Under
# pdflatex those are not "renders differently", they are a build error, and the .tex
# is written by a script nobody watches compile.
_LTX = {"\u2014": "---", "\u2013": "--", "\u2019": "'", "\u201c": "``",
        "\u201d": "''", "&": r"\&", "%": r"\%", "_": r"\_", "#": r"\#"}


def _ltx(text):
    out = str(text)
    for a, b in _LTX.items():
        out = out.replace(a, b)
    return out


def _signed(x, digits=1, escalate=False):
    """A signed number in math mode. Returns the body, without the $.

    Math mode because a hyphen is not a minus: in a column of deltas the text hyphen
    is visibly shorter than the plus it has to line up against.

    `escalate` is for interval bounds. The corrected upper bound on the trajectory
    row is -0.008pp -- clear of zero, which is why the row is starred, but it rounds
    to "0.0" at one decimal, and a bound printed as 0.0 beside a significance mark
    reads as a contradiction. Rather than round it away, the bound borrows a decimal
    until it is not zero. A delta never escalates: there the extra digit would be
    precision the estimate does not have.
    """
    for d in ([digits, digits + 1, digits + 2] if escalate else [digits]):
        v = round(100 * x, d)
        if v != 0:
            return f"{v:+.{d}f}"
    return f"0.{'0' * digits}"


def _pp(x):
    return f"${_signed(x)}$" if x == x else NA


def _ci_pp(lo, hi):
    if lo != lo:
        return NA
    return (f"$[{_signed(lo, escalate=True)},\\ "
            f"{_signed(hi, escalate=True)}]$")


def obfuscation_latex(df, caption=None, label="tab:obfuscation", n_boot=None):
    """Blame under obfuscated identifiers, as a table. Returns LaTeX.

    Sized deliberately, because the first two attempts were not. The figure form
    (fig5) is thirteen rows down a portrait axis: 7.5in of column at \linewidth,
    most of a page. The first table was the same four blocks at \small with five
    columns -- 3.6in, smaller but still half a page for thirteen numbers, which is
    not what a table buys you. Setting the blocks side by side got it to 1.8in tall
    but 6.7in wide against a 5.5in column, and only fits at 7pt.

    This is the third: one stack, three columns, \footnotesize. Before and after
    share a cell (10.3->12.1) and the delta carries its interval, so it is three
    columns rather than five, which is what makes a single group narrow enough to
    need no tricks. About 2.5in -- a third of the figure, at a size a reader can
    read, with the row labels still spelled out.

    Significance is the figure's -- a Bonferroni-corrected interval clear of zero --
    and is marked rather than starred so an unmarked row reads as measured-and-null,
    not as untested.
    """
    from . import figures as F
    from .constants import MODALITY_LABELS
    from .sensitivity import TRAJ_SHORT

    st = F.obfuscation_stats(df, n_boot=n_boot)

    def cond_label(cond):
        if cond.startswith("A-T-"):
            return "trajectory, " + TRAJ_SHORT[cond.rsplit("-", 1)[1]].replace(
                "invalid solver's output", "solver output")
        return MODALITY_LABELS[{"A-C": "C", "A-D": "D", "A-M": "M"}[cond]]

    def row(name, r):
        if not r:
            return None
        mark = r"^{*}" if r.get("significant") else ""
        shift = f"${100 * r['real']:.1f}\\to{100 * r['obf']:.1f}$"
        delta = _pp(r["diff"])[:-1] + mark + "$"
        return (" & ".join([_ltx(name), shift,
                            delta + "\\," + _ci_pp(r.get("clo"), r.get("chi"))])
                + r" \\")

    def block(title, made):
        made = [m for m in made if m]
        if not made:
            return []
        return ([r"\addlinespace",
                 rf"\multicolumn{{3}}{{@{{}}l}}{{\emph{{{_ltx(title)}}}}} \\"] + made)

    # The two over-all-items rows lead, unheaded. They are the headline -- the model
    # goes quieter overall -- and putting them first spends one fewer block heading
    # than filing them under one, which on a table this size is a visible row.
    lines = [r"\toprule", r"& names$\to$obf. & $\Delta$ [95\% CI] \\", r"\midrule"]
    lines += [m for m in (row("declines to blame", st.get("none")),
                          row("names the corrupted code", st.get("guilty"))) if m]
    lines += block("Blame share, of the items it did blame",
                   [row(MODALITY_LABELS.get(r["category"], r["category"]), r)
                    for r in st.get("rows", [])])
    lines += block("Declines to blame, by which view was corrupted",
                   [row(cond_label(r["condition"]), r)
                    for r in st.get("by_condition", [])])

    # Short on purpose. At 10pt across the measure every 95 characters here is
    # another line of float, and on a 2.5in table the caption was costing a fifth of
    # the height to restate what the column heads already say.
    caption = caption or (
        "Obfuscating source-code identifiers: rate with meaningful names, with them "
        "obfuscated, and the difference in points. Intervals bootstrap over "
        f"{st.get('n_solvers', 0)} systems, not items; $^{{*}}$ clears zero at "
        f"Bonferroni $\\alpha={st.get('alpha_corrected', 0.05):.4g}$. Unmarked rows "
        "are null, not untested.")

    return "\n".join([
        r"\begin{table}[t]", r"\centering", r"\footnotesize",
        r"\setlength{\tabcolsep}{4pt}", r"\begin{tabular}{@{}lcl@{}}"]
        + lines
        + [r"\bottomrule", r"\end{tabular}",
           rf"\caption{{{caption}}}", rf"\label{{{label}}}", r"\end{table}"])


def write_obfuscation(df, path="figures/table_obfuscation.tex", **kw):
    import os
    os.makedirs(os.path.dirname(os.path.abspath(path)) or ".", exist_ok=True)
    tex = obfuscation_latex(df, **kw)
    with open(path, "w") as f:
        f.write(tex + "\n")
    return path


def obfuscation_accuracy_latex(df, caption=None, label="tab:obfuscation-accuracy",
                               n_boot=None):
    """The obfuscation DUMBBELL as a table: did it name the right view? Returns LaTeX.

    A different quantity from `obfuscation_latex`, which is about where blame goes.
    This is localization accuracy -- of the items carrying a given corruption, the
    share on which the model named the view that was actually corrupted -- under
    meaningful identifiers versus obfuscated ones.

    The dumbbell spends a full 5.5in axis and eight rows of paired dots to place
    sixteen numbers between 30% and 90%, and then prints the delta beside each row
    anyway. The dots buy an ordering the reader can see; a sorted column buys the
    same ordering and the interval that the dumbbell had no room for.

    Every condition row has the same n by construction, so it is stated once in the
    caption rather than eight times down a column.
    """
    from . import obfuscation as OB
    from .constants import MODALITY_LABELS
    from .sensitivity import TRAJ_SHORT

    kw = {"n_boot": n_boot} if n_boot else {}
    rows = OB.paired_localization_by_condition(df, **kw)
    if not rows:
        rows = []

    def name(r):
        c = r["condition"]
        if c == "ALL":
            return "all corrupted items"
        if c.startswith("A-T-"):
            return "trajectory, " + TRAJ_SHORT[c.rsplit("-", 1)[1]].replace(
                "invalid solver's output", "solver output")
        return MODALITY_LABELS[{"A-C": "C", "A-D": "D", "A-M": "M"}[c]]

    lines = [r"\toprule",
             r"which view was corrupted & real & obfusc. & $\Delta$ [95\% CI] \\",
             r"\midrule"]
    for i, r in enumerate(rows):
        mark = r"^{*}" if r.get("significant") else ""
        lines.append(" & ".join([
            _ltx(name(r)), f"{100 * r['real']:.1f}", f"{100 * r['obf']:.1f}",
            _pp(r["diff"])[:-1] + mark + "$\\," + _ci_pp(r.get("clo"), r.get("chi")),
        ]) + r" \\")
        # The pooled row is a different denominator from the seven below it; a rule
        # says so where a blank line would read as decoration.
        if r["condition"] == "ALL":
            lines.append(r"\midrule")

    ns = {r["n"] for r in rows if r["condition"] != "ALL"}
    n_txt = (f"Each condition row covers $n={ns.pop():,}$ draws"
             if len(ns) == 1 else "Row $n$ varies")
    pooled = next((r for r in rows if r["condition"] == "ALL"), None)
    # Two lines. Every 95 characters here is another line of float at 10pt, and on a
    # nine-row table the caption was costing a third of the height.
    caption = caption or (
        "Naming the corrupted view, real identifiers vs.\\ obfuscated. "
        + (f"{n_txt}, pooled $n={pooled['n']:,}$; " if pooled else f"{n_txt}; ")
        + f"intervals over {rows[0]['n_solvers'] if rows else 0} systems, not items. "
        "$^{*}$ clears zero (pooled at $\\alpha=0.05$, rows Bonferroni-corrected "
        "across the seven); unmarked rows are null, not untested.")

    return "\n".join([
        r"\begin{table}[t]", r"\centering", r"\footnotesize",
        r"\setlength{\tabcolsep}{5pt}", r"\begin{tabular}{@{}lccl@{}}"]
        + lines
        + [r"\bottomrule", r"\end{tabular}",
           rf"\caption{{{caption}}}", rf"\label{{{label}}}", r"\end{table}"])


def write_obfuscation_accuracy(df, path="figures/table_obfuscation_accuracy.tex", **kw):
    import os
    os.makedirs(os.path.dirname(os.path.abspath(path)) or ".", exist_ok=True)
    tex = obfuscation_accuracy_latex(df, **kw)
    with open(path, "w") as f:
        f.write(tex + "\n")
    return path
