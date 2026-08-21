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
    """Rows = condition; columns = detection, localization, judge, n. CIs inline."""
    t = M.main_table(df)
    caption = caption or (
        "Cross-representation consistency by condition. Detection accuracy is scored "
        "against the truth that A0 items agree; localization is conditional on the "
        "item having been correctly flagged; judge-confirmed is the rate at which an "
        "LLM judge finds the justification names the real defect. Brackets give "
        "95\\% Wilson intervals.")
    lines = [
        r"\begin{table}[t]", r"\centering", r"\small",
        r"\begin{tabular}{lrrrr}", r"\toprule",
        (r"condition & detection acc. & localization acc. & judge-confirmed & "
         r"$n$ \\"), r"\midrule",
    ]
    order = [c for c in CONDITIONS if c in set(t.get("condition", []))]
    for c in order:
        r = t[t["condition"].eq(c)].iloc[0]
        lines.append(
            f"{c} & {_fmt(r.detection_rate, r.detection_lo, r.detection_hi)} & "
            f"{_fmt(r.localization_rate, r.localization_lo, r.localization_hi)} & "
            f"{_fmt(r.judge_rate, r.judge_lo, r.judge_hi)} & {int(r.n)} \\\\")
    if not order:
        lines.append(r"\multicolumn{5}{c}{no rows} \\")
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
