"""The new figure must render on synthetic data, sum to 1.0 per row, and stay OFF
by default so the frozen report cannot pick it up."""
import re
from viz.consistency import claim_report as CR
from viz.consistency.synth import generate


def test_renders_and_is_not_clickable():
    d = generate()
    svg = CR.fig_blame_stack_unconditional(d)
    assert svg.startswith("<svg")
    # gids are what wire a segment to the drill-down; this figure must emit none,
    # because its denominators are not the ones build_drilldown indexes.
    assert 'id="tp|' not in svg


def test_miss_category_is_present_and_rows_sum_to_all_corrupted():
    import pandas as pd
    from viz.consistency import metrics as M
    from viz.consistency.sensitivity import SIGNAL_CONDITIONS
    d = generate()
    p = M.prepare(d)
    corrupted = p[p["is_corrupted"]]
    # Every corrupted row lands in exactly one bucket: named a view, named none,
    # unreadable, or never flagged. That is what makes the shares interpretable.
    for cond in SIGNAL_CONDITIONS:
        sub = corrupted[corrupted["condition"].eq(cond)]
        if not len(sub):
            continue
        det = sub["detected"]
        named = sub["pred_outlier"].where(det)
        buckets = int(named.notna().sum()) + int((~det).sum())
        assert buckets == len(sub), (cond, buckets, len(sub))
    assert int((~corrupted["detected"]).sum()) > 0, "synth frame has no misses to show"


def test_defaults_off_so_the_frozen_report_is_unaffected():
    import inspect
    sig = inspect.signature(CR.build)
    assert sig.parameters["blame_unconditional"].default is False
    src = inspect.getsource(CR.build)
    assert "blame_unconditional and svg is not None" in src


def test_legend_omits_categories_with_no_data():
    """`none` is empty by construction -- the parser emits it exactly when the model
    said the views agree, which is already the MISS bucket. A legend entry for a
    segment that can never be drawn is a promise the figure cannot keep."""
    import re
    from viz.consistency.synth import generate
    d = generate()
    svg = CR.fig_blame_stack_unconditional(d)
    text = " ".join(re.findall(r">([^<>]+)<", svg))
    assert "does not identify disagreement" in text
    assert "named none" not in text


def test_no_pooled_row():
    """Pooling averages corruptions that differ by orders of magnitude in
    detectability, so the summary bar would describe the mix, not the model."""
    import re
    from viz.consistency.synth import generate
    svg = CR.fig_blame_stack_unconditional(generate())
    text = " ".join(re.findall(r">([^<>]+)<", svg))
    assert "pooled" not in text.lower()
    assert "was corrupted" in text


def test_hide_empty_defaults_off_so_the_frozen_figure_is_unchanged():
    """hide_empty must be opt-in: consistency_claims.html shows the `none` segment
    and its build script has to keep reproducing it."""
    import inspect
    import re
    from viz.consistency.synth import generate
    sig = inspect.signature(CR.fig_blame_stack)
    assert sig.parameters["hide_empty"].default is False
    d = generate()
    off = " ".join(re.findall(r">([^<>]+)<", CR.fig_blame_stack(d)))
    assert "none" in off, "default form must still carry the none category"


def test_legend_names_the_outlined_correct_answer():
    """The outline is drawn on every row; without a legend entry it reads as
    emphasis rather than as the row's true outlier."""
    import re
    from viz.consistency.synth import generate
    svg = CR.fig_blame_stack_unconditional(generate())
    text = " ".join(re.findall(r">([^<>]+)<", svg))
    assert "correct answer for this row" in text


def test_x_axis_ties_the_colours_to_blame():
    """The axis must name both the denominator and what the segments mean, parallel
    to the conditional figure. Naming only the denominator leaves the legend
    floating."""
    import re
    from viz.consistency.synth import generate
    text = " ".join(re.findall(r">([^<>]+)<",
                               CR.fig_blame_stack_unconditional(generate())))
    assert "share of all items with this corruption" in text
    assert "the view the model blamed" in text


def test_every_row_has_the_same_n_when_no_verdict_draws_are_kept():
    """The design is balanced, so unequal row totals can only come from the
    non-uniform exclusion of no-verdict draws. Keeping them restores equality."""
    import numpy as np
    import pandas as pd
    from viz.consistency import metrics as M
    from viz.consistency.sensitivity import SIGNAL_CONDITIONS
    from viz.consistency.synth import generate
    d = generate()
    d = d.copy()
    # mark a condition-dependent slice as no-verdict, the way real truncation lands
    rng = np.random.default_rng(0)
    d["no_verdict"] = rng.random(len(d)) < 0.05
    p = M.prepare(d)
    p["no_verdict"] = d["no_verdict"].to_numpy()
    cor = p[p["is_corrupted"]]
    totals = {c: int(cor["condition"].eq(c).sum()) for c in SIGNAL_CONDITIONS}
    totals = {k: v for k, v in totals.items() if v}
    # keeping every draw, each condition's total is its full generated count
    assert len(set(totals.values())) == 1, totals


def test_no_verdict_rows_are_not_counted_as_misses():
    """709 of Nemotron's 907 no-verdict draws carry a scavenged `agree=yes`. Folding
    them into the miss bucket would fabricate the figure's headline finding.

    Asserted on the counts rather than on the source text: the classification moved
    out of the figure into `_unconditional_counts` when a second figure started
    drawing these bars, and a test anchored on where the code lived would have failed
    for the move while a test anchored on what it does would not.
    """
    import numpy as np
    from viz.consistency.sensitivity import SIGNAL_CONDITIONS
    from viz.consistency.synth import generate

    d = generate()
    cond = SIGNAL_CONDITIONS[0]
    base = CR._unconditional_counts(d)[cond][0]

    # Every draw of one condition marked no-verdict, and every one of them carrying
    # the scavenged "agree" that would file it as a miss if it were read as an answer.
    d2 = d.copy()
    mark = d2["condition"].eq(cond).to_numpy()
    d2["no_verdict"] = mark
    d2.loc[mark, "pred_agree"] = "yes"
    counts, total = CR._unconditional_counts(d2)[cond]

    assert counts[CR.NOVERDICT] == total
    assert counts[CR.MISS] == 0, "no-verdict draws were filed as misses"
    assert base[CR.MISS] > 0, "fixture cannot show the difference"


def test_a_repo_missing_draws_is_not_treated_as_complete():
    """Qwen3.6's killed backfill arm had all 1,024 items but only 2,981 of 3,072
    draws -- and because the missing ones were exactly the unusable draws, it
    reported ZERO no-verdict rows. An items-only completeness test promotes that
    over the intact source arm and shows a truncated arm as a flawless one."""
    import pandas as pd
    import viz.build_claims_expanded as B
    full = pd.DataFrame({"item_id": [i // 3 for i in range(B.N_ITEMS * 3)]})
    short = pd.DataFrame({"item_id": list(range(B.N_ITEMS))
                          + [i // 2 for i in range(B.N_ITEMS)]})
    assert B._is_complete(full) is True
    assert B._is_complete(short) is False       # 1024 items, too few draws
    assert B._is_complete(None) is False
    assert B._is_complete(pd.DataFrame({"item_id": []})) is False
