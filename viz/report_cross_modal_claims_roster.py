"""Build consistency_claims_expanded.html — the same report, on the new sampled runs.

This is a SIBLING of viz/build_cross_modal_claims_frozen.sh, not a replacement. That script stays pinned
to the frozen repo (bermaneh/pde-llm-eval-xmodal-consistency-frozen-v1) and keeps
producing viz/consistency_claims.html untouched; this one reads the consolidated
generational repo and writes a separate file. Nothing here can overwrite the published
report.

The roster is DECLARED (see ROSTER), not discovered from whichever repos happen to
exist. A run that has not started yet has no repo, so a discovery-based list would
omit it silently and the report would look complete while missing half its models.
Every checkpoint in the intended roster therefore appears in the report's header
panel with its status -- complete, in progress, or queued -- whether or not it has
produced a row. ROSTER is cross-checked against data/model_registry.csv at build
time so the two cannot drift apart.

Two differences from the frozen run drive the filtering below:

1. k = 3. Every item was sampled three times under the uniform regime
   {temperature 0.6, top_p 0.95, top_k 20}, so a row is a DRAW, not an item. The
   draws are kept separate rather than pooled per item, which is what the eval
   recorded; `run_id` therefore carries sample_idx so two draws of one item are not
   collapsed into one run by the drill-down.

2. Some models never emit an opening <think>. Their chat template opens the reasoning
   block in the PROMPT, so only the closing tag appears in the response. When a
   response is truncated before that closing tag is written, the model never reached
   an answer -- but parse_consistency's regex cascade still scavenges a verdict out of
   the unfinished reasoning. Measured 2026-08-21: 907 of Nemotron's 3072 rows (29.5%)
   and 256 of GLM's 2304 (11.1%) carry a verdict the model never actually gave, and
   those verdicts skew toward "agree", which on a 7:1 corrupted design registers as
   misses and depresses both the hit rate and the false-alarm rate.

   Such rows are DROPPED here rather than scored, and the count is printed and shown
   in the report. Dropping is the honest option: scoring them as "said agree" invents
   an answer, and scoring them as failures counts a run that produced no verdict as
   if it had produced a wrong one.

   As of 2026-08-25 every arm reads ZERO here -- the budget draws were continued at a
   larger cap and the looping ones redrawn, and all 24,576 draws carry a verdict the
   model actually gave. The filter stays in place because it is the guard, not the
   fix: a future re-run at a smaller cap would silently reintroduce scavenged
   verdicts, and this is what would catch it.

Usage:
    .tools-venv/bin/python viz/report_cross_modal_claims_roster.py            # complete models only
    .tools-venv/bin/python viz/report_cross_modal_claims_roster.py --partial  # include in-flight runs
"""
import argparse
import pathlib
import html as _h
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from viz.consistency.adapter import from_xmodal                       # noqa: E402
from viz.consistency import claim_report as CR                        # noqa: E402

N_ITEMS = 1024
K_DRAWS = 3        # every arm samples k=3, so a complete arm is 1024 x 3 rows
OUT = "viz/consistency_claims_expanded.html"
REGISTRY = "data/model_registry.csv"

# The single CONSOLIDATED results repo. Until 2026-08-25 this was ~40 per-model repos
# -- one per arm plus a chain of `-backfill`/`-128k`/shard repos -- because concurrent
# jobs each uploaded their own arm and push_dataset_to_hub REPLACES a split, so two
# arms could not share one repo while they were running. That campaign is finished:
# every repair pass has been merged back into its arm, the arms carry `model` as a
# column, and the per-model repos have been deleted. The preference/union/repair
# machinery that used to pick between them is gone with them -- there is nothing left
# to prefer, and a rule that silently picks among repos is exactly what produced the
# stale-arm bugs recorded below.
SOURCE_REPO = "bermaneh/pde-llm-eval-cross-modal-consistency"

# The intended roster, in release order -- which is the axis this expansion exists to
# create. It is DECLARED, not read off the data: a model whose rows are missing must
# show up as an empty row in the report rather than vanishing from it, and a roster
# derived from `df.model.unique()` cannot tell "not run" from "not in the study".
ROSTER = [
    ("R1-Distill-32B",      "deepseek-ai/DeepSeek-R1-Distill-Qwen-32B"),
    ("QwQ-32B",             "Qwen/QwQ-32B"),
    ("Qwen3-32B",           "Qwen/Qwen3-32B"),
    ("Nemotron-3-Nano-30B", "nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16"),
    ("GLM-4.7-Flash",       "zai-org/GLM-4.7-Flash"),
    ("Qwen3.5-27B",         "Qwen/Qwen3.5-27B"),
    ("Qwen3.6-27B",         "Qwen/Qwen3.6-27B"),
    ("Qwen3.8-27B",         "Qwen/Qwen3.8-27B"),
]


def check_roster_matches_registry():
    """The roster and the model registry must name the same checkpoints.

    The registry supplies the release dates the generational figure plots against. A
    model in one and not the other would either plot at an unknown date or be left
    out of the report entirely, and neither failure raises on its own.
    """
    try:
        reg = set(pd.read_csv(REGISTRY)["model_id"].astype(str))
    except (FileNotFoundError, KeyError, ValueError):
        print(f"[expanded] WARNING: cannot read {REGISTRY}; roster unchecked")
        return
    mine = {m for _, m in ROSTER}
    if mine - reg:
        print(f"[expanded] WARNING: in roster but not {REGISTRY}: {sorted(mine - reg)}")
    if reg - mine:
        print(f"[expanded] WARNING: in {REGISTRY} but not roster: {sorted(reg - mine)}")


def load_raw(repo=SOURCE_REPO):
    """The consolidated results frame, or None if the repo is not readable.

    Existence is checked against the Hub BEFORE loading, because load_dataset falls
    back to the local cache when a repo 404s and only says so on stdout. On
    2026-08-22 that silently resurrected an abandoned run: a deleted arm's 384 stale
    rows -- 169 of them unusable -- rebuilt the report as though the run were live. A
    cached copy of a repo that no longer exists is not evidence about the run; it is
    a stale artifact, and this check is the only thing that tells the two apart.
    """
    from datasets import load_dataset
    from huggingface_hub import HfApi
    try:
        HfApi().dataset_info(repo)               # 404s here => genuinely absent
    except Exception as exc:                                          # noqa: BLE001
        print(f"[expanded] {repo} not readable: {type(exc).__name__}: {exc}")
        return None
    df = load_dataset(repo, split="train").to_pandas()
    print(f"[expanded] {repo}: {len(df):,} draws / "
          f"{df['model'].nunique()} model(s) / {df['item_id'].nunique():,} items")
    return df


def has_real_verdict(df):
    """False where the run was cut off before it ever produced an answer.

    A truncated response with no closing </think> is entirely reasoning: the model
    was still deliberating when the budget ran out. See the module docstring.
    """
    truncated = df["finish_reason"].astype(str).eq("length")
    closed = df["response"].astype(str).str.contains("</think>", regex=False)
    return ~(truncated & ~closed)


def sampling_of(df):
    """The decoding parameters this arm actually ran under, read off the rows.

    Taken from the data rather than from a constant in this file: the eval records
    what it used on every row, and a banner asserting the protocol independently
    could disagree with the run it is describing.
    """
    if df is None or "sampling" not in df.columns or not len(df):
        return ""
    label = {"temperature": "T", "top_p": "top_p", "top_k": "top_k"}

    def fmt(s):
        s = dict(s) if isinstance(s, dict) else {}
        return ", ".join(f"{label.get(k, k)}={v:g}" if isinstance(v, (int, float))
                         else f"{label.get(k, k)}={v}" for k, v in sorted(s.items()))

    vals = {fmt(s) for s in df["sampling"].head(2000)}
    k = int(df["k"].iloc[0]) if "k" in df.columns else 1
    # More than one regime in one arm means the run was resumed under changed
    # settings; say so rather than showing whichever came first.
    body = vals.pop() if len(vals) == 1 else " / ".join(sorted(vals)) + " (MIXED)"
    return f"{body}, k={k}"


def survey(raw, include_partial):
    """One status record per ROSTER entry, in release order.

    Sliced out of the one consolidated frame by `model`. A roster entry with no rows
    reports "queued" rather than being dropped, which is the whole reason ROSTER is
    declared rather than read off the data.
    """
    by_model = ({m: g for m, g in raw.groupby("model")}
                if raw is not None and len(raw) else {})
    rows = []
    for short, model_id in ROSTER:
        df = by_model.get(model_id)
        rec = {"short": short, "model": model_id, "repo": SOURCE_REPO, "df": None,
               "df_all": None, "good_mask": None, "repair": None,
               "items": 0, "draws": 0, "dropped": 0, "dropped_loop": 0,
               "dropped_budget": 0, "sampling": "",
               "state": "queued", "used": False}
        if df is not None and len(df):
            good = has_real_verdict(df)
            # The two ways a draw ends without a verdict are not the same defect and
            # do not have the same remedy: a budget-limited draw can be finished by
            # giving it more tokens, a looping one cannot. Reporting them in one
            # column hid that Nemotron's 907 were 650 loops while GLM's 374 were 328
            # budget -- opposite problems presented as the same number. Both are
            # repaired as of 2026-08-25 and every arm now reads zero, but the split
            # stays: it is what a future re-run needs to see if one regresses.
            from cross_modal_consistency.eval.parse_consistency import is_looping
            lost = df[~good]
            loops = int(sum(1 for t in lost["response"] if is_looping(t))) if len(lost) else 0
            rec.update(items=int(df["item_id"].nunique()), draws=int(good.sum()),
                       dropped=int((~good).sum()), dropped_loop=loops,
                       dropped_budget=int((~good).sum()) - loops,
                       sampling=sampling_of(df), df=df[good],
                       # Every draw the model actually wrote, no-verdict ones
                       # included, plus the mask that separates them. The
                       # unconditional blame figure needs these: dropping them is
                       # what makes its row totals unequal, and the drop is not
                       # uniform across conditions, so the exclusion is not
                       # ignorable. With every arm now at zero drops the mask is
                       # all-True and the two figures coincide -- which is the
                       # result, not a reason to stop computing it.
                       df_all=df, good_mask=good)
            complete = _is_complete(df)
            rec["state"] = "complete" if complete else "in progress"
            rec["used"] = complete or include_partial
        rows.append(rec)
    return rows


def _is_complete(df):
    """A repo is complete only with the full ITEM count AND the full DRAW count.

    Testing items alone is not enough, and the difference is a trap rather than a
    nicety. A backfill arm copies every good row through and appends repairs, so a
    run killed before it writes its repairs still covers all 1,024 items -- it is
    just missing draws. Qwen3.6's arm on 2026-08-22 was exactly that: 1,024 items,
    2,981 of 3,072 draws, and because the 91 unusable draws were the ones never
    written it reported ZERO no-verdict rows. An items-only test would have promoted
    it over the complete source arm and shown a truncated arm as a flawless one.
    """
    if df is None or not len(df):
        return False
    return (df["item_id"].nunique() >= N_ITEMS
            and len(df) >= N_ITEMS * K_DRAWS)


def _left_cell(r):
    """Items still to generate for this checkpoint, or a dash when it is done.

    Reported in ITEMS rather than draws because the runner's unit of work is the
    item -- it generates k=3 draws per item in one go and checkpoints on the item --
    so items-left is what maps onto remaining runtime.
    """
    left = max(0, N_ITEMS - r["items"])
    if not left:
        return '<span style="color:var(--muted)">&mdash;</span>'
    return f'<b>{left:,}</b>'


def inject_provenance(path, rows):
    """Put the full roster and the decoding protocol at the top of the report."""
    with open(path, encoding="utf-8") as fh:
        doc = fh.read()

    badge = {"complete": ("var(--accent)", "complete"),
             "in progress": ("var(--muted)", "in progress"),
             "queued": ("var(--muted)", "queued")}
    tr = []
    for r in rows:
        colour, word = badge[r["state"]]
        if r["state"] == "queued":
            state = f'<span style="color:{colour}">queued &mdash; not yet run</span>'
        else:
            label = word
            if r["state"] != "complete":
                label = "{} ({:,}/{:,})".format(word, r["items"], N_ITEMS)
            state = '<span style="color:{}">{}</span>'.format(colour, label)
            if not r["used"]:
                state += ' <span style="color:var(--muted)">&middot; excluded</span>'
        link = (f'<a href="https://huggingface.co/datasets/{r["repo"]}" '
                f'style="color:var(--accent)">{_h.escape(r["model"])}</a>'
                if r["state"] != "queued" else
                f'<span style="color:var(--muted)">{_h.escape(r["model"])}</span>')
        dash = '<span style="color:var(--muted)">&mdash;</span>'

        def _drop_cell(now, key):
            """Current count, and where an in-flight repair has already got to.

            Two numbers rather than one because they answer different questions: the
            left is what the figures on this page are actually conditioning on, the
            right is what the roster will read once the running job lands. Collapsing
            them would either overstate the data in hand or hide the repair.
            """
            if not r["repair"]:
                return f'{now:,}'
            return (f'<span style="color:var(--muted)">{now:,}</span> '
                    f'<b>&rarr; {r["repair"][key]:,}</b>')

        tr.append(
            f'<tr><td style="padding-right:12px">{_h.escape(r["short"])}</td>'
            f'<td style="padding-right:12px">{link}</td>'
            f'<td style="text-align:right">{r["draws"]:,}</td>'
            f'<td style="text-align:right">{_drop_cell(r["dropped_budget"], "budget")}</td>'
            f'<td style="text-align:right">{_drop_cell(r["dropped_loop"], "loop")}</td>'
            f'<td style="text-align:right">{_left_cell(r)}</td>'
            f'<td style="padding-left:12px"><code>{_h.escape(r["sampling"]) or dash}</code></td>'
            f'<td style="padding-left:12px">{state}</td></tr>')

    items_left = sum(max(0, N_ITEMS - r["items"]) for r in rows)
    draws_left = items_left * 3
    n_done = sum(1 for r in rows if r["used"])
    n_complete = sum(1 for r in rows if r["used"] and r["state"] == "complete")
    n_partial = n_done - n_complete
    dropped_total = sum(r["dropped"] for r in rows if r["used"])
    loop_total = sum(r["dropped_loop"] for r in rows if r["used"])
    budget_total = sum(r["dropped_budget"] for r in rows if r["used"])

    # Two different defects, deliberately not summed into one number. A budget-limited
    # draw was still producing new reasoning when the cap hit and can be finished by
    # continuing it; a looping draw stopped terminating and no budget fixes it.
    # Nemotron loops on 21.2% of draws at a 32,768 cap and 21.2% at 131,072.
    drop_note = ""
    if dropped_total:
        drop_note = (
            f'<div style="margin-top:10px;color:var(--muted)">{dropped_total:,} draws '
            'were dropped before scoring: the generation hit its token limit while '
            'still inside the reasoning block, so the model never reached an answer. '
            'Those rows are excluded rather than counted as agreement, because a '
            'verdict a parser recovers from unfinished reasoning is not one the model '
            f'gave. They are <b>{budget_total:,} token-budget</b> draws, recoverable '
            'by continuing the trace from where it stopped, and '
            f'<b>{loop_total:,} decode loops</b>, whose tails repeat a single 12-gram '
            'and which more context does not fix. Both are now being repaired, by '
            'different means: a budget draw is <i>continued</i> from where it stopped, '
            'which is exact; a looping one is <i>redrawn</i> as a fresh sample at a new '
            'seed, which recovers coverage rather than that particular trace. Redrawing '
            'is not cherry-picking &mdash; every figure here already conditions on "the '
            'draw did not loop", so resampling estimates that same conditional '
            'distribution and changes the sample size, not the estimand.</div>')

    if any(r["repair"] for r in rows):
        n_rep = sum(1 for r in rows if r["repair"])
        before = sum(r["dropped"] for r in rows if r["repair"])
        after = sum(r["repair"]["no_verdict"] for r in rows if r["repair"])
        drop_note += (
            f'<div style="margin-top:10px;color:var(--muted)">A <b>&rarr;</b> in those '
            f'two columns marks a repair job still on the GPU. The left number is what '
            f'this page scores on; the right is what the repair arm reads over the draws '
            f'it has rewritten so far. Across {n_rep} model(s) that is '
            f'<b>{before:,} &rarr; {after:,}</b>. The figures deliberately stay on the '
            f'complete arm until the repair finishes: a partly-written repair arm covers '
            f'all 1,024 items while missing the very draws that failed, so promoting it '
            f'early would read as a flawless model rather than an unfinished one.</div>')

    if items_left:
        # The headline number a reader actually wants while the roster is filling:
        # how much is still missing, in the same unit the table reports.
        pct = 100.0 * (len(rows) * N_ITEMS - items_left) / (len(rows) * N_ITEMS)
        left_note = (
            f'<div style="margin-top:10px"><b>Still to generate: '
            f'{items_left:,} items ({draws_left:,} draws)</b> across the roster '
            f'&mdash; {pct:.1f}% of the intended '
            f'{len(rows) * N_ITEMS:,} items are in. The <i>Items left</i> column '
            'says which checkpoints they belong to. Draws left assumes k=3 per '
            'item, which is what every arm here samples.</div>')
    else:
        left_note = ('<div style="margin-top:10px"><b>Roster complete</b> &mdash; '
                     f'all {len(rows)} checkpoints are at the full '
                     f'{N_ITEMS:,} items.</div>')

    panel = (
        '<div class="designnote" style="margin:0 0 22px">'
        '<b>Expanded roster &mdash; sampled decoding.</b> Built from the generational '
        'runs; separate from the published greedy report '
        '(<code>consistency_claims.html</code>), which is unchanged. Every checkpoint '
        'runs reasoning-enabled under one uniform decoding regime, with each draw kept '
        'as its own observation. The figures below are built on '
        f'<b>{n_done} of {len(rows)}</b> checkpoints'
        + (f' &mdash; {n_complete} finished and {n_partial} still generating, folded '
           'in so results are visible early. A partial checkpoint contributes fewer '
           'items than a finished one, so every pooled panel is weighted toward '
           'whichever models are furthest along; treat cross-model comparisons as '
           'provisional until the roster fills.'
           if n_partial else ' , all of them finished.')
        + ' Checkpoints with no rows yet are listed so the intended roster stays '
        'visible while it fills in.'
        '<table style="width:100%;margin-top:10px;border-collapse:collapse">'
        '<tr style="color:var(--muted);text-align:left">'
        '<th style="padding-right:12px">Model</th>'
        '<th style="padding-right:12px">Checkpoint</th>'
        '<th style="text-align:right">Draws</th>'
        '<th style="text-align:right" title="hit the token cap while still '
        'producing new reasoning; recoverable by continuing the trace">Token budget</th>'
        '<th style="text-align:right" title="tail is a repeating 12-gram; the model '
        'stopped terminating, and no budget fixes it">Decode loop</th>'
        '<th style="text-align:right" title="items still to generate before this '
        'checkpoint reaches the full 1,024-item benchmark">Items left</th>'
        '<th style="padding-left:12px">Decoding</th>'
        '<th style="padding-left:12px">Status</th></tr>'
        + "".join(tr) + '</table>' + left_note + drop_note + '</div>')

    # The stock subtitle links the frozen dataset; these results do not come from it.
    doc = doc.replace(
        '<a href="https://huggingface.co/datasets/bermaneh/pde-llm-eval-cross-modal-consistency" '
        'style="color:var(--accent)">source dataset</a>',
        '<b style="color:var(--accent)">sampled runs &mdash; see roster below</b>')

    anchor = "</div>\n\n  \n<section"
    if anchor in doc:
        doc = doc.replace(anchor, "</div>\n\n  " + panel + "\n<section", 1)
    else:                                     # layout changed; never lose the panel
        doc = doc.replace("</h1>", "</h1>\n" + panel, 1)
    with open(path, "w", encoding="utf-8") as fh:
        fh.write(doc)


BROWSER_CSS = """
  .browser { max-width:100%; }
  .controls { display:flex; gap:9px; align-items:center; margin-bottom:12px;
              flex-wrap:wrap; }
  .controls select, .controls button { background:var(--raised); color:var(--text2);
      border:1px solid var(--tagline); border-radius:4px; padding:5px 10px;
      font-size:0.78rem; cursor:pointer; font-family:inherit; }
  .controls button:hover { background:var(--tagbg); }
  .controls span { font-size:0.75rem; color:var(--dim2); min-width:66px;
      text-align:center; font-family:ui-monospace, monospace; }
  .rb-meta { background:var(--panel2); border:1px solid var(--line); border-radius:6px 6px 0 0;
             padding:10px 13px; font-size:0.75rem; line-height:1.85; }
  .rb-line b { color:var(--link2); font-weight:500; }
  .rb-tag { background:var(--tagbg); color:var(--accent); border-radius:3px; padding:1px 7px;
            font-size:0.7rem; }
  .rb-gt { color:var(--green); } .rb-pa { color:var(--orange); }
  .rb-sc { color:var(--dim2); font-family:ui-monospace, monospace; font-size:0.72rem; }
  .rb-just { background:var(--sunk); border:1px solid var(--line); border-top:none;
             padding:11px 14px; font-size:0.78rem; line-height:1.7; color:var(--blue2); }
  .rb-just:empty { display:none; }
  .rb-jlabel { color:var(--accent); font-size:0.68rem; letter-spacing:0.08em;
               text-transform:uppercase; margin-bottom:5px; }
  .rb-text { background:var(--deep); border:1px solid var(--line); border-top:none;
             border-radius:0 0 6px 6px; padding:14px 16px; font-size:0.76rem;
             line-height:1.65; color:var(--text4); white-space:pre-wrap;
             word-wrap:break-word; max-height:620px; overflow-y:auto;
             font-family:ui-monospace, SFMono-Regular, Menlo, monospace; }
"""


VIEWS = ("code", "trajectory", "description", "math")


def named_view(r):
    """The view this draw actually accused, or "" if it accused nothing.

    The stored `outlier` is a SLOT ("view_2"), which means whatever sat in position 2
    for this item and nothing at all without this row's own permutation. Resolving it
    against a fixed order would produce a plausible, wrong answer -- so it is resolved
    here through the row's `slots`, exactly as adapter._resolve_outlier does for the
    figures. Shared helper, so the browser and the figures cannot disagree.
    """
    from viz.consistency.adapter import _slots
    p = str(r.get("outlier") or "").strip().lower()
    if p in ("", "none", "no", "nan"):
        return ""
    if p.startswith("view_"):
        sl = _slots(r.get("slots"))
        try:
            i = int(p.split("_", 1)[1]) - 1
        except ValueError:
            return ""
        return sl[i].lower() if 0 <= i < len(sl) else ""
    return p if p in VIEWS else ""


def is_clean(r):
    """True when nothing was corrupted in this item -- the A0 condition."""
    return str(r.get("corrupted_view") or "").strip().lower() in ("", "none", "nan")


def false_alarm_stats(raw):
    """Per model: how often a CLEAN item drew a named accusation, and which view.

    This is the one cell of the design where localization accuracy is not merely low
    but undefined: nothing was corrupted, so there is no correct view to name and
    every accusation here is wrong by construction. The pooled column is the honest
    answer to "when it names a view, is it right?" -- it puts these draws back in the
    denominator, where the conditional-on-corrupted rate leaves them out.
    """
    per = {}
    for r in raw.to_dict("records"):
        m = str(r.get("model") or "")
        v = named_view(r)
        d = per.setdefault(m, {"clean": 0, "clean_named": 0, "picks": {},
                               "corr_named": 0, "corr_right": 0})
        if is_clean(r):
            d["clean"] += 1
            if v:
                d["clean_named"] += 1
                d["picks"][v] = d["picks"].get(v, 0) + 1
        elif v:
            d["corr_named"] += 1
            d["corr_right"] += int(
                v == str(r.get("corrupted_view") or "").strip().lower())
    return per


def sample_raw(raw, per_cell=3, per_fa=2):
    """Stratified sample of draws for the reader, not the first N rows.

    Taking the head would hand back whichever model's repo loaded first, all of it
    one arm and mostly one outcome. Stratifying over (model x corrupted view x
    detection outcome) guarantees the browser holds misses next to hits for every
    model, which is the comparison the panel exists to support.

    A second stratum is layered on top: clean items where the model named a view
    anyway, keyed by WHICH view it named. The first stratum already admits clean
    draws, but it keys them on detection_correct alone, which cannot tell "said
    nothing is wrong, correctly" from "invented an outlier" -- the two live in the
    same cell and the sampler could return three of the former and none of the
    latter. Since these draws are the whole of the false-alarm claim, they are
    sampled by name so the reader can always find one.
    """
    buckets = {}
    for r in raw.to_dict("records"):
        key = (r.get("model"), r.get("corrupted_view"), r.get("detection_correct"))
        buckets.setdefault(key, []).append(r)
    out = []
    for key in sorted(buckets, key=lambda k: tuple(str(x) for x in k)):
        b = buckets[key]
        step = max(1, len(b) // per_cell)
        out.extend(b[::step][:per_cell])

    fa = {}
    for r in raw.to_dict("records"):
        v = named_view(r)
        if is_clean(r) and v:
            fa.setdefault((str(r.get("model") or ""), v), []).append(r)
    seen = {(str(r.get("model")), str(r.get("item_id")), r.get("sample_idx"))
            for r in out}
    for key in sorted(fa, key=lambda k: (str(k[0]), str(k[1]))):
        b = fa[key]
        step = max(1, len(b) // per_fa)
        for r in b[::step][:per_fa]:
            ident = (str(r.get("model")), str(r.get("item_id")), r.get("sample_idx"))
            if ident not in seen:
                seen.add(ident)
                out.append(r)
    return out


def inject_ladder(path, d, raw):
    """Add the model-comparison panel, built on the declared Qwen ladder.

    Replaces an earlier release-date scatter with a fitted slope. That figure put
    all eight models on a calendar axis, but four of them are singletons from four
    different labs -- the line through them described no trajectory any lab actually
    walked. Here the four consecutive Qwen releases carry the trend claim on an
    ORDINAL axis, and the singletons become the yardstick: how far apart can two
    contemporary 30B models be for reasons unrelated to time.
    """
    from viz.consistency import ladder as L

    cfg = L.load_roles()
    p_lo, p_hi = L.assert_scale_band(cfg)
    lad_ids = [m["model_id"] for m in cfg["ladder"]]
    ref_ids = [m["model_id"] for m in cfg["reference"]]

    t = L.tidy(d, raw)
    present = set(t["model"])
    missing = [m for m in lad_ids + ref_ids if m not in present]
    if missing:
        return {"error": "no rows for " + ", ".join(m.split("/")[-1]
                                                    for m in missing)}

    items, rep = L.common_items(t, lad_ids + ref_ids)
    base, base_view = L.best_constant_baseline(t, items)
    stats = L.per_model(t, items, lad_ids + ref_ids)
    short = {m["model_id"]: m["short"] for m in cfg["ladder"] + cfg["reference"]}
    rel = {m["model_id"]: m["release"] for m in cfg["ladder"] + cfg["reference"]}
    par = {m["model_id"]: m["params_b"] for m in cfg["ladder"] + cfg["reference"]}
    stats["short"] = stats["model"].map(short)
    stats["release"] = stats["model"].map(rel)
    stats["params_b"] = stats["model"].map(par)
    # Ladder rows stay in DECLARED release order, not whatever order they loaded in.
    lad = stats.set_index("model").loc[lad_ids].reset_index()
    ref = stats.set_index("model").loc[ref_ids].reset_index()

    v_loc = L.verdict_line(lad, ref, "loc_acc",
                           "conditional localization accuracy")
    v_agr = L.verdict_line(lad, ref, "agreement_rate", "agreement across 3 draws")
    # The dashed constant-strategy floor is drawn only on the calendar-axis panel
    # in the details drawer. It came off the main figure on 2026-08-24: with the
    # four ladder points, their CIs, the reference band and four margin labels
    # already on two panels, a fifth mark that has to be decoded from a legend
    # was costing more than it explained. The number itself is still stated in
    # the caption below, so nothing is lost -- only the line.
    svg = CR._svg(L.fig8_ladder(lad, ref))
    svg_dated = CR._svg(L.fig8_ladder_dated(lad, ref, baseline=base))

    def _row(r, role):
        f = lambda v: "&mdash;" if not np.isfinite(v) else f"{v * 100:.0f}%"
        ci = lambda a, b: ("" if not (np.isfinite(a) and np.isfinite(b))
                           else f" <span style='color:var(--dim)'>({a * 100:.0f}&ndash;{b * 100:.0f})</span>")
        return (f"<tr><td>{_h.escape(r['short'])}</td><td>{role}</td>"
                f"<td>{r['params_b']:.1f}B</td><td>{r['release']}</td>"
                f"<td>{r['n_items']}</td><td>{r['n_solvers']}</td>"
                f"<td>{f(r['loc_acc'])}{ci(r['loc_lo'], r['loc_hi'])}</td>"
                f"<td>{f(r['agreement_rate'])}{ci(r['agr_lo'], r['agr_hi'])}</td>"
                f"<td>{f(r['modal_accuracy'])}{ci(r['mod_lo'], r['mod_hi'])}</td></tr>")

    table = (
        "<table class='tbl'><tr><th>model</th><th>role</th><th>params</th>"
        "<th>released</th><th>items</th><th>solvers</th>"
        "<th>localization | committed</th><th>agreement</th>"
        "<th>modal accuracy</th></tr>"
        + "".join(_row(r, "ladder") for _, r in lad.iterrows())
        + "".join(_row(r, "reference") for _, r in ref.iterrows())
        + "</table>")

    tbo = L.by_true_outlier(t, items, lad_ids + ref_ids)
    piv = tbo.pivot(index="model", columns="true_outlier", values="loc_acc")
    piv = piv.reindex(lad_ids + ref_ids)
    bo_table = ("<table class='tbl'><tr><th>model</th>"
                + "".join(f"<th>{c}</th>" for c in piv.columns) + "</tr>"
                + "".join("<tr><td>" + _h.escape(short.get(m, m)) + "</td>"
                          + "".join("<td>" + ("&mdash;" if not np.isfinite(v)
                                              else f"{v * 100:.0f}%") + "</td>"
                                    for v in piv.loc[m]) + "</tr>"
                          for m in piv.index) + "</table>")

    lost = ", ".join(f"{short[m]} {rep['lost'][m]}" for m in lad_ids + ref_ids
                     if rep["lost"][m])
    section = (
        '<section id="ladder">'
        '<h2>Model comparison &mdash; the Qwen ladder</h2>'
        f'<div class="verdict v-supported">{v_loc}</div>'
        f'<div class="verdict v-supported">{v_agr}</div>'
        '<p class="sub" style="margin-bottom:14px">'
        'The four consecutive Qwen releases are the only models here that admit an '
        'ordering, so they carry the only trend claim, and they are drawn on an '
        '<b>ordinal</b> axis of release position &mdash; four points do not support '
        'a time axis, and no line is fitted through them. The other four models are '
        'singletons from four different labs; they appear as the shaded band and '
        'its median, showing how far apart two contemporary models of this size can '
        'be for reasons that have nothing to do with time. That band is the '
        'yardstick: a ladder movement smaller than the between-model spread is not '
        'a generational finding.</p>'
        '<p class="sub" style="margin-bottom:14px">'
        'Neither metric is hit rate. QwQ-32B hits 0.93 while false-alarming at 0.75 '
        'and Qwen3.5 hits 0.99 while false-alarming at 0.91 &mdash; both are close '
        'to answering &ldquo;these disagree&rdquo; to everything, which an '
        'accuracy-like measure would score as skill. <b>Localization | committed</b> '
        'asks, once a model has named a view, whether it named the corrupted one; a '
        'model that flags everything still has to point somewhere. '
        '<b>Agreement</b> asks how often all three draws of one item name the same '
        'view &mdash; at T=0.6 a model can be right by luck on a single draw, and '
        'agreement is what separates a localization from a sample of a prior.</p>'
        f'<figure>{svg}</figure>'
        '<p class="sub"><b>Caption.</b> Within-family trajectory across four '
        'releases from one lab. This is not a claim about the field. All eight '
        f'models are {p_lo:.1f}&ndash;{p_hi:.1f}B parameters, asserted at load time '
        'from data/models.yaml, so scale is held roughly constant rather than '
        'varying with date. Every metric is computed on the '
        f'<b>{rep["n_common"]:,} items common to all eight models</b> '
        f'({rep["frac_of_smallest"]:.0%} of the smallest model&rsquo;s '
        f'{rep["smallest_model_items"]:,}); items dropped to the intersection: '
        f'{lost or "none"}. Bootstrap intervals resample the '
        f'{int(lad["n_solvers"].max())} solver systems, not the draws. For scale: '
        f'always naming one fixed view scores {base * 100:.0f}% on this item set '
        f'(the best such view is &ldquo;{base_view}&rdquo;).</p>'
        '<details><summary class="sub">Details &mdash; per-model table, calendar '
        'axis, per-outlier breakdown</summary>'
        f'<div style="overflow-x:auto">{table}</div>'
        '<p class="sub" style="margin-top:18px">The same two panels against calendar '
        'date. <b>Confounded</b> and shown only for completeness: release dates are '
        'unevenly spaced, so a reader sees a slope and infers a rate where only four '
        'measurements exist.</p>'
        f'<figure>{svg_dated}</figure>'
        '<p class="sub" style="margin-top:18px"><b>Exploratory.</b> Localization '
        'accuracy split by which view was actually corrupted. The split divides each '
        'model&rsquo;s corrupted items four ways, so per-cell n is small and no '
        'single cell should be quoted alone.</p>'
        f'<div style="overflow-x:auto">{bo_table}</div>'
        '</details></section>')

    doc = pathlib.Path(path).read_text(encoding="utf-8")
    if "</body>" not in doc:
        raise SystemExit("no </body> in the report; refusing to guess an anchor")
    doc = doc.replace(
        "</body>",
        '<div class="wrap" style="padding-top:0">' + section + "</div></body>", 1)
    pathlib.Path(path).write_text(doc, encoding="utf-8")
    return {"verdict": v_loc, **rep}


def inject_newer_better(path, d, rows):
    """The rates-only generational section, appended directly after the ladder.

    Injection ORDER is document order -- every one of these helpers appends before
    </body> -- so this must be called between inject_ladder and inject_per_model to
    land where it belongs.

    The coverage caveat is MEASURED, never a literal. A hand-written "N items, M
    solvers, X still generating" goes stale the moment an arm finishes, and a stale
    caveat is worse than no caveat: it understates the evidence in the report's own
    voice, and nobody re-reads a warning banner to check whether it is still true.
    """
    from viz.consistency import ladder as L
    from viz.consistency import newer_better as NB

    cfg = L.load_roles()
    known = {m["model_id"] for m in cfg["ladder"] + cfg["reference"]}
    present = set(d["model"].astype(str))
    if not (known & present):
        return {"error": "no ladder/reference models in the scored frame"}
    # Which declared arms had not finished at build time, read off the survey rather
    # than asserted, so the banner names exactly what is provisional and nothing else.
    incomplete = [r["short"] for r in rows
                  if r["model"] in known and not r["used"]]
    section = NB.build_section(d, cfg, provisional={"incomplete": incomplete})
    if not section:
        return {"error": "no rows for the declared roster"}

    doc = pathlib.Path(path).read_text(encoding="utf-8")
    if "</body>" not in doc:
        raise SystemExit("no </body> in the report; refusing to guess an anchor")
    doc = doc.replace(
        "</body>",
        '<div class="wrap" style="padding-top:0">' + section + "</div></body>", 1)
    pathlib.Path(path).write_text(doc, encoding="utf-8")
    res = NB.metrics_by_model(d, cfg)
    order = sorted(res, key=lambda m: res[m]["release"])
    return {"n_models": len(res),
            "trends": {k: NB.trend(res, order, k)
                       for k in (NB.FLAG_C, NB.FLAG_K, "loc_micro", "loc_macro")},
            "incomplete": incomplete}


def inject_tradeoff(path, d, raw):
    """The conditional-vs-unconditional section, appended right after the ladder.

    Reuses the ladder's own tidy frame, common item set and bootstrap so the two
    sections are commensurable: a reader comparing "95%" in one against "80%" in the
    other must be seeing a difference in the DEFINITION, never in which items or
    which resampling unit produced it.
    """
    from viz.consistency import ladder as L
    from viz.consistency import tradeoff as TO

    cfg = L.load_roles()
    t = L.tidy(d, raw)
    ids = [m["model_id"] for m in cfg["ladder"] + cfg["reference"]]
    if not set(ids) <= set(t["model"]):
        return {"error": "not every declared model is in the scored frame"}
    items, _rep = L.common_items(t, ids)
    section = TO.build_section(t, items, cfg)
    if not section:
        return {"error": "no rows for the declared roster"}

    doc = pathlib.Path(path).read_text(encoding="utf-8")
    if "</body>" not in doc:
        raise SystemExit("no </body> in the report; refusing to guess an anchor")
    doc = doc.replace("</style>", PER_MODEL_CSS + "</style>", 1)
    doc = doc.replace(
        "</body>",
        '<div class="wrap" style="padding-top:0">' + section + "</div></body>", 1)
    pathlib.Path(path).write_text(doc, encoding="utf-8")
    fr = TO.per_model(t, items, ids)
    lad, ref = TO._decorate(fr, cfg)
    return {"verdict": TO.verdict(lad, ref)}


def inject_design(path, raw):
    """The item-construction appendix. Reads the delivered rows, asserts nothing."""
    from viz.consistency import design as DS

    section = DS.build_section(raw)
    if not section:
        return {"error": "item ids are not the 4-field design key"}
    f = DS.factorise(raw)
    doc = pathlib.Path(path).read_text(encoding="utf-8")
    if "</body>" not in doc:
        raise SystemExit("no </body> in the report; refusing to guess an anchor")
    doc = doc.replace("</style>", PER_MODEL_CSS + "</style>", 1)
    doc = doc.replace(
        "</body>",
        '<div class="wrap" style="padding-top:0">' + section + "</div></body>", 1)
    pathlib.Path(path).write_text(doc, encoding="utf-8")
    return {"n_items": f["n_items"], "crossed": f["crossed"],
            "balanced": f["balanced"], "draws": f["draws_per_item"]}


# The Q4 subsection is SPLICED INTO the existing answer rather than appended to the
# document, because it belongs under the obfuscation question and every other
# injector in this file appends at </body>. The anchor is the trajectory-split
# figure's caption, which is the last element the Q4 answer emits before its own
# <figcaption>. Anchoring is checked and raises rather than guessing: a silent miss
# would drop the block into the wrong section or lose it entirely.
_Q4_ANCHOR = "Trajectory, split into its four corruptions"


def inject_q4_stability(path, d, raw, rows):
    """Item-level stability under obfuscation, spliced into the Q4 answer.

    Nothing here recomputes or touches the pooled Q4 figures; it is an addition
    below them, on the same scored frame they use.
    """
    from viz.consistency import stability as ST

    used = [r for r in rows if r["used"]]
    present = set(d["model"].astype(str))
    models = [(r["model"], r["short"]) for r in used if r["model"] in present]
    if not models:
        return {"error": "no models present in the scored frame"}

    doc = pathlib.Path(path).read_text(encoding="utf-8")
    i = doc.find(_Q4_ANCHOR)
    if i == -1:
        return {"error": f"Q4 anchor not found: {_Q4_ANCHOR!r}"}
    j = doc.find('<p class="figcap">', i)
    k = doc.find("</p>", j) if j != -1 else -1
    if j == -1 or k == -1:
        return {"error": "Q4 split caption not found after the anchor"}
    cut = k + len("</p>")

    block = ST.build_block(d, raw, models)
    if not block:
        return {"error": "no paired items"}
    pathlib.Path(path).write_text(doc[:cut] + block + doc[cut:], encoding="utf-8")

    t = ST.tidy(d, raw)
    pairs, dropped = ST.paired_answers(t, "outlier", "modal")
    rows, n = ST.direction_rows(pairs, ST.BLAME_LEVELS)
    ch = ST.churn(pairs)
    return {"verdict": ST.direction_verdict(rows, ch, ST.BLAME_LABELS),
            "churn": ST.churn_line(rows, ch, ST.BLAME_LABELS),
            "pairs": n, "dropped": dropped}


PER_MODEL_CSS = """
  h3.pmh { color:var(--text2); font-size:0.95rem; font-weight:600;
           margin:34px 0 10px; padding-top:18px; border-top:1px solid var(--line); }
  #per-model table.tbl td span { font-size:0.86em; }
"""


def inject_per_model(path, d, rows):
    """Append the per-model appendix. Injected, never built inside claim_report.

    Same reason as the ladder and the raw browser: claim_report.build() also emits
    the FROZEN viz/consistency_claims.html, so a section added there would rewrite
    the published report as a side effect. Appending to the finished document is the
    only way to add a section to this report and not that one.

    `d` is the scored frame -- no-verdict draws already dropped -- so every quantity
    here shares the pooled figures' exclusions. The dropped counts come from the
    survey rows and are reported beside the scored n, because effective sample size
    differs across checkpoints and a per-model table that hides that invites
    comparing an arm of 3,072 with an arm of 2,930 as though they were the same.
    """
    from viz.consistency import per_model as PM

    used = [r for r in rows if r["used"]]
    # survey() names this key "model", not "model_id" -- the latter is the ladder
    # config's spelling, and mixing the two silently yields an empty roster here.
    models = [(r["model"], r["short"]) for r in used]
    dropped = {r["model"]: int(r.get("dropped", 0)) for r in used}
    present = set(d["model"].astype(str))
    models = [(m, s) for m, s in models if m in present]
    if not models:
        return {"error": "no models present in the scored frame"}

    section = PM.build_section(d, models, dropped=dropped)

    doc = pathlib.Path(path).read_text(encoding="utf-8")
    if "</body>" not in doc:
        raise SystemExit("no </body> in the report; refusing to guess an anchor")
    doc = doc.replace("</style>", PER_MODEL_CSS + "</style>", 1)
    doc = doc.replace(
        "</body>",
        '<div class="wrap" style="padding-top:0">' + section + "</div></body>", 1)
    pathlib.Path(path).write_text(doc, encoding="utf-8")
    return {"n_models": len(models)}


def _false_alarm_block(raw):
    """The table that answers "when it names a view, is it a correct view?".

    Every headline localization number in this report is conditional on the item
    being corrupted, so the clean items -- one eighth of the design -- never enter
    the denominator. That is the right convention for a localization rate, and it
    also means the reported rate cannot see the failure mode below. A model that
    names an outlier on an item where nothing was corrupted has named a wrong view
    with certainty, not with probability: there is no correct answer to name. The
    pooled column puts those draws back and is the only rate here that can fall for
    that reason.

    Nothing in this block feeds a figure; it reads the same raw frame the browser
    beneath it reads, so the two cannot drift.
    """
    per = false_alarm_stats(raw)
    if not per:
        return ""
    short = {m: sh for sh, m in ROSTER}
    order = [m for _, m in ROSTER if m in per]
    tr = []
    for m in order:
        d = per[m]
        com = d["corr_named"] + d["clean_named"]
        if not com:
            continue
        cond = d["corr_right"] / d["corr_named"] if d["corr_named"] else float("nan")
        pooled = d["corr_right"] / com
        fa = d["clean_named"] / d["clean"] if d["clean"] else float("nan")
        picks = sorted(d["picks"].items(), key=lambda kv: -kv[1])
        top = (f"{picks[0][0]} ({picks[0][1] / max(1, d['clean_named']):.0%})"
               if picks else "&mdash;")
        tr.append(
            f"<tr><td>{_h.escape(short.get(m, m))}</td>"
            f"<td>{cond * 100:.1f}%</td>"
            f"<td>{d['clean_named']:,} / {d['clean']:,}"
            f" <span style='color:var(--dim)'>({fa * 100:.0f}%)</span></td>"
            f"<td>{_h.escape(top)}</td>"
            f"<td>0 / {d['clean_named']:,}</td>"
            f"<td><b>{pooled * 100:.1f}%</b></td></tr>")

    tot_clean = sum(d["clean_named"] for d in per.values())
    tot_com = sum(d["corr_named"] + d["clean_named"] for d in per.values())
    return (
        '<h3 style="margin-top:26px">When it names a view, is it a correct view?</h3>'
        '<p class="sub" style="margin-bottom:14px">No &mdash; and one slice of the '
        'design makes that sharper than a rate suggests. On the <b>A0</b> items '
        'nothing was corrupted, so an accusation there is not a low-probability '
        'guess but a wrong answer with certainty: there is no correct view to name. '
        f'Across the eight models, <b>{tot_clean:,} of {tot_com:,} accusations '
        f'({tot_clean / tot_com:.1%})</b> were made on items where every view was '
        'consistent. The first column is the conditional rate the rest of this '
        'report quotes, which excludes those draws; the last puts them back. Read '
        'them together: the gap between the two is exactly the cost of a model that '
        'would rather name something than say nothing is wrong.</p>'
        '<div style="overflow-x:auto"><table>'
        '<thead><tr><th>Model</th>'
        '<th>Correct, given corrupted<br><span style="font-weight:400;'
        'color:var(--dim)">the reported rate</span></th>'
        '<th>Named a view on a clean item</th>'
        '<th>Which view it named there</th>'
        '<th>Correct there</th>'
        '<th>Correct, all accusations<br><span style="font-weight:400;'
        'color:var(--dim)">pooled</span></th></tr></thead>'
        '<tbody>' + "".join(tr) + '</tbody></table></div>'
        '<p class="sub" style="margin-top:10px">Every draw counted here is in the '
        'browser below and can be read in full: filter the cell column for '
        '<b>false alarm</b> to go straight to them. The accused view is resolved '
        'through each row&rsquo;s own slot permutation, not a fixed order, so '
        '&ldquo;view_2&rdquo; is read as whatever actually sat in position 2 for '
        'that item.</p>')


def inject_raw_browser(path, raw):
    """Append the raw-response reader, mirroring 2.11 of the dual report.

    Injected here rather than added to claim_report.build() on purpose: that
    function also produces the PUBLISHED viz/consistency_claims.html, which is
    frozen. Anything added there would change the frozen report as a side effect.
    """
    from viz.report_freegen_and_cross_modal import response_browser

    rows = []
    for r in sample_raw(raw):
        outcome = {1: "detected", 0: "missed"}.get(r.get("detection_correct"),
                                                   "unscored")
        # Clean items get their own outcome word. "missed" is the right word for a
        # corrupted item the model waved through, and exactly the wrong one for a
        # clean item it accused: nothing was there to miss. The cell string is what
        # the browser filters on, so naming it here is what makes these draws
        # findable rather than scattered through A0.
        nv = named_view(r)
        if is_clean(r):
            outcome = f"false alarm \u2014 named {nv}" if nv else "correctly said agree"
        rows.append({
            "model": str(r.get("model", "")),
            "cell": f"{r.get('condition', '')} / {outcome}",
            "title": str(r.get("item_id", "")),
            "gt_sample": str(r.get("gt_sample", "")),
            "mod_type": f"think={r.get('thinking', 'na')} draw {r.get('sample_idx', '')}",
            "source": f"slots={r.get('slots', '')}",
            "gt": {"corrupted view": str(r.get("corrupted_view", "")),
                   "outlier slot": str(r.get("outlier_slot", "")),
                   "pde class": str(r.get("gt_pde_class", "") or ""),
                   "method": str(r.get("gt_num_method", "") or "")},
            "parsed": {"agree": str(r.get("agree", "") or ""),
                       "outlier": str(r.get("outlier", "") or ""),
                       "pde class": str(r.get("system_pde_class", "") or ""),
                       "method": str(r.get("system_num_method", "") or "")},
            "scores": {k: (None if pd.isna(r.get(k)) else r.get(k))
                       for k in ("detection_correct", "localization_correct",
                                 "pde_class_match", "num_method_match") if k in r},
            "axis": str(r.get("parse_route", "") or ""),
            "conf": str(r.get("traj_level", "") or ""),
            "finish": str(r.get("finish_reason", "") or ""),
            "justification": "" if pd.isna(r.get("justification")) else str(
                r.get("justification") or ""),
            "chars": len(str(r.get("response", "") or "")),
            "text": str(r.get("response", "") or ""),
        })

    section = (
        '<section id="raw-responses">'
        '<h2>Raw responses and justifications</h2>'
        '<p class="sub" style="margin-bottom:16px">Each row shows what the model '
        'answered against what the item actually was: which view was corrupted, '
        'which slot held it, and the model&rsquo;s own account of what it thought '
        'was inconsistent. The justification is where a right answer for the wrong '
        'reason becomes visible &mdash; a model can name the correct outlier while '
        'explaining it by formatting rather than physics. The sample is stratified '
        'over model, corrupted view and outcome, so every model appears with its '
        'misses next to its hits rather than whichever rows happened to load first. '
        f'{len(rows):,} draws are loaded here out of {len(raw):,}; the full text of '
        'each is verbatim and unabridged.</p>'
        + _false_alarm_block(raw)
        + response_browser(rows, "sbatch/run_cross_modal_consistency.sbatch",
                           prefix="xm")
        + '</section>')

    doc = pathlib.Path(path).read_text(encoding="utf-8")
    doc = doc.replace("</style>", BROWSER_CSS + "</style>", 1)
    # Anchor on </body>, NOT on the last </div>. The report's drill-down drawer is
    # built by a <script> that emits markup from a JS template literal, so the final
    # </div> in the file sits INSIDE that script -- injecting there put the whole
    # panel in dead JavaScript, where it rendered as nothing and corrupted the
    # drawer's template into the bargain. </body> is the only anchor guaranteed to
    # be outside every script on the page.
    if "</body>" not in doc:
        raise SystemExit("no </body> in the report; refusing to guess an anchor")
    # Its own .wrap because it lands after the page's, and would otherwise sit at
    # full bleed instead of in the text column.
    doc = doc.replace(
        "</body>",
        '<div class="wrap" style="padding-top:0">' + section + "</div></body>", 1)
    pathlib.Path(path).write_text(doc, encoding="utf-8")
    return len(rows)


def main():
    ap = argparse.ArgumentParser(description="Build the expanded consistency report")
    ap.add_argument("--partial", action="store_true",
                    help="include models whose run has not finished all 1024 items")
    ap.add_argument("--out", default=OUT)
    args = ap.parse_args()

    check_roster_matches_registry()
    print(f"[expanded] loading {SOURCE_REPO}")
    rows = survey(load_raw(), args.partial)

    print(f"\n[expanded] {'model':<22} {'items':>6} {'draws':>7} {'no-verdict':>11}  status")
    for r in rows:
        print(f"[expanded] {r['short']:<22} {r['items']:>6} {r['draws']:>7} "
              f"{r['dropped']:>11}  {r['state']}"
              + ("" if r["used"] or r["state"] == "queued" else "  [excluded]"))

    used = [r for r in rows if r["used"]]
    if not used:
        raise SystemExit("no complete models yet; re-run with --partial to see progress")

    raw_all = pd.concat([r["df"] for r in used], ignore_index=True)
    d = from_xmodal(raw_all)
    # k=3 means item_id|model|thinking is not unique. Without the draw index the
    # drill-down would show three different answers under one run id.
    if "sample_idx" in raw_all:
        d["run_id"] = d["run_id"] + "|s" + raw_all["sample_idx"].astype(str).values

    items = pd.read_csv("data/multimodal_items_v1.csv").drop_duplicates("gt_sample")
    defects = dict(zip(items["gt_sample"].astype(str),
                       items["invalidity_note"].astype(str)))

    print(f"\n[expanded] building on {len(d):,} draws from {len(used)} model(s)")
    # annotate=True is opt-in here only: the shared figure helper defaults to the
    # unlabelled form so viz/build_cross_modal_claims_frozen.sh keeps reproducing the published report
    # byte for byte.
    # The same frame WITH the no-verdict draws, for the unconditional blame figure
    # only. Every other figure keeps dropping them -- scoring a run that produced no
    # verdict as if it had produced a wrong one is worse than excluding it. But the
    # unconditional figure's whole claim is "per opportunity", and a truncated draw
    # is a consumed opportunity, so there it gets its own segment instead.
    parts, masks = [], []
    for r in rows:
        if r["used"] and r.get("df_all") is not None:
            parts.append(r["df_all"])
            masks.append(np.asarray(r["good_mask"], dtype=bool))
    d_all = None
    if parts:
        d_all = from_xmodal(pd.concat(parts, ignore_index=True))
        # from_xmodal builds its output positionally and neither filters nor
        # reorders, so a positional mask is safe; assert the length anyway.
        good_all = np.concatenate(masks)
        assert len(good_all) == len(d_all), (len(good_all), len(d_all))
        d_all["no_verdict"] = ~good_all
        print(f"[expanded] unconditional figure: {len(d_all):,} draws "
              f"({int((~good_all).sum()):,} without a verdict)")
    CR.build(d, out=args.out, defects=defects, annotate=True,
             blame_unconditional=True, d_all=d_all, theme="light", verbose_labels=True)
    inject_provenance(args.out, rows)
    lad = inject_ladder(args.out, d, raw_all)
    if lad.get("error"):
        print(f"[expanded] ladder panel SKIPPED: {lad['error']}")
    else:
        print(f"[expanded] ladder: {lad['n_common']:,} common items "
              f"({lad['frac_of_smallest']:.0%} of smallest)")
        print(f"[expanded] {lad['verdict']}")
    tr = inject_tradeoff(args.out, d, raw_all)
    if tr.get("error"):
        print(f"[expanded] trade-off section SKIPPED: {tr['error']}")
    else:
        print(f"[expanded] {tr['verdict']}")
    dsg = inject_design(args.out, raw_all)
    if dsg.get("error"):
        print(f"[expanded] design appendix SKIPPED: {dsg['error']}")
    else:
        print(f"[expanded] design: {dsg['n_items']:,} items x {dsg['draws']} draws, "
              f"crossed={dsg['crossed']} slot-balanced={dsg['balanced']}")
    # "Are newer models better at this?" removed from the report on 2026-08-24 at the
    # researcher's request. inject_newer_better() and viz/consistency/newer_better.py
    # are left in place and still work; restoring the section is uncommenting this
    # block. Nothing else in the build depends on it -- the trend statistics are
    # computed inside that call and used nowhere else.
    pm = inject_per_model(args.out, d, rows)
    if pm.get("error"):
        print(f"[expanded] per-model appendix SKIPPED: {pm['error']}")
    else:
        print(f"[expanded] per-model appendix: {pm['n_models']} checkpoints")
    q4s = inject_q4_stability(args.out, d, raw_all, rows)
    if q4s.get("error"):
        print(f"[expanded] Q4 stability block SKIPPED: {q4s['error']}")
    else:
        print(f"[expanded] Q4 stability: {q4s['pairs']:,} pairs, "
              f"{q4s['dropped']:,} dropped")
        print(f"[expanded] {q4s['churn']}")
        print(f"[expanded] {q4s['verdict']}")
    n_rb = inject_raw_browser(args.out, raw_all)
    print(f"[expanded] raw-response browser: {n_rb} draws loaded")
    print(f"[expanded] wrote {args.out}")


if __name__ == "__main__":
    main()
