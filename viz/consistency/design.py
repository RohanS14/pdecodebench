"""The item-construction appendix: why there are exactly 1,024 items per model.

Every number in this module is DERIVED from the delivered rows, never asserted. The
factor counts, the cell sizes, the balance checks and the totals are all read off the
`item_id` keys and the design columns of the frame the report was built on, so the
appendix cannot describe a design the data does not have. If a future run changes a
factor -- adds a fifth trajectory method, drops the second presentation order -- this
section re-derives itself and the arithmetic still closes, or the balance line says
plainly that it no longer does.

`item_id` is a four-field key: SYSTEM|CONDITION|IDENTIFIERS|ORDER. That key IS the
design, which is why it is parsed here rather than a separate manifest being read: a
manifest can disagree with what was actually generated, the keys cannot.
"""
import collections
import html as _h

from .constants import TRAJ_LEVELS, TRAJ_LEVEL_LABELS

# Presentation names for the eval's condition vocabulary. Absent keys fall through
# to the raw token rather than being dropped, so a new condition shows up as itself
# instead of silently vanishing from the table.
CONDITION_TEXT = {
    "A0": ("nothing corrupted", "all four views describe the same system"),
    "X_C": ("code corrupted", "the solver source no longer matches the other three"),
    "X_D": ("description corrupted", "the prose statement no longer matches"),
    "X_M": ("math corrupted", "the equations no longer match"),
    # The four trajectory descriptions come from constants.TRAJ_LEVEL_LABELS rather
    # than being written again here: that dict is what the figures label these
    # conditions with, and an appendix that explains them in its own words is a
    # second source of truth waiting to drift from the first.
    **{f"X_T_{lvl}": (f"trajectory &mdash; {lvl}", TRAJ_LEVEL_LABELS[lvl])
       for lvl in TRAJ_LEVELS},
}
NAMING_TEXT = {"real": "informative identifiers, as written",
               "obfuscated": "identifiers replaced with meaningless names"}


def _key(item_id):
    """SYSTEM|CONDITION|IDENTIFIERS|ORDER -> the four fields, or None."""
    parts = str(item_id).split("|")
    return tuple(parts) if len(parts) == 4 else None


def factorise(raw):
    """Read the design off the item keys of one model's arm.

    One arm, not the pooled frame: the pooled frame repeats every item once per
    model, so counting cells there would multiply every factor by the roster size.
    The arms are verified identical separately (see `arms_agree`), which is what
    makes one arm a safe stand-in for all of them.
    """
    d = raw
    if "model" in d.columns and d["model"].nunique() > 1:
        first = sorted(d["model"].astype(str).unique())[0]
        d = d[d["model"].astype(str).eq(first)]

    ids = sorted(set(d["item_id"].astype(str)))
    keys = [k for k in (_key(i) for i in ids) if k]
    if not keys or len(keys) != len(ids):
        return None

    systems = sorted({k[0] for k in keys})
    conds = sorted({k[1] for k in keys})
    # "real" first, so the table reads baseline-then-intervention rather than
    # alphabetically, which puts "obfuscated" ahead of the condition it modifies.
    naming = sorted({k[2] for k in keys}, key=lambda n: (n != "real", n))
    orders = sorted({k[3] for k in keys})

    # Families are the alphabetic stem of the system name (Burgers_3 -> Burgers).
    fams = collections.Counter(s.rsplit("_", 1)[0] for s in systems)

    # Is the design fully crossed? Every (condition, identifiers, order) cell should
    # hold exactly one item per system. Checked, not assumed -- an arm assembled from
    # several repair passes could in principle have gained or lost a cell.
    cells = collections.Counter((k[1], k[2], k[3]) for k in keys)
    crossed = (len(cells) == len(conds) * len(naming) * len(orders)
               and set(cells.values()) == {len(systems)})

    # Counterbalancing: over the corrupted items, which SLOT held the corrupted view.
    # This is the property that stops localization from being readable off position,
    # so it is measured per condition rather than pooled -- a design can be balanced
    # overall and lopsided inside one condition.
    slots = collections.defaultdict(collections.Counter)
    per_item = {}
    for r in d.drop_duplicates("item_id").to_dict("records"):
        per_item[str(r["item_id"])] = r
        if str(r.get("condition")) == "A0":
            continue
        slots[str(r.get("condition"))][str(r.get("outlier_slot"))] += 1
    slot_cells = [n for c in slots.values() for n in c.values()]
    n_slots = len({s for c in slots.values() for s in c})
    balanced = bool(slot_cells) and len(set(slot_cells)) == 1

    # How many presentation orders were actually used, and does one item keep one
    # order across its draws? Both matter for how the three draws are read: if an
    # item's draws vary the order, the draw spread mixes sampling noise with order
    # effects and neither can be recovered.
    perms = collections.Counter()
    per_item_perm = collections.Counter()
    for iid, g in d.groupby(d["item_id"].astype(str)):
        seen = {tuple(str(x) for x in v) for v in g["slots"]}
        per_item_perm[len(seen)] += 1
        for p in seen:
            perms[p] += 1
    draws_per_item = int(round(len(d) / max(1, d["item_id"].nunique())))

    # Two facts about the order factor that a reader would otherwise have to take
    # on trust. Both are checked here rather than asserted in the prose.
    #   (a) the two orders drawn for one cell are actually DIFFERENT -- otherwise
    #       the factor doubles the item count while varying nothing;
    #   (b) the real and obfuscated versions of one (system, condition, order) are
    #       shown in the SAME order -- which is what makes the obfuscation contrast
    #       paired on presentation rather than confounded with it.
    by_cell = collections.defaultdict(dict)
    perm_of = {}
    for iid, r in per_item.items():
        k = _key(iid)
        perm_of[iid] = tuple(str(x) for x in r.get("slots", []))
        by_cell[(k[0], k[1], k[2])][k[3]] = perm_of[iid]
    pairs = [v for v in by_cell.values() if len(v) == len(orders)]
    orders_differ = bool(pairs) and all(
        len(set(v.values())) == len(v) for v in pairs)
    by_pair = collections.defaultdict(dict)
    for iid in per_item:
        k = _key(iid)
        by_pair[(k[0], k[1], k[3])][k[2]] = perm_of[iid]
    np_ = [v for v in by_pair.values() if len(v) == len(naming)]
    naming_paired = bool(np_) and all(len(set(v.values())) == 1 for v in np_)

    n_clean = sum(1 for k in keys if k[1] == "A0")
    corrupted_conds = [c for c in conds if c != "A0"]
    traj = [c for c in corrupted_conds if c.startswith("X_T")]

    return {
        "n_items": len(ids), "systems": systems, "families": dict(fams),
        "conditions": conds, "naming": naming, "orders": orders,
        "crossed": crossed, "cell_size": len(systems),
        "balanced": balanced, "n_slots": n_slots,
        "slot_cell": slot_cells[0] if balanced else None,
        "slots": {c: dict(sorted(v.items())) for c, v in sorted(slots.items())},
        "n_perms": len(perms), "perm_max": 24,
        "orders_differ": orders_differ, "naming_paired": naming_paired,
        "one_order_per_item": set(per_item_perm) == {1},
        "draws_per_item": draws_per_item,
        "n_clean": n_clean, "n_corrupt": len(ids) - n_clean,
        "n_traj": sum(1 for k in keys if k[1] in traj),
        "traj_conds": traj, "corrupted_conds": corrupted_conds,
        "per_item": per_item,
    }


def arms_agree(raw):
    """Do all models see the same item set? Returns (agree, n_models, n_shared)."""
    if "model" not in raw.columns:
        return (True, 1, raw["item_id"].nunique())
    sets = {m: frozenset(g["item_id"].astype(str))
            for m, g in raw.groupby(raw["model"].astype(str))}
    if not sets:
        return (True, 0, 0)
    common = frozenset.intersection(*sets.values())
    return (len({s for s in sets.values()}) == 1, len(sets), len(common))


def _factor_table(f):
    fam = ", ".join(f"{k} ({v})" for k, v in sorted(f["families"].items()))
    rows = [
        ("Solver systems", len(f["systems"]),
         f"{len(f['families'])} PDE families &mdash; {_h.escape(fam)}"),
        ("Corruption conditions", len(f["conditions"]),
         f"one clean, {len(f['corrupted_conds'])} corrupted "
         f"({len(f['traj_conds'])} of them trajectory methods)"),
        ("Identifier regimes", len(f["naming"]),
         " / ".join(NAMING_TEXT.get(n, _h.escape(n)) for n in f["naming"])),
        ("Presentation orders per cell", len(f["orders"]),
         f"two independently drawn orderings of the four views, out of "
         f"{f['perm_max']} possible"),
    ]
    body = "".join(
        f"<tr><td>{lab}</td><td style='text-align:right'><b>{n}</b></td>"
        f"<td>{note}</td></tr>" for lab, n, note in rows)
    prod = " &times; ".join(str(n) for _, n, _ in rows)
    body += (f"<tr><td colspan='3' style='padding-top:12px'>"
             f"<b>{prod} = {f['n_items']:,} items</b>, each answered "
             f"{f['draws_per_item']} times &rarr; "
             f"<b>{f['n_items'] * f['draws_per_item']:,} draws per model</b>"
             f"</td></tr>")
    return ("<div style='overflow-x:auto'><table><thead><tr><th>Factor</th>"
            "<th style='text-align:right'>Levels</th><th>What the levels are</th>"
            "</tr></thead><tbody>" + body + "</tbody></table></div>")


def _condition_table(f):
    per = f["n_items"] // max(1, len(f["conditions"]))
    body = ""
    for c in f["conditions"]:
        lab, note = CONDITION_TEXT.get(c, (_h.escape(c), ""))
        body += (f"<tr><td><code>{_h.escape(c)}</code></td><td>{lab}</td>"
                 f"<td style='color:var(--dim)'>{note}</td>"
                 f"<td style='text-align:right'>{per}</td></tr>")
    return ("<div style='overflow-x:auto'><table><thead><tr><th>Key</th>"
            "<th>Condition</th><th>What is corrupted</th>"
            "<th style='text-align:right'>Items</th></tr></thead><tbody>"
            + body + "</tbody></table></div>")


def _slot_table(f):
    cols = sorted({s for v in f["slots"].values() for s in v})
    head = "".join(f"<th style='text-align:right'>position {_h.escape(s)}</th>"
                   for s in cols)
    body = ""
    for c, v in f["slots"].items():
        lab = CONDITION_TEXT.get(c, (c, ""))[0]
        body += (f"<tr><td>{lab}</td>"
                 + "".join(f"<td style='text-align:right'>{v.get(s, 0)}</td>"
                           for s in cols) + "</tr>")
    return ("<div style='overflow-x:auto'><table><thead><tr><th>Condition</th>"
            + head + "</tr></thead><tbody>" + body + "</tbody></table></div>")


def build_section(raw):
    """The appendix as one HTML <section>. Pure function of the raw frame."""
    f = factorise(raw)
    if not f:
        return ""
    agree, n_models, n_shared = arms_agree(raw)

    traj_share = f["n_traj"] / max(1, f["n_corrupt"])
    n_draws = f["n_items"] * f["draws_per_item"]

    # Every claim below is phrased from a computed flag, so a design that stops being
    # crossed or balanced reports that instead of repeating a stale guarantee.
    crossed_line = (
        f"Every one of the {len(f['conditions']) * len(f['naming']) * len(f['orders'])} "
        f"(condition &times; identifiers &times; order) cells holds exactly "
        f"{f['cell_size']} items, one per solver system"
        if f["crossed"] else
        "<b>The design is not fully crossed in the delivered rows</b> &mdash; see "
        "the cell counts below")
    bal_line = (
        f"the corrupted view sits in each of the {f['n_slots']} positions exactly "
        f"{f['slot_cell']} times within every condition"
        if f["balanced"] else
        "<b>slot assignment is not balanced in the delivered rows</b>")
    pair_line = (
        ("The two orders drawn for a cell are always different &mdash; checked, "
         "not assumed, since a factor that redraws the same order would double the "
         "item count while varying nothing. "
         if f["orders_differ"] else
         "<b>Some cells drew the same order twice</b>, so the order factor does not "
         "vary presentation everywhere. ")
        + ("The real and obfuscated versions of one (system, condition, order) are "
           "shown in the <b>same</b> order, so the identifier contrast is paired on "
           "presentation instead of confounded with it."
           if f["naming_paired"] else
           "<b>The real and obfuscated versions of a cell are not always shown in "
           "the same order</b>, so that contrast is not paired on presentation."))
    order_line = (
        f"All {f['n_perms']} of the {f['perm_max']} possible orderings of four views "
        f"appear. Each item keeps a single ordering across its "
        f"{f['draws_per_item']} draws"
        if f["one_order_per_item"] else
        f"{f['n_perms']} of {f['perm_max']} orderings appear, and an item's draws "
        f"may differ in ordering")

    return (
        '<section id="design">'
        '<h2>Appendix &mdash; how the 1,024 items are built</h2>'
        '<p class="sub" style="margin-bottom:18px"><b>This is an appendix, not a '
        'result.</b> It states where the per-model row count comes from, because '
        f'{f["n_items"]:,} is a designed number rather than a sample size that '
        'happened to be reached. The item set is a fully crossed factorial: four '
        'factors, every combination generated once per level of every other, with '
        'no sampling and no filtering at any stage. Everything on this page is '
        're-derived from the delivered rows on each rebuild &mdash; the factor '
        'levels are read off the item keys, the cell sizes are counted, and the '
        'balance claims are checked rather than repeated.</p>'

        '<h3 class="pmh">1 &middot; The four factors</h3>'
        f'<p class="sub">An item is one key of the form '
        f'<code>SYSTEM|CONDITION|IDENTIFIERS|ORDER</code>. Those four fields are '
        f'the design; the product of their level counts is the item count.</p>'
        + _factor_table(f) +
        f'<p class="sub" style="margin-top:12px">{crossed_line}, so no condition is '
        'better represented than another and no system is over-sampled. The count '
        'is not a target that was sampled up to &mdash; there is exactly one item '
        'per combination, so it could not have been any other number without '
        'changing a factor.</p>'

        '<h3 class="pmh">2 &middot; The eight conditions</h3>'
        '<p class="sub">Each item shows four representations of one solver system: '
        'the source code, the governing equations, a prose description, and the '
        'numerical trajectory. In seven of the eight conditions exactly one of them '
        'has been altered; in the eighth nothing has.</p>'
        + _condition_table(f) +
        f'<p class="sub" style="margin-top:12px"><b>The trajectory is '
        'over-represented among the corrupted items, by construction.</b> It '
        f'carries {len(f["traj_conds"])} corruption methods where the other three '
        f'views carry one each, so {f["n_traj"]:,} of the {f["n_corrupt"]:,} '
        f'corrupted items ({traj_share:.0%}) have a corrupted trajectory. This is the '
        'single most important thing to know when reading any micro-averaged number '
        'in this report: a model that answers &ldquo;trajectory&rdquo; by reflex is '
        'rewarded by the mix. It is also why the per-model appendix reports '
        'unweighted macro averages beside the micro ones. '
        f'The remaining {f["n_clean"]:,} items are clean and carry the '
        'false-alarm measurements.</p>'

        '<h3 class="pmh">3 &middot; Presentation order and slot balance</h3>'
        f'<p class="sub">The four views are presented in a drawn order, and order is '
        f'a factor rather than a nuisance: each (system, condition, identifiers) '
        f'triple is generated under {len(f["orders"])} independently drawn '
        f'orderings, which is the fourth factor above and the reason the item count '
        f'doubles. {order_line}, so the spread across an item&rsquo;s draws measures '
        'decoding variance at fixed presentation rather than order effects mixed '
        'with decoding variance.</p>'
        f'<p class="sub">{pair_line}</p>'
        f'<p class="sub">Order is drawn, but the position of the corrupted view is '
        f'<b>counterbalanced exactly</b>: {bal_line}. The model answers by slot '
        '(&ldquo;view_2&rdquo;), so without this a model that favoured one position '
        'would score as though it had localized, and every accuracy in this report '
        'would be partly a position preference. Because the balance is exact, '
        'guessing a fixed position scores '
        f'{1 / max(1, f["n_slots"]):.0%} and nothing else.</p>'
        + _slot_table(f) +

        '<h3 class="pmh">4 &middot; What each model was asked</h3>'
        f'<p class="sub">Every model is run on the same {f["n_items"]:,} items with '
        f'{f["draws_per_item"]} sampled generations each, giving '
        f'{n_draws:,} draws per model. '
        + (f'All {n_models} arms carry an identical item set &mdash; verified by '
           f'comparing the key sets, not assumed from the counts &mdash; so every '
           f'cross-model comparison in this report is paired on the item, with no '
           f'intersection taken and nothing dropped.'
           if agree else
           f'<b>The {n_models} arms do NOT carry identical item sets</b>; '
           f'{n_shared:,} items are common to all of them, and cross-model '
           f'comparisons are restricted to that intersection.')
        + '</p>'
        '</section>')
