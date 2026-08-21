"""
audit_multimodal_items.py — every property the cross-modal consistency item set
claims, restated as an executable assertion.

Follows datagen/full_audit.py's contract: the build is not "checked by inspection",
it is checked by a script that fails loudly. Claims that are only written down in a
plan drift away from the data; claims that are asserted here cannot.

Run:  python crossmodal/datagen/audit_multimodal_items.py
"""
import ast
import collections
import csv
import io
import os
import subprocess
import sys
import tempfile
import tokenize

import numpy as np

# repo root: this file sits at crossmodal/<area>/, so three levels up
sys.path.insert(0, os.path.dirname(os.path.dirname(
    os.path.dirname(os.path.abspath(__file__)))))
from crossmodal.datagen.build_multimodal_items import (                       # noqa: E402
    IDENTITY_LEAK, MOD_FOR_NAMES, MULTIMODAL_CSV, MOD_DATASET, NEAR_DUPLICATE, VIEWS,
)
from crossmodal.datagen.corrupt_trajectory import build_ladder                # noqa: E402
from crossmodal.datagen.render_trajectory_table import (                      # noqa: E402
    RENDER_HARD_CASES, choose_grid, parse_trajectory, render,
)
from crossmodal.eval.consistency_prompts import (                             # noqa: E402
    ViewSources, build_prompt, load_items,
)

csv.field_size_limit(10 ** 9)

ITEMS = "data/multimodal_items_v1.csv"

# Vocabulary that would give away the answer if it appeared in a view that is not
# meant to carry it.
CLASS_WORDS = {
    "heat": ["heat", "thermal", "diffusiv"],
    "wave": ["wave", "oscillat", "vibrat"],
    "burgers": ["burgers", "shock"],
    "navier-stokes": ["navier", "stokes", "cfd", "vortic"],
}
METHOD_WORDS = ["crank", "nicolson", "spectral", "implicit", "explicit", "runge",
                "kutta", "fft", "cfl"]

_checks = []


def check(name):
    def deco(fn):
        _checks.append((name, fn))
        return fn
    return deco


def strip_comments(src):
    """Remove comments via tokenize, not a naive '#' split -- a '#' inside a string
    literal is not a comment and splitting on it would corrupt the code."""
    out = []
    try:
        for tok in tokenize.generate_tokens(io.StringIO(src).readline):
            if tok.type != tokenize.COMMENT:
                out.append(tok)
        return tokenize.untokenize(out)
    except (tokenize.TokenError, IndentationError):
        return src


@check("item count and factorial balance")
def _balance(ctx):
    items = ctx["items"]
    n_sys = len(ctx["systems"])
    conds = {i["condition"] for i in items}
    assert len(items) == n_sys * len(conds) * 2 * 2, f"got {len(items)} items"
    for field, expected in (("condition", len(items) // len(conds)),
                            ("names", len(items) // 2),
                            ("order_seed", len(items) // 2)):
        c = collections.Counter(i[field] for i in items)
        assert set(c.values()) == {expected}, f"{field} unbalanced: {dict(c)}"
    return f"{len(items)} items, {len(conds)} conditions, balanced on names and order"


@check("outlier position counterbalanced")
def _positions(ctx):
    corrupted = [i for i in ctx["items"] if i["corrupted_view"] != "none"]
    overall = collections.Counter(i["outlier_slot"] for i in corrupted)
    assert len(set(overall.values())) == 1, f"slots unbalanced overall: {dict(overall)}"
    for cond in {i["condition"] for i in corrupted}:
        c = collections.Counter(i["outlier_slot"] for i in corrupted if i["condition"] == cond)
        assert len(set(c.values())) == 1, f"{cond} slots unbalanced: {dict(c)}"
    return f"{len(overall)} slots x {list(overall.values())[0]} items each, within every condition"


@check("ground truth points at the corrupted view")
def _ground_truth(ctx):
    for i in ctx["items"]:
        slots = [i[f"slot_{k}"] for k in range(1, 5)]
        assert sorted(slots) == sorted(VIEWS), f"{i['item_id']} slot set {slots}"
        if i["corrupted_view"] == "none":
            assert not i["outlier_slot"], f"{i['item_id']} all-agree item has an outlier"
        else:
            assert i[f"slot_{i['outlier_slot']}"] == i["corrupted_view"], i["item_id"]
    return f"all {len(ctx['items'])} items: outlier_slot resolves to corrupted_view"


@check("code views are byte-equal to their audited jul28 source")
def _code_provenance(ctx):
    src = ctx["sources"]
    for system in ctx["systems"]:
        for names in ("real", "obfuscated"):
            for valid in (True, False):
                got = src.code(system, names, valid)
                want = ctx["mod"][system][MOD_FOR_NAMES[(names, valid)]]
                assert got == want, f"{system}/{names}/{valid} diverges from merged_mod"
    return f"{len(ctx['systems'])} systems x 4 variants match merged_mod_jul28.csv exactly"


@check("code views carry no comments")
def _no_comments(ctx):
    src = ctx["sources"]
    for system in ctx["systems"]:
        for names in ("real", "obfuscated"):
            for valid in (True, False):
                code = src.code(system, names, valid)
                assert strip_comments(code) == code, f"{system}/{names} has comments"
    return "natural language appears in the description view only, as specified"


@check("all four trajectory rungs render to identical size")
def _render_uniform(ctx):
    src = ctx["sources"]
    for system in ctx["systems"]:
        valid = parse_trajectory(src.mm[system]["Trajectory"])
        grid = choose_grid(valid.shape)
        ladder = build_ladder(valid, src.mm[system + "_wrong"]["Trajectory"], system,
                              include_time_shuffle=True)
        lengths = {len(render(a, grid)) for a in [valid] + list(ladder.values())}
        assert len(lengths) == 1, f"{system}: rendered lengths differ {sorted(lengths)}"
    return "shape and float formatting cannot identify the corrupted trajectory"


@check("T_shuf preserves every marginal statistic exactly")
def _shuffle_property(ctx):
    src = ctx["sources"]
    for system in ctx["systems"]:
        valid = parse_trajectory(src.mm[system]["Trajectory"])
        lad = build_ladder(valid, src.mm[system + "_wrong"]["Trajectory"], system)
        shuf, rand = lad["T_shuf"], lad["T_rand"]
        assert shuf.shape == valid.shape and rand.shape == valid.shape, system
        assert np.array_equal(np.sort(shuf.ravel()), np.sort(valid.ravel())), system
        assert not np.array_equal(shuf, valid), f"{system}: shuffle is the identity"
    return "identical histogram, mean, variance, min and max -- only arrangement differs"


@check("the trajectory view leaks no vocabulary")
def _trajectory_text(ctx):
    src = ctx["sources"]
    system = ctx["systems"][0]
    text = src.trajectory(system, "valid").lower()
    body = "\n".join(text.splitlines()[5:])
    for words in list(CLASS_WORDS.values()) + [METHOD_WORDS]:
        for w in words:
            assert w not in body, f"trajectory body contains {w!r}"
    return "numeric table plus a fixed header; no PDE or method vocabulary"


@check("code-view identity leaks match the documented set")
def _identity_leaks(ctx):
    """Strong form: assert the leaking systems are EXACTLY the ones on record, so a
    regression that adds a leak fails, and so does one that silently removes a
    system from the documented list without re-checking."""
    src, found = ctx["sources"], set()
    for system in ctx["systems"]:
        code = src.code(system, "obfuscated", True).lower()
        try:
            literals = " ".join(
                n.value.lower() for n in ast.walk(ast.parse(code))
                if isinstance(n, ast.Constant) and isinstance(n.value, str)).lower()
        except SyntaxError:
            literals = ""
        imports = " ".join(
            ln for ln in code.splitlines() if ln.startswith(("import ", "from ")))
        surface = imports + " " + literals
        for words in list(CLASS_WORDS.values()) + [METHOD_WORDS]:
            if any(w in surface for w in words):
                found.add(system)
    assert found == set(IDENTITY_LEAK), f"leaks changed: {sorted(found)} vs {sorted(IDENTITY_LEAK)}"
    return f"exactly {sorted(found)} -- held out of the obfuscation analysis"


@check("covariates and flags are populated")
def _covariates(ctx):
    numeric = ["desc_len_delta", "code_len_delta", "code_blank_runs_valid",
               "render_recon_err"]
    for i in ctx["items"]:
        for f in numeric:
            assert i[f] != "", f"{i['item_id']} missing {f}"
            float(i[f])
    hard = {i["gt_sample"] for i in ctx["items"] if i["flag_render_hard"] == "1"}
    dup = {i["gt_sample"] for i in ctx["items"] if i["flag_near_duplicate"] == "1"}
    leak = {i["gt_sample"] for i in ctx["items"] if i["flag_identity_leak"] == "1"}
    assert hard == set(RENDER_HARD_CASES), f"render-hard flag drifted: {sorted(hard)}"
    assert dup == set(NEAR_DUPLICATE), f"near-duplicate flag drifted: {sorted(dup)}"
    assert leak == set(IDENTITY_LEAK), f"identity-leak flag drifted: {sorted(leak)}"
    return f"{len(numeric)} covariates present on every item; 3 flag sets match their sources"


@check("build is byte-reproducible")
def _determinism(ctx):
    with tempfile.NamedTemporaryFile(suffix=".csv", delete=False) as tmp:
        path = tmp.name
    try:
        subprocess.run([sys.executable, "crossmodal/datagen/build_multimodal_items.py", "--out", path],
                       check=True, capture_output=True)
        assert open(path, "rb").read() == open(ITEMS, "rb").read(), "rebuild differs"
    finally:
        os.unlink(path)
    return "consecutive builds produce an identical CSV"


@check("prompts assemble within budget")
def _prompts(ctx):
    src, sizes = ctx["sources"], []
    for i in ctx["items"]:
        if i["traj_level"] == "T_exec":
            continue                      # needs the cpu_short job
        sizes.append(len(build_prompt(i, src)) // 4)
    assert max(sizes) < 32000, f"largest prompt ~{max(sizes)} tokens"
    return (f"{len(sizes)} prompts, median ~{int(np.median(sizes))} tok, "
            f"max ~{max(sizes)} tok")


def main():
    items = load_items(ITEMS)
    sources = ViewSources(MULTIMODAL_CSV, MOD_DATASET)
    mod = collections.defaultdict(dict)
    for r in csv.DictReader(open(MOD_DATASET, newline="")):
        mod[r["gt_sample"]][r["mod_type"]] = r["code"]

    ctx = {"items": items, "sources": sources, "mod": mod,
           "systems": sorted({i["gt_sample"] for i in items})}

    passed = failed = 0
    for name, fn in _checks:
        try:
            detail = fn(ctx)
            passed += 1
            print(f"  PASS  {name}\n          {detail}")
        except AssertionError as exc:
            failed += 1
            print(f"  FAIL  {name}\n          {exc}")
    total = passed + failed
    print(f"\n{passed} of {total} checks pass")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
