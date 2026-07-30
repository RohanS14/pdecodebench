"""
full_audit.py — end-to-end integrity audit of the jul28 release.

Goes well beyond audit_dataset.py's balance checks. Every claim the dataset makes about
itself is restated here as an executable assertion, grouped so a failure names the design
property it breaks. Structure and semantics only; execution-dependent checks live in
full_audit_exec.py because they need the simulation dependencies installed.

Usage:
    python datagen/full_audit.py [path/to/merged_mod_jul28.csv]
Exit code 0 iff every check passes.
"""

import ast
import re
import sys
from collections import Counter, defaultdict

import pandas as pd

MOD_TYPES = ["Comm_Valid", "NoComm_Valid", "CorrComm", "NoComm_CorrVar",
             "Comm_InValid", "NoComm_InValid", "CorrComm_Invalid", "NoComm_CorrVar_InValid"]
VALID_MODS = ["Comm_Valid", "NoComm_Valid", "CorrComm", "NoComm_CorrVar"]
INVALID_MODS = ["Comm_InValid", "NoComm_InValid", "CorrComm_Invalid", "NoComm_CorrVar_InValid"]
NOCOMM_MODS = ["NoComm_Valid", "NoComm_InValid", "NoComm_CorrVar", "NoComm_CorrVar_InValid"]
CORRVAR_MODS = ["NoComm_CorrVar", "NoComm_CorrVar_InValid"]
PDE_CLASSES = ["wave", "heat", "burgers", "navier-stokes"]

FOOBAR = re.compile(r"^(foobar_\d+|fn\d+)$")
results = []


def check(group, name, ok, detail=""):
    results.append({"group": group, "name": name, "ok": bool(ok), "detail": detail})


def N(code):
    return str(code).replace("\\r\\n", "\n").replace("\\n", "\n").replace("\\t", "\t")


def comments(code):
    return [l.strip() for l in N(code).splitlines() if l.strip().startswith("#")]


def strip_comments(code):
    return "\n".join(l for l in N(code).splitlines() if not l.strip().startswith("#"))


def norm_ws(code):
    return "\n".join(l.rstrip() for l in N(code).splitlines() if l.strip())


def identifiers(code):
    """All identifiers, plus which are author-declared."""
    try:
        tree = ast.parse(N(code))
    except SyntaxError:
        return None, None
    alln, declared = set(), set()
    for n in ast.walk(tree):
        if isinstance(n, ast.Name):
            alln.add(n.id)
            if isinstance(n.ctx, (ast.Store, ast.Del)):
                declared.add(n.id)
        elif isinstance(n, ast.arg):
            alln.add(n.arg); declared.add(n.arg)
        elif isinstance(n, ast.keyword) and n.arg:
            alln.add(n.arg)
        elif isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            alln.add(n.name); declared.add(n.name)
    return alln, declared


def ast_shape(code):
    """AST with every identifier blanked — structural fingerprint for pure-rename checks."""
    t = ast.parse(N(code))
    t = ast.parse(ast.unparse(t))
    for n in ast.walk(t):
        for f in ("id", "name", "arg", "attr"):
            if hasattr(n, f) and isinstance(getattr(n, f), str):
                setattr(n, f, "_")
    return ast.dump(t)


def main(path="data/merged_mod_jul28.csv"):
    df = pd.read_csv(path)
    by = {(r.gt_sample, r.mod_type): r for _, r in df.iterrows()}
    gts = sorted(df.gt_sample.unique())

    # ---------------------------------------------------------------- A. structure
    g = "A. Structure"
    check(g, "256 rows", len(df) == 256, f"{len(df)}")
    check(g, "32 base problems", len(gts) == 32, f"{len(gts)}")
    check(g, "8 mod_types, 32 rows each",
          dict(df.mod_type.value_counts()) == {m: 32 for m in MOD_TYPES},
          str(dict(df.mod_type.value_counts())))
    check(g, "every gt_sample has all 8 mod_types",
          all(set(df[df.gt_sample == gt].mod_type) == set(MOD_TYPES) for gt in gts))
    check(g, "64 rows per pde_class",
          dict(df.pde_class.value_counts()) == {c: 64 for c in PDE_CLASSES},
          str(dict(df.pde_class.value_counts())))
    check(g, "128 valid / 128 invalid",
          int(df.phys_valid.sum()) == 128 and int((~df.phys_valid).sum()) == 128)
    check(g, "128 human / 128 synthetic", dict(df.source.value_counts()) == {"human": 128, "synthetic": 128},
          str(dict(df.source.value_counts())))
    check(g, "titles unique", df.title.nunique() == 256, f"{df.title.nunique()} distinct")
    check(g, "phys_valid matches mod_type",
          all(bool(r.phys_valid) == (r.mod_type in VALID_MODS) for _, r in df.iterrows()))
    check(g, "no empty code", not df.code.isna().any() and (df.code.astype(str).str.len() > 50).all())

    # metadata recomputation
    bad_lines = [r.title for _, r in df.iterrows() if r.num_lines != len(N(r.code).split("\n"))]
    check(g, "num_lines matches code", not bad_lines, f"{len(bad_lines)} wrong: {bad_lines[:4]}")
    bad_char = [r.title for _, r in df.iterrows() if r.num_char != len(N(r.code))]
    check(g, "num_char matches code", not bad_char, f"{len(bad_char)} wrong: {bad_char[:4]}")
    bad_com = [r.title for _, r in df.iterrows() if r.num_comments != len(comments(r.code))]
    check(g, "num_comments matches code", not bad_com, f"{len(bad_com)} wrong: {bad_com[:4]}")

    # column sparsity patterns
    check(g, "invalidity_note on all invalid, none valid",
          df[~df.phys_valid].invalidity_note.notna().all()
          and df[df.phys_valid].invalidity_note.isna().all())
    cc = df[df.mod_type.isin(["CorrComm", "CorrComm_Invalid"])]
    non_cc = df[~df.mod_type.isin(["CorrComm", "CorrComm_Invalid"])]
    check(g, "corruption_source_* only on CorrComm rows",
          cc.corruption_source_id.notna().all() and non_cc.corruption_source_id.isna().all())

    # ------------------------------------------------- B. metadata consistency
    g = "B. Metadata consistency"
    for col in ["pde_class", "phys_process", "num_method", "source"]:
        bad = [gt for gt in gts if df[df.gt_sample == gt][col].nunique() != 1]
        check(g, f"{col} constant across a gt_sample's 8 rows", not bad, f"varies for {bad}")
    bad = [gt for gt in gts if df[(df.gt_sample == gt) & (~df.phys_valid)].invalidity_note.nunique() != 1]
    check(g, "invalidity_note identical across a gt_sample's 4 invalid rows", not bad, f"varies for {bad}")
    check(g, "pde_class vocabulary is the 4 slugs", set(df.pde_class) == set(PDE_CLASSES),
          str(set(df.pde_class) - set(PDE_CLASSES)))
    ws = [v for v in set(df.num_method.dropna()) | set(df.phys_process.dropna()) if v != str(v).strip()]
    check(g, "no leading/trailing whitespace in num_method/phys_process", not ws, str(ws))
    # order normalisation
    sets = defaultdict(set)
    for v in set(df.num_method.dropna()):
        sets[frozenset(str(v).split("/"))].add(v)
    dupes = {tuple(sorted(k)): sorted(v) for k, v in sets.items() if len(v) > 1}
    check(g, "num_method is order-normalised", not dupes, f"same set spelled differently: {dupes}")
    check(g, "4 base problems per (class, source)",
          all(len(df[(df.pde_class == c) & (df.source == s) & (df.mod_type == "Comm_Valid")]) == 4
              for c in PDE_CLASSES for s in ["human", "synthetic"]))

    # ------------------------------------------------- C. condition semantics
    g = "C. Condition semantics"
    nc = df[df.mod_type.isin(NOCOMM_MODS)]
    check(g, "all NoComm_* rows are comment-free",
          all(len(comments(r.code)) == 0 for _, r in nc.iterrows()))
    cm = df[df.mod_type.isin(["Comm_Valid", "Comm_InValid", "CorrComm", "CorrComm_Invalid"])]
    check(g, "all commented rows have >=1 comment",
          all(len(comments(r.code)) >= 1 for _, r in cm.iterrows()))
    bad = [gt for gt in gts
           if comments(by[(gt, "Comm_Valid")].code) != comments(by[(gt, "Comm_InValid")].code)]
    check(g, "Comm_Valid and Comm_InValid carry identical comments", not bad, f"differ: {bad}")

    # comments are the ONLY difference between Comm_X and NoComm_X
    for cmod, nmod in [("Comm_Valid", "NoComm_Valid"), ("Comm_InValid", "NoComm_InValid")]:
        bad = [gt for gt in gts
               if norm_ws(strip_comments(by[(gt, cmod)].code)) != norm_ws(by[(gt, nmod)].code)]
        check(g, f"{cmod} minus comments == {nmod}", not bad, f"code differs: {bad}")

    # CorrComm preserves the receiver's code and comment count, only swapping comment text
    for corr, base in [("CorrComm", "Comm_Valid"), ("CorrComm_Invalid", "Comm_InValid")]:
        bad_code = [gt for gt in gts
                    if norm_ws(strip_comments(by[(gt, corr)].code))
                    != norm_ws(strip_comments(by[(gt, base)].code))]
        check(g, f"{corr} leaves {base}'s code untouched", not bad_code, f"code differs: {bad_code}")
        same_text = [gt for gt in gts
                     if comments(by[(gt, corr)].code) == comments(by[(gt, base)].code)]
        check(g, f"{corr} actually replaced the comments", not same_text,
              f"comments unchanged (no-op corruption): {same_text}")

    # CorrVar is a pure rename of its NoComm source
    for cv, base in [("NoComm_CorrVar", "NoComm_Valid"),
                     ("NoComm_CorrVar_InValid", "NoComm_InValid")]:
        bad = []
        for gt in gts:
            try:
                if ast_shape(by[(gt, cv)].code) != ast_shape(by[(gt, base)].code):
                    bad.append(gt)
            except SyntaxError:
                bad.append(f"{gt}(parse)")
        check(g, f"{cv} is a pure rename of {base}", not bad, f"structure differs: {bad}")

    # every author-declared identifier obfuscated
    leaks = {}
    for gt in gts:
        for cv in CORRVAR_MODS:
            alln, decl = identifiers(by[(gt, cv)].code)
            if alln is None:
                leaks[f"{gt}/{cv}"] = ["PARSE ERROR"]; continue
            surv = sorted(x for x in decl if not FOOBAR.match(x))
            if surv:
                leaks[f"{gt}/{cv}"] = surv
    check(g, "no author-declared identifier survives obfuscation", not leaks, str(leaks))

    # shared foobar mapping across validity
    bad = []
    for gt in gts:
        av, _ = identifiers(by[(gt, "NoComm_CorrVar")].code)
        ai, _ = identifiers(by[(gt, "NoComm_CorrVar_InValid")].code)
        if av is None or ai is None:
            continue
        fv = {x for x in av if FOOBAR.match(x)}
        fi = {x for x in ai if FOOBAR.match(x)}
        if not fv & fi:
            bad.append(gt)
    check(g, "CorrVar valid/invalid share their foobar mapping", not bad, f"disjoint for {bad}")

    # donor constraints
    g = "D. CorrComm donors"
    title_to_gt = {r.title: r.gt_sample for _, r in df[df.mod_type == "Comm_Valid"].iterrows()}
    same_class = cc[cc.corruption_source_pde == cc.pde_class]
    check(g, "donor never shares receiver's pde_class", len(same_class) == 0,
          f"{len(same_class)} rows")
    cross_src, same_method = [], []
    for _, r in cc.iterrows():
        dgt = title_to_gt.get(r.corruption_source_id)
        if dgt is None:
            cross_src.append((r.gt_sample, r.corruption_source_id)); continue
        drow = by[(dgt, "Comm_Valid")]
        if drow.source != r.source:
            cross_src.append((r.gt_sample, dgt))
        if set(str(drow.num_method).split("/")) == set(str(r.num_method).split("/")):
            same_method.append((r.gt_sample, dgt, r.num_method))
    check(g, "donor resolves to a real Comm_Valid row in the same source", not cross_src, str(cross_src))
    check(g, "donor's num_method set differs from receiver's", not same_method, str(same_method))
    pair = defaultdict(set)
    for _, r in cc.iterrows():
        pair[r.gt_sample].add(r.corruption_source_id)
    multi = {k: sorted(v) for k, v in pair.items() if len(v) != 1}
    check(g, "same donor serves a receiver's valid and invalid variant", not multi, str(multi))

    # ------------------------------------------------- E. label leakage
    g = "E. Label leakage"
    CLASS_WORDS = {"wave": [r"\bwave\b"], "heat": [r"\bheat\b", r"\bthermal\b"],
                   "burgers": [r"\bburgers?\b"],
                   "navier-stokes": [r"\bnavier\b", r"\bstokes\b"]}
    METHOD_WORDS = [r"\bexplicit\b", r"\bimplicit\b", r"\bspectral\b", r"\bcrank[- ]?nicolson\b",
                    r"\bfinite[- ]difference\b", r"\bupwind\b", r"\bnewmark\b", r"\blax[- ]"]

    def text_only(code):
        """String literals + comments — the parts that can talk about the problem."""
        out = []
        try:
            t = ast.parse(N(code))
            for n in ast.walk(t):
                if isinstance(n, ast.Constant) and isinstance(n.value, str):
                    out.append(n.value)
                if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef, ast.Module)):
                    d = ast.get_docstring(n)
                    if d:
                        out.append(d)
        except SyntaxError:
            pass
        return " ".join(out) + " " + " ".join(comments(code))

    # NoComm_* must not name their own class or method anywhere in code text
    own_class, own_method, docstrings = [], [], []
    for gt in gts:
        for mt in NOCOMM_MODS:
            r = by[(gt, mt)]
            body = N(r.code)
            for pat in CLASS_WORDS[r.pde_class]:
                if re.search(pat, body, re.I):
                    own_class.append((gt, mt, pat))
            for pat in METHOD_WORDS:
                w = pat.strip(r"\b").replace("[- ]?", "").replace("[- ]", "")
                if re.search(pat, body, re.I) and w.lower() in str(r.num_method).lower():
                    own_method.append((gt, mt, w))
            try:
                if ast.get_docstring(ast.parse(body)):
                    docstrings.append((gt, mt))
            except SyntaxError:
                pass
    check(g, "no NoComm_* row names its own pde_class", not own_class, str(own_class[:6]))
    check(g, "no NoComm_* row names its own numerical method", not own_method, str(own_method[:6]))
    check(g, "no NoComm_* row retains a module docstring", not docstrings, str(docstrings[:6]))

    # validity must not be inferable from text
    FAIL_WORDS = re.compile(r"blow.?up|blows? up|onset|nan|diverge|unstable|instability|"
                            r"explode|overflow|violent|wrong|incorrect|broken|bug|invalid", re.I)
    asym = []
    for gt in gts:
        for vmod, imod in zip(VALID_MODS, INVALID_MODS):
            tv, ti = text_only(by[(gt, vmod)].code), text_only(by[(gt, imod)].code)
            wv = {m.group(0).lower() for m in FAIL_WORDS.finditer(tv)}
            wi = {m.group(0).lower() for m in FAIL_WORDS.finditer(ti)}
            if wi - wv:
                asym.append((gt, imod, sorted(wi - wv)))
    check(g, "no failure vocabulary appears only on the invalid side", not asym, str(asym[:8]))

    # incidental (non-physics) differences between valid/invalid twins
    sys.path.insert(0, "datagen")
    try:
        from derive_invalidity_change import diff_changes
        inc = []
        for gt in gts:
            for c in diff_changes(by[(gt, "NoComm_Valid")].code, by[(gt, "NoComm_InValid")].code):
                if c["incidental"]:
                    inc.append((gt, c["category"], c["description"][:60]))
        check(g, "valid/invalid twins differ only in physics",
              not inc, f"{len(inc)} incidental diffs: {inc[:6]}")
    except ImportError as e:
        check(g, "valid/invalid twins differ only in physics", False, f"could not run: {e}")

    # surface conditions must not perturb the physics: all 4 valid conditions should carry
    # the same code modulo comments/names, likewise the 4 invalid ones
    g = "F. Cross-condition code identity"
    for label, mods, base in [("valid", VALID_MODS, "NoComm_Valid"),
                              ("invalid", INVALID_MODS, "NoComm_InValid")]:
        bad = []
        for gt in gts:
            ref = ast_shape(by[(gt, base)].code)
            for mt in mods:
                try:
                    if ast_shape(by[(gt, mt)].code) != ref:
                        bad.append((gt, mt))
                except SyntaxError:
                    bad.append((gt, mt, "parse"))
        check(g, f"all 4 {label} conditions share one program structure", not bad, str(bad[:8]))

    # ------------------------------------------------- G. parseability
    g = "G. Parseability"
    unparseable = []
    for _, r in df.iterrows():
        try:
            ast.parse(N(r.code))
        except SyntaxError as e:
            unparseable.append((r.title, str(e)[:50]))
    check(g, "all 256 rows parse as Python", not unparseable, str(unparseable[:6]))

    # ---------------------------------------------------------------- report
    print("=" * 78)
    print("FULL DATASET AUDIT —", path)
    print("=" * 78)
    cur = None
    for r in results:
        if r["group"] != cur:
            cur = r["group"]; print(f"\n{cur}")
        mark = "PASS" if r["ok"] else "FAIL"
        print(f"  [{mark}] {r['name']}")
        if not r["ok"] and r["detail"]:
            print(f"         -> {r['detail'][:400]}")
    n_fail = sum(1 for r in results if not r["ok"])
    print("\n" + "=" * 78)
    print(f"{len(results) - n_fail}/{len(results)} checks passed"
          + (f"   {n_fail} FAILED" if n_fail else "   ALL PASS"))
    print("=" * 78)
    return 1 if n_fail else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1] if len(sys.argv) > 1 else "data/merged_mod_jul28.csv"))
