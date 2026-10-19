"""
validate_merged.py — standing leak audit for the jul28 evaluation set.

datagen/full_audit.py checks that the dataset is BUILT correctly: balance, donor
constraints, condition semantics, cross-condition program identity. This checks
something different and complementary -- whether the dataset still LEAKS. Those are
separate failure modes. A build can be perfectly consistent and still hand a model
the answer through an import path.

Proposed during a review of the Jul 29 build and never written, which is part of
why the findings below had to be rediscovered by hand. Landing it makes them
reproducible.

What it measures, and why each one earned a place:

  identifiers   How many author-declared names survive obfuscation, per sample. The
                NoComm_CorrVar conditions exist to remove lexical cues; a sample
                where they survive is not testing what the condition claims. This
                caught NavierStokes_4 keeping `viscosity`, `drag` and
                `max_velocity` through a protected-kwargs exemption that turned out
                to be unnecessary.

  imports       Import paths name the problem domain and no renaming hides them.
                `jax_cfd` and `mpi4py_fft` survive every condition including full
                obfuscation.

  literals      String literals survive comment stripping and renaming alike.
                Reported with their content so a reader can judge severity -- most
                are inert, and calling them all leaks would be as wrong as missing
                the real ones.

  notes         invalidity_note coverage. Every invalid row needs one: it is the
                ground truth against which a model's explanation is graded, and a
                blank means that row cannot be scored on reasoning at all.

  equations     verified_by coverage in equations_jul28.csv. The equations are the
                physics ground truth for every cross-modal number, and
                dataset_overview.md warns that a wrong one "corrupts those results
                in a way no downstream statistic can detect". needs_review being
                clear is not the same as a human having signed off.

Run:  python cross_modal_consistency/datagen/validate_merged.py
      python cross_modal_consistency/datagen/validate_merged.py --dataset data/merged_mod_jul28.csv --strict
"""
import argparse
import ast
import collections
import csv
import re
import sys

csv.field_size_limit(10 ** 9)

DEFAULT_DATASET = "data/merged_mod_jul28.csv"
DEFAULT_EQUATIONS = "data/equations_jul28.csv"

OBFUSCATED = ("NoComm_CorrVar", "NoComm_CorrVar_InValid")
FOOBAR = re.compile(r"^(foobar_\d+|fn\d+)$")

# Library kwargs are the library's API, not author vocabulary: renaming
# np.zeros(shape=...) raises TypeError. They survive by design and are not leaks.
LIBRARY_KWARGS = {
    "shape", "endpoint", "indexing", "args", "d", "axis", "dtype", "collapse",
    "out", "rank", "sparse", "view", "divide", "invalid", "s", "length",
    "static_argnums", "domain", "maximum_velocity", "peak_wavenumber", "copy",
    "keepdims", "casting", "order", "subok", "where", "initial", "ddof", "mode",
}

# Import roots that name a physical domain or a numerical method.
DOMAIN_IMPORTS = ("jax_cfd", "mpi4py_fft", "fenics", "firedrake", "dedalus", "pyfftw")

# Vocabulary that would identify the PDE class or the numerical method.
LEAK_WORDS = (
    "heat", "thermal", "diffusiv", "wave", "oscillat", "vibrat", "burgers", "shock",
    "navier", "stokes", "cfd", "vortic", "crank", "nicolson", "spectral",
    "implicit", "explicit", "runge", "kutta", "cfl", "fft",
)


def surviving_identifiers(code):
    """Author-declared names still readable after obfuscation, excluding library
    kwargs, which are the library's API rather than the author's vocabulary."""
    try:
        tree = ast.parse(code)
    except SyntaxError:
        return set()
    out = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Name) and isinstance(node.ctx, ast.Store):
            out.add(node.id)
        elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            out.add(node.name)
            for a in node.args.args + node.args.kwonlyargs + node.args.posonlyargs:
                out.add(a.arg)
        elif isinstance(node, ast.keyword) and node.arg:
            out.add(node.arg)
    return {n for n in out if not FOOBAR.match(n) and n not in LIBRARY_KWARGS}


def domain_imports(code):
    out = set()
    for line in code.splitlines():
        line = line.strip()
        if line.startswith(("import ", "from ")):
            for root in DOMAIN_IMPORTS:
                if root in line:
                    out.add(line)
    return out


def string_literals(code, min_len=4):
    try:
        tree = ast.parse(code)
    except SyntaxError:
        return []
    return [n.value for n in ast.walk(tree)
            if isinstance(n, ast.Constant) and isinstance(n.value, str)
            and len(n.value) >= min_len and n.value != "__main__"]


def leaky_words(text):
    low = text.lower()
    return sorted({w for w in LEAK_WORDS if w in low})


def audit(dataset, equations):
    rows = list(csv.DictReader(open(dataset, newline="")))
    by_sample = collections.defaultdict(dict)
    for r in rows:
        by_sample[r["gt_sample"]][r["mod_type"]] = r

    findings, total_names, total_surviving = [], 0, 0
    per_sample = {}
    for sample, variants in sorted(by_sample.items()):
        surviving, imports, literals = set(), set(), []
        for mt in OBFUSCATED:
            if mt not in variants:
                continue
            code = variants[mt]["code"]
            names = surviving_identifiers(code)
            renamed = set(re.findall(r"\bfoobar_\d+\b|\bfn\d+\b", code))
            total_names += len(names) + len(renamed)
            total_surviving += len(names)
            surviving |= names
            imports |= domain_imports(code)
            literals += string_literals(code)
        # A domain import is a finding in its own right: `mpi4py_fft` names the
        # method even though no single leak word need appear in the line, and no
        # renaming can hide an import path.
        leaks = leaky_words(" ".join(imports) + " " + " ".join(literals))
        if imports and not leaks:
            leaks = ["domain-specific import"]
        per_sample[sample] = {
            "surviving": sorted(surviving), "imports": sorted(imports),
            "literals": literals, "leak_words": leaks,
        }
        if leaks:
            findings.append((sample, leaks, sorted(imports), literals))

    invalid = [r for r in rows if "Invalid" in r["mod_type"] or "InValid" in r["mod_type"]]
    blank_notes = [r["title"] for r in invalid if not r["invalidity_note"].strip()]

    eq_rows, unsigned, flagged = [], [], []
    try:
        eq_rows = list(csv.DictReader(open(equations, newline="")))
        unsigned = [r["gt_sample"] for r in eq_rows if not r.get("verified_by", "").strip()]
        flagged = [r["gt_sample"] for r in eq_rows
                   if r.get("needs_review", "0").strip() not in ("0", "", "False")]
    except FileNotFoundError:
        pass

    return {
        "rows": len(rows), "samples": len(by_sample),
        "total_names": total_names, "total_surviving": total_surviving,
        "per_sample": per_sample, "findings": findings,
        "invalid_rows": len(invalid), "blank_notes": blank_notes,
        "equations": len(eq_rows), "unsigned": unsigned, "flagged": flagged,
    }


def report(a, strict=False):
    print("=" * 72)
    print("jul28 leak audit")
    print("=" * 72)
    print(f"{a['rows']} rows, {a['samples']} samples\n")

    print("-- identifier obfuscation --")
    n_samples_affected = sum(1 for v in a["per_sample"].values() if v["surviving"])
    print(f"   {a['total_surviving']} of {a['total_names']} distinct author-declared "
          f"names survive renaming, across {n_samples_affected} samples")
    if a["total_surviving"] == 0:
        print("   (library kwargs excluded -- they are the library's API, not author")
        print("    vocabulary, and renaming np.zeros(shape=...) would raise TypeError)")
    worst = sorted(((len(v["surviving"]), s) for s, v in a["per_sample"].items()),
                   reverse=True)[:5]
    for n, s in worst:
        if n:
            print(f"     {s:<18}{n:>3}  {a['per_sample'][s]['surviving'][:6]}")
    print()

    print("-- identity leaks surviving every condition --")
    if not a["findings"]:
        print("   none")
    for sample, words, imports, literals in a["findings"]:
        print(f"   {sample}  ->  {words}")
        for i in imports:
            print(f"       import  {i}")
        for l in literals:
            if leaky_words(l):
                print(f"       literal {l!r}")
    print()

    print("-- invalidity_note coverage --")
    print(f"   {a['invalid_rows'] - len(a['blank_notes'])}/{a['invalid_rows']} invalid rows "
          f"carry a note")
    if a["blank_notes"]:
        print(f"   MISSING: {a['blank_notes'][:8]}")
    print()

    print("-- equation ground truth --")
    if not a["equations"]:
        print("   equations file not found")
    else:
        print(f"   {a['equations']} equations, {len(a['flagged'])} flagged needs_review "
              f"{a['flagged'] or ''}")
        print(f"   {a['equations'] - len(a['unsigned'])}/{a['equations']} signed off "
              f"(verified_by populated)")
        if a["unsigned"]:
            print("   These are the physics ground truth for every cross-modal number.")
            print("   A wrong equation corrupts results in a way no downstream statistic")
            print("   can detect, so needs_review being clear is not the same as signed off.")
    print()
    print("=" * 72)

    problems = bool(a["blank_notes"]) or (strict and bool(a["unsigned"]))
    return 1 if problems else 0


def main():
    p = argparse.ArgumentParser(description="Leak audit for the jul28 evaluation set")
    p.add_argument("--dataset", default=DEFAULT_DATASET)
    p.add_argument("--equations", default=DEFAULT_EQUATIONS)
    p.add_argument("--strict", action="store_true",
                   help="also fail when equations are unsigned")
    args = p.parse_args()
    return report(audit(args.dataset, args.equations), args.strict)


if __name__ == "__main__":
    sys.exit(main())
