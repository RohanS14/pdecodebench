"""
parse_newcode.py — Parse data/newcode_jul28.txt (Shreya's 16 new base PDE
solvers, jul 28) into the pdedata_clean schema.

Each of the 16 base problems appears as a "{Class} Simulation Code Example N"
header followed by a valid script, then a "{Class} Simulation Code Example N
invalid (<failure mode note>)" header followed by the invalid variant. Both
scripts are full end-to-end generation artifacts: they carry a module
docstring naming the PDE/method, and trailing matplotlib/animation code whose
titles and saved filenames spell out the PDE class and numerical method
in plain text (e.g. "heat_2d_adi_multisource.png"). Both are stripped before
this code enters the dataset, since the existing NoComm_* perturbation only
strips '#' comment lines and would otherwise leak PDE identity straight
through the docstring/plot text regardless of variable obfuscation.

Assumptions specific to this source file (documented so future edits to
newcode_jul28.txt can be checked against them):
  - Every function-level docstring is a single line (no multi-line inner
    docstrings appear in this file).
  - The plotting/animation block is always a single contiguous block at the
    end of the script, starting at a `plt.figure(` or `fig[, ax] = plt.subplots(`
    line at column 0, and never interleaved with the numerical solve.

Docstring/comment stripping is done via line-based text manipulation, NOT
ast.parse/unparse — the ast module discards all comments, which would erase
the inline '#' comments that this dataset's Comm_Valid/NoComm_Valid axis
depends on.

IMPORTANT — comment style normalization: the existing pipeline's comment
machinery (corrupt_comment.py's extract_comments/count_comments/inject_comments,
and this module's own NoComm_* stripping) all detect comments via
`line.strip().startswith("#")` — i.e. whole-line comments only, matching the
old v1-v4 dataset's style. Shreya's new snippets instead follow the
generation prompt's "5-6 inline comments" instruction, meaning almost every
comment is a *trailing* comment (`code  # comment`), with only one generic
whole-line comment per snippet ("# Domain and physical parameters", reused
verbatim across nearly all 16 samples). Left as-is, this would make
NoComm_Valid/NoComm_InValid retain most of their comments (not actually
comment-free) and make CorrComm/CorrComm_Invalid a near no-op (the only
comment ever touched carries no PDE-identifying information). `clean_code()`
therefore runs `_normalize_inline_comments()` last, which uses `tokenize` to
find every COMMENT token that has real code before it on the same physical
line, and hoists it onto its own line directly above (same indentation),
matching the old dataset's convention exactly so the untouched downstream
scripts work correctly on the new material with zero changes to them.

Public entry points:
  parse_raw_sections(text)   -> list of per-example dicts with raw code spans
  build_snippet_table(text)  -> {(prefix, local_idx): {"valid":.., "invalid":.., "note":..}}
  clean_code(raw_code)       -> code with docstring/plots stripped, comments intact
  write_tag_review_csv(path) -> writes the human-review CSV if it doesn't exist
  get_new_base_rows(tag_df)  -> list of dict rows (Comm_Valid/NoComm_Valid/
                                 Comm_InValid/NoComm_InValid) for the 16 new
                                 gt_samples, tagged from tag_df
"""

import csv
import io
import os
import re
import tokenize
from collections import defaultdict

NEWCODE_PATH = "data/newcode_jul28.txt"
TAG_REVIEW_PATH = "data/descriptions/newcode_v5_tag_review.csv"

HEADER_RE = re.compile(
    r"^(?P<cls>Heat Equation|Wave Equation|Burgers|Navier Sto\w+)\s+"
    r"Simulation Code Example\s+(?P<idx>\d+)\s*"
    r"(?:(?P<invalidflag>[Ii]nvalid)\s*\((?P<note>[^)]*)\))?\s*$"
)

# gt_sample numbering continues from the existing 16 (indices 1-4 per class);
# new material occupies indices 5-8.
GT_NUMBER_OFFSET = 4


def _map_prefix(cls_raw: str) -> tuple[str, str]:
    if cls_raw.startswith("Heat"):
        return "Heat", "heat"
    if cls_raw.startswith("Wave"):
        return "Wave", "wave"
    if cls_raw.startswith("Burgers"):
        return "Burgers", "burgers"
    if cls_raw.startswith("Navier"):
        return "NavierStokes", "navier-stokes"
    raise ValueError(f"Unrecognized class header: {cls_raw!r}")


# ---------------------------------------------------------------------------
# Step 1: split the raw text into per-example (valid/invalid) code spans
# ---------------------------------------------------------------------------

def parse_raw_sections(text: str) -> list[dict]:
    lines = text.split("\n")
    matches = []
    for i, line in enumerate(lines):
        m = HEADER_RE.match(line.strip())
        if not m:
            continue
        prefix, pde_class = _map_prefix(m.group("cls"))
        matches.append(
            {
                "line_idx": i,
                "prefix": prefix,
                "pde_class": pde_class,
                "local_idx": int(m.group("idx")),
                "is_invalid": m.group("invalidflag") is not None,
                "note": (m.group("note") or "").strip(),
            }
        )

    for j, m in enumerate(matches):
        start = m["line_idx"] + 1
        end = matches[j + 1]["line_idx"] if j + 1 < len(matches) else len(lines)
        code_lines = lines[start:end]
        while code_lines and not code_lines[0].strip():
            code_lines.pop(0)
        while code_lines and not code_lines[-1].strip():
            code_lines.pop()
        m["raw_code"] = "\n".join(code_lines)

    return matches


def build_snippet_table(text: str) -> dict:
    """Group parsed sections into {(prefix, local_idx): {"valid":, "invalid":, "note":, "pde_class":}}."""
    matches = parse_raw_sections(text)
    table: dict = defaultdict(dict)
    for m in matches:
        key = (m["prefix"], m["local_idx"])
        table[key]["pde_class"] = m["pde_class"]
        if m["is_invalid"]:
            table[key]["invalid"] = m["raw_code"]
            table[key]["note"] = m["note"]
        else:
            table[key]["valid"] = m["raw_code"]

    for key, entry in table.items():
        if "valid" not in entry or "invalid" not in entry:
            raise ValueError(f"Incomplete valid/invalid pair for {key}: found keys {list(entry)}")

    return dict(table)


# ---------------------------------------------------------------------------
# Step 2: strip docstrings + plotting/animation code, keep inline # comments
# ---------------------------------------------------------------------------

_TRIPLE_QUOTE_RE = re.compile(r'^\s*("""|\'\'\')')
_PLOT_START_RE = re.compile(
    r"^\s*(plt\.figure\(|fig\s*,\s*\w+\s*=\s*plt\.subplots\(|fig\s*=\s*plt\.subplots\()"
)
_MPL_IMPORT_RE = re.compile(
    r"^\s*(import matplotlib(\.\w+)*(\s+as\s+\w+)?|from matplotlib(\.\w+)*\s+import\s+.*|from mpl_toolkits.*import.*)\s*$"
)


def _strip_module_docstring(code: str) -> str:
    lines = code.split("\n")
    i = 0
    while i < len(lines) and not lines[i].strip():
        i += 1
    if i >= len(lines):
        return code
    m = _TRIPLE_QUOTE_RE.match(lines[i])
    if not m:
        return code
    quote = m.group(1)
    # find closing quote, which may be on the same line (after the opening) or a later line
    first_line_rest = lines[i].strip()[len(quote):]
    if quote in first_line_rest:
        end = i
    else:
        end = i + 1
        while end < len(lines) and quote not in lines[end]:
            end += 1
    # drop lines[i..end] (the docstring block) and one following blank line
    remaining = lines[end + 1 :]
    while remaining and not remaining[0].strip():
        remaining.pop(0)
    return "\n".join(remaining)


def _strip_function_docstrings(code: str) -> str:
    lines = code.split("\n")
    out = []
    for idx, line in enumerate(lines):
        stripped = line.strip()
        is_single_line_docstring = bool(
            re.match(r'^("""[^"]*"""|\'\'\'[^\']*\'\'\')$', stripped)
        )
        if is_single_line_docstring and out and out[-1].rstrip().endswith(":"):
            continue
        out.append(line)
    return "\n".join(out)


def _strip_matplotlib_imports(code: str) -> str:
    lines = [l for l in code.split("\n") if not _MPL_IMPORT_RE.match(l)]
    return "\n".join(lines)


def _strip_trailing_plot_block(code: str) -> str:
    lines = code.split("\n")
    for i, line in enumerate(lines):
        if _PLOT_START_RE.match(line):
            return "\n".join(lines[:i])
    print("  WARNING: no plot-block start marker found — code left as-is; check manually.")
    return code


def _collapse_blank_lines(code: str) -> str:
    lines = code.split("\n")
    out = []
    blank_run = 0
    for line in lines:
        if line.strip():
            blank_run = 0
            out.append(line)
        else:
            blank_run += 1
            if blank_run <= 2:
                out.append(line)
    while out and not out[-1].strip():
        out.pop()
    return "\n".join(out) + "\n"


def _normalize_inline_comments(code: str) -> str:
    """Hoist trailing '# comment' code onto its own whole-line comment directly
    above, at the same indentation, so the existing whole-line-only comment
    machinery (extract_comments/count_comments/inject_comments/_strip_comments)
    works correctly on it. Uses tokenize (not a '#' string search) so '#'
    characters inside string literals are never mistaken for a comment."""
    lines = code.split("\n")
    try:
        tokens = list(tokenize.generate_tokens(io.StringIO(code).readline))
    except (tokenize.TokenizeError, IndentationError, SyntaxError):
        print("  WARNING: tokenize failed during comment normalization — code left as-is; check manually.")
        return code

    replace_map = {}
    for tok in tokens:
        if tok.type != tokenize.COMMENT:
            continue
        row, col = tok.start
        row_idx = row - 1
        line = lines[row_idx]
        before = line[:col]
        if before.strip() == "":
            continue  # already a whole-line comment
        indent = before[: len(before) - len(before.lstrip())]
        replace_map[row_idx] = (indent + tok.string, before.rstrip())

    if not replace_map:
        return code

    out = []
    for i, line in enumerate(lines):
        if i in replace_map:
            comment_line, code_line = replace_map[i]
            out.append(comment_line)
            out.append(code_line)
        else:
            out.append(line)
    return "\n".join(out)


def clean_code(raw_code: str) -> str:
    code = _strip_module_docstring(raw_code)
    code = _strip_matplotlib_imports(code)
    code = _strip_function_docstrings(code)
    code = _strip_trailing_plot_block(code)
    code = _collapse_blank_lines(code)
    code = _normalize_inline_comments(code)
    return code


def _extract_docstring_summary(raw_code: str) -> str:
    """Pull the module docstring text out for the human-review CSV (context only,
    never stored in the final dataset code)."""
    lines = raw_code.split("\n")
    i = 0
    while i < len(lines) and not lines[i].strip():
        i += 1
    if i >= len(lines) or not _TRIPLE_QUOTE_RE.match(lines[i]):
        return ""
    quote = lines[i].strip()[:3]
    body = []
    first_rest = lines[i].strip()[3:]
    if quote in first_rest:
        return first_rest.split(quote)[0].strip()
    body.append(first_rest)
    j = i + 1
    while j < len(lines) and quote not in lines[j]:
        body.append(lines[j].strip())
        j += 1
    if j < len(lines):
        body.append(lines[j].split(quote)[0].strip())
    return " ".join(b for b in body if b)


# ---------------------------------------------------------------------------
# Step 3: per-snippet phys_process / num_method proposals (hand-reasoned from
# reading newcode_jul28.txt — see reasoning strings). Reviewer: check the
# "NEEDS REVIEW" entries especially closely.
# ---------------------------------------------------------------------------

TAG_TABLE = {
    ("Heat", 1): dict(phys_process="diffusion", num_method="explicit",
        reasoning="Pure heat/diffusion equation; FTCS is a standard explicit forward-time update."),
    ("Heat", 2): dict(phys_process="diffusion", num_method="implicit",
        reasoning="Crank-Nicolson solves a tridiagonal system each step (implicit)."),
    ("Heat", 3): dict(phys_process="diffusion", num_method="implicit",
        reasoning="ADI solves an implicit tridiagonal system on each half-step sweep."),
    ("Heat", 4): dict(phys_process="diffusion", num_method="spectral",
        reasoning="Exact spectral (FFT) time-advance of each Fourier mode's decay factor."),

    ("Wave", 1): dict(phys_process="oscillation", num_method="explicit",
        reasoning="Standard explicit FDTD leapfrog update, no damping/forcing term."),
    ("Wave", 2): dict(phys_process="oscillation", num_method="spectral",
        reasoning="Each Fourier mode advanced by its exact analytic (spectral) time solution."),
    ("Wave", 3): dict(phys_process="oscillation/restoration", num_method="explicit",
        reasoning="Explicit leapfrog update, but a sponge/absorbing layer damps amplitude near "
                  "the boundaries -> restoration in addition to oscillation."),
    ("Wave", 4): dict(phys_process="oscillation", num_method="implicit",
        reasoning="NEEDS REVIEW: Newmark-beta solves a linear system via LU each step (implicit). "
                  "gamma=0.6 (>0.5) adds mild numerical damping of grid-scale noise -- this is a "
                  "numerical stabilization choice, not a modeled physical restoring force, so tagged "
                  "as pure oscillation; flag if you'd rather call this oscillation/restoration."),

    ("Burgers", 1): dict(phys_process="advection", num_method="explicit",
        reasoning="nu=0 (inviscid): pure advection/shock formation, no diffusion term in the PDE."),
    ("Burgers", 2): dict(phys_process="advection/diffusion", num_method="explicit",
        reasoning="nu=0.005>0: explicit Lax-Wendroff advection flux plus explicit central-difference "
                  "diffusion term."),
    ("Burgers", 3): dict(phys_process="advection/diffusion", num_method="explicit",
        reasoning="nu=0.01>0: explicit upwind advection plus explicit central diffusion term."),
    ("Burgers", 4): dict(phys_process="advection/diffusion", num_method="explicit",
        reasoning="nu=0.02>0, 2D coupled (u,v): explicit upwind advection plus explicit diffusion, "
                  "same reasoning as the 1D viscous cases."),

    ("NavierStokes", 1): dict(phys_process="diffusion/restoration", num_method="explicit",
        reasoning="NEEDS REVIEW: 1D channel start-up flow has no advective nonlinearity (u.grad(u)=0 "
                  "by symmetry in this formulation) -- it diffuses momentum from the walls and relaxes "
                  "toward a steady parabolic profile under a constant pressure gradient (restoration). "
                  "Explicit FTCS time-stepping."),
    ("NavierStokes", 2): dict(phys_process="diffusion", num_method="implicit",
        reasoning="NEEDS REVIEW: Stokes' first problem -- no advective nonlinearity (1D unidirectional "
                  "flow) and no forcing/steady-state target, so tagged as pure diffusion (momentum "
                  "diffusing outward from the impulsively-started wall) rather than diffusion/restoration. "
                  "Crank-Nicolson implicit time-stepping."),
    ("NavierStokes", 3): dict(phys_process="advection/diffusion", num_method="explicit/implicit",
        reasoning="Full 2D advection (u.grad(u)) + viscous diffusion; Chorin projection explicitly "
                  "advances momentum but solves the pressure Poisson equation via an implicit iterative "
                  "(Jacobi) solve each step."),
    ("NavierStokes", 4): dict(phys_process="advection/diffusion", num_method="spectral/explicit",
        reasoning="Nonlinear advection of vorticity plus weak (near-inviscid, nu=1e-4) diffusion; "
                  "spatial derivatives via FFT (spectral), explicit RK2 (Heun) time-stepping -- matches "
                  "the 'spectral/explicit' tag convention already used elsewhere in the dataset."),
}


# ---------------------------------------------------------------------------
# Step 4: write the human-review CSV
# ---------------------------------------------------------------------------

def write_tag_review_csv(newcode_path: str = NEWCODE_PATH, out_path: str = TAG_REVIEW_PATH) -> None:
    with open(newcode_path, "r", encoding="utf-8") as f:
        text = f.read()
    table = build_snippet_table(text)

    rows = []
    for (prefix, local_idx), entry in sorted(table.items()):
        gt_num = local_idx + GT_NUMBER_OFFSET
        gt_sample = f"{prefix}_{gt_num}"
        tags = TAG_TABLE.get((prefix, local_idx))
        if tags is None:
            raise KeyError(f"No proposed tags for {(prefix, local_idx)} -- add an entry to TAG_TABLE")
        rows.append(
            {
                "gt_sample": gt_sample,
                "docstring_summary": _extract_docstring_summary(entry["valid"]),
                "pde_class": entry["pde_class"],
                "phys_process": tags["phys_process"],
                "num_method": tags["num_method"],
                "invalidity_note": entry.get("note", ""),
                "reasoning": tags["reasoning"],
            }
        )

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "gt_sample", "docstring_summary", "pde_class",
                "phys_process", "num_method", "invalidity_note", "reasoning",
            ],
        )
        writer.writeheader()
        writer.writerows(rows)
    print(f"Wrote {out_path} ({len(rows)} rows). Review before running build_v5.py.")


# ---------------------------------------------------------------------------
# Step 5: build the 4 core mod_type rows per new gt_sample, using reviewed tags
# ---------------------------------------------------------------------------

def get_new_base_rows(tag_rows: list[dict], newcode_path: str = NEWCODE_PATH) -> list[dict]:
    """tag_rows: list of dicts read back from the (possibly hand-edited) review CSV."""
    with open(newcode_path, "r", encoding="utf-8") as f:
        text = f.read()
    table = build_snippet_table(text)

    tag_by_gt = {r["gt_sample"]: r for r in tag_rows}

    rows = []
    for (prefix, local_idx), entry in sorted(table.items()):
        gt_num = local_idx + GT_NUMBER_OFFSET
        gt_sample = f"{prefix}_{gt_num}"
        if gt_sample not in tag_by_gt:
            raise KeyError(f"{gt_sample} missing from tag review CSV")
        tag_row = tag_by_gt[gt_sample]

        valid_code = clean_code(entry["valid"])
        invalid_code = clean_code(entry["invalid"])
        invalidity_note = tag_row.get("invalidity_note", "") or None

        for mod_type, code, phys_valid in [
            ("Comm_Valid", valid_code, True),
            ("Comm_InValid", invalid_code, False),
        ]:
            rows.append(
                {
                    "title": f"{prefix}_{mod_type}_{gt_num}",
                    "code": code,
                    "num_lines": len(code.split("\n")),
                    "num_char": len(code),
                    "pde_class": tag_row["pde_class"],
                    "phys_process": tag_row["phys_process"],
                    "phys_valid": phys_valid,
                    "num_method": tag_row["num_method"],
                    "corruption_source_id": None,
                    "corruption_source_pde": None,
                    "injected_comments": None,
                    "delta_comments": None,
                    "num_comments": sum(
                        1 for l in code.split("\n") if l.strip().startswith("#")
                    ),
                    "gt_sample": gt_sample,
                    "mod_type": mod_type,
                    "invalidity_note": invalidity_note if not phys_valid else None,
                }
            )

    # NoComm_* rows: strip '#' comments from the Comm_* rows just built
    nocomm_rows = []
    for row in rows:
        code = "\n".join(
            l for l in row["code"].split("\n") if not l.strip().startswith("#")
        )
        nocomm_rows.append(
            {
                **row,
                "code": code,
                "num_lines": len(code.split("\n")),
                "num_char": len(code),
                "num_comments": 0,
                "mod_type": row["mod_type"].replace("Comm_", "NoComm_"),
                "title": row["title"].replace("Comm_", "NoComm_"),
            }
        )

    return rows + nocomm_rows


if __name__ == "__main__":
    import os as _os
    script_dir = _os.path.dirname(_os.path.abspath(__file__))
    repo_root = _os.path.dirname(script_dir)
    _os.chdir(repo_root)
    write_tag_review_csv()
