"""
derive_invalidity_change.py — describe HOW each invalid variant was made invalid, by
diffing it against its valid twin.

Why derived rather than transcribed: Physics_Code_HumanGen.xlsx carries a hand-written
mechanism note (its unnamed 10th column) for 8 of the 32 base problems, but a line-by-line
cross-check against the live code found 3 correct, 3 incomplete and 2 outright wrong --
they appear to describe an earlier revision of the invalid code. Transcribing them would
put authoritative-looking but incorrect descriptions into the dataset, so the code is
treated as ground truth and the human note is retained separately as provenance.

Emits three columns onto every invalid row:
  invalidity_change        — mechanism derived from the NoComm_Valid -> NoComm_InValid diff
  invalidity_change_human  — the workbook's note, where one exists (8 samples), else NaN
  invalidity_change_agrees — whether the workbook note's quoted spans appear in the diff:
                             "yes" / "partial" / "no", else NaN

Diffing uses the NoComm_* pair so every difference is real code -- comments are absent
from both sides, and Comm_InValid's comments are inherited from Comm_Valid anyway.
"""

import difflib
import re

import pandas as pd

# ---------------------------------------------------------------------------
# Change classification
# ---------------------------------------------------------------------------

# A collector is write-only if its name is only ever read as the receiver of .append().
# Widening the guard around such an append changes how many snapshots accumulate in a list
# nothing consumes -- it cannot change any computed value, so it is instrumentation, not
# physics. Five synthetic samples carry exactly this pattern in their invalid variant only.
_GUARD_RE = re.compile(r"^\s*if\s+.+:\s*$")
_NUM_RE = re.compile(r"(?<![\w.])\d+\.?\d*(?:[eE][-+]?\d+)?")


def _minimal_spans(old: str, new: str) -> list[tuple[str, str]]:
    """Tightest (before, after) excerpts distinguishing two lines, ignoring shared context."""
    sm = difflib.SequenceMatcher(None, old, new, autojunk=False)
    spans = []
    for tag, i1, i2, j1, j2 in sm.get_opcodes():
        if tag == "equal":
            continue
        # widen to token boundaries so excerpts read as code, not fragments
        a, b = i1, i2
        while a > 0 and (old[a - 1].isalnum() or old[a - 1] in "_."):
            a -= 1
        while b < len(old) and (old[b].isalnum() or old[b] in "_."):
            b += 1
        c, d = j1, j2
        while c > 0 and (new[c - 1].isalnum() or new[c - 1] in "_."):
            c -= 1
        while d < len(new) and (new[d].isalnum() or new[d] in "_."):
            d += 1
        spans.append((old[a:b].strip(), new[c:d].strip()))
    return [(o, n) for o, n in spans if o != n]


def _sign_flipped(old: str, new: str) -> bool:
    """True when the only difference is +/- operators or inserted unary minus."""
    strip = lambda s: re.sub(r"[+\-\s]", "", s)
    return strip(old) == strip(new) and old.replace(" ", "") != new.replace(" ", "")


def _excerpt(o: str, n: str) -> str:
    """Readable `before -> after`. Falls back to whole lines when the minimal spans
    degenerate into single characters or empty strings, which happens for structural
    edits and reads as meaningless noise ("`` -> `-`")."""
    spans = _minimal_spans(o, n)
    degenerate = (not spans) or any(len(a) < 3 or len(b) < 3 for a, b in spans[:3])
    if degenerate:
        return f"`{o[:90]}` -> `{n[:90]}`"
    return "; ".join(f"`{a}` -> `{b}`" for a, b in spans[:3])


def classify(old: str, new: str, valid_full: str = "", invalid_full: str = "") -> tuple[str, str]:
    """Return (category, human-readable description) for one changed line pair.

    valid_full/invalid_full are the whole files, needed to tell a cosmetic global rename
    (every occurrence of U0 became B0) from a semantic substitution at one site
    (u_prev became u_curr; np.cos became np.cosh). Only the former is incidental --
    conflating them silently reclassifies the actual physical error as noise.
    """
    o, n = old.strip(), new.strip()
    if o == n:
        return "whitespace", "whitespace only"

    spans = _minimal_spans(o, n)
    excerpt = _excerpt(o, n)

    if _GUARD_RE.match(old) and _GUARD_RE.match(new):
        # guard condition changed; caller decides whether the body is write-only
        return "guard", f"snapshot guard changed: {excerpt}"

    if _sign_flipped(o, n):
        return "sign_flip", f"sign flip: {excerpt}"

    # unary minus inserted before an identifier/call argument, e.g. matmul(A, u) -> matmul(-A, u)
    if any(b.lstrip().startswith("-") and not a.lstrip().startswith("-") for a, b in spans):
        return "sign_flip", f"sign flip: {excerpt}"

    old_nums, new_nums = _NUM_RE.findall(o), _NUM_RE.findall(n)
    if old_nums != new_nums:
        if re.sub(_NUM_RE, "#", o) == re.sub(_NUM_RE, "#", n):
            # Cosmetic only when every literal keeps its VALUE and merely changes spelling
            # (1 -> 1.0, 0 -> 0.0). Comparing rstrip("0") made 100 and 1000 look identical,
            # which silently reclassified Heat_2's 10x grid refinement as formatting.
            same_value = len(old_nums) == len(new_nums) and all(
                float(a) == float(b) for a, b in zip(old_nums, new_nums)
            )
            if same_value:
                return "cosmetic_literal", f"int/float literal formatting: {excerpt}"
            return "coefficient", f"numeric change: {excerpt}"
        return "coefficient", f"numeric and structural change: {excerpt}"

    if re.sub(r"[A-Za-z_]\w*", "@", o) == re.sub(r"[A-Za-z_]\w*", "@", n):
        # Cosmetic only if it is a genuine global rename: the old name is gone from the
        # invalid file and the new name was absent from the valid one. If both names live
        # in both files, one existing symbol was swapped for another -- that is physics.
        # Wave_5 (u_prev -> u_curr) and Wave_6 (np.cos -> np.cosh) are exactly this.
        pairs = [(a, b) for a, b in spans if a and b]
        word = lambda s, t: re.search(rf"(?<![\w.]){re.escape(t)}(?![\w])", s) is not None
        # An attribute access is a call into a library, never an author-chosen name, so
        # changing it swaps the function being invoked. Wave_6's np.cos -> np.cosh turns an
        # oscillatory solution hyperbolic; it passes the disappeared-name test above
        # (no `np.cos` survives in the invalid file) yet is precisely the physical error.
        dotted = any("." in a or "." in b for a, b in pairs)
        global_rename = bool(pairs) and not dotted and all(
            not word(invalid_full, a) and not word(valid_full, b) for a, b in pairs
        )
        if global_rename:
            return "rename", f"identifier renamed throughout: {excerpt}"
        return "substitution", f"symbol substituted: {excerpt}"

    return "other", f"code change: {excerpt}"


# categories that cannot be the physical error
INCIDENTAL = {"whitespace", "cosmetic_literal", "rename", "guard"}


def _write_only_collectors(code: str) -> set[str]:
    """Names only ever read as the receiver of .append() -- i.e. write-only accumulators."""
    import ast
    try:
        tree = ast.parse(code)
    except SyntaxError:
        return set()
    appends, loads = {}, {}
    for node in ast.walk(tree):
        if (isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
                and node.func.attr == "append" and isinstance(node.func.value, ast.Name)):
            appends[node.func.value.id] = appends.get(node.func.value.id, 0) + 1
    for node in ast.walk(tree):
        if isinstance(node, ast.Name) and isinstance(node.ctx, ast.Load):
            loads[node.id] = loads.get(node.id, 0) + 1
    return {n for n, c in appends.items() if loads.get(n, 0) == c}


def diff_changes(valid_code: str, invalid_code: str) -> list[dict]:
    """Classified list of every difference between a valid/invalid code pair."""
    norm = lambda c: str(c).replace("\\r\\n", "\n").replace("\\n", "\n").split("\n")
    v, i = norm(valid_code), norm(invalid_code)
    collectors = _write_only_collectors("\n".join(i)) | _write_only_collectors("\n".join(v))

    out = []
    sm = difflib.SequenceMatcher(None, v, i, autojunk=False)
    for tag, i1, i2, j1, j2 in sm.get_opcodes():
        if tag == "equal":
            continue
        olds = [l for l in v[i1:i2] if l.strip()]
        news = [l for l in i[j1:j2] if l.strip()]
        if tag == "replace" and len(olds) == len(news):
            pairs = list(zip(olds, news))
        else:
            pairs = [(o, "") for o in olds] + [("", n) for n in news]
        for o, n in pairs:
            if not o and not n:
                continue
            if not o:
                cat, desc = "line_added", f"line added: `{n.strip()[:70]}`"
            elif not n:
                cat, desc = "line_removed", f"line removed: `{o.strip()[:70]}`"
            else:
                cat, desc = classify(o, n, "\n".join(v), "\n".join(i))
            # a guard is only incidental if what it gates is a write-only collector
            if cat == "guard" and not collectors:
                cat = "other"
            out.append({"category": cat, "description": desc,
                        "incidental": cat in INCIDENTAL})
    return out


def summarize(changes: list[dict]) -> str:
    """One-line mechanism description built from the non-incidental changes."""
    real = [c for c in changes if not c["incidental"]]
    if not real:
        return "no physical change detected"
    # collapse repeats of the same description (a sign flip applied at many boundary lines)
    seen, parts = set(), []
    for c in real:
        if c["description"] in seen:
            continue
        seen.add(c["description"])
        parts.append(c["description"])
    n_extra = len(real) - len(parts)
    text = "; ".join(parts[:4])
    if len(parts) > 4:
        text += f"; (+{len(parts) - 4} further distinct changes)"
    if n_extra:
        text += f" [same change repeated on {n_extra} further line(s)]"
    return text


# ---------------------------------------------------------------------------
# Human-note cross-check
# ---------------------------------------------------------------------------

def load_human_notes(xlsx_path: str) -> dict[str, str]:
    """{gt_sample: workbook mechanism note} from the unnamed 10th column of Sheet1."""
    df = pd.ExcelFile(xlsx_path).parse("Sheet1")
    col = next((c for c in df.columns if str(c).startswith("Unnamed")), None)
    if col is None:
        return {}
    notes = {}
    for _, row in df.iterrows():
        if pd.isna(row[col]):
            continue
        m = re.match(r"(Wave|Heat)_NoComm_InValid_(\d+)", str(row["Title"]).strip())
        if m:
            notes[f"{m.group(1)}_{m.group(2)}"] = " ".join(str(row[col]).split())
    return notes


# Verdicts from a line-by-line manual comparison of each workbook note against the live
# NoComm_Valid -> NoComm_InValid diff (2026-07-29). Recorded as data rather than computed:
# an automated substring match produced false "no" verdicts for notes that in fact match
# exactly (Wave_2, Heat_4), because the notes paraphrase spacing and quoting. A wrong
# agreement flag is worse than none, so the checked result is stored explicitly.
#
#   yes      — every change the note describes is present, and it describes them all
#   partial  — what it describes is correct but it omits further real changes
#   no       — it describes a change that is NOT what the code actually does
_NOTE_VERDICTS = {
    "Wave_2": "yes",    # -u_jm1->10*u_jm1, -u_jm1[0]->5*u_jm1[0], u_j[1]->2*u_j[1] all match
    "Wave_4": "yes",    # k_abs = np.abs(k); k_abs[0]=1e-14  ->  k_abs = k
    "Heat_4": "yes",    # np.matmul(D, u[t]) -> np.matmul(D, -u[t])
    "Wave_1": "partial",  # sign flip 2*u_n->-2*u_n correct; omits 10*/5*/3* on 3 boundary lines
    "Heat_1": "partial",  # matmul(-A, u) correct; omits k/dx**2->k/dx**3 and t_steps 1001->1000
    "Heat_3": "partial",  # toeplitz [-2.0,1.0]->[-2.0,-1.0] correct; omits U0/Un->B0/Bn rename
    "Wave_3": "no",     # note says hstack zeroed; code actually ADDS np.ones_like to dphidt/dpsidt
    "Heat_2": "no",     # note says matmul(-A, u+ones_like(u)); code is matmul(A, -u) - b
}


def note_agreement(gt_sample: str) -> str | None:
    """Hand-checked verdict for a workbook note; see _NOTE_VERDICTS."""
    return _NOTE_VERDICTS.get(gt_sample)


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

def add_invalidity_change(mod_df: pd.DataFrame, xlsx_path: str | None = None) -> pd.DataFrame:
    """Attach invalidity_change / _human / _agrees to every invalid row of mod_df."""
    valid = mod_df[mod_df["mod_type"] == "NoComm_Valid"].set_index("gt_sample")["code"]
    invalid = mod_df[mod_df["mod_type"] == "NoComm_InValid"].set_index("gt_sample")["code"]
    human = load_human_notes(xlsx_path) if xlsx_path else {}

    derived, agrees = {}, {}
    for gt in invalid.index:
        if gt not in valid.index:
            continue
        changes = diff_changes(valid[gt], invalid[gt])
        derived[gt] = summarize(changes)
        if gt in human:
            agrees[gt] = note_agreement(gt)

    df = mod_df.copy()
    is_invalid = ~df["phys_valid"].astype(bool)
    df["invalidity_change"] = df["gt_sample"].map(derived).where(is_invalid)
    df["invalidity_change_human"] = df["gt_sample"].map(human).where(is_invalid)
    df["invalidity_change_agrees"] = df["gt_sample"].map(agrees).where(is_invalid)
    return df


def audit_incidental(mod_df: pd.DataFrame) -> pd.DataFrame:
    """Every incidental (non-physics) difference between valid/invalid twins."""
    valid = mod_df[mod_df["mod_type"] == "NoComm_Valid"].set_index("gt_sample")["code"]
    invalid = mod_df[mod_df["mod_type"] == "NoComm_InValid"].set_index("gt_sample")["code"]
    rows = []
    for gt in sorted(invalid.index):
        if gt not in valid.index:
            continue
        src = mod_df.loc[mod_df["gt_sample"] == gt, "source"]
        for c in diff_changes(valid[gt], invalid[gt]):
            if c["incidental"]:
                rows.append({"gt_sample": gt,
                             "source": src.iloc[0] if len(src) else "",
                             "category": c["category"],
                             "description": c["description"]})
    return pd.DataFrame(rows)
