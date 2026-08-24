"""
Utilities for extracting variable probe points from PDE code snippets.

Workflow:
  1. build_name_map(clean, corrupt)      -> {corrupt_name: gt_name}
  2. get_pde_semantic_map(clean, corrupt) -> filtered {corrupt: gt}, pde_state/pde_param only
  3. extract_probe_points(code, sem_map, mod_type) -> list of probe dicts
"""

import ast
import io
import re
import tokenize
from typing import Optional


# ── Variable Classification ───────────────────────────────────────────────────

# Known PDE state variable names (solution fields, intermediate states)
_PDE_STATE_NAMES: frozenset[str] = frozenset({
    # Generic state
    "u", "v", "w", "p", "T", "phi", "psi", "rho",
    # Time-stepped variants
    "u_n", "u_np1", "u_nm1", "un", "vn", "pn", "wn",
    "u_prev", "u_next", "u_old", "u_new",
    # Spatial stencil variants
    "u_j", "u_jm1", "u_jp1", "u_jm2", "u_jp2",
    "u_i", "u_im1", "u_ip1",
    "u_left", "u_right", "u_up", "u_down",
    "u_iph_left", "u_iph_right",
    "u_im1_j", "u_ip1_j", "u_i_jm1", "u_i_jp1",
    # Riemann/flux variants
    "u_bar", "u_init", "u_hat", "u_tilde",
    "u_plus", "u_minus",
    # Other fields
    "v0", "v0_hat", "vorticity", "omega", "omega0", "omega0_hat",
    "phi_init", "phi_psi", "phi_psi_init",
    "u_ip2", "u_im1", "u_ip1",
    "solution",
})

# Known PDE parameter names (physical constants, step sizes, numerical params)
_PDE_PARAM_NAMES: frozenset[str] = frozenset({
    # Spatial / temporal steps
    "dx", "dy", "dz", "dr", "dt", "h",
    # Physical constants
    "nu", "mu", "alpha", "beta", "kappa", "sigma", "gamma",
    "rho", "c", "Re", "Pe", "Le", "Pr",
    # Wave speeds / CFL
    "c_1", "c_2", "c_3",
    "Cx2", "Cy2", "Cz2",
    "CFL", "CFL_1", "CFL_2", "CFL_3", "CFL_4",
    "C2",
    # Flux and derivative intermediates
    "dudx", "dudt", "dvdx", "dpdy", "ddx", "ddt",
    "flux", "f", "g",
    "f_bar", "f_left", "f_right", "f_interface",
    "fL", "fR",
    "s",   # numerical speed / wave speed in upwind schemes
    # Spectral / transform
    "FFT", "k", "k_abs",
    # Viscosity / diffusivity tensors
    "A", "A_phi", "A_psi", "B",
    # Misc physical params
    "viscosity", "L", "T", "q",
})

# Grid / domain size names — excluded from pde_state/pde_param focus
_GRID_SIZE_NAMES: frozenset[str] = frozenset({
    "n", "N", "nx", "ny", "nz", "N_x", "N_y", "N_z",
    "Nx", "Ny", "Nz", "nt", "N_t", "L_x", "L_y", "L_t",
    "nit", "t_steps",
})

# Single-char loop indices — always excluded
_LOOP_INDEX_NAMES: frozenset[str] = frozenset({"i", "j", "k", "l", "m"})


def classify_var(gt_name: str) -> str:
    """
    Classify a ground-truth variable name into one of:
      pde_state | pde_param | grid_size | loop_index | other
    """
    if gt_name in _LOOP_INDEX_NAMES:
        return "loop_index"
    if gt_name in _GRID_SIZE_NAMES:
        return "grid_size"
    if gt_name in _PDE_STATE_NAMES:
        return "pde_state"
    if gt_name in _PDE_PARAM_NAMES:
        return "pde_param"

    # Pattern-based fallbacks
    # Step sizes: d + (x|y|z|t|r) variants  e.g. dx2, dt_half
    if re.match(r'^d[xyztrs]\w*$', gt_name):
        return "pde_param"
    # Time-stepped / stencil variants: ends in _n, _np1, _nm1, _hat, _bar, _init
    if re.search(r'_(n|np1|nm1|hat|bar|init|prev|next|old|new|plus|minus)$', gt_name):
        return "pde_state"
    # Spatial stencil index: ends in _j, _jm1, _jp1, _im1, _ip1 etc.
    if re.search(r'_[ij](m\d*|p\d*)?$', gt_name):
        return "pde_state"

    return "other"


# ── Name Mapping ──────────────────────────────────────────────────────────────

def build_name_map(clean_code: str, corrupt_code: str) -> dict[str, str]:
    """
    Build {corrupt_name: gt_name} by aligning NAME tokens from both codes.

    Both codes share identical structure — only variable/function names differ.
    Keywords align perfectly, so divergences are the mapping.
    """
    def get_name_tokens(code: str) -> list[str]:
        tokens = []
        try:
            for tok in tokenize.generate_tokens(io.StringIO(code).readline):
                if tok.type == tokenize.NAME:
                    tokens.append(tok.string)
        except tokenize.TokenError:
            pass
        return tokens

    clean_names  = get_name_tokens(clean_code)
    corrupt_names = get_name_tokens(corrupt_code)

    mapping: dict[str, str] = {}
    for corrupt, clean in zip(corrupt_names, clean_names):
        if corrupt != clean and corrupt not in mapping:
            mapping[corrupt] = clean

    return mapping


def get_pde_semantic_map(clean_code: str, corrupt_code: str) -> dict[str, str]:
    """
    Return {corrupt: gt} restricted to pde_state and pde_param variables.

    Excludes: function definition names, loop indices, grid sizes, and anything
    that doesn't classify as a PDE state field or physical/numerical parameter.
    """
    full_map = build_name_map(clean_code, corrupt_code)

    try:
        tree = ast.parse(clean_code)
    except SyntaxError:
        return full_map

    func_def_names: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            func_def_names.add(node.name)

    return {
        c: g for c, g in full_map.items()
        if g not in func_def_names
        and classify_var(g) in ("pde_state", "pde_param")
    }


# ── Probe Point Extraction ────────────────────────────────────────────────────

def extract_probe_points(
    code: str,
    semantic_map: dict[str, str],
    title: str,
    mod_type: str,
    pde_class: str,
) -> list[dict]:
    """
    For each occurrence of each pde_state/pde_param variable in code, yield a probe dict.

    For NoComm_CorrVar:  semantic_map = {foobar_11: u, foobar_3: dx, ...}
    For NoComm_Valid:    semantic_map = {u: u, dx: dx, ...} (identity map)

    Fields per probe:
      prefix        — code up to (not including) the variable name token
      corrupt_var   — the variable name as it appears in this code
      gt_var        — the ground-truth meaningful variable name
      var_class     — pde_state | pde_param
      occurrence_idx — 0-indexed among all occurrences of THIS variable
      probe_idx     — global index in code order across all variables in snippet
      char_offset   — character offset in code
      line_number   — 1-indexed line
      code_fraction — char_offset / len(code), normalized [0, 1]
    """
    code_len = len(code)
    probes: list[dict] = []

    var_occurrence_counts: dict[str, int] = {}

    for corrupt_var, gt_var in sorted(semantic_map.items()):
        var_cls = classify_var(gt_var)
        pattern = r'\b' + re.escape(corrupt_var) + r'\b'
        for match in re.finditer(pattern, code):
            start  = match.start()
            prefix = code[:start]

            occurrence_idx = var_occurrence_counts.get(corrupt_var, 0)
            var_occurrence_counts[corrupt_var] = occurrence_idx + 1

            probes.append({
                "title":          title,
                "mod_type":       mod_type,
                "pde_class":      pde_class,
                "corrupt_var":    corrupt_var,
                "gt_var":         gt_var,
                "var_class":      var_cls,
                "prefix":         prefix,
                "occurrence_idx": occurrence_idx,
                "char_offset":    start,
                "line_number":    prefix.count('\n') + 1,
                "code_fraction":  round(start / code_len, 4) if code_len > 0 else 0.0,
            })

    probes.sort(key=lambda p: p["char_offset"])

    n = len(probes)
    for idx, p in enumerate(probes):
        p["probe_idx"]    = idx
        p["total_probes"] = n

    return probes


# ── Logprob Extraction ────────────────────────────────────────────────────────

def extract_continuation_logprob(
    prompt_logprobs: list[Optional[dict]],
    prefix_token_count: int,
    full_token_ids: list[int],
) -> Optional[float]:
    """
    Sum logprobs for tokens at positions [prefix_token_count, len(full_token_ids)).

    vLLM prompt_logprobs[i] is a dict {token_id: Logprob} for position i.
    The actual token is always present when prompt_logprobs >= 1.

    Returns None if any required position is missing.
    """
    total = 0.0
    for pos in range(prefix_token_count, len(full_token_ids)):
        if pos >= len(prompt_logprobs) or prompt_logprobs[pos] is None:
            return None
        token_id     = full_token_ids[pos]
        logprob_dict = prompt_logprobs[pos]
        if token_id not in logprob_dict:
            return None
        total += logprob_dict[token_id].logprob
    return total
