"""
Build the symbolic-equation modality for Experiment 2 Part II (plan §15).

Writes data/equations_jul28.csv — one PDE per gt_sample, in three notations, so
cross-modal retrieval can separate "the model matched the physics" from "the model
matched the notation".

WHY THIS FILE IS GENERATED, NOT HAND-WRITTEN
The equation set is the physics ground truth for the whole cross-modal experiment.
If an equation is wrong, every retrieval number built on it is wrong in a way no
downstream statistic can detect. So each row carries the evidence it was derived
from (dimensionality, detected coefficients, dataset labels) and a `needs_review`
flag. Rows flagged for review MUST be signed off by a domain reader before any
cross-modal result is reported. `verified_by` is empty until that happens.

Evidence is extracted from the actual solver code, not from the pde_class name.

Usage:
    python datagen/build_equations.py [--out data/equations_jul28.csv]
"""
import argparse
import csv
import os
import re
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from shared.dataset_io import DEFAULT_MOD_DATASET, load_dataset  # noqa: E402

BASE_CONDITION = "Comm_Valid"


def detect_dim(code: str) -> tuple:
    """
    Return (dim, evidence, weak) for the SPATIAL dimensionality.

    Deliberately does NOT use "the state array has two axes". Many of these
    solvers allocate `u = np.zeros((nt, nx))` to keep the time history of a 1D
    field, which that test misreads as a 2D domain — it mislabelled Burgers_1-4
    before this was fixed.

    Instead we look for a genuine second spatial coordinate, including the
    spectral case where the domain lives in Fourier space and there is no
    meshgrid at all (NavierStokes_4: kx/ky, Nx/Ny, rfftn).
    """
    # 3D first — NavierStokes_3 is a distributed spectral Taylor-Green solver with
    # three velocity components and three wavenumber axes.
    ev3 = []
    if re.search(r"\bkz\b|\bK\[2\]|\bdz\b|\bn_?z\b", code, re.I):
        ev3.append("z-axis")
    if re.search(r"X\[2\]", code):
        ev3.append("X[2]")
    if re.search(r"rank\s*=\s*1", code) and "PFFT" in code:
        ev3.append("vector-PFFT")
    if len(ev3) >= 2:
        return "3D", ev3, False

    ev = []
    if "meshgrid" in code:
        ev.append("meshgrid")
    if re.search(r"\bky\b", code) and re.search(r"\bkx\b", code):
        ev.append("kx+ky")                       # spectral 2D wavenumbers
    if re.search(r"\[\s*1:-1\s*,\s*1:-1\s*\]", code):
        ev.append("2d-stencil")

    # PAIRED coordinate evidence only. `dy`, `ny` and `y = linspace(...)` are all
    # the SAME coordinate, so counting them as independent signals turned two 1D
    # wall-normal solvers into 2D ones: NavierStokes_5 (forced Poiseuille,
    # u[1:-1] updated along a single index) and NavierStokes_6 (Stokes' first
    # problem). A second dimension requires an x counterpart to the y evidence.
    for xpat, ypat, name in [
        (r"\bN_?x\b", r"\bN_?y\b", "Nx+Ny"),
        (r"\bdx\b", r"\bdy\b", "dx+dy"),
        (r"\bx\s*=\s*np\.(linspace|arange)", r"\by\s*=\s*np\.(linspace|arange)",
         "x-grid+y-grid"),
    ]:
        if re.search(xpat, code, re.I) and re.search(ypat, code, re.I):
            ev.append(name)

    strong = {"meshgrid", "kx+ky", "2d-stencil"}
    if ev and (set(ev) & strong or len(ev) >= 2):
        return "2D", ev, False
    if ev:
        return "1D", ev, True                    # single paired signal — flag it
    return "1D", [], False


def has_zero_viscosity(code: str) -> bool:
    """
    `nu = 0.0` means inviscid however many times the name `nu` appears.

    Handles tuple assignment, which is how Burgers_5 does it:
        L, nu = 2.0, 0.0
    A plain `\\bnu\\s*=\\s*0` regex misses that entirely and would have called an
    inviscid solver viscous.
    """
    if re.search(r"\bnu\s*=\s*0(\.0*)?\s*(?![.\d])", code):
        return True
    for lhs, rhs in re.findall(r"^\s*([\w\s,]+?)\s*=\s*([^=\n]+)$", code, re.M):
        names = [n.strip() for n in lhs.split(",")]
        vals = [v.strip() for v in rhs.split(",")]
        if len(names) > 1 and len(names) == len(vals) and "nu" in names:
            v = vals[names.index("nu")]
            if re.fullmatch(r"0(\.0*)?", v):
                return True
    return False


def is_vorticity_formulation(code: str) -> bool:
    """
    A solver that names a vorticity field AND a streamfunction (or solves a
    Poisson equation for one) is not ambiguous — NavierStokes_4 defines
    `vorticity_to_velocity` and `psi_hat` outright.
    """
    # Trailing \w* matters: `psi_hat` and `omega_hat` do NOT match \bpsi\b /
    # \bomega\b, because underscore is a word character so the boundary fails.
    # That is exactly how NavierStokes_4 names them.
    has_w = bool(re.search(r"\bvorticity\w*|\bomega\w*|\bw_hat\b", code, re.I))
    has_psi = bool(re.search(r"\bpsi\w*|streamfunc", code, re.I))
    return has_w and has_psi


def alpha_is_numerical_dissipation(code: str) -> bool:
    """
    Distinguish a physical diffusivity from a Rusanov / local Lax-Friedrichs
    wave-speed coefficient. Burgers_4 computes
        alpha = np.maximum(np.abs(u[:-1]), np.abs(u[1:]))
    which is a numerical flux term, not viscosity. Treating it as viscosity would
    have turned an inviscid Burgers solver into a viscous one in the equation set.
    """
    return bool(re.search(r"alpha\s*=\s*(np|jnp)\.max", code))


def detect_features(code: str) -> set:
    feats = set()
    for name, pat in [
        ("viscosity", r"\bnu\b|viscos"),
        ("alpha", r"\balpha\b"),
        ("wave_speed", r"\bc\s*\*\*\s*2|\bc2\b|\bc_sq\b"),
        ("pressure", r"\bpoisson\b|pressure|\bp\s*=\s*np\."),
        ("spectral", r"\bfft\b"),
        ("damping", r"\bdamp|\bgamma\b"),
    ]:
        if re.search(pat, code, re.I):
            feats.add(name)
    return feats


# Equation templates, keyed by (pde_class, dim, variant). Unicode / LaTeX / ASCII.
# `variant` distinguishes physically different members of the same class.
EQ = {
    ("heat", "1D", "std"): (
        "∂u/∂t = α ∂²u/∂x²",
        r"\frac{\partial u}{\partial t} = \alpha \frac{\partial^2 u}{\partial x^2}",
        "du/dt = alpha * d2u/dx2",
    ),
    ("heat", "2D", "std"): (
        "∂u/∂t = α (∂²u/∂x² + ∂²u/∂y²)",
        r"\frac{\partial u}{\partial t} = \alpha \nabla^2 u",
        "du/dt = alpha * (d2u/dx2 + d2u/dy2)",
    ),
    ("wave", "1D", "std"): (
        "∂²u/∂t² = c² ∂²u/∂x²",
        r"\frac{\partial^2 u}{\partial t^2} = c^2 \frac{\partial^2 u}{\partial x^2}",
        "d2u/dt2 = c**2 * d2u/dx2",
    ),
    ("wave", "2D", "std"): (
        "∂²u/∂t² = c² (∂²u/∂x² + ∂²u/∂y²)",
        r"\frac{\partial^2 u}{\partial t^2} = c^2 \nabla^2 u",
        "d2u/dt2 = c**2 * (d2u/dx2 + d2u/dy2)",
    ),
    ("wave", "2D", "damped"): (
        "∂²u/∂t² + γ ∂u/∂t = c² (∂²u/∂x² + ∂²u/∂y²)",
        r"\frac{\partial^2 u}{\partial t^2} + \gamma \frac{\partial u}{\partial t}"
        r" = c^2 \nabla^2 u",
        "d2u/dt2 + gamma*du/dt = c**2 * (d2u/dx2 + d2u/dy2)",
    ),
    ("burgers", "1D", "inviscid"): (
        "∂u/∂t + u ∂u/∂x = 0",
        r"\frac{\partial u}{\partial t} + u\frac{\partial u}{\partial x} = 0",
        "du/dt + u*du/dx = 0",
    ),
    ("burgers", "1D", "viscous"): (
        "∂u/∂t + u ∂u/∂x = ν ∂²u/∂x²",
        r"\frac{\partial u}{\partial t} + u\frac{\partial u}{\partial x}"
        r" = \nu \frac{\partial^2 u}{\partial x^2}",
        "du/dt + u*du/dx = nu * d2u/dx2",
    ),
    ("burgers", "2D", "inviscid"): (
        "∂u/∂t + u ∂u/∂x + v ∂u/∂y = 0",
        r"\frac{\partial u}{\partial t} + (\mathbf{u}\cdot\nabla)u = 0",
        "du/dt + u*du/dx + v*du/dy = 0",
    ),
    ("burgers", "2D", "viscous"): (
        "∂u/∂t + u ∂u/∂x + v ∂u/∂y = ν (∂²u/∂x² + ∂²u/∂y²)",
        r"\frac{\partial u}{\partial t} + (\mathbf{u}\cdot\nabla)u = \nu \nabla^2 u",
        "du/dt + u*du/dx + v*du/dy = nu * (d2u/dx2 + d2u/dy2)",
    ),
    ("navier-stokes", "2D", "primitive"): (
        "∂u/∂t + (u·∇)u = −(1/ρ)∇p + ν ∇²u,   ∇·u = 0",
        r"\frac{\partial \mathbf{u}}{\partial t} + (\mathbf{u}\cdot\nabla)\mathbf{u}"
        r" = -\frac{1}{\rho}\nabla p + \nu\nabla^2\mathbf{u},\quad \nabla\cdot\mathbf{u}=0",
        "du/dt + (u.grad)u = -(1/rho)*grad(p) + nu*lap(u), div(u) = 0",
    ),
    ("navier-stokes", "2D", "vorticity"): (
        "∂ω/∂t + (u·∇)ω = ν ∇²ω,   ∇²ψ = −ω",
        r"\frac{\partial \omega}{\partial t} + (\mathbf{u}\cdot\nabla)\omega"
        r" = \nu\nabla^2\omega,\quad \nabla^2\psi = -\omega",
        "dw/dt + (u.grad)w = nu*lap(w), lap(psi) = -w",
    ),
    ("navier-stokes", "2D", "vorticity_forced"): (
        "∂ω/∂t + (u·∇)ω = ν ∇²ω − γω + f",
        r"\frac{\partial \omega}{\partial t} + (\mathbf{u}\cdot\nabla)\omega"
        r" = \nu\nabla^2\omega - \gamma\omega + f",
        "dw/dt + (u.grad)w = nu*lap(w) - gamma*w + f",
    ),
    ("navier-stokes", "1D", "stokes"): (
        "∂u/∂t = ν ∂²u/∂y²",
        r"\frac{\partial u}{\partial t} = \nu \frac{\partial^2 u}{\partial y^2}",
        "du/dt = nu * d2u/dy2",
    ),
}


def choose_variant(pde: str, dim: str, feats: set, processes: str,
                   code: str = "") -> tuple:
    """Return (variant, needs_review, reason)."""
    p = processes.lower()
    # `alpha` only counts as diffusivity if it is not a numerical-flux coefficient,
    # and an explicit nu = 0 overrides the mere presence of the name `nu`.
    alpha_phys = "alpha" in feats and not alpha_is_numerical_dissipation(code)
    visc = (("viscosity" in feats and not has_zero_viscosity(code)) or alpha_phys)

    if pde == "heat":
        return "std", False, ""
    if pde == "wave":
        if dim == "2D" and "damping" in feats:
            return "damped", False, ""
        return "std", False, ""
    if pde == "burgers":
        has_diff = "diffusion" in p
        if has_diff and visc:
            return "viscous", False, ""
        if not has_diff and not visc:
            return "inviscid", False, ""
        # label and code disagree about whether a viscous term is present
        return ("viscous" if visc else "inviscid"), True, (
            f"label says process={processes!r} but code features={sorted(feats)} — "
            f"viscous/inviscid is ambiguous")
    if pde == "navier-stokes":
        if dim == "1D":
            return "stokes", True, (
                "1D Navier-Stokes is unusual; treated as a Stokes/diffusion model. "
                "Needs a domain reader to confirm the intended system.")
        if "spectral" in feats and "damping" in feats:
            return "vorticity_forced", True, (
                "spectral + damping suggests a forced vorticity formulation; "
                "confirm forcing term and whether gamma is damping or drag")
        if is_vorticity_formulation(code):
            # names a vorticity field AND a streamfunction — not a guess
            return "vorticity", False, ""
        if "spectral" in feats and "pressure" not in feats:
            return "vorticity", True, (
                "spectral without an explicit pressure solve suggests "
                "vorticity-streamfunction, but the code names no streamfunction; "
                "confirm the formulation")
        return "primitive", False, ""
    raise ValueError(pde)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default=DEFAULT_MOD_DATASET)
    ap.add_argument("--out", default="data/equations_jul28.csv")
    args = ap.parse_args()

    df = load_dataset(args.dataset)
    base = df[df.mod_type == BASE_CONDITION].sort_values("gt_sample")
    if len(base) == 0:
        raise SystemExit(f"no {BASE_CONDITION} rows in {args.dataset}")

    rows, n_review = [], 0
    for _, r in base.iterrows():
        code = str(r["code"])
        dim, dim_ev, dim_weak = detect_dim(code)
        feats = detect_features(code)
        variant, review, reason = choose_variant(
            r["pde_class"], dim, feats, str(r["phys_process"]), code)
        if dim_weak:
            review = True
            reason = (reason + " | " if reason else "") + (
                f"dimensionality ambiguous: only weak evidence {dim_ev} for 2D, "
                f"treated as 1D")

        key = (r["pde_class"], dim, variant)
        if key not in EQ:
            review, reason = True, f"no template for {key}"
            uni = latex = ascii_ = ""
        else:
            uni, latex, ascii_ = EQ[key]

        n_review += int(review)
        rows.append({
            "gt_sample": r["gt_sample"],
            "pde_class": r["pde_class"],
            "source": r.get("source", ""),
            "dim": dim,
            "variant": variant,
            "equation_unicode": uni,
            "equation_latex": latex,
            "equation_ascii": ascii_,
            "evidence_features": "|".join(sorted(feats)),
            "evidence_dim": "|".join(dim_ev),
            "evidence_num_method": r["num_method"],
            "evidence_phys_process": r["phys_process"],
            "needs_review": int(review),
            "review_reason": reason,
            "verified_by": "",
        })

    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    with open(args.out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)

    print(f"Wrote {args.out}: {len(rows)} equations", flush=True)
    by_variant = {}
    for r in rows:
        by_variant.setdefault(f"{r['pde_class']}/{r['dim']}/{r['variant']}", 0)
        by_variant[f"{r['pde_class']}/{r['dim']}/{r['variant']}"] += 1
    for k in sorted(by_variant):
        print(f"  {k:38s} {by_variant[k]}", flush=True)

    print(f"\n{n_review}/{len(rows)} rows FLAGGED FOR DOMAIN REVIEW:", flush=True)
    for r in rows:
        if r["needs_review"]:
            print(f"  {r['gt_sample']:16s} {r['review_reason']}", flush=True)
    if n_review:
        print("\nThese equations are a first pass derived from code features and "
              "dataset labels. They are the physics ground truth for every "
              "cross-modal result, so they must be signed off (fill `verified_by`) "
              "before any of those results are reported.", flush=True)


if __name__ == "__main__":
    sys.exit(main())
