"""
static_bubbles/symbolic.py
==========================
Symbolic analysis of the static spherically-symmetric warp bubble metric.

Reference: Bolívar, Abellán, Vasilev — Annals of Physics 481 (2025) 170147.

Metric:
    ds² = -dt² + (dr − β(r) dt)² + r² dΩ²

From the Einstein field equations one derives:
    p_r  = −ρ
    p_⊥  = −ρ − (r/2) dρ/dr

Energy condition summary
────────────────────────
 NEC radial : ρ + p_r  = 0                      (always satisfied)
 NEC perp   : ρ + p_⊥  = −(r/2) dρ/dr  ≥ 0  ↔  dρ/dr ≤ 0
 WEC        : ρ ≥ 0
 DEC        : ρ ≥ |p_⊥|
 SEC        : ρ + p_r + 2 p_⊥ = 2 p_⊥  ≥ 0  ↔  p_⊥ ≥ 0

→ A profile satisfies WEC + NEC everywhere iff  ρ ≥ 0  AND  ρ is non-increasing.

Shift vector (radial magnitude):
    β²(r) = (8π / r²) ∫₀ʳ ρ(s) s² ds
"""

import sympy as sp
import numpy as np


# ─── Core symbolic engine ────────────────────────────────────────────────────

def analyze(rho_expr, r=None, do_simplify=True, attempt_beta=True):
    """
    Full symbolic analysis of the static warp bubble for a given ρ(r).

    Parameters
    ----------
    rho_expr : sympy Expr
        Symbolic energy density as a function of *r*.
    r : sympy Symbol, optional
        Radial coordinate symbol.  Created as Symbol('r', positive=True)
        if not supplied.
    do_simplify : bool
        Apply sp.simplify() to intermediate results (slower but cleaner).
    attempt_beta : bool
        Try to compute β²(r) symbolically via sp.integrate().
        Can be slow or fail for complicated profiles.

    Returns
    -------
    dict with keys:
        r, rho, drho_dr, p_r, p_perp,
        nec_radial, nec_perp, wec, dec, sec,
        beta_sq, integral_rho_s2
    """
    if r is None:
        r = sp.Symbol('r', positive=True)

    # ── Pressures (exact) ────────────────────────────────────────────────
    drho = sp.diff(rho_expr, r)
    p_r    = -rho_expr
    p_perp = -rho_expr - (r * drho) / 2

    # ── Energy conditions (positive ↔ satisfied) ─────────────────────────
    nec_radial = rho_expr + p_r               # = 0
    nec_perp   = rho_expr + p_perp            # = −(r/2) dρ/dr
    wec        = rho_expr                     # ρ ≥ 0
    dec        = rho_expr - sp.Abs(p_perp)    # ρ − |p_⊥| ≥ 0
    sec        = 2 * p_perp                   # 2 p_⊥ ≥ 0

    if do_simplify:
        p_perp     = sp.simplify(p_perp)
        nec_perp   = sp.simplify(nec_perp)
        dec        = sp.simplify(dec)
        sec        = sp.simplify(sec)

    # ── β²(r) via integration ────────────────────────────────────────────
    integral_expr = None
    beta_sq_expr  = None
    if attempt_beta:
        s = sp.Symbol('s', positive=True)
        integrand = rho_expr.subs(r, s) * s**2
        try:
            integral_expr = sp.integrate(integrand, (s, 0, r))
            beta_sq_expr  = 8 * sp.pi * integral_expr / r**2
            if do_simplify:
                beta_sq_expr = sp.simplify(beta_sq_expr)
        except Exception:
            integral_expr = None
            beta_sq_expr  = None

    return {
        'r'              : r,
        'rho'            : rho_expr,
        'drho_dr'        : drho,
        'p_r'            : p_r,
        'p_perp'         : p_perp,
        'nec_radial'     : nec_radial,
        'nec_perp'       : nec_perp,
        'wec'            : wec,
        'dec'            : dec,
        'sec'            : sec,
        'beta_sq'        : beta_sq_expr,
        'integral_rho_s2': integral_expr,
    }


# ─── Reporting ───────────────────────────────────────────────────────────────

_EC_LABELS = [
    ('nec_radial', 'NEC radial  ρ + p_r',  'trivially 0 — always satisfied'),
    ('nec_perp',   'NEC perp   ρ + p_⊥',   '≥ 0  ↔  dρ/dr ≤ 0'),
    ('wec',        'WEC        ρ',          '≥ 0'),
    ('dec',        'DEC        ρ − |p_⊥|', '≥ 0'),
    ('sec',        'SEC        2 p_⊥',      '≥ 0'),
]


def print_report(results, name=""):
    """Pretty-print the symbolic analysis to stdout (plain + LaTeX)."""
    sep = "═" * 62
    print(f"\n{sep}")
    print(f"  SYMBOLIC ANALYSIS  —  {name}")
    print(sep)

    base_items = [
        ("ρ(r)",        results['rho']),
        ("dρ/dr",       results['drho_dr']),
        ("p_r",         results['p_r']),
        ("p_⊥",         results['p_perp']),
    ]
    for label, expr in base_items:
        _show(label, expr)

    print(f"\n{'─'*62}")
    print("  Energy Conditions")
    print(f"{'─'*62}")
    for key, label, hint in _EC_LABELS:
        expr = results[key]
        _show(f"{label}  ({hint})", expr)

    if results.get('beta_sq') is not None:
        print(f"\n{'─'*62}")
        print("  Shift vector")
        print(f"{'─'*62}")
        _show("β²(r) = 8π/r² ∫₀ʳ ρ s² ds", results['beta_sq'])

    print(f"{sep}\n")


def _show(label, expr):
    print(f"\n  {label}:")
    print(f"    = {expr}")
    print(f"    LaTeX: ${sp.latex(expr)}$")


def latex_summary(results):
    """Return a dict of LaTeX strings for every result."""
    return {k: sp.latex(v) for k, v in results.items()
            if isinstance(v, sp.Basic)}


# ─── Numerical conversion ────────────────────────────────────────────────────

def to_numpy(results):
    """
    Lambdify every symbolic expression in *results* to a numpy callable.

    Returns
    -------
    dict : same keys as *results*, values are callables  f(r_array) → array,
           or None if lambdification failed.
    """
    r = results['r']
    numeric = {}
    scalar_keys = [
        'rho', 'drho_dr', 'p_r', 'p_perp',
        'nec_radial', 'nec_perp', 'wec', 'dec', 'sec',
        'beta_sq',
    ]
    for key in scalar_keys:
        expr = results.get(key)
        if expr is None:
            numeric[key] = None
            continue
        try:
            f = sp.lambdify(r, expr, modules=['numpy'])
            # Wrap to always return an array even for constant expressions
            numeric[key] = lambda r_arr, _f=f: np.broadcast_to(
                np.asarray(_f(r_arr), dtype=float), np.asarray(r_arr).shape
            ).copy()
        except Exception:
            numeric[key] = None
    return numeric


# ─── Convenience: build common symbolic profiles ──────────────────────────────

def make_exponential(A=None, b=None):
    """
    ρ(r) = A · exp(−b·r)

    Symbolic parameters if A/b are None; otherwise substituted numerically.
    NEC perp = A·b·r/2·exp(−b·r)  ≥ 0  always  ✓
    """
    r = sp.Symbol('r', positive=True)
    _A = sp.Symbol('A', positive=True) if A is None else sp.Rational(A).limit_denominator(1000)
    _b = sp.Symbol('b', positive=True) if b is None else sp.Rational(b).limit_denominator(1000)
    expr = _A * sp.exp(-_b * r)
    if A is not None and b is not None:
        expr = sp.Float(A) * sp.exp(-sp.Float(b) * r)
    return expr, r


def make_lorentzian(A=None, r0=None):
    """
    ρ(r) = A / (1 + (r/r0)²)

    NEC perp = A·r·r0²/(r0²+r²)²  ≥ 0  always  ✓
    """
    r = sp.Symbol('r', positive=True)
    _A  = sp.Symbol('A',  positive=True) if A  is None else sp.Float(A)
    _r0 = sp.Symbol('r0', positive=True) if r0 is None else sp.Float(r0)
    expr = _A / (1 + (r / _r0)**2)
    return expr, r


def make_gaussian(A=None, sigma=None):
    """
    ρ(r) = A · exp(−(r/σ)²)

    NEC perp = A·r²/σ²·exp(−(r/σ)²)  ≥ 0  always  ✓
    """
    r = sp.Symbol('r', positive=True)
    _A     = sp.Symbol('A',     positive=True) if A     is None else sp.Float(A)
    _sigma = sp.Symbol('sigma', positive=True) if sigma is None else sp.Float(sigma)
    expr = _A * sp.exp(-(r / _sigma)**2)
    return expr, r


def make_sech_squared(A=None, b=None):
    """
    ρ(r) = A / cosh²(b·r)  =  A · sech²(b·r)

    NEC perp = A·b·r·tanh(b·r)/cosh²(b·r)  ≥ 0  always  ✓
    """
    r = sp.Symbol('r', positive=True)
    _A = sp.Symbol('A', positive=True) if A is None else sp.Float(A)
    _b = sp.Symbol('b', positive=True) if b is None else sp.Float(b)
    expr = _A / sp.cosh(_b * r)**2
    return expr, r
