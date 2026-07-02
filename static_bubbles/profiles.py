"""
static_bubbles/profiles.py
===========================
Energy-density profiles ρ(r) for the static spherically-symmetric warp bubble.

Each function returns ρ evaluated on the input array r (must be a numpy array).

── Paper profiles (Bolívar-Abellán-Vasilev 2025) ────────────────────────────
  profile_single_shell    — piecewise-exponential, NEC violated at junction
  profile_double_shell    — exponential+power-law, limited-support shell

── Physical profiles that satisfy WEC + NEC everywhere ──────────────────────
  Key insight from the Einstein equations:
      NEC perp = −(r/2) dρ/dr  ≥ 0   ↔   dρ/dr ≤ 0
      WEC      = ρ              ≥ 0

  Therefore: any profile that is (i) non-negative and (ii) monotonically
  non-increasing for r ≥ 0 automatically satisfies both WEC and NEC.

  profile_exponential     — A·exp(−b·r)                  [smooth, infinite support]
  profile_lorentzian      — A / (1+(r/r0)²)              [power-law tail]
  profile_gaussian        — A·exp(−(r/σ)²)               [Gaussian, fast decay]
  profile_sech_squared    — A / cosh²(b·r)               [compact-ish, no tail]
  profile_power_law       — A / (1+(r/r0)^n)             [adjustable steepness]
  profile_compact_smooth  — smooth bump on [r1, r2] with guaranteed ρ'≤0 inside
"""

import numpy as np


# ═══════════════════════════════════════════════════════════════════════════════
# Paper profiles (may violate NEC at discontinuities)
# ═══════════════════════════════════════════════════════════════════════════════

def profile_single_shell(r, a=1.0, b=1.0):
    """
    Piecewise-exponential (Single Shell).
    Ref: Eq. (16) — Bolívar-Abellán-Vasilev (2025).

    ρ(r) = 0              for r < 2/b
    ρ(r) = a·exp(−b·r)   for r ≥ 2/b

    ⚠  NEC perp is violated at r = 2/b (jump → ρ' = +∞ as a delta function).
    """
    r = np.asarray(r, dtype=float)
    r_crit = 2.0 / b
    val = np.zeros_like(r)
    mask = r >= r_crit
    val[mask] = a * np.exp(-b * r[mask])
    return val


def profile_double_shell(r, A=1.0, b=1.0, R=0.5):
    """
    Exponential / Power-law decay (Double Shell).
    Ref: Eq. (23) — Bolívar-Abellán-Vasilev (2025).

    ρ(r) = 0                        for r < R
    ρ(r) = A·exp(−b(r−R)) / r²     for R ≤ r ≤ 2/b
    ρ(r) = 0                        for r > 2/b

    ⚠  NEC perp violated at both discontinuities (r=R and r=2/b).
    """
    r = np.asarray(r, dtype=float)
    r_outer = 2.0 / b
    val = np.zeros_like(r)
    mask = (r >= R) & (r <= r_outer)
    r_safe = r[mask]
    with np.errstate(divide='ignore', invalid='ignore'):
        val[mask] = (A * np.exp(-b * (r_safe - R))) / (r_safe**2)
    return val


# ═══════════════════════════════════════════════════════════════════════════════
# Physical profiles  (WEC ✓  NEC ✓  everywhere)
# ═══════════════════════════════════════════════════════════════════════════════

def profile_exponential(r, A=1.0, b=1.0):
    """
    Pure exponential decay.

    ρ(r) = A · exp(−b·r)

    Symbolic NEC perp = A·b·r/2 · exp(−b·r)  ≥ 0  ✓
    WEC: ρ > 0 everywhere  ✓

    β²(r) = 8πA/b³r² · [2 − exp(−br)(b²r² + 2br + 2)]
    No horizon iff  max β < 1  →  choose A/b³ small enough.

    Parameters
    ----------
    A : amplitude (≥ 0)
    b : inverse length scale (> 0); larger b → sharper decay
    """
    r = np.asarray(r, dtype=float)
    return A * np.exp(-b * r)


def profile_lorentzian(r, A=1.0, r0=1.0):
    """
    Lorentzian (Cauchy) profile.

    ρ(r) = A / (1 + (r/r0)²)

    Symbolic NEC perp = A·r·r0² / (r0² + r²)²  ≥ 0  ✓
    WEC: ρ > 0  ✓
    Power-law tail ∝ 1/r² → β converges.

    Parameters
    ----------
    A  : peak energy density at r=0
    r0 : half-width at half-maximum
    """
    r = np.asarray(r, dtype=float)
    return A / (1.0 + (r / r0)**2)


def profile_gaussian(r, A=1.0, sigma=1.0):
    """
    Gaussian centered at the origin.

    ρ(r) = A · exp(−(r/σ)²)

    Symbolic NEC perp = A · (r/σ)² · exp(−(r/σ)²)  ≥ 0  ✓
    WEC: ρ > 0  ✓
    Extremely fast decay beyond r ≈ 2σ.

    Parameters
    ----------
    A     : peak amplitude
    sigma : width parameter
    """
    r = np.asarray(r, dtype=float)
    return A * np.exp(-(r / sigma)**2)


def profile_sech_squared(r, A=1.0, b=1.0):
    """
    Hyperbolic-secant squared (soliton-like).

    ρ(r) = A / cosh²(b·r)  =  A · sech²(b·r)

    Symbolic NEC perp = A·b·r·tanh(br)/cosh²(br)  ≥ 0  ✓  (for r ≥ 0)
    WEC: ρ > 0  ✓
    Decays as 4A·exp(−2br) for large r.

    Parameters
    ----------
    A : peak amplitude at r=0
    b : inverse width; larger b → tighter bubble
    """
    r = np.asarray(r, dtype=float)
    return A / np.cosh(b * r)**2


def profile_power_law(r, A=1.0, r0=1.0, n=2):
    """
    Generalised power-law decay.

    ρ(r) = A / (1 + (r/r0)^n)

    NEC perp = A·n·(r/r0)^n / (r0·(1+(r/r0)^n)²·(r/r0)) · r/2  ≥ 0  ✓
    (positive for r>0, n≥1)
    WEC: ρ > 0  ✓

    Parameters
    ----------
    A  : amplitude
    r0 : scale radius
    n  : power-law index (integer ≥ 1); larger n → sharper edge
    """
    r = np.asarray(r, dtype=float)
    return A / (1.0 + (r / r0)**n)


def profile_compact_smooth(r, A=1.0, r1=0.5, r2=3.0, steepness=5.0):
    """
    Smooth compact-support profile using a logistic envelope.
    Designed to be nearly zero outside [r1, r2] and monotonically
    decreasing in between.

    ρ(r) = A · σ(r1, s) · (1 − σ(r2, s))
    where  σ(r_c, k) = 1 / (1 + exp(k·(r − r_c)))  (reversed logistic)

    This gives a smooth bump that rises near r1, has a flat-ish interior,
    then falls near r2 — approximating a finite-width shell.

    ⚠  This profile is NOT monotonically decreasing from 0; it first rises
    then falls.  NEC perp can be violated near r1 (where ρ'>0).
    Included as an example of a compact shell that approximately satisfies
    conditions in the outer region where ρ'<0.

    Parameters
    ----------
    A          : peak amplitude
    r1, r2     : inner and outer radii of the shell
    steepness  : sharpness of the edges (larger = sharper)
    """
    r = np.asarray(r, dtype=float)
    def logistic(rc, k, r_arr):
        return 1.0 / (1.0 + np.exp(k * (r_arr - rc)))
    inner = logistic(r1, -steepness, r)   # rises at r1
    outer = logistic(r2,  steepness, r)   # falls at r2
    return A * inner * outer


# ═══════════════════════════════════════════════════════════════════════════════
# Catalog helper
# ═══════════════════════════════════════════════════════════════════════════════

#: All physically-valid profiles (WEC + NEC everywhere) with a default
#: parameter set and a short display name.
PHYSICAL_PROFILES = [
    {
        'name'    : 'Exponential',
        'func'    : profile_exponential,
        'kwargs'  : {'A': 0.3, 'b': 1.5},
        'r_max'   : 8.0,
        'color'   : '#2196F3',
        'nec_note': 'NEC ✓  WEC ✓  (ρ′ ≤ 0 everywhere)',
    },
    {
        'name'    : 'Lorentzian',
        'func'    : profile_lorentzian,
        'kwargs'  : {'A': 0.5, 'r0': 1.0},
        'r_max'   : 8.0,
        'color'   : '#4CAF50',
        'nec_note': 'NEC ✓  WEC ✓  (ρ′ ≤ 0 everywhere)',
    },
    {
        'name'    : 'Gaussian',
        'func'    : profile_gaussian,
        'kwargs'  : {'A': 0.5, 'sigma': 1.5},
        'r_max'   : 8.0,
        'color'   : '#FF9800',
        'nec_note': 'NEC ✓  WEC ✓  (ρ′ ≤ 0 everywhere)',
    },
    {
        'name'    : 'Sech²',
        'func'    : profile_sech_squared,
        'kwargs'  : {'A': 0.5, 'b': 1.0},
        'r_max'   : 6.0,
        'color'   : '#9C27B0',
        'nec_note': 'NEC ✓  WEC ✓  (ρ′ ≤ 0 everywhere)',
    },
    {
        'name'    : 'Power-law n=4',
        'func'    : profile_power_law,
        'kwargs'  : {'A': 0.5, 'r0': 1.5, 'n': 4},
        'r_max'   : 8.0,
        'color'   : '#F44336',
        'nec_note': 'NEC ✓  WEC ✓  (ρ′ ≤ 0 everywhere)',
    },
]
