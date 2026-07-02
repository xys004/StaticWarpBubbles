"""
static_bubbles/scanner.py
==========================
Vectorized 2D parameter-space scanner for the static warp bubble metric.

Given a profile ρ(r; p1, p2) it sweeps a (p1, p2) grid and computes
β_max = max_r β(r) for each combination.

Key physics
-----------
For the profiles that satisfy WEC + NEC analytically (ρ ≥ 0, ρ′ ≤ 0),
the only remaining constraint for a physically sensible warp bubble is:

    β_max < 1   (no trapped surface / no horizon)

This module finds the boundary β_max = 1 in parameter space and exposes
analytic approximations for simple profiles.

Scaling laws (from the symbolic β² expressions)
------------------------------------------------
Exponential   ρ = A·exp(−b·r):
    β²(r) = (8πA/b) · f(br)      where f(x) = [2−e^{−x}(x²+2x+2)]/x²
    → β_max = sqrt(8πA/b · C_exp),    C_exp = max_{x>0} f(x) ≈ 0.170
    → No-horizon boundary: A = b / (8π·C_exp)            [A ∝ b, LINEAR]

Lorentzian    ρ = A/(1+(r/r0)²):
    β²(r) = 4πA·r0 · g(r/r0)     g(u) = (u−arctan u)/u²
    → β_max = sqrt(4πA·r0 · C_lor),   C_lor ≈ 0.230
    → No-horizon boundary: A = 1/(4π·r0·C_lor)           [A ∝ 1/r0]

Gaussian      ρ = A·exp(−(r/σ)²):
    β²(r) = 8πA·σ · h(r/σ)
    → β_max = sqrt(8πA·σ · C_gau),    C_gau ≈ 0.190
    → No-horizon boundary: A = 1/(8π·σ·C_gau)            [A ∝ 1/σ]

Sech²         ρ = A/cosh²(br):
    β²(r) = (8πA/b) · k(br)      (no closed form, numerical)
    → β_max = sqrt(8πA/b · C_sech),   C_sech ≈ 0.200
    → No-horizon boundary: A = b/(8π·C_sech)              [A ∝ b, LINEAR]
"""

import numpy as np
from scipy.special import erf as _erf
from functools import lru_cache


# ─── Core integration (vectorised) ──────────────────────────────────────────

def _beta_max_grid(profile_func, p1_arr, p2_arr, p1_name, p2_name,
                   fixed_params, r_arr):
    """
    Fully vectorised β_max computation over a 2D (p1 × p2) parameter grid.

    Broadcasting layout
    -------------------
    r_arr : (1, 1, n_r)
    p1    : (n1, 1, 1)
    p2    : (1, n2, 1)
    rho   : (n1, n2, n_r)
    """
    n1, n2, n_r = len(p1_arr), len(p2_arr), len(r_arr)

    r_3d  = r_arr[np.newaxis, np.newaxis, :]        # (1, 1, n_r)
    p1_3d = p1_arr[:, np.newaxis, np.newaxis]        # (n1, 1, 1)
    p2_3d = p2_arr[np.newaxis, :, np.newaxis]        # (1, n2, 1)

    kwargs = {**fixed_params, p1_name: p1_3d, p2_name: p2_3d}
    rho = profile_func(r_3d, **kwargs)               # (n1, n2, n_r)

    # ∫₀ʳ ρ(s) s² ds  via cumulative trapezoidal rule
    dr = r_arr[1] - r_arr[0]
    integrand = rho * r_3d**2
    half = 0.5 * (integrand[..., :-1] + integrand[..., 1:]) * dr
    integral = np.zeros((n1, n2, n_r), dtype=np.float64)
    integral[..., 1:] = np.cumsum(half, axis=-1)

    # β²(r) = 8π / r² · integral
    r_safe = np.where(r_arr > 0.0, r_arr, np.inf)
    r_safe_3d = r_safe[np.newaxis, np.newaxis, :]
    beta_sq = 8.0 * np.pi * integral / r_safe_3d**2
    beta_sq[..., 0] = 0.0                            # β(0) = 0

    beta_max = np.sqrt(np.maximum(beta_sq, 0.0)).max(axis=-1)  # (n1, n2)
    return beta_max


# ─── Public scan API ────────────────────────────────────────────────────────

def scan_2d(profile_func, p1_name, p1_range, p2_name, p2_range,
            fixed_params=None, r_max=14.0, n_r=2000, n_p=70):
    """
    2D parameter-space scan: compute β_max(p1, p2) over a uniform grid.

    Parameters
    ----------
    profile_func : callable  ρ(r, **params) → ndarray
    p1_name      : str       name of first parameter to scan (x-axis)
    p1_range     : (min,max)
    p2_name      : str       name of second parameter to scan (y-axis)
    p2_range     : (min,max)
    fixed_params : dict      extra fixed parameters for profile_func
    r_max        : float     integration upper limit (set large enough)
    n_r          : int       radial resolution (≥ 1000 recommended)
    n_p          : int       number of points per parameter axis

    Returns
    -------
    p1_arr   : ndarray (n_p,)
    p2_arr   : ndarray (n_p,)
    beta_max : ndarray (n_p, n_p)   rows → p1, cols → p2
    """
    if fixed_params is None:
        fixed_params = {}

    p1_arr = np.linspace(*p1_range, n_p)
    p2_arr = np.linspace(*p2_range, n_p)
    r_arr  = np.linspace(0.0, r_max, n_r)

    beta_max = _beta_max_grid(
        profile_func, p1_arr, p2_arr, p1_name, p2_name, fixed_params, r_arr
    )
    return p1_arr, p2_arr, beta_max


# ─── Analytic scaling coefficients ──────────────────────────────────────────

def _universal_max(f_func, u_max=60.0, n=100_000):
    """Find max_{u>0} f_func(u) numerically."""
    u = np.linspace(1e-4, u_max, n)
    return float(np.max(f_func(u)))


@lru_cache(maxsize=None)
def _C_exponential():
    """C_exp = max_{x>0}  [2 - exp(-x)(x²+2x+2)] / x²"""
    def f(x):
        return (2.0 - np.exp(-x) * (x**2 + 2*x + 2)) / x**2
    return _universal_max(f)


@lru_cache(maxsize=None)
def _C_lorentzian():
    """C_lor = max_{u>0}  (u - arctan(u)) / u²"""
    def f(u):
        return (u - np.arctan(u)) / u**2
    return _universal_max(f)


@lru_cache(maxsize=None)
def _C_gaussian():
    """C_gau = max_{u>0}  [sqrt(π)/2·erf(u) - u·exp(-u²)] / u²  ×  σ/A factor absorbed"""
    # β²(r) for Gaussian A·exp(-(r/σ)²):
    #   = 8πA·∫₀ʳ s²·exp(-(s/σ)²)ds / r²
    # Let u=r/σ: integral = σ³·∫₀ᵘ t²·exp(-t²)dt  = σ³·[sqrt(π)/4·erf(u) - u/2·exp(-u²)]
    # β²(r) = 8πA·σ·[sqrt(π)/4·erf(u) - u/2·exp(-u²)] / u²   where u=r/σ
    # C_gau = max_u {[sqrt(π)/4·erf(u) - u/2·exp(-u²)] / u²}
    def f(u):
        num = (np.sqrt(np.pi) / 4.0) * _erf(u) - (u / 2.0) * np.exp(-u**2)
        return num / u**2
    return _universal_max(f, u_max=20.0)


@lru_cache(maxsize=None)
def _C_sech():
    """C_sech = max_{x>0}  ∫₀ˣ t²/cosh²(t) dt / x²  (numerically)"""
    x_arr = np.linspace(1e-4, 30.0, 5000)
    dx = x_arr[1] - x_arr[0]
    integrand = x_arr**2 / np.cosh(x_arr)**2
    integral = np.zeros_like(x_arr)
    integral[1:] = np.cumsum(0.5*(integrand[:-1]+integrand[1:])*dx)
    f = integral / x_arr**2
    return float(np.max(f))


def no_horizon_boundary(profile_name, p2_arr):
    """
    Return the analytic A_crit(p2) curve where β_max = 1.

    Parameters
    ----------
    profile_name : str   one of 'exponential','lorentzian','gaussian','sech'
    p2_arr       : array values of the second parameter (b or r0 or sigma)

    Returns
    -------
    A_crit : array, same shape as p2_arr
    """
    p2 = np.asarray(p2_arr)
    pn = profile_name.lower()

    if pn == 'exponential':
        C = _C_exponential()
        return p2 / (8.0 * np.pi * C)          # A_crit ∝ b  (linear)

    elif pn == 'lorentzian':
        C = _C_lorentzian()
        return 1.0 / (4.0 * np.pi * p2 * C)

    elif pn == 'gaussian':
        C = _C_gaussian()
        return 1.0 / (8.0 * np.pi * p2 * C)

    elif pn in ('sech', 'sech2', 'sech_squared'):
        C = _C_sech()
        return p2 / (8.0 * np.pi * C)          # A_crit ∝ b  (linear)

    else:
        raise ValueError(f"Unknown profile_name '{profile_name}'. "
                         "Use: 'exponential', 'lorentzian', 'gaussian', 'sech'.")


def scaling_exponents():
    """
    Print a summary of scaling laws and the computed universal constants.

    Returns
    -------
    dict with keys 'C_exp', 'C_lor', 'C_gau', 'C_sech'
    """
    C = {
        'C_exp' : _C_exponential(),
        'C_lor' : _C_lorentzian(),
        'C_gau' : _C_gaussian(),
        'C_sech': _C_sech(),
    }
    print("\nUniversal constants for β_max = 1 boundary:")
    print(f"  Exponential:  C_exp  = {C['C_exp']:.6f}")
    print(f"  Lorentzian:   C_lor  = {C['C_lor']:.6f}")
    print(f"  Gaussian:     C_gau  = {C['C_gau']:.6f}")
    print(f"  Sech²:        C_sech = {C['C_sech']:.6f}")
    print()
    print("No-horizon conditions (A < A_crit):")
    print("  Exponential:  A < b  / (8π·C_exp)   [A ∝ b  — linear]")
    print("  Lorentzian:   A < 1  / (4π·r0·C_lor)[A ∝ 1/r0]")
    print("  Gaussian:     A < 1  / (8π·σ·C_gau) [A ∝ 1/σ]")
    print("  Sech²:        A < b  / (8π·C_sech)  [A ∝ b  — linear]")
    return C
