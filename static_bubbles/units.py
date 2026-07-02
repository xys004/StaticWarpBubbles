"""
static_bubbles/units.py
========================
SI physical-units conversion for the static warp bubble.

In the code all quantities are in geometric/code units (G = c = 1).
This module provides:
  - SI conversion of ρ, β, energies
  - Analytic mass integrals M̃ = ∫ρ r² dr for each profile
  - Analytic β_max formulas (inverse: find A for target β_max)
  - Human-readable energy formatting

Physical derivation
-------------------
β²(r) = 8π/r² ∫₀ʳ ρ(s) s² ds       [geometric → dimensionless β]

For exponential ρ = A·exp(-br), substituting x = br:
    β²(r) = 8πA/b · f(x)   where f(x) = [2 - e^{-x}(x²+2x+2)] / x²
    β²_max = 8πA/b · C_exp             C_exp = max_x f(x) ≈ 0.170
    No-horizon: A < b / (8π·C_exp)     [A ∝ b, LINEAR in b]

For gaussian ρ = A·exp(-(r/σ)²):
    β²(r) = 8πAσ · h(r/σ)
    β²_max = 8πAσ · C_gau              C_gau ≈ 0.190
    No-horizon: A < 1/(8π·σ·C_gau)    [A ∝ 1/σ]

For sech² ρ = A/cosh²(br):
    β²_max = 8πA/b · C_sech            C_sech ≈ 0.200
    No-horizon: A < b/(8π·C_sech)      [A ∝ b]

For Lorentzian ρ = A/(1+(r/r0)²):
    β²(r) = 4πA·r0 · g(r/r0)          from sympy closed form
    β²_max = 4πA·r0 · C_lor            C_lor ≈ 0.230
    No-horizon: A < 1/(4π·r0·C_lor)   [A ∝ 1/r0]
    ⚠ Infinite total energy (power-law tail ∝ 1/r² → ∫ diverges)

Total energy (SI):
    E [J] = 4π · M̃ · L [m] · c⁴/G
    where M̃ = ∫₀^∞ ρ_code(r) r² dr   (dimensionless, computed in code units)
          L  = physical length scale: 1 code-unit = L meters
"""

import numpy as np

# ── Physical constants ────────────────────────────────────────────────────────
c    = 2.99792458e8      # m/s
G    = 6.67430e-11       # m³/(kg·s²)
c4_G = c**4 / G         # J/m  ≈ 1.2135 × 10⁴⁴  (conversion factor)

# ── Reference energy scales ───────────────────────────────────────────────────
J_sun   = 1.98892e30 * c**2   # ≈ 1.788 × 10⁴⁷ J  (solar rest-mass energy)
J_earth = 5.9722e24  * c**2   # ≈ 5.366 × 10⁴¹ J
J_moon  = 7.342e22   * c**2   # ≈ 6.596 × 10³⁹ J
J_Mt    = 4.184e15            # 1 megaton TNT
J_nuke  = 8.368e13            # ~20 kt (Hiroshima)

# ── Length reference scales ───────────────────────────────────────────────────
AU = 1.495978707e11   # m
ly = 9.4607304725e15  # m


# ─── Energy conversion ───────────────────────────────────────────────────────

def total_energy_J(M_tilde: float, L_meter: float) -> float:
    """
    Total SI energy from the dimensionless radial mass integral.

    E [J] = 4π · M̃ · L [m] · c⁴/G

    Parameters
    ----------
    M_tilde : M̃ = ∫₀^∞ ρ_code(r) r² dr    (dimensionless)
    L_meter : physical length scale [m]; 1 code-unit of r = L_meter metres.
    """
    return 4.0 * np.pi * M_tilde * L_meter * c4_G


def rho_SI(rho_code: float, L_meter: float) -> float:
    """Energy density [J/m³]  from code-unit density."""
    return (rho_code / L_meter**2) * c4_G


def format_energy(E_J: float) -> str:
    """Human-readable energy string."""
    a = abs(E_J)
    if a >= J_sun:
        return f"{E_J/J_sun:.3g} M☉c²  ({E_J:.3e} J)"
    if a >= J_earth:
        return f"{E_J/J_earth:.3g} M⊕c²  ({E_J:.3e} J)"
    if a >= J_moon:
        return f"{E_J/J_moon:.3g} M☽c²  ({E_J:.3e} J)"
    if a >= J_Mt:
        return f"{E_J/J_Mt:.3g} Mt TNT  ({E_J:.3e} J)"
    return f"{E_J:.3e} J"


# ─── Analytic mass integrals ──────────────────────────────────────────────────

def M_exponential(A: float, b: float) -> float:
    """M̃ = ∫₀^∞ A·e^{-br} r² dr = 2A/b³"""
    return 2.0 * A / b**3


def M_gaussian(A: float, sigma: float) -> float:
    """M̃ = ∫₀^∞ A·e^{-(r/σ)²} r² dr = A·σ³·√π/4"""
    return A * sigma**3 * np.sqrt(np.pi) / 4.0


def M_sech2(A: float, b: float, r_max: float = 60.0, n: int = 30000) -> float:
    """M̃ = ∫₀^∞ A/cosh²(br) r² dr  (numerical trap rule)"""
    r  = np.linspace(0.0, r_max, n)
    return float(np.trapz(A * r**2 / np.cosh(b * r)**2, r))


# ─── β_max analytic formulas ──────────────────────────────────────────────────
# Correct derivation (see module docstring):
#   Exponential: β²_max = 8π·A/b · C_exp
#   Gaussian:    β²_max = 8π·A·σ  · C_gau
#   Sech²:       β²_max = 8π·A/b  · C_sech
#   Lorentzian:  β²_max = 4π·A·r0 · C_lor

def _C_exp():
    x = np.linspace(1e-5, 80.0, 200_000)
    return float(np.max((2.0 - np.exp(-x)*(x**2+2*x+2)) / x**2))

def _C_gau():
    from scipy.special import erf
    u = np.linspace(1e-5, 20.0, 100_000)
    num = (np.sqrt(np.pi)/4.0)*erf(u) - (u/2.0)*np.exp(-u**2)
    return float(np.max(num / u**2))

def _C_sech():
    x  = np.linspace(1e-5, 40.0, 50_000)
    dx = x[1]-x[0]
    ig = x**2 / np.cosh(x)**2
    I  = np.zeros_like(x); I[1:] = np.cumsum(0.5*(ig[:-1]+ig[1:])*dx)
    return float(np.max(I / x**2))

def _C_lor():
    u = np.linspace(1e-5, 80.0, 200_000)
    return float(np.max((u - np.arctan(u)) / u**2))


# Cache on first call
_CACHE = {}
def _C(name):
    if name not in _CACHE:
        _CACHE[name] = dict(exp=_C_exp, gau=_C_gau, sech=_C_sech, lor=_C_lor)[name]()
    return _CACHE[name]


def beta_max(profile: str, A: float, scale: float) -> float:
    """
    Analytic β_max.

    profile : 'exponential' | 'gaussian' | 'sech2' | 'lorentzian'
    scale   : b (exp/sech2), sigma (gaussian), r0 (lorentzian)
    """
    p = profile.lower()
    if p == 'exponential':
        return np.sqrt(8*np.pi * A / scale * _C('exp'))
    if p == 'gaussian':
        return np.sqrt(8*np.pi * A * scale * _C('gau'))
    if p in ('sech2', 'sech_squared'):
        return np.sqrt(8*np.pi * A / scale * _C('sech'))
    if p == 'lorentzian':
        return np.sqrt(4*np.pi * A * scale * _C('lor'))
    raise ValueError(profile)


def A_for_beta(profile: str, target_beta: float, scale: float) -> float:
    """Invert beta_max(): find A that gives β_max = target_beta."""
    p = profile.lower()
    b2 = target_beta**2
    if p == 'exponential':
        return b2 * scale / (8*np.pi * _C('exp'))
    if p == 'gaussian':
        return b2 / (8*np.pi * scale * _C('gau'))
    if p in ('sech2', 'sech_squared'):
        return b2 * scale / (8*np.pi * _C('sech'))
    if p == 'lorentzian':
        return b2 / (4*np.pi * scale * _C('lor'))
    raise ValueError(profile)


def no_horizon_A(profile: str, scale: float) -> float:
    """A_crit where β_max = 1 (horizon boundary)."""
    return A_for_beta(profile, 1.0, scale)
