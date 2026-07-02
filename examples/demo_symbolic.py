"""
examples/demo_symbolic.py
==========================
Demonstrates the symbolic + numerical analysis of new physically-valid
ρ(r) profiles for the static warp bubble metric.

Sections
--------
1. Symbolic analysis (sympy) — derives pressures and energy conditions
   analytically for each profile and prints a human-readable + LaTeX report.

2. Numerical plots — four panels per profile:
   (a) ρ(r)
   (b) β(r)  with the β=1 horizon line
   (c) Energy conditions: NEC perp, WEC, DEC
   (d) Pressures: p_r, p_⊥

3. Comparison plot — all profiles on a single canvas.

Run:
    python examples/demo_symbolic.py
"""

import sys
import os
import numpy as np
import matplotlib
matplotlib.use('Agg')          # no display window needed
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from scipy.interpolate import interp1d

# ── path setup ───────────────────────────────────────────────────────────────
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.abspath(os.path.join(ROOT, '..')))

from static_bubbles import symbolic as sym
from static_bubbles.profiles import (
    profile_exponential, profile_lorentzian,
    profile_gaussian, profile_sech_squared, profile_power_law,
    PHYSICAL_PROFILES,
)

OUTPUT_DIR = os.path.dirname(os.path.abspath(__file__))

# ─────────────────────────────────────────────────────────────────────────────
#  Helpers
# ─────────────────────────────────────────────────────────────────────────────

def compute_beta(r_arr, rho_arr, k=8 * np.pi):
    """β(r) from numerical integration of the mass function."""
    integrand = rho_arr * r_arr**2
    integral = np.zeros_like(r_arr)
    integral[1:] = np.cumsum(
        0.5 * (integrand[:-1] + integrand[1:]) * np.diff(r_arr)
    )
    with np.errstate(divide='ignore', invalid='ignore'):
        beta_sq = np.where(r_arr > 0, k * integral / r_arr**2, 0.0)
    return np.sqrt(np.maximum(beta_sq, 0.0))


def compute_ec(r_arr, rho_arr):
    """Numerical energy conditions from ρ and its derivative."""
    drho = np.gradient(rho_arr, r_arr)
    p_r     = -rho_arr
    p_perp  = -rho_arr - (r_arr / 2.0) * drho
    nec_r   = rho_arr + p_r          # = 0
    nec_p   = rho_arr + p_perp       # = −r/2 dρ/dr
    wec     = rho_arr
    dec     = rho_arr - np.abs(p_perp)
    return dict(p_r=p_r, p_perp=p_perp,
                nec_radial=nec_r, nec_perp=nec_p, wec=wec, dec=dec)


# ─────────────────────────────────────────────────────────────────────────────
#  1. Symbolic analysis
# ─────────────────────────────────────────────────────────────────────────────

SYMBOLIC_PROFILES = [
    ('Exponential  ρ = A·exp(−b·r)',   sym.make_exponential(A=0.3, b=1.5)),
    ('Lorentzian   ρ = A/(1+(r/r₀)²)', sym.make_lorentzian(A=0.5, r0=1.0)),
    ('Gaussian     ρ = A·exp(−(r/σ)²)', sym.make_gaussian(A=0.5, sigma=1.5)),
    ('Sech²        ρ = A/cosh²(b·r)',   sym.make_sech_squared(A=0.5, b=1.0)),
]


def run_symbolic():
    print("\n" + "▶" * 62)
    print("  SECTION 1 — Symbolic Analysis (sympy)")
    print("▶" * 62)

    for name, (rho_expr, r_sym) in SYMBOLIC_PROFILES:
        print(f"\n⏳  Analysing: {name}  ...")
        results = sym.analyze(rho_expr, r=r_sym, do_simplify=True,
                              attempt_beta=True)
        sym.print_report(results, name=name)


# ─────────────────────────────────────────────────────────────────────────────
#  2. Per-profile numerical plots
# ─────────────────────────────────────────────────────────────────────────────

STYLE = dict(
    figure_facecolor='#0d1117',
    axes_facecolor='#161b22',
    text_color='#e6edf3',
    grid_color='#30363d',
    zero_line='#58a6ff',
)

plt.rcParams.update({
    'figure.facecolor' : STYLE['figure_facecolor'],
    'axes.facecolor'   : STYLE['axes_facecolor'],
    'axes.edgecolor'   : STYLE['grid_color'],
    'axes.labelcolor'  : STYLE['text_color'],
    'xtick.color'      : STYLE['text_color'],
    'ytick.color'      : STYLE['text_color'],
    'text.color'       : STYLE['text_color'],
    'grid.color'       : STYLE['grid_color'],
    'legend.facecolor' : '#21262d',
    'legend.edgecolor' : STYLE['grid_color'],
    'legend.labelcolor': STYLE['text_color'],
    'font.family'      : 'DejaVu Sans',
    'axes.grid'        : True,
    'grid.alpha'       : 0.4,
    'lines.linewidth'  : 2.0,
})


def _hline(ax, y=0, **kw):
    ax.axhline(y, color=STYLE['zero_line'], lw=0.8, ls='--', **kw)


def plot_profile(info):
    """Generate a 4-panel figure for one profile."""
    name     = info['name']
    func     = info['func']
    kwargs   = info['kwargs']
    r_max    = info['r_max']
    color    = info['color']
    nec_note = info['nec_note']
    label    = info.get('label', name)

    r_arr  = np.linspace(0, r_max, 2000)
    rho    = func(r_arr, **kwargs)
    beta   = compute_beta(r_arr, rho)
    ec     = compute_ec(r_arr, rho)

    has_horizon = np.any(beta >= 1.0)

    fig = plt.figure(figsize=(14, 10))
    fig.suptitle(
        f"Static Warp Bubble — {name}\n"
        f"Params: {kwargs}   ·   {nec_note}",
        fontsize=13, y=0.98,
    )

    gs = gridspec.GridSpec(2, 2, hspace=0.45, wspace=0.35)
    ax_rho  = fig.add_subplot(gs[0, 0])
    ax_beta = fig.add_subplot(gs[0, 1])
    ax_ec   = fig.add_subplot(gs[1, 0])
    ax_p    = fig.add_subplot(gs[1, 1])

    # ── Panel A: ρ(r) ──────────────────────────────────────────────────
    ax_rho.plot(r_arr, rho, color=color, lw=2.5)
    ax_rho.fill_between(r_arr, 0, rho, where=(rho >= 0),
                        alpha=0.2, color=color)
    _hline(ax_rho)
    ax_rho.set_xlabel('r')
    ax_rho.set_ylabel('ρ(r)')
    ax_rho.set_title('Energy Density')

    # ── Panel B: β(r) ──────────────────────────────────────────────────
    ax_beta.plot(r_arr, beta, color='#58a6ff', lw=2.5, label='β(r)')
    ax_beta.axhline(1.0, color='#f85149', lw=1.5, ls='--', label='β = 1 (horizon)')
    ax_beta.fill_between(r_arr, 0, beta, where=(beta < 1),
                         alpha=0.15, color='#58a6ff')
    if has_horizon:
        ax_beta.fill_between(r_arr, 1, beta, where=(beta >= 1),
                             alpha=0.25, color='#f85149', label='β ≥ 1 !')
    ax_beta.legend(fontsize=8)
    ax_beta.set_xlabel('r')
    ax_beta.set_ylabel('β(r)')
    ax_beta.set_title('Shift Vector' + (' — ⚠ Horizon present' if has_horizon else ' — ✓ No horizon'))

    # ── Panel C: Energy Conditions ─────────────────────────────────────
    ax_ec.plot(r_arr, ec['nec_perp'], label='NEC⊥ = −r/2 ρ′  ≥ 0?', color='#3fb950')
    ax_ec.plot(r_arr, ec['wec'],      label='WEC  = ρ         ≥ 0?', color='#f0883e')
    ax_ec.plot(r_arr, ec['dec'],      label='DEC  = ρ−|p⊥|   ≥ 0?', color='#d2a8ff')
    _hline(ax_ec)
    ax_ec.fill_between(r_arr, 0, ec['nec_perp'],
                       where=(ec['nec_perp'] < 0),
                       alpha=0.2, color='#f85149', label='NEC violated')
    ax_ec.legend(fontsize=8)
    ax_ec.set_xlabel('r')
    ax_ec.set_ylabel('Value  (≥ 0 = satisfied)')
    ax_ec.set_title('Energy Conditions')

    # ── Panel D: Pressures ─────────────────────────────────────────────
    ax_p.plot(r_arr, ec['p_r'],    label='p_r = −ρ',              color='#79c0ff')
    ax_p.plot(r_arr, ec['p_perp'], label='p_⊥ = −ρ − r/2 ρ′',   color='#ffa657')
    ax_p.plot(r_arr, rho,          label='ρ',  color=color, ls=':')
    _hline(ax_p)
    ax_p.legend(fontsize=8)
    ax_p.set_xlabel('r')
    ax_p.set_ylabel('Pressure / density')
    ax_p.set_title('Stress-Energy Components')

    fname = f"symbolic_{name.lower().replace(' ', '_').replace('²','2')}.png"
    out   = os.path.join(OUTPUT_DIR, fname)
    plt.savefig(out, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"  ✓  Saved  {fname}")
    return out


# ─────────────────────────────────────────────────────────────────────────────
#  3. Comparison plot
# ─────────────────────────────────────────────────────────────────────────────

def plot_comparison():
    """One figure showing ρ(r) and β(r) for all physical profiles."""
    fig, (ax_rho, ax_beta) = plt.subplots(1, 2, figsize=(14, 6))
    fig.suptitle(
        "Static Warp Bubble — Profile Comparison\n"
        "All profiles satisfy WEC + NEC everywhere (ρ ≥ 0, ρ′ ≤ 0)",
        fontsize=13,
    )

    for info in PHYSICAL_PROFILES:
        r_max = info['r_max']
        r_arr = np.linspace(0, r_max, 2000)
        rho   = info['func'](r_arr, **info['kwargs'])
        beta  = compute_beta(r_arr, rho)

        lbl = f"{info['name']}  {info['kwargs']}"
        ax_rho.plot(r_arr, rho,  color=info['color'], label=lbl)
        ax_beta.plot(r_arr, beta, color=info['color'], label=lbl, ls='--')

    ax_rho.axhline(0, color=STYLE['zero_line'], lw=0.8, ls='--')
    ax_rho.set_xlabel('r')
    ax_rho.set_ylabel('ρ(r)')
    ax_rho.set_title('Energy Density Profiles')
    ax_rho.legend(fontsize=8)

    ax_beta.axhline(1.0, color='#f85149', lw=1.5, ls=':', label='Horizon β=1')
    ax_beta.set_xlabel('r')
    ax_beta.set_ylabel('β(r)')
    ax_beta.set_title('Shift Vector Profiles')
    ax_beta.legend(fontsize=8)

    out = os.path.join(OUTPUT_DIR, 'symbolic_comparison.png')
    plt.savefig(out, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"  ✓  Saved  symbolic_comparison.png")
    return out


# ─────────────────────────────────────────────────────────────────────────────
#  Main
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == '__main__':

    # 1. Symbolic
    run_symbolic()

    # 2. Per-profile plots
    print("\n" + "▶" * 62)
    print("  SECTION 2 — Numerical Plots")
    print("▶" * 62)
    for info in PHYSICAL_PROFILES:
        print(f"\n  Plotting: {info['name']} ...")
        plot_profile(info)

    # 3. Comparison
    print("\n  Comparison plot ...")
    plot_comparison()

    print("\n✅  All done.  Check the examples/ folder for PNG outputs.\n")
