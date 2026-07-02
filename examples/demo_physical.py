"""
examples/demo_physical.py
==========================
Concrete warp-bubble examples with SI physical units.

Two things here:
  1. Design concrete examples at β_max = 0.3 / 0.5 / 0.8 for 3 profiles
     (Exponential, Gaussian, Sech²) and visualise ρ(r), β(r).

  2. Energy budget: E(R_bubble) from 1 mm to 1 AU for each profile,
     compared against known energy scales.

Run:
    python examples/demo_physical.py
"""

import sys, os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.lines import Line2D

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, ROOT)

from static_bubbles.profiles import profile_exponential, profile_gaussian, profile_sech_squared
from static_bubbles.units   import (
    total_energy_J, M_exponential, M_gaussian, M_sech2,
    A_for_beta, beta_max, format_energy,
    c, G, c4_G, J_sun, J_earth, J_moon, J_Mt, AU, ly,
)

OUT = os.path.dirname(os.path.abspath(__file__))

# ─── Dark plot style ─────────────────────────────────────────────────────────
BG, PANEL, BORDER, FG = '#0d1117', '#161b22', '#30363d', '#e6edf3'
plt.rcParams.update({
    'figure.facecolor': BG, 'axes.facecolor': PANEL, 'axes.edgecolor': BORDER,
    'axes.labelcolor': FG, 'xtick.color': FG, 'ytick.color': FG,
    'text.color': FG, 'grid.color': BORDER, 'grid.alpha': 0.4, 'axes.grid': True,
    'legend.facecolor': '#21262d', 'legend.edgecolor': BORDER, 'legend.labelcolor': FG,
    'font.family': 'DejaVu Sans', 'font.size': 10, 'lines.linewidth': 2.0,
})

# ─── Profile registry ────────────────────────────────────────────────────────
PROFILES = [
    dict(name='Exponential', kind='exponential',
         func=profile_exponential, scale_key='b',  scale_val=1.5,
         M_func=lambda A, s: M_exponential(A, s),
         color='#58a6ff', char_radius=lambda s: 1.0/s),   # 1/b
    dict(name='Gaussian',    kind='gaussian',
         func=profile_gaussian,    scale_key='sigma', scale_val=1.5,
         M_func=lambda A, s: M_gaussian(A, s),
         color='#f0883e', char_radius=lambda s: s),         # sigma
    dict(name='Sech²',       kind='sech2',
         func=profile_sech_squared, scale_key='b', scale_val=1.0,
         M_func=lambda A, s: M_sech2(A, s),
         color='#d2a8ff', char_radius=lambda s: 1.0/s),    # 1/b
]

TARGETS = [0.3, 0.5, 0.8]
TARGET_LINES = [':', '--', '-']


# ─── 1. Profile plots (ρ and β) ──────────────────────────────────────────────

def plot_profiles():
    fig, axes = plt.subplots(2, 3, figsize=(15, 9))
    fig.suptitle(
        "Concrete warp-bubble examples — Code units\n"
        "Profiles designed for target β_max = 0.3, 0.5, 0.8  (all NEC ✓  WEC ✓)",
        fontsize=12,
    )

    r_arr = np.linspace(0.0, 8.0, 3000)

    for col, pinfo in enumerate(PROFILES):
        ax_rho  = axes[0, col]
        ax_beta = axes[1, col]

        for tgt, ls in zip(TARGETS, TARGET_LINES):
            A = A_for_beta(pinfo['kind'], tgt, pinfo['scale_val'])
            kw = {pinfo['scale_key']: pinfo['scale_val']}
            rho  = pinfo['func'](r_arr, A=A, **kw)

            # β(r) via integration
            integrand = rho * r_arr**2
            dr = r_arr[1] - r_arr[0]
            integral = np.zeros_like(r_arr)
            integral[1:] = np.cumsum(0.5*(integrand[:-1]+integrand[1:])*dr)
            with np.errstate(divide='ignore', invalid='ignore'):
                bsq = np.where(r_arr > 0, 8*np.pi*integral/r_arr**2, 0.0)
            beta_arr = np.sqrt(np.maximum(bsq, 0.0))

            lbl = f'β_max={tgt}  (A={A:.4f})'
            ax_rho.plot(r_arr, rho, ls=ls, color=pinfo['color'], label=lbl)
            ax_beta.plot(r_arr, beta_arr, ls=ls, color=pinfo['color'], label=lbl)

        ax_rho.set_title(f"{pinfo['name']}  (scale={pinfo['scale_val']})")
        ax_rho.set_xlabel('r  [code units]')
        ax_rho.set_ylabel('ρ(r)')
        ax_rho.legend(fontsize=7)

        ax_beta.axhline(1.0, color='#f85149', lw=1.2, ls='--', label='β=1 horizon')
        ax_beta.set_xlabel('r  [code units]')
        ax_beta.set_ylabel('β(r)')
        ax_beta.legend(fontsize=7)

    plt.tight_layout()
    out = os.path.join(OUT, 'physical_profiles.png')
    plt.savefig(out, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  ✓  physical_profiles.png")


# ─── 2. Energy vs bubble radius ──────────────────────────────────────────────

def plot_energy():
    """
    E(R_bubble) for each profile × each β_max target.
    R_bubble = char_radius × L_meter  →  L_meter = R / char_radius(scale)
    """
    R_arr = np.logspace(-3, 11, 500)   # 1 mm → 100 AU in metres

    fig, axes = plt.subplots(1, 3, figsize=(16, 6), sharey=True)
    fig.suptitle(
        "Total energy required vs physical bubble radius\n"
        "E [J] = 4π · M̃ · L [m] · c⁴/G          (all profiles: NEC ✓  WEC ✓)",
        fontsize=12,
    )

    # Horizontal reference lines
    refs = [
        (J_Mt,    '#aaaaff', '1 Mt TNT'),
        (J_earth, '#88cc88', 'Earth mass·c²'),
        (J_sun,   '#ffcc44', 'Solar mass·c²'),
        (1e50,    '#ff7777', '1000 M☉·c²'),
    ]

    for ax, pinfo in zip(axes, PROFILES):
        for tgt, ls in zip(TARGETS, TARGET_LINES):
            A     = A_for_beta(pinfo['kind'], tgt, pinfo['scale_val'])
            Mtild = pinfo['M_func'](A, pinfo['scale_val'])
            R_char = pinfo['char_radius'](pinfo['scale_val'])  # in code units

            # L [m] = R_bubble [m] / R_char [code units]   (since 1 code-unit = L m)
            L_arr = R_arr / R_char
            E_arr = total_energy_J(Mtild, L_arr)

            ax.loglog(R_arr, E_arr, ls=ls, color=pinfo['color'],
                      label=f'β_max={tgt}  (A={A:.4f}  M̃={Mtild:.4f})')

        for E_ref, col, lbl in refs:
            ax.axhline(E_ref, color=col, lw=0.9, ls=':', alpha=0.7)
            ax.text(1e-2, E_ref*1.4, lbl, color=col, fontsize=7, va='bottom')

        # Vertical guides
        ax.axvline(1.0,  color='white', lw=0.6, ls=':')   # 1 m
        ax.axvline(1e3,  color='white', lw=0.6, ls=':')   # 1 km
        ax.axvline(AU,   color='white', lw=0.6, ls=':')   # 1 AU
        ax.text(1.0,   3e26, '1 m', color='white', fontsize=7, rotation=90, va='bottom')
        ax.text(1e3,   3e26, '1 km', color='white', fontsize=7, rotation=90, va='bottom')
        ax.text(AU,    3e26, '1 AU', color='white', fontsize=7, rotation=90, va='bottom')

        ax.set_title(pinfo['name'])
        ax.set_xlabel('Physical bubble radius  R  [m]')
        if ax is axes[0]:
            ax.set_ylabel('Total energy  E  [J]')
        ax.set_xlim(R_arr[0], R_arr[-1])
        ax.set_ylim(1e28, 1e55)
        ax.legend(fontsize=7)

    plt.tight_layout()
    out = os.path.join(OUT, 'physical_energy.png')
    plt.savefig(out, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  ✓  physical_energy.png")


# ─── 3. Summary table ────────────────────────────────────────────────────────

def print_summary():
    print("\n" + "═"*80)
    print("  PHYSICAL SUMMARY — β_max = 0.5 examples")
    print("═"*80)
    print(f"{'Profile':12s} {'Scale':8s} {'A':10s} {'M̃':10s}  "
          f"{'E(R=1 m)':22s}  {'E(R=1 km)':22s}")
    print("─"*80)

    for pinfo in PROFILES:
        tgt   = 0.5
        A     = A_for_beta(pinfo['kind'], tgt, pinfo['scale_val'])
        Mtild = pinfo['M_func'](A, pinfo['scale_val'])
        R_char = pinfo['char_radius'](pinfo['scale_val'])
        sk    = pinfo['scale_key']
        sv    = pinfo['scale_val']

        E_1m  = total_energy_J(Mtild, 1.0    / R_char)
        E_1km = total_energy_J(Mtild, 1.0e3  / R_char)

        print(f"{pinfo['name']:12s} {sk}={sv:<5.1f}  A={A:.4f}    M̃={Mtild:.4f}    "
              f"{format_energy(E_1m):28s}  {format_energy(E_1km)}")

    print("─"*80)
    print()
    print("  Scaling law:  E ∝ R_bubble  (linear in physical size)")
    print("  To halve energy: halve bubble size  OR  choose higher decay rate (b or 1/σ)")
    print()
    print("  Interpretation of β_max:")
    print("    β_max = 0.3  →  well inside valid region, comfortable margin from horizon")
    print("    β_max = 0.5  →  50% of horizon speed, geometrically significant curvature")
    print("    β_max = 0.8  →  80% of horizon — strongly curved, approaching horizon")
    print("    β_max = 1.0  →  HORIZON FORMS — metric becomes degenerate")
    print()
    print("  Corrected scaling laws (β²_max formula):")
    print("    Exponential: β²_max = 8π·A·C_exp / b      [A_crit ∝ b  (linear)]")
    print("    Gaussian:    β²_max = 8π·A·σ·C_gau        [A_crit ∝ 1/σ]")
    print("    Sech²:       β²_max = 8π·A·C_sech / b     [A_crit ∝ b  (linear)]")
    print("═"*80 + "\n")


# ─── 4. Corrected scanner boundary plot ──────────────────────────────────────

def plot_corrected_boundary():
    """Re-plot the exponential no-horizon boundary with the corrected A ∝ b law."""
    from static_bubbles.scanner import scan_2d

    print("  Re-scanning exponential (corrected analytic overlay) …", end='', flush=True)
    p1, p2, bmax = scan_2d(
        profile_exponential, 'A', (0.02, 1.0), 'b', (0.3, 3.5),
        r_max=16.0, n_r=1800, n_p=60,
    )
    print(" done")

    from static_bubbles.scanner import no_horizon_boundary
    A_crit = no_horizon_boundary('exponential', p2)   # now uses corrected formula

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle(
        "Exponential profile — corrected no-horizon boundary\n"
        r"$A_{\rm crit}(b) = b\,/\,(8\pi C_{\rm exp})$   [A ∝ b  (linear)]",
        fontsize=12,
    )

    import matplotlib.colors as mcolors
    CMAP = mcolors.LinearSegmentedColormap.from_list('bm', [
        (0.00, '#0d2b0f'), (0.40, '#3fb950'), (0.50, '#d8c850'),
        (0.65, '#f0883e'), (1.00, '#5a0010'),
    ])
    norm = mcolors.Normalize(0, 2.0)

    for ax, title in zip(axes, ['β_max heatmap (numerical)', 'Valid region + corrected boundary']):
        bmax_c = np.clip(bmax, 0, 2.0)
        ax.pcolormesh(p2, p1, bmax_c, cmap=CMAP, norm=norm, shading='gouraud')
        ax.contour(p2, p1, bmax, levels=[1.0], colors=['white'], linewidths=2.0)
        # corrected analytic curve
        valid = (A_crit >= p1[0]) & (A_crit <= p1[-1])
        ax.plot(p2[valid], A_crit[valid], color='#e3b341', lw=2.2, ls='-.',
                label=r'$A_{\rm crit}=b/(8\pi C_{\rm exp})$  [linear]')
        ax.set_xlabel('b'); ax.set_ylabel('A')
        ax.set_title(title); ax.legend(fontsize=8)

    out = os.path.join(OUT, 'physical_exp_boundary.png')
    plt.tight_layout()
    plt.savefig(out, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  ✓  physical_exp_boundary.png")


# ─── Main ────────────────────────────────────────────────────────────────────
if __name__ == '__main__':
    print("\n▶ Generating profile plots …")
    plot_profiles()

    print("▶ Generating energy plots …")
    plot_energy()

    print("▶ Printing summary table …")
    print_summary()

    print("▶ Corrected boundary plot …")
    plot_corrected_boundary()

    print("\n✅  Done.  Outputs in examples/\n")
