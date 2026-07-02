"""
examples/demo_scanner.py
========================
Parameter-space exploration for the static warp bubble metric.

For each physically-valid profile (WEC + NEC satisfied analytically),
we scan the 2D amplitude–scale parameter space and visualise:

  • Heatmap of β_max(A, b)  — how "fast" the bubble is
  • β_max = 1 contour        — the horizon boundary
  • Analytic boundary curve  — derived from the scaling law
  • Valid / invalid regions  — green (no horizon) vs red (horizon)
  • Mass contours            — M = ∫ ρ r² dr, shown as dashed lines

A separate diagnostic plot shows 1D slices through the boundary.

Run:
    python examples/demo_scanner.py
"""

import sys
import os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.ticker as ticker
from matplotlib.lines import Line2D

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.abspath(os.path.join(ROOT, '..')))

from static_bubbles.profiles import (
    profile_exponential, profile_lorentzian,
    profile_gaussian, profile_sech_squared,
)
from static_bubbles.scanner import (
    scan_2d, no_horizon_boundary, scaling_exponents,
)

OUTPUT_DIR = os.path.dirname(os.path.abspath(__file__))


# ─── Colour palette ─────────────────────────────────────────────────────────
BG      = '#0d1117'
PANEL   = '#161b22'
BORDER  = '#30363d'
FG      = '#e6edf3'
GREEN   = '#3fb950'
RED     = '#f85149'
BLUE    = '#58a6ff'
ORANGE  = '#f0883e'
PURPLE  = '#d2a8ff'
YELLOW  = '#e3b341'

plt.rcParams.update({
    'figure.facecolor' : BG,
    'axes.facecolor'   : PANEL,
    'axes.edgecolor'   : BORDER,
    'axes.labelcolor'  : FG,
    'xtick.color'      : FG,
    'ytick.color'      : FG,
    'text.color'       : FG,
    'grid.color'       : BORDER,
    'grid.alpha'       : 0.35,
    'axes.grid'        : True,
    'legend.facecolor' : '#21262d',
    'legend.edgecolor' : BORDER,
    'legend.labelcolor': FG,
    'font.family'      : 'DejaVu Sans',
    'font.size'        : 10,
})

# Custom diverging colormap: dark-green → yellow → dark-red
_CMAP_VALS = [GREEN, '#7ee787', '#d2a8ff', ORANGE, RED]
_BETA_CMAP = mcolors.LinearSegmentedColormap.from_list('beta_map', [
    (0.00, '#0d2b0f'),   # deep green   (β_max = 0)
    (0.40, GREEN),       # green        (β_max = 0.8)
    (0.50, '#d8c850'),   # yellow       (β_max = 1 — boundary)
    (0.65, ORANGE),      # orange       (β_max = 1.3)
    (1.00, '#5a0010'),   # deep red     (β_max ≥ 2)
])


# ─── Helper: total mass M = ∫₀^∞ ρ(r) r² dr (vectorised) ─────────────────

def _mass_grid(profile_func, p1_arr, p2_arr, p1_name, p2_name,
               fixed_params, r_max=20.0, n_r=3000):
    r  = np.linspace(0, r_max, n_r)
    p1 = p1_arr[:, np.newaxis, np.newaxis]
    p2 = p2_arr[np.newaxis, :, np.newaxis]
    r3 = r[np.newaxis, np.newaxis, :]
    kwargs = {**fixed_params, p1_name: p1, p2_name: p2}
    rho = profile_func(r3, **kwargs)
    integrand = rho * r3**2
    dr = r[1] - r[0]
    mass = np.trapz(integrand, dx=dr, axis=-1)   # (n1, n2)
    return mass


# ─── Main scan spec ──────────────────────────────────────────────────────────

SCANS = [
    dict(
        label       = 'Exponential   ρ = A·e^{−br}',
        short       = 'Exponential',
        func        = profile_exponential,
        p1          = ('A',  (0.02, 1.20)),
        p2          = ('b',  (0.20, 3.50)),
        fixed       = {},
        r_max       = 16.0,
        boundary    = 'exponential',
        ann_eq      = r'$A_{\rm crit} = b^3 / (8\pi C_{\rm exp})$',
        ann_scaling = r'$A \propto b^3$',
        color       = BLUE,
    ),
    dict(
        label       = 'Lorentzian   ρ = A/(1+(r/r₀)²)',
        short       = 'Lorentzian',
        func        = profile_lorentzian,
        p1          = ('A',  (0.01, 0.60)),
        p2          = ('r0', (0.10, 3.00)),
        fixed       = {},
        r_max       = 18.0,
        boundary    = 'lorentzian',
        ann_eq      = r'$A_{\rm crit} = 1 / (4\pi r_0 C_{\rm lor})$',
        ann_scaling = r'$A \propto 1/r_0$',
        color       = GREEN,
    ),
    dict(
        label       = 'Gaussian   ρ = A·e^{−(r/σ)²}',
        short       = 'Gaussian',
        func        = profile_gaussian,
        p1          = ('A',     (0.02, 1.20)),
        p2          = ('sigma', (0.20, 3.50)),
        fixed       = {},
        r_max       = 16.0,
        boundary    = 'gaussian',
        ann_eq      = r'$A_{\rm crit} = 1 / (8\pi \sigma C_{\rm gau})$',
        ann_scaling = r'$A \propto 1/\sigma$',
        color       = ORANGE,
    ),
    dict(
        label       = 'Sech²   ρ = A·sech²(br)',
        short       = 'Sech²',
        func        = profile_sech_squared,
        p1          = ('A', (0.01, 0.80)),
        p2          = ('b', (0.20, 3.00)),
        fixed       = {},
        r_max       = 16.0,
        boundary    = 'sech',
        ann_eq      = r'$A_{\rm crit} = b^3 / (8\pi C_{\rm sech})$',
        ann_scaling = r'$A \propto b^3$',
        color       = PURPLE,
    ),
]


# ─── Per-profile heatmap ────────────────────────────────────────────────────

def plot_scan(spec, n_p=65):
    p1_name, p1_range = spec['p1']
    p2_name, p2_range = spec['p2']

    print(f"  Scanning {spec['short']} ({p1_name} vs {p2_name}) …", end='', flush=True)
    p1_arr, p2_arr, bmax = scan_2d(
        spec['func'], p1_name, p1_range, p2_name, p2_range,
        fixed_params=spec['fixed'], r_max=spec['r_max'],
        n_r=2000, n_p=n_p,
    )
    print(f"  β_max ∈ [{bmax.min():.3f}, {bmax.min(axis=None):.3f}–{bmax.max():.3f}]")

    # mass grid
    mass = _mass_grid(
        spec['func'], p1_arr, p2_arr, p1_name, p2_name,
        spec['fixed'], r_max=spec['r_max'],
    )

    # analytic boundary A_crit(p2)
    A_crit = no_horizon_boundary(spec['boundary'], p2_arr)

    # ── figure ──────────────────────────────────────────────────────────
    fig, axes = plt.subplots(1, 2, figsize=(14, 6),
                              gridspec_kw={'width_ratios': [1.15, 1]})
    fig.suptitle(
        f"Parameter Space — {spec['label']}\n"
        r"Green = $\beta_{\rm max}<1$ (no horizon)  ·  "
        r"Red = $\beta_{\rm max}\geq1$ (horizon)",
        fontsize=12, y=1.01,
    )

    # ── left panel: β_max heatmap ─────────────────────────────────────
    ax = axes[0]
    bmax_plot = np.clip(bmax, 0, 2.0)     # cap at 2 for colormap saturation
    norm = mcolors.Normalize(vmin=0, vmax=2.0)

    im = ax.pcolormesh(p2_arr, p1_arr, bmax_plot,
                       cmap=_BETA_CMAP, norm=norm, shading='gouraud')
    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label(r'$\beta_{\rm max}$', fontsize=11)
    cbar.ax.axhline(1.0, color='white', lw=1.5, ls='--')

    # β_max = 1 contour (numerical)
    ax.contour(p2_arr, p1_arr, bmax, levels=[1.0],
               colors=['white'], linewidths=2.0)
    # β_max = 0.5, 0.75 extra contours
    ax.contour(p2_arr, p1_arr, bmax, levels=[0.5, 0.75, 1.25, 1.5],
               colors=['white'], linewidths=0.6, linestyles='--', alpha=0.45)

    # analytic boundary (only where in p1 range)
    valid_mask = (A_crit >= p1_range[0]) & (A_crit <= p1_range[1])
    ax.plot(p2_arr[valid_mask], A_crit[valid_mask],
            color=YELLOW, lw=2.2, ls='-.', label=spec['ann_eq'])

    ax.set_xlabel(p2_name, fontsize=11)
    ax.set_ylabel('A  (amplitude)', fontsize=11)
    ax.set_title(r'$\beta_{\rm max}(A, \,' + p2_name + r')$')
    ax.legend(fontsize=8, loc='upper left')

    # annotate valid region
    ax.text(0.97, 0.06,
            r'$\beta_{\rm max}<1$' + '\n(valid bubble)',
            transform=ax.transAxes, ha='right', va='bottom',
            color=GREEN, fontsize=9,
            bbox=dict(boxstyle='round,pad=0.3', fc='#0d2b0f', alpha=0.8))

    # ── right panel: mass + boundary ─────────────────────────────────
    ax2 = axes[1]
    # shade the no-horizon region
    filled = ax2.contourf(p2_arr, p1_arr, bmax,
                          levels=[0.0, 1.0], colors=[GREEN], alpha=0.2)
    over   = ax2.contourf(p2_arr, p1_arr, bmax,
                          levels=[1.0, 999], colors=[RED], alpha=0.12)

    # mass contours
    mc = ax2.contour(p2_arr, p1_arr, mass,
                     levels=8, colors=[BLUE], linewidths=0.8, alpha=0.65)
    ax2.clabel(mc, inline=True, fontsize=7, fmt='M=%.2f', colors=BLUE)

    # β_max = 1 boundary
    ax2.contour(p2_arr, p1_arr, bmax, levels=[1.0],
                colors=['white'], linewidths=2.2)

    # analytic
    ax2.plot(p2_arr[valid_mask], A_crit[valid_mask],
             color=YELLOW, lw=2.0, ls='-.', alpha=0.9,
             label=spec['ann_eq'] + '\n' + spec['ann_scaling'])

    ax2.set_xlabel(p2_name, fontsize=11)
    ax2.set_ylabel('A  (amplitude)', fontsize=11)
    ax2.set_title(r'No-horizon region + mass contours  $M=\int\rho r^2\,dr$')

    legend_els = [
        Line2D([0],[0], color='white', lw=2.2, label=r'$\beta_{\rm max}=1$ (numerical)'),
        Line2D([0],[0], color=YELLOW,  lw=2.0, ls='-.', label='Analytic boundary\n' + spec['ann_scaling']),
        Line2D([0],[0], color=BLUE,    lw=0.8, label=r'Mass contours $M$'),
    ]
    ax2.legend(handles=legend_els, fontsize=8, loc='upper left')

    plt.tight_layout()
    name = spec['short'].lower().replace('²','2').replace(' ','_')
    out  = os.path.join(OUTPUT_DIR, f'scanner_{name}.png')
    plt.savefig(out, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"  → Saved scanner_{name}.png")
    return p1_arr, p2_arr, bmax, A_crit, mass


# ─── Diagnostic: 1D slice of β_max vs amplitude at fixed scale ───────────

def plot_beta_vs_A_slices():
    """
    For each profile, show β_max(A) at several fixed values of the scale
    parameter, marking the β_max = 1 crossing clearly.
    """
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle(
        r'$\beta_{\rm max}$ vs amplitude $A$ — fixed scale parameter slices',
        fontsize=13,
    )

    configs = [
        (profile_exponential, 'A', 'b',     [0.5, 1.0, 1.5, 2.0, 3.0], (0.005, 2.0),  16., 'Exponential'),
        (profile_lorentzian,  'A', 'r0',    [0.5, 1.0, 1.5, 2.0, 3.0], (0.005, 1.0),  18., 'Lorentzian'),
        (profile_gaussian,    'A', 'sigma', [0.5, 1.0, 1.5, 2.0, 3.0], (0.005, 2.0),  16., 'Gaussian'),
        (profile_sech_squared,'A', 'b',     [0.5, 1.0, 1.5, 2.0, 3.0], (0.005, 1.5),  16., 'Sech²'),
    ]
    colors_sl = [BLUE, GREEN, ORANGE, PURPLE, RED]

    for ax, (func, amp_name, scale_name, scale_vals, A_range, r_max, title) in \
            zip(axes.flat, configs):

        A_arr = np.linspace(*A_range, 300)
        r_arr = np.linspace(0, r_max, 2000)

        for sv, col in zip(scale_vals, colors_sl):
            # vectorize over A
            A_3d = A_arr[:, np.newaxis, np.newaxis]
            r_3d = r_arr[np.newaxis, np.newaxis, :]
            kwargs = {amp_name: A_3d, scale_name: sv}
            rho = func(r_3d, **kwargs)           # (n_A, 1, n_r)

            dr = r_arr[1] - r_arr[0]
            integrand = rho * r_3d**2
            half = 0.5*(integrand[...,:-1]+integrand[...,1:])*dr
            integral = np.zeros((len(A_arr), 1, len(r_arr)))
            integral[..., 1:] = np.cumsum(half, axis=-1)
            r_safe = np.where(r_arr>0, r_arr, np.inf)[np.newaxis,np.newaxis,:]
            beta_sq = 8 * np.pi * integral / r_safe**2
            beta_sq[...,0] = 0.
            bmax_arr = np.sqrt(np.maximum(beta_sq,0)).max(axis=(1,2))

            # find crossing A* where β_max = 1
            cross = None
            idx = np.where(np.diff(np.sign(bmax_arr - 1.0)))[0]
            if len(idx) > 0:
                i = idx[0]
                # linear interpolation
                cross = A_arr[i] + (A_arr[i+1]-A_arr[i]) * \
                        (1.-bmax_arr[i]) / (bmax_arr[i+1]-bmax_arr[i])

            lbl = f'{scale_name}={sv}'
            ax.plot(A_arr, bmax_arr, color=col, lw=1.8, label=lbl)
            if cross is not None:
                ax.axvline(cross, color=col, lw=0.8, ls=':', alpha=0.6)
                ax.annotate(f'A*={cross:.3f}',
                            xy=(cross, 1.0), xytext=(cross+0.02, 1.05+0.04*scale_vals.index(sv)),
                            fontsize=6.5, color=col, arrowprops=dict(arrowstyle='->', color=col, lw=0.7))

        ax.axhline(1.0, color='white', lw=1.5, ls='--', label=r'$\beta_{\rm max}=1$')
        ax.fill_between(A_arr, 0, 1,  alpha=0.06, color=GREEN)
        ax.fill_between(A_arr, 1, ax.get_ylim()[1] if ax.get_ylim()[1] > 1.5 else 3.0,
                        alpha=0.06, color=RED)
        ax.set_xlim(*A_range)
        ax.set_ylim(0, None)
        ax.set_xlabel('A  (amplitude)', fontsize=10)
        ax.set_ylabel(r'$\beta_{\rm max}$', fontsize=10)
        ax.set_title(title, fontsize=11)
        ax.legend(fontsize=7, ncol=2)

    plt.tight_layout()
    out = os.path.join(OUTPUT_DIR, 'scanner_slices.png')
    plt.savefig(out, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"  → Saved scanner_slices.png")


# ─── Big overview: all 4 boundary curves on one plot ────────────────────────

def plot_boundary_overview(all_scans):
    """
    Overlay the β_max = 1 boundary for all 4 profiles (rescaled to A vs scale).
    """
    fig, ax = plt.subplots(figsize=(10, 6))
    fig.suptitle(
        r'No-horizon boundary $\beta_{\rm max}=1$ — all profiles',
        fontsize=13,
    )

    markers = ['o','s','^','D']
    for (spec, (p1_arr, p2_arr, bmax, A_crit, _)), mk in zip(all_scans, markers):
        p1n, _ = spec['p1']
        p2n, _ = spec['p2']

        # numeric contour points (just collect crossing A for each p2)
        A_cross = []
        for j, p2v in enumerate(p2_arr):
            col = bmax[:, j]
            idx = np.where(np.diff(np.sign(col - 1.0)))[0]
            if len(idx) > 0:
                i = idx[0]
                a = p1_arr[i] + (p1_arr[i+1]-p1_arr[i]) * \
                    (1.-col[i])/(col[i+1]-col[i])
                A_cross.append((p2v, a))

        if A_cross:
            p2c, Ac = zip(*A_cross)
            ax.plot(p2c, Ac, color=spec['color'], lw=2.2, marker=mk,
                    ms=3, label=f"{spec['short']} (numerical)")

        # analytic
        valid = (A_crit >= spec['p1'][1][0]) & (A_crit <= spec['p1'][1][1])
        ax.plot(p2_arr[valid], A_crit[valid],
                color=spec['color'], lw=1.2, ls='--', alpha=0.7,
                label=f"{spec['short']} {spec['ann_scaling']} (analytic)")

    # fill valid region note
    ax.text(0.97, 0.97, 'Region BELOW each curve:\nno horizon  ✓',
            transform=ax.transAxes, ha='right', va='top',
            color=GREEN, fontsize=10,
            bbox=dict(boxstyle='round,pad=0.4', fc='#0d2b0f', alpha=0.85))

    ax.set_xlabel('Scale parameter  (b  or  r₀  or  σ)', fontsize=11)
    ax.set_ylabel('A  (amplitude)', fontsize=11)
    ax.legend(fontsize=8, ncol=2)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    out = os.path.join(OUTPUT_DIR, 'scanner_overview.png')
    plt.savefig(out, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"  → Saved scanner_overview.png")


# ─── Main ────────────────────────────────────────────────────────────────────

if __name__ == '__main__':
    print("\n" + "▶"*60)
    print("  PARAMETER SPACE SCANNER")
    print("▶"*60 + "\n")

    print("Computing universal scaling constants …")
    C = scaling_exponents()

    print("\nRunning 2D scans …\n")
    all_scans = []
    for spec in SCANS:
        result = plot_scan(spec, n_p=65)
        all_scans.append((spec, result))

    print("\nGenerating 1D slices …")
    plot_beta_vs_A_slices()

    print("\nGenerating boundary overview …")
    plot_boundary_overview(all_scans)

    print("\n✅  Scanner done.  PNG files saved in examples/\n")
