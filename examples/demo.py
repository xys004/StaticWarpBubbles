import sys
import os
import numpy as np
import matplotlib.pyplot as plt

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))

from static_bubbles.generator import create_static_bubble_metric
from static_bubbles.analyzer import analyze_static_bubble
from static_bubbles.profiles import profile_single_shell, profile_double_shell


def run_demo(label, rho_func, grid_params, plot_params, r0=None):
    print(f"--- Running {label} Demo ---")

    # 1. Generate Metric
    metric = create_static_bubble_metric(
        grid_params['size'], grid_params['scale'], grid_params['center'],
        rho_profile=rho_func, r0=r0
    )

    # 2. Build radial coordinate from 1D r_samples stored in the metric
    r_1d = metric.params['r_samples']
    beta_1d = metric.params['beta_r']

    # 3. Evaluate rho and energy conditions on the same 1D radial line
    #    (analyzer works on any array of r values)
    rho_1d = rho_func(r_1d)

    # Numerical derivative of rho for p_perp = -rho - r/2 * rho'
    drho_dr = np.gradient(rho_1d, r_1d)
    p_perp_1d = -rho_1d - (r_1d / 2.0) * drho_dr

    nec_1d  = rho_1d + p_perp_1d          # NEC perp: >= 0 ?
    wec_1d  = rho_1d                       # WEC: rho >= 0 ?
    dec_1d  = rho_1d - np.abs(p_perp_1d)  # DEC: rho >= |p_perp| ?

    output_dir = os.path.dirname(os.path.abspath(__file__))

    # ── Plot 1: Density and Shift ──────────────────────────────────────────
    fig, ax1 = plt.subplots(figsize=(10, 6))

    color_rho  = '#d62728'
    color_beta = '#1f77b4'

    ax1.set_xlabel('Radius r', fontsize=12)
    ax1.set_ylabel('Energy Density  ρ(r)', color=color_rho, fontsize=12)
    ax1.plot(r_1d, rho_1d, color=color_rho, lw=2, label='ρ(r)')
    ax1.tick_params(axis='y', labelcolor=color_rho)
    ax1.axhline(0, color='gray', lw=0.8, ls='--')

    ax2 = ax1.twinx()
    ax2.set_ylabel('Shift  β(r)', color=color_beta, fontsize=12)
    ax2.plot(r_1d, beta_1d, color=color_beta, lw=2, ls='--', label='β(r)')
    ax2.axhline(1.0, color='gray', lw=1.2, ls=':', label='Horizon (β=1)')
    ax2.tick_params(axis='y', labelcolor=color_beta)

    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper right')

    plt.title(f'{label}: Density and Shift', fontsize=14)
    plt.tight_layout()
    plot_name = f"static_bubble_{plot_params['name']}_metric.png"
    plt.savefig(os.path.join(output_dir, plot_name), dpi=150)
    print(f"  Saved {plot_name}")
    plt.close()

    # ── Plot 2: Energy Conditions ──────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(10, 6))

    ax.plot(r_1d, wec_1d,  label='ρ  (WEC: ρ ≥ 0)',           lw=2)
    ax.plot(r_1d, nec_1d,  label='ρ + p⊥  (NEC: ≥ 0)',        lw=2)
    ax.plot(r_1d, dec_1d,  label='ρ − |p⊥|  (DEC: ≥ 0)',      lw=2)

    ax.axhline(0, color='black', lw=1.0)
    ax.fill_between(r_1d, 0, nec_1d,
                    where=(nec_1d >= 0), alpha=0.12, color='green', label='NEC satisfied')
    ax.fill_between(r_1d, nec_1d, 0,
                    where=(nec_1d < 0),  alpha=0.12, color='red',   label='NEC violated')

    ax.set_xlabel('Radius r', fontsize=12)
    ax.set_ylabel('Energy Condition value', fontsize=12)
    ax.set_title(f'{label}: Energy Conditions  (positive = satisfied)', fontsize=14)
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plot_name = f"static_bubble_{plot_params['name']}_ec.png"
    plt.savefig(os.path.join(output_dir, plot_name), dpi=150)
    print(f"  Saved {plot_name}")
    plt.close()


# ── Examples ───────────────────────────────────────────────────────────────

def example_1_single_shell():
    """
    Example 1: Piecewise-exponential (Single Shell).
    Params from paper Fig 2: a=1, b=1.
    rho(r) = a*exp(-b*r) for r >= 2/b, else 0.
    With b=1, profile starts at r=2 and decays to ~0 by r~8.
    Grid: 0..12, spacing 0.1 -> 120 points.
    """
    a, b = 1.0, 1.0
    rho_func = lambda r: profile_single_shell(r, a=a, b=b)

    # Grid spans 0..12 (well past the profile's decay)
    # center at (0, 6, 6, 6) with scale 0.1 -> x in [-6, 6), i.e. r in [0, ~8.5]
    grid_params = {
        'size':   (1, 200, 5, 5),    # thin grid: save memory, enough for 1D slice
        'scale':  (1.0, 0.06, 0.06, 0.06),
        'center': (0, 6.0, 0.18, 0.18)  # center in the middle of x axis
    }

    run_demo("Ex 1 (Single Shell, a=1, b=1)", rho_func, grid_params, {'name': 'ex1'})


def example_2_double_shell():
    """
    Example 2: Double Shell — exponential/power-law decay.
    Params from paper Fig 5: A=1, b=1, R=0.5.
    rho(r) = A*exp(-b*(r-R))/r^2  for R <= r <= 2/b, else 0.
    With b=1, R=0.5: profile lives in [0.5, 2.0].
    Grid: 0..4, spacing 0.02 -> 200 points.
    """
    A, b, R = 1.0, 1.0, 0.5
    rho_func = lambda r: profile_double_shell(r, A=A, b=b, R=R)

    grid_params = {
        'size':   (1, 200, 5, 5),
        'scale':  (1.0, 0.02, 0.02, 0.02),
        'center': (0, 2.0, 0.06, 0.06)
    }

    run_demo("Ex 2 (Double Shell, A=1, b=1, R=0.5)", rho_func, grid_params, {'name': 'ex2'})


if __name__ == "__main__":
    example_1_single_shell()
    example_2_double_shell()
    print("\nDone. Check the examples/ folder for the PNG outputs.")
