import sys
import os
import numpy as np
import matplotlib.pyplot as plt

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))

from static_bubbles.generator import create_static_bubble_metric
from static_bubbles.analyzer import analyze_static_bubble
from warpfactory.constants import C

def run_demo(label, rho_func, grid_params, plot_params):
    print(f"--- Running {label} Demo ---")
    
    # 1. Generate Metric
    # Using finer grid for better resolution of discontinuities
    metric = create_static_bubble_metric(
        grid_params['size'], grid_params['scale'], grid_params['center'],
        rho_profile=rho_func
    )
    
    # 2. Analyze
    coords = metric.coords
    center = grid_params['center']
    x = coords['x'] - center[1]
    y = coords['y'] - center[2]
    z = coords['z'] - center[3]
    r_3d = np.sqrt(x**2 + y**2 + z**2)
    
    analysis = analyze_static_bubble(rho_func, r_3d)
    
    # 3. Visualization
    sl_y = grid_params['size'][2] // 2
    sl_z = grid_params['size'][3] // 2
    
    r_slice = x[0, :, sl_y, sl_z]
    mask = r_slice >= 0
    r_plot = r_slice[mask]
    
    # Beta
    if 'beta_r' in metric.params:
        beta_vals = metric.params['beta_r']
        r_vals = metric.params['r_samples']
        beta_plot = np.interp(r_plot, r_vals, beta_vals)
    else:
        beta_plot = np.zeros_like(r_plot)
        
    # Analysis Vars
    rho_plot = analysis['rho'][0, :, sl_y, sl_z][mask]
    nec_plot = analysis['NEC'][0, :, sl_y, sl_z][mask]
    dec_plot = analysis['DEC'][0, :, sl_y, sl_z][mask]
    
    output_dir = os.path.dirname(__file__)
    
    # Plot 1: Standard Metric Components
    fig, ax1 = plt.subplots(figsize=(10, 6))
    
    color = 'tab:red'
    ax1.set_xlabel('Radius r')
    ax1.set_ylabel('Energy Density rho', color=color)
    ax1.plot(r_plot, rho_plot, color=color, label='rho(r)')
    ax1.tick_params(axis='y', labelcolor=color)
    
    ax2 = ax1.twinx()
    color = 'tab:blue'
    ax2.set_ylabel('Shift Beta(r)', color=color)
    ax2.plot(r_plot, beta_plot, color=color, linestyle='--', label='beta(r)')
    ax2.tick_params(axis='y', labelcolor=color)
    ax2.axhline(1.0, color='gray', linestyle=':', label='Horizon Limit')
    
    plt.title(f'{label}: Density and Shift')
    plt.tight_layout()
    plot_name = f"static_bubble_{plot_params['name']}_metric.png"
    plt.savefig(os.path.join(output_dir, plot_name))
    print(f"Saved {plot_name}")
    plt.close()
    
    # Plot 2: Energy Conditions
    plt.figure(figsize=(10, 6))
    plt.plot(r_plot, rho_plot, label='rho (WEC)')
    plt.plot(r_plot, nec_plot, label='rho + p_perp (NEC)')
    plt.plot(r_plot, dec_plot, label='rho - |p_perp| (DEC)')
    
    plt.axhline(0, color='black', linewidth=0.5)
    plt.xlabel('Radius r')
    plt.ylabel('Energy Condition Value')
    plt.title(f'{label}: Energy Conditions (Positive = Satisfied)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plot_name = f"static_bubble_{plot_params['name']}_ec.png"
    plt.savefig(os.path.join(output_dir, plot_name))
    print(f"Saved {plot_name}")
    plt.close()


from static_bubbles.profiles import profile_single_shell, profile_double_shell

def example_1_single_shell():
    """
    Example 1: Piecewise-exponential (Single Shell).
    Params from paper Fig 2: a=1, b=1.
    """
    a = 1.0
    b = 1.0
    
    # Create a callable that only takes r (fixing params)
    rho_func = lambda r: profile_single_shell(r, a=a, b=b)

    grid_params = {
        'size': (3, 120, 120, 120),
        'scale': (1.0, 0.1, 0.1, 0.1),
        'center': (0, 6.0, 6.0, 6.0)
    }
    
    run_demo("Ex 1 (Single Shell)", rho_func, grid_params, {'name': 'ex1'})

def example_2_double_shell():
    """
    Example 2: Exponential/Power law decay (Double Shell).
    Params from paper Fig 5: A=1, b=1, R=0.5.
    """
    A = 1.0
    b = 1.0
    R = 0.5
    
    rho_func = lambda r: profile_double_shell(r, A=A, b=b, R=R)

    grid_params = {
        'size': (3, 120, 120, 120),
        'scale': (1.0, 0.05, 0.05, 0.05),
        'center': (0, 3.0, 3.0, 3.0) 
    }
    
    run_demo("Ex 2 (Double Shell)", rho_func, grid_params, {'name': 'ex2'})

if __name__ == "__main__":
    example_1_single_shell()
    example_2_double_shell()

