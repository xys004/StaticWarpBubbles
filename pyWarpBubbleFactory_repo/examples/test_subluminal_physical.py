import os
import sys

# Ensure imports work from the root of the repo
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from pywarpbubblefactory.ansatze.warp_shell import WarpShellAnsatz
from pywarpbubblefactory.engine import Engine
from pywarpbubblefactory.visualizer import Visualizer

def run_test():
    print("=" * 60)
    print("🚀 EVALUATING CONSTANT VELOCITY PHYSICAL WARP DRIVE (Brooks 2024) 🚀")
    print("=" * 60)

    # The exact architectural setup for the Subluminal Positive Energy Drive
    # We use a 3D grid large enough to encompass the bubble 
    ansatz = WarpShellAnsatz(
        grid_size=(1, 60, 60, 60), 
        grid_scale=(1, 0.1, 0.1, 0.1)
    )

    # Parameters matching the physical paper definition
    # Subluminal velocity (e.g. 10% of light speed), Positive Mass.
    params = {
        'mass': 5.0,           # Sufficient mass to offset shift frame drag (positive energy)
        'r_inner': 1.0,        # Inner bubble edge
        'r_outer': 2.0,        # Outer bubble edge
        'r_buff': 0.5,         # Buffer zone
        'sigma': 2.0,          # Sigmoid sharpness
        'smooth_factor': 3,    # Smoothing iterations
        'v_warp': 0.1          # 10% Subluminal Drive
    }

    print(f"\n[1] Constructing Shell Ansatz (v = {params['v_warp']}c)...")
    
    # Instantiate the Engine (Our ADM 3+1 Evaluator)
    engine = Engine()
    
    print("[2] Running Einstein Field Equations and ADM projections...")
    result = engine.evaluate(ansatz, params)
    
    print("\n[3] Physical Results:")
    print(f"    - Total Observer Energy : {result.total_energy:.4f} J")
    print(f"    - Total Exotic Matter   : {result.exotic_matter:.4f} J")
    
    if result.exotic_matter < 1e-5:
        print("    -> ✅ 100% PHYSICAL WARP DRIVE DETECTED: No negative energy required!")
    else:
        print("    -> ❌ Exotic matter detected.")

    print("\n[4] Generating Visualizations...")
    plot_dir = os.path.join(os.path.dirname(__file__), "subluminal_plots")
    vis = Visualizer(save_dir=plot_dir)
    vis.render_analysis(result)
    
    print(f"Done! Plots saved locally to: {plot_dir}")

if __name__ == "__main__":
    run_test()
