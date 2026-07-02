"""
Search script for finding valid regions in the WarpShell param space.
Uses the pyWarpBubbleFactory framework to scan parameters and check NEC/WEC dynamically.
"""

import os
import sys

# Add the repo root to path so we can import pywarpbubblefactory
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, ROOT)

from pywarpbubblefactory import WarpShellAnsatz, scan

# Use a coarse grid for fast searching to demonstrate functionality.
# grid_size: Nt, Nx, Ny, Nz
size = (1, 30, 30, 30)
scale = (1.0, 0.2, 0.2, 0.2)
ansatz = WarpShellAnsatz(grid_size=size, grid_scale=scale)

# Define the parameter space over which to search.
# We hold constants and vary v_warp for example.
param_grid = {
    'mass': [1.0, 5.0],
    'r_inner': [1.0],
    'r_outer': [2.5],
    'sigma': [1.0, 3.0],
    'v_warp': [0.0, 0.1, 0.5]  # The kinematic element (which makes things non-static)
}

if __name__ == "__main__":
    print(f"Starting generic search on {ansatz.name} metric family.")
    df = scan(ansatz, param_grid)
    
    # Analyze outputs
    print("\n================ SCAN RESULTS ================")
    print(df.to_string(index=False))
    
    valid = df[df['Valid'] == True]
    print("\n--- Valid Solutions (NEC+WEC satisfied) ---")
    if valid.empty:
        print("None found in this param subspace.")
    else:
        print(valid.to_string(index=False))
    
    print("\nNote: ExoticMatter = 0 means purely positive energy/no violations.")
