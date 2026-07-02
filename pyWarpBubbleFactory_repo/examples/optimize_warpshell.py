"""
Demos using ExoticMinimizer on WarpShell.
"""

import os
import sys

# Add the repo root to path
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, ROOT)

from pywarpbubblefactory import WarpShellAnsatz
from pywarpbubblefactory.optimizer import ExoticMinimizer
import pandas as pd

ansatz = WarpShellAnsatz(grid_size=(1, 30, 30, 30), grid_scale=(1.0, 0.2, 0.2, 0.2))

# We want to optimize Mass and Sigma specifically
initial_guess = {
    'mass': 2.0,
    'sigma': 1.0
}

# Keep the others fixed. We give it a slight warp velocity so it's not totally trivial.
fixed = {
    'r_inner': 1.0,
    'r_outer': 2.5,
    'v_warp': 0.1
}

bounds = {
    'mass': (0.1, 10.0),
    'sigma': (0.1, 5.0)
}

if __name__ == "__main__":
    minimizer = ExoticMinimizer(method='Nelder-Mead', maxiter=15, verbose=True)
    
    res = minimizer.optimize(ansatz, initial_guess, bounds, fixed)
    
    print("\n========= OPTIMIZATION DONE =========")
    print(f"Success: {res['success']} ({res['message']})")
    print(f"Best Exotic Matter output: {res['exotic_matter_min']:.3e}")
    print(f"Found at parameters:\n  {res['best_params']}")
    
    print("\nHistory Snapshot (first 5 and last 5):")
    df = pd.DataFrame(res['history'])
    
    if len(df) > 10:
        print(pd.concat([df.head(), df.tail()]).to_string())
    else:
        print(df.to_string())
