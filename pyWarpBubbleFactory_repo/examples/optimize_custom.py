"""
Demos using ExoticMinimizer on a Custom metric.
"""

import os
import sys
import numpy as np
import pandas as pd

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, ROOT)

from pywarpbubblefactory import CustomAnsatz
from pywarpbubblefactory.optimizer import ExoticMinimizer

def my_custom_metric(T, X, Y, Z, A=1.0, width=1.0, mix=0.1):
    shape = (4, 4) + X.shape
    g = np.zeros(shape)
    
    R = np.sqrt(X**2 + Y**2 + Z**2)
    rho_profile = A * np.exp(-(R/width)**2)
    
    g[0, 0] = -1.0 + rho_profile
    g[1, 1] = 1.0
    g[2, 2] = 1.0
    g[3, 3] = 1.0
    
    # Introduce some tx mixing
    g[0, 1] = mix * rho_profile
    g[1, 0] = mix * rho_profile
    
    return g

ansatz = CustomAnsatz(func=my_custom_metric, grid_size=(1, 30, 30, 30), grid_scale=(1.0, 0.2, 0.2, 0.2))

initial_guess = {
    'A': 1.0,
    'width': 1.5,
}
fixed = {
    'mix': 0.2
}
bounds = {
    'A': (0.01, 5.0),
    'width': (0.5, 3.0)
}

if __name__ == "__main__":
    minimizer = ExoticMinimizer(method='Powell', maxiter=10, verbose=True)
    
    res = minimizer.optimize(ansatz, initial_guess, bounds, fixed)
    
    print("\n========= CUSTOM OPTIMIZATION DONE =========")
    print(f"Success: {res['success']} ({res['message']})")
    print(f"Best Exotic Matter output: {res['exotic_matter_min']:.3e}")
    print(f"Found at parameters:\n  {res['best_params']}")
