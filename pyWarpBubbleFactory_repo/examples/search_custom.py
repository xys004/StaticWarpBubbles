"""
Search script for a completely custom metric defined via python lambda.
"""

import os
import sys
import numpy as np

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, ROOT)

from pywarpbubblefactory import CustomAnsatz, scan

# Define a custom metric functionally.
# It takes coordinate grids and any kwargs from the parameter scan.
def my_custom_metric(T, X, Y, Z, A=1.0, width=1.0):
    # Determine the shape from X
    shape = (4, 4) + X.shape
    g = np.zeros(shape)
    
    R = np.sqrt(X**2 + Y**2 + Z**2)
    rho_profile = A * np.exp(-(R/width)**2)
    
    # Let's make an ad-hoc metric
    # Diagonal starts as minkowski
    g[0, 0] = -1.0 + rho_profile
    g[1, 1] = 1.0
    g[2, 2] = 1.0
    g[3, 3] = 1.0
    
    # Introduce arbitrary cross terms (tx mixing)
    g[0, 1] = 0.5 * rho_profile
    g[1, 0] = 0.5 * rho_profile
    
    return g

size = (1, 30, 30, 30)
scale = (1.0, 0.2, 0.2, 0.2)
ansatz = CustomAnsatz(func=my_custom_metric, grid_size=size, grid_scale=scale)

param_grid = {
    'A': [0.1, 0.5, 1.0],
    'width': [1.0, 2.0]
}

if __name__ == "__main__":
    print(f"Starting generic search on Custom metric family.")
    df = scan(ansatz, param_grid)
    
    print("\n================ SCAN RESULTS ================")
    print(df.to_string(index=False))
