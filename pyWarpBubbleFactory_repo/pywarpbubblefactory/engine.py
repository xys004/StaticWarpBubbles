"""
pywarpbubblefactory/engine.py
=============================
Core wrapper that hooks into pyWarpFactory's evaluate functions.
"""

import numpy as np
import sys
import copy

# Ensure we can import warpfactory
try:
    from warpfactory.analyzer.energy_conditions import calculate_energy_conditions
except ImportError:
    print("Warning: Please ensure pyWarpFactory is installed or in PYTHONPATH.")

from .results import ECResult
from .ansatze.base import Ansatz

class Engine:
    def __init__(self, num_null_vectors=50, tolerance=-1e-10):
        self.num_null_vectors = num_null_vectors
        self.tolerance = tolerance

    def evaluate(self, ansatz: Ansatz, params: dict) -> ECResult:
        """
        Builds the metric using the ansatz, feeds it to pyWarpFactory,
        and aggregates the results.
        
        Returns an ECResult object.
        """
        metric = ansatz.build(params)
        
        # calculate_energy_conditions signature: (metric_tensor, grid_scale, num_vecs=50)
        # Returns: results_dict, T_euler
        ec, T_euler = calculate_energy_conditions(metric.tensor, metric.scaling, num_vecs=self.num_null_vectors)
        
        rho = T_euler[0, 0] # Energy density in Eulerian frame, shape (Nt, Nx, Ny, Nz)
        
        nec_map = ec['Null']
        wec_map = ec['Weak']
        
        # We handle numerical noise cleanly
        nec_ok = bool(np.all(nec_map >= self.tolerance))
        wec_ok = bool(np.all(wec_map >= self.tolerance))
        
        # "Exotic matter" is roughly the sum of absolute negative energy density
        # Integrate (volumetric sum) over the spatial grid.
        # wec_map is min(rho, nec). Exotic is when wec_map < 0.
        dv = np.prod(metric.scaling[1:]) # dx * dy * dz
        
        exotic = np.sum(np.clip(-wec_map, 0, None)) * dv
        tot_e  = np.sum(rho) * dv
        return ECResult(
            NEC_map=nec_map,
            WEC_map=wec_map,
            rho_map=rho,
            NEC_ok=nec_ok,
            WEC_ok=wec_ok,
            exotic_matter=float(exotic),
            total_energy=float(tot_e),
            T_euler=T_euler
        )
