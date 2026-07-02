"""
pywarpbubblefactory/ansatze/custom.py
=====================================
Maximum-flexibility ansatz where the user directly provides a function
that returns the components of the metric.
"""

from typing import Dict, Any, Callable
import numpy as np
from .base import Ansatz
from warpfactory.generator.base import Metric
from warpfactory.generator.commons import create_grid

class CustomAnsatz(Ansatz):
    def __init__(self, func: Callable, grid_size=(1, 50, 50, 50), grid_scale=(1, 0.1, 0.1, 0.1), world_center=(0, 0, 0, 0)):
        """
        func: A callable that takes (T, X, Y, Z, **params) and returns a (4, 4, ...) numpy array
              representing the covariant metric tensor g_μν.
        """
        super().__init__("Custom", grid_size, grid_scale, world_center)
        self.func = func
        
        # Precompute the physical coordinates once
        coords = create_grid(grid_size, grid_scale)
        self.coords = coords
        self.T = (coords['t'] + grid_scale[0]) - world_center[0]
        self.X = (coords['x'] + grid_scale[1]) - world_center[1]
        self.Y = (coords['y'] + grid_scale[2]) - world_center[2]
        self.Z = (coords['z'] + grid_scale[3]) - world_center[3]

    def build(self, params: Dict[str, Any]) -> Metric:
        # User function returns g_uv
        g_tensor = self.func(self.T, self.X, self.Y, self.Z, **params)
        
        return Metric(
            tensor=g_tensor,
            coords=self.coords,
            scaling=np.array(self.grid_scale),
            name="Custom User Metric",
            index="covariant",
            params=params
        )
